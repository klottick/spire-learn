# Gym 8726d467 Training Readiness and Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run Learn's gated tensor tests and a fresh-weights ~405k-decision Ironclad baseline on Gym `8726d467`,
through one reusable profile-driven container runner.

**Architecture:**
- A host runner (`scripts/neural/runtime_container.py`) verifies a `runtime.json` profile and launches one
  bounded, offline container.
- Two in-container entries reuse Learn's existing code:
  - `gated_tests_entry.py` runs explicit tensor-test modules;
  - `baseline_entry.py` drives `prepare_stream`, `run_stream_segment`, `build_development_panel` and
    `summarize_development_reports`.
- No training logic is new.

**Tech Stack:** Python 3.12 stdlib (host); the runtime image's CUDA venv (container); Docker.

**Spec:** `docs/superpowers/specs/2026-11-15-gym-8726d467-training-readiness-design.md`

## Global Constraints

- Profile `configs/neural/gym-8726d467/runtime.json`. Image `spire-learn/task8-python:cuda-setup-20261028-a`.
  GPU `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`.
- **Container shape:**
  - `--network none` and `--read-only`;
  - the repository mounted read-only at its host path;
  - exactly one writable output directory under `artifacts/development/neural-combat/`;
  - an owned container name, removed afterwards with absence verified;
  - no retries.
- **Baseline training settings:**
  - target 405,504 learned decisions; segment 4,096 environment attempts; rollout rows 16, steps 32;
  - embedding 16 and hidden 64;
  - PPO learning rate 3e-4, 4 epochs, 64-row minibatches;
  - seeds policy 1103, optimizer 2207, curriculum 3301, noncombat 4409.
- **Baseline limits:**
  - wall cap 4 h overall, with 3.5 h for training;
  - new output at most 12 GiB;
  - at least 30 GiB free disk and 12 GiB free VRAM before launch.
- **Tensor tests:** the modules `tests.neural.test_model`, `test_distribution`, `test_ppo` and `test_checkpoint`.
  The `_sample_guard` task id is `gym-8726d467`, and the shared ledger lifetime cap stays 4,096.
- Unit tests never run Docker, native code or Torch. Run them with
  `PYTHONPATH=src python3.12 -B -m unittest discover -s tests -t . -p '<file>.py'`, with no full discovery.
- Faults are errors, never outcomes. A completed segment is never replayed. Selection and final splits stay
  untouched.
- Implementers do no installs, Docker, native or training runs, pushes or history rewrites. The controller runs
  every real container.

## Review Focus

1. **Runner exit paths.** A runner interrupted or timed out mid-run must remove its container and leave a
   manifest stating the failure. It must never leave a running container or a manifest that claims success.
2. **Segment failure.** A segment ending without a healthy published checkpoint stops the baseline with a
   recorded stop reason. It must never silently restart from scratch.
3. **Budget accounting.** The decision target counts actual learned decisions from saved progress, not
   configured budgets.

---

### Task 1: Profile-driven container runner

**Files:**
- Create: `scripts/neural/runtime_container.py`
- Test: `tests/unit/test_runtime_container.py`

**Interfaces:**
- `load_profile(path: Path) -> RuntimeProfile`, a frozen dataclass with these fields:
  - `path` and `sha256`;
  - `gym_revision` and `build_identity`;
  - `wheel_path` and `wheel_sha256`;
  - `overlay`, `native_sha256` and `dependency_venv`;
  - `dependency_lock_sha256`.

  It rejects any schema other than `spire-learn-gym-runtime-profile-v2`.
- `verify_profile(profile, root: Path) -> dict`: recomputes the wheel and overlay `_native` hashes, and checks
  the venv lock and that the overlay is read-only. It returns the verification record and raises `ValueError` on
  any mismatch.
- `docker_command(profile, *, root, entry: str, args: Sequence[str], output: Path, name: str, gpu_uuid: str | None, cpus: float, memory: str, env: Mapping[str, str]) -> list[str]`.
  The command runs `<venv python> -I -B -c <bootstrap>`. The bootstrap puts the overlay, `src` and the repository
  root first on `sys.path`, then `runpy`-runs `scripts/neural/<entry>`.
- `run(profile_path, *, entry, args, output, authority: str, gpu_uuid, timeout_seconds, cpus, memory, env, docker=subprocess-backed) -> int`:
  1. Exclusively creates `output/manifest.json` and records the start.
  2. Creates and starts a uniquely named container, waits with the timeout, then collects the logs to
     `output/stdout.txt` and `output/stderr.txt`.
  3. Always removes the container and verifies its absence.
  4. Records the end in a separate `output/manifest-final.json`, created exclusively, with:
     - the exit status and whether it timed out;
     - the container's removal;
     - the learner HEAD and dirty state (`git status --porcelain`), plus the profile verification.

  It never retries.
- **CLI:** `python3.12 scripts/neural/runtime_container.py --profile P --entry E --output O --authority TEXT [--gpu UUID] [--timeout S] [--cpus N] [--memory M] -- <entry args>`.

- [ ] **Step 1: Write the failing tests.** Use a fake docker in the style of `tests/unit/test_neural_sustained_controls.py:FakeDocker`.
  - Command shape: `--network none`, `--read-only`, read-only repository mount, a single writable output mount,
    and the GPU flag only when a UUID is given.
  - A profile hash mismatch is rejected before any docker call.
  - The manifests are exclusive; an existing output directory is rejected.
  - A timeout produces kill, remove and verified absence, plus a final manifest with `timed_out: true` and a
    non-zero exit.
  - A docker create failure produces a final manifest and no container.
  - No retry on a non-zero exit.
  - The dirty state is recorded.
- [ ] **Step 2: Implement, run the module, and commit.** Commit message:
  `feat: add a profile-driven runtime container runner`.

### Task 2: Gated tensor tests on 8726d467

**Files:**
- Create: `scripts/neural/gated_tests_entry.py`
- Modify: `tests/neural/_sample_guard.py`, `tests/neural/_support.py` (~454)
- Test: `tests/unit/test_gated_tests_entry.py`

**Interfaces:**
- **Sample guard.** `_sample_guard` accepts task `gym-8726d467` alongside `task4` and `task6`, sharing the same
  ledger and the 4,096 lifetime cap.
- **Real-schema config.** `_support.accepted_schema_and_config` loads `configs/neural/gym-8726d467/run.toml`.
- **Entry behaviour.** `gated_tests_entry.py --run-id ID --ledger PATH --evidence DIR`:
  1. Reserves 64 rows in the ledger under `gym-8726d467-<ID>`, refusing an existing ID.
  2. Sets `SPIRE_LEARN_RUN_NEURAL_TENSOR=1` and the guard environment.
  3. Runs exactly the four modules in one `unittest` process.
  4. Writes `result.json` with the tests run and their outcomes.
  5. Exits 0 only when every expected test ran, with zero skips, failures and errors.

- [ ] **Step 1: Write the failing tests.**
  - The guard accepts the new task id and still enforces the cap.
  - The entry refuses a reused run ID.
  - The entry computes GREEN and RED correctly from a fake unittest result, and treats any skip as RED.
  - `_support` points at the 8726d467 template.
- [ ] **Step 2: Implement, run the modules, and commit.** Commit message:
  `feat: run gated tensor tests on Gym 8726d467`.

### Task 3: Baseline entry (controller)

**Files:**
- Create: `scripts/neural/baseline_entry.py`
- Create: `configs/neural/baseline-8726d467/{recipe.json,campaign.json}`, bound to the 8726d467 contract with the
  Global Constraints settings
- Test: `tests/unit/test_baseline_entry.py`

**Interfaces:**
- `baseline_entry.py --plan configs/neural/baseline-8726d467/campaign.json --output DIR`
- **Phases:**
  1. Prepare and run the qualification segment.
  2. Run a fresh `train` on mixed shard 0, then `resume` on each successive shard from the published checkpoint.
  3. After each segment, append to `progress.jsonl` and check the decision target, wall cap and storage.
  4. Retain the checkpoints nearest 135k and 270k decisions, plus the final one.
  5. Build the development panel and evaluate the fresh-initial policy and the final checkpoint, then the
     intermediates while time remains.
  6. Write `summary.json`.
- **Failure handling.** Any segment that yields no checkpoint, any fault and any cap stops the run with a recorded
  `stop_reason`. Nothing is retried.

- [ ] **Step 1: Write failing tests with fake `prepare_stream`, `run_stream_segment` and evaluation.**
  - The sequence is train, then resume, resume, and so on.
  - The run stops on reaching the decision target, counted from saved prefixes.
  - It stops on a missing checkpoint without restarting.
  - Wall-cap and storage stops work.
  - Checkpoint retention picks the right checkpoints.
  - Panel evaluation covers fresh and final.
  - The summary contents are correct.
- [ ] **Step 2: Implement, run, and commit.** Commit message: `feat: add the Gym 8726d467 baseline entry`.

### Task 4: Real runs and results (controller)

- [ ] **Step 1: Tensor tests.** Run them through the runner on CPU, then on the GPU, and record both results.
- [ ] **Step 2: Baseline.** Check free disk and VRAM and that the GPU is idle, then run the baseline through the
  runner on the GPU with a 4 h timeout.
- [ ] **Step 3: Document the results.** Write `docs/neural-baseline-8726d467.md` with:
  - identities and budgets;
  - actual exposure and the stop reason;
  - panel outcomes, as a paired table of fresh-initial against final, with intermediates;
  - limits: a single replicate, simulator only, RNG-001/TIME-001, and no comparison with pre-8726d467 scores.

  Link it from the README.
- [ ] **Step 4: Commit.** Message: `docs: record the Gym 8726d467 baseline`.
