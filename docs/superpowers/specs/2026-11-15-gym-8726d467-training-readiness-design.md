# Training readiness and first baseline on Gym 8726d467: design

Status: approved direction, 2026-11-15. The user authorised:

- bringing the gated tensor tests up to Gym `8726d467`;
- a short fresh-weights Ironclad baseline of about 405k learned decisions (about 2–3 h on the RTX 5090);
- launching through a new reusable, profile-driven runner rather than the campaign machinery.

## Why

Learn is pinned to and verified against Gym `8726d467`, but nothing can run tests or training against it. Every
existing harness and launcher pins an older runtime:

- the task4/task6 tensor harnesses pin `db72a7eb` / `533dad1c`;
- the curriculum profiles and bootstrap pin `afb51f5`.

The `8726d467` templates and runtime profile explicitly grant no execution authority. Gym changed M10 rules
(`m10-rules-v2`), so the earlier 41/64 baseline is not comparable. A new baseline is needed before any learner
change can be judged.

## Decisions

1. **One reusable host runner: `scripts/neural/runtime_container.py`.** It takes a runtime profile (the
   `runtime.json` schema v2), an in-container entry module, a writable output directory and resource limits.
   - **Before launch** it verifies:
     - the wheel, `_native` and overlay hashes against the profile;
     - the dependency lock;
     - the learner Git HEAD and dirty state.
   - **Container settings:**
     - offline (`--network none`) with a read-only root;
     - the repository mounted read-only at its host path, so venv paths resolve;
     - the overlay first on `sys.path`, and one writable output directory;
     - CPU or exactly one GPU by UUID;
     - CPU/memory limits and an outer timeout;
     - an owned container name, removed afterwards with absence verified.
   - **Run manifest.** It writes an exclusive manifest recording:
     - the user-authority text and date;
     - the profile hash, Gym revision and build identity;
     - the learner HEAD and dirty state;
     - the command, limits, start/end times and exit status.
   - **Reuse.** It never retries. Future Gym builds reuse it by passing a different profile.
2. **Gated tensor tests on the new runtime.**
   - An in-container entry, `scripts/neural/gated_tests_entry.py`, runs an explicit list of tensor modules:
     `tests.neural.test_model`, `test_distribution`, `test_ppo` and `test_checkpoint`. It uses explicit modules
     rather than discovery.
   - `tests/neural/_support.py` builds its real-schema driver config from the current
     `configs/neural/gym-8726d467/run.toml`.
   - **Sampled-row budget.** `tests/neural/_sample_guard.py` accepts a new task id, `gym-8726d467`. Its
     reservations draw on the existing shared ledger and lifetime cap (4,096 rows), so the cap is not raised.
   - **Outcome.** The suite must end GREEN: every listed test runs and none is skipped. Any skip is a failure.
3. **The baseline reuses Learn's own building blocks.** An in-container entry,
   `scripts/neural/baseline_entry.py`, runs the following loop. Faults stop the run and are never retried; a
   completed segment is never replayed.
   - **Configs.** It binds the `configs/neural/curriculum-v1/recipe.json` scenarios to the `8726d467` contract.
     Prepared configs carry the `8726d467` revision, mappings and the GPU runtime table.
   - **Qualification segment.** A small segment of 512 environment attempts validates the scenario catalogs
     natively before training.
   - **Fresh initialisation.** Training starts with `run_stream_segment(purpose="train")` on mixed shard 0.
   - **Exact resume.** Successive mixed shards then run as `purpose="resume"` segments from each published
     checkpoint. It stops at the first segment end at or above **405,504** learned decisions, or at the wall cap.
   - **Segment size.** 4,096 environment attempts, as in the Nov 2 campaign.
   - **Rollout and PPO settings.** The training settings match the Nov 2 fresh-Gym mixed arm:
     - rollout rows 16 × steps 32;
     - model embedding 16 and hidden 64;
     - PPO learning rate 3e-4, 4 epochs, 64-row minibatches;
     - seeds 1103 / 2207 / 3301 / 4409.
   - **Checkpoints.** It retains the checkpoints nearest 135k and 270k decisions, plus the final one.
4. **Evaluation.**
   - The 64-case development panel comes from `build_development_panel(recipe)`, bound to `8726d467`.
   - It is evaluated for the fresh-initial policy (policy seed 1103, untrained) and for the final checkpoint,
     with the two intermediate checkpoints evaluated if wall time allows.
   - Results are summarised with `summarize_development_reports`: per-case outcomes, a paired comparison and
     cohort tallies.
   - Selection and final splits stay untouched.
5. **Resources and limits.**
   - **GPU.** `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d` (RTX 5090), with the existing CUDA venv from the
     `spire-learn/task8-python:cuda-setup-20261028-a` image.
   - **Wall caps.** The whole baseline is capped at 4 h: 3.5 h for training, with the rest for evaluation.
   - **Storage.** Outputs go under `artifacts/development/neural-combat/baseline-8726d467-20261115/`, which Git
     ignores. Training is capped at 12 GiB of new output.
   - **Free space.** At least 30 GiB of free disk and 12 GiB of free VRAM are required before launch.
6. **Records required by AGENTS.md.** The run records:
   - the environment revision and dirty state;
   - content and profile, schema, reward and learner config/seeds;
   - resource budgets.

   Results are labelled simulator-only, with RNG-001 and TIME-001 preserved. A single replicate is reported as
   such, not as a population claim.

## Out of scope

- Campaign machinery changes; curriculum or sampler changes; any learner, model or reward change.
- Standard-mode training: this first baseline is Ironclad standalone, comparable in design to the Nov 2 mixed arm.
- Multiple training seeds and statistical claims.

## Acceptance

- **Runner.** The runner and both entries have unit tests using fakes, with no Docker or native code in unit
  tests.
- **Tensor tests.** They run GREEN in the container against the `8726d467` overlay, with ledger reservations
  recorded.
- **Baseline.** It completes with these recorded:
  - the qualification segment;
  - training to the decision target, or a clearly reported stop reason;
  - the retained checkpoints;
  - the panel results for fresh-initial and final;
  - the run manifest.
- **Docs.** A results doc records identities, budgets, actual exposure, stop reason, panel outcomes and limits.
