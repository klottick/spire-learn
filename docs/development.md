# Development and evaluation checks

Spire Learn now contains dependency-light case/config validation, a fixed-case
evaluation runner, and bounded native/vector/TorchRL probes. It is still not a
trained learner. The configs below are examples with proposed resource ceilings;
they are not permission to construct environments, evaluate, or run probes.

## Interpreter and core tests

Use CPython 3.12. Set `PYTHON312` explicitly to the exact interpreter selected for
the check (for example, the already-qualified interpreter inside an approved
runtime). Do not rely on whichever `python` happens to be first on `PATH`. The
core package has no selected runtime dependencies; this guide does not install
packages or set up a Gym environment.

From the repository root:

```sh
PYTHONPATH=src "$PYTHON312" -B -m unittest tests.unit.test_config tests.unit.test_cases -v
env -u SPIRE_LEARN_RUN_NATIVE_INTEGRATION \
    -u SPIRE_LEARN_RUN_TORCHRL_INTEGRATION \
    -u SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION \
    PYTHONPATH=src "$PYTHON312" -B -m unittest discover -s tests -v
PYTHONPATH=src "$PYTHON312" -B -m spire_learn --help
```

Default discovery leaves all integration opt-in flags unset. Gated integration
tests then skip; a skip is not runtime evidence. Help and core tests do not need
the optional Gym/ML runtime packages.

## Authored configurations

`[configs/cases/selection-smoke.json](../configs/cases/selection-smoke.json)` is a
two-execution **selection** fixture:

| Scenario | Environment seed | Policy seed |
| --- | ---: | ---: |
| `starter-cultist-a0-v1` | 7 | 101 |
| `starter-jaw-worm-a0-v1` | 11 | 103 |

Case and execution IDs are unique. These are diagnostic examples, not a benchmark
corpus, training/model-selection data, final-evaluation cases, or authorization to
run. No train or final manifest is supplied.

All config-relative paths resolve from the config file's directory. Evaluation
profiles use `agent = "random"` and the same selection manifest:

| Config | Profile | Decisions per case | Global wall budget |
| --- | --- | ---: | ---: |
| [`smoke.toml`](../configs/evaluation/smoke.toml) | `smoke` | 128 | 60 s |
| [`development.toml`](../configs/evaluation/development.toml) | `development` | 1,024 | 600 s |
| [`overnight.toml`](../configs/evaluation/overnight.toml) | `overnight` | 4,096 | 28,800 s |

These numbers are proposed ceilings, not a request or permission to evaluate.
Evaluation is single-environment-per-case; the wall budget applies to the whole
invocation. Each profile has a distinct output path under ignored `runs/`.

The separate [`vector-smoke.toml`](../configs/probes/vector-smoke.toml) and
[`torchrl-smoke.toml`](../configs/probes/torchrl-smoke.toml) files both reference
the same two-execution manifest. Each sets `max_batch_steps = 128`,
`max_wall_seconds = 60`, `num_envs = 2`, and `workers = 1`; the `[probe]` table has
no other fields, seed arrays, or partition override. These are probe examples,
also subject to separate runtime authority.

### Side-effect-free CLI validation

The following commands parse the real authored configs and their manifests without
starting evaluation/probes or creating run artifacts:

```sh
PYTHONPATH=src "$PYTHON312" -B -m spire_learn evaluate --config configs/evaluation/smoke.toml --validate-only
PYTHONPATH=src "$PYTHON312" -B -m spire_learn evaluate --config configs/evaluation/development.toml --validate-only
PYTHONPATH=src "$PYTHON312" -B -m spire_learn evaluate --config configs/evaluation/overnight.toml --validate-only
PYTHONPATH=src "$PYTHON312" -B -m spire_learn probe-vector --config configs/probes/vector-smoke.toml --validate-only
PYTHONPATH=src "$PYTHON312" -B -m spire_learn probe-torchrl --config configs/probes/torchrl-smoke.toml --validate-only
```

`--validate-only` checks mode, strict fields, paths, manifest split/overlap,
expectations, and probe execution count. It must not create an environment, run
directory, or optional import. Removing `--validate-only` is an execution, not a
validation step; it requires separate explicit authorization. `--allow-final` is
evaluation-only and does not make the supplied selection cases a final set.

## Runtime gates and current evidence

Initial conformance work was CPU-only. CPU evidence is not a permanent device
architecture decision; no GPU/CUDA or training run is evidenced here.

- **Single native adapter (Task 6):** the recorded protected offline run executed
  all 3 required tests with 0 skips and exit 0. It delivered 49 steps across the
  two recorded A0 selection scenarios; both ended in public victory. This is
  bounded single-environment conformance, not an evaluation-runner execution.
- The recorded native results identify schema `spire-lab-combat-observation-v1`,
  content `spire-lab-combat-reference-v2`, reward contract
  `spire-lab-combat-winloss-v1`, and descriptor fingerprint
  `8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357`. The
  observed build identity was Spire Gym revision
  `4874fb398128c142bd56f10a12e6fca939a88f93`, Rust 1.98.0. These identify those
  retained runs; they are not a claim about a caller's current checkout.
- **Native vector probe:** the separately gated run executed 1 test with 0 skips
  and reported `compatible` for its bounded cohort: 11 batch steps, 22 slot
  decisions, and 1 replacement decision. This is a specific CPU cohort result,
  not a general throughput or learner-strength claim.
- **TorchRL:** the first recorded cohort was `incompatible` on observation
  mapping; its precise causal leaf mismatch was not established. A later bounded
  run with the learner-owned Dict-final reader first passed 3 CPU API tests (0
  skips), then 1 cohort test (0 skips). That run reported `adaptable`, all 8
  required checks passing, 11 batches, 22 slot decisions, 1 replacement, and 22
  diagnostic samples on TorchRL 0.14.1, TensorDict 0.14.3, and Torch 2.14.1+cpu.
  This demonstrates the tested adaptation for that pinned runtime/cohort, not
  blanket TorchRL compatibility.
- **Qualified setup history:** the initial CPU TorchRL setup made 96 diagnostic
  draws against an approved cumulative cap of 32 across three attempts. Setup
  remains recorded as blocked/qualified; later successful probe checks do not
  erase or retroactively cure that deviation. The later cohort's 22 draws were
  accounted under its separate 256-row probe cap.
- **Evaluation integration:** accepted at source `732b714`. The single bounded
  native test executed once with 0 skips and completed the two selection cases:
  Cultist in 18 actions and Jaw Worm in 10, both public terminal victories. The
  28 total actions, case identities, artifact counts, and summary reconcile;
  no faults, secondary failures, or budget overshoot occurred. Parent and fresh
  independent review accepted the evidence. M1 foundation is complete for this
  scoped profile; these diagnostic outcomes are not a strength result.
- **Checkout/build separation:** the installed native build remains revision
  `4874fb398128c142bd56f10a12e6fca939a88f93`. The unmounted Gym checkout was
  independently observed clean at `36fbdc6310945caa194f2aa806c37769275228c3`
  and unchanged across execution. The first wrapper stopped before any container
  or test because it assumed the earlier checkout revision; that failed preflight
  and its incomplete cleanup records remain preserved. The revised wrapper
  recorded the source observation separately and ran the test once.

All learner-owned setup and evidence are inside this repository under
`artifacts/development/foundation/runtime/`. Container isolation does not create a
separate project or workspace. The evaluator reused the existing qualified venv;
no host dependency setup or additional install was needed.

The corresponding local evidence and raw command/output records are retained in
`artifacts/development/foundation/`, including the [native integration report](../artifacts/development/foundation/native-integration-report-2026-10-25.md),
[vector runtime report](../artifacts/development/foundation/task-7-vector-runtime-report.md),
[Dict-final runtime report](../artifacts/development/foundation/task-7-dict-final-runtime-report.md),
[contract-gate record](../artifacts/development/foundation/task-7-contract-freeze.md),
[Task 9 report](../artifacts/development/foundation/task-9-report.md),
[evaluator runtime report](../artifacts/development/foundation/evaluation-native-smoke-v2-runtime-report.md),
[independent runtime review](../artifacts/development/foundation/evaluation-native-smoke-v2-runtime-review.md),
and [M1 acceptance report](../artifacts/development/foundation/m1-acceptance-report.md).
The parent-owned [revised evaluation runbook](../artifacts/development/foundation/evaluation-smoke-runbook-v2.sh)
records the exact accepted invocation. These local development artifacts are ignored by
Git and are not packaged with this documentation.

## Opt-in integration commands

Set `PYTHON312` to the exact CPython 3.12 interpreter in the already-approved,
isolated runtime. The parent-owned runbook has static readiness PASS, but the
command forms below are **gates, not authorization**; do not run them unless a
separate approval identifies the exact attempt. Do not install or repair
dependencies as part of a gate.

The native evaluator test satisfied the Task 9 runtime gate; the command is
recorded here for reference, and any future invocation requires its own authority:

```sh
SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_evaluation -v
```

That test uses the two selection cases with 128 decisions per case and a 60-second
global budget. The opt-in is deliberately distinct from the safe `--validate-only`
commands above.

For a future separately authorized TorchRL shard, the API-reader gate must pass
before the cohort command; stop if it fails:

```sh
SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_torchrl_dict_final -v
SPIRE_LEARN_RUN_TORCHRL_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_torchrl_adapter -v
```

The historical native/vector invocations and their actual mounted interpreter and
paths are recorded in the reports above. These local command forms do not replace
those records or authorize a rerun.

## Artifacts, provenance, and qualifications

An execution creates a unique run directory under its configured output root.
The writer records `config.json`, `manifest.json`, `episodes.jsonl`,
`manifest_updates.jsonl`, `metrics.jsonl`, and `summary.json`. Metrics include
transition records when `traces = true`. `config.json` records learner provenance,
Python/platform, distributions, the environment contract, and RNG/TIME notices;
`manifest_updates.jsonl` carries observed public descriptor and native-build
metadata when the environment is discovered. The shipped profiles leave optional
`gym_checkout` unset, so they do not claim a Gym worktree revision/dirty state.
For a real run, use a run-specific config with `gym_checkout` set to the actual
checkout path when that source provenance is required; retain the environment's
observed public native build identity separately.

Evaluation summaries include `elapsed_wall_seconds` and
`overshoot_wall_seconds`, measured from the shared run budget after case work and
owned artifact streams close, but before `summary.json` serialization. These
values describe that boundary, not total process runtime. If the final clock read
fails, both values are null and the run reports a reliability failure while
retaining delivered case outcomes.

Keep generated runs, logs, rollouts, and checkpoints out of Git. Preserve the
literal command, stdout, stderr, exit status, config/manifest, environment revision
and dirty state, content/profile, schema/reward contract, seeds, learner settings,
runtime identity, resource ceilings, and raw evidence together. The Task 10
verification transcript is in the [Task 10 report](../artifacts/development/foundation/task-10-report.md)
and its adjacent `task-10-logs/` directory in this workspace.

**RNG-001:** project-owned reproducibility is not seed-for-seed or bit-for-bit
original-game RNG parity; no original-runtime transfer is established. **TIME-001:**
the full-run Secret Portal elapsed-time assumption is retained and has not been
validated by these combat scenarios. Simulator conformance, probe success, and
diagnostic smoke results do not establish playing strength or original-game
transfer.
