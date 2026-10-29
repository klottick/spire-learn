# Neural combat training and evaluation

The neural runner consumes the accepted public Spire Gym combat adapter. It
does not implement game rules or expose seeds, routing, RNG state, or lifecycle
diagnostics to the policy. Run mode is pinned to the `m10-ironclad-policy`
profile and observation/replay schema v3.

Task 7 adds bounded training, checkpoint resume and warm start, frozen
evaluation, and two stateless public-action references. The implementation has
pure fake-driven coverage. The single approved native CPU training smoke ran
once: both mode tests passed with zero skips, within the declared decision,
environment, update, and time ceilings. It demonstrates bounded training and
checkpoint plumbing only. Approved Tasks 1–7 are accepted after parent and
independent review, with the documented minor qualifications retained. Task 8
subsequently completed the bounded GPU calibration described below and is
accepted after parent and independent review. No playing-strength or frozen-policy evaluation
is claimed. See the [retained CPU runtime report](../artifacts/development/neural-combat/task-7/task-7-training-smoke-runtime-report.md).

## Commands

Use Python 3.12 and the already qualified runtime when an execution has been
released. Importing the CLI, help, and `--validate-only` use only the standard
library and do not initialize a device, construct a Gym environment, or create
an output directory.

```sh
PYTHONPATH=src python3.12 -B -m spire_learn.neural --help
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/standalone.toml --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/run.toml --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural evaluate --config configs/neural/standalone-selection.toml --policy legal-random --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural evaluate --config configs/neural/run-selection.toml --policy end-turn-last-v1 --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural calibrate --config configs/neural/calibration.toml --validate-only
```

`train` accepts either `--resume CHECKPOINT` or `--warm-start CHECKPOINT`; the
options are mutually exclusive. Resume restores optimizer/RNG state, cumulative
budget counters, policy version, and the collector's next seed/decision/episode
ordinals. It creates a fresh environment and discards in-flight rollout rows.
Warm start loads model weights and records parent lineage while starting new
seeds, episodes, counters, and optimizer state.

`evaluate --policy checkpoint` requires `--checkpoint PATH`. The reference
policies reject checkpoint arguments. Checkpoint evaluation defaults to
deterministic action selection; `--action-selection seeded-sampling` is also
available. `legal-random` uses seeded sampling, while `end-turn-last-v1` chooses
the lowest legal public action whose descriptor verb is not `end_turn`, falling
back to the lowest legal action. Both references use the same accepted public
candidate and selection contract. They produce fixed-controller rows without
fabricated policy values or log probabilities.

Final configs require the explicit `--allow-final` flag. Selection and final
cases are separate, and evaluation never invokes the optimizer. `calibrate`
uses an isolated CUDA path with a strict calibration config and qualified UUID.
The recorded one-use cohort has completed; the command above validates only
and does not authorize another execution. General `train`/`evaluate` now has
an authored device-aware extension. Its pure tests do not establish GPU runtime
compatibility. The single reviewed GPU proof has now passed both standalone
B4/h8 and Run B1/h32: two tests, zero skips, 64 environment actions, 40 learned
decisions and eight accepted optimizer steps. Run contributed eight learned
and 24 fixed-controller decisions. Each mode made one finite changed-weight
update, saved its checkpoint and passed exact same-backend state/RNG/recovery
comparisons plus retained-frame CPU portability. The [retained review](../artifacts/development/neural-combat/task-9/gpu-proof/runtime-review-gpu-proof-20261029-e.md)
records the closed evidence and limitations. No held-out gameplay or longer
learning has run.

General CUDA configs require exactly `device = "cuda:0"` and a `[runtime]`
table containing only the qualified `gpu_uuid`. Actual execution additionally
requires the protected GPU opt-in and matching UUID before Torch loads; config
validation does not initialize hardware. CPU configs and stateless reference
policies remain available without a CUDA query. Resume requires the original
config and matching RNG backend/device metadata; warm start and evaluation use
portable CPU-deserialized weights without restoring training progress.

Exit code `0` means the bounded operation finished and wrote sound artifacts; it
does not mean the policy learned or won. Code `2` marks invalid configuration or
blocked setup. Code `1` marks an operational or artifact failure, including an
evaluation that could not finish its fixed manifest. Keyboard and other
process-control exceptions propagate after owned resources close.

## Authored workloads

| Config | Mode/split | Cases | Rollout | Limits |
| --- | --- | ---: | ---: | --- |
| [`standalone.toml`](../configs/neural/standalone.toml) | standalone/train | train catalog | 4 × 32 | 256 learned decisions, 1,024 environment slots, 16 optimizer calls, 120 s |
| [`run.toml`](../configs/neural/run.toml) | Run/train | train catalog | 1 × 128 | 256 learned decisions, 1,024 environment slots, 16 optimizer calls, 120 s |
| [`standalone-selection.toml`](../configs/neural/standalone-selection.toml) | standalone/selection | 6 fixed cases | 1 step per collect | 768 policy decisions, 4,096 environment slots, 600 s |
| [`run-selection.toml`](../configs/neural/run-selection.toml) | Run/selection | 1 fixed Run | 1 step per collect | 768 policy decisions, 4,096 environment slots, 600 s |
| [`standalone-final.toml`](../configs/neural/standalone-final.toml) | standalone/final | 6 fixed cases | 1 step per collect | 768 policy decisions, 4,096 environment slots, 600 s |
| [`run-final.toml`](../configs/neural/run-final.toml) | Run/final | 1 fixed Run | 1 step per collect | 768 policy decisions, 4,096 environment slots, 600 s |

The completed one-time Task 7 CPU smoke used the two train configs together: at most 512
learned decisions, 2,048 environment slots and 32 optimizer attempts, with two
rollout updates per mode enforced before another collection. One shared
120-second soft deadline covers both modes. Standalone batches contain four
rows over 32 steps; Run uses one row over 128 steps. Each successful mode
requires accepted optimizer work, zero uncertain steps, finite final weights
and at least one changed parameter. The smoke reloads each final checkpoint and
compares deterministic policy outputs on public frames retained from the same
training rollouts. Within those same two mode tests, it creates a resume runtime
from the real checkpoint with a fresh fake driver and compares model,
optimizer, RNG, counters, version and recovery state without environment
effects. It adds no separate evaluation or native cohort.

The run recorded 396 learned decisions, 444 environment slots, and 28 accepted
optimizer steps across two updates per mode. It had no faults, uncertain
effects, or wall-budget overshoot. The [retained runtime report](../artifacts/development/neural-combat/task-7/task-7-training-smoke-runtime-report.md)
contains per-mode accounting and checkpoint evidence.

The frozen evaluation configs use one row and `rollout.steps = 1` so a finished
fixed case is not reset into another scheduled case before the runner accounts
for the terminal transition. Multirow fixed-case evaluation is rejected before
native runtime construction. Reports account for every manifest case as complete,
incomplete, or not started. They separate standalone combat from fights reached
inside a Run and report completion, outcomes, reached-combat denominator, public
HP retention when available, turns/actions, and full-Run terminal progress.

## Bounded GPU calibration

The qualified runtime uses Torch 2.14.1+cu130/CUDA 13.0 on the inventoried
RTX 5090. The single completed cohort used standalone B4/B8/B16 at horizon32,
then one Run capped at 128 actions. It passed one test with zero skips, delivered
948 environment step slots and 942 learned samples, and made three disposable
GPU Adam timing calls. The Run ended naturally in death after 52 steps. There
were no simulator faults or uncertain effects. No trained checkpoint was kept.

B16 was fastest among the tested standalone sizes at 40.724 delivered slots/s,
including collection, same-frame probes, the disposable update and close.
Its same-frame GPU forward/backward probe was 4.381 times faster than CPU;
B4/B8 showed no GPU benefit. This was one fixed-order pass, not an estimate of
training performance or playing strength. Peak Torch allocated/reserved memory
was 205.455/250 MiB; process inventory observed 880 MiB separately.

The [runtime report](../artifacts/development/neural-combat/task-8/calibration/calibration-continuation-runtime-report.md)
records exact input identities, timing boundaries, previous preflight failures,
the one-use continuation, preservation and successful container cleanup.
The shared legacy diagnostic-sampling budget remains separate from these real
actor decisions. The general device-aware runner and checkpoint/RNG extension
is described in the [runtime design](superpowers/specs/2026-10-28-neural-gpu-runtime-design.md).
Its bounded two-mode GPU proof is accepted after parent and independent
source/evidence review. Saved elapsed prefixes end at the checkpoint snapshot;
post-snapshot serialization/finalization/close time is outside the saved prefix.
The proof's shared deadline includes that work in this attempt; it does not
establish whole-lifecycle cumulative charging across future resumes.
No longer learning, frozen selection, final evaluation or new cohort is
authorized by this documentation.

## Checkpoints and artifacts

Each training run writes a `config.json`, `manifest.json`, `events.jsonl`,
`summary.json`, and a final checkpoint directory. Checkpoints retain the accepted
Task 6 `training_state` schema. Task 7 stores the typed collector recovery
projection in `run_metadata`; its seed and decision high-water values must
agree with the settled training-state scalars, and the sum of per-slot next
episode ordinals must equal cumulative delivered reset slots. A resume
checkpoint without consistent recovery metadata is rejected before driver
construction.

Events store accepted action indexes, terminal/bootstrap flags, public-frame
hashes, seed allocations, public combat settlement summaries, optimizer metrics,
and cumulative accepted/uncertain counters. They do not serialize raw public
frames or private simulator state. The manifest records the pinned environment
revision, profile/content/reward/schema identity, learner settings, controller
identity, policy-selection mode, resource caps, and RNG-001/TIME-001
qualifications. Runtime provenance records the learner checkout state; the
environment dirty state remains `unobserved` when no Gym source checkout is
mounted, while the loaded extension must match the configured accepted revision.

Run directories and checkpoints are generated artifacts under
`artifacts/development/neural-combat/` and remain out of Git by default. See the
[Task 7 report](../artifacts/development/neural-combat/task-7/task-7-report.md)
for implementation evidence, the [retained smoke runtime
report](../artifacts/development/neural-combat/task-7/task-7-training-smoke-runtime-report.md)
for the one-time result, and the [historical smoke
runbook](../scripts/neural/task7-training-smoke-runbook.md) for the protocol and
stop conditions.

The persisted summary marks its final lifecycle receipt as required and uses
exit code 1 until that receipt is written after runtime and event-writer close.
The separate exclusive receipt is authoritative for final status and elapsed
time; its timing boundary ends before the receipt's own write.

## Limits

This finite authored catalog is a plumbing and conformance workload. It is not
evidence of generalization, playing strength, seed-for-seed original-game RNG
parity, or original-runtime transfer. RNG-001 qualifies project-owned
reproducibility. TIME-001 qualifies the unvalidated full-Run Secret Portal
elapsed-time assumption. Final evaluation is reserved for a separately reviewed
experiment; it must not be used for model or hyperparameter selection.

The [combined review](../artifacts/development/neural-combat/final-combined-review.md)
records the remaining historical minor qualifications. Task 8 corrected model
initialization to preserve the caller's CPU/CUDA RNG states and passed the
qualified RNG checks before the recorded GPU calibration. Earlier failures and
the missing original calibration RED transcript remain disclosed in its reports.
