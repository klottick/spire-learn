# Device-aware neural training runtime

## Scope and authority

This refines the approved Ironclad learning architecture after Task 8 acceptance
at `541f995`. The user's instruction to continue authorizes the narrow runtime
integration and concrete next configurations. Execution method remains parent
synthesis with task-scoped Luna/MAX workers and substantive independent review.
Existing main/local-commit authority persists from the earlier user handoff
("Work on main" and "Local Git commits"); this continue is not its sole basis.
The first delivery is code/core
tests only. Protected tensor/native proof requires its concrete reviewed gate;
longer Task 9 learning/selection remains a separately approved experiment.

Use the existing direct PyTorch trainer for both standalone and normal Run
collection. No second trainer, new model, Gym change, dependency install, cloud,
or performance rewrite is needed. Preserve the public-only actor boundary,
accepted seed domains, finite-content limits and RNG-001/TIME-001. Task 8's
single calibration is complete; none of its claims or workloads are rearmed.

## Current implementation and selected approach

Model/features/distribution already support tensors on the policy device. PPO
requires a separate CPU generator for minibatch permutations and moves selected
tensors to the policy device. `create_runtime` is the remaining CPU gate. Extend
it using a small stdlib-only `neural/device.py`, leaving PPO, Collector, model,
features, distribution and checkpoint serialization contracts unchanged.

Checkpoint payloads continue to load with CPU `map_location` and
`weights_only=True`. Build the selected-device model and copy named weights
strictly into it. Delegate Adam state loading to the pinned optimizer: default
noncapturable/nonfused Adam retains its scalar `step` on CPU and places moments
with parameters. Never recursively move every optimizer tensor to CUDA.

## Configuration and device contract

General train/evaluate configs support `cpu` or exactly `cuda:0` for the one
qualified device. CUDA configs require an exact `[runtime]` table containing
only `gpu_uuid`; its value must be
`GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`. Add defaulted frozen
`NeuralConfig.runtime_gpu_uuid: str | None = None`. CPU configs omit this table.
Missing/unknown/bool/malformed/wrong fields fail before optional imports or
factories. Existing calibration-only table, config identities and CPU files
remain compatible. No unqualified `cuda` or other index/device is admitted.

`RuntimeDevicePlan` is immutable and contains `device`,
`policy_generator_device`, `shuffle_generator_device`, `policy_rng_backend`
(`cpu` or `cuda`), and `gpu_uuid`. `plan_runtime_device(config)` validates these
without optional imports. Keep the config/device type dependency lazy or under
TYPE_CHECKING so either fresh-process import order works. Learned policies use the selected model/actor
generator device; shuffle remains `cpu`. Stateless reference policies require
CPU configs and remain Torch-free.

`require_runtime_execution_gate(plan)` is stdlib-only, a no-op for CPU, and
checks the exact GPU phase opt-in/UUID before importing Torch. Help/config
validation calls neither this execution gate nor hardware qualification.
For resume, call `preflight_checkpoint` and validate nested run metadata before
any Torch import, including the Torch import inside `load_checkpoint`.

`qualify_runtime_device(torch_module, plan)` is called only during explicit
learned execution, before loading tensor payloads or creating a native driver.
CPU qualification performs no CUDA operation. CUDA requires the protected
execution gate `SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=1`, exact
`SPIRE_LEARN_NEURAL_GPU_UUID`, one visible available device, Torch2.14.1+cu130 /
CUDA13.0, RTX5090 / capability12.0 / compiled sm_120, driver595.91.07, at least
12 GiB free, and four intra-op /one inter-op CPU threads. Query UUID through a
bounded nvidia-smi call; do not assume CUDA ordinal proves UUID. Enforce the
existing 8 GiB allocator ceiling. Return only primitive observed metadata for
runtime provenance. The protected gate supplies the audited runtime/package/
native/lock bindings; do not rerun setup, install variants or silently fall back.

## Checkpoint and RNG semantics

New training checkpoints add `runtime_device`, `policy_rng_backend` and
`runtime_gpu_uuid` to `run_metadata`. Existing strict checkpoint payload/schema,
identity/config/manifest/recovery/counter gates remain intact.

The three new metadata fields are atomic: all absent means legacy CPU; partial
or malformed fields reject. CPU metadata has backend cpu and no UUID; CUDA has
device cuda:0, backend cuda, and the exact qualified UUID.
Exact resume requires matching device/backend and UUID plus the existing exact
config hash. Treat legacy metadata with no device/backend fields as CPU only.
Malformed or incompatible explicit fields reject using JSON preflight before
Torch/device/native construction. Do not transform CPU RNG state into CUDA RNG
state. Construct each correct generator, then give its loaded CPU byte state
unchanged to `set_state`. Python curriculum/controller RNG streams retain their
existing independent recovery.

Warm start/evaluation load portable model weights across CPU/CUDA when the
accepted model/schema/mode mappings match. They do not restore optimizer/RNG/
progress. Warm start creates fresh selected-device actor RNG and CPU shuffle
RNG plus a fresh optimizer; frozen evaluation creates no optimizer. Preserve
zero accepted-effect retries and first process-control precedence on every
assembly/load/close failure.

## Bounded validation proposal

After source and runbook review, one protected GPU functional cohort can qualify
the integration, separately counted from Task 8 calibration. Proposed caps:
one standalone B4/horizon8 and one Run B1/horizon32, at most32 environment step
slots per mode (64 total), at most69 reset slots, one rollout update per mode,
default four PPO epochs/minibatch64, at most4 optimizer attempts per mode
(8 total). Actor sampling at most64 rows plus4 explicitly reserved same-frame
resume diagnostic rows (68 total). No additional native evaluation or outcome
search. A mode with no accepted update is inconclusive/failure, never a pass.

Keep one shared120s soft/300s whole budget including cleanup, 4 CPUs/8 GiB host/
8 GiB GPU/512 PIDs, fresh >=12 GiB GPU and >=30 GiB host free, <=1 GiB new output
under the retained40 GiB cap. Reuse the qualified CUDA venv offline/non-root with
read-only source/runtime and one owned writable child. Bind exact source/config/
manifest/lock/native/Git/UUID identities and one-use claim durably before create.

Use the public train orchestration with `RuntimeOperationLimits(max_updates=1,
monotonic_deadline=shared_deadline)`. Save a final checkpoint per mode. Check
same-backend resume using an inert driver that forbids reset/step, comparing
model, Adam placement/state, RNG/version/counters/recovery. Compare deterministic
and bounded sampled actions on the same retained public frames; no fresh
gameplay. Check portable CPU warm-start/evaluation weights and no optimizer/
RNG restoration, again through inert drivers. Do not count these comparisons as
held-out selection or learning efficacy. Preserve original config hashes on
resume; never alter the config to make recovery pass.

The proof must be authored and reviewed before release. No runtime authority is
inferred from fake tests. Longer four-run learning and held-out selection need
their own exact post-calibration config/manifest/checkpoint approval package.
