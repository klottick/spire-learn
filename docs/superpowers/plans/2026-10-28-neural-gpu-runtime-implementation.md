# Device-aware neural runtime implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Extend the existing trainer to the qualified CUDA device while keeping portable CPU checkpoint loading and exact RNG/recovery semantics.

**Architecture:** Add a small lazy device module, a strict general-GPU config table, and narrow runner assembly/save changes. Keep the existing PPO, Collector, model, serialization and protected runtime as dependencies. First deliver code/core evidence; author and review the bounded GPU proof separately before release.

**Tech Stack:** Python3.12, stdlib unittest; existing Torch2.14.1+cu130/CUDA13.0, NumPy2.5.4, Gymnasium1.4.0 and accepted own-built Spire Lab. No installs.

**Spec:** `docs/superpowers/specs/2026-10-28-neural-gpu-runtime-design.md`.

## Global constraints

- Main/local commits explicitly authorized in the earlier user handoff; that authority persists alongside this continue. Preserve the user foundation report and root diff. One writer, Luna/MAX, parent actual-code synthesis and independent review; no nested workers.
- Do not change Gym, PPO/Collector/model/features/distribution/checkpoint schema or accepted Task8 setup/calibration controls without a concrete parent ruling.
- CPU/import/help/config paths remain optional-free. CPU reference policies never initialize Torch/CUDA.
- Only CPU and the exact qualified CUDA0/UUID are supported; no fallback, new dependencies, cloud or host changes.
- Protected proof held until source/runbook review and concrete phase release; no longer learning/selection authority. Do not rerun Task8 calibration or qualification.
- Canonical execution workspace: `artifacts/development/neural-combat/task-9/`.

## Review focus

1. Wrong CUDA ordinal/UUID or unavailable device fails before native driver/tensors; CPU never queries CUDA (Task1).
2. Actor RNG device and CPU minibatch RNG remain independent; no global RNG seeding (Tasks1/2).
3. Legacy CPU and same-backend resume keep exact states; cross-backend resume fails before optional import (Tasks1/2).
4. Portable warm-start/evaluation load CPU weights into the selected-device model without optimizer/RNG/progress leakage; Adam scalar step staysCPU (Tasks1/2).
5. Post-return errors preserve accepted prefixes and first controls, close once, never retry or claim skipped proof success (Tasks1/2).

### Task 1: Device/config/runtime core

**Files:** Create `src/spire_learn/neural/device.py`, `tests/unit/test_neural_device.py`; modify `src/spire_learn/neural/config.py`, `src/spire_learn/neural/runner.py`, `tests/unit/test_neural_config.py`, `tests/unit/test_neural_runner.py`. Focused CLI tests only if needed; no other source changes.

**Consumes:** `NeuralConfig`, `preflight_checkpoint(path, expected, *, purpose)`, `load_checkpoint(...,device="cpu")`, `build_policy(...,seed,device)`, existing `create_runtime` and `_NeuralTrainingRuntime.save` contracts.

**Produces:** frozen `RuntimeDevicePlan(device,policy_generator_device,shuffle_generator_device,policy_rng_backend,gpu_uuid)`;
`plan_runtime_device(config: NeuralConfig) -> RuntimeDevicePlan`;
`require_runtime_execution_gate(plan: RuntimeDevicePlan) -> None` (stdlib, CPU no-op);
`qualify_runtime_device(torch_module, plan) -> dict[str,object]`;
`validate_resume_device(metadata: Mapping[str,object], plan) -> None`.
Existing `create_runtime` public signature stays unchanged.

- [ ] Write focused pure tests for strict runtime table, accepted CPU/CUDA0 plans, unsupported/missing/wrong/bool UUID, calibration-table isolation, no optional import during help/validation.
- [ ] Run exact focused tests before edits; retain real RED command/stdout/stderr/exit, distinguishing API absence from behavior.
- [ ] Implement defaulted `runtime_gpu_uuid` plus strict `[runtime]` GPU config validation; add lazy device planner/qualification with the exact spec pins. Use inert Torch/inventory boundaries in unit tests, not real optional imports.
- [ ] Test and implement construction order: config-only plan, checkpoint JSON preflight and atomic legacy/new resume metadata compatibility, stdlib execution gate, lazy Torch+device qualification, CPU tensor checkpoint load, then driver/model/generators/optimizer. Invalid config/gate/resume metadata must fail before optional import/factory effects. Keep config/device imports acyclic and test both fresh-process import orders.
- [ ] Test actual runner assembly with strict fakes: selected model/actor generator device, CPU shuffle, CPU loader target, strict weight copies, unchanged state passed to correct generators, no reference-policy Torch, close once with primary control preserved.
- [ ] Add runtime provenance metadata and checkpoint device/backend/UUID fields. Legacy absent device metadata meansCPU; reject malformed/mismatched explicit metadata on resume before optional imports. Keep existing config/identity/counter/recovery gates.
- [ ] Delegate optimizer loading unchanged to Adam; no blanket tensor migration or checkpoint.py edit. Record actual device qualification on runtime/report metadata without exposing it to actor frames.
- [ ] Run focused device/config/runner/CLI suites and fresh optional-import guard; keep exact evidence. Do not run full default suite or optional runtime in the worker first delivery.
- [ ] Freeze owned paths/diff/report at `task-9/gpu-runtime-core/`; parent inspects actual code, independent scoped review, then parent default checks and explicit-path local commit when accepted. Core PASS is not CUDA runtime evidence.

### Task 2: Authored bounded GPU proof and protected runbook

**Files:** Create `configs/neural/standalone-gpu-smoke.toml`, `configs/neural/run-gpu-smoke.toml`, `tests/integration/test_neural_gpu_runtime.py`, `scripts/neural/run-gpu-runtime-proof.sh`, `scripts/neural/gpu_runtime_proof_helpers.py`, `scripts/neural/gpu-runtime-proof-runbook.md`, `tests/unit/test_neural_gpu_runtime_helpers.py`. Add a small fake support module if required, scoped by parent before edit. Do not copy or change old runtime controls.

**Consumes:** accepted Task1 runtime, current train manifest/shared schema, public `train(config,*,runtime,clock)`, `RuntimeOperationLimits`, qualified CUDA venv, CPU weights-only loader.

**Produces initially:** two authored train proof configs with runtime UUID; exactly two opt-in mode tests gated by `SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=1` before optional imports; a fake-tested protected runbook and immutable preparation evidence. A real invocation is a separately released later phase, not part of this first delivery.

- [ ] Derive proof configs from accepted train configs, preserving default PPO/model hashes: standaloneB4/h8, RunB1/h32, per-mode32 learned/32 step slots/4 optimizer attempts. Each config carries a120s ceiling, narrowed by ONE shared absolute deadline started before qualification/config/first-mode work and retained across both modes and checkpoint comparisons. It is never reset per mode. Separate owned output roots; accepted Gym/manifest identities unchanged.
- [ ] Author pure enabled-harness tests with strict fakes and projected imports; prove exact function binding, all CLI/parser argv, source/release/preflight/claim/admin gates, and positive/negative full control flow without Docker/GPU.
- [ ] Author one protected container gate for the two mode tests with the design's cumulative64 steps/69 reset slots/8 optimizer attempts/68 samples/shared120s soft/300s whole bounds. No extra gameplay/evaluation/seed search. Counts and resource controls must gate effects, not merely assert afterward.
- [ ] Require one accepted finite update per mode and changed parameter hashes; record actor/critic hashes honestly. Save final checkpoints; verify same-backend restore and portable weights on retained public frames using inert drivers, with only4 reserved diagnostic sample rows total.
- [ ] For portable CPU checks, use the existing mode-specific CPU train/selection configs explicitly with inert drivers and the new CPU-serialized GPU checkpoint. No held-out case is reset or stepped. A zero-step CPU warm-start checkpoint may provide the reverse CUDA weights-load check; account its bytes in the same output cap and make no CPU-training claim.
- [ ] Require same-config recovery/version/RNG/progress and correct Adam device placement; bad metadata before tensor/native factories. Freeze evaluation/hash checks perform no optimizer update.
- [ ] Reuse read-only source/hash/storage/image primitives from qualified tooling where useful. Keep new controls small and specific, no framework/copies of full historical scripts. One owned RW child, exact helper mounts, non-root offline, actual test PID threads4/interop1, source/lock/native/Git bindings, captured release hash matching durable one-use claim, cleanup by verified full ID only.
- [ ] Record exact RED/GREEN/core/CLI/syntax evidence and freeze at `task-9/gpu-runtime-proof/`; parent and independent source/runbook review before any protected release. A missing API or skipped mode is not a proof pass.
- [ ] After explicit phase release, invoke once. Parent independently verifies closed inventories, counters, tensor/state/device equivalence, cleanup and qualifications; review retained evidence without reruns. Never repeat accepted actions or automatically rerun a failed cohort.

### Task 3: Concrete later experiment package, no execution

**Files:** Draft experiment configs under `configs/neural/experiments/`, a plan under `docs/superpowers/plans/`, and update `docs/neural-combat.md` from actual accepted evidence. Generated checkpoints/logs stay ignored.

**Consumes:** accepted GPU proof and Task8 calibration, approved both-mode architecture and Task9 candidate limits.

- [ ] Resolve fresh independent seed pairs per mode, source/config/manifest/runtime identities, model/PPO defaults, B16h32 standalone and B1h128 Run proposal. Explicitly identify the B16 revision from the oldB<=8 candidate.
- [ ] Bound each proposed learning run by16384 learned decisions/65536 environment step slots/1024 optimizer attempts/600s soft/900s whole; exact checkpoint slots/storage and no automatic extension.
- [ ] Prepare matched round-robin frozen selection/baseline schedule and not-started accounting under the design's shared caps. Preserve final split untouched and never promise full coverage from calibration rates.
- [ ] Validate configs/help without optional imports and record exact identities. Present the concrete reviewed learning/selection package for approval before any longer experiment. No Task3 workload is authorized by this implementation plan.

## Parent self-review

Spec coverage maps device/config/RNG/checkpoints to Task1, actual GPU/lifecycle/
resource evidence to Task2, and later exact experiment identities/approval to
Task3. No serialization/API renaming or second training system is introduced.
Checkpoint deserialization remains CPU-only and legacy CPU files remain valid.
The proof's total64 native step slots/8 optimizer attempts/68 samples are separate
from Task8's948 steps/3 synthetic Adam calls and legacy diagnostic sampling.
Execution method was already chosen by the user; no repeat method selection.
