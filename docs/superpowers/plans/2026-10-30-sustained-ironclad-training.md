# Sustained Ironclad Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved eight-hour, four-replica warm-start campaign with bounded segments, broader training cases, profiling and periodic frozen CPU evaluation.

**Architecture:** Retain the accepted learner and checkpoint contracts. Add a small authored experiment package, bounded profiling/trace instrumentation, and a campaign-specific host/guest scheduler around existing training/evaluation APIs. New segments use one stable guest output path per replica and fresh physical host directories; prior checkpoints are read-only inputs.

**Tech Stack:** Python 3.12 stdlib orchestration/unittest; existing qualified Torch/CUDA, NumPy, Gymnasium and native Gym only inside reviewed containers. No installs.

**Spec:** `docs/superpowers/specs/2026-10-30-sustained-ironclad-training-design.md`, approved by the user's subsequent “I approve”.

## Global Constraints

- In-place on existing main under standing local-commit authority; no worktree, new branch, push, publication, Gym edit, install or host-service stop.
- One source writer at a time: 6.1 Sol/high executor. Controller dispatches Luna/MAX assistants/reviews; workers dispatch no agents.
- Public observations and legal candidates only; preserve masks, feature/gradient meaning, natural death/fault distinction, RNG-001/TIME-001 and existing model/PPO/reward identities.
- Gym `533dad1c8aec86df624b3742b6b3ad1dbf1d0033`; common fingerprint `02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78`; explicit Run `m10-ironclad-policy` v3. Use qualified study-B native/lock/image/CDI pins, never current Gym main.
- Total runtime-stage wall ceiling 28,800 seconds: profile 900, each of four replicas 6,000, evaluation 2,700, shared staging/control/cleanup 1,200. Charge intervals once including cleanup; no sub-budget transfers.
- Per-replica lifetime ceilings: 500,000 learned, 1,000,000 environment attempts, 32,768 optimizer attempts, 256 segments. Per segment: 4,096 environment attempts, 8 standalone or 32 Run updates, whole-run 600 seconds including cleanup, narrowed by remaining budgets.
- Reset cap is environment attempts plus 16 initial slots per admitted segment. Training sample attempts are bounded by each replica's 500,000 learned-row ceiling, reserved before sampling; count diagnostic/fixed-controller randomness separately by its actual kind.
- Profile ceilings: 8,192 environment attempts, 16,384 actor/diagnostic sample rows, 256 disposable optimizer attempts; eight A/B trace episodes, 32 actions each, 64 KiB/decision, 16 MiB total.
- Evaluation: 704 operations, one reset/256 action reservations per operation, 180,224 action reservations total, zero optimizer. Missing/expired operations are explicit unavailable/not_started rows, never losses.
- Pinned RTX5090 UUID `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`; require 12 GiB free before GPU admission, preserve unrelated GPU processes. Four CPU cores, Torch 4/1 threads in actual process, 8 GiB memory/combined swap, 512 PIDs, 8 GiB allocator allowance (not total VRAM guarantee), non-root/network-none/read-only-root and exact qualified CDI exceptions.
- Storage: additional 12 GiB, cumulative 40 GiB, free filesystem floor 30 GiB; no pruning. Never mount Docker config, home, full artifacts, `.git`, trackers or Gym checkout into workloads.
- Existing consumed-ID 65,536, event-log 64 MiB and JSON-node guards stay unchanged. Healthy segments save immutable checkpoints; accepted-effect failures halt the campaign, no retries or substituted parents.
- Preserve foundation report SHA `e7ea42e58871d492d4b5ca2e408f782155526fe80258df1a44bad2cd02285d2a`, root diff SHA `a52df58775d2c5fc0d8f1cbbe1946af787839e5d1379cd752e2663a4557edc29`, old study evidence and `configs/cases/neural/final.json`.
- User design approval covers implementation, internal reviewed stage releases and execution within these limits. Do not ask again for routine hash capture/review. Changes beyond the design remain outside authority.

## Review Focus

1. An all-fixed-controller Run rollout must hit the segment environment cap before another native action (Task 1).
2. A resumed checkpoint with nested immutable metadata, fresh physical output and advancing clock must preserve exact saved prefixes without claiming later lifecycle time (Tasks 1 and 3).
3. Failed or malformed Docker create cannot establish ownership or remove a pre-existing container; deadline/control errors retain primary cause (Task 3).
4. Optional runtime imports and class-bound APIs must work in the actual projected container tree; references remain Torch-free (Tasks 2 and 3).
5. Partial evaluation, unavailable checkpoints, actual sparse ledgers and budget exhaustion must produce honest reconciled rows without repeating native effects (Task 3).

## Files and responsibilities

- `scripts/neural/sustained_package.py`: authored package creation/validation and deterministic evaluation schedule; stdlib only.
- `configs/neural/experiments/sustained-v1/`: campaign JSON, four fixed training TOMLs and single-case CPU evaluation TOMLs.
- `configs/cases/neural/sustained-v1/{train,selection}.json`: new catalog/panel; no edits to old splits.
- `src/spire_learn/neural/runner.py`: narrow optional cumulative operational environment ceiling only.
- `scripts/neural/sustained_profile.py`: lazy profiling, bounded semantic traces and catalog validation, using real accepted learner APIs.
- `scripts/neural/sustained_runtime.py`: lazy guest execution of one reviewed profile/segment/evaluation job and receipt publication.
- `scripts/neural/sustained_controls.py`: source/release verification, budget scheduler, exact containment/ownership/storage controls and evidence reconciliation.
- `scripts/neural/run-sustained-training.sh`, `scripts/neural/sustained-training-runbook.md`: whole-run watchdog and approved stage invocations.
- `scripts/neural/task9_nvidia_cdi.py`: additive runtime-class entry point only if needed to reuse exact CDI validation without pretending new jobs are old Task9 phases.
- Matching `tests/unit/test_neural_sustained_{package,profile,runtime,controls}.py` plus affected runner/CDI tests.
- Ignored execution ledger/reports under `artifacts/development/neural-combat/sustained-v1/`; implementation/review packages also in this plan's SDD workspace.

### Task 1: Fixed experiment package and segment admission

**Files:** Create package/config/case paths above and `tests/unit/test_neural_sustained_package.py`; modify `runner.py`, `tests/unit/test_neural_runner.py`.

**Interfaces:**
- Consume `load_neural_config`, `load_neural_cases`, `RuntimeOperationLimits`, checkpoint compatibility identities and the six verified parent inventories in `sustained-v1/preparation/parent-checkpoints.json`.
- Produce `author_campaign(root: Path) -> dict`, `validate_campaign(root: Path, campaign_path: Path) -> dict`, and `evaluation_schedule(campaign: Mapping[str, object], checkpoints: Mapping[str, object]) -> tuple[dict, ...]` in `sustained_package.py`.
- Add `RuntimeOperationLimits.max_environment_actions: int | None = None`, a validated absolute cumulative attempt high-water mark, without changing default behavior. Effective local collect ceiling subtracts restored prefix and respects the tighter configured lifetime ceiling.
- Campaign policies ordered `standalone-1`, `run-1`, `standalone-2`, `run-2`. Parent Standalone B/Run B respectively. Fresh policy/optimizer/curriculum/noncombat quartets: `(1101103,1102207,1103301,1104409)`, `(1301103,1302207,1303301,1304409)`, `(1201103,1202207,1203301,1204409)`, `(1401103,1402207,1403301,1404409)`.

- [ ] **Step 1: Add failing focused tests.** Assert all four exact replica budgets/parents/seeds and unchanged model/PPO hashes; 78 unique standalone training entries plus one normal Run entry; 64 selection cases and a 16-case sentry subset; no train/selection/final seed or lineage collision; expected 704 operations including explicit unavailable rows. Assert fixed-only Run with recovered prefix 100 and operational ceiling 103 forwards exactly three more actions, a zero-remaining ceiling forwards zero, and a narrower config ceiling wins.
- [ ] **Step 2: Capture RED.** Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_sustained_package tests.unit.test_neural_runner -v`; retain exact command/stdout/stderr/exit before implementation. Failures must demonstrate new API/behavior absence, not a broken loader.
- [ ] **Step 3: Author/validate fixed inputs and narrow the existing admission path.** Preserve raw full-config hashing. Logical output roots are fixed per replica under `/learner/artifacts/development/neural-combat/experiments/sustained-v1/training/<policy>`. Use new case IDs/lineages. Training uses the spec's 13 encounters × existing three decks × HP64/80, plus normal Run. Selection seeds are `2**40 + 2**20 + index` for indices 0–63; first32 standalone, last32 Run. Familiar16 cycle across all13 encounters at starter64; deck8 use encounter indices `(0,2,4,6,8,10,11,12)` with starter+two base Shrug It Off at64; HP8 use indices `(1,3,5,7,9,10,11,12)` alternating56/72 with starter. Encounter order is the spec's order. Sentry standalone indices `(0,4,8,12,16,20,24,28)`, Run indices32–39. Copy exact runtime pins from retained qualified records and verify parent hashes. Package state is prepared, not an execution release.
- [ ] **Step 4: GREEN and review package.** Run the focused command, config validate-only for four train and 64 single-case eval configs, and a fresh six-prefix import guard. Test actual parser round-trips rather than duplicated expected JSON only. Self-review, freeze owned paths, commit only owned source/config/tests, and report results. No optional runtime.

### Task 2: Bounded profiling and public diagnostic producer

**Files:** Create `sustained_profile.py`, `tests/unit/test_neural_sustained_profile.py`; optional small profiling hooks only in `runner.py`, `features.py`, `model.py` if scoped wrappers cannot measure the required boundary. Do not optimize yet.

**Interfaces:**
- Consume validated campaign from Task1, real `create_runtime`, `train`, `evaluate_policy`, checkpoint preflight/load, existing schema/feature/choice APIs.
- Produce `run_profile(root: Path, campaign: Mapping[str, object], context: Mapping[str, object], *, dependencies: Mapping[str, object] | None = None, clock: Callable[[], float] = time.monotonic) -> dict`.
- Produce `public_decision_trace(schema, frame, choice, *, legal_scores, successor, result) -> dict`, a bounded semantic projection without raw tensor objects/private metadata.
- Dependency injection names: `factory`, `train`, `evaluate`, `clock`, `memory`, `synchronize`; defaults resolve lazily on execution. Context supplies source/phase identity, deadline and effect ledger; it cannot widen campaign budgets.

- [ ] **Step 1: Add failing producer tests.** Drive actual wrappers with strict fake policy/driver/optimizer/checkpoint and advancing clock. Assert timing inclusivity/synchronization labels; unique frame-byte accounting avoids double-counting shared successor/source; all78 catalog entries validated once; eight trace episodes selected exactly from old cases00/04/01/03 and A/B; 32-action/64-KiB/16-MiB bounds; unknown/padded candidates never become known/public features; sample/optimizer reservations precede calls; close/control exceptions preserve first cause. Assert imports are lazy and projected package aliases are callable without bound-self errors.
- [ ] **Step 2: Capture RED.** Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_sustained_profile -v`, retain literal failure evidence.
- [ ] **Step 3: Implement instrumentation.** Profile actual training collection/update (one bounded pass each mode, disposable weights), with timing paths covering native/reset, snapshot/validation, features/transfers, act/bootstrap, PPO forward/backward/Adam, events/checkpoint/finalization. Separate CPU clocks, synchronized GPU measurements and inclusive totals. Measure RSS/cgroup and Torch allocated/reserved peaks. Trace only bounded decoded public features/legal candidate scores. Never seed campaign models from diagnostic updates. Reuse existing sampler/effect accounting and normal fault semantics; do not build a second collector or simulator.
- [ ] **Step 4: GREEN and static qualification.** Run focused tests and optional-import isolation, compare producer output through its strict validator, prepare a one-shot profile inventory (all work including optional tests fits900s and effect caps). Self-review/freeze/commit owned source/tests and report. Parent and independent review must precede all Torch/GPU/native invocation.

### Task 3: Real projected guest execution and campaign controls

**Files:** Create `sustained_runtime.py`, `sustained_controls.py`, wrapper/runbook, runtime/control unit tests; additive CDI runtime-class API/tests if necessary. No changes to old study claims/configs or frozen pins.

**Interfaces:**
- `run_segment(root: Path, campaign: Mapping[str, object], context: Mapping[str, object], *, dependencies=None, clock=time.monotonic) -> dict`: real warm_start or resume followed by accepted `train`; publishes actual reports, checkpoint inventory and effect/accounting receipt.
- `run_evaluation(root: Path, campaign: Mapping[str, object], context: Mapping[str, object], *, dependencies=None, clock=time.monotonic) -> dict`: one immutable CPU batch using actual checkpoint hashes and Task1 schedule, references in fresh import-guarded children.
- `run_campaign(root: Path, *, campaign_path: Path, source_snapshot: Path, release_path: Path, stage: str, dependencies=None) -> dict`, where stage is `profile` or `train-evaluate`. Returns a closed campaign receipt; does not mint its own authority. CLI `capture-source`, `validate`, `run` expose these controls without import side effects.
- Parent creates reviewed snapshot/release from existing user approval; releases bind exact source/package/pins/budgets/review and consumed predecessor. Each job claim is exclusive and never reused. Profiling and sustained stage share cumulative resource accounting.

- [ ] **Step 1: Add failing integrated fake-boundary tests.** Run complete projected `_guest entry → factory → real trainer API facade → checkpoint inventory → host receipt validator` for both warm-start and two resumed segments, using actual nested immutable loaded metadata. Change physical output roots while raw config bytes/hash and logical guest root remain equal; reject stale/tampered prefix and missing parent. Advance save/close clocks: checkpoint-prefix equality exact, later outer time separately charged once. Test all-fixed Run cap through real runner admission. Drive real ledger increments into projection/strict validator, materialize defined zeros and reconcile internal delivery counters. No hand-authored success booleans.
- [ ] **Step 2: Add containment/deadline/failure RED tests.** Exact projected guest imports/scripts, no host tree fallback; strict create argv/inspect table; qualified CDI runtime profile for truthful new phase IDs and CPU no-GPU path; existing-name/timeout/malformed-create output produce no owned removal; only full created ID plus reinspection permits once-only removal. Inject TERM during cleanup, process-control errors in setup/write/close, wall exhaustion before first action, optimizer uncertainty, save failure, storage admission failure and evaluation child timeout. Assert primary preserved, all later jobs not_started and no retries. Test actual runtime versions primitive-normalized and installed module/thread evidence collected in the executing process.
- [ ] **Step 3: Implement orchestration.** Reuse low-level bounded subprocess/source inventory/exclusive JSON/image/GPU/CDI utilities; no copying the old entire controller. Generic reusable helpers may be factored additively with backward tests. Stage read-only source/config/helper scopes once. Each segment receives one new RW output bind at fixed logical config root plus exact RO parent checkpoint, runtime, lock and acyclic control input. Write context before create; fresh empty Docker client config stays outside mounts. Validate package/source/release before any claim or workload. Names include bounded campaign/job identities, labels bind raw hashes. Exact absence check precedes create. Inspect immutable image, command, environment, mounts, resources and lifecycle; guest validates actual post-OCI CDI facts and emits bounded retained admission evidence. Never label a new GPU job as an old Task9 phase to pass validation.
- [ ] **Step 4: Implement bounded scheduling/accounting.** Round-robin replicas; per-job high-water marks are recovered prefix+remaining segment delta. Profile jobs debit900s and reserved effect caps. Segment whole deadline ≤600s and narrowed by replica/global remaining; normal operation stops early enough for reserved save/close/remove/postcheck time. Charge replica outer intervals, eval active batches and shared intervals once; controller guard enforces total28,800s. Stop admission before nonpositive remaining allowance or storage/VRAM shortfall. Retain actual counters instead of assumed caps. New stage cannot replay profile or consumed segment jobs. Evaluate parent/reference panel first and milestone/final batches when actual checkpoints become available. Run CPU batches with fresh deadlines from remaining2,700s; full704 schedule remains fixed with honest unavailable/not_started outputs. User-selected weights never overwritten. No automatic checkpoint selection or hyperparameter changes from scores.
- [ ] **Step 5: GREEN, full discovery and independent review package.** Run focused package/profile/runtime/control/runner/CDI tests. Run shell syntax and actual Docker `create --help` parser proof only (no lifecycle). Fresh guard blocks all six optional prefixes during help/validate/discovery. Run the full default suite once with all opt-in flags unset. Freeze complete source/config/pin inventories, record exact commands/exits, self-review, commit only owned files and hand off. No workload before parent review and internal release.

### Task 4: Reviewed qualification, measured choice and campaign execution

**Files:** Ignored reports/releases/receipts under `artifacts/development/neural-combat/sustained-v1/`; after final postchecks update `docs/neural-study.md`, `docs/neural-combat.md`, README and runbook. At most one profile-supported feature/preparation optimization with focused tests if measured necessary and achievable within remaining profile allowance.

**Interfaces:** Consume Task1–3 frozen code, independent source/control review and user-approved design. Parent owns stage releases and real invocation. Worker authority to edit/test does not authorize self-release or runtime retries.

- [ ] **Step 1: Parent synthesis and independent review.** Verify spec coverage, diff and retained focused/default evidence. Resolve scoped findings with same worker/reviewer; no repeated broad task replays. Verify parent payload/native/lock/CDI/source pins and protected file hashes. Capture exact approved design, fixed manifest and executable snapshots; write a bounded one-use profile release. Budget accounting starts on first owned runtime interval, not on source work.
- [ ] **Step 2: Run profiling once under reviewed controls.** Confirm free VRAM/unrelated processes and storage. Execute profile stage, capture streams/status/claims and actual guest identity, verify outputs/cleanup/inventory. A fault halts; no replacement run. If qualification incomplete within900s, stop and report. Independently review retained qualification, no workload rerun by reviewer.
- [ ] **Step 3: Select source.** If measurements justify one batch-local encoding/preparation optimization, use TDD, exact masks/owned-storage/gradient/same-backend-choice checks and paired bounded timing from remaining profile budget. Reject a non-gaining candidate in favor of instrumented reference only if both runs are healthy; an accepted-effect fault halts. Freeze/review selected source and complete config before first sustained segment. No source changes between segments.
- [ ] **Step 4: Execute the sustained stage.** Parent release binds qualification/source and remaining budgets. Invoke once, monitor concise periodic status while active, and let round-robin checkpointed training/CPU evaluation finish within caps. Do not stop/restart to adjust policy scores. If headroom fails before effects, report blocked rather than stopping host services. Preserve every effect prefix and original claim on failure. No automatic native retries.
- [ ] **Step 5: Audit results and finish.** Verify actual outer budgets, action/reset/sample/optimizer accounting, all inventories/checkpoint parent chains, source and unrelated-process postchecks, and owned-container absence. Independent retained-evidence review; no reruns. Report learning curves/paired strata and Run outcomes, actual timing/memory/throughput, availability/cutoffs/faults and explicit limitations. Update prose only after source postchecks; retain executed snapshots. Commit local owned docs/source only, never outputs/user files. No push or publication.

## Plan self-review and execution rulings

- Each design section maps to Task1 package/admission, Task2 measurement/trace, Task3 orchestration or Task4 reviewed execution/reporting. No monolithic-training refactor is required.
- Task1's cumulative ceiling, Task2's profile entry and Task3's real guest factories are the explicit shared interfaces. Any typed helper introduced internally is defined in its owning task, not assumed across tasks.
- Review Focus items have their tests in their owning tasks above. Existing training budget semantics remain checkpoint-boundary semantics, with outer campaign accounting independent.
- Ruling: The user's approval explicitly includes implementation/internal reviewed releases; the plan is published for visibility and execution proceeds without another plan confirmation. The method is already approved. This overrides the writing-plans default handoff question.
- Ruling: Remain in the current checkout and retain all plan/runtime artifacts. Do not create/delete a worktree or delete the SDD workspace; this follows the user's standing workspace/preservation preference.
