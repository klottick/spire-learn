# Ten-hour Ironclad Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the accepted broad curriculum then launch and audit a fresh10hour-bounded standalone training campaign under the user’s explicit approval.
**Architecture:** New immutable campaign package and lazy guest runtime reuse accepted learner APIs; a new host controller reuses existing generic protected Docker/CDI/effect primitives. Existing old controllers and learner behavior remain unchanged.
**Tech Stack:** Python3.12 stdlib control code, qualified offline CUDA runtime, existing unittest/fake patterns. No installs.
**Spec:** docs/superpowers/specs/2026-11-01-ten-hour-ironclad-curriculum-design.md

## Global Constraints

- User approved qualification and approximately10hours training. Fresh campaigncurriculum-20261101-a,36,000second whole runtime cap; original old claims consumed. No repeated approval for routine reviewed work within this scope.
- Existing main checkout, one source writer at a time; parent dispatches sequential Sol6.1/high executors and independent LunaMAX reviewers; workers dispatch no agents, scoped local commits/no push. Preserve all SDD/evidence.
- Protect foundation task6report SHAe7ea42e58871d492d4b5ca2e408f782155526fe80258df1a44bad2cd02285d2a; rootdiff SHAa52df58775d2c5fc0d8f1cbbe1946af787839e5d1379cd752e2663a4557edc29; finalcases SHA5b300377431bf16821799bdbd0a706ed634d3c568323a8a2e2ac1c59e58572b9. Never stage them.
- No ML/native/GPU effects in source tasks. Runtime only parent release after reviews. No installs/Gym/tracker/private sources or actor secrets. Keep accepted feature src/tests/configs unchanged; new controller scripts/data only.
- Source task outputs live in new curriculum-campaign-v1 domain. Actual effects reserve before forwarding, faults stop/no retry, healthy lifecycle+checkpoint gates next.
- Qualification exact512actions/512samples/8Adam/256resets/120soft/600whole/1GiB; campaign global1.2Mactions/1.1Msamples/70kAdam/1.1Mresets/36,000whole/12GiBnew/40GiBaggregate/30GiBdiskfree/12GiBVRAMfree.
- Same originalBweights, fresh state and paired1103/2207/3301/4409 for allarms; matched16,384actions/1,024Adam/3,000s max perarm, then longmixed. No score-based policy switching; final split untouched, Run excluded.

## Review Focus

1. Long-run budget edits retain stale config hashes: emit real TOML and actual load/stream validation across at least two shards before effects.
2. Global10hour accounting misses setup/close/cleanup or doubles recovered prefixes: advancing-clock fake complete flow and interval reconciliation.
3. New phase/helper/mount paths differ in guest: full projected read-only layout/import/admission test with real PhaseSpec/runtime_class and exact command.
4. A failed create or interrupt permits removal of an unrelated resource: fake ownership ambiguity/collision/repeatedsignal/close tests with fullID-only cleanup.
5. Evaluation or censored/faulted fights become invented wins/losses or influence sampler: real one-case reports/summary with unavailable rows, zero optimizer, immutable complete panel hash.

### Task 1: Immutable package and lazy guest operations

**Files:** Create scripts/neural/curriculum_package.py, scripts/neural/curriculum_runtime.py, tests/unit/test_neural_curriculum_package.py, tests/unit/test_neural_curriculum_runtime.py; authored campaign data at configs/neural/experiments/curriculum-v1/campaign.json.
**Interfaces:** `author_package(root: Path, destination: Path, *, campaign_id: str) -> Mapping` writes exclusive destination/package.json and inputs; `validate_package(root: Path, package_path: Path) -> Mapping` takes that package.json file; guest `run_qualification(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping`, `run_segment(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping`, `run_panel(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping`. Small `_guest` entry consumes exact reviewed context. Dependencies are explicit plain callables at actual runtime boundary; default imports lazy.
- [ ] RED focused tests for actual config/package data, real projected guest flow and wrong identities/limits; record literal outputs.
- [ ] Author4fixed/4randomized/256mixed exactshards and common trainingTOML per spec,≤256MiBpackage. Validate four-shard cycles/actual loaders. Common originalB inventory,64case panel,64actual one-case CPUconfigs, fixed8evaluation slots. No stale dataclass replacement.
- [ ] Implement qualification adapter invoking actual held unittest method/class once with hostdeadline narrowing and sharedcaps; archive actual attempt counters. Reuse unchanged base learner/gatedsource.
- [ ] Guest segment uses real run_stream_segment and accepted factory/train/state/receipt APIs, actual lifecycle+preflight+prefix reconciliation and attempt ledgers. Perjob8updates4096actions/256Adam, cumulative high-water marks; exact healthy current parent/successor.
- [ ] CPU panel performs real one-case evaluate_policy calls with deterministic checkpoint, no optimizer/sample effects, actual NeuralReport objects for existing summarizer. Missing/censored/error states truthful; preserve primary control and close once on every setup/assert/serialize path.
- [ ] Whole actual producer flow under strict fakes: qualification→warmstart→save→nextshard→panel→close; staleconfig/4RNG/Adam/model/recovery/caps/latecontrol/importlayout failures before effects. Focused tests/importguard, selfreview/report/local scopedcommit. No full default or actual runtime yet.

### Task 2: Protected host controller, launcher and monitoring protocol

**Files:** Create scripts/neural/curriculum_controls.py, scripts/neural/run-curriculum-campaign.sh, scripts/neural/curriculum-campaign-runbook.md, tests/unit/test_neural_curriculum_controls.py. Modify Task1newfiles narrowly only if a concrete producer contract requires it. Modify scripts/neural/task9_study_controls.py only for appended PhaseSpec.pids_limit=512 and corresponding build_create_argv/_inspect_projection fields; primitive128|512 only, bool rejected, legacy512default unchanged. Existing learner/generator/qualification modules untouched. SharedPIDhelperdelta is separately source-bound; old snapshots and defaults stay intact.
**Interfaces:** `capture_source(root: Path, destination: Path, *, learner_head: str, dirty: Mapping) -> Mapping`; `run_campaign(root: Path, *, package_path: Path, source_snapshot: Path, release_path: Path, dependencies=None, clock=time.monotonic) -> Mapping`; pure validation/capture CLI and exact `_guest`/contextcommand. Release schema exact new single-use reviewedsource/package/userapproval/pins/caps/campaignID, no olddomain adaptation.
- [ ] RED fake source/release/package/deadline validation and collision/ambiguous ownership tests; then implement minimal closed controller. Reuse actual generic PhaseSpec(runtime_class,pids_limit), execute_job transport+signal_scope, protectedCDI/storage/staging primitives without invoking old run_campaign. If using legacy admission marker token, document it as reused transport only; new truthful job IDs/contextschema/labels always explicit.
- [ ] Source/package staging and guestcontrolcontext match actual projected mounts; UID1000,offline,ROroot,4CPU/8GiBnoswap,qualified exactCDIprofile,512PID campaign/qualification128PID. Empty owned Dockerclientcfg only. Guest thread4/1 verified in actual process; no phasealiases or implicit reset/step during setup.
- [ ] Schedule single qualification gate, baseline64 CPUcases,4matchedsegmentsperarm/max3,000s, longmixed≤256segments untilglobal33,600s cutoff,8preregisteredpanelsmax512cases/eval1,800s. Global36kdeadline and hard effects/storage floors; reserveactualsetup/serialization/close/cleanup. No metric-driven branch promotion. Preserve availability matrix and true unmatched prefixes.
- [ ] Append-only fresh one-use claims and source/receipt/checkpoint chain, attempt reservations beforestart/effects. Nameabsence/fullsuccessfulIDownership/reinspect/fullIDonlyremoveonce. Parentlauncher result atactualexit, independent outertimeout TERM at35,910+90graceKILL, cleanup signaldeferral. No unrelatedprocess control or runtime retries.
- [ ] Pure actual projected complete controller→guest→producer→host-reconcile flow with injected boundaries, advancing clocks/caps/storage, interruptedpreflight/close/failedpublication/collision/dangling qualified venvsymlinks. No hand-created passing receipts. Fresh blocked6prefixhelp/validate/discovery; focused suites then onefull default final combined tree withallSPIRE_LEARN_RUN_*unset. Selfreview/report/scopedcommit; parent reviews source and exact release before runtime.

### Task 3: Parent qualification, sustained launch and retained audit

**Files:** New owned records under artifacts/development/neural-combat/curriculum-campaign-v1/; postrun owned runbook/result docs only after closure.
- [ ] Parent verifies final source/pins/userapproval/package/protectedfiles/headroom/storage, reviews prepared source/release manifests, publishes new approval/claim only after source review. No additional user prompt within approved exactscope.
- [ ] Launch one enclosing36k runtime wrapper with one-use qualification then campaign. Record PID/execsession/deadline and immediate-exit-result writer; do not return a merely proposed plan. Qual failure stops/no native retry.
- [ ] Set10minute retained-progress heartbeat viaPaseo; no tests/imports/GPU/Docker/source edits/restarts by heartbeat, controlleraloneownsbudget+cleanup. Preserve unrelated PID3772.
- [ ] Atactualexit audit hashes/pins/fullcheckpoint chain/actual elapsed/effects/inventory/cleanup and independent retained review, distinguish unavailable evaluations and scoped learning outcomes. Deleteheartbeat; owned docs/localcommit afterpostchecks, no push.

## Parent self-review

Operational scope follows user-approved curriculum plus new explicit10h authorization, not a new model/reward architecture. All requirements owned by package/guest, hostcontrols or parentactualrun. Signatures are shared across two source tasks and exactbudget data, with no placeholder authority. Existing generic runtime_class andexecute_job ownership primitives avoid another broad container framework. Two prior accepted feature Minors remain unchanged; source-only work never counts as native compatibility or learning.
