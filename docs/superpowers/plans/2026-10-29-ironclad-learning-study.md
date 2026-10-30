# Bounded Ironclad learning study implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute source preparation task by task; no experiment execution.

**Goal:** Produce a reviewed, executable four-run/192-selection approval package.
**Architecture:** Existing training/evaluation APIs, exact one-case inputs and a small outer schedule controller. New protected controls bind future checkpoints and reserve effects before dispatch; accepted proof controls stay historical.
**Tech Stack:** Python3.12 standard library for preparation/tests, existing qualified PyTorch/Gym only in a later approved protected runtime.
**Spec:** docs/superpowers/specs/2026-10-29-ironclad-learning-study.md

## Global constraints

- Source preparation only; no longer training, selection, installs, GPU/native/Docker lifecycle or Gym writes.
- One sole source writer per task, parent synthesis and fresh independent review; no nested agents. Luna assistants use MAX; the Task C integration senior uses 6.1 Sol/high.
- Work on existing main in place per standing user choice; canonical ledger/artifacts under artifacts/development/neural-combat; no worktree or .superpowers workspace.
- Preserve original foundation/task-6-report.md and root task-6-core-review.diff and all old releases/claims/results. Parent owns explicit staging/commit after acceptance.
- Exact caps/pins, fresh seeds, schedule, output slots and qualifications come from the spec; final split unchanged and unexecuted.

## Review focus

1. Altered/reordered configs or lineage overlap must fail package validation before effects.
2. Missing checkpoint, partial run, or an advancing shared clock must leave honest not_started rows and no retry.
3. Real immutable LoadedCheckpoint metadata and sparse counters must cross result serialization without weakening strict JSON.
4. Actual guest filesystem/argv/process imports must match protected inspection, including Torch-free reference children.
5. Failed create, timeout, write/close/control exception must preserve primary cause and never remove unowned containers or lose reserved-effect evidence.

### Task A: Exact authored study and pure validator

**Files:** Create configs/neural/experiments/task9-v1/ (study.json,4 train TOML,32 selection TOML); configs/cases/neural/task9-selection*.json (parent+32 one-case); scripts/neural/task9_study_package.py; tests/unit/test_neural_study_package.py.
**Consumes:** actual load_neural_config/load_neural_cases and canonical train/selection/final templates.
**Produces:** validate_study(root: Path, study_path: Path)->dict[str,object], build_round_robin(cases: list[dict], policies: list[str])->tuple[dict,...], validation/description-only CLI. Validation returns parsed bound study facts; no runtime dispatch API in this task.

- [x] Write focused failing tests for absent package/API, exact4/32/192, all caps/defaults/seed/identity hashes, round-robin ordering and no duplicate/missing pairs.
- [x] Run stdlib unittest focused module; retain exact RED command/stdout/stderr/exit.
- [x] Author package and minimal stdlib validator. Bind canonical and raw input hashes separately. Reject config/path/hash tampering, inexact one-case membership, lineage leakage and policy substitutions. Validate actual known/heldout catalog labels.
- [x] Run focused module GREEN, fresh optional-import guard plus validate-only CLI, and artifact-free temporary source/config copy validation. Preserve final manifest bytes. Report hashes/diff/evidence.
- [x] Parent source synthesis and fresh independent SPEC/QUALITY review. Resolve scoped findings before B. Parent commits only after final combined acceptance unless needed for a gate.

### Task B: Schedule orchestration and shared evaluator deadline

**Files:** Modify src/spire_learn/neural/runner.py narrowly and tests/unit/test_neural_runner.py; create scripts/neural/task9_study_runtime.py and tests/unit/test_neural_study_runtime.py.
**Consumes:** Task A validated study, actual create_runtime/train/evaluate_policy and RuntimeOperationLimits, exact checkpoint storage schema. Parent will pin complete runtime-facing signatures in the task brief from accepted A output before dispatch.
**Produces:** one training-operation entry and one shared selection-schedule entry, lazy CLI for protected children, durable exclusive receipts and checkpoint binding records. No automatic combined execution from validation CLI.

- [x] RED tests show evaluator ignores an advancing absolute deadline; apply only its existing deadline seam and preserve cleanup/control precedence.
- [x] Write full actual orchestration fake-flow tests using real config objects, immutable checkpoint metadata, sparse counters and strict receipts:4 training records/final bindings and all192 schedule rows, shared deadline prefix, missing policy, primary/control cleanup faults, duplicate claims, no-effect exhaustion and Torch-free refs.
- [x] Implement minimal lazy outer orchestration using accepted trainer/evaluator; no replacement collection logic or new optimizer/checkpoint format. Reserve before dispatch, freeze checkpoints, no retry, validate actual receipts and update evidence.
- [x] Run focused GREEN and clean source-copy/full-flow/import guards; record exactly what uses fakes. Parent synthesis and fresh review before C.

### Task C: Protected controls, runbook and approval packet

**Files:** Create scripts/neural/run-task9-study.sh, scripts/neural/task9_study_controls.py, scripts/neural/task9-study-runbook.md and tests/unit/test_neural_study_controls.py; update docs/neural-combat.md and create docs/neural-study.md; parent prepares ignored source snapshot and unsigned approval proposal.
**Consumes:** validated A identities and B actual guest CLI/receipt protocol; existing qualified runtime inputs and generic proven hash/inventory patterns.
**Produces:** one-use campaign plus five phase releases, exact guest mount/argv checks, whole-run watchdog and ownership-safe cleanup, bounded source/storage/process logs, actual inventory verification and an explicit approval-to-execution binding.

- [x] RED fake-boundary tests exercise the complete projected guest flow and actual helper CLI, all5 phase identities, exact source/package/release claims, current checkpoint bindings, phase/shared deadlines and output admission.
- [x] Implement small phase controls, no proof rearming. Reject altered mounts/commands/device controls and nonlocal daemon; allow no implicit service stop or install.
- [x] Prove creation failure/collision/timeout never owns an existing container; successful fullID cleanup once; primary exceptions and final evidence survive cleanup and hard timeout paths.
- [x] GREEN focused tests, shell syntax, AST, fresh optional-import/config validation, clean source-copy tests and appropriate one final default discovery. No protected runtime.
- [x] Parent actual source synthesis, fresh independent combined SPEC/QUALITY/static-readiness review; prepare exact immutable snapshot, identities and budget/storage disclosure. Only then present explicit separate experiment approval; no live release or claim is created during preparation.

## Preflight and self-review

TaskA->B: pure study schema/digests/schedule consumed exactly; B brief will freeze returned keys after A review.
TaskA->C: raw/canonical hashes and relative paths remain distinct; source snapshot excludes generated outcomes.
TaskB->C: real guest argv and output protocol consumed, tested in projected filesystem before release.
TaskA self-check:32 case configs reused across6 policies; no fictional checkpoint hashes.
TaskB self-check: one-case evaluation+shared deadline; production train unchanged.
TaskC self-check: five bounded phases share campaign claim; no actual experiment in source task.
No contradiction with approved GPU plan: additional details refine its source-preparation task. Main/in-place/no-extra-workspace and Luna/MAX preferences supersede generic skill workflow choices. All decisions are reversible source work; no renewed design permission is needed to prepare the concrete approval package.
