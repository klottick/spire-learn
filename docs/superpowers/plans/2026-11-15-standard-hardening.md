# Standard Adapter Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the non-blocking follow-ups from the PR #1 and PR #2 reviews in Learn's Standard Gym adapters.

**Architecture:** Local changes in `src/spire_learn/neural/envs.py`, plus tests. No contract, schema, feature, model or config identity changes.

**Spec:** the review notes recorded on PR #1 and PR #2, and `docs/superpowers/specs/2026-11-11-gym-v1-standard-design.md` (decision 7, privacy).

## Global Constraints

- Native Gym error text must never reach `DriverFaultEvidence.message`, logs or the exception chain on Standard paths (`standard-standalone` and `standard-run`). Only the error class name and per-slot statuses may be kept.
- Legacy `standalone` and `run` behaviour stays unchanged, including their existing error messages.
- Faults remain driver faults. They are never outcomes, and accepted effects are never retried.
- No installs, training, native, Gym or Docker runs, pushes or history rewrites by implementers.

## Review Focus

1. A Standard native error carrying a sensitive message in `str(error)`, `args`, or an attribute must not appear in evidence, `repr`, `__cause__` or `__context__`.
2. Recoverable input and state errors on Standard paths stay recoverable and keep the driver usable.

---

### Task 1: Standard error redaction and adapter tidy-ups

**Files:**
- Modify: `src/spire_learn/neural/envs.py`
- Test: `tests/unit/test_neural_envs_standard.py`, `tests/unit/test_neural_envs_standard_run.py`

**Interfaces:**
- **Standard combat provider.** `_StandardCombatProvider` raises its redacted `_StandardNativeFault` with no `__cause__` and no `__context__`. Build it outside the `except` block, for example by capturing the error, leaving the block, then raising.
- **Standard Run provider.** `_RunProvider` gains a keyword `redact_native_errors: bool = False`. `_make_standard_run_driver` passes `True`. When set, Run fault errors are redacted the same way. Input and state errors are classified as today. Legacy Run construction is unchanged.
- **Reset annotation.** `_StandardCombatProvider.reset` returns `tuple[_RawBoundary, ...]`, because it always returns every slot.
- **Single info read.** `_StandardCombatProvider.step` reads each slot's info once: use `_settle_standard_terminal`'s raw row instead of calling `_standard_info_slot` twice.
- **Typed case helper.** `_standard_run_case` returns `NeuralCase`, removing the `# type: ignore[attr-defined]` comments. Creation is thawed once.

- [ ] **Step 1: Write the failing tests.**
  - `test_standard_fault_has_no_native_context`: a fake `BatchFaultError` whose message, args and an attribute contain `"SECRET"`. The resulting `DriverFault` evidence, its `repr`, and the full `__cause__`/`__context__` chain (walk it) contain no `"SECRET"`. The class name is still reported.
  - `test_standard_run_fault_is_redacted`: the same check through the `standard-run` driver.
  - `test_legacy_run_error_message_unchanged`: the legacy Run fault message still includes the native text, as today.
  - `test_standard_run_input_error_recoverable`: the error is re-raised and the driver is not frozen.
- [ ] **Step 2: Implement, then run the tests.** Run `tests/unit/test_neural_envs*.py` and the neural, gym and integration sweep (one discover per file, excluding `*continuation*` and `test_neural_curriculum_controls.py`).
- [ ] **Step 3: Commit.** Message: `fix: redact Standard native errors and tidy Standard adapters`.
