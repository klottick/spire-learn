# Task 6 implementation report — core only

**Status:** core implemented and reviewed; actual runtime `BLOCKED`. Task 6 is **not complete**. Stopped at approved-plan Step 5 authority gate; Steps 6–7 were not run and no integration test file was added.

## Scope delivered

- Corrected array-like shape validation to require exact nested `tolist()` rank/shape; immutable `ArrayValue` remains flat but is revalidated against dtype-specific integer ranges, float finiteness and range. `FakeArray.tolist()` now mirrors declared shape.
- Round 2: float dtypes accept only finite built-in `float` values that round-trip losslessly through stdlib `struct` formats for their declared width; no integer acceptance or silent rounding. `make_single` is annotated with `CaseExecution`; construction/reset responsibilities remain unchanged.
- Corrected metadata lookup to already-loaded `spire_lab._native`; build info is recursively copied/frozen. With no injected classifier types, exact documented exception exports are resolved only from that already-loaded module; missing/malformed exports and arbitrary `RuntimeError` remain unmapped.
- Added six regression tests over two rounds (13 core tests total). Test-only fixture descriptor and generic `FakeArray` are from `tests/support.py`.
- RED evidence for regressions: focused run before implementation failed shape, dtype, loaded-export classification, and build-info cases (exit 1; 6 failures, 2 errors); corrected assertions and implementation then passed.
- Round 2 RED: `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_native_contract.NativeContractTests.test_float_arrays_require_exact_representable_float_primitives -v` — expected failures for Python float32 `0.1`, integer `1`, underflow `1e-50`, and immutable float32 `0.1` (4 failures, exit 1). After fix, focused core passed.

## Verification

- `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_native_contract -v` — 13 tests, PASS, exit 0.
- `PYTHONPATH=src python3.12 -B -m unittest discover -v` — 116 tests, PASS, exit 0.
- `PYTHONPATH=src python3.12 -B -c 'import sys; import spire_learn.envs.native; blocked={"numpy","gymnasium","torch","torchrl","spire_lab"}; loaded=sorted(blocked.intersection(sys.modules)); print("blocked modules loaded:", loaded); assert not loaded'` — prints `blocked modules loaded: []`, exit 0.
- `git diff --check` — PASS, zero output, exit 0.
- No optional runtime, Gym, NumPy, factory, packing, simulator, or integration function was invoked.

## Read-only public API evidence; runtime remains unverified

Read-only evidence: `docs/combat-environments.md:131–139` documents `spire_lab._native` exports `CombatInputError`, `CombatStateError`, `CombatSetupError`, and `CombatFaultError`; `docs/development.md:139–151` documents `module-name="spire_lab._native"` and `_native.build_info()`. No `public_native` path is documented. Ledger line 134 was corrected: its `public_native` name was a placeholder/path defect; only an already-loaded `spire_lab._native` module is inspected. This minimizes guessing/import risk and makes a future documented module rename an explicit ledger/code review change; absent metadata/classes returns `None`, not inferred identity or classification.

`/home/klottick/repos/spire-lab/python/spire_lab/gymnasium/single.py:20–21,36–49` confirms constructor partition binding and reset accepts `options`; `_common.py:48–63` confirms the single reset option is `scenario_id`. Disposition of reviewer factory finding: `make_single` constructs only, binding `case.partition`; it must not reset. Task 8 `run_case` at plan line 361 checks descriptor before reset; collector owns the one reset and must pass `options={"scenario_id": case.scenario_id}`. This preserves reset ordering and avoids an extra factory reset. No runtime behavior was tested. Bead remains `in_progress`; Task 6 is not closed/completed.

## Gate and concerns

Actual runtime is blocked pending user-reviewed setup addendum and authorized isolated dependencies. Parent review is required before asking the user for that addendum. No native integration evidence, compatibility verdict, or Task 6 completion is claimed. Snapshot validation is deliberately pure stdlib and protocol-based; real ndarray conformance remains gated.

## Fresh controller verification (2026-10-24)

- Expected HEAD: `6561a57`; actual HEAD: `6561a57`.
- Command: `PYTHONPATH=src python3.12 -B -m unittest discover -s tests -v` — exit 0; **116 tests passed** (`Ran 116 tests in 0.021s`, `OK`). This is actual fresh controller output, not reported prior evidence.
- Import-isolation command: `PYTHONPATH=src python3.12 -B -c 'import sys, pkgutil, spire_learn; names=[m.name for m in pkgutil.walk_packages(spire_learn.__path__, spire_learn.__name__ + ".")]; [__import__(name) for name in names]; blocked={"numpy","gymnasium","torch","torchrl","spire_lab"}; loaded=sorted(blocked.intersection(sys.modules)); print("core modules imported:", names); print("blocked modules loaded:", loaded); assert not loaded'` — exit 0; all 18 discovered current core modules imported; `blocked modules loaded: []`.
- `git diff --check` — exit 0, no output.
- Git: HEAD `6561a57`; tracked changes none; untracked `task-6-core-review.diff` is preserved (SHA-256 `a52df58775d2c5fc0d8f1cbbe1946af787839e5d1379cd752e2663a4557edc29`). Not deleted, moved, staged, or overwritten. Canonical reviewer diff is `artifacts/development/foundation/task-6-core-fix-2-review.diff`.
- Task 6 `spire_learn-d8l.7` is now **blocked**, not complete: pure core review PASS/quality APPROVED and fresh 116-test verification complete; Step 5 needs a user-reviewed isolated-setup addendum before any install/runtime integration. No native calls, dependencies, builds, training, services, or agents were run.
- Tasks 1–5 remain closed per the canonical ledger entries above (Tasks 1–5 closeouts at lines 112, 117, 122–128). Tasks 7–10 remain pending; no dispatch or implementation began.
- Next action is a human decision/review of the isolated-setup addendum before any dependency installation or native/runtime work. Resume continuity at `artifacts/development/foundation/progress.md`; do not treat Task 6 as complete or reimplement the accepted core.

## Native completion and parent acceptance — 2026-10-25

**Latest status: Task 6 COMPLETE; Beads `spire_learn-d8l.7` closed.** Earlier
blocked/core-only status above is historical. Existing report content is
preserved verbatim; only this closeout is appended, and this modified report
is deliberately excluded from implementation/documentation commits.

- User-approved builder repair completed and passed protected non-root/source
  preflight. Derived builder Rust/Cargo1.98.0, Python3.12.14. Original builder
  remains retained; Gym checkout/lock/fixtures remain unchanged.
- One own Gym wheel built from exact read-only inputs; SHA256
  `25cd26001e8cf3034da95434351ac71f92f676dbeeb15784682f809e82d4799a`.
- Fifteen-wheel CPU closure audited/hash-locked and installed offline; pip check
  passed. Lock SHA256
  `d026856db845f105383c91d4901231fd016111f46257999456511e699475ee45`.
  Runtime Python3.12.15/glibc2.36, NumPy2.5.4/Gymnasium1.4.0/Torch2.14.1+cpu,
  torch.version.cuda=None. No host installs or Gym changes.
- Fresh baseline116 core tests passed. Final default discovery119 tests:116
  core pass and3 native-gated skips. Fresh import/discovery isolation loaded
  no optional/native packages. This is separate from actual native evidence.
- Exactly one offline native run executed3 tests,0 skipped, allpassed, exit0.
  Cultist20/JawWorm29 submitted and delivered actions, both public victory/+1,
  each one reset/close;49 total, within128 each/256 total. No faults or retries.
  Actual build_info revision4874fb398128c142bd56f10a12e6fca939a88f93, Rust1.98.0;
  schema fingerprint8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357.
- Parent inspected actual source, artifacts and raw evidence. Independent
  Luna/max review: spec PASS, code quality APPROVED, no actionable findings.
  Source/test commit `f21d90b`. Every Luna worker used verified max reasoning.
- Accepted data/runtime identities and reports are under this foundation
  workspace: native-wheel-identity/native-runtime-identity JSON; native-wheel-
  build/native-runtime/native-integration reports; native-integration-review.
  All carry date2026-10-25. Images, wheels, cache, venv and logs remain ignored
  and retained. Host Docker client config is outside the container DATA bind.
- Single-environment conformance only; no native vector/TorchRL/training,
  playing-strength or original-game transfer claim. RNG-001/TIME-001 retained.
  Tasks7–10 and M1 completion remain pending. No own container remains.
