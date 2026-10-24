# Task 6 implementation report — core only

**Status:** fix round 1 `DONE_WITH_CONCERNS` for pure core; actual runtime `BLOCKED`. Task 6 is **not complete**. Stopped at approved-plan Step 5 authority gate; Steps 6–7 were not run and no integration test file was added.

## Scope delivered

- Corrected array-like shape validation to require exact nested `tolist()` rank/shape; immutable `ArrayValue` remains flat but is revalidated against dtype-specific integer ranges, float finiteness and range. `FakeArray.tolist()` now mirrors declared shape.
- Corrected metadata lookup to already-loaded `spire_lab._native`; build info is recursively copied/frozen. With no injected classifier types, exact documented exception exports are resolved only from that already-loaded module; missing/malformed exports and arbitrary `RuntimeError` remain unmapped.
- Added 5 regression tests (12 core tests total). Test-only fixture descriptor and generic `FakeArray` are from `tests/support.py`.
- RED evidence for regressions: focused run before implementation failed shape, dtype, loaded-export classification, and build-info cases (exit 1; 6 failures, 2 errors); corrected assertions and implementation then passed.

## Verification

- `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_native_contract -v` — 12 tests, PASS, exit 0.
- `PYTHONPATH=src python3.12 -B -m unittest discover -v` — 115 tests, PASS, exit 0.
- `PYTHONPATH=src python3.12 -B -c 'import sys; import spire_learn.envs.native; blocked={"numpy","gymnasium","torch","torchrl","spire_lab"}; loaded=sorted(blocked.intersection(sys.modules)); print("blocked modules loaded:", loaded); assert not loaded'` — prints `blocked modules loaded: []`, exit 0.
- `git diff --check` — PASS, exit 0.
- No optional runtime, Gym, NumPy, factory, packing, simulator, or integration function was invoked.

## Read-only public API evidence; runtime remains unverified

Read-only evidence: `docs/combat-environments.md:131–139` documents `spire_lab._native` exports `CombatInputError`, `CombatStateError`, `CombatSetupError`, and `CombatFaultError`; `docs/development.md:139–151` documents `module-name="spire_lab._native"` and `_native.build_info()`. No `public_native` path is documented. Ledger line 134 was corrected: its `public_native` name was a placeholder/path defect; only an already-loaded `spire_lab._native` module is inspected. This minimizes guessing/import risk and makes a future documented module rename an explicit ledger/code review change; absent metadata/classes returns `None`, not inferred identity or classification.

`/home/klottick/repos/spire-lab/python/spire_lab/gymnasium/single.py:20–21,36–49` confirms constructor partition binding and reset accepts `options`; `_common.py:48–63` confirms the single reset option is `scenario_id`. Disposition of reviewer factory finding: `make_single` constructs only, binding `case.partition`; it must not reset. Task 8 `run_case` at plan line 361 checks descriptor before reset; collector owns the one reset and must pass `options={"scenario_id": case.scenario_id}`. This preserves reset ordering and avoids an extra factory reset. No runtime behavior was tested. Bead remains `in_progress`; Task 6 is not closed/completed.

## Gate and concerns

Actual runtime is blocked pending user-reviewed setup addendum and authorized isolated dependencies. Parent review is required before asking the user for that addendum. No native integration evidence, compatibility verdict, or Task 6 completion is claimed. Snapshot validation is deliberately pure stdlib and protocol-based; real ndarray conformance remains gated.
