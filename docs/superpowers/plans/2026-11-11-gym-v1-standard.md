# Gym 1.0.0 and Standard Profile Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin Learn to Spire Gym 1.0.0 and add the Standard combat view and
two Standard environment modes, so the combat learner can be driven by
`StandardCombatVectorEnv` and the Standard Run.

**Architecture:** Selection happens in three places:
- A revision-selected `GymContract` carries all 1.0.0 identities, including
  the full build identity.
- A view registry (`views.py`) replaces the hard-coded common-view constants
  in the schema, contracts and lifecycle parsing.
- `make_driver` gains two provider branches behind new `[environment] mode`
  values.

The feature planner, model and PPO stay descriptor-driven and unchanged.

**Tech Stack:** Python 3.12, stdlib `unittest`, the existing `spire_learn.neural`
package. The native Gym is only imported lazily inside `envs._load_native_runtime`.

**Spec:** `docs/superpowers/specs/2026-11-11-gym-v1-standard-design.md`

## Global Constraints

- Python 3.12. Imports stay free of environment, device and training side
  effects; unit tests never import `spire_lab`.
- Run a test module with
  `PYTHONPATH=src python3.12 -B -m unittest discover -s tests -p '<module>.py'` from the repo root.
  - Full discovery hashes the 21 GB `artifacts/` tree in the campaign-control
    tests, so run modules individually.
- Native revision `1929e18b9f44372e020877bd62f2a5c9dee3e64e`. Other identity
  values are copied verbatim from the spec's "Verified release identity"
  table.
- `rules_epoch = "gym-1.0.0"`. Mapping strings for `LEGACY_CONTRACT` and
  `CURRENT_CONTRACT` (afb51f5) must stay byte-identical.
- Never read, copy or forward any of these to features, lifecycle or logs:
  - `diagnostic()`, `effective_seed`;
  - `exact_checkpoint`, `exact_restore`, `exact_fork`, `from_record`;
  - `record`, `reproduction`;
  - `take_completed_records`, `take_fault_records`;
  - `np_random`.
- Escape (`"escaped"`) is a natural settlement with reward 0. It is never a
  fault, a win or a retry.
- No dependency installs, training, gameplay, or native execution without
  explicit user approval. Task 3, Step 2 and Task 9, Step 3 require that
  approval.
- Wherever a task says "Task 3 fixture", use `synthetic_standard_descriptor()`
  (Task 4) instead if Task 3 was gated off.
- Commit after each task. Do not push.

## Review Focus

1. **A 1.0.0 config paired with an afb51f5 checkpoint.** Expected: rejected
   before any Torch load, because the fingerprints are equal but the
   `rules_epoch` differs. Tested in Task 1.
2. **A descriptor whose fingerprint matches two contracts.** Expected: parsing
   uses the configured revision and never guesses. Tested in Task 4.
3. **A Standard combat that ends in `escaped`.** Expected: the row settles as
   terminal with outcome `escaped`. The driver does not freeze, and summaries
   count it apart from wins and deaths. Tested in Task 5.
4. **A Standard Run boundary outside combat.** Expected:
   - routing is `noncombat` and the mask is empty;
   - the non-combat legal indices come from the Run mask;
   - the stepped action is a Run index validated against `2048`.

   Tested in Task 8.
5. **An info dict or vector info that carries `effective_seed`, `slot_fault` or
   `fault`.** Expected: none of these appear on `Lifecycle`, `CombatFrame` or
   `EncodedBatch`. Tested in Tasks 7 and 8.

---

### Task 1: Gym 1.0.0 contract and build identity

**Files:**
- Modify: `src/spire_learn/neural/gym_contracts.py`
- Modify: `src/spire_learn/neural/envs.py:1018-1025` (`_verify_native_build`)
- Modify: `src/spire_learn/neural/schema.py:36` (`_schema_contract`), and the
  callers of `contract_for_fingerprint`
- Test: `tests/unit/test_neural_gym_v1.py` (new)

**Interfaces:**
- Produces:
  - `BuildIdentity(revision: str, source_digest: str, file_digest: str, profile: str, features: tuple[str, ...])`, a frozen dataclass.
  - `GymContract` gains these fields, each defaulting to `None`:
    - `build: BuildIdentity | None`
    - `standard_combat_fingerprint: str | None`
    - `standard_run_fingerprint: str | None`
    - `rules_epoch: str | None`
  - `raw_mode_mappings` adds `"standard-standalone"` and `"standard-run"` when
    the matching fingerprint is set.
    - Format: `spire-learn-standard-combat-v1@{sc}` and
      `spire-learn-standard-run-v1@{sr}:{sc}`.
    - Every mapping gets `+{rules_epoch}` appended when `rules_epoch` is set.
  - `GYM_V1_CONTRACT`
  - `CONTRACTS = (LEGACY_CONTRACT, CURRENT_CONTRACT, GYM_V1_CONTRACT)`
  - `contracts_for_fingerprint(fp) -> tuple[GymContract, ...]`, which replaces
    `contract_for_fingerprint`.
  - `contract_for_revision` is unchanged.

- [ ] **Step 1: Write failing tests** in `tests/unit/test_neural_gym_v1.py`.
  - `test_v1_contract_identities`: assert the revision, the five fingerprints
    and the `BuildIdentity` fields equal the spec table.
  - `test_v1_mappings_carry_rules_epoch`:
    ```python
    assertEqual(GYM_V1_CONTRACT.raw_mode_mappings["standalone"],
                f"spire-learn-standalone-common-v1@{COMMON}+gym-1.0.0")
    ```
    Also assert that all four modes are present.
  - `test_historical_mappings_unchanged`: assert the `CURRENT_CONTRACT`
    standalone mapping equals the literal string
    `spire-learn-standalone-common-v1@4c73f7de…f463`, with no `+` suffix.
  - `test_shared_fingerprint_is_ambiguous`: assert that
    `contracts_for_fingerprint(COMMON)` returns both `CURRENT_CONTRACT` and
    `GYM_V1_CONTRACT`.
  - `test_verify_native_build_full_identity`: call
    `envs._verify_native_build(fake, GYM_V1_CONTRACT)` with `fake.build_info()`
    returning the published dict. It passes. Mutating any one of these raises
    `ContractError`:
    - `dirty` set to `True`;
    - `profile` set to `"debug"`;
    - any digest;
    - `features`;
    - `revision`.
  - `test_v1_checkpoint_rejects_afb51f5_mapping`: give
    `checkpoint.validate_checkpoint_metadata` metadata with the afb51f5
    mappings and a 1.0.0 config. It raises before any Torch import.
    - Patch `checkpoint`'s torch loader with `side_effect=AssertionError`.
- [ ] **Step 2: Run the tests and confirm they fail.**
  - Run: `PYTHONPATH=src python3.12 -B -m unittest discover -s tests -p 'test_neural_gym_v1.py'`
  - Expected: `ImportError` for `GYM_V1_CONTRACT` and `BuildIdentity`.
- [ ] **Step 3: Implement.**
  - `_verify_native_build(native, contract: GymContract)` takes the contract
    instead of a revision.
  - When `contract.build` is set, it requires all of these:
    - `info["dirty"] is False`;
    - `profile`, the digests and `tuple(features)` equal the contract's;
    - `extension == "spire-python"`.
  - `_schema_contract` takes an explicit `revision: str` argument and
    resolves it with `contract_for_revision`. Thread the configured revision
    through `parse_common_descriptor(descriptor, *, revision)` and
    `CombatSchema`.
  - Update the callers in `envs.py`, `checkpoint.py` and `tests/neural/_support.py`.
- [ ] **Step 4: Run the new module, then these existing modules, and confirm
  they pass:**
  - `unit.test_neural_gym_migration`
  - `unit.test_neural_schema`
  - `unit.test_neural_checkpoint`
  - `unit.test_neural_envs`
  - `unit.test_neural_config`
- [ ] **Step 5: Commit** with message `feat: add Gym 1.0.0 contract with full build identity`.

### Task 2: 1.0.0 runtime profile and templates for the existing modes

**Files:**
- Create: `configs/neural/gym-v1.0.0/runtime.json`, `standalone.toml`, `run.toml`
- Modify: `scripts/neural/curriculum_profiles.py` only if its runtime-profile
  validator rejects `schema` `spire-learn-gym-runtime-profile-v2`. Add v2
  alongside v1; do not edit v1 behaviour.
- Test: `tests/unit/test_neural_gym_v1.py`

**Interfaces:**
- Consumes: `GYM_V1_CONTRACT.raw_mode_mappings` from Task 1.
- Produces: `runtime.json` with schema `spire-learn-gym-runtime-profile-v2`.
  - It has the v1 fields, plus `build_identity` (the dict published in
    `DELIVERY.md`), `card_metadata_fingerprint` (`effbb810…`) and
    `standard_fingerprints` (`{combat, run}`).
  - Its paths point at `artifacts/development/neural-combat/gym-v1.0.0/inputs/…whl`
    and `…/gym-v1.0.0/site`.

- [ ] **Step 1: Write failing tests.**
  - `test_v1_templates_validate`: run `load_neural_config` on both templates.
    It succeeds, and `environment_revision` equals `1929e18b…`.
  - `test_v1_runtime_profile_matches_contract`: load `runtime.json`. Its
    `native_sha256`, `wheel_sha256`, `build_identity` and fingerprints equal
    `GYM_V1_CONTRACT` and the spec table.
- [ ] **Step 2: Run the tests and confirm they fail** with
  `FileNotFoundError`.
- [ ] **Step 3: Create the three files.**
  - Copy `configs/neural/gym-20261102/*.toml`.
  - Change only `gym_revision`, the `accepted_raw_mode_mappings` and any
    config hash the loader recomputes.
- [ ] **Step 4: Verify.**
  - Run the tests and confirm they pass.
  - Then run `PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/gym-v1.0.0/standalone.toml --validate-only`.
  - Expected output: `configuration valid; no neural runtime or artifacts created`.
  - Run the same for `run.toml`.
- [ ] **Step 5: Commit** with message `feat: add Gym 1.0.0 runtime profile and existing-mode templates`.

### Task 3: Capture real 1.0.0 descriptors (approval-gated)

**Files:**
- Modify: `scripts/neural/probe_gym_release.py`
- Create:
  - `tests/fixtures/neural/standard-combat-1929e18.json`
  - `tests/fixtures/neural/standard-run-1929e18.json`
  - `tests/fixtures/neural/common-1929e18.json`
- Test: `tests/unit/test_gym_release_qualification.py`

**Interfaces:**
- Produces: the fixture JSON files, each the exact `observation_spec()` dict.
  The sources are:

  | Fixture | Source |
  | --- | --- |
  | Standard combat | `StandardCombatEnv(catalog).observation_spec()`, using a one-entry Ironclad A0 `fresh_start` catalog |
  | Standard Run | `RunEnv("ironclad", 0, profile="standard").observation_spec()` |
  | Common | `IroncladCombatVectorEnv(...).observation_spec()` |

  Later tasks load these.

- [ ] **Step 1: Add a `--capture-descriptors DIR` mode to the probe.**
  - It writes the three specs as canonical JSON. That is
    `json.dumps(sort_keys=True, separators=(",", ":"))`, the same canonical
    form `schema.py:30` hashes.
  - It recomputes each fingerprint with the domain from the spec. It exits
    non-zero on a mismatch.
  - It closes each env.
  - It never calls `reset` or `step`.
  - Add a preflight unit test, `test_capture_mode_never_resets`, which drives
    the probe against fake classes.
- [ ] **Step 2: Ask the user for approval, then run the probe** against the
  verified overlay.
  - Use the same container or isolated procedure as the November 2
    qualification (`docs/neural-gym-migration.md`).
  - Use the dependency venv from `configs/neural/gym-v1.0.0/runtime.json`, an
  isolated interpreter (`-I`) and `sys.path` = the overlay only.
  - Expected:
    - `build_info()` equals the spec table;
    - three fixtures are written;
    - exit status 0.
  - **Stop here if approval is withheld.** Tasks 4–8 then use only synthetic
    descriptors, and the plan's result must be reported as "not
    native-qualified".
- [ ] **Step 3: Assert that `common-1929e18.json` is byte-identical to the
  existing `common-afb51f5.json`.** The fingerprint is unchanged, so any
  difference is a finding to report, not something to paper over.
- [ ] **Step 4: Commit** the probe change and fixtures with message
  `test: capture Gym 1.0.0 descriptor fixtures`.

### Task 4: View registry and Standard combat schema parsing

**Files:**
- Create: `src/spire_learn/neural/views.py`
- Modify:
  - `src/spire_learn/neural/schema.py`: `parse_common_descriptor` becomes
    `parse_combat_descriptor`; keep `parse_common_descriptor` as a thin alias
    that requires the common view.
  - `src/spire_learn/neural/schema.py:1002` (`parse_lifecycle`)
  - `src/spire_learn/neural/contracts.py:18-31`, `:107`, `:138-185`
- Test: `tests/unit/test_neural_views.py` (new)

**Interfaces:**
- Produces:
  - `ViewSpec`, a frozen dataclass with these fields:
    - `view_id: str`
    - `profile_id: str`
    - `fingerprint_domain: bytes`
    - `groups: tuple[str, ...]`
    - `lifecycle_view_id: str`
    - `routings: frozenset[str]`
    - `outcomes: frozenset[str]`
  - `COMMON_V1` and `STANDARD_COMBAT_V1` hold the values in the spec,
    decision 4.
  - `view_for_id(view_id) -> ViewSpec`.
  - `tensor_names(view) -> frozenset[str]`: each group, plus `<g>_known` for
    every group, plus `<g>_occupied` for the row groups (whichever groups the
    descriptor marks with an occupied tensor), plus `action_mask`.
  - `parse_combat_descriptor(descriptor, *, revision: str) -> CombatSchema`.
    `CombatSchema` gains `.view: ViewSpec`.
  - `CombatFrame` validates tensor names against `schema.view`.
    `parse_lifecycle(info, schema)` validates routing and outcome against
    `schema.view`.

- [ ] **Step 1: Write failing tests.**
  - `test_standard_fixture_parses`: parse
    `parse_combat_descriptor(load("standard-combat-1929e18.json"), revision=V1)`.
    - `schema.view is STANDARD_COMBAT_V1`.
    - The fingerprint equals `5d8147e1…`.
    - `"orbs" in schema.tensors`.
    - `schema.action_space.n == 2048`.
  - `test_common_fixture_still_parses_under_both_revisions`: parse with
    `CURRENT_CONTRACT` and with `GYM_V1_CONTRACT`. The resulting
    `schema.contract` differs between the two.
  - `test_view_mismatch_rejected`: a Standard descriptor with
    `revision=afb51f5` raises `ContractError`.
  - `test_standard_feature_plan`: `feature_contract(schema)` succeeds.
    Every Standard group appears in `plan.groups`. Use Task 3's fixture, or
    `synthetic_standard_descriptor()` if Task 3 was gated off.
  - `test_frame_rejects_foreign_group`: a common-v1 frame carrying an `orbs`
    tensor raises `ContractError`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `views.py` and thread `ViewSpec` through the schema,
  frame and lifecycle.**
  - The fingerprint is
    `sha256(view.fingerprint_domain + canonical_json(body))`.
  - Add `synthetic_standard_descriptor()` to `tests/neural/_support.py`,
    mirroring `synthetic_descriptor()`.
- [ ] **Step 4: Run `unit.test_neural_views` and all `unit.test_neural_*`
  schema, feature, contract and model modules, and confirm they pass.** Also
  run `tests/neural/test_model.py` with `SPIRE_LEARN_RUN_NEURAL_TENSOR=1`, only
  if Torch is already present; never install it.
- [ ] **Step 5: Commit** with message `feat: add combat view registry and Standard combat schema`.

### Task 5: Escape as a natural settlement

**Files:**
- Modify:
  - `src/spire_learn/neural/contracts.py:182-187`, `:486-487`: allowed outcomes
    come from the view.
  - `src/spire_learn/neural/collector.py`, `runner.py` and the evaluation
    summaries: every `"victory"`/`"death"` match found by
    `grep -rn '"victory"\|"death"' src/spire_learn/neural`.
- Test: `tests/unit/test_neural_escape.py` (new)

**Interfaces:**
- Produces: the outcome counters gain an `escaped` key. Win rate stays
  `victories / settled`, and escapes count in the denominator. The return for
  an escape is `0.0`.

- [ ] **Step 1: Write failing tests.**
  - `test_escape_settles_without_fault`: drive a scripted Standard provider
    whose row terminates with `last_exit.outcome="escaped"` and reward `0.0`.
    `step` returns a terminal transition, and the driver is not frozen.
  - `test_escape_counted_separately`: the summary has
    `{"victory": 1, "death": 0, "escaped": 1}` and a win rate of `0.5`.
  - `test_common_view_still_rejects_escape`: an `escaped` outcome on a
    common-v1 lifecycle raises `ContractError`.
- [ ] **Step 2: Run the tests and confirm they fail. Then implement, run them
  again and confirm they pass.**
- [ ] **Step 3: Commit** with message `feat: treat Standard escape as a natural zero-reward settlement`.

### Task 6: Standard config modes and cases

**Files:**
- Modify: `src/spire_learn/neural/config.py:25-49`, `:117-146`, `:264-297`;
  `src/spire_learn/neural/cases.py:18`, `:109-111`
- Create: `configs/neural/gym-v1.0.0/standard-standalone.toml`, `standard-run.toml`
- Test: `tests/unit/test_neural_config_standard.py` (new)

**Interfaces:**
- Produces:
  - `[environment] mode` accepts `"standard-standalone"` and `"standard-run"`.
  - For these modes, `[environment]` requires `standard_view_fingerprint`.
    `standard-run` also requires `standard_run_fingerprint`.
  - Both modes also take `workers: int`, in the range 1–256, defaulting to 1.
  - `NeuralConfig` gains `standard_view_fingerprint`,
    `standard_run_fingerprint` and `workers`.
  - Standard cases take `character` in `{"ironclad","silent","defect","watcher"}`
    and `ascension` in the range 0–20.
  - `standard-standalone` cases carry a catalog entry: `id`, `character`,
    `ascension`, `encounter`, `hp`, `max_hp`, `gold`, `deck`, `relics`,
    `potions`, and the optional `relic_counters`, `bottled_cards` and
    `burning`.
  - `standard-run` cases carry `creation` (`neow`, `seen_bosses`, `note`) or
    omit it.

- [ ] **Step 1: Write failing tests.**
  - `test_standard_templates_validate`: both templates load. `--validate-only`
    prints the success line.
  - `test_standard_requires_v1_revision`: a Standard mode with
    `gym_revision = afb51f5…` raises `ConfigError` before any runtime import.
    Patch `envs._load_native_runtime` with `side_effect=AssertionError`.
  - `test_standard_fingerprint_must_match_contract`: a tampered
    `standard_view_fingerprint` raises `ConfigError`.
  - `test_legacy_modes_still_ironclad_a0`: a `run` case with
    `character="silent"` is still rejected.
  - `test_workers_bounds`: `workers` of 0 or 257 is rejected.
- [ ] **Step 2: Run the tests and confirm they fail. Then implement, and
  confirm the new and existing `unit.test_neural_config*` modules pass.**
- [ ] **Step 3: Commit** with message `feat: add Standard environment modes and multi-character cases`.

### Task 7: Standard standalone provider

**Files:**
- Modify: `src/spire_learn/neural/envs.py`. This adds:
  - `_NativeRuntime.standard_combat_vector_env`, loaded lazily from
    `spire_lab.gymnasium`;
  - a new `_StandardCombatProvider`;
  - a new `make_driver` branch.
- Test: `tests/unit/test_neural_envs_standard.py` (new)

**Interfaces:**
- Consumes:
  - `parse_combat_descriptor` and `STANDARD_COMBAT_V1` (Task 4);
  - the config fields from Task 6.
- Produces: `_StandardCombatProvider(env, runtime, rows)`, implementing the
  existing `_Provider` protocol (`reset`/`step` → `_RawBoundary` /
  `_RawTransition`).
  - The env is constructed as `StandardCombatVectorEnv(catalog, config.rollout_rows, partition=config.partition, workers=config.workers, autoreset_mode="disabled", fault_policy="freeze")`.
  - The identity check requires:
    - `profile_id="standard-combat-v1"`;
    - `schema_id=STANDARD_COMBAT_V1.view_id`;
    - the observation fingerprint equals the config's;
    - `catalog_schema_id="spire-gym-standard-combat-catalog-v1"`.
  - Native errors map into the existing input/state/fault classification in
    `envs.py:485`, `:539-664`. `BatchFaultError.slots[*]` statuses go into
    `DriverFaultEvidence` as slot indices and statuses only. Messages are not
    included, because they may quote native state.

- [ ] **Step 1: Write failing tests** with a `FakeStandardVector`. It follows
  the `FakeVector` pattern in `test_neural_envs.py:666-701` and serves the
  Task 3 fixture descriptor.
  - `test_standard_factory_constructs_with_disabled_autoreset`: assert the
    constructor arguments that the fake records.
  - `test_standard_reset_and_step_produce_frames`: the frames parse and the
    legal candidates match the mask.
  - `test_standard_identity_mismatch_closes_env`: a wrong `profile_id` raises
    `ContractError`, and `fake.closed` is `True`.
  - `test_standard_info_privacy`: the fake info includes `effective_seed`,
    `diagnostic`, `slot_fault` and `fault`. None of them is an attribute or
    key of `Lifecycle`, `CombatFrame` or the `EncodedBatch` field names.
  - `test_batch_fault_freezes_with_slot_evidence`: a
    `FakeBatchFaultError(slots=[{"status":"faulted",...}])` raises
    `DriverFault`, and the evidence slots equal `(0,)`.
  - `test_never_calls_privileged_methods`: the fake raises `AssertionError`
    from each of these:
    - `diagnostic`, `diagnostics`;
    - `take_completed_records`, `take_fault_records`;
    - `exact_checkpoint`, `public_search`.

    A full reset/step/close cycle passes.
- [ ] **Step 2: Run the tests and confirm they fail. Then implement, and
  confirm the new and existing `unit.test_neural_envs` modules pass.**
- [ ] **Step 3: Commit** with message `feat: drive Standard standalone combat through StandardCombatVectorEnv`.

### Task 8: Standard Run provider

**Files:**
- Modify: `src/spire_learn/neural/envs.py`. This adds:
  - `_NativeRuntime.run_vector_env`;
  - a new `_StandardRunProvider`;
  - a `make_driver` branch.
- Test: `tests/unit/test_neural_envs_standard.py`

**Interfaces:**
- Consumes: the Task 4 schema, the Task 6 config and the Task 7 error
  classification.
- Produces: `_StandardRunProvider(env, runtime, rows)`, built as
  `RunVectorEnv([{"character":c,"ascension":a,"partition":p,"creation":cr?}...], profile="standard", workers=config.workers, autoreset_mode="disabled", fault_policy="freeze")`.
  - `dispatch_action_count = 2048`. This is read from the Run descriptor's
    `action_space.n`, with an int8 `action_mask` of shape `[n]`, through the
    existing `_run_action_count`.
  - After each reset and step it calls `env.combat_views()`. The combat frame
    comes from the view `obs`, and the lifecycle comes from the view `info`.
    The Run `info` is discarded, as `envs.py:370` already does.
  - Non-combat legal indices come from the Run `obs["action_mask"]` through
    `_legal_indices`.
  - The identity check requires:
    - Run `schema_id="spire-gym-standard-run-observation-v1"`;
    - its fingerprint equals `standard_run_fingerprint`;
    - the combat view fingerprint equals `standard_view_fingerprint`.

- [ ] **Step 1: Write failing tests** with `FakeRunVector`, which serves the
  Task 3 Run and combat fixtures.
  - `test_noncombat_boundary_routes_to_run_mask`: with `routing="noncombat"`
    and an empty view mask, `row.noncombat_indices` equals the legal Run
    indices. Action 2047 is accepted when it is legal; action 2048 raises
    `DriverInputError`.
  - `test_combat_boundary_uses_view_candidates`: the view candidates match
    the view mask.
  - `test_escape_in_run_continues`: the view reports
    `last_exit.outcome="escaped"`, and the Run is not terminated. The row
    stays live, and nothing is settled as an episode end.
  - `test_run_terminal_reasons`: `heart_victory`, `death`, `empty_deck` and
    `act3_no_keys` each terminate the row. None of them freezes the driver.
  - `test_run_info_privacy`: the same assertion as in Task 7, applied to the
    Run and view info.
- [ ] **Step 2: Run the tests and confirm they fail. Then implement, and
  confirm `unit.test_neural_envs_standard` and `unit.test_neural_envs` pass.**
- [ ] **Step 3: Commit** with message `feat: drive combat learning inside Standard Runs via RunVectorEnv`.

### Task 9: Qualification probe and documentation

**Files:**
- Modify: `scripts/neural/probe_gym_release.py` (add the Standard adapter
  constructors to the qualification mode);
  `docs/gym-learning-capabilities.md`;
  `docs/learning-investigation-20261104.md` (the "Gym boundary" section only:
  add a dated note that `spire-2o7` and `spire-831` were delivered in Gym
  1.0.0 on Standard roots); `README.md` (current pin)
- Create: `docs/neural-gym-v1-migration.md`

- [ ] **Step 1: Extend the qualification mode** so it calls `make_driver` for
  each of the four 1.0.0 templates and parses one reset boundary, then closes.
  It must not call `step`, and it trains nothing. Add the preflight test
  `test_qualification_constructs_four_modes` with fakes.
- [ ] **Step 2: Write `docs/neural-gym-v1-migration.md`.** Base it on the
  spec's table and decisions. It must include these sections:
  - **Rules changes affecting comparability:** turn order, Louse openings,
    Lagavulin, burning elite `regenerate`, Havoc/Exhume, Stasis, and power row
    order. Explain why old panel scores (41/64 and so on) are not comparable.
  - **Privileged surfaces:** the list from the Global Constraints.
  - **Qualification status:** what ran, or "not native-qualified".
- [ ] **Step 3: Ask the user for approval, then run the qualification once.**
  Expected:
  - exit status 0;
  - four adapters constructed and one reset boundary parsed for each;
  - all envs closed.

  Record the exit code, duration and identities in the doc. If approval is
  withheld, say so in the doc and leave the status as not native-qualified.
- [ ] **Step 4: Run every `unit.test_neural_*` and
  `unit.test_gym_release_qualification` module, plus `git diff --check`.
  Expected: all OK.**
- [ ] **Step 5: Commit** with message `docs: record Gym 1.0.0 Standard adoption and qualification`.

### Task 10: Standard modes through the collector and runner (added during execution)

**Files:**
- Modify: `src/spire_learn/neural/config.py` (`mode_family`), `collector.py`, `contracts.py` (`Rollout` modes), `runner.py`
- Test: `tests/unit/test_neural_standard_runner.py` (new); `tests/unit/test_neural_envs_standard_run.py` (out-of-combat Run terminal)

**Interfaces:**
- Produces: `mode_family(mode) -> "standalone" | "run"`. The collector checks the driver mode against the family, selects cases and identities by `config.mode`, and emits `Rollout.mode == config.mode`. The runner branches on the family and records per-mode `profile_id`/`content_id`/`rules_id`/`reward_id` plus `standard_view_fingerprint`/`standard_run_fingerprint` in run metadata. Curriculum `case_sampler`, scenario streams and calibration stay legacy-standalone-only.

- [ ] Write failing tests (mode family; collector selects Standard cases for both modes; cross-family driver rejected; manifest identities for both Standard modes; standalone fight summary; Run-family completion)
- [ ] Implement, run the module and the neural/gym/integration regression sweep
- [ ] Commit `feat: run Standard modes through the collector and runner`
