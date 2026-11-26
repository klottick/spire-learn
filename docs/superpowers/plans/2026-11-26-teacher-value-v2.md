# Teacher Value v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the search teachers value the state a fight leaves behind, not just the fight's outcome. The value covers HP, max HP and potions priced in HP, with a curved-HP variant run alongside. Measure the v2 teachers against controls that isolate each change, then pick the dataset teacher by a rule fixed in advance.

**Architecture:**
- A pure module, `search_value.py`, owns playout scoring:
  - `flat-v1` returns exactly today's numbers;
  - `resource-v2` is the new rule.
- `View` gains a playout policy that never uses or discards potions and ends the turn only as a last resort.
- `TeacherConfig` and the one-move teacher name a value rule and a playout policy. v1 code paths stay literally unchanged (`float(reward)`, `heuristic_choice`).
- The teacher entry and the audit script learn to load, run and report named v2 teachers.
- v2 runs natively on Gym v1.2.5's `tiered_verbs` playout rule. v1.2.5 delivered the change request `spire-9wa`, and it is adopted as a drop-in within the `gym-1.2.4` rules epoch (Task 9). The Python playout path from Tasks 1–3 stays as a reference implementation.

**Amendment, 2026-11-26:** Gym v1.2.5 shipped `spire-9wa`. The plan now:
- adds Task 9 (adopt v1.2.5 as a drop-in) and Task 10 (native v2 scoring on the tiered rule);
- moves v2 measurement to native playouts;
- drops the Python backend control;
- marks Task 7 done.

Execution order: Tasks 1–5, 9, 10, 6, 8.

**Tech Stack:** Python 3.12, Spire Gym v1.2.5 as a drop-in for v1.2.4 (rules epoch `gym-1.2.4`), the offline runtime container, the existing Learn modules.

**Spec:** `docs/superpowers/specs/2026-11-21-search-distillation-design.md` (search teacher, audit, gates). The v2 value and the potion prices are user rulings of 2026-11-26, which extend the spec's teacher; the exact formula is in Global Constraints. Stage-1 results: `docs/neural-search-teacher-gym-1.2.4.md`. Base: main `cc993dd`.

## Global Constraints

- **Why v2.**
  - Today a won playout scores a flat +1 whatever HP is left, so 62.5% of the tree-3x audit decisions had an exact top tie.
  - `current` keeps 0.38 of max HP on the selection panel, against 0.49–0.56 for the trees.
  - HP and potions carry across an A20 run.
- **Gym v1.2.5 (drop-in, delivers `spire-9wa`).**
  - Identity: tag v1.2.5, commit `731fca0e0341d80e7cf08a8be7a03be1fa32c3f1`.
  - Wheel `spire_lab-1.2.5-cp312-cp312-manylinux_2_34_x86_64.whl`, SHA-256 `02346039385b2135f5e7809b3af98315dca89e07cfd855de59eedfeeaad177cf`. Extension `.so` SHA-256 `67819e119354e3a48e42102d66260055bcde7bed0f22b9e1eb7a84059580cdf2`.
  - `source_digest` `19854fcccf6bd09ceef036297d4a6ecf9d068d9bbc37968a38de07482f2ce117`; `file_digest` `63533371d420314d41b6307169156e5fe715c41920052f4117d225cf758d6a63`.
  - Local packet: `/home/klottick/.local/share/spire-gym/evidence/builds/731fca0/packet/`. It is read-only; check it against `MANIFEST.sha256`.
  - Unchanged from v1.2.4: the rules ids (Standard v9, M10/Gym66 v6, M9 v7), all 14 descriptor fingerprints, the replay digests and the Heart pins. `spire-gym-public-playout-v1` results are byte-identical.
  - New rule `tiered_verbs` (id `spire-gym-public-playout-tiered-v1`) on `playout_many` and `playout_action`:
    - keywords `never_verbs` (default `("use_potion", "discard_potion")`) and `last_resort_verbs` (default `("end_turn",)`);
    - it picks uniformly outside both sets, then among the last-resort actions, then among all legal actions;
    - with one eligible action it picks without a draw, and the per-step keys are counter-based;
    - it cannot be combined with `exclude_verbs`.
  - New summary fields: `player_gold`, plus per-slot `potion_kind` (Standard `potion` vocabulary ids) and `potion_present`. A terminal branch's `player_hp` is documented as HP after combat cleanup.
  - Read the v1.2.5 docs from a fresh `source-ref` clone of the tag under `artifacts/development/neural-combat/gym-1.2.5/`. Never from a Gym checkout.
- **Native playouts before v1.2.5 could not keep potions out.**
  - The Gym rule `spire-gym-public-playout-v1` picks uniformly among legal actions whose verb is not excluded. When none remain, it picks among all legal actions (`standard-run-api.md`, "Rule").
  - Today it excludes `end_turn` and `discard_potion`, so whenever no card is playable, a playout drinks a potion before it can end the turn.
  - Adding `use_potion` does not fix this: the fallback reopens potions.
  - The native summary also lacks the potions held.
  - That is why Tasks 1–3 built the Python path. From v1.2.5, `no-potions-v2` maps to native `tiered_verbs` with its defaults. The Python policy implements the same tiers, but its RNG streams differ, so native and Python agree statistically, not bit for bit. v1 teachers keep their behaviour.
- **Potion prices** (user ruling: a typical potion is worth about 10–15 HP; extremes differ):
  - default `potion_hp = 12`;
  - overrides by Gym potion id: `fairy_potion` (Fairy in a Bottle) 30, and `fruit_juice` 7.5, which offsets its +5 max HP gain at weight 1.5;
  - sensitivity variants use a default of 10 and of 15, with the same overrides.
  - Ids are resolved from the descriptor's potion vocabulary by `View.potion_price_table(rule)`, called once per decision. An unknown id raises a `ValueError` there.
- **Value rule `resource-v2`, exact.**
  - Definitions:
    - `M` is the player's max HP at the search root.
    - `hp_end` is the HP at the branch's end. A terminal branch uses the player `hp_after_cleanup` column: combat-end hooks such as Burning Blood and Black Blood have applied, on a Smoke Bomb escape too. A live branch uses `hp`.
    - `max_end` is the max HP at the branch's end.
    - `e` is the remaining enemy HP share, as in `View.balance`.
  - Potion charge, in HP: `P = Σ price(kind) over potions held at the root but not at the end − Σ price(kind) over potions held at the end but not at the root`. It compares the multisets of the `potions` rows' `potion_kind`, over `potions_occupied` rows.
  - HP term: `h = clip(hp_end / M, 0, 1)`. Apply `h ← sqrt(h)` when `hp_curve == "sqrt"`.
  - Bonus, outside the clip: `b = 0.4 × (1.5 × (max_end − M) − P) / M`.
  - Values:
    - victory: `clamp(0.55 + 0.4 × h + b, 0.41, 1.0)`. This is 0.95 at full HP, leaving 0.05 of headroom for Feed and gained potions. 1 HP at the end and 1 HP of potion price weigh exactly the same.
    - Smoke Bomb escape: `clamp(−0.2 + 0.4 × h + b, −0.99, 0.4)`.
    - death: `−1`.
    - cut-off, meaning a live branch at the step or depth limit: `clamp(0.4 × clip(h − e, −1, 1) + b, −0.99, 0.4)`.
    - refused playout: `−1`, flagged.
  - Exchange rates: with the linear curve, every term is in HP at the same 0.4/M scale, so the stated potion prices and the 1.5× max-HP weight hold exactly. The sqrt curve changes the HP side by design.
  - Ordering, guaranteed by the clamps: victory ≥ 0.41 > any cut-off or escape ≥ −0.99 > death. The floors only bind in extreme potion spends; a floored victory no longer separates how many potions were burned.
  - `hp_curve` is `"linear"` by default. The `max_hp_weight` 1.5 and the band constants (0.55, 0.4, 0.41, −0.2, −0.99) are fixed for v2.
  - The full-HP headroom saturates: at `M ≤ 96`, one gained typical potion already fills the 0.05, so further gains at full HP tie. This is accepted for v2.
- **Playout and fallback policy `no-potions-v2`.**
  - Choose uniformly among legal actions whose verb is not `use_potion`, `discard_potion` or `end_turn`.
  - If there are none, choose `end_turn` if it is legal.
  - Otherwise choose uniformly among all legal actions.
  - It reads only the legal table's verbs and the playout RNG.
  - It also replaces `heuristic_choice` for v2 teachers when a search root is refused.
- **v1 must not change.**
  - `flat-v1` and `uniform-v1` keep today's code paths. Terminal scoring stays `float(reward)` and needs no `terminal_reason`. Playouts call `view.heuristic_choice`.
  - Every existing teacher's `config_sha256` is unchanged: `current`, `tree-1x`, `tree-3x` and `tree-10x` in `configs/neural/search-teacher-gym-1.2.4/teachers.json`.
  - Past records stay valid.
- **Two independent switches.** Scoring branches only on `value_rule.name`. Playout choice, the refused-root fallback and the full-view guard branch only on `playout_policy`. So the control `flat-v1` with `no-potions-v2` scores like v1 but plays out with no potions.
- **Guards.**
  - `resource-v2` requires `playout_policy == "no-potions-v2"` and a root context.
  - On the Python backend, any `playout_policy` other than `uniform-v1` requires branch view `"full"`.
  - On the native backend (Task 10), `no-potions-v2` and `resource-v2` require a live Gym whose contract declares the tiered rule (v1.2.5 and later). Older Gyms refuse them with a `ValueError` that names the rule. This check runs only where a live Gym exists; config construction stays offline.
  - `resource-v2` requires branch view `"full"` on both backends.
  - `ValueRule("flat-v1", …)` rejects non-default v2 parameters.
- Rules epoch `gym-1.2.4`. Learner settings: 16 rows, exact settings, compact view, `max_turn_actions = 500`.
- Never feed seeds, hidden RNG state or unseen draw order to actors or critics. Faults are errors, never losses or censors. Only aggregates are published. Baalorlord-derived data stays local.
- `search_teacher.py` must keep none of the words that `tests/unit/test_search_teacher.py` forbids: `checkpoint`, `restore`, `fork_session`, `seed`, `record` and `snapshot`, including in comments.
- **Reviews:**
  - two reviewers (Claude Sonnet medium plus Gemini 4 Neon Low) for Tasks 1–5, 9 and 10;
  - one reviewer for configs and docs.
- Never edit the repo while a container run is in progress. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- PRs and merges run under the user's standing authority for this effort.

## Review Focus

1. **A terminal victory or escape whose `hp_after_cleanup` is not known** (`player_known` 0). The value raises; it never falls back to `hp`. Death needs no HP. Owner: Task 1, `test_terminal_without_cleanup_hp_raises`.
2. **Root max HP of 0 or unknown,** as in a synthetic state. `RootContext.from_observation` raises. Owner: Task 1, `test_root_context_rejects_nonpositive_or_unknown_max_hp`.
3. **Potions gained during the search** (Alchemize, Entropic Brew). The credit applies, but a victory stays ≤ 1.0, an escape ≤ 0.4 and a cut-off ≤ 0.4. Owner: Task 1, `test_gained_potion_credit_is_capped`.
4. **A legal table holding only potion and end-turn actions.** `no-potions-v2` ends the turn. With potion actions only, it falls back to them. Owner: Task 2, `test_policy_ends_turn_before_any_potion` and `test_policy_falls_back_when_only_potions_legal`.
5. **A v1 teacher after this change.** Its config hash, its values and its choices on the existing fakes equal those before the change, and the existing test modules pass unmodified apart from the one hash-test update named in Task 3. Owner: Task 3, `test_v1_teachers_are_bit_identical`.

---

### Task 1: The value module

**Files:**
- Create `src/spire_learn/neural/search_value.py`.
- Modify `src/spire_learn/neural/search_view.py`:
  - column lookups for player `hp_after_cleanup` and `player_known`;
  - the potions `potion_kind` column;
  - the potion vocabulary.
- Test: create `tests/unit/test_search_value.py`.

**Interfaces:**
- `ValueRule`, a frozen dataclass:
  - fields `name` (`"flat-v1"` or `"resource-v2"`), `potion_hp: float = 12.0`, `potion_overrides: tuple[tuple[str, float], ...] = (("fairy_potion", 30.0), ("fruit_juice", 7.5))` and `hp_curve: str = "linear"`;
  - validates names, the curve, and finite, non-negative prices;
  - `flat-v1` rejects non-default fields;
  - `potion_overrides` is sorted by id, and duplicates and `"padding"` are rejected;
  - `as_identity() -> dict` gives JSON-safe output, with the overrides as a dict such as `{"fairy_potion": 30.0, "fruit_juice": 7.5}`. `from_identity(dict)` builds one from it.
- `RootContext`, a frozen dataclass with fields `max_hp: int`, `potions: tuple[int, ...]` (sorted potion-kind ids) and `prices: Mapping[int, float]`. It is built once per decision by `RootContext.from_observation(view, observation, rule)`.
- `View` methods:
  - `potion_kinds(observation) -> tuple[int, ...]`;
  - `cleanup_hp(observation) -> int`, which raises when the value is unknown. Branch-end `hp` and `max_hp` reads also check `player_known` and raise when unknown;
  - `potion_price_table(rule) -> dict[int, float]`, which resolves ids and raises on unknown ones.
- Scoring functions:
  - `terminal_value(rule, root, view, observation, reward, reason) -> float`. For `flat-v1` it returns `float(reward)` and ignores `reason`. For `resource-v2`, `reason` must be `"victory"`, `"death"` or `"escaped"`.
  - `cutoff_value(rule, root, view, observation) -> float`. For `flat-v1` it returns `0.5 * view.balance(observation)`.
  - `REFUSED_VALUE = -1.0`.

- [ ] **Step 1: Write the failing tests,** with fake observations in the style of the existing `test_search_*` fakes:
  - `test_flat_v1_returns_reward_and_half_balance`. A fractional reward such as 0.25 is returned as given.
  - `test_v2_victory_full_hp_is_0_95`: hp_end equals M, with no potions and no max-HP change, gives 0.95.
  - `test_v2_victory_same_end_hp_scores_same_regardless_of_start`: root HP 1 and root HP 80, both with M 80, ending at 1 HP, give equal values.
  - `test_v2_potion_price_in_hp`: one typical potion at M 75 costs `0.4 × 12/75`, at any HP.
  - `test_v2_fairy_and_fruit_juice_overrides`: Fruit Juice drunk with +5 max HP, at equal end HP, is value-neutral.
  - `test_v2_feed_headroom_at_full_hp`: +3 max HP at full HP raises the victory from 0.95 to `min(1.0, 0.95 + 0.4 × 4.5 / M)`.
  - `test_v2_hp_and_potion_price_trade_one_to_one`: below full HP, 12 more HP at the end exactly offsets one typical potion drunk.
  - `test_v2_sqrt_curve`: h 0.25 becomes 0.5, so the victory is `0.55 + 0.4 × 0.5`.
  - `test_v2_ordering_holds_on_a_grid`: every combination of hp_end, max-HP change, potions spent (0–5, including fairy) and `e` keeps victory ≥ 0.41 > every cut-off and escape ≥ −0.99 > death.
  - `test_v2_escape_uses_cleanup_hp_and_charges_smoke_bomb`.
  - `test_flat_v1_rejects_v2_parameters`.
  - `test_value_rule_identity_round_trip`.
  - Review Focus 1–3.
- [ ] **Step 2: Run them to verify they fail:**
  `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_search_value -v`
- [ ] **Step 3: Implement.** Pure arithmetic over the view's columns, with no Gym import at module load.
- [ ] **Step 4: Run them to verify they pass.** Commit `feat: resource-aware search value rule`.

### Task 2: The no-potions policy

**Files:**
- Modify `src/spire_learn/neural/search_view.py`.
- Test: create `tests/unit/test_search_view_policy.py`.

**Interfaces:**
- `PLAYOUT_POLICIES = ("uniform-v1", "no-potions-v2")`.
- `View.playout_choice(observation, rng, policy) -> int`. `"uniform-v1"` returns `self.heuristic_choice(observation, rng)`.
- `View` resolves the `use_potion` verb index.

- [ ] **Step 1: Write the failing tests:**
  - `test_uniform_v1_is_heuristic_choice`: the same RNG gives the same 200 choices.
  - `test_policy_never_picks_potions_when_cards_playable`.
  - `test_policy_ends_turn_before_any_potion`: the legal set {end_turn, use_potion×2, discard_potion×2} gives `end_turn` every time.
  - `test_policy_falls_back_when_only_potions_legal`.
  - `test_policy_uses_only_verbs_and_rng`.
- [ ] **Step 2: Implement, run the tests and commit** `feat: no-potions playout policy`.

### Task 3: Wire v2 into the teachers, keeping v1 literal

**Files:**
- Modify `src/spire_learn/neural/search_teacher.py`: `TeacherConfig`, `config_identity`, `_simulate`, `_playout` and `teach`.
- Modify `src/spire_learn/neural/search_one_move.py`: `one_move_values` and `_native_playouts` (guards only).
- Modify `tests/unit/test_search_native_playouts.py`: in `test_python_hash_is_unchanged_and_native_is_distinct`, also delete `"value_rule"` and `"playout_policy"` from the legacy dict.
- Test: create `tests/unit/test_search_value_wiring.py`.

**Interfaces:**
- `TeacherConfig` gains `value_rule: ValueRule = ValueRule("flat-v1")` and `playout_policy: str = "uniform-v1"`, with the guards from Global Constraints.
  - `config_identity` omits both fields at their v1 defaults.
  - Otherwise it adds `value_rule.as_identity()` and `playout_policy`.
- `teach(..., root: RootContext | None = None)`. `one_move_values(..., value_rule=ValueRule("flat-v1"), playout_policy="uniform-v1", root=None)`.
- **v1 paths stay as they are:** under `flat-v1` and `uniform-v1`, every scoring and playout line is today's expression (`return reward`, `0.5 * view.balance(...)`, `view.heuristic_choice(...)`).
- **v2 paths,** gated independently:
  - When `value_rule.name == "resource-v2"`, scoring calls `terminal_value` or `cutoff_value` with the step's reward and `info["terminal_reason"]`. A missing reason on a terminal step raises, and `root=None` raises.
  - When `playout_policy != "uniform-v1"`, playouts call `view.playout_choice(..., playout_policy)`, and a branch view other than `"full"` raises.

- [ ] **Step 1: Write the failing tests:**
  - `test_v1_teachers_are_bit_identical`:
    - the `config_sha256` of each teachers.json entry equals its value at `cc993dd` (hard-coded in the test);
    - on the existing fakes in `test_search_teacher.py` and `test_search_one_move.py`, `teach` and `one_move_values` give the same `ActionStat`s and choice as `cc993dd`.
  - `test_v2_guards`: covers the native backend, a non-v1 policy on native, the `"mask"` view, and a missing root.
  - `test_v2_charges_potion_used_on_search_path`: the root action drinks a typical potion and the playout wins at full HP. The value is `0.95 − 0.4 × 12 / M`.
  - `test_v2_terminal_reason_from_info`.
  - `test_flat_v1_with_no_potions_policy`: scores as v1, plays out with `no-potions-v2`, and needs no root context.
- [ ] **Step 2: Run them to verify they fail,** then implement, then run them to verify they pass. Then run the full suite:
  `env -u SPIRE_LEARN_RUN_NATIVE_INTEGRATION -u SPIRE_LEARN_RUN_TORCHRL_INTEGRATION -u SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION PYTHONPATH=src python3.12 -B -m unittest discover -s tests`
  The only known failure is `test_complete_controller_projects_real_gate_baseline_training_and_time_cutoff`. No existing test file changes except the one hash test above.
- [ ] **Step 3: Commit** `feat: resource-aware value in the search teachers`.

### Task 4: The teacher entry runs named v2 teachers and records resources

**Files:**
- Modify `scripts/neural/search_teacher_entry.py`.
- Tests: modify `tests/unit/test_search_teacher_entry.py`. Update `Fight`, `process_job` and `SearchFightParityTests`, and the key set in `test_outputs_have_no_hidden_state`.

**Interfaces:**
- `--teacher` accepts any name in the given teachers JSON. The hard-coded choices go, and an unknown name is an error that lists the valid names.
- Dispatch on `row.get("kind") == "one-move"`, not on `name == "current"`.
  - `current`'s validation and identity dict stay byte-identical, so its `config_sha256` holds.
  - One-move rows accept optional `value_rule` (a `ValueRule` identity object) and `playout_policy`.
  - Tree rows accept the same inside `config`.
  - Both are converted with `ValueRule.from_identity`.
  - `provenance["teacher_config"]` and `config_sha256` use the JSON-safe identity.
- `--allow-uncalibrated`, off by default, lets a teacher marked `"calibrated": false` run on `--panel`. It is recorded as `provenance["uncalibrated_allowed"] = true`. Without it, today's guard holds.
- `_decision` builds `RootContext.from_observation(view, observation)` for v2 teachers. On a refused root it falls back to `view.playout_choice(observation, rng, policy)` whenever the teacher's `playout_policy` is not `uniform-v1`. Otherwise it keeps `heuristic_choice`.
- Fight rows gain the ints `hp_start`, `max_hp_start`, `max_hp_end`, `potions_start` and `potions_end`. They come from the reset and final observations, and they are present for censored fights and fights terminal at reset too. They also gain `potion_hp_spent`, a float: the potion charge in HP between reset and end. It always uses the fixed reference prices (12 HP, `fairy_potion` 30, `fruit_juice` 7.5) whatever the teacher's own prices, so every run is measured on the same yardstick.

- [ ] **Step 1: Write the failing tests:**
  - `test_v2_teacher_names_load_from_json`;
  - `test_current_hash_unchanged_after_dispatch_change`;
  - `test_uncalibrated_panel_run_needs_flag_and_is_recorded`;
  - `test_value_rule_round_trips_into_provenance`;
  - `test_refused_root_uses_policy_for_v2`;
  - `test_fight_row_records_resources`, covering censored and terminal-at-reset fights.
- [ ] **Step 2: Implement, run the tests and the full suite, and commit** `feat: run named v2 teachers and record fight resources`.

### Task 5: The audit report admits v2 teachers and applies the choice rule

**Files:**
- Modify `scripts/neural/teacher_audit.py`.
- Test: `tests/unit/test_teacher_audit.py`.

**Interfaces:**
- `--run NAME:PANEL:DIR` accepts any teacher name found in the run's provenance. Sorting puts the v1 teachers first in their current order, then other names alphabetically.
- `load_run` still requires `calibrated: true`, unless the report has `--choose-v2` and the run records `uncalibrated_allowed`. When the new fight-row fields are present, `load_run` validates their types: ints, with `potion_hp_spent` a finite float. Partial sets are rejected.
- `consistency --teacher` accepts any teacher name, as `report` does.
- G1, `teacher_choice` and `stop_rule` stay exactly as they are and run when the v1 inputs are present.
- `--choose-v2` adds a `v2_choice` block to the JSON and Markdown. `REPORT_SCHEMA` gains it additively. `stop_rule` is also reported for the v2-chosen teacher.
- Per teacher and panel, the report adds signed means over completed fights (wins, deaths, escapes):
  - HP lost: `(hp_start − hp_after_cleanup) / max_hp_start`, with a death counting `hp_after_cleanup` as 0;
  - potion HP spent: `potion_hp_spent / max_hp_start`;
  - resource loss: their sum;
  - plus the signed potions used per won fight (`potions_start − potions_end`), and the share of searched decisions with an exact top-value tie.
  - Censored fights are counted and listed by reason. Runs without the new fields show `-`. Max HP comes from the fight row when it is present.
- **`v2_choice` rule.** The controls and candidates are named on the command line: `--v1-control NAME` and `--candidate NAME`, repeatable. The control and every candidate must have runs on both panels, or the report raises `AuditInputError`.
  1. A candidate is **ineligible** on a panel if any of these holds:
     - it wins 6 or more fewer cases than the v1 control (counting censored as not won);
     - it is significantly worse than the control by a one-sided exact McNemar test at 0.10, with `only_control > only_candidate`. Use a dedicated helper (`one_sided_mcnemar(only_control, only_candidate)`, the upper tail over 2^n) that counts censored cases as not won and drops escaped cases. This differs from `PAIRED_OUTCOME`, which drops censored cases;
     - it has more than 3 censored cases.
  2. **Choose:** among eligible candidates, the lowest resource loss, as an unweighted mean over all completed fights of both panels. The report notes that 31 base states are shared between the panels.
  3. **Ties:** among candidates within 0.01 of the best (`≤ best + 0.01`), the first in `--candidate` order wins. Task 8 passes them cheapest-first in expected native cost: `current-v2`, `current-v2-sqrt`, `tree-1x-v2`.
  4. **No eligible candidate:** `chosen` is the `--v1-control` name, with `fell_back_to_control: true`.

- [ ] **Step 1: Write the failing tests:**
  - `test_v2_names_admitted_and_sorted`;
  - `test_uncalibrated_only_with_choose_v2`;
  - `test_resource_columns_signed_and_deaths_lose_all`;
  - `test_old_runs_show_dash`;
  - `test_choose_v2_eligibility_floor_mcnemar_and_censors`;
  - `test_choose_v2_tie_band_and_order`;
  - `test_g1_unchanged_with_v1_inputs`.
- [ ] **Step 2: Implement, run the tests and the full suite, and commit** `feat: v2 teacher report and choice rule`.

### Task 6: Configs, the native check and the UCB check

**Files:**
- Create `configs/neural/search-teacher-gym-1.2.4/teachers-v2.json`.
- Test: extend `EntryConfigTests` in `tests/unit/test_search_native_playouts.py`.

**Teachers.** All are `"calibrated": false` and use `playout_backend` `"native"` (Gym v1.2.5).
- **Control:** `current-v1-nopot`, which is `flat-v1` with `no-potions-v2` (native `tiered_verbs`). It isolates the playout policy.
- **Candidates**, all with `resource-v2` and `no-potions-v2`:
  - `current-v2` (32 samples, 80 playout steps, linear, potion_hp 12);
  - `current-v2-sqrt`;
  - `tree-1x-v2`: tree-1x's config (22 per action, `min_root_visits` 8, `worlds` 8, `max_depth` 4, `widening` (2.0, 0.5)), with `ucb_c` set in Step 2.
- **Sensitivity:** `current-v2-p10` and `current-v2-p15`, with potion_hp 10 and 15.

- [ ] **Step 1:** Write the failing test `test_v2_file_teachers_load_and_differ_from_v1_only_in_value_policy_calibration_and_ucb`, write the file, run the test to verify it passes, and commit `configs: v2 search teachers`.
- [ ] **Step 2: UCB check (controller, container).** The v2 win band is narrower than v1's −1 to 1, so `ucb_c` 1.0 may over-explore.
  - Run `tree-1x-v2` at `ucb_c` 0.25, 0.5 and 1.0 on `--train-sample 100`, seed 2026112602 (training split, never panel cases).
  - Fix the `ucb_c` with the lowest mean resource loss among values not 6 or more wins below the best. Ties go to 1.0.
  - Commit the value as a config change, reviewed by Sonnet.
- [ ] **Step 3: Native and Python agreement (controller, container).**
  - Write a temporary teachers file under `~/scratch/spire-learn/` with `current-v2-py`, identical to `current-v2` but on the Python backend.
  - Run both on `--train-sample 100`, seed 2026112603.
  - They must agree statistically: wins within 6, mean resource loss within 0.02 on fights both won, and within 0.04 overall. They are not bit-identical, because the RNG streams differ. Task 10 Step 3 already gives the exact check.
  - A larger gap stops the task. It points to a scoring or rule mismatch to find before the measurement.

### Task 7: Gym change request (controller) — DONE

Posted as `spire-9wa` on 2026-11-26 and delivered in Gym v1.2.5 (M42, PR #132), with the bead closed. Item 3, permanent-card deltas, was deferred by the Gym. File a new spire-learn-labelled bead with the wanted schema only when Learn is ready to score those effects. The original steps follow for the record.

- [ ] **Step 1: Draft a request for the Gym tracker:**
  1. A native playout rule with a never-pick verb set and a last-resort verb set (never `use_potion` or `discard_potion`; `end_turn` only when nothing else is legal), plus a note on how it consumes the RNG when only one action is legal.
  2. Additive summary fields: potions held (kinds), gold, and terminal `hp_after_cleanup` stated explicitly.
  3. Permanent-card `misc` and upgrade changes, for later scoring of Ritual Dagger, Genetic Algorithm and Lesson Learned.

  Ship it as an additive drop-in under the 2026-11-25 ruling. Include no Baalorlord data.
- [ ] **Step 2:** Post it only with the user's go-ahead, then read it back. After it ships, native v2 replaces the Python path for labelling, after a parity check of native against Python on the Task 6 sample.

### Task 8: Measurement runs and the dataset-teacher choice (controller)

- [ ] **Step 1: Launch the runs** on the Gym v1.2.5 runtime profile. Each uses `search_teacher_entry.py --recipe <v1.2.4 recipe> --panel P --seed 20261117 --teacher NAME --teachers <file> --workers 1 --parallel N --output DIR`, adding `--allow-uncalibrated` for the v2 file.
  - **Both panels:**
    - `current` (the v1 native re-run on v1.2.5: the same decisions as stage 1, now with resource fields; this also extends Task 9's determinism check to both panels);
    - `current-v1-nopot`;
    - `current-v2`;
    - `current-v2-sqrt`;
    - `tree-1x-v2`.
  - **Selection only:** `current-v2-p10` and `current-v2-p15`.
  - That is 12 runs.
  - **Expected cost:** the native one-move runs take about 3,800–4,400 CPU s each, as in stage 1. Playouts that keep their potions run longer. That is about 15–20 core-hours in total, or about 1–2 hours at 16 threads, with the stage-1 tail fight setting the wall time.
  - Run them concurrently in containers at 16 threads in total, with per-run timeouts of 43,200 s, from a detached script under `~/scratch/spire-learn/`.
  - Each run must exit 0 with `learner_changed` false and 256 cases. The repo stays untouched until all end.
- [ ] **Step 2: Report.**
  - First verify that the new `current` re-run reproduces stage-1 `current` exactly on both panels: the same decisions, outcomes and `hp_after_cleanup`. Any difference stops the task.
  - Run `teacher_audit.py report --choose-v2 --v1-control current --candidate current-v2 --candidate current-v2-sqrt --candidate tree-1x-v2`. Its inputs are the new `current` re-run in place of stage-1 `current`, the stage-1 tree runs, the new v2 and control runs, and PPO.
  - Check for zero faults, and that censored cases are reported by reason.
  - **Effect attribution,** with all runs native:
    - the playout policy: `current` against `current-v1-nopot`;
    - the value rule: `current-v1-nopot` against `current-v2`;
    - the curve: `current-v2` against `current-v2-sqrt`;
    - the potion price: p10, p12 and p15, on selection only.
- [ ] **Step 3: Write `docs/neural-teacher-value-v2.md`.**
  - **Provenance:** Gym revision and dirty state, learner revision, schema and profile, config hashes, seeds, CPU and wall budgets, RNG-001/TIME-001, and that results are simulator-only (no original-game transfer).
  - **Content:**
    - the formula and prices;
    - the v1.2.5 drop-in and its determinism check;
    - the native and Python agreement check;
    - the strength, HP-lost, potion, resource-loss and tie tables;
    - the attribution;
    - the curve and price sensitivity;
    - the UCB check;
    - the choice;
    - the deferred permanent-card request.
  - Aggregates only. Sonnet review, then commit `docs: teacher value v2 results and dataset-teacher choice`, open the PR and merge.
- [ ] **Step 4: Stop.** Report the choice to the user. Then write the parts 3–5 plan (dataset, students, PPO arm with the bounded decline diagnostic, gates G2–G4) around the chosen teacher.

### Task 9: Adopt Gym v1.2.5 as a drop-in (after Tasks 4–5 merge, before Task 10)

**Files:**
- Modify `src/spire_learn/neural/gym_contracts.py`: add the contract and `same_rules_epoch`.
- Modify the strict revision guards, which become epoch checks:
  - `src/spire_learn/neural/recipes.py`: `gym_validator` and `require_recipe_revision`;
  - `scripts/neural/search_teacher_entry.py`: the contract lookup, the build check and the recorded revision;
  - `scripts/neural/teacher_audit.py`: `SHARED_PROVENANCE`, `_shared_provenance` and the `run_consistency` live check;
  - `scripts/neural/baseline_entry.py` (around line 724);
  - `scripts/neural/boss_winnability.py` (around line 386).
- Create `configs/neural/gym-1.2.5/`, modelled on `configs/neural/gym-1.2.4/`: the runtime profile with the v1.2.5 overlay and `delivery_manifest_sha256` pinned, and the same templates.
- Modify `docs/neural-gym-v1-migration.md`: add a "Drop-in v1.2.5" section.
- Test: create `tests/unit/test_neural_gym_1_2_5.py`, following `tests/unit/test_neural_gym_1_2_4.py`.

**Interfaces:**
- `GYM_V1_2_5_CONTRACT`, with v1.2.5's revision and build identity.
  - It keeps `rules_epoch="gym-1.2.4"`, the same rules ids and every fingerprint of `GYM_V1_2_4_CONTRACT`.
  - It adds the capability flag `tiered_playouts=True`. The flag is False for every older contract.
- `same_rules_epoch(revision_a: str, revision_b: str) -> bool`. It is true only when both revisions are registered and share a rules epoch that is not `None`.
- **Guards.** Each strict check of a recorded revision against the live one becomes `same_rules_epoch(live, recorded)`. Three things change with it:
  - The entry looks up the contract of the live Gym, for the compact-view and `tiered_playouts` checks.
  - The entry records the live `gym_revision` and `gym_build` in provenance. `recipe_sha256` and `states_table_sha256` still point at the `gym-1.2.4` recipe and table.
  - `teacher_audit.py` keeps `rules_epoch` shared across runs. It checks that each run's `gym_revision` belongs to that epoch, and records `gym_revision` and `gym_build` per run instead of requiring one shared build. `run_consistency` keeps exact build parity against the run it replays.
- Records from a different epoch still fail closed.

- [ ] **Step 1: Write the failing tests:**
  - `test_v1_2_5_contract_identities` (the hashes in Global Constraints, plus the runtime profile's `delivery_manifest_sha256`);
  - `test_v1_2_5_shares_epoch_and_fingerprints_with_v1_2_4`;
  - `test_same_rules_epoch`;
  - `test_gym_1_2_4_table_and_recipe_accepted_on_v1_2_5`;
  - `test_checkpoints_cross_v1_2_4_and_v1_2_5`, using `validate_checkpoint_metadata` across the two profiles' templates, with `gym-1.2.0` still refused;
  - `test_other_epochs_still_refused`;
  - `test_tiered_playouts_flag_only_on_v1_2_5`;
  - `test_entry_records_live_revision`;
  - `test_audit_accepts_mixed_revisions_in_one_epoch`.
- [ ] **Step 2: Implement.** Stage the wheel from the local packet, verify both SHA-256 values and the manifest, and build the overlay the way v1.2.4 did. Clone tag v1.2.5 into `artifacts/development/neural-combat/gym-1.2.5/source-ref`. Run the tests and the full suite, then commit `feat: adopt Gym v1.2.5 as a drop-in in the gym-1.2.4 epoch`.
- [ ] **Step 3: Determinism check (controller, container).** This is the user's 2026-11-25 condition for a drop-in.
  - Check the release identity. The descriptor capture must be byte-identical to v1.2.4. Run one reset per mode and the card-metadata fingerprint.
  - Run v1 `current` on `--train-sample 200`, seed 2026112501, on v1.2.5. Compare it with the v1.2.4 run in `artifacts/development/neural-combat/gym-1.2.4/t3/cost-current`, on these projections:
    - `decisions.jsonl` without `cpu_seconds`;
    - `fights.jsonl` restricted to `case_id`, `outcome`, `censor_reason`, `steps` and `hp_after_cleanup`. Task 4's new columns and the timing fields are projected out.
  - They must be equal.
  - Any difference stops the plan and is filed as a spire-learn-labelled bead.
  - Record the check in the migration doc. Note that the Gym's `delivery.json` compares v1.2.5 with its predecessor build `673887a`, not with `673eb8f`. This check is what bridges `673eb8f` to `731fca0` for Learn.

### Task 10: Native v2 scoring on the tiered rule (after Task 9)

**Files:**
- Modify `src/spire_learn/neural/search_one_move.py`:
  - `check_value_options`;
  - `playout_branches` and `_score_playout`;
  - `_native_playouts`;
  - new helpers `playout_rule_id(policy)` and `require_tiered_playouts(contract)`.
- Modify `src/spire_learn/neural/search_teacher.py`: `_native_playout` and `config_identity`.
- Modify `src/spire_learn/neural/search_value.py`: add the summary-row scorers.
- Modify `scripts/neural/search_teacher_entry.py`: the one-move identity, the provenance `playout_rule_id`, and the live tiered check.
- Tests:
  - `tests/unit/test_search_native_playouts.py`;
  - `tests/unit/test_search_value_wiring.py`, updating `test_v2_guards`, which today refuses native v2 unconditionally;
  - create the gated test `tests/integration/test_search_value_native.py`.

**Interfaces:**
- `TIERED_PLAYOUT_RULE_ID = "spire-gym-public-playout-tiered-v1"`.
- `playout_rule_id(policy)` returns `PLAYOUT_RULE_ID` for `uniform-v1` and the tiered id for `no-potions-v2`. Both identity sites use it:
  - `config_identity`, for tree configs;
  - the entry's one-move identity and its provenance.

  Native v1 identities and hashes stay unchanged.
- **Pure config validation.** `check_value_options` stays pure and offline.
  - It allows the native backend with `no-potions-v2` and `resource-v2`.
  - `resource-v2` requires branch view `"full"` on both backends.
  - It still rejects `resource-v2` with `uniform-v1`.
- **The live capability check** happens only where a live Gym exists:
  - in `search_teacher_entry._run`, against the live contract, before preflight;
  - in `playout_branches`, before a tiered call, through a mockable helper in the style of `search_view._native_errors()`.

  An older Gym raises a `ValueError` naming the rule. `TeacherConfig.__post_init__` never imports `spire_lab`.
- **Native calls.** Under `no-potions-v2`, `playout_branches` calls `playout_many(..., rule="tiered_verbs", never_verbs=("use_potion", "discard_potion"), last_resort_verbs=("end_turn",), max_steps=..., playout_seed=...)`. Under `uniform-v1` it keeps today's call byte for byte, and `_score_playout` stays literally unchanged for `flat-v1`.
- **Summary scorers:** `summary_terminal_value(rule, root, result, index)` and `summary_cutoff_value(rule, root, result, index)`.
  - They take no `view`, because `RootContext` holds the max HP, the potions and the prices.
  - `player_hp` is HP after cleanup on terminal rows, and `player_max_hp` is the max HP.
  - Potion kinds come from the `potion_kind` rows where `potion_present` is set.
  - `e` comes from a shared helper, `summary_enemy_share(result, index)`, in `search_view.py`.
  - The reason is `result["end_reason"][index]`:
    - on terminated rows it must be one of `victory`, `death` or `escaped`, with `rewards` equal to +1, −1 or 0 respectively;
    - on live rows it must be `step_limit` or `max_steps`;
    - anything else raises.
- **Root step and native tree playouts.**
  - `_native_playouts` takes `(value_rule, playout_policy, root)`. Under `resource-v2` it scores root-step terminals and cut-offs with the observation scorers (`terminal_value` / `cutoff_value`). Under `flat-v1` it keeps `result.reward` / `0.5 * view.balance(...)` verbatim.
  - `_native_playout` in `search_teacher.py` receives the root context and passes the value rule, policy and root to `playout_branches`.

- [ ] **Step 1: Write the failing tests:**
  - `test_native_v1_call_and_hash_unchanged`;
  - `test_native_no_potions_uses_tiered_rule_with_explicit_sets`;
  - `test_playout_rule_id_in_identity_and_provenance`;
  - `test_summary_scoring_matches_observation_scoring` (equal values for equal states, including potions gained and spent);
  - `test_summary_reason_validation`;
  - `test_native_v2_root_step_terminal_scored_v2` (a lethal Feed and a Smoke Bomb at the root);
  - `test_native_v2_refused_without_tiered_contract` (offline, through the mockable helper);
  - `test_v2_guards`, updated.
- [ ] **Step 2: Implement, run the tests and the full suite, and commit** `feat: native resource-aware value on the tiered playout rule`.
- [ ] **Step 3: Exact parity (controller, container, gated test)** on v1.2.5, over 20 training-split fights:
  - Replay each native playout's `branch.actions()` with full-view stepping. Assert exact float equality between the summary scorers and the observation scorers, across wins, deaths, escapes, cut-offs and potion changes.
  - Assert that every action `playout_action(rule="tiered_verbs")` picks belongs to the tier that `View.playout_choice(..., "no-potions-v2")` would pick from.
