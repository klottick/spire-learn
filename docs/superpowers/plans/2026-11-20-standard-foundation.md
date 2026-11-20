# Standard Combat Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an Ironclad A20 combat policy on the Gym's Standard rules, using a curriculum built from Baalorlord's reconstructed per-floor fight states plus a card and relic fuzz ladder.

**Architecture:**
- A pure, Gym-free reconstruction module turns run files into per-floor fight states.
- An extraction script converts them to `standard-combat-v1` entries, checks them against the Gym, and writes a local, hashed table.
- A new generator (`baalorlord-standard-v1`) turns that table into deterministic train shards and panels.
- The existing stream, collector and baseline machinery are generalized to run it in the existing `standard-standalone` mode.

**Tech Stack:** Python 3.12, the existing `spire_learn.neural` package, Spire Gym v1.1.1 (`e3c13836`) in the offline runtime container, and `unittest`: `PYTHONPATH=src python3.12 -B -m unittest discover -s tests`, per docs/development.md.

**Spec:** `docs/superpowers/specs/2026-11-20-standard-foundation-design.md`

## Global Constraints

- **Gym checkout.** Gym revision `e3c13836bd1458be8a87b93160f5c350f78fcd80` (rules epoch `gym-1.1.1`). Never modify `/home/klottick/repos/spire-lab`.
- **Run data stays local:**
  - raw runs are in `artifacts/development/neural-combat/baalorlord-runs-20261119/` (`raw/`, and `raw-other/{silent,defect,watcher}/`);
  - nothing under `artifacts/` is ever committed;
  - the reconstructed table, shards and panels are local and regenerable.
- **Untrusted input.** Run files are parsed only with `json` and run with `python3.12 -I`. Tests use hand-written synthetic fixtures under `tests/fixtures/neural/`, never real run data.
- **Public observations only** for the actor and critic. Gym faults are errors, and accepted effects are never retried.
- **Splits are by whole run:** a `run_id` hash assigns 80 / 10 / 10 to train / selection / final. No state or fuzzed variant from a selection or final run ever enters training.
- **Character and ascension:** the curriculum is Ironclad only, at ascension 20. Reconstruction covers all four characters.
- **Fuzz ladder:**
  - Level *k* replaces *k* cards and *k* non-starter relics, capped by what exists.
  - Replacement pools are cards and relics seen in train-split runs of that character.
  - Bottle relics are excluded from the pool. Replacing a bottle leaves its card unbottled; replacing a bottled card replaces its bottle too.
  - New relics take the Gym's default counters. Curses seen in his decks are in the card pool.
  - The schedule over quarters of training is {0}, {0,1}, {0,1,2}, {0,1,2,4}.
- **Potions.** A state with no potion logs, or an ambiguous potion state, is kept with no potions and tagged `potions_assumed_empty`.
- **Burning.** An elite is `burning` only on the floor in `green_key_taken_log`.
- **Drop rule.** A state that can't be rebuilt exactly is dropped and counted by reason. If more than 10% of a character's states drop for reasons other than scope exclusions (event fights with no Gym encounter row), stop and report.
- **Panels:**
  - selection panel of 256 cases from selection runs, stratified by act × kind, 64 per level at levels 0, 1, 2 and 4;
  - final panel built but not scored;
  - each panel deterministic and pinned by hash.
- **Escapes.** `escaped` (Smoke Bomb) is neither a win nor a loss; it is reported separately.
- **Throughput gate.** Below 100 decisions per second at the chosen row count means the throughput fix task (7b) runs before the baseline.
- **Commit trailer.** Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Delivery.** One PR per task, Sonnet review, merge when clean. No dependency installs.

## Review Focus

1. **A run whose `master_deck` doesn't match its history.** For example, a deck-changing event the undo logic doesn't know. The state must be dropped with reason `deck_unrecoverable`, not emitted with a wrong deck (Task 2).
2. **A Gym rejection of one entry in a catalog.** Probe entries one at a time, or bisect, so one bad entry drops only itself, with its reason, rather than failing or dropping the whole batch (Task 3).
3. **Fuzzing a deck smaller than *k*, or a relic list holding only the starter.** Level *k* must cap without error, and the case records the actual counts replaced (Task 5).
4. **Resume mid-schedule.** A run resumed from a shard past a schedule boundary must draw the same levels as an uninterrupted run (Task 6).
5. **A panel case censored by `case_actions`** on a long A20 boss fight. It is censored, not an operational failure, exactly as in PR #20 (Task 6, panel evaluation).

---

### Task 1: Gym name mapping

**Files:**
- Create: `src/spire_learn/neural/standard_names.py`
- Create: `configs/neural/standard/name-aliases.json` (name pairs only)
- Test: `tests/unit/test_neural_standard_names.py`

**Interfaces:**
- Produces:
  - `load_aliases(path: Path) -> NameAliases`;
  - `NameAliases.card(run_name: str) -> tuple[str, int] | None`, which returns the Gym card id and upgrade count (`"Strike_R+1"` → (the Gym's Ironclad Strike id, 1), `"Searing Blow+12"` → upgrades 12);
  - `NameAliases.relic(run_name) -> str | None` and `NameAliases.potion(run_name) -> str | None`;
  - `NameAliases.sha256: str`.
- Card ids come from the Gym's card metadata, which Learn already pins via `card_metadata_fingerprint`. The table supplies only the names that differ from the normalized name, for example `Ghostly` → `ghostly_armor`. Relic and potion entries are explicit pairs for all four characters. The normalization from `scripts/neural/real_boss_decks.py:normalize` is reused.

- [ ] **Step 1: Write failing tests.** Cover:
  - `test_card_upgrades_parsed`: "Bash+1" → `(id, 1)`;
  - `test_starter_aliases`: `Strike_R`, `Defend_G`, `Strike_B`, `Strike_P`;
  - `test_unknown_returns_none`;
  - `test_alias_table_hash_stable`;
  - `test_alias_targets_exist_in_gym`, gated like other Gym-backed tests: every relic and potion target is accepted by a one-entry Standard catalog for its character.
- [ ] **Step 2: Run them and confirm they fail.**

  ```
  PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_standard_names -v
  ```

  Expected: FAIL (module missing).
- [ ] **Step 3: Implement and build the table.**
  - Seed the relic and potion list by collecting every relic and potion name in all 2,400 runs with a scratch script outside the repo. Map each to a Gym id, and check each with the gated test inside the runtime container.
  - Print the unmapped names in the report. Unmapped names are allowed; they become drops later.
- [ ] **Step 4: Confirm the tests pass**, both the plain test run and the gated test in the container.
- [ ] **Step 5: Commit.** `feat: Gym name aliases for run reconstruction`.

### Task 2: Per-floor reconstruction (pure, no Gym)

**Files:**
- Create: `src/spire_learn/neural/run_states.py`
- Modify: `scripts/neural/real_boss_decks.py`. Import the moved rebuild helpers (`deck_at`, `relics_at`, `_later_card_changes`, the untracked-relic sets) from `run_states`. The existing real boss table must regenerate byte-identically (sha `fd28ab417875fb80955076d55e9cdb84583b2aadcaca5d615b7a1df8c7c7526a`).
- Test: `tests/unit/test_neural_run_states.py`, with fixtures `tests/fixtures/neural/runs/*.json` (synthetic).

**Interfaces:**
- Consumes: `NameAliases` (Task 1).
- Produces:
  - `fight_floors(run: dict) -> list[FightFloor]`. `FightFloor` has `floor`, `enemies`, `room_kind` (`monster`, `elite`, `boss` or `event`), `act`, and `won`.
  - `state_at(run: dict, fight: FightFloor, names: NameAliases) -> FightState | Drop`.
    - `FightState` fields: `run_id`, `character`, `floor`, `act`, `room_kind`, `enemies`, `deck: list[{card, upgrades, misc}]`, `relics: list[str]` (acquisition order), `relic_counters: dict`, `bottled_cards: dict`, `potions: list[str]`, `hp`, `max_hp`, `gold`, `burning: bool`, `won`, and `tags: list[str]`.
    - `Drop(reason: str, detail: str)`.
- **Floor convention:** apply every change recorded on floors before F. Potions used or discarded on F were held entering the fight. Everything obtained on F is post-fight.
- **Field sources:**
  - Bottled cards and counters come from `relic_stats`.
  - Potions come from `potions_obtained`, `potion_use_per_floor`, `potion_discard_per_floor`, `potions_obtained_alchemize` and `potions_obtained_entropic_brew`.
  - `burning` comes from `green_key_taken_log`.
- **Drop reasons:** `deck_unrecoverable`, `untracked_deck_relic`, `unmapped_card:<name>`, `unmapped_relic:<name>`, `bottle_card_missing` and `event_fight_reordered`.
- **Tag:** `potions_assumed_empty`.

- [ ] **Step 1: Write failing tests**, each against a synthetic fixture:
  - `test_deck_before_floor_undoes_picks_purges_smiths`;
  - `test_potions_used_on_fight_floor_were_held`;
  - `test_potion_obtained_on_floor_is_post_fight`;
  - `test_missing_potion_logs_keep_state_with_no_potions_tagged`;
  - `test_more_potions_than_slots_keeps_state_empty_tagged`;
  - `test_potion_belt_allows_four_slots` (2 at A20, plus 2);
  - `test_bottled_card_from_relic_stats`;
  - `test_bottle_card_absent_drops`;
  - `test_burning_only_on_green_key_floor`;
  - `test_hp_entering_floor`;
  - `test_lost_fight_kept_with_won_false`;
  - `test_unmapped_card_drops_with_name`;
  - `test_master_deck_mismatch_drops_deck_unrecoverable` (Review Focus 1);
  - `test_real_boss_table_regenerates_identically`, which is gated on the local table and skips when it is absent.
- [ ] **Step 2: Run them and confirm they fail.**

  ```
  PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_run_states -v
  ```

  Expected: FAIL.
- [ ] **Step 3: Implement** `fight_floors` and `state_at`, and move the helpers.
- [ ] **Step 4: Confirm the tests pass.** Then run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_real_boss_decks -v`, which must also pass.
- [ ] **Step 5: Commit.** `feat: per-floor fight state reconstruction`.

### Task 3: Extraction, Gym acceptance and the state table

**Files:**
- Create: `scripts/neural/run_states.py`, with the `extract` subcommand.
- Test: `tests/unit/test_neural_run_states_extract.py`

**Interfaces:**
- Consumes: `fight_floors`, `state_at` (Task 2), and `NameAliases` (Task 1).
- Produces:
  - **Command:**

    ```
    run_states.py extract --raw DIR [DIR...] --aliases PATH --output FILE --report FILE
    ```

  - **Table:** a canonical JSON document, sha-reproducible:

    ```
    {"schema": "spire-learn-run-states-v1",
     "aliases_sha256", "gym_revision",
     "states": [{...FightState, "entry": <standard-combat-v1 entry>}]}
    ```

    States are sorted by `(character, run_id, floor)`.
  - **Report:** counts per character of states kept, dropped by reason and scope-excluded, plus `potions_assumed_empty` counts and the 10% drop-rule verdict.
  - **Conversion:** `to_standard_entry(state) -> dict`, with `character`, `ascension` 20, `encounter`, `hp`, `max_hp`, `gold`, `deck`, `relics`, `relic_counters`, `bottled_cards` and `potions`. Burning elites get a `burning` kind.
    - The run doesn't record which burning buff applied. *Ruling, from the plan:* draw it uniformly from the Gym's 4 kinds, using a hash of `(run_id, floor)`, and tag `burning_kind_sampled`.
  - **Encounter mapping:** run enemy names map to the Gym's encounter rows by the Gym's environment spec support listing. An event fight with no row gets `scope_excluded:<name>`.
- **Gym acceptance.** Validate in catalogs of at most 256 entries. If a catalog is rejected, bisect to find the bad entries, and drop each with the Gym's error as its reason (Review Focus 2).

- [ ] **Step 1: Write failing tests:**
  - `test_canonical_output_sha_stable`;
  - `test_report_counts_and_drop_rule_verdict`;
  - `test_scope_excluded_not_in_drop_denominator`;
  - `test_burning_kind_deterministic`;
  - `test_one_rejected_entry_drops_only_itself`, which uses a fake validator injected by parameter.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Gym validation is injected as `validate(catalog) -> None | raises`. The CLI binds it to the Gym inside the runtime container.
- [ ] **Step 4: Confirm the tests pass.**
- [ ] **Step 5: Run the real extraction (controller).** Run it inside the runtime container over `raw/` and all of `raw-other/*`. Write the table to `artifacts/development/neural-combat/run-states/run-states.json` and the report beside it.
  - If any character trips the 10% rule, stop: investigate before Task 5.
  - Record the table's sha in the PR description.
- [ ] **Step 6: Commit.** `feat: extract Gym-validated fight states from runs`.

### Task 4: Baalorlord statistics report

**Files:**
- Modify: `scripts/neural/run_states.py`, adding the `stats --table FILE --output FILE` subcommand.
- Create: `docs/baalorlord-run-statistics.md` (aggregates only)
- Test: `tests/unit/test_neural_run_states_stats.py`

**Interfaces:**
- Consumes: the Task 3 table schema.
- Produces: a JSON of aggregates, per character × act × room kind:
  - states;
  - deck size (median and IQR);
  - upgrade fraction;
  - card-type mix;
  - top 15 cards and top 15 relics, by share of decks;
  - mean potions;
  - HP fraction;
  - his win rate;
  - `potions_assumed_empty` share.

- [ ] **Step 1: Write a failing test**, `test_aggregates_on_synthetic_table`. On a three-state synthetic table it gives the exact medians and shares.
- [ ] **Step 2: Run it and confirm it fails.**
- [ ] **Step 3: Implement it, run it on the real table, and write the doc.** The doc presents the numbers as tables, with a short "what his decks look like" reading for each character. It contains no per-run rows.
- [ ] **Step 4: Confirm the test passes.**
- [ ] **Step 5: Commit.** `docs: Baalorlord per-floor deck statistics`.

### Task 5: The `baalorlord-standard-v1` generator

**Files:**
- Create: `src/spire_learn/neural/standard_stream.py`
- Create: `configs/neural/baalorlord-standard-v1/recipe.json`
- Test: `tests/unit/test_neural_standard_stream.py`

**Interfaces:**
- Consumes: the Task 3 table, loaded by path and sha.
- Produces:
  - **`StandardRecipe`**, loaded by `load_standard_recipe(path) -> StandardRecipe`. Recipe fields:
    - `schema`;
    - `generator_version` = `"baalorlord-standard-v1"`;
    - `states_table` {path, sha256};
    - `character` = `"ironclad"`;
    - `ascension` = 20;
    - `generator_seed`;
    - `shard_size` (≤ 256);
    - `split_fractions` = [0.8, 0.1, 0.1];
    - `fuzz_levels_max` = 4;
    - `fuzz_schedule` = `[{"from_shard": 0, "levels": [0]}, {"from_shard": Q1, "levels": [0,1]}, {"from_shard": Q2, "levels": [0,1,2]}, {"from_shard": Q3, "levels": [0,1,2,4]}]`;
    - `redraw_bound`.

    It also exposes `sha256`, `payload`, `category_keys` (`room:<act>:<kind>` and `level:<k>`), and `mode` = `"standard-standalone"`.
  - `run_split(run_id: str) -> "train" | "selection" | "final"`.
  - `fuzz(state, level: int, pools, rng) -> tuple[entry, FuzzRecord]`. `FuzzRecord` has `cards_replaced` and `relics_replaced`, the actual counts.
  - `generate_standard_shard(recipe, *, split, shard_index, arm="mixed") -> ScenarioShard`, using the existing `ScenarioShard` type. Cases are `standard-standalone` cases (`common | {"entry"}`, no `entry_id`), with annotations `{run_id, floor, act, room_kind, level, fuzz, tags}`.
  - `build_standard_panel(recipe, split: "selection" | "final") -> Mapping`: 256 cases, 64 per level at levels 0, 1, 2 and 4. Each level is stratified by act × kind in proportion to the available base states, with shortfalls redistributed within the level. Base states are distinct within a level.
- **Q1–Q3.** `from_shard` thresholds are quarters of the baseline's expected shard count. Task 9 sets them, so the recipe ships with placeholders and Task 9 fixes the values.
  - *Ruling, from the plan:* the quarter boundaries are by shard index, because the stream consumes shards in order. The counting unit is shards, not decisions.

- [ ] **Step 1: Write failing tests:**
  - `test_split_by_run_disjoint`: no `run_id` appears in two splits, and the fractions are within ±3% on 1,000 synthetic ids;
  - `test_train_shards_contain_only_train_runs`;
  - `test_level_zero_is_exact_state`;
  - `test_fuzz_level_k_changes_exactly_k_cards_and_k_relics`;
  - `test_fuzz_keeps_starter_relic`;
  - `test_fuzz_caps_on_small_deck_and_starter_only_relics` (Review Focus 3);
  - `test_fuzz_pool_excludes_bottles`;
  - `test_replacing_bottled_card_replaces_bottle`;
  - `test_schedule_levels_by_shard_index`;
  - `test_shard_deterministic`;
  - `test_panel_quotas_and_distinct_bases`;
  - `test_panel_deterministic_hash`;
  - `test_missing_or_mismatched_table_fails_closed`.

  All of these use a synthetic states table, with a fake validator for the redraw path.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement it, plus the recipe JSON**, pointing at `../../../artifacts/development/neural-combat/run-states/run-states.json` with the Task 3 sha. Tests that load the shipped recipe skip when the table is absent.
- [ ] **Step 4: Confirm the tests pass.** Then generate one real train shard and one panel inside the runtime container, and confirm the Gym accepts both.
- [ ] **Step 5: Commit.** `feat: baalorlord-standard-v1 curriculum generator`.

### Task 6: Run the stream in `standard-standalone` mode

**Files:**
- Modify:
  - `src/spire_learn/neural/collector.py:131-135`;
  - `curriculum.py:201`;
  - `stream_resume.py:116-177`;
  - `stream_runner.py:18-175`;
  - `calibration.py:80-95`;
  - `config.py:290-291, 333-336`;
  - `scripts/neural/baseline_entry.py`, for the panel builder dispatch and per-panel `case_actions`;
  - anything else the implementer finds that hard-codes `"standalone"` on the stream path.
- Create: `src/spire_learn/neural/recipes.py`
- Test: `tests/unit/test_neural_standard_stream_runtime.py`

**Interfaces:**
- Consumes: `StandardRecipe`, `generate_standard_shard` and `build_standard_panel` (Task 5).
- Produces:
  - `load_recipe(path) -> ScenarioRecipe | StandardRecipe`, dispatching on `generator_version`;
  - `generate_shard(recipe, *, split, shard_index, arm) -> ScenarioShard`;
  - `build_panel(recipe, split) -> Mapping`, which for M10 recipes is the existing development panel.

  Every stream, resume, collector and baseline caller uses these.
- **Standard identity.** Standard configs declare a `standard-standalone` mapping, with new `features_id` and `model_id` values for the Standard view (for example `spire-learn-combat-features-standard-v1` and `spire-learn-combat-ff-standard-v1`). The common-view fingerprint is no longer required in Standard-only configs.
- **Outcomes.** `escaped` counts as neither a win nor a loss, and is reported. Censoring follows `is_censored_case_report`.
- **M10 unchanged.** M10 recipes and configs behave byte-identically: the existing pins and tests pass unchanged.

- [ ] **Step 1: Write failing tests:**
  - `test_load_recipe_dispatch`;
  - `test_m10_paths_unchanged`, which runs the existing stream pins;
  - `test_standard_config_validates_without_common_fingerprint`;
  - `test_resume_across_schedule_boundary_matches_uninterrupted` (Review Focus 4);
  - `test_escaped_reported_not_scored`;
  - `test_long_case_censored_not_failure` (Review Focus 5);
  - `test_standard_stream_smoke`, gated on the Gym: 2 updates at 16 rows on a synthetic table, which must finish cleanly.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** This is the judgment task, so the implementer uses the most capable model.
- [ ] **Step 4: Confirm the whole suite passes.** Run `PYTHONPATH=src python3.12 -B -m unittest discover -s tests`. The only allowed failure is the known pre-existing one, `test_complete_controller_projects_real_gate_baseline_training_and_time_cutoff`. Then run the gated smoke test in the runtime container.
- [ ] **Step 5: Commit.** `feat: run the curriculum stream in standard-standalone mode`.

### Task 7: Throughput measurement (7b only if needed)

**Files:**
- Create: `configs/neural/profile-standard-1.1.1/`
- Create: `docs/neural-throughput-standard.md`

- [ ] **Step 1: Profile.** Run `scripts/neural/profile_entry.py` on the Standard stream at 16 and 64 rows, in the runtime container on the GPU with the `gpu_wait` gate. Record decisions per second and the phase split against the M10 figure (about 285 decisions per second at 16 rows).
- [ ] **Step 2: Write the doc and commit.** `docs: Standard combat throughput`.
- [ ] **Step 3 (7b): Fix throughput, only if below 100 decisions per second at the chosen row count.**
  - First profile the hotspot. Then apply a fix in the style of PRs #8, #10 and #13: a fast path for validation, or trimming rows known to be empty.
  - Include a CPU byte-identity equivalence test, then re-measure.
  - It is a separate PR: `perf: Standard view <fix>`.

### Task 8: Winnability reference on the selection panel

**Files:**
- Modify: `scripts/neural/boss_winnability.py`. Add a `--standard-panel RECIPE` mode that plays panel entries directly as Standard entries, with no conversion.
- Create: `docs/neural-standard-foundation-results.md`, the winnability section.
- Test: extend `tests/unit/test_neural_boss_winnability.py` with `test_standard_panel_cases_passed_through_unchanged`.

- [ ] **Step 1: Write the failing test, implement, and confirm it passes.**
- [ ] **Step 2: Run search on the selection panel.** Use 32 samples, 80 playout steps and `max_steps` 1200. Cut the budget if one pass would take more than about 6 hours of CPU time.
- [ ] **Step 3: Write the doc section.**
  - Show search wins per act × kind × level, beside his recorded win rate on the level-0 cases.
  - If a stratum has 0 search wins and 0 of his wins, it is reported and excluded from the schedule, through a recipe change in this task.
- [ ] **Step 4: Commit.** `feat: Standard panel winnability reference`.

### Task 9: Baseline (two seed pairs) and results

**Files:**
- Create: `configs/neural/baseline-standard-1.1.1/` and `configs/neural/baseline-standard-1.1.1-r2/`, each with `plan.json` and `template.toml`.
- Modify: `configs/neural/baalorlord-standard-v1/recipe.json`, fixing `from_shard` Q1–Q3.
- Modify: `docs/neural-standard-foundation-results.md` and `README.md`.

- [ ] **Step 1: Write the configs.**
  - Copy the `baseline-gym-1.1.1-cv2` model and PPO settings, and its seeds (r1 and r2).
  - Set the budgets from Task 7's measured throughput, so that training fits in about 3.5 hours of wall time excluding waits.
  - Q1–Q3 are 25%, 50% and 75% of the expected shard count.
  - Panels: the selection panel is primary, with `case_actions` 1024. Retain the checkpoints at fresh, 25%, 50%, 75% and final.
- [ ] **Step 2: Run both baselines sequentially** in the runtime container with `--ledger-bind` and the `gpu_wait` gate. GPU time not used by Hermes is allowed. The ledger records `learner_changed` false and `target_reached`.
- [ ] **Step 3: Write the results.**
  - Learning curves.
  - Per act × kind × level wins against search and against his recorded wins.
  - Fresh-vs-final exact McNemar tests.
  - Replicate agreement.
  - Escapes and censoring.
  - The reconstruction drop summary.
  - Qualifications: simulator only, RNG-001/TIME-001, one player's data, burning-elite labelling, sampled burning kind, and potions assumed empty.

  Link the results doc from the README.
- [ ] **Step 4: Commit and open the PR.** `docs: Standard foundation baseline results`.
