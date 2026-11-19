# Curriculum v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a v2 curriculum whose late boss and elite fights are winnable (checked by a search gate), reweight what stays unwinnable, add a potential-based HP-shaped objective, and run four fresh-weight arms evaluated on the v1 and v2 panels.

**Architecture:**
- `scenario_stream` dispatches on `generator_version`. v1 stays byte-identical, and v2 adds encounter-first stage choice, prepared boss/elite decks and tier weights, all from new recipe tables.
- The winnability diagnostic gains a panel mode for ceilings.
- The collector applies an optional shaped reward keyed by a new objective id.
- `baseline_entry` evaluates extra panels.

**Tech Stack:** Python 3.12 stdlib (generator, diagnostic, entry); Torch only where it already runs (collector inside the runtime); Gym v1.1.1 via the runtime container.

**Spec:** `docs/superpowers/specs/2026-11-19-curriculum-v2-design.md`

## Global Constraints

- The v1 recipe hash, v1 shard manifests and the v1 development panel must be byte-identical to `main`.
- Generator version strings: `"ironclad-curriculum-v1"` (unchanged) and `"ironclad-curriculum-v2"`.
- New recipe tables `prepared_profiles` and `tier_weights` are accepted only when `generator_version == "ironclad-curriculum-v2"`. They are part of the payload, and so the hash, only for v2.
- Objective ids: `spire-learn-combat-winloss-v1` (unchanged) and `spire-learn-combat-winloss-hpshaped-v1`.
- Configs without `[objective]` behave exactly as now.
- Gate thresholds are search wins of at least: act 1 boss 5/10, act 2 boss 4/10, act 3 boss 3/10, act 3 elite 3/10, act 1 elite 8/10. The heuristic must stay below 9/10 on every boss tier. There are at most 6 tuning iterations, after which stop and report.
- Reweighting: tiers with gate search wins below 2/10 get weight 0.25; all others get 1.0.
- Shaping: β = 0.5, γ = the config's PPO gamma, and Φ = 0 on terminal successors.
- No Gym checkout changes, no new dependencies. Search is diagnostic only.
- Never commit or edit while a `runtime_container.py` run is in progress.
- Each task lands as its own PR, with a Sonnet review and a merge when clean. Tasks 1 and 4 also get a Sol 6.1 high adversarial review.

## Review Focus

1. **v1 bytes.** Any refactor of `_build` that changes RNG draw order or field order silently changes v1 shards and invalidates every v1 checkpoint binding. Task 1 pins the v1 recipe hash, shard 0 and 37 manifest hashes, and the panel manifest hash.
2. **Prepared-deck validity.** Every generated v2 entry must pass `support.validate_entry`, keep the reserved-pair exclusion and the composition split, and stay within deck bounds. Task 1 tests 4 full v2 shards.
3. **Shaping on padded and dead enemies.** Unoccupied rows and `max_hp = 0` must not divide by zero or count toward Φ. Task 4 tests these frames.
4. **Shaped reward at task boundaries.** Φ = 0 on a terminal successor, with no shaping across a reset. Task 4 tests a terminal and a reset transition.
5. **Extra panels on failure.** An `additional_panels` evaluation fault must not mark the primary result complete. Task 5 tests a failing extra panel.

---

### Task 1: Generator v2 (encounter-first stage, prepared boss/elite decks, tier weights)

**Files:**
- Modify: `src/spire_learn/neural/scenario_stream.py` (`ScenarioRecipe`, `load_scenario_recipe`, `payload`, `_build`)
- Create: `configs/neural/curriculum-v2/recipe.json`, with `support_file` set to `"../curriculum-v1/support.json"`
- Test: `tests/unit/test_neural_scenario_stream_v2.py`

**Interfaces:**
- `ScenarioRecipe` gains `prepared_profiles: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None` and `tier_weights: Mapping[str, float] | None = None`. Both must be None for v1 and present for v2.
- `prepared_profiles[act][kind]`, with act in {act1, act2, act3} and kind in {boss, elite}, has exact keys:
  - `core_cards: [min, max]`, ints in 0..12;
  - `starter_removals: int` in 0..9;
  - `upgrade_percent: int` in 0..100;
  - `rarity_weights: {starter, common, uncommon, rare}`, non-negative ints;
  - `hp_floor_percent: int` in 1..100.
- `tier_weights`: keys from the support's encounter tiers (`act{n}:{kind}`), values finite in (0, 1].
- `generate_scenario_shard` signature is unchanged.

- [ ] **Step 1: Write failing tests.**
  - `test_v1_bytes_are_unchanged` asserts the v1 `recipe.sha256` and the `manifest_sha256` of train shards 0 and 37 and of `build_development_panel(v1)['manifest']` equal the values captured from `main` before editing. Capture them with a one-off script and paste them as literals.
  - `test_v2_requires_and_v1_rejects_v2_tables`.
  - `test_v2_stage_follows_encounter_act`: every v2 case has deck_stage early, middle or developed for act 1, 2 or 3–4.
  - `test_v2_prepared_decks_follow_profile`: for every boss and elite case in v2 shards 0–3, the deck is within bounds and has at least `core_cards[0]` cards from one mechanic group; hp ≥ `hp_floor_percent`% of max_hp; validate_entry passes; no reserved pair in train.
  - `test_tier_weights_shift_the_noncoverage_draw`: with act3:boss weight 0.25 vs 1.0, the act3:boss share over 8 shards falls; the first 11 coverage entries are unchanged.
  - `test_v2_shards_are_deterministic_and_unique`.
- [ ] **Step 2:** Run the tests and see them fail: `PYTHONPATH=src:. python3.12 -m unittest tests.unit.test_neural_scenario_stream_v2`.
- [ ] **Step 3: Implement.**
  - Keep the v1 `_build` body untouched. Route v2 to `_build_v2`, which copies the v1 loop.
  - Draw the encounter first, using `tier_weights` via `encounter_rng.choices` for the non-coverage draw, then set the stage from its act.
  - For boss and elite kinds, build the prepared deck:
    - remove Strikes first, then other starters, up to the profile's `starter_removals`;
    - add `core_cards` from one deck_rng-chosen mechanic group, weighted by the profile's `rarity_weights`;
    - fill to the stage size with the profile's rarity weights;
    - upgrade each card with probability `upgrade_percent`;
    - set hp to `max(hp, ceil(max_hp*hp_floor_percent/100))`.
  - Use RNG labels distinct from v1's (`f"v2-deck:{local}"` and so on).
  - Initial knobs:
    - act2 boss: core 3–5, removals 4, upgrades 60, weights {0,4,4,2}, hp floor 70;
    - act3 boss and act3 elite: core 5–8, removals 6, upgrades 80, weights {0,3,5,3}, hp floor 75;
    - act1 boss: core 2–4, removals 3, upgrades 40, hp floor 65;
    - elites one step below their act's boss.
  - Set `tier_weights` to all 1.0 initially.
- [ ] **Step 4:** Run the new tests and `tests.unit.test_neural_scenario_stream`, `test_neural_stream_caches` and `test_neural_curriculum`. All pass.
- [ ] **Step 5:** Commit as `feat: curriculum v2 generator with prepared boss and elite decks`.

### Task 2: Gate and tuning, plus the panel ceiling mode

**Files:**
- Modify: `scripts/neural/boss_winnability.py`, adding `--panel`, `panel_ceiling()` and a `gate_report()`
- Modify: `configs/neural/curriculum-v2/recipe.json`, with tuned knobs and `tier_weights`
- Test: `tests/unit/test_boss_winnability.py`

**Interfaces:**
- `--panel` makes the script build `build_development_panel(recipe)` and play all 64 cases with the search player only. It writes `panel.jsonl` and `panel_ceiling.json` as `{search_wins, cases, by_tier, by_cohort}`.
- `gate_report(by_tier) -> {passed: bool, failures: [str]}` applies the Global Constraints thresholds.

- [ ] **Step 1:** Write failing tests:
  - `test_gate_report_thresholds` covers the exact pass and fail edges of every threshold;
  - `test_panel_mode_plays_every_panel_case_with_search_only`, with fakes.
- [ ] **Step 2:** Implement, and confirm the tests pass.
- [ ] **Step 3: Gate loop.** This is controller work, not a subagent's. For iteration i ≤ 6, run the gate in the container on the v2 recipe:
  `scripts/neural/boss_winnability.py --recipe configs/neural/curriculum-v2/recipe.json --output <artifacts>/curriculum-v2-gate-<i>`.
  If it fails, adjust only `prepared_profiles` knobs and re-run. Commit each knob change before its run. If the gate passes, set `tier_weights` per the reweighting rule from that run's by_tier.
- [ ] **Step 4:** Run `--panel` for the v1 and v2 recipes, and record both ceilings.
- [ ] **Step 5:** Commit as `feat: gate curriculum v2 and record panel ceilings`. Add the gate and ceiling summary to `docs/neural-curriculum-v2-results.md`, which this task starts.

### Task 3: v2 baseline templates and extra-panel evaluation

**Files:**
- Modify: `scripts/neural/baseline_entry.py`, where `load_plan` accepts optional `additional_panels: [recipe path]` and `_evaluate` handles the extra panels
- Create:
  - `configs/neural/baseline-gym-1.1.1-cv2/{plan.json,template.toml}`, with replicate-1 seeds;
  - `configs/neural/baseline-gym-1.1.1-cv2-r2/…`, with replicate-2 seeds.
  Both use recipe `curriculum-v2`, `additional_panels: ["configs/neural/curriculum-v1/recipe.json"]`, and `gpu_wait` as in the r2 plan.
- Test: `tests/unit/test_baseline_entry.py`

**Interfaces:**
- Extra panels are evaluated for `final` and `fresh-initial` only. They go under `summary['additional_panels'][<recipe path>] = {'evaluations': {label: …}, 'paired_fresh_vs_final': …}`.
- `result == 'complete'` additionally requires every extra-panel evaluation to be complete.

- [ ] **Step 1:** Write failing tests:
  - `test_additional_panel_is_evaluated_for_final_and_fresh`;
  - `test_failing_additional_panel_keeps_result_incomplete`;
  - `test_cv2_plans_differ_from_r1_r2_only_in_recipe_and_panels`.
- [ ] **Step 2:** Implement, and confirm the tests pass.
- [ ] **Step 3:** Commit as `feat: evaluate additional panels and add curriculum v2 baseline plans`.

### Task 4: HP-shaped objective

**Files:**
- Modify:
  - `src/spire_learn/neural/config.py`: an optional `[objective]` table with `kind = "hp_shaped"`, `coefficient` finite in (0, 1]; adds the objective id to the allowed set.
  - `src/spire_learn/neural/returns.py`: `hp_potential(frame, schema) -> float` and `shaped_reward(reward, source_frame, successor_frame, *, terminal, gamma, beta, schema) -> float`.
  - `src/spire_learn/neural/collector.py`: apply `shaped_reward` when the config objective is hp_shaped.
  - `src/spire_learn/neural/model.py` and `contracts.py`, only if the objective id is pinned there.
- Create: `configs/neural/baseline-gym-1.1.1-cv2-hp/` and `-cv2-hp-r2/`, which are the Task 3 plans with `[objective]` and `objective_id = "spire-learn-combat-winloss-hpshaped-v1"`.
- Test: `tests/unit/test_neural_shaped_reward.py`

- [ ] **Step 1:** Write failing tests:
  - `test_potential_ignores_unoccupied_and_zero_max_rows`: Φ is −0.25 for enemies of 100/400 plus a padded row;
  - `test_shaped_reward_terminal_potential_is_zero`: at terminal, r' = r + β·(0 − Φ(s));
  - `test_shaped_reward_damage_step`: 400 to 300 of 400 with γ 0.99 and β 0.5 gives 0.5·(0.99·(−0.75) − (−1.0));
  - `test_winloss_configs_are_unchanged_without_objective`: same TaskSignal rewards for a scripted rollout;
  - `test_objective_ids_do_not_cross`: a hp-shaped checkpoint is rejected under a win/loss config, and the reverse.
- [ ] **Step 2:** Implement.
  - Shaping changes only `TaskSignal.reward` in the collector.
  - Boundary flags are untouched.
  - The source and successor frames are public CombatFrames, read with `schema._unpack`.
- [ ] **Step 3: Evidence.**
  - Unit tests pass.
  - A CPU byte-identity check for an existing win/loss config, against `main`, using `cache-equivalence-20261117/inputs/plan-cpu.json`.
  - The gated tensor tests are green, through `gated_tests_entry.py` with a new 64-row reservation.
- [ ] **Step 4:** Commit as `feat: potential-based HP-shaped combat objective`.

### Task 5: Runs and results

This is controller work, done sequentially with no repo edits during runs.

- [ ] **Step 1:** Launch, in order, each from merged `main` with `--timeout 37800`:
  1. `cv2`;
  2. `cv2-r2`;
  3. `cv2-hp`;
  4. `cv2-hp-r2`.
- [ ] **Step 2:** For each run, check:
  - `result`, the stop reason, `learner_changed: false` and no post manifest;
  - the v2 and v1 panel scores with paired tests;
  - per-tier wins and training-time boss win rates.
- [ ] **Step 3:** Complete `docs/neural-curriculum-v2-results.md` with:
  - the four-arm table on both panels, against the v1 baselines (39 and 38 on the v1 panel) and both ceilings;
  - case-level agreement within each arm;
  - limits.

  Link it from the README. Open the PR, get a Sonnet review, and merge.
