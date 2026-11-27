# Search Distillation Parts 3–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four steps.
1. Measure on full runs whether resource-aware combat helps.
2. Build a dataset labelled by the search teacher `current-v2`.
3. Distil it into three students: the current network and two entity transformers.
4. Start PPO from distilled weights, and give verdicts on gates G2–G4.

**Architecture:** Five milestones.
- **M1, data plumbing and pilot.** A numpy shard format, and a labelling entry. The entry plays fuzzed train-split fights with the teacher; one fight in five is mixed with PPO moves. It records compact observations, the teacher's value for each action, and fight outcomes.
- **MR, full runs.** A full-run harness with fixed seeds, a pluggable combat player and a total, pre-registered non-combat heuristic. Its first experiment, v1 `current` against `current-v2`, is gate R1. R1 decides whether full labelling goes ahead.
- **M2, S1 first.** A supervised trainer. Then S1, the existing feed-forward network with extra value heads. Then a measure-first readout.
- **M3, transformers.** The entity transformer S2, and S3, which adds card metadata. Then the searches and the final two-seed students.
- **M4, PPO and gates.** Export S1 to a PPO checkpoint and run the PPO arm. Run a bounded diagnostic of PPO's late decline. Then compute the gates and write the results.

**Tech Stack:**
- Python 3.12.
- numpy 2.5.4 and PyTorch 2.14.1+cu130, inside the runtime container only. The user ruled on 2026-11-27 that container code may use them and that their tests are gated. `tests/unit` stays stdlib-only.
- Spire Gym v1.2.5, a drop-in in rules epoch `gym-1.2.4`.
- The RTX 5090, shared, behind `gpu_wait`.

**Spec:** `docs/superpowers/specs/2026-11-21-search-distillation-design.md`: Parts 3–5, Gates and Constraints. Results from earlier stages: `docs/neural-search-teacher-gym-1.2.4.md`, `docs/neural-teacher-value-v2.md` and `docs/neural-standard-baseline-gym-1.2.4.md`.

## Pinned constants

Every task uses these values. Tasks refer to them by name.

| Name | Value |
| --- | --- |
| `GYM_REVISION` | `731fca0e0341d80e7cf08a8be7a03be1fa32c3f1` (v1.2.5) |
| `LABEL_SHARD_START` | 1000 |
| `PILOT_SHARDS` | 1000–1007 |
| `CHUNK_SHARDS` | 8 |
| `ROUNDTRIP_EVERY` | 50 |
| `HOLDOUT_FRACTION` | 0.05 of Ironclad train runs (25 of 505) |
| `MIXED_EVERY` | 5 |
| `BATCH_ROWS` | 512 |
| `EVAL_EVERY` | 1000 steps, and also at the last step |
| `RUN_PANEL_SIZE` | 64 |
| `RUN_PANEL_SEED_OFFSET` | 2**34 (selection-split ordinals reserved for the run panel; disjoint from shard ordinals below 2**32, the M10 panel at 2**32 and the combat panels at 2**33) |
| `STOP_S1` | 173 selection wins |
| `DECLINE_BAND` | 6 wins |
| `TURN_CAP` | 500 actions in one public turn |
| `FIGHT_CAP` | 1,024 actions |
| `RUN_CAP` | 10,000 decisions |
| `KEPT_PREFIX_MAX` | 256 decisions |

## Spec deltas (rulings, with why and the cost if wrong)

The spec was written on Gym v1.2.0. Each delta below replaces a spec statement or fills a gap in it.

1. **Gym and data.**
   - Gym `GYM_REVISION`, a drop-in in epoch `gym-1.2.4`.
   - Recipe `configs/neural/baalorlord-standard-gym-1.2.4/recipe.json`; table `3121570f…`.
   - *Why:* the user re-pinned, and v1.2.5 passed the determinism check.
2. **Teacher.**
   - `current-v2`: one-move search, 32 native tiered playouts per legal action, value rule `resource-v2`, 80 playout steps. It is defined in `configs/neural/search-teacher-gym-1.2.4/teachers-v2.json`.
   - The spec's "visit count" is the sample count, 32.
   - *Why:* G1 failed, and the v2 choice rule picked this teacher.
3. **Comparison points.**
   - The PPO controls are the v1.2.4 baselines r1 and r2. Their selection-panel finals are 179 and 166.
   - The mixed-fight player is r1's final checkpoint.
4. **Throughput work is done.** Padding trim and sampled validation came in stage-1 Tasks 2–4.
5. **Three value heads, with value weight split equally.**
   - The heads:
     - **outcome:** +1 for a win, −1 for a death, 0 for an escape, trained by MSE;
     - **HP kept:** `hp_after_cleanup / max_hp_start`, which is 0 on a death;
     - **realised v2:** the `resource-v2` formula applied to the fight's actual end, with each decision as its own root. The root's max HP is the decision's own max HP, and potions spent count from that decision.
   - Each value head has weight ⅓, so policy and value weigh 1:1, as the spec says.
   - The outcome head is undiscounted. PPO's critic is discounted, so the Task 13 export rescales the head (delta 18).
6. **Censored teacher fights.**
   - **Turn cap:** a fight that hits the turn cap keeps its decisions from earlier public turns, as policy-only labels, if there are at most `KEPT_PREFIX_MAX` of them. If there are more, it keeps none. The decisions in the capped turn are dropped.
   - **Fight cap:** a fight that hits the fight cap keeps nothing. It is a stalemate across many turns.
   - **Value targets:** no censored fight gets any.
   - **Reporting:** decisions record their public turn number. Kept and dropped counts are reported by fight type and by cap.
   - *Why:* a stalling loop emits hundreds of near-identical labels, and the fights that stall are the hard ones.
   - *Cost if wrong:* a few labels near a stall remain.
7. **All-way ties.** When every legal action has exactly the same teacher value, the decision gets policy weight 0 and keeps its value labels. Partial ties stay in the softmax target with equal mass.
8. **Validation holdout.**
   - The holdout is exactly `HOLDOUT_FRACTION` of Ironclad train-split runs. Runs are ranked by `sha256("spire-learn-distill-validation-v1:" + run_id)` read big-endian, and the first `round(0.05 × N)` are taken.
   - Their fights are labelled into the validation set only. PPO still trains on all train runs.
   - *Why:* the spec's 2% gives only 6 Ironclad runs and 92 states.
9. **Labelling cases.**
   - Cases come from train-split shards `LABEL_SHARD_START` and up, from the recipe's only arm, `mixed`, through `generate_standard_shard` with the fuzz pools. That gives the final-quarter level mix: 0, 1, 2 and 4.
   - "Mixed arm" is the recipe's term. A "mixed fight" is one where PPO plays some of the moves.
10. **Transformer parameter cap.** 16,000,000. The feed-forward cap stays 10,000,000.
11. **How τ is chosen.**
    - **Candidates:** the 25th, 50th and 75th percentiles of strictly positive gaps between the top two distinct teacher values. They are computed on the pilot's training rows only, with a floor of 0.01.
    - **Choice:** the τ with the lowest mean validation teacher-value regret, `max v − v(student argmax)`, over searched decisions that are not all-way ties.
    - **Ties:** the tied candidates are those within one run-level SE of the best. Among them, choose the one with the smallest `|τ − τ_mid|`. If two are equally close, choose the smaller τ.
    - **Run-level SE,** used here and in delta 12. Take the candidate's regret difference from the best over the validation rows, and group the rows by `labels.run_hash` into G runs, where run g has `M_g` rows and summed difference `D_g`. Let M be the total row count and `d̄ = ΣD_g / M`. Then `SE = (1/M)·sqrt(G/(G−1)·Σ(D_g − M_g·d̄)²)`, or 0 when G < 2. A candidate is tied when `d̄ ≤ SE`.
    - *Why:* cross-entropy against `softmax(v/τ)` falls as τ falls, purely by construction, so it can't pick τ.
12. **Training length and checkpoint choice.**
    - Every student trains for the same `gradient_steps` at `BATCH_ROWS`.
    - Validation runs every `EVAL_EVERY`. The checkpoint with the lowest regret is the result.
    - Near-ties, within one run-level standard error, go to the lower value loss. Value MSE at the chosen checkpoint is reported.
    - Only the best checkpoint and the final one are kept on disk.
    - `gradient_steps` is fixed once, from the S1 pilot, at about 6 passes over the data.
    - S2 gets more selection pressure on the same 25 validation runs than S1 does. The doc notes this.
13. **Gate R1 and the MR milestone.**
    - Before full labelling, v1 `current` and `current-v2` play the same `RUN_PANEL_SIZE` full Ironclad A20 runs with the same heuristic.
    - **Rank key per run:** floors reached, then victory. A capped run is kept with the floors it reached at the cap and is flagged.
    - **Primary test:** an exact two-sided sign test on the paired rank keys at α 0.05. Labelling stops for a user ruling if either:
      - v2 is significantly worse; or
      - v2 has at least 3 more capped runs than v1.
    - **Reported alongside:** the mean and median floor difference, with a bootstrap 95% CI (10,000 resamples, seed 20261127).
    - **What a pass means:** "no large harm detected." The test detects about a 64% directional share. The doc says so.
    - *Why:* the user's 2026-11-27 ruling. v2's prices are a proxy, so they are tested before 1.5M labels are built on them.
14. **Measure-first stop (not a gate).**
    - If both S1 seeds win fewer than `STOP_S1` selection cases, stop for a user ruling before M3. (`STOP_S1` is 173; the PPO mean is 172.5.)
    - Validation regret and agreement are reported with it. They tell an S1 capacity limit, which G3 tests, apart from weak labels.
    - The readout uses panel results only from runs whose settings were chosen on validation.
15. **Compute.** Labelling and full runs use at most 16 threads, and torch is pinned to 1 thread per worker. The spec said 24 of 32 cores, but the host has 16 physical cores with 2 threads each.
16. **Deferred.** The spec's transformer PPO arm, which runs if G3 passes, goes to a follow-up plan after user review.
17. **PPO decline diagnostic (bounded, branched).**
    - **Setup:** four quarter-length runs, all branched from r2's own 75% checkpoint `near-1216512` (186 selection wins). They start at shard 297 with a decision offset of 1,216,512 and a target of 405,504 more decisions.
      - Arm A keeps the recipe, with two seeds.
      - Arm B uses a recipe copy that never adds level 4, with two seeds.
      - All four warm-start, so Adam and RNG are fresh. Resume across recipes is refused by design, and warm-starting everything treats both arms the same.
    - **Reading,** on selection wins, applied top to bottom; the first line that holds decides:
      1. Both A runs score at least `186 − DECLINE_BAND`. → The decline did not reproduce after branching. Inconclusive.
      2. Both B runs beat their seed-paired A runs, and the pooled exact McNemar test of B against A gives p < 0.05. → Level 4 is implicated. Called "suggestive", because B's case draws also differ.
      3. Both B runs fall more than `DECLINE_BAND` below 186. → Level 4 is not implicated.
      4. Anything else. → Inconclusive.
    - The boss panel is reported alongside.
    - *Why:* curriculum tuning is a spec non-goal. This only diagnoses, and branching isolates the final quarter.
18. **Critic export.**
    - The outcome head's weight and bias are both multiplied by `c = mean(γ^remaining)` over teacher-only value rows, where `remaining` counts decisions to the fight's end. `c` is recorded.
    - The early value-loss transient of the PPO arm is reported.
    - *Why:* PPO's critic predicts discounted returns. The head was also fit on teacher play, not student play; the doc notes the mismatch.

## Global Constraints

- **AGENTS.md:**
  - networks get public observations and legal masks only;
  - Gym faults are errors, never losses or censors;
  - never modify a Gym checkout;
  - artifacts and Baalorlord-derived data stay out of Git;
  - no dependency installs. Ask the user whenever a new or updated dependency would help.
- **Labels.** Search labels are allowed under the spec's "A rule that changes". Research seeds and search statistics are labels and provenance, never inputs.
- **Game settings.** Ironclad, A20, compact view `standard-combat-compact-v1`, `TURN_CAP`, `FIGHT_CAP`. Profile `configs/neural/gym-1.2.5/runtime.json`. Every new plan and template sets `gym_revision` = `GYM_REVISION` in both `plan.json` and `template.toml`.
- **Identities.** Identity strings are written out in full everywhere, never abbreviated:
  - `spire-learn-combat-ff-standard-v1`
  - `spire-learn-combat-features-standard-v1`
  - `spire-learn-combat-dispatch-v1`
  - `spire-learn-combat-winloss-v1`
  - `spire-learn-combat-distill-v1`
- **Splits.** Labels come from train-split runs only. The selection and final panels are never labelled, and the final panel is never scored. Run-panel seeds come from their own domain (Task 4) and are not Baalorlord states.
- **Determinism.**
  - The same recipe, seeds and teacher version give the same shard files.
  - Deterministic manifests (file hashes, counts, provenance) are kept separate from run reports (CPU, wall time, host).
  - Supervised training is seeded. GPU nondeterminism is not claimed away; seeds are reported.
- **Gates are pre-registered as in the spec.**
  - Exact two-sided McNemar on selection-panel pairs, final checkpoint against final checkpoint, seed 1 paired with seed 1.
  - "Beats" means both seed pairs favour the challenger on paired wins *and* the pooled test is significant.
  - A case censored or escaped on either side is dropped from that pair. Censor counts are reported beside every gate.
  - α: G2 is 0.05/3 per test, G3 is 0.025, G4 is 0.05.
  - The boss panel and the run panel are reported, not gated.
- **Choices are made on validation, never on a panel.** The panels are used only for the final two-seed runs, R1, and the measure-first stop.
- **Compute.** At most 16 threads for CPU work. GPU work goes through `gpu_wait`, re-checked at every validation interval and every panel case.
- **Provenance for every experiment:**
  - Gym revision and dirty state;
  - learner revision and dirty state;
  - recipe, table, panel and dataset hashes;
  - seeds and budgets;
  - CPU and GPU time, and GPU waits.
- **Tests.**
  - `tests/unit` is stdlib-only.
  - Each module that needs numpy or torch names its suite:
    - **The tensor suite** (`gated_tests_entry.py` `MODULES`). Deterministic. Sampling only through fake generators, inside the sampled-row budget.
    - **A native suite**, with its own flag. New flags: `--distill-native`, `--full-run-native`, `--card-metadata-live`.
  - `tests/unit/test_gated_tests_entry.py` is updated whenever the suites change.
- **Process.**
  - Code that changes training, labelling, evaluation, model or identity behaviour gets two reviewers: Claude Sonnet medium and Gemini 4 Neon Low. Docs, configs and tests get one.
  - Never edit the repo while a container run is in progress.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - PRs and merges run under the user's standing authority.

## Review Focus

1. **The observation round trip.**
   - A stored decision expands to exactly the compact observation the Gym returned, byte for byte. That covers every group, both masks (including the known empty slots of `potions` and `orbs`) and the action mask.
   - The unit test uses the real compact fixture. The labeller asserts the round trip on every `ROUNDTRIP_EVERY`-th real decision; a mismatch is an error.
   - The run-panel bridge (Task 4) is pinned the same way against the Gym's own compact frames.
   - Owners: Tasks 1, 2 and 4.
2. **Information safety.**
   - No label field reaches a model input. Label arrays are separate files, and batches are built from observation files only.
   - Masked-out labels carry sentinels or NaN, and they never enter a loss expression. The loss uses boolean row selection, so a NaN in a masked row can't poison a gradient.
   - Owners: Tasks 1 and 7.
3. **Split integrity.** No selection-split, final-split or holdout run appears in a training shard. The labeller refuses non-train cases. Owner: Task 2.
4. **No cross-loading.**
   - Distillation, S2, S3 and PPO checkpoints never load into one another. S2 and S3 have distinct features ids.
   - The only bridge is Task 13's export, and it accepts only an S1 distillation checkpoint.
   - Owners: Tasks 7, 10, 11 and 13.
5. **No silent defaults.**
   - A player that can't act is an error. That covers a student with no finite legal logit, and a heuristic with no rule for a boundary kind.
   - The heuristic is total over the run descriptor's choice kinds. Its declared fallback, the lowest legal index, is counted and reported. It is never silent.
   - Owner: Task 4.

---

## M1: data plumbing and pilot

### Task 1: Sparse decision shards

**Files:**
- Create `src/spire_learn/neural/distill_data.py` (numpy, container-only).
- Test: `tests/neural/test_distill_data.py`, in the tensor suite.

**Interfaces:**
- `SHARD_SCHEMA = "spire-learn-distill-shard-v1"`.
- **Shard layout.** A shard is a directory of `.npy` files written with `np.save`, never pickled, plus `manifest.json`.
  - **Row groups** are stored CSR-style:
    - `<group>.offsets`: int64, with decisions + 1 entries;
    - `<group>.rows`;
    - `<group>.values`: the group's dtype;
    - `<group>.known`.
  - Groups in `optional_known_empty_groups` also store their full `_known` array.
  - Single-row groups and `action_mask` are stored dense.
- **Labels** are separate files:
  - `labels.legal_offsets`, `labels.legal`;
  - `labels.values`: float64 per legal action, NaN when there are none;
  - `labels.samples`, `labels.choice`, `labels.played`, `labels.root` (an enum);
  - `labels.outcome`: an enum (win, death, escaped, none);
  - `labels.hp_kept` and `labels.realised_v2`: NaN when `value_ok` is false;
  - `labels.value_ok`, `labels.policy_ok`, `labels.research_seed`;
  - `labels.case`: shard and local index;
  - `labels.run_hash`: uint64, the big-endian first 8 bytes of `sha256(run_id)`. It is stored in label files only;
  - `labels.turn`: the public turn.
- **API:**
  - `encode_decision(frame, schema)` and `expand_decision(shard, index, schema) -> CombatFrame`.
  - `ShardWriter(path, schema, provenance)`: `.add(frame, labels)`, and `.close() -> manifest`. A zero-row shard writes a manifest with rows 0; readers skip it.
  - `ShardReader(path)`: `.verify()`, `.frame(i)`, `.labels(i)`, with arrays opened `mmap_mode="r"`.
  - `validation_runs(train_run_ids) -> frozenset[str]` (delta 8).

- [ ] **Step 1: Write the failing tests:**
  - `test_round_trip_exact_on_compact_fixture`, using the builders in `tests/unit/test_neural_standard_compact.py`. It covers non-prefix occupied rows and non-uniform empty-slot `_known` in `potions` and `orbs`.
  - `test_label_files_disjoint_from_observation_files`
  - `test_shard_files_deterministic`
  - `test_verify_detects_tamper`
  - `test_zero_row_shard`
  - `test_validation_runs_exact_fraction` (25 of 505 when the real table is mounted, otherwise synthetic ids)

  Register the module in the tensor suite.
- [ ] **Step 2: Implement it, run the tests in the container, and commit** `feat: sparse decision shards for distillation`.

### Task 2: The labelling entry and the tie-aware audit

**Files:**
- Create `scripts/neural/distill_label_entry.py`.
- Modify `scripts/neural/search_teacher_entry.py`. `play_fight` gains three optional hooks:
  - `chooser(observation, legal, teacher_result, rng) -> action`
  - `observer(observation, decision, turn)`
  - `finisher(final_observation, info, outcome, censor_reason)`, which gives `realised_v2` its end state: potion kinds, max HP and cleanup HP

  The cap, censor and resource logic stay in one place. The default behaviour is byte-identical.
- Create `src/spire_learn/neural/policy_player.py` with `CheckpointPlayer(path, schema, *, seed)`:
  - It loads with `load_checkpoint(purpose="evaluate", device="cpu")`, then `build_policy` and a strict `load_state_dict`.
  - `act(frame, legal) -> action` samples with a seeded generator, or takes the argmax when `deterministic=True`.
  - It takes an `expected_identity` argument and refuses a mismatched checkpoint.
  - Workers pin torch to 1 thread and 1 interop thread. The parent entry process sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1 before spawning workers.
  - In Task 2 it handles the FF family only. Task 8 extends it to dispatch through the `ModelFamily` registry.
  - With a single legal action it plays that action without sampling.
  - It never calls `create_runtime`.
- Modify `scripts/neural/teacher_audit.py`. `compare` and `summarize_consistency` gain these output keys, added without changing existing ones:
  - `tie_aware_agreement`: run B's choice is in run A's top-valued set;
  - `non_tie_decisions`;
  - `non_tie_agreement`.

  `consistency` also gains `--allow-uncalibrated`. It is passed to `_teacher` and recorded in provenance.
- Tests:
  - `tests/unit/test_search_teacher_entry.py`: the default hooks leave behaviour unchanged.
  - `tests/unit/test_teacher_audit.py`: the new keys.
  - `tests/integration/test_distill_label_native.py`, in the `--distill-native` suite.
  - `tests/neural/test_policy_player.py`, in the tensor suite, with `deterministic=True`.

**Interfaces:**
- **CLI:** `--recipe --teacher current-v2 --teachers <teachers-v2.json> --allow-uncalibrated --shards FIRST:COUNT --seed S --mixed-checkpoint <r1 final> --mixed-every MIXED_EVERY --output DIR --parallel N --roundtrip-every ROUNDTRIP_EVERY`.
- **Cases** come from `generate_standard_shard(recipe, split="train", shard_index=i, validate=gym_validator)`. Holdout runs go to the validation shard series. Any non-train case raises.
- **Scheduling.** A rolling pool runs over all cases of the chunk. Shards are written in index order.
- **Seeds:**
  - research seed: `H(seed, shard, local)`;
  - mixed-move stream: `H(seed, "mixed", shard, local)`.
- **Mixed fights.** A fight is mixed when `(shard × 256 + local) % MIXED_EVERY == MIXED_EVERY − 1`.
  - In a mixed fight, each move is played by `CheckpointPlayer` with probability ½, and otherwise by the teacher.
  - The teacher labels every decision, and `played` records the move actually made.
  - Mixed fights get `value_ok = False`.
- **Per decision:**
  - The compact frame is built with `snapshot_common` and the parsed schema.
  - `RootContext` is built before the search, so single-legal and refused rows get `realised_v2` too.
  - A terminal branch with no cleanup HP raises.
  - Single-legal, refused and all-tie rows get `policy_ok = False`.
  - `labels.run_hash` comes from the case's run id in the shard annotations.
- **At the end of a fight:** fill `outcome`, `hp_kept` and `realised_v2`, then apply delta 6.
- **Resuming is by chunk.** Each container call writes `COUNT` shards to a fresh `--output` under `artifacts/development/neural-combat/distill/`. The host driver skips chunks whose manifests verify.
- **The manifest records:**
  - recipe, table, teacher config sha256, Gym revision and build;
  - learner revision and dirty state;
  - shard range, seeds, and the mixed checkpoint's sha256;
  - counts by root kind and by fight type;
  - censored, kept and dropped counts by cap;
  - the all-tie share and the round-trip count.

- [ ] **Step 1: Write the failing tests:**
  - `test_only_train_cases_and_holdout_routed`
  - `test_mixed_schedule_played_vs_choice`
  - `test_mixed_has_no_value_targets`
  - `test_turn_cap_keeps_bounded_prefix_policy_only`
  - `test_fight_cap_keeps_nothing`
  - `test_turn_cap_on_turn_one_writes_cleanly`
  - `test_realised_v2_matches_search_value_formula`
  - `test_single_legal_refused_and_all_tie_rows`
  - `test_seed_independent_of_chunking`
  - `test_roundtrip_assertion_fires_on_mismatch`
  - `test_default_hooks_unchanged` (host)
  - `test_tie_aware_consistency_keys` (host)
  - `test_consistency_allow_uncalibrated` (host)
  - `test_finisher_receives_end_state` (host)
- [ ] **Step 2: Implement it, run the host suite and the gated suites, and commit** `feat: distillation labelling entry`.

### Task 3: Pilot and the teacher's self-agreement (controller)

- [ ] **Step 1: Pilot.**
  - Label `PILOT_SHARDS` (2,048 cases) at 16 threads. These shards become part of the dataset.
  - Report:
    - CPU per searched decision, the p99 and maximum fight CPU, and the CPU of the top five fights;
    - the share of CPU spent in censored fights;
    - censored, kept and dropped counts by type and cap;
    - decisions per fight and the all-tie share;
    - bytes per decision against free disk.
  - Verify every pilot shard.
  - **Stop for a ruling if** more than 2% of fights are censored, more than 25% of CPU went to censored fights, or there is any fault.
- [ ] **Step 2: Self-agreement reference.**
  - Command: `teacher_audit.py consistency --decisions <current-v2 selection run>/decisions.jsonl --sample 2000 --seed 2026112701 --teacher current-v2 --teachers configs/neural/search-teacher-gym-1.2.4/teachers-v2.json --allow-uncalibrated --recipe <v1.2.4 recipe> --parallel 12 --output <dir>/consistency.json`. It runs in the container and is for audit only.
  - Report strict, tie-aware and non-tie agreement.
- [ ] **Step 3: Size and cost.**
  - Choose the shard count so the train decisions reach about 1,500,000, counting all stored rows. That is about 215 shards; recompute it from the pilot.
  - State CPU-hours and wall time as a range. The prior estimate is 125–330 core-hours, which is 8–21 hours at 16 threads.

## MR: full runs

### Task 4: The full-run harness

**Files:**
- Create `scripts/neural/full_run_entry.py`.
- Create `src/spire_learn/neural/noncombat_heuristic.py` (stdlib).
- Create `src/spire_learn/neural/compact_bridge.py` (numpy).
- Tests:
  - `tests/unit/test_noncombat_heuristic.py` (stdlib, on boundaries built from `tests/fixtures/neural/standard-run-1929e18.json`);
  - `tests/neural/test_compact_bridge.py` (tensor suite);
  - `tests/integration/test_full_run_native.py` (`--full-run-native`).

**Interfaces:**
- **Run panel.**
  - `RUN_PANEL_SIZE` Ironclad A20 runs, partition `selection`.
  - Seeds come from `environment_seed_for("selection", replica=0, ordinal=RUN_PANEL_SEED_OFFSET + i)` for `i < RUN_PANEL_SIZE`.
  - Experienced creation, full Neow.
  - The panel identity is the sha256 of the seed list, the partition and the settings.
- **Run caps:** `TURN_CAP` and `FIGHT_CAP` per combat, and `RUN_CAP` per run. A cap censors the run, recording the floors reached and the reason. Faults are errors.
- **Combat players** (`--combat-player`):
  - **Routing.** Combat and non-combat boundaries are told apart by `parse_lifecycle(view["info"], view=STANDARD_COMBAT_V1).active`, which also catches combats started by events.
  - `teacher:<name>` loads the teacher with `allow_uncalibrated=True`, recorded in provenance, so that `current-v2` is accepted. It sends each combat decision through `search_teacher_entry._decision`, with `RunEnv.public_search(..., observation="compact", capacity_preset=...)` as the search. (Python `RunEnv` has no `public_root()`.)
    - It reuses the single-legal shortcut and the refused-root fallback: on `RunUnsupportedBoundaryError` or `RunStateError`, it plays the teacher's `playout_choice` on the bridged observation.
    - Refused roots are normal, especially under Runic Dome, and never raise. They are counted per run.
  - `checkpoint:<path>` plays by reflex on `compact_bridge(dense combat projection)`, through `CheckpointPlayer(deterministic=True)`.
- **The bridge** trims each group's rows to the compact preset's capacities and casts to the compact dtypes.
  - It raises a typed error if an occupied row exceeds a capacity or a value is out of range. For `potions` and `orbs`, a known empty slot beyond capacity also raises.
  - It returns both forms: the numpy dict that `_decision`'s `View` reads, and a `CombatFrame` for `CheckpointPlayer`.
  - Gated test (a): widen a compact fixture frame to dense, which the Gym documents as exact. Bridge it back and require byte equality.
  - Gated test (b): at real Run combat boundaries, compare the bridged frame with the root decision view of `public_search(observation="compact")`. The action mask and the deterministic groups must be equal, and the root's legal set must agree with the Run's legal indices.
- **Non-combat player: `noncombat-heuristic-v1`.**
  - It is a total function over every `room` and `selection_operation` value of the run descriptor.
  - It is deterministic given the public state, with ties broken by the lowest legal index.
  - It never plays `use_potion` or `discard_potion` outside combat.
  - A free potion slot means `any((potions_known[:, 0] == 1) & (potions_occupied == 0))`: an owned slot with nothing in it.
  - **Rewards, treasure and boss treasure:** take gold, then the relic, then open the card reward, then take a potion if a slot is free, then proceed. Keys are skipped, so runs end at Act 3 (`act3_no_keys`).
  - **Map:** score the kinds of the immediately reachable nodes in this order: `rest` (if HP < 55%), `treasure`, `monster`, `question` (or `event`), `shop`, `elite`, `boss`. An elite drops below `shop` when HP < 70%. Ties go to the lowest legal index.
  - **Card reward:** the highest card on the committed Ironclad priority list, whose file sha256 is recorded. Skip if no listed card is offered.
  - **Shop:** at most one purge (the first of Curse, Strike, Defend) and at most one card off the list if affordable, then leave. `full_run_entry.py` keeps per-room flags for whether it has purged and whether it has bought, keyed by `(act, floor)`, so the shop ends within 3 steps even with The Courier.
  - **Campfire:**
    - rest if HP < 50%;
    - otherwise smith the highest-listed upgradable card;
    - Recall, Lift, Toke and Dig are never chosen unless they are the only options.
  - **Non-combat card selections**, by operation family:
    - remove, transform or toke: Curse, then Strike, then Defend, then the lowest-listed card;
    - upgrade, duplicate or bottle: the highest-listed eligible card, or the lowest legal `select` index if none is listed;
    - confirm as soon as it is legal;
    - never cancel while any `select` or `confirm` is legal.
  - **Events:**
    - Always exclude repeatable self-harm options: Knowing Skull options 0–2, and Scrap Ooze option 0 when HP < 50%.
    - Exclude options whose `param_semantics` or `event_constants` show HP or max-HP loss while HP < 50%, or a curse.
    - Take the first remaining option.
    - If every option is excluded, take the non-lethal option with the smallest HP plus max-HP loss, breaking ties by the highest `event_option`. In Knowing Skull that means leaving in one step.
    - If every option would be lethal, take the smallest-loss option anyway. A resulting death is a real outcome.
  - **Neow:** the first option without an HP or max-HP cost.
  - **Boss relics:** the first listed.
  - **Fallback:** any boundary not covered above uses the lowest legal index that is not a potion verb, counted and reported. A boundary with no legal action raises.
- **Execution.** `--parallel N` runs whole runs per worker, in chunks of 8 runs per container, each writing to a fresh `--output`.
- **Output:**
  - per run: floors reached, the death act and floor, HP at each boss entry, potions used, censor reason, victory, refused roots and fallback counts;
  - a `fights.jsonl` in the teacher entry's format;
  - a summary with aggregates only.

- [ ] **Step 1: Write the failing tests:**
  - `test_heuristic_total_over_descriptor_enums`
  - `test_heuristic_rules_on_synthetic_boundaries`
  - `test_shop_terminates_with_courier`
  - `test_knowing_skull_leaves`
  - `test_unlisted_smith_never_cancels`
  - `test_no_noncombat_potion_actions_and_owned_slot_rule`
  - `test_run_panel_seeds_disjoint_from_panel_ordinals`
  - `test_fallback_counted_not_silent`
  - `test_run_panel_seeds_stable`
  - `test_bridge_equals_gym_compact` (gated)
  - `test_bridge_overflow_raises` (gated)
  - `test_run_root_legal_set_matches` (gated)
  - `test_full_run_smoke_two_seeds` (gated)
- [ ] **Step 2: Implement it, run the tests, and commit** `feat: full-run harness with a pre-registered non-combat heuristic`. Commit the Ironclad priority-list file and record its hash before any R1 run.

### Task 5: Gate R1 (controller)

- [ ] **Step 1: Run the panel** with `teacher:current` and `teacher:current-v2` on v1.2.5, using the same heuristic. For reference only, also run `checkpoint:<PPO r1 final>`. The expected cost is 20–80 core-hours, about 1.5–5 hours at 16 threads. Late-boss fights can take up to about 2,500 CPU s. Run the first chunk of 8 runs as a measured pilot, restate the range, then continue. Report the p99 and maximum run CPU.
- [ ] **Step 2: Apply R1** (delta 13). Report mean floors, deaths by act, Act 1 and Act 2 boss clears, HP at boss entry, potions per run, fallback counts and censors by player.
- [ ] **Step 3: Write `docs/neural-full-run-panel.md`.** It covers the method, the heuristic and its priority-list hash, and the R1 result, with aggregates only. One reviewer.

### Task 6: Full labelling (controller)

- [ ] **Step 1:** Run the remaining shards from 1008 in `CHUNK_SHARDS` chunks as a detached chain under `~/scratch/spire-learn/`.
- [ ] **Step 2:** Record the totals in the ledger:
  - decisions by root and fight type;
  - censored, kept and dropped counts by cap;
  - CPU hours and bytes;
  - validation-set size.

## M2: S1 and the measure-first readout

### Task 7: Distillation config, identities, trainer and loader

**Files:**
- Modify `src/spire_learn/neural/config.py`.
- Modify `src/spire_learn/neural/model.py`.
- Modify `src/spire_learn/neural/ppo.py`.
- Modify `src/spire_learn/neural/checkpoint.py`.
- Modify `src/spire_learn/neural/runner.py`.
- Create `src/spire_learn/neural/distill_train.py` and `scripts/neural/distill_train_entry.py`.
- Tests:
  - `tests/unit/test_distill_config.py` (stdlib);
  - `tests/neural/test_distill_train.py` (tensor suite);
  - `tests/neural/test_model.py`, extended (tensor suite).

**Interfaces:**
- **Model config.**
  - `ModelConfig.value_heads: int = 1` is admitted in the `[model]` exact-key set.
  - A single helper, `model_config_hash_payload(model_config)`, leaves out `value_heads` when it is 1. An explicit `value_heads = 1` hashes the same as an absent one.
  - Every hash site uses that helper: the config model hash, `_stream_runtime_projection`, `CombatPolicy._model_config_hash`, and `ppo_update`. Existing hashes stay byte-identical.
- **Outputs.** `PolicyOutput.values` stays the outcome head. A new `aux_values` attribute holds the other two heads. `_compile_model_plan` counts the extra heads. `act` and `values` are unchanged.
- **Training configs** are loaded by a dedicated `load_distill_config`, with no `manifest_path` or `[rollout]`.
  - `DISTILL_OBJECTIVE_ID = "spire-learn-combat-distill-v1"`.
  - `config_hashes` is `{model, distill}`. The distill hash covers only `tau`, `learning_rate`, `batch_rows`, `gradient_steps` and `data_fraction`. Shards, seed and `eval_every` are recorded as run metadata.
- **Evaluation configs** go through `load_neural_config`, which accepts the distill objective with `config_hashes = {model}` and either `[model]` or `[transformer]`.
  - `NeuralConfig.ppo` becomes optional.
  - `runner._config_projection` writes `ppo` and `distill` only when they are present, and writes the model through a family projection that keeps today's keys for `ModelConfig`.
- **Checkpoints.** `checkpoint.py` checks `training_state.hyperparameter_id` against the objective's hash key (`distill` or `ppo`). A distill `training_state` defines the four RNG domains, `next_environment_seed_ordinal` = 0 and the counters.
- **Loss.** Masked terms use boolean row selection, so a masked NaN never enters an expression.
  - **Policy:** cross-entropy to `softmax(v/τ)` over legal actions, on `policy_ok` rows.
  - **Value:** three MSE terms, each with weight ⅓, on `value_ok` rows.
  - Each term is a mean over its selected rows, and 0 when none are selected.
  - **Optimiser:** Adam at a fixed learning rate with `max_grad_norm` 1.0. The same for every student.
- **Microbatching.** `microbatch_rows` accumulates gradients up to `BATCH_ROWS` when memory requires. The gradient is the same as one full batch.
- **The loader:**
  - It memory-maps the shards and collates trimmed batches straight from the CSR arrays. A test pins exact equality with `encode_frames([expand_decision(…)], trim_padding=True)`.
  - It reads observation files only.
  - Its resident memory stays under 8 GiB.
  - `data_fraction` is a prefix of the training shards in index order.
- **Validation,** at every `EVAL_EVERY`:
  - regret;
  - tie-aware and strict top-1 agreement;
  - each loss term.

  The chosen checkpoint follows delta 12.
- **The entry** runs with `--gpu` and `gpu_wait`. It writes `train_log.jsonl` and `summary.json`, recording rows per second, GPU time and waits, and peak VRAM.

- [ ] **Step 1: Write the failing tests:**
  - `test_existing_hashes_unchanged_at_every_site`
  - `test_value_heads_one_hashes_as_absent`
  - `test_distill_config_without_ppo`
  - `test_eval_config_projection_distill_and_transformer`
  - `test_distill_checkpoint_refused_as_ppo`
  - `test_aux_values_and_param_count`
  - `test_masked_nan_rows_give_finite_zero_grads`
  - `test_soft_target_and_policy_mask`
  - `test_loader_equals_encode_frames`
  - `test_loader_reads_no_label_file`
  - `test_best_regret_checkpoint_with_value_tiebreak`
  - `test_final_step_evaluated`
  - `test_microbatch_gradient_equal`
  - `test_same_seed_same_batches`
- [ ] **Step 2: Implement it, run the host suite and the gated suites, and commit** `feat: distillation trainer and identities`.

### Task 8: The evaluation seam

**Files:**
- Create `src/spire_learn/neural/model_families.py`.
- Modify `src/spire_learn/neural/runner.py`: `create_runtime` dispatches on the model family for `purpose="evaluate"`.
- Create `src/spire_learn/neural/panel_eval.py`, extracted from `baseline_entry._evaluate`. `baseline_entry` calls it, and its outputs stay byte-identical.
- Create `scripts/neural/student_eval_entry.py`.
- Tests:
  - `tests/unit/test_model_families.py`
  - `tests/unit/test_baseline_entry.py`
  - `tests/neural/test_model.py`

**Interfaces:**
- **The `ModelFamily` registry,** keyed by `model_id`, gives each family's `build`, `config_type` and `parameter_cap`. The FF family's path is unchanged.
  - Every built policy exposes `act`, `values`, `forward`, `_behavior_policy_version` and the identity attributes `model_id`, `features_id`, `action_id` and `objective_id`, as `Collector` requires.
  - `CheckpointPlayer` (Task 2) dispatches through this registry.
- **Student evaluation:**
  - reflex argmax over legal actions;
  - on the selection and boss panels at `FIGHT_CAP` and `TURN_CAP`;
  - the GPU gate is re-checked for each case.
- **`summary.json`:** `evaluations.final` in the baselines' per-case format, readable by `standard_results.load_outcomes(…, "final")`, `teacher_audit.load_ppo` and the gate script.
- **Inference time:** warmed batch-1 forward passes on CPU and GPU, median of 1,000.
- **A model with no finite legal logit raises.**

- [ ] **Step 1: Write the failing tests:**
  - `test_ff_family_unchanged`
  - `test_unknown_family_refused`
  - `test_baseline_eval_outputs_unchanged`
  - `test_student_summary_format`
  - gated `test_no_finite_legal_logit_raises`
- [ ] **Step 2: Implement it, run the tests and commit** `feat: student evaluation seam`.

### Task 9: S1 searches, final S1 and the readout (controller)

- [ ] **Step 1: Budget.** Run one S1 pilot: one seed, the middle τ, learning rate 3e-4. Measure rows per second and peak VRAM. Fix `gradient_steps = round(6 × train_rows / BATCH_ROWS)`. Record a GPU-hours estimate for M2–M4. The prior estimate is 20–70 GPU-hours supervised plus about 25 for PPO and the diagnostic.
- [ ] **Step 2: Learning rate.** Try 1e-4, 3e-4 and 1e-3 at the middle τ and pick by regret. Candidates within one run-level SE (delta 11) of the best go to 3e-4 if it is among them, and otherwise to the smaller rate. The pilot counts as the 3e-4 run.
- [ ] **Step 3: τ.** Train the other two candidates at the chosen learning rate and pick by delta 11. All students share the result.
- [ ] **Step 4: Final S1.** Train two seeds. Evaluate them on the selection and boss panels, and on the run panel as `checkpoint:` players, which is reported only.
- [ ] **Step 5: Readout** (delta 14). Report:
  - wins;
  - regret and value MSE;
  - strict and tie-aware agreement against the Task 3 reference;
  - HP kept;
  - inference time;
  - run-panel floors.

  Then apply the stop rule.

## M3: transformers

### Task 10: The entity transformer (S2)

**Files:**
- Create `src/spire_learn/neural/transformer.py`.
- Register it in `model_families.py`.
- Add the identities and the `[transformer]` table in `config.py`.
- Tests: `tests/unit/test_transformer_config.py`, and `tests/neural/test_transformer.py` (tensor suite).

**Interfaces:**
- **Identities:**
  - `TRANSFORMER_MODEL_ID = "spire-learn-combat-entity-transformer-v1"`
  - `TRANSFORMER_FEATURES_ID = "spire-learn-combat-features-entity-v1"`
- **`TransformerConfig`:**
  - `layers`, `width` and `heads`;
  - `number_encoding`: `bucket`, `scaled` or `sinusoidal`;
  - `value_heads`: 3;
  - a parameter cap of 16M.
- **Sizes,** all pre-norm:

  | Size | Layers | Width | Heads |
  | --- | ---: | ---: | ---: |
  | Small | 2 | 128 | 4 |
  | Medium | 4 | 256 | 8 |
  | Large | 6 | 384 | 8 |
- **Tokens:**
  - one per occupied row of every non-action row group in the schema, enumerated from the descriptor;
  - plus one context token built from the single-row groups;
  - padding rows are never tokens.
- **References are resolved before the trunk.** Each reference column below is replaced by a learned projection of the referenced token's pre-trunk embedding, masked where the reference is −1:
  - `powers.owner_row`
  - `selection.source_row`
  - `known_draw.combat_card_row`
  - `combat_cards.permanent_row`
  - `combat_cards.stasis_enemy_row`

  Raw row indices are never used as features.
- **Action scoring.** Each legal action's query is its action row plus the gathered post-trunk embeddings for `source_row`, `target_row` and `stance_choice_ref`. The query attends over the trunk and produces a logit. Only legal actions are scored.
- **Value heads:** 3 linear heads over the mean-pooled trunk.

- [ ] **Step 1: Write the failing tests:**
  - `test_parameter_counts`
  - `test_every_row_group_tokenised`
  - `test_padding_rows_never_tokens`
  - `test_permutation_invariance_with_reference_remap`
  - `test_only_legal_scored`
  - `test_refuses_ff_and_s3_checkpoints`
  - `test_transformer_through_create_runtime_evaluate`
  - `test_number_encodings_build`
- [ ] **Step 2: Implement it, run the tests and commit** `feat: entity transformer student`.

### Task 11: Card metadata features (S3)

**Files:**
- Create `scripts/neural/card_metadata_snapshot_entry.py`. It calls `standard_names.gym_card_metadata()` and writes to `$SPIRE_LEARN_RUNNER_OUTPUT`. The controller copies the result to `configs/neural/standard/card-metadata-gym-1.2.4.json`.
- Extend `transformer.py`.
- Tests:
  - `tests/integration/test_card_metadata_live.py` (`--card-metadata-live`);
  - `tests/unit/test_card_metadata_snapshot.py`.

**Interfaces:**
- **Identity.** `TRANSFORMER_CARDMETA_FEATURES_ID = "spire-learn-combat-features-entity-cardmeta-v1"`. The snapshot's fingerprint `effbb810…` is part of the config hash.
- **Metadata** goes on the card tokens of `combat_cards`, `permanent_cards` and `combat_card_offers`.
- **Fields:**
  - `color`, `rarity`, `type` and the tags;
  - for each face: keywords, `cost`, target, `base_damage`, `base_block` and `magic_number`.
- **Face choice.**
  - The face is `faces[min(max(upgrades, 0), len(faces) − 1)]`.
  - `has_face = 0` when a card has no faces.
  - Offers use upgrade level 0.
- **Value encoding:**
  - `cost` is a kind (int, `x` or `unplayable`) plus a number;
  - nullable integers are a presence bit plus a value.

- [ ] **Step 1: Test, snapshot, implement, and commit** `feat: card metadata features for S3`.

### Task 12: Transformer searches and finals (controller)

- [ ] **Step 1: Learning rate per size.** Try 1e-4, 3e-4 and 1e-3, with one seed, the shared τ and `bucket` encoding. Pick by regret.
- [ ] **Step 2: Number encoding,** on Medium at its learning rate. If the winner isn't `bucket`, rerun Small and Large with the winner at their learning rates. The winner is used by all transformers.
- [ ] **Step 3: Size.** The size with the lowest regret becomes S2's. S3 uses S2's size, encoding and learning rate.
- [ ] **Step 4: Finals.** Two seeds of S2, S3 and every size, evaluated on the selection, boss and run panels. Report inference times.
- [ ] **Step 5: Data curve.** S1 and S2 at 25% and 50% of the training shards (an index prefix), two seeds each, same `gradient_steps`, best-regret checkpoint. The 100% runs are reused. Evaluate on the selection and boss panels. This is reported as a trend.
- [ ] **Step 6:** Record GPU hours and waits.

## M4: PPO from distilled weights, the diagnostic and the gates

### Task 13: Export, warm-start and branching plumbing

**Files:**
- Create `scripts/neural/distill_export.py`. It runs in the container, once per template.
- Modify `scripts/neural/baseline_entry.py`:
  - add optional `plan.json` fields, validated by `load_plan`, used by `_reevaluation_preflight`, and persisted in `summary['plan']` and `_TRAINING_RECORD` so `--reevaluate --panel boss` keeps them:
    - `initial_checkpoint`;
    - `start_shard`, default 0;
    - `decision_offset`, default 0;
    - `evaluate`, default true;
  - write a settled training-stream win rate over the second half of the run to `summary.json`.
- Configs:
  - `configs/neural/baseline-standard-1.2.5-distill{,-r2}/`;
  - `configs/neural/baseline-standard-1.2.5-decline-{a,b}{1,2}/`.
- Recipe: `configs/neural/baalorlord-standard-gym-1.2.4-nolevel4/recipe.json`.
- Tests:
  - `tests/unit/test_baseline_entry.py`
  - `tests/unit/test_distill_plans.py`
  - `tests/neural/test_distill_export.py` (tensor suite)

**Interfaces:**
- **`export_ppo_checkpoint(distill_ckpt, template, c) -> path`, with `c = compute_critic_scale(shards) -> float`:**
  - Copy the trunk, the policy head and the outcome head into the FF `value_head`, then scale its weight and bias by `c` (delta 18). Drop the other two heads.
  - `distill_export.py` computes `c` once from the training shards.
    - `remaining = (fight_rows − 1) − row_in_fight`, which is 0 on a fight's last decision. Both counts come from `labels.case` and row order.
    - The same `c` is used for every template, and it is recorded.
  - Write identity (`spire-learn-combat-ff-standard-v1`, `spire-learn-combat-features-standard-v1`, `spire-learn-combat-dispatch-v1`, `spire-learn-combat-winloss-v1`) with the template's hashes.
  - Write a fresh optimiser, the template's RNG seeds, and `parent_checkpoint` with the source checkpoint's path and sha256.
  - Refuse anything but an S1 distillation checkpoint.
- **`initial_checkpoint`:**
  - Preflighted for both `warm_start` and `evaluate`.
  - Evaluated as checkpoint 0.
  - Recorded with its sha256.
- **`start_shard` and `decision_offset`:**
  - Training and qualification start at `start_shard`, with learned-decision counters starting at `decision_offset`. The target counts from that offset.
- **`evaluate: false`:** skips all panel evaluation, and the results check does not require evaluations. The pilot uses it.
- **Checkpoint 0** keeps the label `fresh-initial`, with `learned_decisions = decision_offset`, so `load_prior` and `_paired` work unchanged. `_reevaluation_preflight` prepares at `start_shard`.
- **Plan comparison.** `test_plans_differ_only_in_allowed_fields` also allows `accepted_identities.config_hashes.ppo`, `baseline_id` and `template` to differ when the pilot changes the learning rate.
- **Plans.** The distilled plans copy the v1.2.4 baselines and change only:
  - the initial checkpoint (S1 seed 1 with r1's seeds, S1 seed 2 with r2's);
  - `gym_revision`;
  - the learning rate, after the pilot;
  - the output root.
- **Decline plans:**
  - start from r2's `near-1216512` with `start_shard` 297, `decision_offset` 1216512 and a target of 405,504;
  - set `retain_near_decisions` to `[]`, so only checkpoint 0 and the final checkpoint are evaluated;
  - A uses the recipe, B the no-level-4 recipe;
  - each has two seeds: r2's, and new seeds 9101 / 9203 / 9307 / 9419.
- **Recipe check.** A test pins that the no-level-4 recipe differs from the original only in `fuzz_schedule`, and that its panel identity is equal.

- [ ] **Step 1: Write the failing tests:**
  - `test_export_refuses_non_s1`
  - gated `test_export_logits_equal_and_value_scaled_by_c`
  - `test_initial_checkpoint_preflight_and_provenance`
  - `test_evaluate_false_skips_panels`
  - `test_plan_fields_survive_reevaluate`
  - `test_start_shard_and_offset`
  - `test_settled_win_rate_window`
  - `test_plans_differ_only_in_allowed_fields`
  - `test_nolevel4_recipe_diff_and_panel_identity`
- [ ] **Step 2: Implement it, run the tests and commit** `feat: start PPO from distilled S1`.

### Task 14: PPO pilot, distilled arm and decline diagnostic (controller)

- [ ] **Step 1: Pilot.** About 100,000 decisions from exported S1 seed 1, with `evaluate: false`, at learning rates 3e-4 and 1e-4. Each needs its own export. Choose by the settled win rate. That criterion favours the smaller rate; the doc says so. No panels are used.
- [ ] **Step 2: Distilled arm.** Two full runs with r1's and r2's seeds. Score them on the selection and boss panels at the five checkpoints. Report the early value-loss transient.
- [ ] **Step 3: Decline diagnostic** (delta 17). Run the four branch runs, score them on both panels, and report against the reading.

### Task 15: Gates and results

**Files:**
- Create `scripts/neural/distill_gates.py`, reusing `standard_results.mcnemar_exact`.
- Test: `tests/unit/test_distill_gates.py`.
- Create `docs/neural-distillation-gym-1.2.4.md`, plus a README link.

- [ ] **Step 1: Write the failing tests:**
  - `test_beats_needs_both_pairs_on_paired_wins_and_pooled_p`
  - `test_drop_censored_escaped_per_pair_and_report_counts`
  - `test_alpha_per_gate`
  - `test_g2_any_of_three`
- [ ] **Step 2: Implement and commit,** then run the gates.
- [ ] **Step 3: The doc.** Aggregates only. It covers:
  - provenance and the dataset;
  - R1 and the run panel;
  - the searches;
  - strength on both panels: the students, PPO, the distilled arm, the teacher, and Baalorlord L0;
  - HP kept;
  - agreement against the reference;
  - G2–G4, with p-values, effect sizes and censor counts;
  - the size and data trends, and inference times;
  - the decline diagnostic against its reading.

  Sonnet reviews it, then a PR and merge.
- [ ] **Step 4: Stop.** Report the verdicts to the user, with two follow-up decisions: the transformer PPO arm (delta 16), and run-sourced combat scenarios.
