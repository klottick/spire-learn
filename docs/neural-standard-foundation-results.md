# Standard foundation results

Everything here is a simulator-only diagnostic on one player's reconstructed fight states. Tables are regenerable
with `scripts/neural/standard_results.py` from the local artifacts; only aggregates are published, never run
or case identifiers.

## Winnability reference

Measured 2026-11-20 on the 256-case selection panel of `configs/neural/baalorlord-standard-v1`
(panel manifest SHA256 `e536d53dd17464940d8fdec6ad3a1ffe9d3dbbb8f7e858e323335b6388aab8a2`; 64 cases at each
fuzz level 0, 1, 2 and 4, stratified by act x room kind). Gym revision
`e3c13836bd1458be8a87b93160f5c350f78fcd80` (rules epoch `gym-1.1.1`), Standard rules, Ironclad ascension 20.

**Why this replaces the first measurement.** An earlier version of this section reported 219 of 256 on a panel
seeded from the recipe hash before the recipe's fuzz-schedule boundaries (`from_shard` 99, 198, 297) were set in
Task 9. Changing the schedule changes the recipe hash, and the panel is seeded from that hash, so the panel
changed (different cases, same stratification). Those numbers are superseded. The numbers below come from the
same panel that both baseline runs were evaluated on; the manifest SHA256 above is identical in both baseline
summaries and in the search output.

**Method.** `scripts/neural/boss_winnability.py --standard-panel RECIPE` plays each panel entry directly as a
Standard catalog entry (no M10 conversion) with Gym `PublicSearch`: at every decision it samples worlds
consistent with public information and scores each legal action by mean heuristic playout value
(common random numbers across actions). A decision whose root the search cannot model (hidden intents, draw
knowledge beyond the exact model, nested targeting) falls back to a play-before-end-turn heuristic.
Search is a measurement, never a training input.

**Budget.** 32 samples, 80 playout steps, `max_steps` 1200, one search-played fight per case, 4 CPUs, seed
20261117. The full pass took about 56 min wall (0.94 h summed fight time). Container run exit status 0, learner
unchanged.

**Overall.** Search won 231 of 256 (23 deaths, 2 escapes; escapes are neither wins nor losses). By level it won
59, 57, 58 and 57 of 64 at levels 0, 1, 2 and 4. On the 64 level-0 cases (his unmodified states) search won 59
and 5 died. His recorded outcome on the same 64 fights is 63 wins and 1 loss.

Search wins / cases by act x kind x fuzz level, beside his recorded wins on the level-0 cases (level-0
cases are his unmodified states; fuzzed levels have no recorded outcome):

| Act | Kind | L0 | L1 | L2 | L4 | All levels | His recorded wins, L0 |
|---|---|---|---|---|---|---|---|
| 1 | monster | 14/14 | 14/14 | 14/14 | 14/14 | 56/56 | 14/14 |
| 1 | elite | 6/7 | 6/7 | 7/7 | 7/7 | 26/28 | 7/7 |
| 1 | boss | 1/2 | 2/2 | 1/2 | 2/2 | 6/8 | 1/2 |
| 2 | monster | 10/11 | 11/11 | 10/11 | 10/11 | 41/44 | 11/11 |
| 2 | elite | 5/5 | 5/5 | 3/5 | 4/5 | 17/20 | 5/5 |
| 2 | boss | 2/2 | 0/2 | 2/2 | 2/2 | 6/8 | 2/2 |
| 3 | monster | 10/10 | 9/10 | 10/10 | 10/10 | 39/40 | 10/10 |
| 3 | elite | 4/5 | 5/5 | 5/5 | 4/5 | 18/20 | 5/5 |
| 3 | boss | 4/4 | 3/4 | 4/4 | 2/4 | 13/16 | 4/4 |
| 4 | elite | 1/2 | 1/2 | 1/2 | 1/2 | 4/8 | 2/2 |
| 4 | boss | 2/2 | 1/2 | 1/2 | 1/2 | 5/8 | 2/2 |

A recorded win means the fight was won in his run. His Ironclad rate over all kept states is 228 lost fights out of 14,645
(1.56%; 842 of 49,895, 1.7%, across all four characters), so 63/64 here is near the top by construction and a property of this sample, not a ceiling. On
this panel search is below his recorded outcome in a few cells; at level 0 the one-fight gaps are act 1 elite, act 2 monster, act 3 elite and act 4 elite; with
2 to 14 cases per cell these differences are mostly one or two fights.

**Excluded strata.** None. Every stratum has at least one search win or one recorded win, so the recipe's
`excluded_strata` stays empty.

**Caveats.**
- Search with heuristic playouts underestimates the achievable ceiling; its numbers are a lower bound.
- One player's data: states come from Baalorlord's runs only, and his recorded win rate reflects selection
  into reaching those states.
- Simulator results only; no claim about transfer to the original game.
- The per-stratum counts are small (2 to 14 per cell), so single-cell differences are noise.
- RNG-001 and TIME-001 qualifications apply: simulator randomness and timing/turn-clock behaviour are not
  claimed to match the original game.
- Search decided 5843 decisions with the search proper and fell back to the heuristic on 498 unmodelled roots
  (438 draw knowledge beyond the exact model, 59 hidden intents, 1 nested targeting); 190 playout branches
  were refused by the engine and scored as losses.

## Baseline setup

Two replicates of the same baseline, differing only in seed streams, trained with PPO on the Standard foundation
curriculum (`standard-standalone`, Ironclad ascension 20) and evaluated on the panel above.

| | r1 | r2 |
|---|---|---|
| Plan | `configs/neural/baseline-standard-1.1.1/plan.json` | `configs/neural/baseline-standard-1.1.1-r2/plan.json` |
| Baseline id | `baseline-standard-1.1.1-20261120` | `baseline-standard-1.1.1-r2-20261120` |
| Seeds (policy / optimizer / curriculum / noncombat) | 1103 / 2207 / 3301 / 4409 (the cv2 seeds) | 5101 / 6203 / 7307 / 8419 (the cv2-r2 seeds) |
| Run summary config SHA256 | `2acd35ff...f232` | `5b143fa6...0684` |
| Training wall incl. GPU waits | 14,372 s | 12,219 s |
| GPU-wait total during training | 3,510 s (5 waits) | about 1,440 s over the whole run |
| Training wall excl. waits | about 10,860 s | about 10,780 s |
| Learner dirty / changed | clean / `learner_changed` false | clean / `learner_changed` false |

The plan SHA256 values of the committed `plan.json` files are
`e37c073eca5173c41af35055667e27ab19175bede89eab9289c9a698e4340576` (r1) and
`eae508fdf5d0a8e009eb19a523f4946ca2f98ea44ea580aa35cc1196e3c199a5` (r2).

- **Recipe.** `configs/neural/baalorlord-standard-v1/recipe.json` (file SHA256
  `d75e2e09c9102f9665b4de664bbb1a2a55d99a594f983ffcf526bb224213a3aa`; the evaluator records the recipe
  hash `d0e08c447919ff6133bb11f0d2c1184eb8612412512e9dcf15441daee7a51b5b` in every summary). States table
  SHA256 `30e6e64441e3c5ec72d6c227982f583c47ed4535dcfdfe48a745cb99a2c4b44f`. Gym revision
  `e3c13836bd1458be8a87b93160f5c350f78fcd80`.
- **Budget.** 1,622,016 learned decisions (4 x 405,504; 396 shards of 4,096 decisions), 1 worker, 16 rows,
  planned training wall 12,600 s plus evaluation headroom. Both runs stopped with `target_reached`.
- **Fuzz schedule.** Shards 0 to 98 use level {0}; 99 to 197 levels {0, 1}; 198 to 296 levels {0, 1, 2}; 297
  onward levels {0, 1, 2, 4}. The three intermediate checkpoints are the weights at the end of shards 98, 197
  and 296 (405,504, 811,008 and 1,216,512 decisions), before each new level joins.
- **Fuzz pools.** Replacement cards and relics are drawn from those held in the kept train-split Ironclad states
  (the reconstructed table), not from every card or relic seen anywhere in his train-split runs: a card or relic
  that appears only on dropped floors is not in the pool. The spec's wording ("appearing in train-split runs")
  is broader; the pool is narrower by exactly the dropped floors.
- **Evaluation.** Selection panel, 256 cases, deterministic (argmax) action selection, 1024 actions and 120 s
  per case, on `cuda:0`. Checkpoints retained and evaluated: fresh (untrained), the three intermediates and
  final.
- **r1 evaluation path.** r1's original run reported `incomplete` only because a bug skipped the three
  intermediate-checkpoint evaluations (fixed in PR #31). Its checkpoints were re-evaluated with
  `--reevaluate`, with no retraining; the numbers below are from that complete re-evaluation (fresh and final
  match the original run's). r1 trained at learner revision `8444d68` and was re-evaluated at `0c08ed4`; r2
  ran at `0c08ed4`. Both manifests record a clean learner tree.
- **Accounting.** An outcome is a win, a death, an escape, or censored (the case hit its action or time cap).
  Win counts below are out of 256; escapes and censored cases count as not won. Paired tests exclude escaped and
  censored cases.

## Learning curves

Wins out of 256 on the selection panel, with escaped and censored cases in the same columns.

| Checkpoint | Decisions | Fuzz levels in training so far | r1 wins | r1 esc / cens | r2 wins | r2 esc / cens |
|---|---:|---|---:|---:|---:|---:|
| Fresh (untrained) | 0 | none | 35 | 0 / 0 | 154 | 0 / 3 |
| 25% | 405,504 | {0} | 177 | 3 / 3 | 188 | 3 / 3 |
| 50% | 811,008 | {0, 1} | 170 | 3 / 2 | 182 | 1 / 3 |
| 75% | 1,216,512 | {0, 1, 2} | 184 | 3 / 1 | 181 | 3 / 4 |
| Final | 1,622,016 | {0, 1, 2, 4} | 182 | 3 / 2 | 184 | 3 / 0 |

Mean HP retention over completed cases at fresh and final: r1 0.06 and 0.41; r2 0.33 and 0.43. Search wins 231
of 256 on the same panel. The 25% checkpoint has seen only level-0 shards, the 50% checkpoint levels 0 and 1, and
so on; each checkpoint is the weights at the end of the shard named, and was evaluated on all four panel
levels.

## Paired tests

Exact two-sided McNemar tests on the per-case win/not-win pairing, excluding cases that are censored or escaped
in either checkpoint.

| Comparison | Paired cases | Both win | Only first wins | Only second wins | Neither | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| r1 fresh vs final | 251 | 28 | 7 | 154 | 62 | 3.5e-37 |
| r2 fresh vs final | 250 | 145 | 8 | 37 | 60 | 1.5e-05 |
| r1 final vs r2 final | 251 | 162 | 20 | 21 | 48 | 1.0 |

The fresh-vs-final values are the `paired_fresh_vs_final` blocks of the two run summaries; the replicate row was
computed with the script on cases that are a win or a loss in both finals. Final r1 and r2 agree on 210 of 251
paired cases (162 wins, 48 losses) and disagree on 41, nearly symmetrically. The two finals' near-identical win
counts (182 and 184) therefore do not come from the same set of cases; each differs from the other on about one
case in six of the 251.

## Final wins by stratum and level

Final-checkpoint wins out of the panel cases in each cell, beside search wins and the fresh checkpoints.
"His L0" is the recorded outcome of the same states at level 0 only, out of the level-0 cases ("L0 cases").

By act x kind:

| Act / kind | Cases | r1 final | r2 final | Search | r1 fresh | r2 fresh | L0 cases | His L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 monster | 56 | 50 | 55 | 56 | 22 | 54 | 14 | 14 |
| 1 elite | 28 | 13 | 15 | 26 | 1 | 11 | 7 | 7 |
| 1 boss | 8 | 4 | 3 | 6 | 0 | 2 | 2 | 1 |
| 2 monster | 44 | 41 | 40 | 41 | 4 | 32 | 11 | 11 |
| 2 elite | 20 | 11 | 10 | 17 | 1 | 8 | 5 | 5 |
| 2 boss | 8 | 2 | 3 | 6 | 0 | 1 | 2 | 2 |
| 3 monster | 40 | 36 | 35 | 39 | 6 | 30 | 10 | 10 |
| 3 elite | 20 | 15 | 16 | 18 | 1 | 11 | 5 | 5 |
| 3 boss | 16 | 7 | 4 | 13 | 0 | 3 | 4 | 4 |
| 4 elite | 8 | 1 | 2 | 4 | 0 | 2 | 2 | 2 |
| 4 boss | 8 | 2 | 1 | 5 | 0 | 0 | 2 | 2 |

By fuzz level (each level has 64 cases; his recorded outcome exists only at level 0):

| Level | r1 final | r2 final | Search | r1 fresh | r2 fresh | His L0 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 52 | 54 | 59 | 13 | 45 | 63 |
| 1 | 46 | 47 | 57 | 5 | 38 | n/a |
| 2 | 42 | 44 | 58 | 11 | 36 | n/a |
| 4 | 42 | 39 | 57 | 6 | 35 | n/a |

By act: act 1 (92 cases) 67 and 73 final wins for r1 and r2 against 88 for search; act 2 (72) 54 and 53 against 64;
act 3 (76) 58 and 55 against 70; act 4 (16) 3 and 3 against 9.

## Reading

What the data support:

- **Both finals are well above r1's untrained checkpoint (35 wins) and about the same as each other.** Final wins are
  182 (r1) and 184 (r2) of 256, with a replicate disagreement that is symmetric (20 against 21 discordant cases,
  exact p = 1.0). Both are below search's 231 and below his recorded 63 of 64 level-0 outcomes; at level 0 the finals
  win 52 and 54 of 64 against search's 59.
- **The untrained policy is already strong in r2.** r2's fresh checkpoint, a deterministic argmax over a random
  initialisation, won 154 of 256, against 35 for r1's. This is consistent with many panel fights being winnable by simple play, and
  the fresh policy's win count depends heavily on the seed. That is consistent with the fresh-vs-final gain differing between runs
  (+147 wins for r1, +30 for r2) while the finals match. The fresh-vs-final tests are significant for both runs
  (p = 3.5e-37 and 1.5e-05), so training helped in both in the paired fresh-vs-final sense, but r2's gain is much smaller than r1's.
- **Most of the gain arrives by the 25% checkpoint, then the curve is flat.** r1 goes 35 to 177 by 405,504
  decisions and then 170, 184, 182; r2 goes 154 to 188 and then 182, 181, 184. After 25% the differences between
  checkpoints (at most 14 wins in r1 and 7 in r2) are of the size of the replicate disagreement and are not
  tested here. The 25% checkpoint has trained on level-0 states only, yet it already wins 44, 41 and 38 (r1) and 47, 43
  and 45 (r2) of 64 at levels 1, 2 and 4, within a few wins of the finals at each level.
- **Where the finals lose.** Both finals are close to search in monsters (act 1 to 3: 127 and 130 of 140 against
  136) and far below it in elites and bosses (for example act 1 to 3 elites 39 and 41 of 68 against 61; act 3 bosses
  7 and 4 of 16 against 13) and in act 4 (3 and 3 of 16 against 9). Per-cell counts are 2 to 14, so only the
  broad pattern is informative.
- **Fuzzed states are lost more than unmodified ones.** Final wins fall from 52 and 54 of 64 at level 0 to 39 to
  44 at levels 2 and 4, while search stays at 57 to 59. Hypothesis, not tested here: the policy
  generalises less well to states with swapped cards and relics, or the fuzzed states are harder for any
  player; the search column suggests the latter is at most a small part.
- Plausible, untested explanations for the plateau include model capacity (embedding 16, hidden 64), the
  per-shard schedule (levels 1, 2 and 4 are only trained on in later quarters), PPO optimisation settings
  and the single-fight reward. This baseline does not distinguish between them.

## Reconstruction drops

From `run-states-report.json` (reconstruction of the same runs into fight states; 49,895 states kept across the
four characters). Ironclad: 17,262 states considered, 14,645 kept and 2,617 dropped (15.2%); 94 Mind Bloom boss
battles were excluded by scope and not counted. Drop reasons for Ironclad: deck unrecoverable 1,456, untracked
deck relic 373, relic counter unrecorded 295, bottled card missing 211, HP unrecorded 193, gold unrecorded 74,
event fight reordered 15. Per-character drop rates are Defect 14.0%, Ironclad 15.2%, Silent 15.1% and Watcher
24.5%. Each stratum's rate differs: for Ironclad, act 1 elites drop 8.5%, act 1 bosses 11.9%, act 2 and 3
monsters 16.0% and 18.4%, act 3 elites 17.2%, and event fights (not in the panel) 20% to 35%. Drops are
therefore not uniform across strata, so the kept sample over-represents the strata that reconstruct more easily.
Potions are assumed empty (tagged `potions_assumed_empty`) for 3,174 of the 14,645 kept Ironclad states
(21.7%); 11 of the 64 level-0 panel cases are tagged, and 6 of the 64 are burning elites.

## Limits and qualifications

- **Simulator only.** Every number here is a simulator result on reconstructed states; there is no claim about
  the original game. The RNG-001 and TIME-001 qualifications apply.
- **One player's data.** The panel and curriculum use Baalorlord's Ironclad A20 states only; the
  reconstructed table covers all four characters. His recorded Ironclad outcomes (63 of 64
  at level 0; 1.56% of kept Ironclad fights lost) reflect selection into reaching those states.
- **Labelling.** Burning elites are labelled only from the green-key log, and the burning elite kind is sampled
  (371 Ironclad states carry `burning_kind_sampled`). Potions are assumed empty where tagged (above). Reconstruction
  drops differ by character and stratum (above).
- **Persistent relic counters are defaulted.** No run file records the counters of Pen Nib, Nunchaku, Ink Bottle,
  Incense Burner, Happy Flower, Sundial or Inserter (`relic_stats` holds lifetime trigger totals, not the
  counter), so every state holding one starts at the Gym's default counter, not at the value he had. In the
  current states table 5,678 of the 14,645 kept Ironclad states (38.8%) hold at least one of these seven relics
  (6,535, 44.6%, counting the out-of-combat Wing Boots, Tiny Chest and Matryoshka, which also default). For
  those states, level 0 is his state except for these counters, so "his exact state" and the "His L0"
  comparisons are not like for like there: a Pen Nib he entered at 9 does not double his first attack. The
  current table predates the tag; reconstruction now tags such states `relic_counter_defaulted`, and the
  generator recomputes the tag for every shard and panel case, so panel summaries stratify by it. The real
  counters are not recovered (deferred to project 2).
- **Thin cells.** The panel is proportional to the kept states, so bosses are few (40 of 256 cases, with 2 to
  4 per level in each act cell). Cells hold 2 to 14 cases per level; do not read single-cell
  differences.
- **Two replicates.** The r2 and r1 finals agree overall, but two seeds do not bound seed variance, and the fresh
  checkpoints differ by 119 wins between them.
- **Search is a lower bound.** It uses heuristic playouts and falls back to a heuristic on unmodelled roots, so
  231 of 256 underestimates what is winnable.
- **One panel, one metric.** All figures use the same 256-case selection panel with deterministic argmax
  evaluation; final-evaluation cases were not used. Checkpoint choice here is by schedule, not by selection.
- **Not comparable with M10.** Rules, observation view, panels and training cases differ from the M10 results;
  this baseline starts a new series.
