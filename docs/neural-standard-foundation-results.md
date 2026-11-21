# Standard foundation results

Everything here is a simulator-only diagnostic on one player's reconstructed fight states. Tables are regenerable
with `scripts/neural/standard_results.py` from the local artifacts; only aggregates are published, never run
or case identifiers.

## Panel history

The selection panel has moved twice, and all numbers in this document are on the latest one. Panel manifest
SHA256 `66c9835c11732fafe2f34f0aef554b1835145d9c8c0c73368769d654504d17d0`, panel identity SHA256
`1f1d96f78114b7ec1b78cd0cb22905bcc1695ec3ec31d3889ea260e332fe7c28` (256 cases, 64 at each fuzz level 0, 1,
2 and 4, stratified by act x room kind; monster, elite and boss rooms only). Earlier versions of this document
used two older panels:

- `0bf7b55e1f5ae243e05fe653456b2a982b5820e1325da7ed73c6c9f0aaef8b7a`, the panel of the first winnability run
  (219 of 256 for search). The panel is seeded from the recipe hash, and Task 9 then set the recipe's
  fuzz-schedule boundaries (`from_shard` 99, 198, 297), which changed the hash and so the panel.
- `e536d53dd17464940d8fdec6ad3a1ffe9d3dbbb8f7e858e323335b6388aab8a2`, on which the two baseline runs and the
  second search were first evaluated (search 231 of 256). The final review then gave panels an identity that
  depends on the panel alone and removed event rooms, which moved the panel again.

The training runs are unchanged: the same checkpoints were evaluated again on the current panel (evaluation
only, no retraining), and search was re-run on it. Numbers from the older panels are superseded and are not
comparable case by case; they are not reproduced here.

## Winnability reference

Measured 2026-11-21 on the 256-case selection panel of `configs/neural/baalorlord-standard-v1` (panel above).
Gym revision `e3c13836bd1458be8a87b93160f5c350f78fcd80` (rules epoch `gym-1.1.1`), Standard rules,
Ironclad ascension 20. The manifest SHA256 is identical in both baseline summaries (all five checkpoints each)
and in the search output; `scripts/neural/standard_results.py` checks this before printing any table.

**Method.** `scripts/neural/boss_winnability.py --standard-panel RECIPE` plays each panel entry directly as a
Standard catalog entry (no M10 conversion) with Gym `PublicSearch`: at every decision it samples worlds
consistent with public information and scores each legal action by mean heuristic playout value
(common random numbers across actions). A decision whose root the search cannot model (hidden intents, draw
knowledge beyond the exact model, nested targeting) falls back to a play-before-end-turn heuristic.
Search is a measurement, never a training input.

**Budget.** 32 samples, 80 playout steps, `max_steps` 1200, one search-played fight per case, 4 CPUs, seed
20261117. The full pass took about 51 min wall (0.85 h summed fight time). Container run exit status 0, learner
unchanged.

**Overall.** Search won 220 of 256 (33 deaths, 3 escapes; escapes are neither wins nor losses). By level it won
57, 56, 55 and 52 of 64 at levels 0, 1, 2 and 4. On the 64 level-0 cases (his unmodified states) search won 57,
lost 5 and escaped 2. His recorded outcome on the same 64 fights is 64 wins.

Search wins / cases by act x kind x fuzz level, beside his recorded wins on the level-0 cases (level-0
cases are his unmodified states; fuzzed levels have no recorded outcome):

| Act | Kind | L0 | L1 | L2 | L4 | All levels | His recorded wins, L0 |
|---|---|---|---|---|---|---|---|
| 1 | monster | 14/14 | 14/14 | 14/14 | 13/14 | 55/56 | 14/14 |
| 1 | elite | 7/7 | 7/7 | 5/7 | 6/7 | 25/28 | 7/7 |
| 1 | boss | 1/2 | 2/2 | 2/2 | 0/2 | 5/8 | 2/2 |
| 2 | monster | 10/11 | 10/11 | 10/11 | 11/11 | 41/44 | 11/11 |
| 2 | elite | 4/5 | 3/5 | 5/5 | 2/5 | 14/20 | 5/5 |
| 2 | boss | 1/2 | 1/2 | 2/2 | 1/2 | 5/8 | 2/2 |
| 3 | monster | 10/10 | 9/10 | 10/10 | 10/10 | 39/40 | 10/10 |
| 3 | elite | 4/5 | 4/5 | 4/5 | 4/5 | 16/20 | 5/5 |
| 3 | boss | 4/4 | 3/4 | 2/4 | 4/4 | 13/16 | 4/4 |
| 4 | elite | 2/2 | 2/2 | 1/2 | 1/2 | 6/8 | 2/2 |
| 4 | boss | 0/2 | 1/2 | 0/2 | 0/2 | 1/8 | 2/2 |

A recorded win means the fight was won in his run. His Ironclad rate over all kept states is 228 lost fights out
of 14,645 (1.56%; 842 of 49,895, 1.7%, across all four characters), so 64/64 here is near the top by construction
and a property of this sample, not a ceiling. At level 0 search is below his recorded outcome in act 1 boss
(1/2), act 2 monster (10/11), act 2 elite (4/5), act 2 boss (1/2), act 3 elite (4/5) and act 4 boss (0/2), seven
fights in all; with 2 to 14 cases per cell these are one or two fights each. Act 4 bosses are the weakest cell for
search (1 of 8 over all levels), which is the expected underestimate of the ceiling and not a statement that
those states are unwinnable.

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
- Search decided 5573 decisions with the search proper and fell back to the heuristic on 569 unmodelled roots
  (225 draw knowledge beyond the exact model, 339 hidden intents, 5 nested targeting); 282 playout branches
  were refused by the engine and scored as losses.

## Baseline setup

Two replicates of the same baseline, differing only in seed streams, trained with PPO on the Standard foundation
curriculum (`standard-standalone`, Ironclad ascension 20) and evaluated on the current panel (see Panel history).

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
  `d75e2e09c9102f9665b4de664bbb1a2a55d99a594f983ffcf526bb224213a3aa`; the summaries from the earlier panel recorded the
  recipe hash `d0e08c447919ff6133bb11f0d2c1184eb8612412512e9dcf15441daee7a51b5b`; current summaries record the panel
  identity instead). States table
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
- **Evaluation path.** r1's original run reported `incomplete` only because a bug skipped the three
  intermediate-checkpoint evaluations (fixed in PR #31). r1 trained at learner revision `8444d68` and r2 at
  `0c08ed4`. All numbers below come from later evaluation-only passes of the existing checkpoints on the current
  panel (r1 with `--reevaluate` of its original run; r2 likewise), run at learner revision `2d5508b` with a clean
  tree, exit status 0 and `learner_changed` false; there was no retraining.
- **Accounting.** An outcome is a win, a death, an escape, or censored (the case hit its action or time cap).
  Win counts below are out of 256; escapes and censored cases count as not won. Paired tests exclude escaped and
  censored cases.

## Learning curves

Wins out of 256 on the selection panel, with escaped and censored cases in the same columns.

| Checkpoint | Decisions | Fuzz levels in training so far | r1 wins | r1 esc / cens | r2 wins | r2 esc / cens |
|---|---:|---|---:|---:|---:|---:|
| Fresh (untrained) | 0 | none | 24 | 0 / 0 | 149 | 0 / 8 |
| 25% | 405,504 | {0} | 169 | 6 / 6 | 184 | 6 / 5 |
| 50% | 811,008 | {0, 1} | 173 | 6 / 6 | 179 | 4 / 6 |
| 75% | 1,216,512 | {0, 1, 2} | 184 | 6 / 1 | 180 | 4 / 5 |
| Final | 1,622,016 | {0, 1, 2, 4} | 186 | 6 / 0 | 181 | 6 / 1 |

Mean HP retention over completed cases at fresh and final: r1 0.04 and 0.44; r2 0.32 and 0.44. Search wins 220
of 256 on the same panel. Censored cases (hit the action or time cap) are 0 to 8 per checkpoint, at most 3%.
The 25% checkpoint has seen only level-0 shards, the 50% checkpoint levels 0 and 1, and so on; each checkpoint
is the weights at the end of the shard named, and was evaluated on all four panel levels. Wins by level
(L0 / L1 / L2 / L4, 64 cases each):

| Checkpoint | r1 | r2 |
|---|---|---|
| Fresh | 3 / 6 / 6 / 9 | 38 / 39 / 36 / 36 |
| 25% | 45 / 42 / 43 / 39 | 49 / 44 / 46 / 45 |
| 50% | 46 / 46 / 43 / 38 | 48 / 42 / 44 / 45 |
| 75% | 47 / 47 / 45 / 45 | 46 / 44 / 46 / 44 |
| Final | 53 / 46 / 45 / 42 | 46 / 47 / 44 / 44 |

## Paired tests

Exact two-sided McNemar tests on the per-case win/not-win pairing, excluding cases that are censored or escaped
in either checkpoint.

| Comparison | Paired cases | Both win | Only first wins | Only second wins | Neither | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| r1 fresh vs final | 250 | 24 | 0 | 162 | 64 | 3.4e-49 |
| r2 fresh vs final | 243 | 144 | 4 | 34 | 61 | 6.0e-07 |
| r1 final vs r2 final | 249 | 169 | 16 | 12 | 52 | 0.57 |

The fresh-vs-final values are the `paired_fresh_vs_final` blocks of the two run summaries; the replicate row was
computed with the script on cases that are a win or a loss in both finals (the summary and curve counts differ
slightly because paired counts leave out censored and escaped cases). The two finals agree on 221 of 249 paired
cases (169 wins, 52 losses) and disagree on 28, with 16 won only by r1 and 12 only by r2. The finals' win counts
(186 and 181) are not drawn from the same set of cases; they differ on about one paired case in nine, and the
test does not separate them.

## Final wins by stratum and level

Final-checkpoint wins out of the panel cases in each cell, beside search wins and the fresh checkpoints.
"His L0" is the recorded outcome of the same states at level 0 only, out of the level-0 cases ("L0 cases").

By act x kind:

| Act / kind | Cases | r1 final | r2 final | Search | r1 fresh | r2 fresh | L0 cases | His L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 monster | 56 | 51 | 51 | 55 | 14 | 49 | 14 | 14 |
| 1 elite | 28 | 21 | 16 | 25 | 0 | 10 | 7 | 7 |
| 1 boss | 8 | 3 | 1 | 5 | 0 | 2 | 2 | 2 |
| 2 monster | 44 | 37 | 35 | 41 | 5 | 33 | 11 | 11 |
| 2 elite | 20 | 11 | 12 | 14 | 2 | 6 | 5 | 5 |
| 2 boss | 8 | 3 | 3 | 5 | 0 | 1 | 2 | 2 |
| 3 monster | 40 | 36 | 38 | 39 | 3 | 33 | 10 | 10 |
| 3 elite | 20 | 15 | 14 | 16 | 0 | 10 | 5 | 5 |
| 3 boss | 16 | 6 | 6 | 13 | 0 | 3 | 4 | 4 |
| 4 elite | 8 | 3 | 3 | 6 | 0 | 2 | 2 | 2 |
| 4 boss | 8 | 0 | 2 | 1 | 0 | 0 | 2 | 2 |

By fuzz level (each level has 64 cases; his recorded outcome exists only at level 0):

| Level | r1 final | r2 final | Search | r1 fresh | r2 fresh | His L0 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 53 | 46 | 57 | 3 | 38 | 64 |
| 1 | 46 | 47 | 56 | 6 | 39 | n/a |
| 2 | 45 | 44 | 55 | 6 | 36 | n/a |
| 4 | 42 | 44 | 52 | 9 | 36 | n/a |

By act: act 1 (92 cases) 75 and 68 final wins for r1 and r2 against 85 for search; act 2 (72) 51 and 50 against 60;
act 3 (76) 57 and 58 against 68; act 4 (16) 3 and 5 against 7.

## Reading

What the data support:

- **Both finals are well above r1's untrained checkpoint (24 wins) and in the same range as each other.** Final wins
  are 186 (r1) and 181 (r2) of 256, with a replicate disagreement that is close to symmetric (16 against 12
  discordant cases, exact p = 0.57). Both are below search's 220 and below his recorded 64 of 64 level-0 outcomes;
  at level 0 the finals win 53 and 46 of 64 against search's 57.
- **The untrained policy is already strong in r2.** r2's fresh checkpoint, a deterministic argmax over a random
  initialisation, won 149 of 256, against 24 for r1's. This is consistent with many panel fights being winnable
  by simple play, and the fresh policy's win count depends heavily on the seed. That is consistent with the
  fresh-vs-final gain differing between runs (+162 wins for r1, +32 for r2) while the finals are close. The
  fresh-vs-final tests are significant for both runs (p = 3.4e-49 and 6.0e-07), so training helped in both in the
  paired fresh-vs-final sense, but r2's gain is much smaller than r1's.
- **Most of the gain arrives by the 25% checkpoint.** r1 goes 24 to 169 by 405,504 decisions (145 of its 162
  gained wins) and then 173, 184, 186; r2 goes 149 to 184 and then 179, 180, 181. After 25% the two runs differ:
  r1 gains a further 17 wins, and r2's 25% checkpoint is its highest, 3 above its final. These later changes are
  small next to the replicate disagreement and are not tested here. The 25% checkpoint has trained on level-0
  states only, yet it wins 42, 43 and 39 (r1) and 44, 46 and 45 (r2) of 64 at levels 1, 2 and 4, within a few wins
  of the finals at those levels (46, 45, 42 and 47, 44, 44).
- **Where the finals lose.** Search is at or above both finals in every act x kind cell except act 4 bosses (r2 2,
  search 1). In act 1 to 3 monsters the finals win 124 and 124 of 140 against 135 for search; in elites 47 and 42 of
  68 against 55; in bosses (all acts) 12 and 12 of 40 against 24. Act 4 is 3 and 5 of 16 against 7. Per-cell counts
  are 2 to 14, so only the broad pattern (a gap in elites and bosses, a smaller one in monsters) is informative.
- **Level trend.** Final wins at level 0 against level 4: r1 53 then 42, r2 46 then 44, against search 57 then 52.
  Search also falls with level, and r2's final is nearly flat across levels (46, 47, 44, 44). With 64 cases per level
  and a few wins' difference, these data do not show whether fuzzed states are harder for the policy than for
  search. Hypothesis, not tested here: swapped cards and relics make states harder for any player, or the policy
  generalises less well to them.
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
(21.7%). On the current panel, 35 of the 256 cases carry `potions_assumed_empty` (10 of the 64 level-0
cases), 8 carry `burning_kind_sampled` (3 of the 64 level-0 cases are burning elites), and 112 carry
`relic_counter_defaulted` (see Limits).

## Limits and qualifications

- **Simulator only.** Every number here is a simulator result on reconstructed states; there is no claim about
  the original game. The RNG-001 and TIME-001 qualifications apply.
- **One player's data.** The panel and curriculum use Baalorlord's Ironclad A20 states only; the
  reconstructed table covers all four characters. His recorded Ironclad outcomes (64 of 64
  at level 0; 1.56% of kept Ironclad fights lost) reflect selection into reaching those states.
- **Labelling.** Burning elites are labelled only from the green-key log, and the burning elite kind is sampled
  (371 Ironclad states, and 8 of the 256 panel cases, carry `burning_kind_sampled`). Potions are assumed empty where tagged (above). Reconstruction
  drops differ by character and stratum (above).
- **Persistent relic counters are defaulted.** No run file records the counters of Pen Nib, Nunchaku, Ink Bottle,
  Incense Burner, Happy Flower, Sundial or Inserter (`relic_stats` holds lifetime trigger totals, not the
  counter), so every state holding one starts at the Gym's default counter, not at the value he had. In the
  current states table 5,678 of the 14,645 kept Ironclad states (38.8%) hold at least one of these seven relics
  (6,535, 44.6%, counting the out-of-combat Wing Boots, Tiny Chest and Matryoshka, which also default). For
  those states, level 0 is his state except for these counters, so "his exact state" and the "His L0"
  comparisons are not like for like there: a Pen Nib he entered at 9 does not double his first attack. The
  current table predates the tag; reconstruction now tags such states `relic_counter_defaulted`, and the
  generator recomputes the tag for every shard and panel case, so panel summaries stratify by it: 112 of the 256
  current panel cases (43.8%) carry it. The real
  counters are not recovered (deferred to project 2).
- **Thin cells.** The panel is proportional to the kept states, so bosses are few (40 of 256 cases, with 2 to
  4 per level in each act cell). Cells hold 2 to 14 cases per level; do not read single-cell
  differences.
- **Two replicates.** The r1 and r2 finals are in the same range, but two seeds do not bound seed variance, and the fresh
  checkpoints differ by 125 wins between them.
- **Search is a lower bound.** It uses heuristic playouts and falls back to a heuristic on unmodelled roots, so
  220 of 256 underestimates what is winnable.
- **One panel, one metric.** All figures use the same 256-case selection panel with deterministic argmax
  evaluation; final-evaluation cases were not used. Checkpoint choice here is by schedule, not by selection.
- **Not comparable with M10.** Rules, observation view, panels and training cases differ from the M10 results;
  this baseline starts a new series.
