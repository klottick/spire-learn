# Standard panels, search reference and baselines on Gym v1.2.0

Gym v1.2.0 revision `4d198064b3b568b6f0d08d6e3937d236fbc48e9b` uses Standard rules `spire-gym-native-standard-rules-v7`. These runs used the clean tracked Learn tree at `a262c443aa6b93195b5f42776d0089d857aaae51` and `configs/neural/baalorlord-standard-gym-1.2.0/recipe.json` with state-table SHA-256 `f4c2ae7a5d5d48277d460ec27a18adf50c437c60660e6e8595fb7ad0045350d0`. The runs are under `artifacts/development/neural-combat/`. They measure simulation under RNG-001 and TIME-001. Baalorlord's recorded outcomes provide context for reconstructed entries. Original-game transfer remains unmeasured. All panel entries come from one player, and the smallest act and room cells have two cases per level.

## Panels

`t4-final-panels` built two Ascension 20 Ironclad panels of 256 cases each, with 64 cases per fuzz level (0, 1, 2 and 4). Every level has the composition below and zero skipped fuzz draws. The selection panel was scored by search. The final panel was built only.

The shared panel identity SHA-256 is `2247fe2be2b32b66a746483b974c514301de19b0bb88defaca9f2c785094c041`.

| Split | Manifest SHA-256 |
| --- | --- |
| Selection | `eb4a50d82eaf2289a318a80ab1d730f11f736edef0893c70cc8044b813fac0b2` |
| Final | `af943313f5cae7ca8a37eaaea4865e839291b36f3b62cb05572da9e94ca34418` |

| Act / room | Selection | Final |
| --- | ---: | ---: |
| 1 / boss | 2 | 2 |
| 1 / elite | 7 | 8 |
| 1 / monster | 14 | 15 |
| 2 / boss | 2 | 2 |
| 2 / elite | 5 | 5 |
| 2 / monster | 11 | 10 |
| 3 / boss | 4 | 4 |
| 3 / elite | 5 | 5 |
| 3 / monster | 10 | 9 |
| 4 / boss | 2 | 2 |
| 4 / elite | 2 | 2 |

## Selection search reference

`t4-final-search` ran one fight per case with 32 samples per action, 80 playout steps, a 1,200-step fight cap, seed 20261117, dense branch observations and a four-CPU allocation. The reference uses heuristic playouts. Commit `3101375` later changed search-branch faults from losses to errors, and the final search recorded zero such faults, so its results still hold. Search won **221 of 256** (34 deaths and one escape). On the original level-0 entries it won **58 of 64** versus **63 of 64** recorded wins for Baalorlord. The tables use the `standard_results.py` join on (run_id, floor) and count his outcomes only at level 0. Escapes contribute neither wins nor deaths.

The search used the sole legal action directly on 646 decisions. It attempted a root on the remaining decisions. When Gym refused a root, the driver recorded the refusal class and chose the action with the heuristic. The root table uses attempted roots as its denominator. Gym refused 27 branch playouts. The run recorded zero branch faults.

**By act and room kind**

| Act / kind | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 monster | 56 | 55 | 14 | 14 |
| 1 elite | 28 | 26 | 7 | 7 |
| 1 boss | 8 | 6 | 2 | 2 |
| 2 monster | 44 | 42 | 11 | 11 |
| 2 elite | 20 | 14 | 5 | 5 |
| 2 boss | 8 | 6 | 2 | 2 |
| 3 monster | 40 | 38 | 10 | 10 |
| 3 elite | 20 | 15 | 5 | 5 |
| 3 boss | 16 | 11 | 4 | 4 |
| 4 elite | 8 | 6 | 2 | 2 |
| 4 boss | 8 | 2 | 2 | 1 |

**By act, room kind and fuzz level**

| Act / kind / level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 monster L0 | 14 | 14 | 14 | 14 |
| 1 monster L1 | 14 | 13 | 0 | 0 |
| 1 monster L2 | 14 | 14 | 0 | 0 |
| 1 monster L4 | 14 | 14 | 0 | 0 |
| 1 elite L0 | 7 | 7 | 7 | 7 |
| 1 elite L1 | 7 | 7 | 0 | 0 |
| 1 elite L2 | 7 | 7 | 0 | 0 |
| 1 elite L4 | 7 | 5 | 0 | 0 |
| 1 boss L0 | 2 | 1 | 2 | 2 |
| 1 boss L1 | 2 | 2 | 0 | 0 |
| 1 boss L2 | 2 | 2 | 0 | 0 |
| 1 boss L4 | 2 | 1 | 0 | 0 |
| 2 monster L0 | 11 | 10 | 11 | 11 |
| 2 monster L1 | 11 | 11 | 0 | 0 |
| 2 monster L2 | 11 | 11 | 0 | 0 |
| 2 monster L4 | 11 | 10 | 0 | 0 |
| 2 elite L0 | 5 | 4 | 5 | 5 |
| 2 elite L1 | 5 | 4 | 0 | 0 |
| 2 elite L2 | 5 | 3 | 0 | 0 |
| 2 elite L4 | 5 | 3 | 0 | 0 |
| 2 boss L0 | 2 | 2 | 2 | 2 |
| 2 boss L1 | 2 | 1 | 0 | 0 |
| 2 boss L2 | 2 | 2 | 0 | 0 |
| 2 boss L4 | 2 | 1 | 0 | 0 |
| 3 monster L0 | 10 | 10 | 10 | 10 |
| 3 monster L1 | 10 | 9 | 0 | 0 |
| 3 monster L2 | 10 | 9 | 0 | 0 |
| 3 monster L4 | 10 | 10 | 0 | 0 |
| 3 elite L0 | 5 | 5 | 5 | 5 |
| 3 elite L1 | 5 | 4 | 0 | 0 |
| 3 elite L2 | 5 | 4 | 0 | 0 |
| 3 elite L4 | 5 | 2 | 0 | 0 |
| 3 boss L0 | 4 | 3 | 4 | 4 |
| 3 boss L1 | 4 | 4 | 0 | 0 |
| 3 boss L2 | 4 | 3 | 0 | 0 |
| 3 boss L4 | 4 | 1 | 0 | 0 |
| 4 elite L0 | 2 | 1 | 2 | 2 |
| 4 elite L1 | 2 | 2 | 0 | 0 |
| 4 elite L2 | 2 | 2 | 0 | 0 |
| 4 elite L4 | 2 | 1 | 0 | 0 |
| 4 boss L0 | 2 | 1 | 2 | 1 |
| 4 boss L1 | 2 | 0 | 0 | 0 |
| 4 boss L2 | 2 | 1 | 0 | 0 |
| 4 boss L4 | 2 | 0 | 0 | 0 |

**By fuzz level**

| Level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| L0 | 64 | 58 | 64 | 63 |
| L1 | 64 | 57 | 0 | 0 |
| L2 | 64 | 58 | 0 | 0 |
| L4 | 64 | 48 | 0 | 0 |

**By act**

| Act | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| act 1 | 92 | 87 | 23 | 23 |
| act 2 | 72 | 62 | 18 | 18 |
| act 3 | 76 | 64 | 19 | 19 |
| act 4 | 16 | 8 | 4 | 3 |

**Search-reference root outcomes (share of decisions with a root attempt)**

| Act / kind | decisions | root | draw knowledge exceeds the exact model | hidden intents are not modelled | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- |
| 1 monster | 864 | 97.454% | 1.968% | 0.579% | 0.000% |
| 1 elite | 519 | 99.615% | 0.385% | 0.000% | 0.000% |
| 1 boss | 259 | 100.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 920 | 95.109% | 2.174% | 2.609% | 0.109% |
| 2 elite | 345 | 94.783% | 5.217% | 0.000% | 0.000% |
| 2 boss | 231 | 98.268% | 1.732% | 0.000% | 0.000% |
| 3 monster | 1103 | 91.478% | 4.986% | 3.536% | 0.000% |
| 3 elite | 643 | 93.935% | 4.510% | 1.555% | 0.000% |
| 3 boss | 978 | 89.980% | 5.010% | 5.010% | 0.000% |
| 4 elite | 284 | 83.451% | 13.028% | 3.521% | 0.000% |
| 4 boss | 244 | 81.967% | 18.033% | 0.000% | 0.000% |
| all | 6390 | 93.537% | 4.304% | 2.144% | 0.016% |

The selection search refused 413 of 6,390 root attempts (6.46%). Gym v1.1.1 refused 569 of 6,142 attempts (9.3%) on its own selection panel. These are separate series with different cases, so the change in shares cannot isolate a Gym release effect.

## Training-fight root availability

`t4-final-roots` opened and immediately closed a public search handle with minimal limits at every decision to test whether the binding can construct a public root. It ran zero branches, including at single-legal-action decisions.

These tables include every decision, while the search root table excludes 646 single-legal-action decisions. A direct comparison of the root columns would mix those denominators.

The corpus covers every train-split level-0 fight state for Ironclad, Silent, Defect and Watcher in the recipe's pinned v1.2.0 table. The split hashes the source run into eight train buckets out of ten. Each entry was replayed once with the play-before-end-turn heuristic, seed 20261117 plus the entry's sorted index within its character, and a 1,200-step cap. It includes event-room fights. Four entries began terminal and contribute zero decisions. The table reports decision-weighted shares. Gym's room kind `monster` is normal combat.

| Character | Fights | Decisions | Root | Dome fights | Dome decisions | Root with Dome | Root without Dome | Terminal at reset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defect | 10,322 | 306,418 | 89.75% | 434 (4.20%) | 14,532 (4.74%) | 0.00% | 94.22% | 1 |
| ironclad | 11,724 | 329,311 | 91.08% | 611 (5.21%) | 19,995 (6.07%) | 0.00% | 96.97% | 0 |
| silent | 10,290 | 344,907 | 93.25% | 220 (2.14%) | 8,735 (2.53%) | 0.00% | 95.67% | 3 |
| watcher | 6,742 | 209,932 | 94.42% | 220 (3.26%) | 7,639 (3.64%) | 0.00% | 97.99% | 0 |
| all four | 39,078 | 1,190,568 | 91.96% | 1,485 (3.80%) | 50,901 (4.28%) | 0.00% | 96.06% | 4 |

Every decision in a fight holding Runic Dome was refused as `hidden intents are not modelled`. Dome matters for Learn's training coverage because it appears in 611 of 11,724 Ironclad training fights (5.21%) and accounts for 19,995 of their decisions.

### All four

**All entries**

| Act / room | decisions | root | hidden intents are not modelled | draw knowledge exceeds the exact model | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target | generated cards are held unoffered | a queued Mayhem play holds a random target | a queued repeat holds a randomly chosen target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 154178 | 98.261% | 0.999% | 0.708% | 0.003% | 0.029% | 0.000% | 0.000% | 0.000% |
| 1 elite | 98985 | 98.000% | 0.897% | 1.086% | 0.007% | 0.010% | 0.000% | 0.000% | 0.000% |
| 1 boss | 55243 | 96.731% | 1.035% | 2.201% | 0.027% | 0.005% | 0.000% | 0.000% | 0.000% |
| 1 event | 2262 | 99.116% | 0.000% | 0.840% | 0.044% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 193338 | 93.282% | 3.881% | 2.779% | 0.021% | 0.037% | 0.000% | 0.000% | 0.000% |
| 2 elite | 82591 | 92.884% | 4.102% | 2.936% | 0.042% | 0.035% | 0.000% | 0.000% | 0.000% |
| 2 boss | 60730 | 91.154% | 3.562% | 5.231% | 0.026% | 0.025% | 0.000% | 0.000% | 0.002% |
| 2 event | 4453 | 94.341% | 2.807% | 2.807% | 0.000% | 0.000% | 0.045% | 0.000% | 0.000% |
| 3 monster | 203699 | 88.796% | 6.415% | 4.615% | 0.067% | 0.059% | 0.036% | 0.009% | 0.001% |
| 3 elite | 137026 | 87.888% | 6.116% | 5.909% | 0.042% | 0.040% | 0.003% | 0.002% | 0.000% |
| 3 boss | 112242 | 86.214% | 6.918% | 6.812% | 0.022% | 0.034% | 0.000% | 0.000% | 0.000% |
| 3 event | 7877 | 89.260% | 6.995% | 3.567% | 0.140% | 0.038% | 0.000% | 0.000% | 0.000% |
| 4 elite | 38287 | 88.855% | 6.344% | 4.628% | 0.133% | 0.039% | 0.000% | 0.000% | 0.000% |
| 4 boss | 39657 | 87.969% | 6.375% | 5.525% | 0.101% | 0.030% | 0.000% | 0.000% | 0.000% |
| all | 1190568 | 91.957% | 4.275% | 3.686% | 0.037% | 0.035% | 0.007% | 0.002% | 0.000% |

**Runic Dome holders**

| Act / room | decisions | root | hidden intents are not modelled |
| --- | --- | --- | --- |
| 1 monster | 1541 | 0.000% | 100.000% |
| 1 elite | 888 | 0.000% | 100.000% |
| 1 boss | 572 | 0.000% | 100.000% |
| 2 monster | 7503 | 0.000% | 100.000% |
| 2 elite | 3388 | 0.000% | 100.000% |
| 2 boss | 2163 | 0.000% | 100.000% |
| 2 event | 125 | 0.000% | 100.000% |
| 3 monster | 13068 | 0.000% | 100.000% |
| 3 elite | 8380 | 0.000% | 100.000% |
| 3 boss | 7765 | 0.000% | 100.000% |
| 3 event | 551 | 0.000% | 100.000% |
| 4 elite | 2429 | 0.000% | 100.000% |
| 4 boss | 2528 | 0.000% | 100.000% |
| all | 50901 | 0.000% | 100.000% |

### Ironclad

**All entries**

| Act / room | decisions | root | hidden intents are not modelled | draw knowledge exceeds the exact model | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 42409 | 98.465% | 0.764% | 0.766% | 0.002% | 0.002% |
| 1 elite | 28474 | 98.040% | 1.004% | 0.948% | 0.004% | 0.004% |
| 1 boss | 15505 | 97.394% | 0.626% | 1.980% | 0.000% | 0.000% |
| 1 event | 738 | 99.458% | 0.000% | 0.407% | 0.136% | 0.000% |
| 2 monster | 52411 | 92.494% | 4.995% | 2.463% | 0.034% | 0.013% |
| 2 elite | 23292 | 91.555% | 5.723% | 2.653% | 0.056% | 0.013% |
| 2 boss | 15583 | 90.766% | 5.660% | 3.497% | 0.058% | 0.019% |
| 2 event | 1301 | 93.313% | 3.766% | 2.921% | 0.000% | 0.000% |
| 3 monster | 54153 | 86.955% | 9.357% | 3.610% | 0.068% | 0.009% |
| 3 elite | 35519 | 86.193% | 9.755% | 4.006% | 0.031% | 0.014% |
| 3 boss | 32262 | 85.240% | 10.610% | 4.122% | 0.022% | 0.006% |
| 3 event | 2493 | 88.087% | 7.621% | 4.212% | 0.040% | 0.040% |
| 4 elite | 11617 | 87.527% | 8.892% | 3.529% | 0.052% | 0.000% |
| 4 boss | 13554 | 86.417% | 9.060% | 4.434% | 0.081% | 0.007% |
| all | 329311 | 91.084% | 6.072% | 2.800% | 0.035% | 0.009% |

**Runic Dome holders**

| Act / room | decisions | root | hidden intents are not modelled |
| --- | --- | --- | --- |
| 1 monster | 324 | 0.000% | 100.000% |
| 1 elite | 286 | 0.000% | 100.000% |
| 1 boss | 97 | 0.000% | 100.000% |
| 2 monster | 2618 | 0.000% | 100.000% |
| 2 elite | 1333 | 0.000% | 100.000% |
| 2 boss | 882 | 0.000% | 100.000% |
| 2 event | 49 | 0.000% | 100.000% |
| 3 monster | 5067 | 0.000% | 100.000% |
| 3 elite | 3465 | 0.000% | 100.000% |
| 3 boss | 3423 | 0.000% | 100.000% |
| 3 event | 190 | 0.000% | 100.000% |
| 4 elite | 1033 | 0.000% | 100.000% |
| 4 boss | 1228 | 0.000% | 100.000% |
| all | 19995 | 0.000% | 100.000% |

Ironclad supplies the Learn recipe and both panels. The local `root_availability.json` and `standard_results.py --roots` provide separate cell breakdowns for each of the four characters. Percentages in the detailed tables are rounded to three decimal places.

## Baselines

Two replicates of the Standard foundation baseline ran on Gym v1.2.0 and were scored on the selection panel above (256 cases, deterministic argmax evaluation, 1,024 actions and 120 s per case, plus a 500-action cap on any one public turn). The learning-curve, censor-reason, level, act and room-kind stratum and paired-test tables regenerate from the local run summaries with `scripts/neural/standard_results.py` (`--run1`, `--run2`, `--panel`, `--run-states` and `--recipe` pointing at the v1.2.0 recipe). The room-kind-by-checkpoint table, the 25%-to-final and 75%-to-final tests, the tag table and the wall-time table come from one-off aggregation of the same summaries and `progress.jsonl`, and the per-eighth training diagnostics come from a one-off aggregation of the segment event streams, described below; neither is a repository script. Only aggregates are published.

| | r1 | r2 |
| --- | --- | --- |
| Plan | `configs/neural/baseline-standard-1.2.0/plan.json` | `configs/neural/baseline-standard-1.2.0-r2/plan.json` |
| Plan SHA-256 | `73413ec7ee68d930c0872f633110e494a1ebe3db21594eaf64a04eebfd7aa678` | `48806da2f5511c64badaf1952d64cbdbef6f510ec723c0c0486b3967b2524eb8` |
| Seeds (policy / optimizer / curriculum / noncombat) | 1103 / 2207 / 3301 / 4409 | 5101 / 6203 / 7307 / 8419 |
| Gym revision | `4d198064b3b568b6f0d08d6e3937d236fbc48e9b` | same |
| Learner revision, training | `9574f60` (no turn cap) | `ccf7738` (turn cap 500) |
| Learner revision, evaluations | `ccf7738`, by `--reevaluate` of the r1 checkpoints | `ccf7738` |
| Learned decisions | 1,624,192 (397 segments) | 1,622,016 (396 segments) |
| Stop reason, result | `target_reached`, `complete` | `target_reached`, `complete` |
| Learner changed | false | false |

The seeds are those of the v1.1.1 baselines; model, PPO settings, the curriculum schedule (levels {0} from shard 0, {0, 1} from shard 99, {0, 1, 2} from shard 198, {0, 1, 2, 4} from shard 297) and the checkpoint schedule are unchanged. Both learner trees were clean apart from the untracked search-distillation design note. Decisions per segment vary slightly, so r1's last segment ended 2,176 decisions (0.13%) past the 1,622,016 target. The intermediate checkpoints are the weights after 405,504, 811,008 and 1,216,512 decisions, taken before levels 1, 2 and 4 join.

r1's own training run reached its target, but its first evaluation stopped on a Gym fault (see the turn-action cap below) and that evaluation is superseded. The r1 evaluations below are a `--reevaluate` of the same five checkpoints with the turn cap. r1 did not retrain.

An outcome is a win, a death, an escape or a censored case (the case hit a cap). Win counts are out of 256; escapes and censored cases count as not won. Paired tests exclude escaped and censored cases.

### Learning curves

Wins out of 256 (escaped / censored in the second column of each replicate):

| Checkpoint | Decisions | Levels trained on | r1 wins | r1 esc / cens | r2 wins | r2 esc / cens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fresh | 0 | none | 34 | 0 / 0 | 146 | 0 / 7 |
| 25% | 405,504 | {0} | 190 | 2 / 0 | 183 | 2 / 4 |
| 50% | 811,008 | {0, 1} | 183 | 2 / 1 | 179 | 2 / 3 |
| 75% | 1,216,512 | {0, 1, 2} | 178 | 2 / 4 | 181 | 2 / 3 |
| Final | about 1.62 M | {0, 1, 2, 4} | 166 | 2 / 2 | 168 | 2 / 4 |

Search wins 221 of 256 on the same panel. Deaths are the remainder (r1 final 86, r2 final 82). Mean HP retention over completed cases at fresh and final is 0.06 and 0.37 for r1 and 0.31 and 0.38 for r2.

Wins by fuzz level (L0 / L1 / L2 / L4, 64 cases each):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 9 / 8 / 10 / 7 | 37 / 38 / 38 / 33 |
| 25% | 50 / 51 / 47 / 42 | 48 / 50 / 45 / 40 |
| 50% | 48 / 46 / 46 / 43 | 51 / 48 / 45 / 35 |
| 75% | 47 / 46 / 44 / 41 | 51 / 47 / 48 / 35 |
| Final | 46 / 40 / 42 / 38 | 42 / 47 / 42 / 37 |

Wins by room kind (monster / elite / boss, out of 140 / 76 / 40; search 135 / 61 / 25):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 31 / 3 / 0 | 111 / 27 / 8 |
| 25% | 124 / 53 / 13 | 123 / 47 / 13 |
| 50% | 124 / 47 / 12 | 124 / 47 / 8 |
| 75% | 122 / 45 / 11 | 121 / 47 / 13 |
| Final | 116 / 38 / 12 | 121 / 37 / 10 |

Fresh against final, exact two-sided McNemar tests on the paired win/not-win outcomes (censored and escaped cases excluded in either checkpoint):

| Comparison | Paired | Both win | Only fresh wins | Only final wins | Neither | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 fresh vs final | 252 | 33 | 1 | 133 | 85 | 1.2e-38 |
| r2 fresh vs final | 247 | 134 | 12 | 31 | 70 | 0.0054 |

These match the `paired_fresh_vs_final` blocks of the two run summaries. Training helped in both replicates against the untrained checkpoint, but r2's untrained policy already wins 146 cases, so its gain is small next to r1's.

### Replicate agreement

The two finals agree on 211 of 248 paired cases (147 wins by both, 64 losses by both) and differ on 37: 17 won only by r1 and 20 won only by r2 (exact p = 0.74). Their win counts (166 and 168) are not distinguishable. Escapes are the same two cases at every trained checkpoint of both replicates, and every case that r1 had censored at some checkpoint (seven distinct cases) was also censored at some checkpoint of r2, which has nine.

### Final wins by stratum and level

Final-checkpoint wins in each cell beside search and the fresh checkpoints. "L0 cases" and "his L0" count Baalorlord's recorded outcomes of the same states at level 0 only. Cells hold 2 to 14 cases per level, so single-cell differences are noise.

**By act and room kind**

| Act / kind | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 monster | 56 | 49 | 51 | 55 | 20 | 50 | 14 | 14 |
| 1 elite | 28 | 17 | 14 | 26 | 3 | 14 | 7 | 7 |
| 1 boss | 8 | 5 | 3 | 6 | 0 | 4 | 2 | 2 |
| 2 monster | 44 | 37 | 36 | 42 | 7 | 31 | 11 | 11 |
| 2 elite | 20 | 10 | 11 | 14 | 0 | 4 | 5 | 5 |
| 2 boss | 8 | 2 | 0 | 6 | 0 | 0 | 2 | 2 |
| 3 monster | 40 | 30 | 34 | 38 | 4 | 30 | 10 | 10 |
| 3 elite | 20 | 8 | 9 | 15 | 0 | 9 | 5 | 5 |
| 3 boss | 16 | 4 | 6 | 11 | 0 | 4 | 4 | 4 |
| 4 elite | 8 | 3 | 3 | 6 | 0 | 0 | 2 | 2 |
| 4 boss | 8 | 1 | 1 | 2 | 0 | 0 | 2 | 1 |

**By fuzz level**

| Level | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 64 | 46 | 42 | 58 | 9 | 37 | 64 | 63 |
| L1 | 64 | 40 | 47 | 57 | 8 | 38 | 0 | 0 |
| L2 | 64 | 42 | 42 | 58 | 10 | 38 | 0 | 0 |
| L4 | 64 | 38 | 37 | 48 | 7 | 33 | 0 | 0 |

**By act**

| Act | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| act 1 | 92 | 71 | 68 | 87 | 23 | 68 | 23 | 23 |
| act 2 | 72 | 49 | 47 | 62 | 7 | 35 | 18 | 18 |
| act 3 | 76 | 42 | 49 | 64 | 4 | 43 | 19 | 19 |
| act 4 | 16 | 4 | 4 | 8 | 0 | 0 | 4 | 3 |

Both finals trail search in every act. At level 0 they win 46 and 42 of 64 against search's 58 and Baalorlord's recorded 63. By room kind they win 116 and 121 of 140 monster cases against search's 135, 38 and 37 of 76 elite cases against 61, and 12 and 10 of 40 boss cases against 25. The act 2 boss cell is the weakest (2 and 0 of 8 against search's 6).

### Escapes, censoring and panel tags

Escapes are two per trained checkpoint and none at the untrained one. Censored cases by reason (a case is censored when it reaches the 1,024-action cap per fight, `environment_action_budget`, or the 500-action cap per public turn, `turn_action_budget`):

| Checkpoint | r1 turn cap | r1 fight cap | r2 turn cap | r2 fight cap |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 0 | 0 | 7 | 0 |
| 25% | 0 | 0 | 4 | 0 |
| 50% | 1 | 0 | 3 | 0 |
| 75% | 4 | 0 | 3 | 0 |
| Final | 2 | 0 | 4 | 0 |

No evaluation case reached the 1,024-action fight cap, and no case or evaluation faulted. Every censored case was a turn-cap case, and censoring is at most 7 of 256 (2.7%) at any checkpoint.

Panel tag counts (the panel carries `potions_assumed_empty` on 47 cases, `burning_kind_sampled` on 10, `misc_defaulted` on 6 and `relic_counter_defaulted` on 110) and wins by tag at 25% and at the final checkpoint:

| Tag | Cases | r1 25% | r1 final | r2 25% | r2 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| `potions_assumed_empty` | 47 | 33 | 27 | 31 | 30 |
| `burning_kind_sampled` | 10 | 7 | 6 | 7 | 6 |
| `misc_defaulted` | 6 | 3 | 2 | 2 | 2 |
| `relic_counter_defaulted` | 110 | 71 | 62 | 64 | 63 |
| All cases | 256 | 190 | 166 | 183 | 168 |

Tagged cases overlap, so rows are not additive. Tag cells hold 6 to 110 cases and are not tested.

### Wall time and GPU waits

Times are from `progress.jsonl` and the run manifests. A wait is the gate that holds a stage until the GPU has 13,958,643,712 free bytes.

| Run | Process wall | Of which GPU waits | Training elapsed (excluding waits) | Waits |
| --- | ---: | ---: | ---: | --- |
| r1 training and first evaluation | 10,672 s | 1,230 s | 10,081 s (8,851 s) | 3 in training (450, 390 and 390 s) |
| r1 evaluation with the turn cap | 1,092 s | 360 s | not applicable | 1 before the first evaluation |
| r2 training and evaluations | 10,757 s | 1,320 s | 9,196 s (8,296 s) | 2 in training (390 and 510 s), 1 of 420 s before the 75% evaluation |

r1's first run spent 587 s on evaluations before the fault and the re-evaluation of all five checkpoints took 1,089 s of progress time including its wait. r2's five evaluations took 1,549 s including a 420 s wait. The runs used one worker and 16 environment rows.

### Turn-action cap

r1's first evaluation hit a Gym fault on one panel case. A fight with no damage looped within a single turn (Shrug It Off and Finesse cycling with Sundial energy) until it reached the 1,000th card of that turn. The Gym found the root cause: Giant Head's Slow was capped at 999 stacks in its engine, which the original game does not do. Gym v1.2.1 fixes it (bead `spire-sjh`).

By the user's ruling, Learn censors a fight that reaches 500 actions in one public turn. In panels it is a censored case, like the 1,024-action cap. In training it is a time-limit truncation, not a loss. A simulator fault stays an error.

- r1 trained without the cap (learner `9574f60`; the committed r1 plan now carries the cap that the re-evaluation used) and its checkpoints were evaluated again with it. The first evaluation's partial results are not used.
- r2 trained with the cap. It hit the cap **0 times in training**: `turn_action_budget_hits` is absent from all 396 segment summaries, and the runner writes that field only when it is non-zero. The training streams of r1 carry no such counter, because it did not train with the cap.
- Evaluation censors for both replicates are in the table above. They are all turn-cap cases. The untrained r2 policy accounts for the most (7), and 2 to 4 cases per trained checkpoint remain, so the cap changes at most a few percent of cases at any checkpoint.

### Decline after early training

Both replicates peak early and end lower. r1 wins 34, 190, 183, 178 and 166 at fresh, 25%, 50%, 75% and final; r2 wins 146, 183, 179, 181 and 168. From 25% to final r1 lost 24 wins (190 to 166) and r2 lost 15 (183 to 168). The tests below are post hoc, not pre-registered. For 25% against final they give exact p = 6.1e-4 (33 cases won only at 25%, 10 only at final) for r1 and p = 0.014 (24 and 9) for r2. For 75% against final they give p = 0.047 (25 and 12) and p = 0.053 (26 and 13).

Where the wins went (25% to final): of the 34 cases r1 won at 25% and not at final, 20 were elites, 10 monsters and 4 bosses; it gained 10 elsewhere (5 elite, 3 boss, 2 monster). For r2 the 25 losses were 16 elites, 4 monsters and 5 bosses against 10 gained. Elite wins fall from 53 to 38 (r1) and from 47 to 37 (r2); most of the net loss is in elite fights. The decline appears at every fuzz level from 25% to final (r1 50 / 51 / 47 / 42 to 46 / 40 / 42 / 38; r2 48 / 50 / 45 / 40 to 42 / 47 / 42 / 37), and is largest at level 1 for r1 and level 0 for r2.

Training-stream diagnostics per eighth of training (each eighth is 49 or 50 consecutive segments of about 4,096 decisions; eighths 7 and 8 are the final quarter, after level 4 joins at shard 297). These come from the `optimizer_update` and `combat_settlement` records of the training segments. The win rate is wins over all settled fights (wins, deaths and escapes) under the training policy's seeded sampling; mean HP is the HP left after cleanup on a won fight. Entropy and value loss are means over the optimizer updates.

| Eighth | r1 win rate | r1 entropy | r1 value loss | r1 HP on wins | r2 win rate | r2 entropy | r2 value loss | r2 HP on wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.656 | 0.997 | 0.037 | 47.5 | 0.645 | 0.945 | 0.034 | 47.2 |
| 2 | 0.698 | 0.778 | 0.038 | 47.2 | 0.692 | 0.790 | 0.037 | 48.2 |
| 3 | 0.676 | 0.665 | 0.044 | 47.7 | 0.672 | 0.807 | 0.040 | 48.3 |
| 4 | 0.693 | 0.680 | 0.052 | 46.8 | 0.690 | 0.699 | 0.046 | 47.8 |
| 5 | 0.686 | 0.598 | 0.049 | 46.8 | 0.681 | 0.669 | 0.050 | 46.3 |
| 6 | 0.683 | 0.634 | 0.054 | 46.4 | 0.665 | 0.622 | 0.053 | 45.9 |
| 7 | 0.656 | 0.577 | 0.049 | 46.5 | 0.671 | 0.587 | 0.052 | 45.7 |
| 8 | 0.677 | 0.534 | 0.057 | 44.5 | 0.664 | 0.547 | 0.054 | 46.3 |

Each eighth holds 7,300 to 8,100 settled fights. The training win rate stays between 0.65 and 0.70 and does not fall with the evaluation wins. Entropy falls by 42 to 46% in both replicates and value loss rises by more than half (r1 0.037 to 0.057, r2 0.034 to 0.054), and mean HP on wins drifts down by 1 to 3 points. Training and evaluation differ: training samples actions while evaluation takes the argmax, and the training stream mixes more fuzz levels over time, so the training win rate is not an evaluation proxy.

What the data support and weaken:

- **Classic overfitting to the training fights. Weakened.** The training win rate is flat while the evaluation wins fall, and the decline spans every fuzz level. This does not exclude a train/evaluation mismatch.
- **Turn-cap stalling. Not supported.** r2 never hit the turn cap in training. In evaluation the final checkpoints have 2 and 4 turn-cap censors, which is too few to account for the 15 (r2) to 24 (r1) wins lost between the 25% and final checkpoints, and most lost cases are deaths (r1 86 and r2 82 losses at final against 64 and 67 at 25%).
- **Final-quarter curriculum shift (level 4 joins). Weakly supported.** Half of r1's decline (12 of 24 wins, 190 to 178) predates level 4, and r2 is flat to 75% (183, 179, 181) before dropping 13 in the last quarter. The r2 drop does not concentrate at level 4: its level-4 wins are 35 at 75% and 37 at final, and the drops are at levels 0 and 2 (51 to 42 and 48 to 42). A shift in the training mixture is therefore consistent with r2's timing, but r1 and the level breakdown do not single it out.
- **PPO drift while entropy falls and the critic worsens. Untested (consistent with the data).** Both replicates show it: entropy falls and value loss rises over the whole run, in step with the slow decline in evaluation wins. These data only show that the quantities move together; they do not show cause.

r1 trained without the cap, so stalling in r1's training cannot be measured from its records. Neither remaining explanation is tested. The final checkpoint is the pre-registered comparison point; checkpoint selection is a later decision. The v1.1.1 baselines went the other way: r1 rose from 169 at 25% to 186 at the end and r2 stayed flat around 180. That series is separate, and its numbers are for orientation only.

### Series separation

The v1.2.0 and v1.1.1 series are separate and are not paired. The cases, panel and rules differ (the Standard rules version and the turn cap), so differences between the series cannot be read as a Gym-release effect. For orientation only, the v1.1.1 finals were 186 (r1) and 181 (r2) wins of 256 on their own panel, against 166 and 168 here. Gym v1.2.1 now exists and changes the rules id, so adopting it would be a later re-pin with its own baselines.
