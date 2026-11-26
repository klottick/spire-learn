# Standard panels, search reference and baselines on Gym v1.2.4

Gym v1.2.4 revision `673eb8f220023b0c95401bbf96bec814a359457b` uses Standard rules `spire-gym-native-standard-rules-v9` and rules epoch `gym-1.2.4`. The search references below ran on the clean tracked Learn tree at `b4f8a36e42d49ebb593ccc5de07121481fba244d`. The panels, the cost unit and the root-availability run used the clean tree at `2987e5016750c2e70801d451afcd71e345d0dcea`, and the teacher budget verifications used the clean tree at `d31727c0eb8d88cd07b9afb880f679d5c227d47d`. `2987e50` and `d31727c` precede the baseline plans, and `d31727c` added the per-legal-action tree budgets (`simulations_per_action`); the `current` teacher path changed only to record each decision's simulation count. All runs used `configs/neural/baalorlord-standard-gym-1.2.4/recipe.json` (file SHA-256 `5d53d06566ae8de898de78140d10aaffe6ac39204b1d245ec98747ed09403fb5`) with state-table SHA-256 `3121570f1b970d96acd76e9c6c7681ca32789e22c9b11af74ab5aa9283dfb6a0`. The runs are under `artifacts/development/neural-combat/gym-1.2.4/t3/`. They measure simulation under RNG-001 and TIME-001. Baalorlord's recorded outcomes provide context for reconstructed entries. Original-game transfer remains unmeasured. All panel entries come from one player, and the smallest act and room cells have two cases per level.

The v1.2.0 and v1.2.4 series are separate and are not paired. The rules epoch differs and the panels were rebuilt on the new state table, so the cases are not the same cases. Where v1.2.0 numbers appear below, they are for orientation only.

## Panels

`panels` built two Ascension 20 Ironclad panels of 256 cases each, with 64 cases per fuzz level (0, 1, 2 and 4). Every level has the composition below and zero skipped fuzz draws. The selection panel was scored by search. The final panel was built only and has never been scored.

The shared panel identity SHA-256 is `338fcbef860cb83fc35251c10825280e19f257a025752bdbec60e2e84088e47a`.

| Split | Manifest SHA-256 |
| --- | --- |
| Selection | `95727539babe276a115dd8ef03a735554bc2ef80e6c7fcd83d82558ab137298d` |
| Final | `5cc0094c964b6fc441084ba205aa45af42771b9e875a964c4dadca056276ca4b` |

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

The per-level composition matches the v1.2.0 panels cell for cell. The cases differ, because the state table and the panel seed differ.

The boss panel draws 256 Ironclad A20 cases from the selection split. Each of the four fuzz levels holds 40 boss and 24 elite cases, with distinct base states within a level and act quotas proportional to availability within each room kind. Its identity is separate, so it seeds its own case selection and fuzzing. The identity and manifest come from the boss search reference's provenance (`panels` printed only the selection and final panels).

| Field | Value |
| --- | --- |
| Boss panel SHA-256 | `5f9a0f41f77cb892c6913a105132846284e3fda8ae7276dcb50d7d0c231e4b7e` |
| Boss manifest SHA-256 | `124f202e9c1f134498be71c775ca268b71bdafe5eff131a4e0fdc905700bdf57` |
| Distinct base states, boss panel / selection panel | 206 / 245 `(run_id, floor)` pairs |
| Base states shared by the two panels | 31 distinct `(run_id, floor)` pairs |

| Act / room | Cases per level |
| --- | ---: |
| 1 / elite | 9 |
| 1 / boss | 10 |
| 2 / elite | 6 |
| 2 / boss | 8 |
| 3 / elite | 6 |
| 3 / boss | 15 |
| 4 / elite | 3 |
| 4 / boss | 7 |

The sharing count and the boss composition come from a one-off, read-only rebuild of both panels in memory from the recipe, with the Gym catalog check replaced by a no-op (no fuzz draw was skipped in the recorded builds, and the rebuilt selection and boss manifest hashes match the recorded ones above). The same rebuild joins search cases to their act, room kind and level in the tables below.

## Selection search reference

`searchref-selection` ran one fight per case with teacher `current` from `configs/neural/search-teacher-gym-1.2.4/teachers.json`: 32 samples per action, 80 playout steps, native Gym playouts (`spire-gym-public-playout-v1`), the compact branch layout, one worker per search, a 1,024-step fight cap, a 500-action turn cap, seed 20261117 and `--parallel 12` fights in a 12-CPU container. Search won **233 of 256** (23 deaths, no escapes, no censored cases). On the original level-0 entries it won **58 of 64** versus **63 of 64** recorded wins for Baalorlord. The tables below count Baalorlord's outcomes only at level 0: "L0 cases" is the number of level-0 panel entries in the cell, and "his L0" is how many of those his recorded run won. Escapes contribute neither wins nor deaths.

The tables come from the `render_search` function of `scripts/neural/standard_results.py`, fed by a one-off in-memory join of `fights.jsonl` to the rebuilt panel (the script's own command line needs a `standard_panel.jsonl` that this teacher entry does not write). The recorded outcomes join on (run_id, floor) against `run-states-gym-1.2.4.json`, whose SHA-256 is the recipe pin. The root tables below are a one-off aggregation of `decisions.jsonl`.

The search used the sole legal action directly on 588 of 7,541 decisions. It attempted a root on the remaining 6,953 decisions and Gym refused 79 of them (1.136%). When Gym refused a root, the driver recorded the refusal class and chose the action with the heuristic. A root is a public search root that Gym could construct at that decision. The root table uses attempted roots as its denominator. Gym refused 11 branch playouts, on 7 decisions. A refused branch playout scores −1 in its action's mean and is counted in that action's statistics. The run recorded no branch faults. The evidence is that the process exited 0 with empty stderr: a fault stops the run as an error, and the summary has no fault counter. No case was censored. The search used 4,335.6 CPU s over 6,874 searched decisions, **0.630 CPU s per searched decision**, in 2,353 s of fight wall time.

One fight dominates that cost: an act 4 boss case at level 0, which search won in 437 steps with 435 searched decisions. It used 2,326 CPU s, 54% of the run's CPU, at 5.35 CPU s per searched decision. Its mean simulation count per decision was 353, against 204 for the other fights. Its CPU time roughly equals its wall time, so it ran on a core of its own, and for most of the run's 2,353 s it was the only fight left running. That is why the run kept only about 1.8 CPUs busy on average. The other 255 fights used 2,009.5 CPU s over 6,439 searched decisions, **0.312 CPU s per searched decision**, and the next-longest fights took 75, 55 and 54 s.

**By act and room kind**
| Act / kind | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 monster | 56 | 56 | 14 | 14 |
| 1 elite | 28 | 27 | 7 | 7 |
| 1 boss | 8 | 6 | 2 | 1 |
| 2 monster | 44 | 44 | 11 | 11 |
| 2 elite | 20 | 17 | 5 | 5 |
| 2 boss | 8 | 4 | 2 | 2 |
| 3 monster | 40 | 39 | 10 | 10 |
| 3 elite | 20 | 16 | 5 | 5 |
| 3 boss | 16 | 13 | 4 | 4 |
| 4 elite | 8 | 7 | 2 | 2 |
| 4 boss | 8 | 4 | 2 | 2 |

**By act, room kind and fuzz level**
| Act / kind / level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 monster L0 | 14 | 14 | 14 | 14 |
| 1 monster L1 | 14 | 14 | 0 | 0 |
| 1 monster L2 | 14 | 14 | 0 | 0 |
| 1 monster L4 | 14 | 14 | 0 | 0 |
| 1 elite L0 | 7 | 7 | 7 | 7 |
| 1 elite L1 | 7 | 7 | 0 | 0 |
| 1 elite L2 | 7 | 6 | 0 | 0 |
| 1 elite L4 | 7 | 7 | 0 | 0 |
| 1 boss L0 | 2 | 1 | 2 | 1 |
| 1 boss L1 | 2 | 2 | 0 | 0 |
| 1 boss L2 | 2 | 2 | 0 | 0 |
| 1 boss L4 | 2 | 1 | 0 | 0 |
| 2 monster L0 | 11 | 11 | 11 | 11 |
| 2 monster L1 | 11 | 11 | 0 | 0 |
| 2 monster L2 | 11 | 11 | 0 | 0 |
| 2 monster L4 | 11 | 11 | 0 | 0 |
| 2 elite L0 | 5 | 5 | 5 | 5 |
| 2 elite L1 | 5 | 3 | 0 | 0 |
| 2 elite L2 | 5 | 4 | 0 | 0 |
| 2 elite L4 | 5 | 5 | 0 | 0 |
| 2 boss L0 | 2 | 1 | 2 | 2 |
| 2 boss L1 | 2 | 1 | 0 | 0 |
| 2 boss L2 | 2 | 1 | 0 | 0 |
| 2 boss L4 | 2 | 1 | 0 | 0 |
| 3 monster L0 | 10 | 10 | 10 | 10 |
| 3 monster L1 | 10 | 10 | 0 | 0 |
| 3 monster L2 | 10 | 10 | 0 | 0 |
| 3 monster L4 | 10 | 9 | 0 | 0 |
| 3 elite L0 | 5 | 3 | 5 | 5 |
| 3 elite L1 | 5 | 5 | 0 | 0 |
| 3 elite L2 | 5 | 5 | 0 | 0 |
| 3 elite L4 | 5 | 3 | 0 | 0 |
| 3 boss L0 | 4 | 4 | 4 | 4 |
| 3 boss L1 | 4 | 3 | 0 | 0 |
| 3 boss L2 | 4 | 2 | 0 | 0 |
| 3 boss L4 | 4 | 4 | 0 | 0 |
| 4 elite L0 | 2 | 1 | 2 | 2 |
| 4 elite L1 | 2 | 2 | 0 | 0 |
| 4 elite L2 | 2 | 2 | 0 | 0 |
| 4 elite L4 | 2 | 2 | 0 | 0 |
| 4 boss L0 | 2 | 1 | 2 | 2 |
| 4 boss L1 | 2 | 2 | 0 | 0 |
| 4 boss L2 | 2 | 1 | 0 | 0 |
| 4 boss L4 | 2 | 0 | 0 | 0 |

**By fuzz level**
| Level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| L0 | 64 | 58 | 64 | 63 |
| L1 | 64 | 60 | 0 | 0 |
| L2 | 64 | 58 | 0 | 0 |
| L4 | 64 | 57 | 0 | 0 |

**By act**
| Act | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| act 1 | 92 | 89 | 23 | 22 |
| act 2 | 72 | 65 | 18 | 18 |
| act 3 | 76 | 68 | 19 | 19 |
| act 4 | 16 | 11 | 4 | 4 |

**Search-reference root outcomes (share of decisions with a root attempt)**

| Act / kind | decisions | root | a card read a hidden intent | draw knowledge exceeds the exact model | a hidden pre-rolled value was shown by damage | a nested card play holds an enemy target | an executed enemy move was not identifiable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 902 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 elite | 438 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 boss | 236 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 1003 | 99.701% | 0.000% | 0.000% | 0.000% | 0.299% | 0.000% |
| 2 elite | 379 | 99.736% | 0.000% | 0.000% | 0.000% | 0.264% | 0.000% |
| 2 boss | 273 | 91.575% | 0.000% | 8.059% | 0.000% | 0.366% | 0.000% |
| 3 monster | 1200 | 95.833% | 2.083% | 0.000% | 1.583% | 0.083% | 0.417% |
| 3 elite | 675 | 99.704% | 0.000% | 0.000% | 0.000% | 0.296% | 0.000% |
| 3 boss | 818 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 4 elite | 353 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 4 boss | 676 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| all | 6953 | 98.864% | 0.360% | 0.316% | 0.273% | 0.115% | 0.072% |

Refusals by class over the 6,953 attempts: a card read a hidden intent 25, draw knowledge exceeds the exact model 22, a hidden pre-rolled value was shown by damage 19, a nested card play holds an enemy target 8, and an executed enemy move was not identifiable 5. Two cells hold most of the refusals: 2 boss (8.425%, mostly draw knowledge) and 3 monster (4.167%).

For orientation only: the v1.2.0 selection search won 221 of 256 and Gym refused 413 of 6,390 root attempts (6.46%). The v1.2.4 selection search won 233 of 256 and Gym refused 79 of 6,953 (1.136%). These are not comparable. The panel cases differ, the Standard rules differ, and v1.2.0 used heuristic playouts run in Python while v1.2.4 uses native Gym playouts. The shift in wins and refusals cannot be assigned to one cause.

## Boss panel search reference

`searchref-boss` ran the same teacher, settings and seed (`current`, 32 samples, 80 playout steps, native playouts, compact layout, one worker, 20261117, `--parallel 12`, 12 CPUs) on the boss panel. Search won **176 of 256** (78 deaths and 2 escapes, no censored cases). On the original level-0 entries it won **45 of 64** versus **63 of 64** recorded wins for Baalorlord. The two escapes are act 1 elite cases at levels 2 and 4. The tables come from the same one-off join as above.

The search used the sole legal action directly on 682 of 8,748 decisions. It attempted a root on the remaining 8,066 decisions and Gym refused 16 of them (0.198%). Gym refused 205 branch playouts, on 38 decisions: 192 in act 3 boss fights, 12 in act 2 boss fights and 1 in an act 4 elite fight. The run recorded no branch faults (exit 0, empty stderr). The search used 3,780.4 CPU s over 8,050 searched decisions, **0.469 CPU s per searched decision**, in 430 s of fight wall time. This run kept about 8.8 CPUs busy on average. Its tail is milder: the two longest fights took 252 and 249 s, and without the longest the cost is 0.446 CPU s per searched decision. The two per-decision costs are for different panels and are not a like-for-like comparison.

**By act and room kind**
| Act / kind | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 elite | 36 | 32 | 9 | 9 |
| 1 boss | 40 | 21 | 10 | 9 |
| 2 elite | 24 | 21 | 6 | 6 |
| 2 boss | 32 | 19 | 8 | 8 |
| 3 elite | 24 | 22 | 6 | 6 |
| 3 boss | 60 | 41 | 15 | 15 |
| 4 elite | 12 | 9 | 3 | 3 |
| 4 boss | 28 | 11 | 7 | 7 |

**By act, room kind and fuzz level**
| Act / kind / level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| 1 elite L0 | 9 | 9 | 9 | 9 |
| 1 elite L1 | 9 | 9 | 0 | 0 |
| 1 elite L2 | 9 | 6 | 0 | 0 |
| 1 elite L4 | 9 | 8 | 0 | 0 |
| 1 boss L0 | 10 | 6 | 10 | 9 |
| 1 boss L1 | 10 | 6 | 0 | 0 |
| 1 boss L2 | 10 | 5 | 0 | 0 |
| 1 boss L4 | 10 | 4 | 0 | 0 |
| 2 elite L0 | 6 | 6 | 6 | 6 |
| 2 elite L1 | 6 | 6 | 0 | 0 |
| 2 elite L2 | 6 | 4 | 0 | 0 |
| 2 elite L4 | 6 | 5 | 0 | 0 |
| 2 boss L0 | 8 | 5 | 8 | 8 |
| 2 boss L1 | 8 | 4 | 0 | 0 |
| 2 boss L2 | 8 | 6 | 0 | 0 |
| 2 boss L4 | 8 | 4 | 0 | 0 |
| 3 elite L0 | 6 | 5 | 6 | 6 |
| 3 elite L1 | 6 | 6 | 0 | 0 |
| 3 elite L2 | 6 | 6 | 0 | 0 |
| 3 elite L4 | 6 | 5 | 0 | 0 |
| 3 boss L0 | 15 | 8 | 15 | 15 |
| 3 boss L1 | 15 | 13 | 0 | 0 |
| 3 boss L2 | 15 | 11 | 0 | 0 |
| 3 boss L4 | 15 | 9 | 0 | 0 |
| 4 elite L0 | 3 | 2 | 3 | 3 |
| 4 elite L1 | 3 | 3 | 0 | 0 |
| 4 elite L2 | 3 | 2 | 0 | 0 |
| 4 elite L4 | 3 | 2 | 0 | 0 |
| 4 boss L0 | 7 | 4 | 7 | 7 |
| 4 boss L1 | 7 | 2 | 0 | 0 |
| 4 boss L2 | 7 | 3 | 0 | 0 |
| 4 boss L4 | 7 | 2 | 0 | 0 |

**By fuzz level**
| Level | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| L0 | 64 | 45 | 64 | 63 |
| L1 | 64 | 49 | 0 | 0 |
| L2 | 64 | 43 | 0 | 0 |
| L4 | 64 | 39 | 0 | 0 |

**By act**
| Act | cases | search | L0 cases | his L0 |
| --- | --- | --- | --- | --- |
| act 1 | 76 | 53 | 19 | 18 |
| act 2 | 56 | 40 | 14 | 14 |
| act 3 | 84 | 63 | 21 | 21 |
| act 4 | 40 | 20 | 10 | 10 |

**Search-reference root outcomes (share of decisions with a root attempt)**

| Act / kind | decisions | root | a card read a hidden intent | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- |
| 1 elite | 626 | 100.000% | 0.000% | 0.000% | 0.000% |
| 1 boss | 979 | 100.000% | 0.000% | 0.000% | 0.000% |
| 2 elite | 474 | 100.000% | 0.000% | 0.000% | 0.000% |
| 2 boss | 1018 | 99.804% | 0.000% | 0.196% | 0.000% |
| 3 elite | 751 | 100.000% | 0.000% | 0.000% | 0.000% |
| 3 boss | 2650 | 99.585% | 0.377% | 0.038% | 0.000% |
| 4 elite | 614 | 100.000% | 0.000% | 0.000% | 0.000% |
| 4 boss | 954 | 99.686% | 0.000% | 0.105% | 0.210% |
| all | 8066 | 99.802% | 0.124% | 0.050% | 0.025% |

Refusals by class over the 8,066 attempts: a card read a hidden intent 10, a nested card play holds an enemy target 4, and a queued Distilled Chaos play holds a random target 2. Draw knowledge, the pre-rolled value and the unidentifiable executed move classes did not occur on this panel.

## Training-fight root availability

`roots` opened and immediately closed a public search handle with minimal limits at every decision to test whether the binding can construct a public root. It ran zero branches, including at single-legal-action decisions, in a 4-CPU container (it does not measure cost). The tables come from `root_availability.json` (seed 20261117, 1,200-step cap) and regenerate with `standard_results.py --roots`. The v1.2.0 counterpart is `t4-final-roots/root_availability.json`.

These tables include every decision, while the search root tables exclude single-legal-action decisions. A direct comparison of the root columns would mix those denominators.

The corpus is the same train-split recipe definition as v1.2.0 (every level-0 fight state for Ironclad, Silent, Defect and Watcher, with the same eight-of-ten train buckets), re-extracted on v1.2.4. The counts are identical to v1.2.0: 39,078 fights and 1,190,568 decisions overall, and the same per-character fights and decisions. Each entry was replayed once with the play-before-end-turn heuristic. Four entries began terminal (their fight was already over at reset) and contribute zero decisions. "Dome" means the player holds Runic Dome, which hides enemy intents. The table reports decision-weighted shares. Gym's room kind `monster` is normal combat.

| Character | Fights | Decisions | Root | Dome fights | Dome decisions | Root with Dome | Root without Dome | Terminal at reset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defect | 10,322 | 306,418 | 99.23% | 434 (4.20%) | 14,532 (4.74%) | 89.13% | 99.73% | 1 |
| ironclad | 11,724 | 329,311 | 99.53% | 611 (5.21%) | 19,995 (6.07%) | 93.64% | 99.91% | 0 |
| silent | 10,290 | 344,907 | 99.79% | 220 (2.14%) | 8,735 (2.53%) | 93.84% | 99.95% | 3 |
| watcher | 6,742 | 209,932 | 99.43% | 220 (3.26%) | 7,639 (3.64%) | 93.21% | 99.66% | 0 |
| all four | 39,078 | 1,190,568 | 99.51% | 1,485 (3.80%) | 50,901 (4.28%) | 92.32% | 99.83% | 4 |

v1.2.0 had 91.96% overall, 91.08% for Ironclad and 0.00% rooted in Runic Dome fights (every Dome decision was refused as `hidden intents are not modelled`). Gym v1.2.4 now builds a root in 92.32% of Dome decisions. Every boss and elite cell of the all-four table is at or above 99.223%, the event cells are at or above 99.302%, and the lowest cell is 3 monster at 98.510%.

### All four

**All entries**

| Act / room | decisions | root | a hidden pre-rolled value was shown by damage | a card read a hidden intent | draw knowledge exceeds the exact model | a nested card play holds an enemy target | an executed enemy move was not identifiable | a queued Distilled Chaos play holds a random target | generated cards are held unoffered | a queued Mayhem play holds a random target | a queued repeat holds a randomly chosen target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 154178 | 99.875% | 0.083% | 0.006% | 0.004% | 0.003% | 0.000% | 0.029% | 0.000% | 0.000% | 0.000% |
| 1 elite | 98985 | 99.969% | 0.000% | 0.007% | 0.007% | 0.007% | 0.000% | 0.010% | 0.000% | 0.000% | 0.000% |
| 1 boss | 55243 | 99.884% | 0.000% | 0.007% | 0.076% | 0.027% | 0.000% | 0.005% | 0.000% | 0.000% | 0.000% |
| 1 event | 2262 | 99.956% | 0.000% | 0.000% | 0.000% | 0.044% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 193338 | 99.808% | 0.000% | 0.100% | 0.033% | 0.021% | 0.000% | 0.038% | 0.000% | 0.000% | 0.000% |
| 2 elite | 82591 | 99.757% | 0.000% | 0.107% | 0.059% | 0.042% | 0.000% | 0.035% | 0.000% | 0.000% | 0.000% |
| 2 boss | 60730 | 99.574% | 0.000% | 0.089% | 0.285% | 0.026% | 0.000% | 0.025% | 0.000% | 0.000% | 0.002% |
| 2 event | 4453 | 99.955% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.045% | 0.000% | 0.000% |
| 3 monster | 203699 | 98.510% | 0.860% | 0.188% | 0.047% | 0.070% | 0.217% | 0.062% | 0.036% | 0.009% | 0.001% |
| 3 elite | 137026 | 99.488% | 0.000% | 0.253% | 0.169% | 0.045% | 0.000% | 0.040% | 0.003% | 0.002% | 0.000% |
| 3 boss | 112242 | 99.650% | 0.000% | 0.128% | 0.146% | 0.028% | 0.014% | 0.034% | 0.000% | 0.000% | 0.000% |
| 3 event | 7877 | 99.302% | 0.000% | 0.521% | 0.000% | 0.140% | 0.000% | 0.038% | 0.000% | 0.000% | 0.000% |
| 4 elite | 38287 | 99.436% | 0.000% | 0.353% | 0.037% | 0.136% | 0.000% | 0.039% | 0.000% | 0.000% | 0.000% |
| 4 boss | 39657 | 99.223% | 0.000% | 0.151% | 0.494% | 0.101% | 0.000% | 0.030% | 0.000% | 0.000% | 0.000% |
| all | 1190568 | 99.510% | 0.158% | 0.123% | 0.087% | 0.038% | 0.038% | 0.036% | 0.007% | 0.002% | 0.000% |

**Runic Dome holders**

| Act / room | decisions | root | a hidden pre-rolled value was shown by damage | a card read a hidden intent | an executed enemy move was not identifiable | draw knowledge exceeds the exact model | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 1541 | 91.110% | 8.306% | 0.584% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 elite | 888 | 99.212% | 0.000% | 0.788% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 boss | 572 | 99.301% | 0.000% | 0.699% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 7503 | 97.401% | 0.000% | 2.572% | 0.000% | 0.000% | 0.000% | 0.027% |
| 2 elite | 3388 | 97.285% | 0.000% | 2.597% | 0.000% | 0.118% | 0.000% | 0.000% |
| 2 boss | 2163 | 97.503% | 0.000% | 2.497% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 event | 125 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 3 monster | 13068 | 80.188% | 13.407% | 2.923% | 3.382% | 0.015% | 0.046% | 0.038% |
| 3 elite | 8380 | 95.632% | 0.000% | 4.141% | 0.000% | 0.191% | 0.036% | 0.000% |
| 3 boss | 7765 | 97.489% | 0.000% | 1.854% | 0.206% | 0.373% | 0.077% | 0.000% |
| 3 event | 551 | 92.559% | 0.000% | 7.441% | 0.000% | 0.000% | 0.000% | 0.000% |
| 4 elite | 2429 | 94.401% | 0.000% | 5.558% | 0.000% | 0.000% | 0.041% | 0.000% |
| 4 boss | 2528 | 96.321% | 0.000% | 2.373% | 0.000% | 1.305% | 0.000% | 0.000% |
| all | 50901 | 92.320% | 3.693% | 2.876% | 0.900% | 0.165% | 0.031% | 0.014% |

### Ironclad

**All entries**

| Act / room | decisions | root | a hidden pre-rolled value was shown by damage | a card read a hidden intent | an executed enemy move was not identifiable | draw knowledge exceeds the exact model | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 42409 | 99.932% | 0.059% | 0.000% | 0.000% | 0.005% | 0.002% | 0.002% |
| 1 elite | 28474 | 99.993% | 0.000% | 0.000% | 0.000% | 0.000% | 0.004% | 0.004% |
| 1 boss | 15505 | 99.813% | 0.000% | 0.000% | 0.000% | 0.187% | 0.000% | 0.000% |
| 1 event | 738 | 99.864% | 0.000% | 0.000% | 0.000% | 0.000% | 0.136% | 0.000% |
| 2 monster | 52411 | 99.901% | 0.000% | 0.050% | 0.000% | 0.000% | 0.034% | 0.015% |
| 2 elite | 23292 | 99.738% | 0.000% | 0.107% | 0.000% | 0.086% | 0.056% | 0.013% |
| 2 boss | 15583 | 99.647% | 0.000% | 0.160% | 0.000% | 0.116% | 0.058% | 0.019% |
| 2 event | 1301 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 3 monster | 54153 | 98.161% | 1.174% | 0.199% | 0.332% | 0.048% | 0.072% | 0.013% |
| 3 elite | 35519 | 99.631% | 0.000% | 0.242% | 0.000% | 0.076% | 0.037% | 0.014% |
| 3 boss | 32262 | 99.774% | 0.000% | 0.139% | 0.000% | 0.053% | 0.028% | 0.006% |
| 3 event | 2493 | 99.840% | 0.000% | 0.080% | 0.000% | 0.000% | 0.040% | 0.040% |
| 4 elite | 11617 | 99.466% | 0.000% | 0.473% | 0.000% | 0.000% | 0.060% | 0.000% |
| 4 boss | 13554 | 99.587% | 0.000% | 0.258% | 0.000% | 0.066% | 0.081% | 0.007% |
| all | 329311 | 99.529% | 0.201% | 0.124% | 0.055% | 0.045% | 0.037% | 0.010% |

**Runic Dome holders**

| Act / room | decisions | root | a hidden pre-rolled value was shown by damage | a card read a hidden intent | an executed enemy move was not identifiable | draw knowledge exceeds the exact model | a nested card play holds an enemy target | a queued Distilled Chaos play holds a random target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 monster | 324 | 92.284% | 7.716% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 elite | 286 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 1 boss | 97 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 monster | 2618 | 98.969% | 0.000% | 0.993% | 0.000% | 0.000% | 0.000% | 0.038% |
| 2 elite | 1333 | 97.824% | 0.000% | 1.875% | 0.000% | 0.300% | 0.000% | 0.000% |
| 2 boss | 882 | 97.166% | 0.000% | 2.834% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 event | 49 | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 3 monster | 5067 | 81.646% | 12.552% | 2.131% | 3.552% | 0.039% | 0.039% | 0.039% |
| 3 elite | 3465 | 97.460% | 0.000% | 2.482% | 0.000% | 0.000% | 0.058% | 0.000% |
| 3 boss | 3423 | 98.393% | 0.000% | 1.315% | 0.000% | 0.234% | 0.058% | 0.000% |
| 3 event | 190 | 98.947% | 0.000% | 1.053% | 0.000% | 0.000% | 0.000% | 0.000% |
| 4 elite | 1033 | 94.579% | 0.000% | 5.324% | 0.000% | 0.000% | 0.097% | 0.000% |
| 4 boss | 1228 | 97.150% | 0.000% | 2.850% | 0.000% | 0.000% | 0.000% | 0.000% |
| all | 19995 | 93.638% | 3.306% | 2.036% | 0.900% | 0.070% | 0.035% | 0.015% |

Percentages in the detailed tables are rounded to three decimal places. The local `root_availability.json` and `standard_results.py --roots` provide separate cell breakdowns for each of the four characters.

### Change from v1.2.0

Refusals fell from 95,754 of 1,190,568 decisions (8.04%) to 5,830 (0.490%). By class:

| Class | v1.2.0 decisions | v1.2.4 decisions |
| --- | ---: | ---: |
| hidden intents are not modelled | 50,901 | not present |
| draw knowledge exceeds the exact model | 43,889 | 1,041 |
| a hidden pre-rolled value was shown by damage | not present | 1,880 |
| a card read a hidden intent | not present | 1,464 |
| an executed enemy move was not identifiable | not present | 458 |
| a nested card play holds an enemy target | 442 | 458 |
| a queued Distilled Chaos play holds a random target | 417 | 424 |
| generated cards are held unoffered | 80 | 80 |
| a queued Mayhem play holds a random target | 22 | 22 |
| a queued repeat holds a randomly chosen target | 3 | 3 |

The blanket Runic Dome class is gone. Three narrower classes replace it, and they occur only in Dome fights (3,802 decisions, 0.319% of all decisions; 7.47% of Dome decisions). Draw knowledge shrank by 97.6%, from 3.686% to 0.087% of decisions, and 84 of its 1,041 decisions are in Dome fights. The long-tail classes outside Dome fights (nested card play, queued Distilled Chaos, queued Mayhem, queued repeat, generated cards unoffered) are unchanged to the decision. The 16 extra nested-card and 7 extra Distilled Chaos decisions are Dome-fight decisions (the same classes appear in the Dome-holder tables below). v1.2.0 refused those decisions under the Dome class. The remaining refusals, about 0.5% of decisions, are tracked on the Gym side as issue `spire-e58`, which covers the refusals that are there by design. At this coverage no Learn follow-up for the teacher is planned. Including the 107 decisions in the older classes, 7.68% of Dome decisions are refused, which complements the 92.32% rooted.

## Teacher budgets

`current` is the cost unit. On `--train-sample 200` (seed 2026112501, native playouts, compact layout, one worker, `--parallel 12`) it used 1,326.5 CPU s over 4,698 searched decisions: **0.2815 CPU s per searched decision** (174 of 200 wins, 36 branch refusals, 476 single-legal-action decisions).

The tree teachers (`tree-1x`, `tree-3x` and `tree-10x` in `configs/neural/search-teacher-gym-1.2.4/teachers.json`) use per-action budgets, `simulations_per_action` k = 22, 66 and 220, with `min_root_visits` 8, `ucb_c` 1.0, `max_depth` 4, 8 worlds, widening (2.0, 0.5), 80 playout steps, compact layout, one worker and native playouts. The config file's SHA-256 is `dc5d969c9b5e85dd2232ec5f2ed48bfe9de78a9dcba8e94491d08d2da34a5925`. Each budget was verified on `--train-sample 100` (seed 2026112502) against a target multiple of the cost unit, with a band of plus or minus 15%:

| Teacher | k | Target | CPU s per searched decision | Measured ratio | Wins of 100 | Branch refusals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tree-1x` | 22 | 1 | 0.2844 | 1.01 | 84 | 0 |
| `tree-3x` | 66 | 3 | 0.9362 | 3.33 | 87 | 0 |
| `tree-10x` | 220 | 10 | 3.1352 | 11.14 | 90 | 0 |

The ratios divide each verification's CPU per searched decision by 0.2815. The 3x and 10x results sit near the top of the band (3.45 and 11.5). The verification wins are on a different 100-fight sample from the cost-unit run, so they do not compare with the 174 of 200. Each verification also had one escape and no censored case. The cost-unit run, the verifications and the search references all used the same 12-CPU container allocation.

Budgets scale with the legal-action count because cost does. `current` spends 32 playouts per legal action. A fixed tree budget does not. An early `tree-3x` probe with a fixed per-decision budget (`probe-3x`, run before the per-action budgets existed; the container was killed at exit status 137 after 24 fights, so it is partial, and its config is not in the artifacts) cost 14.7 times `current` at 2 to 4 legal actions (1.838 against 0.125 CPU s per searched decision) and 3.2 times at 17 or more (3.617 against 1.123). Between those bins, at 5 to 16 legal actions, it cost 7.4 times (2.510 against 0.341). Binned the same way against the cost-unit run, the verified `tree-3x` with per-action budgets costs 2.4, 3.4 and 3.1 times `current` in the three bins:

| Legal actions | `current` CPU s | `tree-3x` CPU s (decisions) | Ratio |
| --- | ---: | ---: | ---: |
| 2 to 4 | 0.125 | 0.297 (800) | 2.4 |
| 5 to 16 | 0.341 | 1.164 (1,756) | 3.4 |
| 17 or more | 1.123 | 3.466 (44) | 3.1 |

These ratios bracket the target of 3, while the fixed-budget probe's 14.7, 7.4 and 3.2 do not. The 17-or-more bins are small: 44 decisions here and 17 in the probe. The bin ratios are one-off aggregations of the `decisions.jsonl` files over different fight samples, and the probe ran on only 24 fights, so they are indicative. Per-action budgets mirror `current`'s 32 playouts per action.

Native playouts cost less than v1.2.0's Python playouts. On the same cost-unit sample size and seed (`--train-sample 200`, seed 2026112501, 12 CPUs), v1.2.0 `current` used 1.4773 CPU s per searched decision (`s1t7-calibration/cost-current`), against 0.2815 here, about 5.2 times less. This is not a single-variable comparison: the rules epoch differs as well. The v1.2.0 section "Search reference cost" in `docs/neural-standard-baseline-gym-1.2.0.md` reports 0.821 CPU s per searched decision for compact, one-worker search on the v1.2.0 selection panel under a four-CPU allocation. The v1.2.4 selection panel costs 0.630 with its longest fight, 1.3 times less, and 0.312 without it, 2.6 times less. One fight moves the panel ratio by a factor of two, so the panel figures are not a stable measure. The sample-matched 5.2 is the better estimate of the playout change. Neither figure is a Gym-release effect on its own.

The long tail matters for the tree teachers. A `tree-10x` search of that one selection fight should take about ten times its 2,326 CPU s, roughly six hours on one core. That is an estimate, not a measurement.

## Baselines

Two replicates of the Standard foundation baseline trained fresh on Gym v1.2.4. Each was scored on the selection panel above at five checkpoints (256 cases, deterministic argmax evaluation, 1,024 actions and 120 s per case, plus a 500-action cap on any one public turn). `baseline_entry.py --reevaluate PRIOR --panel boss` then scored the same five checkpoints of each replicate on the boss panel. The runs are under `artifacts/development/neural-combat/runs/gym-1.2.4/` (`standard-baseline`, `standard-baseline-r2`, `standard-baseline-boss` and `standard-baseline-r2-boss`).

Some tables come from `scripts/neural/standard_results.py`, run unchanged with `--run1`, `--run2`, `--panel`, `--search`, `--run-states run-states-gym-1.2.4.json` and `--recipe` set to the v1.2.4 recipe. These are the learning curves with escapes and censored cases, the censor reasons, the wins by level per checkpoint, the fresh-to-final tests, the final-to-final replicate test and the act, room-kind and level tables, for both panels. The script needs a `standard_panel.jsonl` that no v1.2.4 entry writes. A one-off step wrote it from the panel files the baseline runs saved (`development-panel.json` and `boss-panel.json`, whose manifest SHA-256s match the search references) and the search references' `fights.jsonl`. It also wrote a stub that carries the panel manifest SHA-256 for `--search`. Other tables come from one-off aggregation of the same summaries, `progress.jsonl` and the container manifests: deaths and HP retention, wins by room kind per checkpoint, the 25%-to-final and 75%-to-final tests, where wins went, replicate agreement at other checkpoints, escape and censor case sets, the tag table and the wall-time table. The per-eighth training diagnostics come from a one-off aggregation of the segment event streams, described below. None of these one-off aggregations is a repository script. Only aggregates are published.

| | r1 | r2 |
| --- | --- | --- |
| Plan | `configs/neural/baseline-standard-1.2.4/plan.json` | `configs/neural/baseline-standard-1.2.4-r2/plan.json` |
| Plan SHA-256 | `dac35c28ed5523ba57d154e9c1317c95e9db22331d0b3ac2da6600d8aac2f219` | `f3ed156da5d50c9286bbd57679f98474d161a8e498c27449b583594590768773` |
| Seeds (policy / optimizer / curriculum / noncombat) | 1103 / 2207 / 3301 / 4409 | 5101 / 6203 / 7307 / 8419 |
| Gym revision | `673eb8f220023b0c95401bbf96bec814a359457b` | same |
| Learner revision, training and both evaluations | `b4f8a36`, clean | `b4f8a36`, clean |
| Turn cap in training | 500 actions | 500 actions |
| Learned decisions | 1,622,016 (396 segments) | 1,622,016 (396 segments) |
| Stop reason, result | `target_reached`, `complete` | `target_reached`, `complete` |
| Learner changed | false | false |

The seeds are those of the v1.2.0 and v1.1.1 baselines. Apart from IDs and paths, the v1.2.4 plans and templates differ from the v1.2.0 ones only in the Gym revision, the recipe, the raw-mode mapping suffix (`+gym-1.2.4`) and the template's turn cap. The model, PPO settings, curriculum schedule and checkpoint schedule are unchanged. The recipe schedule is levels {0} from shard 0, {0, 1} from shard 99, {0, 1, 2} from shard 198 and {0, 1, 2, 4} from shard 297. The training streams confirm it: in both runs levels 1, 2 and 4 first appear among settled training fights at shards 99, 198 and 297, and no shard holds a level outside its schedule. Every segment learned exactly 4,096 decisions, so neither run overshot the target. The intermediate checkpoints are the weights after segments 99, 198 and 297 (405,504, 811,008 and 1,216,512 decisions), taken before levels 1, 2 and 4 join.

All four runs exited 0 with `exit_status` 0, `failure` null, `timed_out` false, an empty `stderr.txt` and no `manifest-post.json`. Each ran in a container with 4 CPUs, 16 GB and one GPU, with the learner tree at `b4f8a36e42d49ebb593ccc5de07121481fba244d` and no dirty paths before and after (`learner_changed` false). Both boss re-evaluations record the SHA-256 of the prior run's `summary.json` they read, and both match the training runs' summaries. r1's first attempt (`standard-baseline-interrupted-host-restart-20261126`) stopped when the host restarted late on 2026-11-25, after 309 of its 396 segments. `baseline_entry.py` cannot resume a run, so r1 retrained from scratch with the same plan and seeds. That attempt is not used. Its untrained checkpoint is byte-identical to the retrain's, but the trained weights differ from the first segment on, so the retrain is not a bitwise replay. Learn does not claim bit-for-bit reproducible training.

An outcome is a win, a death, an escape or a censored case (the case hit a cap; `incomplete` in the summaries). Win counts are out of 256, and escapes and censored cases count as not won. Paired tests exclude escaped and censored cases. Faults are errors, not outcomes, and there were none. All 792 training segment summaries have status `complete`, a null fault and a fault count of 0. Every evaluation of the four runs reports no failures and zero operational failures.

### Learning curves

Wins out of 256 on the selection panel (escaped / censored in the second column of each replicate):

| Checkpoint | Decisions | Levels trained on | r1 wins | r1 esc / cens | r2 wins | r2 esc / cens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fresh | 0 | none | 19 | 0 / 0 | 141 | 0 / 6 |
| 25% | 405,504 | {0} | 178 | 1 / 3 | 147 | 0 / 4 |
| 50% | 811,008 | {0, 1} | 179 | 1 / 0 | 172 | 1 / 5 |
| 75% | 1,216,512 | {0, 1, 2} | 176 | 1 / 3 | 186 | 1 / 3 |
| Final | 1,622,016 | {0, 1, 2, 4} | 179 | 1 / 4 | 166 | 1 / 1 |

Search wins 233 of 256 on the same panel. Deaths are the remainder (r1 final 72, r2 final 88). Mean HP retention over completed cases at fresh and final is 0.04 and 0.39 for r1 and 0.31 and 0.35 for r2.

r1 rises from 19 to 178 by 25% and then stays between 176 and 179. r2 starts high (141 untrained), barely moves by 25% (147), peaks at 75% (186) and drops to 166 at the final.

Wins by fuzz level (L0 / L1 / L2 / L4, 64 cases each; search 58 / 60 / 58 / 57):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 7 / 5 / 2 / 5 | 35 / 34 / 37 / 35 |
| 25% | 45 / 46 / 47 / 40 | 39 / 44 / 35 / 29 |
| 50% | 44 / 48 / 45 / 42 | 42 / 47 / 43 / 40 |
| 75% | 44 / 45 / 48 / 39 | 46 / 49 / 45 / 46 |
| Final | 45 / 47 / 47 / 40 | 45 / 39 / 42 / 40 |

Wins by room kind (monster / elite / boss, out of 140 / 76 / 40; search 139 / 67 / 27):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 18 / 1 / 0 | 110 / 26 / 5 |
| 25% | 125 / 46 / 7 | 106 / 31 / 10 |
| 50% | 123 / 41 / 15 | 117 / 44 / 11 |
| 75% | 123 / 43 / 10 | 126 / 47 / 13 |
| Final | 129 / 38 / 12 | 119 / 34 / 13 |

Fresh against final, exact two-sided McNemar tests on the paired win/not-win outcomes (censored and escaped cases excluded in either checkpoint):

| Comparison | Paired | Both win | Only fresh wins | Only final wins | Neither | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 fresh vs final | 251 | 19 | 0 | 160 | 72 | 1.4e-48 |
| r2 fresh vs final | 249 | 123 | 17 | 42 | 67 | 0.0015 |

These match the `paired_fresh_vs_final` blocks of the two run summaries. Training helped both replicates against the untrained checkpoint. r2's untrained policy already wins 141 cases, so its gain is small next to r1's. The same r2 seeds also gave a strong untrained policy on v1.2.0 (146 wins), but that series is separate.

### Replicate agreement

The two finals agree on 205 of 251 paired cases (149 wins by both, 56 losses by both). They differ on 46: 30 won only by r1 and 16 won only by r2 (exact p = 0.054). Their win counts (179 and 166) differ by 13, and the test does not reach 0.05. The replicates differ most at 25%, where r1 learned faster (44 cases won only by r1 against 14 only by r2, p = 1.0e-4). At 50% and 75% they do not differ (29 against 25, p = 0.68; 14 against 23, p = 0.19).

Between any two trained checkpoints of r1, 30 to 58 paired cases change outcome even where the totals barely move (25% against 50%: 26 cases won only at each). A checkpoint-to-checkpoint difference of a few wins is within this turnover.

One selection case escapes: an act 1 elite at level 2. Both replicates escape it at every trained checkpoint except r2 at 25%. Search did not escape it. Every case that r1 had censored at some checkpoint (five distinct cases) was also censored at some checkpoint of r2, which has seven.

### Final wins by stratum and level

Final-checkpoint wins in each cell beside search and the fresh checkpoints. "L0 cases" and "his L0" count Baalorlord's recorded outcomes of the same states at level 0 only. Cells hold 2 to 14 cases per level, so single-cell differences are noise.

**By act and room kind**

| Act / kind | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 monster | 56 | 55 | 54 | 56 | 17 | 52 | 14 | 14 |
| 1 elite | 28 | 17 | 14 | 27 | 1 | 10 | 7 | 7 |
| 1 boss | 8 | 1 | 3 | 6 | 0 | 2 | 2 | 1 |
| 2 monster | 44 | 39 | 36 | 44 | 0 | 29 | 11 | 11 |
| 2 elite | 20 | 8 | 9 | 17 | 0 | 9 | 5 | 5 |
| 2 boss | 8 | 3 | 2 | 4 | 0 | 1 | 2 | 2 |
| 3 monster | 40 | 35 | 29 | 39 | 1 | 29 | 10 | 10 |
| 3 elite | 20 | 10 | 11 | 16 | 0 | 7 | 5 | 5 |
| 3 boss | 16 | 6 | 7 | 13 | 0 | 2 | 4 | 4 |
| 4 elite | 8 | 3 | 0 | 7 | 0 | 0 | 2 | 2 |
| 4 boss | 8 | 2 | 1 | 4 | 0 | 0 | 2 | 2 |

**By fuzz level**

| Level | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 64 | 45 | 45 | 58 | 7 | 35 | 64 | 63 |
| L1 | 64 | 47 | 39 | 60 | 5 | 34 | 0 | 0 |
| L2 | 64 | 47 | 42 | 58 | 2 | 37 | 0 | 0 |
| L4 | 64 | 40 | 40 | 57 | 5 | 35 | 0 | 0 |

**By act**

| Act | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| act 1 | 92 | 73 | 71 | 89 | 18 | 64 | 23 | 22 |
| act 2 | 72 | 50 | 47 | 65 | 0 | 39 | 18 | 18 |
| act 3 | 76 | 51 | 47 | 68 | 1 | 38 | 19 | 19 |
| act 4 | 16 | 5 | 1 | 11 | 0 | 0 | 4 | 4 |

Both finals trail search in every act. At level 0 they each win 45 of 64, against search's 58 and Baalorlord's recorded 63. By room kind they win 129 and 119 of 140 monster cases against search's 139, 38 and 34 of 76 elite cases against 67, and 12 and 13 of 40 boss cases against 27. Monster fights are close to search. Elite and boss fights are not. Act 4 is the weakest act (5 and 1 of 16 against search's 11), and r2 wins none of the 8 act 4 elite cases (search 7).

### Escapes, censoring and panel tags

Escapes are one per trained checkpoint, except none at r2's 25%, and none at either untrained checkpoint. Censored cases by reason (a case is censored when it reaches the 1,024-action cap per fight, `environment_action_budget`, or the 500-action cap per public turn, `turn_action_budget`):

| Checkpoint | r1 turn cap | r1 fight cap | r2 turn cap | r2 fight cap |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 0 | 0 | 6 | 0 |
| 25% | 3 | 0 | 4 | 0 |
| 50% | 0 | 0 | 5 | 0 |
| 75% | 3 | 0 | 3 | 0 |
| Final | 4 | 0 | 1 | 0 |

No evaluation case reached the 1,024-action fight cap or ended with any other censor reason. Every censored case was a turn-cap case, and censoring is at most 6 of 256 (2.3%) at any checkpoint. The seven distinct censored cases are two act 1 elites, two act 3 bosses, an act 2 monster, an act 2 elite and an act 4 boss.

Panel tag counts (the selection panel carries `potions_assumed_empty` on 42 cases, `burning_kind_sampled` on 10, `misc_defaulted` on 10 and `relic_counter_defaulted` on 113) and wins by tag at 25%, at 75% and at the final checkpoint:

| Tag | Cases | r1 25% | r1 75% | r1 final | r2 25% | r2 75% | r2 final | search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `potions_assumed_empty` | 42 | 28 | 28 | 29 | 26 | 31 | 25 | 38 |
| `burning_kind_sampled` | 10 | 6 | 6 | 3 | 4 | 5 | 5 | 10 |
| `misc_defaulted` | 10 | 5 | 4 | 6 | 2 | 8 | 5 | 8 |
| `relic_counter_defaulted` | 113 | 75 | 77 | 76 | 57 | 75 | 63 | 99 |
| All cases | 256 | 178 | 176 | 179 | 147 | 186 | 166 | 233 |

Tagged cases overlap, so rows are not additive. The counts match the tag strata of the run summaries. Tag cells hold 10 to 113 cases and are not tested.

### Wall time and GPU waits

Times are from `progress.jsonl` and the run manifests. A wait is the gate that holds a stage until the GPU has 13,958,643,712 free bytes.

| Run | Process wall | Of which GPU waits | Training elapsed (excluding waits) | Waits |
| --- | ---: | ---: | ---: | --- |
| r1 training and selection evaluations | 9,607 s | 840 s | 8,727 s (7,887 s) | 2 in training (450 and 390 s) |
| r2 training and selection evaluations | 9,962 s | 1,080 s | 8,931 s (7,851 s) | 3 in training (360, 390 and 330 s) |
| r1 boss re-evaluation | 1,012 s | 0 s | not applicable | none |
| r2 boss re-evaluation | 1,535 s | 330 s | not applicable | 1 of 330 s before the fresh evaluation |

r1's five selection evaluations took 877 s of progress time and r2's took 1,029 s, with no waits. The boss re-evaluations took 1,011 s (r1) and 1,534 s including the 330 s wait (r2). The runs used one worker and 16 environment rows.

### Turn-action cap

Both replicates trained and were evaluated with the 500-action cap per public turn from the start. Both plans set `max_turn_actions` to 500, and so do both templates' budgets. In training a capped fight is a time-limit truncation, not a loss. In panels it is a censored case. A simulator fault stays an error.

- Training hit the cap **0 times in either replicate**. Each `progress.jsonl` has 396 `segment_complete` records, and every one carries `turn_action_budget_hits: 0`. No segment summary's collection block carries the field, and the runner writes it there only when it is non-zero.
- Evaluation censors are in the table above and, for the boss panel, below. They are all turn-cap cases. The untrained r2 policy accounts for 6 on each panel. Trained checkpoints have 0 to 5 on the selection panel and 0 to 7 on the boss panel, so the cap changes at most 2.7% of cases at any checkpoint.

### Boss panel

The boss re-evaluations used the same checkpoints, deterministic argmax, 1,024 actions per fight and the 500-action turn cap. The per-case time limit was the plan's 19,800 s overall deadline, as `--panel boss` sets it. The panel has 96 elite and 160 boss cases, and search won 176 of 256 on it (78 deaths, 2 escapes). The tables come from the same script and one-off aggregation as above.

Wins out of 256 on the boss panel (escaped / censored in the second column of each replicate):

| Checkpoint | Levels trained on | r1 wins | r1 esc / cens | r2 wins | r2 esc / cens |
| --- | --- | ---: | ---: | ---: | ---: |
| Fresh | none | 4 | 0 / 0 | 62 | 0 / 6 |
| 25% | {0} | 82 | 2 / 3 | 68 | 2 / 5 |
| 50% | {0, 1} | 88 | 2 / 0 | 85 | 2 / 6 |
| 75% | {0, 1, 2} | 107 | 1 / 7 | 108 | 2 / 6 |
| Final | {0, 1, 2, 4} | 102 | 2 / 4 | 89 | 2 / 0 |

Deaths are the remainder (r1 final 148, r2 final 165). Mean HP retention over completed cases at fresh and final is 0.00 and 0.18 for r1 and 0.09 and 0.16 for r2. Both replicates climb to 75%, where they are nearly equal (107 and 108), and both end lower: r1 by 5 and r2 by 19.

Wins by fuzz level (L0 / L1 / L2 / L4, 64 cases each; search 45 / 49 / 43 / 39):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 1 / 3 / 0 / 0 | 20 / 15 / 12 / 15 |
| 25% | 21 / 25 / 21 / 15 | 19 / 16 / 14 / 19 |
| 50% | 23 / 24 / 21 / 20 | 20 / 24 / 23 / 18 |
| 75% | 27 / 29 / 28 / 23 | 31 / 27 / 33 / 17 |
| Final | 30 / 28 / 28 / 16 | 23 / 28 / 23 / 15 |

Wins by room kind (elite / boss, out of 96 / 160; search 84 / 92):

| Checkpoint | r1 | r2 |
| --- | --- | --- |
| Fresh | 2 / 2 | 35 / 27 |
| 25% | 56 / 26 | 35 / 33 |
| 50% | 51 / 37 | 50 / 35 |
| 75% | 60 / 47 | 62 / 46 |
| Final | 59 / 43 | 46 / 43 |

| Comparison | Paired | Both win | Only fresh wins | Only final wins | Neither | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 fresh vs final | 250 | 4 | 0 | 98 | 148 | 6.3e-30 |
| r2 fresh vs final | 248 | 41 | 21 | 47 | 139 | 0.0022 |

These match the `paired_fresh_vs_final` blocks of the boss re-evaluation summaries. The two finals agree on 194 of 250 paired cases (67 wins by both, 127 losses by both) and differ on 56: 35 won only by r1 and 21 only by r2 (exact p = 0.081). At 75% they differ on 49 (24 against 25, p = 1.0).

Censored cases on the boss panel, all at the turn cap (none at the fight cap or for any other reason):

| Checkpoint | r1 turn cap | r2 turn cap |
| --- | ---: | ---: |
| Fresh | 0 | 6 |
| 25% | 3 | 5 |
| 50% | 0 | 6 |
| 75% | 7 | 6 |
| Final | 4 | 0 |

Every censored boss-panel case is a boss fight (eight distinct for r1 and nine for r2, seven shared). The escapes are the two cases search also escaped, both act 1 elites (levels 2 and 4). Each replicate escapes both at every trained checkpoint, except r1 at 75%, where it escapes one.

Panel tags (the boss panel carries `potions_assumed_empty` on 57 cases, `burning_kind_sampled` on 8, `misc_defaulted` on 3 and `relic_counter_defaulted` on 155):

| Tag | Cases | r1 25% | r1 75% | r1 final | r2 25% | r2 75% | r2 final | search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `potions_assumed_empty` | 57 | 18 | 23 | 21 | 18 | 21 | 15 | 38 |
| `burning_kind_sampled` | 8 | 6 | 5 | 4 | 4 | 6 | 6 | 6 |
| `misc_defaulted` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `relic_counter_defaulted` | 155 | 47 | 62 | 61 | 38 | 61 | 50 | 107 |
| All cases | 256 | 82 | 107 | 102 | 68 | 108 | 89 | 176 |

**By act and room kind**

| Act / kind | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 elite | 36 | 23 | 17 | 32 | 2 | 12 | 9 | 9 |
| 1 boss | 40 | 8 | 12 | 21 | 1 | 7 | 10 | 9 |
| 2 elite | 24 | 13 | 11 | 21 | 0 | 11 | 6 | 6 |
| 2 boss | 32 | 12 | 10 | 19 | 0 | 4 | 8 | 8 |
| 3 elite | 24 | 18 | 16 | 22 | 0 | 10 | 6 | 6 |
| 3 boss | 60 | 20 | 19 | 41 | 1 | 14 | 15 | 15 |
| 4 elite | 12 | 5 | 2 | 9 | 0 | 2 | 3 | 3 |
| 4 boss | 28 | 3 | 2 | 11 | 0 | 2 | 7 | 7 |

**By fuzz level**

| Level | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 64 | 30 | 23 | 45 | 1 | 20 | 64 | 63 |
| L1 | 64 | 28 | 28 | 49 | 3 | 15 | 0 | 0 |
| L2 | 64 | 28 | 23 | 43 | 0 | 12 | 0 | 0 |
| L4 | 64 | 16 | 15 | 39 | 0 | 15 | 0 | 0 |

**By act**

| Act | cases | r1 final | r2 final | search | r1 fresh | r2 fresh | L0 cases | his L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| act 1 | 76 | 31 | 29 | 53 | 3 | 19 | 19 | 18 |
| act 2 | 56 | 25 | 21 | 40 | 0 | 15 | 14 | 14 |
| act 3 | 84 | 38 | 35 | 63 | 1 | 24 | 21 | 21 |
| act 4 | 40 | 8 | 4 | 20 | 0 | 4 | 10 | 10 |

On the boss panel the finals win 102 and 89 against search's 176. They win 59 and 46 of 96 elite cases against search's 84, and 43 each of 160 boss cases against 92. At level 0 they win 30 and 23 of 64, against search's 45 and Baalorlord's recorded 63. The gap to search is widest at level 4 (16 and 15 against 39) and in act 4 bosses (3 and 2 of 28 against 11). Act 3 bosses, the largest cell, are won 20 and 19 times of 60 against search's 41.

### Change after the 75% checkpoint

On v1.2.0 both replicates peaked at 25% and declined (r1 190 to 166, r2 183 to 168). That pattern does not recur here. Neither replicate peaks at 25%:

- r1 on the selection panel is flat after 25% (178, 179, 176, 179).
- r2 on the selection panel peaks at 75% (186) and drops to 166.
- Both boss-panel curves peak at 75% (107 and 108) and drop at the final (102 and 89).

Every 75%-to-final drop falls in the final quarter, when level 4 joins the training mixture (r1 on the selection panel has none). The tests below are post hoc, not pre-registered, and are not corrected for multiplicity. The final checkpoint remains the pre-registered comparison point; checkpoint selection is a later decision.

| Panel | Comparison | Paired | Both win | Only earlier wins | Only final wins | Neither | Exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selection | r1 25% vs final | 250 | 153 | 23 | 26 | 48 | 0.78 |
| Selection | r1 75% vs final | 250 | 153 | 21 | 26 | 50 | 0.56 |
| Selection | r2 25% vs final | 251 | 130 | 17 | 36 | 68 | 0.013 |
| Selection | r2 75% vs final | 251 | 154 | 31 | 12 | 54 | 0.0054 |
| Boss | r1 25% vs final | 249 | 68 | 14 | 34 | 133 | 0.0055 |
| Boss | r1 75% vs final | 247 | 82 | 25 | 19 | 121 | 0.45 |
| Boss | r2 25% vs final | 249 | 44 | 24 | 44 | 137 | 0.021 |
| Boss | r2 75% vs final | 248 | 69 | 39 | 18 | 122 | 0.0075 |

Two of the four 75%-to-final drops are clear (r2 on both panels). r1's boss-panel drop of 5 wins is not distinguishable from checkpoint turnover (25 against 19). Against 25%, the finals are better on the boss panel in both replicates and on the selection panel for r2, and equal for r1 on the selection panel.

Where the wins went (75% to final):

- **r2, selection.** It lost 32 wins (31 became deaths and 1 a turn-cap censor) and gained 12, all from deaths. The losses were 18 elites, 11 monsters and 3 bosses. The gains were 5 elites, 4 monsters and 3 bosses. Elite wins fall from 47 to 34. By level the net change is −1 / −10 / −3 / −6 (L0 / L1 / L2 / L4), so the largest loss is at level 1, not level 4.
- **r2, boss.** It lost 39 wins, all to deaths, and gained 20 (18 from deaths and 2 from censored cases). The losses were 20 elites and 19 bosses. The gains were 4 elites and 16 bosses. Elite wins fall from 62 to 46 and boss wins from 46 to 43. By level the net change is −8 / +1 / −10 / −2, largest at levels 0 and 2.
- **r1, boss.** It lost 25 wins, all to deaths, and gained 20 (19 from deaths and 1 from a censored case). By kind the net change is −1 elite and −4 boss. By level it is +3 / −1 / 0 / −7, so this small drop sits at level 4.
- **r1, selection.** It lost 23 wins (21 to deaths, 2 to censors) and gained 26. Elite wins fall from 43 to 38 while monster wins rise from 123 to 129. The net change by level is +1 / +2 / −1 / +1.

Elite wins fall in the final quarter in three of the four curves: 43 to 38 (r1, selection), 47 to 34 (r2, selection) and 62 to 46 (r2, boss). On the fourth they hold (60 to 59, r1, boss). On v1.2.0 most of the net loss was also in elite fights.

Training-stream diagnostics per eighth of training. Each eighth is 49 or 50 consecutive segments of 4,096 decisions. Level 1 joins at the start of eighth 3, level 2 at eighth 5 and level 4 at eighth 7, so eighths 7 and 8 are the final quarter. These come from the `optimizer_update` and `combat_settlement` records of the training segments. The win rate is wins over all settled fights (wins, deaths and escapes) under the training policy's seeded sampling. Mean HP is the HP left after cleanup on a won fight. Entropy and value loss are means over the optimizer updates (392 to 400 per eighth).

| Eighth | r1 win rate | r1 entropy | r1 value loss | r1 HP on wins | r2 win rate | r2 entropy | r2 value loss | r2 HP on wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.650 | 0.946 | 0.036 | 46.9 | 0.644 | 0.936 | 0.039 | 45.4 |
| 2 | 0.687 | 0.726 | 0.042 | 47.7 | 0.634 | 0.862 | 0.049 | 46.6 |
| 3 | 0.689 | 0.672 | 0.047 | 47.9 | 0.652 | 0.811 | 0.046 | 45.7 |
| 4 | 0.703 | 0.647 | 0.053 | 47.0 | 0.690 | 0.676 | 0.053 | 46.8 |
| 5 | 0.702 | 0.584 | 0.047 | 46.5 | 0.650 | 0.659 | 0.052 | 46.6 |
| 6 | 0.698 | 0.591 | 0.051 | 46.1 | 0.666 | 0.572 | 0.051 | 46.6 |
| 7 | 0.684 | 0.540 | 0.049 | 46.5 | 0.671 | 0.561 | 0.051 | 46.7 |
| 8 | 0.665 | 0.587 | 0.050 | 46.1 | 0.677 | 0.509 | 0.053 | 46.0 |

Each eighth holds 6,982 to 8,049 settled fights. The training win rate by fuzz level links each settled fight to its training case through the segment's seed allocations and the shard's annotations. Every settled fight in both runs linked.

| Eighth | r1 L0 / L1 / L2 / L4 | r2 L0 / L1 / L2 / L4 |
| ---: | --- | --- |
| 1 | 0.650 / - / - / - | 0.644 / - / - / - |
| 2 | 0.687 / - / - / - | 0.634 / - / - / - |
| 3 | 0.690 / 0.688 / - / - | 0.655 / 0.649 / - / - |
| 4 | 0.704 / 0.702 / - / - | 0.689 / 0.692 / - / - |
| 5 | 0.725 / 0.689 / 0.690 / - | 0.667 / 0.637 / 0.645 / - |
| 6 | 0.693 / 0.702 / 0.699 / - | 0.673 / 0.672 / 0.653 / - |
| 7 | 0.721 / 0.682 / 0.684 / 0.648 | 0.703 / 0.670 / 0.686 / 0.625 |
| 8 | 0.681 / 0.677 / 0.658 / 0.646 | 0.711 / 0.681 / 0.663 / 0.652 |

From eighth 1 to eighth 8, entropy falls by 38% in r1 (0.946 to 0.587, with its lowest value, 0.540, in eighth 7) and by 46% in r2 (0.936 to 0.509). Value loss rises by 36 to 39% in both (r1 0.036 to 0.050, r2 0.039 to 0.053). Mean HP on wins stays between 45.4 and 47.9. In the final quarter r1's training win rate falls (0.698 in eighth 6 to 0.665 in eighth 8) while its selection wins hold. r2's rises slightly (0.666 to 0.677) while its selection wins fall by 20. Level-4 training fights are won least often (0.625 to 0.652). Training and evaluation differ: training samples actions while evaluation takes the argmax, and the training mixture changes at each join, so the training win rate is not an evaluation proxy.

What the data support, weaken or leave untested:

- **Final-quarter curriculum shift (level 4 joins). Supported in timing, not in the level breakdown.** Every drop falls in the final quarter, and the two clear drops (r2, p = 0.0054 and 0.0075) both start after the 75% checkpoint, which precedes level 4. In training, level-4 fights have the lowest win rate. But r1's selection curve does not drop at all. r2's losses do not concentrate at level 4: its largest net losses are at level 1 on the selection panel and at levels 0 and 2 on the boss panel. Only r1's small, non-significant boss-panel drop sits at level 4. A shift in the training mixture is consistent with the timing. The level breakdown does not single out level 4.
- **PPO drift while entropy falls and the critic worsens. Weakened as a sufficient explanation.** On v1.2.0 this was untested and consistent with the data. Here r1 shows the same trends as r2 (entropy down 38%, value loss up 39%), yet its selection wins are flat from 25% to final. These trends alone do not produce a decline. They could still contribute in r2. That is untested.
- **Classic overfitting to the training fights. Not excluded.** r2's training win rate rises slightly in the final quarter while its evaluation wins fall, which is the pattern overfitting would produce. The rise is about one point, the training rate uses sampled actions and the mixture changed, so these data cannot test it. r1's training win rate falls while its evaluation holds, so the two do not track each other in either direction.
- **Turn-cap stalling. Not supported.** Neither replicate hit the turn cap in training. In evaluation, r2's final checkpoint has fewer turn-cap censors than its 75% checkpoint (1 against 3 on the selection panel and 0 against 6 on the boss panel). The lost wins became deaths (31 of 32 and 39 of 39 for r2).
- **A favourable 75% checkpoint. Possible, untested.** r2's 75% count (186) is the highest of any selection checkpoint of either replicate. Its 50% and final counts (172 and 166) are both below r1's plateau of 176 to 179, so 186 stands out in both replicates, and the finals of the two replicates are not distinguishable (p = 0.054). Checkpoint turnover of 30 to 68 cases per checkpoint pair across the four curves also limits how much one checkpoint can show. Two replicates cannot separate a real final-quarter regression from a high draw at 75%.

Neither the curriculum nor the drift explanation is tested by these runs. A test would need more replicates or a control run without the level-4 join. These analyses do not change the reported results: the finals (179 and 166 on the selection panel, 102 and 89 on the boss panel) are the comparison points.

### Series separation

The v1.2.0 and v1.2.4 series are separate and are not paired. The Standard rules differ (`spire-gym-native-standard-rules-v7` against `-v9`), and the panels were rebuilt on a new state table, so the cases differ. The search references also differ (221 and 233 of 256, with Python playouts against native playouts). Differences between the series cannot be read as a Gym-release effect. For orientation only, the v1.2.0 finals were 166 (r1) and 168 (r2) of 256 on their own selection panel, against 179 and 166 here. The untrained checkpoints won 34 and 146 there, against 19 and 141 here. v1.2.0 r1 also trained without the turn cap, which both v1.2.4 replicates had from the start.
