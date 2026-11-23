# Standard panels and search reference on Gym v1.2.0

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

`t4-final-search` ran one fight per case with 32 samples per action, 80 playout steps, a 1,200-step fight cap, seed 20261117, dense branch observations and a four-CPU allocation. The reference uses heuristic playouts. Search won **221 of 256** (34 deaths and one escape). On the original level-0 entries it won **58 of 64** versus **63 of 64** recorded wins for Baalorlord. The tables use the `standard_results.py` join on (run_id, floor) and count his outcomes only at level 0. Escapes contribute neither wins nor deaths.

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
