# Search teacher audit and gate G1 on Gym v1.2.4

This is the stage-1 teacher audit of `docs/superpowers/specs/2026-11-21-search-distillation-design.md` (plan `docs/superpowers/plans/2026-11-24-search-teacher-audit.md`, Task 10). The plan names Gym v1.2.3. The pin is now Gym v1.2.4, so every run here uses Gym revision `673eb8f220023b0c95401bbf96bec814a359457b` (clean release build), Standard rules epoch `gym-1.2.4`, recipe `configs/neural/baalorlord-standard-gym-1.2.4/recipe.json` (SHA-256 `5d53d06566ae8de898de78140d10aaffe6ac39204b1d245ec98747ed09403fb5`) and state table SHA-256 `3121570f1b970d96acd76e9c6c7681ca32789e22c9b11af74ab5aa9283dfb6a0`. The panels, the `current` search references, the PPO baselines and the teacher budgets are described in `docs/neural-standard-baseline-gym-1.2.4.md`. The results measure simulation under RNG-001 and TIME-001. Original-game transfer remains unmeasured. Only aggregates are published.

## Results in short

- **G1 fails.** Neither tree teacher beats `current` on the selection panel. `tree-1x` against `current`: 5 against 5 discordant cases, p = 1. `tree-3x` against `current`: 8 against 7, p = 1.
- **The chosen teacher is `current`.** It has the most selection-panel wins of the three candidates (233, against 231 for `tree-1x` and 232 for `tree-3x`). Per the gate table, the work continues with this teacher.
- **The stop rule gives CONTINUE.** `current` built a search root at 7,565 of 7,604 boss-fight decisions that needed one (0.9949).
- **The teacher agrees with itself 68.5% of the time.** This is the spec's "consistency ceiling", a reference level rather than a strict upper bound. Two `tree-3x` searches of the same 2,000 selection-split decisions chose the same action 1,369 times.
- The selection panel is near its ceiling for every search (231 to 240 wins of 256), so it separates teachers poorly. The tree teachers keep more HP than `current` on both panels. `tree-10x` is the strongest teacher on both panels, but it is outside the candidate set and is on the strength curve only.
- Simulator faults are errors. There were none.

## Teachers

The four teachers are in `configs/neural/search-teacher-gym-1.2.4/teachers.json` (file SHA-256 `dc5d969c9b5e85dd2232ec5f2ed48bfe9de78a9dcba8e94491d08d2da34a5925`). All use native Gym playouts (`spire-gym-public-playout-v1`) of 80 steps, the compact branch layout and one worker per search.

| Teacher | Kind | Budget per decision | Other settings | Config SHA-256 |
| --- | --- | --- | --- | --- |
| `current` | one-move search | 32 samples per legal action | none | `2f29eef0…` |
| `tree-1x` | public tree search | `simulations_per_action` 22 | `min_root_visits` 8, 8 worlds, `ucb_c` 1.0, `max_depth` 4, widening (2.0, 0.5) | `602c1b5d…` |
| `tree-3x` | public tree search | `simulations_per_action` 66 | as `tree-1x` | `7db31be6…` |
| `tree-10x` | public tree search | `simulations_per_action` 220 | as `tree-1x` | `2faa984d…` |

The config hashes are the per-teacher `config_sha256` values the runs recorded. Budgets scale with the number of legal actions, as `current`'s does. The tree constants were carried from the v1.2.0 calibration and not re-tuned.

**Calibration.** `current` is the cost unit: 0.2815 CPU s per searched decision on `--train-sample 200` (seed 2026112501). Each tree budget was verified on `--train-sample 100` (seed 2026112502) against its target multiple, within plus or minus 15%: measured ratios 1.01, 3.33 and 11.14 for targets 1, 3 and 10. All calibration fights were training-split fights. The measurements, the legal-action bins and the earlier fixed-budget probe are in `docs/neural-standard-baseline-gym-1.2.4.md`, section "Teacher budgets". They are not repeated here.

Mean simulations per searched decision on the panels were 214 and 208 for `current` (selection and boss), 136 and 129 for `tree-1x`, 386 and 385 for `tree-3x`, and 1,386 and 1,375 for `tree-10x`. A tree simulation descends up to four tree levels before its playout, so simulation counts do not compare directly with `current`'s.

## Runs

Every run played one fight per case with seed 20261117, a 1,024-decision fight cap and the 500-action cap on one public turn. A fight that reaches either cap is censored.

| Teacher | Panels | Learner revision | Container | Wall time, selection / boss |
| --- | --- | --- | --- | --- |
| `current` | selection, boss | `b4f8a36e42d49ebb593ccc5de07121481fba244d` | 12 CPUs, `--parallel 12`, one panel at a time | 2,357 / 434 s |
| `tree-1x` | selection, boss | `556eef68004b9a81ee273f5d8a95399bda8e2729` | 1 CPU, `--parallel 1`, per panel | 2,146 / 2,978 s |
| `tree-3x` | selection, boss | `556eef6` | 2 CPUs, `--parallel 2`, per panel | 3,487 / 4,668 s |
| `tree-10x` | selection, boss | `556eef6` | 5 CPUs, `--parallel 5`, per panel | 4,486 / 6,483 s |

The `current` runs are the search references of the baseline document. Between `b4f8a36` and `556eef6` nothing under `src/` or `scripts/` changed, so all eight runs used the same search code. The six tree runs ran at once, 16 CPUs in total, on a host with 16 physical cores and 32 hardware threads. The plan asked for one detached chain of runs on at most 24 cores. The ruling kept the total at 16 because above that, concurrent fights would share physical cores through SMT and inflate CPU seconds per decision. The container limits are CPU quotas, not pinned cores.

All eight runs and the consistency audit exited 0 with `exit_status` 0, `failure` null, `timed_out` false, empty `stderr.txt` and clean learner trees before and after (`learner_changed` false). A simulator fault stops a run as an error, so the clean exits show there were none.

The tables below come from `teacher_audit.py report` (`report.json`, SHA-256 `8f76c36a6d049683541b18381a4a924b3c3bcbcef4ac32bd9b265414d2b048f7`):

```
python scripts/neural/teacher_audit.py report \
  --run TEACHER:PANEL:RUN_DIR [--run TEACHER:PANEL:RUN_DIR ...] \
  --panel selection:DEVELOPMENT_PANEL --panel boss:BOSS_PANEL \
  --ppo r1:selection:SUMMARY --ppo r2:selection:SUMMARY \
  --ppo r1:boss:SUMMARY --ppo r2:boss:SUMMARY \
  --run-states RUN_STATES --output OUT
```

There is one `--run` for each of the eight teacher and panel runs. The panel documents are the `development-panel.json` and `boss-panel.json` that the baseline runs saved. The per-stratum tables, the cost without the longest fight, the HP comparisons on shared wins, the post hoc tests and the tie counts come from one-off read-only aggregations of the same `fights.jsonl`, `decisions.jsonl` and PPO summaries. They are not repository scripts.

## Strength against budget

An outcome is a win, a death, an escape or a censored case. Wins are out of 256. "L0 wins" counts wins on the 64 level-0 (unfuzzed) entries. HP kept is HP after the fight over the case's starting maximum HP, averaged over completed fights (a death counts 0). For PPO it is the evaluation summary's mean HP retention over completed cases. PPO r1 and r2 are the final checkpoints of the two v1.2.4 baselines. Their censored cases also all hit the turn cap (see the baseline document). CPU s per searched decision sums each search's CPU time over decisions with a search root. Fight CPU s per decision divides each fight's whole CPU time by all its decisions, including the single-legal-action ones.

**Selection panel.** Baalorlord's recorded runs won 63 of the 64 level-0 fights.

| Player | Wins | Deaths | Escaped | Censored | L0 wins | HP kept | HP kept on wins | CPU s / searched decision | Fight CPU s / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 233 | 23 | 0 | 0 | 58 | 0.380 | 0.417 | 0.630 | 0.575 |
| `tree-1x` | 231 | 23 | 0 | 2 | 57 | 0.494 | 0.543 | 0.283 | 0.252 |
| `tree-3x` | 232 | 21 | 0 | 3 | 59 | 0.510 | 0.556 | 0.859 | 0.774 |
| `tree-10x` | 240 | 15 | 0 | 1 | 61 | 0.556 | 0.591 | 2.918 | 2.597 |
| PPO r1 | 179 | 72 | 1 | 4 | 45 | 0.386 | – | – | – |
| PPO r2 | 166 | 88 | 1 | 1 | 45 | 0.348 | – | – | – |

**Boss panel.** Baalorlord's recorded runs won 63 of the 64 level-0 fights.

| Player | Wins | Deaths | Escaped | Censored | L0 wins | HP kept | HP kept on wins | CPU s / searched decision | Fight CPU s / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 176 | 78 | 2 | 0 | 45 | 0.253 | 0.362 | 0.469 | 0.432 |
| `tree-1x` | 165 | 87 | 2 | 2 | 42 | 0.300 | 0.455 | 0.321 | 0.290 |
| `tree-3x` | 179 | 72 | 2 | 3 | 47 | 0.340 | 0.474 | 0.952 | 0.864 |
| `tree-10x` | 197 | 56 | 2 | 1 | 50 | 0.378 | 0.484 | 3.485 | 3.153 |
| PPO r1 | 102 | 148 | 2 | 4 | 30 | 0.179 | – | – | – |
| PPO r2 | 89 | 165 | 2 | 0 | 23 | 0.155 | – | – | – |

Every teacher escaped the same two boss-panel cases, act 1 elites at levels 2 and 4. Every censored teacher case hit the turn cap (`turn_cap`), none the fight cap:

| Teacher | Selection panel | Boss panel |
| --- | --- | --- |
| `current` | none | none |
| `tree-1x` | 2: act 1 elite, act 3 boss | 2: act 3 boss, act 4 boss |
| `tree-3x` | 3: act 1 elite, act 2 elite, act 3 boss | 3: act 1 boss, act 3 boss, act 4 boss |
| `tree-10x` | 1: act 1 elite | 1: act 1 boss |

No case was censored by all three tree teachers. The censored cases are five distinct selection cases and four distinct boss cases.

### Wins

On the selection panel every search wins 231 to 240 of 256 (90% to 94%), and all four win every act 1 and act 2 monster case. The panel has little room left to separate teachers. `tree-10x` wins most, 240, and has the fewest deaths, 15. The three candidates differ by at most two wins.

The boss panel spreads the teachers further: 165, 176, 179 and 197 wins for `tree-1x`, `current`, `tree-3x` and `tree-10x`. Wins rise with the tree budget, from 165 at 1x to 179 at 3x and 197 at 10x. `tree-1x` is below `current` there and `tree-3x` slightly above it. Every search is well above both PPO finals on both panels. At level 0, every search remains below Baalorlord's recorded 63 of 64 on both panels. The closest is `tree-10x`'s 61 on the selection panel.

### HP kept

The tree teachers keep more HP than `current` at every budget. Over completed fights on the selection panel, `current` keeps 0.380 of maximum HP and the trees keep 0.494, 0.510 and 0.556. On the boss panel the figures are 0.253 against 0.300, 0.340 and 0.378.

Part of that gap could come from which cases each teacher wins. Restricted to cases both sides won, the gap remains. The table compares each tree teacher with `current` on those cases. It is a post hoc description, and the sign-test p-values are unadjusted.

| Panel | Tree teacher | Cases both won | Tree HP kept | `current` HP kept | Tree kept more / less / equal | Sign-test p |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Selection | `tree-1x` | 226 | 0.550 | 0.416 | 160 / 59 / 7 | 5.9e-12 |
| Selection | `tree-3x` | 224 | 0.561 | 0.413 | 163 / 50 / 11 | 3.6e-15 |
| Selection | `tree-10x` | 228 | 0.601 | 0.415 | 180 / 39 / 9 | 7.7e-23 |
| Boss | `tree-1x` | 151 | 0.468 | 0.378 | 92 / 50 / 9 | 5.3e-4 |
| Boss | `tree-3x` | 160 | 0.499 | 0.373 | 105 / 46 / 9 | 1.8e-6 |
| Boss | `tree-10x` | 171 | 0.503 | 0.369 | 119 / 48 / 4 | 3.8e-8 |

HP kept is not a gate metric. The pre-registered teacher choice uses wins only.

### Cost

On the matched calibration samples, `tree-1x` and `current` cost the same (ratio 1.01), and `tree-3x` and `tree-10x` cost 3.33 and 11.14 times as much. The panel figures do not repeat those ratios, for two reasons.

First, one fight inflates `current`'s selection-panel cost. That act 4 boss case at level 0 used 2,326 CPU s, 54% of `current`'s search CPU on the panel, at 5.35 CPU s per searched decision. Without it `current` costs 0.311 CPU s per searched decision (the baseline document's 0.312 divides fight CPU instead). Every teacher won that case:

| Teacher | Decisions | CPU s | CPU s / searched decision | Share of the teacher's search CPU on the panel |
| --- | ---: | ---: | ---: | ---: |
| `current` | 437 | 2,326 | 5.35 | 54% |
| `tree-1x` | 82 | 47 | 0.61 | 2% |
| `tree-3x` | 230 | 1,139 | 4.99 | 16% |
| `tree-10x` | 70 | 448 | 6.69 | 2% |

It was also `tree-3x`'s costliest fight. The baseline document estimated that `tree-10x` would need about six hours on this fight if it played it as long as `current`. It won in 70 decisions and 448 CPU s. Both parts of the estimate failed: the fight was shorter, and on it `tree-10x` cost 6.69 CPU s per searched decision against `current`'s 5.35, about 1.25 times, not 11.

Second, the panel runs did not share a container allocation. `current` ran 12 fights in a 12-CPU container. The tree teachers ran six containers on 16 CPUs at once. CPU time per decision can depend on how busy the host is.

| Teacher | Selection, ratio to `current` | Selection without the case above | Boss panel | Calibration |
| --- | ---: | ---: | ---: | ---: |
| `tree-1x` | 0.45 | 0.90 | 0.68 | 1.01 |
| `tree-3x` | 1.36 | 2.38 | 2.03 | 3.33 |
| `tree-10x` | 4.63 | 9.27 | 7.43 | 11.14 |

The ratios divide CPU s per searched decision. On the panels each tree teacher costs less, relative to `current`, than on the calibration samples. The calibration is the matched measurement. The panel ratios depend on which fights run long and on the host load, so they are not a re-calibration.

### Wins by stratum

Cells hold 2 to 14 cases per level, and single-cell differences are noise. "L0 cases" and "his L0" count Baalorlord's recorded outcomes at level 0 only.

**Selection panel, by act and room kind**

| Act / kind | Cases | `current` | `tree-1x` | `tree-3x` | `tree-10x` | PPO r1 | PPO r2 | L0 cases | His L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 monster | 56 | 56 | 56 | 56 | 56 | 55 | 54 | 14 | 14 |
| 1 elite | 28 | 27 | 26 | 27 | 25 | 17 | 14 | 7 | 7 |
| 1 boss | 8 | 6 | 6 | 6 | 7 | 1 | 3 | 2 | 1 |
| 2 monster | 44 | 44 | 44 | 44 | 44 | 39 | 36 | 11 | 11 |
| 2 elite | 20 | 17 | 16 | 15 | 17 | 8 | 9 | 5 | 5 |
| 2 boss | 8 | 4 | 5 | 6 | 6 | 3 | 2 | 2 | 2 |
| 3 monster | 40 | 39 | 39 | 39 | 40 | 35 | 29 | 10 | 10 |
| 3 elite | 20 | 16 | 16 | 17 | 19 | 10 | 11 | 5 | 5 |
| 3 boss | 16 | 13 | 12 | 12 | 15 | 6 | 7 | 4 | 4 |
| 4 elite | 8 | 7 | 7 | 7 | 8 | 3 | 0 | 2 | 2 |
| 4 boss | 8 | 4 | 4 | 3 | 3 | 2 | 1 | 2 | 2 |
| Monster | 140 | 139 | 139 | 139 | 140 | 129 | 119 | 35 | 35 |
| Elite | 76 | 67 | 65 | 66 | 69 | 38 | 34 | 19 | 19 |
| Boss | 40 | 27 | 27 | 27 | 31 | 12 | 13 | 10 | 9 |

**Selection panel, by fuzz level**

| Level | Cases | `current` | `tree-1x` | `tree-3x` | `tree-10x` | PPO r1 | PPO r2 | His L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 64 | 58 | 57 | 59 | 61 | 45 | 45 | 63 |
| L1 | 64 | 60 | 62 | 61 | 61 | 47 | 39 | – |
| L2 | 64 | 58 | 59 | 59 | 60 | 47 | 42 | – |
| L4 | 64 | 57 | 53 | 53 | 58 | 40 | 40 | – |

**Boss panel, by act and room kind**

| Act / kind | Cases | `current` | `tree-1x` | `tree-3x` | `tree-10x` | PPO r1 | PPO r2 | L0 cases | His L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 elite | 36 | 32 | 28 | 30 | 32 | 23 | 17 | 9 | 9 |
| 1 boss | 40 | 21 | 19 | 21 | 25 | 8 | 12 | 10 | 9 |
| 2 elite | 24 | 21 | 19 | 19 | 21 | 13 | 11 | 6 | 6 |
| 2 boss | 32 | 19 | 20 | 22 | 23 | 12 | 10 | 8 | 8 |
| 3 elite | 24 | 22 | 23 | 23 | 24 | 18 | 16 | 6 | 6 |
| 3 boss | 60 | 41 | 41 | 46 | 48 | 20 | 19 | 15 | 15 |
| 4 elite | 12 | 9 | 9 | 10 | 11 | 5 | 2 | 3 | 3 |
| 4 boss | 28 | 11 | 6 | 8 | 13 | 3 | 2 | 7 | 7 |
| Elite | 96 | 84 | 79 | 82 | 88 | 59 | 46 | 24 | 24 |
| Boss | 160 | 92 | 86 | 97 | 109 | 43 | 43 | 40 | 39 |

**Boss panel, by fuzz level**

| Level | Cases | `current` | `tree-1x` | `tree-3x` | `tree-10x` | PPO r1 | PPO r2 | His L0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 64 | 45 | 42 | 47 | 50 | 30 | 23 | 63 |
| L1 | 64 | 49 | 43 | 45 | 52 | 28 | 28 | – |
| L2 | 64 | 43 | 44 | 48 | 51 | 28 | 23 | – |
| L4 | 64 | 39 | 36 | 39 | 44 | 16 | 15 | – |

On the boss panel, `tree-10x` wins 109 of 160 boss cases against `current`'s 92. Act 4 bosses remain the hardest cell for every search (6 to 13 of 28).

## Gate G1

G1 follows the spec: an exact two-sided McNemar test on the selection panel for each tree candidate against `current`, at alpha 0.025 each (Bonferroni over two tests). A case censored or escaped on either side is dropped from the pair. A teacher "beats" `current` if it has more wins on the paired cases and p < 0.025. There is one teacher seed, so each test pairs the single run at that budget with the single `current` run. The effect is the difference in discordant cases as a share of paired cases, in percentage points.

| Challenger | Paired | Dropped | Both win | Only challenger | Only `current` | Neither | Paired wins, challenger / `current` | Effect (points) | Exact p | Beats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `tree-1x` | 254 | 2 | 226 | 5 | 5 | 18 | 231 / 231 | 0.0 | 1 | no |
| `tree-3x` | 253 | 3 | 224 | 8 | 7 | 14 | 232 / 231 | +0.4 | 1 | no |

The dropped cases are the tree runs' censored cases. `current` won both cases dropped from the `tree-1x` pair and two of the three dropped from the `tree-3x` pair, which is why its paired wins are 231 rather than 233.

**G1: FAIL.** Per the gate table, the work continues with whichever teacher the selection rule picks.

The selection panel separates these teachers poorly. Only 10 and 15 paired cases are discordant, and both effects are well inside the 4 to 5 point range that reads as no difference (see "Reading the differences").

**Post hoc comparisons (not part of G1).** These were not pre-registered and are not adjusted for multiple comparisons. The table is a chosen subset of the pairwise comparisons computed, so an unadjusted p near 0.01 should not be read as significant. They describe the strength curve and the boss panel, which the spec reports beside the gates but does not use for pass or fail. `tree-10x` is outside the candidate set by rule.

| Panel | Comparison | Paired | Only first | Only second | Effect (points) | Exact p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selection | `tree-10x` vs `current` | 255 | 12 | 4 | +3.1 | 0.077 |
| Boss | `tree-1x` vs `current` | 252 | 14 | 24 | −4.0 | 0.14 |
| Boss | `tree-3x` vs `current` | 251 | 19 | 15 | +1.6 | 0.61 |
| Boss | `tree-10x` vs `current` | 253 | 26 | 5 | +8.3 | 0.00019 |
| Boss | `tree-3x` vs `tree-1x` | 251 | 21 | 7 | +5.6 | 0.013 |
| Boss | `tree-10x` vs `tree-3x` | 250 | 22 | 6 | +6.4 | 0.0037 |

The first two boss rows (`tree-1x` and `tree-3x` against `current`) come from `report.json` (`g1.boss_panel`), which marks them as not used. The other rows are one-off computations with the same exact test. On the boss panel the tree's wins grow with its budget, and `tree-10x` is ahead of `current` by 8.3 points. On the selection panel the same comparison gives +3.1 points with p = 0.077.

## Teacher choice

The spec's rule picks the candidate with the most selection-panel wins among `current`, `tree-1x` and `tree-3x`, all within 3x the cost of `current`. Ties go to the cheaper teacher. `tree-10x` is measured for the strength curve only.

| Candidate | Selection wins | Nominal budget | CPU s / searched decision |
| --- | ---: | --- | ---: |
| `current` | 233 | 1x | 0.630 |
| `tree-1x` | 231 | 1x | 0.283 |
| `tree-3x` | 232 | 3x | 0.859 |

**Chosen teacher: `current`**, with the most wins outright. No tie-break was needed. The margins are one and two wins, far below anything the panel can resolve. The rule was fixed before the results, and the choice follows it.

## Self-consistency (the spec's consistency ceiling)

The consistency audit drew 2,000 searched decisions from the `tree-3x` selection-panel run. The sample was drawn with `random.Random(2026112601).sample` from the 8,088 searched decisions, ordered by case and step. It covers 252 of the 256 cases, and its SHA-256 is `7a22432af61ebdfa78b33fbb393c490e723ba2098209ee9928906e3111c51b3b`. These are selection-split decisions: audit only, never training data. Each decision's state was rebuilt by resetting the case and replaying the recorded choices, and any mismatch would have stopped the run. Then `tree-3x` searched each state again (run B) with the recorded research seed plus one. The recorded search is run A.

```
python scripts/neural/teacher_audit.py consistency \
  --decisions TREE3X_SELECTION_RUN/decisions.jsonl --sample 2000 --seed 2026112601 \
  --teacher tree-3x --teachers configs/neural/search-teacher-gym-1.2.4/teachers.json \
  --recipe configs/neural/baalorlord-standard-gym-1.2.4/recipe.json --parallel 12 --output OUT
```

It ran at learner revision `696c4844a07c68ca135de66fad7e6b892ea05cc5` (clean, unchanged after) in a 12-CPU container, in 301 s, at 0.80 CPU s per re-searched decision. `consistency.json` has SHA-256 `f6311d976834658b00f50614ca5cbfabfb528c4a8a82af1641878988d09d8fab`.

**Seeds.** The teacher run gives each decision the research seed `(seed + case index) × 1024 + step`. Seed plus one is therefore the seed run A used at the same case's next decision. That holds for 1,933 sampled decisions. The other 67 are the last decision of their fight, and A never used their seed plus one. No fight reached 1,024 decisions (the longest had 580), so seed plus one never crosses into another case. Where A did use the seed, it searched a different state. At each audited state, A and B searched with different seeds, so the two searches used different seeds.

**Value.** A search scores each root action by the mean return of that action's simulations. A simulation returns +1 if the simulated fight ends in victory, −1 on death and 0 on escape. If the fight has not ended when the simulation stops (at the end of its 80-step playout, or earlier if the branch is truncated), it returns half the HP balance. The HP balance is the player's HP fraction minus the remaining HP fraction of the living enemies, in [−1, 1] (`View.balance` in `src/spire_learn/neural/search_view.py`). A refused branch returns −1 (`teach` in `src/spire_learn/neural/search_teacher.py`). An action's value therefore lies in [−1, 1], and the teacher chooses the action with the highest value. "Value lost" is A's value of A's own choice minus A's value of B's choice, and symmetrically for B. It is zero when the runs agree.

| Measure | Value |
| --- | ---: |
| Decisions | 2,000 |
| Top-choice agreement | 1,369 (68.5%) |
| Mean value lost, run A's values | 0.0445 |
| Mean value lost, run B's values | 0.0421 |
| Mean value lost on the 631 disagreements, run A's values | 0.141 |
| Mean value lost on the 631 disagreements, run B's values | 0.134 |
| Branch refusals in the reruns | 14 |

**Reading.** 68.5% is how often the teacher matches itself at a fresh seed. It is a reference level for students, not a strict upper bound. A student that learns the teacher's most likely choice can match a single run more often than a second noisy run does. A student far below this level points to a learning problem, and one near it points to the teacher's own noise (the spec's G2 reading). Most disagreements cost little by the teacher's own estimate. Over all 2,000 decisions the loss averages 0.04 on a scale where a certain win and a certain loss differ by 2. When the runs disagree, it averages 0.13 to 0.14. For two actions whose simulations all end the fight, 0.14 is a seven-point difference in simulated win share. So many disagreements are between actions the search rates as close, though not all are. The audit output keeps only these aggregates, so the distribution of the loss is not measured.

**Ties.** In run A, 1,250 of the 2,000 sampled decisions (62.5%) had two or more actions tied exactly for the top value. In 986 of them the tie was at 1.0 (every simulation of the tied actions won), in 216 at −1.0 and in 48 elsewhere. In 942, every legal action tied. The tree teacher breaks an exact tie by the lowest action index, so when both runs tie on the same set of actions, they agree by that rule, not by preference. The agreement on decisions without a tie may therefore be well below 68.5%. The output does not split agreement by tie status, so this is not measured and the aggregates cannot bound it. Exact top ties are common in every teacher's selection-panel decisions: 71.4% of `current`'s searched decisions, and 65.6%, 62.9% and 53.3% of `tree-1x`'s, `tree-3x`'s and `tree-10x`'s. `current` breaks ties at random.

The audit measured `tree-3x`, as the plan specifies. The self-agreement of `current`, the chosen teacher, was not measured.

## Root refusals and label coverage

A root is a public search root that Gym could build at a decision. Decisions with one legal action take it directly and need no root. When Gym refused a root, the teacher recorded the refusal class and played the heuristic choice. "Root" below is the share of decisions with more than one legal action that got a root. "Labelled" is the share of all decisions that got a search, which is the share that would carry a search label.

| Panel | Teacher | Decisions | Single legal action | Root attempts | Refused | Root | Labelled | Branch refusals (decisions) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Selection | `current` | 7,541 | 588 | 6,953 | 79 | 98.86% | 91.2% | 11 (7) |
| Selection | `tree-1x` | 8,485 | 884 | 7,601 | 60 | 99.21% | 88.9% | 69 (36) |
| Selection | `tree-3x` | 8,990 | 840 | 8,150 | 62 | 99.24% | 90.0% | 48 (20) |
| Selection | `tree-10x` | 7,854 | 802 | 7,052 | 65 | 99.08% | 89.0% | 33 (13) |
| Boss | `current` | 8,748 | 682 | 8,066 | 16 | 99.80% | 92.0% | 205 (38) |
| Boss | `tree-1x` | 10,241 | 982 | 9,259 | 19 | 99.79% | 90.2% | 145 (31) |
| Boss | `tree-3x` | 10,775 | 958 | 9,817 | 38 | 99.61% | 90.8% | 360 (58) |
| Boss | `tree-10x` | 9,922 | 911 | 9,011 | 36 | 99.60% | 90.5% | 979 (78) |

A refused branch inside a search scores −1 for that simulation and is counted, as in the baseline document.

Refused roots by class:

| Class | Selection: `current` / `tree-1x` / `tree-3x` / `tree-10x` | Boss: `current` / `tree-1x` / `tree-3x` / `tree-10x` |
| --- | --- | --- |
| a card read a hidden intent | 25 / 27 / 33 / 20 | 10 / 14 / 36 / 21 |
| a hidden pre-rolled value was shown by damage | 19 / 13 / 16 / 26 | 0 / 0 / 0 / 0 |
| draw knowledge exceeds the exact model | 22 / 10 / 6 / 0 | 0 / 0 / 0 / 14 |
| an executed enemy move was not identifiable | 5 / 6 / 3 / 12 | 0 / 0 / 0 / 0 |
| a nested card play holds an enemy target | 8 / 4 / 4 / 7 | 4 / 5 / 1 / 1 |
| a queued Distilled Chaos play holds a random target | 0 / 0 / 0 / 0 | 2 / 0 / 1 / 0 |

Root and labelled shares by act and room kind (root % / labelled %):

**Selection panel**

| Act / kind | `current` | `tree-1x` | `tree-3x` | `tree-10x` |
| --- | --- | --- | --- | --- |
| 1 monster | 100.0 / 87.6 | 100.0 / 81.0 | 100.0 / 81.0 | 100.0 / 81.3 |
| 1 elite | 100.0 / 85.4 | 100.0 / 91.4 | 100.0 / 90.7 | 100.0 / 90.2 |
| 1 boss | 100.0 / 87.4 | 100.0 / 83.3 | 100.0 / 84.0 | 100.0 / 85.4 |
| 2 monster | 99.7 / 91.1 | 100.0 / 85.1 | 99.9 / 85.5 | 100.0 / 86.0 |
| 2 elite | 99.7 / 89.8 | 100.0 / 88.9 | 100.0 / 95.0 | 100.0 / 89.9 |
| 2 boss | 91.6 / 83.1 | 96.2 / 85.2 | 97.6 / 87.5 | 99.4 / 90.6 |
| 3 monster | 95.8 / 89.7 | 96.9 / 87.4 | 96.0 / 87.3 | 94.8 / 85.5 |
| 3 elite | 99.7 / 93.3 | 99.8 / 91.5 | 99.9 / 92.6 | 99.8 / 94.0 |
| 3 boss | 100.0 / 95.9 | 99.8 / 95.9 | 100.0 / 96.0 | 99.8 / 94.8 |
| 4 elite | 100.0 / 97.0 | 96.7 / 91.1 | 98.4 / 93.3 | 100.0 / 95.8 |
| 4 boss | 100.0 / 98.3 | 100.0 / 93.7 | 100.0 / 95.5 | 100.0 / 93.9 |

**Boss panel**

| Act / kind | `current` | `tree-1x` | `tree-3x` | `tree-10x` |
| --- | --- | --- | --- | --- |
| 1 elite | 100.0 / 86.7 | 100.0 / 84.0 | 100.0 / 83.1 | 100.0 / 83.7 |
| 1 boss | 100.0 / 85.9 | 100.0 / 83.0 | 100.0 / 87.9 | 99.1 / 87.4 |
| 2 elite | 100.0 / 88.4 | 99.8 / 86.6 | 100.0 / 86.4 | 100.0 / 86.8 |
| 2 boss | 99.8 / 92.0 | 99.9 / 89.4 | 100.0 / 89.8 | 100.0 / 92.1 |
| 3 elite | 100.0 / 93.5 | 100.0 / 91.1 | 100.0 / 92.0 | 100.0 / 92.1 |
| 3 boss | 99.6 / 95.1 | 99.9 / 93.2 | 99.4 / 93.1 | 99.3 / 92.8 |
| 4 elite | 100.0 / 91.9 | 100.0 / 92.5 | 100.0 / 91.7 | 100.0 / 92.4 |
| 4 boss | 99.7 / 95.2 | 99.0 / 92.3 | 98.9 / 92.8 | 99.8 / 90.6 |

Every cell has a root at 91.6% or more of its attempts. The lowest cells are selection 2 boss (91.6% for `current`, mostly draw knowledge, as in the baseline document, and 96.2% to 99.4% for the trees), selection 3 monster for every teacher (94.8% to 96.9%) and selection 4 elite for `tree-1x` (96.7%). Each teacher reaches different states, so the cells hold different decisions. The labelled share is lower than the root share mainly because of single-legal-action decisions, which need no label.

## Stop rule

The spec stops before the dataset if fewer than half of the boss-fight decisions in the chosen teacher's own fights have a root. The share counts decisions with more than one legal action, over both panels.

| Teacher | Boss-fight decisions with a root | Root attempts | Share | Verdict |
| --- | ---: | ---: | ---: | --- |
| `current` (chosen) | 7,565 | 7,604 | 0.9949 | **CONTINUE** |

For reference, the same share is 0.9967 for `tree-1x`, 0.9954 for `tree-3x` and 0.9954 for `tree-10x`. The rule passes by a wide margin.

## Reading the differences

The spec fixes this caution before the results. With about 256 paired cases, differences below about 4 to 5 percentage points read as no difference. G1 paired 254 and 253 cases, with effects of 0.0 and +0.4 points. The win counts of the three candidates differ by one or two cases. The HP-kept gap is not covered by this caution, which concerns win rates. It is large and consistent, but it is a post hoc description, not a gate.

## Open questions for the user

These are not recommendations. The next plan (parts 3 to 5: dataset, students, PPO arm, gates G2 to G4) decides the dataset teacher.

- `tree-10x` is the strongest teacher on both panels (240 and 197 wins, against `current`'s 233 and 176), and post hoc it is 8.3 points ahead of `current` on the boss panel. By the spec it is on the strength curve only, because it costs about 11 times `current` on calibration and would not fit the labelling budget. Whether that holds is for the next plan.
- Exact top-value ties are common: 53% to 71% of searched decisions, depending on the teacher. `current` breaks them at random and the tree teacher by the lowest action index. A label on a fully tied decision carries no preference. How the dataset treats such labels is open.
- The consistency audit measured `tree-3x`, not the chosen `current`. It kept only aggregates, so agreement cannot be split by tie status or by act and room kind.
