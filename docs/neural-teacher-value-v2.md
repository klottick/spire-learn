# Teacher value v2 on Gym v1.2.5

This is the results document of teacher value v2 (plan `docs/superpowers/plans/2026-11-26-teacher-value-v2.md`, Task 8). It measures search teachers that value the state a fight leaves behind, not just the outcome, and it picks the dataset teacher by the rule fixed in the plan. Stage 1 is in `docs/neural-search-teacher-gym-1.2.4.md`, and the panels, PPO baselines and budgets are in `docs/neural-standard-baseline-gym-1.2.4.md`.

## Provenance

- **Gym.** Release v1.2.5, revision `731fca0e0341d80e7cf08a8be7a03be1fa32c3f1`, clean release build (`dirty` false, `source_digest` `19854fcc…`, `file_digest` `63533371…`). It is a drop-in inside rules epoch `gym-1.2.4`, so the v1.2.4 recipe, state table and panels are used unchanged. The six stage-1 tree runs are reused from Gym v1.2.4, revision `673eb8f…`, in the same epoch.
- **Learner.** All twelve Task 8 runs used the clean tracked tree at `b612e35c27975a261e083e93cbc385eae4f70c67`, unchanged before and after. The Task 6 checks used `1f8bc1f`, and the native parity test used `d3b0550`, both clean. The stage-1 tree runs used `556eef6`.
- **Profile.** `configs/neural/gym-1.2.5/runtime.json` (SHA-256 `b57420db…`), with the verified v1.2.5 wheel (`02346039…`) and extension (`67819e11…`).
- **Inputs.** Recipe `configs/neural/baalorlord-standard-gym-1.2.4/recipe.json` (SHA-256 `5d53d065…`) and state table `3121570f…`.
- **Identifiers.**
  - Standard rules `spire-gym-native-standard-rules-v9`.
  - Observation schema `spire-gym-standard-combat-view-compact-v1`, profile `StandardCombat`, capacity preset `standard-combat-compact-v1`.
  - Playout rules: `spire-gym-public-playout-v1` for the v1 teacher `current`, and `spire-gym-public-playout-tiered-v1` for every other teacher.
  - Value rules `flat-v1` and `resource-v2`, and playout policies `uniform-v1` and `no-potions-v2`.
  - Entry schema `spire-learn-search-teacher-entry-v1`, report schema `spire-learn-teacher-audit-report-v1`.
- **Teachers.**
  - v1: `configs/neural/search-teacher-gym-1.2.4/teachers.json` (SHA-256 `dc5d969c…`).
  - v2: `configs/neural/search-teacher-gym-1.2.4/teachers-v2.json` (SHA-256 `c270eb24…`).
  - Task 6 checks: a temporary file, `artifacts/development/neural-combat/gym-1.2.5/t6/teachers-check.json` (SHA-256 `09eec39c…`).
- **Seeds.**
  - The panel runs use 20261117.
  - The UCB check uses `--train-sample 100` with seed 2026112602.
  - The native and Python check uses `--train-sample 100` with seed 2026112603.
  - The determinism check uses `--train-sample 200` with seed 2026112501.
- **Budgets.** Section "Runs" gives CPU and wall time.
- **Scope.** The results measure simulation under RNG-001 and TIME-001. They are simulator results only; no original-game transfer is claimed. Only aggregates are published.

## Results in short

- **The chosen dataset teacher is `current-v2`.** Both v2 one-move teachers are eligible on both panels. `current-v2` has the lowest pooled resource loss, 0.4536, and `current-v2-sqrt`, at 0.4584, is within the 0.01 band behind it. The tie goes to `current-v2`, the first in candidate order. The control `current` has 0.6994.
- **The stop rule gives CONTINUE** for `current-v2`: it built a search root at 7,530 of the 7,547 boss-fight decisions that needed one (0.9977).
- **v2 keeps resources without losing wins.** `current-v2` wins 233 and 180 against `current`'s 233 and 176. It keeps 0.623 and 0.391 of max HP over completed fights, against 0.380 and 0.253, and uses 0.901 and 1.339 potions per won fight, against 1.373 and 1.523.
- **The value rule carries essentially all of the effect** (a post hoc attribution). Changing the playout policy alone (`current-v1-nopot`) changes nothing measurable. The sqrt curve and potion prices of 10, 12 and 15 HP give results within the choice rule's 0.01 tie band. One unadjusted comparison, 15 HP against 12, reaches p 0.019 on a −0.005 shared-win gap.
- **`tree-1x-v2` is ineligible.** It fails both the win floor and the McNemar test on both panels.
- **v2 removes most exact ties.** Exact top-value ties fall from 0.714 to 0.303 of searched decisions on the selection panel, and from 0.606 to 0.353 on the boss panel. No v2 decision ties at +1.
- There were no simulator faults. Every censored case hit the turn cap.

## Why v2, and the formula

Under v1 a won playout scores a flat +1 whatever HP is left. That causes two problems:
- **Ties.** 62.5% of the stage-1 `tree-3x` audit decisions had an exact top tie, and 71.4% of `current`'s selection-panel decisions did.
- **Careless HP and potion use.** In a decision where every action wins all its playouts, the teacher cannot tell keeping 70 HP from keeping 5, or drinking a potion from keeping it. `current` kept 0.38 of max HP on the selection panel.

HP and potions carry across an A20 run, so v2 scores them.

**Value rule `resource-v2`.** This is copied from the plan.
- **Definitions.**
  - `M` is the player's max HP at the search root.
  - `hp_end` is the HP at the branch's end. A terminal branch uses HP after combat cleanup, so combat-end hooks such as Burning Blood have applied, on a Smoke Bomb escape too. A live branch uses `hp`.
  - `max_end` is the max HP at the branch's end.
  - `e` is the remaining enemy HP share, as in `View.balance`.
- **Potion charge, in HP.** `P = Σ price(kind)` over potions held at the root but not at the end, minus `Σ price(kind)` over potions held at the end but not at the root. It compares multisets of potion kinds.
- **HP term.** `h = clip(hp_end / M, 0, 1)`. Apply `h ← sqrt(h)` when `hp_curve` is `"sqrt"`.
- **Bonus, outside the clip.** `b = 0.4 × (1.5 × (max_end − M) − P) / M`.
- **Values.**
  - Victory: `clamp(0.55 + 0.4 × h + b, 0.41, 1.0)`. That is 0.95 at full HP, leaving 0.05 of headroom for Feed and gained potions.
  - Smoke Bomb escape: `clamp(−0.2 + 0.4 × h + b, −0.99, 0.4)`.
  - Death: −1.
  - Cut-off, meaning a live branch at the step or depth limit: `clamp(0.4 × clip(h − e, −1, 1) + b, −0.99, 0.4)`.
  - Refused playout: −1, flagged.
- **Exchange rates.** With the linear curve, every term is in HP at the same 0.4/M scale. So 1 HP at the end and 1 HP of potion price weigh exactly the same, and max HP weighs 1.5. The sqrt curve changes the HP side by design.
- **Ordering.** The clamps guarantee victory ≥ 0.41 > any cut-off or escape ≥ −0.99 > death. The floors bind only in extreme potion spends.
- **Fixed constants.** The `max_hp_weight` 1.5 and the band constants 0.55, 0.4, 0.41, −0.2 and −0.99 are fixed for v2. At full HP the headroom saturates: with `M ≤ 96`, one gained typical potion fills the 0.05. This was accepted.

**Potion prices** (user ruling: a typical potion is worth about 10–15 HP).
- The default is `potion_hp = 12`.
- Overrides by Gym potion id:
  - `fairy_potion` (Fairy in a Bottle) is 30;
  - `fruit_juice` is 7.5, which offsets its +5 max HP gain at weight 1.5.
- The sensitivity variants use a default of 10 and of 15, with the same overrides.

**Playout policy `no-potions-v2`.** It chooses uniformly among legal actions other than `use_potion`, `discard_potion` and `end_turn`. If none remain, it ends the turn. Otherwise it chooses among all legal actions. It also replaces the heuristic choice when a v2 teacher's root is refused. Natively it is the Gym rule `tiered_verbs` with its defaults. The v1 rule could not keep potions out of playouts, because its fallback reopens them.

**Measuring resources.** Each fight row records its start and end HP, max HP and potions. The audit then computes, per completed fight and over the fight's starting max HP:
- HP lost is `hp_start − hp_after_cleanup`, with a death counting 0 after;
- potion HP spent is the potion charge at the fixed reference prices (12, `fairy_potion` 30, `fruit_juice` 7.5), whatever the teacher's own prices;
- resource loss is their sum.

All three are signed: gained potions and max HP count as negative loss.

## Gym v1.2.5 drop-in and determinism

v1.2.5 delivered the requested native tiered playout rule and the potion summary fields. It changes no rules id, fingerprint or replay digest, so Learn adopted it as a drop-in in epoch `gym-1.2.4`. The adoption and its determinism check are recorded in [`neural-gym-v1-migration.md`, "Drop-in v1.2.5"](neural-gym-v1-migration.md#drop-in-v125-2026-11-26). The check passed:
- All ten descriptor captures are byte-identical to v1.2.4's.
- One reset per mode passed.
- The card-metadata fingerprint is unchanged.
- v1 `current` on `--train-sample 200`, seed 2026112501, matched the v1.2.4 run exactly: 5,217 decision rows, excluding CPU time, and 200 fight rows.

**The panel re-run.** Task 8 extended the check to both panels. The v1 `current` re-run on v1.2.5 reproduced stage-1 `current` exactly:
- 7,541 selection-panel and 8,748 boss-panel decisions are equal on every recorded field except CPU time, including the choice and every action's value.
- All 256 fights on each panel are equal on outcome, censor reason, steps and HP after cleanup.

Only CPU time differs: 0.677 and 0.490 CPU s per searched decision, against stage 1's 0.630 and 0.469. The container size and host load differed between the two runs. The stage-1 tree runs from v1.2.4 are therefore compared directly with the v1.2.5 runs below.

## Exact native parity

The gated test `tests/integration/test_search_value_native.py` ran on v1.2.5 in a 4-CPU container. It passed in 5 s; the output is in `artifacts/development/neural-combat/gym-1.2.5/value-native-gate/stdout.txt`.

**Method.**
- The test replays each native playout's actions on a fresh branch with full-view stepping.
- Every playout pick had to be the rule's pick, and it had to lie in the tier that `View.playout_choice(..., "no-potions-v2")` draws from.
- For both the linear and the sqrt rule, the native summary scorers had to equal the observation scorers with exact float equality.
- On the production rule, the teacher's own `playout_branches` path also had to give the same value.

**Coverage.**
- **Production rule** (`tiered_verbs` with its defaults): 404 playouts were compared. The test module draws them from 20 training roots plus two targeted training entries, one holding a Smoke Bomb and one with Alchemize; the stdout reports only the totals. There were 229 victories, 170 deaths and 5 cut-offs at the step limit, and 54 playouts gained a potion. No production playout spent one. Eight root steps ended the fight and needed no playout.
- **Probe rule** (potions in the first tier, for scorer coverage only): 48 playouts were compared from the Smoke Bomb entry. All 48 were escapes, and all spent a potion.

There were no mismatches. This is the exact check that native v2 scoring is correct.

## Native and Python agreement (Task 6)

The Python backend implements the same tiers with different RNG streams, so the plan asked for statistical agreement only. Both ran `current-v2` on the same 100 training fights (seed 2026112603). The runs are in `artifacts/development/neural-combat/gym-1.2.5/t6/`.

| Measure | Native | Python | Gap | Limit |
| --- | ---: | ---: | ---: | --- |
| Wins | 84 | 85 | −1 | within 6 |
| Resource loss, all 100 fights | 0.309 | 0.318 | −0.009 | within 0.04 |
| Resource loss, 83 fights both won | 0.193 | 0.217 | −0.024 | within 0.02: **exceeded** |

**The shared-win gap.**
- The gap breaks down as −0.025 in HP lost and +0.0009 in potion HP, so all of it is HP.
- Its paired SE is 0.0188, so the gap is 1.3 SE.
- Of the 83 fights, native lost less in 25, more in 27, and the same in 31.
- The limit itself is 1.06 paired SE. Two teachers with identical play would exceed it about 29% of the time under a normal approximation.

**Controller ruling: proceed. This was a post hoc waiver of a pre-set stop condition.** The plan said a gap above 0.02 stops the task, and this one exceeded it. The controller judged it noise, for these reasons:
- The limit was tighter than this sample's noise.
- The win counts differ by one, the split is even, and the potion terms differ by under 0.001.
- The exact parity test proves that native summary scoring equals observation scoring on the same trajectories.

That test does not cover differences between the native and Python playout RNG streams, which is exactly what this check measures. So a small real difference in the Python reference path cannot be ruled out. The measurement runs are all native, so such a difference would not affect them.

The native run cost 0.294 CPU s per searched decision, against 0.899 for Python.

## UCB check (Task 6)

The v2 win band is narrower than v1's, so `ucb_c` 1.0 might over-explore. `tree-1x-v2` ran at three values on 100 training fights (seed 2026112602).

| `ucb_c` | Wins | Resource loss |
| ---: | ---: | ---: |
| 0.25 | 85 | 0.455 |
| 0.5 | 87 | 0.413 |
| 1.0 | 84 | 0.443 |

No value is 6 or more wins below the best. 0.5 has the lowest resource loss, so **`ucb_c` is 0.5**, committed in `38dfd9d`. The 0.5 check config has the same hash as the panel `tree-1x-v2`.

## Runs

The twelve runs used `search_teacher_entry.py` with seed 20261117, one worker per search, a 1,024-decision fight cap and the 500-action cap on one public turn. A fight that reaches either cap is censored. They ran at once in runtime containers, 16 CPUs in total, from 20:48 to 22:15 UTC on 2026-11-26. Each had a 43,200 s timeout and 3 GB of memory.

| Teacher | Value / policy | Panels | CPUs per run | Config SHA-256 | Wall s, selection / boss | Fight CPU s, selection / boss |
| --- | --- | --- | ---: | --- | --- | --- |
| `current` | `flat-v1` / `uniform-v1` | both | 2 | `2f29eef0…` | 2,789 / 2,005 | 4,656 / 3,948 |
| `current-v1-nopot` | `flat-v1` / `no-potions-v2` | both | 1 | `f32d1db3…` | 4,434 / 3,479 | 4,427 / 3,473 |
| `current-v2` | `resource-v2` linear, 12 HP | both | 1 | `df81b2d7…` | 5,195 / 3,554 | 5,189 / 3,547 |
| `current-v2-sqrt` | `resource-v2` sqrt, 12 HP | both | 1 | `4be49616…` | 2,406 / 3,654 | 2,399 / 3,647 |
| `tree-1x-v2` | `resource-v2` linear, 12 HP | both | 2 | `f5724496…` | 1,094 / 1,706 | 2,147 / 3,370 |
| `current-v2-p10` | `resource-v2` linear, 10 HP | selection | 1 | `460ef967…` | 5,183 / – | 5,176 / – |
| `current-v2-p15` | `resource-v2` linear, 15 HP | selection | 1 | `b7997fa7…` | 5,214 / – | 5,207 / – |

- **Teacher settings.** The one-move teachers use 32 samples per legal action and 80 playout steps. `tree-1x-v2` is `tree-1x`'s config (22 simulations per action, `min_root_visits` 8, 8 worlds, `max_depth` 4, widening (2.0, 0.5)) with `ucb_c` 0.5.
- **Config hashes.** They are the `config_sha256` values in each run's summary provenance. The reused stage-1 runs are `tree-1x` (`602c1b5d…`), `tree-3x` (`7db31be6…`) and `tree-10x` (`2faa984d…`).
- **Cost.** The twelve runs used about 47,185 fight CPU s (13.1 core-hours) and 5,214 s of wall time. The plan expected 15–20 core-hours.
- **Clean exits.** Every run exited 0 with `exit_status` 0, `failure` null, `timed_out` false, empty `stderr.txt`, 256 cases and `learner_changed` false. A simulator fault stops a run as an error, so there were none.

The runs are under `artifacts/development/neural-combat/runs/gym-1.2.5/value-v2/`, and the stage-1 tree runs are under `artifacts/development/neural-combat/runs/gym-1.2.4/teacher/`. The tables come from `teacher_audit.py report` (`report.json` in the value-v2 directory, SHA-256 `d1548d8b…`):

```
python scripts/neural/teacher_audit.py report \
  --run TEACHER:PANEL:RUN_DIR [--run TEACHER:PANEL:RUN_DIR ...] \
  --panel selection:DEVELOPMENT_PANEL --panel boss:BOSS_PANEL \
  --ppo r1:selection:SUMMARY --ppo r2:selection:SUMMARY \
  --ppo r1:boss:SUMMARY --ppo r2:boss:SUMMARY \
  --run-states RUN_STATES \
  --choose-v2 --v1-control current \
  --candidate current-v2 --candidate current-v2-sqrt --candidate tree-1x-v2 \
  --output OUT
```

There is one `--run` for each of the twelve new runs and the six stage-1 tree runs. Paired comparisons, shared-win figures, tie breakdowns and CPU splits come from one-off read-only aggregations of the same `fights.jsonl` and `decisions.jsonl`. They are not repository scripts.

## Strength against budget

Columns are as in the stage-1 document:
- "L0 wins" counts the 64 level-0 cases.
- HP kept is HP after cleanup over the case's starting max HP, averaged over completed fights, with a death counting 0.
- CPU s per searched decision sums search CPU over decisions with a root.
- Stage-1 tree rows are from v1.2.4.
- PPO r1 and r2 are the final checkpoints of the two v1.2.4 baselines.
- Baalorlord's recorded runs won 63 of the 64 level-0 fights on each panel.

**Selection panel**

| Player | Wins | Deaths | Escaped | Censored | L0 wins | HP kept | HP kept on wins | CPU s / searched decision | Fight CPU s / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 233 | 23 | 0 | 0 | 58 | 0.380 | 0.417 | 0.677 | 0.617 |
| `current-v1-nopot` | 233 | 23 | 0 | 0 | 59 | 0.370 | 0.407 | 0.641 | 0.586 |
| `current-v2` | 233 | 22 | 0 | 1 | 57 | 0.623 | 0.681 | 0.848 | 0.783 |
| `current-v2-sqrt` | 233 | 23 | 0 | 0 | 58 | 0.613 | 0.673 | 0.416 | 0.384 |
| `current-v2-p10` | 234 | 21 | 0 | 1 | 57 | 0.626 | 0.682 | 0.846 | 0.780 |
| `current-v2-p15` | 233 | 22 | 0 | 1 | 57 | 0.622 | 0.680 | 0.850 | 0.786 |
| `tree-1x-v2` | 224 | 30 | 0 | 2 | 52 | 0.551 | 0.624 | 0.305 | 0.282 |
| `tree-1x` | 231 | 23 | 0 | 2 | 57 | 0.494 | 0.543 | 0.283 | 0.252 |
| `tree-3x` | 232 | 21 | 0 | 3 | 59 | 0.510 | 0.556 | 0.859 | 0.774 |
| `tree-10x` | 240 | 15 | 0 | 1 | 61 | 0.556 | 0.591 | 2.918 | 2.597 |
| PPO r1 | 179 | 72 | 1 | 4 | 45 | 0.386 | – | – | – |
| PPO r2 | 166 | 88 | 1 | 1 | 45 | 0.348 | – | – | – |

**Boss panel**

| Player | Wins | Deaths | Escaped | Censored | L0 wins | HP kept | HP kept on wins | CPU s / searched decision | Fight CPU s / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 176 | 78 | 2 | 0 | 45 | 0.253 | 0.362 | 0.490 | 0.451 |
| `current-v1-nopot` | 176 | 78 | 2 | 0 | 45 | 0.258 | 0.370 | 0.443 | 0.411 |
| `current-v2` | 180 | 74 | 2 | 0 | 47 | 0.391 | 0.550 | 0.470 | 0.434 |
| `current-v2-sqrt` | 180 | 74 | 2 | 0 | 47 | 0.387 | 0.544 | 0.480 | 0.444 |
| `tree-1x-v2` | 166 | 85 | 2 | 3 | 44 | 0.343 | 0.516 | 0.333 | 0.306 |
| `tree-1x` | 165 | 87 | 2 | 2 | 42 | 0.300 | 0.455 | 0.321 | 0.290 |
| `tree-3x` | 179 | 72 | 2 | 3 | 47 | 0.340 | 0.474 | 0.952 | 0.864 |
| `tree-10x` | 197 | 56 | 2 | 1 | 50 | 0.378 | 0.484 | 3.485 | 3.153 |
| PPO r1 | 102 | 148 | 2 | 4 | 30 | 0.179 | – | – | – |
| PPO r2 | 89 | 165 | 2 | 0 | 23 | 0.155 | – | – | – |

Both v2 one-move teachers keep more HP than any stage-1 tree, `tree-10x` included, on both panels. Every censored case hit the turn cap (`turn_cap`), none the fight cap:
- `current-v2`, `current-v2-p10` and `current-v2-p15`: one selection case, the same act 4 boss fight at level 0 (see "CPU").
- `tree-1x-v2` on selection: two cases, an act 1 elite and an act 2 monster fight.
- `tree-1x-v2` on boss: three cases, act 1, 3 and 4 bosses.

## Resources

These are signed means over completed fights, per starting max HP. Stage-1 runs have no resource fields.

**Selection panel**

| Teacher | Completed | Censored | HP lost | Potion HP spent | Resource loss | Potions used / win |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `current` | 256 | 0 | 0.403 | 0.221 | 0.624 | 1.373 |
| `current-v1-nopot` | 256 | 0 | 0.412 | 0.216 | 0.628 | 1.348 |
| `current-v2` | 255 | 1 (turn cap) | 0.159 | 0.144 | 0.303 | 0.901 |
| `current-v2-sqrt` | 256 | 0 | 0.169 | 0.139 | 0.309 | 0.850 |
| `current-v2-p10` | 255 | 1 (turn cap) | 0.155 | 0.149 | 0.304 | 0.936 |
| `current-v2-p15` | 255 | 1 (turn cap) | 0.160 | 0.139 | 0.298 | 0.858 |
| `tree-1x-v2` | 254 | 2 (turn cap) | 0.231 | 0.164 | 0.395 | 0.991 |

**Boss panel**

| Teacher | Completed | Censored | HP lost | Potion HP spent | Resource loss | Potions used / win |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `current` | 256 | 0 | 0.538 | 0.237 | 0.775 | 1.523 |
| `current-v1-nopot` | 256 | 0 | 0.532 | 0.233 | 0.765 | 1.523 |
| `current-v2` | 256 | 0 | 0.399 | 0.204 | 0.603 | 1.339 |
| `current-v2-sqrt` | 256 | 0 | 0.404 | 0.204 | 0.608 | 1.333 |
| `tree-1x-v2` | 253 | 3 (turn cap) | 0.448 | 0.214 | 0.661 | 1.434 |

## Ties

The tie share is the share of searched decisions where two or more actions tie exactly for the top value. The columns split it by the tied value. "All tied" means every legal action tied. The breakdown is a one-off aggregation, and the shares match the report's `top_tie_share`.

| Panel | Teacher | Top tie | At +1 | At −1 | Elsewhere | All tied |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selection | `current` | 0.714 | 0.536 | 0.038 | 0.140 | 0.344 |
| Selection | `current-v1-nopot` | 0.716 | 0.545 | 0.042 | 0.129 | 0.337 |
| Selection | `current-v2` | 0.303 | 0.000 | 0.046 | 0.257 | 0.071 |
| Selection | `current-v2-sqrt` | 0.264 | 0.000 | 0.046 | 0.218 | 0.068 |
| Selection | `tree-1x` (stage 1) | 0.656 | 0.470 | 0.125 | 0.062 | 0.443 |
| Selection | `tree-1x-v2` | 0.209 | 0.000 | 0.070 | 0.139 | 0.164 |
| Boss | `current` | 0.606 | 0.343 | 0.132 | 0.131 | 0.295 |
| Boss | `current-v1-nopot` | 0.618 | 0.354 | 0.133 | 0.131 | 0.296 |
| Boss | `current-v2` | 0.353 | 0.000 | 0.140 | 0.213 | 0.182 |
| Boss | `current-v2-sqrt` | 0.348 | 0.000 | 0.141 | 0.208 | 0.176 |
| Boss | `tree-1x` (stage 1) | 0.575 | 0.305 | 0.244 | 0.026 | 0.455 |
| Boss | `tree-1x-v2` | 0.452 | 0.000 | 0.233 | 0.220 | 0.386 |

The report also gives 0.304 for `current-v2-p10` and 0.302 for `current-v2-p15` on selection, and 0.629 and 0.533 for stage-1 `tree-3x` and `tree-10x` there.

**What v2 changes.** v2 removes the ties at +1, which were over half of `current`'s selection-panel decisions. Ties at −1, where all the tied actions' simulations died, stay about where they were. Exact ties elsewhere remain in a quarter of v2's selection decisions. Their cause was not measured.

## Dataset-teacher choice

**The rule.** It was fixed in the plan before the runs.
- **Eligibility.** A candidate is ineligible on a panel if any of these holds:
  - it wins 6 or more fewer cases than the control `current`;
  - a one-sided exact McNemar test finds it worse at p < 0.10, with censored cases counted as not won and escaped cases dropped;
  - it has more than 3 censored cases.
- **Choice.** Among candidates eligible on both panels, the lowest resource loss wins. Resource loss is an unweighted mean over every completed fight of both panels. The two panels share 31 base states, which are counted on each.
- **Ties.** Candidates within 0.01 of the best go to the first in `--candidate` order.

| Candidate | Panel | Wins | Control wins | Deficit | Only control | Only candidate | Escaped dropped | One-sided p | Censored | Ineligible by |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `current-v2` | selection | 233 | 233 | 0 | 7 | 7 | 0 | 0.605 | 1 | – |
| `current-v2` | boss | 180 | 176 | −4 | 7 | 11 | 2 | 0.881 | 0 | – |
| `current-v2-sqrt` | selection | 233 | 233 | 0 | 6 | 6 | 0 | 0.613 | 0 | – |
| `current-v2-sqrt` | boss | 180 | 176 | −4 | 7 | 11 | 2 | 0.881 | 0 | – |
| `tree-1x-v2` | selection | 224 | 233 | 9 | 13 | 4 | 0 | 0.0245 | 2 | win floor, McNemar |
| `tree-1x-v2` | boss | 166 | 176 | 10 | 22 | 12 | 2 | 0.0607 | 3 | win floor, McNemar |

| Candidate | Eligible | Resource loss, both panels | Completed fights |
| --- | --- | ---: | ---: |
| `current-v2` | yes | 0.4536 | 511 |
| `current-v2-sqrt` | yes | 0.4584 | 512 |
| `tree-1x-v2` | no | 0.5278 | 507 |
| `current` (control) | – | 0.6994 | 512 |

**Chosen dataset teacher: `current-v2`.** `current-v2-sqrt` is inside the tie band, so the choice rests on candidate order: `current-v2` was listed first as the cheaper teacher in expected cost. The rule did not fall back to the control.

## Stop rule

| Teacher | Boss-fight decisions with a root | Root attempts | Share | Verdict |
| --- | ---: | ---: | ---: | --- |
| `current-v2` (chosen) | 7,530 | 7,547 | 0.9977 | **CONTINUE** |

## Attribution

All runs here are native. Every comparison below is post hoc and was not pre-registered. The p-values are unadjusted. The pre-registered parts are the choice rule and the stop rule.

**Comparison method.**
- Win pairs use a two-sided exact McNemar test over cases neither side escaped.
- "Shared wins" are the cases both sides won.
- The resource-loss gap is first minus second, with its paired SE.
- "Lower / higher / equal" counts the shared wins where the first side's loss was lower, higher or the same.

| Panel | Comparison | Only first / only second | McNemar p | Shared wins | Resource-loss gap (SE) | of which HP / potions | Lower / higher / equal | Sign-test p |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| Selection | `current-v1-nopot` vs `current` | 5 / 5 | 1 | 228 | +0.003 (0.012) | +0.008 / −0.005 | 59 / 68 / 101 | 0.48 |
| Boss | `current-v1-nopot` vs `current` | 9 / 9 | 1 | 167 | −0.009 (0.016) | −0.004 / −0.005 | 56 / 53 / 58 | 0.85 |
| Selection | `current-v2` vs `current-v1-nopot` | 4 / 4 | 1 | 229 | −0.356 (0.020) | −0.275 / −0.081 | 202 / 13 / 14 | 9.4e-45 |
| Boss | `current-v2` vs `current-v1-nopot` | 10 / 6 | 0.45 | 170 | −0.226 (0.021) | −0.184 / −0.042 | 126 / 14 / 30 | 1.0e-23 |
| Selection | `current-v2-sqrt` vs `current-v2` | 2 / 2 | 1 | 231 | −0.001 (0.004) | +0.004 / −0.005 | 23 / 23 / 185 | 1 |
| Boss | `current-v2-sqrt` vs `current-v2` | 1 / 1 | 1 | 179 | +0.008 (0.006) | +0.008 / +0.000 | 19 / 18 / 142 | 1 |
| Selection | `current-v2-p10` vs `current-v2` | 1 / 0 | 1 | 233 | +0.001 (0.003) | −0.003 / +0.005 | 5 / 7 / 221 | 0.77 |
| Selection | `current-v2-p15` vs `current-v2` | 0 / 0 | 1 | 233 | −0.005 (0.002) | +0.001 / −0.006 | 15 / 4 / 214 | 0.019 |
| Selection | `tree-1x-v2` vs `current-v2` | 4 / 13 | 0.049 | 220 | +0.087 (0.012) | +0.067 / +0.020 | 43 / 135 / 42 | 2.9e-12 |
| Boss | `tree-1x-v2` vs `current-v2` | 7 / 21 | 0.013 | 159 | +0.059 (0.019) | +0.045 / +0.014 | 53 / 88 / 18 | 0.0040 |

**Playout policy: no change.** Keeping potions out of the playouts changes nothing measurable on either panel. `current-v1-nopot` matches `current` exactly in wins, and the shared-win resource gaps are within one SE. Potion use per won fight is 1.348 against 1.373 on selection, and 1.523 against 1.523 on boss.

The v1 teacher's potion use therefore does not come from its playouts. It comes from the teacher's own root choices under tied values. Under `flat-v1`, at 54% of selection decisions and 34% of boss decisions every tied action won all its simulations, and `current` breaks ties at random. A potion action that ties at +1 is picked as often as any other tied action. The link from ties to potion use is an inference. The decision rows do not record action verbs, so the share of tied choices that were potions was not measured.

**Value rule: the whole effect.** `current-v2` against `current-v1-nopot` isolates the value rule, and it carries the whole change.
- **Selection.** On the 229 shared wins, resource loss falls by 0.356 of max HP: 0.275 from HP and 0.081 from potions. 202 of those fights improve and 13 get worse.
- **Boss.** It falls by 0.226: 0.184 from HP and 0.042 from potions.
- **Wins.** They do not change: 4 against 4 discordant on selection, and 10 against 6 on boss (p 0.45).

**Curve: no real difference.** Linear and sqrt have the same wins and differ by under 0.01 of resource loss on each panel. Most shared wins have identical resource loss: 185 of 231 on selection and 142 of 179 on boss. The sqrt curve values HP more steeply at low HP. On these panels it changed few outcomes.

**Potion price: flat.** On selection, prices of 10, 12 and 15 HP give the following.

| Price | Wins | Resource loss | Potions used / win |
| ---: | ---: | ---: | ---: |
| 10 | 234 | 0.304 | 0.936 |
| 12 | 233 | 0.303 | 0.901 |
| 15 | 233 | 0.298 | 0.858 |

Potion use drifts down with price, as it should. At 15 HP against 12, the shared-win gap is −0.005 (sign-test p 0.019, unadjusted). That is half the choice rule's tie band and immaterial for the choice.

**`tree-1x-v2`: ineligible.** It fails the win floor and the McNemar test on both panels. On selection it wins 9 fewer cases than `current`, with 13 against 4 discordant and one-sided p 0.025. On boss it wins 10 fewer, with 22 against 12 and p 0.061. Under v2 it also keeps less than `current-v2`: on shared wins it loses 0.087 and 0.059 more of max HP. In stage 1, by contrast, the trees kept more HP than `current`.

These hypotheses are post hoc and untested:
- **The boss deficit may be the 1x tree's, not v2's.** The stage-1 `tree-1x` won 165 boss cases, and `tree-1x-v2` won 166 (19 against 18 discordant). Under the same eligibility test, stage-1 `tree-1x` would also fail the boss win floor (11 below `current`, one-sided p 0.054). On selection, `tree-1x-v2` won 7 fewer cases than `tree-1x` (12 against 5 discordant, two-sided p 0.14), which may be noise.
- **The tree's v1 HP advantage may not carry over.** Under `flat-v1`, `current` ignored HP whenever its playouts won. The tree's deeper look may have recovered some of that. Under v2 the one-move search sees HP directly, so the tree's edge may vanish while its smaller root sample stays. It used 140 simulations per searched decision on selection, against 221 for `current-v2`, though a tree simulation is not the same unit as a one-move sample.
- **The tree constants were tuned for v1.** Only `ucb_c` was re-tuned for v2, on 100 training fights, where the three values were 3 wins apart. `min_root_visits`, worlds, depth and widening were carried from the v1.2.0 calibration.
- **Hopeless boss states stay tied.** `tree-1x-v2`'s boss ties at −1 (0.233 of decisions) are as common as stage-1 `tree-1x`'s (0.244). The tree breaks exact ties by the lowest action index.

## CPU

**Selection: one fight explains `current-v2`'s cost.** `current-v2` costs 0.848 CPU s per searched decision on selection, against 0.677 for `current`. The stage-1 tail fight explains the difference. That case is an act 4 boss fight at level 0. It took 2,326 CPU s in stage 1, 54% of `current`'s search CPU on the panel. The table's 2,535 s is the same fight's search CPU in the v1.2.5 re-run. Its decisions are identical; the re-run ran in a 2-CPU container alongside eleven other runs, so the per-core CPU time is higher.

| Teacher | Tail fight outcome | Decisions | Search CPU s | Share of the run's search CPU | CPU s / searched decision without it |
| --- | --- | ---: | ---: | ---: | ---: |
| `current` | victory | 437 | 2,535 | 55% | 0.329 |
| `current-v1-nopot` | victory | 437 | 2,487 | 56% | 0.299 |
| `current-v2` | censored, turn cap | 531 | 3,399 | 66% | 0.320 |
| `current-v2-p10` | censored, turn cap | 531 | 3,404 | 66% | 0.316 |
| `current-v2-p15` | censored, turn cap | 531 | 3,394 | 65% | 0.323 |
| `current-v2-sqrt` | victory | 171 | 517 | 22% | 0.336 |
| `tree-1x-v2` | victory | 55 | 42 | 2% | 0.301 |

- **Without the fight**, the one-move teachers cost 0.30–0.34 CPU s per searched decision, v1 and v2 alike.
- **`current-v2`'s one censor is this fight.** It reached the 500-action cap within one public turn at decision 531, after more search CPU than `current` spent winning it.
- **The sqrt teacher won it in 171 decisions.** Its selection figure, 0.416, is low because of that, not because sqrt is cheaper.
- **On the boss panel there is no such fight.** `current-v2` costs 0.470 there, against `current`'s 0.490.

Why the linear v2 teachers stalled in one turn of that fight was not examined.

**No-potion playouts did not run longer.** The plan expected playouts that keep their potions to run longer. The measurements do not show that. The first decision of every fight searches the same state with the same research seed under every one-move teacher. At those 256 matched roots per panel, CPU per simulation was lower for the no-potion playouts:
- `current-v1-nopot`: 0.89 of `current`'s on selection and 0.94 on boss;
- `current-v2`: 0.89 and 0.93.

`current-v2` against `current-v1-nopot` is 0.99 on both. Over whole runs, `current-v1-nopot` also cost less per simulation than `current` (3.03 against 3.17 ms on selection, 2.17 against 2.35 ms on boss).

There are two caveats. `current` ran two fights in a 2-CPU container and the others one fight in a 1-CPU container. And playout lengths are not recorded, so this is CPU, not steps.

## Next

Per the plan, Task 8 Step 4 stops here and reports the choice. The next plan covers parts 3 to 5 of the spec around `current-v2` as the dataset teacher:
- the dataset;
- the students;
- the PPO arm with the bounded decline diagnostic;
- gates G2 to G4.

## Open items

- **Permanent-card effects.** The Gym deferred item 3 of change request `spire-9wa`: permanent-card `misc` and upgrade deltas, for scoring Ritual Dagger, Genetic Algorithm and Lesson Learned. v2 does not score these effects; it scores Feed only through max HP. File a new spire-learn-labelled bead with the wanted schema only when Learn is ready to score them.
- **Tree search under v2.** A later look is optional. The untested hypotheses under "Attribution" name what it could examine: the budget, the carried tree constants and the tie-break.
- **`current-v2`'s stalled turn.** The selection-panel censor above was not examined. It does not affect eligibility, which allows 3 censored cases per panel.
