# Full-run panel and gate R1

This document reports Task 5 of the distillation parts 3–5 plan (`docs/superpowers/plans/2026-11-27-distillation-parts-3-5.md`), gate R1. The plan's delta 13 adds the gate. Before 1.5M labels are built on `current-v2`'s resource prices, `current-v2` and the v1 teacher `current` each play the same panel of full Ironclad A20 runs, with the same pre-registered non-combat heuristic, to check that the prices do no large harm over a whole run. The harness is Task 4 (`scripts/neural/full_run_entry.py`). The teachers are described in `docs/neural-teacher-value-v2.md`.

## Provenance

- **Gym.** Release v1.2.5, revision `731fca0e0341d80e7cf08a8be7a03be1fa32c3f1`, a clean release build, inside rules epoch `gym-1.2.4`. Profile: `configs/neural/gym-1.2.5/runtime.json`.
- **Learner.** Every panel run used the clean tracked tree at `2ee5d779745b18741b3a0684c462ad8698204a5c`, unchanged before and after.
- **Panel.**
  - 64 Ironclad A20 runs from the `selection` partition. The seeds are `environment_seed_for("selection", replica=0, ordinal=2**34 + i)` for `i < 64`. A unit test (`test_run_panel_seeds_disjoint_from_panel_ordinals`) asserts that they are disjoint from the shard and combat-panel ordinals.
  - Runs use experienced creation and a full Neow, and keys are skipped.
  - Panel identity: SHA-256 `098eb73c9968a36db773b76e88027ba2400c446b94c32bf0ea9e92fda35a2b58`.
- **Caps.** 500 actions in one public turn, 1,024 actions in one fight, and 10,000 decisions in one run. A cap censors the run.
- **Non-combat player.** `noncombat-heuristic-v1`, a deterministic rule set fixed before any panel run. The card priority list is `configs/neural/full-run/ironclad-priority-v1.json` (SHA-256 `9dd28b208c16e6fb05daced54ca264d287c9437c95d8aad4500e861448f36548`), and every run recorded it as pre-registered.
- **Combat players.**
  - `teacher:current`: v1 values, from `configs/neural/search-teacher-gym-1.2.4/teachers.json`, with teacher config `2f29eef0…`.
  - `teacher:current-v2`: `resource-v2` values, from `teachers-v2.json`, with teacher config `df81b2d7…`.
  - For reference only, `checkpoint:` r1: the PPO baseline r1's final checkpoint, played deterministically on the bridged compact frame.
  - Both teachers are one-move searches with 32 native playouts per legal action. They differ in three ways, not only in the value rule:
    - `current` uses the `flat-v1` value rule, the `uniform-v1` playout policy and the v1 playout rules, and it is calibrated;
    - `current-v2` uses `resource-v2`, the `no-potions-v2` playout policy and the tiered playout rules, and it was loaded with `allow_uncalibrated`.
  - R1 therefore compares the two teacher configurations as a whole.
- **Seeds.** Research seed 2026112703 for every player. Decision `d` of run `i` searches with `H(2026112703, i, d)`, so the results do not depend on chunking.
- **Budget.** 8 CPUs per container, one container per chunk of 8 runs, with up to two players' chains running at once.
- **Scope.** These are simulator results only, under RNG-001 and TIME-001. No original-game transfer is claimed, and only aggregates are published.

## Results in short

- **R1 passes.** The `current-v2` teacher is not worse than `current`; over these runs it is clearly better. Its rank key (floors reached, then victory) is higher in 42 of the 64 paired runs and lower in 2, with 20 ties. The exact two-sided sign test gives p = 1.1 × 10⁻¹⁰.
- **Both teachers die early.** With this heuristic, 59 of `current-v2`'s runs and 63 of `current`'s end in Act 1. No run of any player won, and no run was capped.
- **`current` loses more HP per fight.** In Act 1 fights other than the boss, `current` loses about 16.2 HP per fight and `current-v2` about 8.3. It reaches the Act 1 boss in 8 runs against 28, with 0.485 of max HP against 0.720. This is consistent with the v1 value rule's indifference to HP (`docs/neural-teacher-value-v2.md`, "Careless HP and potion use"). The runs follow different paths after their first fight, so this is a correlation, not an isolated cause.

## Gate R1

The plan fixed this rule before the runs. Each run's rank key is its floors reached, then victory. A capped run would keep the floors it reached at the cap and be flagged.

The primary test is an exact two-sided sign test on the paired keys, at α 0.05. Labelling stops for a user ruling if `current-v2` is significantly worse, or if it has at least 3 more capped runs than `current`.

| Pairs | v2 higher | v2 lower | Ties | Sign test p | Capped, v1 / v2 | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 64 | 42 | 2 | 20 | 1.1 × 10⁻¹⁰ | 0 / 0 | **PASS** |

The floor difference (`current-v2` minus `current`), with bootstrap 95% CIs from 10,000 resamples with seed 20261127:

| Statistic | Value | 95% CI |
| --- | ---: | --- |
| Mean | +4.09 | 2.84 to 5.44 |
| Median | +2.0 | 1.0 to 4.5 |

A pass is defined as "no large harm detected". The plan sized the test to detect a directional share of about 64% over 64 decisive pairs. Here 20 pairs tie, leaving 44 decisive pairs, and a significant result at α 0.05 then needs at least 29 of them (66%) to go one way. In this run `current-v2` is significantly better.

## Per player

| Player | Mean floors | Median | Min–max | Deaths, act 1 / 2 | Act 1 boss entries | Act 1 boss clears | HP at boss entry | Potions per run | Refused roots | Fallbacks | Censored |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current` | 7.81 | 7 | 4–20 | 63 / 1 | 8 | 1 | 0.485 | 1.30 | 0 | 0 | 0 |
| `current-v2` | 11.91 | 10 | 6–28 | 59 / 5 | 28 | 5 | 0.720 | 2.36 | 0 | 0 | 0 |
| PPO r1 (reference) | 7.23 | 6 | 3–16 | 64 / 0 | 4 | 0 | 0.530 | 1.78 | – | 0 | 0 |

HP at boss entry is the mean of current HP over max HP at the start of the Act 1 boss fight, over 8, 28 and 4 entries. No run reached a later boss. Potions per run counts potions used in combat. The heuristic has no rule that uses or discards a potion outside combat, and none fired. Search roots were never refused, and the heuristic never reached its counted fallback.

## CPU

Run CPU (seconds) over the 64 runs:

| Player | Total | p99 = max |
| --- | ---: | ---: |
| `current` | 443 | 18.8 |
| `current-v2` | 868 | 31.2 |
| PPO r1 | 108 | 2.4 |

The plan estimated 20–80 core-hours on the assumption that runs reach late bosses. Because runs end in Act 1, R1 cost under half a core-hour in total.

## What this does and does not show

- R1 was built to catch one risk: that `current-v2`'s configuration, chosen on single fights, would hurt whole runs, for example by hoarding HP or potions at the cost of later survival. Within Act 1 there is no sign of that, and `current-v2` uses more potions per run, not fewer. Only 5 runs reached Act 2, so later-act effects are untested. On this result, labelling proceeds with `current-v2` (Task 6).
- The runs are short, so R1 tests Act 1 play almost entirely. Acts 2 and 3 are barely represented (5 `current-v2` runs reached Act 2). The heuristic's later-act rules are covered only by unit tests.
- These runs may be limited as much by the decks and paths the heuristic produces as by combat search; R1 does not separate the two. The Standard combat panels start from recorded states of an expert player's runs (`docs/neural-standard-baseline-gym-1.2.4.md`), so they measure combat with the decks that player built. A full-run learner will likely need a much stronger non-combat policy than `noncombat-heuristic-v1`.
- The PPO r1 checkpoint is weaker than both teachers over runs, which agrees with its combat-panel results. It is a reference only and takes no part in the gate.
