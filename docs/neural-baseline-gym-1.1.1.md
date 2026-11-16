# Fresh-weights baseline on Gym v1.1.1

Run on **2026-11-16**, 19:42–20:31 UTC. This is one fresh-weights mixed-stream
Ironclad combat baseline on Spire Gym release v1.1.1. It used the validation fast
path, and its plan is the [8726d467 baseline](neural-baseline-8726d467.md)'s,
re-bound to the new build. It trained to its learned-decision target, then
evaluated four checkpoints on the development panel. The final policy won
**39 of 64** paired cases; the same untrained weights won 18. Read the
[limits](#limits) before comparing these numbers with anything else.

## What ran

| Item | Value |
| --- | --- |
| Gym build | `v1.1.1`, `e3c13836bd1458be8a87b93160f5c350f78fcd80`, `rules_epoch` `gym-1.1.1` |
| Learner commit | `33c9f72`, clean at start and end (`learner_changed: false`) |
| Plan | `configs/neural/baseline-gym-1.1.1/plan.json` (`baseline-gym-1.1.1-20261116`) |
| Template | `configs/neural/baseline-gym-1.1.1/template.toml`, prepared config SHA-256 `3d3f0505…` |
| Stream | `configs/neural/curriculum-v1/recipe.json`, arm `mixed`, train split, shards 0–98 |
| Runner | `scripts/neural/runtime_container.py` with entry `baseline_entry.py`; offline, read-only root, 4 CPUs, 16 GiB |
| Device | RTX 5090 (`GPU-c05e0ce3…`), training and evaluation on `cuda:0`; co-resident GPU services left running by user decision |
| Rollout | 16 environments × 32 steps; 8 PPO updates per 4,096-decision segment |
| Budgets | 405,504 learned decisions; training wall 12,600 s, total 14,400 s, runner timeout 15,600 s; 12 GiB output cap |
| Evaluation | Development panel `435a236e…`, 64 cases, deterministic actions, 256 actions or 30 s per case |

Plan, panel, budgets and model settings are identical to the 8726d467 baseline.
Only the Gym build, its rules epoch and the learner code differ. The learner code
now includes the validation fast path, which accepts exactly the frames the
earlier validator accepted.

## Outcome

| Measure | Value |
| --- | --- |
| Result | `complete`, stop reason `target_reached` |
| Learned decisions | 405,504 across 99 segments, plus a 512-decision qualification |
| Training wall | 2,798 s, about 145 learned decisions/s (8726d467 baseline: 6,989 s, about 58/s) |
| Evaluation wall | 122 s for four panels |
| Faults | None: no segment fault, evaluation fault, censored case or failed report |
| Runner | `manifest-final.json` exit status 0, not timed out, `learner_changed: false`; no `manifest-post.json` |

The training wall includes per-segment setup. The throughput profile measured
it at about 7 s per segment; this run's segments averaged 28.3 s, against the
profile's 23.7 s. That setup cost is the next throughput target ([plan](neural-throughput-plan-20261116.md), fix 3).
Outputs stay outside Git, under
`artifacts/development/neural-combat/baseline-gym-1.1.1-20261116/`.

### Paired fresh vs final

| | Final wins | Final loses |
| --- | ---: | ---: |
| **Fresh wins** | 17 | 1 |
| **Fresh loses** | 22 | 24 |

The final policy won 22 cases that the fresh weights lost. The fresh weights won
1 that the final policy lost. The exact two-sided McNemar p-value is 5.7 × 10⁻⁶.
All 64 cases were paired; none were censored or excluded.

### Checkpoints along the run

| Checkpoint | Learned decisions | Panel wins | Mean HP retention |
| --- | ---: | ---: | ---: |
| fresh-initial | 0 | 18/64 | 0.144 |
| near-135168 | 135,168 | 34/64 | 0.271 |
| near-270336 | 270,336 | 36/64 | 0.281 |
| final | 405,504 | **39/64** | **0.305** |

### By panel family and cohort

| Stratum | fresh | 135k | 270k | final |
| --- | ---: | ---: | ---: | ---: |
| family `run_like` | 16/48 | 26/48 | 29/48 | 31/48 |
| family `reserved_pair` | 2/16 | 8/16 | 7/16 | 8/16 |
| cohort `familiar_new_seed` | 9/16 | 10/16 | 11/16 | 12/16 |
| cohort `novel_deck` | 5/16 | 11/16 | 12/16 | 13/16 |
| cohort `resource_later_tier` | 2/16 | 5/16 | 6/16 | 6/16 |

By encounter tier, the final policy won:
- all 16 act 1 monsters, 5 of 5 act 1 elites, 7 of 8 act 2 monsters and 3 of 4
  act 2 elites;
- 2 of 17 boss cases (one each in acts 1 and 2), and 0 of 3 act 3 elites.

Bosses and late elites remain largely unsolved.

## Limits

- One replicate, one seed schedule and a 64-case development panel. The protected
  final split was not run.
- Simulator-only standalone Ironclad combat. It makes no claim about full Runs or
  transfer to the original game. RNG-001 and TIME-001 qualifications apply.
- Panel scores are bound to `rules_epoch` `gym-1.1.1`. v1.1.1 changes gameplay in
  twelve places, so these scores are **not comparable** with the 8726d467
  baseline's 37/64 or with any earlier build. The matching fresh-initial 18/64
  reflects the same initial weights and seeds, not equivalence of the builds.
- One replicate cannot separate the two baselines' final scores (39 vs 37) from
  noise, even setting the rule change aside.
- The panel ran on `cuda:0`, like the 8726d467 baseline and unlike the earlier
  CPU campaign panels.
