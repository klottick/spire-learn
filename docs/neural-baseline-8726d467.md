# Fresh-weights baseline on Gym 8726d467

Run **2026-11-16**, 04:08–06:08 UTC. This is one fresh-weights mixed-stream Ironclad
combat baseline on Spire Gym 1.0.0 native build `8726d467`. It trained to its
learned-decision target, then evaluated four checkpoints on the development panel.
The final policy won **37 of 64** paired cases. The same untrained weights won 18.
Read the [limits](#limits) before comparing these numbers with anything else.

## What ran

| Item | Value |
| --- | --- |
| Gym build | `8726d467ad8579730ddf4fa236b24aeb2206b71a`, `rules_epoch` `gym-1.0.0+8726d467` |
| Learner commit | `1e6e0f7`, clean tree |
| Plan | `configs/neural/baseline-8726d467/plan.json` (`baseline-8726d467-20261115`) |
| Template | `configs/neural/baseline-8726d467/template.toml`, prepared config SHA-256 `d3a4e35e…` |
| Stream | `configs/neural/curriculum-v1/recipe.json`, arm `mixed`, train split, shards 0–98 |
| Runner | `scripts/neural/runtime_container.py` with entry `baseline_entry.py`; offline, read-only root, 4 CPUs, 16 GiB |
| Runtime image | `spire-learn/task8-python:cuda-setup-20261028-a` |
| Device | RTX 5090 (`GPU-c05e0ce3…`), training and evaluation on `cuda:0` |
| Rollout | 16 environments × 32 steps; 8 PPO updates per 4,096-decision segment |
| Budgets | 405,504 learned decisions; training wall 12,600 s, total 14,400 s, runner timeout 15,600 s; 12 GiB output cap |
| Evaluation | Development panel `435a236e…`, 64 cases, deterministic actions, 256 actions or 30 s per case |

The co-resident GPU services (ollama, comfyui, immich machine learning, kokoro) were
left running by user decision. No VRAM fault occurred. The baseline peaked at about
6.3 GiB.

## Outcome

| Measure | Value |
| --- | --- |
| Result | `complete`, stop reason `target_reached` |
| Learned decisions | 405,504 across 99 segments, plus a 512-decision qualification |
| Training wall | 6,989 s (about 58 learned decisions/s) |
| Evaluation wall | 181 s for four panels |
| Faults | None: no segment fault, evaluation fault, censored case or failed report |
| Runner | `manifest-final.json` exit status 0, container removed, not timed out; no `manifest-post.json` |

Outputs stay outside Git under
`artifacts/development/neural-combat/baseline-8726d467-20261115/`. The key files are
`summary.json`, `progress.jsonl`, `development-panel.json` and the run manifests.

### Paired fresh vs final

| | Final wins | Final loses |
| --- | ---: | ---: |
| **Fresh wins** | 16 | 2 |
| **Fresh loses** | 21 | 25 |

The final policy won 21 cases that the fresh weights lost. The fresh weights won 2
that the final policy lost. The exact two-sided McNemar p-value is 6.6 × 10⁻⁵. All
64 cases were paired; none were censored or excluded.

### Checkpoints along the run

| Checkpoint | Learned decisions | Panel wins | Mean HP retention |
| --- | ---: | ---: | ---: |
| fresh-initial | 0 | 18/64 | 0.144 |
| near-135168 | 135,168 | 33/64 | 0.282 |
| near-270336 | 270,336 | **40/64** | **0.315** |
| final | 405,504 | 37/64 | 0.296 |

The 270k checkpoint scored above the final one. One replicate and 64 cases cannot
tell a plateau from noise. The three-win gap is not evidence of regression.

### By panel family and cohort

| Stratum | fresh | 135k | 270k | final |
| --- | ---: | ---: | ---: | ---: |
| family `run_like` | 16/48 | 24/48 | 31/48 | 28/48 |
| family `reserved_pair` | 2/16 | 9/16 | 9/16 | 9/16 |
| cohort `familiar_new_seed` | 9/16 | 11/16 | 11/16 | 10/16 |
| cohort `novel_deck` | 5/16 | 9/16 | 14/16 | 12/16 |
| cohort `resource_later_tier` | 2/16 | 4/16 | 6/16 | 6/16 |

By encounter tier, the final policy won 15 of 16 act 1 monsters, 5 of 5 act 1
elites and 8 of 8 act 2 monsters. It won 1 of 17 boss cases and 0 of 3 act 3 elites.
Bosses and late elites remain unsolved at every checkpoint.

## Limits

- One replicate, one seed schedule and a 64-case development panel. The protected
  final split was not run.
- Simulator-only standalone Ironclad combat. It makes no claim about full Runs or
  transfer to the original game. RNG-001 and TIME-001 qualifications apply.
- Panel scores are bound to `rules_epoch` `gym-1.0.0+8726d467`. They are not
  comparable with panel scores from earlier builds, including the 2026-11-03
  campaign's 41/64 on build `afb51f5`.
- The panel ran on `cuda:0`. The earlier campaign panels ran on CPU. Deterministic
  evaluation should not depend on the device, but this was not cross-checked here.
- Throughput was about 58 decisions/s with GPU utilization near zero. That is far
  below the simulator's native rate. A profiling pass will follow (see
  `scripts/neural/profile_entry.py`).

## Readiness checks before the run

The gated tensor tests ran on this build through the same runner. They passed 34
of 34 on CPU (ledger reservation `gym-8726d467-cpu-20261116-a`) and 34 of 34 on GPU
(`gym-8726d467-gpu-20261116-a`).
