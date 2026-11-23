# Standard combat throughput on Gym v1.2.0: dense and compact views

Measured 2026-11-23 on the Standard stream (`configs/neural/baalorlord-standard-gym-1.2.0/recipe.json`,
`*-standard-v1` learner identity, Gym `4d198064b3b568b6f0d08d6e3937d236fbc48e9b`, rules epoch
`gym-1.2.0`). Simulator timings only; not a training or panel result.

## Result

Decisions per second for the three fresh-weight segments (synchronized, synchronized,
unsynchronized); the headline is the last, unsynchronized segment.

| Rows | View | Segments (dec/s) | Headline | vs dense | Segment wall | Decisions |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | dense | 156.3, 157.3, 163.1 | **163.1** | 1.00 | 25.1 s | 4,096 |
| 16 | compact `standard-combat-compact-v1` | 208.0, 216.9, 222.0 | **222.0** | 1.36 | 18.5 s | 4,096 |
| 16 | compact `standard-combat-compact-play-v1` (measurement only) | 231.6, 235.1, 243.2 | **243.2** | 1.49 | 16.8 s | 4,096 |
| 64 | dense | 174.7, 174.1, 172.2 | **172.2** | 1.00 | 95.2 s | 16,384 |
| 64 | compact `standard-combat-compact-v1` | 253.2, 256.3, 254.5 | **254.5** | 1.48 | 64.4 s | 16,384 |
| 64 | compact `standard-combat-compact-play-v1` (measurement only) | 284.6, 292.5, 290.5 | **290.5** | 1.69 | 56.4 s | 16,384 |

The dense figures match the v1.1.1 profile (`docs/neural-throughput-standard.md`: 165.1 and
174.3), so the release itself did not change throughput. The compact view, which training and
panels use, is about 36% faster at 16 rows and 48% faster at 64 rows. Widening to 64 rows now
pays more (compact: +15% over 16 rows, dense: +6%).

## Phase split (unsynchronized segment, share of segment wall time)

| Phase (spans) | 16 dense | 16 compact | 16 play | 64 dense | 64 compact | 64 play |
| --- | --- | --- | --- | --- | --- | --- |
| Env step (native step/reset public export, driver step/reset) | 6.8% | 6.2% | 6.3% | 9.9% | 10.0% | 10.5% |
| Snapshot/validation (`snapshot_validation`, `frame_hash`, artifact events) | 24.1% | 21.7% | 20.1% | 32.7% | 34.4% | 33.3% |
| Features (`features_transfer`, `encode_frames`) | 6.3% | 4.6% | 4.4% | 6.2% | 3.9% | 3.4% |
| Policy forward (`policy_act`, `bootstrap_values`) | 19.9% | 26.0% | 28.3% | 8.1% | 8.5% | 9.5% |
| PPO update (forward/loss, backward, clipping, behaviour validation, Adam) | 37.6% | 34.8% | 33.8% | 40.2% | 40.3% | 40.0% |
| Other (checkpoint save, collection, unattributed) | 5.3% | 6.6% | 7.1% | 2.9% | 2.9% | 3.2% |

Shares hide that every phase except policy inference shrank in absolute time. Exclusive seconds
per segment for the larger spans:

| Span | 16 dense | 16 compact | 16 play | 64 dense | 64 compact | 64 play |
| --- | --- | --- | --- | --- | --- | --- |
| `native_step_public_export` | 1.13 | 0.59 | 0.55 | 5.00 | 2.30 | 2.22 |
| `snapshot_validation` | 3.62 | 2.94 | 2.58 | 21.44 | 17.75 | 15.51 |
| `frame_hash` | 2.34 | 1.01 | 0.74 | 9.38 | 4.12 | 3.00 |
| `features_transfer` | 1.58 | 0.85 | 0.73 | 5.92 | 2.52 | 1.93 |
| `policy_act` | 2.53 | 2.45 | 2.44 | 4.09 | 2.87 | 2.82 |
| `bootstrap_values` | 2.48 | 2.35 | 2.33 | 3.63 | 2.59 | 2.55 |
| `ppo_forward_loss` | 3.76 | 2.58 | 2.57 | 15.25 | 10.47 | 10.29 |
| `ppo_backward` | 3.66 | 2.39 | 1.98 | 14.76 | 9.61 | 7.72 |
| `ppo_validate_behavior` | 1.04 | 0.50 | 0.18 | 4.15 | 2.00 | 0.72 |

The compact view roughly halves public export, frame hashing and feature transfer, and cuts the
PPO forward/backward by about a third (fewer action, power, relic and offer rows). Snapshot
validation falls least (about 17-19%): most of its cost is the per-row check of occupied rows,
which the layout does not change; it is now the largest single span at 64 rows (28%). At 16 rows
policy inference is fixed per-call overhead and does not shrink.

## Why training uses the bounds preset

`standard-combat-compact-v1` is sized to the entry bounds, so every valid entry fits and a
capacity fault would mean a Gym or entry defect, not a long fight. The play preset is a further
10-14% faster, but its smaller capacities (96 permanent cards, 832 combat cards, 64 powers,
32 relics, 192 actions) can overflow in legitimate late-game states; an overflow is a typed Gym
`Capacity` fault that Learn surfaces as a `DriverFault` (an operational failure of the segment,
never a loss or truncation), so a stream on it could stop mid-run. Shard and panel entry
validation (`standard_catalog.bind_gym`) builds a dense environment and never resets, so it
cannot catch a compact overflow in advance; with the play preset the first sign would be a
fault during training or a panel. The play preset was measured only: baseline plans refuse it
(`baseline_entry._refuse_overflowing_preset`), and no Learn config uses it for training or panels.

## Method

- Entry `scripts/neural/profile_entry.py`: three fresh-weight stream segments per plan
  (synchronized, synchronized, unsynchronized), each 8 updates of one 32-step rollout.
- Plans/templates: `configs/neural/profile-standard-1.2.0/{plan,template}-{16,64}-{dense,compact,play}.*`;
  model, PPO, seeds and segment allowance as `profile-standard-1.1.1`, recipe
  `baalorlord-standard-gym-1.2.0`. The three views of each row count differ only in the
  `[environment]` layout keys and the identity's raw-mode mapping.
- Offline runtime container (Gym v1.2.0 profile), `cuda:0` on the shared GPU, 4 CPUs, 16 GiB.
  All six runs exit 0 with `learner_changed` false, at learner commit `0bba825`. Local evidence:
  `artifacts/development/neural-combat/t3-profile-{16,64}-{dense,compact,play}/` (not committed).
- The `gpu_wait` gate (min free 13,958,643,712 B, poll 30 s, cap 21,600 s) was on; the GPU had
  about 30 GiB free throughout, so no wait occurred in any run.
- The gated test `test_compact_and_dense_give_identical_policy_outputs` found byte-equal encoded
  features, coverage, critic values, logits and sampled and greedy actions on CPU for 2,400
  decisions from real v1.2.0 entries of all four characters (3,128 frames, 104 terminal) and 480
  decisions under the play preset at its capacity edge. On that sample the speed difference is
  the data path only, not a different model.

## Caveats

- One replicate each; segment-to-segment spread is about 1-4%.
- Every segment sees only the first shards of the stream (fuzz level 0), so the combat mix
  differs from late training; later, fuller states narrow the compact view's margin where rows
  are occupied, and throughput will drift with the mix.
- The runs ran back to back on a shared host and GPU; other load was light but not controlled.
- Simulator throughput only; no original-game transfer claim. RNG-001 and TIME-001
  qualifications apply.
