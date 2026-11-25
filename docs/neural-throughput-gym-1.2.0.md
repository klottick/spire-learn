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

## Padding trim, sampled validation and workers (stage 1)

Measured 2026-11-25 at learner commit `8861836` using the same Gym v1.2.0
Standard compact preset, recipe and offline CUDA runtime. Each plan started
with fresh weights and resumed across three early shards. The first two
segments synchronized CUDA for timing, and the last ran unsynchronized.
All ten profiles exited 0 with three segments and `learner_changed` false.
The `gpu_wait` gate was active and recorded zero waits. Local evidence is
under `artifacts/development/neural-combat/s1t4-profile-{16,64}-{compact,compact-fast,compact-w8,compact-fast-w8,compact-fast-inexact}/`.

The `fast` plans enable exact padding trim and training validation with stride
16. The `w8` plans use eight Gym batch workers, with the same four-CPU container
limit across all plans. `fast-inexact` uses one worker and enables experimental
trim compute in addition to trim and sampled validation.

### Throughput

Rates were checked by dividing learned decisions by segment wall time. The
table uses the profiler's rates, calculated before its wall times are rounded
to milliseconds. Speed-ups use the unsynchronized compact control at the same
row count. Spread is the fastest segment divided by the slowest, minus one.

| Rows | Variant | Segments (dec/s) | Headline | vs compact | Segment wall | Decisions | Spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | compact | 209.4, 221.5, 233.1 | **233.1** | 1.00 | 17.6 s | 4,096 | 11.3% |
| 16 | compact-fast | 247.9, 233.9, 218.1 | **218.1** | 0.94 | 18.8 s | 4,096 | 13.7% |
| 16 | compact-w8 | 191.2, 196.1, 228.5 | **228.5** | 0.98 | 17.9 s | 4,096 | 19.5% |
| 16 | compact-fast-w8 | 243.0, 246.8, 228.6 | **228.6** | 0.98 | 17.9 s | 4,096 | 8.0% |
| 16 | compact-fast-inexact | 222.6, 229.8, 231.7 | **231.7** | 0.99 | 17.7 s | 4,096 | 4.1% |
| 64 | compact | 253.4, 271.7, 272.4 | **272.4** | 1.00 | 60.1 s | 16,384 | 7.5% |
| 64 | compact-fast | 315.5, 281.5, 311.7 | **311.7** | 1.14 | 52.6 s | 16,384 | 12.0% |
| 64 | compact-w8 | 271.8, 262.6, 273.5 | **273.5** | 1.00 | 59.9 s | 16,384 | 4.2% |
| 64 | compact-fast-w8 | 331.8, 309.6, 324.3 | **324.3** | 1.19 | 50.5 s | 16,384 | 7.2% |
| 64 | compact-fast-inexact | 337.5, 347.1, 350.0 | **350.0** | 1.28 | 46.8 s | 16,384 | 3.7% |

At 16 rows the headlines range from 218.1 to 233.1 decisions/s, with variants
within 6.4% of compact. Each plan's segment spread is 4.1–19.5%, and the runs
mix timing modes and shards. The 16-row differences are within observed noise.
At 64 rows the spread is 3.7–12.0%. The exact fast plans gained 14% and 19%
over compact in their unsynchronized segments, and the inexact plan gained
28%. Eight workers alone changed the 64-row headline by about 0.4%.

### Phase split (unsynchronized segment, share of segment wall time)

The phase groups sum exclusive span times. Other includes checkpoint save,
collection, and the remainder of segment wall time.

| Phase (spans) | 16 compact | 16 fast | 16 w8 | 16 fast-w8 | 16 fast-inexact | 64 compact | 64 fast | 64 w8 | 64 fast-w8 | 64 fast-inexact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Env step (native step/reset public export, driver step/reset) | 5.9% | 7.7% | 4.4% | 5.3% | 8.2% | 10.7% | 12.3% | 6.8% | 7.9% | 13.2% |
| Snapshot/validation (`snapshot_validation`, `frame_hash`, artifact events) | 20.2% | 9.3% | 20.0% | 9.9% | 9.9% | 29.5% | 16.9% | 30.8% | 18.0% | 18.6% |
| Features (`features_transfer`, `encode_frames`) | 4.6% | 4.5% | 4.9% | 4.7% | 4.8% | 4.3% | 5.1% | 4.6% | 5.3% | 5.3% |
| Policy forward (`policy_act`, `bootstrap_values`) | 26.5% | 29.4% | 27.1% | 30.5% | 30.0% | 8.9% | 10.6% | 9.3% | 11.3% | 11.1% |
| PPO update (forward/loss, backward, clipping, behaviour validation, Adam, plus the `ppo_update`, `ppo_metrics` and `ppo_freeze_rollout` spans) | 36.0% | 41.9% | 35.9% | 42.1% | 38.2% | 43.6% | 50.9% | 45.3% | 53.2% | 48.0% |
| Other (checkpoint save, collection, unattributed) | 6.8% | 7.2% | 7.7% | 7.5% | 8.9% | 3.0% | 4.2% | 3.2% | 4.3% | 3.8% |

Exclusive seconds per segment for the larger spans follow.

| Span | 16 compact | 16 fast | 16 w8 | 16 fast-w8 | 16 fast-inexact | 64 compact | 64 fast | 64 w8 | 64 fast-w8 | 64 fast-inexact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `native_step_public_export` | 0.53 | 0.75 | 0.29 | 0.34 | 0.74 | 2.36 | 2.34 | 0.87 | 0.85 | 2.28 |
| `snapshot_validation` | 2.45 | 1.60 | 2.51 | 1.63 | 1.61 | 13.48 | 8.38 | 14.06 | 8.57 | 8.21 |
| `frame_hash` | 0.98 | 0.08 | 1.01 | 0.08 | 0.08 | 4.00 | 0.30 | 4.12 | 0.30 | 0.29 |
| `features_transfer` | 0.80 | 0.84 | 0.87 | 0.84 | 0.84 | 2.60 | 2.65 | 2.74 | 2.68 | 2.46 |
| `policy_act` | 2.38 | 2.81 | 2.48 | 2.80 | 2.71 | 2.80 | 2.92 | 2.93 | 2.99 | 2.73 |
| `bootstrap_values` | 2.27 | 2.70 | 2.38 | 2.66 | 2.59 | 2.54 | 2.62 | 2.64 | 2.70 | 2.45 |
| `ppo_forward_loss` | 2.51 | 3.31 | 2.59 | 3.15 | 3.08 | 10.60 | 11.05 | 11.16 | 11.06 | 10.34 |
| `ppo_backward` | 2.40 | 2.81 | 2.38 | 2.69 | 2.31 | 9.63 | 9.71 | 9.75 | 9.76 | 7.65 |
| `ppo_validate_behavior` | 0.50 | 0.51 | 0.50 | 0.52 | 0.17 | 2.00 | 2.02 | 2.02 | 2.02 | 0.59 |

At 64 rows the exact fast plan cut snapshot validation from 13.48 to 8.38 s
and hashing from 4.00 to 0.30 s. At 16 rows they fell from 2.45 to 1.60 s and
from 0.98 to 0.08 s, while PPO forward/loss and backward took longer in that
segment. Eight workers reduced native step public export at 64 rows from
2.36 to 0.87 s on compact and from 2.34 to 0.85 s on compact-fast.

Exact trim reduces encoded tensor widths for encoding, transfer and storage.
The model restores full widths before computing. The measured 64-row exact
fast segment spent 2.65 s in feature transfer versus 2.60 s for compact.
Validation and hashing accounted for the largest measured savings in that
comparison. The fast comparison bundles exact trim and sampled validation.

Inexact trim runs the model on the trimmed rows. Comparing the two one-worker
fast plans at 64 rows, feature transfer fell from 2.65 to 2.46 s, PPO
forward/loss from 11.05 to 10.34 s, backward from 9.71 to 7.65 s, and PPO
behaviour validation from 2.02 to 0.59 s. These are exclusive host wall times
from the unsynchronized segment.

On 64 synthetic CPU frames across one Adam update, the maximum absolute
legal-logit difference was 2.24e-8 and the maximum value difference was
1.49e-8. Greedy choices matched, and 22 parameter tensors had unequal
gradients (the test counts unequal tensors; it does not measure the size of the differences). Inexact trim is measurement-only and refused outside
`profile_entry`.

### Caveats

- One replicate per plan on a shared host and GPU. Other load was uncontrolled.
- The first shards use fuzz level 0. Later fights can have more occupied rows
  and a different combat mix.
- These are simulator throughput measurements. RNG-001 and TIME-001 apply.
