# Standard combat throughput (Gym 1.1.1)

Measured 2026-11-20 on the Standard stream (`baalorlord-standard-v1`, `*-standard-v1` learner
identity, Gym `e3c13836bd1458be8a87b93160f5c350f78fcd80`, rules epoch `gym-1.1.1`). Simulator
timings only; not a training or panel result.

## Result

| Rows | Segment decisions/s (synced, synced, unsynced) | Headline (unsynced) | Decisions per segment |
| --- | --- | --- | --- |
| 16 | 158.2, 159.5, 165.1 | **165.1** | 4096 |
| 64 | 176.0, 174.4, 174.3 | **174.3** | 16384 |
| M10 reference, 16 rows | 170.7 and 285.4 (before and after the PR #8/#10/#13 fixes) | 285.4 | |

Both Standard figures are well above the 100 decisions/s trigger, so no throughput fix (7b) is
needed. Standard at 16 rows is about 58% of the M10 post-fix figure; widening to 64 rows gives
only about 6% more, so row count is not the lever here.

## Phase split (unsynchronized segment 2, share of segment wall time)

| Phase (spans) | 16 rows | 64 rows |
| --- | --- | --- |
| Env step (native step/reset public export, driver step/reset) | 6.9% | 10.0% |
| Snapshot/validation (`snapshot_validation`, `frame_hash`, artifact events) | 23.5% | 31.9% |
| Features (`features_transfer`, `encode_frames`) | 6.5% | 6.8% |
| Policy forward (`policy_act`, `bootstrap_values`) | 19.8% | 8.2% |
| PPO update (forward/loss, backward, clipping, behavior validation, Adam) | 37.6% | 40.0% |
| Other (checkpoint save, collection, unattributed) | 5.6% | 3.0% |

Segment wall: 24.8 s (16 rows), 94.0 s (64 rows). Largest single spans: PPO forward/loss and
backward (about 15% each), then snapshot validation (14% at 16 rows, 22% at 64) and frame hash
(9-10%). The native engine is under 11% of the time at either width. The split of the synced
segments is within about two points of the above.

## GPU waits

None. The `gpu_wait` gate (min free 13,958,643,712 B, poll 30 s, cap 21,600 s) was enabled and
the GPU had about 30 GiB free throughout, so `gpu_waits.events` is empty in both runs.

## Method

- Entry `scripts/neural/profile_entry.py`: three fresh-weight stream segments per plan
  (synchronized, synchronized, unsynchronized), each at most 8 updates, one 32-step rollout per
  update. Segment 0 carries CUDA warm-up; the last segment is the unperturbed headline.
- Plans/templates: `configs/neural/profile-standard-1.1.1/{plan,template}-{16,64}.*`; model,
  PPO and seeds as `baseline-gym-1.1.1-cv2` (embedding 16, hidden 64, minibatch 64). The
  64-row plan raises the per-segment action allowance to 16384 so each segment still performs
  8 updates.
- Offline runtime container, `cuda:0` on the shared GPU, 4 CPUs, 16 GiB; both runs exit 0 with
  `learner_changed` false. Local evidence: `artifacts/development/neural-combat/profile-standard-{16,64}-1/`
  (not committed).
- `profile_entry.py` previously validated `gpu_wait` without applying it. It now waits before
  each segment (outside the segment clock) and records `gpu_waits`; M10 behaviour is otherwise
  unchanged and covered by tests.

## Caveats

- One replicate each; segment-to-segment spread is about 2-4%. The M10 figures come from an
  earlier profile on a different Gym build, GPU load and recipe (Standard states are more
  varied), so the 58% ratio is indicative.
- Every segment sees only the first shards of the stream (fuzz level 0), so the combat mix
  differs from late training and throughput will drift with it.
- Snapshot validation plus frame hashing is about a third of the time at 64 rows and is the
  next hotspot if throughput is ever needed; not pursued because the trigger was not met.
- Simulator throughput only; no original-game transfer claim. RNG-001 and TIME-001 qualifications apply.
