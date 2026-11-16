# Throughput profile and plan (2026-11-16, Gym v1.1.0)

The 8726d467 baseline trained at about 58 learned decisions/s with the GPU mostly
idle. Spire Gym's native engine steps far faster than that, so this note measures
where the learner's time goes and ranks the fixes. It measures throughput only;
nothing here is a learning result.

## Measurements

**Segment profile.** `scripts/neural/profile_entry.py` ran three baseline-shaped
segments from fresh weights on Gym v1.1.0, using
`configs/neural/profile-gym-1.1.0/plan.json`. Each segment was 4,096 decisions:
16 environments × 32 steps, 8 PPO updates. It ran on the RTX 5090 through
`runtime_container.py`, with exit 0 and `learner_changed: false`. Segments 0–1
synchronize the device at every span. Segment 2 does not, so its wall time is
unperturbed: 63.96 s, or 64.0 decisions/s, matching the baseline's rate.

Exclusive time in segment 2:

| Span | Seconds | Share |
| --- | ---: | ---: |
| `snapshot_validation` (frame snapshot, validation, legal candidates), 6,160 calls | 40.9 | 64% |
| Unattributed (runtime creation, checkpoint preflight, loop code) | 7.3 | 11% |
| PPO forward, backward, clipping, behaviour check and Adam | 5.5 | 9% |
| Policy inference (`policy_act`) and bootstrap values | 4.2 | 7% |
| Collection loop bookkeeping | 1.9 | 3% |
| `frame_hash` | 1.1 | 2% |
| Feature transfer to the device | 1.1 | 2% |
| Checkpoint save | 0.9 | 1% |
| Native step and reset, including public export | 0.8 | 1% |

Native Gym time is about 1% of the segment. The GPU-bound work, inference plus
PPO, is about 15%.

**Validation profile.** `scripts/neural/validation_profile.py` drove the same
prepared standalone config for 200 random-action batch steps under `cProfile`,
with no training. cProfile inflates per-call overhead, so only the proportions
matter here:

| Function | Cumulative s | Note |
| --- | ---: | --- |
| `_row_from_raw` | 86.0 of 86.7 | everything after the native call |
| `_validate_frame_contents` | 75.4 | one full check per new frame (5,184 frames) |
| `_unpack` | 41.0 | 191,259 calls; decodes every element through `struct.iter_unpack` and a generator |
| the `any(...)` binary checks | 16.3 | per element, int8 tensors |
| `candidates` | 8.4 | re-unpacks tensors that validation already decoded (15,003 `_tensor_values` calls) |
| native `_step_deliver` and `_reset_deliver` | 0.4 | the simulator itself |

Each frame decodes about 46,000 int8 and 40,000 int64 values one element at a
time in Python. A host micro-benchmark on buffers of that size gave these per-frame
costs:
- Unpacking drops from 2.02 ms to 0.48 ms with one `struct.unpack` call.
- The binary check drops from 0.70 ms to 0.27 ms as a set comparison.

## Plan, in order

Each step must keep validation fail-closed with identical accept and reject
decisions. The current validator stays in the test suite as the reference. Each
step adds differential tests: the reference and the new path must agree on real
captured frames and on mutated frames, such as out-of-range values, non-binary
masks, unknown-but-occupied rows, bad references and mask/occupancy mismatches.

1. **Bulk decoding and set checks in `schema.py`.**
   - Replace the per-element `_unpack` with one `struct.unpack` per tensor.
   - Replace the per-element binary `any(...)` with a set comparison.
   - Let `candidates` reuse the values validation already decoded instead of
     unpacking again.

   This is pure standard-library Python, with no new dependency. Expected:
   validation falls from about 41 s to about 12–15 s per segment, giving
   roughly 100–110 decisions/s (1.6–1.7×).
2. **Vectorised column checks.** Move the per-row, per-column loop in
   `_validate_frame_contents` to whole-tensor NumPy operations on the decoded
   arrays. NumPy is already in the runtime and used by `snapshot.py`. Expected:
   validation under about 3 s per segment, giving roughly 150–170 decisions/s
   (2.5×) at the same rollout width. This is a larger correctness surface, so it
   gets its own review.
3. **Amortise per-segment overhead.** About 7 s per segment is unattributed:
   runtime creation, checkpoint preflight and the save. Longer segments, for
   example 16,384 decisions, would spread it out. This changes checkpoint
   granularity, not learning; the baseline's retention points can still be hit.
4. **Wider rollouts.** Use 64 environments instead of 16. Inference and PPO cost
   per decision fall because 64-row batches cost about the same as 16-row ones on
   this GPU. This changes learning dynamics (batch statistics and update cadence),
   so it is an experiment with its own fresh baseline, not a pure speedup.
5. **Later.**
   - Gym 1.1.0's opt-in compact observation, at about 137 KB per slot instead of
     764 KB, would shrink everything validation touches. Learn's parser assumes
     the dense int64 layout, so this is a separate schema adoption.
   - Parallel collectors would follow once Python-side cost per decision is no
     longer the bottleneck.

Steps 1–3 leave model, PPO, rollout width, schema and checkpoint identities
unchanged. Their throughput gains can be measured by re-running the same profile
plan. Training results from before and after stay comparable, apart from the
documented RNG-001 and TIME-001 qualifications.

## Evidence

All of this is outside Git, under `artifacts/development/neural-combat/`:
- `profile-gym-1.1.0-20261116/` holds `profile.json`, the prepared configs and
  the manifests.
- `validation-profile-gym-1.1.0-20261116/` holds `validation-profile.json` and
  `.txt`.

One profile run and one cProfile run, on a shared host with the co-resident GPU
services running.
