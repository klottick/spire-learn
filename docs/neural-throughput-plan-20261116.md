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
unperturbed: 63.96 s, or 64.0 decisions/s, close to the baseline's overall
58 decisions/s, which also includes qualification and per-segment setup.

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
| Native step and reset (`driver_step`, `driver_reset` and both public-export spans) | 0.9 | 1% |

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
   - Replace the per-element `_unpack` with one `struct.unpack` per tensor,
     `f"={n}b"` for int8 and `f"={n}q"` for every other dtype. That is the same
     fall-through the current code uses, and with standard sizes and no padding
     it decodes byte-identically. `BufferArray` already enforces the buffer
     length.
   - Replace the per-element binary `any(...)` with a set comparison.
   - Stop `candidates` decoding again. `validate_frame` currently discards its
     decoded values and returns early on a memo hit, so this needs either a
     decoded-values cache keyed like the validation memo, or one decode passed
     to both. Its ceiling is the `candidates` share: 8.4 cProfile seconds of
     the 41 spent unpacking.

   This is pure standard-library Python, with no new dependency.
   - **Expected gain.** The micro-benchmark saves about 2 ms of the 6.6 ms each
     validation call costs. That leaves about 25–29 s of validation per segment,
     and about 80–85 decisions/s (about 1.3×).
   - **What it leaves.** The per-row, per-column loop in
     `_validate_frame_contents`, the largest single cost at 24.6 s of own time
     under cProfile, stays for step 2.
2. **Vectorised column checks.** Move the per-row, per-column loop in
   `_validate_frame_contents` to whole-tensor NumPy operations on the decoded
   arrays. NumPy is already in the runtime and used by `snapshot.py`.
   - **Target, not a measurement.** Validation under about 3 s per segment would
     give roughly 150–170 decisions/s (2.5×) at the same rollout width. Measure
     it with the same profile plan before claiming it.
   - **Risks the differential tests must pin:**
     - Python integers are unbounded and NumPy int64 is not, so range and
       vocabulary comparisons must not overflow.
     - Error order and error text change; tests should check the decision, not
       the message.
     - `_selector_group` and reference resolution depend on each row and on
       occupancy, so they need care to vectorise.
     - Unoccupied rows have three cases: required known-empty, optional, and
       padding.
   - This is a larger correctness surface than step 1, so it gets its own
     review.
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

## Result: validation fast path (fixes 1 and 2)

Implemented on 2026-11-16 as one change. NumPy turned out not to be needed:
`schema.py` stays dependency-free.

**Design.** `_validate_frame_contents` first tries `_frame_is_valid_fast`, which
can only accept:
- **Unoccupied rows.** These are most of every capacity, with up to 1,024 rows
  per tensor. Whole runs of them are compared as bytes against their required
  padding, and int8 tensors are checked binary with one `bytes.translate`.
- **Occupied rows.** These run the reference's own per-row check, now shared
  code, over zero-copy `memoryview` views.
- **Fallback.** Any mismatch, unexpected shape or exception falls back to the
  unchanged reference validator, `_validate_frame_contents_reference`. The
  reference remains the only source of rejections and their messages.

Tensors decode in one `struct.unpack` call, and `candidates` reads the same
zero-copy views.

**Evidence that decisions are unchanged.**
- **Unit differential tests** (`tests/unit/test_neural_validation_fast_path.py`).
  Every column of sampled rows is probed with about a dozen boundary values,
  plus known-bit and int8 flips, on synthetic, common and Standard frames: more
  than 4,000 cases, about three-quarters rejected. The fast path accepts exactly
  what the reference accepts, and sampled composite calls match its messages.
  During development they caught each of six deliberately injected fast-path
  defects; that mutation check was manual and is not part of the suite. Two
  parser-rejected layouts found in adversarial review, non-int8 or short
  occupancy tensors, are pinned by tests and go to the reference.
- **Real frames** (`scripts/neural/validation_differential.py` on Gym v1.1.1,
  1,000 batch steps in each of the four modes):
  - all 11,719 real frames were accepted by the fast path, with no fallbacks;
  - 9,360 random single-cell mutations of them were checked, of which 8,420
    were rejected;
  - there were zero disagreements in either direction;
  - checking took 90.9 s in the reference against 2.1 s on the fast path.

**Throughput.** The same three-segment profile on v1.1.1, using
`configs/neural/baseline-gym-1.1.1/plan.json`. The table compares segment 2,
the unsynchronized one, in each run. The synchronized segments 0–1 went from
63.6 and 66.7 to 182.0 and 181.1 decisions/s.

| | Before (v1.1.0) | After (v1.1.1) |
| --- | ---: | ---: |
| Segment wall, unsynchronized | 63.96 s | 23.65 s |
| Decisions/s | 64.0 | 173.2 (2.7×) |
| `snapshot_validation` | 40.9 s (64%) | 1.9 s (8%) |
| Unattributed per-segment overhead | 7.3 s | 7.3 s (31%) |

The builds differ. v1.1.1's `DELIVERY.md` states that observation and action
layouts are unchanged, and its captured descriptors match v1.1.0's apart from
`rules_id`, so validation reads the same tensors. Per-segment overhead (fix 3) is now the largest single cost.

## Result: per-segment setup (fix 3, revised) — 2026-11-17

Fix 3 as planned would have lengthened segments, but that changes the training
data mix. Each segment is bound to one 256-scenario stream shard: the config pins
`shard_index`, and the successor offered is the next index
(`stream_runner.py`, `run_stream_segment`). A segment draws about 186 episodes
(case choices) from its shard. Longer segments would draw more episodes per shard
and visit fewer distinct shards over a fixed budget, so the idea was dropped.

**Diagnosis.** `profile_entry.py` now splits each segment into runtime creation,
training and finalization, and can run cProfile over one segment
(`--cprofile-segment`). The roughly 7 s of setup per segment was deterministic
recomputation:
- `generate_scenario_shard` ran about 19 times per segment. Config admission,
  `stream_identity`, resume validation and checkpoint provenance regenerate the
  same current and next shards through pure-Python canonical-JSON hashing.
- `CurriculumSampler.validate_cases` revalidated all 256 cases, rehashing every
  deck, on each of about 186 case choices (episodes) per segment.

**Change.**
- Shards are deeply frozen, and pure functions of the recipe's content hash,
  split, index and arm, each with its own derived RNG. A 16-entry in-process
  cache keyed by exactly those values reuses them, so a changed recipe never
  hits.
- The sampler skips revalidation only for the same tuple object of exact
  `NeuralCase` records (entries frozen at construction) that already validated
  against its fixed annotations.
- Adversarial review found bypasses, all deliberate. Two are now closed:
  - On every hit, both caches check that each field is still the exact stored
    object, so a record rebound with `object.__setattr__` is regenerated or
    revalidated.
  - The shard cache is skipped for recipe or support subclasses.

  Two stay out of scope: patching generator module constants at runtime, which
  only tests do and which then clear the cache, and mutating the sampler's
  private `_annotations`.

**Evidence that training is unchanged.**
- **Unit tests** (`tests/unit/test_neural_stream_caches.py`) cover:
  - cache identity, key misses (including a changed recipe), bounds and
    uncached failures;
  - memo identity rules;
  - identical curriculum choices with and without the memo.
- **CPU byte comparison.** CPU training is deterministic. Three 1,024-decision
  segments from fresh weights, using `configs/neural/baseline-gym-1.1.1/` with
  `device = "cpu"`, ran on `main` (`c17e560`) and on this change (`25e3eb3`,
  twice). All three runs produced byte-identical `model.pt` and `training.pt`
  for every segment. The checkpoint stream, sampler and training state differ
  only in wall-clock `elapsed_seconds`. The GPU path is not byte-checked, because
  CUDA kernels are not deterministic run to run, but the caches act only on CPU
  stream work, which is the same on both paths.

**Throughput** (same profile plan as above, on the RTX 5090):

| | Validation fast path (overhead run) | Plus stream caches |
| --- | ---: | ---: |
| Runtime creation per segment | 5.1 s | 0.13 s |
| Finalization per segment | 1.8 s | 0.03 s |
| Segment wall (segment 2, unsynchronized) | 24.0 s | 14.4 s |
| Decisions/s | 170.7 | 285.4 |

That is about 4.5× the original 64 decisions/s. Training is now mostly the
policy, PPO and validation spans.

## Evidence

All of this is outside Git, under `artifacts/development/neural-combat/`:
- `profile-gym-1.1.0-20261116/` holds `profile.json`, the prepared configs and
  the manifests.
- `validation-profile-gym-1.1.0-20261116/` holds `validation-profile.json` and
  `.txt`.
- `validation-differential-gym-1.1.1-20261116/` and
  `profile-gym-1.1.1-fastpath-20261116/` hold the fast-path evidence, from
  commit `84fce8e`.
- The `-b` directories repeat both runs on the reviewed code, commit `5e926ed`.
  The differential gave identical counts and agreement. The profile gave
  170.6 decisions/s in segment 2 (193.4 and 178.9 in the synchronized
  segments), with validation at 2.1 s.
- `profile-gym-1.1.1-overhead-20261117/` holds the phase split and cProfile;
  `profile-gym-1.1.1-caches-20261117/` holds the cached result;
  `cache-equivalence-20261117/` holds the CPU byte-comparison runs and inputs.

One profile run and one cProfile run, on a shared host with the co-resident GPU
services running.

`validation_profile.py` was still uncommitted when it produced the cProfile
evidence: that run's final manifest records head `534f941` with the script
untracked. The committed script adds only the runner output-path check. Its
profiling logic is the same.
