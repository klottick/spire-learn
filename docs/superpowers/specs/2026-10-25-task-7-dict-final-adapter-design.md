# Task 7 dictionary final-observation reader revision

**Status: Proposed; NOT approved for implementation or another runtime run.**
This is a narrow revision of M1 §7 after the first actual framework probe reported
incompatible. No framework adoption, training or broader native experiment follows
from approving it. Luna workers remain max reasoning, with substantive parent review.

## Evidence and uncertainty

The accepted native probe passed 1 test/0 skips:11 batches,22 slot decisions,
one replacement decision. The sole TorchRL probe failed 1 test/0 skips at the
first mixed batch (10):20 raw native decisions were delivered and preserved,
matching the native probe's first20 public transitions. Wrapper next observations
did not match the independently captured old-final/continuing rows. CPU mask
sampling/log-probability passed; constructor calls were0, reset1, steps10,
close1, denials0;20 diagnostic draws stayed within256. No run was retried.

The retained report proves a mapping mismatch, not its precise cause: mapped
TorchRL tensors were not separately persisted. Inspection of the **installed**
TorchRL0.14.1 terminal reader suggests a cause: it nests Dict terminal tensors
under `observation`, then looks up each declared leaf at the outer final-info
level. Missing leaves keep zero buffers; a same-named info field such as
`outcome` can also collide with an observation leaf. This is a source-based
hypothesis to test, not a claimed measurement of each failed mapped tensor.

Evidence: `artifacts/development/foundation/task-7-torchrl-runtime-report.md`,
`task-7-torchrl-review.md`, runtime root
`95b65383-0d3d-409d-a011-de1a082de926`; tested source commit `a9cdf09`.
The separate setup96vs32 draw-budget breach remains a historical qualification.

## Approaches and recommendation

1. **Recommended: learner-owned dictionary-aware public info reader.** Use
   TorchRL's `BaseInfoDictReader`/registration extension points, keep GymWrapper
   and VecGymEnvTransform, and copy only declared public `final_obs` leaves into
   TensorDict `final`. Reuse the existing guarded probe. This is a semantic
   adapter and therefore needs new approval; success is not presumed.
2. **PyTorch-native fallback.** Keep raw Gym batching and add a separately tested
   Torch tensor/distribution path. M1 allows this after explicit approval and
   actual boundary/mask evidence. It changes the selected integration approach
   and is not included in this proposed reader revision.
3. **Retain the current incompatible result.** Make no changes or new calls;
   Task7/collector-contract freeze/M1 remain incomplete.

The recommendation is the parent's proposal, not a user-selected architecture.

## Proposed behavior

- A small new optional adapter module owns the reader; `run_torchrl_probe` keeps
  its public signature. Native-vector code remains unchanged.
- The passive wrapper override substitutes exactly one learner reader for the
  vendor reader supplied by TorchRL's constructor. Register the learner reader
  once through `set_info_dict_reader`; never register both. Assert vendor reader
  absence, learner registration count1 and a matching final leaf spec.
- The reader receives the parsed public schema and known public observation spec.
  It exposes an already-known `info_spec` for `final`; registration uses no
  environment reset/step, sampling, shape-discovery rollout or check_env_specs.
- Empty registration info and initial reset have no final rows. On actual steps,
  done flags and full-width final presence masks must agree. Partial/missing keys,
  wrong shapes/dtypes, live-row payloads or missing ended rows fail explicitly.
- Prepare owned snapshots of all declared final observation tensors before
  assigning output. Write every declared leaf for ended rows under `final`,
  with exact CPU dtype/shape/value preservation. Never source observation leaves
  from `final_info`, including the colliding `outcome` name. No synthetic final
  values may replace delivered ended-row values.
- Zero buffers for absent rows/spec registration are placeholders only. Presence
  and done masks govern use; they are never conformance or terminal evidence.
  Do not mutate incoming final_obs objects, None placeholders, main reset arrays
  or routing/state records. Keep old-final and replacement memory separate.
- Preserve the pre-dispatch guard, raw capture before wrapper processing,
  attempted/native-delivered/wrapper-delivered accounting, error priority,
  diagnostic pre-draw reservations and exactly-once native close.
- Persist a versioned adapter identifier and per-key/row mapping mismatch details
  on any new failure. Keep old failed output immutable. Report `adaptable` only
  after all eight required checks pass; otherwise remain incompatible/blocked.
  Authored-data tests assert the persisted ID and mismatch row/key plus expected
  and observed dtype/shape/value details where available, without partial output.

## Approval envelope

Approval would authorize only the companion plan's code/tests plus one new
protected offline validation shard: three framework-reader checks using authored
numeric data (zero `spire_lab`/native-binding imports, environments, actions or
diagnostic draws; the stdlib learner adapter remains importable), followed
only if green by one fresh B2/workers1 selection cohort: Cultist env7/policy101,
JawWorm env11/policy103, <=128 full-batch native steps, <=256 diagnostic draws,
60s probe soft limit and600s hard shard including cleanup. The failed old cohort
is closed and is never resumed/retried; this would be a separately approved run.

Reuse the exact qualified23-wheel runtime/lock and pinned Python image, non-root
1000:1000, network-none, root/source/venv read-only, only new output writable,
4CPUs/8GiB/512PIDs/cap-drop/no-new-privileges. Check >=30GiB free and existing
20GiB cumulative soft storage budget; no pull/prune/budget expansion. No new
packages/images, host installs, GPU, training, fallback, Gym/vendor edits,
shared-service changes, push or publication. Preserve original learner changes.

[Companion implementation plan](../plans/2026-10-25-task-7-dict-final-adapter.md).
