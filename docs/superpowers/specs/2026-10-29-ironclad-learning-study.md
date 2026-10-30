# Bounded Ironclad learning study package

This refines Task 3 of the approved 2026-10-28 neural GPU plan. Source and
configuration preparation is authorized; experiment execution is not. The
accepted GPU proof at commit 3431f20 establishes plumbing and portability,
not playing strength. No installation, Gym change, service stop or proof replay.

## Study and interpretation

Four fresh, non-resumed learners use the accepted model/PPO unchanged. Training
order: standalone-a, run-a, standalone-b, run-b. Standalone uses B16/h32,
explicitly revising the old B<=8 proposal from the accepted calibration; Run is
B1/h128. Each run stops at its first ceiling: 16,384 learned decisions, 65,536
environment step slots, 1,024 optimizer attempts, 600 seconds soft, 900 seconds
whole including setup and cleanup. One final training checkpoint per run, zero
midpoints. Its immutable weights are the frozen-policy input; there is no extra
lean-export format. No automatic extension, retry, replacement seed or resume.

Training learner seed quartets (policy, optimizer, curriculum, noncombat):
standalone-a=(101103,102207,103301,104409),
standalone-b=(201103,202207,203301,204409),
run-a=(301103,302207,303301,304409),
run-b=(401103,402207,403301,404409).
These are independent learner RNG seeds. Native training resets retain the
accepted environment-seed-for-replica0-v1 ordinal policy, so native seed streams
are matched, not disjoint independent simulator streams. Record actual seeds.

Selection is CPU for all four learned policies and both references, while
training is GPU. Keep this backend difference explicit. References are
legal-random and end-turn-last-v1. Each operation runs in a fresh process;
reference processes must neither attempt nor load Torch/CUDA. Frozen neural
choices use highest legal logit, smallest original action index on ties. Final
split remains untouched and unused.

Author 16 standalone selection cases: four training-catalog encounters
(cultist, jaw_worm, two_louse, small_slimes) with two seeds each, and two
excluded encounters (large_slime, three_louse) with four seeds each. Reuse the
existing selection entry templates, not simulator rules. Held-out encounter
means excluded from the standalone training catalog; no global non-exposure
claim is made for Run-trained policies. Author 16 Ironclad A0 normal Run cases
using explicit m10-ironclad-policy. Selection seeds are (1<<40)+4096+i for
standalone i=0..15 and Run i=16..31. Unique IDs, lineages and entry IDs.
Eval seed quartet is (91003,92009,93011,94007), matched across policies.

Each of 32 cases has one actual one-case manifest/config; each is paired with
all six policies, yielding 192 operations. For case index 0..15, rotate policy
order [standalone-a,run-a,standalone-b,run-b,legal-random,end-turn-last-v1] by
index modulo six; for each policy, visit both modes, alternating first mode by
case-index parity. Reserve 256 step slots before each operation, maximum 49,152
reserved within the shared 65,536 ceiling. Actual delivery and uncertainty stay
separate. Shared selection soft deadline is 600 seconds from before setup and
whole phase cap 900 seconds including cleanup. Unstarted rows are persisted.
No selection optimizer calls or training resets. Selection may be partial.

## Binding and storage

Authored package: configs/neural/experiments/task9-v1/study.json, four train
TOMLs, 32 selection TOMLs, parent32 selection JSON and 32 one-case JSONs in
configs/cases/neural/. Canonical config/manifest digests must come from actual
loader inputs, never dataclasses.replace with an old digest. Raw byte hashes
are separately named. All one-case records equal their parent member exactly.
Validate train/new-selection/final identity separation without executing final.

Future checkpoint hashes are unknown. After each healthy training operation,
bind the exact final checkpoint directory, metadata and all payload hashes
and accepted update evidence. Before selection, freeze these bindings and
verify them on every use and after evaluation. A policy lacking a valid finite
accepted update/checkpoint is unavailable; its schedule rows are not_started,
with no substitute. Any operational fault halts the remaining campaign and
records not_started rows; genuine game death remains a valid outcome.

Proposed additional storage ceilings: 1 GiB each training output and 2 GiB
shared selection output, 6 GiB total; retained aggregate <=40 GiB and >=30 GiB
free before admission. These are accounting/admission controls, not filesystem
quotas. Bound subprocess output and stop when accounting fails; don't prune
retained evidence to make room. Existing event-log ceilings remain unchanged.

## Runtime and controls

Reuse accepted CPython3.12 CUDA venv, Torch2.14.1+cu130, native Gym revision
533dad1c8aec86df624b3742b6b3ad1dbf1d0033 and common fingerprint
02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78.
Run observation fingerprint stays
30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8.
Lock SHA256 92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45.
Native SHA256 071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9.
Image python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0.
GPU UUID GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d, min12GiB free,
4 CPU threads/1 interop, <=8GiB host memory/8GiB Torch GPU allocator, <=512 PIDs.
The allocator cap is not a hard limit on all driver/context VRAM; record device
and process memory separately, with >=12GiB free before training admission.

Separate container per training operation and one selection phase container;
five phases at <=900 seconds each, campaign <=4500 seconds. No phase begins
before explicit package approval bound to reviewed source/package bytes.
Protected source/runtime read-only, exactly one owned output child writable;
non-root1000:1000, root read-only, network none, cap-drop ALL/no-new-privileges,
pull never. Local unix Docker daemon only with fresh empty client config;
no home, socket, Gym checkout, credential or client-config mounts. Reject an
existing exact container name before create. Only successful create returning
one full verified ID establishes cleanup ownership; never remove by guessed
name after failed/ambiguous creation. Hard watchdog includes bounded cleanup.
Fresh source/release/one-use campaign and phase claims; no rearming old proofs.

Minimal runner change: evaluate_policy must consume the existing
RuntimeOperationLimits.monotonic_deadline, including setup admission, so the
selection schedule cannot reset its shared deadline. The outer controller
owns schedule reservations, durable operation receipts and not_started rows.
No second trainer, simulator, checkpoint schema or model architecture.

Keep RNG-001, TIME-001, finite supported content, simulator-only evidence,
partial coverage, and saved checkpoint elapsed-prefix limitations explicit.
Saved elapsed ends at checkpoint snapshot; later finalization is outside that
prefix. Fresh training has no resume, while whole-phase deadlines still cover
all current invocation work. Calibration rates don't promise cap completion.
