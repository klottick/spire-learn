# Task 9 protected study: historical completed protocol

The reviewed protocol below was invoked once for approved campaign
`task9-study-20261030-b` at source `17e71f07d736469955ce09a7b0dbb3643052ec9a`.
It completed all five phases and 192 selection operations in 47.47 minutes;
see the [study results](../../docs/neural-study.md). The original a and successful
b claims are consumed, their evidence/outputs remain retained, and no rerun or
extension is authorized. The executed source snapshot remains immutable.

The remaining sections preserve the reviewed pre-execution protocol and its
qualification history. Their preparation/proposal wording grants no current
execution authority. Another campaign requires separately reviewed fresh input
and output identities plus explicit approval. No installs, Gym edits, service
stops or original-game-transfer claims are implied.

## Exact proposed campaign

One approval covers five sequential fresh phases: standalone-a, run-a,
standalone-b, run-b, selection. The base proof anchor is
3431f20d8a22ca3bcae4781974d742444c299808; exact execution HEAD and Git dirty
porcelain are captured separately after source acceptance. Model/PPO identities
remain those in study.training.identity_hashes; no architecture change.
The exact [standalone-a config](../../configs/neural/experiments/task9-v1/standalone-a.toml),
[run-a](../../configs/neural/experiments/task9-v1/run-a.toml),
[standalone-b](../../configs/neural/experiments/task9-v1/standalone-b.toml) and
[run-b](../../configs/neural/experiments/task9-v1/run-b.toml) specify
embedding_width=16, hidden_width=64, max_parameters=10,000,000. The accepted proof
measured 265,122 actual parameters. PPO is learning_rate=0.0003,gamma=0.99,
gae_lambda=0.95,clip_epsilon=0.2,value_coefficient=0.5,entropy_coefficient=0.01,
max_grad_norm=0.5,epochs=4,minibatch_rows=64,adam_eps=0.00001.

| Policy | Mode | Policy/optimizer/curriculum/noncombat seeds | Rollout |
| --- | --- | --- | --- |
| standalone-a | standalone | 101103 / 102207 / 103301 / 104409 | B16/h32 |
| run-a | Run | 301103 / 302207 / 303301 / 304409 | B1/h128 |
| standalone-b | standalone | 201103 / 202207 / 203301 / 204409 | B16/h32 |
| run-b | Run | 401103 / 402207 / 403301 / 404409 | B1/h128 |

Standalone B16 explicitly revises the older B<=8 proposal using accepted
calibration. Training stops at the first ceiling:16,384 learned decisions,
65,536 environment steps,1,024 optimizer attempts,600 seconds soft. One final
checkpoint/no midpoint per phase; no resume, extension, retry or replacement.
Each phase starts its 600/900 soft/whole clocks before preflight/setup; cleanup
is included. Campaign4500 whole starts before initial preflight. Host ordinary
phase commands stop by870, retaining30 for cleanup. Launcher TERM4470/KILL4500
remains active during cleanup, and every subprocess has bounded output/time.
The launcher sets SPIRE_LEARN_TASK9_WATCHDOG=4470+30; direct live CLI refuses
without it. This marker is a watchdog handoff, not experiment approval.

Selection CPU uses all four frozen neural policies plus legal-random and
end-turn-last-v1. Eval quartet:91003/92009/93011/94007. Cases/seeds/order exactly
follow study.json:32 one-case configs×6 policies=192 pairs; rotate policies
by case-index modulo6 and alternate first mode by parity. Reserve256 slots
before eligible child entry, planned49152<=65536 shared. One absolute600 soft
limit includes setup; whole900 includes cleanup. Fresh process per operation;
no optimizer or training reset in selection. Neural action rule is highest
legal logit/smallest original index on ties. References stay Torch-free.

## Pins and filesystem

CPython3.12, Torch2.14.1+cu130/CUDA metadata13.0, Gym
533dad1c8aec86df624b3742b6b3ad1dbf1d0033, common fingerprint
02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78,
Run observation fingerprint
30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8.
Lock SHA256:92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45.
Native SHA256:071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9.
Image:docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0.
GPU:GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d, floor12GiB free at each train
admission. Containers have 4 CPUs, 4 Torch intra-op threads, 1 inter-op thread,
8 GiB host memory, an 8 GiB combined memory-and-swap ceiling and 512 PIDs.
8GiB Torch allocator cap is not hard driver/context isolation. Record observed
process/device memory separately. Selection reuses the qualified CUDA-wheel
venv on CPU; never use legacy+cpu validators or install a CPU stack.

The existing CUDA venv and lock under task-8/runtime-output/cuda-setup-20261028-a
are separate RO mounts at /runtime/venv-cuda and /runtime/requirements-cuda.lock.
Only public src/configs and six actual imported helpers (including task9_nvidia_cdi) are staged under the
host control directory and mounted RO at original /learner paths. No tests,
.git, private artifacts, runtime binaries or old claims enter source staging.
Approval/snapshot/claims/context mount RO at /control. Only the exact authored
training root (or selection root) is the sole operator-declared RW bind; no
writable ancestor, extra application bind, host home/client config/Gym mount or
Docker-management socket. The pinned CDI runtime exceptions below are separate
from application binds; one is an actor-writable tmpfs and one is NVIDIA IPC.

TrainingRoot is strictly empty at guest entry. B reserves once then creates
.runtime/{tmp,cache,torch,torch-extensions,cuda,triton,home} before factory.
C sets HOME/TMPDIR/cache env paths there but never precreates training scratch.
Selection may precreate its scratch/control/case skeleton. Frozen binding file
is an immutable RO overlay on selection/control/checkpoint-bindings.json.
Only actual eligible checkpoint-final directories are RO mounted at original
/learner/<repo-relative-path>; whole training roots are never exposed.

## Approval and proposed commands

Pure preparation example, in an explicitly selected evidence directory:

```sh
PYTHONPATH=src:. python3.12 -B scripts/neural/task9_study_controls.py describe
PYTHONPATH=src:. python3.12 -B scripts/neural/task9_study_controls.py snapshot --root "$REPO" --snapshot "$SNAPSHOT" --learner-head "$REVIEWED_HEAD" --dirty-json "$DIRTY_RECORD"
```

Default snapshot state source_only cannot execute. After acceptance parent
creates a reviewed immutable snapshot and an approval with exactly these keys:
schema,state,single_use,campaign_id,source_snapshot_sha256,study_raw_sha256,
review_path,review_sha256,phase_order,runtime_pins,budgets,checkpoint_binding.
Schema=spire-learn-task9-campaign-approval-v1; state=approved; single_use=true;
phase_order is the exact five IDs; pins/budgets equal validated study; binding
procedure=actual-post-training-final-inventory-once. Review report is separately
hashed and excluded from source snapshot, preventing a self-digest cycle.
Campaign ID uses lowercase alphanumerics/hyphens,max40. No fictional future
checkpoint hash occurs in approval.

Only following explicit approval, proposed live command (NOT RUN):

```sh
scripts/neural/run-task9-study.sh --root "$REPO" --snapshot "$REVIEWED_SNAPSHOT" --approval "$APPROVED_CAMPAIGN" --campaign-id "$CAMPAIGN_ID" --control-root "$FRESH_HOST_CONTROL_ROOT"
```

Host control root must be a fresh directory under artifacts/development/neural-combat,
outside authored experiment outputs. Every Docker call uses explicit local
unix:///var/run/docker.sock and a fresh empty client directory; sanitized host
environment preserves original HOME. Pinned image/pullnever,nonroot1000:1000,
rootRO,netnone,cap-dropALL,no-new-privileges. Actual inspect validates new
commands/env/mounts/resources, never invented proof-compatible values.

The acyclic identity chain is snapshot→approval→campaign claim→phase claim
(predecessor receipt hashes,deadlines)→context→actual create env/labels→phase
receipt→closed inventory. Context raw hash is added to actual env/labels only
after publication; no payload includes its own digest. Absolute floats use
JSON/repr roundtrip, not rounded display strings. Capture approval rawhash once.
The stable exclusive campaign slot is
artifacts/development/neural-combat/task-9/study-campaign-claims/<campaign-id>/campaign.claim.json.
Its per-ID directory is independent of the caller's host evidence directory;
never delete/rearm it. Exact claim bytes/hash are copied into evidence/contexts
and counted/frozen along with that per-campaign directory, so another evidence
path cannot reuse approval. Invalid preflight inputs leave it unclaimed.
All contexts/claims are exclusive; a fault never authorizes replay/replacement.

## Final checkpoint and failure handling

After each healthy training phase verify actual writer UUID (distinct from
policy ID), summary rawhash, authoritative lifecycle, positive finite accepted
update evidence and exact metadata/model.pt/training.pt inventory. Zero-update
healthy output remains unavailable; it gets no substitute. Freeze four slots
at selection setup and independently verify actual prior receipt hashes and
checkpoint metadata. Mount final checkpoint directories and binding file RO;
verify before/after each B use and again host-side after phase.

Before create an exact-name absence query must succeed empty. Only successful
create returning exactly one full64hex ID establishes ownership. Capture ID
before inspect/start; cleanup is by re-inspected ID/name/labels with one durable
removal authorization. Collision/create125/timeout/ambiguous ID never authorizes
name recovery/deletion. Preserve bounded failed-command prefixes, process-control
precedence and unknown accepted effects; no silent retry. All unstarted phase
and selection identities remain recorded. Unreturned reservations remain
unresolved, with conservative256 upper bounds.

Accounting ceilings:1GiB each training output,2GiB shared selection,6GiB total
additional including host/source/logs,40GiB retained and30GiB free admission.
These are accounting controls,not quotas. Sequential actual usage plus overhead
may refuse later admission before theoretical caps complete. Check progress;
stop rather than prune. Retained symlinks count lstat bytes without following;
phase/source symlinks are rejected. Close all referenced records before final
inventory; no self-hash cycle.

Final source fixes address B R7/R8. Dispatch samples the clock after output
accounting, reservation and argv setup; timeout is the remaining time until the
unchanged shared deadline plus60 seconds. If that absolute grace has expired,
no child launches: any published reservation stays256/unresolved with known
not_dispatched/no-effects evidence, without refund or retry. A child admitted
within grace can return the supported reserved+entered pre-factory deadline
not_started receipt. C validates that exact no-runner shape, retains256 and
unknown counters, and permits healthy partial closure; it does not require
runner artifacts for an operation that never entered a runner.

B also tracks reservation publication independently of a returned digest. A
file/file-fsync followed by directory-fsync failure retains256/unresolved in
its row and aggregate, with publication uncertainty separate from known absence
of child dispatch. Partial/unreadable footprints remain unresolved. C's audit
agrees with this producer accounting; readable bytes do not prove durability.
These are source/fault-injection results, not crash-recovery or runtime evidence.

Interpretation remains simulator-only: RNG-001,TIME-001,finite supported content,
CPU-selection/GPU-training difference,matched ordinal native training reset
streams,standalone-only heldout definition,partial coverage and saved elapsed
prefix ending at checkpoint snapshot. Genuine death is valid; simulator faults
halt campaign. Final split untouched. No native pass, learning, full coverage,
playing-strength or original-game-transfer claim follows from preparation.

## Source-review admission and closure qualifications

Initial preflight failures retain the five-phase/192-row plan, bounded command
prefixes and explicit unowned/unclaimed state. Final control records reserve
32MiB inside existing storage ceilings, with each published record bounded to
8MiB. Post-publication/inventory accounting and clocks precede a successful
return. A persisted campaign-result is provisional until closed-inventory
validation and successful host return. Any closure-failure.json is authoritative
and invalidates an earlier inventory; prior records are never overwritten.

Receipt footprints alone cannot resolve effects. The host audit requires a
matching reconciled schedule return, receipt identity/status/hash, reservation
and entry bindings and authoritative lifecycle/summary evidence. Corrupt or
unreadable evidence retains256/unresolved and bounded diagnostics. This consumer
check agrees with B's corrected unresolved reservation accounting; neither a
readable file nor this audit establishes filesystem durability after a failed sync.

The failed approved attempt a is stopped with a consumed claim. Preserve its
closed source/evidence and empty `experiments/task9-v1/training/standalone-a`.
Replacement authored outputs use `experiments/task9-v1-attempt-b`, while config
filenames remain under `configs/neural/experiments/task9-v1`. Proposed campaign
ID is `task9-study-20261030-b`; new review and explicit user approval are required
before training. Never rearm, resume, delete or populate the original attempt.

The raw-hash-bound profile at `configs/runtime/task9-nvidia-cdi-profile.json`
qualifies the measured Toolkit1.20.0/driver595.91.07 transformation. Host admission
hashes the fixed public CDI spec and hook executable without running them, before
claim acquisition and during failed/successful phase and final closure checks.
Requested Docker Config.Env and DeviceRequests retain the exact GPU UUID. Guest
validation separately accepts only exact legacy values or the pinned CDI void
sentinel and exact LIBCUDA_DIR; observed facts are never rewritten. Runtime GPU
UUID/count and qualified native/lock/device checks stay independent and exact.
Bounded field-specific diagnostics name a mismatched controlled value or mount;
unknown extra environment values are withheld.

CDI GPU mode requires the nineteen exact regular RO driver/firmware/OpenCL
mounts with lstat/nonwritable access evidence. Firmware mountinfo uses the pinned
image's `/usr/lib/firmware` spelling. No generic /usr or /run directory is allowed.
The exact `/run/nvidia-persistenced/socket` exception is a RO-mounted Unix socket
UID105/GID108,0777,with effective write access: driver IPC remains connectable.
No other socket, including any Docker-management endpoint, is allowed.

One `/run/nvidia-ctk-hook<UUID>` runtime tmpfs is root0:0,01777,actor-writable,
capacity<=4096 bytes,zero available bytes at admission. It must contain only the
root-owned regular0444/nonwritable `nvct-params`, whose device/inode/size match
the nonwritable params override at `/proc/driver/nvidia/params`. Type, filesystem,
owner, mode, access, inode accounting and no nested/link/extra mount checks fail
closed; the /proc exemption cannot bypass those checks. Inode capacity is observed,
not a separate hard inode quota. Existing8GiB memory/PID caps remain. Exactly one
RW application bind remains, not one globally writable mount. CPU selection gets
none of these CDI environment, driver-file, socket, scratch or params exceptions.

Two metadata-only probes established the admission mismatch and runtime mount
shape without optional runtime imports or training calls. The correction is
source/fake-tested preparation only. The replacement approval must explicitly
include the persistence IPC and writable4096-byte runtime scratch exceptions;
no metadata probe or source test grants a replacement training invocation.

Successful parent-phase admission emits one C-owned stderr line beginning
`SPIRE_LEARN_TASK9_ADMISSION_V1 `, with JSON schema
`spire-learn-task9-guest-admission-v1`. The record binds campaign/phase, raw
context/study/profile identities and actual process/group, safe CDI environment
and socket/scratch/params observations, including inode/access accounting and
the no-hard-inode-quota qualification. Existing bounded host command capture and
closed inventory retain it. Serialization, a payload including marker/newline
above16KiB, short/failed write or failed flush denies entry before learner/factory
setup. Process-control exceptions retain precedence. No application file is
created before B's empty-output check, and B's receipt schema is unchanged.
Selection children remain verified and emit no duplicate phase admission lines.
A record proves those admission checks passed; subsequent transport/learner/closure
results still determine effects and campaign completion.
