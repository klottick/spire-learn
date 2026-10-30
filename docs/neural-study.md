# Bounded Ironclad learning study

The single approved replacement `task9-study-20261030-b` completed at source
`17e71f07d736469955ce09a7b0dbb3643052ec9a`: four eligible checkpoints,
43,163 learned decisions, 3,447 accepted optimizer steps, zero simulator faults
and all 192 CPU selection operations completed in 47.47 minutes. All five owned
containers were removed. The original a attempt stopped before training and its
closed evidence remains preserved. Both one-use claims are consumed; no new
training or evaluation is authorized.

## Selection results

Each policy was evaluated on the same 16 isolated-combat and 16 normal-Run cases.
The last column counts combat victories per Run case, not Run victories.

| Policy | Isolated-combat wins | Mean combat victories per Run case |
| --- | ---: | ---: |
| standalone-a | 7/16 | 3.5625 |
| run-a | 16/16 | 5.3125 |
| standalone-b | 16/16 | 4.8750 |
| run-b | 16/16 | 5.5625 |
| legal-random | 15/16 | 2.2500 |
| end-turn-last-v1 | 16/16 | 4.6250 |

All 96 normal-Run trials ended in death. Standalone A won 7/8 seen cases and
0/8 held-out-encounter cases; the other three neural checkpoints won 16/16
isolated fights. Held-out encounters are excluded from standalone training
only; Run training may encounter them. Run B is a provisional candidate for
further validation. These small, seed-sensitive selection results establish
neither significance, final-split performance nor original-game transfer.
In-Run fight denominators depend on policy survival, and post-cleanup HP includes
healing. The final split remains untouched. Exact checkpoints, per-case results,
recorded soft overshoots and prior qualifications are in the [retained runtime report](../artifacts/development/neural-combat/task-9/experiment-package/execution-20261030-b/runtime-report.md).

## Reviewed protocol

The remaining sections describe the reviewed inputs and controls used by the
completed campaign. They grant no authority to invoke it again. The
[runbook](../scripts/neural/task9-study-runbook.md) is a historical protocol.

The authoritative authored inputs are [study.json](../configs/neural/experiments/task9-v1/study.json)
and its exact four training configs, 32 selection configs and parent/one-case
manifests. The pure package validator binds raw bytes separately from canonical
loader identities. The final split is validated for separation and never run.

The five phases are standalone-a, run-a, standalone-b, run-b, selection. Each
training phase starts fresh with the accepted model/PPO, one final checkpoint
and no midpoint/resume/retry/seed replacement. Standalone uses B16/h32, a
calibration-informed revision from the older B<=8 proposal; Run uses B1/h128.
Each stops at the first of 16,384 learned decisions, 65,536 step slots, 1,024
optimizer attempts or 600 seconds soft. Each whole phase includes setup and
cleanup within 900 seconds; the campaign includes all work within 4,500 seconds.

Selection evaluates four frozen policies and legal-random/end-turn-last-v1 on
32 exact cases, 192 scheduled pairs. All selection executes on CPU using the
qualified CUDA-wheel stack; training uses GPU. Every operation gets a fresh
process and a 256-step reservation; 49,152 planned reservations remain below
65,536 shared. One 600-second soft deadline starts before selection setup and
never restarts between children. Missing healthy zero-update policies are
unavailable; their rows are not_started. Faults halt the remaining campaign.
Reserved operations without a trustworthy receipt remain unresolved. Partial
selection is valid and is never given a fabricated full denominator.

Cases contain 16 standalone encounters (8 training-catalog and 8 held-out
encounter cases) and 16 Ironclad A0 normal Run cases with m10-ironclad-policy.
The held-out label means excluded from the standalone training catalog, not
unseen by Run policies. Policy order is rotated by case index modulo six; both
modes are visited per policy, alternating first mode by case-index parity.
Frozen neural action selection takes the highest legal logit, breaking ties
by smallest original index. References never attempt or load Torch.

One separate user approval binds reviewed source, study, runtime inputs,
resource/storage limits, five phase identities and mechanical post-training
binding. Actual checkpoint hashes are unknown until healthy training produces
them. Controls verify actual receipt run UUIDs, authoritative lifecycle,
finite accepted updates, metadata and all checkpoint files, then freeze those
bindings before selection and verify them before/after use.

Outputs are accounting-limited to 1 GiB per training phase, 2 GiB selection
and 6 GiB total including control records, source staging and logs. These are
not filesystem quotas or promised completion capacity. Admission requires
retained aggregate <=40 GiB, >=30 GiB free and >=12 GiB GPU free before training.
Containers use 4 CPUs, 8 GiB host memory with an 8 GiB combined memory-and-swap ceiling and 512 PIDs. The 8 GiB Torch GPU
allocator ceiling does not hard-limit driver/context VRAM; actual memory
observations remain separate. No other model/service is stopped to make room.

Interpret results with RNG-001, TIME-001, finite supported simulator content,
matched native training reset streams, CPU/GPU backend difference and partial
coverage explicit. Learner RNG quartets differ; the native reset ordinal
policy matches simulator seed streams. Checkpoint saved elapsed is a prefix
ending at snapshot; finalization happens later inside the whole phase cap.

Prepared source fixes address the final review's R7/R8 findings. Child timeout
now uses the remaining time to the absolute shared deadline plus 60 seconds,
including prior accounting/reservation/argv work. Expired grace prevents child
dispatch while retaining any reservation as 256/unresolved. Failed reservation
publication also retains its 256-slot bound in B's aggregate independently of a
returned digest; dispatch/effect certainty stays separate from publication
uncertainty, and partial/unreadable footprints never become completion.

A healthy child that entered its reservation but expired before factory setup
has a distinct trusted not_started receipt. C validates the actual identities,
reservation/entry linkage, supported deadline reason, explicit no-effects and
absence of runner artifacts; all 256 slots remain reserved and counters remain
unknown. This supports healthy partial selection without retry or refund.
Those source tests use strict fakes. The completed study adds actual bounded
runtime evidence; durability under a real crash remains unqualified.

The replacement routes artifacts to
`artifacts/development/neural-combat/experiments/task9-v1-attempt-b`; config paths
remain under `configs/neural/experiments/task9-v1`. Only output routing and its
canonical/raw config bindings changed; cases, seeds, schedule, model/PPO and
budgets are unchanged. The proposed future campaign identity is
`task9-study-20261030-b`. No replacement approval or claim is created by this
source correction.

The [authored NVIDIA CDI profile](../configs/runtime/task9-nvidia-cdi-profile.json)
is bound by raw hash in the study and staged context/source. Host checks pin the
CDI spec and hook binary before claim acquisition and throughout closure. Docker
continues to request the exact qualified GPU UUID. GPU guest admission recognizes
only legacy exact values or the pinned post-OCI `NVIDIA_VISIBLE_DEVICES=void` plus
`NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu`; all other controlled values and
unknown-key rejection remain strict. CPU selection gets no CDI exceptions.

The measured CDI profile admits nineteen exact regular RO driver files, one
specific NVIDIA persistence-daemon socket, and one UUID-named runtime tmpfs.
The RO socket remains connectable driver IPC (UID105/GID108,0777), not read-only
IPC. The tmpfs is root-owned01777 and actor-writable, limited to4096 bytes with
zero available bytes at admission and one root-owned0444 `nvct-params` child
bound to the same params file. Its inode capacity/access are recorded; no separate
hard inode quota is claimed. Container memory/PID bounds remain. Exactly one
operator-declared RW application bind remains; this is not a claim of one
globally writable mount. A replacement approval must explicitly disclose both
runtime exceptions. Metadata probes and pure/fake source tests do not establish
learning or release another training invocation.

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
