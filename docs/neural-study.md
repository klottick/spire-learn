# Prepared bounded Ironclad learning study

This Task 9 package is prepared source, not an executed study or an execution
approval. The accepted proof/calibration establish bounded plumbing and measured
calibration only. No learning, held-out playing strength, native coverage or
original-game transfer is claimed here. The [runbook](../scripts/neural/task9-study-runbook.md)
defines the protected proposed invocation and artifact interpretation.

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
The evidence is strict-fake source coverage; actual runtime and durability under
a real crash remain unqualified, and final source re-review is still required.
