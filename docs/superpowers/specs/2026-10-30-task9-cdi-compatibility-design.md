# Task 9 measured NVIDIA CDI compatibility correction

Source preparation only. The study at9cd716d was approved once, failed before
training at guest identity admission, and its claim is consumed. No replacement
training is authorized. Preserve that run's complete source/claims/inventory and
empty output directory. Current source fixes remain reversible; a new reviewed
source/approval packet is required before another study invocation.

## Evidence and intent

The two bounded metadata-only probes, with no optional runtime imports or learner
calls, found Toolkit1.20.0 (commit5505e2f94d9aaa08561490db974ba3cd676af209)
rewrites NVIDIA_VISIBLE_DEVICES from the requested GPU UUID to void and injects
NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu. The public installed CDI file
explicitly declares both. UID/GID1000 and the qualified venv interpreter match.
The original failure was a control/runtime contract mismatch, not failed learning.
Evidence: artifacts/development/neural-combat/task-9/experiment-package/execution-20261030-a/.

Installed /var/run/cdi/nvidia.yaml SHA256:
8b1609ef31b3c4a6fd60991ce491d6c59f427df669c7c6ee3fb93697fab691ec.
Installed /usr/bin/nvidia-cdi-hook SHA256:
12556bcbc901d5d698f126f38200c90553e4ddf71bfd768b7f5e6460b8cb454f.
Native/lock/image/GPU UUID/driver/model/PPO and study caps remain unchanged.

## Approach and boundary decisions

Keep the installed host toolkit/configuration unchanged. Correct the consumer to
recognize the pinned CDI post-OCI profile. Changing host runtime mode would alter
an independently managed service; broadly ignoring environment/mount differences
would lose the promised boundary. Neither is selected. This is a parent source
repair decision, not an assertion that the user approved new runtime exceptions.
The explicit replacement approval must disclose the measured exceptions below.

Use one small stdlib-only task9_nvidia_cdi helper and an authored runtime-profile
JSON; no YAML parser/dependency or second container framework. Bind its raw hash
in study.runtime_pins; host validation verifies CDI/hook input hashes before and
after work. Context/source/approval bind the profile acyclically. Preserve exact
Docker Config.Env, requested UUID, DeviceRequests, nonroot1000:1000, rootRO,
netnone, capdropALL, no-new-privileges and exact application bind table.

GPU guest validation may recognize legacy exact controlled values, or the exact
pinned CDI rewrite (void plus exact LIBCUDA_DIR). Other controlled values stay
exact; no arbitrary extra keys. Keep actual facts, never rewrite observed evidence
into an invented legacy state. Existing runtime GPU UUID/count qualification
remains independent and unchanged. CPU selection gets no CDI/device exceptions.

Successful parent-phase admission must retain its actual observations through
the existing captured guest stderr channel, bound to phase, context and profile.
Emit at most 16 KiB per record, with safe observed environment and
socket/scratch/params metadata; unknown environment values stay withheld.
Serialization, size or emission failure must stop before the learner factory.
Do not create application output before its empty-directory startup check or
change B's public interfaces/receipt schema to retain these observations.

Admit exact qualified RO driver/firmware/OpenCL files declared by this profile,
with type/access checks. Admit only the exact RO NVIDIA persistence-daemon socket,
which is driver IPC (not a Docker-management socket). Its observed socket UID105,
GID108,mode0777 and effective write access mean RO mounting does not make its IPC
API read-only. Document this explicitly; allow no other host socket.

The measured disable-device-node-modification hook creates one UUID-named
/run/nvidia-ctk-hook<uuid> tmpfs: root0:0,mode01777,4096-byte capacity, writable
access for UID1000,0 available bytes. Its sole nvct-params child is root0:0,
regular0444,not writable and bind-exposed at /proc/driver/nvidia/params.
Do not describe it as root-only or as one globally writable mount. Permit only
this exact runtime-managed bounded scratch class on GPU CDI: one full UUID path,
tmpfs type, capacity<=4096, expected ownership/sticky mode, exact immutable child,
no nested mounts/links, and matching params-file identity where observable. Record
inode capacity and effective access; no separate hard inode quota is claimed.
Container8GiB memory/PID limits remain; all operator-declared writable binds stay
exactly one owned output child. Refuse larger or arbitrary writable runtime mounts.
Additional observed CDI groups44/991 plus primary1000 are explicit metadata.

## Preservation-safe replacement output namespace

Original artifacts/.../experiments/task9-v1/training/standalone-a exists empty
and belongs to the first closed inventory. Never delete, rename or populate it.
Prepare replacement artifacts/development/neural-combat/experiments/task9-v1-attempt-b.
Keep current authored config filenames under configs/neural/experiments/task9-v1,
all case JSON/seeds/lineages/order/model/PPO unchanged. Update only authored
artifact routing in36 TOMLs and the study's canonical/raw config hashes/output
fields, validator and B/C output constants. Recompute through real loaders.
Old committed9cd716d and staged closed source remain the original inputs.
New campaign ID is task9-study-20261030-b; no claim/release/output is created now.

## Verification and remaining authority

Use source-only deterministic tests for exact env projection, captured-profile
semantics, mismatched hashes/UUID/libdir, CPU rejection, RO/type/ownership/socket
exceptions, 4096-capacity/sticky/child/link/nested-mount/inode qualifications,
unknown extras and preserved one-use/output namespaces. Actual raw metadata
remains ignored evidence; commit only small authored semantic fixtures/profile.
Exercise the real projected B/C flow; no Torch/Gym/GPU/native/container runs.
Preserve user foundation report/rootdiff and every original runtime record.
No training-budget, simulator/schema/reward or optimizer architecture change.

Primary references: NVIDIA CDI documentation:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
Pinned hook implementation:
https://github.com/NVIDIA/nvidia-container-toolkit/blob/5505e2f94d9aaa08561490db974ba3cd676af209/cmd/nvidia-cdi-hook/disable-device-node-modification/params_linux.go
