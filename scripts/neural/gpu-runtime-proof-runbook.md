# Task 9 bounded GPU runtime proof

Historical protocol: the approved `gpu-proof-20261029-e` invocation completed
with two passing tests and zero skips. Parent and independent evidence review
accepted its bounded result. This document does not authorize another invocation.
The exact executed runbook remains bound by the retained e source snapshot;
this status note was added after its successful source postcheck.

This is a one-cohort proof of the accepted Task 1 runtime API. It does not
authorize longer training, model selection, final evaluation, another seed, or a
repeat after any accepted effect. Parent source/runbook review and a matching
parent release are required before the wrapper can run.

## Workload and shared accounting

| Test | Config | Rollout | Max learned decisions | Max environment slots | Max reset slots | Max optimizer attempts |
|---|---|---:|---:|---:|---:|---:|
| test_01_standalone_b4_h8 | standalone-gpu-smoke.toml | B4/h8 | 32 | 32 | 36 | 4 |
| test_02_run_b1_h32 | run-gpu-smoke.toml | B1/h32 | 32 | 32 | 33 | 4 |

The single shared monotonic deadline starts in the host runner before source,
config, storage, image and GPU preflight. It is passed as an absolute monotonic
timestamp to the container. The integration class uses the earlier of that
timestamp and its own setup start plus 120 seconds. Both tests, both CUDA
qualifications, all checkpoint comparisons and every guarded effect use that
same deadline. The 300-second host limit includes cleanup and evidence writes;
the runner reserves at least 30 seconds for cleanup.

The cumulative ceilings are 64 step slots, 69 reset slots, 8 optimizer
attempts, 64 actor sample rows and 4 diagnostic rows. Each mode reserves its
two-row original/restored diagnostic pair before sampling either policy.
Counters reserve reset replacements, steps, actor rows and Adam attempts before
forwarding to their delegate. Rejected over-budget calls fail before delegation.
If the first test fails, the second method is recorded as halted before it
creates a runtime or performs another workload.

The saved elapsed prefix is the checkpoint counter snapshot taken at the save
boundary. It includes time up to that snapshot; checkpoint serialization after
the snapshot, runner finalization and runtime/writer close are outside the saved
prefix. A resumed training invocation subtracts that saved prefix once from its
configured wall allowance and still uses the earlier shared absolute deadline.
For this proof, that shared deadline bounds the entire attempt, including save,
finalization and close. The lifecycle receipt after runtime and writer close is
authoritative and must report a complete status, zero close errors and zero
wall overshoot. This qualification does not claim cumulative charging of
post-snapshot overhead across later invocations.

The configs preserve the train manifest, environment pins, model/PPO hashes and
seed streams from the accepted CPU configs. Each mode must make one finite
accepted update that changes weights and must save its final checkpoint.

The retained public frames from each training collection support deterministic
and one sampled original/restored action comparison on the same frame. Same
backend resume uses the original config and an inert driver; it checks full
action/log-probability/value/behavior-version equality, weights, Adam state,
device placement, Torch and Python RNG state, cumulative counters, policy
version, collector recovery and close-once. Portable CPU warm-start and frozen
evaluation use the existing CPU train and selection configs with inert drivers.
They compare imported weights exactly and deterministic outputs with 1e-4
absolute tolerance. Warm-start optimizer, RNG, progress and behavior version
are fresh; frozen evaluation has no optimizer and retains the checkpoint
behavior version. Neither path resets or steps an environment or performs a
CPU optimizer step. No extra diagnostic samples are permitted.

## Qualified inputs and resource limits

The host runner verifies the read-only Task 8 records and hashes before claiming
the release:

- CUDA lock: 92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45
- Native binary: 071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9
- Image: docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
- GPU: GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d, RTX 5090, driver
  595.91.07, compute capability 12.0 / sm_120
- CUDA stack: Torch 2.14.1+cu130, CUDA 13.0, NumPy 2.5.4, Gymnasium 1.4.0,
  SpireLab 0.1.0, Python 3.12

The image must already exist locally with the exact image ID. The runner never
pulls, rebuilds, installs or starts a service. Immediately before create it
requires at least 12 GiB free on the exact GPU and 30 GiB free on the repository
filesystem. It admits at most 40 GiB retained neural-combat data and a new
writable proof output child with at most 1 GiB of accounted output. The output
cap is enforced by known producer reservations, bounded host stream capture,
polling the actual output tree while the container runs, and closed-output
checks. It is an accounting/detection limit, not a filesystem quota.

Before create, the host creates the guest temp and cache targets below the owned
output child. They are available before the gated test imports optional packages
and are writable through the one output bind. `HOME` is not changed, and no home
directory bind is added.

| Container variable | Guest path | Host location |
|---|---|---|
| `TMPDIR` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/tmp` | `<output-child>/.runtime/tmp` |
| `XDG_CACHE_HOME` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/cache` | `<output-child>/.runtime/cache` |
| `TORCH_HOME` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/torch` | `<output-child>/.runtime/torch` |
| `TORCH_EXTENSIONS_DIR` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/torch-extensions` | `<output-child>/.runtime/torch-extensions` |
| `CUDA_CACHE_PATH` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/cuda` | `<output-child>/.runtime/cuda` |
| `TRITON_CACHE_DIR` | `/work/artifacts/development/neural-combat/task-9/gpu-proof/.runtime/triton` | `<output-child>/.runtime/triton` |

The 1 GiB admission calculation reserves 384 MiB for each mode's final
checkpoint, 64 MiB for each neural event log, 2 MiB for each captured host log,
16 MiB for metadata, and 96 MiB for temp/cache growth: 1,012 MiB total. The
remaining 12 MiB is headroom. The accepted checkpoint and event writers enforce
their own size limits; temp/cache growth has no filesystem quota and is polled
and counted with all other output.

The created container must inspect exactly as follows before start:

| Control | Required value |
|---|---|
| Image | Exact immutable reference and matching sha256 image ID above |
| User / workdir | Current non-root numeric host UID:GID / /work |
| Entrypoint and command | /runtime/venv-cuda/bin/python3.12 and -B -m unittest -v test_neural_gpu_runtime.NeuralGpuRuntimeProofTests |
| Network / root | none / read-only root filesystem |
| Capabilities | Drop ALL; no-new-privileges:true; no privileged mode |
| CPU / memory / PIDs | 4 CPUs, 8 GiB memory, 8 GiB memory+swap, 512 PIDs |
| GPU | One device request naming only the released UUID |
| Torch worker | 4 intra-op threads, 1 inter-op thread; allocator fraction capped at 8 GiB |

The only bind mounts are:

| Host source | Container destination | Access |
|---|---|---|
| src/ | /work/src | read-only |
| tests/ | /work/tests | read-only |
| configs/ | /work/configs | read-only |
| scripts/neural/gpu_runtime_proof_helpers.py | /work/gpu_runtime_proof_helpers.py | read-only |
| Task 8 requirements-cuda.lock | /runtime/requirements-cuda.lock | read-only |
| Task 8 venv-cuda/ | /runtime/venv-cuda | read-only |
| Task 8 _native.cpython-312-x86_64-linux-gnu.so | matching site-packages path under /runtime/venv-cuda | read-only |
| One new host output child | /work/artifacts/development/neural-combat/task-9/gpu-proof | read-write |

No whole checkout, home directory, Docker socket/config/credentials, Gym
checkout or unrelated artifact tree is mounted. Public Config, HostConfig,
Mounts, state and full ID inspection of this proof container is required and
permitted. The host Docker client credential/config directories, including
.token_seed, are never read or mounted.

Docker commands always select `unix:///var/run/docker.sock`, pass a fresh empty
task-owned client config directory, and run with a four-variable host
environment allowlist. The caller's Docker context, TLS, endpoint and config
overrides do not select the daemon. Every Docker CLI call uses those same global
flags, and create includes `--pull=never`.

## Snapshot, release, claim and one invocation

After parent review, create a source-only snapshot. It records source bytes and
Git modes for src/, tests/, configs/, this helper, wrapper and runbook; the
accepted Task 1 learner revision 7e37450c84f0893f5dc798ca4f23304c8b30fb14 and
the dirty-state hash; both proof configs and their manifests;
and the exact Task 8 lock, native, image, GPU and release-binding inputs.
Snapshot raw-file and canonical-JSON hashes are distinct fields. Mutable
reports and the run output child stay outside the source inventory.

The parent release must identify that exact snapshot by both raw and canonical
hash, the run ID, and the full limits map. The claim path must be new and
durable. The host runner validates the release and all inputs before the claim,
then fsyncs the one-use claim file and its parent directory. It will not
overwrite or reuse a claim.

The release object has exactly these binding fields in addition to its
schema-specific metadata: schema_id, status=released, release_id, run_id,
source_snapshot_raw_sha256, source_snapshot_canonical_sha256, and limits.
The limits object must equal the complete table in the helper. The durable
claim records the raw and canonical release hashes, both snapshot hashes,
release_id and run_id.

Invoke the wrapper once after the parent supplies those exact paths and run ID:

~~~bash
SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=1 \
SPIRE_LEARN_NEURAL_GPU_UUID=GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d \
SPIRE_LEARN_GPU_PROOF_REPO_ROOT=/absolute/path/to/spire-learn \
SPIRE_LEARN_GPU_PROOF_SOURCE_SNAPSHOT=/absolute/path/to/gpu-proof/source-snapshot.json \
SPIRE_LEARN_GPU_PROOF_PARENT_RELEASE=/absolute/path/to/gpu-proof/parent-release.json \
SPIRE_LEARN_GPU_PROOF_EXCLUSIVE_CLAIM=/absolute/path/to/gpu-proof/claims/release.claim.json \
SPIRE_LEARN_GPU_PROOF_RUN_ID=the-parent-released-run-id \
SPIRE_LEARN_GPU_PROOF_OUTPUT_CHILD=/absolute/path/to/gpu-proof/the-new-output-child \
scripts/neural/run-gpu-runtime-proof.sh
~~~

Use a non-root host account. The output child must not exist and must be below
the Task 9 proof directory. No wrapper retry is allowed. A failed validation
before create consumes no container. A failed or ambiguous create without one
full returned ID establishes no ownership and never permits name-based
deletion. After a full ID is returned, cleanup re-inspects that ID and name,
stops only that owned ID if needed, and removes only by that ID. The runner
queries exact-name absence before create and again after cleanup.

## Evidence and stop conditions

The host captures separate container stdout and stderr, requires the exact two
unskipped unittest methods and a zero exit, and rejects skipped, failed,
uncertain, incomplete, timed-out or zero-update results. It records the source
postcheck, full-ID cleanup result, final name absence, output/retained byte
counts, independently reads both closed mode result files, reconciles their
cumulative counters and creates a closed-file evidence inventory. Logs are
closed before inventory creation; no file covered by that inventory is appended
afterward. The mutable host control report is outside the inventory. Raw and
canonical hashes remain separately named.

Any failure ends this cohort. Preserve its accepted prefixes, claim, logs,
checkpoint and cleanup evidence; do not retry an environment action or rerun
the cohort automatically. A pass establishes only that this bounded runtime
path and its checkpoint comparisons completed on the qualified stack. It does
not establish learning efficacy, GPU compatibility beyond this path, a
selection result, original-game transfer or longer training authority.
