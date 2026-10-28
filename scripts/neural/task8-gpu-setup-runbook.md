# Task 8 GPU setup runbook

This runbook describes the Task 8 controls. It is not phase authorization. Each
phase needs its own fresh parent release after review. The parent already
resolved the closure once, installed it offline, passed `pip check`, and invoked
the original RNG RED once. That attempt ran three tests but was a harness error:
both `ModelConfig` construction sites omitted required `max_parameters`, so the
two errors and one child failure are not product RED. Its output, inventory,
`.started` marker, and `container_started=0` field are historical and remain
unchanged. This fix round did not replay setup or invoke a container.

## Frozen inputs and exact candidates

- Python image: `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
  Resolution checks Linux/amd64, the canonical or qualified `RepoDigest`, and
  the image `.Id`; the run-scoped alias must keep that same `.Id`.
- CUDA Torch: `2.14.1+cu130`, CPython 3.12, Linux x86_64; URL
  `https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl`;
  advertised SHA256
  `5359b255ac68c0cabb8293952b38e00f0d1a570815cd87f19b61af702c071410`;
  retained HEAD size `554621169` bytes. The retained METADATA SHA256 is
  `f50fae9c2fd1e1d1e22f52426296a598741c51378935d0905c2c60288671729a`. This
  metadata is not a payload download or payload-hash verification.
- Direct pins: NumPy `2.5.4`, Gymnasium `1.4.0`, and the accepted Spire Lab
  `0.1.0` wheel. Its exact basename is
  `spire_lab-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`.
  It is mounted read-only at `/inputs/<that exact basename>` and referenced by
  the resolver input, closure provenance, and lock with the same filename.
  Neither the CPU venv nor Gym source/native artifacts are mounted into the new
  venv.
- Approved GPU: UUID `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`, NVIDIA
  GeForce RTX 5090, driver `595.91.07`, expected total `32607 MiB`. Every
  workload phase requires at least `12288 MiB` free before container creation.

## Source and phase releases

The source snapshot is separate from mutable report, release, and run evidence.
It covers the complete projected `src/` and `tests/` trees, including imported
learner contracts/schema/support, plus the Task8 runner, runbook, device check,
and helper. It records the exact path set, file bytes, and normalized executable
mode. Before container creation and after cleanup the runner recomputes the
snapshot. A changed or missing path stops the phase.

After the final fix5 source changes are complete, capture the fresh canonical
corrected-RED snapshot for same-reviewer and parent review. Keep every earlier
fix snapshot unchanged:

```bash
cd /home/klottick/repos/spire-learn
REPO_ROOT="$(pwd -P)"
TASK_ROOT="$REPO_ROOT/artifacts/development/neural-combat/task-8"
python3.12 -B "$REPO_ROOT/scripts/neural/task8_gpu_setup_helpers.py" \
  capture-source-snapshot "$REPO_ROOT" \
  "$TASK_ROOT/task-8-corrected-rng-reviewed-source-snapshot.json"
```

The runner accepts only canonical absolute, non-symlink release and snapshot
files inside the Task8 evidence root. `source_snapshot_sha256` in every release
is the SHA256 of the snapshot JSON file. The completed resolve/install phases
are retained under run ID `cuda-setup-20261028-a`; never replay either phase.

Each single-use release contains the following common fields and the exact
phase-specific fields described below:

```json
{
  "decision": "internal_task8_phase_release",
  "phase": "resolve",
  "preparation_review": "passed",
  "independent_review": "passed",
  "source_snapshot_sha256": "<sha256 of the reviewed snapshot JSON>",
  "single_use": true,
  "release_id": "<unique release id>"
}
```

Install also binds the resolve lock and setup budget state with
`closure_lock_sha256`, `setup_budget_state_sha256`,
`setup_active_seconds_before`, `setup_soft_remaining_seconds`,
`setup_hard_remaining_seconds`, and
`external_parent_review_pause_seconds`. The pause value is supplied by the
parent and kept separate from measured setup execution.

The one corrected RNG RED release is phase `rng-red` and additionally binds
`run_id: "cuda-setup-20261028-a"`, `attempt_id: "corrected-1"`,
`corrected_source_snapshot_sha256` (equal to the common source snapshot hash),
the retained `closure_lock_sha256`, and the exact original failed inventory:

```json
{
  "previous_failed_attempt_inventory_path": "/home/klottick/repos/spire-learn/artifacts/development/neural-combat/task-8/runtime-admin/cuda-setup-20261028-a/rng-red/evidence-inventory.json",
  "previous_failed_attempt_inventory_sha256": "7973eb6489daa2660f1f05921e4cd5a44217abe93cdb62144eed4873f74a297f"
}
```

The corrected attempt gets fresh paths under `rng-red-corrected-1` and the
exact container name
`spirelearn-task8-rng-red-corrected-1-cuda-setup-20261028-a`. Its `.started`
marker is separate; the original `rng-red.started` and evidence are never
removed or rewritten. There is no automatic retry. The optional argument is
accepted only for this exact run and RNG RED phase. New cleanup records include
`container_start_invocation_returned`, which says only that control returned
from the bounded start invocation; the original `container_started` field is
not reinterpreted.

Before `docker create`, the runner performs a bounded exact-name lookup and
requires a successful command with byte-empty output. A nonempty or malformed
result, daemon error, or timeout stops before create. If a name is claimed in
the race after the empty lookup, failed or uncertain create output does not
establish ownership. The create status and stdout remain recorded and cleanup
keeps a final exact-name listing, but it never recovers an ID by name. Only a
successful create returning exactly one full 64-digit ID establishes
ownership. Start and cleanup inspect that exact ID against the reviewed spec;
removal also requires a successful cleanup-time assertion and a one-use
authorization record, then removes that exact ID once.

RNG green and device-check releases must bind all of the following corrected
RED identities in addition to `run_id: "cuda-setup-20261028-a"` and their own
phase/source/lock fields:
`rng_red_attempt_id`, `rng_red_transcript_path`,
`rng_red_transcript_sha256`, `rng_red_inventory_path`,
`rng_red_inventory_sha256`, `rng_red_source_snapshot_sha256`,
`rng_red_closure_lock_sha256`, `previous_failed_attempt_inventory_path`, and
`previous_failed_attempt_inventory_sha256`. The runner accepts only the exact
canonical corrected transcript and inventory paths, validates their hashes and
contents, checks the corrected inventory is unchanged, and confirms its input
identity records the original failed inventory. The original failed RED path
is never a fallback. Device-check also binds the green transcript and
`rng_regressions: "passed"`.

The resolve and offline install are complete. After same-reviewer review of the
fix5 source and snapshot, and after the parent issues the one-use corrected RED
release bound to the canonical corrected source snapshot, invoke only this
correction. The runner checks the snapshot tree and release bindings before
claiming phase roots or its one-use marker. This is not a setup replay:

```bash
cd /home/klottick/repos/spire-learn
REPO_ROOT="$(pwd -P)"
TASK_ROOT="$REPO_ROOT/artifacts/development/neural-combat/task-8"
RUNNER="$REPO_ROOT/scripts/neural/run-task8-gpu-setup.sh"
SOURCE_SNAPSHOT="$TASK_ROOT/task-8-corrected-rng-reviewed-source-snapshot.json"
RUN_ID="cuda-setup-20261028-a"

"$RUNNER" rng-red "$RUN_ID" \
  "$TASK_ROOT/phase-releases/rng-red-corrected-1.json" "$SOURCE_SNAPSHOT" \
  --attempt-id corrected-1
```

After authentic corrected expected RED evidence and a separate product-source
review, only then may a product RNG fix be authored. Keep the corrected RED
evidence and its original failed inventory immutable. Freeze changed projected
source separately and obtain fresh later-phase releases. Their releases include
the corrected path/hash fields above and the old failed inventory path/hash:

```bash
cd /home/klottick/repos/spire-learn
REPO_ROOT="$(pwd -P)"
TASK_ROOT="$REPO_ROOT/artifacts/development/neural-combat/task-8"
RUNNER="$REPO_ROOT/scripts/neural/run-task8-gpu-setup.sh"
SOURCE_SNAPSHOT="$TASK_ROOT/task-8-post-rng-fix-source-snapshot.json"
RUN_ID="cuda-setup-20261028-a"

"$RUNNER" rng-green "$RUN_ID" "$TASK_ROOT/phase-releases/rng-green.json" "$SOURCE_SNAPSHOT"
"$RUNNER" device-check "$RUN_ID" "$TASK_ROOT/phase-releases/device-check.json" "$SOURCE_SNAPSHOT"
```

## Phase boundaries and evidence

Resolve is the only network-enabled phase. It writes the exact resolver input,
uses pip's vendored `packaging.tags` in the pinned CPython 3.12 Linux x86_64
container to audit target-compatible tags, performs the binary-only single
index resolution/download, captures wheel hashes/metadata, audits the complete
closure, and exclusively writes the hash lock. The closure records the runtime
identity and compatible-tag digest. Host control tests inject representative
compatible platform tags; host controls do not import pip or select new
dependencies.

Install is offline. It creates `/work/venv-cuda` with `venv --copies`, installs
only from the audited wheelhouse with required hashes, then runs `pip check`.
Both resolve and install set `TMPDIR=/work/tmp` and
`PIP_CACHE_DIR=/work/pip-cache`, creating them under the fresh owned `/work`
bind before pip runs. The Docker `/tmp` remains a 64 MiB tmpfs and is not used
for pip staging/cache.

Every phase or attempt has exclusive output, control, and admin roots. Its
immutable inventory is written only at
`runtime-admin/<run-id>/<phase-key>/evidence-inventory.json` after the phase is
closed; the corrected RNG attempt uses the distinct `rng-red-corrected-1`
phase key. Each inventory covers only that phase's roots and excludes only its
own inventory writer's four log files. Later phases reference prior
inventories, the lock, and installed venv by recorded identity, then verify the
earlier inventory bytes remain unchanged. The runner checks only that the
owned Docker config directory is empty; it never reads or hashes credential
config contents.

The retained CPU venv has exact pre-existing interpreter aliases at
`bin/python -> /usr/local/bin/python`, `bin/python3 -> python`, and
`bin/python3.12 -> python`. Total neural storage accounting counts only each
of those symlink inode sizes with `lstat`; it never resolves or traverses those
targets. Any target or path change fails the storage check. These historical
aliases are not phase-inventory records. Phase-owned venv inventory permits
only the known sibling `lib64 -> lib` link, with link text and `lstat` identity;
all other phase-owned venv links fail closed.

For workload phases, the runner verifies the single visible approved UUID and
free-memory floor using the retained host inventory, passes only that UUID to
Docker, and validates the container-visible GPU query inside the device check.
Docker inspect may represent absent `Devices` and non-GPU
`DeviceRequests` as either JSON `null` or `[]`; only those two absence shapes
are accepted. GPU phases require one exact UUID request with GPU capability.
Added devices, groups, or capabilities fail.

The device check performs one synthetic float32 forward/backward/Adam update.
The qualification has exactly one Adam attempt and separately records one
diagnostic call; it is not the later calibration probe. It records attempted,
accepted, and uncertain step counts, inspects tensor values in Adam state, and
writes failed status/fault evidence if anything fails after the step. It does
not call Gym or the learner, use AMP/compile/fallback, retry, or attribute
GPU-wide process memory to this check. The preserved 498 MiB process remains
unowned. The RNG RED parser requires initialized approved CUDA markers and the
two exact expected regression causes; missing/unavailable CUDA, load failures,
unrelated assertion text, skips, errors, and changed labels fail the gate.

## Time, storage, and stop conditions

- Dependency setup uses one cumulative `600` second soft / `1200` second hard
  allowance across resolve and install. Invocation preflight and cleanup count
  toward the active elapsed budget. Install receives the remaining allowance
  and a parent release bound to the resolve budget state; it cannot reset the
  timers. Any external parent review pause is reported separately and is not
  counted as measured setup execution. Each phase retains elapsed time,
  soft overshoot, remaining allowance, cleanup result, and outer timeout result.
- RNG RED, RNG GREEN, and device-check are separate invocations with `60`
  second soft / `120` second whole bounds, including preflight and cleanup.
  These limits do not include gameplay, samples, or calibration.
- Storage uses a conservative logical-byte measure: regular-file sizes and
  allowed symlink inode sizes, without following links. The 40 GiB neural cap
  is checked against all retained `neural-combat` contents, including the
  previous CPU setup and all Task8 outputs; no pre-Task8 baseline is subtracted.
  Task8-owned roots are counted separately against 24 GiB. Before resolution,
  reserve the full remaining 24 GiB maximum Task8 allocation and require the
  40 GiB total plus reservation to fit with at least 60 GiB free. After
  resolution, offline-install admission uses the sum of wheel ZIP
  uncompressed sizes, the largest expanded wheel for staging, and a bounded
  1 GiB temp/cache/venv overhead. This projection must fit the remaining 24
  GiB and 40 GiB caps before install. This is conservative accounting, not a
  filesystem quota; stop if projected or measured usage exceeds a cap. Later
  workload phases reserve 1 GiB for their bounded transcript, report, and
  final inventory before dispatch.
- Setup requires at least 60 GiB free disk; workload phases require 30 GiB.
  Every workload phase requires at least 12 GiB free GPU memory; the device
  check verifies the retained reading again inside the container. RNG GREEN
  requires the exact three reviewed test labels, all passing with zero
  skips/errors. Torch allocator peak is capped at 8 GiB, which is not
  driver-wide isolation.
- The runner never deletes old data, pulls an image, changes dependency/index
  versions, switches devices, retries an accepted action, stops unrelated
  services, or sweeps containers globally. Any identity, hash, resolver,
  closure, lock, source, budget, storage, VRAM, inspect, test-transcript, or
  device-report mismatch blocks that phase.

The retained GPU inventory has `2337 MiB` free while Ollama uses `29262 MiB`;
a separate `498 MiB` process (PID `3772`) must be preserved. Resolve that VRAM
prerequisite before releasing `rng-red`; changing only the later device phase
is insufficient. No runtime phase or calibration cohort is included in this
preparation delivery.
