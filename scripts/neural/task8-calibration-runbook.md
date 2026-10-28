# Task 8 CUDA calibration continuation runbook

This is an authored control plan, not a runtime authorization. It prepares one
explicit continuation of the still-unstarted fixed Task 8 calibration cohort.
Parent source review, independent review, and a fresh release are still required
before invocation. This delivery does not start Torch, the CUDA runtime, a Docker
container, or Spire Gym. It does not install packages, download data, retry an
accepted native operation, or run an efficacy, selection, or final evaluation.

The previous v2 invocation is recorded as a failed one-test attempt, not as a
successful calibration or skipped work. It stopped at manifest identity
verification before CUDA initialization, model or driver construction, native
factory/reset/step, policy sampling, or optimizer calls. Its wrapper wall time
was 4.119860398 seconds; the host lifecycle boundary through cleanup and closed
inventory was 3.957788467 seconds. Those times describe the failed preflight,
not a cohort measurement. There have been zero calibration trials and no
performance result.

## Reviewed inputs

- Continuation source snapshot: `artifacts/development/neural-combat/task-8/calibration/calibration-source-snapshot-v3.json`.
- Parent continuation release: `artifacts/development/neural-combat/task-8/calibration/phase-release-v3.json`.
- Config and manifest: `configs/neural/calibration.toml` and
  `configs/cases/neural/train.json`.
- Existing CUDA venv: `artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a/install-work/venv-cuda`.
- Existing lock SHA256:
  `92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45`.
- Existing qualification report and summary:
  `task-8-gpu-qualification-report.md` and
  `task-8-gpu-qualification-summary.json` under the Task 8 evidence root.
- Accepted native identity SHA256:
  `071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`.
- Local image alias:
  `spire-learn/task8-python:cuda-setup-20261028-a`; the checked image must also
  carry the pinned Python image digest
  `sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
- Only approved device: UUID
  `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`.
- Required predecessor claim:
  `calibration/phase-release.claim.json`, SHA256
  `cd023e95b890cbd5f9b1c8604e8c2ed35001fe0644c12cd1af0a1fe47cde2164`.
- Required failed-run proof:
  `calibration/parent-invocation-v2/runtime-parent-proof.json`, SHA256
  `4ab61300d71d06ad2c4dc434a0c2741e8f61d162756871a52ea39e3350c1f5ee`.
- Required closed runtime inventory:
  `calibration/runtime-admin/calibration-20261028-b/evidence-inventory.json`,
  SHA256 `a3eb9438f730092b5ad29b24f4299efb04bafeeca682dbcf67e603b3fef4189e`.

The setup scripts and evidence from CUDA setup are immutable inputs. The
calibration runner uses `task8_gpu_setup_helpers.py` only for its established
hash, source snapshot, image, inventory, and storage primitives; it does not
replay setup phases.

The v1 and v2 snapshots, releases, consumed `phase-release.claim.json`, failure
records, and runtime trees remain immutable. The original claim is required
predecessor evidence. This is exactly one fixed continuation generation: v3
uses only `calibration-source-snapshot-v3.json`, `phase-release-v3.json`, and
the exclusive `phase-release-continuation-1.claim.json` marker. This control
never deletes, resets, renames, or reuses that marker, and defines no second
continuation generation, including after uncertainty or failure.

## Source snapshot and parent release

After the implementation is frozen, create a new source snapshot. It binds the
complete `src/` and `tests/` trees, the exact calibration config and manifest,
this runbook, this runner and helper, and the reused Task 8 setup helper. It
records paths, bytes, and executable modes. The snapshot output is exclusive;
old snapshots are never overwritten.

```bash
cd /home/klottick/repos/spire-learn
ROOT="$(pwd -P)"
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-8"
python3.12 -B scripts/neural/task8_calibration_run_helpers.py \
  capture-source-snapshot "$ROOT" \
  "$TASK_ROOT/calibration/calibration-source-snapshot-v3.json"
```

The parent release is created only after the parent source review and a fresh
independent review both pass. It must be a regular, canonical file inside the
Task 8 calibration evidence directory, and its JSON keys and values must match
this shape exactly:

```json
{
  "decision": "one_task8_calibration_cohort",
  "parent_source_review": "passed",
  "independent_review": "passed",
  "single_use": true,
  "release_id": "<unique-release-id>",
  "source_snapshot_sha256": "<reviewed snapshot file SHA256>",
  "runbook_sha256": "<reviewed runbook SHA256>",
  "calibration_config_sha256": "<calibration TOML file SHA256>",
  "train_manifest_sha256": "<train manifest SHA256>",
  "qualification_report_sha256": "<retained report SHA256>",
  "qualification_summary_sha256": "<retained summary SHA256>",
  "cuda_lock_sha256": "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45",
  "native_sha256": "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9",
  "previous_claim_sha256": "cd023e95b890cbd5f9b1c8604e8c2ed35001fe0644c12cd1af0a1fe47cde2164",
  "previous_failure_proof_sha256": "4ab61300d71d06ad2c4dc434a0c2741e8f61d162756871a52ea39e3350c1f5ee",
  "previous_inventory_sha256": "a3eb9438f730092b5ad29b24f4299efb04bafeeca682dbcf67e603b3fef4189e",
  "gpu_uuid": "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d",
  "learner_head": "<full 40- or 64-character lowercase Git object ID>",
  "learner_dirty": true
}
```

The host verifies that the retained qualification is still accepted, that the
exact lock and native identities remain unchanged, and that the source
snapshot, runbook, config, manifest, and three exact predecessor hashes match
the release. Before creating the claim or any container, it verifies the named
original claim, parent failure proof, and closed runtime inventory as regular
canonical non-symlink files with the exact hashes above. It checks the prior
claim's exact run ID and v2 release hash, the proof's one failed enabled test,
zero calibration/native/policy/optimizer effects, and successful cleanup and
removal. It also queries only `git rev-parse --verify HEAD` and
`git status --porcelain=v1 --untracked-files=all` on the host, with five-second
timeouts and optional Git locks disabled. It compares the full HEAD and dirty
boolean to the calibration-only release fields and retains no status paths or
source contents. The historical source-snapshot schema and format remain
unchanged. The generated `release-binding.json` retains the learner identity,
source-snapshot hash, parent release-file hash, continuation generation, prior
run/release identity, and all three predecessor evidence hashes. It does not
add those prior-run hashes to the runtime environment; the container receives
the existing source/config/manifest/qualification/lock/native release fields.
The final host lifecycle receipt copies those values and adds the release-
binding file hash after container cleanup and closed-output inventory. No `.git`
directory is mounted into the container. The host checks the fixed config
values before a container name or output directory is claimed. The release is
claimed once, after the exact container name is proven absent and immediately before the single
`docker create`, using the exclusive calibration-level marker
`phase-release-continuation-1.claim.json`. The old consumed claim stays intact
as evidence. A failed or uncertain create consumes the one continuation marker
and there is no retry path.

## Host preflight and container boundary

The wrapper requires an explicit fresh run ID and
`SPIRE_LEARN_TASK8_CALIBRATION_RELEASED=1`. The absent release file therefore
fails closed. It uses the retained installed venv and existing local image
only. Before container creation it verifies:

- Source and parent release hashes against the exact reviewed projection.
- At least 30 GiB host free space, 12 GiB device free memory, 40 GiB retained
  neural-combat storage headroom, and a total 1 GiB calibration-output cap.
  Existing wrapper/control/admin logs count toward that cap; initial admission
  reserves only the remaining bytes up to 1 GiB. The final storage check keeps
  its explicit 16 MiB inventory/receipt reservation and rejects if it does not
  fit.
- The approved UUID/name/driver and the pinned local image digest and alias.
- A successful exact-name container lookup with empty output.
- Fresh runtime, control, admin, and output paths; existing run IDs are never
  reused. The output directory must already belong to UID:GID `1000:1000`; the
  wrapper does not change ownership or permissions on retained paths.

The created container uses the pinned already-local image alias, `--pull=never`,
`--network none`, only the exact UUID via `--gpus device=<UUID>`, UID:GID
`1000:1000`, 4 CPUs, 8 GiB memory and swap, 512 PIDs, a read-only rootfs,
`--cap-drop ALL`, `no-new-privileges`, and a 64 MiB no-exec/no-suid `/tmp`.
There is no privileged mode, supplemental group, host socket, host home,
credential mount, broad checkout mount, or other device request.

Only these input mounts are allowed, all read-only:

- `/learner/src`, `/learner/tests`.
- The exact calibration TOML and train manifest at their canonical guest paths.
- The calibration control helper and reused Task 8 setup helper.
- The exact CUDA lock and retained qualification report and summary.
- The installed venv at `/runtime/venv-cuda`.

The sole writable bind is the fresh run-owned host output child mounted at
`/learner/artifacts/development/neural-combat/task-8/calibration/outputs`, the
exact output root resolved by the authored config. The root filesystem is
read-only; `/tmp` is a bounded tmpfs. Docker inspect is checked against the
full-ID create result and this complete mount/resource/device allowlist before
the container starts.

Inside that same container, the entrypoint first runs `pip check` and verifies
Python 3.12.15 plus exact package metadata for Torch `2.14.1+cu130`, NumPy
`2.5.4`, Gymnasium `1.4.0`, and Spire Lab `0.1.0`. It then runs exactly this
unittest label once:

```text
tests.integration.test_neural_calibration.NeuralCalibrationIntegrationTests.test_task8_cuda_calibration_runs_one_fixed_cohort
```

The gated test records its actual PID, package versions, Torch intra-op/interop
thread counts, BLAS environment, UUID gate, and reviewed input hashes before
calling a learner or environment factory. The host accepts only one executed
test, zero skips/errors/failures, and container exit code zero. Any mismatch
stops the attempt; there is no automatic retry.

## Deadline, ownership, and retained evidence

The learner has one shared 300-second soft budget across CUDA qualification,
trial creation, collection, comparisons, disposable optimizer probes,
serialization, and per-trial close. It reserves the approved final 210 seconds
for process teardown. The host wrapper applies one 510-second TERM deadline
and KILL after another 90 seconds, making 600 seconds inclusive of cleanup.
The deadline is not reset per trial or control command.

The cohort and resource limits stay fixed: 1,024 requested native action slots
under the existing absolute 2,048-slot reservation guard, at most three Adam
optimizer steps, 4 CPUs, and the existing 8 GiB host/container and 8 GiB GPU
allowances. The existing 30 GiB host free-space floor, 12 GiB GPU free-memory
floor, 40 GiB cumulative neural-storage bound, 1 GiB calibration-output cap,
16 MiB final inventory/receipt reservation, and owned-container cleanup checks
remain in force.

Only a successful `docker create` returning exactly one full 64-hex ID records
container ownership. The wrapper inspects that ID, verifies ownership labels
and the full container spec, and invokes `docker start --attach` once. Cleanup
inspects and re-verifies the recorded full ID, stops it only if it is still
running, and removes only that same ID. It never resolves or removes a
container by name after an uncertain create. The final exact-name listing is a
read-only audit. Container and host failures retain their logs and status.

After the container has stopped/been removed, all command and wrapper log
descriptors are closed. Only then does the wrapper write an exclusive
immutable inventory for its run, control, and output roots. The inventory is
kept in the separate per-run admin root; previous run records are never
overwritten or pruned. A final storage check enforces the 1 GiB output growth
and 40 GiB cumulative neural storage bounds, with 16 MiB reserved for the
closed inventory and final host receipt. The inventory writer rejects a larger
inventory. The separate `host-lifecycle.json` records total wrapper elapsed
time through container cleanup and inventory, with its boundary stated
explicitly. When release verification succeeded, it retains `learner_head`,
`learner_dirty`, `source_snapshot_sha256`, `release_sha256`, and
`release_binding_sha256` for later audit. The lifecycle's `continuation` object
records generation 1, prior run `calibration-20261028-b`, prior v2 release hash,
the exact claim/proof/inventory hashes, and the failed-before-effects outcome.

The wrapper captures `release_sha256` from that verified release binding once,
passes it to `claim-continuation` as `--expected-release-sha256`, and the helper
compares it with a fresh release verification before writing the exclusive
marker. It then checks the claim output against the captured value before
setting `CREATE_ATTEMPTED` or running `docker create`. A mismatch stops the
wrapper; a marker already written remains consumed. The exclusive writer fsyncs
the claim file and its parent directory in that order. A parent-directory fsync
failure is fatal before container creation and leaves the claim record intact.

The wrapper sends the inventory helper the required `--logs-closed` flag and
the two closed run/control roots. It sends the lifecycle helper the saved
numeric wrapper exit code and numeric `CLEANUP_FAILED` value (`0` or `1`),
followed by `--release-binding`. This matches argparse's accepted CLI values.
An inventory or receipt command error changes a zero wrapper result to `2` and
preserves any already nonzero wrapper result.

## One explicit invocation

Do not invoke this example until the source diff, this runbook, the frozen v3
source snapshot, the retained CUDA qualification, and the fresh parent release
have all passed parent and independent review. This authored delivery creates
neither the v3 snapshot nor parent release.

```bash
cd /home/klottick/repos/spire-learn
SPIRE_LEARN_TASK8_CALIBRATION_RELEASED=1 \
SPIRE_LEARN_TASK8_CALIBRATION_RUN_ID="<fresh-unique-id>" \
  scripts/neural/run-task8-calibration.sh
```

No Docker lifecycle command, device check, or calibration cohort was run while
authoring this control plan.
