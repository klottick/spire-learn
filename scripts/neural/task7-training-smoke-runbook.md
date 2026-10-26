# Task 7 CPU training smoke runbook

Status: **the single approved invocation completed once; no further execution is
authorized by this runbook**. The two mode tests passed with zero skips. The
run completed in 41.328 seconds and recorded 396 learned decisions, 444
environment slots, 28 accepted optimizer steps, and two updates per mode, with
no faults, uncertain effects, or wall-budget overshoot. This is plumbing evidence
only; it is not a playing-strength result, and no frozen-policy evaluation or
GPU run is claimed. Parent verification matched 105 inventory entries, 127
mounted input files, and all 20 reviewed source files. Task 7 acceptance and
final combined review remain pending with the parent. See the [retained runtime
report](../../artifacts/development/neural-combat/task-7/task-7-training-smoke-runtime-report.md).

At the pre-release fix-round source freeze on 2026-10-26, the retained CPU setup
report was historical and native conformance remained a separate parent gate.
The wrapper had not yet been invoked at that point. No Docker lifecycle, venv
command, Torch import, Gym environment, native reset/step, or training smoke
was run during that fix wave.

## Historical protocol for the completed invocation

This section records the one-time protocol used for the completed run. It does
not authorize a repeat.

The completed protected invocation was one combined unittest command. It ran
exactly these two methods, in the test module's standard order:

```text
tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_run_bounded_training_and_reload_parity
tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_standalone_bounded_training_and_reload_parity
```

The exact host command that was executed once was:

```sh
SPIRE_LEARN_TASK7_REVIEWED=1 SPIRE_LEARN_TASK7_RUN_ID=training-smoke-20261026-a bash scripts/neural/run-task7-training-smoke.sh
```

Before that invocation, `SPIRE_LEARN_TASK7_REVIEWED=1` and the unused unique
run ID were insufficient by themselves. The wrapper also required the parent-authored
`artifacts/development/neural-combat/task-7/parent-smoke-release.json`. That
record said `internal_cpu_smoke_release`, pinned the exact Task 7 source
snapshot, marked first-delivery and fresh independent review passed, and marked
Task 3 native conformance and Task 5 native collection accepted. The helper
rejected any release record whose source snapshot hash did not match.

The one-time wrapper used the already retained, offline-verified runtime:

- Runtime location: `artifacts/development/neural-combat/cpu-runtime-location.json`.
- Venv: `artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc/data/venv-cpu`, mounted read-only at `/work/venv-cpu`.
- CPU lock SHA-256: `51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0`.
- Native extension SHA-256: `071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`.
- Runtime image: `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`; creation uses `--pull=never` and verifies Docker's canonical `python@sha256:...` local digest alias first.

The wrapper refuses less than 30 GiB free space. It verifies the venv link,
lock, native binary and exact container inspection. The container uses UID:GID
1000:1000, a read-only root, no network, four CPUs, 8 GiB memory, 512 PIDs, all
capabilities dropped, no-new-privileges, and a 64 MiB noexec/nosuid `/tmp`.
There is no GPU, Docker socket, host home, credential path or Gym source mount.
The runtime venv and lock, learner `src/`, `tests/` and `configs/`, and the
single `scripts/neural/task7_training_run_helpers.py` file are read-only binds.
The helper file is mounted at
`/learner/scripts/neural/task7_training_run_helpers.py` because gated
`setUpClass` imports its thread recorder from the `/learner` working directory.
The wrapper and inspect gate require that exact host file, guest file path and
read-only bind; they do not mount the host `scripts/` directory. Its hash is
already covered by the Task 7 source-control snapshot. The only writable bind
is a fresh output child mounted at `/learner/artifacts`; it contains the two
run output roots and no setup parent.

The container starts `/bin/sh -ec`. Before unittest imports the integration
module or constructs a Gym driver, one in-container preflight verifies Python
3.12.15, Torch 2.14.1+cpu with `torch.version.cuda is None`, NumPy 2.5.4,
Gymnasium 1.4.0, six BLAS/OpenMP environment caps at four, preflight-process
Torch intra-op threads four and inter-op threads one, and a successful
`pip check`. Its identity record is explicitly labeled as the preflight
process; its process-local Torch settings do not stand in for the unittest
process. The runtime identity is written to the fresh output child and the
preflight/version/pip markers are captured in the test transcript.

When the gated test class starts in the actual unittest process, `setUpClass`
imports Torch, reapplies and verifies four intra-op and one inter-op thread
before model or environment factory construction, then records that process ID,
versions, thread values and environment in the same runtime-identity artifact.
It emits a separate training-process marker before either smoke method starts.
Transcript parsing and final artifact validation require both process records
and reject a preflight-only record. These checks all use the same container and
venv as the smoke.

The in-container unittest command used for the completed invocation was:

```sh
SPIRE_LEARN_RUN_NEURAL_NATIVE=1 SPIRE_LEARN_RUN_NEURAL_TENSOR=1 \
PYTHONPATH=/learner/src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_training -f -v
```

The runner shares one 120-second monotonic soft deadline across both modes,
starting at integration class setup. Each mode receives only the remaining
time, and an exhausted deadline stops before another collection or update. The
container attach ceiling is 170 seconds so the runner can persist artifacts and
close normally. The host wrapper
enforces the approved 210-second TERM point and 90-second cleanup grace, for a
300-second whole-run ceiling including evidence closure. Each mode is capped at
256 learned decisions, 1,024 environment slots, 16 optimizer attempts and two
rollout updates; the update limit is checked before another collection starts.
Together they cannot exceed 512 learned decisions, 2,048 environment slots or
32 optimizer attempts. Each successful mode requires two updates, positive
accepted optimizer steps, attempts equal accepted steps, and zero uncertain
steps. Thirty-two is a ceiling, not an exact work target.

Each mode exports one checkpoint and compares deterministic reload outputs on
public frames retained from that mode's training rollouts. It also requires
finite final named parameters with at least one changed parameter. In each of
the same two tests, a real saved CPU checkpoint is reopened with
`create_runtime(purpose="resume")` against a fresh fake driver. The regression
compares model, optimizer, RNG domains, counters, policy version and collector
recovery without reset, step, sample or optimizer update. This is not another
native cohort. REDO, rerun, retry, alternate seeds, extra evaluation and
post-failure resumption are prohibited for this smoke.
Any missing method, skip, failed test, cap violation, checkpoint/parity issue,
unexpected mount, cleanup uncertainty, source drift or incomplete evidence
stops the invocation as failed. Keep all outputs for parent diagnosis.

## Evidence and cleanup

The wrapper installs ownership-aware cleanup before per-run mutations and uses
the tested output-scaffold helper to create every nested output parent. It
records literal command, stdout, stderr and exit files for each host control
and the exact in-container transcript. Before any container create it captures
Git HEAD/full dirty state, Task 7 controls and rulings, the two preserved user
files, and every regular file plus Git-compatible mode under the read-only
mounted `src/`, `tests/` and `configs/` trees. Symlinks and unsupported entries
are rejected. It checks the pinned image without pulling it, requires the
immutable image ID and linux/amd64 platform, then verifies the created
container by its full ID, exact command, ownership labels, resource controls,
empty privileged/device fields and complete mount table before start.

After the test command, the wrapper inspects the container exit state, validates
the two exact method IDs, fail-fast behavior, preflight ordering and zero skips,
and reconciles two run summaries, lifecycle receipts, smoke-check records, the
same-container package identity and two complete checkpoint directories.
Combined learned decisions, environment slots and optimizer attempts must
remain inside the global caps. `summary.json` stays exit-code 1 while the
exclusive `lifecycle.json` receipt is pending; the receipt written after runtime
and writer close is authoritative for final status, elapsed time and overshoot.
Its timing boundary is before its own write. It removes only a
container whose full ID and ownership/configuration match this run ID. If
`docker create` times out or returns an invalid ID, exact-name lookup may find
zero or one full ID; an uncertain or mismatched container is preserved for
manual review. A final exact-name listing must be empty.

Source HEAD, working-tree status, full mounted-input hashes/modes, Task 7 file
hashes and preserved-file hashes must match after cleanup. The final inventory covers the run and control trees
after their streams close. Only the narrow final inventory command/output/error/
exit files are excluded from their own inventory. The qualified venv, lock and
runtime remain read-only. The shared Task 4/6 diagnostic sampling ledger is not
used by this workload; actual training decisions are counted by the runner and
checked against the 512-decision global ceiling.

Do not rerun this wrapper or smoke command. This runbook records the completed
single invocation; no further execution is authorized. Static source review and
pure helper tests alone do not count as native, tensor, optimizer, checkpoint
execution, or training evidence.
