# Task 4 CPU tensor-test runbook — first-delivery candidate

This runbook and `run-task4-tensor-tests.sh` are authored for parent review.
The first delivery has run only focused host Python 3.12 checks and the
zero-effect Docker `create --help` parser proof described below. It has not
created, started or inspected a container, executed the qualified venv, imported
Torch/Gym/native code, or run a tensor test. Parent must review the corrected
runbook, wrapper, inspect fixture, test files and source hashes, then internally
release the already-authorized Task 4 CPU TDD phase. The wrapper's
`SPIRE_LEARN_TASK4_REVIEWED=1` gate records that release; it is not a request for
another human approval.

## Qualified inputs

Use only the existing runtime at
`artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc`
and its `/data/venv-cpu`. Do not rebuild, reinstall, update, or execute that
venv on the host. The wrapper reads only the Python symlink text and hashes the
CPU lock/native extension before Docker. The lock SHA-256 is
`51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0`; the
native extension SHA-256 is
`071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`.

The locally present base image is pinned as
`docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
The workload uses `--pull=never`. Inside the container, offline `pip check`
and package identity checks must match `spire-lab 0.1.0`, `gymnasium 1.4.0`,
`numpy 2.5.4`, and `torch 2.14.1+cpu`. No package install or network access is
part of this run.

Gym provenance is the accepted source/package revision
`533dad1c8aec86df624b3742b6b3ad1dbf1d0033`, source-map hash
`dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3`, and
636-file durable export with normalized Git modes matching. The original
managed source/manifest was removed externally after archival, so its original
post-build source hash is unavailable and is not claimed. The runner verifies
that qualification and does not mount any Gym checkout, source tree, build
input, fixture, or native artifact into the container.

## Exact one-shot invocation

After parent release, use a fresh run ID and invoke once from the repository
root:

```sh
SPIRE_LEARN_TASK4_REVIEWED=1 \
SPIRE_LEARN_TASK4_EXPECT=red \
SPIRE_LEARN_TASK4_RUN_ID=cpu-red-20261026-a \
bash scripts/neural/run-task4-tensor-tests.sh
```

After the RED result and implementation review, a later GREEN invocation must
use `SPIRE_LEARN_TASK4_EXPECT=green` and a different fresh run ID, for example
`cpu-green-20261026-a`. Each invocation is one-shot and reserves its own
64-row maximum from the shared lifetime ledger.

The wrapper rejects an existing run directory/name and never reuses logs,
container IDs or reservations. It records the current Git HEAD, complete dirty
status, each regular-file hash under mounted `src/`, `tests/`, and
`configs/`, and a canonical input-tree digest. Before reservation or container
creation it also requires at least **30 GiB free** on the run filesystem and
hashes the host runner, helper, inspect checker and this runbook. It records the
qualified CPU and Gym provenance, then verifies the same inputs, control files
and runtime identity after execution. `.git` is not mounted; the container
receives only the captured read-only inputs.

The wrapper reserves 64 rows in the shared append-only ledger before Docker
create, using reservation ID `task4-<run-id>`. The lifetime allowance is 4096
sampled rows shared by Task 4 and Task 6. Reservations are permanent, including
failed and uncertain invocations. Every later invocation needs a new ID and a
new reservation; a started or uncertain unittest cohort is never retried by the
wrapper. The in-process guard installs one `torch.multinomial` hook for all
three test modules, counts leading batch rows times requested samples before
calling Torch, and refuses any draw past the same 64-row ceiling. It writes
attempted, completed, and refused counts atomically under the owned writable
run child. The host appends those actual counts separately to
`artifacts/development/neural-combat/diagnostic-sampled-row-ledger.jsonl`;
if a process stops while a sample call is in flight, the ledger records an
uncertain completion and leaves the completion count null. It does not refund
reservations when a run fails.

## Container boundary and exact test command

The wrapper uses Docker's local `create --help` path with the reviewed flags,
mounts, environment and command. `--help` precedes the image reference, so this
parser proof makes no daemon/container request. This proof has already exited 0;
Docker printed `Usage: docker create [OPTIONS] IMAGE [COMMAND] [ARG...]`. The
raw command, help output and exit status are retained in
`task-4-logs/docker-create-help-001.*`. An earlier helper invocation failed
before Docker because its Python import path omitted the repository root; that
failure is also retained. The corrected helper produced the successful
zero-effect proof. After parent release, the runner repeats the local parser
check, then uses the same flags for a real `--pull=never` create and inspects
the fresh container before start.

The container runs as UID/GID 1000:1000, with a read-only root filesystem, all
capabilities dropped, `no-new-privileges`, four CPUs, 8 GiB memory, 512 PIDs,
no network, and a 64 MiB `/tmp` tmpfs with `noexec,nosuid`. Docker uses
`--entrypoint /bin/sh IMAGE -ec COMMAND`, which must inspect as entrypoint
`["/bin/sh"]` and command `["-ec", COMMAND]`. Read-only bind mounts are
`src/`, `tests/`, `configs/`, and the existing CPU venv. The only writable
bind is a fresh run child, mounted at `/run` and aliased at
`/learner/artifacts/development/neural-combat`; both destinations must report
the same source and `RW: true`. The alias is never the setup data parent. No
home, credential, socket, `.git`, source-Gym, or writable learner-source mount
is present.

After offline runtime identity checks, the bounded command is:

```text
SPIRE_LEARN_RUN_NEURAL_TENSOR=1 PYTHONPATH=/learner/src /work/venv-cpu/bin/python -B -m unittest discover -s tests/neural -v
```

The wrapper sets `SPIRE_LEARN_TENSOR_TASK=task4`, the unique reservation ID,
`SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64`, sampling evidence at
`/run/tensor-sampling.json`, and model metrics at `/run/model-metrics.json`.
OpenBLAS is limited to one thread; OMP, MKL, NumExpr, and Torch use at most four.
The protected unittest command has a 120-second TERM limit plus a 15-second
kill grace; attached container start has a 140-second TERM limit plus a
5-second kill grace. The complete wrapper has a 210-second TERM limit plus a
90-second cleanup grace and re-execs through `bash`, so the script need not be
executable. Cleanup ignores later INT/TERM only after finalization starts.
Container removal, final-name inventory, source postcheck, ledger finalization,
cleanup/summary writers and the final inventory writer all have explicit
timeouts. Their timeout ceilings plus kill graces total 78 seconds, within the
90-second outer cleanup grace.

The tests use learner-authored frames. They may construct and close the
qualified public facade only to obtain/authenticate its descriptor; they do
not call reset or step, dispatch actions, sample a gameplay rollout, or train a
policy. The model-size case reports the actual accepted-descriptor default
model's parameter count, bytes per authored frame, B1/B4 CPU forward and
backward times, and peak process memory. It rejects more than the effective
10,000,000-parameter cap or process use over 8 GiB. This is a tensor correctness
and resource check, not gameplay or learning evidence.

## Evidence and first-delivery hold

The wrapper stores unique command, stdout, stderr, and exit files under
`artifacts/development/neural-combat/task-4/runtime/<run-id>/logs/`, plus
preflight, source-postcheck, result files, authored-frame metrics, sample guard
evidence, cleanup status, run summary and a final SHA-256 inventory in that same
run child. Before the evidence inventory closes the child, `test-result.json`
records actual unittest IDs/counts/skips/failures/errors, expected-case
coverage, sampling evidence and model metrics. `cleanup-status.json` records
the actual remove exit, final exact-name inventory exit/output and whether the
resource is absent, present or unknown. `run-summary.json` records actual
command exits and both result objects. The final inventory call is the last
run-child writer, is not itself logged into that child, and sends command,
stdout, stderr and exit files to the sibling `runtime-control/<run-id>/` area.
No run-child files are written afterward.

There are **27** expected authored tensor test IDs, derived from the three
gated test modules and matched against narrow unittest discovery without
optional imports. A GREEN classification requires all 27 IDs exactly once,
zero skips/failures/errors, exit 0, coherent nonzero completed sample evidence
within 64 rows with no refusals, and accepted-schema model measurements. A
complete RED may classify only as a behavioral assertion failure or explicit
missing product API. Missing optional runtime/packages, incomplete discovery,
loader failures and skips are separate non-behavioral outcomes.

This first delivery ran only the host Python 3.12 pure feature-contract RED,
focused config/frame fixtures, stdlib sampling-ledger/guard/run-result checks,
source syntax/import parsing, shell and embedded `/bin/sh` syntax parsing,
authored inspect fixtures, expected-test inventory validation and the permitted
Docker `create --help` parser proof. All 27 tensor tests remain authored and
gated; none were executed. The qualified venv, Torch/Gym/native packages, tensor
RED/GREEN, model measurements and sampled draws remain unrun pending parent
review/release. The real shared ledger remains at zero reserved rows.
