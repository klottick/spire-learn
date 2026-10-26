# Task 3 CPU native conformance runbook

This is the sole approved native workload for Task 3. The qualified CPU runtime
already exists at
`artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc`.
Do not rebuild, reinstall, pull an image, or use another venv. The two recorded
parent wrapper attempts both stopped before container creation and before any
test cohort. They remain in the lifetime attempt record. A later parent release
after scoped review must use a new unique run ID. The script never retries a
started cohort or an invocation that may have accepted native effects.

## Wrapper attempts and test cohorts

A wrapper attempt is a host launch of this script. A native cohort starts only
when the configured three-test unittest command starts inside the pinned
container. Wrapper attempt 1 exited 2 during host preflight and started zero
containers, zero unittest cohorts and zero native tests, resets, steps or
actions. Its exact command and result remain in
`artifacts/development/neural-combat/task-3/parent-native-invocation/`.
Wrapper attempt 2 exited 2 when Docker rejected the two writable bind-mount
arguments ending in `,rw`; Docker reported `invalid field 'rw' must be a
key=value pair`. It also started zero containers, unittest cohorts, native
tests, resets, steps or actions. Its exact command and result remain in
`artifacts/development/neural-combat/task-3/parent-native-invocation-v2/`, with
the container-create error and cleanup evidence under
`artifacts/development/neural-combat/task-3/runtime/task3-native-20261026-b/`.
This failed CLI parse does not authorize a replay after a cohort has started or
any accepted native effect. A future release for the still-unstarted cohort
requires review of this fix and a fresh run ID; it must not overwrite either
attempt record.

## Review gate and one invocation

The parent reviews the frozen learner source, scripted tests, conformance
configs, this runbook, the source tree digest, CPU lock/runtime hashes and Gym
source-reconciliation qualification before setting `SPIRE_LEARN_TASK3_REVIEWED=1`.
That internal gate is already authorized by the task; no further user approval is
needed.

Choose a unique run ID and invoke the script once from the repository root:

```sh
SPIRE_LEARN_TASK3_REVIEWED=1 \
SPIRE_LEARN_TASK3_RUN_ID=reviewed-20261026-a \
scripts/neural/run-task3-native-conformance.sh
```

The outer GNU `timeout` covers the entire host wrapper, including preflight,
tests, postchecks, cleanup and inventory finalization. It sends TERM at 210
seconds and KILL at 300 seconds. Cleanup ignores later TERM/INT only after
cleanup begins. Each Docker cleanup command has an 8 second timeout plus a 2
second kill grace; source and result postchecks have 15 second bounds. The script
never retries a failed invocation. The integration suite stops before another
reset or action at 120 seconds and records the elapsed time and overshoot. Its
4096 slot-action cap and 1024-per-fight cap are shared across all three tests.

## Host review and runtime gates

Before creating a container, the host verifies that the qualified
`$VENV/bin/python` entry is a symlink and that `readlink` reports the reviewed
text `/usr/local/bin/python`. This metadata check uses `[[ -L ... ]]` and
`readlink`; it does not resolve or execute the container-absolute target on the
host. The actual interpreter existence, readability, version and offline
package checks remain inside the pinned container. Before creating a container,
the script also requires 30 GiB free and a new run folder. It records
`git rev-parse HEAD`, the full dirty status, each SHA-256 for
all regular files under the mounted `src/`, `tests/` and `configs/` trees, and a
canonical digest of that complete tree. The canonical digest must match the
reviewed constant in the script. After the workload, it rehashes every file and
requires identical per-file hashes and tree digest. The accepted Gym source is
not mounted.

The host checks the exact CPU lock and extension hashes and reads the
source-reconciliation qualification. Executable package-version and offline
`pip check` gates run only inside the protected container. The host does not
execute the qualified venv, import `spire_lab` or reset an environment. The original
managed upstream worktree/manifest was removed externally after archival; its
original post-build source hash is unavailable and is not claimed. The durable
accepted-Git export was independently verified at 636 files with normalized Git
modes matching. Its reconstructed source-map hash is
`dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3`.

The loaded CPU extension must report Gym revision
`533dad1c8aec86df624b3742b6b3ad1dbf1d0033`, extension SHA-256
`071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`, and the
same locked package versions. This refers to the loaded CPU build, not advancing
Gym `main`.

## Container boundary

The script refuses an existing run ID or container name and never pulls the
pinned base image. Before start, it parses and asserts the actual container
inspection: the immutable local image ID; UID/GID 1000:1000; `/learner` working
directory; read-only root filesystem; no privilege; all capabilities dropped;
`no-new-privileges`; 4 CPUs; 8 GiB RAM; 512 PIDs; no network; and the 64 MiB
`/tmp` tmpfs with `noexec,nosuid`. The reviewed create argv uses
`--entrypoint /bin/sh IMAGE -ec COMMAND`, so the inspected entrypoint must be
`["/bin/sh"]` and the command must be `["-ec", COMMAND]`. The assertion checks
that exact command and requires its final line to invoke
`/work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_envs -v`.

Only `src/`, `tests/`, `configs/` and the qualified CPU venv are mounted
read-only. One fresh per-run writable child is mounted at `/run` and aliased at
`/learner/artifacts/development/neural-combat` so the already-authored config
output paths resolve inside that same child. Those are the only writable bind
mounts. Docker's `--mount` syntax makes a bind writable by default, so the
create argv omits the invalid bare `,rw` token while inspection still requires
both destinations to report `RW: true`. No source/private Gym tree,
credentials, Docker socket, `.git`, setup parent, or native build inputs are
mounted. The container runs with networking disabled. Its `pip check` and
runtime-version check run offline before unittest.

The working directory and workload are explicit:

```text
cwd: /learner
PYTHONPATH=/learner/src
SPIRE_LEARN_RUN_NEURAL_NATIVE=1
SPIRE_LEARN_TASK3_EVIDENCE_PATH=/run/task3-counters.json
/work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_envs -v
```

`make_driver` only constructs and authenticates each facade. The standalone
conformance first validates train, selection and final authored catalog metadata
through the public constructor/catalog descriptor without resetting those
entries. The three tests then exercise the selection probe entries and exactly
one normal `m10-ironclad-policy` Run. They perform no policy updates and never
search seeds. Mixed-vector conformance deliberately submits a batch containing a
terminal row, requires the driver's pre-dispatch rejection and unchanged
accepted-step/action counters, then masked-resets only that row. Run traversal
uses the one configured seed and public legal-random control; only a verified
victory counts as continue-after-combat evidence. A death or cap before victory
is reported as inconclusive, never reinterpreted or retried.

The native tests write aggregate machine-readable counters for attempted and
delivered slot actions, uncertain actions, reset calls/selected slots,
pre-dispatch rejections, lifecycle observations/settlements/outcomes, driver
construction, close calls/failures and the cohort fault latch. Host result
finalization also records unittest count, skips and exit status. Acceptance
requires exactly 3 tests, 0 skips, successful machine evidence, no cohort fault,
all three catalog splits validated, a passing source postcheck and an empty
post-cleanup container inventory.

## Logs, postchecks and closure

Every recorded host, runtime, image, container, native, result and cleanup command
has a literal command file plus stdout, stderr and exit files. Run artifacts live
uniquely under
`artifacts/development/neural-combat/task-3/runtime/<run-id>/`. The final
`evidence-inventory.json` hashes every other run artifact and is written only
after container cleanup and all other writers finish; it excludes only itself.
The final Docker inventory is captured after bounded removal. If any gate,
workload, postcheck or cleanup fails, preserve the logs and do not run this
script again automatically.
