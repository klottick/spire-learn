# Task 5 protected collection runbook

This is the first-delivery runbook for the persistent collector. The authored
core tests and wrapper control tests run on host Python 3.12 without Torch,
Gymnasium, NumPy, `spire_lab`, a Docker daemon, or a policy draw. The native
wrapper is held until internal parent and fresh independent review release it.
That release is an internal gate under the existing Tasks 1–7 CPU authority; it
does not request another user approval.

## Accepted inputs

- Learner source is the current review base `449d41fbd4383bd01c4c974c38e3d4e9c43a1fcc` plus the frozen Task 5 diff. The wrapper records the actual HEAD, dirty status, mounted source/test/config hashes, exact config hashes and a source tree digest before and after execution.
- Gym is the accepted `533dad1c8aec86df624b3742b6b3ad1dbf1d0033` build with common-view fingerprint `02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78`. Run mode requires `m10-ironclad-policy`, Run observation/replay v3 and fingerprint `30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8`.
- Use the already qualified runtime at `artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc/data/venv-cpu`. The venv and `requirements-cpu.lock` are separate read-only mounts. Lock SHA-256 is `51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0`; loaded Gym native SHA-256 is `071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`.
- The pinned local image is `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`, used with `--pull=never`. The protected command verifies `spire-lab 0.1.0`, Gymnasium `1.4.0`, NumPy `2.5.4`, Torch `2.14.1+cpu`, the lock hash and native hash while running inside the container. The CPU report records the accepted Gym source-map hash `dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3` and its verified 636-file durable source export. The original source's post-build hash was unavailable and is not claimed.
- The container receives `src/`, `tests/`, and `configs/`. The single existing
  helper file `scripts/neural/task5_collection_control.py` is separately
  bind-mounted read-only at exactly
  `/learner/scripts/neural/task5_collection_control.py`. The container also
  receives the read-only CPU venv and lock file, and one fresh Task 5 runtime
  child. No broader `scripts/` directory, checkout/home/credential parent,
  `.git`, Gym source/fixture, socket or writable source mount is exposed. The
  exact helper source, destination and read-only bit are part of inspected
  mount-set equality; missing, writable, wrong-source, duplicate or extra
  mounts fail before start. The wrapper's pre/post source snapshots already
  include the helper file hash. The fresh child is mounted at `/run` and at
  `/learner/artifacts/development/neural-combat`; both mounts must resolve to
  that same child. It is never the runtime setup parent.
- A settlement `ResourceCheckpoint` carries the owned immediate successor frame and primitive `visit`, `outcome`, `encounter`, `hp_before_healing`, and `hp_after_cleanup` copied from that same immutable public `Lifecycle.last_exit`. Tasks 6/7 should consume these fields directly; the collector does not infer prior HP from a frame or apply game rules.

## Bounded workload

The only native unittest module is `tests.integration.test_neural_collection`. It
contains exactly these two opt-in methods:

```text
tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_run_natural_task_end_without_run_reset
tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_standalone_manual_resets
```

The disabled gate is checked in `setUpClass` before product/runtime imports.
After that gate, setup creates exactly one `Budget(120.0, time.monotonic)` for
the whole two-test cohort, before importing the collector, driver, config or
model. Both test methods share the same Budget object; neither mode resets its
clock. The harness sets `rollout_steps=1` for each test so it can preserve exact
before/after action boundaries while calling the persistent collector.

Each mode has a 2,048 slot-action ceiling, for at most 4,096 attempted slot
actions across the suite; the accepted experiment caps may stop it earlier.
The tests make no optimizer updates. Standalone must observe a natural task
result and a later actual delivered manual reset, with reset-operation, reset-
slot, seed-ordinal and completed-task counts around that reset. This particular
gate makes no mixed-row continuity claim unless its evidence shows one.

The pure gate-off availability regression builds a temporary layout matching
the reviewed container paths: symlinked `src/`, `tests/`, and `configs/`, plus
only the copied control helper at
`/learner/scripts/neural/task5_collection_control.py`. It runs outside the
repository working directory with `PYTHONPATH` limited to
the mounted `src/`, blocks imports of Torch, NumPy, Gymnasium, `spire_lab`, and
`spire_learn`, and verifies that both native method IDs are discoverable while
the opt-in flag is absent. The class skips before a test body runs, and the
helper resolves from the single-file mount. This is import-availability and
gate-isolation evidence only; it does not inspect or start a container or
enter the runtime.

The Run method uses the explicitly named `PublicLowestLegalDiagnosticPolicy`,
not the untrained neural network. Its behavior is fully public and deterministic:
it reads only each combat frame's public int8 `action_mask`, selects the minimum
legal action index, reports a point-mass distribution (`log_probability=0.0`),
uses a constant-zero value estimate, and reports behavior version
`public-diagnostic-lowest-legal-v1`. The distinct identity is
`spire-learn-public-diagnostic-lowest-legal-v1` with action ID
`spire-learn-public-lowest-legal-action-v1`; the policy specification hash is
`120e6ce8fafd10b2b4468c6640635a5da5ffb71c7cd38b087edba65bf51345fc`. It makes
no policy RNG draw and receives no seed, route or hidden state. The exact policy
identity and statistics are included in the evidence validator. This policy
may fail to win; such a run is reported as a failed gate. The test does not
search seeds, fabricate a result, or reset the Run before a victory.

The Run evidence gate requires a victory checkpoint whose native `terminated`
and `truncated` flags are both false, then the immediately following delivered
action from that checkpoint's exact successor frame, with the same stream and
Run episode, a new decision ID, and no intervening reset. It records both
boundaries' visit/outcome/frame/HP/native flags, complete before/after counters
and reset counts, and the immediate checkpoint hash. The test stops after that
continuation. It does not restart after a true Run terminal. In normal later
Task 7 collection, a truly terminal Run is reset only on a subsequent collector
call when the full batch fits the remaining action/decision capacities and the
shared time budget permits it. Seed and decision ordinals remain cumulative;
Run task and started-visit IDs include the Run episode so visit zero in a new
Run is a new task. This remains B=1 collection, not Python Run vectorization.

The accepted cohort is one `unittest -v` invocation with a shared 120-second
soft Budget and a 300-second whole-run ceiling. The wrapper sends TERM at 210
seconds and KILL after a further 90 seconds. Its bounded cleanup child deadlines
sum to 44 seconds: exact-name recovery inventory 8, removal 10, final name
inventory 7, source snapshot 7, source comparison 5, and final evidence
inventory 7. A further 45-second shell/filesystem reserve keeps the documented
worst-case cleanup budget at 89 seconds. Cleanup disables EXIT recursion and
ignores later INT/TERM signals. Inventory output and final status files are
written outside the run child being inventoried. The result gate requires both
full stdlib `TextTestRunner` IDs exactly once, two executed tests, zero
skips/failures/errors, exit 0, valid per-mode evidence, no optimizer updates,
no simulator fault, actual manual-reset and Run-victory/continuation evidence,
and the combined slot-action bound. Missing boundary evidence is a failed or
inconclusive collection, even if the process exits 0.

## Invocation after review release

The first invocation must use a fresh run ID. The reviewer records release
internally; then invoke once from the repository root:

```sh
SPIRE_LEARN_TASK5_REVIEWED=1 \
SPIRE_LEARN_TASK5_RUN_ID=collection-20261026-a \
bash scripts/neural/run-task5-collection.sh --run
```

`--help` is always zero-effect. `--run` exits before filesystem or Docker
operations unless the internal review marker is present. The wrapper refuses a
reused run ID/name, requires at least 30 GiB free, reads only the venv Python
symlink text on the host, verifies the local image without pulling, and checks
the exact name is absent. After the local `docker create --help` parser proof,
it creates one UID/GID `1000:1000` container with a read-only root, all
capabilities dropped, `no-new-privileges`, four CPUs, 8 GiB memory, 512 PIDs, no
network and a 64 MiB `noexec,nosuid` `/tmp`. It inspects the exact container and
mount set before start, including the separately mounted read-only control
helper file at its exact reviewed source and destination. Inspection includes
`.Config.Cmd` and requires exactly
`["-ec", EXACT_REVIEWED_COMMAND]` with Entrypoint `["/bin/sh"]`; the expected
command is extracted from the wrapper's reviewed heredoc, and altered command
text or argv is rejected before start.

All Docker calls have timeouts. The wrapper removes only its fresh exact
container ID/name, verifies that exact name is absent, compares post-run source
and dirty-state snapshots, then inventories the run child. Run summary, cleanup
status, final inventory output and wrapper exit are stored outside the child,
after inventory, so exact cleanup outcomes are retained without self-inclusion.
Outer commands, stdout, stderr, exit codes and final status live under
`artifacts/development/neural-combat/task-5/runtime-control/<run-id>/`;
container output, runtime identity, native collection evidence, classification
and other run-child files live under
`artifacts/development/neural-combat/task-5/runtime/<run-id>/`.

The runtime identity's `cuda_available: false` is an inferred qualification
from the pinned CPU-only package/image; it is not a runtime measurement. This
delivery does not query CUDA.

No container creation, inspection, start, native import, reset, step, gameplay,
policy sampling or runtime cohort belongs to the first delivery. Simulator
collection remains subject to RNG-001 and TIME-001: it is not original-game
transfer evidence.
