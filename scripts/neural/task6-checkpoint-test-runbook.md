# Task 6 checkpoint CPU test runbook

Status: **authored controls, not released for protected execution**. This is a
checkpoint-only test runner. It does not create a game environment, reset or
step Gym, draw policy actions, train, install packages, or write a checkpoint.
The protected CPU phase is already within the user's Tasks 1–7 approval; the
parent and independent reviewer must first release this exact source/control
snapshot as required by the Task 6 handoff.

## First-delivery checks

Run only the focused stdlib tests and the shell syntax check on the host:

```sh
PYTHONPATH=src python3.12 -B -m unittest \
  tests.unit.test_neural_artifacts \
  tests.unit.test_neural_checkpoint \
  tests.unit.test_neural_checkpoint_run_helpers -v
bash -n scripts/neural/run-task6-checkpoint-tests.sh
```

The gated Torch module is deliberately excluded here. Default discovery imports
no optional runtime prefix; the gated test class skips before importing Torch
only while the Task 6 gate is off. With the gate on, a Torch, sample-guard or
checkpoint-module import error is an error and cannot be counted as a skip.
`parse_container_command` passes the embedded `/bin/sh -n` check without
executing its body. `runtime_identity` performs read-only availability/hash
checks against the retained CPU venv, wheel lock, native extension and accepted
Gym source reconciliation. It does not invoke that venv or load Gym.

The wrapper derives the four required verbose test labels from the authored
`NeuralCheckpointTensorTests` class and checks them against the standard library
runner format. GREEN requires all four exact IDs, zero failures/errors/skips,
the pinned runtime identity marker and child exit 0. RED is accepted only for
either the exact missing `save_checkpoint`/`load_checkpoint` setup condition
after the runtime identity check, or four executed test methods with at least
one assertion failure and no skips/errors. Other nonzero exits, missing runtime
identity, setup/import errors and incomplete transcripts fail the wrapper.

## Protected phase after internal release

Run one invocation per fresh ID; never reuse an output or reservation ID. The
command tests only `tests.neural.test_checkpoint`:

```sh
SPIRE_LEARN_TASK6_REVIEWED=1 \
SPIRE_LEARN_TASK6_RUN_ID=checkpoint-red-unique \
SPIRE_LEARN_TASK6_EXPECT=red \
scripts/neural/run-task6-checkpoint-tests.sh

SPIRE_LEARN_TASK6_REVIEWED=1 \
SPIRE_LEARN_TASK6_RUN_ID=checkpoint-green-unique \
SPIRE_LEARN_TASK6_EXPECT=green \
scripts/neural/run-task6-checkpoint-tests.sh
```

The `red` example is only for a source-reviewed tensor-dependent RED phase. The
`green` example is only for the final compatible checkpoint implementation.
For each invocation the host reserves 64 rows from the existing shared Task 4/6
ledger before creating a container. The in-process Torch guard enforces the same
64-row ceiling before any multinomial call; these checkpoint tests should use
zero sampled rows. Reservations never refund, including when a container does
not start. There is no gameplay/action budget for Task 6.

The runner has an internal review gate and pins the retained CPU runtime:

- image `python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`;
- lock SHA-256 `51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0`;
- native extension SHA-256 `071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`;
- Python 3.12, `spire-lab 0.1.0`, NumPy 2.5.4, Gymnasium 1.4.0 and
  Torch 2.14.1+cpu, verified read-only inside the container.

Before reserving diagnostic rows or creating a container, the host inspects the
exact pinned image locally and checks its immutable ID, repository digest and
`linux/amd64` identity. `docker create --pull=never` prevents image acquisition;
a missing local image stops the run.

The container runs as UID:GID 1000:1000 with a read-only root, all capabilities
dropped, no-new-privileges, no network, four CPUs, an 8 GiB memory limit and 512
PIDs. The only writable host bind is the fresh `/run` child; `/tmp` is a capped
64 MiB noexec/nosuid tmpfs. Learner `src/`, `tests/`, the CPU venv and lock are
explicit read-only mounts; no `.git`, Gym source, home, credentials, socket,
setup parent or GPU is mounted. Before start, the host
asserts Docker's exact `Cmd`, `Entrypoint`, resource controls and complete mount
table from `docker inspect`. Container test execution is bounded to 120 seconds.
The wrapper sends TERM at 210 seconds and allows at most 90 seconds for
signal-safe stop/remove, source postcheck, reservation completion and evidence
closure. If create times out or returns an invalid ID, cleanup looks up only the
fresh exact container name, requires a single valid ID, inspects its exact ID,
image, command, run ID and ownership labels, and removes it only after that
identity check. A conflicting or unowned object is preserved and marks the run
failed. There is no retry.

The runner captures literal commands, separate stdout/stderr/exit files, source
and control hashes before/after, the container inspection, sampling evidence,
cleanup status, run summary and a closed evidence inventory over both the run
root and sibling `runtime-control/<run_id>` tree. The final manifest and its
command/stdout/stderr/exit bookkeeping live in a narrow sibling
`runtime-admin/<run_id>` output directory, explicitly excluded from its own hash
scope to avoid a self-changing inventory. All other wrapper and test evidence,
including the pre-inventory wrapper status, is covered. The qualified runtime
remains read-only. A missing or changed runtime identity, insufficient 30 GiB
free space, control/source drift, unexpected mount, container left after
cleanup, unacceptable test transcript or missing evidence makes that invocation
fail. Simulator and training evidence are not produced by this phase.

## Checkpoint payload contract for the next reviewed phase

The pure module currently validates `spire-learn-neural-checkpoint-v1` metadata,
identity mappings and byte inventories without importing Torch. Tensor save and
load APIs are intentionally deferred until the reviewed CPU TDD release.
Checkpoint directories will contain only `metadata.json`, `model.pt`, and, for
training state, `training.pt`. The JSON inventory names each payload's byte
length and SHA-256 and lists each named model tensor's name, dtype, shape and
content hash. Preflight requires compatible model/features/action/objective IDs,
an exact requested raw-mode mapping, matching model config hash, and (for resume)
matching full config hashes and hyperparameter identity. Evaluation and warm
start may carry different PPO hashes, modes already represented by a verified
mapping, seeds, partitions and budgets as transfer provenance. Resume additionally
requires explicit optimizer, RNG-domain, counter, update/version and seed/decision
sequence metadata; loading drops saved in-flight records and reports the dropped
count while returning `inflight_record_count == 0`.

Before any optional deserialization, the loader must verify directory contents,
safe filenames, strict JSON, identity compatibility, payload length and hash,
ZIP member paths, and that both aggregate stored bytes and aggregate uncompressed
archive bytes are independently no more than 256 MiB. Before any optional loader,
the metadata also checks each named tensor's explicit supported dtype, rank,
dimension and checked dense logical byte size; aggregate logical tensor bytes
have their own 256 MiB cap. Scalar shapes count as one element and zero-sized
shapes count as zero elements after every dimension and rank are bounded. Every
Torch payload must then load on CPU with
`torch.load(..., weights_only=True)`. The loader must reject non-tensors,
unexpected/missing tensor names, geometry/dtype/hash mismatches and non-finite
tensor values before applying any state. It must never use unrestricted pickle.
Checkpoint save must use a fresh owned temporary sibling and atomic publish,
preserving any pre-existing destination. These tensor rules are review criteria,
not execution claims in this first delivery.
