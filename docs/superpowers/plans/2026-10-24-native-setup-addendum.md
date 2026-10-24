# Task 6 Native Setup Implementation Plan Addendum — Stage A Preflight Authority

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`
> only after explicit user approval of this exact addendum. Do not spawn nested
> workers.

**Status: Stage A CPU-only Task 1 preflight approved, including the exact narrow
fixture-overlay exception recorded below. Task 6 (`spire_learn-d8l.7`) remains
incomplete. This approval authorizes only Task 1 source/container preflight and
its conditional pull/use of the two exact pinned images. It does not authorize
Task 2 wheel build, Task 3 package download/install, Task 4 integration tests, or
any Task 7/TorchRL/GPU/training/shared-service work.**

Approval record (local record time `2026-10-24T21:16:02+00:00`; user message in
the parent session): “yes, as long as that decision is fine long term”. The
approved scope is the Stage A CPU-only Task 1 preflight with only the fixture
directory overlay exception below. All other pins, source mounts, and gates in
this addendum remain unchanged. Conditional image pulls are permitted only after
all source, resource, storage, Docker and mount preflight checks pass. No Task 2
build, package installation, runtime import, integration test, service change,
or cleanup is authorized here.

## Goal

If separately approved, create the Gym-owned native wheel in a disposable,
network-enabled build container; resolve a CPU-only learner runtime in a second
disposable container; and run only the opt-in Task 6 public-interface conformance
tests offline in that same Python runtime image. Keep all generated artifacts in
a newly created, ignored learner work directory for review. Do not claim Task 6
complete unless the actual integration evidence and remaining acceptance gates
are separately reviewed.

## Architecture

Use **two pinned images and three phases**: (1) build the Gym wheel in the pinned
official Maturin manylinux image, (2) resolve/install a CPU-only dependency
closure in the pinned Python runtime image, and (3) run conformance offline in
that same Python runtime image. The manylinux build image is not the candidate
runtime: its glibc baseline cannot establish that a wheel runs on the selected
slim Python image. Host-venv setup is rejected because verified current controls
do not provide a bwrap/source-filesystem read-only guarantee. Installing Gym's
default dependencies is rejected because its Torch dependency can select CUDA
packages. No Dockerfile, Compose file, learner dependency metadata, shared
service, or source helper is proposed.

## Tech Stack and pinned inputs

- Learner source HEAD observed: `6561a575318c6a8b5cf68fa8aae8423f8939bfd1`.
  Preserve the existing dirty report and untracked review diff; record actual
  learner dirty state at approval/execution time, not an assumed clean tree.
- Gym clean source revision observed: `4874fb398128c142bd56f10a12e6fca939a88f93`.
  Python requirement is `>=3.12,<3.13`; Gym Cargo MSRV is 1.89. Gym
  `pyproject.toml` declares NumPy `2.5.4`, Torch `2.14.1`, Gymnasium extra
  `1.4.0`, and build backend requirement `maturin>=1.15,<2`. Do not adopt Gym
  dev tools or `uv.lock`.
- Build image:
  `ghcr.io/pyo3/maturin@sha256:0d1711efd2d3aab276f738bdcda48d378bceef2eeb7f8a8caf59451defddee72`
  (observed v1.15.0 linux/amd64 manifest). Use its
  `/opt/python/cp312-cp312/bin/python`. Rust toolchain version is unverified;
  preflight must prove existing Rust >=1.89. No rustup installation/bootstrap.
- Runtime image:
  `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`
  (observed Python 3.12.15 slim-bookworm linux/amd64). Prove runtime glibc
  >=2.28 before installing packages. Both runtime Rust/ABI assumptions remain
  preflight checks, not established facts. The runtime image does not need Rust.
- CPU Torch wheel exact URL/hash, confirmed from the official CPU index:
  `https://download-r2.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl`
  SHA-256 `5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f`.
  Download only this exact file after approval, verify its digest before use.
  Pin `torch==2.14.1+cpu`; public `==2.14.1` accepts this local version under
  PEP 440, but still inspect wheel metadata and final resolution. Never use
  PyPI CUDA fallback, `--extra-index-url`, or upgrade substitution.
- NumPy `2.5.4` and Gymnasium `1.4.0` have public Python 3.12-compatible
  metadata; this is not runtime verification. Verify actual wheel metadata,
  dependency closure, and `pip check` in the approved isolated runtime.

## Spec / Plan authority

This draft is subordinate to [M1 foundation design](../specs/2026-10-24-foundation-design.md)
§§5, 7–8 and [foundation implementation plan](2026-10-24-foundation-implementation.md)
Task 6 Steps 5–7 and the shared interface ledger. It does not change the
approved M1 scope, Gym ownership, interfaces, or Task 7 gate. The canonical
ignored continuity ledger is `artifacts/development/foundation/progress.md`;
Task 6 report is `artifacts/development/foundation/task-6-report.md`.

## Global Constraints

- Human approval of **this exact written addendum** is a prerequisite, not
  implied by parent review or any earlier M1/Task 6 authority. Until then, no
  setup, image-pull, dependency, build or native-runtime commands may run.
  Read-only planning checks and already-authorized local documentation commits
  are permitted. Approval is limited to this Stage A CPU setup and Task 6
  conformance below.
- No host Python environment/package installs, Rust/OS installs, Gym checkout
  edits, lockfile copies, learner dependency edits, copying Gym source into
  learner-authored source, fixture copying, training, game-asset/VOD/model
  downloads, Dockerfile/Compose, or shared Beads service changes. Preparing this
  draft makes no tracker changes; after setup approval, Task 6 bookkeeping
  continues under the existing learner-only Beads authority described in
  Task 5. No GPU, CUDA, TorchRL, vector probe, or Task 7 work; later
  framework/native-vector work needs a separate gate.
- Do not author/copy Gym code into learner source or build a second simulator.
  Read-only Gym inputs and consumption of the resulting own Gym wheel are
  permitted within this proposal; do not copy fixture catalogs, private
  diagnostics, hidden RNG/state, or privileged inputs. Consume only the
  documented public interface and legal masks. Genuine death remains an
  outcome; simulator faults are errors; never retry an accepted environment
  action. Preserve RNG-001/TIME-001 qualifications.
- Persistent new images and caches are within explicit approval scope; no
  automatic prune. Use Docker client/server 29.7.2 only as the observed local
  read-only query context; that observation authorizes no other daemon action.
- A new unique ID creates an exclusive root under
  `artifacts/development/foundation/runtime/<unique-new-id>/` (ignored). Never
  reuse, overwrite, or clobber. Stages: `build/`, `wheelhouse/`, `venv/`,
  `logs/`, `provenance/`, and generated `requirements-native.lock`.
- New-storage budget is a **20 GiB soft approval budget, not a quota**. Before
  each phase check and record >=30 GiB free; record free space and growth,
  including image/cache growth. Stop at 20 GiB growth or resource failure and
  ask; never expand budget or prune to make room. No claim of hard disk/network
  caps. Preserve existing images and shared resources.
- Each container is named uniquely, `--rm`, `--platform linux/amd64`,
  `--read-only`, caller UID:GID, `--cap-drop ALL`,
  `--security-opt no-new-privileges`, `--cpus 4`, `--memory 8g`,
  `--pids-limit 512`; bind only the owned work root read-write and reviewed
  build inputs read-only. No Docker socket, credentials, home, checkout root,
  GPU, privileged mode, or extra capabilities. If mount protections cannot be
  proven before build, stop without starting a build.
- **Approved narrow fixture overlay exception:** the otherwise approved
  read-only `$GYM/crates/spire-env` mount remains unchanged. In the disposable
  Task 1 source-preflight container, use the same source projection intended for
  the release-build container, and at the matching container path
  `/gym/crates/spire-env/src/tests/fixtures`, place an empty kernel tmpfs mount
  with read-only mode (Docker CLI form: `--tmpfs
  /gym/crates/spire-env/src/tests/fixtures:ro,size=64k,mode=755`). This obscures
  only the existing fixture directory inside that disposable container; do not
  bind/copy host fixture files or change the Gym checkout. Prove the overlay is
  empty and read-only from container mount metadata, directory listing and
  ownership/mode inspection; do not write a canary to Gym. The approved source
  crate mounts remain read-only. The overlay is only for the release-build
  source projection; future `cargo test` must use the original checkout fixtures
  without this overlay. If a normal release source/API or build script begins
  requiring these test fixtures, stop for review rather than approximating or
  supplying fixture content. Cost if this source-based assumption is wrong:
  build inputs would be hidden and the release build would fail, requiring
  review; Gym source and upstream tests remain unchanged.
- Define `$GYM` as the absolute read-only Gym checkout path, `$WORK` as the
  absolute unique ignored learner runtime root mounted at `/work`, and `$UID` /
  `$GID` as the invoking non-root host user's numeric IDs. Use a fresh
  work-root-local `DOCKER_CONFIG` for anonymous pulls; never mount host Docker
  credentials or change the global Docker context/daemon. If the selected
  context or anonymous registry access fails, stop; do not inspect environment
  secrets or try alternate contexts/credentials.
- Network is available only in explicitly approved build/dependency-download
  phases, limited by host policy to GHCR/Docker Hub, PyPI and files.pythonhosted.org,
  official PyTorch CPU redirect, and crates.io/index/static.crates.io. Docker
  does not enforce this DNS/host allowlist: this is policy, not a firewall claim.
  No network in conformance or gameplay.
- Per-build wall cap 30 minutes including graceful stop; total native session
  cap 10 minutes. At expiry stop only the run's own named container, wait and
  inspect; no continued dangling work and no rerun/retry of an accepted native
  action. Wall timeout is a diagnostic error, not a loss, truncation, or
  and does not remove or alter the project's TIME-001 qualification.

## Review Focus — principal hazards

1. **Source mutation/leakage:** narrowly mount only reviewed files read-only;
   no full checkout, `.git`, `.beads`, `.venv`, Cargo `target`, fixture catalog,
   raw fixtures, host binary, or copied lockfile. Stop on escaping symlinks or
   generated native binaries before mounting.
2. **Wrong ABI/runtime:** manylinux builder is not the runtime. Establish Rust
   MSRV, runtime glibc, Python tag, wheel tags and actual import only through
   approved preflight/conformance; never infer from image labels.
3. **CPU Torch provenance:** exact hash-pinned CPU wheel and audited closure;
   no index fallback, CUDA/NVIDIA/Triton packages, or broad upgrades.
4. **Scope/time/storage creep:** unique ignored output, bounded commands and
   resources, stop-and-ask on cap; no prune or work on Task 7/TorchRL.
5. **Evidence semantics:** opt-in tests must actually be discovered/executed;
   faults/timeouts are not losses, and skipped/blocked tests are not green.

## Proposed execution plan (only after explicit approval)

### Task 1 — Authority, source and container preflight

- [ ] **Preflight:** record explicit addendum approval; define `$GYM`, `$WORK`,
  `$UID`, `$GID`; confirm new unique ignored root, source state, images, mounts,
  resource bounds and storage/network policy before any build.
- **Files/artifacts:** create only the unique ignored runtime root and its
  stage directories; write approval, command/image identities, source revisions
  and dirty states, lock hash, free-space/growth, and pre/post source status to
  `provenance/` and `logs/`. Do not alter tracked report/diff or canonical
  progress ledger.
- **Inputs/mounts:** inspect Gym root read-only. Verify its exact HEAD and clean
  state against the recorded revision; compute Cargo.lock SHA-256
  `3fd2a3fcd907f3dd25fcc79e9fb338e51e0f04c5610c0f74d40d6d53708e32d6` without
  copying it. Scan allowed trees for escaping symlinks/private paths and
  `.so/.pyd/.dll`; any finding stops. Allowed source mounts only:
  `$GYM/Cargo.toml`, `$GYM/Cargo.lock`, `$GYM/pyproject.toml`; directories
  `crates/spire-core`, `crates/spire-env`, `crates/spire-python`,
  `crates/spire-cli`; and `python/spire_lab`, each mounted at the same path
  under `/gym` and read-only. Apply only the approved empty read-only tmpfs
  overlay at `/gym/crates/spire-env/src/tests/fixtures` described above; this is
  not a host mount or a Gym filesystem change. Add top-level README/license only
  if packaging demonstrably requires them; stop for any other unreviewed source
  input.
- **Container checks:** pull only the two pinned images after approval; verify
  linux/amd64 identity. Check existing builder Rust >=1.89 and runtime Python
  3.12.15/glibc >=2.28; Rust is required only in builder. Use root-read-only plus read-only source mounts and,
  where necessary, root-owned tmpfs `/gym` parent for mount points; non-root
  must not write `/gym`. Inspect Docker mounts only (no env/secret dump), and
  prove source mounts read-only by mount metadata/stat/access checks, never by
  writing a canary into Gym. Define isolated `CARGO_TARGET_DIR=/work/build/target`,
  `CARGO_HOME=/work/cargo`, `TMPDIR=/work/tmp`, `HOME=/work/home`; bytecode off.
- **Pass/stop:** proceed only when all source, image, ABI, resource, identity,
  writable-workspace and read-only-mount checks are evidenced. Any mismatch,
  inaccessible mount, insufficient free space, or needed input outside the
  table stops before build. Do not bootstrap Rust or patch source.

### Task 2 — Build only the Gym-owned wheel

- [ ] **Build and audit:** build the configured own Gym wheel without changing
  source; verify exact wheel identity, metadata, dependency list and digest.
- **Files/commands:** use the reviewed exact mount table; work output only in
  new `build/`, `wheelhouse/`, `logs/`. Explicit container name, `--rm`,
  hardened flags above; Maturin needs `/bin/sh` entrypoint for preflight, then
  invoke `/usr/bin/maturin` explicitly (or its verified absolute path). In
  `/gym`, command template:

  ```sh
  maturin build --release --locked \
    --manifest-path /gym/crates/spire-python/Cargo.toml \
    --interpreter /opt/python/cp312-cp312/bin/python \
    --target-dir /work/build/target --out /work/wheelhouse
  ```

  Honor the actual `pyproject.toml` Maturin configuration, including
  `python-source=python`, `module-name=spire_lab._native`, and its exact
  `features=['extension']`; do not substitute a guessed inner PyO3 feature
  name or override configured features manually. Pass only
  `SPIRE_REVISION=4874fb398128c142bd56f10a12e6fca939a88f93`
  as the documented build revision; no `.git` mount or forged dirty metadata.
  `CARGO_TARGET_DIR`, Cargo home, temp and home remain in work root. Network is
  allowed for Cargo lockfile-specified downloads; do not claim initial build is
  offline. Workspace members are required, but do not build all workspace
  packages.
- **Verify:** inspect wheel filename/tags, `METADATA`, version, Python ABI,
  platform/glibc requirements, dependencies and build metadata. Reject any
  unexpected package/version/tag or dependency; do not patch Gym or learner
  metadata. Record compiler, Python/image IDs, source SHA/dirty state, lock SHA,
  wheel SHA-256 and metadata. The wheel's compatibility with runtime remains
  unverified until Task 3/4.
- **Pass/stop:** exactly the expected Gym wheel, metadata and hashes pass audit;
  any build error, source write, unexpected wheel or dependency stops. Do not
  retry a failed build automatically or modify source; stop and report it.

### Task 3 — Resolve and lock the CPU-only runtime closure

- [ ] **Resolve, pin and install:** hash-verify the exact CPU Torch wheel before
  use; resolve only the pinned CPU dependency closure, generate its hash lock,
  install offline and verify it.
- **Files/mounts:** runtime image sees learner `src/` at `/learner/src:ro` and
  `tests/` at `/learner/tests:ro` only (no learner repository root). It sees
  reviewed wheelhouse and owned venv/requirements outputs read-write. Never
  mount host-native package artifacts. Create venv only under `/work/venv` with
  image Python's built-in venv/pip; do not upgrade global pip. Set
  `PYTHONDONTWRITEBYTECODE=1`.
- **Resolution:** obtain the exact CPU Torch wheel URL above once after
  approval, using `pip download --no-deps` or a standard-library verified
  stream into the owned wheelhouse; verify SHA-256 immediately before any use
  or install. No alternative URL/file. Pin exactly `torch==2.14.1+cpu`,
  `numpy==2.5.4`, and `gymnasium==1.4.0`, plus the audited own Gym wheel.
  Generate `/work/constraints-native.txt` with exactly these three lines:
  `numpy==2.5.4`, `torch==2.14.1+cpu`, `gymnasium==1.4.0`. Resolve using
  binary wheels only. After auditing the unique Gym wheel, set
  `GYM_WHEEL=/work/wheelhouse/<exact-wheel-filename>` by matching wheel
  `METADATA` Name `spire-lab` and Version `0.1.0`; require exactly one match
  and validate its filename/tags and recorded digest. Do not select an arbitrary
  glob match. Use that variable in closure resolution:

  ```sh
  pip download --only-binary=:all: --find-links /work/wheelhouse \
    --constraint /work/constraints-native.txt --dest /work/wheelhouse \
    "$GYM_WHEEL" gymnasium==1.4.0
  ```

  Never `--extra-index-url`, allow CUDA fallback, source-build, or silently
  substitute/upgrade a pin. Audit every resolved wheel and dependency closure;
  reject duplicates, unexpected names/versions, `nvidia-*`, CUDA, Triton,
  TorchRL, or any dependency not justified by audited metadata. Generate a new
  learner runtime `requirements-native.lock` containing exact names/versions
  and SHA-256 for each wheel payload. This is not a copied Gym Cargo/uv lock.
- **Install/verify:** install only from this wheelhouse with
  `--no-index --find-links /work/wheelhouse --require-hashes -r
  /work/requirements-native.lock`. Generate the lock from exact wheel
  METADATA versions and SHA-256 payload hashes for the complete closure, then
  confirm offline resolution/install and `pip check`. Record the venv's
  built-in pip version separately; do not globally upgrade pip.
  Run `pip check`; record interpreter, pip, installed versions and CPU Torch
  metadata (`torch.version.cuda is None`). Do not initialize CUDA or require a
  GPU. Any hash mismatch, unresolved dependency, unexpected package, CUDA
  selection or pip-check failure stops without fallback.
- **Pass/stop:** closure fully audited, hash-locked, offline-installable and
  pip-check clean. Keep the lock, wheels, logs and venv in the unique ignored
  root for review; no automatic cleanup.

### Task 4 — Add opt-in Task 6 conformance and run offline

- [ ] **Test and conformance:** baseline-check the expected integration module
  absence if applicable, author the two permitted integration paths, run the
  core suite and the exact opt-in tests, and record all counts/results.
- **Files:** only after approval, author minimal `tests/integration/__init__.py`
  if needed and `tests/integration/test_native_envs.py`; no product code or
  setup helper. Keep default unittest discovery safe without optional imports;
  native tests are safe when opt-in flag is unset; isolated subprocess checks
  verify default discovery/import does not load optional/native modules. Missing
  flag is not a native pass.
- **Exact test cases/bounds:** author exactly these three new test methods in
  `test_native_envs.py`: `test_pack_actions_native_int64` (no environment),
  `test_starter_cultist_public_contract`,
  `test_starter_jaw_worm_public_contract`. The scenario tests each create one
  environment, reset once, and close exactly once in `finally`; each checks
  public descriptor/profile, observations/masks, action behavior, delivered
  outcome/reward evidence, and already-loaded documented native `build_info()`.
  Use the two scenarios `starter-cultist-a0-v1` and `starter-jaw-worm-a0-v1`,
  selection partition, independent environment and policy seed 0, at most 128
  accepted steps per scenario and 256 total. `pack_actions` checks native-endian
  int64 only; no vector implication. No final cases, fault injection, catalog
  enumeration, vector/TorchRL integration, or assertion every case wins or
  completes before cap. A valid ongoing episode at the cap can pass the checks
  actually performed; that is not episode completion or victory.
- **TDD/evidence:** before authoring, confirm baseline core suite is 116 tests;
  run it before and after integration additions and report native-gated skips
  separately from core passes. Test a real compatibility failure diagnostically;
  do not manufacture a RED result or treat setup failure as test failure evidence.
  With the opt-in flag, require exactly 3 discovered, 3 executed, 0 skipped and
  successful assertions; never infer pass from an exit status/test count alone.
- **Execution:** run only the approved integration command in the runtime image,
  from `/learner`, with network disabled and approved venv:

  ```sh
  SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=/learner/src \
    /work/venv/bin/python -B -m unittest tests.integration.test_native_envs -v
  ```

  Run ordinary stdlib tests as already authorized by the existing approved plan;
  no added permission is required. Capture actual discovered/run/skipped counts,
  exit status and logs. Container
  wall cap 10 minutes; stop only its own named container at expiry.
- **Pass/stop:** each selected integration test was discovered and executed;
  all assertions pass and no source checkout changes occurred. A delivered
  simulator fault/error is diagnostic, never loss; accepted environment actions
  are never retried. Timeout, blocked import, skip, zero tests, ABI failure,
  uncaught fault, or missing required check is not a pass. The 128-step cap
  itself does not fail a case if all bounded conformance checks have passed;
  record ongoing status without claiming completion/victory. Timeout remains a
  diagnostic under existing RNG-001/TIME-001 qualifications. Record outcomes
  and limits honestly; setup alone is not Task 6 completion.

### Task 5 — Review evidence and gate closeout

- [ ] **Evidence and closeout:** append verified execution evidence to the
  canonical ledger and Task 6 report without overwriting existing content;
  parent review and approved-plan acceptance determine Task 6 status.
- **Files (future approved execution only):** after approval, write new ignored
  runtime provenance/logs and append evidence to the canonical ledger and Task 6
  report while preserving existing modified contents and the root untracked
  review diff. Tracker updates are limited to the learner `spire_learn`
  namespace (`BEADS_DIR` absolute to the owned `.beads` directory,
  `BEADS_DOLT_SERVER_DATABASE=spire_learn`); parent acceptance may close Task 6
  under existing authority. Preparing this document performs no Beads writes.
  Local documentation commits are already allowed; implementation commits
  follow tests/review; never commit runtime payloads. Only the two integration
  test paths are new source files proposed.
- **Record:** approval text/time; exact command arguments and container names;
  image digests/platforms; source revision/dirty before and after; mount table;
  Rust/Python/glibc facts; lock/wheel hashes and metadata; per-phase free-space
  and storage growth; installed closure; test names/counts/statuses; accepted
  step totals, outcomes/faults, timeouts and remaining unverified claims.
- **Pass/stop:** once this written setup addendum has been approved and its
  checks completed, parent reviews actual evidence and applies the approved
  plan's existing acceptance/closeout process; no extra user decision gate is
  added by this addendum. Task 7/TorchRL remains a separate gate. No push, broad
  cleanup, runtime-payload commit, or broader work is included.

## Evidence sources

Read-only local sources: root `AGENTS.md`, `README.md`, `docs/reference/README.md`,
M1 foundation spec §§5, 7–8, foundation plan Task 6 and shared interface ledger,
`artifacts/development/foundation/progress.md`, and Task 6 report. Task 6 report
records actual controller suite result (116 tests) and current public API/source
inspection; it explicitly says native runtime remains unverified.

Official/public evidence consulted for candidate facts (not proof of local
runtime behavior):

- Python venv: https://docs.python.org/3.12/library/venv.html
- Maturin documentation/distribution: https://www.maturin.rs/ and
  https://github.com/PyO3/maturin
- Gym public source metadata and locked dependencies: local Gym revision and
  files; source manifest/API docs in `docs/reference/spire-gym/`.
- PyPI JSON metadata: https://pypi.org/pypi/numpy/2.5.4/json and
  https://pypi.org/pypi/gymnasium/1.4.0/json
- Official PyTorch CPU wheel index:
  https://download.pytorch.org/whl/cpu/torch/
- PEP 440 local version matching: https://peps.python.org/pep-0440/
- Registry digests: read-only GHCR/Docker registry metadata for the exact image
  digests recorded above. Reconfirm digest/platform only after authorization.
- Docker Engine tmpfs mount syntax and behavior:
  https://docs.docker.com/engine/storage/tmpfs/ (read-only `ro`, `size`, and
  `mode` options; a mount obscures the underlying directory in the container).
  This public documentation does not replace verification of the actual local
  container mount metadata and empty directory.

Context7 quota was exhausted during evidence collection; official project
documentation and public registry metadata were used as fallback. Candidate
package metadata, Rust version, glibc, wheel ABI and real import remain
preflight/runtime checks, not asserted results.

## Approval checklist — Stage A only

- [x] Task 1 source/container preflight; conditionally pull/use only the two
  exact pinned linux/amd64 images after all preflight checks; retain outputs in
  the new ignored runtime root subject to storage/resource/network bounds.
- [x] Exact approved empty read-only tmpfs overlay at
  `/gym/crates/spire-env/src/tests/fixtures` in the disposable release-build
  container only; no host fixture mount/copy and no Gym filesystem change.
- [ ] Task 2 Maturin wheel build — **not authorized by this Stage A decision**.
- [ ] Task 3 CPU wheel downloads, dependency resolution/install, or venv —
  **not authorized by this Stage A decision**.
- [ ] Task 4 opt-in Task 6 conformance tests — **not authorized by this Stage A
  decision**.
- [ ] Task 5 Task 6 acceptance/closeout — remains subject to review and is not
  authorized by this decision.

No GPU, CUDA, TorchRL, Task 7, training, or shared services are authorized.
Any different source/mount, pin, command, or scope requires fresh explicit
approval. Failed preflight stops before Task 2. Task 6 remains incomplete.
