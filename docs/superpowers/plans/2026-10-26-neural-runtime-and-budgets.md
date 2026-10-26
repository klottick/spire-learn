# Neural combat runtime and budget proposal

Status: proposed companion to `2026-10-26-ironclad-combat-learning.md`, not an
execution record. The user approved the learner design and preparation of these
plans. No dependency payload, new image, environment, native call, GPU workload
or training run was created while writing this proposal.

## Phases and approval scope

| Phase | Deliverable | Proposed limit | Required approval |
| --- | --- | --- | --- |
| CPU setup | New accepted-source Gym wheel, audited CPU closure, separate venv | 8GiB new retained/temporary storage; build/setup600s soft/1,200s whole phase | Explicit CPU setup approval; no GPU/training implied |
| Native conformance | Task3 three tests; Task5 two tests in a later invocation | Each invocation4,096 total native slot actions,120s soft/300s whole run; zero updates | Scoped integration authorization after core/source review |
| CPU update smoke | Task7 two mode tests, export/reload | 512 learned decisions,2,048 environment slots total; two rollout updates per mode/32 optimizer steps total;120s soft/300s whole run | Scoped update-smoke approval after code review |
| GPU setup | Separate CUDA wheel closure/venv, finite device work | Additional24GiB storage;600s soft/1,200s whole setup;60s soft/120s device check | Explicit GPU setup/workload approval |
| Calibration | One declared standalone B4/8/16 cohort plus one Run driver | 2,048 environment slots,3 disposable update microbenchmarks;300s soft/600s whole run;8GiB peak GPU allocation | Calibration approval, no learning experiment implied |
| Initial learning | Four independent mode/seed runs | Candidate16,384 learned /65,536 total env actions,600s soft/900s whole run EACH | Exact post-calibration configs and budgets approved separately |
| Selection/transfer | Matched frozen baselines/exports | Shared65,536 env-action cap,600s soft/900s whole selection phase | Included explicitly in experiment approval; final split excluded |

One approval may explicitly cover Tasks1–7 plus this CPU phase, including their
listed native/model/update invocations. Source/evidence review between them is
the parent's internal responsibility, not a reason to ask the user again for
already approved scope. GPU phases and post-calibration learning stay separate.

Propose a **new neural-phase40GiB cumulative soft storage budget** over the
recorded pre-phase filesystem/container baseline:8GiB CPU setup,24GiB GPU setup,
8GiB outputs/checkpoints/logs. This requires explicit approval; it does not
silently expand M1's historical20GiB allowance. Require≥60GiB free before each
setup phase and≥30GiB before workloads. Account new images, caches, wheels,
targets, venvs and transient outputs, not only final files. No pruning shared or
historical resources to fit. Stop before a proposed allocation exceeds a cap.

Action caps are cumulative slot actions, not Python batch-call counts. Reserve a
whole batch before dispatch; insufficient remaining budget stops without action.
Count attempted/forwarded/known accepted/delivered and uncertain prefixes. Count
diagnostic sampling and optimizer steps separately with explicit cumulative caps.
Workload soft budgets are checked before reset/action/minibatch/serialization and
record elapsed/overshoot. A hard wrapper enforces the stated total INCLUDING
cleanup, with TERM at total minus90s and final kill at the total. Cleanup ignores
subsequent TERM only once it has begun and uses bounded stop/remove/postcheck calls
within that grace; missing cleanup evidence is a failed run, never fabricated pass.
Never retry a failed/partially accepted native action or automatically rerun a
cohort, download another package variant, or extend a budget.

## CPU images and immutable source inputs

Propose reuse of the already retained linux/amd64 builder image ID
`sha256:470a1d6751fa197eaeb99319d9a2f2f24f3d7da11347659c7b24b4ff5527c738`
and runtime `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
They remain different roles. Verify actual image IDs, Python3.12 and bundled
Rust≥1.89 before use; no new derivative, root execution, supplemental root group
or toolchain installation. Rootfs read-only; UID:GID1000:1000; drop all
capabilities; no-new-privileges;4 CPUs/8GiB RAM/512 PIDs; owned temp/cache/output
only. No Docker socket, host home, credentials or broad checkout mounts.

Accepted Gym source is `533dad1c8aec86df624b3742b6b3ad1dbf1d0033`, canonical
636-file map root `dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3`.
Use the already frozen public source at
`/home/klottick/.paseo/worktrees/2ae6xl12/gym66-senior-20261026/.superpowers/gym66/review-v3/source`
only after verifying its manifest hashes/modes against accepted Git again.
This avoids building from an advancing main/M12 checkout. Do not create/edit a
Gym worktree or copy sources into Learn. If retained inputs disappear or differ,
stop and obtain a new reviewed read-only input location.

Builder mount table (all source mounts read-only; exact paths saved before run):

| Host input | Guest path |
| --- | --- |
| Frozen source `Cargo.toml`, `Cargo.lock`, `pyproject.toml` | Corresponding file under `/gym/` |
| Frozen `crates/spire-core`, `spire-env`, `spire-python`, `spire-cli`, `spire-report` | Corresponding directory under `/gym/crates/` |
| Each of22 tracked `.py`/`.pyi` files under frozen `python/spire_lab/`, selected from accepted manifest | Corresponding individual file under `/gym/python/spire_lab/` |
| Fresh owned learner runtime build child | `/work` read-write |

Mount public Python files individually, excluding existing `.so`/`.pyd`/`.dll`,
target directories, caches, reproduction/private outputs and `.git`. Parent dirs
are container tmpfs, not writable source. Preserve the previously approved empty
read-only tmpfs overlay at `/gym/crates/spire-env/src/tests/fixtures` in the
release-build container only; never use it for upstream tests. No other private
paths or source overlays. Preflight exact workspace members (now five), package
inputs, escaping symlinks and fixture references; release needing hidden inputs
stops, no fabricated substitutes. No upstream test suite rerun is needed here.

Explicit container home, Cargo home/target and temp live under `/work`; default
Docker credential config is outside every data bind and never inspected/copied.
Network is allowed only during approved Cargo lockfile downloads and dependency
resolution/download phases. Build honoring accepted pyproject:

```sh
SPIRE_REVISION=533dad1c8aec86df624b3742b6b3ad1dbf1d0033 \
CARGO_HOME=/work/cargo CARGO_TARGET_DIR=/work/target \
HOME=/work/home TMPDIR=/work/tmp \
/usr/bin/maturin build --release --locked \
  --manifest-path /gym/crates/spire-python/Cargo.toml \
  --interpreter /opt/python/cp312-cp312/bin/python \
  --target-dir /work/target --out /work/wheelhouse
```

Prove the pinned tool actually uses the documented revision variable from accepted
build.rs; never forge dirty-state fields or assume a `.git` mount. Audit the new
wheel's tags/metadata/package contents/hash; record loaded binary/build identity
after offline install. It must expose both exact accepted fingerprints and support
catalog. CPU setup is not allowed to install Torch's CUDA fallback.

## Audited Python dependency closure

CPU runtime versions are chosen for accepted public package compatibility:
NumPy2.5.4, Gymnasium1.4.0 and Torch2.14.1+cpu. No Gym dev environment, lockfile,
venv or native artifact is copied; no TorchRL/TensorDict/torchvision/torchaudio.
Existing ordinary dependency-wheel caches may be mounted read-only by exact
payload path after hash verification, never entire M1 runtime roots.

Exact CPU Torch candidate URL:
`https://download-r2.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl`
SHA256 `5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f`.
Reverify bytes; earlier M1 payload evidence is not a new venv qualification.

Exact GPU candidate is Torch2.14.1+cu130, CPython3.12/linux x86_64:
`https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl`
Advertised SHA256 `5359b255ac68c0cabb8293952b38e00f0d1a570815cd87f19b61af702c071410`.
The [official index](https://download.pytorch.org/whl/cu130/torch/) and separate
wheel metadata were read, but this GPU payload was not downloaded/verified locally.
Metadata declares CUDA toolkit extras13.0.3, CUDA bindings≥13.0.3,<14,
cuDNN9.24.0.43, cuSPARSELt0.8.1, NCCL2.30.7, NVSHMEM3.4.5 and Triton~=3.8.0
on this platform, as well as ordinary Python dependencies. Resolve/audit all
declared transitive runtime requirements, including unused-by-model libraries;
never prune them to pretend a broken wheel has a complete closure.

Each approved resolver phase writes unique nonoverwritten logs with source URL,
original response metadata, expected/observed hashes, wheel ABI/platform and
Requires-Dist/markers. Resolve CPU and GPU separately using exact Torch payload
constraints; binary wheels only, no sdists, extra indexes or silent replacements.
Emit new complete `requirements-cpu.lock` / `requirements-cuda.lock` hash locks.
Offline install uses `--no-index --find-links /work/wheelhouse --require-hashes`,
then `pip check`; absence of a compatible binary stops the phase. Locks are
generated evidence reviewed before installation, never guessed package pins.

Runtime sees only fresh owned phase child at `/work:rw`, learner src/tests/configs
at `/learner/{src,tests,configs}:ro` and exact allowed dependency-wheel inputs
read-only. No Gym source, M1 venv, credential/config directory or old outputs are
writable. New venv paths `/work/venv-cpu` and `/work/venv-cuda` are separate.
All import/tests/calibration/gameplay/training run with `--network none` and
`PYTHONDONTWRITEBYTECODE=1`. `pip check` and exact versions/builds are gates.
Pure optional tensor-test development (Tasks4/6) has a cumulative4,096-row
diagnostic sampling ceiling across its logged RED/GREEN invocations; reserve
counts before draws and stop if exhausted. Bootstrap/value queries never sample.
Task7 native policy sampling is separately bounded by its512 learned decisions.

## GPU access and qualification

Read-only inventory was RTX5090/32607MiB/driver595.91.07. The candidate declares
CUDA13.0.3; NVIDIA's [13.0.3 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
list Linux driver≥580.126.20. The observed driver clears that version comparison;
actual container/GPU/compiled-kernel support remains untested, not inferred.

Require existing host GPU container support and inventory current UUID/free memory.
Pass only that device through the existing Docker GPU runtime, same non-root and
filesystem protections. No host driver/toolkit/container-runtime install, new
device group privilege workaround or GPU use during CPU phases. Require≥12GiB
free device memory before starting; never evict other processes. Record device
capability, `torch.version.cuda`, `torch.cuda.get_arch_list()`, actual finite
forward/backward/Adam and synchronized timing. Enforce allocator peak≤8GiB and
record process/device memory; allocator limits are not a claim of hard isolation
from every driver/context allocation. Stop and report excess usage.

Use a fresh model and stored public frames for disposable microbenchmarks.
Sampling is bounded by actual learned decisions; no separate diagnostic draws
unless their explicit cap is reserved before the call. Keep the prior M1
96/32 TorchRL sampling breach in history; it grants no new sampling allowance.

## Completion evidence and stop conditions

Save exact approval text, runbook/source hashes, preflight, full literal
commands/output/exits, schema/catalog/build/package identities, unique logs,
action/sample/update reservations, close/cleanup/postcheck and an inventory
finalized only after logs are closed. Never hash a still-changing file and claim
the original inventory passed; a correction is a separately labeled record.

Zero/skipped gated tests are not a native/tensor pass. Cap-limited ongoing fights
are incomplete, not losses/truncations/victories. Faults stop without retry; no
seed search or second cohort for nicer outcomes. A changed checkout that is not
mounted is separately observed provenance, not proof the installed build changed.
Source changes after review require a scoped diff review and matching execution
snapshot, not an unrelated blanket rerun. No final evaluation/cloud/GRU/LLM/demo
work is authorized by these proposed limits.

The parent reviews the materialized lock/runbook and code before each applicable
phase. New requirements outside these mounts, versions, privileges or caps require
a concrete revised proposal. Routine core tests and reversible implementation
within an approved task do not require repeated permission.

References used for API choices: PyTorch's [restricted checkpoint loading](https://docs.pytorch.org/docs/2.14/generated/torch.load.html),
[generator-aware sampling](https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html)
and [gradient clipping](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html).
