# Task 7 CPU probe dependency setup plan

> **For agentic workers:** Use superpowers:subagent-driven-development after
> explicit approval of the new dependency setup. Luna workers use max reasoning;
> no nested agents. Keep artifacts/development/foundation as canonical workspace.

**Status:** Tasks A–B explicitly user-approved on 2026-10-25 and executed.
Task A's locked runtime is accepted. Task B's final availability checks passed,
with a recorded compliance failure: retries drew 96 authored CPU samples against
the 32-sample bound. The parent accepted qualified artifacts for the already
authorized probe, without retroactively authorizing that overrun. Full evidence,
review and disposition are in `artifacts/development/foundation/task-7-runtime-report.md`,
`task-7-runtime-review.md` and `task-7-setup-parent-review.md`. The
user's “Let's continue” resumes Task7 implementation and bounded native-vector
work in the existing runtime. This request covers only adding the separately
selected TorchRL/TensorDict CPU probe runtime, not training or framework adoption.

**Goal:** Create a separately locked CPU probe venv without changing the accepted
Task6 runtime, then establish imports/MaskedCategorical availability offline.

**Architecture:** Reuse the exact pinned Python runtime image and accepted own
Gym/CPU Torch wheels. Keep Task6 wheelhouse/venv/lock intact; resolve a new
binary-only closure into new owned outputs and install in a new venv offline.
Actual GymWrapper mapping is tested by the separately bounded Task7 probe.

**Tech stack:** CPython3.12.15/glibc2.36, existing Torch2.14.1+cpu,
NumPy2.5.4, Gymnasium1.4.0, own Gym0.1.0; candidate TorchRL0.14.1 and
TensorDict0.14.3. No optional extras, trainer, optimizer, GPU or CUDA packages.

**Spec:** [M1 foundation design](../specs/2026-10-24-foundation-design.md) §7 and
[foundation implementation plan](2026-10-24-foundation-implementation.md) Task7.
This is subordinate setup, not a new learner architecture.

## Global constraints

- New package payload download/install is authorized only for Tasks A–B as
  explicitly approved on 2026-10-25. Read-only metadata/source inspection, pure Task7 implementation,
  reviews and local documentation commits remain within current authority.
- Host Python/Rust/OS and Gym checkout remain unchanged. No borrowing Gym
  environments/artifacts, copied Gym code/fixtures/lockfiles, new Docker image,
  root/group workaround, Dockerfile change, service modification, push or prune.
- Existing runtime image, linux/amd64:
  `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
- Create a fresh exclusive Task7 DATA root under
  `artifacts/development/foundation/runtime/<new-uuid>` as the sole `/work:rw`.
  Mount ONLY accepted StageA wheelhouse at `/inputs/stagea-wheelhouse:ro` and
  accepted StageA requirements-native.lock at
  `/inputs/requirements-native.lock:ro`. These are the only extra input binds;
  StageA venv, root, home and caches are not mounted. StageA host root remains
  `runtime/2e4242dc-1852-4d78-a13f-1907b42fc94f`. Host-only Docker client config
  stays under its separate original root, never mounted/read. No source/Gym/
  home/socket/device mounts. Prove input RO and output RW via Docker inspect,
  mountinfo and access checks before downloads; no write canary to old data.
- Every container: unique name, --rm, --platform linux/amd64, --read-only,
  --user1000:1000, --cap-dropALL, no-new-privileges, 4CPUs, 8GiB,512PIDs.
  Use network only for approved public PyPI/files.pythonhosted.org resolution;
  installation/import checks run network-none. This is policy, not DNS firewall.
- New setup storage <=4GiB soft, and cumulative StageA+Task7 growth <=20GiB
  soft from the recorded StageA baseline. Check >=30GiB free before phases and
  record data/image/cache growth; neither is a hard quota. Stop on cap/failure;
  never prune or expand budget. Resolution/install wall cap30min including
  own-container shutdown; offline import check cap5min. No gameplay in setup.
- Preserve accepted Task6 venv,15-wheel lock/wheelhouse, source/report/diff
  changes. New exclusive ignored setup log/provenance and wheelhouse dirs,
  new `/work/venv-task7`; refuse pre-existing nonempty/new-target paths. The
  old accepted StageA runtime is physically outside the writable bind.
- No `--extra-index-url`, CUDA/NVIDIA/Triton, torchvision/torchaudio/TorchRL
  extras, pytest/dev/LLM/video/rendering/Atari/game assets. In particular
  `torchrl[gym-continuous]` is excluded: its optional Gymnasium<1 requirement
  conflicts with the selected public facade. No pin replacement or fallback.

## Exact candidate inputs and provenance

Read-only PyPI release metadata on2026-10-25 reports TorchRL0.14.1 requiring
TensorDict>=0.14.3,<0.15.0. TensorDict0.14.3 is the available matching0.14
release, while current0.15.3 is outside that range. Metadata is not runtime
compatibility evidence. Use only these two root wheel payloads:

| Distribution | Exact wheel | SHA256 |
| --- | --- | --- |
| TorchRL0.14.1 | torchrl-0.14.1-cp312-cp312-manylinux_2_28_x86_64.whl | d2bea667dee3abb76dd7cc2e5c064f3012c125adad06af36603c3c79e63618b2 |
| TensorDict0.14.3 | tensordict-0.14.3-cp312-cp312-manylinux_2_28_x86_64.whl | b5c48a21ee84b83e69f1ff6ac06e371bdc20b01aa2c05af76c7e5020144cb71a |

Exact root URLs:

- https://files.pythonhosted.org/packages/33/d6/5c91e919c4261e1bfe13c833a356b26c5d79102d018930f85444cf0117b4/torchrl-0.14.1-cp312-cp312-manylinux_2_28_x86_64.whl
- https://files.pythonhosted.org/packages/73/7c/f8d6570ef5b0a859316e38329993ef540fe53ea711234647d2f38d33af5b/tensordict-0.14.3-cp312-cp312-manylinux_2_28_x86_64.whl

Keep all15 StageA distributions pinned to their existing exact versions.
Retain exact CPU Torch payload SHA256
`5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f`
and own Gym wheel SHA256
`25cd26001e8cf3034da95434351ac71f92f676dbeeb15784682f809e82d4799a`.
StageA lock SHA256 is
`d026856db845f105383c91d4901231fd016111f46257999456511e699475ee45`.

Expected additional runtime requirements from root metadata include packaging,
pyvers, hoptorch, importlib_metadata, and orjson for Python<3.13; transitive
requirements are admitted only through the audited active metadata closure.
Resolve binary wheels, then pin and hash every selected payload before install.
Do not adopt a whole source repository's dev environment. Current metadata for
hoptorch0.1.4 requires Torch>=2.7 and pyvers>=0.2.2; pyvers0.2.3 has no declared
runtime dependencies. These are planning observations, not install evidence.

## Review focus

1. Wrong TensorDict minor/ABI: preserve exact candidates and Torch+cpu; import
   checks must actually execute, and any symbol/import failure is a blocker.
2. Resolver creep: preserve all StageA versions, no extras/CUDA/source builds;
   every added wheel must be required by active platform/extra markers.
3. Runtime mutation: new venv/wheelhouse/lock only, old payload hashes unchanged;
   host Docker client credentials stay outside every bind.
4. False compatibility: successful imports/masked distribution checks are setup
   evidence, not GymWrapper SAME_STEP compatibility or trainer adoption.
5. Hidden rollout: setup imports must never construct an environment, call
   check_env_specs/rollout/collectors, reset or step a simulator.

## Task A — Resolve and hash-lock isolated CPU probe dependencies

**Files:** new ignored DATA setup directories, constraints-task7.txt,
requirements-task7.lock, venv-task7; report under canonical foundation workspace.
**Consumes:** accepted native-runtime-identity-2026-10-25.json and StageA lock.
**Produces:** audited new closure/lock and unchanged-old-runtime hash proof.

- [ ] Record explicit approval/time; verify pins, source/learner state, owner,
  untouched targets, image/platform and storage. Inspect container protections
  and the exact three binds before downloads: exclusive `/work:rw`, accepted
  `/inputs/stagea-wheelhouse:ro`, and `/inputs/requirements-native.lock:ro`.
  No other binds are allowed; HOME/tmp/pip caches stay work-owned.
- [ ] Download exactly the two root URLs once with stdlib streaming, verify
  SHA256 immediately before metadata/resolver use. No substitute/root fallback.
- [ ] Derive constraints from the15 StageA versions plus the two new exact
  roots. Run binary-only pip resolution with `/inputs/stagea-wheelhouse` as RO input
  and new output wheelhouse. Reusing/copying owned approved dependency wheels
  here is consumption, never borrowing artifacts from the Gym checkout.
- [ ] Audit complete closure: filename/METADATA/WHEEL/runtime tags/Python,
  unique distributions, exact roots/old pins, active marker/specifier closure,
  payload digests and justified URLs. Reject forbidden/unexplained packages.
  Generate NEW exact-version/hash requirements-task7.lock. Preserve old lock.
- [ ] Stop network resolver; use the same protected image network-none to
  create/install into new `/work/venv-task7` using --no-index/--require-hashes
  and the new wheelhouse. No pip/global upgrade. Require pip check exit0.

## Task B — Offline package/API availability and evidence

**Consumes:** accepted locked TaskA closure. **Produces:** actual import/version/
CPU/masked-distribution availability, not Gym integration verdict.

- [ ] Run imports/version capture for TorchRL0.14.1, TensorDict0.14.3,
  Torch2.14.1+cpu, NumPy2.5.4 and Gymnasium1.4.0; require torch.version.cuda=None.
  Record any binary-extension warnings accurately; do not hide incompatibility.
- [ ] Carry prior accepted Gym0.1.0 wheel hash, source revision
  `4874fb398128c142bd56f10a12e6fca939a88f93`, native extension label
  `spire-python` and Rust1.98.0 into setup evidence as PRIOR accepted identity.
  Do not import Gym native code during setup. The actual probe separately
  reads already-loaded native build_info and actual descriptor fingerprint.
- [ ] Import public GymWrapper and MaskedCategorical without constructing an
  environment. With explicit nonempty authored boolean mask and CPU logits,
  check finite legal log-probabilities and that sampled actions are legal;
  bounded32 samples, no optimizer/backprop/model/trainer. Zero gameplay.
- [ ] Close only own containers; recheck old wheelhouse/lock and original
  report/diff/source states, no active task process, storage within bounds.
  Full literal commands/stdout/stderr/exit records and new closure manifest
  remain ignored. Parent reviews actual artifacts before any runtime probe.

## Task7 probe boundary (existing implementation scope)

Current continuation authorizes implementing the planned native-vector and
framework probe; new dependency installation above remains gated on approval.
Use at most one initial B=2 cohort per probe, workers1, selection scenarios
Cultist(env7/policy101) and JawWorm(env11/policy103), max128 full-batch steps and
60s soft probe budget each, hard10min container timeout, same protections.
Every slot needs a current legal action; no step_mask, filler/drain evaluation
scheduler, extra final cases or accepted-effect retries. Automatic replacements
are diagnostic overhead and counted. No observed mixed boundary means blocked.

The versioned TorchRL source indicates two hazards to verify: direct
autoreset_mode attribute access while Gym publishes public metadata, and
auto_register_info_dict calling check_env_specs with implicit rollouts. Probe
construction must guard against unapproved reset/step, and record raw public
data from the SAME accepted calls rather than resubmitting for comparison.
The guard is learner-owned and sits immediately before every call forwarded to
the public facade: during wrapper construction, `reset` and `step` raise a
typed `SetupBlocked` BEFORE delegating, with zero native calls; after construction
exactly one explicit initial cohort reset is permitted, then only controller-
requested full-batch legal steps within the configured budget. Any additional
reset/step request is rejected before native dispatch, recorded as blocked and
ends that attempt, with no retry. The guard does not repair/rewrite actions.
Its pure call-count tests use a counter-only protocol object (no simulation)
to assert denied operations never forward, permitted operations forward once,
and post-failure operations remain denied. This safety check precedes any
wrapper/native integration. If TorchRL registration would require implicit
operations, block and report it. Any non-rollout registration bridge is allowed
only if its exact passive path is reviewed and its zero-construction-call count
is demonstrated; direct `set_info_dict_reader` is not assumed passive.
Any learner-owned metadata/registration bridge must be explicit and bounded,
not a Gym edit or simulator rewrite; call the verdict adaptable only after all
required boundary/mask checks actually pass. Further semantic changes or a
PyTorch-native fallback require separate review/approval, not silent adoption.

## Sources and approval

- https://pypi.org/pypi/torchrl/0.14.1/json
- https://pypi.org/pypi/tensordict/0.14.3/json
- https://pypi.org/pypi/hoptorch/0.1.4/json
- https://pypi.org/pypi/pyvers/0.2.3/json
- https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.GymWrapper.html
- https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.MaskedCategorical.html
- https://raw.githubusercontent.com/pytorch/rl/v0.14.1/torchrl/envs/libs/gym.py
- https://raw.githubusercontent.com/pytorch/rl/v0.14.1/torchrl/envs/gym_like.py

Source metadata/source inspection was read-only planning. No package payloads,
new venv, imports or probe results are claimed by this plan. User approval on
2026-10-25 covers TasksA–B only; it neither adopts a trainer nor changes existing
user boundaries. Review: artifacts/development/foundation/task-7-setup-review.md
(Specification PASS; Plan quality APPROVED before user approval).
