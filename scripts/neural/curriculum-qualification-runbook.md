# Held curriculum qualification proposal — UNEXECUTED

This is a concrete proposal for a fresh protected phase, not a release or an
executable host controller. No native/tensor workload, installation, service
operation, GPU query, Docker lifecycle, Run effect or old-claim reuse is
authorized. Every prior Task 9/sustained claim remains consumed. No residual
eight-hour campaign authority is asserted. A later host controller, its source
review and a new one-use approval/claim are prerequisites to execution.

## Immutable inputs to bind in a future release

The retained pure preparation is
`artifacts/development/neural-combat/curriculum-v1/task-5/`.
`qualification-input-inventory.json` binds every emitted input's raw SHA256 and
size. The original `source-sha256.json` remains the historical Task 5 snapshot
at `6a3cb142f6ee1d4bc08d0bfd129915d275bfdc40`; current fix source is
bound by `fix-1/source-sha256.json` and `fix-1/qualification-binding.json`.
The Task 5 report records their detached hashes and source commits. The
producer base is `bed896c55c02e3ee5fa57982de4a6262398b4b0e`. Rehash these actual
bytes at admission. A later controller must add its own exact reviewed source
inventory and preserve this source/input identity, never stage a moving checkout.

Use `prepared-mixed/qualification-0.toml` and `qualification-1.toml`. These are
real parser-accepted CUDA configs, each B4 × h16, unchanged model/PPO/features/
reward, with cumulative caps 512 decisions, 512 environment actions, eight Adam
attempts and 120 seconds. They share logical `output_root="runs"`, recipe,
support, seeds and distribution; only manifest path and shard index differ.
The descriptive default pilot's 16-Adam budget is not used.

| Binding | Canonical SHA256 |
| --- | --- |
| Recipe | `3279664729328894fc995c997ccc54282f8d33483459e91d4da1b7ed43a92f66` |
| Support | `5b2d0c144c9cd375ded221a38b1685986d0edd3639f88cedbe8706a807edb469` |
| Mixed distribution | `ac8060dc58683ff48644981080858d51cf976baee435e3ef081bcc04220b3ea2` |
| Qualification config 0 | `02643c5784a2c7d9bee3b7323b3583a53d49fdd93e419a6bcc5b8fcb06296f31` |
| Qualification config 1 | `612172ae211ba5d3d7efd6d0799a6da6053565fdb0f72b1fc6a01e0d68dc7586` |
| Shard 0 | `d644d2a113034641bc12cd06e50c7817cad23b5268c3c9014ef3f3f4406c7909` |
| Shard 1 | `6d1d3920fbba67df15d176336ab170ce5259dd9385729e51484a4954f10302b9` |
| Development manifest, prepared only | `435a236e60ac54f0b04f742b6d019023f49670c7730ba47545b73b23d2fb620e` |

`prepared-mixed/pilot-parents.json` binds original standalone B at
`artifacts/development/neural-combat/experiments/task9-v1-attempt-b/training/standalone-b/20261030T210525Z-c7d37fcacba143709d2e08ca9eb9c21c/checkpoint-final`.
Verify all three files before tensor loading:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| metadata.json | 17,150 | `d8c5c7e9da0262134cd6e383d76654bbd57517a75171becb5d26fde1f0dd2ff7` |
| model.pt | 1,089,216 | `fc44f89afd409abd68bf2397d28c11f49584ea2c017f090e6da0b2408c16ecd9` |
| training.pt | 2,211,765 | `ecff01b35c77c9da4aae7a4ff0506276aa91db36a9f972b725ba2ce061ada6fb` |

The qualification warm-starts weights only. Old Adam/RNG/progress are discarded.
Original Run B is recorded in the parent inventory solely as the future frozen
evaluation reference; this phase does not load it into a Run or evaluate it.
No development/final evaluation, replay or diagnostic optimizer probe is included.

Existing runtime pins are inputs to reverify, never permission to reinstall:

- Python image `docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`.
- Existing venv `artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a/install-work/venv-cuda`; guest Python `/runtime/venv-cuda/bin/python3.12`.
- Lock `artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a/resolve-work/requirements-cuda.lock`, SHA256 `92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45`.
- Native `artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a/install-work/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so`, SHA256 `071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`; Gym revision `533dad1c8aec86df624b3742b6b3ad1dbf1d0033`.
- Locked Torch `2.14.1+cu130`, CUDA `13.0`, NumPy `2.5.4`, Gymnasium `1.4.0` and source `spire-lab` wheel hash from that unchanged lock. No new TorchRL/TensorDict dependency.
- CDI profile `configs/runtime/task9-nvidia-cdi-profile.json`, SHA256 `83214eaa2b2568accc8350ac1cd05955d5cfdf1594a6234d383863d6ceabe937`, profile `task9-nvidia-cdi-1.20.0-595.91.07-v1`; UUID `GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d`, driver `595.91.07`. Reverify exact host CDI/hook and post-OCI mounts/environment against that profile before any GPU effect.

## Proposed bounded workload

One new phase named `curriculum-qualification-v1-<fresh-id>` would run exactly
the single method in `tests.neural.test_curriculum.CurriculumIntegrationTests`.
Its dedicated curriculum flag and the existing GPU flag would be enabled only
by that new release; all unrelated `SPIRE_LEARN_RUN_*` flags stay absent.
`SPIRE_LEARN_CURRICULUM_INPUT_ROOT` would name the immutable prepared-mixed
mount. No approved launcher exists yet.

| Shared ceiling across the entire phase | Limit |
| --- | ---: |
| Native action slots, attempts including faults | 512 |
| Actor sampled rows, including probes | 512 |
| Adam attempts, including failures/probes | 8 |
| Reset slots, including compatibility probes | 256 |
| Soft workload time, all guest work | 120 seconds |
| Whole phase including staging/preflight/save/close/cleanup/evidence | 600 seconds |
| New outputs including control/log/scratch evidence | 1 GiB |
| Existing aggregate retained-output ceiling | 40 GiB |
| Minimum filesystem free space | 30 GiB |
| Minimum GPU free memory | 12 GiB |

These are single shared ceilings, not per-test or per-segment allowances.
The future host starts the outer 120-second workload deadline before guest
entry, refuses new effects after it, and enforces a 600-second whole watchdog
including cleanup. The test's in-process counters additionally charge before
forwarding native actions/resets, multinomial rows and Adam calls; failed
attempts are not refunded. The host must enforce space/time admission and
persist attempt counters and first failures even if the guest dies. An
uncertain native/optimizer effect ends the phase without retry or salvage.
The host must reserve control/final evidence within the 1-GiB ceiling and stop
before crossing any aggregate/free-space floor.

Sequence:

1. Rehash exact source/config/support/shard/parent/runtime inputs; verify no
   final cases are runtime inputs. Admit one fresh claim and a fresh empty
   owned output/scratch directory. No container or claim exists from this proposal.
2. Construct/close public standalone drivers for both 256-entry catalogs. The
   real `make_driver` verifies the installed build, public observation/profile
   and Gym-normalized entries. No reset or action is requested by these probes.
3. Call real `run_stream_segment(config0, purpose="warm_start", checkpoint=B, ...)`
   with cumulative `max_updates=1` and action high-water512. Require fresh
   optimizer/version/progress and original B weights. One 64-row rollout has
   at most four epochs × one minibatch = four Adam attempts. Require healthy
   published final checkpoint, finite changed weights and accepted optimizer work.
4. Load its real tensor archive. Construct/close a same-shard resume runtime
   without reset/step/sample/update. Compare model, Adam, all four RNG domains,
   behavior version, sampler, saved progress and collector recovery exactly.
5. Resume exactly shard1 from that checkpoint using cumulative `max_updates=2`,
   the same absolute deadline and action high-water512. Check exact restored
   state before any effect; then allow one 64-row update, at most four additional
   Adam attempts. Require healthy publication and cumulative update_count2,
   no uncertain effects, bounded actual attempt totals and finite changed weights.
6. Close owned resources; retain configs, input/source inventories, counters,
   event/summary/lifecycle/checkpoint inventories and first-error evidence.
   Compare saved elapsed prefix only to saved counters; charge later close and
   serialization against the external phase budget.

Nominal actor/action usage is 128 rows/slots; a terminal at every decision
requires at most128 reset slots. These are scheduling estimates, not measured
results. Safety caps remain512/512/8/256, including every probe. Two normalized
catalogs comprise512 entries; this does not mean512 resets or fought cases.
Four generated coverage shards remain preparation evidence only.

## Proposed isolation, admission and cleanup

The reviewed controller must create one non-root `1000:1000`, offline,
read-only-rootfs container from the exact local image. Use no network, no new
privileges, all capabilities dropped, private IPC, bounded CPU/RAM/PIDs and
bounded tmpfs scratch; proposed ceilings are4 CPUs,8 GiB RAM with no added
swap,128 PIDs and128 MiB temporary scratch. No Docker socket, home directory,
credentials, host source checkout, Gym checkout, tracker or broad artifact tree
is mounted. Use fresh phase-specific labels/name and retain the full container
ID. CDI exceptions must be exactly those inventoried by the pinned profile.

Mount reviewed learner source/tests at `/learner` read-only, the existing venv
at `/runtime/venv-cuda` read-only, lock and profile read-only, only the original
B checkpoint directory at its exact `/learner/artifacts/.../checkpoint-final`
path read-only, and prepared inputs at their exact `/learner/artifacts/.../prepared-mixed`
path read-only. Overlay only its `runs` child with a fresh owned writable output
directory so logical config/output paths stay unchanged across resume. Mount
owned scratch separately. Use an allowlisted clean environment and fixed Python
and working directory. Reinspect exact image, UID, mounts, labels, CDI and
network/security settings after creation and before entry.

The future watchdog must stop only its verified full-ID container, reinspect
ownership before removal, preserve first faults and record cleanup uncertainty.
It must never operate on unrelated containers or delete retained inputs/history.
Whole deadline includes cleanup; inability to prove clean closure blocks success.
Existing sustained/proof controllers are historical reference only and cannot
be reused under their consumed IDs or silently adapted phase vocabularies.

## Acceptance and later scope

A successful run would qualify only these two native catalogs, these tensor
payloads and this bounded continuation path on the pinned runtime. Any skip,
loader/setup error, changed input, missing lifecycle receipt, uncertain effect,
cap violation or failed cleanup rejects qualification. Report generated,
normalized, reset, sampled and naturally completed coverage separately.

A separately reviewed replicated fixed/randomized/mixed learning pilot needs
its own training/evaluation/resource budget, matched original B weight-only
initialization and development strata. Qualification alone cannot establish
learning. The native Run fault remains a separate dependency; this standalone
phase has no Run effects and does not replace the fixed noncombat controller.
Both deferred Minors in [the curriculum guide](../../docs/neural-curriculum.md)
remain disclosed. Final evaluation stays frozen and unexecuted. Preserve
RNG-001/TIME-001 and no original-game-transfer claim.
