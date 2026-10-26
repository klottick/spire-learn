# M1 native evaluation smoke execution plan

> **For agentic workers:** Use superpowers:subagent-driven-development, fresh
> Luna at max reasoning, no nested workers. Canonical workspace remains
> `artifacts/development/foundation/`; main and local commits are authorized.

**Status:** Historical v1 plan, superseded for execution. The approved v1 wrapper
stopped at host preflight before any container or native test. The reviewed v2
wrapper then executed the still-unstarted evaluator once at source `732b714`:
1 test, 0 skips, two cases and 28 actions. Parent and independent reviews accepted
M1 on 2026-10-26. The original runbook and failed evidence below remain historical;
do not rerun them. Current evidence and qualifications are in
`artifacts/development/foundation/m1-acceptance-report.md` and the v2 runtime
report/review. This closeout grants no further execution authority.

**Goal:** Verify the implemented single-case evaluator produces reconciled public
case artifacts for the two documented selection scenarios.
**Architecture:** One fresh protected offline container runs exactly one opt-in
unittest. It calls `evaluate` once, constructing two single environments
sequentially. Reuse the installed qualified CPU venv; no new setup or packages.
**Tech stack:** Python3.12.15, NumPy2.5.4, Gymnasium1.4.0, own SpireLab0.1.0 wheel
in the retained Task7 runtime. Torch packages remain installed dependencies;
this evaluation performs no TorchRL probe, diagnostic sampling or training.
**Spec:** Approved [foundation design](../specs/2026-10-24-foundation-design.md),
[implementation plan Task9](2026-10-24-foundation-implementation.md), and accepted
Task7 contract gate at `artifacts/development/foundation/task-7-contract-freeze.md`.

## Global constraints

- Approval covers only the one test below, after code review. No automatic retry,
  second evaluation, extra case, constructor warmup, native fault manufacture,
  completed Task6/7 repeat, training, final cases, GPU or scheduler experiment.
- Exactly two selection cases, random policy: `starter-cultist-a0-v1` environment
  seed7/policy101 and `starter-jaw-worm-a0-v1` environment11/policy103.
  At most128 accepted actions per case,256 total. Global evaluation wall budget
 60s soft; whole shard600s hard including cleanup, outer watchdog510s plus90s cleanup grace.
  Keep delivered effects/outcomes on deadline; faults are errors, never losses.
- On a passing run, one factory/reset/close per scheduled case; no automatic
  replacement case. Each declared case is attempted at most once. A case error
  remains an error result; another already-declared case may run within the
  shared budget. Artifact failure/global expiry stops future cases.
  Public observations/candidates/masks only to actors. No seeds, routing IDs,
  private diagnostics or unseen state as features; opaque state not serialized.
- Existing runtime pins, venv and lock stay unchanged. No downloads, installation,
  image pull, host/Gym/source change, root/group workaround, shared-service
  change, prune, push or publication. Check >=30GiB free and20GiB cumulative soft
  storage budget before launch; do not expand either limit.
- Preserve the original Task6 report/root diff and every earlier run/report,
  including the96/32 setup qualification and incompatible framework attempt.
  Earlier approvals do not retroactively cure those qualifications.

## Exact inputs and containment

Image:
`docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0`,
`linux/amd64` (daemon spelling `x86_64` is the same architecture).
Venv: `artifacts/development/foundation/runtime/94039f1c-8ecc-4284-8919-25faa1d9fe15/venv-task7`.
Lock SHA256: `9cbdaa5ff1a2ca7f5dfae975543a3b7f1e68bf92ad8000c750cdfc86fa562959`.

Exactly four binds, using absolute resolved host paths:

| Input | Container path | Mode |
| --- | --- | --- |
| Qualified venv above | `/work/venv-task7` | read-only |
| Learner `src` | `/learner/src` | read-only |
| Learner `tests` | `/learner/tests` | read-only |
| Fresh owned output root | `/probe-work` | read-write |

Non-root1000:1000; networknone; read-only root; cap-dropALL;
no-new-privileges;4CPUs/8GiB/512PIDs; auto-remove; unique name/owner label.
Inspect all controls and exact mounts before start. No whole checkout/runtime,
home, credential, Docker socket, Gym source or device mounts. Host-only Docker
config is used only through Docker CLI `--config`; its contents are not
inspected or mounted. Exact host-only path:
`/home/klottick/repos/spire-learn/artifacts/development/foundation/runtime/faba7179-7213-4141-ab74-fe4ca660ac5d/docker-config`. HOME/tmp/cache are under fresh output; workdir `/learner`.

Expected public schema `spire-lab-combat-observation-v1`, content
`spire-lab-combat-reference-v2`, reward `spire-lab-combat-winloss-v1`, fingerprint
`8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357`.
Expected loaded native build: extension`spire-python`,
revision`4874fb398128c142bd56f10a12e6fca939a88f93`, Rust`1.98.0`.
Record actual loaded identity separately from the read-only checkout observation.

## Review focus

1. Run only the declared module; discovery would repeat other native tests.
2. Assert one test executed/zero skipped, not merely exit0.
3. Require factory/reset/close counts2, per-case steps<=128 and total<=256;
   accepted errors never cause action/case retry.
4. Reconcile exactly two unique episode records, summary/status/outcome totals,
   actual schema/build discovery and resolved case/config identity. Victory and
   death are both valid; do not require wins or infer outcomes from rewards.
5. Freeze source/test hashes before/after; preserve output and stable inventory.
   Finish captures before hashing them; never hash a still-open capture of the
   inventory command itself. Cleanup and qualifications are part of evidence.

## Exact host-side runbook

The prepared wrapper is
`artifacts/development/foundation/evaluation-smoke-runbook.sh`. It contains the
literal Docker create/inspect/start/timeout/cleanup commands, a pre-start JSON
control checker, writable-directory setup, and a post-run result gate. It has
been syntax-checked only (`bash -n` and compilation of its five embedded stdlib
Python blocks); no Docker or test command was run for that check.

After the source and runbook reviews pass and the user approves this run, the
parent records the exact approval in
`artifacts/development/foundation/evaluation-native-smoke-approval.json` and
records the reviewed Git commit. The structured approval has `approved=true`,
the user's verbatim `user_approval`, and `scope` exactly matching the runbook's
`approved_scope` dictionary: reviewed commit, current runbook SHA256, pinned
image/venv/lock, exact test and case seed pairs, one invocation/evaluate call,
128/256 action caps,60s evaluation/600s shard and zero retries. A stale or partial
record fails before creating the attempt marker or container. The preflight
also asserts Python3.12.15 and all six qualified installed distribution versions
using stdlib metadata from the mounted venv, and compares every Python source/
test hash to the approved commit. No mismatch is repaired by this run.
Only then invoke, from the repository root:

```sh
timeout --signal=TERM --kill-after=90s 510s bash artifacts/development/foundation/evaluation-smoke-runbook.sh <reviewed-commit>
```

Replace `<reviewed-commit>` with the actual accepted source commit recorded by
the parent, never an inferred branch tip. The wrapper verifies HEAD and clean
tracked, staged and untracked `src`/`tests` against the approved commit, then creates a UUID child of canonical runtime.
Its absolute resolved path is the only `/probe-work` bind. Host UID:GID must
already be1000:1000; it creates owned home/tmp/pip-cache/runs/records directories
with mode0700. It never uses chown, a root container, or supplemental groups.

The create command sets container-only `HOME=/probe-work/home`,
`TMPDIR=/probe-work/tmp`, `PIP_CACHE_DIR=/probe-work/pip-cache`,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and the exact native/output
flags below. Before `docker start --attach`, its stdlib JSON checker asserts
the exact image, created/not-running state, entrypoint/test arguments, owner
label, non-root user, limits/security flags, four resolved bind sources/modes,
and writable-temp environment. Failure prevents start and cleans only its own
CID; no weakened-control fallback is permitted. Create and inspect have30s/20s
timeouts; read-only metadata subprocesses have8s timeouts. The outer510s
timeout sends TERM, allowing90s for cleanup before the600s hard stop. The inner
work watchdog expires at480s (plus5s kill grace), leaving room for bounded
removal and postcheck. Cleanup starts with an incomplete marker and only marks
complete after preservation checks; a hard kill leaves that marker incomplete.
Cleanup ignores repeated TERM/INT during its reserved grace, with1s kill grace
on Docker command timeouts so ignored TERM cannot make a cleanup command hang.

A fixed, exclusive approval-consumption record
`evaluation-native-smoke-attempt.json` prevents wrapper reinvocation under the
same approval. Do not delete it to retry. Any pre-launch correction must be
reviewed against the same unused attempt and may never repeat a started test.
The remaining480-second work watchdog is calculated from monotonic time before
create. EXIT/INT/TERM cleanup checks/removes only the returned owned CID (or the exact unique owner-label/name
match if create timed out before returning its CID), records
the final query, and verifies source/protected/Gym preservation and600s total.

The wrapper snapshots itself and all Python source/test hashes in the output.
It captures Docker/test output and the actual test exit; the Bash command trace
is deliberately **outside** the runtime root while open. Any outer invocation
stdout/stderr capture must also be outside that root. The stable payload
inventory is written outside the root after captures and cleanup; once the
wrapper exits, record the closed outer captures/command-trace hashes separately.
No inventory may include a capture that is still being written.

## Execution and review

- [ ] Parent accepts Task9/10 code and independent reviews; freeze actual tested
  commit and source/test hashes. Resolve any source findings before runtime.
- [ ] User explicitly approves this bounded run. No dependent execution before
  approval; preparation and pure verification remain authorized.
- [ ] Fresh Luna/max executor creates one exclusive output/marker before native
  construction; records commands/stdout/stderr/exits and pre-start inspect.
- [ ] Run exactly this command inside the protected container, once:

```sh
env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=/learner/src SPIRE_LEARN_EVALUATION_OUTPUT_ROOT=/probe-work/runs /work/venv-task7/bin/python -B -m unittest tests.integration.test_evaluation -v
```

The module has exactly one test,
`EvaluationIntegrationTests.test_two_bounded_selection_executions_reconcile_artifacts`.
It authors its case/config inputs under owned temporary output; no configs bind
is needed. Require1executed0skipped, exit0, two natural terminal records with
recognized public outcomes, no primary/secondary reliability failure, correct
public build/schema, persisted artifacts and reconciled counts. Keep the
structured `EVALUATION_EVIDENCE` record and raw files; absence is not a pass.

- [ ] Stop after test failure/incomplete or ambiguous evidence; per-case/global
  caps stop their corresponding work. Preserve output, close/remove only the
  owned container, verify source/protected/Gym hashes and storage. No rerun.
- [ ] Parent and fresh independent reviewer inspect actual evidence. Only after
  acceptance may required Task9/10 integration/M1 gates be closed. Retain
  RNG-001/TIME-001; no strength, transfer or permanent architecture claim.

Reports and identity belong in canonical foundation, raw runtime data under its
fresh UUID root. Keep small authored approvals separate from ignored payloads.
