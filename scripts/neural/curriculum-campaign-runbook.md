# Reviewed curriculum campaign

The user approved a fresh qualification and approximately ten hours of standalone
Ironclad curriculum work. Source tests are pure evidence. The parent publishes
one new release after source review, then launches `curriculum-20261101-a` once.
Historical claims remain consumed. No install, Run environment, final case,
Gym checkout change, service operation or push belongs to this campaign.

## Freeze and release

Run these from the repository root **after** the final reviewed source commit.
They read metadata and copy no runtime environment. The captured Git revision and
dirty-state hash describe the real checkout, including the two preserved user
changes. Source capture includes the new PID helper delta explicitly.
The production capture CLI always reads actual Git provenance; `--head` and
`--dirty-json` overrides are rejected. Explicit provenance remains only in the
pure capture API used for authored test fixtures.

```sh
PYTHONPATH=src python3.12 -B scripts/neural/curriculum_controls.py capture \
  --root . \
  --destination artifacts/development/neural-combat/curriculum-campaign-v1/release/source-snapshot.json
PYTHONPATH=src python3.12 -B scripts/neural/curriculum_controls.py release-template \
  --root . \
  --package artifacts/development/neural-combat/curriculum-campaign-v1/task-1/package-fixture/package.json \
  --source artifacts/development/neural-combat/curriculum-campaign-v1/release/source-snapshot.json \
  --user-approval artifacts/development/neural-combat/curriculum-campaign-v1/preparation/user-approval.json \
  --destination artifacts/development/neural-combat/curriculum-campaign-v1/release/release-template.json
```

The existing 264-shard package is reused. Its raw SHA256 is
`fb548fb4564b6955131914aa66a31a62fa70092459fd87ef46f9f9d7f4113afe`.
Do not regenerate or modify it. `release-template` does metadata/hash checks;
the full parser/coverage validation occurs once during actual host admission.

The release has exactly these keys: `schema`, `state`, `campaign_id`,
`single_use`, `source_snapshot_sha256`, `package_sha256`, `campaign_raw_sha256`,
`qualification_input_inventory`, `runtime_pins`, `caps`, `user_approval`, `review`.
Its schema is `spire-learn-curriculum-release-v1`; campaign ID is
`curriculum-20261101-a`; `single_use` is the JSON boolean `true`.
All template bindings stay exact. The parent changes only `state` to `approved`
and `review` to `{"approved": true, "reports": ["<actual source review report>",
"<actual independent review report>"]}`. The two report references are nonempty
strings describing the reviews of this exact captured source. Publish that JSON
exclusively as
`artifacts/development/neural-combat/curriculum-campaign-v1/release/release.json`.
An existing file is evidence, never an overwrite target.

`runtime_pins` binds the qualified image/GPU, lock/native/copied Python hashes,
installed distribution metadata, prior qualified device evidence, exact CDI
profile, 4 CPUs, 8 GiB memory/no swap, 4/1 Torch threads and PID limits 128/512.
`caps` binds the whole/training/evaluation/effect/storage/qualification/segment
limits. `user_approval` has exactly a root-relative `path` and raw `sha256`.
Numeric fields retain their JSON types; booleans cannot substitute for numbers.

Optional read-only release validation is available below. It runs the expensive
full 264-config/cycle validator. The launcher always repeats full admission of
the actual current bytes, so a separate validation is unnecessary for launch.

```sh
PYTHONPATH=src python3.12 -B scripts/neural/curriculum_controls.py validate \
  --root . \
  --package artifacts/development/neural-combat/curriculum-campaign-v1/task-1/package-fixture/package.json \
  --source artifacts/development/neural-combat/curriculum-campaign-v1/release/source-snapshot.json \
  --release artifacts/development/neural-combat/curriculum-campaign-v1/release/release.json
```

## Launch once

The parent launches only after verifying the reviewed release. This command is
the workload entry; help, capture, template and validation never start Docker or
resolve optional ML packages.

```sh
scripts/neural/run-curriculum-campaign.sh \
  --root . \
  --package artifacts/development/neural-combat/curriculum-campaign-v1/task-1/package-fixture/package.json \
  --source artifacts/development/neural-combat/curriculum-campaign-v1/release/source-snapshot.json \
  --release artifacts/development/neural-combat/curriculum-campaign-v1/release/release.json \
  --result artifacts/development/neural-combat/curriculum-campaign-v1/launcher-result.json
```

The independent parent starts one 36,000-second monotonic allowance at its first
module timestamp, including imports, argument handling and child setup.
GNU timeout sends TERM by parent second 35,900 (rounded earlier
to include supervisor setup), then KILL after 90 seconds. The child narrows every
job by the same parent deadline, reserves closure/cleanup time, and stops new
training before second 33,600. The last ten seconds cover independent parent
result publication and exit. `launcher-result.json` records the **actual GNU
timeout watchdog wait status** in `watchdog_exit_status`, including124/137 and
control statuses. The campaign Python process is a grandchild: its OS exit is
explicitly `unavailable`, never presented as directly observed. Its independently
read logical seal and finalization veto are separate fields. Whole-run completion
requires watchdog0, a matching complete logical seal, no finalization veto and
no parent control. The elapsed measurement ends before parent publication and
does not pretend to include its own write.
Parent interrupt forwarding targets only its newly owned process group; repeated
signals cannot bypass waiting. No unrelated PID is signaled.
Before claim or Docker admission, production `run` verifies the actual Linux
controller→GNU timeout→launcher PID/start-time/cmdline/session chain, the exact
TERM/KILL duration, controller source binding and the launcher's still-open owned
result descriptor. Public metadata is read only for this known chain. Caller
timestamps alone cannot bypass the independent watchdog. The launcher creates an
exclusive sibling `<result>.supervision.json` and binds its raw hash; there is
no public test/skip-admission mode.

Static admission checks the full source/package/parent/held-input/runtime/user
chain before acquiring the durable exclusive campaign claim. Once claimed, this
runtime domain cannot be resumed or retried. Operational failures stop the
campaign. Failed/ambiguous Docker create never establishes container ownership.
Successful create must return a full immutable ID; inspect must verify the exact
image, resource, environment, label and mount contract. Cleanup reinspects and
removes that full ID at most once, then checks absence. A container with changed
ownership/configuration is retained as a failure rather than removed by name.
The Task 2 signal scope wraps the unchanged generic scope and preserves its
actual first control object. Different/repeated signals within one cleanup
command cannot replace it. A local transaction adapter promotes that first
control over an earlier ordinary fault only after generic full-ID cleanup;
all handlers and context state are restored on exit. This protection remains
active through campaign seal and fsync. Signals during sealing defer until the
owned publication finishes; the actual first control then stops the controller.
Any sealing failure publishes `campaign-seal-failure.json` when writable and
vetoes success. A logical `campaign-result.json` explicitly has
`whole_run_claim=false`; a partial/late seal can never establish completion by
itself, including when writing its failure veto also fails.

## Frozen schedule and accounting

1. Invoke the unchanged held qualification method once, with shared ceilings
   of512 actions/512 samples/8 Adam/256 resets, 120-second soft and at most
   600-second whole lease; output is within 1 GiB of the total 12 GiB allowance.
   Failure stops everything. Actual totals may be lower; no filler effects run.
   Host independently reconciles actual effects with both real reports/saved
   prefixes and the expected update1→2 progression. Qualification weights never
   become a campaign parent.
2. Evaluate the original B checkpoint on the fixed 64-case CPU development panel.
3. Run fixed, randomized and mixed arms in that order. Each starts from original
   B weights with fresh state and paired seeds 1103/2207/3301/4409. Each short arm
   gets at most four segments, 16,384 actions, 1,024 Adam attempts and 3,000 whole
   seconds. Evaluate each available matched checkpoint afterward. Short/censored
   prefixes remain unequal when actual resource use differs.
4. Continue the planned mixed branch, including its matched counters, up to 256
   total shards, 1,000,000 actions/65,536 Adam attempts or the training cutoff.
   Evaluate at cumulative mixed charged training opportunities 6,000/12,000/18,000
   seconds and finally. This includes matched mixed job wall, setup, execution,
   cleanup and host attempt publication/closure, and excludes CPU evaluations,
   other arms and shared controller time. These are opportunities, not promises
   of completed evaluations.

There are eight preregistered panel opportunities, at most 512 case executions,
and **one shared 1,800-second evaluation lease** including setup, guest lifecycle,
cleanup and host attempt publication. Panels use actual deterministic CPU
evaluation, no sample draws or optimizer. Unreached, unavailable, censored and
incomplete rows remain explicit. No score changes the chosen training branch.
A healthy save that advances no native actions stops its arm instead of spinning.

Global caps are 1.2M native attempts, 1.1M stochastic sample rows, 70K Adam
attempts and 1.1M resets; random controller choices are zero. Durable reservations
precede forwarding. Actual attempts, delivered effects, reserved leases and
uncertainty remain distinct. Unknown/failed attempts retain their whole lease;
there is no refund or retry based on an absent receipt. A successor requires the
actual saved prefix, state/config/RNG/checkpoint/receipt chain, zero uncertainty,
clean process/lifecycle evidence and successful Docker cleanup.

Source is byte-copied to new inodes **once**. Each job rechecks that frozen source
and copies only its selected package inputs, or all 64 panel configs/manifests,
or the held qualification inventory. The full package is metadata-only in the
guest. Current parents, control records, source, runtime and selected inputs are
read-only. Qualification uses its exact historical prepared input/parent paths;
campaign jobs use `/inputs/parent`. Exactly one writable mount is the real config
`runs` output and scratch root. Every TMPDIR/cache/Torch/CUDA/HOME environment
root is this existing, initially empty output directory itself. Optional
libraries may create their own children only after real context/empty-root
validation. This avoids creating scratch directories before the held producer's
empty-root checks; there is no shared cache or reuse across jobs. All nested targets under RO parents already
exist. A new empty owned Docker client directory excludes user credentials.

Every job reserves at most 1 GiB output within the global 12 GiB new/40 GiB
aggregate/30 GiB disk-free ceilings; GPU admission requires at least 12 GiB free
VRAM and the pinned CDI profile. Output is polled during execution; aggregate
space is rechecked at admission, every five seconds and closure. These are host
admission/monitoring checks, not a filesystem quota or an inode quota. The
qualified CDI scratch mount retains its separately reviewed small exception.
Unrelated GPU process observations, including PID 3772, are retained and never
treated as cleanup authority.

## Retained outputs

The fresh runtime root is
`artifacts/development/neural-combat/curriculum-campaign-v1/runtime/`:

- `campaign-claim.json` and `job-claims/*.json`: exclusive durable authority and
  reservations; these are never rearmed.
- `source/`: one frozen common source copy, including exact modes/inventory.
- `jobs/<job-id>/control/`: raw context, release, snapshot, claims, layout and
  optional immediate predecessor receipt; all mounted RO.
- `jobs/<job-id>/output/`: actual reports/checkpoints, `job-receipt.json`, failure
  evidence and `campaign-control/attempts.json` plus bounded effect journals.
- `attempts/<job-id>.json`: actual Docker command stdout/stderr/exits, owned
  cleanup/admission outcome, reserved/actual effects and failure/control notes.
- `progress/NNNN.json`: append-only prefix/effect/availability snapshots.
- `campaign-result.json`: controller logical result, all true prefixes,
  availability, charged/reserved attempts and error, with no whole-run claim.
  `campaign-seal-failure.json`, when present, vetoes completion. The independent
  sibling `launcher-result.json` records actual watchdog wait/accounting and
  separately reports controller logical evidence and unavailable controller OS exit.

Inspect these files read-only. No monitoring command changes a service or
signals a GPU process. If a timeout, native fault, close failure or publication
failure occurs, preserve every byte for review. No automatic continuation runs.
Results remain simulator-only standalone diagnostics with RNG-001/TIME-001,
one paired replicate and unknown historical exposure; no final-policy or
original-game transfer claim follows.
