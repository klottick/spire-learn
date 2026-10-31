# Sustained Ironclad controls

These commands prepare and consume the approved sustained-v1 package. They do
not authorize another campaign or reuse a historical study claim. Workloads
require the parent's externally reviewed stage release. No dependency install,
Gym edit or checkpoint selection is part of these controls.

Pure package validation:

```sh
PYTHONPATH=src python3.12 -B scripts/neural/sustained_controls.py --root "$PWD" validate
```

Capture source only (the destination must be new; dirty.json is a strict JSON
description of the reviewed working tree):

```sh
PYTHONPATH=src python3.12 -B scripts/neural/sustained_controls.py --root "$PWD" \
  capture-source --learner-head FULL_COMMIT --dirty-json PATH/dirty.json \
  --destination PATH/source-only.json
```

The parent creates a new snapshot with `state=reviewed`, reviews the exact raw
snapshot hash, and supplies a release. `spire-learn-sustained-stage-release-v1`
has exactly `schema`, `state=approved`, `single_use=true`, `stage` (`profile` or
`train-evaluate`), `campaign_id` (1–24 lowercase letters/digits/hyphens),
`source_snapshot_sha256` (raw snapshot file), `campaign_raw_sha256`, `stage_wall_seconds`,
`runtime_pins`, `budgets`, `review={path,sha256}`, `predecessor`, and
`initial_accounting={path,sha256}`. Pins and budgets equal campaign.json exactly.
The initial file is preparation/runtime-budget-initial.json, raw SHA
cfb5595c860dc9f753cac2f58477a6bc9e109b6c44a7ce347bb211fbe590ffd5.
Profile predecessor is null. Train/evaluate predecessor has `path`, `sha256`,
and `outer_accounting={path,sha256}`. It binds the profile's `closure.json` seal,
which binds the complete `closed-stage.json` result. Both stages
use the same campaign and source-snapshot identities; stage directories are
exclusive and cannot be reused, including after interruption.

The parent times each complete host child process externally through actual
exit. Its `spire-learn-sustained-outer-accounting-v1` receipt has exactly
`schema`, `stage`, `campaign_id`, `seal_sha256`, `exit_code=0`,
`enclosing_elapsed_seconds`, `accounted_interval_seconds` (the seal's
`stage_elapsed_seconds`), and `residual_shared_seconds` (enclosing minus
accounted, nonnegative). The next release binds that raw receipt and seal. Only
the residual is added to shared; existing profile/replica/evaluation debits are
carried unchanged. The seal's boundary is before its own serialization and
process exit; no fixed-point serialization claim is made. The same external
receipt audits the final stage. A terminal-closure-failure.json invalidates a
seal even if earlier provisional bytes were written.

The parent also chooses `stage_wall_seconds`, a primitive integer matching the
wrapper's numeric first argument. For profile it cannot exceed remaining profile
plus shared; for train/evaluate it cannot exceed the remaining four replicas,
evaluation and shared. Both are narrowed by remaining global time; unused profile
time never enlarges the sustained stage. The wrapper encloses Python startup,
all release/package/source/predecessor validation, and execution in one timeout,
with TERM ten seconds before the numeric ceiling and KILL at the ceiling. There
is no unguarded allowance-derivation subprocess. The controller rechecks the
numeric value against the reviewed release before effects and narrows internal
admission to the same absolute stage deadline.

After review, the parent invokes each stage once:
Set each seconds variable to that stage's exact reviewed numeric release value.

```sh
bash scripts/neural/run-sustained-training.sh "$PROFILE_STAGE_SECONDS" --root "$PWD" run \
  --source-snapshot PATH/reviewed-source.json --release PATH/profile-release.json --stage profile
bash scripts/neural/run-sustained-training.sh "$TRAIN_STAGE_SECONDS" --root "$PWD" run \
  --source-snapshot PATH/reviewed-source.json --release PATH/train-release.json --stage train-evaluate
```

Outputs are ignored under sustained-v1/runtime/CAMPAIGN/STAGE. Retain every
claim, context, command stream, callback journal, raw runner report, checkpoint
and closed-stage.json. No pruning or name-based cleanup is permitted. A failed
or uncertain accepted effect halts the stage; do not recreate a stage/job ID.
GPU headroom is read before each GPU admission. Other processes remain owned
by their users; insufficient headroom stops admission.

`retained-inventory.json` binds bounded path/hash/size entries for every owned
stage file, including staged source, controls, all profile/trace outputs, callback
batches, raw runner files, all checkpoints, command logs and host-job results.
It excludes only itself and the explicitly named future closed-stage/seal/failure
records, whose raw hashes are bound by the enclosing seal and external receipt.
Docker client configuration must remain empty and is never read or copied.
Links, nonregular entries, unreadable directories, added/missing/changed files
and altered static Task1 schedule fields or checkpoint bindings all reject.
Closure and predecessor admission repeat the inventory and actual producer
receipt validators; they do not rely on an earlier successful check.

Each segment has one fresh physical output bound at the unchanged config's
logical /learner output path, with only its current parent mounted read-only at
/inputs/parent. Parent hashes and prior receipt hashes live in the external
context chain; checkpoint metadata retains the actual guest mount path.
Collection counts are invocation-local; optimizer and checkpoint counts are
cumulative. Saved elapsed prefixes stop at save. Outer intervals also charge
save completion, writer/runtime close, publication, Docker cleanup and checks.
Per-job package/source/storage/parent admission starts inside the selected job
bucket and its 600-second segment/evaluation or 900-second profile ceiling.
Expired admission cannot create/start a workload; early exits restore shared
and charge the job interval once. Stage-wide release validation remains shared.

Evaluation retains all 704 fixed rows, with unavailable/not_started states for
missing checkpoints or exhausted allowances. A death remains a game outcome;
faults are errors. Scores never select or overwrite the user's chosen weights.
RNG-001/TIME-001 and simulator-only/model-selection qualifications apply.
