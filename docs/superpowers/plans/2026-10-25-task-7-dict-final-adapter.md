# Task 7 dictionary final-observation reader implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task by
> task after explicit user approval. Luna max reasoning, no nested workers.
> Canonical workspace remains artifacts/development/foundation; main authorized.

**Status:** User-approved on 2026-10-25 ("I approve all changes"); execution may
proceed within the exact scope and gated runtime limits below.
**Goal:** Test a bounded public Dict-final reader adaptation against the observed
TorchRL terminal mapping mismatch, without modifying Gym or installed packages.
**Architecture:** A new lazy optional reader module supplies a known-spec
BaseInfoDictReader through the existing passive GymWrapper registration hook.
The existing guarded diagnostic probe verifies actual same-call mapping.
**Tech stack:** Existing Python3.12/Torch2.14.1+cpu/TorchRL0.14.1/
TensorDict0.14.3/Gymnasium1.4.0/NumPy2.5.4 and own Gym0.1.0 only.
**Spec:** [Proposed narrow design](../specs/2026-10-25-task-7-dict-final-adapter-design.md)
and approved foundation design§7. Original plan Task7 says incompatible results
stop for a user-approved revision; the user has now approved this revision.

## Global constraints

- Explicit approval covers this revision and its gated new validation shard;
  no automatic rerun of the failed cohort. Retain incompatible run and setup96/32
  qualification.
- Public observations/masks only; seeds/route IDs/private diagnostics never actors.
  No game rules, copied Gym code/fixtures, native edits or opaque-state copying.
- Keep `probes/vector.py`, public contracts and installed venv/wheels unchanged.
  No generic Env framework or broad rewrite of the already large TorchRL probe.
- Existing guards, immutable per-attempt logs/source snapshots, raw capture and
  reservation-before-draw limits remain mandatory. Stop on new error/cap; no
  semantic fallback, pin change, implicit rollout or accepted-effect repair.
- Runtime controls and exact new-run authority are those in the companion design.
  One shard, no host dependencies/training/GPU/shared-service changes/publication.

## Review focus

1. `outcome` exists in both observation and info: observation final leaf must
   come from final_obs only, not info.outcome.
2. Mixed ended/live rows: None/presence and input arrays stay unchanged; terminal
   leaf dtype/shape and final/reset separation remain faithful.
3. Missing/partial final fields or wrong masks fail before assigning prepared
   output, not with fabricated terminal zeroes or a hidden reset.
4. Known-spec registration remains passive; no accidental lazy setter rollout.
5. Returned native effects survive reader/wrapper errors; guard/draw/close counts
   and artifact failure diagnostics remain truthful.

## Task 1 — Small reader module and core contracts

**Create:** `src/spire_learn/probes/torchrl_dict_final.py`,
`tests/unit/test_torchrl_dict_final.py`.
**Modify narrowly:** `src/spire_learn/probes/torchrl.py` public-runtime imports,
reader-factory selection, adapter identity and per-key mismatch diagnostics only.
**Interfaces:** Keep `run_torchrl_probe(config,writer,*,clock)->ProbeReport`.
New lazy factory:
`make_dict_final_reader(*, observation_spec: object, schema: ObservationSchema,
 torch: object, info_reader_type: type, composite_type: type) -> object`.
It returns an initialized public BaseInfoDictReader implementation with known
`info_spec`, `__call__(info_dict,tensordict)` and adapter ID
`spire-learn-dict-final-reader-v1`. No optional import at module load.

Private injection interface in `torchrl.py`:
`_passive_gym_wrapper_class(gym_wrapper_class: type, *,
 reader_factory: Callable[[object], object] | None = None) -> type`.
When explicitly configured, its `auto_register_info_dict` override ignores the
supplied vendor reader and calls `reader_factory(self.observation_spec)` once.
The factory binds the already parsed schema and imported public
BaseInfoDictReader/Composite/Torch types, then calls `make_dict_final_reader`.
Pass that learner reader to the existing passive setter path. Require exactly
one registered learner reader, zero registered vendor readers and exact final
leaf-spec agreement. The None default retains the previous hook for core tests;
the revised production probe explicitly selects the new reader.

- [x] Write focused pure tests first for owned leaf preparation, missing/presence/
  dtype/shape rejection, no input mutation, info/observation outcome collision,
  and passive known-spec registration. Use authored protocol fixtures, not Gym
  fixtures or a fake simulator. Assert actual expected values, not implementation.
- [x] Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_torchrl_dict_final -v`;
  retain real RED commands/output/exits. Missing runtime is not behavioral RED.
- [x] Implement only declared public Dict tensor leaves from final_obs; validate
  all rows before assigning prepared output. Reuse existing core snapshot and
  presence validation. Derive ended mask from actual TD done when present;
  absence is allowed only for registration/reset with no final fields. Require
  equality with public final masks. No reading routing/hidden fields as features.
- [x] Inject via GymWrapper's public info-reader registration hook, with known
  Composite final spec. No vendor/installed-package monkeypatch, private reader
  storage edits, observation-key renaming or replacement of VecGymEnvTransform.
- [x] Assert in focused tests that exactly the learner reader is registered;
  the constructor-supplied vendor reader is never registered or called. A failed
  spec check must register no reader and perform no environment operation.
- [x] Add per-key/row dtype/shape/value mismatch diagnostics from existing owned
  raw and mapped snapshots without new calls/samples; persist before stopping.
- [x] Assert adapter ID persistence in discovery/summary using a real RunWriter
  in an owned temporary directory. Assert deliberately mismatching authored
  leaves persist a `mapping_mismatch` probe metric identifying `row`, `key`,
  `expected_dtype`, `observed_dtype`, `expected_shape`, `observed_shape` and finite
  expected/observed values when available (otherwise explicit unavailable).
  Never assign a partial final output before validation succeeds. This metadata
  is a diagnostic, never an actor input or a replacement native observation.
- [x] Run focused GREEN, current TorchRL guard tests, one full default suite with
  native/framework opt-in flags unset, and fresh import/discovery isolation.
  Commit only explicit task paths after parent/independent spec+quality review.

## Task 2 — One bounded zero-native/API and native validation shard

**Create:** `tests/integration/test_torchrl_dict_final.py` with exactly three tests,
opt-in `SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION=1`. Import optional packages
only inside opted-in tests. Missing package with flag set is error, never skip.
**Keep:** existing single `tests.integration.test_torchrl_adapter` cohort test.

- [x] Add `test_mixed_dict_final_values_and_presence`: real CPU TensorDict/spec,
  authored B2 fields including player and action_mask; only ended row filled,
  absent None stays absent and input/main reset arrays are unchanged.
- [x] Add `test_info_outcome_does_not_replace_observation_outcome`: deliberately
  different authored info outcome and nested observation outcome; assert nested
  observation wins with exact dtype/shape/value. Zero environment/sampler calls.
- [x] Add `test_reader_registration_has_zero_environment_calls`: known spec,
  empty registration/reset info and no-final ongoing data; no implicit reset,
  step/check_env_specs/rollout. Use a counter-only registration owner plus actual
  public reader/spec/TensorDict types; real GymWrapper passivity is proved only
  by the following native probe's construction guard. Assert malformed masks/
  fields fail explicitly.
- [x] Review new code/tests before runtime. Use one fresh exclusive output/marker
  and pinned protected offline container; interpreter `/work/venv-task7/bin/python`.
  Venv/src/tests RO, fresh output RW, no whole-root/home/credential/socket mount.
- [x] Run the three reader tests first (require3executed0skipped, <=60s, zero
  `spire_lab`/native-binding imports, environments, actions or draws). The stdlib
  learner adapter may be imported. Then, only if green, run exactly the existing one
  framework cohort test (require1executed0skipped) within the same600s shard.
  No new constructor experiment or sampler smoke beyond these declared checks.
  Exact first command inside the protected container, cwd `/learner`:

  ```sh
  env SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION=1 PYTHONPATH=/learner/src /work/venv-task7/bin/python -B -m unittest tests.integration.test_torchrl_dict_final -v
  ```

  Run as its own subprocess, retain stdout/stderr/exit and require exit0 plus
  exactly3 tests and0 skips; the tests assert no `spire_lab` or binding imports
  before/after. The orchestrator verifies these postconditions and then, only on
  success, runs the following separate subprocess. Do not invoke both modules
  via discovery or rely on shell exit0 alone; missing/ambiguous result evidence
  stops before any native construction.

  ```sh
  env SPIRE_LEARN_RUN_TORCHRL_INTEGRATION=1 PYTHONPATH=/learner/src SPIRE_LEARN_PROBE_OUTPUT_ROOT=/probe-work/runs /work/venv-task7/bin/python -B -m unittest tests.integration.test_torchrl_adapter -v
  ```

  Both commands share one fresh exclusive output/marker, protected container
  and600s hard shard including cleanup. Failure of the first command ends the
  shard with no cohort; failure of the second ends without another invocation.
- [x] Require constructor nativeforward0, reset1, <=128 steps, <=256 reserved
  draws, <=60s probe, actual mixed row plus later fresh replacement decision,
  all8checks pass and exact old-final/main mapping; verdict must be adaptable.
  Any failure stops with preserved evidence and no retry or compatibility claim.
- [x] Parent and independent reviewer inspect source hashes, actual output,
  payload mapping diagnostics, counters/protections/cleanup and schema/build.
  Only then consider Task7 acceptance/contract freeze. Tasks8–10 remain pending
  until that gate; this approval does not waive the historical setup deviation.

Reports/logs stay in `artifacts/development/foundation/`; do not overwrite prior
reports, venv, run artifacts or user changes. Use fresh task-specific report names.

## Accepted execution

Parent accepted on2026-10-25 after independent code/runtime reviews. Tested source`51194e1`,3APItests/0skips then1cohort/0skips; adaptable, all8checks pass,11batch22slot1replacement22diagnostic samples. Full record and retained historical qualifications: `artifacts/development/foundation/task-7-contract-freeze.md`. Original race-era TDD logs are qualified; actual final core and native evidence is distinguished in reports. This completes this narrow revision, not M1.
