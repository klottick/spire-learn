# Task 2 report: Gym 1.0.0 runtime profile and existing-mode templates

## Implemented

- Added `configs/neural/gym-v1.0.0/standalone.toml` and `run.toml`, based on the afb51f5 templates. They preserve the learner settings and component hashes, bind revision `1929e18b9f44372e020877bd62f2a5c9dee3e64e`, and carry only the existing `standalone` and `run` raw-mode mappings with the `+gym-1.0.0` rules epoch.
- Added `configs/neural/gym-v1.0.0/runtime.json` using schema `spire-learn-gym-runtime-profile-v2`. It records the release build identity, wheel/native hashes, common/run/card metadata and Standard fingerprints, required local artifact paths, and keeps dependency venv/lock values from the afb51f5 profile. Qualification-specific values are null because qualification is outside this task.
- Updated the fixed-campaign runtime schema gate in `scripts/neural/curriculum_profiles.py` to allow v1 and v2 while retaining the existing v1 checks. Added a regression test for v2 acceptance and tests for the new templates/profile in `tests/unit/test_neural_gym_v1.py`.

## TDD evidence

### RED

Added template/runtime tests before creating the three files, then ran:

```text
PYTHONPATH=src python3.12 -B -m unittest discover -s tests -t . -p 'test_neural_gym_v1.py'
```

Result: 8 tests run, 3 errors. The missing runtime JSON raised `FileNotFoundError`; template loading wrapped missing-file `FileNotFoundError` as `ValueError`. After adding the v2 validator regression test, it also failed at `overlay_pins()` with `ValueError: fixed Gym runtime profile differs`, confirming the v1-only schema gate rejected v2.

### GREEN

The focused Gym v1 module passed after implementation:

```text
Ran 9 tests in 0.021s
OK
```

Other required verification:

```text
PYTHONPATH=src python3.12 -B -m unittest discover -s tests -t . -p 'test_neural_gym_migration.py'
Ran 10 tests ... OK

PYTHONPATH=src python3.12 -B -m unittest discover -s tests -t . -p 'test_neural_config.py'
Ran 15 tests ... OK

PYTHONPATH=src python3.12 -B -m unittest discover -s tests -t . -p 'test_neural_curriculum_fixed_profile.py'
Ran 16 tests in 92.100s
OK
```

Both validate-only invocations succeeded and printed `configuration valid; no neural runtime or artifacts created`:

```text
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/gym-v1.0.0/standalone.toml --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/gym-v1.0.0/run.toml --validate-only
```

The wheel and native library were hashed without importing or executing Gym. Observed hashes match the spec:

```text
wheel:  d157065d9389c92094bef34fc3a58024dd7c2632486dd60a7b4a9f6ace4774ad
native: 5398d2fd591e025f84435efb4d80909995a764303ce0bc6d87d74c68d4be88cf
```

The delivery manifest SHA-256 used in the profile was read from the local release input: `a3bda273c3c4999232abb640ca7c2241bad922b8d575e519ec2e800410eb3427`.

Per the task brief, `test_neural_curriculum_controls.py` was run once after changing the validator. It ran 49 tests and failed in two existing assertions:

- `test_actual_owned_launch_proc_access_reaches_only_rejected_static_admission` expected `source snapshot schema differs` but received `ValueError: whole campaign deadline differs`.
- `test_complete_controller_projects_real_gate_baseline_training_and_time_cutoff` expected a shorter job list and got three additional panel/final jobs. This is the pre-existing failure named in the task brief.

These failures are outside the runtime schema gate and neither test was modified.

## Files in this task

- `configs/neural/gym-v1.0.0/runtime.json`
- `configs/neural/gym-v1.0.0/standalone.toml`
- `configs/neural/gym-v1.0.0/run.toml`
- `scripts/neural/curriculum_profiles.py`
- `tests/unit/test_neural_gym_v1.py`
- This report

## Self-review and concerns

- `git diff --check` passed. The new templates load against the actual contract and only admit standalone/run mappings; the runtime fingerprints and release hashes match the contract/spec values.
- The requested validators created no runtime or artifacts. No dependency installation, training, Gym/native execution, Docker, or edits to a Gym checkout occurred.
- The campaign-controls module has the two failures listed above. In addition, concurrent work appeared in unrelated `src/spire_learn/neural/` and `tests/unit/test_neural_views.py` paths while this task was running. Those files were left untouched and will not be included in this task's commit. The pre-existing untracked `task-6-core-review.diff` was also preserved and excluded.
