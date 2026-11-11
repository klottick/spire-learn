# Spire Learn

Machine-learning agents, training and experiments for Slay the Spire 1, consuming
[Spire Gym](https://github.com/klottick/spire-gym) as the environment.

## Repository boundary

- **spire-learn:** models, optimizers, rollout collection, training, evaluation,
  experiment configuration and checkpoint management.
- **spire-gym:** authoritative Rust simulation, game rules, native batching,
  public observations, legal actions/masks, rewards, termination and replay.

Do not duplicate gameplay rules here or modify the gym as part of learner work.
Use supported public interfaces; seeds, hidden RNG state, unseen draw order and
privileged simulator diagnostics are not policy or critic inputs.

The eventual objective is Heart victory across all four standard characters and
Ascension 0–20. Report Act 3 completion separately. Each experiment must name the
environment revision, supported content/profile, observation schema and reward
contract; planned gym capabilities are not implemented learner support.

## Current state

This is a local, dependency-light learner with initial Ironclad combat checkpoints. The
public combat model, PPO update path, collection, checkpoint and Task 7 training
and evaluation runner are authored. The single approved Task 7 CPU smoke ran
once: both mode tests passed with zero skips, recording 396 learned decisions,
444 environment slots, 28 accepted optimizer steps, and two updates per mode.
This is bounded training-plumbing evidence. Approved Tasks 1–7 are accepted
after parent and independent review. The Task 8 RTX 5090 calibration also
completed: 948 environment actions, 942 policy samples and three disposable
GPU optimizer probes, with no simulator faults. B16 was the fastest tested
standalone batch. Its [retained calibration report](artifacts/development/neural-combat/task-8/calibration/calibration-continuation-runtime-report.md)
records performance, memory, cleanup and qualifications. Task 8 is accepted
after parent and independent source/evidence review. These are bounded plumbing and performance results;
no playing-strength or frozen-policy evaluation is claimed. The general
training runner now has an authored device-aware extension with pure-test
coverage. The reviewed bounded GPU proof passed both standalone and normal Run
mode: two tests, 64 environment actions, 40 learned decisions and eight accepted
optimizer steps. Both final checkpoints passed same-backend state restoration
and CPU portability comparisons. See the [retained runtime
review](artifacts/development/neural-combat/task-9/gpu-proof/runtime-review-gpu-proof-20261029-e.md).
The separately approved bounded study then trained four checkpoints across
43,163 learned decisions and completed all 192 CPU selection comparisons, with
zero simulator faults. Results are mixed and seed-sensitive: see the
[study results](docs/neural-study.md). All normal-Run trials ended in death;
no full-Run victory, final-evaluation or original-game-transfer claim is made.
The earlier one-use study is complete. The subsequently approved overnight
campaign halted on a native Run fault after 79 healthy training segments
(238,574 learned decisions). Its 350 complete and two incomplete evaluation
rows show mixed intermediate results; no final-policy evaluation ran. The
[partial campaign results](docs/neural-study.md) preserve the fault, checkpoints,
accounting and coverage limits. No retry or further workload is authorized.
See the [retained CPU runtime
report](artifacts/development/neural-combat/task-7/task-7-training-smoke-runtime-report.md),
[development and evaluation checks](docs/development.md), and [neural combat
guide](docs/neural-combat.md) for evidence and qualifications. The M1 evaluator
and bounded CPU/native probes remain separate from this training evidence.
Historical PPO and two-policy proposals remain references, not implementation
commitments.

The randomized Ironclad combat curriculum has passed its bounded native/CUDA
qualification. The approved ten-hour campaign then stopped after about 22 minutes
on a native standalone combat fault in its first randomized segment. Four fixed
segments saved 16,384 training actions; development-panel wins rose from 22/64 to
31/64 in one paired replicate. Randomized training published no checkpoint and
mixed training never started, so their learning benefit remains unverified. The
[curriculum results](docs/neural-curriculum.md#2026-11-01-campaign-outcome) preserve
the failure, saved work and limits. The campaign is closed; no retry or further
workload is authorized.

Gym's November 1 native-fault fixes at `a86df922` were staged in a separate
Learn runtime overlay. Both adapters and the unchanged schema identities passed
a zero-gameplay container check. See the [runtime upgrade](docs/neural-runtime-upgrade.md)
for the exact pins, diagnosed causes and new config templates. Historical
campaign records remain unchanged. A separately approved campaign on this
release then stopped after about 25 minutes on another native standalone fault.
Six healthy segments saved 24,576 decisions and 1,536 optimizer steps, including
two randomized checkpoints. Fixed training's paired development panel moved
from 31/64 to 35/64 wins; randomized evaluation and mixed training never started.
The [fixed-Gym campaign results](docs/neural-curriculum.md#fixed-gym-campaign-c)
record the retained audit, independent review and P1 Gym issue `spire-14g`.
The original Learn audit did not determine the internal cause. Training is
stopped; the consumed campaign cannot be retried.

Learn has now migrated both adapters to the November 2 Gym build `afb51f5f`,
which includes later fault fixes and reordered public vocabularies. A released
CPU-only check verified the exact binary, complete schema and card metadata,
and constructed/closed both real adapters with zero gameplay. Future runs on
this release require fresh weights; old checkpoint IDs cannot be relabeled.
See the [migration record](docs/neural-gym-migration.md) for the new templates,
pins and limits. No new training was launched by this migration.

The separately approved fresh-weight campaign on this release has now completed
its development comparisons. Across 438,272 learned decisions, the final mixed
combat policy improved from **18/64 to 41/64 development wins**. All eight panels
completed, with no simulator faults or uncertain effects in the reconciled jobs.
A user-requested checkpoint cutover preserved model/optimizer/RNG state and
raised observed mixed-training throughput from 13.60 to 61.92 decisions/s.
See the [complete results and limits](docs/neural-curriculum.md#fresh-gym-campaign-results-2026-11-03).
This is one paired standalone Ironclad experiment; the protected final split
remains unrun. The original interrupted execution and both source phases are
retained. No additional training is authorized by this completion.

A later sustained campaign and its repaired continuation saved further mixed
training, but stopped on Learn orchestration failures. The available later
development comparison was 32/64 wins versus its parent checkpoint's 35/64;
latest saved shard442 is unevaluated. The
[November 4 investigation](docs/learning-investigation-20261104.md) records the
Docker timeout/cleanup issues, learning-diagnostic gaps and proposed comparisons.
Training and monitoring are stopped; no full-campaign or final-split result is
claimed for that continuation.

Python **3.12** matches the gym's current Python package requirement. Spire Learn
declares no runtime dependencies, and this project has not installed packages
into the host Python. Separate protected CPU runtime environments used for the
recorded native/probe evidence do contain qualified locked dependencies; see the
[development guide](docs/development.md). The gym is not a declared dependency.
Its GitHub repo name is `spire-gym`, but its Python distribution/import names
remain `spire-lab` / `spire_lab`; do not assume a PyPI package named `spire-gym`
exists.

From this directory, check the package without installing anything:

```sh
PYTHONPATH=src python3.12 -B -c "import spire_learn; print(spire_learn.__doc__)"
```

Hatchling is declared only as the Python packaging backend. Building/installing
the package will require that backend; no build or installation is part of this
bootstrap. Choose and lock learner dependencies in a separately scoped setup.

## Documentation

Start with the [copied reference index](docs/reference/README.md). It explains
which gym documents were copied, their provenance and the historical requirements
that must not be mistaken for current APIs or learner decisions. These are copies,
not a fork of simulator ownership; the gym originals remain unchanged.

Local code lives in `src/spire_learn/`. Keep large/generated datasets, checkpoints,
rollouts and experiment logs out of Git; document their provenance and storage
when actual experiments are introduced.

The [learning-strategy review](docs/learning-strategies.md) explains what the
expanded online literature search supports, what has failed in particular
settings, and the implications for our PPO, imitation and curriculum comparisons.
The [November 4 investigation](docs/learning-investigation-20261104.md) connects
those findings to the later training plateau/regression, the Docker timeout and
cleanup bugs, and proposed short comparisons before further sustained training.
The [Gym capability assessment](docs/gym-learning-capabilities.md) separates
supported strategies from environment gaps and learner-side work, including
the public-information requirements for any future search interface.

## Distribution

Local Git history exists. A license for new learner code has not been selected.
Copied gym documentation retains its upstream attribution/license, recorded in
the reference index. Original game code, binaries, assets and private verification
artifacts do not belong in this repository.
