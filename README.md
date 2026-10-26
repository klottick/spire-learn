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

This is a local, dependency-light evaluation scaffold, not a trained learner. A
fixed-case evaluator and bounded CPU/native probes are present, but no learning
framework, training algorithm, learned architecture, or playing-strength result
has been selected or produced. The two-case selection smoke config is diagnostic,
not a benchmark corpus or permission to run. M1 foundation is complete for its
scoped CPU/native profile: the bounded evaluator smoke executed one test with
zero skips and reconciled both selection cases. See the
[development and evaluation checks](docs/development.md) for evidence and
qualifications. Historical PPO and two-policy proposals
remain references, not implementation commitments.

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

## Distribution

Local Git history exists. A license for new learner code has not been selected.
Copied gym documentation retains its upstream attribution/license, recorded in
the reference index. Original game code, binaries, assets and private verification
artifacts do not belong in this repository.
