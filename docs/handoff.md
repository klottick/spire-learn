# Spire Learn session handoff — 2026-10-23

## Start the next session here

**Working directory:** `/home/klottick/repos/spire-learn`.

Read, in order:

1. [Agent guidance](../AGENTS.md).
2. [Repository README](../README.md).
3. [Reference index and historical-design caveats](reference/README.md).

The previous session ran from `/home/klottick/repos/spire-lab` but created this
separate sibling repository. Do not accidentally continue learner work in the gym.

## User intent and completed request

The user chose **spire-learn** for the ML side of Spire Gym and authorized:

- Creating a folder under `/home/klottick/repos` with basic Python/repository setup.
- Copying relevant documentation from the gym, explicitly **not moving/deleting
  it or changing anything in the gym**.
- Producing this handoff for a new session.

The agreed bootstrap is complete. No further feature implementation is underway.
The next session should establish the first learner milestone with the user,
not treat this handoff as authorization to implement a training system.

## What exists

```text
spire-learn/
  .git/                         local repository, main, no commits
  .gitignore                    Python/build files, secrets, generated ML outputs
  .python-version               3.12
  AGENTS.md                     learner/simulator boundary and safety guidance
  README.md                     scope, scaffold status and import smoke command
  pyproject.toml                spire-learn 0.1.0, no runtime dependencies
  src/spire_learn/__init__.py   docstring only; no behavior or import side effects
  docs/handoff.md               this handoff
  docs/reference/README.md      reading order, provenance, historical qualifications
  docs/reference/spire-gym/
    LICENSE                     upstream MIT license for copied material
    SOURCE.json                 exact source paths/revision and SHA-256 hashes
    docs/
      combat-environments.md
      scope.md
      intentional-deviations.md
      superpowers/specs/
        2026-10-11-spire-lab-ml-first-design.md
        2026-10-16-project-randomness-design.md
```

Everything remains **untracked/uncommitted**. There is no configured remote,
GitHub repository, lockfile, virtual environment, test suite, trained agent,
dataset, experiment or checkpoint in this new repo.

GitHub checks earlier in the session found no exact public `spire-learn` name
match, and `klottick/spire-learn` did not resolve. That was an availability check,
not a reservation; recheck if remote creation is later authorized.

## Scope and decisions not yet made

- **spire-learn owns:** models, optimizers, rollout collection, training,
  evaluation, experiment configuration and checkpoint management.
- **spire-gym owns:** authoritative Rust gameplay, native batching, public
  observations/actions/masks, environment rewards/termination/reset and replay.
- Long-term goal: Heart victory across all four standard characters at A0–20;
  distinguish Act 3 completion, Heart victory and genuine death.
- No learner framework, algorithm, model architecture, first training profile,
  budget or evaluation corpus has been selected. Historical PPO defaults and
  separate combat/run-policy ideas were copied for context, not adopted wholesale.
- Python requirement is `>=3.12,<3.13`, matching the gym package at bootstrap.
  `python3.12` is available locally; the unqualified `python3` was 3.14.4, so use
  the explicit interpreter. Hatchling is declared as the build backend only;
  no backend or runtime dependency was installed.
- No license has been chosen for newly authored learner code. Retaining the
  upstream MIT license does not license all new code or third-party datasets.

## Gym integration cautions

The gym's GitHub name is `klottick/spire-gym`, its local checkout is
`/home/klottick/repos/spire-lab`, and its historical Python distribution/import
names are `spire-lab` / `spire_lab`. Do not invent a dependency on a PyPI package
named `spire-gym` or copy the gym's environment/lockfiles/native binaries here.

The copied interface guide documents the bounded supplied-entry Ironclad A0
combat/Gymnasium API. Newer native run milestones do not imply that a full-run
Python extension exists or is installed. Gym development is proceeding separately;
early M1–M3/M4 status descriptions from the conversation are stale. Check actual
current gym interfaces/revision read-only when choosing a consumer path. The
copied docs are snapshots, not a live task board or proof of installed support.

Keep policies and critics restricted to public information. Seeds, hidden RNG,
unseen draw order and privileged replay/debug metadata are not model features.
Action indices are decision-local; obey masks and actual final/reset observation
semantics. Simulator faults are errors, not gameplay deaths or fabricated
truncations. Do not silently retry accepted effects.

Disclose RNG-001/TIME-001 in relevant results. Source-grounded/reference-informed
simulation is not original-runtime validation or evidence of original-game
transfer. Copied historical tensor shapes, rewards, budgets and capture gates are
not necessarily the current environment contracts.

## Verification already performed

At bootstrap, direct offline checks passed for:

- Importing `spire_learn` with Python 3.12, without importing gym/ML libraries.
- Parsing the TOML and checking package name/layout and empty runtime dependencies.
- Comparing all five copied documents plus license byte-for-byte with the originals
  and the hashes recorded in `SOURCE.json`.
- Resolving links in the newly written README and reference index. Relative links
  inside exact upstream copies may refer to files intentionally not copied.
- Ignore rules for environments, secrets, datasets, rollouts/checkpoints/logs,
  while allowing source, configs, small authored fixtures and documentation.
- Authored-file whitespace, no symlinks, no commits and no remotes.

No package build/install, gym integration test, upstream suite or GPU/training
run was performed. There is no test-suite pass claim for this docstring scaffold.

Quick rechecks from the repository root:

```sh
PYTHONPATH=src python3.12 -B -c "import spire_learn; print(spire_learn.__doc__)"
python3.12 -B -c "import pathlib, tomllib; p = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(p['project'])"
git --no-optional-locks status --short --branch
git remote -v
```

## Recommended next conversation

Choose the smallest useful ML deliverable against an actually available gym API.
For example, consider a public-information random/scripted baseline and evaluation
caller before selecting a training algorithm. This is a suggestion, not a selected
milestone or permission to begin implementation. Discuss whether to begin with
the existing Python combat interface or wait for/use newer run-level interfaces,
then agree on dependencies, tests and resource limits for that first deliverable.

Do not restart gym mechanics research or build a second simulator to fill API
gaps. Raise a concrete missing consumer contract instead.

## Authority and preservation

- **Do not modify any gym file** as part of this handoff/learner bootstrap.
- No commits, pushes, remote creation/publication, dependency installs, training
  runs, original-game execution or broad cleanup have been authorized here.
- Do not write to the gym's Beads board or inherit its old workers/task ownership.
  No learner board was created; choose tracking only if requested.
- Preserve all current uncommitted files. Do not copy private game sources/assets,
  credentials, task databases, agent scratch/evidence, `.venv` or build artifacts.
- Reference snapshots retain exact upstream text. Refresh them deliberately with
  updated provenance, rather than editing historical plans into learner specs.
