# Spire Learn: agent guidance

## Purpose and boundaries

- Read `README.md` and `docs/reference/README.md` first.
- This repository owns ML agents, training, evaluation, experiment configuration
  and checkpoint management. It consumes Spire Gym; it does not own game rules.
- Keep all authoritative simulation and environment behavior in Spire Gym. Do not
  copy its engine, implement a second approximate simulator or modify any gym
  checkout without a separate user request.
- Consume only public observations and legal actions/masks. Never feed seeds,
  hidden RNG state, unseen draw order or privileged diagnostics to actors/critics.
- The current contents are a scaffold, not an implemented learner. Copied designs
  are historical/reference material, not authority to install, train or implement
  their proposed architecture. Recheck the actual gym API before integration.

## Development

- Use Python 3.12 to match the current gym package. The package is `spire_learn`
  under `src/`; keep imports free of environment/device/training side effects.
- No ML/runtime dependencies are selected yet. Do not copy the gym's dependency
  list, environment, lockfiles or native artifacts as a shortcut.
- Test learner arithmetic, masks, trajectory boundaries and checkpoint/schema
  compatibility independently when those features are introduced. Keep tests
  deterministic, focused and fast; do not require GPU training to test logic.
- Genuine gameplay death is a valid episode outcome. Simulator faults are errors,
  not losses or successful truncations; do not silently retry accepted effects.
- Keep train/model-selection/final-evaluation cases separate. Record environment
  revision and dirty state, content/profile, schema, reward, learner config/seeds
  and resource budgets for experiments; distinguish simulator results from
  original-game transfer. Preserve RNG-001/TIME-001 qualifications.

## Preservation and authority

- Keep datasets, rollouts, checkpoints and generated experiment outputs out of
  Git by default; small authored fixtures/configs/reports may be versioned.
- Never copy original game code, binaries/assets, private sources, raw verification
  artifacts, credentials, `.beads`, `.venv` or historical agent scratch directories.
- `docs/reference/spire-gym/` contains exact reference copies. Update provenance
  when refreshing them; do not silently edit them into new learner specifications.
- Gym task IDs, historical agent allocations and source-repo instructions are not
  learner assignments. Do not write to its shared tracker for this scaffold.
- No dependency installs, training runs, commits, pushes, remote creation or
  publication without explicit user authority. Preserve unfamiliar/user changes.
