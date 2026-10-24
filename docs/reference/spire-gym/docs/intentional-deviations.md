# Intentional deviations from the base game

These are explicit project choices, not simulator bugs or claims of exact base-game
equivalence. Preserve this distinction in designs, tests and evaluation reports.

## RNG-001 — Decorrelated, reproducible project randomness

**Decision date:** 2026-10-16

**Status:** User-approved policy; the Python randomness foundation and narrow
headless combat reference are implemented/reviewed, and the bounded Rust combat/
environment slice implements project-owned randomness with reviewed parity tests.
Original-runtime combat validation remains incomplete. See
[repository scope](scope.md) for the Rust/learning goals this policy serves.

### Decision and rationale

Exact replication of the base game's PRNG, seed-to-outcome mapping, stream coupling
and RNG-call consumption is not required. Use project-owned, reproducible randomness
with independently derived streams for unrelated mechanics, rather than carrying
forward incidental seed/stream correlations. The purpose is to train policies on
gameplay rather than incidental properties of the original RNG implementation.

Preserve the supported game's intended probabilities and conditional rules,
including intentional dependencies such as drawing without replacement and enemy
move-history restrictions. This is not permission to make every outcome IID,
change encounter rules or replace shuffling with independent replacement draws.

The user explicitly reaffirmed conditional enemy-intention chance fidelity on
2026-10-16. Match each supported next-move law given relevant gameplay state and
selected-move history, not just overall move frequencies. Where finite sampler
semantics determine the branch mass, preserve that mass with exact rational
sampling rather than rounding the displayed decimal or choosing cleaner weights.
This does not require the original PRNG calculation or restore incidental seed/
stream coupling. The combat design records the exact supported chance tables.

Keep seeds and internal RNG state out of policy observations. Retain deterministic
replay and snapshot restoration under the project's own randomness contract.
Independent streams here are a design objective, not a claim that deterministic
PRNG streams are mathematically independent or impossible to infer.

### Correctness and reporting consequences

- Seed-for-seed or bit-for-bit agreement with the original RNG is **not** a project
  acceptance criterion. The superseded original-RNG core/full vector gates and
  exceptional-NaN replication work do not block the replacement RNG design.
- Probability laws and intentional gameplay dependencies still require evidence
  and tests. Altering incidental coupling changes the overall joint process;
  do not describe this simulator as stochastically identical to the base game.
- Validate mechanics against original evidence, using matched supplied random
  outcomes where feasible rather than assuming a shared seed produces the same
  outcomes. This does not waive existing original-game validation requirements.
- Training/evaluation reports must disclose RNG-001 and distinguish simulator
  results from original-game transfer results. Do not label an expected RNG-induced
  trajectory difference as a mechanics defect or use this exception to excuse one.
- Later Python/Rust parity should target the approved project randomness contract,
  not original PRNG parity; agreement still does not establish game correctness.

### Scope and retained history

This decision supersedes conflicting exact-original-RNG requirements only. It does
not waive non-RNG arithmetic, card/enemy rules, effect ordering, visibility, fault
semantics or other correctness/privacy constraints.

Retain prior accepted RNG evidence and failed histories unchanged as reference
material. The 2026-10-15 exact seeded-RNG design remains a superseded draft, not
approved implementation. The replacement design/plan separately specifies the
algorithm, stream boundaries, seed derivation and state representation; this
policy alone does not authorize simulator edits, new evidence access or Git writes.

Approved replacement design:
[`2026-10-16-project-randomness-design.md`](superpowers/specs/2026-10-16-project-randomness-design.md).
The user separately approved its design and
[implementation plan](superpowers/plans/2026-10-16-project-randomness.md) on
2026-10-16. The standalone module `python/spire_lab/combat/rng.py` is implemented
with fixed project/public vectors and reviewed reproducibility, sampling, shuffle
and exhaustion tests. Its standalone milestone recorded 395 passing Python tests.
The subsequent [combat integration](superpowers/specs/2026-10-16-python-combat-behavior-design.md)
implemented typed stream ownership, public-information boundaries and schema-2
snapshots, with 735 tests passing in its final implementation verification. The
opaque schema-1 RNG slot is rejected rather than retained. These historical test
results establish authored behavior/replay, not original-runtime validation or
learned playing strength; independently authored Rust parity remains future work.

## TIME-001 — Secret Portal's elapsed-time requirement is assumed satisfied

**Decision date:** 2026-10-22. **Status:** user-approved full-run gym policy;
implementation remains subject to the full-run plan/start gate.

Treat Secret Portal's original 800-second run-playtime requirement as always
satisfied. The selected gym models ordinary play, not competitive speedrunning;
this narrow eligibility simplification is intentional, not elapsed-time parity.

Keep Act 3 and all other ordinary eligibility, event-pool/history and random
selection rules. Eligible does not mean guaranteed: do not force the event,
restore it after consumption, or expose its future selection. Accepting still
routes to the actual boss fight; declining remains available. Keys, A20's second
Act 3 boss, rewards and terminal rules are unchanged.

No run clock, frame simulation, wait action, elapsed-time feature or host-speed
dependency is required for this policy. Focused event tests must cover the assumed
time condition and unchanged non-time eligibility; existing replay/batch checks
protect independence from caller delay and worker scheduling. Retain historical
source/time findings unchanged; they describe the original rule, not this gym
choice. No product code or original-runtime-validation claim follows from recording
the decision.
