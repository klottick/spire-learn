# Repository scope and current direction

## Current repository boundary and priority (2026-10-22)

**Approved progressive delivery:** the [36-milestone roadmap](superpowers/specs/2026-10-22-narrow-gym-milestones-design.md)
supersedes broad family-wave scheduling, not final scope. The user approved the
[migration/M1 execution plan](superpowers/plans/2026-10-22-roadmap-migration-and-m1-combat.md)
and native single-owner start. M1 authored Ironclad A0 combat with damage/Block/
Weak/Vulnerable is accepted, not a normal run. Beads M1–36 are
`spire-dxo.23`–`.58`; M1 `.23`/`.23.1`, M2 `.24`/`.24.1`/`.24.2` and M3
`.25`/`.25.1`/`.25.2`, M4 `.26`, M5 `.27`, M6 `.28` and M7 `.29` are closed.
M7's approved [written plan](superpowers/plans/2026-10-23-m7-ironclad-act2.md) has
an [accepted captured Act2 profile](milestones/m7-act2.md) after ONE independent
milestone review. M8 `.30`/`.30.1` are planning-only/open/unassigned, not product
start. Natural Death is a valid episode outcome, not
a simulator failure; all three actual boss/reward/handoff paths complete on the
already planned seed11 while seed7 retains its natural Champ loss. The
[M2 implementation plan](superpowers/plans/2026-10-23-m2-shared-combat-mechanics.md)
is user-approved; the [M2 authored combat profile](milestones/m2-combat.md) is
accepted after integrated verification and isolated review, not full-run or
workspace-green acceptance. Only the next milestone is
decomposed. Old unfinished components remain coverage umbrellas, not early gates.
All four characters/A0–20/Acts1–4/Heart and normally reachable content remain the
final target; intermediate profiles state their limitations explicitly.

The [M3 implementation plan](superpowers/plans/2026-10-23-m3-persistent-combat-run.md)
is user-approved; [its authored consecutive-fight profile](milestones/m3-run.md)
is accepted after isolated review with no findings. It connects real fights/resource
settlement, not Neow/map/reward progression or complete content.

The [M4 implementation plan](superpowers/plans/2026-10-23-m4-act1-room-loop.md)
is user-approved; its [accepted room-loop profile](milestones/m4-rooms.md) connects actual
standard starting resources and a named authored nine-floor route/offer profile
to real Neow/map/combat/reward/shop/shrine/campfire/treasure decisions. Normal
random offer/pool generation remains explicitly incomplete. One isolated Luna6
milestone review found no issues; accepted M4 is not Act1/boss acceptance. M5
normal-start-through-Act1 implementation follows the approved plan.

The [M5 written plan](superpowers/plans/2026-10-23-m5-ironclad-act1.md) was approved
with "I approve the plan". Its [accepted native profile](milestones/m5-act1.md)
executes full generated-map/public standard-resource paths through all three
elite/boss alternatives, real rewards/relic children and same-run Act2Ready.
One isolated GPT-6 Luna milestone review found no issues; final core428/7 retains
exactly seven old draft failures. M6 planning `.28.1` is now closed after approval.
Act2Ready is nonterminal with no actions
or success bonus, not complete random pools/Act2 gameplay/full-run acceptance.
No new commit, install or cleanup authority is implied.

The [M6 plan](superpowers/plans/2026-10-23-m6-act2-mechanics.md) is user-approved:
concrete ordinary/all-three-elite/all-three-boss mechanics, actual
summon/card-custody/event effects and a generated-map prefix through floor26.
M5 remains unchanged; M7 retains complete Act2 progression. `.28.1` is closed;
native implementation `.28.2` is accepted/closed after one isolated milestone review.
The [M6 runnable profile](milestones/m6-act2.md) records three normal-start seed7
and three seed11 early-prefix completions, real encounters/events/loot and separate
six-encounter diagnostics. Independent review found no issues at the declared scope. Its floor26 cutoff
is nonterminal/no-action/no-bonus, not a full Act2 or standard-pool success claim.
M7 planning-only `.29.1` is open/unassigned; no M7 product code/start is authorized
until its actual written plan is reviewed. No commit/install/cleanup authority.

The runnable [M1 demonstration](milestones/m1-combat.md) documents its exact
four-card/three-enemy fixture profile, public action API, observed outcomes and
known pre-existing draft failures. It does not establish normal-run support.

**This repository delivers the gym, not ML code.** It owns the headless Rust
simulator, environment/batching layer, thin Python/Gymnasium interfaces, replay,
correctness tests, representative performance checks and small scripted/random
rollout examples. Models, optimizers, training loops, learner checkpoints and ML
experiments belong outside this repository. Earlier ML-first documents are
historical context; their in-repo training requirements are superseded.

The user selected a **full-run gym PoC through Acts 1–4 and the Heart**, rather
than a separately accepted Act 1-only release. It must provide seeded reset,
public observations, legal actions/masks, stepping, rewards/termination, replay
and native batching across the run lifecycle. The user selected all four standard
fully unlocked characters and all Ascension 0–20, A20-primary, not a reduced
Ironclad/A0 profile. Include all normally reachable content, including shared and
off-color acquisition paths. Reuse current combat/batching and qualified research.
The full-run written design and implementation plan are user-approved; explicit
start authority is recorded in closed `spire-dxo.3`. Workspace setup precedes code
edits. User-approved TIME-001 assumes Portal's time condition satisfied while
retaining other eligibility and random selection, without a clock or wait action.

Beads `spire-dxo`/`spire-dxo.1` track the PoC/scope. `spire-8jm` remains the broader
four-character, Acts 1–4, A0–20/A20-primary specification backlog. The custom
selector verifier and inactive standing worker/reviewer claims are deferred.
Completing a generic verification engine or the entire research backlog must not
replace the next runnable gym milestone. Keep unknown mechanics and unsupported
content explicit; deferral is not acceptance or permission to invent behavior.

Historical plans/reports retain their evidence, approvals and failures. Existing
implemented contracts still apply; no original-runtime, publication, dependency
change or new product-implementation approval is inferred. See
[work status](work-status.md) for the administrative handoff and
[environment usage](combat-environments.md) for the implemented API.

## Approved source-first sequencing amendment (2026-10-16)

The user explicitly approved moving original-game validation to a later promotion
gate rather than blocking Rust implementation. This supersedes earlier requirements
that original combat captures must precede Rust development/handoff. Build from
reviewed, independently worded mechanics, checking ambiguous decompiler details
against bytecode only under applicable access approval; do not copy or mechanically
translate original methods into product code.

The source-first milestone is governed by the
[narrow Rust combat specification](superpowers/specs/2026-10-16-source-first-rust-combat-design.md)
(approved and implemented) and its separately approved
[implementation plan](superpowers/plans/2026-10-16-source-first-rust-combat.md),
not Linux capture readiness. Its core-only handoff is verified; capture-helper/
feasibility work remains deferred.
Rust development, differential testing, then separately scoped native batching,
bindings and benchmarks may precede original captures. This sequencing approval
does not approve an unwritten Rust spec/plan or authorize implementation, training,
new original access/execution, dependencies, staging, commits, pushes or cleanup.

Until reviewed original comparisons satisfy the applicable evidence gates, label
the simulator **source-grounded, not original-runtime validated**. Python/Rust
agreement, invariants and authored cases are implementation evidence, not proof of
original fidelity or demonstrated original-game transfer. Original seed/interaction/
terminal/platform requirements remain requirements for later fidelity promotion;
they are not prerequisites for beginning the source-first Rust milestone. Preserve
all historical evidence and qualifications without retroactive repair.

## Goals, in priority order

1. **A usable gym for external ML consumers.** Deliver complete supported episodes
   with reliable observations/actions/masks, rewards/terminal outcomes, reset and
   replay, thin bindings and native batching. A runnable interface and focused
   correctness checks, not a trained agent, establish the gym milestone.
2. **An optimized, game-correct Rust simulator.** Keep one authoritative engine;
   profile representative release simulation/encoding/batching workloads and state
   the content, resource and fidelity limits of any result. Long-term coverage is
   four characters, Acts 1–4 and Ascension 0–20, with A20 the performance target and
   Heart victory distinct from Act 3 completion. Coverage and throughput do not
   prove original-game fidelity or learned playing strength.
3. **Only the supporting tools needed for those goals.** A readable Python combat
   reference reduces mechanics risk during Rust development; original-game
   comparisons later validate fidelity. They are not competing production
   simulators, a Python optimization project or a general-purpose capture/validation
   platform.

## Headless architecture

- **Rust core:** owned combat state, content/rules, effects, randomness and replay;
  no renderer, Python interpreter, tensor-library or training dependency.
- **Rust environment layer:** scenarios, public observations, legal actions and
  native batching. Keep hot-path work native, not a Python call per effect.
- **Thin bindings and Python interfaces:** numeric batch interchange, Gymnasium
  facades and small rollout examples. Models/training/evaluation are external.
- **Separate offline validation:** original observations, reviewed semantic
  comparison cases and diagnostics; not a runtime dependency of training.

The product is entirely **headless and programmatic**. No graphics, animation,
audio, human-playable client or interactive game UI. Semantic actions and decision
boundaries are APIs, not human input prompts. Original-game observation tooling may
need the original game's own runtime/platform; that does not make the simulator
playable or require a running game, display or Windows connection for training.

Policies use player-visible information, never hidden RNG state, unseen draw order
or privileged evidence metadata. Preserve intended conditional probabilities,
gameplay dependencies and project replay under the approved
[RNG-001 deviation](intentional-deviations.md); original seed/PRNG parity is not the
target. Do not describe the changed joint random process as identical to the game.

## Approved two-level learning direction (2026-10-17)

**Historical external-consumer context, not current repository deliverables.**
The policy/checkpoint/training discussion in this section is retained as history.
The 2026-10-22 boundary above supersedes any requirement to implement ML here.
The gym-facing principle remains one authoritative detailed Rust engine and a
player-visible-information boundary. No external learner implementation or new
callback/training interface is approved by retaining this section.

The user approved **separate combat and run-level policies initially, backed by
one authoritative Rust simulator**, toward one system that can play the full game.
This records the approved architectural direction, not an implemented ML system
or a detailed training/API specification.

### Policies and model boundaries

- **Combat policy:** cards, targets, play order, potion use and ending turns;
  learn to survive while preserving HP and other resources for the run.
- **Run policy:** card rewards and skips, map/path choices, shops, events,
  resting/upgrading and other supported between-combat decisions; learn to build
  a deck and resource position that can win the run.
- Start with separately trainable models/checkpoints so one policy can be frozen
  while evaluating or training the other. Keep interfaces modular and replaceable.
  Shared card/deck/relic representations with separate policy/value heads are a
  later measured option, not an initial requirement. A single unified network is
  also an experiment, not the required architecture.
- Both policies use player-visible information only. Combat decisions may need
  public run context for resource conservation; neither policy may consume hidden
  RNG state, unseen draw order or privileged simulator diagnostics.

### One simulation, two training interfaces

All production gameplay rules, state transitions, randomness and native batching
remain in Rust. Python owns ML training/evaluation and thin numeric interchange;
the Python combat reference remains an offline correctness aid, not a production
simulation layer. The Rust simulator can run without Python.

Combat training can start at a supplied combat state and step individual combat
decisions. Run training can step strategic decisions and delegate each fight to
the combat policy until combat completes, returning the actual resulting run state
including consumed resources and terminal cleanup. The run learner may treat a
fight as one larger transition, but the same detailed Rust combat engine still
executes it; there is no second approximate simulator or initial fixed-HP-loss
shortcut. A learned combat-outcome predictor may be investigated later as a
planning accelerator, with its errors measured against detailed simulation.

### Training progression and common objective

1. Establish combat competence and evaluate it on the supported content.
2. Once run simulation/content is supported, train strategic choices against a
   frozen combat checkpoint to make their consequences more stable.
3. Alternate policy improvements, retaining older checkpoints for robustness and
   evaluating the combined system rather than assuming independent gains compose.
4. Evaluate held-out full runs as support expands, ultimately targeting Heart
   success and playing-strength gains per wall-clock hour; report Act 3 separately.

Combat victory alone is a bootstrap objective, not the eventual full-game target.
HP, potions and other remaining resources affect future success. Card/draft value
also depends on the combat policy's ability to use the resulting deck; account for
this coupling instead of optimizing two incompatible local objectives. Exact
rewards, value targets, model sizes, update schedules and interfaces require later
specification and evaluation; this amendment does not silently set them.

This approval does not expand the current starter-card combat content, require a
full-game engine before the first combat learner, or authorize implementation,
training, dependencies, original access, staging, commits, pushes or cleanup.
Keep the existing progression and obtain separately reviewed specifications/plans
and start approval for later implementation milestones. Original-runtime fidelity
remains unvalidated until its own evidence gates pass.

## Repository boundary and Gymnasium direction (2026-10-17)

The user approved tested Gymnasium compatibility as part of the next batched
combat environment/bindings milestone: thin single-environment and native-vector
interfaces over the same Rust engine and public observation encoder, not a Python
simulation or a Python loop over environments. Concrete interface/reset/action
contracts were specified, reviewed and separately authorized for implementation.
The supplied-entry combat slice now has native batching and tested single/vector
Gymnasium facades with masked actions. This is a bounded interface implementation,
not full-run support or a promise of compatibility with arbitrary learners, mask
conventions or vector adapters; framework-specific adapters are added only when a
selected learner needs them. Fresh candidate-revision extension verification passed
all 167 environment Python tests; the full suite retained the same two baseline
readiness-test failures as main (995 passed), not a fully green suite.

The repository is now named **spire-gym** on GitHub under **klottick**. The user
separately authorized public MIT-licensed repository creation and publication of
the relevant authored files. Local directories, existing package/crate names and
historical documents need not be renamed. ML training, experiments and learner
checkpoints are now explicitly outside this repository's scope, not a deferred
repository-split decision. Existing readiness smoke code and pinned dependencies
are retained unchanged; this boundary clarification is not dependency-removal or
code-migration authorization.
This administrative publication does not authorize new simulation implementation,
training, dependencies, original access, evidence promotion or cleanup.

## Delivery path

1. Reuse the implemented Rust combat core, native batching and Gymnasium facades;
   do not rebuild already merged functionality or mistake historical dirty copies
   for unmerged product work.
2. Specify the smallest useful gym PoC gameplay path and runnable acceptance checks.
   Identify actual missing content/transitions/interfaces, reusing accepted facts
   at their existing grade. Resolve only ambiguities that affect that milestone.
3. Obtain the narrow design/plan/start decisions and implement the selected path
   with focused correctness, hidden-information, lifecycle/replay and integration
   tests. Do not require a custom semantic-verification platform to proceed.
4. Demonstrate complete supported rollouts and batching from an external-style
   caller; measure representative release performance when scoped. Learner training,
   learned-strength evaluation and training checkpoints are not gym deliverables.
5. Expand the supported game content toward the long-term roadmap, preserving
   explicit unsupported behavior and fidelity limitations.

Perform original-game comparisons as a later fidelity-promotion milestone under
separately approved capture scope. They remain necessary for validated-fidelity and
original-game-transfer claims, not for starting Rust. Each later implementation or
training milestone still requires its own reviewed plan and start approval.

The existing gym slice is deliberately narrow: supplied-entry Ironclad A0 starter
cards against Cultist/Jaw Worm. It is not full-run support or unrestricted StS.
The next PoC must add useful gym capability beyond merely renaming this existing
slice; exact scope and acceptance remain in `spire-dxo.1`. Existing design,
plan/start and evidence approvals still apply to each actual implementation step.

Content progression remains combat → Act 1 Ironclad → full-game/Heart Ironclad →
A0–A20 → the same progression for Silent → Defect/Watcher → Daily modifiers under
their original mode rules. Content support is not evidence of agent competence.

## Current implementation status

This table describes the implemented source tree and its recorded verification,
not a claim that the planned interfaces or full-game content already exist.

| Area | Status |
|---|---|
| Rust/Python packaging and device readiness | Implemented smoke/readiness tooling; not a trained agent or simulator-performance result |
| Narrow Python combat reference | Implemented and independently reviewed: supplied ready state, base/upgraded Strike/Defend/Bash, Burning Blood, one Cultist/Jaw Worm, necessary powers/piles/turns, owned project RNG and schema-2 replay |
| Python verification | Reviewed reference baseline: 735 tests, including 52 whole-combat acceptance cases. Final minor-fix verification: 830 full-suite tests and all 95 differential tests pass, including above-CPython-threshold handles and isolated reveal execution. Native provenance correction and adapter fix are independently reviewed; historical failed exchanges remain retained. The unchanged default Python product engine still has its enormous-handle reveal conversion limit. Not original-runtime correctness evidence |
| Original combat validation | Incomplete; the reference remains static-evidence-grounded, **not original-runtime validated** |
| Narrow Rust combat core | User plan/start approval executed; supplied-entry combat, project RNG, public actions/views, snapshots, native acceptance and test-only differential harness implemented. Historical core-handoff controller workspace/release runs each passed 124 native tests; complete differential checkpoints agreed, including both formerly failing fault cases. Task reviews, whole-slice review, final correction review and controller verification were complete for that core-only handoff, not the later batching/Gymnasium integration. **Source-grounded, not original-runtime validated**; no optimization/throughput claim |
| Native environment pool/batched interface | Implemented for bounded supplied-entry Ironclad A0 combat with public observations, replay, native workers and single/vector Gymnasium facades. Fresh verification passed 169 offline Rust workspace tests and all 167 environment Python tests with a newly built candidate-revision extension. Full Python:995 passed and the same two baseline readiness-test failures as main, not fully green. Historical CUDA allocation OOMs are separate; neither CUDA readiness nor full-run support is claimed |
| Learning/training/evaluation system | External; not an unfinished repository deliverable. No learned playing-strength claim |
| Linux capture readiness | Approved specification retained; helper/feasibility work deferred in favor of source-first Rust. A concrete capture consumer remains required if resumed |
| Windows real capture adapter/platform gates | Incomplete/fail-closed; Linux readiness does not satisfy them |

## Validation scope and stopping rules

The [Linux readiness specification](superpowers/specs/2026-10-16-linux-validation-readiness-design.md)
is supporting work for the next actual StS comparison, not a standalone product.
Before planning helper implementation, identify the concrete next validation case,
intended invocation boundary, observations, unknown runtime/adapter facts and
required approvals. If there is no usable capture path yet, propose a bounded
feasibility investigation rather than building unused generic infrastructure.

Keep necessary identity/privacy checks, timeouts, independent child/descriptor
cleanup and truthful failure retention. Stop readiness work once its required
controls and original-validation handoff are complete, if resumed; do not add
speculative helper layers or block the now-approved source-first Rust path on them.
Do not create new promotion gates without evidence or user-reviewed scope changes,
and do not waive existing gates to claim completion sooner.

Original validation still needs complete supported combats at no fewer than three
original seeds per enemy, the required interactions and both terminal HP
observations, plus applicable adapter/export/load/visibility/non-interference and
Windows gates. Exact conditional AI laws remain required despite RNG-001. Python
self-tests and later Python/Rust agreement cannot replace original observations.

No playable client, generic card-game engine, generic capture framework, unrelated
infrastructure, private game redistribution or installation is implied by scope
approval. Preserve private raw evidence outside Git and historical failures as
recorded. Approval of a written spec permits planning, not execution of a plan
that has not yet been reviewed; no automatic staging, commit, push or cleanup.
