# Spire Lab: ML-first simulator design

> **Approved learning-direction amendment (2026-10-17):** Start with separate
> combat and run-level policies/models, keeping interfaces modular. One authoritative
> Rust simulator serves both: run-level training delegates fights to the combat
> policy executing the same detailed engine, not a second approximate simulation.
> Python is for ML and offline reference validation, not production gameplay rules.
> Train combat first, then strategic decisions against a frozen combat checkpoint
> as run content is supported; later alternate improvements and evaluate combined
> held-out runs toward Heart success. Shared representations/unified networks are
> later experiments. See the [approved direction](../../scope.md#approved-two-level-learning-direction-2026-10-17)
> for resource-aware objectives, scope and remaining design gates. The combat-only
> learner defaults below are bootstrap choices, not a mandated full-game policy or
> reward. The dated body is preserved; this amendment grants no new implementation,
> training, dependency, Git-write or original-access authorization.

> **Approved sequencing amendment (2026-10-16):** The user approved
> [source-first Rust development](../../scope.md#approved-source-first-sequencing-amendment-2026-10-16).
> Original captures are a later fidelity-promotion gate, not a prerequisite for
> Rust implementation. Conflicting sequencing below is superseded; mechanics,
> privacy and original-evidence requirements for fidelity claims remain. This is
> not approval of an unwritten Rust design/plan or permission to begin coding.

> **Approved RNG policy amendment (2026-10-16):**
> [RNG-001](../../intentional-deviations.md#rng-001--decorrelated-reproducible-project-randomness)
> supersedes exact original PRNG/seed/stream-sharing requirements below. Use
> reproducible project randomness decorrelated across unrelated mechanics while
> preserving intended probabilities and gameplay dependencies. The replacement
> [algorithm/API design](2026-10-16-project-randomness-design.md) was separately
> approved on 2026-10-16 and its standalone Python foundation implemented under
> the separately approved plan. The subsequent narrow Python combat integration
> is implemented/reviewed; original-runtime validation and Rust parity remain
> incomplete. Other correctness requirements remain.

> **Scope reaffirmed (2026-10-16):** The end goal is an AI that plays StS and
> improves, enabled by an optimized, game-correct **headless Rust simulator** and
> Python learning. The Python reference and validation are supporting work, not
> separate production/platform goals; no playable client or rendering is planned.
> See [current repository scope](../../scope.md) for present status, the bounded
> validation-to-Rust-to-learning path and stopping rules. Original dated contracts
> below remain historical unless explicitly superseded; this note grants no new
> execution, waives no evidence gate and reports no performance/learning result.

Date: 2026-10-11

Status: Approved by the user on 2026-10-11 for implementation planning;
implementation still requires plan review and an execution choice.

## 1. Purpose and priorities

Build a Rust Slay the Spire 1 simulator and Python learning system in one
repository. The ultimate agent objective is **Heart victory**, eventually across
Ascension 0–20, all four characters, and supported Daily modifiers. Report Act 3
completion separately from Heart victory. Daily modes follow the original mode's
rules; do not invent access to Act 4 where the game does not permit it.

Primary optimization target: held-out playing strength gained per wall-clock
hour, subject to game-correct simulation and player-visible information.
Simulation throughput and sample efficiency are diagnostic metrics, not
substitutes for this objective. Learning Rust is a secondary benefit; Rust
simulation and Python training are requirements.

This document specifies shared architectural contracts, the phase 1 boundary,
and the first executable slice, 1a. Later phases receive their own detailed plans
using measurements from completed stages. Approval of this document permits
writing the 1a implementation plan, not implementing it.

### Operating environment

- Primary development/training target: Linux server, Ryzen 9 9950X (16 cores /
  32 threads), approximately 59.5 GiB usable RAM, RTX 5090 with about 32 GB VRAM.
- Windows: user's original game installation, capture and controlled testing.
- Resource observations are not exclusive allocations. Configure worker/memory
  limits and leave capacity for other workloads.
- Training does not depend on an active Windows connection. Transfer captured
  traces to Linux for offline validation.
- Free Colab is an optional portability target, not an initial dependency.

## 2. Architecture and repository boundaries

```text
crates/spire-core/          combat rules, state, effects, RNG, snapshots
crates/spire-env/           scenarios, public encoding, actions, native pool
crates/spire-python/        thin extension bindings
crates/spire-cli/           replay, comparison, benchmark tools
python/spire_lab/           model, rollout/PPO, evaluation, experiment commands
validation/windows-mod/    project-owned game mod and build/install guidance
validation/fixtures/       reviewed, redistributable game traces
configs/                   experiment configurations and scenario manifests
docs/                      design, content/test matrices, benchmark reports
artifacts/                 ignored checkpoints, raw logs, generated outputs
```

`spire-env` depends on `spire-core`. Bindings and CLI consume those crates.
The core has no Python, tensor-library, renderer, or training dependency.
Immutable rule definitions can be shared; mutable environments cannot share
cards, enemy state, RNG, or pending effects. No mutable game globals.

Use PyO3 for bindings and maturin for the mixed package, with the extension
inside `spire_lab._native`. Use PyTorch for models/training and NumPy-compatible
numeric interchange. Keep PPO small and auditable, adapting established
algorithm mechanics and preserving attribution rather than building a new RL
framework. An optional conventional single-environment adapter is for tests and
compatibility; the training path uses the native pool directly.

Pin the Rust toolchain and resolved dependencies during implementation setup.
Commit Rust/Python/build lockfiles. Verify a compatible PyTorch/CUDA installation
with a real forward/backward/optimizer computation and synchronization on the
5090, not just GPU enumeration. No dependency installation is part of this design
step. Windows mod compilation targets the original game's JVM compatibility;
game and loader dependencies are external, not checked into this repository.

## 3. Core state and effect resolution

### Ownership model

`CombatState` owns:

- Ruleset/content identity, character, Ascension, turn and terminal status.
- Player HP, block, energy, powers, turn counters, relics/counters, potion slots.
- Card instances and ordered hand/draw/discard/exhaust piles.
- Enemy instances, intents, powers, AI state and move history.
- Named RNG streams and initialization/consumption state.
- Ordered effects/card-play work, pending decisions and suspended continuations.
- Public knowledge needed by observation encoding, including revealed top cards
  and observed enemy history, kept distinct from privileged internal state.

Use stable runtime card/enemy IDs rather than references across state mutation.
Do not reuse an ID within a combat. Enemy-slot indices and card-pile indices are
locations, not identities. Card kinds are immutable definitions; instances own
upgrades, temporary costs, and other mutable properties.

Start with normal owned collections and `Clone` snapshots. Avoid speculative
custom allocators, ECS frameworks, per-card inheritance, and shared ownership
inside a combat. Optimize allocations/state layout only from profiles.

### Resolution contract

Player actions and automatic effects are explicit enums with data. Effects
identify their source/target and carry either a precomputed quantity or an
explicit calculation instruction, matching the original calculation timing.
Do not universally recompute damage at execution or universally cache it at
card selection; both can change interactions.

Use an ordered scheduler with explicit phases for card use, queued effects,
turn ending, enemy actions, end-of-round work, and victory/death cleanup. It must
support front/back insertion and pausing for choices without queued closures.
Suspended work is serializable data. Implement only phases/hooks required by
current supported content; do not build a generic plugin engine.

An external step validates a decision/action before mutation, then runs until
the next original player choice, a fully resolved terminal outcome, or an error.
Intermediate discard/exhaust/upgrade prompts remain decisions. Multi-card picks
are staged and committed according to the original UI semantics: no effects
between provisional picks, no added hidden information, and no invented confirm
action where the game commits automatically.

Use game-equivalent integer arithmetic and rounding; explicitly reproduce
floating-point behavior where the game uses it. RNG overflow is deliberate
wrapping arithmetic, not dependent on debug/release compiler behavior.

Invalid actions leave state unchanged. Unsupported mechanics, broken invariants,
or an automatic-work budget violation are typed errors, never no-ops or fabricated
losses. Initial automatic-work guard: 100,000 resolved effects per external step;
hitting it fails the experiment and preserves diagnostic information.

Victory/death ordering and relic cleanup follow the game. Record both ending HP
before combat-end healing and HP after cleanup. Enemy escape is recorded
separately even when the game's combat outcome counts as player victory.

## 4. RNG, snapshots, and determinism

Implement the game-relevant Java/libGDX RNG algorithms, bounded sampling, and
shuffle procedures. Never substitute Rust's default RNG for game transitions.
Maintain the original named streams and their sharing/consumption rules for AI,
shuffling, card randomness, monster HP, potions, and other supported game uses.
Do not invent independent streams where the game shares one. Snapshot exact
stream states/counters.

Scenario sampling and model sampling use separate randomness. Given a supported
initial state and actions, worker count, scheduling, tracing, and inference
device cannot change game transitions. Full training bitwise determinism is a
separate optional setting; GPU learning nondeterminism must not be mislabeled as
a simulator replay failure.

Snapshots include all mutable state and continuations plus rules/content/schema
fingerprints. Restore verifies identity and invariants, rejecting incompatible
data. Snapshotting is a debugging/search operation, not a normal rollout step.
Initially import Windows snapshots only at supported clean combat-entry decision
boundaries. Arbitrary mid-animation original-game state import is not supported.
Native Rust snapshots can preserve choices once those mechanics are implemented.

## 5. Information and action contract

Actors and the initial critic receive only player-visible information and
legitimately remembered history. No seed, RNG state, unseen draw order, unpublished
enemy rolls, internal UUIDs, scenario IDs, or environment IDs as model features.
Validation metadata is a separate interface, not an extra observation dictionary
that the model can accidentally consume.

Public draw-pile composition is unordered. Preserve any order actually revealed
by effects such as Headbutt/Warcry in a separate known-top representation. Do not
expose hidden instance properties merely because the simulator stores them.
Use deterministic public ordering for sets/candidates; ordering must not depend
on hidden draw order or memory allocation. A stochastic search teacher added in
a later experiment must use uncertainty-consistent information, not future RNG.

### Candidate actions

Enumerate legal actions from the current decision. An action index selects one
candidate in that decision, not a permanent global vocabulary entry. A decision
token prevents executing an index from an obsolete observation.

Kinds cover card play, potion use/discard when relevant, card selection, original
choice confirmation/cancellation when legal, and end turn. Source/target fields
refer to public observation slots; core execution resolves stable internal IDs.
Candidate descriptors include action kind and currently visible numeric card,
target, potion, and selection-context features. No speculative future-outcome
scores from hidden state. Padded candidates are masked in sampling, log-probability,
entropy, and loss. Terminal states do not require policy inference.

Multi-card choices are sequential rather than a powerset of candidates. Return
the remaining required/allowed count and selected public cards. With the initial
undiscounted objective, selection bookkeeping does not discount combat outcomes.

## 6. Numeric observation protocol

The protocol includes stable feature names, kind vocabularies, units, normalization,
and a schema fingerprint. The binding exposes `observation_spec()` so Python
constructs tensor shapes from the protocol, not duplicated magic constants.
Checkpoint loading rejects incompatible schemas by default; migration requires
an explicit kind/feature mapping and its own evaluation.

Initial logical arrays (batch size `B`):

| Array | Type / shape | Meaning |
|---|---|---|
| global | float32 `[B,16]` | Player resources and turn/pile counts |
| hand_kind | int32 `[B,10]` | Public card-kind vocabulary indices |
| hand_features | float32 `[B,10,16]` | Current visible card attributes |
| hand_mask | bool `[B,10]` | Occupied hand slots |
| enemy_kind / intent | int32 `[B,5]` each | Published enemy and intent kinds |
| enemy_features | float32 `[B,5,16]` | Published combat quantities/history |
| enemy_history | int32 `[B,5,2]` | Previous two observed intent kinds |
| enemy_mask | bool `[B,5]` | Occupied enemy slots |
| pile_counts | float32 `[B,3,V,2]` | Known draw/discard/exhaust counts by kind/upgrade |
| known_top | float32 `[B,K,16]` plus bool mask/int32 kinds | Publicly revealed top sequence |
| choice_cards | float32 `[B,S,16]` plus bool masks/int32 kinds | Public card-selection options and staged picks |
| player/enemy powers | float32 `[B,P]` / `[B,5,P]` | Manifest-ordered public power quantities |
| relic features | float32 `[B,R,2]` | Presence and public counter |
| potion kind / mask | int32 / bool `[B,5]` | Potion slots; phase 1a slots are empty |
| decision context | numeric `[B,8]` plus kind | Choice counts/options/staged selection summary |
| action descriptors | numeric `[B,C,8]` plus kind/source/target | Legal candidate features/references |
| action_mask | bool `[B,C]` | Valid candidate positions |

`V`, `P`, and `R` come from the versioned supported-content registry. `K`, `S`, and
`C` are padded batch capacities, not game limits. Five potion slots reserve the
original maximum slot layout; only validated capacities/content are accepted.
Categorical/reference arrays use int32, quantities use float32, and masks use
bool. Enemy history has its own observation-validity mask. Choice cards have
occupied, selectable, and staged-selection masks; action source references carry
their zone (hand, choice cards, or potion slots). Thus non-hand card prompts do
not reuse hand indices or expose the underlying hidden pile order. Choice options
are ordered from public attributes only. Card features reflect only attributes
visible or legitimately known in that zone, with an attribute-visibility mask
where needed; an unknown current cost is not exposed as a private value.

Global fields, in order: current HP / max HP, max HP / 80, block / 100, energy / 10,
energy per turn / 10, base draw / 10, turn / 50, hand size / 10, draw/discard/exhaust
sizes / 64, total/attack/skill cards played this turn / 20, occupied potion slots
/ 5, Ascension / 20. Powers are separate rather than hidden in those fields.

Hand fields: current cost / 10, printed cost / 10, upgrade count, printed damage
/ 50, printed block / 50, attack/skill/power flags, exhaust/ethereal/current-retain/
innate/free-play/status/curse flags, and currently playable flag. Sentinel costs
remain explicitly interpretable; no normalization-based clipping.

Enemy fields: current HP / max HP, max HP / 200, block / 100, published damage per
hit / 50, hit count / 10, published intent block / 100, strength / 20, vulnerable
/ 10, weak / 10, artifact / 5, ritual / 10, observed turns / 50, and previous
observed intent damage / 50, hits / 10, actual attack damage / 50, block gain / 100.
Unobserved history is masked/zero, not read from private AI history. Additional
supported public powers use registry fields.

Decision context contains minimum/maximum/remaining picks, staged pick count,
whether skip/confirm/cancel are legal, and original-choice completion status.
Action numeric fields contain visible cost/damage/block, upgrade indicator,
target requirement, remaining picks, free-play indicator, and potion amount;
kind and public source/target indices are separate categorical/reference arrays.

Padding is zeroed and masked, including every reset. Means/maxima in encoders
exclude padding. Scaling constants do not saturate real values. For powers,
counts, and relic counters, units/scales are declared in the registry.

Phase 1a preallocates `C=32`, `K=0`, `S=0`; phase 1b/1c start with `C=128`,
`K=10`, `S=64`.
If valid content exceeds capacity, safely grow at a batch boundary and publish
new shapes, or return an explicit capacity error if the configured memory budget
prevents growth. Never truncate data, drop legal actions, or silently resample.
Capacity changes are recorded; observation feature semantics remain versioned.

## 7. Native pool and Python memory boundary

Proposed interface names:

- `reset_batch(indices, scenarios)` initializes selected slots and returns data
  for those slots in the requested order.
- `step_batch(action_indices, decision_tokens)` advances every active slot once.
- `observation_spec()` returns schema, vocabularies, shapes, and normalization.
- Single-environment reset/step/snapshot/replay use the same encoder/rules engine.

Validate all batch input actions/tokens before advancing any slot. A runtime
engine error may leave a partially advanced batch; mark the pool faulted and
stop the run rather than pretending the batch completed or rolling back silently.

Start with one Python process and a configurable persistent native CPU pool,
synchronous stepping, and one batched model call. Native execution detaches from
Python; workers access Rust data only. Each slot is advanced exclusively by one
worker per call. Output rows remain slot-ordered regardless of scheduling.

Every transition returns reward, `terminated`, `truncated`, decision/reset
metadata, and next observations. Auto-reset happens after retaining final
observations; return final rows with their slot indices separately, so the
learner cannot bootstrap from reset state. Reset episodic counters/history and
any recurrent state at the correct boundary. Simulator errors are not truncations.

Native buffers are reusable. **Initial binding outputs are safely owned numeric
arrays copied per batch**, not unguarded views into subsequently mutated Rust
memory. Python retains rollout copies in preallocated storage. This bounded batch
copy is preferable to lifetime bugs; measure it before adding zero-copy complexity.
An owned-copy reference path remains available permanently.

Only add leased/reusable external views if profiling justifies them. Such a path
must prevent native reuse while Python readers or GPU transfers retain the lease;
readonly flags alone do not prevent races with native writes. Begin with completed
transfers/synchronous ownership, then measure pinned-memory/overlap improvements.
No JSON, card-level Python objects, state clones, or debug formatting in rollouts.

## 8. Learning baseline and defaults

Use synchronous masked PPO with actor and critic sharing a compact public-state
encoder. Card/enemy kind embeddings start at width 16; shared entity encoders use
two 64-unit layers; masked mean/max pooling forms state context. A two-layer
128-unit state encoder feeds a scalar value head and a candidate scorer using
state plus relevant source-card/target-enemy embeddings, action kind, and numeric
descriptors. Evaluate state context once, not once per candidate. Known-top
features preserve public ordering rather than treating them as a set.

No initial recurrence, attention, dropout, search, or imitation dependency. Supply
observable history explicitly. Recurrence/attention and search/imitation are
later ablations measured by held-out learning per hour. No claim that this model
or PPO is the universally best eventual learner.

Baseline reward is +1 for resolved victory, -1 for death, zero otherwise; gamma
is 1 for finite-combat returns. Track HP before/after healing, potion use, and
enemy escape separately. Resource-aware rewards or potential-based shaping get
distinct experiment identities and cannot change the canonical win metric.
Full-run learning must replace/reassess combat-only targets rather than treating
combat wins as the ultimate objective.

Initial pilot defaults for written-spec review, not measured optimums:

| Setting | Initial value |
|---|---|
| Environment slots / native workers | 256 / 8 |
| Rollout decisions per slot | 128 |
| PPO epochs / minibatch size | 4 / 2,048 |
| Learning rate / clipping | 3e-4 / 0.2 |
| GAE lambda / value coefficient | 0.95 / 0.5 |
| Entropy coefficient / gradient norm cap | 0.01 / 0.5 |
| Training seeds for promoted configuration | 7, 17, 29 |
| Administrative combat limits | 100 turns or 5,000 policy decisions |
| Initial per-seed training budget | 2 hours or 20 million decisions |

Stop at whichever training-budget limit is reached first. Budgets are checked
between complete updates; report any one-update overshoot.
These are configurable starting points, not completion-time or win-rate promises.
Bounded pilots can replace them with recorded values using model-selection data.

For advantage calculation, genuine termination eliminates value bootstrap.
Administrative truncation bootstraps from the final pre-reset observation but
stops trajectory recursion across reset. Rollout boundaries bootstrap normally.
Store the exact observations, action masks, actions, and old log probabilities
used for sampling. Test these calculations independently of the game.

Scenario generation admits only validated content closure. Initial mixing is
uniform over declared configurations/tiers, not whatever is fastest to simulate.
Later curricula retain earlier cases and publish their mixture weights. A model
that wins only trivial fights is not evidence of broad phase 1 competence.

## 9. Evaluation and efficient experimentation

Use three separate scenario namespaces/manifests: training, model selection, and
final evaluation. Check manifests for configuration/seed overlap. No final-set
hyperparameter selection. If final results motivate model/curriculum changes,
retire that corpus to diagnostic use and freeze a fresh final corpus before the
next promoted comparison. Fit any learned normalization on training data only.

### Phase 1a scenarios

- Canonical deck: five Strikes, four Defends, one Bash; max HP 80, three energy,
  five cards drawn per ordinary turn, Burning Blood, no potions.
- Train across both enemies, HP 20/40/60/80, and starter-only decks with either
  no upgrades or a single upgraded Strike, Defend, or Bash.
- Model selection uses HP 50 and one-upgrade starter configurations.
- Final evaluation uses HP 30/50/70 and two-upgrade starter configurations,
  distinct from the single-upgrade model-selection configurations.
- Generate enemy HP/moves/shuffles with validated original-game procedures.
  Synthetic scenario constructors are not claims of seed-accurate full runs.
- Use 200 model-selection scenarios and 1,000 final scenarios, with 500 final
  cases per enemy and a deterministic balanced allocation across configurations.

### Full phase 1 evaluation

Freeze 1,000 scenarios: 600 normal, 200 elite, 200 boss cases, balanced within
tiers across supported encounters and realistic deck/resource configurations.
Build and freeze the manifest before tuning on its results. Decks use only the
approved card manifest; include common supported synergies, not arbitrary decks
that ignore Act 1 acquisition constraints. Separate captured-natural and controlled
synthetic evaluation reports instead of claiming identical distributions.

Uniform-random and fixed public-information heuristic baselines use identical
initial cases and reproducible action randomness. The 1a heuristic prioritizes
immediate lethal attacks, necessary block against displayed incoming damage,
useful Bash, then other damage and end turn. Its precise tie-breaking/configuration
is frozen and versioned before comparison. Extend the heuristic explicitly for
new choice mechanics; it never uses simulator RNG or unseen draws.

Evaluate the three independently trained models. Select checkpoints and greedy
versus stochastic evaluation mode using model-selection scenarios, then freeze
that rule. Report every training seed, the aggregate, and tier-specific win rates.
Use paired 95% bootstrap intervals for model-minus-baseline win rates, resampling
scenario configurations/cases and reporting training-seed variation rather than
treating repeated models on the same cases as independent observations.

Learning gate: positive lower 95% bound for aggregate improvement over random,
with improvement in at least two of the three training runs. Matching/beating
the heuristic is a separate performance target, not assumed guaranteed.
For full phase 1, report tier improvements separately; aggregate success cannot
be described as mastery of a tier where the learner fails.

Count every timeout in the denominator of attempted-case success rates, report
it separately, and never relabel it as a game death. Any internal error invalidates
the evaluation run rather than being dropped. Repeat affected runs after fixes.
HP statistics include both all-attempt and win-conditioned summaries so survival
conditioning does not conceal poor play. Report potion use and escape outcomes.

Plot score versus training decisions and wall-clock time, including evaluation
and update overhead; report setup and pilot budgets separately. Keep raw counters
so throughput comparisons cannot substitute internal effects for policy decisions.

Benchmark release builds after warm-up: simulation+encoding separately, data
movement, policy inference, policy updates, and full training decisions/second.
Sweep worker counts 1/4/8/12 and slot counts 64/256/512; measure CPU and GPU
inference for the small model. Use three timed repetitions per configuration.
Select two practical configurations for bounded 10-minute learning pilots,
then promote using validation learning progress rather than raw throughput alone.
Start with synchronous execution; asynchronous execution, GPU simulation, mixed
precision, compilation, and distributed training require a measured bottleneck
and correctness comparison before adoption.

## 10. Windows mod, traces, and validation gates

The project-owned mod observes the original game's stable decision boundaries.
Passive capture does not consume RNG, replace mechanics, or change effect order.
A separate controlled-test mode can set up supported scenarios; label every
injected fixture. Mod/loader dependencies and the game JAR remain user-provided.
Record game build identification and binary hash locally, export only metadata.

Trace format: versioned UTF-8 JSON-lines, optional compression outside training.
First record is a header with schema/content versions, game/mod/loader versions,
capture mode, scenario provenance and case ID. A complete clean initial-state
record follows, then semantic-action/expected-decision-state pairs, and finally
a terminal/truncation record. Encode 64-bit RNG words/seeds losslessly as hex.

Initial state includes privileged ordered piles, per-instance attributes, relevant
enemy AI state and exact RNG state. Also capture public knowledge separately;
never infer policy-visible history from all privileged fields. Normalize identities
to trace-local card/enemy symbols with a replay mapping. Do not compare game heap
addresses, UUID implementations, timing, animation state, or renderer state.

Canonical comparison checks HP, block, energy, card identities/properties/pile
order, powers, relic/potion state, intents, turn/pending choice, terminal result,
and supported RNG streams. Reject fixtures requiring unsupported mechanics;
do not silently delete mismatching fields. CLI reports the first action/field
divergence and relevant before/after values. Add effect-level diagnostics only
when needed, without altering RNG consumption.

Layered requirements:

1. RNG primitive/shuffle fixtures produced by the reference runtime.
2. Focused rule/interaction tests, including calculation timing and rounding.
3. Invariants: valid card ownership/pile membership, stable IDs, valid energy/HP,
   legal pending-choice/effect state, and explicit exceptions required by rules.
4. Randomized supported scenarios, native snapshot/replay, worker-count and
   single-versus-batch identity, correct masks, reset semantics, and memory lifetime.
5. Hidden-information tests: identical public histories with different hidden RNG
   or unknown draw ordering produce identical observations and legal candidates.
   Permuting public entity sets with corresponding action references preserves
   semantic decisions; genuinely revealed order is retained.
6. Real-game differential corpus covering every implemented mechanic/encounter
   family and required interactions; no known mismatches in supported behavior.

Phase 1a includes full Cultist/Jaw Worm combat traces across at least three RNG
seeds per enemy, base/upgraded starter-card checks, block reset, Bash vulnerability,
enemy buffs, shuffle boundaries, victory healing and death. Controlled traces
exercise edge cases natural play misses. RNG fixtures and required game captures
are blockers for validated promotion, not replaced by comparing two copies of
the same simulator. Tiny optimizer smoke tests may run beforehand but cannot
be presented as validated learning results.

Every experiment/checkpoint records code revision/dirty status, rules/content and
observation fingerprints, scenario split hash, reward/learner version, complete
config, seeds, dependency/driver/device identities, decisions and elapsed time.
Rules/schema changes invalidate compatibility by default. Resume requires matching
versions and optimizer/sampler state; deliberate migrations start a separately
labeled experiment. Do not quietly continue a checkpoint under changed mechanics.

## 11. Approved phase 1 manifest

Normal Ironclad A0 combat only; reject unsupported Ascensions, characters, or
modifiers. Generated effects/cards must remain within the validated closure.

### 1a: first executable training slice

Strike, Defend, Bash (base/upgraded); Cultist, Jaw Worm; Burning Blood. Include
enemy Strength/Ritual, Vulnerable, block/energy/turn reset, exact draw/shuffle,
victory/death and healing. No acquisition screens, potions, other cards, relics,
or arbitrary run state. The first implementation plan targets this slice only.

### 1b: normal encounters and expanded mechanics

Normal templates: Cultist, Jaw Worm, Two Louse, Small Slimes, Blue Slaver,
Gremlin Gang, Looter, Large Slime, Lots of Slimes, Exordium Thugs,
Exordium Wildlife, Red Slaver, Three Louse, Two Fungi Beasts. Match supported
game template variants and their construction/RNG rules, not merely enemy names.

All 20 Ironclad commons: Anger, Armaments, Body Slam, Clash, Cleave, Clothesline,
Flex, Havoc, Headbutt, Heavy Blade, Iron Wave, Perfected Strike, Pommel Strike,
Shrug It Off, Sword Boomerang, Thunderclap, True Grit, Twin Strike, Warcry,
Wild Strike. Selected uncommons: Battle Trance, Bloodletting, Burning Pact,
Disarm, Entrench, Feel No Pain, Flame Barrier, Inflame, Metallicize, Seeing Red.
Support base/upgraded forms and generated Wound/Slimed/status behavior as needed.

Relics: Burning Blood, Vajra, Oddly Smooth Stone, Anchor, Bag of Preparation,
Lantern. Potions: Fire, Block, Strength, Dexterity, Energy, Weak, Explosive.
Support relevant powers, selection prompts, exhaust/autoplay, temporary buffs,
AOE/random targeting, hand caps and generated cards. No random-card potions.

### 1c: elites, bosses and phase 1 promotion

Gremlin Nob, Lagavulin, Three Sentries; Slime Boss, Guardian, Hexaghost. Add their
required mechanics, splitting/new identities, Dazed/Burn/upgraded Burn and
generated-effect closure. Complete differential fixtures, tiered curriculum,
1,000-case evaluation and three-run learning/throughput reports.

Do not claim unrestricted Act 1 support at phase 1 exit: rare cards, colorless
cards, curses, other relics/potions and acquisition paths can appear in Act 1.
Phase 2 must expand those before its unrestricted-run completion claim. Training
an explicit subset is acceptable; silently modifying the game's reward pools
and presenting that as full compatibility is not.

## 12. Long-term roadmap and exit definitions

| Phase | Capability / reporting boundary |
|---|---|
| 1 | Validated supported Ironclad A0 combat; 1a first, then 1b/1c |
| 2 | Ironclad A0 Act 1 navigation, events, shops, rewards, deckbuilding; expand content for unrestricted supported original runs |
| 3 | Ironclad A0 full runs, keys, Act 4 and Heart; report Act 3 and Heart separately |
| 4 | Ironclad A0–A20 full rules and learning/evaluation; stratify by Ascension |
| 5 | Repeat combat → Act 1 → full/Heart → A0–A20 progression for Silent |
| 6 | Same progression for Defect and Watcher; plan their order after Silent measurements |
| 7 | Original Daily modifier combinations across applicable supported content; preserve original mode restrictions |

Each phase adds a validated manifest, observation/checkpoint migration policy,
scenario coverage, and learning benchmark before promotion. Simulator content
coverage does not demonstrate agent competence. Strong Heart performance is the
ultimate benchmark, not a claim attached to an implemented Heart enemy.

Later run state coordinates battle entry/exit and run-level decisions without
reimplementing combat. Reuse learned encoders/actor initialization where measured
useful; recalibrate/retrain value targets for resource preservation and full-run
success. Character-specific heads or policies are allowed; a single universal
model is an experiment, not a required architecture. Add versioned ruleset/modifier
identity to configurations, but do not implement generic modifier hooks early.

## 13. Acceptance and handoff

Phase 1a delivery requires supported mechanics and capture/replay fixtures,
safe numeric batching/bindings, optimizer smoke tests, reproducible benchmarks,
and the specified held-out learning comparison. A failed learning gate is a
research finding requiring iteration, not permission to call agent strength
demonstrated. Full phase 1 additionally requires its complete approved manifest
and tiered completion corpus.

No throughput or win-rate result has been measured. Numerical pilot defaults,
array layouts, and evaluation procedures above were proposed in this written
design and approved with it; they remain starting points, not measured optimums.

After written-spec approval, produce a test-driven 1a implementation plan with
small tasks, corresponding Windows captures, toolchain/readiness checks,
performance measurement, and training/evaluation. Ask the user to select its
execution method. Do not scaffold/install/implement before those gates.

## 14. Reference and redistribution policy

Use existing simulators as mechanics references, not ground-truth oracles:

- https://github.com/gamerpuppy/sts_lightspeed
- https://github.com/SnyderConsulting/sts_lightspeed
- https://github.com/ben-w-smith/sts_lightspeed

Native-batching and integration references:

- https://envpool.readthedocs.io/en/latest/
- https://puffer.ai/docs.html
- https://pyo3.rs/
- https://www.maturin.rs/project_layout.html
- https://pytorch.org/docs/stable/notes/cuda.html

Reference-project tests were inspected, not run. Published unrelated throughput
figures are not performance estimates for this project. EnvPool/PufferLib are
design references, not selected dependencies.

Retain applicable licenses/attribution for any adapted code. Keep game binaries,
assets and decompiled game source outside the repo. Include only project-owned
mod/tool source and reviewed redistributable fixtures. This phase creates a local
repository; public hosting/publication and the new project's distribution license
require a separate explicit decision before release.
