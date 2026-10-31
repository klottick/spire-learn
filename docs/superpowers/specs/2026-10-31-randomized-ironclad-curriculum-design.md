# Broad randomized Ironclad combat curriculum

Status: proposed design for user review. No implementation or runtime is released
by this document. The user selected mostly realistic decks, with some unusual
combinations, after considering broad randomization and card-focused lessons.

## Purpose and evidence

Teach a combat policy that can use varied Ironclad decks against varied enemies,
including combinations outside a small authored catalog. Keep both standalone
combat and combat inside a normal Run as targets.

The preceding standalone training catalog had 78 entries but only three deck
templates, seven distinct card IDs, 13 encounters and two starting-HP settings.
The collector sampled those fixed entries uniformly; it did not generate decks.
Increasing the number of training actions did not expand that distribution.
The overnight campaign saved 238,574 learned decisions, but intermediate
selection showed no consistent improvement and one regression. Its normal-Run
branch stopped on a native fault tracked as **spire-tsw**; it was not retried.
See [the partial results](../../neural-study.md).

The pinned Gym66 public contract supports 75 Red cards and 43 finite encounter
setups. This is the initial executable-content boundary, not the entire game.
Support must be checked against the actual qualified artifact before runtime.
Do not substitute a newer Gym revision or infer a rebuilt extension from source
acceptance. The existing native, source, content and schema identities remain
recorded until a separately qualified delivery changes them.

## Approaches and selected direction

1. Uniform arbitrary decks and enemies: broad coverage, but it can devote much
   of the budget to mismatched fights and combinations outside ordinary runs.
2. A sequential card-by-card syllabus: easier to attribute lessons, but slow to
   expose interactions and liable to neglect previously covered situations.
3. **Mostly realistic randomized decks, mixed with focused lessons and unusual
   combinations:** selected direction. Breadth is present from the beginning;
   focused lessons give difficult mechanics repeated exposure. No global gate
   requires mastering one card before another is introduced.

The initial mixture below is a builder proposal, not an empirically established
optimum. Public research on task sampling and learning progress motivates the
approach, but does not establish effectiveness in this game:
[Portelas et al.](https://proceedings.mlr.press/v100/portelas20a.html),
[Florensa et al.](https://proceedings.mlr.press/v80/florensa18a.html).

## Distribution and generation

Use three explicitly labeled scenario families:

| Family | Initial scenario share | Construction |
| --- | ---: | --- |
| Run-like randomized | 70% | Starter-derived decks with additions, removals, duplicates and upgrades, varied resources, and ordinary/elite/boss encounters |
| Focused lessons | 20% | Starter-derived single-card exercises, then pairs and small mechanic combinations; continued earlier-task practice |
| Unusual combinations | 10% | Broader supported-card mixtures, atypical sizes and resource combinations, bounded by the public construction contract |

These describe scenario allocation, not a promise about equal action or gradient
shares: longer fights produce more transitions. Report allocated scenarios,
actual reset exposure, actions and completed outcomes separately by family.

Run-like is a declared **training heuristic**, not a reimplementation of reward
screens, drop tables, card acquisition or other game rules. Begin with the
canonical Ironclad starter deck; vary removals and additions using small authored
stage profiles (early, middle, developed). Prefer common cards more often than
rare cards, preserve varied damage/block/utility composition, and occasionally
sample coherent combinations. Card-family labels guide sampling only; Gym still
executes every effect. Do not claim these recipes reproduce the true Run-state
distribution. Future public Run observations can inform that distribution under
a separately recorded recipe revision.

Initial run-like decks contain 8–35 cards; unusual decks contain 5–40. These are
learner sampling bounds, narrower than Gym's 1–128 supported range. They can be
revised through a versioned recipe rather than silently changed mid-experiment.
Ordinary upgrades use 0/1; additional Searing Blow upgrade levels are a later
explicit lesson, not required for initial base/upgraded coverage.

Pair early, middle and developed decks with coarse public encounter tiers and
HP ranges. Keep all tiers represented, including a small number of difficult
pairs; do not require an oracle that proves a fight winnable. Use recorded
training completion statistics to determine practice priorities, not simulator
lookahead, hidden state or a second approximate combat engine.

HP and max HP are public supplied resources, with HP no greater than max HP.
Initial max HP is bounded to 80–120 and current HP to 24–max HP. Relics and
potions come only from the qualified public support list. Start with the
character's supported starter relic plus a small varied set of owned items;
reject incompatible combinations. Active Neow's Lament is excluded from the
initial recipe; a supplied Lament must be spent. Acquisition effects are never
applied twice. No supplied curses/statuses, private snapshots, hidden draw order
or RNG cursor are added to bypass Gym's accepted constructor.

## Card coverage and lesson progression

Use all 75 accepted cards as the pool from the beginning. Single-card lessons
cycle across that pool; they are not 75 consecutive training projects.

Focused lessons mix single additions, pairs and small groups. Their initial
shares are 50%, 30% and 20% within the lesson family. Each four-shard coverage
cycle must include every supported card in base and ordinary-upgraded form and
every accepted encounter setup. Counts identify **generated** coverage; actual
exposure is reported separately and must not be inferred from generation.
Pair coverage is measured, not claimed exhaustive.

Within a loaded catalog, preserve ordinary broad sampling while reserving half
of focused selections for the least-exposed lesson categories. The other half
prefers categories with intermediate observed success (20–80%) once at least
32 completed training fights are available. Before that count, use coverage
balancing. Retain a nonzero floor for all lesson categories. This small stdlib
sampler avoids adding a learned teacher or new dependency.

The sampler uses training outcomes only. Selection/final outcomes never adjust
its task weights. Success statistics distinguish natural victory/death from
incomplete fights; simulator faults halt collection and never enter a loss
count. Store sufficient statistics and selection RNG state in checkpoints.
Retention means **fresh on-policy rollouts on earlier tasks**, not replaying
stale trajectories through PPO.

## Bounded catalogs and sampler identity

Gym accepts an immutable catalog of at most 256 entries and at most 1 MiB JSON
per environment. Its reset selects an existing entry; it does not accept a new
deck. Respect that interface.

A deterministic learner generator produces bounded catalog shards. A full shard
has 179 run-like, 51 lesson and 26 unusual entries, subject to the byte bound.
If full entries exceed 1 MiB, reduce the count deterministically while retaining
family quotas through an explicit accumulator; never truncate serialized JSON.
Use bounded generation attempts and fail preparation if the declared coverage
cycle cannot be built. Unsupported content does not trigger silent substitution.

Freeze the generator version, public support snapshot, sampling parameters,
split domain and generator seed in a content-addressed distribution manifest.
Every generated entry has a canonical identity and records recipe family,
deck stage, targeted mechanic/card, encounter tier and lineage. Those labels,
seeds, IDs and difficulty statistics are **sampler metadata, not actor inputs**.
Gym owns shuffling, startup, enemies and all simulation randomness.

Derive generator substreams by purpose and shard index independently of actor,
optimizer, environment and noncombat-controller randomness. Native episode seed
allocation retains the accepted monotonic domain rules. A resume reconstructs
the exact same next shard and sampler choice sequence without drawing from the
policy generator. Same inputs produce byte-identical manifests.

Rotate catalogs only when constructing a fresh standalone runtime at a normal
checkpointed segment boundary. Do not replace a live driver's catalog or reset
ongoing rows to make an inconvenient case disappear. As in the existing segment
contract, an unfinished task at closure is explicitly incomplete, with accepted
bootstrap/censoring accounting; it is not a natural terminal or a loss. Record
censored fractions by deck stage and encounter so rotation cannot hide poor
performance on long fights. An uncertain effect prevents the next segment.

## Checkpoints and integration

Keep the accepted model, feature schema, public action semantics, PPO settings
and combat reward unchanged in the first distribution comparison. The model's
card vocabulary already covers the accepted schema. This design changes data
generation and sampling, not the game engine or the policy architecture.

Add an optional, versioned scenario-stream config and manifest path. Existing
fixed-manifest configs and their strict resume behavior remain unchanged.
Proposed implementation seams are a stdlib generator/support manifest, pure
sampler/state contracts, collector sampler injection, and the existing runner's
checkpoint/orchestration boundary. Names are builder choices; no placeholder
API is an accepted implementation yet.

The current runner checks exact config and case-manifest hashes during resume.
Blindly swapping shards would fail this check, and weakening it globally would
lose provenance. Introduce an **explicit scenario-stream resume contract**:

- Stable identity binds recipe, support, mode, model/PPO/objective, rollout
  shape, budgets and RNG domains. Actual raw config and concrete shard hashes
  remain separately recorded; no stale dataclass hash or fabricated path.
- `run_metadata` carries the distribution identity, current and next shard,
  concrete manifest hash, coverage/success statistics, and sampler state. Keep
  the existing closed training-state schema and optimizer/RNG payload contract.
- A stream-specific validator verifies both predecessor and next shard from
  the same frozen recipe before allowing that declared transition. No other
  config, mode, schema, budget or arbitrary manifest change is accepted.
- Preserve accepted model/Adam state, behavior version, environment/decision/
  episode high-water marks and cumulative counters. Never warm-start weights
  at each shard and call it exact optimizer continuation.
- Old checkpoints can initialize a new experiment by the existing weight-only
  warm-start path, with fresh Adam/RNG/progress. They cannot claim an exact
  scenario-stream resume without the new state. The original B parents remain
  frozen references; do not automatically promote unevaluated latest weights.

Use a small pilot to compare fixed sampling, realistic randomization alone, and
the mixed curriculum under matched action and optimizer budgets. Report
throughput and exposure as well as score. This separates the value of breadth
from the additional sampler; a positive result is not assumed.

All three standalone pilot arms start from the **same original standalone B
model weights**, with fresh Adam state, behavior version zero, zero progress and
fresh RNG domains. The parent is
`artifacts/development/neural-combat/experiments/task9-v1-attempt-b/training/standalone-b/20261030T210525Z-c7d37fcacba143709d2e08ca9eb9c21c/checkpoint-final`;
its `model.pt` SHA-256 is
`fc44f89afd409abd68bf2397d28c11f49584ea2c017f090e6da0b2408c16ecd9`.
Validate all three parent files against the retained original parent inventory,
not the model hash alone. Use the same initial RNG quartet across arms for each
paired replicate, with distinct domains within that quartet. Divergent task
choices naturally consume randomness differently; this is paired initialization,
not a promise of identical trajectories. The eventual runtime proposal fixes the
replicate count and seed quartets before execution.

This is the author's proposed initialization choice: it isolates distribution
effects during continuation of an existing standalone policy. It is not a claim
of learning from scratch. The original standalone B and Run B checkpoints remain
immutable evaluation references; Run B is not a different starting point for one
pilot arm. Latest unevaluated overnight checkpoints are not silently promoted.
Any later Run fine-tuning parent is explicitly named in that separate proposal.

## Evaluation and the wider learning goal

Author fixed development panels before training: familiar recipes with new
seeds, unseen canonical decks, selected held-out card combinations, resource
perturbations and later encounter tiers. Split by composition/lineage as well
as seed. Do not claim that different seeds alone establish deck generalization.
For reserved card pairs, enforce exclusion during generation with bounded
rejection and explicit coverage checks; keep the reserved set small enough to
retain the declared generation support.

Parent checkpoints may have encountered combinations during earlier Run
training. Therefore held-out means excluded from **this experiment's** training
generator; historical pretraining exposure is unknown. Keep a fresh-initialized
control available when a claim specifically requires learning from scratch.
The existing final split remains untouched; any future generated final panel
gets its own preregistered recipe, domain and immutable identity.

Measure wins, completion, HP before healing and after cleanup, potion/resource
use, actions, and outcomes by deck family/card/encounter/tier. Compare matched
cases at fixed checkpoints and preserve regressions. Report worst-stratum
results as well as averages, with denominators and small-sample limitations.
Health/resource metrics initially remain evaluation diagnostics: changing
reward shaping simultaneously would confound the distribution comparison.

Combat-local learning is a component of full-game learning. Current Run mode
still routes map, reward, shop and other noncombat decisions through a
legal-random controller. Do not interpret all Run deaths as an isolated measure
of combat quality, or claim this curriculum teaches deckbuilding/path planning.
A future strategic controller is a separate learner capability.

Develop and validate the broad standalone stream first. Once Gym issue
**spire-tsw** has a source-matched qualified resolution, evaluate the shared-view
combat policy inside Run and consider a separate Run fine-tuning branch. The
old uncertain action/consumed campaign is never replayed or resumed. This
preserves both target modes without blocking pure learner preparation on Gym.

## Verification and authority

Pure tests cover deterministic generation, support/byte/count limits, all-card
and encounter coverage, split/combination exclusion, curriculum bookkeeping,
fresh on-policy attribution, censored/faulted outcomes and exact stream resume.
Use adversarial fake driver/model/optimizer boundaries for lifecycle and budget
checks. Existing fixed-manifest checkpoint rejection tests remain intact.

Before optional execution, separately qualify generated catalogs against the
actual public support/normalized catalog and run bounded CPU/GPU learner checks
on reviewed source. Sampling changes must preserve legal masks, feature
ownership, actor privacy and strict budget/fault behavior. No install or Gym
checkout edit is part of learner preparation.

This proposal does **not** spend the remaining overnight allowance, create a
release, rerun a test cohort, restart training or approve source changes in Gym.
After design approval, an implementation plan will cover source work and pure
verification. A concrete fresh runtime proposal will name the qualified Gym
artifact, exact phases, cases, effects, storage/time ceilings and stop rules
before execution is requested. Existing consumed authorities remain consumed.

## Goals and exclusions

Success means reproducible coverage beyond three decks; correct resumable
sampling; measurable comparisons on new deck/enemy compositions; and honest
evidence about whether breadth and lessons improve retention and transfer.
Coverage alone is not a playing-strength claim.

No approximate simulator, hidden actor features, unbounded catalogs, quiet
budget extension, success-seed search, automatic promotion of a regressed model,
simultaneous PPO/reward/model sweep, all-character/A20 claim or original-game
parity claim is included. RNG-001 and TIME-001 remain explicit.
