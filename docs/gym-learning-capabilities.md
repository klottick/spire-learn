# Gym capabilities for evidence-informed learning

Updated **2026-11-12** for Gym 1.0.0, native revision
`1929e18b9f44372e020877bd62f2a5c9dee3e64e`. Learn adopts its Standard combat
and Run views through the four templates under
`configs/neural/gym-v1.0.0/`. Gym 1.0.0 delivered public-information branching
(`PublicSearch`, request `spire-2o7`) and tactical combat starts (request
`spire-831`) on Standard roots. Learn has not integrated either capability:
the PublicSearch teacher and tactical-start curricula remain follow-up work.
The [Gym 1.0.0 migration record](neural-gym-v1-migration.md) lists identities,
rules changes, qualification scope and current limits. The November 2
assessment below is retained as historical context where its interface claims
were specific to the earlier release.

The [learning-strategy review](learning-strategies.md) motivates the questions
below. The delivered API supplies the foundation for the existing PPO experiment and many
imitation/curriculum experiments. Gym now supplies safe decision-time branching
and tactical starts on Standard roots; integrating and evaluating them is Learn
work. Learner compatibility and fault evidence need attention before further
training.

## Capability map

| Strategy or requirement | Delivered interface | Assessment / owner |
| --- | --- | --- |
| PPO, actor–critic, recurrence, attention, auxiliary learned predictors | Public numeric observations, known/occupancy masks, legal candidates, step/reset and outcomes. | Supported foundation. Model, losses, memory, replay buffers and optimizers belong to Learn; no new Gym algorithm API is necessary. |
| Imitation from a public-information policy | A teacher can read the same observation and choose a legal action through the existing facade. | Collecting demonstrations, scoring teacher quality and training on them are Learn work. A built-in expert or dataset is not required from Gym. |
| Randomized decks and focused entry-deck lessons | Strict catalogs let callers choose supported encounters, decks/upgrades, HP, relics and potions; Gym owns shuffle/startup/enemies. | Already available for the finite admitted content. Sampling distributions, difficulty allocation and lesson outcomes belong to Learn. |
| Learned card acquisition and Run decisions | `RunEnv` exposes the full Run observation/legal dispatch; `combat_view()` supplies a shared combat-facing view and lifecycle. | The existing fixed noncombat controller is a Learn limitation. Broader content remains bounded by the selected Run profile. |
| Public-information MCTS/lookahead or search-generated demonstrations | Gym 1.0.0 delivers `PublicSearch` branching on Standard roots. | Gym capability delivered (`spire-2o7`); Learn's search teacher and expert-iteration integration remain follow-up work. Use only the public-information contract, not exact-world state. |
| Late-turn curricula / expert iteration on the same position | Gym 1.0.0 delivers validated tactical combat starts on Standard roots. | Gym capability delivered (`spire-831`); Learn has not added tactical-start curricula and must validate its distributions before use. |
| Text/mechanic-aware teachers or semantic encoders | New `spire_lab.card_metadata.load()` / `_native.card_metadata()` exports versioned static cost, target, keywords, base values, upgrades and support metadata, joined by public card ID. | `spire-90f` is delivered upstream. Complete card-effect structure and natural-language mechanics are still absent. Live observations and profile admission remain authoritative; Learn owns encoding and teacher use. |
| Long campaigns and useful fault diagnosis | Native faults freeze owners; existing diagnostics and reproduction APIs retain fault context. The latest packet reports passing replays at the earlier `spire-tsw`, `spire-bhl` and `spire-14g` fault points. | These are upstream fixes/replays, not a completed Learn reliability campaign. Learn still needs pending-action/first-fault capture and explicit qualification of its new adapter path. |
| More normal-Run throughput | Standalone has a native serial batch, B=1..256, with explicit masked resets. The inspected Python `RunEnv` is single-session. | Potential delivery/performance gap for full-Run scaling, not a correctness blocker. Profile first; independent workers and current batching remain options. |

## Search needs a public-information contract

The November 2 audit of the earlier Python surfaces found that those interfaces
exposed
[`IroncladCombatEnv`](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/python/spire_lab/gymnasium/ironclad.py),
[`IroncladCombatVectorEnv`](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/python/spire_lab/gymnasium/ironclad_vector.py),
[`RunEnv`](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/python/spire_lab/gymnasium/run.py)
and their [`_native.pyi` signatures](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/python/spire_lab/_native.pyi),
but no independent public branch interface. Gym 1.0.0 subsequently delivered
`PublicSearch` for Standard roots, so that earlier availability statement no
longer describes the current release. Learn has not integrated that API yet.
Normal Run recording/replay remains useful for reproduction and is separate
from the player-information search contract.

Internal snapshots and the older Python combat model are not substitutes for
Gym's public planning contract. Learn must not duplicate engine state or infer
hidden future draws from exact-world branches.

Gym owns authoritative transitions and branch creation. Learn's integration
must use independent branch ownership, bounded effects/costs, cleanup,
unchanged live-root state, legal-action mapping and declared chance semantics.
Search outcomes that drive action selection must distinguish:

- **Exact-world reproduction**, used only to diagnose/reproduce an execution.
- **Player-information planning**, whose sampled worlds are consistent with
  public knowledge and do not reveal the real hidden draw order or RNG cursor.

Opaque exact-state handles alone do not solve this: a planner can exploit the
actual hidden future even if the policy never receives raw RNG bytes. Search
outcomes that drive action selection must satisfy the declared player-information
contract. Gym owns that semantics; Learn owns MCTS, action selection and teacher
distillation. The literature motivates this capability but does not establish
that it is the next most cost-effective experiment.

## Historical coverage: do not request work Gym already contains

This November 2 snapshot predates Standard support in Gym 1.0.0 and records the
Ironclad-only coverage assessment at that time.

The failed campaign used the then-frozen `ironclad-combat-v1` / common-view-v1 profile:
75 Red cards and 43 admitted encounter setups, with finite relic/potion support.
The accepted source already includes an opt-in **`ironclad-combat-v2` catalog**
and **combat-policy-view-v2**, covering 57 standalone encounter setups. The
old adoption record confirms its then-current imported v2 descriptor fingerprint
`02433813e1297244060d9852f342c6eb4c9c4d62f3dde9763ff82191867eceec`.
That record forwarded zero gameplay actions; it does not establish gameplay
conformance for all v2 setups. The freeze has now been lifted. In the delivered
November 2 build the v2 fingerprint is
`5ba729fd1b10a067fb8fc5d777d78b47fc9526ec0af0030d6868a32b52a7824b`;
the old fingerprint must not be used for the new binary.

Learn's parser and driver now select the exact new v1 contract and
catalog shape. This required migration: categorical IDs changed,
combat-card/known-draw capacity grew to 1,024 and public lifecycle phase fields
were added. Old checkpoints need retraining or an explicitly tested conversion;
changing their fingerprint labels would corrupt their meaning. Using v2 also
requires an explicit learner compatibility change and
qualification, not weakening those checks or asking Gym to recreate the extra
encounters. Existing checkpoints are not automatically compatible with the
expanded vocabulary. [Gym encounter inventory](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/docs/encounter-coverage.md),
[Learn parser](../src/spire_learn/neural/schema.py),
[driver](../src/spire_learn/neural/envs.py).

The inspected Python delivery remains Ironclad A0, with finite supported
resources and captured normal-Run profiles. Broader character/Ascension support
in Rust does not establish a Python training interface. The delivered Run
Python stub still lists only `m9-ironclad-act3`, `m10-ironclad-heart` and
`m10-ironclad-policy`; M28 standard-run source work is not a new Python profile.
V2 standalone closure also does not switch normal Run to complete standard pools.
These are real scope limits for eventual full-game transfer, not reasons that
we cannot learn useful Ironclad A0 combat now.

Resource-entry lessons are also narrower than authentic arbitrary Run states:
only supported public counters can be supplied, and there is no arbitrary
mid-combat import. More realistic captured public starting distributions could
be useful later, but they need validated Gym semantics and separately measured
benefit. No private snapshot or hidden draw-order import should be requested.
[Gym construction contract](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/docs/combat-training.md#validated-public-construction).

## Public card metadata is now delivered

The new [card metadata API](https://github.com/klottick/spire-gym/blob/afb51f5f222b4c4331ed34067c5eebeda606db91/docs/card-metadata.md)
addresses an important part of `spire-90f`. It describes printed faces,
including cost class, target, keywords and constructor numbers, with its own
content hash and descriptor fingerprint. The Python helper indexes by the same
string IDs as the observation vocabulary and renders a short description.
This enables public-state explanations and future semantic encoders without
importing wiki text or exposing hidden game state.

It is not a complete effect language or rules-text export. Eleven card kinds
have fields unresolved by the neutral probe; display names, full mechanic
summaries and upgrades past the first are outside v1. A global metadata row
does not admit that card to a particular profile. Use the session's support and
vocabulary, and let live observations/legal masks override printed values.
A Learn semantic model still needs its own design and evaluation.

## Reliability and diagnosis come first

Campaign c faulted during a B16 native step after 160 accepted batch calls.
Its 16 submitted action slots remain uncertain; it was not a gameplay loss.
Gym issue **`spire-14g`** records that incident; its fix and a replay crossing
the fault point are now included in the delivered release. This does not turn
the failed campaign into a success. Learn issue
**`spire_learn-pa7`** tracks pending-action/first-fault evidence capture.
The original Learn audit did not identify the failing public case. Later
upstream replay/fix evidence is a separate record, not information the original
campaign retained.

The current batch diagnostic already reports per-session accepted counts and
first-fault/failing-slot metadata. Learn should preserve a bounded submitted
operation intent and attempt to retain safe existing diagnostics before close,
while preserving the primary exception and never retrying an uncertain effect.
If that capture reveals insufficient Gym evidence, a precise upstream request
can follow. Adding a generic new debug API before using the current one would
misdiagnose the immediate gap. Diagnostic or provenance fields must never enter
actor/critic features.

## What to request upstream, and what stays here

1. **Current:** Learn adopts the source-matched Gym 1.0.0 release and Standard
   views. Complete the one-reset qualification and preserve the capture gap
   evidence. Do not retain old binary/descriptor pairings or reuse old
   checkpoints under new categorical IDs.
2. **Follow-up in Learn:** integrate Gym 1.0.0 `PublicSearch` into a bounded,
   public-information-safe search teacher and measure teacher quality before
   expert iteration. Gym owns authoritative branch/chance semantics; Learn
   owns search, action selection and distillation.
3. **Follow-up in Learn:** evaluate Gym's delivered tactical starts as a
   curriculum with validated public scenarios.
4. **Optional:** consume the delivered card metadata in a semantic encoder or
   teacher; request richer mechanics against a concrete missing field.
   Broader Python profiles and normal-Run batching remain separate improvements
   needing a consumer and cost/benefit review.

PPO variants, imitation losses, replay datasets, curricula, adaptive sampling,
model encoders, reward shaping, teacher choice and evaluation design remain
Learn responsibilities. Full-game LLM demonstrations do not require a new Gym
MCP server: a teacher can be integrated through Learn's existing observation/action
boundary, subject to a suitable public-state representation and measured quality.

The source assessment changed no Gym checkout. The separate released
zero-gameplay check qualified the new package and both adapter constructors;
it ran no gameplay or training and grants no future execution authority.
Original-game parity, RNG-001 and TIME-001 remain separate limitations.

## Tracking follow-up — 2026-11-02

At the user's request, three P2 feature issues were created and read back in
Gym's shared Docker-backed Beads database (`spire`):

| Issue | Capability |
| --- | --- |
| `spire-2o7` | Bounded public-information combat branching for search |
| `spire-831` | Validated mid-combat tactical starting scenarios |
| `spire-90f` | Public card metadata delivered and hash-qualified in Learn; semantic model use remains future work |

They include consumer evidence, acceptance criteria, compatibility and
information boundaries, and links to related prior Gym work. They do not
duplicate the existing fault/capture issues or start implementation or training.
