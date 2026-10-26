# Gym request: Ironclad combat training in two modes

Status: acknowledged by the existing Gym controller as `spire-dxo.66`, queued
after M11 acceptance (`spire-dxo.33.2`). Exact API/schema delivery remains pending.
Consumer: Spire Learn. This file requests a public environment capability, not
training code in Gym. Preserve the controller's current M11 work and schedule
this ticket at a safe ownership boundary.

## User request and execution setup

The user wants both independently resettable Ironclad combat training and combat
learning during normal runs. They explicitly requested communication to the
Paseo agent currently owning Gym, implementation as a tracked ticket with
GPT-6.1 Sol at high reasoning as executor and GPT-6 Luna assistants at max, and a
completion notification back to the learner agent. The existing Gym controller
already reports GPT-6.1 Sol/high; retain that setup and verify assistant max.

Create/scope the ticket in Gym's own tracker and follow its design, preservation,
implementation and review workflow. This request authorizes the compatibility
work; it is not a request to stop or overwrite current accepted/in-flight work.
Avoid competing writers or a second controller. Report substantive API/scope
questions back here; routine implementation choices belong to the Gym owner.

## Required consumer capabilities

1. A versioned standalone combat profile over the broader authoritative engine,
   initially Ironclad A0. Support varied valid encounters, decks, upgrades, HP,
   relics and potions within the declared supported content. Gym owns legal entry
   construction, shuffle, startup hooks, enemy initialization and seeded RNG.
   Use validated public configuration/catalog requests; no private snapshot
   import or learner-supplied hidden RNG/draw order. Unsupported inputs fail.
2. Keep normal RunEnv progression available. The learner must be able to route
   combat and combat-related selections/potion decisions to a learned policy,
   while a separately recorded fixed controller handles non-combat decisions.
   Public phase/lifecycle/outcome/final-resource information must let a consumer
   identify combat boundaries without confusing them with run termination.
3. Make the combat policy/encoder reusable across both modes through compatible
   public combat views/candidates, or an explicit documented mapping. Do not
   force identical full-run and combat schemas; keep feature meaning, known/
   occupancy masks, legal candidate references and version identities explicit.
4. Public Python reset/step/close, legal masks, descriptor/profile/content/rules/
   reward identities and owned numeric transport. Include public combat outcomes
   and resource checkpoints needed for analysis. Preserve real death vs faults,
   accepted-effect no-retry rules and final-state ownership. Clarify single vs
   batch/autoreset semantics. An efficient Python batch path for standalone
   combats is useful; explicitly identify any full-run batching follow-up rather
   than claiming the Rust RunBatch already supplies a Python vector facade.
5. Preserve frozen combat-v1 and existing Run profiles; introduce an explicit new
   profile/schema when required. Reuse engine owners, not a parallel simulator.
   PPO, reward shaping, curriculum choice, models and checkpoints stay in Learn.

## Acceptance and delivery

Demonstrate a small varied public standalone profile (not only Strike/Defend/Bash
against Cultist/Jaw Worm) and the same combat-facing representation during a
normal run. Verify seeds, legal actions, selections, ownership/finals, masks,
resource outcomes and faults with focused tests. No strategy win-rate gate,
training, seed search or claim of complete standard pools is required. Document
public entry/reset limits and unsupported combinations; retain RNG/TIME limits.

Return the ticket ID and intended API early so learner preparation can proceed.
At completion send: accepted commit/dirty state, build/package instructions,
public names/signatures and examples for both modes, profile/schema/fingerprint/
rules/content/reward IDs, supported entry configuration, batching/lifecycle and
error semantics, actual test results and remaining qualifications. Distinguish
source acceptance from a rebuilt extension; do not claim learner installation.
Do not install dependencies, modify files or run training in spire-learn.

## Completion callback

Destination server: srv_ps8wMRjjDjZ3
Destination learner agent: 0db1cb9a-ad7c-445b-91ae-0dd451322364
Learner workspace: wks_77e5470b3865c682

Send a Paseo message headed SPIRE LEARN COMBAT INTERFACE READY when this ticket
is actually accepted, with the delivery fields above. An acknowledgement or an
unrelated M11 completion is not this ticket's completion notification.
