# Curriculum v2: winnable late fights, reweighting and a shaped-reward arm

Date: 2026-11-19. Status: approved in conversation ("Maybe all 3?" then "yes"). This
records the agreed layering.

## Problem

On Gym v1.1.1 the baselines win 38–39 of 64 development cases, with two replicates
agreeing on 59 of 64. They almost never win late bosses. The boss winnability
diagnostic (`docs/neural-boss-winnability-20261117.md`) gives search wins of 0/10
act 3 bosses, 1/10 act 2 bosses and 1/10 act 3 elites, against 10/10 act 1 elites.
The generated late decks are random, rarity-weighted draws. They lack the damage
and scaling those fights need, so many boss scenarios give no win signal. That is
about 23% of the scenarios.

## Goal

Make late fights learnable, then measure what helps, keeping each layer's effect
separable:
1. **Winnable late fights.** Curriculum v2 builds boss and elite decks for their
   fight. A winnability gate checks it before any training.
2. **Reweighting.** The v2 stream draws fewer of any tier that stays mostly
   unwinnable. Panel scores are reported against a search-estimated ceiling.
3. **Shaped reward.** A separate experiment arm, on top of v2.

## Decisions

**A. Generator v2** (`generator_version = "ironclad-curriculum-v2"`)
- A new recipe, `configs/neural/curriculum-v2/recipe.json`, with the same support
  file. The v1 recipe, its generator path and every v1 shard and panel stay
  byte-identical. The generator dispatches on `generator_version`.
- **Stage follows the encounter.** v2 assigns the deck stage from the
  encounter's act: act 1 early, act 2 middle, act 3–4 developed. Encounters are
  drawn first. Coverage rules for the first 11 and 38 entries are kept.
- **Prepared decks for boss and elite encounters.** These replace the random
  fill:
  - one core mechanic group (strength, exhaust, block, self_damage, draw), with
    `core_cards` drawn from that group's cards;
  - extra starter removals, Strikes first;
  - a higher upgrade percent;
  - rarity weights shifted toward uncommon and rare;
  - a starting HP floor.

  All knobs live in a new recipe table, `prepared_profiles`, keyed by `act1`,
  `act2` and `act3` (act 4 uses `act3`) and by kind (`boss`, `elite`). Monster
  encounters keep v1 construction.
- **Unchanged:** reserved-pair exclusion, the composition split, uniqueness,
  bounds and entry validation.

**B. Gate, reweighting and ceilings**
- **Gate.** `scripts/neural/boss_winnability.py --recipe <v2>`, with the same
  search settings (10 per tier, 8 samples, 40-step playouts, shards 0–19). It
  must reach search wins of at least:
  - 5/10 act 1 bosses;
  - 4/10 act 2 bosses;
  - 3/10 act 3 bosses;
  - 3/10 act 3 elites;
  - act 1 elites still at least 8/10.

  The heuristic must stay below 9/10 on every boss tier, so fights do not become
  trivial. Knobs are tuned with at most 6 gate iterations. If the gate still
  fails, stop and report rather than lower it.
- **Reweighting.** A recipe table `tier_weights` maps encounter tier to a weight
  in (0, 1] for v2's non-coverage encounter draw. Every tier defaults to 1. After
  the gate, tiers with search wins below 2/10 get weight 0.25. The gate result
  and weights are recorded in the recipe notes and the results doc.
- **Panel ceilings.** A `--panel` mode in the diagnostic plays all 64 cases of a
  recipe's development panel with the search player. The search-win count is the
  panel's estimated ceiling, a lower bound. It is run for the v1 and v2 panels.

**C. Shaped-reward objective**
- Objective id `spire-learn-combat-winloss-hpshaped-v1`; the existing
  `spire-learn-combat-winloss-v1` is unchanged.
- **Potential-based shaping**, from public observation only: Φ(f) = −(sum of
  occupied enemy rows' hp) / (sum of their max_hp), with Φ = 0 on a terminal
  successor. The reward is `r' = r + β·(γ·Φ(s′) − Φ(s))`, with γ the PPO gamma.
  β is set in a new optional `[objective]` template table
  (`kind = "hp_shaped"`, `coefficient = 0.5`).
- Potential-based shaping leaves the optimal policy unchanged in theory, and it
  gives every damaging action a gradient. No hidden state is read.
- Configs without `[objective]` behave exactly as now. Checkpoints carry the
  objective id, so the arms never mix.

**D. Experiments**
- Four fresh-weight runs on Gym v1.1.1, curriculum v2, with the existing plan
  budgets and `gpu_wait`:
  - v2 + win/loss, with replicate 1's seeds and with replicate 2's seeds;
  - v2 + hp-shaped, with each of the same two seed sets.
- Each run is evaluated on the v2 panel, which the plan's recipe builds, and
  also on the v1 panel for continuity. The baseline plan gains an
  `additional_panels` list of recipe paths.
- Results go in `docs/neural-curriculum-v2-results.md`:
  - per arm and replicate, the v1 and v2 panel scores with paired tests;
  - per-tier wins;
  - the panel ceilings;
  - training-time boss win rates.

## Constraints

- No Gym checkout changes. No new dependencies.
- Simulator-only claims. RNG-001 and TIME-001 apply.
- Search and diagnostics are never training inputs.
- Each layer lands as its own PR with a Sonnet review, merged when clean. This
  plan is correctness-sensitive, so add a Sol 6.1 adversarial review for the
  generator and objective changes.
- Commits never happen while a container run is reading the repo.
- Training is second class on the shared GPU (`gpu_wait`).
