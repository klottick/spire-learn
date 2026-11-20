# Standard combat foundation, guided by Baalorlord's runs: design

Date: 2026-11-20. This is project 1 of the A20 Heart roadmap below.

## Why

The user's goal is a policy that wins A20 runs from Neow to the Heart with all
four characters. A separate model per character is acceptable.

Spire Learn currently trains combat on the Gym's older M10 Ironclad-only
profile, using a hand-tuned 75-card deck generator. The Gym (v1.1.1,
`e3c13836`) supports all four characters, every card, relic and potion, and the
real ascension rules under its Standard profile.

Baalorlord's 2,400 public runs give real data to build from. All of them are
A20, 1,433 are wins, and they cover every character. They are already
downloaded locally (`artifacts/development/neural-combat/baalorlord-runs-20261119/`,
`raw/` for Ironclad and `raw-other/<character>/` for the others). Each run file
records the deck changes, relics, potions (obtained, used, discarded), HP and
gold for every floor, plus the encounter on each floor.

The user's direction:
- Use Baalorlord as the guide for the foundation.
- Start from his exact deck at every floor.
- Then fuzz it: replace one random card, then two, and so on, and the same for
  relics.

## Roadmap (context only; this design covers project 1)

1. **Standard combat foundation, Ironclad A20.** This document.
2. **The other three characters.** A combat model each. The pipeline here is
   built character-generic so this is mostly configuration.
3. **Run decisions.** Card picks, paths, shops, campfires, events and Neow
   choices.
   - Imitate Baalorlord's recorded decisions first, then fine-tune by self-play
     in `RunEnv`.
   - Measure acts cleared, bosses beaten and Heart wins per character.
   - It reuses this project's per-floor reconstruction.

The headroom panel and scaling probes design
(`2026-11-19-scaling-probes-design.md`) is on hold. It will be rewritten for
this foundation once project 1 lands.

## Goals

1. **Per-floor reconstruction.** For every combat floor of every downloaded
   run, rebuild the state the player entered that fight with:
   - the encounter;
   - the deck, with upgrades and `misc`;
   - relics in acquisition order, with counters where recorded;
   - potions;
   - current and max HP, and gold.

   All four characters are covered (this is cheap, and needed by projects 2–3).
   Output is a canonical, hashed local table, like the existing real boss deck
   table.
2. **Baalorlord statistics report.** Per character, act and fight kind, report:
   - deck size, upgrade fraction and card-type mix;
   - the most common cards and relics;
   - potion counts and HP fraction;
   - his win rate per encounter.

   This is the "what real decks look like" insight. It is committed as a doc
   with aggregate numbers only.
3. **A Standard-rules Ironclad A20 curriculum built from the reconstructed
   states**, with the fuzzing ladder below, held-out panels split by run, and a
   winnability check.
4. **Training on Standard combat.** Train through Learn's existing, never-trained
   `standard-standalone` mode, after a throughput measurement and any fixes it
   calls for.
5. **A fresh Ironclad A20 baseline**, two seed pairs, evaluated on the new panels
   with a search reference.

## Non-goals

- Silent, Defect and Watcher training (project 2); reconstruction covers them
  only.
- Run decisions, imitation learning and `RunEnv` training (project 3).
- Architecture changes or scaling probes.
- Removing the M10 path. It keeps working for old results, but gets no new
  features.
- Comparing new scores with M10 results. Rules, view and panels all differ, so
  the new baseline starts a new series.

## Anti-goals

- **Leaking held-out runs into training.** Splits are by whole run, never by
  fight. No state, or fuzzed variant of a state, from a selection or final run
  ever enters training.
- **Silent reconstruction errors.** A state that cannot be rebuilt exactly is
  dropped and counted, never guessed. This includes unknown card names,
  unrecorded deck changes, and entries the Gym rejects.
- **Publishing his data.** The raw runs, the reconstructed table and the
  curriculum shards stay local under `artifacts/`, are regenerable, and are
  never committed or published. Only code, synthetic test fixtures and
  aggregate statistics are committed.

## Reconstruction

- **Generalize, don't rewrite.** Extend `scripts/neural/real_boss_decks.py`'s
  rebuild logic from boss floors to every floor that had a fight. That logic
  undoes later picks, purchases, purges, smiths and event changes, and drops
  decks that unrecorded relics changed.
- **Fight floors.**
  - Fights come from `damage_taken` (monster, elite and boss rooms, including
    the A20 double boss and act 4).
  - The fight's starting HP is the HP entering the floor.
  - Elite fights with the emerald key flame are marked `burning` where the run
    records it.
- **Floor convention.** The state entering a fight on floor F applies every
  recorded change on floors before F.
  - On floor F itself, potions used or discarded on F were held entering the
    fight.
  - Potions, cards, relics and gold obtained on F are post-fight rewards.
  - *My call:* event fights whose event changes the deck or relics before the
    fight, where the run does not record the order, are dropped.
- **Potions.** Rebuild the potions held from `potions_obtained` (floor and
  key), `potion_use_per_floor` and `potion_discard_per_floor`; the last two
  are per-floor lists of potion names.
  - About 80% of runs have these logs. For runs without them, the user's
    ruling applies: every fight state is kept, with no potions. Those states
    are tagged `potions_assumed_empty` and reported separately in the
    statistics.
  - Potions also come from Alchemize and Entropic Brew; the
    `potions_obtained_alchemize` and `potions_obtained_entropic_brew` fields
    record them.
  - *My call, following the user's ruling:* if a floor's potion state is still
    ambiguous, for example more potions than slots, the state is kept with no
    potions and tagged the same way.
  - Potion slots follow A20 (2) plus Potion Belt.
- **Bottles and counters.**
  - Bottled cards come from `relic_stats` (for example
    `"Bottled Tornado": "<card>"`). A bottle whose card is missing, or is not
    in the deck, causes a drop.
  - Relic counters come from `relic_stats` where it records them. Where it doesn't,
    relics take the Gym's entry defaults, and the drop/default policy for each
    such relic is listed in the report.
- **Names.**
  - Card names map to Gym ids by `card_metadata()`.
  - The Gym publishes no relic or potion lists, so relics and potions use a
    committed alias table. It holds only name pairs, and each pair is checked
    by Gym catalog validation.
  - Unmapped names cause a drop and are counted by name.
- **Event fights.** Fights from events (Colosseum, Masked Bandits, Mind Bloom,
  the Lagavulin event, Mushroom Lair) are kept if the Gym lists a matching
  encounter row. Otherwise they are dropped by name as a scope exclusion,
  reported, and kept out of the drop-rate denominator.
- **Burning elites.** An elite is marked `burning` on the floor in
  `green_key_taken_log`. *My call:* other elites are treated as non-burning.
  A burning elite he fought but whose key he did not take would be mislabelled,
  so this is a stated qualification, not a silent guess.
- **Gym acceptance.** Every kept state is converted to a `standard-combat-v1`
  entry (character, ascension 20) and must pass Gym catalog validation.
  Rejections are dropped and counted by reason.
  - *My call:* the drop rate is measured over fight states in runs that have
    full logs, excluding the scope exclusions above. If more than 10% of a
    character's states drop for reasons other than those, stop: report the
    breakdown and investigate before building a curriculum, because the
    reconstruction is probably wrong somewhere.
- **Outcome.** Each state records whether he won that fight. All fights are
  kept, including the one a lost run died to.

## Curriculum (Ironclad, A20)

**Splits.** Every run is assigned to train, selection or final by a hash of its
`run_id`, 80 / 10 / 10. That is roughly 540 / 68 / 68 Ironclad runs and about
40 fights per run. The final split is not touched in this project except to be
built.

**Fuzz ladder.** A case is a reconstructed state at fuzz level *k*:
- **Level 0:** his exact state.
- **Level *k* ≥ 1:**
  - Replace *k* random cards of the deck (sampled without replacement; capped at
    deck size). Each replacement is drawn uniformly from the Ironclad cards that
    appear anywhere in his train-split Ironclad runs, and keeps the replaced
    card's upgrade count where the new card allows it.
  - Replace *k* random relics, never the starter relic (capped at the number of
    non-starter relics). Each replacement is drawn uniformly from the relics
    seen in his train-split Ironclad runs.
  - Encounter, HP, gold and potions are unchanged.
  - Fuzzed entries must pass Gym validation; a rejected draw is redrawn, up to
    a bound, then the case is skipped and counted.
  - *My calls:*
    - Bottle relics are not in the replacement pool.
    - If an existing bottle is replaced, its bottled card stays in the deck,
      unbottled.
    - If a bottled card is replaced, its bottle is replaced too.
    - A replacement relic takes the Gym's default counters.
    - The card pool includes the curses that appear in his decks.
- *My calls:*
  - cards and relics move together under one *k*;
  - the replacement pools are "seen in his runs" rather than everything the Gym
    allows, to stay realistic;
  - the maximum level is *k* = 4 for this project.

**Schedule.**
- Training starts on level 0 only.
- At each quarter of the training budget, the next level joins the mix, and
  cases are drawn uniformly over the enabled levels:
  - first quarter: {0};
  - second: {0, 1};
  - third: {0, 1, 2};
  - last: {0, 1, 2, 4}.
- *My call:* level 3 is skipped to reach 4 within the budget.
- The schedule is part of the recipe and its hash.

**Stream.** Cases are drawn from train-split states with the existing shard,
recipe-hash and resume machinery, generalized from the M10 `standalone` mode to
`standard-standalone`.
- Shards are deterministic for a recipe.
- There is a new generator id, `baalorlord-standard-v1`.

**Panels.**
- **Selection panel:** 256 cases from selection-split runs, stratified by act and
  fight kind (monster, elite, boss), at fuzz levels 0, 1, 2 and 4 (64 each).
  - *My call:* if an act × kind cell has too few base states, it takes all of
    them, and the shortfall goes to the other cells of the same level in
    proportion to their size.
  - Base states are distinct within a level. The same base state may appear at
    different levels.
- **Final panel:** built the same way from final-split runs, and only built, not
  scored, in this project.
- Panels are deterministic and pinned by hash.

**Winnability check.**
- Run the Gym's `PublicSearch` (already used by `scripts/neural/boss_winnability.py`)
  on the selection panel. With Standard rules this now matches the training
  rules exactly.
- Report search wins per stratum and level, beside his own recorded win rate on
  the level-0 cases.
- This is a reference, not a gate, unless a stratum is at 0 search wins and 0 of
  his wins. Such a stratum is reported and excluded from the schedule.

## Training on Standard combat

- Use the existing `standard-standalone` mode and the Gym's
  `StandardCombatVectorEnv`.
- Lift the guards that currently admit only M10 `standalone` in collector,
  curriculum, stream resume, stream runner and calibration, so Standard runs
  through the same paths.
- Smoke Bomb escapes are a separate outcome (`escaped`). They count as neither
  a win nor a loss in panel scores and are reported.
- **Throughput first.**
  - Before the baseline, measure decisions per second at 16 and 64 rows on the
    Standard view against the M10 figure, which was about 285 decisions per
    second at 16 rows. Use the existing `profile_entry.py`.
  - *My call:* if Standard is below 100 decisions per second at the chosen row
    count, a throughput task comes before the baseline.
  - That task applies the same kind of fixes as before: validation fast paths,
    and trimming rows that are known to be empty.
- Identity: new `features_id` and `model_id` values for the Standard view. M10
  checkpoints are not loadable into it, and that is expected.

## Baseline

- Two seed pairs, reusing the v1.1.1 baselines' seeds.
- Model and PPO settings as in `baseline-gym-1.1.1-cv2`. Budgets are scaled so
  wall time fits; the builder sets them from the throughput measurement.
- Each run evaluates the selection panel at fresh, intermediate and final
  checkpoints, using the existing paired evaluation, censoring and `gpu_wait`
  gate.
- **GPU time:** the user allows all GPU time not used by Hermes.
- **Results doc:**
  - learning curves;
  - per-stratum and per-level wins against search and against his recorded wins;
  - fresh-vs-final paired tests;
  - drop counts from reconstruction;
  - qualifications (simulator only, RNG-001/TIME-001, one player's data, the
    burning-elite labelling).

## Constraints

- AGENTS.md applies. In particular:
  - the actor and critic see public observations only;
  - Gym faults are errors;
  - do not modify the Gym checkout;
  - artifacts stay out of Git;
  - no original game code or assets.
- Tests are deterministic and CPU-only, and use hand-written synthetic run
  fixtures, never real run data. They cover:
  - per-floor rebuild, including potions and bottles;
  - name mapping and drop counting;
  - split-by-run disjointness;
  - fuzz-level correctness: exactly *k* cards and *k* relics differ, the
    starter relic is kept, and the result is valid;
  - schedule progression;
  - panel determinism;
  - the lifted Standard-mode paths.
- Experiments record the gym revision and dirty state, table and recipe hashes,
  panel hashes, seeds, budgets and GPU waits.

## Left to the builder

- Module layout.
- The exact potion-reconstruction field logic.
- Which relic counters need explicit handling.
- The redraw bound.
- Shard sizes.
- Budget numbers after the throughput measurement.
- Whether the statistics report is a script plus doc or part of extraction.
