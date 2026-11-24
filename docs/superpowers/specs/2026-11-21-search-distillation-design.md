# Search teacher and distillation: design

Date: 2026-11-21; updated 2026-11-23 for Spire Gym v1.2.0. Follows
`docs/neural-standard-foundation-results.md` and
`docs/neural-standard-baseline-gym-1.2.0.md`.
This is the first project of the search-plus-learning direction (roadmap below).

## Why

The Standard foundation baselines stop well short of what the fights allow.

**On Gym v1.2.0**, the series this project runs on, the fixed 256-case selection
panel has manifest `eb4a50d8…` (panel identity `2247fe2b…`):

| Player | Wins / 256 | Boss wins / 40 |
| --- | ---: | ---: |
| PPO final, r1 and r2 (v1.2.0 baselines) | 166 and 168 | 12 and 10 |
| PPO best checkpoint (25%), r1 and r2 | 190 and 183 | 13 and 13 |
| Search reference (one move deep, heuristic playouts) | 221 | 25 |
| Baalorlord, recorded, level-0 cases only | 63 / 64 | |

**The v1.1.1 series**, for orientation only. It is a separate series and is not
paired with the table above. Its selection panel, manifest `66c9835c…`, gave:
- PPO finals: 186 and 181 wins, with 12 and 12 boss wins.
- Search reference: 220 wins, with 24 boss wins.

The PPO policy acts by reflex: one forward pass, no lookahead. A crude search
that looks one move deep beats the v1.2.0 finals by 53–55 fights.

**Both v1.2.0 baselines peak early and decline.** They are best at 25% of
training and lose 15–24 wins by the final checkpoint, mostly in elite fights
(`docs/neural-standard-baseline-gym-1.2.0.md`):
- the training-stream win rate stays flat;
- entropy falls;
- the critic's value loss rises.

The final checkpoint stays the pre-registered comparison point. The PPO arm in
part 5 uses the same objective and may drift the same way, so its learning
curve is reported at every checkpoint. The user chose to
pursue search plus learning (AlphaZero-style expert iteration) with an entity
transformer.

Outside evidence is mixed, so this project is staged as cheap, measured bets.

- **No public success in Slay the Spire.** The literature review
  (`artifacts/development/neural-combat/literature-20261101/expanded/`) found
  no credible Slay the Spire result for expert iteration or network-guided
  search.
- **One reported failure.** The Jialeiv project reports six failed attempts to
  learn combat from a search teacher, with imitation accuracy capped near 44%.
  Its teacher could see the future draw order, so it is not a clean test of
  this plan.
- **Positive evidence from another card game.** A 2026 Legends of Code and
  Magic preprint (Rubin) went from 26.8% to 51.35% wins by adding
  belief-sampled search to policy and value networks.
- **Imitation then RL helps.** Haluška and Schmid (2024) report 54.2% against
  41.8% for PPO started from behaviour-cloned weights against random weights.
- **Lessons from Silver Automaton** (`repos/references/sts_lightspeed-daniel-ziegler`):
  - a search that peeks at hidden draws overstates strength by tens of points;
  - a real tree search with chance nodes matters;
  - per-fight gains can reverse over full runs;
  - HP kept matters, not only winning.

## Roadmap (context only; this design covers the first item)

1. **Search teacher and distillation.** This document.
2. **Learned value inside the search**, then the closed loop (search, train,
   repeat).
3. **Other characters.** The v1.2.0 re-extraction already covers all four
   characters and tags relic counters (`relic_counter_defaulted`).
4. **Run decisions.**
   - Imitate Baalorlord first, then self-play.
   - Use the combat value as a deck evaluator.
   - Add a full-run A20 check. About 20% Heart wins is the public state of the
     art (Silver Automaton 18.6%; Epoch SpireClimb about 20% with an LLM).

The scaling-probes design (`2026-11-19-scaling-probes-design.md`) is superseded
by gate G3 below.

## A rule that changes

Earlier specs and the `boss_winnability.py` docstring say search is a
measurement and never a training input. **From this project on, search results
may be training targets** (approved by the user on 2026-11-21), under these
conditions:

- The search uses only the Gym's `PublicSearch` roots and branches. These
  sample worlds consistent with public information.
- It never uses exact checkpoints, forks of the live session, seeds, records or
  any other privileged surface.
- Networks receive only public observations and legal-action masks, as before.
- Research seeds and search statistics are stored as labels and provenance.
  They are never network inputs.

AGENTS.md's rule on privileged information is unchanged and still binds.

## Goals

1. **A better teacher.** A public-information tree search with chance handled
   by sampled worlds, measured against the current one-move search.
2. **A teacher audit.** How consistent are the teacher's own choices, and how
   does its strength grow with budget? This bounds what any imitator can reach.
3. **A boss panel.** A second fixed panel weighted to bosses and elites,
   because that is where the gap is and the main panel has only 40 boss cases.
4. **A search-labelled dataset** of training-split decisions.
5. **Three distilled students** trained on identical data:
   - the current network;
   - an entity transformer with pointer-style action scoring;
   - the same transformer with public card-metadata features.
6. **A PPO arm started from distilled weights**, against PPO from scratch.
7. **Verdicts on four pre-registered gates**, in a results doc.

## Non-goals

- A learned value or policy inside the search, and the closed loop (next
  project).
- Run decisions, full-run evaluation, other characters.
- Scoring the final-split panel. It stays untouched.
- Re-extracting the state table. This project uses the v1.2.0 table
  `f4c2ae7a…`, through recipe `baalorlord-standard-gym-1.2.0` (`34a80641…`),
  as is.
- Tuning PPO or the curriculum.

## Anti-goals

- **A teacher that peeks.** Any path that lets hidden draw order, enemy RNG or
  seeds influence a label is a defect, not a strength gain.
- **Leaking held-out fights.** Labelled decisions come only from train-split
  states. Selection and final states are never labelled for training.
- **Moving the goalposts.** The gates below are fixed before results exist.
- **Publishing derived data.** The labelled dataset derives from Baalorlord's
  runs. It stays local under `artifacts/`, like the state table.

## Part 1: the teacher

**Current teacher** (`scripts/neural/boss_winnability.py`, `search_choice`):
for each legal action, 32 sampled worlds, a heuristic playout of up to 80
steps, pick the best mean. It looks one move deep and falls back to the
heuristic when the Gym refuses a search root. On v1.2.0's selection panel that
was 413 of 6,390 attempted roots (6.46%), against 9.3% on v1.1.1.

**Tree teacher.** A Monte Carlo tree search over public observations:

- **Sampled worlds.** Each simulation runs in a sampled world, `branch(k)` of
  the root's `PublicSearch`, with `k` cycling over a fixed set of worlds.
- **Nodes keyed by public view.** Tree nodes are keyed by the public combat
  view reached, not by the world, so statistics are shared across worlds that
  look the same to the player. Random events (draws, enemy rolls) lead to
  different child nodes by producing different views. This keeps decisions
  conditioned on public information only.
- **Selection** by an upper-confidence rule.
- **Leaf evaluation** by the existing heuristic playout, scored as now
  (terminal reward, else half the HP balance).
- **Root actions.** Every legal root action receives at least a fixed minimum
  number of simulations, so each has a mean value. *My call:* the minimum is 8;
  the builder may spend the root budget by successive halving.
- **Root choice** is the action with the highest mean value.
- **Per-action record.** The teacher records the mean value and simulation
  count of every legal root action; the means become the soft labels in part 3.
  The current one-move search fits the same record: every action has a mean
  over its 32 worlds.
- **Unavailable roots.** Where the Gym refuses a root (hidden intents, draw
  knowledge outside its exact model, held nested targets), the teacher plays
  the heuristic move and the decision gets no label. The rate is reported per
  reason.
- **Faults.** Gym faults are errors. A branch refusal inside a simulation
  counts against that simulation as now and is reported.

**Use the Gym's batched search.** The current search steps one branch at a time
from a Python loop. The Gym already provides more, and the teacher must use it:
- `step_many` steps many branches in one call on a worker pool
  (`public_search(..., workers=n)`);
- `view="mask"` skips observation encoding where the playout needs only the
  legal mask;
- `fork` copies a branch in about 2 µs, so the tree forks instead of replaying
  from the root;
- compact branch observations. A `PublicSearch` selects its compact combat
  layout independently of the session. The v1.2.0 search reference used dense
  branch observations, so whether compact ones give the same results and cost
  less is unmeasured. The builder measures both before using them.

The speed-up over the current search is measured and reported before budgets
are fixed. These measurements (time per decision, and the share spent in Python
round trips) are also what the Gym's bead `spire-n6b` (native playouts) waits
for, so they are posted there.

*My calls:*
- the search stays in Learn's scripts and uses only the Gym's public API; the
  Gym checkout is not modified;
- the heuristic playout policy is unchanged in this project;
- the upper-confidence constant, the tree depth limit and progressive widening
  are left to the builder, tuned on training-split fights only.

**Budgets.** Three teacher budgets are measured on both panels: roughly 1×,
3× and 10× the CPU cost per decision of the current search. The v1.2.0 search
reference took about 0.68 s of wall time per searched decision: 4,046 s for
5,977 searched decisions, in one process on a four-CPU container. The builder
re-measures with the batched calls.

**Caps for teacher fights.** A teacher fight is capped at 1,024 decisions, the
same action cap the policies get, and has no per-case wall-clock cap. A fight
that reaches the cap is censored. The same applies when the teacher generates
the dataset.

**Choosing the dataset's teacher.** The dataset uses the teacher with the most
selection-panel wins among the candidates costing at most 3× the current
search: the current search, and the tree teacher at 1× and 3×. Ties go to the
cheaper one. The 10× budget is measured for the strength curve only; it would
not fit the labelling budget in part 3.

## Part 2: audit and panels

**Teacher audit**, on a fixed sample of 2,000 labelled decisions from
selection-split fights (audit only, never training data):

- **Self-consistency.** Run the teacher twice with different research seeds.
  Report top-choice agreement and the value lost when one run follows the
  other's choice. This is the ceiling for imitation accuracy.
- **Strength against budget.** Wins, HP retained and CPU seconds per decision
  at each budget, on both panels.
- **Root refusals** by reason, and by act and fight kind on both panels.
- **Label coverage.** The share of decisions with a search root, by act and
  fight kind. *My call (stop rule):* if fewer than half of boss-fight decisions
  in the teacher's own fights have a root, stop and report before building the
  dataset. Otherwise the teacher would mostly be the heuristic exactly where the
  gap is.
  - **Measured on v1.2.0.** Learn replayed every train-split Ironclad level-0
    fight with the heuristic (`docs/neural-standard-baseline-gym-1.2.0.md`).
    The share of boss-fight decisions with a root was:
    - act 1: 97.4%;
    - act 2: 90.8%;
    - act 3: 85.2%;
    - act 4: 86.4%.
  - Across all fights, roots were available at 91.08% of Ironclad decisions and
    91.96% for all four characters.
  - The refusals are hidden intents and draw knowledge outside the Gym's exact
    model. Hidden intents means every decision in a Runic Dome fight: 5.21% of
    Ironclad fights.
  - The rule is therefore expected to pass with a wide margin. It still binds
    on the teacher's own fights, because a stronger teacher reaches different
    positions than the heuristic does.

**Boss panel.** A second fixed panel from selection-split runs, built by the
existing panel machinery with its own panel identity:

- 256 cases: 64 at each fuzz level 0, 1, 2 and 4.
- Per level: 40 boss cases and 24 elite cases, stratified by act in proportion
  to available base states.
- Base states are distinct within a level. Cases may share base states with
  the main selection panel; the results doc says so.
- The v1.2.0 selection panel (`eb4a50d8…`) is unchanged and stays primary.

Both v1.2.0 PPO baselines and the current search are re-scored on the boss
panel, by evaluation only.

## Part 3: the labelled dataset

- **Source fights.** Train-split Ironclad fights from the
  `baalorlord-standard-gym-1.2.0` recipe, drawn with the final-quarter level mix
  (fuzz levels 0, 1, 2, 4).
- **Who plays.** The teacher chosen in part 1. *My call:* one fight in five is
  a mixed fight, so the dataset covers positions a weaker player reaches. In a
  mixed fight, each move played comes from the v1.2.0 PPO r1 final policy with
  probability one half. Every decision is still labelled by the teacher.
- **Value targets come from teacher-only fights.** A mixed fight's outcome is
  not the teacher's, so mixed fights carry policy labels only.
- **What is stored per decision:**
  - the public observation and legal mask, as the learner already records them;
  - the teacher's value and visit count for every legal action, and its choice;
  - the fight's eventual outcome (win, loss, escape) and the fraction of HP
    kept at the end;
  - provenance: recipe, table and panel identities, teacher version and budget,
    research seed, Gym revision.
- **Size.** 1.5 million labelled decisions, comparable to one PPO baseline.
  There is no wall-time cap on labelling (user's ruling). At the v1.2.0
  reference's 0.68 s per decision, that is roughly 12 hours on 24 cores at 1×
  cost and 35 hours at 3×. The builder re-measures with the batched calls.
- **Sparse storage.** A dense observation is about 72,000 values, nearly all
  padding, so 1.5 million of them would be hundreds of gigabytes. The compact
  view (bounds preset) removes some of that. The dataset still stores only
  occupied rows per group. A stored decision must expand back to exactly the
  observation the student is trained on, and a test pins that. Widening a
  compact observation and zero-padding it gives exactly the dense observation,
  as the re-pin showed.
- **Held-out labels.** 2% of train-split *runs* are set aside as a validation
  set for supervised training (early stopping, temperature). They are still
  train-split runs, so the selection and final splits stay untouched.
- **Determinism.** The same recipe, seeds and teacher version reproduce the
  same dataset bytes. Shards carry a hashed manifest.
- **Storage.** Local, under `artifacts/development/neural-combat/`, never
  committed.

## Part 4: the students

All three train on the same dataset, with the same number of gradient steps
and two seeds each.

| Student | Network |
| --- | --- |
| S1 | The current feed-forward network (`spire-learn-combat-ff-standard-v1`). |
| S2 | Entity transformer with pointer-style action scoring. |
| S3 | S2 plus public card-metadata features on card tokens. |

**Entity transformer (S2).**
- **Tokens.** One token per occupied row of the public view: each card in each
  zone, each enemy, each power, each potion, each relic, each orb, plus one
  context token. Padding rows are not tokens.
- **Trunk, three sizes** (user's request, to show a trend). All are pre-norm.

  | Size | Layers | Width | Heads |
  | --- | ---: | ---: | ---: |
  | Small | 2 | 128 | 4 |
  | Medium | 4 | 256 | 8 |
  | Large | 6 | 384 | 8 |

  No published combat transformer exists for this game, so the right size is
  unknown; supervised training is cheap enough to try all three.
- **Size selection.** *My call:* the size with the lowest validation loss
  becomes S2, and S3 uses the same size. The choice is made on the validation
  set, never on a panel. All three sizes are scored on the panels with two
  seeds each, so the trend is visible; only S2 and S3 enter the gates.
- **Inference time** per observation is reported for every size, because the
  next project puts the network inside the search.
- **Action scoring.** Each legal action becomes a query built from its action
  row and the tokens it refers to (the card played, the target). Its score
  comes from that query attending over the trunk output. Only legal actions are
  scored.
- **Value heads.** Two outputs from the pooled trunk: the probability of
  winning the fight, and the fraction of HP kept.

**Card-metadata features (S3).** Card tokens also receive the Gym's public
card metadata, in addition to the card id embedding: colour, rarity, type and
tags, and per upgrade face the keywords, cost, target, base damage, base block
and magic number. Nothing from run data is used.
- The metadata is read from the pinned Gym build inside the runtime container.
  `card_metadata()` is present at v1.2.0 (`4d198064`, `docs/card-metadata.md`),
  and the name-alias work already reads it there.
- A committed snapshot carries the Gym's fingerprint, and a Gym-gated test
  checks the snapshot against the live build, as the alias table does.

**Training.**
- **Policy loss.** Cross-entropy to soft targets: a softmax over the teacher's
  per-action values with temperature τ. τ is the one tuned hyperparameter,
  chosen on the validation set and shared by all students.
- **Value loss.** Win probability against the fight outcome; HP fraction against
  the HP kept. Escapes count as neither win nor loss, with HP kept as recorded.
  Only teacher-only fights contribute; the value heads therefore estimate
  outcomes under teacher play.
- **Decisions without a search root** carry no policy label and contribute to
  the value loss only.
- **S1's value head** gains the second output; its trunk is unchanged.
- **Identities.** New model, features and objective identities for S2 and S3
  and for the distillation objective. Checkpoints of different families must
  not load into each other, as today.

**Hyperparameter searches.** Every choice below is made on the validation set
with one seed. Panels are used only for the final two-seed runs.

| Search | Values | Applied to |
| --- | --- | --- |
| Learning rate | 1e-4, 3e-4, 1e-3 | Each architecture separately (S1 and the three transformer sizes) |
| Soft-target temperature τ | three values spanning sharp to soft; the builder picks them from the spread of teacher values | Chosen once on S1, shared by all |
| Number encoding | learned embedding of bucketed values, against scaled and sinusoidal features | Medium transformer; the winner is used by all transformers. This covers HP, block, energy, costs and power amounts. |
| Dataset size | 25%, 50%, 100% of the labelled decisions | S1 and S2, final settings, two seeds; reported as a learning curve |

- The dataset-size curve shows whether the students are limited by data or by
  the network. It is reported on the panels as a trend and is not a gate.
- *My calls:*
  - loss weights between policy and value are fixed at 1 and are not searched;
  - the share of mixed fights and the teacher's constants are not searched
    here, because changing them means relabelling.

**Evaluation.** Each student plays by reflex (its top-scoring legal action, no
search) on both panels. Reported: wins, escapes, censored, HP kept, boss wins,
and top-choice agreement with the teacher on the validation set beside the
self-consistency ceiling.

## Part 5: PPO from distilled weights

- **Arm.** PPO on the existing curriculum and budget (1,622,016 decisions),
  started from distilled S1 weights. There are two runs: S1's first seed starts
  the run with baseline r1's seeds, and S1's second seed starts the run with
  r2's.
- **Control.** The v1.2.0 from-scratch baselines r1 and r2
  (`configs/neural/baseline-standard-1.2.0{,-r2}/`, compact view).
- PPO's objective stays win/loss. The win-probability head seeds the critic;
  the builder maps it to PPO's value scale.
- **Pilot.** PPO started from trained weights can undo them early. *My call:*
  before the two full runs, a pilot of about 100,000 decisions compares the
  baseline PPO settings with a learning rate one third as large. The setting
  with the higher training-stream win rate is used for both full runs. The
  panels are not used for this choice.
- *My call:* a transformer PPO arm (from distilled weights, with a from-scratch
  transformer control) runs only if gate G3 passes. It is otherwise deferred.

## Gates (pre-registered)

All tests are exact two-sided McNemar tests on paired selection-panel cases,
final against final. The boss panel is reported beside them and is not used for
pass or fail.

- **Pairing.** Seed 1 of one side is paired with seed 1 of the other, and seed
  2 with seed 2. PPO r1 counts as seed 1 and r2 as seed 2. The single teacher
  run is paired with itself at each budget.
- **"Beats"** means both: each seed pair has more wins on the challenger's side,
  and the test on the two pairs' pooled discordant cases is significant.
- **Dropped cases.** A case censored or escaped on either side is dropped from
  that pair.
- **Significance levels**, with Bonferroni correction where a gate runs several
  tests:
  - G1: two tests (tree teacher at 1× and at 3× against the current search),
    0.025 each.
  - G2: three tests (S1, S2 and S3, each against PPO), 0.0167 each. The gate
    passes if any passes.
  - G3: two tests (S2 against S1, S3 against S1), 0.025 each.
  - G4: one test, 0.05.
- **Known limit.** The two seeds share the same 256 cases, so pooled cases are
  not independent. The both-seeds-agree condition is the guard, and the doc
  states the limit.

| Gate | Question | Pass condition | If it fails |
| --- | --- | --- | --- |
| G1 | Is the tree teacher better than the one-move search? | The tree teacher beats the current search at 1× or 3× the current CPU cost. | Continue with whichever teacher the selection rule in part 1 picks. |
| G2 | Does learning from search beat learning from win/loss? | At least one student architecture beats PPO. | Use the audit: is the student far below the consistency ceiling (a learning problem) or near it (a teacher problem)? |
| G3 | Does the transformer beat the current network on identical data? | S2 or S3 beats S1. | Keep the current network; it is cheaper. |
| G4 | Does starting PPO from distilled weights help? | The distilled-start arm beats the from-scratch baselines. | Report; distillation alone is the result. |

Effect sizes, per-stratum tables and HP kept are reported whatever the
verdicts. With 256 paired cases per seed, differences below about 4–5
percentage points will read as no difference; the doc says so.

## Gym feature audit

Before the teacher is built, one read-only pass over the Gym's v1.2.0
documentation lists every feature relevant to
search, labelling, batching and observation size, and says for each whether
Learn uses it. The documentation is read from the `source-ref` clone under
`artifacts/development/neural-combat/gym-1.2.0/`, not from the Gym checkout,
which stays at v1.1.1.

The batched search above was found this way, late. The re-pin already adopted
the compact view, combat-entry resources and the wider search roots. The
audit's output is a short committed note, and the plan is adjusted if it finds
more.

## Throughput

Measured on Gym v1.2.0 (`docs/neural-throughput-gym-1.2.0.md`). Training on the
compact view (bounds preset `standard-combat-compact-v1`) runs at:
- 222 decisions per second at 16 rows, against 163 for the dense view;
- 255 at 64 rows, against 172.

The v1.2.0 baselines train at 16 rows on the compact view. At 64 rows the
compact path's time splits as follows:

| Share of training time | What |
| ---: | --- |
| about 40% | PPO update on padded batches |
| about 34% | Per-observation schema validation and frame hashing in Python |
| about 10% | Gym step and export |
| about 9% | Policy forward |
| about 4% | Feature transfer |

The compact view roughly halved export, hashing and feature transfer, and cut
the PPO forward and backward passes by about a third. Validation fell least,
and it is now the largest single span. Both changes below are still justified,
and both are in scope, because the PPO arm and every later project use this
path:

1. **Trim padding in batches.** Build model inputs from occupied rows, padded
   only to the largest occupied count in the batch, not to the preset's
   capacities. This must be exact: the current network's outputs and a full
   training segment on CPU must be byte-identical before and after.
2. **Sampled validation during training.** *My call:* training validates and
   hashes one observation in 16 (the first of every rollout always); evaluation
   and all tests keep full validation. A failed sampled check is an error, as
   now.

Both are measured with the existing profiler before and after, and reported.
The Gym batch's `workers` setting (currently 1) is also measured at 1 and 8.
PPO hyperparameters, including minibatch size, are not changed here.

**Gym requests.** Learn does not modify the Gym. At the user's request, four
feature beads were filed in the Gym's tracker on 2026-11-21 (bodies under
`artifacts/development/neural-combat/gym-feature-issues-20261121/`). Their
status at v1.2.0:

| Request | Status |
| --- | --- |
| `spire-pqs`: compact combat observation for standalone combat and search branches | Delivered in v1.2.0. Training uses it; search branches are measured by this project. |
| `spire-e0l`: published per-character relic and potion lists | Delivered in v1.2.0 as the combat-entry resource descriptor (fingerprint `4eafca3b…`). |
| `spire-61l`: search roots where the Gym refused them | Stage 1 (held targets, identical-copy picks) was delivered in v1.2.0. Learn measured the coverage and posted it on the bead on 2026-11-23: roots at 91.96% of train decisions. Stage 2 (draw knowledge, 3.7% of decisions) and stage 3 (Runic Dome hidden intents, 4.3%) are open. |
| `spire-n6b`: a native playout call | Pending. The Gym waits for Learn's `step_many` measurements, which this project produces (part 1). |
| `spire-sjh`: a legal infinite turn ended in an untyped native fault (Learn's bug report, 2026-11-23) | Closed. v1.2.1 fixed the cause, Giant Head's Slow being capped at 999 stacks. v1.2.2 (`7d62f131`, rules ids Standard v9 and M10 v6) reports every engine limit that legal play can reach as a typed `bound` on `RunFaultError`. Until a re-pin, Learn censors a fight at 500 actions in one public turn. |

**Re-pin checkpoint.** Done. This project starts on v1.2.0 (rules epoch
`gym-1.2.0`), and the v1.2.0 baselines and search reference are its comparison
points.

A further Gym release (v1.2.2 already exists; later ones may bring
`spire-61l` stages 2 and 3, or `spire-n6b`)
gets its own rules epoch, so adopting it mid-project would orphan the labelled
dataset and its comparison points. The only point where this project may adopt
one is after the teacher audit and before labelling starts. Adopting it there
means re-running the two PPO baselines and the search reference first.
Otherwise the project finishes on v1.2.0.

## Compute

- **CPU.** Search labelling is CPU work, parallel across fights. *My call:* at
  most 24 of the 32 cores, at low priority, so other services keep headroom.
  There is no wall-time cap.
- **GPU.** Training and evaluation use the existing `gpu_wait` gate. Learn
  remains second to Hermes.
- Labelling and GPU training may run at the same time.

## Constraints

- AGENTS.md applies:
  - public observations only to networks;
  - Gym faults are errors;
  - no modification of the Gym checkout;
  - artifacts and Baalorlord-derived data stay out of Git;
  - no dependency installs without approval.
- Gym revision `4d198064` (v1.2.0), rules epoch `gym-1.2.0`, Standard rules,
  Ironclad, ascension 20, compact combat view `standard-combat-compact-v1`.
- Fights, teacher fights and PPO training use Learn's per-turn action budget
  (`max_turn_actions = 500`) while on v1.2.0. At a re-pin to v1.2.2 or later,
  the budget is reviewed: limits then arrive as typed `bound` faults, which
  remain errors.
  A fight that reaches it is censored in evaluation and truncated in training,
  never scored as a loss.
- Tests are deterministic and CPU-only, on synthetic fixtures. They cover:
  - tree statistics are keyed by public view, and two worlds with the same view
    share a node;
  - a refused root yields no label;
  - no privileged Gym surface is called by the teacher;
  - dataset determinism and train-split-only sourcing;
  - the validation runs are excluded from training shards;
  - token construction ignores padding rows and is invariant to row order
    within a group;
  - sparse dataset rows expand to the exact dense observation;
  - padding-trimmed batches give byte-identical outputs for the current
    network;
  - sampled validation still raises on a corrupted sampled observation;
  - only legal actions are scored;
  - checkpoint families do not cross-load;
  - the boss panel's quotas, determinism and independence from training-only
    recipe fields.
- Experiments record Gym revision and dirty state, recipe, table, panel and
  dataset hashes, seeds, budgets, CPU and GPU time, and GPU waits.
- M10 and existing Standard behaviour stay byte-identical where untouched.

## Risks

- **The teacher may be hard to imitate.** A noisy teacher gives noisy labels.
  The audit measures this before any student is trained.
- **Distillation is capped near the teacher's level** when playing by reflex.
  Exceeding the teacher is the next project's job.
- **Unlabelled positions.** About 9% of Ironclad train decisions have no
  search root. They are concentrated in acts 3 and 4, and in Runic Dome fights,
  where no decision has one. The students see no labels there; the PPO arm does
  train on them.
- **Per-fight results can mislead about full runs.** There is no full-run check
  in this project. HP kept is reported as a partial guard.
- **Selection-panel reuse.** The selection panel has now guided several
  decisions. The final panel stays unscored for a later clean test.

## Left to the builder

- Search constants (upper-confidence constant, depth limit, widening, world
  count), tuned on training-split fights.
- How labelling is parallelised across cores and containers.
- Dataset shard format and size within the bounds above.
- Token feature details and the exact form of the action query.
- Optimiser, learning-rate schedule and batch size for supervised training,
  identical across students.
- How the win-probability head seeds PPO's critic.
- Module layout.
