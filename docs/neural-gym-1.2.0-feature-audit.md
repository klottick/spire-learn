# Gym v1.2.0 feature audit for the search teacher

This audit uses the pinned Gym v1.2.0 documentation snapshot at
`artifacts/development/neural-combat/gym-1.2.0/source-ref/docs/`, revision
`4d198064b3b568b6f0d08d6e3937d236fbc48e9b`. Stage 1 targets Standard rules,
Ironclad, ascension 20, rules epoch `gym-1.2.0`, and runtime profile
`configs/neural/gym-1.2.0/runtime.json`. The training layout is compact with
preset `standard-combat-compact-v1`. Adoption markers below distinguish
existing use from planned teacher work. This is a documentation audit.

## Search and labelling

| Feature and Gym doc section | Learn use |
| --- | --- |
| `branch(k)` and common random numbers, `standard-run-api.md`, “Public belief and search” | **used**. Equal roots and equal `(research_seed, k)` fix the same sampled world across candidate actions. A different root digest changes the stream. The one-move baseline also pairs its heuristic RNG by sample. Keep research seeds and sample bookkeeping in search provenance and out of network inputs. |
| `step_many` and `workers`, `batching.md` and `standard-run-api.md`, “Batched public search” | **used from Task 5**. Call `step_many(branches, tokens, actions, view="full")` on distinct live branches with current tokens. Results are returned in request order. Gym accepts 1–256 workers, with serial execution at 1 and a lazily shared pool above 1. Grouping and worker count preserve branch results, tokens and counters. The baseline uses sequential `branch.step`. |
| `view="mask"`, `standard-run-api.md`, “Public belief and search” and “Batched public search” | **not used, because the unchanged heuristic needs action rows and the nonterminal leaf score needs HP**. Mask mode skips full observation encoding and suits playouts that need only the mask. Fetching a full `decision()` each step would bring encoding back. |
| `fork()`, `standard-run-api.md`, “Public belief and search” | **used from Task 5**. Fork a live branch at a reusable prefix to copy its sampled world with fresh IDs and tokens. Forks inherit the step count and arena layout. Terminal or truncated branches cannot be forked. Gym measured 1.3–2.4 µs per fork in its release-build workloads. |
| Compact branch layout, `standard-run-api.md`, “Compact Standard combat view (M39b)”, and `releases/v1.2.0.md`, “What is new” | **used from Task 5**, subject to dense/compact agreement and cost measurement. Select `public_search(..., observation="compact", capacity_preset="standard-combat-compact-v1")` independently of the live session. Check `observation_spec()`. The documented array footprint is 229,926 bytes per slot versus 644,378 dense. Root digests, sample keys and action indices are layout-independent. Widen dtypes and zero-pad rows for dense-equivalent comparisons. |
| Smaller play preset and capacity overrides, `standard-run-api.md`, “Compact Standard combat view (M39b)” | **not used, because the stage pins standard-combat-compact-v1**. The `standard-combat-compact-play-v1` preset is 143,313 bytes per slot. Gym also supports row-capacity overrides. Compact admission validates the first full view. A committed publication overflow freezes the branch. Mask mode can bypass non-action successor overflow, so a mask-only pass cannot qualify full-view publication. |
| Four arena limits, `standard-run-api.md`, “Public belief and search” | **used**. The baseline supplies all four through `limits`. Set explicit limits for the teacher too. A missing limit defaults to its hard cap. |
| `counters()`, `limits()` and `spec()["root_digest"]`, `standard-run-api.md`, “Public belief and search” | **used from Task 5**. Reconcile live, created and closed branches and total steps against the search budget. Record the root digest, layout identity and resolved limits in provenance. Use the arena's retained-memory estimate to size concurrency. The estimate excludes caller-owned outputs, Python wrappers and transient worker/encoder memory. Node keys use public combat views. |
| Root and branch refusals, `standard-run-api.md`, “Public belief and search”, and `releases/v1.2.0.md`, “Known differences and limits” | **used**. The baseline counts root unavailability and falls back to the heuristic. Stage 1 reports root refusals by reason and omits labels for those decisions. Count an in-search refusal against the affected simulation. |
| Nested `branch.public_search(...)`, `standard-run-api.md`, “Public belief and search” | **not used, because the teacher carries each sampled world through its simulation**. A nested root re-determinises and starts separate limits and counters. |
| Combat terminal reward and cleanup HP, `standard-run-api.md`, “Public belief and search” and “Standard combat view and sessions” | **used**. Score terminal victory +1, death −1 and escape 0. The unchanged heuristic estimates nonterminal leaves with half the HP balance. Branches end at combat cleanup before settlement. Keep the 1,024-decision fight cap and `max_turn_actions = 500` distinct from search-arena budgets. |
| Exact checkpoints and records, `standard-run-api.md`, “Exact checkpoints (privileged)” and “Records and semantic replay” | **not used, because these surfaces expose privileged state or seeds**. The teacher receives a public search root from the prepared combat and networks receive public observations and legal-action masks. |

| PublicSearch limit | Hard cap |
| --- | ---: |
| `max_live_branches` | 4,096 |
| `max_branches` | 1,048,576 |
| `max_steps_per_branch` | 100,000 |
| `max_total_steps` | 100,000,000 |

Creating past a branch limit or stepping past the total limit raises
`RunBudgetError` with no effect. Reaching the per-branch limit truncates that
branch with `step_limit`. Close branches promptly to free live slots. Closing
does not replenish the cumulative creation budget.

### Root availability and belief limits L1–L6

Gym documents L1–L6 as belief limitations. L1 and L2 directly reject roots.
The other limits describe what a supported search can represent.

- **L1** refuses a root when exact draw knowledge exceeds its component or
  2 MiB byte bound, or after a grid pick that leaves knowledge unrepresentable
  until the next shuffle.
- **L2** refuses hidden enemy intents under Runic Dome.
- **L3** exposes sampled worlds only, with no chance-node enumeration or
  probability API.
- **L4** restricts the horizon to the current combat and excludes settlement.
- **L5** uses setup pre-roll laws narrowed by published intent values at
  decision boundaries. Executed damage does not narrow the sets.
- **L6** conditions on public history finer than the encoded draw rows. The
  consistent sets stay internal to Gym.

Root construction also checks lifecycle and profile, consistent pre-roll sets
and the continuation allow-list. In v1.2.0, nested plays and repeats carrying a
player-chosen target can admit a root. Randomly targeted autoplay and repeats
of those plays, held random bounces or orb rolls, unoffered generated cards,
Alchemize proposals, pending enemy spawn or replacement, and the thief's
pending rolled branch can still prevent admission. A branch can separately
refuse `HIDDEN_ORDER_PICK` without mutation. Gym supports exact single-pick
grids for wholly unpublished exchangeable copies under its label-free rule.
Two-pick Seek+, Frozen Eye and other unsupported mixtures remain outside that
rule where the candidate copies are not all published.

## Entry construction and vector batching

| Feature and Gym doc section | Learn use |
| --- | --- |
| `card_metadata()`, `card-metadata.md`, “Access”, “Sources and identity” and “Semantics and limits” | **used** for static name/catalog tooling and fingerprint checks through `spire_lab.card_metadata.load()`. The native entry point is `spire_lab._native.card_metadata()`. Static faces describe printed values. Use live public action rows and masks for current costs, targets and legality, and the session spec for profile admission. |
| Combat-entry resource descriptor, `combat-entry-resources.md`, opening contract and “Resource fields” | **used** through the pinned descriptor snapshot for Standard entry construction. Gym exposes it via `spire_lab.combat_entry_resources.load()`. Validate counter intervals, bottle links, relic exclusivity, potion-slot allowances and bounds against the descriptor. Match profile/catalog schema and pin descriptor schema and fingerprint. Native construction validates the complete combination and startup effects. Reachable recipes take resources from their validated prefix. |
| Vector `reset_mask`, `batching.md`, “Reset and autoreset”, `combat-training.md`, “Native batch and final ownership”, and `combat-environments.md`, “Reset, seeds and legal actions” | **used** through Learn's vector collector/driver. Initialize all slots on the first reset. Later vector resets accept a NumPy bool[B] mask and preserve unselected boundaries and tokens, with full-width results. Respect terminal ownership and next-step autoreset rows. This facility applies to episode slots. Search branches are created, forked and closed through their arena. |
| Native vector worker pools and fault status, `batching.md`, “Slots, seeds and workers” and “Faults, recovery and export”, and `combat-training.md`, “Native batch and final ownership” | **used** for vector collection. Per-slot status identifies what happened after faults. Committed-result recovery is **not used, because stage 1 surfaces simulator and delivery faults as errors**. Treat delivery and simulator faults as errors and never resubmit a possibly committed effect. Search uses its own worker and fault contract. |
| Standard Run compact observation, `standard-run-api.md`, “Compact observation (M38e)”, and `performance.md`, “A20 Standard checks after M38e” | **not used, because the teacher consumes standalone Standard combat views**. The upstream Run measurements motivate measuring encoding and worker scaling. They do not establish throughput for this combat teacher. |

## Adjustments to the plan

- In Task 5, make whole-call preflight refusal handling explicit. A
  `HIDDEN_ORDER_PICK` can reject every request in a batch before dispatch, and
  the error carries no offending request index. Use an accounted split or
  singleton dispatch only after identifying this no-effect refusal. Count the
  refused simulation once and preserve unaffected simulations. For successful
  dispatch, inspect per-entry fault details before consuming observations or
  masks. A fault row can carry zero-filled output. Close the search and surface
  the error without replaying any possibly accepted action.
- In Task 5, keep full views for the unchanged heuristic and qualify dense
  against compact using normalized public views, decision-local action
  alignment, values and outcomes at fixed worlds before comparing cost.
  Measure worker scaling and Python round-trip cost on the teacher workload
  before setting the Task 6 and 7 budgets.
- State the fault contract by pin. Use v1.2.0's documented fault classes and
  typed capacity details. v1.2.2 typed `bound` faults are not adopted; relevant at the re-pin.
