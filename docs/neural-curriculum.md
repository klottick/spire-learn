# Randomized Ironclad combat curriculum

The broad curriculum passed its bounded native/CUDA qualification, including
checkpoint continuation. Both the original campaign and the later campaign on
Gym's delivered fixes stopped on native standalone faults before the broad
learning comparison could complete. The latest campaign saved two randomized
checkpoints, but their evaluation never started. The results below separate
saved work from incomplete comparisons. The [qualification protocol](../scripts/neural/curriculum-qualification-runbook.md)
is historical preparation material; this document authorizes no retry or new
workload. The earlier sustained campaign's native Run fault remains a separate
unresolved result.

## 2026-11-01 campaign outcome

Campaign `curriculum-20261101-a`, executed from source `61c48e4`, stopped after
21 minutes 38 seconds with observed launcher exit 1. Its qualification completed
128 native actions, 128 samples, eight Adam steps and 14 resets within the
approved ceilings. Four fixed-control segments then published four checkpoints:
16,384 learned actions, 1,024 accepted Adam steps and 32 PPO updates.

The original-B and post-fixed checkpoints each completed the same 64-case CPU
development panel:

| Cohort | Cases | Original B wins | After fixed training wins |
| --- | ---: | ---: | ---: |
| Familiar, new seed | 16 | 9 | 11 |
| Novel deck | 16 | 8 | 8 |
| Reserved pair | 16 | 5 | 7 |
| Resource / later tier | 16 | 0 | 5 |
| **Total** | **64** | **22 (34.375%)** | **31 (48.4375%)** |

Ten cases changed from loss to win and one from win to loss. This is encouraging
development evidence from one paired replicate, with unknown historical parent
exposure; it does not establish seed-stable improvement or final performance.

The first randomized segment delivered 1,488 actions before the next B16 batch
faulted, leaving 16 submitted action slots uncertain. Its two PPO updates were
not checkpointed. Mixed training and six later panel opportunities (384 cases)
never started; these are unavailable results, not losses or zero scores. The
public standalone vector traceback reports a native `RunFaultError`, but the
internal cause and any relationship to the earlier Run fault remain unproven.

The retained audit reconciled source, pins, effects, time, storage, checkpoint
ancestry and all eight owned-container cleanup records. The last healthy saved
checkpoint belongs to `curriculum-fixed-003`. See the [detailed postmortem](../artifacts/development/neural-combat/curriculum-campaign-v1/audit/runtime-report.md)
and [independent retained-evidence review](../artifacts/development/neural-combat/curriculum-campaign-v1/audit/runtime-review.md).
No workload was replayed during the audit.

Broader training's benefit remains unanswered: the simulator fault stopped the
comparison. Gym-owned diagnosis and sufficient fault-operation evidence are
needed before another campaign. This result does not show that randomized decks
are the wrong learning approach. Saved checkpoints are retained; the consumed
campaign will not retry. Final evaluation is untouched, and RNG-001, TIME-001,
finite Ironclad A0 content and simulator-only transfer limits remain explicit.

## Fixed-Gym campaign c

Campaign `curriculum-20261101-c` ran from Learn source `9ad64019` against Gym
`a86df922` / native SHA256 `96d54621…`, after the separately recorded fixes to
the earlier incidents. Its launcher measured 1,507.798 seconds (about 25 minutes)
before failure; the primary later retrieved actual launcher exit 1. That retrieval
timestamp is an observation time, not the time the process exited.

All arms started from campaign a's healthy `curriculum-fixed-003` weights with
fresh optimizer, RNG and progress state. This is a new warm-start comparison,
not exact continuation of the earlier failed randomized job.

| Arm | Saved segments | Learned decisions | Accepted Adam steps | PPO updates |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 4 | 16,384 | 1,024 | 32 |
| Randomized | 2 | 8,192 | 512 | 16 |
| **Total** | **6** | **24,576** | **1,536** | **48** |

The same 64 development cases, lineages and environment seeds paired exactly:

| Checkpoint | Wins | Deaths |
| --- | ---: | ---: |
| Warm-start fixed-003 | 31/64 | 33 |
| After four new fixed segments | 35/64 | 29 |

Five cases changed to wins and one to a loss. This is a small improvement in one
paired development replicate, with unknown historical parent exposure. It does
not establish a reliable generalization gain. The two randomized checkpoints
were not evaluated; mixed training never began. Six planned panels, or 384 case
opportunities, remain unavailable rather than losses or zero scores.

The third randomized segment faulted after 160 accepted B16 calls / 2,560
delivered actions. The next 16 action slots remain uncertain. Its raw journals
also record 2,576 delivered samples, 160 accepted Adam steps and 124 resets;
this partial work produced no healthy receipt or checkpoint. Its full lease
remains charged, separate from the six saved segments. Prior campaign b's
qualification-only host-reconciliation failure and its charged lease remain
separate and unchanged.

The internal Gym cause and failing public case are unknown. P1 Gym issue
`spire-14g` tracks the incident; Learn issue `spire_learn-pa7` tracks the missing
pending-action/first-fault capture needed for stronger diagnosis. No replay or
retry was performed. A generic `RunFaultError` does not identify a shared cause
with the earlier defects.

The [final retained audit](../artifacts/development/neural-combat/curriculum-campaign-v3/audit/final-runtime-report.md)
and [independent review](../artifacts/development/neural-combat/curriculum-campaign-v3/audit/final-runtime-review.md)
reconcile the source, runtime pins, checkpoints, effects and cleanup for all ten
campaign containers. The review matched all 2,018 inventory entries and records
37 passing evidence checks. It clarifies the audit's timing wording: delayed
status retrieval does not establish clock drift or an execution overrun. No
post-run free-disk sample exists; retained output sizes and removal records are
reported separately.

The campaign is closed as failed and training is stopped. The saved randomized
work shows that broader training executed, but its learning benefit remains
unmeasured. It does not show that random decks are ineffective or that more
fixed training solves broad combat. Final evaluation remains untouched;
simulator-only, finite-content, RNG-001 and TIME-001 limits remain in force.
No further workload is authorized by this result report.

## Inputs and coverage

[recipe.json](../configs/neural/curriculum-v1/recipe.json) binds the canonical
support snapshot, generator version, seed, held-out pairs, lesson categories,
stage profiles and rarity weights. [support.json](../configs/neural/curriculum-v1/support.json)
records source support at Gym revision
`533dad1c8aec86df624b3742b6b3ad1dbf1d0033`: 75 Ironclad cards, 43 encounters,
30 relics and 13 potions. This is a source inventory, not installed-extension
qualification. Rarity and mechanic labels are learner sampling annotations;
Gym still owns validation, shuffle, enemies, game rules and native RNG.

The fixed arm retains the exact 78-case narrow control. Randomized uses only
run-like generation. Mixed generates approximately 70% run-like decks, 20%
lessons and 10% unusual combinations. At the authored 256-entry shard size,
each mixed shard has 179/51/26 entries. Catalogs are bounded at 256 entries and
1 MiB. An aligned four-shard cycle targets all 75 cards in base and upgraded
forms and all 43 encounters. The retained cycle has 1,024 generated entries
and 150 base/upgraded forms for each broad arm. Fixed has 312 entries per
cycle and covers seven cards and 13 encounters; it is explicitly exempt from
the broad coverage claim.

Run-like early/middle/developed decks range from 8–35 cards and perturb starter
removals, additions and upgrades with coarse encounter-stage affinity. Unusual
decks range from 5–40 cards. Public max HP ranges from 80–120 and current HP
from 24 through max HP. Public relics, counters and potion slots vary within
source support. Generated decks exclude statuses/curses and active Neow's
Lament. Status cards can still arise through Gym-owned gameplay effects.
These heuristics do not estimate the distribution of decks reached by a real
Run policy. Difficult/off-stage encounters, unusual compositions and authored
resources deliberately expand coverage.

Generated entries, native-normalized entries, accepted resets, actor samples
and completed natural fights are separate denominators. Four generated
manifests do not establish that their cases were presented to the policy.
Catalog construction can normalize 256 entries without resetting or stepping
any of them. Collector delivered-reset/action counters and retained case IDs
are the evidence for actual exposure. Sampler allocation counts record choices;
a failed reset can leave an allocation without accepted exposure.

## Sampling and public-information boundary

Mixed standalone training uses the actual `CurriculumSampler`. Family selection
follows a repeating 70/20/10 cycle, with missing families renormalized over the
represented choices. Lesson kinds follow 50% single cards, 30% pairs and 20%
mechanic groups. Within a represented kind, 50% of probability balances the
least allocated categories. After 32 global natural completions, 45% targets
categories with at least 32 outcomes and a recent victory rate from 20–80%;
before eligibility, that share also balances allocation. The final 5% is
uniform over represented categories. Outcome windows retain at most 128
natural outcomes. Counters saturate at 2**63−1, which becomes a lower bound.
Faults and censored fights never become losses or successful lessons.

Fixed and randomized retain the existing uniform case sampler. Evaluation
never adapts the training sampler. Recipe/family/difficulty IDs, category
statistics, seeds, hidden order and private diagnostics stay outside actor and
critic tensors. Model, features, PPO and reward remain unchanged. Resources
already visible through the accepted public observation can affect decisions.

## Preparation and continuation

The pure entry prepares reproducible files; choose an absent destination below
`artifacts/development/neural-combat` whose parent already exists:

```sh
PYTHONPATH=src python3.12 -B -m spire_learn.neural curriculum prepare --recipe configs/neural/curriculum-v1/recipe.json --destination artifacts/development/neural-combat/my-prepared-inputs --arm mixed --shard-index 0 --validate-only
```

Remove `--validate-only` to publish preparation files. This performs only
stdlib generation, parsing and file writes; it does not train. Preparation
writes the recipe/support, four manifests, current catalog/annotations,
development panel, pilot plan, actual config and an input inventory.
Validate-only writes nothing and explicitly does not claim an on-disk loader
roundtrip. Existing destinations and final-split preparation are rejected.
[`pilot.toml`](../configs/neural/curriculum-v1/pilot.toml) is a preparation
template containing placeholder paths. Use the real preparation output before
loading a train config. Its descriptive 16-Adam pilot budget is unapproved and
is different from the held qualification's eight-Adam shared ceiling.

The appended optional `scenario_stream` table binds recipe, arm, shard index
and distribution hash. Real TOML and case loaders compute actual config and
manifest identities; changing dataclass values while retaining old hashes is
not a valid configuration. Only standalone/train supports this stream.
Fixed legacy manifests retain strict exact-config and exact-manifest resume.

Warm start takes only original standalone B's weights, with fresh Adam,
behavior version, progress and four RNG streams. The future three-arm pilot
must use that same original parent and matched seeds/budgets for all arms.
The original Run B remains a frozen evaluation reference. Both exact
three-file parent inventories are retained in
[pilot-parents.json](../configs/neural/curriculum-v1/pilot-parents.json).
Historical parent exposure is unknown.

Exact stream resume restores the current healthy checkpoint's model, Adam
moments, behavior version, four RNG domains, sampler windows/counters,
collector seed/decision/episode ordinals and saved cumulative progress. It
admits only the same shard or its exactly derived successor with stable
training settings. Changed concrete config/manifest hashes are bound to the
actual destination. Checkpoints retain one-hop transition provenance rather
than a growing embedded ancestry. Environments start fresh; unfinished rollout
rows are discarded. Snapshot elapsed time excludes later serialization and
close time; an external whole-lifecycle budget must include that work.
`run_stream_segment` runs exactly one segment and offers a successor only after
actual checkpoint preflight, settled saved progress and healthy final lifecycle.
Its `max_updates` limit uses cumulative update high-water values on resume.
There is no campaign loop or retry after a simulator fault.

Deferred Minor: the last accepted stream shard is 16,777,214. Its metadata can
declare the generator's terminal shard 16,777,215, which stream config loading
rejects. Preparation marks `successor_config_eligible=false`; orchestration
offers no eligible successor at this boundary. Advancement is finite.

## Development evidence and limits

The emitted development panel has four 16-case cohorts: familiar composition
with new seeds, novel deck, reserved pair and resource/later-tier diagnostic.
Familiar/resource cohorts deliberately reuse training compositions. Novel and
reserved-pair compositions are disjoint from this experiment's training
partition; that says nothing about historical parent exposure. The existing
final split remains unchanged and unexecuted.

`summarize_development_reports(panel, reports)` consumes actual per-case
standalone `NeuralReport` objects and preserves scheduled/reported/started/
completed denominators, censoring, faults, missing fight metrics and actual HP
sample counts. Potion-use data is unavailable, with a null value and zero
samples. Small strata and worst-stratum rates remain diagnostics. Deferred
Minor: the direct API checks the manifest hash but does not authenticate a
separate mutable annotations mapping. Consume the verified immutable emitted
panel, bound in its preparation inventory, for trustworthy stratification.

Retained source tests cover generation/support, real config parsing, sampler
outcomes, checkpoint metadata/recovery, one-segment lifecycle and failures.
[The integration class](../tests/neural/test_curriculum.py) has a dedicated
`SPIRE_LEARN_RUN_NEURAL_CURRICULUM_INTEGRATION` flag, unset by default.
[Its pure harness test](../tests/unit/test_neural_curriculum_integration.py)
drives the same enabled method through actual producers with all six optional
packages blocked, substituting native/tensor boundaries. Synthetic outcomes
and fake Adam state are plumbing evidence only. Pure transcripts and generated
input/source inventories are retained under
`artifacts/development/neural-combat/curriculum-v1/task-5/`.

The completed bounded qualification establishes compatibility and exact
continuation plumbing within its tested scope. The interrupted campaign above
does not establish a replicated fixed/randomized/mixed learning comparison. Broad
combat support does not fix the native Run fault or replace the fixed
noncombat controller with a learned Run policy. Full-Run continuation depends
on separately scoped Gym fault investigation and approval. RNG-001 and
TIME-001 qualifications, final-evaluation separation and original-game-transfer
limits remain in force.
