# Combat learning investigation: 4 November 2026

This investigation separates campaign reliability from playing strength. The
latest campaign stopped because Docker creation timed out. Its completed
training also failed to demonstrate an improvement over its parent. Neither
finding establishes that PPO cannot learn Slay the Spire.

The recommendation is to make learning measurable and preserve the best-scoring
evaluated reference, then compare conservative PPO with a quality-controlled
public-information teacher. Test representation and curriculum changes
separately. Another uninterrupted day with the same settings would give more
experience, but would not explain the current result.

This document proposes research directions. It does not launch training, change
the learner, or authorize reuse of a consumed campaign. The investigation used
retained reports, checkpoint metadata, source inspection, and online primary
sources. No checkpoints were deserialized and no ML runtime was executed.

## What the experiments actually show

| Stage | New mixed training decisions | Development result | Interpretation |
| --- | ---: | --- | --- |
| Fresh-Gym mixed arm | 405,504 | 18/64 initially; 41/64 at its final development panel | Useful learning in one lineage; short fixed/randomized/mixed comparisons did not establish a superior curriculum. |
| Original twenty-hour campaign | 335,872 | Started from the 41/64 weights; healthy shard 337 later scored 35/64 | Weights-only warm start, with fresh Adam and RNG state. This differs from exact resume. |
| Continuation-3 | 425,984 | Parent 337: 35/64; evaluated shard 439: 32/64 | Exact continuation of model, Adam, RNG, sampler and progress; latest saved shard 442 was not evaluated. |

The last comparison has 32 wins shared by both policies, three parent-only
wins, no candidate-only wins, and 29 shared losses. The observed change is
−4.7 percentage points. With only three discordant pairs, an exact two-sided
paired sign/McNemar test gives **p = 0.25**. This is a warning worth investigating,
not a demonstrated population-wide decline or proof of catastrophic forgetting.
The cases are a repeated development panel, not independent training replicates.
Likewise, 41/64 is the best observed score in this lineage, not proof that its
policy is strongest on the wider game distribution. Preserve it as a reference,
without turning that development maximum into a final-performance claim.

Cross-campaign receipt checks confirm that **41 → 35 → 32 used the same 64
case IDs, environment seeds, config hashes and manifest hashes**, with deterministic
selection and the same Gym/common-view identities. The new twenty-hour baseline
also reproduced 41/64 from identical parent model bytes. From that baseline to
shard337, six previously won cases became losses and none improved; continuation
to shard439 lost another three without gaining a case. Thus nine of the original
41 wins were lost on this panel. This is an observed retention problem across
both continuation stages, although its mechanism and population generality are
unestablished. Resetting Adam/RNG cannot by itself explain the later exact-resume
losses. The [cross-campaign evidence](../artifacts/development/neural-combat/learning-investigation-20261104/metrics-evidence.json)
retains these comparisons; the earlier development maximum and repeated panel
use prevent treating them as a fresh, unbiased algorithm comparison.

The earlier gains on novel decks and reserved card pairs are evidence that
broad combat learning is possible in this setup. The game being large is not,
by itself, a reason to abandon realistic deck variety or introduce cards only
one at a time. Keep broad coverage and test whether targeted assistance improves
sample efficiency without erasing those gains.

| Same-case cohort, 16 fights each | Parent 337 wins | Candidate 439 wins |
| --- | ---: | ---: |
| Familiar composition, new seed | 12 | 10 |
| Novel deck | 11 | 11 |
| Reserved card pair | 7 | 7 |
| Resource / later-tier diagnostic | 5 | 4 |

The three lost fights were `lots_of_slimes`, `slime_boss`, and `four_shapes`.
Across all 64 fights, mean reported HP retention fell from 31.2% to 27.1%, while
mean action count rose from 20.47 to 21.77. These are descriptive diagnostics,
not an explanation of the mistakes. The reports identify public-frame hashes
but do not preserve enough model inputs/probabilities to reconstruct the exact
bad decisions. Recalculation confirmed identical case, seed, config and
manifest bindings and equal input inventories for this paired comparison;
the [paired-check record](../artifacts/development/neural-combat/learning-investigation-20261104/root-paired-check.json)
retains the case IDs and fight summaries.

Do not add all checkpoint counters as independent experience: continuation-3
inherits the original twenty-hour prefix. Shard 442 contains 761,856 decisions
in that counter lineage; the earlier warm-start model separately had 405,504
mixed decisions. Disposable qualification updates and failed-job leases are
separate accounting categories.

The [completed fresh-Gym results](neural-curriculum.md#fresh-gym-campaign-results-2026-11-03)
and [continuation-3 terminal audit](../artifacts/development/neural-combat/twenty-hour-20261103-continuation-3/monitor/terminal-audit-report.md)
retain the experiment identities, paired outcomes and limitations. The later
milestone panels and protected final split did not run. These results cover
standalone Ironclad A0 combat in the simulator, not learned deckbuilding,
full-Run victory, other characters, A20, or original-game transfer. RNG-001 and
TIME-001 still apply.

## Reliability findings and Beads

| Issue | Finding | Required repair |
| --- | --- | --- |
| **spire_learn-637**, P1 bug | Shard 443's Docker create exceeded the short command allowance; the retained command has a timeout, empty output, and no output truncation. | Use appropriate explicit finite lifecycle deadlines within job/campaign and cleanup reserves. Establish daemon/client latency causes where possible; do not remove deadlines or retry gameplay. |
| **spire_learn-38o**, P1 bug | Create timed out before returning a full ID, leaving `cid=None`; automatic cleanup skipped a container the daemon had actually created. | Reconcile ambiguous creation using exact ownership evidence and a full ID. Never adopt a name collision or remove by name alone. Preserve the primary failure and prove absence after cleanup. |
| **spire_learn-e6f**, P2 investigation | More mixed training did not improve the available development comparison. | Reconcile evidence, identify observability/design gaps, and propose discriminating comparisons. This document and the linked research reports are its deliverable. |
| **spire_learn-c0l**, P2 task | Logged PPO losses/entropy describe only the last minibatch; drift and critic-fit diagnostics are absent. | Record explicit, correctly weighted update diagnostics, with honest partial/zero-step boundaries. |
| **spire_learn-5wz**, P2 task | Latest resumable state and strongest evaluated candidate are different concepts. | Maintain separate provenance-bound resume and model-selection records; leave unevaluated/inconclusive candidates explicit. |

The relevant source is
[the bounded command helper](../scripts/neural/task9_study_controls.py) and
[the container transaction](../scripts/neural/sustained_controls.py).
The daemon's underlying latency cause remains unknown. Later exact-label
inspection found the leftover container created but never started; it was
removed and absence verified. That resolves the resource leak for this attempt,
without changing the failed attempt's historical accounting. Shard 443's full
lease remains reserved; missing receipts never mean zero effects.

The previous live-output accounting race and host-history dependency inside
thin-guest admission were different faults, already repaired before this run.
The latest evidence does not demonstrate another simulator fault.

## What the learner already does

The policy is a small typed feed-forward actor–critic. It scores legal action
candidates using public state and gathered source/target entities. It already
receives card damage, block and cost, enemy intents, powers, and known/missing
field masks. Numeric inputs use signed `log1p`; describing the model as using
only card IDs, or as blindly clipping all numeric inputs, would be incorrect.

PPO uses terminal combat rewards (+1 victory, −1 death), GAE, 512-row rollouts
(16 environments × 32 steps), four epochs, 64-row minibatches, and learning rate
0.0003. The current update event records policy loss, value loss, entropy and
step/version counts. It does not record approximate KL, clip fraction or critic
explained variance. Source inspection of task-boundary GAE did not establish a
reward leakage or reset-bootstrap bug. A source audit is not a numerical runtime
validation of every path.

A limited independent endpoint sample illustrates the metric limitation:

| Segment | Mean logged policy loss | Mean logged value loss | Mean logged entropy |
| --- | ---: | ---: | ---: |
| First resumed shard 339 | 0.0151 | 0.0306 | 0.6493 |
| Last saved shard 442 | −0.0219 | 0.0281 | 0.5593 |

Each cell averages eight **final-minibatch** scalars, one per PPO update; these
are not full-update estimates. Both event files matched their receipt hashes.
The sample does not show an obvious value-loss explosion. Lower entropy alone
cannot distinguish improved decisiveness from harmful narrowing, particularly
when visited states and legal-action counts differ. KL and critic calibration
cannot be recovered from these scalars. The
[endpoint sample](../artifacts/development/neural-combat/learning-investigation-20261104/root-metric-sample.json)
records the paths, hashes and ranges.

The sampler already does more than uniform randomization: its family cycle is
70% run-like, 20% lessons and 10% unusual entries, with lesson allocation balancing
and intermediate-success targeting. Proposed failure replay must therefore be
compared with this actual baseline. It should resample **training scenarios and
collect fresh on-policy trajectories**, not feed arbitrary old rollout rows back
into PPO.

Those mixture percentages describe reset allocations, not decision-row shares.
A difficult long fight contributes more decisions than an easy short one.
Delivered resets, natural settlements and accepted actions must be joined to
scenario annotations before claiming that training experience was 70/20/10.

## Ranked hypotheses, not established causes

1. **Policy drift is poorly observed and latest is confused with strongest.**
   Constant-rate continuation can change behavior without improving the target
   panel. Missing KL/clip diagnostics prevent attributing the change to excessive
   updates. Keep a training-resume checkpoint and a separate strongest evaluated
   candidate; preserve uncertainty and per-cohort scores when selecting the latter.
2. **The training objective and evaluation distribution need a closer match.**
   Natural episode outcomes, not raw sampled deck counts, determine what the
   policy learns. Measure accepted exposure, settlements, segment-cut rate,
   encounter/resource cohorts and lesson outcomes. More hard scenarios may simply
   produce more sparse losses; easier lessons may fail to transfer.
3. **Public semantics and interactions may be inefficiently represented.**
   The model already has meaningful numeric and relational features, but global
   mean/max pooling and learned identity embeddings may make long-tail mechanics
   harder to learn. Static card metadata is available at the pinned Gym release;
   it is not a complete rule interpreter. Metadata, attention or public-history
   recurrence are ablations to test, not proven necessities.
4. **Exploration may need a better starting signal.**
   A competent public-information teacher could expose useful action sequences
   that sparse reward seldom discovers. A weak heuristic could instead erase
   learned skills. Measure the teacher before imitating it.

Sparse reward and the per-action discount also deserve measurement. With
gamma 0.99, equal eventual outcomes reached after different numbers of card
decisions have different discounted targets. That is an objective choice, not
a demonstrated arithmetic bug. Record fight lengths, terminal-outcome coverage
and critic fit before proposing a different discount or a shaping reward.
The direct lambda trace ends at a 32-step rollout boundary; credit across
longer fights relies on critic bootstrapping. Sparse reward does not mean every
early decision lacks a learning signal.

## How the literature changes the options

| Option | Why it is credible | What could fail here |
| --- | --- | --- |
| **A. Instrumented, conservative PPO** | PPO implementation choices materially affect outcomes; retain the current successful baseline and change one optimization factor at a time. | Better diagnostics and a lower update rate may only reveal a representation or exploration limit. |
| **B. Public teacher → imitation → restrained PPO** | In one LoCM study, behavior-cloned initialization plus PPO beat random initialization at the largest tested deck pool: 54.2% versus 41.8%, averaged over five runs. | Fixed opponent/pool and early stopping limit transfer; our teacher must be competent on our information and task. Self-warm-start is not stronger-teacher imitation. |
| **C. Public-information search and expert iteration** | A LoCM preprint improved its network-only policy substantially with belief-sampled decision-time search. | It exceeded the old competition memory cap. Current pinned Gym lacks a supported public branching/belief API; implementing a second simulator in Learn is prohibited. |

The evidence supports these **comparisons**, not a guarantee that one transfers
to Spire. See [Engstrom et al.](https://arxiv.org/abs/2005.12729),
[Andrychowicz et al.](https://arxiv.org/abs/2006.05990),
[Haluška and Schmid, Table 1](https://ala2024.github.io/papers/ALA2024_paper_33.pdf),
and [Rubin's search study and limitations](https://arxiv.org/html/2609.06816).

In option B, behavior cloning means supervised training to predict a teacher's
legal move from the public position. PPO then learns from newly played fights
and their real outcomes. An optional imitation/KL anchor discourages large
departures from the reference policy, but can also preserve its mistakes.
Cloning our own 41/64 policy could help retain behavior; it supplies no new
expert knowledge by itself.

Semantic representations and focused mechanic curricula can complement A or B,
but should not be bundled into the first comparison. Existing card-game work
does not establish an optimal 70/20/10 mixture, one-new-card-at-a-time schedule,
or a general benefit from a larger network or dense damage reward. Human draft
prediction and full-game proprietary LLM performance answer different questions.
The [literature guide](learning-strategies.md) preserves those distinctions.

Successful card-game systems also often use much more simulation and training
compute than this project has used. Our results do not show that the learner
has reached its ultimate limit or that long training is unnecessary. The
purpose of short comparisons is to find a recipe worth scaling, rather than
to substitute a tiny experiment for eventual large-scale learning.
[ByteRL](https://arxiv.org/abs/2303.04096) is evidence for the value of scale and
careful game-specific design together, not a promise of comparable results
from one overnight GPU run.

## Recommended sequence of discriminating experiments

These are proposed experiments, not execution authority or a new GPU budget.

1. **Repair reliability and add learning diagnostics first.** Close the two P1
   orchestration bugs with fake-boundary tests. Record update-weighted KL/clip
   statistics, advantage/return scale, critic fit and actual accepted exposure.
   Separate last-minibatch metrics from full-update aggregates. Preserve the
   41/64 evaluated checkpoint, shard 337 and unevaluated shard 442 separately.
2. **Run a short PPO control comparison.** From the same authenticated parent
   weights, compare current settings with one conservative update change, such
   as a lower learning rate. Use the supported weights-only warm-start path for
   both conditions, with fresh Adam and paired RNG initialization in both. Exact
   resume requires matching PPO hashes; do not relabel a changed learning rate
   as exact resume. Use three paired training seeds and, as an initial screening
   budget, 65,536 new decisions per seed/condition. Match admitted scenario
   distributions, paired initial seed streams and effect ceilings; report actual
   delivered exposure, since policy-dependent episode lengths and adaptive
   sampling can diverge. Log wall time and evaluate at fixed decision intervals.
   A KL stop requires
   its own validated implementation and predeclared threshold. Do not change
   learning rate, reward, architecture and curriculum simultaneously.
3. **Qualify a teacher, then compare imitation plus PPO.** Evaluate its legal,
   public-information behavior against the frozen control by cohort. If it is
   not broadly stronger, use only independently validated tactical lessons or
   do not use it. Count teacher generation/search time and environment effects
   in the comparison budget. Demonstrations use a separate supervised objective;
   they are not fabricated on-policy PPO samples. Test a BC/KL anchor separately
   because it can preserve both good and bad teacher behavior.
4. **Ablate semantics and lesson selection independently.** Compare existing
   features with authenticated static metadata before increasing model size.
   Compare the current sampler with capped training-failure scenario replay,
   retaining broad realistic coverage. Never use development/final outcomes to
   steer the training sampler. Recompute feature/model identities for changed
   representations; do not silently relabel old checkpoints.

Training-scenario replay has a useful precedent in
[Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html),
which selects procedurally generated levels for their estimated learning
potential. Its results are from different games; it does not establish that
repeatedly selecting our hardest losses will help. A Spire comparison should
retain broad coverage, cap any replay share, and measure learning progress.

For each comparison, prerecord the primary win metric, cohort guardrails,
decision/compute budgets, evaluation intervals and selection rule. Report each
training seed and paired outcomes, not just the best checkpoint or a pooled
confidence interval treating all cases as independent training runs. A candidate
advances only after a replicated development gain or a compelling
sample-efficiency gain without a material cohort loss. Otherwise expand the
evaluation or reject the hypothesis rather than automatically train longer.
The protected final split remains untouched until selection is complete.
[Agarwal et al.](https://arxiv.org/abs/2108.13264) explain why few-run RL point
estimates alone are unreliable.

## Gym boundary

Options A, metadata features, training-scenario replay, and a public heuristic
teacher fit the existing pinned interface. A complete search teacher needs Gym's
authoritative branching/uncertainty support; arbitrary tactical mid-combat starts
also need a supported Gym interface. Existing Gym requests `spire-2o7` and
`spire-831` cover those gaps; static metadata request `spire-90f` is delivered.
This investigation creates no duplicate Gym issue or upstream change.

Deck acquisition remains a separate later policy: realistic supplied decks help
combat training, but do not teach reward selection, shopping or path planning.
For the eventual full-game objective, the natural extension is hierarchical:
the Run policy chooses acquisitions/routes from public deck and resource state,
while the combat policy executes fights. The combat pilot affects which decks
appear strong, so a deckbuilding optimizer must be tested with a stable pilot
and then with more than one pilot; otherwise it may specialize to its evaluator.
This is a future design direction, not an inference that our present generated
decks or 64-case combat scores measure deckbuilding ability.
See the [Gym capability assessment](gym-learning-capabilities.md).

## Detailed research and reproducibility

The local [investigation folder](../artifacts/development/neural-combat/learning-investigation-20261104/)
contains tracker creation records, the source audit, retained metric analysis,
targeted online verification and root source/evidence checks. Those generated
artifacts are ignored by Git. This maintained document carries the conclusions
and direct primary links without changing historical campaign records.

- [Source audit](../artifacts/development/neural-combat/learning-investigation-20261104/code-audit.md): reward/GAE boundaries, PPO cadence, representation, sampling and metric semantics.
- [Retained metrics audit](../artifacts/development/neural-combat/learning-investigation-20261104/metrics-audit.md): checkpoint/panel comparability, paired cases and observed training trends.
- [Targeted literature review](../artifacts/development/neural-combat/learning-investigation-20261104/literature-implications.md) and [verification registry](../artifacts/development/neural-combat/learning-investigation-20261104/literature-verification.json): 17 primary/source records, access grades and search notes.
