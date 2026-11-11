# Learning strategies: evidence and current direction

Literature survey: **2026-11-01**; local evidence updated **2026-11-04**. This guide summarizes an expanded online search
by three Luna/MAX researchers and primary-agent source checks. It explains the
evidence behind possible learner changes; it does not authorize new runs or
declare a winning architecture.

**Interface update, 2026-11-02:** Gym now delivers static public card metadata,
and the observation-shape freeze has been lifted. The release also reorders
categorical IDs, so existing checkpoints cannot be silently reused with it.
See the [updated capability assessment](gym-learning-capabilities.md) and
[migration record](neural-gym-migration.md).
These interface changes do not add evidence about training efficacy.

The [November 4 learning investigation](learning-investigation-20261104.md)
connects that literature to the subsequent training results and current source.
It recommends measured PPO controls before a teacher-assisted comparison, with
representation and curriculum changes tested separately. The latest campaign
stopped on a Learn Docker-creation timeout; that infrastructure failure is
separate from its disappointing development comparison.

The [expanded synthesis](../artifacts/development/neural-combat/literature-20261101/expanded/synthesis.md)
links the detailed reports, exact-query logs and screened-source registries for
general card-game play, deckbuilding and Slay the Spire. Those generated files
are retained locally outside Git. This guide keeps the central findings and
direct primary-source links in the maintained repository documentation.

## What we are trying today

The learner is a small, typed feed-forward **actor–critic**. The actor scores
currently legal candidate actions; the critic estimates future combat return.
Shared card/entity encodings and public references describe the current
decision. PPO updates the policy, using GAE to estimate which decisions were
better or worse than expected. Combat reward is +1 for victory, −1 for death,
and otherwise zero. Public legal masks prevent invalid action selection.

It supports standalone combat and combat inside a normal Run. Noncombat Run
choices still use a fixed controller: we are not yet learning card acquisition,
shopping or path selection. Generated training decks are supplied scenarios,
not decisions made by a learned deckbuilder.

The curriculum experiment compares a narrow fixed control with mostly
realistic randomized decks and a mixed distribution. The mixed design targets
70% realistic decks, 20% card/pair/mechanic lessons and 10% unusual combinations.
The completed fresh-Gym comparison arms started from the same fresh weights.
The later twenty-hour campaign warm-started the strongest evaluated compatible
weights with fresh optimizer, randomness and progress; continuation-3 instead
resumed its healthy shard337 state exactly. These are different treatments.
The ratio is an authored hypothesis,
not a literature-established optimum. See [curriculum details and outcomes](neural-curriculum.md).

## What the literature supports

| Approach | Evidence | Limitation and implication |
| --- | --- | --- |
| RL policies with structured legal actions | ByteRL won both LoCM competition tracks; DouZero achieved strong DouDizhu play using encoded actions and deep Monte Carlo learning. | Successful systems combine algorithms, representations and substantial experience. Their adversarial/team objectives differ from single-player survival. Keep our candidate-scoring baseline; these results do not identify a universal best algorithm. [ByteRL](https://arxiv.org/abs/2303.04096), [DouZero](https://proceedings.mlr.press/v139/zha21a.html) |
| Imitation followed by RL | A LoCM study's 1,024-deck-pool condition reached 54.2% against ByteRL with behavior-cloned initialization plus PPO, versus 41.8% with random initialization; five runs per condition. | Fixed opponent/pool, early stopping and development evaluation limit generalization claims. This is a strong reason to compare a quality-controlled teacher with exploration from scratch. [Haluška and Schmid, §5/Table 1](https://ala2024.github.io/papers/ALA2024_paper_33.pdf) |
| Contextual and semantic card representations | Cardsformer combines card text, transition prediction and learned play. Wins against lookahead fell from 78.0% on familiar cards to 47.5% with 30 unseen cards; component ablations supported its representations. | Restricted historical Hearthstone simulator and substantial training. It demonstrates useful but incomplete transfer, not arbitrary rule understanding or a need for a larger model here. [Cardsformer, §6](https://journals.sagepub.com/doi/pdf/10.3233/FAIA230581) |
| Focused combination starts | A Dominion SAC agent struggled to acquire enough engine cards despite being able to use them. Engine-like starts and other interventions helped expose the strategy. | Interventions were combined, and one engine run failed. This motivates playable mechanic lessons, not a claim that one-card-at-a-time training or our mixture is optimal. [Gerigk and Engels](https://jaspergerigk.com/uploads/final.pdf) |
| Search plus a learned policy | A LoCM preprint reports 26.8% wins against ByteRL without search versus 51.35% with sampled-belief search over 10,000 games. | Search exceeded the old competition memory cap. It trades simulator/inference cost for decision quality and must respect uncertainty rather than inspect the actual hidden world. [Rubin](https://arxiv.org/html/2609.06816) |
| Full-game pretrained LLM agents | Recent Spire benchmarks demonstrate full-game play. SpireClimb reports 20% A20 Heart wins for its leading model over 60 attempts across four characters. | Screenshots, wiki access, proprietary pretraining and expensive inference differ from training our compact policy. This suggests possible baselines/teachers, not an overnight training recipe. [SpireClimb methods](https://epoch.ai/benchmarks/spireclimb), [results](https://epoch.ai/publications/ai-systems-are-improving-rapidly-at-slay-the-spire-and-its-sequel) |

## What does not reliably work, or remains unproven

**A high training score need not mean transferable play.** One LoCM PPO study
reported about 76% against recent copies of itself but only about 51% against
max-attack and 36% against one-step lookahead. Training against one fixed
opponent also encouraged specialization. Spire has no learned adversary, so our
corresponding robustness tests concern unfamiliar decks, encounters, resources
and initial states. [Vieira et al., SBGames 2022, §V–VI](https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2022.pdf).

**More randomization is not automatically a curriculum.** Random opening
deckbuilding picks in one Hearthstone study changed deck style without raising
its measured win rate. A small LoCM thesis experiment found gradual deck-pool
expansion less efficient than its fixed-pool comparison; it is weak, unreplicated
evidence. Neither experiment tests randomized Spire combat decks. Focused starts,
random decks and incremental card introduction remain different treatments.
[Hearthstone random-CB ablation](https://arxiv.org/html/2303.05197),
[curriculum thesis, §5.3.2](https://dodo.is.cuni.cz/bitstream/handle/20.500.11956/193622/120485962.pdf?isAllowed=y&sequence=1).

**Dense reward can favor an unhelpful proxy.** Tested damage-based shaping in
LoCM did not consistently improve a win/loss PPO baseline; one variant was
worse during parts of training. Any shaping experiment here must still be
judged on unshaped combat outcomes, HP and held-out scenarios, rather than its
own reward alone. This does not rule out all shaping.
[Vieira et al., 2023, §5](https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/entcom-2023.pdf).

**A teacher must solve the same information problem.** Jialeiv's Spire project
reports unsuccessful attempts to replace combat MCTS with learned policies,
but says its teacher knows the actual future draw order. That is not a fair
public-state imitation target. Its published scores also predate a rules patch
and were not remeasured. These observations do not establish that PPO fails at
Spire combat. [Project limitations](https://github.com/Jialeiv/sts-rl-agent).

**Human-pick agreement, deck-search fitness and game wins are distinct.**
Contextual draft models can predict human choices on unseen expansions without
demonstrating stronger decks. A deck optimizer can favor combinations that its
particular pilot executes well. DSA-ME's 98.4% is an archive maximum against a
selected opponent suite, not average strength across decks or a new metagame.
[DraftFM](https://arxiv.org/html/2608.19568),
[DSA-ME, Table 1/§6](https://arxiv.org/html/2112.03534).

## How to interpret the newer Spire results

The first research pass missed several October 2026 reports. It should not be
used to claim that only hand-coded/search hybrids can complete the game.
SpireBench evaluates 20 LLMs on ten A20 Ironclad seeds each; its main table uses
renamed game terminology and reports 0–10% Heart wins. Preference Model reports
79% wins across five fixed A0/A1 scenarios and 20% in an added 50-run A20 panel.
SpireClimb instead uses screenshots, wiki access and a four-character panel.
These are not interchangeable win rates. Pretraining exposure and proprietary
training are unknown; renamed cards, fixed seeds, game modifications, difficulty,
character and inference budget all affect the measured task.
[SpireBench](https://arxiv.org/html/2610.12596),
[Preference Model](https://www.preferencemodel.com/blog/slay-the-spire/).

## Priorities suggested by the evidence

These are research directions, not approved implementation or execution:

1. **Make comparisons reliable.** The migrated Gym campaign completed its
   fixed/randomized/mixed development comparisons; later sustained runs exposed
   Learn orchestration failures. Repair the Docker timeout/ambiguous-create
   cleanup gaps and measure PPO drift before another sustained comparison.
   Retain submitted-action/first-fault evidence for future simulator faults.
   Operational failures remain separate from playing-strength outcomes.
2. **Retain PPO as a baseline and compare teacher-assisted learning.** Good
   public-information demonstrations could reduce exploration demands. Compare
   imitation-plus-PPO with PPO alone at matched environment and compute budgets.
   Warm-starting our own weights is not the same as learning from a stronger
   teacher. Teacher suitability and public-state access need their own checks.
3. **Compare broad exposure with broad exposure plus focused lessons.** Supply
   complete legal decks where the target combination can occur, then measure
   transfer to ordinary decks and unfamiliar pairings. Do not assume adding one
   card at a time is superior or that a higher variety count implies learning.
4. **Test representation changes before increasing model size.** Existing typed
   features already encode public context. Check irrelevant-order sensitivity
   and mechanic/context generalization; preserve meaningful order and references.
   Semantic pretraining, attention and recurrence are candidates for controlled
   comparisons, not required architecture changes.
5. **Keep acquisition learning separate.** A later card-reward/shop/path policy
   should condition choices on the current deck and resources and be evaluated
   through actual Run outcomes. Combat training on generated decks does not
   establish that such a policy exists.

Measure held-out wins with uncertainty, HP retention and results by scenario
stratum. Compare multiple training seeds where affordable. Track both simulator
decisions and wall time, especially when comparing search and policy-only play.
Keep model-selection and final cases separate.

Subsequent fresh-Gym mixed training saved 405,504 decisions and improved the
repeated development panel from 18/64 to 41/64. A weights-only continuation then
added 335,872 decisions; its healthy shard337 scored 35/64. Exact continuation
added another 425,984 decisions; evaluated shard439 scored 32/64, while saved
shard442 remains unevaluated. The latter comparison lost three previously won
cases and gained none; its small sample does not establish a general decline.
Cross-campaign receipts confirm the same case/seed/config/manifest bindings for
41 → 35 → 32: nine original wins were lost and none gained across those stages.
That is an observed development-panel retention problem, with its cause and
population generality still unresolved.
More experience therefore has not reliably produced a stronger evaluated
policy. These results neither prove mixed curricula superior nor show PPO is
incapable of learning. See the [investigation](learning-investigation-20261104.md)
for checkpoint boundaries, retained metrics, candidate redesigns and proposed
matched comparisons. Finite Ironclad A0, one training lineage, repeated
development selection, simulator-only transfer, RNG-001 and TIME-001 remain;
the protected final split is unrun.

## Search limits and Gym implications

The expanded records contain 85 contemporaneously logged query formulations
across three tracks, plus 80 explicitly reconstructed Spire formulations.
Screening records overlap across tracks and include implementation-only and
abstract-only sources. Some publisher and forward-citation access failed.
This is a structured, auditable review, not an exhaustive or replicated one.

See [Gym capability assessment](gym-learning-capabilities.md) for what the
adopted environment already supports, which strategies need a new public
interface, and which limitations belong to Learn rather than Gym.
