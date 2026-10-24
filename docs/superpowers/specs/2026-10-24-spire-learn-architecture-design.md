# Spire Learn architecture design

**Status:** Approved by the user after review and correction of six architectural
findings. Approval covers the master architecture and roadmap; implementation
execution is not authorized.

This is the master architecture and roadmap. Each implementation milestone needs
its own reviewed specification and plan later. It records decisions made for this
scaffold; it does not select package versions, promise capability not present in
Spire Gym, or authorize installs, training, data acquisition, commits, or release.

## Intent and boundaries

Optimize for exploration and learning: make strategies easy to try, inspect, and
change. Maximum win rate and heavyweight formal research are secondary, while
reproducible evidence remains important. Begin with individual combat; leave a
path to complete runs. The long-term objective is Heart victory for all four
standard characters at A0–A20, with Act 3, Heart, genuine death, and unfinished
outcomes reported separately. There is no present coverage or strength claim.

The currently documented Python gym slice is supplied-entry Ironclad A0, using
Strike, Defend, and Bash against Cultist/Jaw Worm. It is useful for integration
and learning mechanics, not broad gameplay conclusions. A previous read-only
source inspection observed gym HEAD `4874fb398128c142bd56f10a12e6fca939a88f93`
clean at that time. This is historical evidence, not proof of the installed
version or its present capabilities. Reverify the actual public API and revision
for every integration milestone.

Spire Gym owns authoritative simulation, rules, public observations, legal
actions/masks, rewards, endings, and reset behavior. Spire Learn owns policies,
models, collection, optimization, evaluation, experiment records, and learner
checkpoints. Do not implement a second simulator, patch a gym checkout, or infer
full-run support from planned native capabilities. Consume only public state and
legal actions. Seeds, hidden RNG, unseen draw order, private diagnostics, and
privileged replay data are not actor or critic inputs.

Use a single Python 3.12 project with replaceable modules, not separate services.
Local RTX 5090 is the preferred compute target, Colab second, other cloud third;
cloud scheduling infrastructure is not required. Smoke, development, and
overnight profiles are all useful. Each experiment must set and record explicit
budgets; numeric limits are calibrated per experiment, not fixed here. Do not
assume recurrent throughput or LLM fit from hardware alone.

## Dependency and integration direction

Gymnasium is the selected environment-interface direction; PyTorch is the
selected neural foundation direction. Neither has yet been installed, pinned, or
declared as a dependency. TorchRL is the first compatibility candidate, not an
adopted trainer. Do not force a training framework on LLM inference or evaluation.
Algorithm, model/backend, and version pins are decisions for milestone
investigations, not inherited historical PPO defaults.

Consume the existing public Gymnasium `CombatEnv` / `CombatVectorEnv` facade
(distribution/import names historically `spire-lab` / `spire_lab`), not a new
generic environment API or replacement simulator. The selected dependency
versions and adapters must be tested against the actual gym version. The copied
guide documents a tested Gymnasium 1.4.0 subset; do not silently upgrade it or
assume wrappers are compatible. In particular, the documented generic vector
observation wrapper is rejected.

For single environments, preserve the returned observation and end flags exactly.
For vector `SAME_STEP`, the step reward and termination/truncation flags describe
the old episode, while ended rows' main observations describe replacement
episodes. `final_obs` and `final_info` carry old final state. Collectors must
preserve row-specific final state and the old episode's appropriate recurrent
context before resetting ended-slot memory for the replacement episode. If a
truncation permits bootstrapping under the selected environment/learner contract,
final-state value evaluation must use that retained old context, never reset or
replacement-episode memory. Genuine termination suppresses bootstrap under the
selected learner target contract. Reset ended-slot memory before processing the
replacement observation. A normal collector cut is not an episode boundary:
retain episode identity and memory while execution continues. Fresh-episode
checkpoint recovery is a separate, explicitly recorded event. Termination,
truncation, and collector cuts remain distinct; this does not assert that the
current combat environment generates truncations.

Genuine game death is a valid loss. Simulator faults are errors, not losses,
truncations, or successful episodes; do not silently retry accepted effects.
Invalid agent/LLM responses and inference failures are distinct from simulator
faults and must not trigger silent random fallback. If a later experiment adds
bounded pre-submit correction, it must be explicit, configured, measured policy;
this design selects no implicit retries. External wall-time or step-budget stops
stop collection; they do not automatically truncate/reset an environment or turn
an incomplete episode into a loss. Handling continuation is a milestone decision.

## Proposed responsibilities

These are responsibility boundaries and suggested folder names, not a mandate to
create a universal plugin system. Dependencies flow from orchestration toward
the components it composes; lower-level models and policies do not construct
environments or start training at import time.

- **`envs`:** construct the existing public gym environments, expose verified
  public schema/capabilities, and validate integration. Gym remains authority for
  state, action, reward, and ending contracts.
- **`encoders`:** own versioned selection, grouping, normalization, embeddings,
  derived public features, and feature layouts for numeric/structured models or
  readable LLM views. Schema-derived lookup mappings are allowed. Gym remains
  authoritative for observation meanings, categorical vocabularies, legal actions,
  and gameplay effects; do not maintain competing environment definitions or
  shadow game rules. Encoder and action-mapping identities belong in dataset,
  model-export, and checkpoint compatibility metadata. Text rendering uses
  verified public metadata; missing effect descriptions are consumer capability
  gaps, not permission to approximate mechanics.
- **`models`:** feedforward and GRU models initially, transformer experiments
  later, and a separate inference boundary for LLM backends. Imports have no
  eager environment, device, or training effects. A trainable model is independent
  of its optimization objective.
- **`agents` / policies:** choose a current legal action from public observation
  and agent-owned state. Provide scripted/random choices, neural masked action
  distributions, and LLM response validation/history. Memory is per environment
  slot and reset at genuine episode boundaries, not collector cuts. In later
  full runs preserve memory across combat boundaries that are not episode ends.
  Candidate/action identifiers are decision-local: never reuse stale indices or
  assume stable card identity.
- **`collection`:** provide distinct sequential LLM and batched neural execution
  paths with shared environment/transition conventions, not one forced tensor
  loop. Collector owns reset/step and records exact observation, action, reward,
  termination, truncation, next state, public final-observation semantics, and
  row-specific boundaries. Preserve candidate masks both during selection and
  training/log-probability evaluation. On-policy extras such as log-probability,
  value, or recurrent state are algorithm-specific, not mandatory fields on every
  transition.
- **`training` / algorithms:** own behavior cloning (BC), provisional PPO/actor
  critic, and later alternatives: losses, optimizers, batches/sequences, and
  update requirements. Do not construct environments directly. A compatible
  numeric model should be reusable for BC and RL. Recurrent PPO needs intentional
  sequence-reset, burn-in, padding, and bootstrap correctness tests; GRU is not
  an effortless swap.
- **`evaluation`:** own frozen cases and run policies using collector/session
  components without depending on the RL trainer. Keep outcome, reliability, and
  compute measurements separate; use matched initial cases and never tune on the
  final set.
- **`data`:** keep versioned imported demonstrations, provenance, unknown and
  confidence flags, and episode/decision boundaries distinct from on-policy
  rollouts. An importer maps to gym schema/candidates only when compatibility is
  proven. Never fabricate legal masks, rewards, log-probabilities, or hidden
  state. Keep raw video outside environment observations; extraction is a
  separate workstream.
- **`experiments` / artifacts:** compose config-driven factories with explicit
  small contracts where real variation exists. No general plugin framework or
  cloud service is needed now. Keep artifacts relocatable rather than tied to
  absolute machine paths. Centralize experiment records, but namespace
  framework-specific checkpoint internals; do not invent a framework-neutral
  tensor universe.

### Minimum cross-component contracts

These semantic contracts define interchangeability without prescribing Python
signatures, tensor containers, or a second environment API. Version/fingerprint
the record contracts, encoders, and action mappings; reject incompatible imports,
exports, and checkpoints explicitly.

- **Decision request:** public pre-action observation, verified schema/action
  contract identity, and current legal candidates/mask. Slot, episode-start,
  and opaque decision-correlation metadata route execution but are not learned
  features. The agent owns the meaning of its opaque per-slot memory; the runner
  stores/passes it and signals lifecycle boundaries. LLM and neural memory need
  not share a storage format.
- **Decision result:** one choice from the current legal candidates, updated
  opaque agent state, and optional strategy-specific diagnostics. Correlate it
  with its request to reject stale results at the consumer boundary; do not
  invent native decision tokens as observations. Failures are explicit rather
  than fallback actions. Any configured pre-submit correction for one decision
  must not repeatedly advance history/recurrent state; its precise commit rule
  belongs in the agent/collector milestone design.
- **Collected transition:** source public observation and candidate mask,
  submitted action, reward, separate termination/truncation flags, old-episode
  successor/final observation, and a distinct replacement reset observation when
  supplied by `SAME_STEP`. Keep routing, episode-boundary, and collector-cut
  metadata separate from model inputs. Algorithm-specific extensions can retain
  behavior-policy identity, log-probabilities, values, and recurrent contexts;
  these are not fabricated or required for all imported examples. Required
  in-memory learning records are distinct from optional persisted debug traces.
- **Demonstration example:** causal public pre-action observation/history (which
  may be partial), semantic chosen action and verified candidate mapping where
  available, known/unknown/confidence and audit fields, and source/run/timestamp/
  extraction provenance. Include rewards, masks, or outcomes only when known;
  unknown is not zero or false. Keep later-frame/commentary labeling evidence
  outside policy inputs. Import validation determines which learning objectives
  an example can support; do not coerce incomplete examples into rollout records.

## Strategy sequence

1. **Random legal and simple scripted baselines.** Diagnostic references, not an
   elaborate solver or strength target.
2. **Prompted local instruction LLM.** Render readable public observations,
   present explicit legal candidate IDs, use bounded history, validate responses,
   and report prompt/history/model comparisons, tokens, latency, and memory.
   Fine-tuning is not a prerequisite. No model/backend is selected; measured
   hardware and schema compatibility are gates.
3. **Neural policy.** Start with feedforward masked candidate-scoring
   actor-critic, then test a GRU under comparable encoder, cases, reward, and
   budget. PPO is provisional, not established Slay the Spire best practice.
   GRU is the first recurrence candidate; LSTM may be considered later. Do not
   assume recurrence improves this narrow content.
4. **Demonstration learning.** Keep optional neural BC and LLM supervised
   fine-tuning (SFT) first-class. Assess teacher quality and exclude
   hidden-information teachers. Compare scratch training with imitation
   initialization. Transformer/hybrid strategies are later options after useful
   evidence; public-information-safe gym capability is required, with no hidden
   snapshot branching.

### Trainer compatibility gate

TorchRL provides candidate recurrent modules, collectors, masked categorical
distributions, and PPO loss building blocks, but integrated suitability is
unproven. Split investigation into two gates:

1. **Early adapter/trajectory probe, before freezing milestone 1 collection
   contracts:** a separately authorized non-training spike checks public Dict
   observation/mask mapping, single/manual and vector `SAME_STEP` final-state
   preservation, and per-slot lifecycle/recurrent-context routing. Use small
   deterministic fixtures and limited actual consumer integration as appropriate;
   no complete PPO trainer or long GPU experiment is required. Do not freeze a
   contract around a lossy adapter or one that conflates episode boundaries or
   weakens public-information isolation. Revise the adapter or use scoped PyTorch
   collection if necessary. Basic artifact/evaluation work need not wait on it.
2. **Trainer adoption gate in milestone 4:** verify masks during sampling and
   log-probability recomputation, recurrent targets/resets and sequence padding,
   model reuse for BC/RL, and checkpoint save/load semantics before adopting a
   trainer and beginning its training experiments.

If unsuitable, retain these module boundaries and use scoped PyTorch trainers
informed by established patterns. Custom recurrent optimization is substantial
work and requires its own design and tests. SB3 plus contrib is not the first
default: combined masked and recurrent PPO support is not established here.
Precisely, RecurrentPPO does support Dict observations via
`MultiInputLstmPolicy`; RecurrentPPO and MaskablePPO are separate features and
algorithms, not a combined guarantee.

## Experiments, evaluation, and artifacts

Training, model-selection, and final-evaluation cases are separate immutable,
versioned manifests. Harness test fixtures are separate from experiment final
evaluation. Evaluation seeds may be held by orchestration, never exposed to the
policy. Case counts and numeric budgets are approved per experiment; this design
claims no arbitrary statistical power. Matched initial cases do not guarantee
identical later randomness.

Demonstration partitions have their own source-lineage manifests, separate from
environment case manifests. Split by whole underlying run/recording: adjacent
decisions, linked segments, and duplicate or re-encoded excerpts stay in the same
partition. Preserve source/run identity through derivatives; conservatively group
uncertain lineage and do not claim a clean holdout when identity cannot be
established. Where feasible, creator/layout holdouts test extraction generality;
they are not requirements for the single-video exploratory pilot. Do not leak
final examples through prompt selection, demonstrations, fine-tuning, or memory
construction. Offline held-out action-prediction/extraction evaluation and live
gym gameplay evaluation answer different questions and are reported separately.

Account for every scheduled case execution, including repetitions, with an
orchestration execution identity that is not a model feature. Report planned,
started, and not-started counts, naming setup failures or cancellations. Among
started executions, use explicit execution statuses for terminal completion,
valid environment truncation (only if supported), budget-stopped unfinished,
agent failure, and simulator fault; distinguish any other execution error rather
than dropping it. Keep trustworthy observed game outcomes separate from execution
status, and define non-overlapping status counts in the evaluation schema.

Always show raw counts and denominators, including failed/incomplete executions.
If reporting `wins / (wins + deaths)`, label it as completed-game win rate and
show terminal-completion coverage among started cases alongside it. Future
terminal types need explicit outcome mapping. Faults are not gameplay losses;
repeated attempts must not erase aborted attempts or silently change the sample.
On broader tasks report Act 3 and Heart separately, not an unspecified win label.

Step lengths and HP, when available, are diagnostics, not extra reward. Measure
sample efficiency as environment interactions and/or demonstration counts.
Record wall time, update speed or steps/second as applicable, LLM latency/tokens,
and peak GPU memory when available; explicitly mark measurements not collected.
Cost matters but is not the sole ranking goal.

Record code and gym revisions plus dirty state; content/profile/schema
fingerprint/reward; model, tokenizer, backend, and quantization where relevant;
prompt and encoder versions; resolved configuration, seeds, budgets, hardware,
case manifest, and checkpoint identity. Do not log sensitive values. A possible
per-run layout is `config.yaml`, `manifest.json`, `metrics.jsonl`,
`episodes.jsonl`, `summary.json`, and optional traces/checkpoints. This is a
suggestion, not a serialization dependency or fixed filename contract. Small
authored configs/reports may be versioned; generated records, datasets, videos,
models, and checkpoints stay outside Git by default.

Distinguish policy export (weights plus architecture, encoder, action, and schema
metadata for evaluation) from a training checkpoint (learner weights, optimizer,
update counters, scheduler/config/provenance, and relevant learner RNG where
supported). Recovery may start fresh episodes and explicitly drop incompatible
or in-flight rollouts. Record that resume event; do not promise exact trajectory
or cross-machine bitwise continuation. Exact environment resume is out of initial
scope and depends on a public gym capability.

## One-VOD feasibility task

In parallel with learner work, plan one representative combat/reward-choice
inspection from the desired Baalorlord Unedited YouTube channel. Verify exact
channel/video, permissions, and an appropriate platform acquisition route in a
separately authorized task before downloading. No extractor is assumed to exist;
this document does not authorize acquisition. Keep footage/frames out of Git and
do not publish without authorization.

The pilot may try small manual, OCR, vision, or combined probes and produce a
few traceable annotated examples plus a feasibility report—not a production
pipeline or guaranteed dataset. Separate visible evidence from inference/unknown
and ambiguity. Future frames or commentary can assist offline labels but cannot
become earlier actor inputs; commentary is not necessarily true rationale. Do
not reconstruct hidden RNG, draw order, or simulator-complete states. Broad VOD
content may not fit the current gym slice; do not claim compatibility. One video
is neither a final benchmark nor generalization evidence. Full pipeline design
is intentionally deferred.

## Milestones and gates

Each milestone below needs its own specification and implementation plan,
acceptance criteria, and separately approved authority. No schedule or strength
claim is implied.

1. **Environment/evaluation/artifact foundation and baselines:** deterministic
   CPU contract/unit tests; actual public single and vector integration; fault,
   mask, and final-observation handling; inspectable configuration and case
   outcomes. Complete the early adapter/trajectory probe before freezing collector
   contracts. Basic artifact/evaluation work can proceed independently; this
   milestone does not require full trainer infrastructure.
2. **Prompted local LLM:** first measure hardware/backend feasibility, then use
   supported combats through legal choices. Save prompt/model configuration and
   invalid-output and latency data. Training is not required.
3. **One-VOD pilot:** may run alongside milestone 2; not a prerequisite for
   milestones 1, 2, or 4. Gate: a few audited examples and a feasibility report.
   Rights/acquisition authorization precedes download.
4. **Neural learning:** complete the trainer adoption checks on the early-tested
   collection contract, then a learnable feedforward baseline and controlled GRU
   comparison. Test masks, sequences, old-context final values, and recovery.
   Measure gym/learner throughput and approve budgets before training. If gym
   content is too narrow, report plumbing results and identify the missing
   consumer contract; do not copy rules.
5. **Demonstration learning:** once compatible audited data exists, separately
   scope BC and/or LLM SFT, compare with scratch learning on frozen
   selection/final cases. Other permitted, validated demonstrations can be used
   without a complete VOD pipeline.
6. **Broader combats/full runs:** proceed only as public gym interfaces become
   available; recheck schema and capabilities. Do not assume this follows from
   native Rust milestones.

## Verification expectations and non-goals

Fast deterministic CPU tests should cover arithmetic, masks (including padding
and no-legal-action edges), per-slot memory, sequence length/padding, episode
ends versus collector cuts, and bootstrap endings. Include a final-value test
where old and reset recurrent contexts yield different values, mixed ended and
ongoing slots, and fresh-recovery boundaries. Test record/encoder/action-mapping
compatibility, demonstration lineage/split validation, evaluation status/count/
denominator reconciliation, config/provenance round-trip, and checkpoint/schema
incompatibility. Scope reproducibility appropriately; do not promise cross-device
bitwise results. Actual gym, selected framework, and single/vector wrapper
behavior need separate explicit integration tests. General core tests should not
require ML/GPU; neural arithmetic can use CPU. GPU smoke, model-fit, and
learning-curve experiments require separate authority and are not the test suite.
An overnight run is not evidence that tests pass.

Out of scope: duplicate engine or gym edits; original-game runtime execution;
cloud/distributed scheduler; required dashboard/service; automatic full VOD
scraping; exact resume; license selection; and any install, training, commit, or
publication authority. Preserve the RNG-001 and TIME-001 qualifications: simulator
results are not proof of original-game transfer.

## Research context

Research claims and source links are collected in
[`docs/research/2026-10-24-prior-art.md`](../../research/2026-10-24-prior-art.md).
It is a review, not independently reproduced evidence. The copied gym reference
index at [`docs/reference/README.md`](../../reference/README.md) explains source
provenance and historical-design limits. Borrowed concepts inform experiments;
they do not commit this project to copied engine/code or historical architectures.
