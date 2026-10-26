# Ironclad combat learning in standalone and normal-run modes

**Status: PROPOSED — preparation, not implementation or training authority.**
The user selected both modes and requested Gym compatibility ticket
`spire-dxo.66`. It is now accepted and merged at
`533dad1c8aec86df624b3742b6b3ad1dbf1d0033` (PR #47), following the requested
parallel senior execution and coordinated M11 integration. Learn has audited
source/package bytes; no new runtime has been installed or exercised here.
This advances the neural track ahead of LLM/VOD work; neither is a prerequisite.
M1 remains accepted and its frozen contracts/evidence remain unchanged.

## Intent and scope

Build a small, inspectable Ironclad A0 neural combat learner, optimized for
experimentation and understanding. One combat-policy architecture and explicit
encoder contract should work with both standalone fights and combats reached
through normal runs. Do not claim complete standard pools, original-game parity,
playing strength or generalization from a small learning smoke.

User choices: both modes; Gym owns environment changes; preferred executor is
GPT-6.1 Sol/high with GPT-6 Luna/max assistants; setup stays in this repository.
Proposals below are the parent's design choices, not additional user decisions.

Initial proposal: a feedforward masked actor-critic, PPO as the first algorithm
candidate, combat-local objective in both modes, and a recorded fixed controller
outside combat. GRU comparison follows a credible feedforward learning result;
LLMs, demonstrations and a learned full-run controller remain later tracks.

## Approaches considered

| Approach | Benefit | Limitation / disposition |
| --- | --- | --- |
| Standalone combats only | Controlled curricula and cheap resets | Excludes in-run behavior; user wants both |
| Normal runs only, fixed noncombat controller | Uses existing broader RunEnv progression | Biased reached states and slower curriculum control; insufficient alone |
| Shared combat policy, two explicit environment/collection modes | Supports controlled learning and in-run use without duplicate models or simulator rules | Selected direction from user's both-modes instruction; accepted common Gym view now available |

## Current evidence and upstream boundary

M1 source `732b714` passed 267 core tests with 9 gated skips; the accepted native
evaluator executed one test with no skips. Its two starter combats are interface
evidence, not a broad training corpus. Accepted Gym M10 source inspected at
`36fbdc6310945caa194f2aa806c37769275228c3` includes the broader Ironclad A0
engine profile. Gym66 and its M11 prerequisite are now accepted; this does not
change the older installed M1 runtime or establish Learn training compatibility.

`ArrayValue` and numeric validation already support int64. However, M1's schema
reader and candidate extraction expect combat-v1's five action tensors. M10 Run
uses grouped value/known/occupied tensors and action rows with typed group/row
references. `RunEnv.reset` has no scenario/resource override, and its termination
means run termination. Do not alias these contracts or merely widen schema IDs.
See the read-only audit in `artifacts/development/neural-combat/schema-audit.md`.

Gym controller `d5b72748-ea4e-4d34-b6a0-4b7b54dfafc5` delivered the accepted
READY callback for `spire-dxo.66`. Learn independently verified all 636 accepted
source hashes/modes, the delivered wheel/native hashes and all 22 packaged
Python/stub files against accepted Git. The native build label still identifies
its original dirty build context; accepted source and built-byte provenance are
recorded separately in `artifacts/development/neural-combat/accepted-gym66-audit.json`.
Upstream retains one historical Rust failure and 60 baseline-matched legacy
comparison failures; no entirely green suite or original-game parity is claimed.
The request is `docs/requests/2026-10-26-ironclad-combat-training-gym-request.md`.

Accepted public guide: `docs/combat-training.md` in Gym at the revision above.
Exports are from `spire_lab.gymnasium`:

| Mode / identity | Accepted contract |
| --- | --- |
| Standalone | `IroncladCombatEnv(entry_catalog, partition="train")`; reset with `seed` and `options={"entry_id": ...}` |
| Standalone vector | `IroncladCombatVectorEnv(num_envs, entry_catalog, partition="train")`; B=1..256, explicit masked reset, DISABLED autoreset |
| Normal Run | `RunEnv("ironclad", 0, profile="m10-ironclad-policy", partition=...)`; `combat_view()`, `combat_observation_spec()`, `environment_spec()` |
| Common view | `spire-gym-combat-policy-view-v1`; fingerprint `02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78` |
| Normal policy transport | `M10IroncladPolicy`, observation-v3 / replay-v3; fingerprint `30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8` |
| Standalone identities | profile `ironclad-combat-v1`; content `spire-gym-ironclad-combat-content-v1`; rules `spire-gym-native-m10-rules-v1`; reward `spire-gym-combat-outcome-reward-v1` |

The common descriptor has int64 values and int8 known/occupied masks, typed row
references and 1,024 action rows. It includes the public `lose_strength` power.
Use exact descriptor columns/vocabularies, not the old full-Run audit's dimensions.
Original action-row indices remain dispatch indices. Frozen combat-v1, M9 and
M10-v2 remain distinct; no silent profile substitution. Finite standalone support
is 75 Red cards, 43 native setups, 30 relics and 13 potions; bind exact supported
IDs/limits from the public catalog, not registry rows or aggregate counts.

Gym owns valid entry construction, shuffle, startup hooks, encounters, game
rules/RNG, public views/masks, outcomes and native batch/fault behavior. Learn owns
encoders, policies, task objectives, curricula, learning, evaluation and checkpoints.
Request public contract identities for provenance; do not obtain them through
privileged reproduction records. No private snapshots or a second simulator.

## Two modes, distinct lifecycle

### Standalone

A versioned case refers to a Gym-validated entry/catalog configuration and explicit
independent environment/policy seeds. Gym decides legal initialization. The combat
policy handles all declared combat actions, including selection and potion choices.
A single facade resets only when collection explicitly requests it. A native batch
uses only its documented reset/autoreset/final-state protocol. Natural combat end
is also the standalone environment episode end.

For the accepted vector, a terminal row rejects the entire step before dispatch.
Record old terminal results, then masked-reset finished rows before the next
full-batch step. Unselected ongoing rows keep exact state/token and do not advance.
There are no SAME_STEP replacements or inactive action placeholders. Input errors
are recoverable; accepted runtime/export faults freeze the owner and prohibit retry.

### Combat learning inside normal runs

A versioned Run case names character, ascension, profile, partition, supported
public constructor options and seed; it does not pretend to accept combat-v1
scenario reset options. A separate fixed public-information controller handles
map, reward, shop and other noncombat decisions. Its identity/configuration/RNG
are recorded independently from the learned combat policy. Proposed initial
noncombat controller is seeded legal-random, frozen for each comparison; its RNG
seed is explicit and separate from environment and combat-policy seeds. It is a
reproducible reference, not a claim of competent run strategy.

Gym's public phase and selection ownership determine routing. Never classify all
selection screens as combat or infer victory from a phase change/positive HP.
Use explicit accepted public lifecycle/outcome facts. Every environment call is
accounted for, including fixed-controller work and its effects on later state.
Only learned combat decisions receive actor-loss/log-probability attribution.

The sidecar routes `combat` and `combat_selection` to the learned policy,
including card/potion-owned selectors; `noncombat` has an empty common legal
mask. Dispatch through the original Run action index. Observe settled-count
increments once, paired with sticky `last_exit` visit/outcome and HP before
healing/after cleanup. Snapshot current numeric deck/items/HP at that increment,
before later reward/map actions; those arrays are not historical checkpoints.
Keep routing, tokens, visits and seeds outside actor/critic features. No Python
full-Run vector facade is included; initial in-run collection is explicitly single.

Proposed initial objective is local to each fight: a natural combat end closes a
learner task segment even when the Run continues. Preserve original Gym terminal
and truncation flags unchanged; add a separate learner task-boundary flag in the
neural rollout representation. Never reset the Run on combat victory or count a
combat victory as Heart/run victory. Runs and their contained combats have distinct
routing identities and accounting. The feedforward policy initially has no state.
Future GRU work must explicitly handle run memory and noncombat observation gaps;
no automatic recurrent reset at a continuing Run's combat boundary is selected.

## Policy, encoder and compatibility

Use descriptor-driven public feature selection, with separate typed handling for
continuous quantities, categorical embeddings, known bits, occupied rows and row
references. Unknown is not known zero; padding is not a real card/enemy/candidate.
No hidden RNG, seeds, routing IDs, replay records or unseen draw order enter actor
or critic features. Public derived ratios/counts may be selected and versioned;
do not predict effects by copying simulator rules.

Proposed first model: small feedforward row encoders, masked pooling for public
card/enemy/power groups, a global context embedding, one shared candidate scorer
conditioned on source/target public features, and a value head. Do not flatten
all large padded Run arrays into a fixed dense input or learn absolute action-slot
semantics. Widths and embedding sizes are selected only after inspecting delivered
schemas and a small memory/throughput calibration.

Sampling and PPO recomputation use the exact stored current legal mask and
candidate mapping. Invalid empty nonterminal masks fail; terminal observations
are not acted on. Evaluation sampling mode and tie behavior are explicit.
Preserve M1 baselines; reuse only where their menu contract applies. Add an
explicitly named selection/potion-complete reference baseline if needed, rather
than silently making old play-first repair unsupported menus.

Share architecture and an explicit learner feature contract, not an assumption
that both raw Gym descriptors are identical. Each mode needs a verified mapping.
Checkpoint loading requires compatible model, encoder, action mapping, accepted
profile/schema mappings and objective IDs. A matching character name is insufficient.

## Weight ownership and cross-mode experiments

Parent proposal after builder check: share the architecture and verified feature/
checkpoint contract, not a live optimizer across modes. Each training run selects
one mode and owns its weights, optimizer, RNG streams and data. Initially train
standalone and in-run modes as separate experiments, from scratch by default.
Both configurations must support actual bounded training, not only evaluation.

A frozen standalone checkpoint can be evaluated through the compatible in-run
adapter, and vice versa, to measure transfer without updating it. Optional warm
start is explicit: record the parent checkpoint and transferred model weights,
start a fresh optimizer/RNG state and mark it as initialization rather than exact
resume. Training resume is a separate operation with its checkpoint-state rules.
No checkpoint is assumed compatible until both mode mappings are verified.

Joint pooling/mixed-mode updates are outside the initial implementation: that
would require explicit sampling ratios, balancing and on-policy version rules.
This is a proposed experimental default, not a claim the user chose separate
weights or that either distribution will generalize to the other.

## Reward, returns and rollout correctness

Proposed first task reward: +1 for a verified public combat victory, -1 for public
combat death, 0 otherwise. No HP/damage/turn shaping initially. HP before/after
cleanup, turns and actions are diagnostics; baseline ceiling/floor must be checked
on a sufficiently varied curriculum. Store native environment rewards separately:
Run floor/key rewards must not silently become combat-policy reward. A Run
ending outside an active combat closes the Run without inventing a combat loss
or retroactively penalizing an already closed combat. Any additional natural
combat ending exposed by the delivered profile needs an explicit objective
mapping before that profile is accepted; unknown outcomes are not coerced into
victory/death.

Neural rollout data wraps or references the existing public transition. It adds
policy version, stored action log-probability, value, learner-task reward/objective,
task boundary and training masks. Do not overload M1 `Transition.terminated` or
serialize opaque policy state through the generic JSON codec. Cut/buffer metadata
can be separate; a new field on every core record is not presumed necessary.

For the proposed local objective, both native environment termination and natural
combat-task termination suppress value bootstrap. For an ongoing rollout cut,
bootstrap from its actual successor and preserve lifecycle state. If a delivered
profile supports valid environment truncation, bootstrap from its old final state
under the declared contract while stopping episode recursion. Never use an
autoreset replacement as an old episode's successor/value input.

Define bootstrap mask b and continuation mask c explicitly:
- delta = task_reward + gamma * b * next_value - value
- advantage = delta + gamma * lambda * c * next_advantage

A natural combat-task end or true environment termination has b=0 and c=0. A rollout horizon cut has a valid bootstrap
but no recursion past the recorded horizon. Supported truncation uses the verified
old final with c=0. Fixed noncombat actions outside a closed task segment retain
native provenance but receive no combat actor loss. A future run-return objective
would need an explicit new contract for intervening rewards/discount time; it is
not smuggled into this initial combat objective.

TorchRL passed the earlier trajectory adaptation gate, not trainer adoption for
these new modes. Verify masked ratios, GAE, value targets, batching and recovery
before adopting its PPO components. Keep a small direct PyTorch implementation
an alternative if actual trainer integration is unsuitable; do not select two
trainers or implement recurrent optimization in parallel.

Mode-specific configuration and manifests must express their actual constructor,
entry and seed domains. Keep M1 case/config contracts intact; do not pass a
scenario override to RunEnv or truncate seeds to fit a different profile. Neural
CLI/config validation stays dependency-free; model/device initialization remains
behind explicit neural execution. Exact new record fields belong to the approved
implementation plan after the upstream handshake.

## Experiments and evaluation

Start with a small valid A0 curriculum and widen encounters/deck compositions,
HP and item contexts gradually. Vary environment seeds and independent learner
seeds. Gym owns generation; Learn owns sampling priorities and budgets. Do not
use privileged snapshot branching or seed search to manufacture successful tests.

Keep train/selection/final manifests separate. All fragments of an underlying Run
stay in one split; a combat seen in a training Run cannot reappear as a claimed
independent held-out case. Track entry lineage when Gym supports derived cases,
and reject unclear cross-split collisions conservatively. Future final evaluation
remains explicit opt-in and is not used for curriculum or hyperparameter tuning.

Report standalone combat outcomes and in-run combat outcomes separately, alongside
full-run outcomes/progression and the fixed controller identity. Show denominators,
coverage, faults, HP retention, actions/turns and sample/wall efficiency. Compare
matched legal-random/reference policies. A small overfit/calibration experiment
establishes plumbing only; a broader frozen selection set is needed to assess
learning. Stop at approved budgets and report inconclusive results honestly.

## Runtime, exports and recovery

All learner-owned environments, caches, data and outputs stay under this repo;
Gym is built from an accepted revision using read-only inputs, never a live dirty
M11 checkout or borrowed Gym venv/artifact. Preserve M1's qualified CPU runtime.
Use a separate scoped neural runtime when dependencies are selected and locked.

Read-only device inventory found RTX5090,32607MiB,driver595.91.07. This is not
Torch/CUDA compatibility or free-memory evidence. Verify the GPU runtime and measure CPU simulation,
transfer, model/update time and memory before setting batch sizes or run budgets.
Preferred compute remains local5090, thenColab, thenothercloud. No GPU workload,
install, training budget or cloud setup is authorized by this proposed design.

Add minimal policy export and training checkpoint support when implementing the
learner: named tensor state plus explicit versioned JSON metadata, optimizer/update
state where needed, and separately recorded RNG state. No arbitrary object loading
or opaque-state serialization through core artifacts. Recovery starts fresh
accepted environment episodes, drops in-flight on-policy data and records that
fact; exact environment or cross-machine trajectory continuation is not promised.

## Preparation, gates and acceptance

Completed preparation: Gym request/ticket/callback routing, M1 reuse audit,
read-only hardware inventory and this proposed design. A builder check identified one material question about weight ownership. The
parent proposed the separate-run/frozen-transfer default above; no other
material planning question was raised. The accepted upstream contract has now
been reconciled; learner adapter/runtime verification remains future work.

Implementation sequence after design approval:
1. Pure neural rollout/objective/return and compatibility metadata tests; no Gym
   simulation or optional-package import in ordinary core commands.
2. Versioned descriptor-driven adapters/encoders against the accepted Gym delivery;
   verify both reset/step/lifecycle contracts before accepting a new profile.
3. Optional neural arithmetic/model/checkpoint tests in a separately qualified CPU
   runtime; masked distribution and trainer-adoption gate.
4. Both-mode collection/evaluation integration and bounded GPU runtime calibration.
5. Explicitly budgeted feedforward learning in each mode with separate weights,
   followed by frozen evaluation and cross-mode checkpoint evaluation. Numeric
   limits are approved after calibration, not implied here. Only then scope a
   matched GRU experiment.

Accept when both modes execute through their real public contracts, target/mask/
boundary arithmetic and checkpoint compatibility pass, fault/error accounting is
honest, resources/provenance are recorded, and a budgeted learning result (including
an inconclusive one) is documented against baselines. A code test pass is not a
learning result. A failed/incomplete experiment never becomes a fabricated loss.

Gym IDs/signatures are now fixed above. GPU/runtime package pins and concrete
experiment budgets remain pending separate setup/calibration work. Preserve the
completed M1 contracts and runtime when adding the new neural track.
The user must approve the design, then the implementation plan and concrete runtime
setup/experiment budgets can be prepared under the normal workflow.
