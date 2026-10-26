# Ironclad Combat Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and inspect one feedforward combat-policy architecture through standalone Ironclad A0 fights and combats inside normal Runs, with honest boundaries, provenance and bounded experiments.

**Architecture:** A new `neural` package consumes the accepted common combat view without changing M1 contracts. Dependency-free configuration, byte snapshots and return arithmetic surround lazy native adapters and a small direct PyTorch masked actor-critic/PPO implementation. Each mode has its own experiment weights/optimizer; a frozen legal-random controller handles noncombat Run decisions.

**Tech Stack:** Python 3.12; stdlib unittest/TOML/JSON; optional NumPy 2.5.4, Gymnasium 1.4.0, accepted source-built Spire Lab and PyTorch 2.14.1. CPU correctness precedes CUDA calibration. TorchRL is not needed for this trainer.

**Spec:** `docs/superpowers/specs/2026-10-26-ironclad-combat-learning-design.md` (user approved 2026-10-26).

## Global Constraints

- Preparation of this plan is authorized. Implementation and each runtime/experiment phase below await plan review; no install or workload occurred while writing it.
- Use Python 3.12; preserve optional-import isolation, completed M1 contracts/runtime, user-modified Task 6 report and root review diff.
- Work locally on main with explicit-path commits; no pushes, publication, Gym writes, host dependency installs or shared-service changes.
- Accepted Gym revision: `533dad1c8aec86df624b3742b6b3ad1dbf1d0033`. Common fingerprint: `02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78`.
- Run requires explicit `m10-ironclad-policy`, observation/replay v3, fingerprint `30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8`; never substitute frozen M1/M9/M10-v2 profiles.
- Actors/critics see descriptor-defined public features/candidates/masks only. Seeds, tokens, visits, routing, RNG and diagnostics stay in metadata.
- Preserve true death vs faults, accepted-effect no-retry behavior, train/selection/final separation, RNG-001/TIME-001 and finite-content limitations.
- All environments, caches and outputs stay under `artifacts/development/neural-combat/`; generated data/checkpoints remain ignored. Mount reviewed public Gym build inputs read-only; never copy its source/fixtures/environment/native binary into learner-authored code.
- Parent executes synthesis/review as GPT-6.1 Sol/high. Fresh task-scoped Luna6/MAX workers, one implementation writer at a time, no nested workers; fresh Luna/MAX independent review plus substantive parent review before acceptance. Reuse the same worker/reviewer for scoped fixes.
- Workers read AGENTS/README/reference index, this spec/plan and their preceding interfaces. Report exact commands/stdout/stderr/exits under `artifacts/development/neural-combat/task-N/`; distinguish authored, core-tested, optional-tested, native-tested and trained.

## Review Focus

1. A batch containing a terminal row must never advance ongoing rows before masked reset; test mixed rows, reset failure and insufficient remaining full-batch budget in Tasks 3/5.
2. Sticky Run exits can be paired with later inventory accidentally; test repeated exits, reward-induced resource changes, potion selectors and outside-combat death in Tasks 2/5.
3. Mask/reference mistakes can select a legal-looking wrong action or leak metadata; test unknown/padding, invalid targets, dispatch permutations and poisoned metadata in Tasks 2/4.
4. A budget/artifact/close failure can conceal accepted effects or mutate a frozen evaluation model; test each phase, process-control preservation and parameter/optimizer hashes in Tasks 5/6/7.
5. Resume and transfer can silently reuse incompatible encoders, optimizers or seed lineages; test identity mismatches, cross-split Run lineage and warm-start vs resume in Tasks 1/6/7.

## File and interface map

New files are under `src/spire_learn/neural/`; leave existing M1 modules unchanged.

| Files | Responsibility |
| --- | --- |
| `__init__.py`, `contracts.py`, `returns.py` | Pure immutable records, lifecycle/task boundaries, GAE |
| `config.py`, `cases.py` | Strict TOML, authored catalogs, split and seed domains |
| `schema.py`, `snapshot.py` | Common descriptor reader, compact owned byte arrays, typed candidates |
| `envs.py` | Lazy accepted single/vector/Run adapters and bounded fault projection |
| `features.py`, `model.py`, `distribution.py` | Lazy tensorization, public row encoders, candidate scoring and masked sampling |
| `ppo.py` | Direct clipped PPO, deterministic minibatch ordering and update accounting |
| `collector.py` | Persistent collection, explicit resets, separate task/environment boundaries |
| `artifacts.py`, `checkpoint.py` | Neural JSON/events, policy export/training state and compatibility |
| `runner.py`, `evaluation.py`, `cli.py`, `__main__.py` | Bounded orchestration and lazy execution commands |
| `tests/unit/test_neural_*.py` | Dependency-free config/contract/driver-fake/runner tests |
| `tests/neural/test_*.py`, `tests/integration/test_neural_*.py` | Explicitly gated tensor/model and native checks |
| `configs/neural/`, `configs/cases/neural/`, `docs/neural-combat.md` | Authored initial profiles, manifests and usage |

Core records use frozen dataclasses with copied/immutable containers. `BufferArray(dtype: str, shape: tuple[int,...], data: bytes)` supports native-endian int8/int64 with exact byte-length validation. `CombatFrame(tensors: Mapping[str,BufferArray], schema_fingerprint: str)` owns actor inputs only. Bytes avoid retaining native storage or millions of Python scalar objects in rollouts. Tensor modules import optional packages only when called by explicit execution; gated tests import them inside enabled test methods, not at discovery.

`Lifecycle` records `active`, `routing`, `selection_owner`, `active_visit`, `combats_entered`, `combats_settled`, `last_exit` separately. `TaskSignal` records reward, task terminal, bootstrap/continuation flags. `PolicyChoice` records original action index, log-probability, value and behavior-policy version. `StepRecord` records stream/task/decision IDs, immutable source/successor frames, native reward/flags, task signal, choice and actor-training mask; fixed-controller records have no fabricated log-probability/value. These are new neural records, not changes to M1 `Transition`.

## Execution gates and defaults

Tasks 1–2 need only host Python 3.12. Task 3's setup/native calls require the reviewed CPU phase of `2026-10-26-neural-runtime-and-budgets.md`. Tasks 4/6 use its offline CPU environment; Task 5 has pure tests before native execution; Task 7 permits its bounded CPU update smoke only. Approval explicitly covering Tasks1–7 and the companion CPU phase covers their listed CPU invocations together: parent source/evidence reviews are internal gates, not repeated user permission requests. Task 8 needs GPU setup/calibration approval. Task 9 is a proposal finalized after calibration, requiring its own experiment approval.

First calibration candidate: embedding width16, row/global hidden width64, two-layer tanh MLPs, no dropout/batchnorm, float32, no AMP/compile, maximum 10M parameters. Task4 measures its small CPU footprint and Task8 measures runtime/memory before final training sizes are accepted; these widths are not a pre-calibration performance claim. Initial PPO defaults: gamma0.99, lambda0.95, Adam lr0.0003/eps0.00001, clip0.2, value coefficient0.5, entropy coefficient0.01, four epochs, minibatch64, global gradient norm0.5. Normalize advantages once over training rows with population variance and epsilon1e-8. Values/returns stay unnormalized; no value clipping initially. Record these as versioned config choices, not tuned results.

### Task 1: Pure configuration, lineage, task signals and return arithmetic

**Files:** Create `neural/{__init__,contracts,config,cases,returns}.py`; tests `tests/unit/test_neural_{contracts,config,returns}.py`; initial `configs/cases/neural/{train,selection,final}.json` and `configs/neural/{standalone,run}.toml`.

**Interfaces:** `load_neural_config(path: Path) -> NeuralConfig`; `load_neural_cases(path: Path) -> NeuralCases`; `task_signal(previous: Lifecycle, current: Lifecycle, *, terminated: bool, truncated: bool, supports_truncation: bool=False) -> TaskSignal`; `compute_gae(records: Sequence[StepRecord], *, gamma: float, lam: float) -> ReturnBatch`. Task1 also defines frozen `ModelConfig`, `PPOConfig`, `NeuralIdentity`, `Rollout` and its cumulative-counter record for later tasks. `NeuralIdentity` carries model/features/action/objective IDs and declared accepted raw-mode mappings/config hashes. `StepRecord` includes the next value captured from its actual successor; `ReturnBatch` supplies aligned advantages/targets/train mask. Records are grouped by stream/task, never by flat arrival order.

Config fields: mode `standalone|run`, split/partition, manifest path, output root, accepted identities, independent seed streams, model/PPO settings, rollout rows/steps, device and budgets. Reject unknown keys, numeric bools, nonfinite/range-invalid values and unsupported profiles. Record resolved paths and canonical authored-content hashes. Final is evaluation-only and requires a later explicit opt-in.

- [ ] Write RED tests `test_gae_hand_example`, `test_cut_uses_old_successor`, `test_interleaved_streams_do_not_mix`, `test_run_task_end_is_not_run_end`, `test_outside_combat_death_has_no_new_task_loss`, `test_invalid_config_and_split_lineage`.
  ```python
  # gamma=.9, lambda=.8; V=[.5,.4], reward=[0,1], second task-terminal:
  assert advantages == approx_tuple((.292, .6))
  assert targets == approx_tuple((.792, 1.0))
  # Ongoing one-row cut: next V=.4, never replacement V=9:
  assert abs(targets_at_cut[0] - .36) < 1e-12
  ```
  `approx_tuple` is a local test helper; no pytest dependency. Also reject unsupported truncation at runtime, while testing explicit old-final bootstrap in the pure supported-truncation branch. Native end and natural task end both use bootstrap/continuation0. Unknown natural outcomes error; do not coerce to victory/death.
  Accepted public `last_exit.outcome` strings are `victory` and `death`, mapping to+1/-1 exactly once per settled visit. No exit means no natural task reward.
- [ ] Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_returns tests.unit.test_neural_config tests.unit.test_neural_contracts -v`; record observed missing-feature/assertion failure, never invent RED.
- [ ] Implement records/config/cases/signals/GAE; mask fixed-controller rows out of return/loss arrays. Within each stream, only recurse into the next recorded row of the same task; cuts bootstrap the actual successor, not another slot/episode.
- [ ] Author 24 standalone train entries: encounters `cultist,jaw_worm,two_louse,small_slimes` × three decks × HP64/80 (max80), Burning Blood and three empty potion slots. Decks: starting 5Strike/4Defend/1Bash; that deck plus upgraded Immolate/Armaments; that deck plus Warcry/Whirlwind. Validate exact support through Gym later. Selection uses explicit new IDs and held-out seeds, with seen encounters and `large_slime,three_louse` reported separately. Final entries are reserved and not executed. Normal Run cases use the explicit policy profile; no catalog/resource reset options.
- [ ] Define deterministic nonoverlapping environment seed domains: train `[0,2**40)`, selection `[2**40,2**48)`, final `[2**48,2**64)`; replica train offsets `replica*2**32` with monotonically allocated seeds, bounded before domain exhaustion. Keep policy, optimizer/shuffle, curriculum and noncombat RNG seeds separate. A Run and all its combats share one split/lineage; reject duplicate lineage across splits and final training.
- [ ] GREEN focused tests, parent review, independent Luna/MAX review, explicit-path commit. No runtime assertions yet.

### Task 2: Accepted common descriptor, owned snapshots and candidate mapping

**Files:** Create `neural/{schema,snapshot}.py`; `tests/unit/test_neural_schema.py`, `test_neural_snapshot.py`.

**Interfaces:** `parse_common_descriptor(raw: Mapping[str,object]) -> CombatSchema`; `validate_frame(frame: CombatFrame, schema: CombatSchema) -> None`; `candidates(frame: CombatFrame, schema: CombatSchema) -> tuple[CombatCandidate,...]`; `parse_lifecycle(info: Mapping[str,object]) -> Lifecycle`. `CombatCandidate` has original index, verb, typed optional source/target group/row, choice/scalars and applicability bits. `CombatSchema` freezes actual tensor columns, known/occupied keys, references, capacities/vocabularies and accepted identity; no old combat-v1 parser alias.

- [ ] Write RED tests `test_common_descriptor_requires_exact_identity`, `test_unknown_is_not_zero_or_padding`, `test_reference_must_resolve_to_occupied_row`, `test_dispatch_rows_are_not_compacted`, `test_mutated_source_cannot_change_snapshot`, `test_metadata_is_not_actor_input`.
  ```python
  assert tuple(c.index for c in candidates(frame_with_legal_rows_2_9, schema)) == (2, 9)
  assert known_zero_encoding != unknown_zero_encoding
  assert 'token' not in frame.tensors and 'active_visit' not in frame.tensors
  ```
  Use small learner-authored descriptors/byte fixtures for behavioral tests; never copy Gym fixtures or a full upstream descriptor into source/tests. Require known/occupied bits0/1, exact dtype/shape, finite registered units, valid categories and typed -1 unknown references. A noncombat or terminal empty mask is valid transport, but cannot be sent to an actor.
- [ ] Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_schema tests.unit.test_neural_snapshot -v` and record genuine RED.
- [ ] Implement descriptor parsing from actual `tensors[].columns`, `known`, `occupied`, `reference_columns`, `vocabularies` and `action_space`. Keep a separate canonical descriptor digest; do not invent a replacement algorithm for Gym's supplied fingerprint. Parse lifecycle `combats_entered`, `combats_settled`, `active_visit`, `last_exit` exactly. Runtime snapshots copy native C arrays into immutable bytes; unknown transport fields fail closed.
- [ ] Test potion source-slot semantics, card/potion selection ownership, missing/unknown target applicability and terminal empty actors with retained resources. Public references resolve against the current frame, never historical/native instance handles.
- [ ] GREEN focused tests and six-prefix import isolation, parent/independent review, commit. No optional import needed in pure validation.

### Task 3: Protected CPU runtime and real public adapter conformance

**Files:** Create `neural/envs.py`; `tests/unit/test_neural_envs.py`; gated `tests/integration/test_neural_envs.py`; authored runtime helper/runbook under `scripts/neural/` only after CPU setup approval.

**Interfaces:** `make_driver(config: NeuralConfig, cases: NeuralCases) -> CombatDriver` constructs only. `CombatDriver.reset(*, seeds: Sequence[int], entry_ids: Sequence[str|None], reset_mask: Sequence[bool]|None=None) -> tuple[DriverRow,...]`; `step(actions: Sequence[int]) -> tuple[DriverTransition,...]`; `close() -> None`. `DriverRow` contains common frame/lifecycle plus `noncombat_indices` for the fixed controller; actors receive only frame/candidates. Run driver has exactly one row. `DriverTransition` preserves source/successor/native flags/reward and immediate public lifecycle; accepted failures raise `DriverFault` with bounded public accepted-prefix/delivery evidence.

- [ ] RED scripted-driver tests: exact constructors/options/partition, no implicit construction reset, u64 seeds, full-batch int64 actions, mixed terminal rejection, masked reset retaining ongoing state/token, close/fault paths, no retry. Run fake requires `combat_view()` immediately after every successful reset/step, before any next action.
  ```python
  # test_terminal_row_rejects_before_dispatch / test_masked_reset_preserves_ongoing
  assert native_dispatches_after_rejected_step == 0
  assert ongoing_state_after_reset == ongoing_state_before_reset
  assert ongoing_token_after_reset == ongoing_token_before_reset
  ```
- [ ] Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_envs -v`; implement lazy adapters; GREEN and import isolation.
- [ ] Execute approved CPU setup from the runtime companion: build from hash-verified read-only accepted inputs, isolated hash-locked venv, offline `pip check`, exact schema/build/catalog verification. Stop if package/toolchain/mount constraints differ; do not reuse/copy upstream native binary or M1 venv.
- [ ] With `SPIRE_LEARN_RUN_NEURAL_NATIVE=1`, offline command: `/work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_envs -v` using `PYTHONPATH=/learner/src`.
  Require exactly three tests, zero skips: `test_standalone_public_profile`, `test_manual_vector_mixed_reset`, `test_normal_run_common_view`. Shared invocation counters enforce ≤4,096 total native slot actions, ≤1,024 per natural fight, 120s soft / 300s whole-run. No policy updates. Two scripted public catalog entries exercise selections/potions; native Run traversal uses fixed seeded legal-random without seed search. If required lifecycle evidence is not reached at cap, report inconclusive, never rerun automatically.
- [ ] Confirm real descriptor mapping, snapshots after settlement and next noncombat action, current dispatch indices, source/build pins, close and no remaining container. Review evidence and commit. Native tests prove only scoped integration, not learning.

### Task 4: Typed public features, masked actor-critic and direct PPO

**Files:** Create `neural/{features,model,distribution,ppo}.py`; `tests/neural/__init__.py`, gated `tests/neural/test_model.py`, `test_distribution.py`, `test_ppo.py`; pure `tests/unit/test_neural_feature_contract.py`.

**Interfaces:** `encode_frames(frames: Sequence[CombatFrame], schema: CombatSchema, *, device: str) -> EncodedBatch`; `build_policy(schema: CombatSchema, model_config: ModelConfig, *, seed: int, device: str) -> CombatPolicy`; `CombatPolicy.forward(batch: EncodedBatch) -> PolicyOutput` (logits in original1024 row order, values[B]); `act(frames, *, generator, deterministic: bool=False) -> tuple[PolicyChoice,...]`; `values(frames: Sequence[CombatFrame]) -> tuple[float,...]` (no sampling); `ppo_update(policy, optimizer, rollout: Rollout, config: PPOConfig, *, shuffle_generator, budget: Budget) -> UpdateMetrics`. Define `EncodedBatch` in features, `PolicyOutput` in model and `UpdateMetrics` in ppo; these modules remain import-isolated until explicit tensor execution.

- [ ] Write RED tests for illegal logit1000 vs legal-1000/-1001, singleton log-prob/entropy0, empty ongoing mask rejection before sampling, source/target candidate permutation, finite gradient and unchanged metadata poison. Disabled optional tests skip before importing NumPy/Torch.
  ```python
  # test_illegal_extreme_logit / test_unchanged_policy_ratio
  assert illegal_probability.item() == 0.0
  assert torch.isfinite(legal_log_probs).all()
  assert torch.allclose(ratios_before_update, torch.ones_like(ratios_before_update))
  ```
- [ ] Pure feature contract lists descriptor fields by meaning: categorical embeddings; numeric `sign(x)*log1p(abs(x))` (known values only); known bits; occupied row pooling; current typed references. No learned action-slot or metadata embeddings. Per-field categorical tables follow descriptor vocabularies. Reference rows gather row embeddings with explicit unknown/not-applicable flags; row numbers never become scalar features.
- [ ] Implement width16 categorical embeddings, width64 two-layer tanh row MLPs, masked mean and max pooling per row group (empty max=0), a width64 global MLP, shared candidate MLP over global/verb/source/target/choice/scalars, and linear scalar value head. No running normalization/dropout. Reject >10M parameters or incompatible feature IDs; record `spire-learn-combat-features-v1` and `spire-learn-combat-ff-v1` with config/descriptor fingerprints.
- [ ] Mask logits with negative infinity, compute stable legal log-softmax, calculate entropy over legal rows only, sample via `torch.multinomial(..., generator=...)`; deterministic choice is highest logit then smallest original row index. Terminal rows bypass actor. Reject nonfinite applicable logits/values before sampling; never repair masks.
- [ ] Implement ratio `exp(new_logp-old_logp)` and loss `-mean(min(ratio*A,clamp(ratio,.8,1.2)*A)) + .5*mean((V-target)**2) - .01*mean(entropy)` with only training rows. Normalize advantages once per rollout, freeze old inputs/log-probs/values, shuffle using a separate saved CPU generator. Check losses/gradients finite before clipping0.5 and stepping. Count each accepted optimizer step; stop on fault, never roll back/retry an uncertain step. No actor collection during updates.
  Admit one matching behavior-policy version at update entry and compute frozen returns with Task1; version advances during internal optimizer steps without rejecting that admitted batch. Mark rollout consumed on admission so an aborted/finished update cannot replay it. Value-only successor queries must not consume policy RNG or create phantom decisions.
- [ ] Gated CPU command: `SPIRE_LEARN_RUN_NEURAL_TENSOR=1 PYTHONPATH=/learner/src /work/venv-cpu/bin/python -B -m unittest discover -s tests/neural -v`. Require model/distribution/PPO cases execute with zero skips. Ratio1 with unchanged weights; positive/negative advantage clipping directions; finite value/entropy losses; fixed rows do not affect gradients. Report original/updated parameter hashes.
  Record candidate parameter count, snapshot bytes per frame and CPU B1/B4 forward/backward timings/peak process memory during these authored-frame checks. No gameplay or extra sampled rollout; reject a candidate exceeding process8GiB or the parameter cap before adopting it for later smoke/calibration.
- [ ] Parent and independent review, explicit-path commit. GPU is not required or authorized by this task.

### Task 5: Persistent both-mode collector and task-boundary accounting

**Files:** Create `neural/collector.py`; `tests/unit/test_neural_collector.py`; gated `tests/integration/test_neural_collection.py`.

**Interfaces:** `Collector(driver: CombatDriver, policy, config: NeuralConfig, *, controller_rng: random.Random, curriculum_rng: random.Random, budget: Budget)`; `collect(*, max_learned_decisions: int, max_env_actions: int, policy_version: int) -> Rollout`; `close() -> None`. `Rollout` owns StepRecords and counters: attempted/forwarded/known-accepted/delivered environment slots, learned/fixed decisions, resets, tasks/completions/faults, elapsed/overshoot and incomplete work. Counters accumulate across calls, not just each horizon.

- [ ] RED counter-only fakes test mixed row finals/reset continuity, learned vs fixed policy attribution, one task reward per sticky exit, natural Run victory without Run reset, later reward-resource mutation, outside-combat death, horizon successor bootstrap and insufficient full-batch capacity. Verify every accepted source/final snapshot precedes any reset/next step.
  ```python
  # test_run_victory_closes_task_once_and_keeps_run
  assert (task_reward_sum, task_end_count, run_reset_count) == (1.0, 1, 0)
  assert checkpoint_inventory == inventory_at_settlement
  assert checkpoint_inventory != inventory_after_reward_choice
  ```
- [ ] Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_collector -v`; implement collector. Stop before dispatch when remaining action/decision budget cannot reserve the whole batch; reset only when further collection will occur. Do not pad/truncate a native batch to fit caps. Externally stopped episodes stay incomplete; preserve native flags.
  Populate ongoing next values with `policy.values(actual_successor_frames)` under the unchanged behavior policy; task-terminal next value is zero. Never call `act` just to obtain a bootstrap value.
- [ ] For Run mode, choose from `noncombat_indices` with the separate frozen legal-random RNG until combat routing. Account fixed actions/resources but exclude them from actor loss. Observe each exit/visit once and immediately snapshot resources; do not reset the Run at victory. Treat unsupported/unknown outcomes and lifecycle jumps as errors.
- [ ] On fault, retain bounded diagnostics/known prefix and incomplete decisions, never retry accepted calls or reset a frozen owner. Close once; preserve primary exceptions including process-control, attach close/artifact failures as secondary notes/evidence. No generic policy-state serialization.
- [ ] GREEN scripted tests, then one approved offline collection integration invocation with two mode tests/zero skips, combined ≤4,096 native slot actions,120s soft/300s whole-run, no updates. Exercise actual manual reset and normal Run task end; a cap without required evidence is inconclusive. Parent/independent review and commit.

### Task 6: Neural artifacts, compatible exports and checkpoint recovery

**Files:** Create `neural/{artifacts,checkpoint}.py`; `tests/unit/test_neural_artifacts.py`; gated `tests/neural/test_checkpoint.py`.

**Interfaces:** `NeuralWriter.create(root: Path, run_id: str) -> NeuralWriter`; `start(config: Mapping, manifest: Mapping)`, `event(record: Mapping)`, `finalize(summary: Mapping)`, `close()`. `save_checkpoint(path: Path, *, policy, optimizer, rng_state: Mapping, metadata: Mapping) -> None`; `load_checkpoint(path: Path, expected: NeuralIdentity, *, purpose: str, device: str) -> LoadedCheckpoint`, purpose `evaluate|warm_start|resume`. Define `LoadedCheckpoint` in checkpoint with validated metadata, named model tensors, optional optimizer/RNG state according to purpose, `optimizer_step: int` (zero except resume) and `inflight_record_count: int` (always zero on load).

- [ ] RED pure tests: exclusive safe run IDs, explicit JSON whitelist/nonfinite rejection, failing append/finalization/close, source provenance unavailable vs dirty, wrong compatibility IDs or mismatched hashes before optional loading. Reuse `canonical_fingerprint` and `collect_provenance`, not M1 schema envelopes or a widened generic dataclass serializer.
- [ ] Implement a small neural writer with exclusive config/manifest/events and atomic summary; `spire-learn-neural-run-v1` envelopes. Persist action/lifecycle/counter traces and bounded metrics, not full raw observations by default. Preserve learner revision/dirty status plus exact code/config hashes, native build/source/package IDs, schema/rules/content/reward/objective, seeds/partitions/device/budgets and controller identity.
- [ ] Export named tensor `state_dict` plus strict JSON metadata/hash inventory; training checkpoint additionally contains optimizer and explicit Python/Torch policy/minibatch/curriculum/controller RNG state, counters and update/version identities. Enforce ≤256MiB aggregate checkpoint and finite tensor state; load on CPU using `torch.load(..., weights_only=True)`, never arbitrary object or unrestricted-pickle loading. Validate model/features/action/objective/profile mappings and hyperparameter IDs before applying weights/state.
  Evaluation/warm start require compatible model/features/action/objective and verified raw-mode mappings, while allowing different mode/controller/seeds/budgets recorded as transfer context. Full config hashes are provenance, not blanket equality checks that prohibit transfer. Resume additionally requires matching optimizer/hyperparameter contracts and restores its counters/RNG. Test `test_transfer_allows_mode_seed_change_but_rejects_encoder_change`. Save new checkpoint directories through an owned temporary sibling and atomic publish; never overwrite an unfamiliar checkpoint on failure.
- [ ] Gated tests round-trip model/optimizer/RNG; frozen evaluation mutates neither weights nor optimizer; warm start transfers only compatible model weights and creates fresh optimizer/RNG/counters with parent lineage. Resume restores declared learner state, drops in-flight on-policy rollout, creates fresh environment episodes using the next unused recorded seeds, and records the discontinuity. Never claim exact environment/cross-device continuation.
  ```python
  # test_warm_start_differs_from_resume / test_frozen_load_preserves_weights
  assert warm_start.optimizer_step == 0 and resume.optimizer_step == saved_step
  assert parameter_hash_before_evaluation == parameter_hash_after_evaluation
  assert resumed_inflight_record_count == 0
  ```
- [ ] GREEN pure/gated checkpoint tests with zero skips when enabled; parent/independent review, commit. No extensive checkpoint service or public upload.

### Task 7: Training runner, frozen evaluation and lazy neural CLI

**Files:** Create `neural/{runner,evaluation,cli,__main__}.py`, pure `tests/unit/test_neural_runner.py`, `test_neural_cli.py`; gated `tests/integration/test_neural_training.py`; `docs/neural-combat.md`; finalize authored neural configs/manifests.

**Interfaces:** `train(config: NeuralConfig, *, runtime: TrainingRuntime, clock: Callable[[],float]) -> NeuralReport`; `evaluate_policy(config: NeuralConfig, *, runtime: TrainingRuntime, clock: Callable[[],float], allow_final: bool=False) -> NeuralReport`; `choose_reference(frame: CombatFrame, schema: CombatSchema, *, kind: str, rng: random.Random) -> int` in evaluation; `main(argv: Sequence[str]|None=None) -> int`. `TrainingRuntime` provides `collector`, `update(rollout)->UpdateMetrics`, `save(path,purpose)`, `close()`; lazy `create_runtime(config, *, purpose: str, checkpoint: Path|None)` assembles real components. Pure runner tests inject scripted runtimes.

- [ ] RED tests enforce finite caps before resets/calls/minibatches, partial rollout updating only valid learned records, no stale-policy rollout reuse, reserved final evaluation, exhaustive not-started/incomplete accounting, artifact faults stopping further actions, and close preserving primary process-control errors. Exact completed optimizer-step counts determine policy-version changes even on a mid-update stop.
  ```python
  # test_artifact_failure_stops_actions / test_close_preserves_primary_control
  assert native_calls_after_artifact_failure == 0
  assert propagated_exception is original_keyboard_interrupt
  assert 'close' in '\n'.join(propagated_exception.__notes__)
  ```
- [ ] Implement CLI `python -m spire_learn.neural {train,evaluate,calibrate} --config PATH`, `--validate-only`; evaluation selects `--policy checkpoint|legal-random|end-turn-last-v1` (default checkpoint), requires `--checkpoint` only for that kind, and final requires `--allow-final`. Baselines reject checkpoint arguments and use the same common candidate/selection/potion contract. Training `--resume` and `--warm-start` are mutually exclusive. Validation/help must not create outputs, import optional packages or initialize devices. Exit0 means bounded operation completed with sound artifacts (not learning achieved),2 invalid/setup-blocked,1 operational/artifact failure; process-control propagates after cleanup.
- [ ] Persist all accepted effects, metrics, unfinished work and final elapsed/overshoot. Evaluate frozen exports with deterministic and seeded-sampling modes explicitly named. Report standalone and in-run fights separately, reached-combat denominators/HP retention/turns/actions/full-Run progress and controller identity. Match baseline cases/seeds/budgets; do not tune on final cases. Selection/model choice cannot mutate the evaluated checkpoint.
- [ ] GREEN pure runner/CLI tests plus fresh-process imports/discovery/help/all configs with six optional prefixes blocked (`numpy,gymnasium,torch,torchrl,tensordict,spire_lab`); full default unittest discovery only after the coherent task is finished. Existing gated tests remain skips; new optional tests must never import at discovery.
  Default suite command: `env -u SPIRE_LEARN_RUN_NATIVE_INTEGRATION -u SPIRE_LEARN_RUN_TORCHRL_INTEGRATION -u SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION -u SPIRE_LEARN_RUN_NEURAL_NATIVE -u SPIRE_LEARN_RUN_NEURAL_TENSOR PYTHONPATH=src python3.12 -B -m unittest discover -s tests -v`.
- [ ] One approved CPU training smoke: `SPIRE_LEARN_RUN_NEURAL_NATIVE=1 SPIRE_LEARN_RUN_NEURAL_TENSOR=1 PYTHONPATH=/learner/src /work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_training -v`. Exactly two mode tests/zero skips; ≤512 learned decisions,2,048 total environment slots, two rollout updates per mode and32 optimizer steps total,120s soft/300s whole-run. Standalone B4/horizon32 and Run B1/horizon128 yield ≤128 learned rows per rollout; four epochs/minibatch64 yield ≤8 optimizer steps per rollout. Two exports/checkpoints and deterministic reload forward parity on stored public frames are required; this adds no native evaluation cohort or sampled decisions. Actual frozen environment evaluation follows Task9. Changed finite weights plus correct accounting are plumbing evidence, not improved play. No retry.
- [ ] Parent/independent review, commit. CPU evidence cannot substitute for both-mode later learning/evaluation results.

### Task 8: Qualified RTX5090 runtime and bounded calibration

**Files:** Author/review `scripts/neural/` GPU runbook and `configs/neural/calibration.toml`; implement `calibrate(config: NeuralConfig, *, clock: Callable[[],float]) -> NeuralReport` in `neural/runner.py`; pure `tests/unit/test_neural_calibration.py`.

- [ ] Test reservation math with fakes: cumulative trial caps, insufficient free-device memory, device failure, synchronization timing, no fallback package/device switch and no automatic repeated cohort. Config selects only the inventoried GPU by recorded UUID.
  ```python
  # test_calibration_cap_reserves_before_dispatch
  assert reserved_slot_actions <= 2048
  assert disposable_optimizer_steps <= 3
  assert native_calls_after_exhaustion == 0
  ```
- [ ] After GPU phase approval, resolve/audit/hash-lock the exact candidate CUDA wheel and full closure in a separate venv. Require existing host Docker GPU support; no host driver/toolkit install. Verify CUDA availability, device name/capability, reported compiled architectures, and real finite forward/backward/Adam work. Keep model float32/no compile/AMP. A version or `is_available()` print alone is insufficient.
- [ ] One calibration cohort: standalone B4/8/16 at horizon32, plus one Run driver, total ≤2,048 native slot actions, ≤3 disposable update microbenchmarks,300s soft/600s whole-run; peak device memory≤8GiB, host RAM≤8GiB,4 CPUs. Record simulation, snapshot/collation, H2D, forward/backward/update, serialization and synchronized wall times separately. CPU-vs-GPU comparison uses the same stored public frames/model sizes, no new gameplay cohort or learning claim.
- [ ] Select the fastest tested size within caps, never infer threaded simulation scaling from native serial batching. If no GPU benefit, report it; do not change user's compute preference or launch cloud work. Finalize Task9 run limits from actual calibration and review/commit evidence, not executable training authority.

### Task 9: Approved initial both-mode learning, baselines and frozen transfer

**Files:** Final authored experiment configs and `docs/neural-combat.md`; ignored `artifacts/development/neural-combat/experiments/` and acceptance report. No new simulator/trainer architecture.

- [ ] After Task8, present exact resolved config/manifest/checkpoint identities and per-cohort accounting for approval. Candidate ceiling: two learner seeds in each mode (four independent runs), ≤16,384 learned decisions per run,≤65,536 total environment actions per run,600s soft/900s whole-run per run; stop at the first cap. Separate fresh weights/optimizer per run. No retries, seed search or automatic budget extensions.
- [ ] Proposed standalone rollout B≤8/horizon64 and Run B1/horizon128, bounded by the tested calibration result; four PPO epochs/minibatch64 defaults remain unless explicitly revised before run approval. Collect/update/sample counts include calibration/CPU smokes separately; never reset lifetime accounting to hide an overrun.
  Save at most one midpoint and one final training checkpoint plus one final policy export per run; no unbounded per-update checkpoints. A wall/budget stop saves the final valid learner state in the final slot. No deletion or replacement of unfamiliar existing outputs to meet storage limits.
- [ ] Proposed selection gate:16 standalone and16 normal-Run seed cases × four trained exports and two baselines =192 scheduled evaluations. Each has ≤256 total environment actions; shared selection cap65,536 actions,600s soft/900s whole-run. Order round-robin by case/policy/mode; mark unstarted cases when time/caps prevent coverage. Compare legal-random and explicit `end-turn-last-v1` (lowest legal original row excluding end_turn when alternatives exist; covers selections/potions). Each export uses deterministic highest-logit/smallest-row evaluation through both adapters, weights/optimizer hashes unchanged. A case budget stopping inside combat leaves it incomplete; native/task flags stay honest. Final split stays unused.
- [ ] Report each seed/mode learning curve, matched baseline coverage, win/death/incomplete/fault denominators, native vs task reward, HP/resources, sampled decisions/update counts, wall/sample efficiency and fixed-controller effects. Distinguish seen-encounter selection, unseen-encounter selection and cross-mode transfer. An inconclusive or negative result satisfies honest experiment reporting, not playing-strength claims.
- [ ] Parent acceptance and independent final feature review; one final default suite/import-isolation pass after source changes, no gratuitous repeated native/training runs. Record actual checks/results/limitations and leave GRU/LLM/demo/full-run-policy work outside this milestone.

## Plan self-review and authority

Coverage: public mapping and both lifecycles (Tasks2/3/5), masks/model/PPO (4), return/task math and splits (1), recovery/provenance (6), actual both-mode training/evaluation (7/9), device/performance/budgets (8). All five Review Focus cases have named owning tests. The 256MiB checkpoint and compact-byte rollout limits prevent raw tuple/JSON blowups; memory is measured at calibration. No Gym gameplay implementation is duplicated.

Execution method is already selected by the user: parent GPT-6.1 Sol/high with task-scoped Luna6/MAX assistance and substantive review. Plan review precedes Task1. The runtime companion contains concrete phase boundaries and proposed limits; its GPU closure is a resolution deliverable, not a fabricated complete lock. Initial learning budgets are re-presented after calibration, as required by the approved design.
