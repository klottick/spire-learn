# Randomized Ironclad Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reproducible broad scenario generation, adaptive lessons and exact checkpointed stream continuation without running a new training campaign.

**Architecture:** Pure support/recipe/generator modules prepare immutable Gym-sized catalogs. A pure curriculum sampler plugs into the existing collector; explicit stream metadata extends the existing runner at its config/checkpoint boundary. A small orchestration/CLI layer prepares reproducible experiments and exposes a lazy segment API, preserving fixed-manifest behavior.

**Tech Stack:** Python 3.12, stdlib, existing unittest/core neural contracts; no dependencies or runtime installation.

**Spec:** `docs/superpowers/specs/2026-10-31-randomized-ironclad-curriculum-design.md` at `50d068444a6780258308b29ca647727ee6f7d6fe` (user approved).

## Global Constraints

- Work in the existing main checkout, one source writer, scoped local commits; no push, worktree, service changes, tracker writes or Gym edits. Preserve all retained campaigns and consumed claims.
- Preserve `artifacts/development/foundation/task-6-report.md` SHA256 `e7ea42e58871d492d4b5ca2e408f782155526fe80258df1a44bad2cd02285d2a` and `task-6-core-review.diff` SHA256 `a52df58775d2c5fc0d8f1cbbe1946af787839e5d1379cd752e2663a4557edc29`; never stage them.
- No Torch/NumPy/Gymnasium/SpireLab/TorchRL/TensorDict runtime, native effects, Docker lifecycle, installs, training or replay. Author gated qualification tests/proposal only; never execute their enabled lane.
- Actor/critic receive only accepted public tensors. Seeds, recipe/family/difficulty IDs, curriculum statistics, private state and hidden order never become actor features.
- Initial family mix 70/20/10, catalog maximum256 entries/1MiB, four-shard coverage cycle, realistic decks8–35/unusual5–40, maxHP80–120/currentHP24..maxHP. All75 supported cards and43 encounters are generation targets; generated coverage and actual exposure are separate.
- Gym owns rules, shuffle, enemies, native RNG and validation. Reference rarity/mechanic labels are sampling annotations, not executable mechanics or support proof. No generated statuses/curses or active Neow's Lament.
- Keep model/features/PPO/reward unchanged. Fixed manifests retain strict resume checks. New stream resume preserves actual config/shard identities, Adam/RNG/version/progress and fault semantics.
- The existing final split remains unchanged and unexecuted. Development panels and held-out combinations are new-experiment-disjoint only; historical parent exposure is unknown.
- Original standalone B is the common weight-only pilot parent, fresh Adam/version/progress/RNG. Original Run B is a frozen evaluation reference. Exact parent inventories remain in the approved design and retained preparation file.
- Same preserved delegation: GPT-6.1 Sol/high executor, Luna/MAX reviewers, no nested agents. Parent resolves routine interfaces; no repeated plan approval is required after the user's explicit design/implementation approval.

## Review Focus

1. A valid card vocabulary is mistaken for support, or unsupported item combinations reach dispatch: Task1 binds executable public support separately and rejects invalid generated entries before optional runtime.
2. A held-out pair contains a starter card and makes all-card coverage impossible, or duplicate compositions cross splits: Task1/4 reject impossible recipes and verify composition exclusions with bounded construction.
3. Curriculum callbacks count faults/censoring as losses or leak metadata into frames: Task2 injects real collector fakes and proves natural-settlement-only updates and identical actor frames.
4. A changed manifest silently resets Adam/RNG or weakens legacy identity checks: Task3 validates real save/recovery boundaries across two shard transitions, tampering and wrong modes.
5. A new CLI path imports runtime packages or dispatches during validation: Task4/5 fresh blocked-import guards and enabled harness fakes prove zero effects for all preparation/invalid-input paths.

## File responsibilities

- `scenario_support.py` + a small authored `configs/neural/curriculum-v1/support.json`: pure finite support/annotation loading and provenance.
- `scenario_stream.py` + `configs/neural/curriculum-v1/recipe.json`: recipe validation, deterministic bounded generation, identity and coverage.
- `curriculum.py`: bounded pure sampler state and outcome statistics; optional collector integration.
- `stream_resume.py`: explicit stream binding/state and predecessor-to-next-shard checks; narrow config/runner edits.
- `stream_runner.py`: prepare artifacts and lazy single-segment orchestration using the accepted runner; no new host container controller.
- `stream_cli.py`: pure preparation/validation/pilot description, routed by the existing lazy CLI.
- New focused unittest files mirror these modules. Existing tests change only to cover the additive producer boundary owned by that task.

---

### Task 1: Finite support and deterministic scenario generation

**Files:** Create `src/spire_learn/neural/scenario_support.py`, `src/spire_learn/neural/scenario_stream.py`, `configs/neural/curriculum-v1/{support,recipe}.json`, `tests/unit/test_neural_scenario_stream.py`.

**Interfaces:** Produce immutable `ScenarioSupport`, `ScenarioRecipe`, `ScenarioShard`; `load_scenario_support(path: Path) -> ScenarioSupport`; `load_scenario_recipe(path: Path) -> ScenarioRecipe`; `generate_scenario_shard(recipe: ScenarioRecipe, *, split: str, shard_index: int, arm: str = "mixed") -> ScenarioShard`; `validate_coverage_cycle(shards: Sequence[ScenarioShard], recipe: ScenarioRecipe) -> Mapping[str, object]`. Recipe exposes a finite `category_keys` tuple. Shard fields: `manifest`, `catalog`, `annotations` keyed by case ID, `shard_index`, `arm`, `manifest_sha256`, `distribution_sha256`. Annotations include family, deck stage, lesson kind/category, target card IDs and encounter tier; seeds/IDs stay outside public entry fields. All payloads JSON-convertible, owned and immutable. Arms are `fixed`, `randomized`, `mixed`; only mixed uses adaptive lessons. Tasks2–4 consume these exact names. Broad coverage requirements apply to randomized/mixed arms; fixed retains and reports its narrow control coverage without pretending to contain all75cards.

- [ ] **Step1 — RED:** Author tests for byte-identical same-seed output, different shard/split identity, 179/51/26 full-shard quotas, four-shard base/upgraded all-card and all-encounter coverage, rarity/stage variety, and canonical catalog parser roundtrip. Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_scenario_stream -v`; retain literal failure.
- [ ] **Step2 — Support:** Bind exact public accepted Gym66 support from the durable accepted source (`crates/spire-env/src/combat_training.rs` and referenced core `TRAINING_*` tables). Record source revision/path/hash provenance in the small authored support file. Add only selected public ID/upgrade/act/kind/item fields and explicitly learner-authored rarity/mechanic tags; do not copy engines, private sources or raw research artifacts. No installed package import. Parse/validate the75/43/30/13 inventories; source-reference metadata does not itself establish runtime qualification.
- [ ] **Step3 — Generator:** Implement the interfaces, isolated hash-derived generator domains, bounded rejection (4096 attempts per entry), canonical IDs/lineages, existing case-schema validation and byte/count enforcement. Mixed full shards use179/51/26; deterministic smaller shards carry fractional family allocation across the four-shard cycle. Failed coverage rejects preparation. Fixed arm uses the old three-deck/13-encounter distribution; randomized uses the realistic family without adaptive lessons. All arms share support validation and separate experiment identities.
- [ ] **Step4 — Boundary GREEN:** Add invalid bool/NaN/unknown-key/unsupported-card/upgrades/potions/relic-counter tests; forbid active Lament and BurningBlood+BlackBlood; tests force oversized payload, bounded rejection exhaustion, reserved starter-pair impossibility and cross-split canonical composition collisions. Reserved pair lists are small, explicit nonstarter pairs with support feasibility checked before generation. Run focused suite and fresh six-prefix import guard; no optional execution.
- [ ] **Step5 — Freeze:** Self-review, preserve user hashes, record commands/exits/provenance/API exports in task report and commit only owned paths (`feat: generate broad Ironclad scenario catalogs`). No broad default suite yet.

### Task 2: Bounded lesson sampler and collector attribution

**Files:** Create `src/spire_learn/neural/curriculum.py`, `tests/unit/test_neural_curriculum.py`; modify `src/spire_learn/neural/collector.py`, `tests/unit/test_neural_collector.py` narrowly.

**Interfaces:** Consume Task1 shard annotations. Produce `CurriculumSampler(annotations: Mapping[str, Mapping[str, object]], *, distribution_sha256: str, categories: Sequence[str], state: Mapping[str, object] | None = None)` with `choose_case(cases: Sequence[NeuralCase], *, rng: random.Random) -> NeuralCase`, `observe(case_id: str, outcome: str) -> None`, `snapshot() -> Mapping[str, object]`. Categories are the recipe's finite `category_keys`, shared across shards; state binds the distribution hash rather than deriving a growing universe from new cases. Collector gains appended optional keyword `case_sampler=None`; default behavior is identical. Existing `curriculum_rng` remains the selection RNG and existing actor RNG stays separate.

- [ ] **Step1 — RED:** Tests for repeatable choices/state restore, family allocation, least-exposed versus 20–80% success preference after32 completed fights, and a nonzero floor. Focused submix is50/30/20 single/pair/small-group. State uses fixed category keys, bounded counters and at most128 recent outcomes per category; no historical trajectories or per-ever-generated-case growth.
- [ ] **Step2 — Sampler:** Implement fixed finite category state keyed by card/mechanic/encounter buckets with current catalog mapping, not case IDs that grow across shards. Explicitly choose family then within-family case so adaptive priority cannot change70/20/10 into an undocumented mixture. A state/annotation mismatch rejects before sampling. Sparse/small catalogs retain declared family accumulator behavior.
- [ ] **Step3 — Collector:** Inject only for standalone train. At reset use `choose_case`; on already-validated natural settlement call `observe` once using the retained slot case. Never update it for fault, truncation, censored segment closure, sticky exit rereads or evaluation. Sampler failures latch collection failure without repeating accepted driver effects. Preserve existing counters and actor frames.
- [ ] **Step4 — GREEN:** Fake-driven actual Collector tests verify no metadata in policy frames, no stale off-policy data, no repeated settlements, exception prefix preservation and exact default path when sampler absent. Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_curriculum tests.unit.test_neural_collector -v` and fresh import guard. No native workload.
- [ ] **Step5 — Freeze:** Report exact sampler state/exports and covering evidence; commit only owned changes (`feat: sample fresh on-policy curriculum lessons`).

### Task 3: Explicit scenario-stream config and exact resume

**Files:** Create `src/spire_learn/neural/stream_resume.py`, `tests/unit/test_neural_stream_resume.py`; modify `config.py`, `runner.py` and their focused tests. Only a narrow sampler-projection accessor may be added to collector if Task2 cannot expose its public snapshot.

**Interfaces:** Append `NeuralConfig.scenario_stream: Mapping[str, object] | None = None`, parsed from optional exact `[scenario_stream]` with `schema`, `recipe_path`, `arm`, `shard_index`, `distribution_sha256`. Loaded config resolves/verifies recipe and actual case manifest; stream is standalone/train only. Produce `stream_identity(config: NeuralConfig) -> Mapping[str, object]`, `validate_stream_resume(metadata: Mapping[str, object], config: NeuralConfig) -> Mapping[str, object]`, `stream_checkpoint_state(config: NeuralConfig, sampler_state: Mapping[str, object]) -> Mapping[str, object]`. Return normalized validated state; never mutate loaded metadata. Preserve legacy exact hash checks when table absent.

- [ ] **Step1 — RED:** Real config-loader tests reject arbitrary hash/path changes, wrong modes, evaluation/final recipes, missing support bindings and fake shard content. Resume tests cover same shard and exactly next derived shard, reject skips/backwards/changed recipe/support/model/PPO/budgets/device/rollout, and retain original fixed-manifest tamper rejections.
- [ ] **Step2 — Validation:** Stream identity binds every stable training setting including RNG domains/output-root semantics and excludes only concrete derived shard index/manifest fields. Store actual predecessor and next config/manifest raw and canonical hashes separately. Verify derivation rather than overwriting hashes in dataclasses. Global checkpoint/PPO payload validation stays unchanged.
- [ ] **Step3 — Runner boundary:** Resolve stream sampler before driver reset. Save its state and stream binding in `run_metadata` while the collector is healthy, preserving closed training-state fields. Restore exact Adam/RNG/version/progress and sampler state for explicit stream resume. Warm-start old checkpoints uses fresh state and cannot claim exact resume. Resume normal Run is rejected for this implementation while spire-tsw remains outside scope.
- [ ] **Step4 — GREEN:** Strict fakes exercise the actual accepted `TrainingRuntime.save`/`_checkpoint_recovery` path through warm start and two shard transitions; compare saved snapshot with restored state, not a later clock reading or closed collector property. Assert no repeated seed/decision/task IDs, exact independent RNG advancement, wrong parent rejection, exhausted budgets forward zero effects, and unchanged actual config identities. Run stream/config/runner/collector focused suites; no Torch import or protected runtime.
- [ ] **Step5 — Freeze:** Report compatibility additions and exact metadata, tests and outputs. Commit scoped paths (`feat: resume generated curriculum streams without losing state`).

### Task 4: Preparation CLI, development panels and lazy segment API

**Files:** Create `src/spire_learn/neural/stream_runner.py`, `src/spire_learn/neural/stream_cli.py`, `tests/unit/test_neural_stream_runner.py`, `tests/unit/test_neural_stream_cli.py`; modify `src/spire_learn/neural/cli.py` only to route the new command. Create authored pilot/development recipe files under `configs/neural/curriculum-v1/` as needed.

**Interfaces:** `prepare_stream(recipe_path: Path, destination: Path, *, arm: str, split: str, shard_index: int) -> Mapping[str, object]`; `build_development_panel(recipe: ScenarioRecipe) -> Mapping[str, object]`; `pilot_plan(recipe: ScenarioRecipe) -> Mapping[str, object]`; lazy `run_stream_segment(config: NeuralConfig, *, purpose: str, checkpoint: Path | None, operation_limits: RuntimeOperationLimits, dependencies: Mapping[str, object] | None = None) -> object`. Dependencies are `create_runtime`, `train`, `clock`, used as plain callables; default imports occur only inside explicit execution. Return actual accepted report/prefix/checkpoint references, not synthetic success.

- [ ] **Step1 — RED:** CLI help, recipe validation and `curriculum prepare --recipe PATH --destination PATH --arm mixed --shard-index N` work without optional import attempts; malformed input/occupied destination causes zero output/runtime effects. Add explicit `--validate-only` that writes nothing. Existing train/evaluate/calibrate CLI remains unchanged.
- [ ] **Step2 — Preparation:** Publish bounded recipe/support/shard/config/annotation files exclusively under the requested destination, with a raw/canonical inventory and actual loader roundtrip. Generated config uses real paths and accepted identities. Do not put temporary output in Git or copy parent weights. Pilot plan binds all three arms to the original standalone B inventory and paired initialization; it grants no execution authority.
- [ ] **Step3 — Evaluation:** Deterministically author64 standalone development cases (16 new-seed familiar,16 novel canonical decks,16 reserved-pair combinations,16 resource/later-tier cases), disjoint from new training composition/lineage/seed domains. Every panel entry must satisfy public support. Report stratified win/HP/resource/censoring/coverage denominators from actual supplied reports; preserve operational failure and do not infer full-Run strength. Existing final.json is untouched.
- [ ] **Step4 — Orchestration GREEN:** Actual factory/train call contract with strict fakes proves all three operation-limit fields survive, only the current parent is passed, next shard advances only after healthy checkpoint publication, and setup/update/close/write/control failures halt without retry. No duplicate trainer, simulator or container controller. Run focused CLI/runner suites, whole projected entry fake, import isolation and JSON boundedness checks.
- [ ] **Step5 — Freeze:** Report APIs and concrete generated artifact identities, no runtime claim; commit owned paths (`feat: prepare and orchestrate reproducible curriculum segments`).

### Task 5: Integration, documentation and an unexecuted qualification proposal

**Files:** New `tests/neural/test_curriculum.py` with one opt-in class; relevant focused tests; `docs/neural-combat.md`, README, new `docs/neural-curriculum.md`, and new `scripts/neural/curriculum-qualification-runbook.md`. No executable host wrapper or release is required until approved runtime scope exists.

- [ ] **Step1 — Integration fixtures:** Add an enabled-harness fake test and a gate-off test that prove actual API names/call binding with all six optional packages blocked. Author gated tests for public support/catalog normalization, tensor save/resume parity and one tiny healthy training segment; use a new flag `SPIRE_LEARN_RUN_NEURAL_CURRICULUM_INTEGRATION`, unset by default. Do not execute it enabled.
- [ ] **Step2 — Concrete proposal:** Document a fresh protected qualification phase with actual source/support/shard/parent identities, existing runtime pins, ≤512 native action slots, ≤512 actor samples, ≤8 Adam attempts, ≤256 reset slots,120-second soft/600-second whole limit,1GiB new outputs, unchanged40GiB aggregate/30GiB free/12GiB GPU floor. No Run effects or old claim reuse. Explain that a later three-arm replicated learning pilot gets its own reviewed budget; qualification alone cannot establish learning.
- [ ] **Step3 — Verification:** Run all covering pure suites, then the full default suite once with every `SPIRE_LEARN_RUN_*` flag unset. Fresh guard blocks torch/numpy/gymnasium/spire_lab/torchrl/tensordict during CLI help, validate-only and discovery. Verify all authored configs/recipes/panels through real parsers and source/protected hashes. No tests mirror implementation merely to raise counts.
- [ ] **Step4 — Documentation:** Explain current versus proposed coverage, sampler semantics, exact resume, initialization, generated/exposed counts, realistic-distribution limits, Run fault dependency and noncombat-controller boundary. All measured/runtime claims remain explicitly unverified for this feature. Link retained tests and proposal, never claim the old campaign completed.
- [ ] **Step5 — Finish:** Self-review/commit owned paths (`docs: document broad curriculum and held qualification`); parent arranges one final whole-branch source review and addresses concrete findings through the scoped fix loop. Preserve SDD workspace and all evidence. No training, installs, push or additional approval request inside pure implementation.

## Plan self-review and rulings

- Task1 owns breadth/validation and Task4 owns immutable evaluation; Task2 owns actual exposure, Task3 its continuation, Task4 real orchestration, Task5 source verification/proposal. All design sections have owners.
- Interfaces use the same Task1 shard and Task2 sampler throughout. Config is appended/default-None; no global weakening of legacy compatibility. Actual target versions remain the accepted public schema.
- Parent resolves ordinary implementation details against the approved design. Explicit new runtime authority is absent; preparing a concrete proposal is necessary before asking for it.
- The user approved design **for learner implementation**. Preserve the already selected in-place executor/reviewer method and proceed without another plan confirmation. This overrides the writing-plans default execution-handoff pause; the plan adds no new authorized effects.
