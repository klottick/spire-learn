# Evaluation and Artifact Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Preserve the user-selected Luna tool-worker workflow and parent synthesis/self-review; do not spawn nested worker agents.

**Goal:** Implement M1 fixed-case single-environment evaluation, public-native diagnostic probes, baseline policies, and inspectable artifacts without implementing a learner.

**Architecture:** Keep records, schema parsing, manifests, configuration, budgets, serialization, and aggregation standard-library-only. Isolate native dependencies in lazy adapters and gate real integrations on separately authorized setup and observed evidence; the early probes precede freezing collection contracts.

**Tech Stack:** Python 3.12 (`python3.12 -B`), `unittest`, `tomllib`, `argparse`, JSON/JSONL, lazy NumPy/Gymnasium/native/TorchRL adapters. No dependency pins or installs are included.

**Spec:** [M1 foundation design](../specs/2026-10-24-foundation-design.md); [approved master architecture](../specs/2026-10-24-spire-learn-architecture-design.md).

**Status:** User-approved plan; implementation in the main session is authorized. Native integration/setup remains separately gated as described below.

## Global Constraints

- Plan approved by the user; implementation on `main` is explicitly authorized. Approved M1 spec/master alone did not authorize execution.
- Fixed-case evaluator is single-environment only. Native vector is diagnostic/conformance/probe only; no refill/drain scheduler, learner training, optimizer, LLM/VOD, checkpoints, or game simulation.
- Python `>=3.12,<3.13`; package under `src/spire_learn`; package import/help/core tests have no optional dependency, device, or training side effects.
- Core tests use deterministic stdlib `unittest`. Do not add `pytest` as a dependency.
- Do not edit dependency metadata or install dependencies. Current Python metadata lacks NumPy, Gymnasium, Torch, TorchRL, pytest and spire-lab. Real integration requires explicit separate setup/compatibility authorization. If absent or dependencies are missing, stop and ask for a reviewed setup addendum; report blocked, not pass.
- Do not borrow a Gym virtual environment, alter Gym, copy its code/fixtures/dependencies/lockfiles/native artifacts, or use seeds, hidden RNG, unseen draw order or private diagnostics as actor inputs.
- Gym owns simulation, rules, observations, legal masks, reward, endings and resets. Test doubles are scripted protocol records only.
- Genuine death is an outcome; simulator faults are errors. Never retry accepted actions/effects or classify faults as losses/truncations.
- Keep train/selection/final separate; conservatively reject split overlap; final requires explicit `--allow-final`. Preserve RNG-001/TIME-001 qualifications; no transfer or strength claim.
- Preserve existing/user files and generated data. No installs, real integration, training, pushes or publication without separate authority. Local Git commits are authorized; commit task work after its tests within approved scope. Review fixes may follow commits.
- Plan review is complete and implementation is authorized on `main`. No tests are claimed run by this status update.

## Review Focus

1. **Import/help isolation:** no optional imports or install behavior. Task 9 test: `test_import_and_help_do_not_import_optional_dependencies`.
2. **Mixed vector endings:** final/reset observations and contexts remain distinct; absent final keys are valid only when no rows end. Tasks 3, 7 tests: `test_vector_routes_final_and_replacement_with_old_state`, `test_probe_verdict_requires_boundary_evidence`.
3. **Deadline during an in-flight step:** delivered terminal/truncation is preserved; future cases become not_started. Tasks 8–9 tests: `test_terminal_step_delivered_after_deadline_is_preserved`, `test_deadline_marks_remaining_cases_not_started`.
4. **Artifact/close errors:** no retry or overwrite; primary failure/outcome retained. Tasks 5, 8–9 tests: `test_writer_refuses_existing_run`, `test_trace_and_close_errors_preserve_terminal_result_without_retry`, `test_writer_error_preserves_prior_case_and_stops_without_retry`.
5. **Split leakage:** scenario/environment-seed overlap is rejected despite changed partition/policy seed; final is opt-in. Tasks 4, 9 tests: `test_split_overlap_ignores_partition_and_policy_seed`, `test_final_split_requires_explicit_opt_in`.

## File Map

All listed files are new unless marked Modify. Package initializers stay minimal and optional-import-free.

- `src/spire_learn/contracts.py`: immutable public records and contract exceptions.
- `src/spire_learn/schema.py`: stdlib descriptor parser and schema-derived candidates.
- `src/spire_learn/agents/base.py`, `baselines.py`: policy protocol and baselines.
- `src/spire_learn/collection/boundaries.py`, `single.py`: old/final/reset routing and fixed-case collector.
- `src/spire_learn/experiments/cases.py`, `config.py`, `budget.py`: manifest, strict TOML, budget.
- `src/spire_learn/artifacts/codec.py`, `writer.py`, `provenance.py`: safe JSON, run writing, provenance.
- `src/spire_learn/envs/native.py`: lazy public Gym facade, snapshots, action validation/packing, info and typed error mapping.
- `src/spire_learn/probes/vector.py`, `torchrl.py`: diagnostic probes, no optimizer.
- `src/spire_learn/evaluation/report.py`, `runner.py`: result accounting and single-env schedule.
- `src/spire_learn/cli.py`, `__main__.py`: dependency-free help/validation and guarded commands.
- `tests/support.py`, `tests/unit/test_{contracts,schema,baselines,boundaries,cases,config,budget,artifacts,provenance,native_contract,probes,single,report,runner,cli}.py` plus test package initializers.
- Opt-in only: `tests/integration/test_{native_envs,vector_probe,torchrl_adapter,evaluation}.py` and package initializer.
- Authored data/config: `configs/cases/selection-smoke.json`, `configs/evaluation/{smoke,development,overnight}.toml`, `configs/probes/{vector-smoke,torchrl-smoke}.toml`.
- `docs/development.md`; Modify `README.md` only in Task 10 if evidence justifies a status change.
- `pyproject.toml` and dependency metadata remain unchanged; changes require separately authorized setup addendum.

## Shared Interface Ledger

Use these names/signatures across tasks. Do not invent helpers in later tasks. Collection interfaces remain provisional until Task 7 probe evidence and user review.

### Records and schema — Task 1

- `ArrayValue(dtype: str, shape: tuple[int, ...], values: tuple[int | float, ...])`, frozen; validate shape/count, numeric primitives and finiteness; reject bool-as-int and unsupported objects.
- `Observation = Mapping[str, ArrayValue]`; `frozen_mapping(values: Mapping[str, ArrayValue]) -> Observation` returns immutable owned mapping.
- `Candidate(index: int, kind: str, source: int | None, target: int | None, features: tuple[tuple[str, int], ...])`.
- `PublicDecision(observation: Observation, candidates: tuple[Candidate, ...], mask: tuple[bool, ...], schema_fingerprint: str, action_mapping_id: str)`.
- `Route(execution_id: str, slot: int, episode_id: str, decision_id: str, episode_start: bool)` is orchestration-only.
- `DecisionRequest(public: PublicDecision, route: Route)`; `Choice(action: int, next_state: object | None)`; `DecisionResult(decision_id: str, choice: Choice)`.
- `Transition(route: Route, source: Observation, action: int, reward: float, terminated: bool, truncated: bool, successor: Observation, replacement: Observation | None, state_after_source: object | None)`. Context is after processing source and before old successor/final, never replacement state. Opaque context is not serialized by default.
- `CaseExecution(execution_id: str, case_id: str, split: str, partition: str, scenario_id: str, environment_seed: int, policy_seed: int, expected_schema_id: str | None = None, expected_fingerprint: str | None = None)`.
- `Failure(kind: str, reason: str, phase: str, exception_type: str | None = None)`.
- `CaseResult(execution: CaseExecution, status: str, started: bool, outcome: str | None, steps: int, reward_sum: float, stop_reason: str, final_resources: Mapping[str, int | None], primary_failure: Failure | None = None, secondary_errors: tuple[Failure, ...] = (), observed_descriptor: Mapping[str, object] | None = None, native_build_info: Mapping[str, object] | None = None)`.
- Outside-start statuses: `not_started`, `setup_error`. Started statuses: `terminal`, `env_truncated`, `budget_stopped`, `agent_error`, `simulator_fault`, `contract_error`, `exec_error`. Outcome only delivered verified `victory`, `death`, or null.
- Define `ContractError` and `SetupBlocked` once in `contracts.py`.
- `CategorySpec = str | tuple[str | None, ...] | None`; `TensorSpec(key: str, dtype: str, shape: tuple[int, ...], fields: tuple[str, ...], axes: tuple[str, ...], category: CategorySpec, mask: str | None)`. A category tuple aligns vocabulary names/None with the named `field` axis (for example public `context` metadata); `mask` is a descriptor field reference such as `action_mask` or `enemy_damage_mask:intent_damage`, not a boolean.
- `ObservationSchema(schema_id: str, content_id: str, fingerprint: str, action_count: int, tensors: tuple[TensorSpec, ...], vocabularies: Mapping[str, tuple[str, ...]])`.
- `parse_descriptor(raw: Mapping[str, object], *, allowed_schema_ids: frozenset[str]) -> ObservationSchema` reads descriptor root `schema_id`, `content_id`, `action_space.n`, tensors (`key/dtype/shape/fields/axes/category/mask`), vocabularies and binding-provided hex fingerprint. Never recompute Gym fingerprint. Validate required keys, unique tensor/field names, category-to-vocabulary references and field counts against named field axes where those axes exist. Runtime observation category codes are checked during candidate/snapshot validation. Do not assume action-kind vocabulary length equals candidate/action-space count.
- `candidates_for(observation: Observation, schema: ObservationSchema) -> tuple[Candidate, ...]` derives candidates from action mask width/action-space count; resolves kind codes through descriptor vocabulary; extracts feature/source/target fields by names/axes. Category vocabulary cardinality is independent of candidate-slot count. Do not hardcode `n=32`, enum IDs, or descriptor column positions.

### Test support — Task 1

- `tiny_descriptor(*, action_count: int = 3) -> dict[str, object]`: authored fixture schema `spire-learn-fixture-v1`, content `spire-learn-fixture-content-v1`, fingerprint `'a'*64`, action vocabulary exactly three kind labels, outcome vocabulary where integer code 1 means `victory`, candidate tensors sized from `action_count`, named axes/string mask references; action-mask tensor has `mask=None`. Native allowlist never includes fixture schema.
- `observation(*, mask: tuple[int, ...] = (1, 1, 0), kinds: tuple[int, ...] = (1, 2, 0), hp: int = 10) -> Observation`; require equal mask/kinds widths, derive candidate metadata widths from mask; no simulated dynamics.
- `FakeArray(dtype_name: str, shape: tuple[int, ...], values: tuple[int | float, ...], *, isnative: bool = True)` exposes `.dtype.name`, `.dtype.isnative`, `.shape`, `.tolist()`; numeric-only public-array fixture, not NumPy emulator.
- `ScriptedEnv(reset_result: tuple[object, dict], steps: Sequence[tuple[object, float, bool, bool, dict] | Exception], *, descriptor: Mapping[str, object] | None = None, on_step: Callable[[], None] | None = None)`. Methods `reset(*, seed=None, options=None)`, `step(action)`, `close()`, `observation_spec()`. Logs: `reset_calls` stores `(seed, options)`; `step_calls` stores actions (append before scripted exception); integer `close_calls`; default descriptor is `tiny_descriptor()`. `on_step()` runs only when a result is delivered. No game dynamics.
- `FakeClock(start: float = 0)` is callable; `advance(delta: float) -> None`.
- `ScriptedAgent(choices: Sequence[Choice | Exception])` implements `init_state() -> object | None` and `act(public: PublicDecision, state: object | None) -> Choice`.
- `FakeNativeErrors` in support defines only concrete `InputError`, `StateError`, `SetupError`, `FaultError` exception classes; Task 6 constructs `NativeErrorTypes` explicitly from them.
- `case_execution(**overrides: object) -> CaseExecution` supplies valid fixture IDs, selection split/partition, fixture expected schema and independent seeds; named overrides replace fields.
- `case_result(**overrides: object) -> CaseResult` defaults to terminal started victory, one step/reward and a `case_execution()`; override execution for unique IDs.
- `public_decision(*, mask: tuple[int, ...] = (1, 1, 0), kinds: tuple[int, ...] = (1, 2, 0)) -> PublicDecision` builds immutable observation/candidates using a descriptor sized to mask.
- `fixture_contract() -> EnvironmentContract` allows exactly `spire-learn-fixture-v1` / `spire-learn-fixture-content-v1` and fixture-protocol reward label; fixture `action_mapping_id` is `spire-learn-fixture-actions-v1`.

### Policies, routing, cases and config — Tasks 2–4

- In `agents/base.py`: `Policy.init_state() -> object | None`, `Policy.act(public: PublicDecision, state: object | None) -> Choice`, `resolve_decision(request: DecisionRequest, policy: Policy, state: object | None) -> DecisionResult`, `validate_result(request: DecisionRequest, result: DecisionResult) -> None`. Result validation rejects bool, illegal/padding and stale ID.
- In `agents/baselines.py`: `RandomPolicy(seed: int)`, `PlayFirstPolicy()`, each `init_state() -> None`. Random owns `random.Random(seed)`; legal-random uses current legal tuple. Play-first uses lowest legal `play_card`, else lowest legal `end_turn`; no game rules.
- `BoundaryResult(transition: Transition, active_observation: Observation, active_state: object | None, episode_started: bool)`. A single ended row has `episode_started=False`, `active_state=None`, and its terminal `active_observation` is not acted on. A vector ended row retains old final/context in `transition`, while its active replacement has `episode_started=True` and reset `active_state`. An ongoing vector row has `episode_started=False` and unchanged state.
- `route_single_step(*, route: Route, source: Observation, action: int, reward: float, terminated: bool, truncated: bool, successor: Observation, state_after_source: object | None) -> BoundaryResult`.
- `route_vector_step(*, routes: Sequence[Route], sources: Sequence[Observation], actions: Sequence[int], rewards: Sequence[float], terminated: Sequence[bool], truncated: Sequence[bool], main_observations: Sequence[Observation], final_observations: Sequence[Observation | None], final_presence: Sequence[bool], final_info_presence: Sequence[bool], states_after_source: Sequence[object | None], reset_state: Callable[[int], object | None]) -> tuple[BoundaryResult, ...]`. Validate row widths and matching presence masks. Ended rows route final to old successor and main to replacement; live rows use main successor. Reset ended slot state before replacement processing, retain old state in transition. Missing final keys are valid when no row ended.
- `CaseManifest(path: Path, split: str, executions: tuple[CaseExecution, ...])`; `load_cases(path: Path) -> CaseManifest`; `validate_splits(manifests: Sequence[CaseManifest]) -> None`.
- Manifest root schema `spire-learn-cases-v1`, split `train|selection|final`, case records with unique case ID, scenario ID, partition, optional expected schema/fingerprint and explicit executions (unique execution ID, environment seed, policy seed). Reject bool/negative seeds. Repetitions are explicit executions; seeds never derived or carried across cases. Reject duplicate case IDs within/across manifests; repeated executions within one declared case/split are allowed. Conservative cross-split collision key is `(scenario_id, environment_seed)`, ignoring partition/policy seed/content claims; this may reject independent content variants. Do not claim missing content proves independence. Content/profile is provenance, not a `CaseExecution` field.
- `ProbeSettings(num_envs: int, workers: int)` and `ProbeConfig(num_envs: int, workers: int, partition: str, executions: tuple[CaseExecution, ...], max_batch_steps: int, max_wall_seconds: float)` are stdlib dataclasses defined in `experiments/config.py`; Task 4 owns both, so `to_probe_config` can construct its return value before probe implementation exists. Task 7 imports `ProbeConfig` from `experiments.config` (never the reverse).
- `RunConfig(mode: str, profile: str, agent: str, cases: Path, output_root: Path, per_case_decisions: int | None, max_batch_steps: int | None, max_wall_seconds: float, traces: bool, expected_partition: str | None = None, expected_schema_id: str | None = None, expected_fingerprint: str | None = None, gym_checkout: Path | None = None, split_guard_manifests: tuple[Path, ...] = (), probe: ProbeSettings | None = None)`.
- `load_config(path: Path, *, expected_mode: str) -> RunConfig` strictly validates TOML, unknown fields and bool/numeric types; resolves relative paths from config and permits explicitly supplied absolute paths. Required root keys: `mode`, `profile`, `agent`, `cases`, `output_root`, `max_wall_seconds`, `traces`. Evaluate requires positive `per_case_decisions`, forbids `max_batch_steps` and `[probe]`; probe modes require positive `max_batch_steps` and `[probe] num_envs, workers`, forbid `per_case_decisions`, and require `num_envs >= 2`. `max_wall_seconds` is positive finite for all modes. Config partition/schema are assertions, not overrides.
- `validate_expectations(config: RunConfig, manifest: CaseManifest) -> None` performs static config/manifest split, partition and expected-schema/fingerprint conflict checks before environment factory.
- `to_probe_config(config: RunConfig, manifest: CaseManifest) -> ProbeConfig` requires probe settings, exactly `num_envs` explicit executions in manifest, one partition and selection split; scenario IDs and both seeds come from manifest. No probe TOML seed arrays or partition override.
- Task 4b binding erratum: `bind_case_expectations(config, execution) -> CaseExecution` rejects declared conflicts (including asserted partition) and returns a copy filling only missing schema ID/fingerprint assertions. Task 9 must pass these bound cases to `run_case`; `validate_expectations` alone does not return the bound cases. Probe conversion also binds, without synthesizing or overriding execution identity/seeds/partition.
- Task 4c budget: `Budget(max_wall_seconds, clock)` validates positive finite primitive numeric duration and finite primitive numeric clock reads (excluding bool), captures its start once, and provides clamped `remaining()` / `exhausted()` soft checks. It does not interrupt effects or track batch/slot/step counts.
- `Budget(max_wall_seconds: float, clock: Callable[[], float])`; `exhausted() -> bool`, `remaining() -> float`; clock is injectable. Per-case accepted step limits are distinct from vector batch-step limits.

### Environment contract — Task 1 schema, Task 6 native adapter

- `EnvironmentContract(schema_ids: frozenset[str], content_ids: frozenset[str], reward_contract_id: str, reward_evidence: str)` describes allowed profile metadata; it is not an environment API.
- `fixture_contract() -> EnvironmentContract` allows only fixture schema/content and a fixture-protocol reward label. `NATIVE_CONTRACT` in `envs/native.py` allows only the documented native schema/content and documented reward profile. Production CLI cannot configure fixture IDs.

### Artifacts, environment and probes — Tasks 5–7

- `to_json_value(value: object) -> object`: JSON primitives, mappings/sequences and explicit whitelist `ArrayValue`, `Candidate`, `CaseExecution`, `CaseResult`, `Failure`; rejects other dataclasses, opaque state, arbitrary objects and non-finite values. `canonical_fingerprint(value: object) -> str`: sorted compact JSON SHA-256 for learner-owned fingerprints only.
- `transition_to_json(transition: Transition) -> dict[str, object]` emits public source/action/reward/flags/successor/replacement/routing and the transition schema ID, explicitly excluding `state_after_source`.
- `RunWriter.create(root: Path, run_id: str) -> RunWriter`; methods `write_start(config: Mapping[str, object], manifest: Mapping[str, object]) -> None`, `write_case(result: CaseResult) -> None`, `write_metric(record: Mapping[str, object]) -> None`, `write_transition(transition: Transition) -> None`, `update_manifest(record: Mapping[str, object]) -> None`, `finalize(summary: Mapping[str, object]) -> None`, `close() -> None`. Files: `config.json` uses config envelope ID, `manifest.json` cases ID, each `episodes.jsonl` result uses evaluation ID, transition/metric lines use transition/probe IDs, `summary.json` uses evaluation ID. Refuse existing path/unsafe ID, flush JSONL, final summary temp/replace inside owned dir. `update_manifest` is append-only discovery, not rewrite of initial identity.
- `collect_provenance(*, learner_root: Path, gym_checkout: Path | None) -> dict[str, object]`: read-only Git subprocess and `importlib.metadata`, no optional imports; exact keys `{learner: {revision, head_state, dirty}, gym_checkout: {revision, dirty}|None, distributions: {name: version|null}, native_build_info: null}`. No inferred native version/hash.
- Persistent schema IDs: `spire-learn-decision-v1`, `spire-learn-transition-v1`, `spire-learn-cases-v1`, `spire-learn-evaluation-v1`; config/probe envelope IDs `spire-learn-config-v1`, `spire-learn-probe-v1`. Writer supplies envelope IDs; records need no undefined schema property.
- Profile constants: schema `spire-lab-combat-observation-v1`, content `spire-lab-combat-reference-v2`, documented reward profile `spire-lab-combat-winloss-v1`; reward evidence is documentation, not a nonexistent runtime reward API.
- `NativeErrorTypes(input: type[Exception], state: type[Exception], setup: type[Exception], fault: type[Exception])`.
- `validate_actions(actions: Sequence[int]) -> tuple[int, ...]` is pure stdlib; rejects bool, non-int, negative values. Current-mask/action-count legality is checked by caller before packing.
- `make_single(case: CaseExecution) -> object`; `make_vector(num_envs: int, partition: str, workers: int = 1) -> object`; lazy imports public `spire_lab.gymnasium`, uses `catalog=None`, replay off, manifest partition. Load actual native exception types only after importing authorized binding.
- `native_snapshot(raw: Mapping[str, object], schema: ObservationSchema) -> Observation` accepts either already immutable `Observation` (validated) or public numeric array-like objects exposing dtype name/native flag, shape, and `tolist()`. Flatten and validate with stdlib against descriptor; no NumPy import required. Do not silently coerce unknown/object arrays.
- `pack_actions(actions: Sequence[int]) -> object` validates first and lazily returns native-endian NumPy `int64` vector. Its dtype/shape assertion is opt-in integration only.
- `normalize_public_info(info: Mapping[str, object], schema: ObservationSchema) -> Mapping[str, object]` returns only delivered public outcome/steps/resources.
- `native_failure(error: BaseException, *, error_types: NativeErrorTypes | None = None) -> Failure | None` maps only injected or already-loaded actual typed classes; no class-name matching. No types/no match returns `None`; arbitrary RuntimeError is never simulator fault.
- `ProbeReport(verdict: str, checks: Mapping[str, str], versions: Mapping[str, str | None], batch_steps: int, slot_decisions: int, replacement_decisions: int, failures: tuple[Failure, ...])`.
- `probe_report_to_json(report: ProbeReport) -> dict[str, object]` in `probes/vector.py` explicitly maps every report field and encodes each `Failure` through the safe codec. `ProbeReport` is not added to the generic dataclass whitelist. `run_vector_probe` and `run_torchrl_probe` use this mapping for `writer.write_metric` and the final summary; do not use blanket `dataclasses.asdict` encoding.
- `public_build_info() -> Mapping[str, object] | None` in `envs/native.py` calls only the documented `spire_lab._native.build_info` getter on an already-loaded binding module (`sys.modules` lookup only); otherwise returns `None`, without optional imports. The `public_native` name in the original ledger was a placeholder/path defect, not a Gym export. The same loaded binding's documented exception exports are resolved only if all four exact typed classes are present; never import it for classification. Fake-environment core tests inject a test module under this qualified name but do not load the Gym binding. `run_case` populates `CaseResult.observed_descriptor` and `native_build_info` where available; `evaluate` appends recorded discovery through `RunWriter.update_manifest` (there is no `on_schema` callback).
- `NATIVE_CHECKS` in `probes/vector.py`: `mixed_rows`, `old_final`, `replacement_reset`, `presence_masks`, `old_context`. `TORCHRL_CHECKS` in `probes/torchrl.py`: native checks plus `observation_mapping`, `mask_sampling`, `mask_log_prob`.
- `probe_verdict(checks: Mapping[str, str], *, required: frozenset[str], runtime_available: bool, adaptation_verified: bool = False) -> str`: runtime absent/missing or unavailable required checks -> blocked; any required failure -> incompatible even if adaptation flag set; all checks pass -> adaptable only for demonstrated bounded conversion, otherwise compatible.
- `run_vector_probe(config: ProbeConfig, writer: RunWriter, *, clock: Callable[[], float]) -> ProbeReport`; `run_torchrl_probe(config: ProbeConfig, writer: RunWriter, *, clock: Callable[[], float]) -> ProbeReport`. Opt-in runtime functions use real APIs; unit tests test pure report/verdict helpers only. Each probe persists report by `writer.write_metric({"schema_id": "spire-learn-probe-v1", ...})` and includes final report in summary; native vector verdict does not imply TorchRL verdict.

### Collector, evaluator and CLI — Tasks 8–10

- `run_case(case: CaseExecution, policy: Policy, *, env_factory: Callable[[CaseExecution], object], budget: Budget, max_decisions: int, on_transition: Callable[[Transition], None] | None = None, classify_failure: Callable[[BaseException], Failure | None] = native_failure, contract: EnvironmentContract = NATIVE_CONTRACT) -> CaseResult`. Scripted tests inject `fixture_contract()`; production CLI cannot.
- `summarize(results: Sequence[CaseResult]) -> dict[str, object]` returns scheduled, started, not_started, setup_failures, status_counts, outcome_counts, completed_game_win_rate, terminal_completion_coverage, secondary_failure_count. Rates null for zero denominator; scheduled equals started plus not_started.
- `EvaluationReport(summary: Mapping[str, object], exit_code: int, run_dir: Path | None, fatal_error: Failure | None)`.
- `evaluate(config: RunConfig, *, allow_final: bool = False, env_factory: Callable[[CaseExecution], object] = make_single, clock: Callable[[], float] = time.monotonic, writer_factory: Callable[[Path, str], RunWriter] = RunWriter.create, run_id: str | None = None, contract: EnvironmentContract = NATIVE_CONTRACT) -> EvaluationReport` runs cases sequentially in single envs, creates fresh policy RNG per execution from manifest policy seed, passes the injected `contract` through to `run_case`, and preserves every result. Runner tests inject `fixture_contract()` with `ScriptedEnv`; production CLI always uses `NATIVE_CONTRACT` and accepts no fixture allowed IDs through config or flags.
- `main(argv: Sequence[str] | None = None) -> int`; CLI `evaluate`, `probe-vector`, `probe-torchrl`, `--help`; only evaluate supports `--allow-final`; `--validate-only` has no env/run-directory side effects. `__main__.py` invokes only under main guard.
- Exit 0 only when all executions naturally complete (victories and deaths are valid) and no execution/secondary reliability errors; 2 for preflight validation/setup/blocked with no execution failure; 1 for execution failure, budget-unfinished, fatal artifact or secondary reliability failure. Mixed setup plus execution failure => 1. Before-reset expiry is `not_started`; a started ongoing case may be `budget_stopped`. CLI returns `EvaluationReport.exit_code`.

## Task 1: Immutable Contracts and Descriptor Schema

**Depends on:** none. **Files:** `src/spire_learn/contracts.py`, `src/spire_learn/schema.py`, package initializers, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/support.py`, `tests/unit/test_contracts.py`, `tests/unit/test_schema.py`.

**Interfaces:** Records/schema/test-support from ledger. Parser must represent native 3 action-kind labels independently of candidate count.

- [ ] **Step 1 — Red:** add tests including this exact distinction, malformed schema and immutable records:

```python
def test_action_kind_vocabulary_is_not_candidate_count(self):
    descriptor = tiny_descriptor(action_count=32)
    schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
    self.assertEqual(schema.action_count, 32)
    self.assertEqual(len(schema.vocabularies["action"]), 3)
    self.assertIsNone(next(t for t in schema.tensors if t.key == "action_mask").mask)
    self.assertEqual(next(t for t in schema.tensors if t.key == "action_kind").mask, "action_mask")
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_contracts tests.unit.test_schema -v`; expected missing symbols/failing assertions.
- [ ] **Step 3 — Implement:** frozen owned numeric records, string mask reference and axes-aware schema parser/candidate extraction. Validate category codes using vocabulary, mask width independently against action count, and named field-axis counts only where declared. Never infer vocab/action-space equality.
- [ ] **Step 4 — Green:** repeat exact command; expected all Task 1 tests pass without optional imports. `test_fixture_contract_is_explicitly_scoped` asserts fixture schema/content allowed and native schema not allowed. Candidate category codes are validated against vocabularies when observations are parsed, not when descriptor metadata is parsed.

## Task 2: Policy Protocol and Baselines

**Depends on:** Task 1. **Files:** `src/spire_learn/agents/base.py`, `src/spire_learn/agents/baselines.py`, `tests/unit/test_baselines.py`.

**Interfaces:** `Policy.init_state`, `Policy.act`, `resolve_decision`, `validate_result` are in `agents/base.py`; baselines in `agents/baselines.py`.

- [ ] **Step 1 — Red:** create named tests `test_random_policy_matches_independent_seeded_legal_sequence`, `test_play_first_uses_lowest_legal_kind`, `test_empty_nonterminal_mask_errors`, `test_result_rejects_bool_illegal_padding_and_stale_id`, `test_policy_receives_no_route_or_seed`.
- [ ] **Step 1a — Test assertion:**

```python
public = public_decision(mask=(1, 0, 1), kinds=(1, 0, 2))
policy, expected = RandomPolicy(19), random.Random(19)
self.assertEqual([policy.act(public, None).action for _ in range(8)],
                 [expected.choice((0, 2)) for _ in range(8)])
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_baselines -v`; expected failure.
- [ ] **Step 3 — Implement** the exact protocol and baseline signatures in ledger; random sequence uses its own `random.Random(seed)`, play-first resolves descriptor kind only. No fallback or action-effect rules.
- [ ] **Step 4 — Green:** same command; assert expected choices against independent `random.Random(seed).choice(legal)` and captured policy arguments contain `PublicDecision`/state but no route or seeds.
- [ ] **Step 5 — Import check:** `PYTHONPATH=src python3.12 -B -c 'import sys; import spire_learn.agents.baselines; assert not any(n in sys.modules for n in ("numpy", "torch", "torchrl", "gymnasium", "spire_lab"))'`; expected exit 0.

## Task 3: Boundary Routing

**Depends on:** Tasks 1–2. **Files:** `src/spire_learn/collection/boundaries.py`, `tests/unit/test_boundaries.py`.

**Interfaces:** `BoundaryResult`, `route_single_step`, `route_vector_step` from ledger.

- [ ] **Step 1 — Red:** create `test_single_step_preserves_flags`, `test_vector_routes_final_and_replacement_with_old_state`, `test_vector_live_row_preserves_state`, `test_missing_final_keys_allowed_when_no_row_ends`, `test_presence_masks_must_match`, `test_terminal_empty_mask_is_not_acted_on`, `test_cut_does_not_reset_memory`. Assert a single ended row has `episode_started=False`, `active_state=None`, and terminal `active_observation` is not acted on; a vector ended row retains old final/context in `transition` while its active replacement has `episode_started=True` and reset `active_state`; an ongoing vector row has `episode_started=False` and unchanged state. No new field or live flag.
- [ ] **Step 1a — Fixture interface:** add `vector_step_fixture() -> dict[str, object]` to `tests/support.py`. It returns exactly the keyword arguments accepted by `route_vector_step`: two routes/sources/actions/rewards/flag rows/main observations/final observations/presence masks/states and reset callback. Row 0 terminates with final HP 10 and old state 7; row 1 continues with state 9; reset callback returns 0 only for row 0. This contains protocol records only.
- [ ] **Step 1b — Test assertion:**

```python
rows = route_vector_step(**vector_step_fixture())
self.assertEqual((rows[0].transition.state_after_source, rows[0].active_state), (7, 0))
self.assertEqual(rows[1].active_state, 9)
self.assertIsNot(rows[0].transition.successor, rows[0].transition.replacement)
self.assertEqual((10 + rows[0].transition.state_after_source, 10 + rows[0].active_state), (17, 10))
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_boundaries -v`; expected failure.
- [ ] **Step 3 — Implement** pure routing with row-width checks. Ended transition retains old final and old state; main is replacement and reset callback applies before its decision. Live rows retain main successor and state.
- [ ] **Step 4 — Green:** same command. In `test_vector_routes_final_and_replacement_with_old_state`, assert old final HP 10 + old state 7 = 17 while replacement HP 10 + reset state 0 = 10; assert reset callback is called only for ended row. No trainer/value-target helper is introduced.

## Task 4: Manifests, Config and Budget

**Depends on:** Task 1. **Files:** `src/spire_learn/experiments/cases.py`, `src/spire_learn/experiments/config.py`, `src/spire_learn/experiments/budget.py`; `tests/unit/test_cases.py`, `tests/unit/test_config.py`, `tests/unit/test_budget.py`.

**Interfaces:** `CaseManifest`, manifest functions, `ProbeSettings`, `ProbeConfig`, `RunConfig`, `load_config`, `validate_expectations`, `to_probe_config`, `Budget` from ledger. Both probe dataclasses are defined in `experiments/config.py`; `to_probe_config` constructs `ProbeConfig` directly, and Task 7 imports it from this module.

- [ ] **Step 1 — Red cases:** create `test_manifest_preserves_explicit_seeds`, `test_bool_and_negative_seeds_rejected`, `test_duplicate_case_and_execution_ids_rejected`, `test_cross_split_case_id_rejected`, `test_split_overlap_ignores_partition_and_policy_seed`, and `test_same_scenario_different_environment_seed_is_allowed`.
- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_cases -v`; expected failure.
- [ ] **Step 3 — Implement** declared JSON schema and conservative `(scenario_id, environment_seed)` cross-split collision. No content field on `CaseExecution`; content/profile is provenance only. Repeated explicit executions in one case/split remain allowed.
- [ ] **Step 4 — Green cases:** repeat command; assert duplicate case IDs reject while distinct execution IDs under one repeated case preserve each policy seed independently.
- [ ] **Step 5 — Red config:** tests cover strict unknown/bool numeric rejection, relative paths, mode mismatch, assertion-only partition/schema conflicts, probe-table mode rules, seed non-overrides, positive finite caps, exact B manifest execution count and one partition.
- [ ] **Step 5a — Test assertion:**

```python
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "bad.toml"
    path.write_text('mode="evaluate"\nprofile="smoke"\nagent="random"\n'
                    'cases="cases.json"\noutput_root="runs"\n'
                    'max_wall_seconds=60.0\ntraces=false\nper_case_decisions=true\n')
    with self.assertRaises(ValueError):
        load_config(path, expected_mode="evaluate")
```

- [ ] **Step 6 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_config -v`; expected failure.
- [ ] **Step 7 — Implement** TOML root fields, `ProbeSettings` and `ProbeConfig` stdlib dataclasses in `experiments/config.py`, and `[probe] num_envs, workers`; probe modes require that table and `num_envs>=2`; evaluate rejects it. `to_probe_config` takes exactly B explicit manifest executions and constructs `ProbeConfig`; no seed/partition fields in probe TOML. Config expected values are static assertions only. Task 7 imports `ProbeConfig` from this module.
- Task 4b Task 9 consumer pointer: scheduler must pass `bind_case_expectations` results to `run_case`, not only validate the original manifest.
- [ ] **Step 8 — Green config:** repeat command; static mismatch errors occur before env factory (tested in Task 9); runtime descriptor mismatch is a separate post-construction check.
- [ ] **Step 9 — Red budget:** `test_budget_remaining_and_expiry` uses FakeClock to assert remaining at start/cap, exhaustion and zero-clamped remaining after expiry; no batch/slot counters are assigned to Budget.
- [ ] **Step 9a — Test assertion:**

```python
clock = FakeClock()
budget = Budget(3.0, clock)
clock.advance(4.0)
self.assertTrue(budget.exhausted())
self.assertEqual(budget.remaining(), 0.0)
```

- [ ] **Step 10 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_budget -v`; expected failure.
- [ ] **Step 11 — Implement:** implement `Budget` only; probe batch/slot counters belong to Task 7 report; accepted step count belongs to Task 8 result.
- [ ] **Step 12 — Green:** repeat `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_budget -v`; expected pass.

## Task 5: Artifacts and Provenance

**Depends on:** Tasks 1 and 4. **Files:** `src/spire_learn/artifacts/codec.py`, `src/spire_learn/artifacts/writer.py`, `src/spire_learn/artifacts/provenance.py`; `tests/unit/test_artifacts.py`, `tests/unit/test_provenance.py`.

**Interfaces:** Ledger codec, RunWriter (including append-only `update_manifest`) and `collect_provenance`.

- [ ] **Step 1 — Red codec:** create `test_array_json_preserves_dtype_shape_values`, `test_codec_rejects_opaque_nonfinite_and_arbitrary_values`, and `test_canonical_fingerprint_ignores_mapping_order`; assertions below pin order and unsupported-object behavior.
- [ ] **Step 1a — Test assertion:**

```python
self.assertEqual(canonical_fingerprint({"a": 1, "b": 2}),
                 canonical_fingerprint({"b": 2, "a": 1}))
with self.assertRaises(TypeError):
    to_json_value(object())
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_artifacts -v`; expected missing symbols.
- [ ] **Step 3 — Implement codec:** explicit JSON conversion only, with record whitelist and `transition_to_json` excluding opaque context.
- [ ] **Step 4 — Green codec:** repeat `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_artifacts -v`; expected pass.
- [ ] **Step 5 — Red writer:** create `test_writer_flushes_jsonl`, `test_writer_refuses_existing_run`, `test_writer_rejects_unsafe_id`, `test_summary_replace_is_atomic`, `test_manifest_discovery_is_append_only`, and `test_writer_error_does_not_retry_action`. Use `tempfile.TemporaryDirectory()` and `pathlib.Path` from stdlib. Assert second create raises `FileExistsError` and original `config.json` bytes are unchanged.
- [ ] **Step 6 — Run red writer:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_artifacts -v`; expected writer tests fail.
- [ ] **Step 7 — Implement writer:** exclusive run directory, flushed output, stable schema envelopes, safe IDs, atomic summary; errors never retry environment actions.
- [ ] **Step 8 — Green writer:** repeat exact command; expected pass.
- [ ] **Step 9 — Red provenance:** tests cover unborn/dirty unavailable state, metadata lookup without optional imports, separate checkout/build identities and exact result keys from ledger.
- [ ] **Step 10 — Run red provenance:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_provenance -v`; expected failure.
- [ ] **Step 11 — Implement provenance:** read-only Git subprocess and `importlib.metadata`; unavailable values explicit.
- [ ] **Step 12 — Green provenance:** repeat exact command; expected pass without optional imports.

## Task 6: Lazy Public Native Adapter and Gated Integration

**Depends on:** Tasks 1 and 5. Actual runtime requires separately authorized isolated setup. **Files:** `src/spire_learn/envs/native.py`, `tests/unit/test_native_contract.py`, opt-in `tests/integration/test_native_envs.py`.

**Interfaces:** Native adapter signatures and `NativeErrorTypes` in ledger. Core uses FakeArray/public immutable observations and injected exact error classes; no optional imports.

- [ ] **Step 1 — Red core tests:** create `test_import_is_lazy`, `test_snapshot_copies_public_numeric_array`, `test_snapshot_rejects_object_shape_and_endian_mismatch`, `test_validate_actions_rejects_bool_float_and_negative`, `test_native_failure_uses_injected_types_only`, `test_info_allowlist_and_fixture_outcome_decode`, `test_unsupported_schema_is_setup_error`.
- [ ] **Step 1a — Test assertion:**

```python
self.assertRaises(ContractError, validate_actions, [0, True])
schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
obs = observation()
raw = {key: FakeArray(value.dtype, value.shape, value.values) for key, value in obs.items()}
self.assertEqual(native_snapshot(raw, schema)["action_mask"], obs["action_mask"])
errors = NativeErrorTypes(input=FakeNativeErrors.InputError, state=FakeNativeErrors.StateError,
                          setup=FakeNativeErrors.SetupError, fault=FakeNativeErrors.FaultError)
self.assertEqual(native_failure(FakeNativeErrors.FaultError(), error_types=errors).kind, "simulator_fault")
self.assertIsNone(native_failure(RuntimeError("not a typed simulator fault"), error_types=errors))
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_native_contract -v`; expected failure.
- [ ] **Step 3 — Implement** pure stdlib `validate_actions`, snapshot support for numeric public array-like protocol, `native_failure(..., error_types=...)`; actual exception types loaded only with authorized binding. Construct `NativeErrorTypes` explicitly in this task, not in Task 1. `pack_actions` imports NumPy lazily. No fake NumPy helper.
- [ ] **Step 4 — Green core tests:** repeat command; expected pass with no NumPy/Gym/native imports. Actual NumPy ndarray dtype=`int64`/shape assertions are not core tests.
- [ ] **Step 5 — Authority gate:** if missing authority or dependencies, stop; do not install/borrow env/edit metadata. Record blocked, not integration pass. Source Gymnasium 1.4.0 tested subset is not a learner pin; source NumPy/Torch versions are not automatic dependencies.
- [ ] **Step 6 — Only with authority, add opt-in tests** for supported public descriptor/profile, both documented starter scenarios, reset/close, masks/action dtype and delivered public outcome/reward-profile evidence. Bound integration case to 128 steps. Fixture info uses integer outcome 1 decoded through fixture vocabulary to `victory`; native integration uses actual descriptor vocabulary. Use `catalog=None`, replay off; no private fault hooks or catalog enumeration.
- [ ] **Step 7 — Run authorized integration only:** `SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_native_envs -v`. `$PYTHON312` must be explicitly set to the authorized interpreter path; otherwise stop. If flag is absent, tests report blocked/not-run; zero discovered tests is not pass.

## Task 7: Native Vector and TorchRL Probes; Contract Gate

**Depends on:** Tasks 1–6. Runtime requires Task 6 authority. Basic stdlib artifacts/reports/CLI are not blocked by this gate; single collector contract freeze is. **Files:** `src/spire_learn/probes/vector.py`, `src/spire_learn/probes/torchrl.py`, `tests/unit/test_probes.py`, opt-in `tests/integration/test_vector_probe.py`, `tests/integration/test_torchrl_adapter.py`.

**Interfaces:** `ProbeConfig` imported from `experiments.config` plus report/verdict/probe functions from ledger; no dependency from config back into probes.

- [x] **Step 1 — Red pure tests:** create `test_probe_verdict_requires_boundary_evidence`, `test_native_and_torchrl_require_distinct_check_sets`, and `test_probe_report_counts_batches_slots_and_replacements`. Missing required checks are blocked; runtime unavailable is blocked; failed required checks incompatible; all required passes compatible absent adaptation proof.
- [x] **Step 1a — Test assertion:**

```python
self.assertEqual(probe_verdict({}, required=NATIVE_CHECKS, runtime_available=True), "blocked")
checks = {name: "pass" for name in NATIVE_CHECKS}
self.assertEqual(probe_verdict(checks, required=NATIVE_CHECKS, runtime_available=True), "compatible")
checks["old_final"] = "fail"
self.assertEqual(probe_verdict(checks, required=NATIVE_CHECKS, runtime_available=True,
                                adaptation_verified=True), "incompatible")
```

- [x] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_probes -v`; expected failure.
- [x] **Step 3 — Implement pure report/verdict:** implement separate native/TorchRL required check sets and `probe_report_to_json` explicit field mapping (including safe `Failure` encoding). Reports persist via `writer.write_metric` with `spire-learn-probe-v1` and appear in final summary; native verdict does not imply TorchRL verdict.
- [x] **Step 3a — Verify pure report/verdict:** run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_probes -v`; assert native check set and TorchRL check set separately, mapping output fields and failure encoding.
- [x] **Step 4 — Runtime probe only after gate:** use exactly B manifest executions from `to_probe_config`, full-width native-endian int64 actions, legal current masks, batch cap and wall cap. Compare raw delivered public data with final/reset rows and masks; observe actual mixed ended/live row, old context and replacement state. If not observed within cap, report incomplete/blocked, do not synthesize evidence.
- [x] **Step 4b — Authorized native vector probe:** `test_vector_probe.py` invokes the real `run_vector_probe` against the public facade, requires mixed-row old/final/reset evidence and records `ProbeReport`. Run `SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_vector_probe -v`; `$PYTHON312` must be explicitly authorized. Missing opt-in is blocked/not-run, not green.
- [x] **Step 5 — TorchRL opt-in only:** separately inspect actual GymWrapper mapping and MaskedCategorical CPU sampling/log-probability; required checks include observation mapping and mask semantics. No optimizer or empty-mask sampling. Record candidate versions and native build separately.
- [x] **Step 6 — Integration command:** `SPIRE_LEARN_RUN_TORCHRL_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_torchrl_adapter -v`; require authorized `$PYTHON312`, never fallback to bare Python. Missing flag/dependencies is blocked, not skipped pass.
- [x] **Step 7 — Gate:** freeze provisional collection interfaces only after evidence and user/parent review. Compatible/adaptable/incompatible/blocked meanings follow ledger. If blocked, pure stdlib work may continue after implementation authorization, but M1 and contract freeze remain incomplete. If incompatible, stop for user-approved revision. A PyTorch-native fallback requires separate approval and demonstrated boundary/mask evidence.

**Acceptance (2026-10-25):** Parent accepted the bounded native and adapted TorchRL evidence after independent Luna/max code and runtime reviews. Tested reader commit `51194e1`; actual reader API3/0 and cohort1/0, adaptable/all8checks pass. Canonical acceptance record: `artifacts/development/foundation/task-7-contract-freeze.md`, including retained historical setup, race and audit qualifications. Provisional M1 collection contracts are frozen; M1 remains incomplete.

## Task 8: Single-Case Collector

**Depends on:** Tasks 1–7 and reviewed contract-freeze evidence. **Files:** `src/spire_learn/collection/single.py`, `tests/unit/test_single.py`.

**Interfaces:** Exact `run_case` signature in ledger; injected factory and classifier allow dependency-free scripted tests.

- [ ] **Step 1 — Red tests:** create `test_terminal_transition_records_public_outcome`, `test_truncation_is_distinct`, `test_decision_cap_is_budget_stop`, `test_illegal_action_has_zero_steps`, `test_empty_nonterminal_mask_is_contract_error`, `test_typed_fault_is_not_retried`, `test_terminal_step_delivered_after_deadline_is_preserved`, `test_trace_and_close_errors_preserve_terminal_result_without_retry`. Inject typed fault classes; arbitrary RuntimeError remains unmapped.
- [ ] **Step 1a — Test assertion:**

```python
clock = FakeClock()
env = ScriptedEnv((observation(), {}), [
    (observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False,
     {"outcome": 1, "episode_steps": 1})],
    on_step=lambda: clock.advance(2.0))
result = run_case(case_execution(), PlayFirstPolicy(), env_factory=lambda case: env,
                  budget=Budget(1.0, clock), max_decisions=4, contract=fixture_contract())
self.assertEqual((result.status, result.outcome, result.steps), ("terminal", "victory", 1))
self.assertEqual(len(env.step_calls), 1)
self.assertEqual(env.close_calls, 1)
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_single -v`; expected failure.
- [ ] **Step 3 — Implement:** validate case fields/profile and actual descriptor/content/fingerprint after factory construction and before reset. Runtime mismatch: factory count 1, reset count 0, close count 1; `setup_error` with primary failure. Runner owns config-vs-manifest assertions before factory (Task 9). Factory failure has no env to close; if factory returns, close exactly once. Reset failure or unsupported runtime profile before first successful reset is unstarted `setup_error`, preserving the primary failure kind. No hidden retry.
- [ ] **Step 4 — Preserve accepted outcomes and classify errors:** invalid agent choice (including bool, stale ID, or illegal action) rejected by `validate_result` -> `agent_error`; malformed public observation or empty nonterminal mask -> `contract_error`; factory/reset/unsupported runtime profile before first successful reset -> `setup_error`, preserving the primary failure kind. Do not classify arbitrary `RuntimeError` as simulator fault. Upon delivered step, record accepted step/reward/terminal outcome before invoking trace callback. Ongoing callback error -> `exec_error` primary; terminal callback/close error -> retain terminal outcome and attach secondary. Close never replaces primary. Typed simulator failure remains error, never death. One reset, close exactly once.
- [ ] **Step 5 — Green:** repeat focused command; assert terminal step overshooting deadline remains terminal, ongoing deadline stops before another action, and call logs show no retries or extra resets.

## Task 9: Reports, Evaluation Runner and CLI

**Depends on:** Tasks 1–8 for collector; pure report/config/CLI work may proceed after implementation authority while runtime gate is pending. **Files:** `src/spire_learn/evaluation/report.py`, `src/spire_learn/evaluation/runner.py`, `src/spire_learn/cli.py`, `src/spire_learn/__main__.py`; `tests/unit/test_report.py`, `tests/unit/test_runner.py`, `tests/unit/test_cli.py`; opt-in `tests/integration/test_evaluation.py`.

**Interfaces:** `summarize`, `EvaluationReport`, `evaluate`, `main` signatures and exit policy from ledger.

- [ ] **Step 1 — Red report test:** name it `test_status_and_outcome_counts_reconcile`; create terminal victory/death, truncation, budget stop, fault, setup error and not-started records. Assert scheduled reconciliation, disjoint started statuses, faults absent from outcomes, win rate `wins/(wins+deaths)`, coverage `terminal/started`, null rates for zero denominators, and secondary error count.
- [ ] **Step 1a — Test assertion:**

```python
fault = case_result(execution=case_execution(execution_id="fault"), status="simulator_fault",
                    outcome=None, steps=0,
                    primary_failure=Failure("simulator_fault", "fixture", "step"))
victory = case_result(execution=case_execution(execution_id="victory"))
summary = summarize([victory, fault])
self.assertEqual((summary["scheduled"], summary["started"]), (2, 2))
self.assertEqual(summary["outcome_counts"], {"victory": 1, "death": 0})
self.assertEqual(summary["completed_game_win_rate"], 1.0)
self.assertEqual(summary["terminal_completion_coverage"], 0.5)
```

- [ ] **Step 2 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_report -v`; expected failure.
- [ ] **Step 3 — Implement report:** only recognized verified public outcomes count; reject duplicate execution IDs.
- [ ] **Step 3a — Green report:** repeat the command; expect exact counts/rates above and `ValueError` for duplicate result IDs.
- [ ] **Step 4 — Red runner tests in `tests/unit/test_runner.py`:** `test_evaluation_uses_fresh_policy_seed_per_execution`, `test_final_split_requires_explicit_opt_in`, `test_deadline_marks_remaining_cases_not_started`, `test_static_partition_mismatch_precedes_factory`, `test_runtime_fingerprint_mismatch_constructs_then_closes_without_reset`, `test_writer_error_preserves_prior_case_and_stops_without_retry`, `test_single_scheduler_never_counts_vector_replacements`.
- [ ] **Step 5 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_runner -v`; expected failure.
- [ ] **Step 6 — Implement runner:** static `validate_expectations(config, manifest)` before factory; actual descriptor check after factory/before reset. Inject `fixture_contract()` in runner tests with `ScriptedEnv`; production CLI always supplies `NATIVE_CONTRACT`, never fixture IDs from config/flags. `run_case` populates observed descriptor and available public build info; append that recorded discovery via writer update. Persist each case once. On writer error retain current result in memory and mark future executions not_started with artifact-failure reason; best-effort stderr/return report if disk is unavailable, do not pretend summary artifact was written. Count secondary failures and set exit 1 even if natural terminal outcome is retained.
- [ ] **Step 7 — Green runner:** repeat command; assert exact factory/reset/close counts for static versus runtime mismatch, retained primary/outcome and no action retry.
- [ ] **Step 8 — Red CLI tests:** create `test_import_and_help_do_not_import_optional_dependencies`, `test_validate_only_has_no_run_side_effects`, `test_final_flag_is_evaluation_only`, `test_exit_codes_distinguish_blocked_and_incomplete`. `main(['--help'])` returns 0 by catching argparse's help `SystemExit(0)`; malformed args return 2. Subprocesses set `PYTHONPATH=src` so source tree, not installed package, is tested.
- [ ] **Step 9 — Run red:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_cli -v`; expected failure.
- [ ] **Step 10 — Implement CLI:** implement CLI and keep `__main__` main-guard-only; CLI production evaluation always uses `NATIVE_CONTRACT`.
- [ ] **Step 10a — Verify CLI:** run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_cli -v`; then run full core suite `PYTHONPATH=src python3.12 -B -m unittest discover -s tests/unit -v`; expected all unit tests pass without optional packages.
- [ ] **Step 11 — Integration only after authority:** run `SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 PYTHONPATH=src "$PYTHON312" -B -m unittest tests.integration.test_evaluation -v`; smoke cases cap at 128 decisions. Verify tests were discovered/executed, artifacts reconcile, no final split and no exact wall-time claim. Missing flag/setup is blocked, not green.

## Task 10: Authored Configs and Documentation

**Depends on:** Tasks 1–9. **Files:** `configs/cases/selection-smoke.json`, `configs/evaluation/smoke.toml`, `configs/evaluation/development.toml`, `configs/evaluation/overnight.toml`, `configs/probes/vector-smoke.toml`, `configs/probes/torchrl-smoke.toml`, `docs/development.md`; conditional `README.md` modification.

- [ ] **Step 1 — Red config tests:** in `tests/unit/test_config.py`, assert profiles parse with exact mode-specific caps and probe settings derive from manifest executions. Example assertion:

```python
config = load_config(Path("configs/evaluation/smoke.toml"), expected_mode="evaluate")
self.assertEqual((config.profile, config.per_case_decisions, config.max_wall_seconds),
                 ("smoke", 128, 60.0))
```

- [ ] **Step 2 — Run red config tests:** `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_config tests.unit.test_cases -v`; expect missing authored config files to fail.
- [ ] **Step 3 — Add selection fixture:** exact scenarios `starter-cultist-a0-v1` (env 7/policy 101), `starter-jaw-worm-a0-v1` (env 11/policy 103), split selection, unique IDs. Diagnostic examples, not benchmark/authorization.
- [ ] **Step 4 — Add evaluation profiles:** smoke 128 decisions/60 sec, development 1024/600, overnight 4096/28800; explicit mode/agent/manifest/output/profile. Proposed ceilings, not permission to run.
- [ ] **Step 5 — Add probe examples:** each references same two-execution manifest; root `max_batch_steps=128`, `max_wall_seconds=60`; `[probe]` contains only `num_envs=2`, `workers=1`. No seed arrays or partition override; separate vector/TorchRL mode files.
- [ ] **Step 6 — Green config tests:** repeat the command and expect pass; evaluation forbids probe fields, probes forbid per-case cap.
- [ ] **Step 7 — Document** core tests, authorized integration gates/commands using `PYTHONPATH=src "$PYTHON312"`, flags, artifacts, provenance and RNG/TIME qualifications; no implicit install/run instructions.
- [ ] **Step 8 — README:** only change if actual evidence supports precise status; no learner strength claim. If integration blocked, say incomplete/blocked.
- [ ] **Step 9 — Acceptance review:** require core suite, actual single and vector evidence, reconciled artifacts, and TorchRL compatible/adaptable evidence or separately approved, tested PyTorch fallback. Blocked integration is not acceptance. Inspect changed/untracked paths.
- [ ] **Step 10 — Final execution report:** record actual acceptance evidence and remaining blockers for M1; no automatic next milestone or implementation authorization.

## Coverage and Stop Conditions

- Contracts/schema/masks: Tasks 1–2; endings and old/reset context: Tasks 3, 7–8; cases/splits/config/budgets: Task 4; artifacts/provenance: Tasks 5, 9; public native facade: Task 6; early framework gate: Task 7; evaluator/report/CLI: Task 9; authored profiles/docs: Task 10.
- The user approved this plan and separately authorized implementation on `main`. Local commits are authorized; no push or publication is authorized. Spec approval alone was not treated as implementation authority.
- Before dependency setup or real Gym/TorchRL integration, require explicit authority and reviewed setup addendum. If public API differs or evidence is insufficient, stop for user review; do not guess APIs or use private attributes.
- If vector mapping loses final data or masks cannot be established, stop and revise scope/adapter. Fallback is not automatically approved.
- On writer/close error preserve primary error and accepted result; never repeat action or overwrite prior run.
- Final cases require concrete user-approved run/config and `--allow-final`. No install, real integration, probe, training, push or publication is authorized by this plan document.

This plan captures the approved M1 scope; the user has explicitly authorized execution on `main` and local Git commits. Any real integration remains separately gated by required user authority and setup addendum. Preserve the Luna tool-worker workflow and parent self-review.
