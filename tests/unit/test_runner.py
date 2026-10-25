import json
import random
from importlib import import_module
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.experiments.config import RunConfig
from tests.support import FakeClock, ScriptedEnv, case_execution, fixture_contract, observation


def _runner_api(test_case):
    try:
        module = import_module("spire_learn.evaluation.runner")
    except ImportError as error:
        test_case.fail(f"evaluation runner API is missing: {error}")
    evaluate = getattr(module, "evaluate", None)
    report_type = getattr(module, "EvaluationReport", None)
    test_case.assertTrue(callable(evaluate), "evaluation runner API is missing")
    test_case.assertTrue(report_type is not None, "EvaluationReport API is missing")
    return module, evaluate


def _write_manifest(path: Path, executions, *, split: str = "selection") -> None:
    first = executions[0]
    case = {
        "case_id": first.case_id,
        "scenario_id": first.scenario_id,
        "partition": first.partition,
        "executions": [
            {
                "execution_id": execution.execution_id,
                "environment_seed": execution.environment_seed,
                "policy_seed": execution.policy_seed,
            }
            for execution in executions
        ],
    }
    if first.expected_schema_id is not None:
        case["expected_schema_id"] = first.expected_schema_id
    if first.expected_fingerprint is not None:
        case["expected_fingerprint"] = first.expected_fingerprint
    path.write_text(
        json.dumps({"schema": "spire-learn-cases-v1", "split": split, "cases": [case]}),
        encoding="utf-8",
    )


def _config(root: Path, case_path: Path, *, split_guards=(), **overrides) -> RunConfig:
    values = {
        "mode": "evaluate",
        "profile": "smoke",
        "agent": "random",
        "cases": case_path,
        "output_root": root / "runs",
        "per_case_decisions": 20,
        "max_batch_steps": None,
        "max_wall_seconds": 60.0,
        "traces": False,
        "expected_partition": None,
        "expected_schema_id": None,
        "expected_fingerprint": None,
        "gym_checkout": None,
        "split_guard_manifests": tuple(split_guards),
        "probe": None,
    }
    values.update(overrides)
    return RunConfig(**values)


def _terminal_step(outcome: int = 1):
    return (observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": outcome})


class _Writer:
    def __init__(
        self,
        run_dir: Path,
        *,
        fail_case_write: bool = False,
        fail_not_started_write: bool = False,
        transition_error: Exception | None = None,
        control_phase: str | None = None,
        control_error: BaseException | None = None,
        close_error: Exception | None = None,
    ):
        self.run_dir = run_dir
        self.fail_case_write = fail_case_write
        self.fail_not_started_write = fail_not_started_write
        self.transition_error = transition_error
        self.control_phase = control_phase
        self.control_error = control_error
        self.close_error = close_error
        self.case_writes = []
        self.manifest_updates = []
        self.final_summary = None
        self.close_calls = 0
        self.transition_writes = []

    def write_start(self, config, manifest):
        self.start = (config, manifest)
        if self.control_phase == "start":
            raise self.control_error

    def update_manifest(self, record):
        self.manifest_updates.append(record)

    def write_transition(self, transition):
        self.transition_writes.append(transition)
        if self.transition_error is not None:
            raise self.transition_error

    def write_case(self, result):
        self.case_writes.append(result)
        if self.control_phase == "case":
            raise self.control_error
        if self.fail_not_started_write and result.status == "not_started":
            raise OSError("injected not-started write failure")
        if self.fail_case_write:
            raise OSError("injected case-write failure")

    def finalize(self, summary):
        self.final_summary = dict(summary)

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _writer_factory_for(writer):
    def create(root, run_id):
        return writer
    return create


def _episode_results(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)["record"]
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]


class EvaluationRunnerTests(unittest.TestCase):
    def _evaluate(self, *args, **kwargs):
        module, evaluate = _runner_api(self)
        return module, evaluate(*args, **kwargs)

    def _patch_provenance(self, module):
        return patch.object(
            module,
            "collect_provenance",
            return_value={"learner": {"revision": None, "dirty": None}},
        )

    def test_evaluation_uses_fresh_policy_seed_per_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="first", environment_seed=7, policy_seed=103),
                case_execution(execution_id="second", environment_seed=11, policy_seed=103),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path)
            envs = []

            def factory(execution):
                steps = [
                    (
                        observation(), 0.0, False, False, {},
                    )
                    for _ in range(19)
                ] + [_terminal_step()]
                env = ScriptedEnv((observation(), {}), steps)
                envs.append(env)
                return env

            expected_rng = random.Random(103)
            expected = [expected_rng.choice([0, 1]) for _ in range(20)]
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    run_id="fresh-seed",
                )

            self.assertEqual(report.exit_code, 0)
            self.assertEqual([env.step_calls for env in envs], [expected, expected])
            self.assertEqual([env.reset_calls[0][0] for env in envs], [7, 11])

    def test_final_split_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "final.json"
            _write_manifest(
                case_path,
                (case_execution(execution_id="final-case"),),
                split="final",
            )
            config = _config(root, case_path)
            factory_calls = []
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: factory_calls.append(execution),
                    contract=fixture_contract(),
                    run_id="blocked-final",
                )

            self.assertEqual(report.exit_code, 2)
            self.assertEqual(factory_calls, [])
            self.assertFalse(config.output_root.exists())
            self.assertIsNotNone(report.fatal_error)

    def test_deadline_marks_remaining_cases_not_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="deadline-first", environment_seed=7),
                case_execution(execution_id="deadline-next", environment_seed=11),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path, max_wall_seconds=1.0)
            clock = FakeClock()
            envs = []

            def factory(execution):
                env = ScriptedEnv(
                    (observation(), {}),
                    [(observation(), 0.0, False, False, {})],
                    on_step=lambda: clock.advance(1.1),
                )
                envs.append(env)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    clock=clock,
                    run_id="deadline",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(len(envs), 1)
            self.assertEqual(len(envs[0].reset_calls), 1)
            self.assertEqual(report.summary["started"], 1)
            self.assertEqual(report.summary["not_started"], 1)
            self.assertEqual(report.summary["status_counts"]["budget_stopped"], 1)
            self.assertIsNotNone(report.run_dir)
            self.assertEqual(
                [
                    (
                        item["execution"]["execution_id"],
                        item["status"],
                        item["stop_reason"],
                    )
                    for item in _episode_results(report.run_dir)
                ],
                [
                    ("deadline-first", "budget_stopped", "wall_time_budget"),
                    ("deadline-next", "not_started", "wall_time_budget"),
                ],
            )

    def test_budget_clock_failure_is_an_execution_reliability_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="clock-error-first", environment_seed=7),
                case_execution(execution_id="clock-error-next", environment_seed=11),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path, max_wall_seconds=1.0)
            clock_reads = 0

            def failing_clock():
                nonlocal clock_reads
                clock_reads += 1
                if clock_reads == 2:
                    raise OSError("injected clock read failure")
                return 0.0

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: self.fail("factory must not run"),
                    contract=fixture_contract(),
                    clock=failing_clock,
                    run_id="clock-error",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(report.summary["started"], 0)
            self.assertEqual(report.summary["not_started"], 2)
            self.assertEqual(report.fatal_error.kind, "exec_error")
            self.assertIsNotNone(report.run_dir)
            self.assertEqual(
                [
                    (
                        item["execution"]["execution_id"],
                        item["status"],
                        item["stop_reason"],
                    )
                    for item in _episode_results(report.run_dir)
                ],
                [
                    ("clock-error-first", "not_started", "budget_error"),
                    ("clock-error-next", "not_started", "budget_error"),
                ],
            )

    def test_deadline_before_reset_persists_current_and_remaining_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="before-reset-first", environment_seed=7),
                case_execution(execution_id="before-reset-next", environment_seed=11),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path, max_wall_seconds=1.0)
            clock = FakeClock()

            class ExpiringDescriptorEnv(ScriptedEnv):
                def observation_spec(self):
                    clock.advance(1.1)
                    return super().observation_spec()

            envs = []

            def factory(execution):
                env = ExpiringDescriptorEnv((observation(), {}), [_terminal_step()])
                envs.append(env)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    clock=clock,
                    run_id="before-reset-deadline",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(len(envs), 1)
            self.assertEqual(envs[0].reset_calls, [])
            self.assertEqual(envs[0].close_calls, 1)
            self.assertIsNotNone(report.run_dir)
            self.assertEqual(
                [
                    (
                        item["execution"]["execution_id"],
                        item["status"],
                        item["stop_reason"],
                    )
                    for item in _episode_results(report.run_dir)
                ],
                [
                    ("before-reset-first", "not_started", "wall_time_budget"),
                    ("before-reset-next", "not_started", "wall_time_budget"),
                ],
            )

    def test_not_started_write_failure_stops_drain_without_retries_or_factories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = tuple(
                case_execution(
                    execution_id=f"drain-{index}",
                    environment_seed=20 + index,
                    policy_seed=40 + index,
                )
                for index in range(3)
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path, max_wall_seconds=1.0)
            clock_reads = 0

            def expires_before_first_case():
                nonlocal clock_reads
                clock_reads += 1
                return 0.0 if clock_reads == 1 else 1.1

            writer = _Writer(root / "not-started-drain-run", fail_not_started_write=True)
            factory_calls = []
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: factory_calls.append(execution),
                    contract=fixture_contract(),
                    clock=expires_before_first_case,
                    writer_factory=_writer_factory_for(writer),
                    run_id="not-started-drain-failure",
            )

            self.assertEqual(report.exit_code, 1)
            self.assertIsNotNone(report.fatal_error)
            self.assertEqual(report.fatal_error.kind, "artifact_error")
            self.assertEqual(report.summary["scheduled"], 3)
            self.assertEqual(report.summary["not_started"], 3)
            self.assertEqual(report.summary["secondary_failure_count"], 1)
            self.assertEqual(factory_calls, [])
            self.assertEqual(len(writer.case_writes), 1)
            self.assertEqual(writer.case_writes[0].stop_reason, "wall_time_budget")
            self.assertEqual(writer.close_calls, 1)

    def test_static_partition_mismatch_precedes_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="wrong-partition"),))
            config = _config(root, case_path, expected_partition="other")
            factory_calls = []
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: factory_calls.append(execution),
                    contract=fixture_contract(),
                )

            self.assertEqual(report.exit_code, 2)
            self.assertEqual(factory_calls, [])
            self.assertFalse(config.output_root.exists())

    def test_split_guard_overlap_is_checked_before_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "selection.json"
            guard_path = root / "train-guard.json"
            _write_manifest(
                case_path,
                (case_execution(execution_id="selection-case"),),
                split="selection",
            )
            _write_manifest(
                guard_path,
                (
                    case_execution(
                        execution_id="train-overlap",
                        case_id="train-case",
                        split="train",
                    ),
                ),
                split="train",
            )
            config = _config(root, case_path, split_guards=(guard_path,))
            factory_calls = []
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: factory_calls.append(execution),
                    contract=fixture_contract(),
                )

            self.assertEqual(report.exit_code, 2)
            self.assertEqual(factory_calls, [])
            self.assertFalse(config.output_root.exists())

    def test_runtime_fingerprint_mismatch_constructs_then_closes_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            execution = case_execution(
                execution_id="runtime-mismatch",
                expected_fingerprint="b" * 64,
            )
            _write_manifest(case_path, (execution,))
            config = _config(root, case_path)
            env = ScriptedEnv((observation(), {}), [_terminal_step()])
            factory_calls = []

            def factory(bound_execution):
                factory_calls.append(bound_execution)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    run_id="runtime-mismatch",
                )

            self.assertEqual(report.exit_code, 2)
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(env.reset_calls, [])
            self.assertEqual(env.step_calls, [])
            self.assertEqual(env.close_calls, 1)
            self.assertEqual(report.summary["setup_failures"], 1)

    def test_config_schema_expectations_are_bound_before_run_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="config-expectation"),))
            config = _config(root, case_path, expected_fingerprint="b" * 64)
            env = ScriptedEnv((observation(), {}), [_terminal_step()])
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: env,
                    contract=fixture_contract(),
                    run_id="config-expectation",
                )

            self.assertEqual(report.exit_code, 2)
            self.assertEqual(env.reset_calls, [])
            self.assertEqual(env.step_calls, [])
            self.assertEqual(env.close_calls, 1)
            self.assertEqual(report.summary["setup_failures"], 1)

    def test_writer_error_preserves_prior_case_and_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="writer-first", environment_seed=7),
                case_execution(execution_id="writer-future", environment_seed=11),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path)
            writer = _Writer(root / "fake-run", fail_case_write=True)
            envs = []

            def factory(execution):
                env = ScriptedEnv((observation(), {}), [_terminal_step()])
                envs.append(env)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    writer_factory=_writer_factory_for(writer),
                    run_id="writer-failure",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(len(envs), 1)
            self.assertEqual(envs[0].step_calls, [0])
            self.assertEqual(len(writer.case_writes), 1)
            self.assertEqual(writer.close_calls, 1)
            self.assertEqual(report.summary["scheduled"], 2)
            self.assertEqual(report.summary["started"], 1)
            self.assertEqual(report.summary["not_started"], 1)
            self.assertEqual(report.summary["outcome_counts"], {"victory": 1, "death": 0})
            self.assertEqual(report.summary["secondary_failure_count"], 1)

    def test_transition_writer_error_stops_ongoing_case_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            executions = (
                case_execution(execution_id="trace-first", environment_seed=7),
                case_execution(execution_id="trace-future", environment_seed=11),
            )
            _write_manifest(case_path, executions)
            config = _config(root, case_path, traces=True)
            writer = _Writer(root / "trace-run", transition_error=OSError("injected trace failure"))
            envs = []

            def factory(execution):
                env = ScriptedEnv(
                    (observation(), {}),
                    [(observation(), 0.0, False, False, {}), _terminal_step()],
                )
                envs.append(env)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    writer_factory=_writer_factory_for(writer),
                    run_id="trace-failure",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(len(envs), 1)
            self.assertEqual(len(envs[0].step_calls), 1)
            self.assertEqual(envs[0].close_calls, 1)
            self.assertEqual(len(writer.transition_writes), 1)
            self.assertEqual(len(writer.case_writes), 1)
            failed_case = writer.case_writes[0]
            self.assertEqual(failed_case.status, "exec_error")
            self.assertIsNone(failed_case.outcome)
            self.assertEqual(failed_case.primary_failure.kind, "artifact_error")
            self.assertEqual(failed_case.secondary_errors, ())
            self.assertEqual(report.summary["started"], 1)
            self.assertEqual(report.summary["not_started"], 1)
            self.assertEqual(report.summary["secondary_failure_count"], 0)

    def test_terminal_trace_writer_error_preserves_outcome_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="trace-terminal"),))
            config = _config(root, case_path, traces=True)
            writer = _Writer(root / "terminal-trace-run", transition_error=OSError("injected trace failure"))
            env = ScriptedEnv((observation(), {}), [_terminal_step()])
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=lambda execution: env,
                    contract=fixture_contract(),
                    writer_factory=_writer_factory_for(writer),
                    run_id="terminal-trace-failure",
                )

            self.assertEqual(report.exit_code, 1)
            self.assertEqual(env.step_calls, [0])
            self.assertEqual(env.close_calls, 1)
            self.assertEqual(len(writer.transition_writes), 1)
            self.assertEqual(len(writer.case_writes), 1)
            failed_case = writer.case_writes[0]
            self.assertEqual((failed_case.status, failed_case.outcome), ("terminal", "victory"))
            self.assertIsNone(failed_case.primary_failure)
            self.assertEqual(len(failed_case.secondary_errors), 1)
            self.assertEqual(failed_case.secondary_errors[0].kind, "artifact_error")
            self.assertEqual(report.summary["secondary_failure_count"], 1)

    def test_writer_start_process_control_closes_once_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="start-control"),))
            config = _config(root, case_path)
            primary = KeyboardInterrupt("write_start control")
            writer = _Writer(
                root / "start-control-run",
                control_phase="start",
                control_error=primary,
                close_error=OSError("close also failed"),
            )
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module.evaluate(
                        config,
                        env_factory=lambda execution: self.fail("factory must not run"),
                        contract=fixture_contract(),
                        writer_factory=_writer_factory_for(writer),
                        run_id="start-control",
                    )

            self.assertIs(raised.exception, primary)
            self.assertEqual(writer.close_calls, 1)

    def test_writer_start_failure_propagates_close_process_control_with_primary_note(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="start-close-control"),))
            config = _config(root, case_path)
            start_error = OSError("write_start failed")
            close_control = KeyboardInterrupt("close control")
            writer = _Writer(
                root / "start-close-control-run",
                control_phase="start",
                control_error=start_error,
                close_error=close_control,
            )
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module.evaluate(
                        config,
                        contract=fixture_contract(),
                        writer_factory=_writer_factory_for(writer),
                        run_id="start-close-control",
                    )

            self.assertIs(raised.exception, close_control)
            self.assertEqual(writer.close_calls, 1)
            self.assertTrue(any("write_start failed" in note for note in getattr(close_control, "__notes__", ())))

    def test_close_process_control_keeps_prior_artifact_failure_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="close-after-artifact"),))
            config = _config(root, case_path)
            close_control = KeyboardInterrupt("close after case-write failure")
            writer = _Writer(
                root / "close-after-artifact-run",
                fail_case_write=True,
                close_error=close_control,
            )
            env = ScriptedEnv((observation(), {}), [_terminal_step()])
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module.evaluate(
                        config,
                        env_factory=lambda execution: env,
                        contract=fixture_contract(),
                        writer_factory=_writer_factory_for(writer),
                        run_id="close-after-artifact",
                    )

            self.assertIs(raised.exception, close_control)
            self.assertEqual(writer.close_calls, 1)
            self.assertEqual(env.step_calls, [0])
            self.assertTrue(any("injected case-write failure" in note for note in getattr(close_control, "__notes__", ())))
            self.assertTrue(
                any(
                    "execution_id=close-after-artifact" in note and "outcome=victory" in note
                    for note in getattr(close_control, "__notes__", ())
                )
            )

    def test_first_process_control_survives_process_control_from_writer_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="double-control"),))
            config = _config(root, case_path)
            primary = KeyboardInterrupt("write_start control")
            close_control = SystemExit("close control")
            writer = _Writer(
                root / "double-control-run",
                control_phase="start",
                control_error=primary,
                close_error=close_control,
            )
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module.evaluate(
                        config,
                        contract=fixture_contract(),
                        writer_factory=_writer_factory_for(writer),
                        run_id="double-control",
                    )

            self.assertIs(raised.exception, primary)
            self.assertEqual(writer.close_calls, 1)
            self.assertTrue(
                any(
                    "SystemExit: close control" in note
                    for note in getattr(primary, "__notes__", ())
                )
            )

    def test_after_case_process_control_closes_once_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="case-control"),))
            config = _config(root, case_path)
            primary = KeyboardInterrupt("write_case control")
            writer = _Writer(
                root / "case-control-run",
                control_phase="case",
                control_error=primary,
                close_error=OSError("close also failed"),
            )
            env = ScriptedEnv((observation(), {}), [_terminal_step()])
            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module.evaluate(
                        config,
                        env_factory=lambda execution: env,
                        contract=fixture_contract(),
                        writer_factory=_writer_factory_for(writer),
                        run_id="case-control",
                    )

            self.assertIs(raised.exception, primary)
            self.assertEqual(writer.close_calls, 1)
            self.assertEqual(env.step_calls, [0])
            self.assertEqual(env.close_calls, 1)
            self.assertEqual(len(writer.case_writes), 1)

    def test_single_scheduler_never_counts_vector_replacements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_manifest(case_path, (case_execution(execution_id="single"),))
            config = _config(root, case_path)
            envs = []

            def factory(execution):
                env = ScriptedEnv((observation(), {}), [_terminal_step()])
                envs.append(env)
                return env

            module, _ = _runner_api(self)
            with self._patch_provenance(module):
                report = module.evaluate(
                    config,
                    env_factory=factory,
                    contract=fixture_contract(),
                    run_id="single-scheduler",
                )

            self.assertEqual(report.exit_code, 0)
            self.assertEqual(len(envs), 1)
            self.assertEqual(len(envs[0].reset_calls), 1)
            self.assertEqual(len(envs[0].step_calls), 1)
            self.assertEqual(envs[0].close_calls, 1)
            self.assertEqual(report.summary["scheduled"], 1)
            self.assertEqual(report.summary["started"], 1)
            self.assertEqual(report.summary["status_counts"]["terminal"], 1)


if __name__ == "__main__":
    unittest.main()
