"""Pure unit tests for the in-container gated tensor-test entry."""

from __future__ import annotations

import io
import json
import os
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.neural import _sample_guard


MODULES = (
    "tests.neural.test_model",
    "tests.neural.test_distribution",
    "tests.neural.test_ppo",
    "tests.neural.test_checkpoint",
)


class _Suite(unittest.TestSuite):
    def __init__(self, count):
        super().__init__([unittest.FunctionTestCase(lambda: None) for _ in range(count)])


class _FakeResult:
    def __init__(self, *, tests_run=5, skipped=(), failures=(), errors=(),
                 unexpected_successes=(), expected_failures=(), successful=True):
        self.testsRun = tests_run
        self.skipped = list(skipped)
        self.failures = list(failures)
        self.errors = list(errors)
        self.unexpectedSuccesses = list(unexpected_successes)
        self.expectedFailures = list(expected_failures)
        self.successful = successful
        self.was_successful_calls = 0

    def wasSuccessful(self):
        self.was_successful_calls += 1
        return self.successful


class _Runner:
    def __init__(self, result=None, *, evidence_status="sample_completed", error=None):
        self.result = result or _FakeResult()
        self.evidence_status = evidence_status
        self.error = error
        self.suites = []

    def run(self, suite):
        self.suites.append(suite)
        self.environment = dict(os.environ)
        if self.evidence_status is not None:
            evidence = Path(os.environ["SPIRE_LEARN_TENSOR_EVIDENCE_PATH"])
            evidence.write_text(json.dumps({
                "schema": "spire-learn-tensor-sampling-v1",
                "reservation_id": os.environ["SPIRE_LEARN_TENSOR_RESERVATION_ID"],
                "reserved_rows": 64,
                "attempted_rows": 3,
                "completed_rows": 3 if self.evidence_status == "sample_completed" else 0,
                "rejected_rows": 1 if self.evidence_status == "budget_rejected" else 0,
                "status": self.evidence_status,
            }))
        if self.error is not None:
            raise self.error
        return self.result


class _RealRunner:
    def __init__(self):
        self.runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
        self.suites = []

    def run(self, suite):
        self.suites.append(suite)
        result = self.runner.run(suite)
        evidence = Path(os.environ["SPIRE_LEARN_TENSOR_EVIDENCE_PATH"])
        evidence.write_text(json.dumps({
            "schema": "spire-learn-tensor-sampling-v1",
            "reservation_id": os.environ["SPIRE_LEARN_TENSOR_RESERVATION_ID"],
            "reserved_rows": 64,
            "attempted_rows": 0,
            "completed_rows": 0,
            "rejected_rows": 0,
            "status": "ready",
        }))
        return result


class GatedTestsEntryTests(unittest.TestCase):
    def _invoke(self, directory, run_id, result=None, *, counts=(2, 1, 1, 1),
                runner=None, loader=None, evidence_name="evidence"):
        from scripts.neural import gated_tests_entry as entry

        root = Path(directory)
        ledger = root / "ledger.jsonl"
        if not ledger.exists():
            _sample_guard.initialize_ledger(ledger)
        evidence = root / evidence_name
        runner = runner if runner is not None else _Runner(result)
        loaded = []

        def load(name):
            loaded.append(name)
            self.assertTrue((evidence / "claim.json").is_file())
            self.assertTrue(any(
                event.get("event") == "reservation"
                and event.get("reservation_id") == f"gym-8726d467-{run_id}"
                for event in self._ledger(ledger)
            ))
            if loader is not None:
                return loader(name, len(loaded))
            return _Suite(counts[len(loaded) - 1])

        with (
            patch.dict(os.environ, {"SPIRE_LEARN_TENSOR_METRICS_PATH": "/outside/metrics.json"}, clear=True),
            patch.object(entry.unittest.defaultTestLoader, "loadTestsFromName", side_effect=load),
            patch.object(entry.unittest, "TextTestRunner", return_value=runner) as runner_factory,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            status = entry.main([
                "--run-id", run_id, "--ledger", str(ledger), "--evidence", str(evidence)
            ])
        return status, root, evidence, loaded, runner, runner_factory

    @staticmethod
    def _ledger(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_entry_claims_owned_evidence_sets_all_module_environment_and_finishes_green(self):
        with tempfile.TemporaryDirectory() as directory:
            status, root, evidence, loaded, runner, runner_factory = self._invoke(
                directory, "fresh-run"
            )
            self.assertEqual(status, 0)
            self.assertEqual(loaded, list(MODULES))
            self.assertEqual(len(runner.suites), 1)
            runner_factory.assert_called_once()
            claim = json.loads((evidence / "claim.json").read_text())
            self.assertEqual(claim["run_id"], "fresh-run")
            self.assertGreater(claim["pid"], 0)
            self.assertTrue(claim["recorded_at_utc"].endswith("Z"))
            self.assertEqual(runner.environment["SPIRE_LEARN_RUN_NEURAL_TENSOR"], "1")
            self.assertEqual(runner.environment["SPIRE_LEARN_TENSOR_TASK"], "gym-8726d467")
            self.assertEqual(runner.environment["SPIRE_LEARN_TENSOR_RESERVATION_ID"], "gym-8726d467-fresh-run")
            self.assertEqual(runner.environment["SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS"], "64")
            self.assertEqual(runner.environment["SPIRE_LEARN_TENSOR_EVIDENCE_PATH"], str(evidence / "tensor-sampling.json"))
            self.assertEqual(runner.environment["SPIRE_LEARN_TENSOR_METRICS_PATH"], str(evidence / "model-metrics.json"))
            self.assertNotEqual(runner.environment["SPIRE_LEARN_TENSOR_METRICS_PATH"], "/outside/metrics.json")
            events = self._ledger(root / "ledger.jsonl")
            reservation, completion = events[1:]
            self.assertEqual((reservation["task"], reservation["reservation_id"], reservation["reserved_rows"]),
                             ("gym-8726d467", "gym-8726d467-fresh-run", 64))
            self.assertEqual((completion["status"], completion["attempted_rows"], completion["completed_rows"]),
                             ("succeeded", 3, 3))
            result = json.loads((evidence / "result.json").read_text())
            self.assertEqual(result["status"], "green")
            self.assertEqual(result["tests"]["run"], 5)
            self.assertEqual(runner.result.was_successful_calls, 1)
            self.assertEqual(result["modules"][0]["module"], MODULES[0])
            self.assertEqual(result["modules"][0]["collected"], 2)
            self.assertEqual(len(result["modules"][0]["test_ids"]), 2)
            self.assertTrue(result["unittest"]["was_successful"])

    def test_entry_refuses_reused_run_id_even_with_a_new_evidence_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            first, *_ = self._invoke(directory, "same-run")
            self.assertEqual(first, 0)
            second, _root, evidence, loaded, runner, _factory = self._invoke(
                directory, "same-run", evidence_name="second-evidence"
            )
            self.assertNotEqual(second, 0)
            self.assertEqual(loaded, [])
            self.assertEqual(runner.suites, [])
            self.assertFalse((evidence / "result.json").exists())

    def test_competing_run_id_cannot_claim_an_already_owned_evidence_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            first, root, evidence, *_ = self._invoke(directory, "owner-a")
            self.assertEqual(first, 0)
            second, _root, _evidence, _loaded, runner, _factory = self._invoke(
                directory, "owner-b"
            )
            self.assertNotEqual(second, 0)
            self.assertEqual(runner.suites, [])
            reservations = [event for event in self._ledger(root / "ledger.jsonl")
                            if event["event"] == "reservation"]
            self.assertEqual([event["reservation_id"] for event in reservations], ["gym-8726d467-owner-a"])
            self.assertEqual(json.loads((evidence / "claim.json").read_text())["run_id"], "owner-a")

    def test_preexisting_sampling_or_metrics_evidence_is_refused_before_reservation(self):
        for filename in ("tensor-sampling.json", "model-metrics.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _sample_guard.initialize_ledger(root / "ledger.jsonl")
                evidence = root / "evidence"
                evidence.mkdir()
                (evidence / filename).write_text("stale")
                from scripts.neural import gated_tests_entry as entry
                with (patch.object(entry.unittest, "TextTestRunner") as runner,
                      redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                    status = entry.main(["--run-id", "stale", "--ledger", str(root / "ledger.jsonl"),
                                         "--evidence", str(evidence)])
                self.assertNotEqual(status, 0)
                self.assertFalse(runner.called)
                self.assertEqual(len(self._ledger(root / "ledger.jsonl")), 1)

    def test_one_empty_module_and_all_empty_modules_are_red_without_running_partial_suite(self):
        for counts in ((2, 0, 1, 1), (0, 0, 0, 0)):
            with self.subTest(counts=counts), tempfile.TemporaryDirectory() as directory:
                status, _root, evidence, loaded, _runner, runner_factory = self._invoke(
                    directory, "empty-case", counts=counts
                )
                self.assertNotEqual(status, 0)
                self.assertEqual(loaded, list(MODULES))
                runner_factory.assert_not_called()
                result = json.loads((evidence / "result.json").read_text())
                self.assertEqual(result["status"], "red")
                self.assertEqual([module["collected"] for module in result["modules"]], list(counts))
                self.assertEqual(result["tests"]["expected"], sum(counts))
                completion = self._ledger(Path(directory) / "ledger.jsonl")[-1]
                self.assertEqual((completion["status"], completion["attempted_rows"]), ("not_started", 0))

    def test_collection_exception_and_failed_test_import_are_recorded_red(self):
        from scripts.neural import gated_tests_entry as entry
        real_loader = unittest.TestLoader()

        def failed_import(name, index):
            if index == 1:
                return real_loader.loadTestsFromName("tests.neural.no_such_gated_module")
            return _Suite(1)

        with tempfile.TemporaryDirectory() as directory:
            status, _root, evidence, _loaded, _runner, _factory = self._invoke(
                directory, "failed-import", loader=failed_import, runner=_RealRunner()
            )
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertTrue(any("_FailedTest" in test_id for module in result["modules"]
                                for test_id in module["test_ids"]))
            self.assertGreater(len(result["unittest"]["errors"]), 0)

        for fail_loader in (True, False):
            with self.subTest(fail_loader=fail_loader), tempfile.TemporaryDirectory() as directory:
                def loader(name, index):
                    if fail_loader and index == 2:
                        raise ImportError("injected module import failure")
                    if not fail_loader and index == 2:
                        return _Suite(0)
                    return _Suite(1)

                status, _root, evidence, *_ = self._invoke(
                    directory, "load-error", loader=loader
                )
                self.assertNotEqual(status, 0)
                result = json.loads((evidence / "result.json").read_text())
                self.assertTrue(result["modules"][1]["load_error"] if fail_loader
                                else result["modules"][1]["collected"] == 0)

    def test_unittest_unexpected_success_cannot_be_green_and_outcomes_are_recorded(self):
        class TinyExpectedOutcomes(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected_failure_is_recorded(self):
                self.fail("expected failure")

            @unittest.expectedFailure
            def test_unexpected_success_is_not_green(self):
                pass

        def loader(_name, index):
            return (unittest.defaultTestLoader.loadTestsFromTestCase(TinyExpectedOutcomes)
                    if index == 1 else _Suite(1))

        with tempfile.TemporaryDirectory() as directory:
            status, _root, evidence, *_ = self._invoke(
                directory, "unexpected", loader=loader, runner=_RealRunner()
            )
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertFalse(result["unittest"]["was_successful"])
            self.assertEqual(len(result["unittest"]["unexpected_successes"]), 1)
            self.assertEqual(len(result["unittest"]["expected_failures"]), 1)

    def test_test_failure_error_and_runner_exception_are_red_and_close_reservation(self):
        skipped = _FakeResult(
            skipped=[(unittest.FunctionTestCase(lambda: None), "skip")], successful=True
        )
        outcomes = (
            skipped,
            _FakeResult(failures=[(unittest.FunctionTestCase(lambda: None), "failure")], successful=False),
            _FakeResult(errors=[(unittest.FunctionTestCase(lambda: None), "error")], successful=False),
        )
        for index, outcome in enumerate(outcomes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                status, _root, evidence, *_ = self._invoke(directory, f"outcome-{index}", outcome)
                self.assertNotEqual(status, 0)
                result = json.loads((evidence / "result.json").read_text())
                self.assertEqual(result["status"], "red")
                self.assertEqual(self._ledger(Path(directory) / "ledger.jsonl")[-1]["status"], "failed")

        with tempfile.TemporaryDirectory() as directory:
            status, _root, evidence, *_ = self._invoke(
                directory, "runner-error", runner=_Runner(error=RuntimeError("runner broke"))
            )
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertIn("runner broke", result["runner_error"])
            self.assertEqual(self._ledger(Path(directory) / "ledger.jsonl")[-1]["status"], "failed")

    def test_interrupt_during_tests_settles_reservation_and_publishes_red_then_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                self._invoke(directory, "interrupted", runner=_Runner(error=KeyboardInterrupt()))
            result = json.loads(next(Path(directory).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "red")
            self.assertIn("KeyboardInterrupt", result["runner_error"])
            self.assertEqual(self._ledger(Path(directory) / "ledger.jsonl")[-1]["status"], "failed")

    def test_malformed_completion_event_is_recorded_red_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            from scripts.neural import gated_tests_entry as entry
            with patch.object(entry, "_completion_for", return_value={"event": "completion"}):
                status, _root, evidence, *_ = self._invoke(directory, "malformed")
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertEqual(result["status"], "red")
            self.assertIn("KeyError", result["reservation_completion"]["error"])

    def test_dirty_sampling_and_failed_ledger_completion_force_red(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _root, evidence, *_ = self._invoke(
                directory, "dirty-sampling", runner=_Runner(evidence_status="budget_rejected")
            )
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertFalse(result["sampling"]["clean"])
            self.assertEqual(self._ledger(Path(directory) / "ledger.jsonl")[-1]["status"], "failed")

        with tempfile.TemporaryDirectory() as directory:
            from scripts.neural import gated_tests_entry as entry
            with patch.object(_sample_guard, "_finish_from_evidence",
                              side_effect=OSError("ledger finish failed")):
                status, _root, evidence, *_ = self._invoke(directory, "finish-error")
            self.assertNotEqual(status, 0)
            result = json.loads((evidence / "result.json").read_text())
            self.assertIn("ledger finish failed", result["reservation_completion"]["error"])

    def test_missing_ledger_refuses_before_claim_or_test_collection(self):
        from scripts.neural import gated_tests_entry as entry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            with (patch.object(entry.unittest, "TextTestRunner") as runner,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                status = entry.main(["--run-id", "missing-ledger", "--ledger", str(root / "missing.jsonl"),
                                     "--evidence", str(evidence)])
            self.assertNotEqual(status, 0)
            self.assertFalse(evidence.exists())
            self.assertFalse(runner.called)

    def test_relative_ledger_and_evidence_paths_are_refused_before_claim(self):
        from scripts.neural import gated_tests_entry as entry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (ledger_relative, evidence_relative, expected) in enumerate((
                (True, False, "--ledger must be an absolute path"),
                (False, True, "--evidence must be an absolute path"),
            )):
                case = root / str(index)
                case.mkdir()
                ledger = case / "ledger.jsonl"
                _sample_guard.initialize_ledger(ledger)
                evidence = case / "evidence"
                ledger_arg = "relative-ledger.jsonl" if ledger_relative else str(ledger)
                evidence_arg = "relative-evidence" if evidence_relative else str(evidence)
                with self.subTest(expected=expected):
                    stderr = io.StringIO()
                    with chdir(case), \
                         patch.object(entry.unittest, "TextTestRunner") as runner, \
                         redirect_stderr(stderr):
                        status = entry.main(["--run-id", f"relative-path-{index}", "--ledger", ledger_arg,
                                             "--evidence", evidence_arg])
                    self.assertNotEqual(status, 0)
                    self.assertIn(expected, stderr.getvalue())
                    self.assertFalse(evidence.exists())
                    self.assertFalse(runner.called)
                self.assertEqual(len(self._ledger(ledger)), 1)

    def test_support_loads_the_8726d467_run_template(self):
        from spire_learn.neural import config
        from tests.neural import _support

        with patch.object(config, "load_neural_config", return_value=object()) as loader, \
                patch.object(_support, "schema_from_driver_factory", return_value="schema"):
            schema, loaded_config = _support.accepted_schema_and_config()
        self.assertEqual(schema, "schema")
        self.assertIsNotNone(loaded_config)
        self.assertEqual(
            loader.call_args.args[0].as_posix().split("/configs/neural/")[-1],
            "gym-8726d467/run.toml",
        )


class StandardTurnCapGateTests(unittest.TestCase):
    def test_selects_only_the_native_turn_cap_module(self):
        from scripts.neural import gated_tests_entry as entry

        suite = _Suite(2)
        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NATIVE_INTEGRATION": "1"}), \
             patch.object(unittest.defaultTestLoader, "loadTestsFromName", return_value=suite) as load:
            self.assertEqual(entry.main(["--standard-turncap"]), 0)
        load.assert_called_once_with("tests.integration.test_standard_turncap")

    def test_selects_only_the_native_search_value_module(self):
        from scripts.neural import gated_tests_entry as entry

        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NATIVE_INTEGRATION": "1"}), \
             patch.object(unittest.defaultTestLoader, "loadTestsFromName", return_value=_Suite(1)) as load:
            self.assertEqual(entry.main(["--search-value-native"]), 0)
        load.assert_called_once_with("tests.integration.test_search_value_native")

    def test_selects_only_the_distill_data_module(self):
        from scripts.neural import gated_tests_entry as entry

        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NATIVE_INTEGRATION": "1"}), \
             patch.object(unittest.defaultTestLoader, "loadTestsFromName", return_value=_Suite(1)) as load:
            self.assertEqual(entry.main(["--distill-data"]), 0)
        load.assert_called_once_with("tests.neural.test_distill_data")
        self.assertNotIn("tests.neural.test_distill_data", entry.MODULES)

    def test_native_route_requires_opt_in(self):
        from scripts.neural import gated_tests_entry as entry

        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NATIVE_INTEGRATION": "0"}), \
             patch.object(unittest.defaultTestLoader, "loadTestsFromName") as load, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(entry.main(["--standard-turncap"]), 2)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
