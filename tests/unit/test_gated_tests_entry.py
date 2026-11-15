"""Pure unit tests for the in-container gated tensor-test entry."""

from __future__ import annotations

import json
import os
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

    def countTestCases(self):
        return super().countTestCases()


class _Result:
    def __init__(self, *, tests_run=5, skipped=(), failures=(), errors=()):
        self.testsRun = tests_run
        self.skipped = list(skipped)
        self.failures = list(failures)
        self.errors = list(errors)


class _Runner:
    def __init__(self, result):
        self.result = result
        self.suites = []

    def run(self, suite):
        self.suites.append(suite)
        return self.result


class GatedTestsEntryTests(unittest.TestCase):
    def _invoke(self, directory, run_id, result):
        from scripts.neural import gated_tests_entry as entry

        root = Path(directory)
        ledger = root / "ledger.jsonl"
        if not ledger.exists():
            _sample_guard.initialize_ledger(ledger)
        evidence = root / "evidence"
        evidence.mkdir(exist_ok=True)
        runner = _Runner(result)
        loaded = []

        def load(name):
            loaded.append(name)
            return _Suite((2, 1, 1, 1)[len(loaded) - 1])

        with (
            patch.object(entry.unittest.defaultTestLoader, "loadTestsFromName", side_effect=load),
            patch.object(entry.unittest, "TextTestRunner", return_value=runner) as runner_factory,
        ):
            status = entry.main([
                "--run-id", run_id, "--ledger", str(ledger), "--evidence", str(evidence)
            ])
        return status, root, loaded, runner, runner_factory

    def test_entry_reserves_and_records_green_in_one_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            status, root, loaded, runner, runner_factory = self._invoke(
                directory, "fresh-run", _Result()
            )
            self.assertEqual(status, 0)
            self.assertEqual(loaded, list(MODULES))
            self.assertEqual(len(runner.suites), 1)
            runner_factory.assert_called_once()
            self.assertEqual(os.environ["SPIRE_LEARN_RUN_NEURAL_TENSOR"], "1")
            self.assertEqual(os.environ["SPIRE_LEARN_TENSOR_TASK"], "gym-8726d467")
            self.assertEqual(os.environ["SPIRE_LEARN_TENSOR_RESERVATION_ID"], "gym-8726d467-fresh-run")
            self.assertEqual(os.environ["SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS"], "64")
            self.assertEqual(
                os.environ["SPIRE_LEARN_TENSOR_EVIDENCE_PATH"],
                str(root / "evidence" / "tensor-sampling.json"),
            )
            ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines()]
            reservation = next(row for row in ledger if row["event"] == "reservation")
            self.assertEqual((reservation["task"], reservation["reservation_id"], reservation["reserved_rows"]),
                             ("gym-8726d467", "gym-8726d467-fresh-run", 64))
            result = json.loads((root / "evidence" / "result.json").read_text())
            self.assertEqual(result["status"], "green")
            self.assertEqual(result["tests"]["run"], 5)

    def test_entry_refuses_a_reused_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first, *_ = self._invoke(directory, "same-run", _Result())
            self.assertEqual(first, 0)
            (Path(directory) / "evidence" / "result.json").unlink()
            second, _root, _loaded, runner, _factory = self._invoke(
                directory, "same-run", _Result()
            )
            self.assertNotEqual(second, 0)
            self.assertEqual(runner.suites, [])

    def test_entry_is_red_for_any_skip_or_incomplete_result(self):
        for fake in (_Result(skipped=[("a test", "unavailable")]), _Result(tests_run=4)):
            with self.subTest(fake=fake), tempfile.TemporaryDirectory() as directory:
                status, root, *_ = self._invoke(directory, "red-run", fake)
                self.assertNotEqual(status, 0)
                result = json.loads((root / "evidence" / "result.json").read_text())
                self.assertEqual(result["status"], "red")

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


if __name__ == "__main__":
    unittest.main()
