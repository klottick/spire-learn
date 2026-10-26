"""Stdlib-only tests for the shared Task 4/6 diagnostic sampling ledger."""

from __future__ import annotations

import json
import builtins
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.neural import _sample_guard as guard


class _ShapedInput:
    def __init__(self, shape):
        self.shape = shape


class NeuralSamplingBudgetTests(unittest.TestCase):
    def test_failed_reservation_stays_spent_against_the_shared_lifetime_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            guard.initialize_ledger(path)
            guard.reserve_rows(task="task4", reservation_id="task4-red-a", path=path)
            guard.complete_reservation(
                reservation_id="task4-red-a",
                status="failed",
                attempted_rows=3,
                completed_rows=2,
                path=path,
            )
            for index in range(1, guard.LIFETIME_CAP // guard.MAX_INVOCATION_ROWS):
                guard.reserve_rows(
                    task="task6",
                    reservation_id=f"task6-green-{index:02d}",
                    path=path,
                )
            with self.assertRaises(guard.SampleBudgetError):
                guard.reserve_rows(task="task4", reservation_id="task4-over-cap", path=path)

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            reservations = [event for event in events if event["event"] == "reservation"]
            self.assertEqual(len(reservations), guard.LIFETIME_CAP // guard.MAX_INVOCATION_ROWS)
            self.assertEqual(reservations[0]["reserved_before"], 0)
            self.assertEqual(reservations[-1]["reserved_after"], guard.LIFETIME_CAP)
            first_finish = next(event for event in events if event["event"] == "completion")
            self.assertEqual(first_finish["status"], "failed")
            self.assertEqual(first_finish["attempted_rows"], 3)
            self.assertEqual(first_finish["completed_rows"], 2)

    def test_uncertain_completion_keeps_its_reservation_and_known_attempt_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            guard.initialize_ledger(path)
            reserved = guard.reserve_rows(task="task4", reservation_id="task4-uncertain", path=path)
            finished = guard.complete_reservation(
                reservation_id="task4-uncertain",
                status="uncertain",
                attempted_rows=4,
                completed_rows=None,
                path=path,
            )
            self.assertEqual(reserved, {"reserved_total": 64, "remaining": 4032})
            self.assertEqual(finished, reserved)
            completion = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(completion["status"], "uncertain")
            self.assertEqual(completion["attempted_rows"], 4)
            self.assertIsNone(completion["completed_rows"])

    def test_guard_counts_before_call_and_refuses_overshoot(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "sample-evidence.json"
            guard_instance = guard.SamplingGuard(
                reservation_id="task4-unit-guard",
                max_rows=2,
                evidence_path=evidence,
            )
            calls = []

            def sampler(input_tensor, num_samples, **kwargs):
                calls.append((input_tensor.shape, num_samples))
                return "sampled"

            guarded = guard_instance.wrap(sampler)
            self.assertEqual(guarded(_ShapedInput((2, 5)), 1), "sampled")
            self.assertEqual((guard_instance.attempted_rows, guard_instance.completed_rows), (2, 2))
            with self.assertRaises(guard.SampleBudgetError):
                guarded(_ShapedInput((1, 5)), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(guard_instance.rejected_rows, 1)
            saved = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(saved["attempted_rows"], 2)
            self.assertEqual(saved["completed_rows"], 2)
            self.assertEqual(saved["rejected_rows"], 1)

    def test_failed_sampler_keeps_attempted_count_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "sample-evidence.json"
            guard_instance = guard.SamplingGuard(
                reservation_id="task4-unit-failure",
                max_rows=4,
                evidence_path=evidence,
            )

            def fail(input_tensor, num_samples, **kwargs):
                raise RuntimeError("injected sample failure")

            with self.assertRaisesRegex(RuntimeError, "injected sample failure"):
                guard_instance.wrap(fail)(_ShapedInput((3, 8)), 1)
            saved = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual((saved["attempted_rows"], saved["completed_rows"]), (3, 0))
            self.assertEqual(saved["status"], "sample_failed")

    def test_guard_support_import_and_disabled_gate_use_only_the_standard_library(self):
        optional_prefixes = ("numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab")
        before = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NEURAL_TENSOR": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "gate is not enabled"):
                guard.activate_from_environment()
        after = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        self.assertEqual(after, before)

    def test_all_opt_in_tensor_modules_discover_without_optional_import_attempts(self):
        optional_prefixes = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
        before = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        original_import = builtins.__import__

        def reject_optional(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in optional_prefixes:
                raise AssertionError(f"optional import attempted before tensor gate: {name}")
            return original_import(name, globals, locals, fromlist, level)

        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_NEURAL_TENSOR": "0"}, clear=False):
            with patch.object(builtins, "__import__", new=reject_optional):
                importlib.import_module("tests.neural.test_model")
                importlib.import_module("tests.neural.test_distribution")
                importlib.import_module("tests.neural.test_ppo")

        after = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
