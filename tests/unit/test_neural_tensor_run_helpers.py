"""Focused stdlib checks for Task 4 protected-run evidence helpers."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural.task4_tensor_run_helpers import (
    CONTROL_FILES,
    MIN_FREE_BYTES,
    assess_run,
    close_inventory,
    control_hashes,
    expected_test_names,
    extract_container_command,
    parse_container_command,
    parse_unittest_output,
    require_free_space,
    validate_model_metrics,
    validate_sampling_evidence,
    verify_control_hashes,
    write_cleanup_status,
    write_run_summary,
)


def _good_metrics() -> dict[str, object]:
    return {
        "schema_fingerprint": "a" * 64,
        "parameter_count": 1234,
        "snapshot_bytes_per_frame": 4096,
        "forward_backward_ms": {
            "B1": {"forward_ms": 1.0, "backward_ms": 2.0},
            "B4": {"forward_ms": 3.0, "backward_ms": 4.0},
        },
        "peak_process_bytes": 16 * 1024**2,
        "native_resets": 0,
        "native_steps": 0,
        "policy_samples": 0,
    }


def _good_sampling() -> dict[str, object]:
    return {
        "schema": "spire-learn-tensor-sampling-v1",
        "reservation_id": "task4-result-a",
        "reserved_rows": 64,
        "attempted_rows": 4,
        "completed_rows": 4,
        "rejected_rows": 0,
        "status": "sample_completed",
    }


class NeuralTensorRunHelperTests(unittest.TestCase):
    def test_free_space_boundary_requires_at_least_thirty_gibibytes(self):
        with self.assertRaisesRegex(ValueError, "30 GiB"):
            require_free_space(MIN_FREE_BYTES - 1)
        self.assertIsNone(require_free_space(MIN_FREE_BYTES))

    def test_literal_container_body_passes_bin_sh_parse_without_execution(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "scripts/neural/run-task4-tensor-tests.sh"
        command = extract_container_command(script)
        self.assertTrue(command.startswith("set -eu\n"))
        parsed = parse_container_command(script)
        self.assertTrue(parsed["parsed"], parsed["stderr"])
        self.assertEqual(parsed["exit_code"], 0)

    def test_control_file_hashes_detect_changed_wrapper_or_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in CONTROL_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"before:{relative}\n", encoding="utf-8")
            before = control_hashes(root)
            verify_control_hashes(root, before)

            changed = root / CONTROL_FILES[0]
            changed.write_text("changed control script\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control files changed"):
                verify_control_hashes(root, before)

    def test_expected_test_inventory_comes_from_all_three_authored_modules(self):
        root = Path(__file__).resolve().parents[2] / "tests/neural"
        names = expected_test_names(root)
        self.assertGreater(len(names), 20)
        self.assertTrue(any("NeuralModelTensorTests" in name for name in names))
        self.assertTrue(any("NeuralDistributionTensorTests" in name for name in names))
        self.assertTrue(any("NeuralPPOTensorTests" in name for name in names))
        self.assertEqual(len(names), len(set(names)))

    def test_expected_inventory_matches_narrow_unittest_discovery_without_optional_imports(self):
        optional = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
        original_import = builtins.__import__

        def reject_optional(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in optional:
                raise AssertionError(f"optional import attempted during gated discovery: {name}")
            return original_import(name, globals, locals, fromlist, level)

        root = Path(__file__).resolve().parents[2] / "tests/neural"
        with patch.object(builtins, "__import__", new=reject_optional):
            suite = unittest.defaultTestLoader.discover(str(root), pattern="test*.py")

        actual: list[str] = []

        def collect(node):
            if isinstance(node, unittest.TestSuite):
                for child in node:
                    collect(child)
            else:
                module, test_class, method = node.id().rsplit(".", 2)
                actual.append(f"{method} ({module}.{test_class}.{method})")

        collect(suite)
        expected = expected_test_names(root)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(set(actual), set(expected))

    def test_expected_inventory_matches_real_verbose_text_runner_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_names = ("test_model", "test_distribution", "test_ppo")
            cases = []
            for module_name in module_names:
                (root / f"{module_name}.py").write_text(
                    "class ProbeTests:\n    def test_probe(self):\n        pass\n",
                    encoding="utf-8",
                )
                probe = type(
                    "ProbeTests",
                    (unittest.TestCase,),
                    {"__module__": module_name, "test_probe": lambda self: None},
                )
                cases.append(probe("test_probe"))

            expected = expected_test_names(root)
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                unittest.TestSuite(cases)
            )
            self.assertTrue(result.wasSuccessful(), stream.getvalue())
            parsed = parse_unittest_output(
                "", stream.getvalue(), expected, exit_code=0
            )

        self.assertTrue(parsed["suite_complete"], stream.getvalue())
        self.assertEqual(parsed["missing_expected_tests"], [])
        self.assertEqual(parsed["unexpected_tests"], [])
        self.assertTrue(parsed["green_candidate"])

    def test_verbose_unittest_result_preserves_actual_counts_and_missing_cases(self):
        expected = (
            "test_first (test_model.ModelTests.test_first)",
            "test_second (test_model.ModelTests.test_second)",
        )
        stderr = (
            f"{expected[0]} ... ok\n"
            f"{expected[1]} ... skipped 'gate disabled'\n\n"
            "Ran 2 tests in 0.010s\n\nOK (skipped=1)\n"
        )
        result = parse_unittest_output("", stderr, expected, exit_code=0)
        self.assertEqual(result["ran"], 2)
        self.assertEqual(result["counts"], {"ok": 1, "failure": 0, "error": 0, "skipped": 1})
        self.assertEqual(result["missing_expected_tests"], [])
        self.assertFalse(result["green_candidate"])

    def test_green_assessment_requires_all_tests_and_coherent_runtime_evidence(self):
        expected = ("test_only (test_model.ModelTests.test_only)",)
        parsed = parse_unittest_output(
            "",
            f"{expected[0]} ... ok\n\nRan 1 test in 0.010s\n\nOK\n",
            expected,
            exit_code=0,
        )
        sampling = validate_sampling_evidence(
            _good_sampling(), reservation_id="task4-result-a", max_rows=64
        )
        metrics = validate_model_metrics(_good_metrics())
        assessment = assess_run(
            parsed,
            expected_outcome="green",
            sampling=sampling,
            metrics=metrics,
        )
        self.assertEqual(assessment["classification"], "green")
        self.assertTrue(assessment["accepted"])

        missing_metrics = validate_model_metrics(None)
        rejected = assess_run(
            parsed,
            expected_outcome="green",
            sampling=sampling,
            metrics=missing_metrics,
        )
        self.assertNotEqual(rejected["classification"], "green")
        self.assertFalse(rejected["accepted"])

        skipped = parse_unittest_output(
            "",
            f"{expected[0]} ... skipped 'not run'\n\nRan 1 test in 0.010s\n\nOK (skipped=1)\n",
            expected,
            exit_code=0,
        )
        skipped_assessment = assess_run(
            skipped,
            expected_outcome="green",
            sampling=sampling,
            metrics=metrics,
        )
        self.assertEqual(skipped_assessment["classification"], "skipped_tests")
        self.assertFalse(skipped_assessment["accepted"])

        zero = parse_unittest_output("", "\nRan 0 tests in 0.000s\n\nOK\n", expected, exit_code=0)
        zero_assessment = assess_run(
            zero,
            expected_outcome="green",
            sampling=sampling,
            metrics=metrics,
        )
        self.assertNotEqual(zero_assessment["classification"], "green")
        self.assertFalse(zero_assessment["accepted"])

    def test_product_api_absence_is_distinct_from_runtime_or_loader_errors(self):
        expected = ("test_only (test_model.ModelTests.test_only)",)

        def classify_import_message(message: str) -> tuple[dict[str, object], dict[str, object]]:
            stderr = (
                f"{expected[0]} ... FAIL\n\n"
                f"AssertionError: {message}\n\n"
                "Ran 1 test in 0.010s\n\nFAILED (failures=1)\n"
            )
            parsed = parse_unittest_output("", stderr, expected, exit_code=1)
            assessment = assess_run(
                parsed,
                expected_outcome="red",
                sampling=validate_sampling_evidence(
                    None, reservation_id="task4-result-a", max_rows=64
                ),
                metrics=validate_model_metrics(None),
            )
            return parsed, assessment

        direct, direct_assessment = classify_import_message(
            "opt-in neural model API is unavailable: "
            "No module named 'spire_learn.neural.model'"
        )
        self.assertTrue(direct["product_api_missing"])
        self.assertEqual(direct_assessment["classification"], "product_api_missing")
        self.assertTrue(direct_assessment["accepted"])
        self.assertFalse(direct_assessment["behavioral_failure"])

        # Exact parent reproducer: Python's `from package import missing_name`
        # error differs from the direct `No module named package.module` form.
        from_package, from_package_assessment = classify_import_message(
            "opt-in neural model API is unavailable: cannot import name 'features' "
            "from 'spire_learn.neural' "
            "(/home/klottick/repos/spire-learn/src/spire_learn/neural/__init__.py)"
        )
        self.assertTrue(from_package["product_api_missing"])
        self.assertEqual(from_package_assessment["classification"], "product_api_missing")
        self.assertTrue(from_package_assessment["accepted"])
        self.assertFalse(from_package_assessment["behavioral_failure"])

        for package in ("torch", "numpy", "gymnasium"):
            package_missing, package_assessment = classify_import_message(
                f"opt-in neural model API is unavailable: No module named '{package}'"
            )
            self.assertFalse(package_missing["product_api_missing"], package)
            self.assertEqual(
                package_assessment["classification"], "runtime_or_loader_error", package
            )
            self.assertFalse(package_assessment["accepted"], package)

        circular, circular_assessment = classify_import_message(
            "opt-in neural model API is unavailable: cannot import name 'features' "
            "from partially initialized module 'spire_learn.neural' "
            "(most likely due to a circular import)"
        )
        self.assertFalse(circular["product_api_missing"])
        self.assertTrue(circular["api_import_failure"])
        self.assertEqual(circular_assessment["classification"], "runtime_or_loader_error")
        self.assertFalse(circular_assessment["accepted"])

        unrelated, unrelated_assessment = classify_import_message(
            "opt-in neural model API is unavailable: cannot import name 'helper' "
            "from 'spire_learn.neural' (/tmp/spire_learn/neural/__init__.py)"
        )
        self.assertFalse(unrelated["product_api_missing"])
        self.assertEqual(unrelated_assessment["classification"], "runtime_or_loader_error")
        self.assertFalse(unrelated_assessment["accepted"])

    def test_sampling_evidence_rejects_uncertain_or_oversized_counts(self):
        valid = validate_sampling_evidence(
            _good_sampling(), reservation_id="task4-result-a", max_rows=64
        )
        self.assertTrue(valid["coherent"])
        self.assertTrue(valid["completed_draws"])

        uncertain_record = dict(_good_sampling(), status="sample_started", completed_rows=2)
        uncertain = validate_sampling_evidence(
            uncertain_record, reservation_id="task4-result-a", max_rows=64
        )
        self.assertFalse(uncertain["coherent"])
        self.assertFalse(uncertain["completed_draws"])

        oversized_record = dict(_good_sampling(), attempted_rows=65, completed_rows=65)
        oversized = validate_sampling_evidence(
            oversized_record, reservation_id="task4-result-a", max_rows=64
        )
        self.assertFalse(oversized["coherent"])

    def test_model_metrics_reject_nonfinite_or_missing_benchmark_values(self):
        self.assertTrue(validate_model_metrics(_good_metrics())["coherent"])
        self.assertFalse(validate_model_metrics(
            _good_metrics(), expected_schema_fingerprint="b" * 64
        )["coherent"])
        nonfinite = _good_metrics()
        nonfinite["forward_backward_ms"] = {
            "B1": {"forward_ms": float("inf"), "backward_ms": 1.0},
            "B4": {"forward_ms": 3.0, "backward_ms": 4.0},
        }
        self.assertFalse(validate_model_metrics(nonfinite)["coherent"])

    def test_inventory_hashes_all_closed_run_files_and_excludes_only_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "logs").mkdir()
            (run_root / "logs/result.exit.log").write_text("0\n", encoding="utf-8")
            (run_root / "run-summary.json").write_text('{"status":"complete"}\n', encoding="utf-8")
            inventory = close_inventory(run_root)
            self.assertEqual(set(inventory["sha256"]), {"logs/result.exit.log", "run-summary.json"})
            saved = json.loads((run_root / "evidence-inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, inventory)
            self.assertTrue(saved["inventory_self_excluded"])

    def test_cleanup_and_summary_preserve_actual_statuses_without_claiming_unknown_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_root = base / "absent"
            logs = run_root / "logs"
            logs.mkdir(parents=True)
            (logs / "container-remove.exit.log").write_text("1\n", encoding="utf-8")
            (logs / "container-final-inventory.exit.log").write_text("0\n", encoding="utf-8")
            (logs / "container-final-inventory.stdout.log").write_text("", encoding="utf-8")
            (logs / "source-postcheck.exit.log").write_text("0\n", encoding="utf-8")
            (logs / "sample-row-finish.exit.log").write_text("0\n", encoding="utf-8")
            test_result = {"classification": "behavioral_red", "accepted": True}
            (run_root / "test-result.json").write_text(
                json.dumps(test_result) + "\n", encoding="utf-8"
            )

            cleanup = write_cleanup_status(
                run_root,
                create_attempted=True,
                container_created=False,
                cleanup_failed=False,
            )
            self.assertEqual(cleanup["remove_exit_status"], 1)
            self.assertEqual(cleanup["final_named_container_inventory_exit_status"], 0)
            self.assertEqual(cleanup["resource_state_after_cleanup"], "absent")
            self.assertFalse(cleanup["cleanup_failed"])

            summary = write_run_summary(
                run_root,
                pre_inventory_exit_status=1,
                expected_outcome="red",
                container_created=False,
                container_started=True,
            )
            self.assertEqual(summary["pre_inventory_exit_status"], 1)
            self.assertEqual(summary["test_result"], test_result)
            self.assertEqual(summary["cleanup_status"], cleanup)
            self.assertEqual(summary["command_exit_statuses"]["container-remove"], 1)
            self.assertEqual(summary["command_exit_statuses"]["container-final-inventory"], 0)

            present_root = base / "present"
            present_logs = present_root / "logs"
            present_logs.mkdir(parents=True)
            (present_logs / "container-final-inventory.exit.log").write_text(
                "0\n", encoding="utf-8"
            )
            (present_logs / "container-final-inventory.stdout.log").write_text(
                "container-id\n", encoding="utf-8"
            )
            present = write_cleanup_status(
                present_root,
                create_attempted=True,
                container_created=True,
                cleanup_failed=False,
            )
            self.assertEqual(present["resource_state_after_cleanup"], "present")
            self.assertTrue(present["cleanup_failed"])

            unknown_root = base / "unknown"
            unknown_root.mkdir()
            unknown = write_cleanup_status(
                unknown_root,
                create_attempted=True,
                container_created=False,
                cleanup_failed=False,
            )
            self.assertEqual(unknown["resource_state_after_cleanup"], "unknown")
            self.assertTrue(unknown["cleanup_failed"])


if __name__ == "__main__":
    unittest.main()
