"""Pure probe report and verdict contract tests."""

import json
from pathlib import Path
import tempfile
import unittest

from spire_learn.artifacts.writer import RunWriter
from spire_learn.contracts import ContractError, Failure


def _vector(test_case):
    try:
        from spire_learn.probes import vector
    except ImportError as exc:
        test_case.fail(f"pure vector probe module is not implemented: {exc}")
    return vector


def _torchrl(test_case):
    try:
        from spire_learn.probes import torchrl
    except ImportError as exc:
        test_case.fail(f"pure TorchRL check contract is not implemented: {exc}")
    return torchrl


def _report_values(**changes):
    values = {
        "verdict": "compatible",
        "checks": {"mixed_rows": "pass"},
        "versions": {"spire_lab": "1.2.3"},
        "batch_steps": 17,
        "slot_decisions": 31,
        "replacement_decisions": 5,
        "failures": (),
    }
    values.update(changes)
    return values


class ProbeVerdictTests(unittest.TestCase):
    def test_probe_verdict_requires_boundary_evidence(self):
        probe_verdict = _vector(self).probe_verdict

        required = frozenset({"mixed_rows", "old_final"})
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass"},
                required=required,
                runtime_available=True,
            ),
            "blocked",
        )
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass", "old_final": "pass"},
                required=required,
                runtime_available=True,
            ),
            "compatible",
        )

    def test_probe_verdict_rejects_empty_required_checks(self):
        probe_verdict = _vector(self).probe_verdict
        with self.assertRaises(ContractError):
            probe_verdict({}, required=frozenset(), runtime_available=True)

    def test_probe_verdict_failure_precedes_missing_and_adaptation(self):
        probe_verdict = _vector(self).probe_verdict
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "fail"},
                required=frozenset({"mixed_rows", "old_final"}),
                runtime_available=True,
                adaptation_verified=True,
            ),
            "incompatible",
        )

    def test_probe_verdict_unavailable_runtime_is_blocked(self):
        probe_verdict = _vector(self).probe_verdict
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "fail", "old_final": "pass"},
                required=frozenset({"mixed_rows", "old_final"}),
                runtime_available=False,
                adaptation_verified=True,
            ),
            "blocked",
        )

    def test_probe_verdict_unknown_status_and_unrequested_checks_do_not_pass(self):
        probe_verdict = _vector(self).probe_verdict
        required = frozenset({"mixed_rows", "old_final"})
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass", "old_final": "passed"},
                required=required,
                runtime_available=True,
            ),
            "blocked",
        )
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass", "old_final": "unavailable"},
                required=required,
                runtime_available=True,
            ),
            "blocked",
        )
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass", "old_final": "pass", "extra": "fail"},
                required=required,
                runtime_available=True,
            ),
            "compatible",
        )

    def test_probe_verdict_adaptation_requires_explicit_verification(self):
        probe_verdict = _vector(self).probe_verdict
        self.assertEqual(
            probe_verdict(
                {"mixed_rows": "pass"},
                required=frozenset({"mixed_rows"}),
                runtime_available=True,
                adaptation_verified=True,
            ),
            "adaptable",
        )

    def test_probe_verdict_rejects_nonprimitive_control_values(self):
        probe_verdict = _vector(self).probe_verdict
        with self.assertRaises(ContractError):
            probe_verdict(
                {"mixed_rows": "pass"},
                required=frozenset({"mixed_rows"}),
                runtime_available=1,
            )
        with self.assertRaises(ContractError):
            probe_verdict(
                {"mixed_rows": "pass"},
                required=frozenset({"mixed_rows"}),
                runtime_available=True,
                adaptation_verified=1,
            )
        with self.assertRaises(ContractError):
            probe_verdict(
                {"mixed_rows": object()},
                required=frozenset({"mixed_rows"}),
                runtime_available=True,
            )

    def test_native_and_torchrl_require_distinct_check_sets(self):
        vector = _vector(self)
        torchrl = _torchrl(self)
        native = frozenset(
            {"mixed_rows", "old_final", "replacement_reset", "presence_masks", "old_context"}
        )
        torchrl_only = frozenset({"observation_mapping", "mask_sampling", "mask_log_prob"})
        self.assertEqual(vector.NATIVE_CHECKS, native)
        self.assertEqual(torchrl.TORCHRL_CHECKS, native | torchrl_only)

        native_passes = {name: "pass" for name in native}
        self.assertEqual(
            vector.probe_verdict(
                native_passes, required=native, runtime_available=True
            ),
            "compatible",
        )
        self.assertEqual(
            vector.probe_verdict(
                native_passes, required=torchrl.TORCHRL_CHECKS, runtime_available=True
            ),
            "blocked",
        )
        native_passes.update({name: "pass" for name in torchrl_only})
        self.assertEqual(
            vector.probe_verdict(
                native_passes, required=torchrl.TORCHRL_CHECKS, runtime_available=True
            ),
            "compatible",
        )


class ProbeReportTests(unittest.TestCase):
    def test_probe_report_counts_batches_slots_and_replacements(self):
        vector = _vector(self)
        report = vector.ProbeReport(**_report_values())
        encoded = vector.probe_report_to_json(report)
        self.assertEqual(encoded["batch_steps"], 17)
        self.assertEqual(encoded["slot_decisions"], 31)
        self.assertEqual(encoded["replacement_decisions"], 5)

    def test_probe_report_rejects_empty_verdict(self):
        vector = _vector(self)
        with self.assertRaises(ContractError):
            vector.ProbeReport(**_report_values(verdict=""))

    def test_probe_report_rejects_unknown_verdict(self):
        vector = _vector(self)
        with self.assertRaises(ContractError):
            vector.ProbeReport(**_report_values(verdict="pass"))

    def test_probe_report_owns_input_mappings_and_failure_state(self):
        vector = _vector(self)
        checks = {"mixed_rows": "pass"}
        versions = {"spire_lab": "1.2.3"}
        failure = Failure("simulator_fault", "fault", "step", "FaultError")
        failures = [failure]
        report = vector.ProbeReport(
            **_report_values(checks=checks, versions=versions, failures=failures)
        )

        checks["mixed_rows"] = "fail"
        versions["spire_lab"] = "mutated"
        failures.clear()
        self.assertEqual(report.checks, {"mixed_rows": "pass"})
        self.assertEqual(report.versions, {"spire_lab": "1.2.3"})
        self.assertEqual(report.failures, (failure,))
        with self.assertRaises(TypeError):
            report.checks["mixed_rows"] = "fail"
        with self.assertRaises(TypeError):
            report.versions["spire_lab"] = "mutated"

    def test_probe_report_rejects_invalid_primitives_and_counters(self):
        vector = _vector(self)
        invalid_values = (
            {"verdict": False},
            {"checks": {"mixed_rows": True}},
            {"checks": [("mixed_rows", "pass")]},
            {"versions": {"spire_lab": 1}},
            {"versions": [("spire_lab", "1.2.3")]},
            {"batch_steps": True},
            {"slot_decisions": -1},
            {"replacement_decisions": 32},
            {"failures": (object(),)},
            {"failures": (Failure([], "fault", "step"),)},
        )
        for change in invalid_values:
            with self.subTest(change=change), self.assertRaises(ContractError):
                vector.ProbeReport(**_report_values(**change))

    def test_probe_report_rejects_non_builtin_string_values(self):
        vector = _vector(self)

        class Label(str):
            pass

        invalid_reports = (
            ("verdict", _report_values(verdict=Label("compatible"))),
            ("check value", _report_values(checks={"mixed_rows": Label("pass")})),
            ("version value", _report_values(versions={"spire_lab": Label("1.2.3")})),
            (
                "failure kind",
                _report_values(failures=(Failure(Label("kind"), "reason", "phase"),)),
            ),
            (
                "failure reason",
                _report_values(failures=(Failure("kind", Label("reason"), "phase"),)),
            ),
            (
                "failure phase",
                _report_values(failures=(Failure("kind", "reason", Label("phase")),)),
            ),
            (
                "failure exception type",
                _report_values(failures=(Failure("kind", "reason", "phase", Label("Error")),)),
            ),
        )
        for field, values in invalid_reports:
            with self.subTest(field=field), self.assertRaises(ContractError):
                vector.ProbeReport(**values)

    def test_accepted_probe_report_encodes_with_safe_writer(self):
        vector = _vector(self)
        report = vector.ProbeReport(
            **_report_values(
                failures=(Failure("simulator_fault", "bad state", "step", "FaultError"),)
            )
        )
        expected_record = {
            "schema_id": "spire-learn-probe-v1",
            "verdict": "compatible",
            "checks": {"mixed_rows": "pass"},
            "versions": {"spire_lab": "1.2.3"},
            "batch_steps": 17,
            "slot_decisions": 31,
            "replacement_decisions": 5,
            "failures": [
                {
                    "kind": "simulator_fault",
                    "reason": "bad state",
                    "phase": "step",
                    "exception_type": "FaultError",
                }
            ],
        }
        record = vector.probe_report_to_json(report)
        self.assertEqual(record, expected_record)

        with tempfile.TemporaryDirectory() as temporary:
            writer = RunWriter.create(Path(temporary), "probe")
            try:
                writer.write_metric(record)
            finally:
                writer.close()
            written = json.loads(
                (Path(temporary) / "probe" / "metrics.jsonl").read_text()
            )
        self.assertEqual(
            written,
            {"schema_id": "spire-learn-probe-v1", "record": expected_record},
        )

    def test_probe_report_serialization_is_explicit_safe_and_fresh(self):
        vector = _vector(self)
        failure = Failure("simulator_fault", "fault", "step", "FaultError")
        report = vector.ProbeReport(
            **_report_values(
                verdict="incompatible",
                checks={"mixed_rows": "fail"},
                versions={"spire_lab": None},
                failures=(failure,),
            )
        )
        expected = {
            "schema_id": "spire-learn-probe-v1",
            "verdict": "incompatible",
            "checks": {"mixed_rows": "fail"},
            "versions": {"spire_lab": None},
            "batch_steps": 17,
            "slot_decisions": 31,
            "replacement_decisions": 5,
            "failures": [
                {
                    "kind": "simulator_fault",
                    "reason": "fault",
                    "phase": "step",
                    "exception_type": "FaultError",
                }
            ],
        }
        encoded = vector.probe_report_to_json(report)
        self.assertEqual(encoded, expected)
        json.dumps(encoded, allow_nan=False)

        encoded["checks"]["mixed_rows"] = "pass"
        encoded["versions"]["spire_lab"] = "mutated"
        encoded["failures"][0]["reason"] = "mutated"
        encoded["failures"].append({"kind": "extra"})
        self.assertEqual(vector.probe_report_to_json(report), expected)


if __name__ == "__main__":
    unittest.main()
