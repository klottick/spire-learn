"""Single opt-in native conformance test for the bounded vector probe."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid


_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"
_EXPECTED_BUILD_INFO = {
    "extension": "spire-python",
    "revision": "4874fb398128c142bd56f10a12e6fca939a88f93",
    "rust_version": "1.98.0",
}
_EXPECTED_FINGERPRINT = "8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357"


@unittest.skipUnless(
    _RUN_NATIVE,
    "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the separately approved runtime",
)
class NativeVectorProbeIntegrationTests(unittest.TestCase):
    def test_bounded_vector_probe_records_real_boundary_evidence(self):
        from spire_learn.artifacts.writer import RunWriter
        from spire_learn.contracts import CaseExecution
        from spire_learn.experiments.cases import CaseManifest
        from spire_learn.experiments.config import ProbeSettings, RunConfig, to_probe_config
        from spire_learn.probes.vector import NATIVE_CHECKS, probe_report_to_json, run_vector_probe

        executions = (
            CaseExecution(
                "vector-cultist-selection-7",
                "vector-cultist",
                "selection",
                "selection",
                "starter-cultist-a0-v1",
                7,
                101,
            ),
            CaseExecution(
                "vector-jaw-worm-selection-11",
                "vector-jaw-worm",
                "selection",
                "selection",
                "starter-jaw-worm-a0-v1",
                11,
                103,
            ),
        )
        manifest = CaseManifest(Path("learner-authored-selection.json"), "selection", executions)

        output_override = os.environ.get("SPIRE_LEARN_PROBE_OUTPUT_ROOT")
        if output_override:
            output_context = None
            output_root = Path(output_override)
            run_id = f"task-7-vector-{uuid.uuid4().hex}"
        else:
            output_context = tempfile.TemporaryDirectory(prefix="spire-learn-vector-probe-")
            output_root = Path(output_context.name)
            run_id = "task-7-vector"

        try:
            repository_root = Path(__file__).resolve().parents[2]
            for protected in (repository_root / "src", repository_root / "tests"):
                if output_root.resolve().is_relative_to(protected.resolve()):
                    self.fail(f"probe output root must not be inside {protected}")
            run_config = RunConfig(
                mode="probe-vector",
                profile="smoke",
                agent="random",
                cases=Path("learner-authored-selection.json"),
                output_root=output_root,
                per_case_decisions=None,
                max_batch_steps=128,
                max_wall_seconds=60.0,
                traces=False,
                expected_partition="selection",
                expected_schema_id="spire-lab-combat-observation-v1",
                expected_fingerprint=_EXPECTED_FINGERPRINT,
                probe=ProbeSettings(num_envs=2, workers=1),
            )
            config = to_probe_config(run_config, manifest)
            writer = RunWriter.create(output_root, run_id)
            try:
                writer.write_start(
                    {
                        "mode": "probe-vector",
                        "profile": "smoke",
                        "agent": "random",
                        "max_batch_steps": 128,
                        "max_wall_seconds": 60.0,
                        "probe": {"num_envs": 2, "workers": 1},
                    },
                    {"split": "selection", "executions": executions},
                )
                report = run_vector_probe(config, writer, clock=time.monotonic)
            finally:
                writer.close()

            run_dir = output_root / run_id
            summary_envelope = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary = summary_envelope["summary"]
            metric_records = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            probe_metrics = [
                item for item in metric_records if item.get("schema_id") == "spire-learn-probe-v1"
            ]

            self.assertEqual(report.verdict, "compatible")
            self.assertEqual(set(report.checks) & NATIVE_CHECKS, NATIVE_CHECKS)
            self.assertTrue(all(report.checks[name] == "pass" for name in NATIVE_CHECKS))
            self.assertGreater(report.replacement_decisions, 0)
            self.assertGreaterEqual(report.batch_steps, 2)
            self.assertEqual(report.slot_decisions, 2 * report.batch_steps)
            self.assertEqual(report.versions["fingerprint"], _EXPECTED_FINGERPRINT)
            self.assertEqual(summary["native_build_info"], _EXPECTED_BUILD_INFO)
            self.assertEqual(summary["action_batch_contract"], {
                "dtype": "int64",
                "native_endian": True,
                "shape": [2],
            })
            self.assertEqual(summary["report"], probe_report_to_json(report))
            self.assertEqual(len(probe_metrics), 1)
            self.assertEqual(probe_metrics[0]["record"], summary["report"])
            self.assertEqual(summary["delivered"]["batch_steps"], report.batch_steps)
            self.assertEqual(summary["delivered"]["slot_decisions"], 2 * report.batch_steps)
            self.assertGreater(summary["delivered"]["replacement_decisions"], 0)
            self.assertEqual(summary["stop_reason"], "required_evidence_complete")

            print(
                "NATIVE_VECTOR_EVIDENCE "
                + json.dumps(
                    {
                        "verdict": report.verdict,
                        "checks": dict(report.checks),
                        "batch_steps": report.batch_steps,
                        "slot_decisions": report.slot_decisions,
                        "replacement_decisions": report.replacement_decisions,
                        "schema": summary["schema"],
                        "native_build_info": summary["native_build_info"],
                        "stop_reason": summary["stop_reason"],
                    },
                    sort_keys=True,
                )
            )
        finally:
            if output_context is not None:
                output_context.cleanup()


if __name__ == "__main__":
    unittest.main()
