"""One opt-in real-runtime test for the bounded TorchRL compatibility probe."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
import uuid


_RUN_TORCHRL = os.environ.get("SPIRE_LEARN_RUN_TORCHRL_INTEGRATION") == "1"
_EXPECTED_BUILD_INFO = {
    "extension": "spire-python",
    "revision": "4874fb398128c142bd56f10a12e6fca939a88f93",
    "rust_version": "1.98.0",
}
_EXPECTED_FINGERPRINT = "8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357"


@unittest.skipUnless(
    _RUN_TORCHRL,
    "set SPIRE_LEARN_RUN_TORCHRL_INTEGRATION=1 in the separately approved runtime",
)
class TorchRLAdapterIntegrationTests(unittest.TestCase):
    def test_bounded_torchrl_probe_records_real_mapping_mask_and_boundary_evidence(self):
        from spire_learn.artifacts.writer import RunWriter
        from spire_learn.contracts import CaseExecution
        from spire_learn.experiments.cases import CaseManifest
        from spire_learn.experiments.config import ProbeSettings, RunConfig, to_probe_config
        from spire_learn.probes.torchrl import TORCHRL_CHECKS, run_torchrl_probe
        from spire_learn.probes.vector import probe_report_to_json

        executions = (
            CaseExecution(
                "torchrl-cultist-selection-7",
                "torchrl-cultist",
                "selection",
                "selection",
                "starter-cultist-a0-v1",
                7,
                101,
            ),
            CaseExecution(
                "torchrl-jaw-worm-selection-11",
                "torchrl-jaw-worm",
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
            output_root = Path(output_override)
            run_id = f"task-7-torchrl-{uuid.uuid4().hex}"
            cleanup_root = False
        else:
            output_root = Path(tempfile.mkdtemp(prefix="spire-learn-torchrl-probe-"))
            run_id = "task-7-torchrl"
            cleanup_root = True
        run_directory = output_root / run_id
        success = False
        try:
            repository_root = Path(__file__).resolve().parents[2]
            for protected in (repository_root / "src", repository_root / "tests"):
                if output_root.resolve().is_relative_to(protected.resolve()):
                    self.fail(f"probe output root must not be inside {protected}")
            run_config = RunConfig(
                mode="probe-torchrl",
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
                        "mode": "probe-torchrl",
                        "profile": "smoke",
                        "agent": "random",
                        "max_batch_steps": 128,
                        "max_wall_seconds": 60.0,
                        "probe": {"num_envs": 2, "workers": 1},
                    },
                    {"split": "selection", "executions": executions},
                )
                report = run_torchrl_probe(config, writer, clock=time.monotonic)
            finally:
                writer.close()

            summary_envelope = json.loads(
                (run_directory / "summary.json").read_text(encoding="utf-8")
            )
            summary = summary_envelope["summary"]
            metric_records = [
                json.loads(line)
                for line in (run_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            probe_metrics = [
                item for item in metric_records if item.get("schema_id") == "spire-learn-probe-v1"
            ]

            self.assertIn(report.verdict, {"compatible", "adaptable"})
            self.assertEqual(set(report.checks) & TORCHRL_CHECKS, TORCHRL_CHECKS)
            self.assertTrue(all(report.checks[name] == "pass" for name in TORCHRL_CHECKS))
            self.assertGreater(report.replacement_decisions, 0)
            self.assertGreaterEqual(report.batch_steps, 2)
            self.assertEqual(report.slot_decisions, 2 * report.batch_steps)
            self.assertEqual(report.versions["fingerprint"], _EXPECTED_FINGERPRINT)
            self.assertEqual(summary["native_build_info"], _EXPECTED_BUILD_INFO)
            self.assertEqual(
                summary["action_batch_contract"],
                {"dtype": "int64", "native_endian": True, "shape": [2]},
            )
            self.assertEqual(summary["report"], probe_report_to_json(report))
            self.assertEqual(len(probe_metrics), 1)
            self.assertEqual(probe_metrics[0]["record"], summary["report"])
            self.assertEqual(summary["delivered"]["batch_steps"], report.batch_steps)
            self.assertEqual(summary["delivered"]["slot_decisions"], 2 * report.batch_steps)
            self.assertEqual(summary["delivered"]["wrapper_batch_steps"], report.batch_steps)
            self.assertGreater(summary["delivered"]["replacement_decisions"], 0)
            self.assertEqual(summary["stop_reason"], "required_evidence_complete")
            self.assertLessEqual(summary["budgets"]["diagnostic_samples_reserved"], 256)
            guard = summary["adapter"]["guard_counts"]
            construction_guard = summary["adapter"]["construction_guard_counts"]
            self.assertEqual(construction_guard["reset_forwarded"], 0)
            self.assertEqual(construction_guard["step_forwarded"], 0)
            self.assertEqual(construction_guard["denied_calls"], 0)
            self.assertEqual(guard["reset_forwarded"], 1)
            self.assertEqual(guard["reset_attempts"], 1)
            self.assertEqual(guard["reset_delivered"], 1)
            self.assertEqual(guard["reset_captured"], 1)
            self.assertEqual(guard["step_forwarded"], report.batch_steps)
            self.assertEqual(guard["step_delivered"], report.batch_steps)
            self.assertEqual(guard["step_captured"], report.batch_steps)
            self.assertEqual(guard["denied_calls"], 0)
            self.assertEqual(
                summary["call_counts"],
                {
                    "controller_attempted_steps": report.batch_steps,
                    "native_forwarded_steps": report.batch_steps,
                    "native_delivered_steps": report.batch_steps,
                    "native_captured_steps": report.batch_steps,
                    "wrapper_delivered_steps": report.batch_steps,
                },
            )

            print(
                "TORCHRL_PROBE_EVIDENCE "
                + json.dumps(
                    {
                        "verdict": report.verdict,
                        "checks": dict(report.checks),
                        "batch_steps": report.batch_steps,
                        "slot_decisions": report.slot_decisions,
                        "replacement_decisions": report.replacement_decisions,
                        "wrapper_batch_steps": summary["delivered"]["wrapper_batch_steps"],
                        "guard_counts": guard,
                        "adaptation_verified": summary["adapter"]["adaptation_verified"],
                        "diagnostic_samples_reserved": summary["budgets"]["diagnostic_samples_reserved"],
                        "versions": dict(report.versions),
                        "native_build_info": summary["native_build_info"],
                        "stop_reason": summary["stop_reason"],
                    },
                    sort_keys=True,
                )
            )
            success = True
        finally:
            if not success:
                print(f"TORCHRL_PROBE_FAILURE_OUTPUT {run_directory}")
            elif cleanup_root:
                shutil.rmtree(output_root)


if __name__ == "__main__":
    unittest.main()
