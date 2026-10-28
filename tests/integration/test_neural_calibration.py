"""The single explicit, opt-in Task 8 CUDA calibration invocation."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest


_GATE = "SPIRE_LEARN_RUN_TASK8_CALIBRATION"
_GPU_GATE = "SPIRE_LEARN_TASK8_GPU_UUID"
_GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
_EVIDENCE_ROOT = Path(os.environ.get("SPIRE_LEARN_TASK8_EVIDENCE_DIR", "/learner/evidence"))
_MEMORY_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_RELEASE_ENV = {
    "source_snapshot_sha256": "SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256",
    "runbook_sha256": "SPIRE_LEARN_TASK8_RUNBOOK_SHA256",
    "calibration_config_file_sha256": "SPIRE_LEARN_TASK8_CONFIG_SHA256",
    "train_manifest_sha256": "SPIRE_LEARN_TASK8_MANIFEST_SHA256",
    "qualification_report_sha256": "SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256",
    "qualification_summary_sha256": "SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256",
    "cuda_lock_sha256": "SPIRE_LEARN_TASK8_LOCK_SHA256",
    "native_sha256": "SPIRE_LEARN_TASK8_NATIVE_SHA256",
}
_LOCK_SHA256 = "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45"
_NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"


def _assert_completed_calibration_report(testcase, report) -> None:
    summary = report.summary
    expected_ids = [
        "standalone-b4-h32", "standalone-b8-h32", "standalone-b16-h32", "run-b1-h128",
    ]
    reservations = (128, 256, 512, 128)
    testcase.assertEqual(report.exit_code, 0, summary)
    testcase.assertEqual(summary["status"], "complete")
    testcase.assertEqual(summary["cohort"]["reserved_action_slots"], 1024)
    delivered_total = summary["cohort"]["delivered_environment_slots"]
    testcase.assertGreater(delivered_total, 0)
    testcase.assertLessEqual(delivered_total, 1024)
    testcase.assertEqual(summary["cohort"]["optimizer_accepted_steps"], 3)
    testcase.assertEqual(summary["cohort"].get("uncertain_environment_slots", 0), 0)
    testcase.assertGreaterEqual(
        summary["cohort"].get("attempted_environment_slots", delivered_total), delivered_total,
    )
    testcase.assertLessEqual(summary["cohort"].get("attempted_environment_slots", delivered_total), 1024)
    trials = summary["trials"]
    testcase.assertEqual([trial["trial_id"] for trial in trials], expected_ids)
    testcase.assertEqual(len(trials), 4)
    observed_total = 0
    for index, (trial, reserved) in enumerate(zip(trials, reservations, strict=True)):
        collection = trial["collection"]
        delivered = collection["delivered_environment_slots"]
        observed_total += delivered
        testcase.assertEqual(trial["reserved_action_slots"], reserved)
        testcase.assertEqual(trial["status"], "complete")
        testcase.assertTrue(trial["fully_tested"])
        testcase.assertEqual(collection.get("faults", 0), 0)
        testcase.assertEqual(collection.get("uncertain_environment_slots", 0), 0)
        testcase.assertEqual(collection.get("environment_truncations", 0), 0)
        testcase.assertNotIn(collection.get("stop_reason"), {"wall_budget", "fault"})
        if index < 3:
            testcase.assertEqual(delivered, reserved)
            testcase.assertEqual(collection.get("stop_reason"), "horizon")
            probe = trial.get("optimizer_probe")
            testcase.assertIsInstance(probe, dict)
            testcase.assertEqual(probe.get("attempted_steps"), 1)
            testcase.assertEqual(probe.get("accepted_steps"), 1)
            testcase.assertEqual(probe.get("uncertain_steps"), 0)
            testcase.assertIs(probe.get("finite"), True)
            testcase.assertIs(probe.get("parameters_changed"), True)
        else:
            testcase.assertLessEqual(delivered, reserved)
            testcase.assertGreater(delivered, 0)
            testcase.assertIsNone(trial.get("optimizer_probe"))
            if delivered < reserved or collection.get("stop_reason") == "run_episode_terminal":
                testcase.assertEqual(collection.get("stop_reason"), "run_episode_terminal")
                testcase.assertGreater(collection.get("environment_terminations", 0), 0)
            else:
                testcase.assertEqual(collection.get("stop_reason"), "horizon")
    testcase.assertEqual(observed_total, delivered_total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NeuralCalibrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get(_GATE) != "1":
            raise unittest.SkipTest(f"set {_GATE}=1 only inside the reviewed calibration container")
        if os.environ.get(_GPU_GATE) != _GPU_UUID:
            raise RuntimeError("calibration container did not pass the exact qualified GPU UUID gate")
        if any(os.environ.get(name) != "4" for name in _MEMORY_ENV):
            raise RuntimeError("calibration container must set all BLAS/OpenMP thread caps to four")

        import gymnasium
        import numpy
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cls.torch = torch
        cls.root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.root / "configs/neural/calibration.toml"
        cls.started = time.monotonic()
        identity = {
            "event": "task8_calibration_process_identity",
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "spire_lab": importlib.metadata.version("spire-lab"),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "blas_environment": {name: os.environ[name] for name in _MEMORY_ENV},
            "gpu_uuid_gate": os.environ[_GPU_GATE],
            "recorded_before_calibration_factory": True,
        }
        release_binding = {
            key: os.environ.get(environment_name)
            for key, environment_name in _RELEASE_ENV.items()
        }
        if any(not isinstance(value, str) or len(value) != 64
               or any(character not in "0123456789abcdef" for character in value)
               for value in release_binding.values()):
            raise RuntimeError("parent release hash bindings are absent or malformed")
        if release_binding["cuda_lock_sha256"] != _LOCK_SHA256 or release_binding["native_sha256"] != _NATIVE_SHA256:
            raise RuntimeError("container lock/native bindings differ from the accepted CUDA runtime")
        mounted_hashes = {
            "cuda_lock_sha256": _sha256(_EVIDENCE_ROOT / "requirements-cuda.lock"),
            "qualification_report_sha256": _sha256(
                _EVIDENCE_ROOT / "task-8-gpu-qualification-report.md"
            ),
            "qualification_summary_sha256": _sha256(
                _EVIDENCE_ROOT / "task-8-gpu-qualification-summary.json"
            ),
            "calibration_config_file_sha256": _sha256(cls.config_path),
            "train_manifest_sha256": _sha256(cls.root / "configs/cases/neural/train.json"),
        }
        if any(release_binding[key] != observed for key, observed in mounted_hashes.items()):
            raise RuntimeError("read-only mounted config/runtime evidence differs from the parent release")
        qualification = json.loads(
            (_EVIDENCE_ROOT / "task-8-gpu-qualification-summary.json").read_text(encoding="utf-8")
        )
        if qualification.get("run_id") != "cuda-setup-20261028-a" or qualification.get("lock_sha256") != _LOCK_SHA256:
            raise RuntimeError("mounted CUDA qualification summary is not the accepted retained run")
        identity["parent_release_binding"] = release_binding
        sys.stdout.write("TASK8_CALIBRATION_PROCESS_IDENTITY=" + json.dumps(identity, sort_keys=True) + "\n")
        sys.stdout.flush()
        if (identity["torch"] != "2.14.1+cu130" or not identity["torch_cuda"].startswith("13.0")
                or identity["numpy"] != "2.5.4" or identity["gymnasium"] != "1.4.0"
                or identity["spire_lab"] != "0.1.0"):
            raise RuntimeError("installed packages differ from the accepted CUDA lock")
        if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("Torch thread caps differ from the calibration runbook")

    def test_task8_cuda_calibration_runs_one_fixed_cohort(self):
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.runner import calibrate

        config = load_neural_config(self.config_path)
        report = calibrate(config)
        _assert_completed_calibration_report(self, report)
        self.assertIsNotNone(report.run_dir)
        self.assertTrue((report.run_dir / "summary.json").is_file())
        self.assertTrue((report.run_dir / "lifecycle.json").is_file())


if __name__ == "__main__":
    unittest.main()
