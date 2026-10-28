"""Pure CLI preflight and lazy dispatch checks for neural operations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from spire_learn.neural.config import load_neural_config


ROOT = Path(__file__).resolve().parents[2]


class NeuralCliTests(unittest.TestCase):
    def setUp(self):
        from spire_learn.neural import cli

        self.cli = cli

    def call(self, *args):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = self.cli.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_help_and_config_validation_are_optional_import_free_in_fresh_process(self):
        program = r'''
import builtins
from pathlib import Path
import sys
import tempfile
import unittest

blocked = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
attempts = []
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        attempts.append(name)
        raise AssertionError("optional import attempted: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from spire_learn.neural.cli import main
import spire_learn.neural.__main__
assert main(["--help"]) == 0
for mode in ("standalone", "run"):
    assert main(["train", "--config", f"configs/neural/{mode}.toml", "--validate-only"]) == 0
    for split in ("selection", "final"):
        args = ["evaluate", "--config", f"configs/neural/{mode}-{split}.toml",
                "--policy", "legal-random", "--validate-only"]
        if split == "final":
            args.append("--allow-final")
        assert main(args) == 0
source = Path("configs/neural/standalone.toml").read_text()
source = source.replace(
    'manifest_path = "../cases/neural/train.json"',
    f'manifest_path = "{Path("configs/cases/neural/train.json").resolve()}"',
).replace(
    'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
    f'output_root = "{Path("artifacts/development/neural-combat/runs/config-validation-gpu").resolve()}"',
)
source = source.replace('device = "cpu"', 'device = "cuda:0"')
source += '\n[runtime]\ngpu_uuid = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"\n'
with tempfile.TemporaryDirectory() as directory:
    gpu_config = Path(directory) / "standalone-gpu.toml"
    gpu_config.write_text(source)
    assert main(["train", "--config", str(gpu_config), "--validate-only"]) == 0
assert main(["calibrate", "--config", "configs/neural/calibration.toml", "--validate-only"]) == 0
unittest.defaultTestLoader.discover("tests")
assert attempts == [], attempts
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", program],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_validate_only_never_creates_output_or_assembles_runtime(self):
        train_config = load_neural_config(ROOT / "configs/neural/standalone.toml")
        evaluation_config = load_neural_config(ROOT / "configs/neural/standalone-selection.toml")
        self.assertFalse(train_config.output_root.exists())
        self.assertFalse(evaluation_config.output_root.exists())
        with patch("spire_learn.neural.runner.create_runtime",
                   side_effect=AssertionError("validation must not create a runtime")):
            self.assertEqual(self.call("train", "--config", str(ROOT / "configs/neural/standalone.toml"),
                                       "--validate-only")[0], 0)
            self.assertEqual(self.call("evaluate", "--config", str(ROOT / "configs/neural/standalone-selection.toml"),
                                       "--policy", "end-turn-last-v1", "--validate-only")[0], 0)
        self.assertFalse(train_config.output_root.exists())
        self.assertFalse(evaluation_config.output_root.exists())

    def test_evaluation_policy_checkpoint_and_final_guards(self):
        selection = str(ROOT / "configs/neural/standalone-selection.toml")
        final = str(ROOT / "configs/neural/standalone-final.toml")
        self.assertEqual(self.call("evaluate", "--config", selection)[0], 2)
        self.assertEqual(self.call("evaluate", "--config", selection, "--policy", "legal-random",
                                   "--checkpoint", "/not/allowed")[0], 2)
        self.assertEqual(self.call("evaluate", "--config", selection, "--policy", "legal-random",
                                   "--action-selection", "deterministic")[0], 2)
        self.assertEqual(self.call("evaluate", "--config", selection, "--policy", "end-turn-last-v1",
                                   "--action-selection", "seeded-sampling")[0], 2)
        self.assertEqual(self.call("evaluate", "--config", final, "--checkpoint", "/checkpoint")[0], 2)
        self.assertEqual(self.call("evaluate", "--config", final, "--checkpoint", "/checkpoint",
                                   "--allow-final", "--validate-only")[0], 0)

    def test_training_sources_are_mutually_exclusive_and_calibration_dispatch_is_lazy(self):
        config = str(ROOT / "configs/neural/standalone.toml")
        self.assertEqual(self.call("train", "--config", config, "--resume", "/resume",
                                   "--warm-start", "/warm")[0], 2)
        self.assertEqual(self.call("train", "--config", config, "--resume", "/resume",
                                   "--validate-only")[0], 0)
        self.assertEqual(self.call("calibrate", "--config", config)[0], 2)
        calibration = str(ROOT / "configs/neural/calibration.toml")
        status, stdout, _ = self.call("calibrate", "--config", calibration, "--validate-only")
        self.assertEqual(status, 0)
        self.assertIn("calibration configuration valid", stdout)
        self.assertIn("were not executed", stdout)
        from spire_learn.neural.runner import NeuralReport

        report = NeuralReport({"status": "complete", "cohort": {"reserved_action_slots": 1024}}, 0, None)
        with (
            patch("spire_learn.neural.runner.create_runtime",
                  side_effect=AssertionError("calibration must not use the training runtime")),
            patch("spire_learn.neural.runner.calibrate", return_value=report) as calibrate,
        ):
            status, output, _ = self.call("calibrate", "--config", calibration)
        self.assertEqual(status, 0)
        self.assertIn('"reserved_action_slots": 1024', output)
        calibrate.assert_called_once()

    def test_train_dispatch_selects_resume_or_warm_start_without_installing_runtime_early(self):
        from spire_learn.neural.runner import NeuralReport

        runtime = object()
        with (
            patch("spire_learn.neural.runner.create_runtime", return_value=runtime) as create,
            patch("spire_learn.neural.runner.train", return_value=NeuralReport({"status": "complete"}, 0, None)) as train,
        ):
            status, stdout, _ = self.call("train", "--config", str(ROOT / "configs/neural/standalone.toml"),
                                          "--warm-start", "/parent/checkpoint")
        self.assertEqual(status, 0)
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["purpose"], "warm_start")
        self.assertEqual(create.call_args.kwargs["checkpoint"], Path("/parent/checkpoint"))
        train.assert_called_once_with(load_neural_config(ROOT / "configs/neural/standalone.toml"),
                                      runtime=runtime)
        self.assertIn('"status": "complete"', stdout)


if __name__ == "__main__":
    unittest.main()
