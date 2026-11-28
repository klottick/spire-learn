"""The S1 distillation training config and student evaluation template and plan (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

from spire_learn.neural import cases
from spire_learn.neural.config import (
    DistillHyperparameters, _canonical_hash, distill_config_hash, load_distill_config, load_neural_config,
    model_config_hash, model_config_hash_payload,
)
from spire_learn.neural.contracts import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.neural.student_eval_entry import check_template, load_plan  # noqa: E402

FOLDER = ROOT / "configs/neural/distill-gym-1.2.5"
R1_MODEL_HASH = "4fb628e06bf4952847795ac5d8548d12dfeede4c97e874ab843857ae0f91cd2f"
R1_TEMPLATE = ROOT / "configs/neural/baseline-standard-1.2.4/template.toml"
STUDENT = ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000, value_heads=3)


class S1ConfigTests(unittest.TestCase):
    def test_training_config_refuses_each_placeholder(self):
        text = (FOLDER / "s1-pilot.toml").read_text()
        self.assertEqual(len(re.findall(r'"PLACEHOLDER"', text)), 3)
        with self.assertRaisesRegex(ValueError, "distill.tau"):
            load_distill_config(FOLDER / "s1-pilot.toml")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "s1-pilot.toml"
            tau_only = text.replace('tau = "PLACEHOLDER"', "tau = 0.05")
            path.write_text(tau_only.replace('output_root = "../../../', f'output_root = "{ROOT}/'))
            with self.assertRaisesRegex(ValueError, "distill.gradient_steps"):
                load_distill_config(path)

    def test_filled_training_config_loads_with_the_pilot_settings(self):
        text = (FOLDER / "s1-pilot.toml").read_text()
        hyper = DistillHyperparameters(tau=0.05, learning_rate=0.0003, batch_rows=512, gradient_steps=18000,
                                       data_fraction=1.0)
        filled = (text.replace('tau = "PLACEHOLDER"', "tau = 0.05")
                  .replace('gradient_steps = "PLACEHOLDER"', "gradient_steps = 18000")
                  .replace('distill = "PLACEHOLDER"', f'distill = "{distill_config_hash(hyper)}"'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s1-pilot.toml"
            path.write_text(filled.replace('output_root = "../../../', f'output_root = "{ROOT}/'))
            config = load_distill_config(path)
        self.assertEqual(config.model, STUDENT)
        self.assertEqual((config.seed, config.eval_every, config.device), (1, 1000, "cuda:0"))
        self.assertEqual((config.distill.learning_rate, config.distill.batch_rows, config.distill.data_fraction),
                         (0.0003, 512, 1.0))
        self.assertEqual(config.identity.model_id, "spire-learn-combat-ff-standard-v1")

    def test_model_matches_r1_apart_from_value_heads(self):
        r1 = _canonical_hash(model_config_hash_payload(ModelConfig(16, 64, 10_000_000)))
        self.assertEqual(r1, R1_MODEL_HASH)
        payload = dict(model_config_hash_payload(STUDENT))
        self.assertEqual(payload.pop("value_heads"), 3)
        self.assertEqual(_canonical_hash(payload), R1_MODEL_HASH)
        import tomllib
        r1_model = tomllib.loads(R1_TEMPLATE.read_text())["model"]
        self.assertEqual(payload, r1_model)
        for name in ("s1-pilot.toml", "student-eval-template.toml"):
            document = tomllib.loads((FOLDER / name).read_text())
            self.assertEqual(document["model"], {**r1_model, "value_heads": 3})
        self.assertEqual(model_config_hash(STUDENT), tomllib.loads(
            (FOLDER / "s1-pilot.toml").read_text())["identity"]["config_hashes"]["model"])

    def test_plan_loads_and_matches_the_baseline_gpu_wait(self):
        plan = load_plan(FOLDER / "student-eval-plan.json")
        self.assertEqual((plan["evaluation"]["case_actions"], plan["max_turn_actions"]), (1024, 500))
        self.assertEqual(plan["evaluation"]["action_selection"], "deterministic")
        baseline = json.loads((ROOT / "configs/neural/baseline-standard-1.2.4/plan.json").read_text())
        self.assertEqual(plan["gpu_wait"], baseline["gpu_wait"])
        self.assertEqual(plan["gym_revision"], "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1")
        self.assertTrue((ROOT / plan["recipe"]).is_file())
        self.assertEqual(ROOT / plan["template"], FOLDER / "student-eval-template.toml")

    def test_template_passes_the_entry_check_and_loads_as_a_case_config(self):
        plan = load_plan(FOLDER / "student-eval-plan.json")
        identity = check_template(FOLDER / "student-eval-template.toml", plan)
        self.assertEqual(identity["config_hashes"], {"model": model_config_hash(STUDENT)})
        import tomllib
        document = tomllib.loads((FOLDER / "student-eval-template.toml").read_text())
        self.assertEqual((document["operation"], document["split"], document["partition"], document["device"]),
                         ("evaluate", "selection", "selection", "cuda:0"))
        # What panel_configs does per case: a one-case selection manifest in place of the placeholder.
        manifest = json.loads((ROOT / "configs/cases/neural/standard-standalone-train.json").read_text())
        manifest.update(split="selection", partition="selection")
        manifest["cases"][0].update(environment_seed=cases._SELECTION_START, case_id="s1-selection-case",
                                    lineage_id="lineage:standard:selection:standalone:ironclad:a0")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "case.json").write_text(json.dumps(manifest))
            text = (FOLDER / "student-eval-template.toml").read_text()
            text = text.replace('manifest_path = "../../cases/neural/standard-standalone-train.json"',
                                f'manifest_path = "{root / "case.json"}"')
            text = text.replace('output_root = "../../../', f'output_root = "{ROOT}/')
            path = root / "config.toml"
            path.write_text(text)
            config = load_neural_config(path)
        self.assertEqual(config.model.value_heads, 3)
        self.assertEqual(config.operation, "evaluate")


if __name__ == "__main__":
    unittest.main()
