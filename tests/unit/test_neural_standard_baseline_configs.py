"""Standard baseline plans and templates remain bound to the Standard learner contract."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.neural.baseline_entry import load_plan
from spire_learn.neural.config import load_neural_config


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/neural"
TARGET = 1_622_016
RETAINED = [405_504, 811_008, 1_216_512]
SEEDS = {
    "baseline-standard-1.1.1": (1103, 2207, 3301, 4409),
    "baseline-standard-1.1.1-r2": (5101, 6203, 7307, 8419),
}


class StandardBaselineConfigTests(unittest.TestCase):
    def _load_template(self, template_path: Path):
        """Validate the authored template against the existing static Standard fixture."""
        text = template_path.read_text()
        manifest = ROOT / "configs/cases/neural/standard-standalone-train.json"
        output = ROOT / "artifacts/development/neural-combat/runs/config-validation"
        text = text.replace(
            'manifest_path = "../../cases/neural/standard-standalone-train.json"',
            f'manifest_path = "{manifest}"',
        )
        text = text.replace(
            'output_root = "../../../artifacts/development/neural-combat/runs/gym-1.1.1/standard-baseline"',
            f'output_root = "{output}"',
        ).replace(
            'output_root = "../../../artifacts/development/neural-combat/runs/gym-1.1.1/standard-baseline-r2"',
            f'output_root = "{output}"',
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "template.toml"
            path.write_text(text)
            return load_neural_config(path)

    def test_r1_and_r2_plans_and_templates_bind_standard_baseline(self):
        plans = {}
        configs = {}
        for name, seeds in SEEDS.items():
            with self.subTest(config=name):
                directory = CONFIGS / name
                plan = load_plan(directory / "plan.json")
                config = self._load_template(directory / "template.toml")
                plans[name], configs[name] = plan, config

                self.assertEqual(plan["target_learned_decisions"], TARGET)
                self.assertEqual(plan["retain_near_decisions"], RETAINED)
                self.assertEqual(plan["evaluation"]["case_actions"], 1024)
                self.assertEqual(plan["evaluation"]["case_seconds"], 120)
                self.assertEqual(plan["recipe"], "configs/neural/baalorlord-standard-v1/recipe.json")
                self.assertNotIn("additional_panels", plan)
                self.assertEqual(plan["gym_revision"], "e3c13836bd1458be8a87b93160f5c350f78fcd80")
                self.assertEqual(plan["gpu_wait"], load_plan(
                    CONFIGS / "baseline-gym-1.1.1-cv2/plan.json")["gpu_wait"])
                self.assertEqual(plan["wall_seconds"], {"training": 12_600, "total": 19_800})
                self.assertGreaterEqual(plan["runner_timeout_seconds"],
                                        19_800 + 21_600 + 30 + 1_200)

                budget = config.budgets
                self.assertEqual(budget.max_training_decisions, 1_638_400)
                self.assertGreater(budget.max_training_decisions, TARGET)
                self.assertEqual(budget.max_optimizer_steps, 262_144)
                self.assertEqual(budget.max_environment_actions, 8_000_000)
                self.assertEqual(budget.max_wall_seconds, 12_600.0)
                self.assertEqual(config.mode, "standard-standalone")
                self.assertEqual(config.environment_revision,
                                 "e3c13836bd1458be8a87b93160f5c350f78fcd80")
                self.assertEqual(config.rollout_rows, 16)
                identity = config.accepted_identities[0]
                self.assertEqual(identity.model_id, "spire-learn-combat-ff-standard-v1")
                self.assertEqual(identity.features_id, "spire-learn-combat-features-standard-v1")
                self.assertEqual(identity.action_id, "spire-learn-combat-dispatch-v1")
                self.assertEqual(identity.objective_id, "spire-learn-combat-winloss-v1")
                self.assertEqual(tuple(config.seed_streams.__dict__.values()), seeds)

        first, second = (configs["baseline-standard-1.1.1"],
                         configs["baseline-standard-1.1.1-r2"])
        self.assertEqual(first.model, second.model)
        self.assertEqual(first.ppo, second.ppo)
        cv2_seeds = tomllib.loads((CONFIGS / "baseline-gym-1.1.1-cv2/template.toml").read_text())["seeds"]
        cv2_r2_seeds = tomllib.loads((CONFIGS / "baseline-gym-1.1.1-cv2-r2/template.toml").read_text())["seeds"]
        self.assertEqual(tuple(first.seed_streams.__dict__.values()), tuple(cv2_seeds.values()))
        self.assertEqual(tuple(second.seed_streams.__dict__.values()), tuple(cv2_r2_seeds.values()))

    def test_recipe_schedule_uses_training_quartiles_in_shard_units(self):
        recipe_path = CONFIGS / "baalorlord-standard-v1/recipe.json"
        schedule = json.loads(recipe_path.read_text())["fuzz_schedule"]
        self.assertEqual(schedule, [
            {"from_shard": 0, "levels": [0]},
            {"from_shard": 99, "levels": [0, 1]},
            {"from_shard": 198, "levels": [0, 1, 2]},
            {"from_shard": 297, "levels": [0, 1, 2, 4]},
        ])
        self.assertEqual(TARGET // 4096, 396)
        self.assertEqual([396 // 4, 396 // 2, 3 * 396 // 4], [99, 198, 297])


if __name__ == "__main__":
    unittest.main()
