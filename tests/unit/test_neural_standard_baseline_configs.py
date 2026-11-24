"""Standard baseline plans and templates remain bound to the Standard learner contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.neural.baseline_entry import load_plan
from spire_learn.neural.config import load_neural_config


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/neural"
GYM_120_REVISION = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"
GYM_120_RECIPE = "configs/neural/baalorlord-standard-gym-1.2.0/recipe.json"
GYM_120_RECIPE_SHA = "b06000c9c53679d4eb32014f5307d3636ecffdb26a6791202a40ad62c153d690"
COMPACT_PRESET = "standard-combat-compact-v1"
COMPACT_FINGERPRINT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
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
        template_output = tomllib.loads(text)["output_root"]
        text = text.replace(f'output_root = "{template_output}"', f'output_root = "{output}"')
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

    def test_gym_120_baselines_change_only_the_pin_layout_and_versioned_paths(self):
        recipe_bytes = (ROOT / GYM_120_RECIPE).read_bytes()
        self.assertEqual(hashlib.sha256(recipe_bytes).hexdigest(), GYM_120_RECIPE_SHA)
        runtime = json.loads((CONFIGS / "gym-1.2.0/runtime.json").read_text())
        self.assertEqual(runtime["gym_revision"], GYM_120_REVISION)
        self.assertEqual(runtime["standard_fingerprints"]["combat_compact"], COMPACT_FINGERPRINT)

        for old_name, seeds in SEEDS.items():
            new_name = old_name.replace("1.1.1", "1.2.0")
            with self.subTest(config=new_name):
                old_dir, new_dir = CONFIGS / old_name, CONFIGS / new_name
                expected_plan = load_plan(old_dir / "plan.json")
                expected_plan.update(
                    baseline_id=expected_plan["baseline_id"].replace(old_name, new_name),
                    gym_revision=GYM_120_REVISION,
                    recipe=GYM_120_RECIPE,
                    max_turn_actions=500,
                    template=f"configs/neural/{new_name}/template.toml",
                )
                self.assertEqual(load_plan(new_dir / "plan.json"), expected_plan)

                expected_template = tomllib.loads((old_dir / "template.toml").read_text())
                expected_template["output_root"] = expected_template["output_root"].replace(
                    "/gym-1.1.1/", "/gym-1.2.0/")
                expected_template["environment"].update(
                    gym_revision=GYM_120_REVISION,
                    standard_view_layout="compact",
                    standard_view_preset=COMPACT_PRESET,
                    standard_view_fingerprint=COMPACT_FINGERPRINT,
                )
                mappings = expected_template["accepted_identities"][0]["accepted_raw_mode_mappings"]
                mappings["standard-standalone"] = (
                    f"spire-learn-standard-combat-compact-v1@{COMPACT_FINGERPRINT}+gym-1.2.0"
                )
                del mappings["standard-run"]
                template = ROOT / expected_plan["template"]
                self.assertEqual(tomllib.loads(template.read_text()), expected_template)

                config = self._load_template(template)
                self.assertEqual(config.environment_revision, GYM_120_REVISION)
                self.assertEqual(config.standard_view_layout, "compact")
                self.assertEqual(config.standard_view_preset, COMPACT_PRESET)
                self.assertEqual(tuple(config.seed_streams.__dict__.values()), seeds)

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
