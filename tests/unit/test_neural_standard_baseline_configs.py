"""Standard baseline plans and templates remain bound to the Standard learner contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.neural.baseline_entry import _refuse_overflowing_preset, load_plan
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.runner import create_runtime
from spire_learn.neural.stream_resume import _stable_document


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
    def _load_template(self, template_path: Path, **options):
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
            return load_neural_config(path, **options)

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

    def test_gym_124_baselines_change_only_the_pin_recipe_and_versioned_names(self):
        revision = "673eb8f220023b0c95401bbf96bec814a359457b"
        recipe_path = "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
        runtime = json.loads((CONFIGS / "gym-1.2.4/runtime.json").read_text())
        self.assertEqual(runtime["gym_revision"], revision)
        self.assertEqual(runtime["standard_fingerprints"]["combat_compact"], COMPACT_FINGERPRINT)
        recipe = json.loads((ROOT / recipe_path).read_text())
        self.assertEqual(recipe["states_table"]["sha256"],
                         "3121570f1b970d96acd76e9c6c7681ca32789e22c9b11af74ab5aa9283dfb6a0")

        for old_name, seeds in (("baseline-standard-1.2.0", (1103, 2207, 3301, 4409)),
                                ("baseline-standard-1.2.0-r2", (5101, 6203, 7307, 8419))):
            new_name = old_name.replace("1.2.0", "1.2.4")
            with self.subTest(config=new_name):
                old_dir, new_dir = CONFIGS / old_name, CONFIGS / new_name
                expected_plan = load_plan(old_dir / "plan.json")
                expected_plan.update(
                    baseline_id=expected_plan["baseline_id"].replace(old_name, new_name),
                    gym_revision=revision,
                    recipe=recipe_path,
                    template=f"configs/neural/{new_name}/template.toml",
                )
                plan = load_plan(new_dir / "plan.json")
                self.assertEqual(plan, expected_plan)
                self.assertEqual(plan["max_turn_actions"], 500)
                self.assertEqual(plan["target_learned_decisions"], TARGET)
                self.assertEqual(plan["retain_near_decisions"], RETAINED)
                self.assertEqual(plan["evaluation"]["case_actions"], 1024)

                expected_template = tomllib.loads((old_dir / "template.toml").read_text())
                expected_template["output_root"] = expected_template["output_root"].replace(
                    "/gym-1.2.0/", "/gym-1.2.4/")
                expected_template["environment"]["gym_revision"] = revision
                expected_template["budgets"]["max_turn_actions"] = 500
                mappings = expected_template["accepted_identities"][0]["accepted_raw_mode_mappings"]
                mappings["standard-standalone"] = (
                    f"spire-learn-standard-combat-compact-v1@{COMPACT_FINGERPRINT}+gym-1.2.4")
                template = ROOT / plan["template"]
                document = tomllib.loads(template.read_text())
                self.assertEqual(document, expected_template)
                self.assertEqual(document["budgets"]["max_turn_actions"], 500)
                self.assertEqual(document["rollout"], {"rows": 16, "steps": 32})

                config = self._load_template(template)
                self.assertEqual(config.environment_revision, revision)
                self.assertEqual(config.standard_view_preset, COMPACT_PRESET)
                self.assertEqual(config.budgets.max_turn_actions, 500)
                self.assertEqual(tuple(config.seed_streams.__dict__.values()), seeds)
                _refuse_overflowing_preset(template)

    def test_stage_one_profiles_change_only_the_named_settings(self):
        directory = CONFIGS / "profile-standard-1.2.0"
        variants = {
            "compact": (),
            "compact-fast": (("model", "trim_padding", True), ("budgets", "validation_stride", 16)),
            "compact-w8": (("environment", "workers", 8),),
            "compact-fast-w8": (("model", "trim_padding", True), ("budgets", "validation_stride", 16),
                                ("environment", "workers", 8)),
            "compact-fast-inexact": (("model", "trim_padding", True), ("budgets", "validation_stride", 16),
                                     ("model", "experimental_trim_compute", True)),
        }
        for rows in (16, 64):
            source = load_plan(directory / f"plan-{rows}-compact.json")
            for variant, changes in variants.items():
                with self.subTest(rows=rows, variant=variant):
                    plan = load_plan(directory / f"plan-{rows}-{variant}.json")
                    expected_plan = dict(source, template=source["template"].replace("compact", variant),
                                         baseline_id=source["baseline_id"].replace("compact", variant))
                    self.assertEqual(plan, expected_plan)
                    expected = tomllib.loads((ROOT / source["template"]).read_text())
                    for section, key, value in changes:
                        expected[section][key] = value
                    template = ROOT / plan["template"]
                    self.assertEqual(tomllib.loads(template.read_text()), expected)
                    experimental = variant.endswith("-inexact")
                    config = self._load_template(template, allow_experimental_trim_compute=experimental)
                    self.assertEqual(config.workers, expected["environment"]["workers"])
                    self.assertEqual(config.rollout_rows, rows)
                    self.assertIs(config.trim_padding, expected["model"].get("trim_padding"))
                    self.assertEqual(config.effective_validation_stride,
                                     expected["budgets"].get("validation_stride", 1))
                    self.assertIs(config.experimental_trim_compute, True if experimental else None)
                    if experimental:
                        exact = self._load_template(directory / f"template-{rows}-compact-fast.toml")
                        self.assertEqual(config.accepted_identities, exact.accepted_identities)
                        self.assertNotEqual(config.config_sha256, exact.config_sha256)
                        with self.assertRaisesRegex(ValueError, "restricted to profile_entry"):
                            self._load_template(template)
                        with self.assertRaisesRegex(ValueError, "restricted to profile_entry"):
                            create_runtime(config, purpose="train")
                        with self.assertRaisesRegex(ValueError, "baseline plans refuse"):
                            _refuse_overflowing_preset(template)

    def test_experimental_trim_requires_trim_padding_and_is_outside_stream_identity(self):
        template = CONFIGS / "profile-standard-1.2.0/template-16-compact-fast-inexact.toml"
        text = template.read_text()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "template.toml"
            path.write_text(text.replace("trim_padding = true", "trim_padding = false"))
            with self.assertRaisesRegex(ValueError, "requires model.trim_padding"):
                self._load_template(path, allow_experimental_trim_compute=True)

        document = tomllib.loads(text)
        document["scenario_stream"] = dict(schema="spire-learn-scenario-stream-v1", recipe_path="recipe.json",
                                           arm="mixed", shard_index=0, distribution_sha256="a" * 64)
        experimental = _stable_document(document)
        del document["model"]["experimental_trim_compute"]
        self.assertEqual(_stable_document(document), experimental)

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
