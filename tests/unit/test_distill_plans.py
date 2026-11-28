"""Task 13 plans: PPO from distilled S1 and the branched decline diagnostic copy the v1.2.4 baselines."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest

from scripts.neural import baseline_entry
from spire_learn.neural.config import load_neural_config

ROOT = Path(__file__).resolve().parents[2]
N = ROOT / "configs/neural"
GYM = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"
RECIPE = "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
NOLEVEL4 = "configs/neural/baalorlord-standard-gym-1.2.4-nolevel4/recipe.json"
R2_SUMMARY = ROOT / "artifacts/development/neural-combat/runs/gym-1.2.4/standard-baseline-r2/summary.json"
R2_NEAR = ("artifacts/development/neural-combat/runs/gym-1.2.4/standard-baseline-r2/prepared/train-296-shard-296/"
           "runs/20261126T044104Z-e3c9a38a47f84b4ba588d3cc721d786e/checkpoint-final")
SEEDS = {"r1": {"policy": 1103, "optimizer": 2207, "curriculum": 3301, "noncombat": 4409},
         "r2": {"policy": 5101, "optimizer": 6203, "curriculum": 7307, "noncombat": 8419},
         "new": {"policy": 9101, "optimizer": 9203, "curriculum": 9307, "noncombat": 9419}}
# Every plan may differ from its source in these; the template path and baseline id name the new run.
PLAN_OWN = {"baseline_id", "template", "gym_revision", "initial_checkpoint"}
DECLINE_OWN = PLAN_OWN | {"recipe", "start_shard", "decision_offset", "target_learned_decisions",
                          "retain_near_decisions", "notes"}
# Template leaves that may differ: the Gym, the output root and the pilot's learning rate with its hash.
TEMPLATE_OWN = {("environment", "gym_revision"), ("output_root",), ("ppo", "learning_rate"),
                ("accepted_identities", 0, "config_hashes", "ppo")}


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (*path, key))
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, index))
    else:
        yield path, value


def _template_diff(old: Path, new: Path) -> set[tuple]:
    a = dict(_leaves(tomllib.loads(old.read_text())))
    b = dict(_leaves(tomllib.loads(new.read_text())))
    return {path for path in set(a) | set(b) if a.get(path, KeyError) != b.get(path, KeyError)}


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _plan(name: str) -> dict:
    return baseline_entry.load_plan(N / name / "plan.json")


class DistillPlanTests(unittest.TestCase):
    DISTILL = (("baseline-standard-1.2.5-distill", "baseline-standard-1.2.4", "r1",
                "artifacts/development/neural-combat/distill/export/s1-seed-1-r1/checkpoint",
                "runs/gym-1.2.5/standard-distill"),
               ("baseline-standard-1.2.5-distill-r2", "baseline-standard-1.2.4-r2", "r2",
                "artifacts/development/neural-combat/distill/export/s1-seed-2-r2/checkpoint",
                "runs/gym-1.2.5/standard-distill-r2"))
    DECLINE = (("a1", RECIPE, "r2"), ("a2", RECIPE, "new"), ("b1", NOLEVEL4, "r2"), ("b2", NOLEVEL4, "new"))

    def _check_template(self, name, source, seeds, *, output_root):
        template = N / name / "template.toml"
        changed = _template_diff(N / source / "template.toml", template)
        self.assertLessEqual(changed - {("seeds", key) for key in SEEDS["r1"]}, TEMPLATE_OWN, name)
        config = load_neural_config(template)  # its declared model and PPO hashes match its settings
        self.assertEqual(config.environment_revision, GYM)
        self.assertEqual(vars(config.seed_streams), SEEDS[seeds])
        self.assertEqual(config.ppo.learning_rate, 3e-4)  # the v1.2.4 rate until the Task 14 pilot
        self.assertTrue(str(config.output_root).endswith(output_root), config.output_root)
        identity = config.accepted_identities[0]
        self.assertEqual((identity.model_id, identity.features_id, identity.action_id, identity.objective_id),
                         ("spire-learn-combat-ff-standard-v1", "spire-learn-combat-features-standard-v1",
                          "spire-learn-combat-dispatch-v1", "spire-learn-combat-winloss-v1"))
        return config

    def test_plans_differ_only_in_allowed_fields(self):
        for name, source, seeds, checkpoint, output_root in self.DISTILL:
            with self.subTest(plan=name):
                plan, old = _plan(name), _plan(source)
                self.assertEqual({k for k in set(plan) | set(old) if plan.get(k) != old.get(k)},
                                 PLAN_OWN, name)
                self.assertEqual((plan["gym_revision"], plan["initial_checkpoint"]), (GYM, checkpoint))
                self.assertEqual(plan["template"], f"configs/neural/{name}/template.toml")
                self.assertTrue(plan.get("evaluate", True))
                self._check_template(name, source, seeds, output_root=output_root)
        r2 = _plan("baseline-standard-1.2.4-r2")
        for arm, recipe, seeds in self.DECLINE:
            name = f"baseline-standard-1.2.5-decline-{arm}"
            with self.subTest(plan=name):
                plan = _plan(name)
                self.assertLessEqual({k for k in set(plan) | set(r2) if plan.get(k) != r2.get(k)}, DECLINE_OWN)
                self.assertEqual({key: plan[key] for key in ("recipe", "initial_checkpoint", "start_shard",
                                                             "decision_offset", "target_learned_decisions",
                                                             "retain_near_decisions", "gym_revision")},
                                 {"recipe": recipe, "initial_checkpoint": R2_NEAR, "start_shard": 297,
                                  "decision_offset": 1216512, "target_learned_decisions": 405504,
                                  "retain_near_decisions": [], "gym_revision": GYM})
                self.assertEqual(plan["decision_offset"] + plan["target_learned_decisions"],
                                 r2["target_learned_decisions"])
                self._check_template(name, "baseline-standard-1.2.4-r2", seeds,
                                     output_root=f"runs/gym-1.2.5/standard-decline-{arm}")

    def test_pilot_learning_rate_edit_is_allowed(self):
        # The Task 14 pilot edits ppo.learning_rate; the identity's PPO hash follows it (the loader checks it).
        import tempfile

        from spire_learn.neural.config import _canonical_hash

        for name, source in (("baseline-standard-1.2.5-distill", "baseline-standard-1.2.4"),
                             ("baseline-standard-1.2.5-distill-r2", "baseline-standard-1.2.4-r2")):
            text = (N / name / "template.toml").read_text()
            ppo = tomllib.loads(text)["ppo"]
            old_hash = _canonical_hash(ppo)
            new_hash = _canonical_hash({**ppo, "learning_rate": 1e-4})
            edited = text.replace("learning_rate = 0.0003", "learning_rate = 0.0001").replace(old_hash, new_hash)
            with self.subTest(plan=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / name / "template.toml"
                path.parent.mkdir()
                path.write_text(edited)
                self.assertEqual(_template_diff(N / source / "template.toml", path) - TEMPLATE_OWN, set())
                self.assertIn(("ppo", "learning_rate"), _template_diff(N / name / "template.toml", path))
                with self.assertRaisesRegex(ValueError, "component hash"):
                    load_neural_config(_write(path.with_name("stale.toml"),
                                              text.replace("learning_rate = 0.0003", "learning_rate = 0.0001")))

    def test_decline_start_is_r2s_near_1216512_checkpoint(self):
        # r2's quarter thresholds: near-1216512 is segment 297 on shard 296, so the branch resumes at shard 297.
        if not R2_SUMMARY.is_file():
            self.skipTest("r2's run output exists only in the main tree")
        rows = {row["label"]: row for row in json.loads(R2_SUMMARY.read_text())["checkpoints"]}
        near = rows["near-1216512"]
        self.assertEqual((near["learned_decisions"], near["shard_index"] + 1), (1216512, 297))
        self.assertEqual(Path(near["path"]).relative_to(R2_SUMMARY.parents[6]).as_posix(), R2_NEAR)


class NoLevel4RecipeTests(unittest.TestCase):
    def test_nolevel4_recipe_diff_and_panel_identity(self):
        old = json.loads((ROOT / RECIPE).read_text())
        new = json.loads((ROOT / NOLEVEL4).read_text())
        self.assertEqual({key for key in set(old) | set(new) if old.get(key) != new.get(key)}, {"fuzz_schedule"})
        self.assertEqual(old["fuzz_schedule"][:-1], new["fuzz_schedule"][:-1])
        self.assertEqual(old["fuzz_schedule"][-1], {"from_shard": 297, "levels": [0, 1, 2, 4]})
        self.assertEqual(new["fuzz_schedule"][-1], {"from_shard": 297, "levels": [0, 1, 2]})
        self.assertFalse(any(4 in step["levels"] for step in new["fuzz_schedule"]))
        # Same table path from the same directory depth, so both resolve the same states table.
        self.assertEqual(Path(NOLEVEL4).parent.parent, Path(RECIPE).parent.parent)
        from spire_learn.neural.standard_stream import load_standard_recipe, panel_identity, panel_sha256

        try:
            a, b = load_standard_recipe(ROOT / RECIPE), load_standard_recipe(ROOT / NOLEVEL4)
        except ValueError as error:
            if "run-states table is missing" in str(error):
                self.skipTest("the local v1.2.4 run-states table is absent")
            raise
        self.assertNotEqual(a.sha256, b.sha256)  # training shards differ (seeded from the full recipe)
        for kind in ("selection", "boss"):
            self.assertEqual(panel_identity(a, kind=kind), panel_identity(b, kind=kind))
            self.assertEqual(panel_sha256(a, kind=kind), panel_sha256(b, kind=kind))
        self.assertNotIn("level:4", b.category_keys)


if __name__ == "__main__":
    unittest.main()
