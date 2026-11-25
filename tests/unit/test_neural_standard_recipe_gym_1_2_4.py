"""The v1.2.4 Standard recipe differs from the v1.2.0 one only in its table, and each is bound to its own Gym."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural import gym_contracts, recipes
from spire_learn.neural.standard_stream import load_standard_recipe

ROOT = Path(__file__).resolve().parents[2]
R120 = ROOT / "configs/neural/baalorlord-standard-gym-1.2.0/recipe.json"
R124 = ROOT / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
TABLE_124 = ROOT / "artifacts/development/neural-combat/run-states/run-states-gym-1.2.4.json"
TABLE_120 = ROOT / "artifacts/development/neural-combat/run-states/run-states-gym-1.2.0.json"
SHA_124 = "3121570f1b970d96acd76e9c6c7681ca32789e22c9b11af74ab5aa9283dfb6a0"
REV_120 = gym_contracts.GYM_V1_2_0_CONTRACT.source_revision
REV_124 = gym_contracts.GYM_V1_2_4_CONTRACT.source_revision


class RecipeGym124Test(unittest.TestCase):
    def test_differs_from_v120_only_in_the_table(self):
        old, new = json.loads(R120.read_text()), json.loads(R124.read_text())
        self.assertEqual(set(old), set(new))
        self.assertEqual({k for k in old if old[k] != new[k]}, {"states_table"})
        self.assertEqual(set(new["states_table"]), {"path", "sha256"})
        self.assertEqual(new["states_table"]["sha256"], SHA_124)
        self.assertEqual(new["states_table"]["path"],
                         old["states_table"]["path"].replace("gym-1.2.0", "gym-1.2.4"))

    def _recipes(self):
        if not (TABLE_124.is_file() and TABLE_120.is_file()):
            self.skipTest("the local v1.2.0 and v1.2.4 run-states tables are absent")
        return load_standard_recipe(R120), load_standard_recipe(R124)

    def test_loads_and_resolves_to_its_contract(self):
        old, new = self._recipes()
        self.assertEqual(new.table_sha256, SHA_124)
        self.assertEqual(new.gym_revision, REV_124)
        self.assertEqual(old.gym_revision, REV_120)
        self.assertIs(gym_contracts.contract_for_revision(new.gym_revision), gym_contracts.GYM_V1_2_4_CONTRACT)
        self.assertEqual(len(old.states), len(new.states))  # byte-identical states, same character
        self.assertGreater(len(new.states), 0)

    def test_each_recipe_is_refused_under_the_other_contract_and_live_gym(self):
        old, new = self._recipes()
        recipes.require_recipe_revision(new, REV_124)
        recipes.require_recipe_revision(old, REV_120)
        for recipe, other in ((new, REV_120), (old, REV_124)):
            with self.assertRaisesRegex(ValueError, "Standard recipe"):
                recipes.require_recipe_revision(recipe, other)
        for recipe, live, other in ((new, REV_124, REV_120), (old, REV_120, REV_124)):
            with patch.object(recipes, "_VALIDATOR", SimpleNamespace(live_gym_revision=live)):
                self.assertIsNotNone(recipes.gym_validator("test", recipe))
            with patch.object(recipes, "_VALIDATOR", SimpleNamespace(live_gym_revision=other)):
                with self.assertRaisesRegex(ValueError, "run it in that Gym's runtime"):
                    recipes.gym_validator("test", recipe)


if __name__ == "__main__":
    unittest.main()
