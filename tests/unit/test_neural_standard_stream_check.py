"""The stream-check script validates only recipes of the live Gym's revision (injected build info)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import standard_stream_check as check
from spire_learn.neural import recipes
from tests.unit.test_neural_standard_stream import GYM_1_1_1, GYM_1_2_0, recipe_document, write_table


class StreamCheckRevisionTests(unittest.TestCase):
    def setUp(self):
        previous = recipes.bind_validator(None)
        self.addCleanup(recipes.bind_validator, previous)

    def run_check(self, directory, recipe_revision, build_info):
        _, sha = write_table(Path(directory), revision=recipe_revision)
        path = Path(directory) / f"recipe-{recipe_revision[:4]}.json"
        path.write_text(json.dumps(recipe_document(sha)))
        recipes.bind_validator(None)  # force a fresh binding for every case
        with patch.object(recipes, "bind_gym", lambda: (lambda catalog: None, [])):
            return check.main(["--recipe", str(path), "--output", str(Path(directory) / "out"), "--shards", "0",
                               "--panels"], build_info=build_info)

    def test_refuses_the_other_revision_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for recipe_revision, live in ((GYM_1_1_1, GYM_1_2_0), (GYM_1_2_0, GYM_1_1_1)):
                with self.subTest(recipe=recipe_revision[:8], live=live[:8]):
                    with self.assertRaisesRegex(ValueError, "loaded Spire Gym"):
                        self.run_check(tmp, recipe_revision, lambda live=live: {"revision": live})
                    self.assertFalse((Path(tmp) / "out" / "summary.json").exists())

    def test_a_missing_live_revision_is_a_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no revision"):
                self.run_check(tmp, GYM_1_1_1, lambda: {"extension": "spire-python"})
            with self.assertRaisesRegex(ValueError, "no revision"):
                self.run_check(tmp, GYM_1_2_0, lambda: {"revision": None})

    def test_matching_revision_runs_the_check(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(check, "gym_check", lambda manifest, validate: {}):
            self.assertEqual(self.run_check(tmp, GYM_1_1_1, lambda: {"revision": GYM_1_1_1}), 0)
            self.assertEqual(self.run_check(tmp, GYM_1_2_0, lambda: {"revision": GYM_1_2_0}), 0)
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
