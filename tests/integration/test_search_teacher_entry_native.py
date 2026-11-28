"""Opt-in fight-entry smoke on the pinned Standard runtime."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.neural import search_teacher_entry as entry


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "requires the pinned Gym native runtime")
class TeacherEntryNativeTests(unittest.TestCase):
    def test_current_and_uncalibrated_tree_on_identical_training_fights(self):
        root = entry._ROOT
        recipe = root / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
        defaults = root / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
        parent = Path(os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT", root / "artifacts"))
        with tempfile.TemporaryDirectory(prefix="teacher-entry-", dir=parent) as temp:
            output = Path(temp)
            configs = json.loads(defaults.read_text())
            configs["teachers"]["tree-1x"]["config"].update(worlds=4)
            teachers = output / "teachers.json"
            teachers.write_text(json.dumps(configs))
            results = {}
            for name in ("current", "tree-1x"):
                for parallel in (1, 2):
                    with self.subTest(teacher=name, parallel=parallel):
                        destination = output / f"{name}-{parallel}"
                        self.assertEqual(entry.main([
                            "--recipe", str(recipe), "--teachers", str(teachers), "--teacher", name,
                            "--train-sample", "4", "--seed", "2026112500", "--output", str(destination),
                            "--workers", "1", "--parallel", str(parallel),
                        ]), 0)
                        summary = json.loads((destination / "summary.json").read_text())
                        fights = self.read_rows(destination / "fights.jsonl")
                        decisions = self.read_rows(destination / "decisions.jsonl")
                        self.assertEqual(len(fights), 4)
                        self.assertEqual(sum(fight["steps"] for fight in fights), len(decisions))
                        self.assertEqual(sum(summary["roots"].values()), len(decisions))
                        self.assertGreater(summary["searched_decisions"], 0)
                        self.assertGreater(summary["cpu_seconds_per_searched_decision"], 0)
                        for decision in decisions:
                            searched = decision["root"] == "searched"
                            self.assertEqual(searched, decision["values"] is not None)
                            self.assertEqual(searched, decision["cpu_seconds"] is not None)
                            if searched:
                                self.assertIn(str(decision["choice"]), decision["values"])
                        for row in fights + decisions:
                            row.pop("cpu_seconds")
                            row.pop("wall_seconds", None)
                        if parallel == 1:
                            results[name] = (fights, decisions, summary["provenance"])
                        else:
                            self.assertEqual((fights, decisions), results[name][:2])

            current, tree = (results[name][2] for name in ("current", "tree-1x"))
            self.assertEqual(current["case_manifest_sha256"], tree["case_manifest_sha256"])
            self.assertEqual(current["environment_seeds"], tree["environment_seeds"])
            self.assertFalse(tree["calibrated"])
            self.assertNotEqual(current["config_sha256"], tree["config_sha256"])

    @staticmethod
    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
