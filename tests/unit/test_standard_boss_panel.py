import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from collections import Counter
from unittest.mock import patch

from scripts.neural import baseline_entry
from spire_learn.neural import recipes, standard_stream as ss
from spire_learn.neural.standard_catalog import CatalogRejected
from tests.unit.test_baseline_entry import PLAN, ROOT, _Clock, _Fakes, _prior_output
from tests.unit.test_neural_standard_stream import (
    NO_FUZZ, SHIPPED_1_2_0, _Fixture, recipe_document, synthetic_states, write_table,
)


class BossPanelTests(_Fixture):
    def test_boss_panel_quotas(self):
        panel = ss.build_boss_panel(self.recipe, validate=NO_FUZZ)
        self.assertEqual(len(panel["manifest"]["cases"]), 256)
        sizes = Counter(ss.stratum(state) for state in self.recipe.states_for("selection"))
        for level in (0, 1, 2, 4):
            rows = [row for row in panel["annotations"].values() if row["level"] == level]
            self.assertEqual(len(rows), 64)
            self.assertEqual(len({(row["run_id"], row["floor"]) for row in rows}), 64)
            self.assertTrue(all(ss.run_split(row["run_id"]) == "selection" for row in rows))
            self.assertEqual(Counter(row["room_kind"] for row in rows), {"boss": 40, "elite": 24})
            strata = Counter(ss.stratum(row) for row in rows)
            for kind, total in (("boss", 40), ("elite", 24)):
                expected = ss.allocate_quotas(total, {key: size for key, size in sizes.items()
                                                     if key.endswith(f":{kind}")})
                self.assertLessEqual(max(abs(strata[key] - count) for key, count in expected.items()), 1)

    def test_boss_panel_is_deterministic(self):
        first = ss.build_boss_panel(self.recipe, validate=NO_FUZZ)
        second = ss.build_boss_panel(self.load(), validate=NO_FUZZ)
        self.assertEqual(first, second)
        self.assertEqual(first["panel_sha256"], ss.panel_sha256(self.recipe, kind="boss"))
        self.assertNotEqual(first["panel_sha256"], ss.panel_sha256(self.recipe))

    def test_boss_panel_independent_of_training_only_fields(self):
        first = ss.build_boss_panel(self.recipe, validate=NO_FUZZ)
        recipe = self.load(fuzz_schedule=[{"from_shard": 0, "levels": [0, 1, 2, 4]}],
                           shard_size=128, generator_seed=37)
        self.assertNotEqual(recipe.sha256, self.recipe.sha256)
        second = ss.build_boss_panel(recipe, validate=NO_FUZZ)
        self.assertEqual(first["panel_identity"], second["panel_identity"])
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_boss_shortfall_stays_within_room_kind(self):
        def reject(document):
            if any(entry["encounter"] == "book_of_stabbing" for entry in document["entries"]):
                raise CatalogRejected("rejected synthetic act 2 elite")

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            _, sha = write_table(directory, synthetic_states(runs=700))
            path = directory / "recipe.json"
            path.write_text(json.dumps(recipe_document(sha)))
            panel = ss.build_boss_panel(recipes.load_recipe(path), validate=reject)
        for level in (1, 2, 4):
            rows = [row for row in panel["annotations"].values() if row["level"] == level]
            self.assertEqual(Counter(row["room_kind"] for row in rows), {"boss": 40, "elite": 24})
            self.assertFalse(any(row["room_kind"] == "elite" and row["act"] == 2 for row in rows))
            self.assertGreater(panel["skipped"][str(level)], 0)

    def test_reevaluate_boss_panel_scores_on_boss_cases(self):
        panel = ss.build_boss_panel(self.recipe, validate=NO_FUZZ)
        case_ids = {case["case_id"] for case in panel["manifest"]["cases"]}
        fakes = _Fakes()
        deps = {**fakes.deps(), "load_recipe": lambda path: self.recipe,
                "build_panel": recipes.build_panel,
                "load_config": lambda path: SimpleNamespace(
                    **{**vars(fakes.load_config(path)), "environment_revision": self.recipe.gym_revision})}
        with tempfile.TemporaryDirectory() as folder:
            # The fake plan needs the synthetic recipe's Gym pin for the panel preflight.
            plan = json.loads((ROOT / PLAN).read_text())
            plan["gym_revision"] = self.recipe.gym_revision
            plan_path = Path(folder) / "plan.json"
            plan_path.write_text(json.dumps(plan))
            prior, previous = _prior_output(folder, plan=plan)
            output = Path(folder) / "boss"
            output.mkdir()
            with patch.object(recipes, "gym_validator", return_value=NO_FUZZ):
                baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=deps,
                                            clock=_Clock(), reevaluate=prior, panel="boss")
            summary = json.loads((output / "summary.json").read_text())
            events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            case_config = tomllib.loads((output / "panel" / "config-0.toml").read_text())
            panel_path = output / "boss-panel.json"
            written_panel = json.loads(panel_path.read_text()) if panel_path.is_file() else {}
        with self.subTest(output="deadline"):
            self.assertEqual(summary.get("boss_case_seconds"), plan["wall_seconds"]["total"])
            self.assertEqual(summary["boss_case_seconds"], case_config["budgets"]["max_wall_seconds"])
            self.assertEqual(summary["plan"]["evaluation"]["case_seconds"], plan["evaluation"]["case_seconds"])
        with self.subTest(output="journal"):
            completed = [row for row in events if row["event"] == "evaluation_complete"]
            self.assertEqual(len(completed), len(previous["checkpoints"]))
            self.assertTrue(all(row.get("panel") == "boss" for row in events))
        with self.subTest(output="panel file"):
            self.assertEqual(written_panel.get("manifest_sha256"), panel["manifest_sha256"])
        self.assertEqual(summary["result"], "complete")
        self.assertEqual(summary["panel"], "boss")
        self.assertEqual(summary["manifest_sha256"], panel["manifest_sha256"])
        self.assertEqual(summary["checkpoints"], previous["checkpoints"])
        self.assertEqual(len(summary["evaluations"]), len(previous["checkpoints"]))
        for evaluation in summary["evaluations"].values():
            self.assertEqual(set(evaluation["cases"]), case_ids)
        self.assertEqual(fakes.segments, [])
        self.assertFalse(any(row[0] == "save" for row in fakes.log))


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "requires the pinned Gym runtime and state table")
class PinnedSelectionPanelTests(unittest.TestCase):
    def test_selection_panel_identity_unchanged(self):
        from spire_lab import _native

        recipe = recipes.load_recipe(SHIPPED_1_2_0)
        self.assertEqual(_native.build_info()["revision"], recipe.gym_revision)
        selection = recipes.build_panel(recipe, "selection")
        self.assertEqual(selection["panel_sha256"],
                         "2247fe2be2b32b66a746483b974c514301de19b0bb88defaca9f2c785094c041")
        self.assertEqual(selection["manifest_sha256"],
                         "eb4a50d82eaf2289a318a80ab1d730f11f736edef0893c70cc8044b813fac0b2")
        boss = recipes.build_panel(recipe, "selection", kind="boss")

        def bases(panel):
            return {(row["run_id"], row["floor"]) for row in panel["annotations"].values()}

        self.assertEqual(len(boss["manifest"]["cases"]), 256)
        for level in (0, 1, 2, 4):
            rows = [row for row in boss["annotations"].values() if row["level"] == level]
            self.assertEqual(Counter(row["room_kind"] for row in rows), {"boss": 40, "elite": 24})
            self.assertEqual(len({(row["run_id"], row["floor"]) for row in rows}), 64)
        print(json.dumps({"boss_panel_sha256": boss["panel_sha256"],
                          "boss_manifest_sha256": boss["manifest_sha256"],
                          "shared_base_states": len(bases(selection) & bases(boss))}, sort_keys=True))
