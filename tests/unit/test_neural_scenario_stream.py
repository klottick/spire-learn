"""Pure preparation contracts: changing support or stream domains must fail here."""

from collections import Counter
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/neural/curriculum-v1"


class ScenarioStreamTests(unittest.TestCase):
    def setUp(self):
        # A missing implementation is an assertion failure in the initial RED run.
        try:
            self.support_module = importlib.import_module("spire_learn.neural.scenario_support")
            self.stream = importlib.import_module("spire_learn.neural.scenario_stream")
        except ModuleNotFoundError as exc:
            self.fail(f"scenario generation is not implemented: {exc}")
        self.recipe = self.stream.load_scenario_recipe(CONFIG / "recipe.json")

    def shard(self, **kwargs):
        return self.stream.generate_scenario_shard(
            kwargs.pop("recipe", self.recipe), split=kwargs.pop("split", "train"),
            shard_index=kwargs.pop("shard_index", 0), **kwargs)

    def plain(self, value):
        return self.support_module.json_value(value)

    def test_same_inputs_have_byte_identical_manifests_and_owned_metadata(self):
        first, second = self.shard(), self.shard()
        self.assertEqual(self.plain(first.manifest), self.plain(second.manifest))
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(len(first.manifest_sha256), 64)
        with self.assertRaises(TypeError):
            first.catalog["entries"][0]["hp"] = 1
        with self.assertRaises(TypeError):
            first.annotations[next(iter(first.annotations))]["family"] = "bad"

    def test_equivalent_recipe_mapping_order_cannot_change_same_identity_output(self):
        reordered = replace(self.recipe, mechanic_groups=dict(reversed(list(self.recipe.mechanic_groups.items()))))
        self.assertEqual(self.recipe.sha256, reordered.sha256)
        self.assertEqual(self.shard().manifest_sha256, self.shard(recipe=reordered).manifest_sha256)

    def test_shard_split_and_arm_have_separate_identities_and_seed_domains(self):
        shards = [self.shard(), self.shard(shard_index=1), self.shard(split="selection"),
                  self.shard(arm="randomized"), self.shard(arm="fixed")]
        self.assertEqual(len({s.manifest_sha256 for s in shards}), 5)
        self.assertEqual(len({s.distribution_sha256 for s in shards}), 4)
        self.assertLess(shards[0].manifest["cases"][0]["environment_seed"], 2**40)
        self.assertGreaterEqual(shards[2].manifest["cases"][0]["environment_seed"], 2**40)
        for shard in shards:
            for entry in shard.catalog["entries"]:
                self.assertEqual(set(entry), {"id", "encounter", "hp", "max_hp", "deck",
                                              "relics", "potions", "relic_counters"})

    def test_full_mixed_shard_has_179_51_26_and_lesson_categories(self):
        shard = self.shard()
        self.assertEqual(Counter(a["family"] for a in shard.annotations.values()),
                         {"run_like": 179, "lesson": 51, "unusual": 26})
        self.assertEqual(Counter(a["lesson_kind"] for a in shard.annotations.values()
                                 if a["family"] == "lesson"),
                         {"single": 26, "pair": 15, "group": 10})
        for annotation in shard.annotations.values():
            if annotation["family"] == "lesson":
                self.assertIn(annotation["category"], self.recipe.category_keys)
                self.assertTrue(annotation["target_card_ids"])

    def test_four_shards_cover_all_base_upgraded_cards_and_encounters(self):
        for arm in ("mixed", "randomized"):
            with self.subTest(arm=arm):
                shards = [self.shard(shard_index=i, arm=arm) for i in range(4)]
                entries = [e for s in shards for e in s.catalog["entries"]]
                cards = {(c["card"], c["upgrades"]) for e in entries for c in e["deck"]}
                self.assertEqual(len({c for c, _ in cards}), 75)
                self.assertTrue({(c, u) for c in self.recipe.support.cards for u in (0, 1)} <= cards)
                self.assertEqual(len({e["encounter"] for e in entries}), 43)
                summary = self.stream.validate_coverage_cycle(shards, self.recipe)
                self.assertTrue(summary["broad_coverage"])
                self.assertEqual(summary["exposure"], "generated_only")
                self.assertEqual({a["deck_stage"] for s in shards for a in s.annotations.values()},
                                 {"early", "middle", "developed"})
                self.assertEqual({self.recipe.support.cards[c]["rarity"] for c, _ in cards},
                                 {"starter", "common", "uncommon", "rare"})

    def test_fixed_is_narrow_control_and_randomized_has_no_lessons(self):
        fixed = [self.shard(shard_index=i, arm="fixed") for i in range(4)]
        summary = self.stream.validate_coverage_cycle(fixed, self.recipe)
        self.assertFalse(summary["broad_coverage"])
        self.assertEqual(summary["card_count"], 7)
        self.assertEqual(summary["encounter_count"], 13)
        self.assertEqual({a["family"] for a in self.shard(arm="randomized").annotations.values()},
                         {"run_like"})

    def test_fixed_control_preserves_old_78_entry_composition_resource_distribution(self):
        old = json.loads((ROOT / "configs/cases/neural/sustained-v1/train.json").read_text())
        old_entries = [c["entry"] for c in old["cases"] if c["mode"] == "standalone"]

        def public_composition(entry):
            return (entry["encounter"], entry["hp"], entry["max_hp"],
                    tuple(sorted((c["card"], c["upgrades"]) for c in entry["deck"])),
                    tuple(entry["relics"]), tuple(entry["potions"]))

        fixed = self.shard(arm="fixed")
        self.assertEqual(len(fixed.catalog["entries"]), 78)
        self.assertEqual({e["hp"] for e in fixed.catalog["entries"]}, {64, 80})
        self.assertEqual(Counter(public_composition(e) for e in fixed.catalog["entries"]),
                         Counter(public_composition(e) for e in old_entries))

    def test_canonical_manifest_roundtrip_uses_existing_parser(self):
        from spire_learn.neural.cases import load_neural_cases
        shard = self.shard()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text(json.dumps(self.plain(shard.manifest)))
            loaded = load_neural_cases(path)
        self.assertEqual(len(loaded.cases), 256)
        self.assertEqual(loaded.sha256, shard.manifest_sha256)
        self.assertEqual(set(shard.annotations), {c.case_id for c in loaded.cases})

    def test_small_shards_carry_family_fraction_across_cycle(self):
        recipe = replace(self.recipe, shard_size=64)
        shards = [self.shard(recipe=recipe, shard_index=i) for i in range(4)]
        self.assertEqual(Counter(a["family"] for s in shards for a in s.annotations.values()),
                         {"run_like": 179, "lesson": 51, "unusual": 26})
        self.assertTrue(self.stream.validate_coverage_cycle(shards, recipe)["broad_coverage"])

    def test_invalid_recipes_fail_before_generation(self):
        invalid = ({"shard_size": True}, {"shard_size": 0}, {"shard_size": 257},
                   {"generator_seed": True}, {"generator_seed": -1},
                   {"maximum_catalog_bytes": float("nan")},
                   {"reserved_pairs": (("strike", "anger"),)},
                   {"reserved_pairs": (("wound", "anger"),)})
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.recipe, **fields)
        raw = json.loads((CONFIG / "recipe.json").read_text())
        for key, value in (("unknown", 2), ("generator_seed", True), ("generator_seed", float("nan"))):
            changed = dict(raw, **{key: value})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "recipe.json"
                path.write_text(json.dumps(changed))
                (path.parent / "support.json").write_bytes((CONFIG / "support.json").read_bytes())
                with self.assertRaises(ValueError):
                    self.stream.load_scenario_recipe(path)

    def test_supported_entry_validation_rejects_invalid_public_resources(self):
        entry = self.plain(self.shard().catalog["entries"][0])
        changes = ({"hp": True}, {"hp": float("nan")}, {"secret": 0},
                   {"deck": [{"card": "wound", "upgrades": 0}]},
                   {"deck": [{"card": "anger", "upgrades": 2}]},
                   {"deck": [{"card": "anger", "upgrades": True}]},
                   {"potions": ["unknown", None, None]},
                   {"relics": ["burning_blood", "black_blood"]},
                   {"relic_counters": {"other": 0}},
                   {"relics": ["neows_lament"], "relic_counters": {}},
                   {"relics": ["neows_lament"], "relic_counters": {"neows_lament": 1}},
                   {"relics": ["neows_lament"], "relic_counters": {"neows_lament": True}})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.recipe.support.validate_entry(dict(entry, **change))
        valid = dict(entry, relics=["neows_lament"], relic_counters={"neows_lament": -2})
        self.recipe.support.validate_entry(valid)

    def test_support_snapshot_inventory_and_unknown_fields(self):
        support = self.recipe.support
        self.assertEqual((len(support.cards), len(support.encounters), len(support.relics),
                          len(support.potions)), (75, 43, 30, 13))
        self.assertEqual(support.cards["searing_blow"]["maximum_upgrades"], 10)
        raw = json.loads((CONFIG / "support.json").read_text())
        for mutate in (lambda r: r.update(unknown=True),
                       lambda r: r["cards"][0].update(maximum_upgrades=True),
                       lambda r: r["cards"].append(dict(r["cards"][0]))):
            copy = json.loads(json.dumps(raw))
            mutate(copy)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "support.json"
                path.write_text(json.dumps(copy))
                with self.assertRaises(ValueError):
                    self.support_module.load_scenario_support(path)

    def test_oversized_catalog_reduces_deterministically_or_rejects(self):
        small = replace(self.recipe, maximum_catalog_bytes=90_000)
        first = self.shard(recipe=small)
        self.assertLess(len(first.catalog["entries"]), 256)
        encoded = json.dumps(self.plain(first.catalog), separators=(",", ":"), sort_keys=True).encode()
        self.assertLessEqual(len(encoded), 90_000)
        self.assertEqual(first.manifest_sha256, self.shard(recipe=small).manifest_sha256)
        with self.assertRaisesRegex(ValueError, "byte|fit"):
            self.shard(recipe=replace(self.recipe, maximum_catalog_bytes=10))

    def test_bounded_rejection_and_incomplete_coverage_are_errors(self):
        # A hash partition that rejects every composition must terminate at the bound.
        from unittest.mock import patch
        with patch.object(self.stream, "_composition_split", return_value="selection") as partition:
            with self.assertRaisesRegex(ValueError, "4096"):
                self.shard(recipe=replace(self.recipe, reserved_pairs=()))
            self.assertEqual(partition.call_count, 4096)
        with self.assertRaisesRegex(ValueError, "four|4"):
            self.stream.validate_coverage_cycle([self.shard()], self.recipe)
        with self.assertRaises(ValueError):
            self.stream.validate_coverage_cycle([self.shard(shard_index=i) for i in (0, 1, 2, 4)], self.recipe)

    def test_reserved_pairs_excluded_and_compositions_disjoint_across_splits(self):
        shards = [self.shard(split=split) for split in ("train", "selection", "final")]
        compositions = []
        for shard in shards:
            compositions.append({tuple(sorted((c["card"], c["upgrades"]) for c in e["deck"]))
                                 for e in shard.catalog["entries"]})
        for i in range(3):
            for j in range(i):
                self.assertFalse(compositions[i] & compositions[j])
        for entry in shards[0].catalog["entries"]:
            ids = {c["card"] for c in entry["deck"]}
            for pair in self.recipe.reserved_pairs:
                self.assertFalse(set(pair) <= ids)
        # Force the first training deck into a selection attempt: the actual
        # composition partition must reject it, even with a new environment seed.
        from unittest.mock import patch
        training_first = tuple(sorted((c["card"], c["upgrades"])
                                     for c in shards[0].catalog["entries"][0]["deck"]))
        rejected = []
        original_partition = self.stream._composition_split
        original_rng = self.stream._rng

        def same_first_training_deck(recipe, split, arm, index, purpose):
            return original_rng(recipe, "train" if purpose == "deck:0" else split,
                                arm, index, purpose)

        def observe_partition(deck):
            composition = tuple(sorted((c["card"], c["upgrades"]) for c in deck))
            if composition == training_first:
                rejected.append(composition)
            return original_partition(deck)

        with patch.object(self.stream, "_rng", side_effect=same_first_training_deck), \
             patch.object(self.stream, "_composition_split", side_effect=observe_partition):
            selection = self.shard(split="selection")
        self.assertTrue(rejected)
        self.assertNotIn(training_first, {
            tuple(sorted((c["card"], c["upgrades"]) for c in e["deck"]))
            for e in selection.catalog["entries"]})

    def test_coverage_rejects_wrong_recipe_and_modified_annotations(self):
        shards = [self.shard(shard_index=i) for i in range(4)]
        with self.assertRaises(ValueError):
            self.stream.validate_coverage_cycle(shards, replace(self.recipe, generator_seed=3))
        annotations = self.plain(shards[0].annotations)
        annotations[next(iter(annotations))]["family"] = "lesson"
        changed = replace(shards[0], annotations=annotations)
        with self.assertRaises(ValueError):
            self.stream.validate_coverage_cycle([changed, *shards[1:]], self.recipe)

    def test_generated_resources_and_deck_bounds_remain_public_and_bounded(self):
        for arm in ("mixed", "randomized"):
            shard = self.shard(arm=arm)
            for case in shard.manifest["cases"]:
                entry, annotation = case["entry"], shard.annotations[case["case_id"]]
                self.assertLessEqual(80, entry["max_hp"])
                self.assertLessEqual(entry["max_hp"], 120)
                self.assertLessEqual(24, entry["hp"])
                self.assertLessEqual(entry["hp"], entry["max_hp"])
                lower, upper = (5, 40) if annotation["family"] == "unusual" else (8, 35)
                self.assertLessEqual(lower, len(entry["deck"]))
                self.assertLessEqual(len(entry["deck"]), upper)
                self.assertTrue(all(c["upgrades"] in (0, 1) for c in entry["deck"]))
                self.assertNotIn("neows_lament", entry["relics"])
                self.recipe.support.validate_entry(entry)

    def test_catalog_byte_reduction_carries_shared_cycle_quotas(self):
        recipe = replace(self.recipe, maximum_catalog_bytes=90_000)
        shards = [self.shard(recipe=recipe, shard_index=i) for i in range(4)]
        sizes = [len(s.catalog["entries"]) for s in shards]
        # A changing reduced size would reset the fractional family accumulator.
        self.assertEqual(len(set(sizes)), 1)
        run_count = sum(a["family"] == "run_like" for s in shards for a in s.annotations.values())
        self.assertEqual(run_count, (sum(sizes) * 179) // 256)
        self.assertTrue(self.stream.validate_coverage_cycle(shards, recipe)["broad_coverage"])

    def test_invalid_generator_arguments_and_infeasible_combinations(self):
        for values in ({"shard_index": True}, {"shard_index": -1}, {"split": "test"},
                       {"arm": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.shard(**values)
        with self.assertRaises(ValueError):
            replace(self.recipe, reserved_pairs=(("inflame", "heavy_blade"),))
        with self.assertRaises(ValueError):
            replace(self.recipe, reserved_pairs=(("dark_embrace", "feel_no_pain"),))
        tiny = replace(self.recipe, shard_size=1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.stream.validate_coverage_cycle([self.shard(recipe=tiny, shard_index=i)
                                                 for i in range(4)], tiny)


if __name__ == "__main__":
    unittest.main()
