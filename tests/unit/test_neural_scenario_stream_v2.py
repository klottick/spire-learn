"""Curriculum v2 generator: encounter-first stages, prepared boss/elite decks, tier weights."""

from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from spire_learn.neural import scenario_stream as stream
from spire_learn.neural.scenario_support import json_value
from spire_learn.neural.stream_runner import build_development_panel

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "configs/neural/curriculum-v1/recipe.json"
V2 = ROOT / "configs/neural/curriculum-v2/recipe.json"

# Captured from the unmodified v1 generator (commit ba3c178) before any v2 edit.
V1_RECIPE_SHA256 = "3279664729328894fc995c997ccc54282f8d33483459e91d4da1b7ed43a92f66"
V1_TRAIN_SHARD_0 = "d644d2a113034641bc12cd06e50c7817cad23b5268c3c9014ef3f3f4406c7909"
V1_TRAIN_SHARD_37 = "8df900ac9a7ec8375c39628675ee799d8def93d445f0fe70bdc0cdcdf20af7a7"
V1_PANEL_MANIFEST = "435a236e60ac54f0b04f742b6d019023f49670c7730ba47545b73b23d2fb620e"

_STAGE_FOR_ACT = {1: "early", 2: "middle", 3: "developed", 4: "developed"}


class ScenarioStreamV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = stream.load_scenario_recipe(V1)
        cls.v2 = stream.load_scenario_recipe(V2)
        stream._SHARD_CACHE.clear()
        cls.mixed = [stream._generate_scenario_shard(cls.v2, split="train", shard_index=i, arm="mixed")
                     for i in range(4)]

    def setUp(self):
        stream._SHARD_CACHE.clear()
        self.addCleanup(stream._SHARD_CACHE.clear)

    def tier(self, encounter_id):
        encounter = self.v2.support.encounters[encounter_id]
        return encounter["act"], encounter["kind"]

    def test_v1_bytes_are_unchanged(self):
        self.assertEqual(self.v1.generator_version, "ironclad-curriculum-v1")
        self.assertEqual(self.v1.sha256, V1_RECIPE_SHA256)
        self.assertEqual(stream.generate_scenario_shard(self.v1, split="train", shard_index=0).manifest_sha256,
                         V1_TRAIN_SHARD_0)
        self.assertEqual(stream.generate_scenario_shard(self.v1, split="train", shard_index=37).manifest_sha256,
                         V1_TRAIN_SHARD_37)
        panel = build_development_panel(self.v1)
        self.assertEqual(panel["manifest_sha256"], V1_PANEL_MANIFEST)
        self.assertNotIn("prepared_profiles", self.v1.payload)
        self.assertNotIn("tier_weights", self.v1.payload)

    def test_v2_requires_and_v1_rejects_v2_tables(self):
        self.assertEqual(self.v2.generator_version, "ironclad-curriculum-v2")
        self.assertIn("prepared_profiles", self.v2.payload)
        self.assertIn("tier_weights", self.v2.payload)
        self.assertNotEqual(self.v1.sha256, self.v2.sha256)
        profiles = json_value(self.v2.prepared_profiles)
        weights = json_value(self.v2.tier_weights)
        for kwargs in ({"prepared_profiles": profiles}, {"tier_weights": weights}):
            with self.subTest(v1=kwargs.keys()), self.assertRaises(ValueError):
                replace(self.v1, **kwargs)
        for name in ("prepared_profiles", "tier_weights"):
            with self.subTest(v2_missing=name), self.assertRaises(ValueError):
                replace(self.v2, **{name: None})

        def bad_profile(act, kind, **changes):
            copy = json.loads(json.dumps(profiles))
            for key, value in changes.items():
                if value is None:
                    del copy[act][kind][key]
                else:
                    copy[act][kind][key] = value
            return copy

        bad_profiles = [
            {k: v for k, v in profiles.items() if k != "act3"},
            {**profiles, "act4": profiles["act3"]},
            {**profiles, "act1": {"boss": profiles["act1"]["boss"]}},
            bad_profile("act1", "boss", core_cards=[3, 2]),
            bad_profile("act1", "boss", core_cards=[0, 13]),
            bad_profile("act1", "boss", core_cards=[1]),
            bad_profile("act1", "boss", starter_removals=10),
            bad_profile("act2", "elite", upgrade_percent=101),
            bad_profile("act2", "elite", upgrade_percent=True),
            bad_profile("act3", "boss", hp_floor_percent=0),
            bad_profile("act3", "boss", hp_floor_percent=101),
            bad_profile("act3", "boss", rarity_weights={"starter": 0, "common": 1, "uncommon": 1}),
            bad_profile("act3", "boss", rarity_weights={"starter": 0, "common": -1, "uncommon": 1, "rare": 1}),
            bad_profile("act3", "boss", rarity_weights={"starter": 0, "common": 0, "uncommon": 0, "rare": 0}),
            bad_profile("act3", "elite", hp_floor_percent=None),
            bad_profile("act3", "elite", extra=1),
        ]
        for index, value in enumerate(bad_profiles):
            with self.subTest(bad_profile=index), self.assertRaises(ValueError):
                replace(self.v2, prepared_profiles=value)
        bad_weights = [
            {k: v for k, v in weights.items() if k != "act4:boss"},
            {**weights, "act5:boss": 1.0},
            {**weights, "act3:boss": 0.0},
            {**weights, "act3:boss": 1.5},
            {**weights, "act3:boss": math.nan},
            {**weights, "act3:boss": math.inf},
            {**weights, "act3:boss": True},
            {**weights, "act3:boss": "1"},
        ]
        for index, value in enumerate(bad_weights):
            with self.subTest(bad_weights=index), self.assertRaises(ValueError):
                replace(self.v2, tier_weights=value)

        support = (ROOT / "configs/neural/curriculum-v1/support.json").resolve()
        v1_raw = json.loads(V1.read_text())
        v2_raw = json.loads(V2.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            cases = [
                {**v1_raw, "support_file": str(support), "tier_weights": weights},
                {**v1_raw, "support_file": str(support), "prepared_profiles": profiles},
                {**{k: v for k, v in v2_raw.items() if k != "tier_weights"}, "support_file": str(support)},
                {**{k: v for k, v in v2_raw.items() if k != "prepared_profiles"}, "support_file": str(support)},
            ]
            for index, raw in enumerate(cases):
                path.write_text(json.dumps(raw))
                with self.subTest(load=index), self.assertRaises(ValueError):
                    stream.load_scenario_recipe(path)
            path.write_text(json.dumps({**v2_raw, "support_file": str(support)}))
            self.assertEqual(stream.load_scenario_recipe(path).sha256, self.v2.sha256)

    def test_v2_stage_follows_encounter_act(self):
        shards = self.mixed + [stream.generate_scenario_shard(self.v2, split=split, shard_index=0, arm=arm)
                               for split, arm in (("selection", "mixed"), ("train", "randomized"),
                                                  ("train", "fixed"))]
        for shard in shards:
            for case in shard.manifest["cases"]:
                act, _kind = self.tier(case["entry"]["encounter"])
                self.assertEqual(shard.annotations[case["case_id"]]["deck_stage"], _STAGE_FOR_ACT[act])

    def test_v2_prepared_decks_follow_profile(self):
        groups = {name: set(cards) for name, cards in self.v2.mechanic_groups.items() if name != "status"}
        prepared = 0
        for shard in self.mixed:
            for local, case in enumerate(shard.manifest["cases"]):
                entry = case["entry"]
                annotation = shard.annotations[case["case_id"]]
                self.v2.support.validate_entry(entry)
                ids = {row["card"] for row in entry["deck"]}
                self.assertFalse(any(set(pair) <= ids for pair in self.v2.reserved_pairs))
                self.assertLessEqual(len(entry["deck"]), 40)
                act, kind = self.tier(entry["encounter"])
                if kind == "monster" or annotation["family"] == "unusual":
                    continue
                prepared += 1
                profile = self.v2.prepared_profiles[f"act{min(act, 3)}"][kind]
                stage = self.v2.stage_profiles[_STAGE_FOR_ACT[act]]
                self.assertGreaterEqual(len(entry["deck"]), stage["minimum_size"])
                self.assertLessEqual(len(entry["deck"]), stage["maximum_size"])
                counts = Counter(row["card"] for row in entry["deck"])
                self.assertGreaterEqual(max(sum(counts[c] for c in group) for group in groups.values()),
                                        profile["core_cards"][0])
                self.assertGreaterEqual(entry["hp"] * 100, entry["max_hp"] * profile["hp_floor_percent"])
                self.assertTrue(set(annotation["target_card_ids"]) <= ids)
                # Strikes go first; any left come only from a lesson target or coverage card.
                strikes = max(0, 5 - profile["starter_removals"]) + list(annotation["target_card_ids"]).count("strike")
                if local < 38:
                    coverage = (shard.shard_index % 4) * 38 + local
                    row = {"card": sorted(self.v2.support.cards)[(coverage // 2) % 75],
                           "upgrades": coverage % 2}
                    self.assertIn(row, json_value(entry["deck"]))
                    strikes += row["card"] == "strike"
                self.assertLessEqual(counts["strike"], strikes)
        self.assertGreater(prepared, 100)

    def test_tier_weights_shift_the_noncoverage_draw(self):
        lowered = replace(self.v2, tier_weights={**json_value(self.v2.tier_weights), "act3:boss": 0.25})

        def draw(recipe):
            shards = [stream.generate_scenario_shard(recipe, split="train", shard_index=i) for i in range(8)]
            stream._SHARD_CACHE.clear()
            tiers = [shard.annotations[case["case_id"]]["encounter_tier"]
                     for shard in shards for case in shard.manifest["cases"][11:]]
            coverage = [case["entry"]["encounter"] for shard in shards for case in shard.manifest["cases"][:11]]
            return Counter(tiers)["act3:boss"] / len(tiers), coverage

        base_share, base_coverage = draw(self.v2)
        low_share, low_coverage = draw(lowered)
        self.assertLess(low_share, base_share * 0.6)
        self.assertEqual(low_coverage, base_coverage)

    def test_unit_tier_weights_keep_v1_kind_mix(self):
        def mix(recipe):
            kinds = Counter(a["encounter_tier"].split(":")[1]
                            for i in range(8)
                            for a in stream.generate_scenario_shard(recipe, split="train",
                                                                    shard_index=i).annotations.values())
            stream._SHARD_CACHE.clear()
            total = sum(kinds.values())
            return {kind: kinds[kind] / total for kind in ("monster", "elite", "boss")}

        self.assertEqual(set(json_value(self.v2.tier_weights).values()), {1.0})
        v1_mix, v2_mix = mix(self.v1), mix(self.v2)
        for kind in v1_mix:
            with self.subTest(kind=kind):
                self.assertLessEqual(abs(v1_mix[kind] - v2_mix[kind]), 0.02)

    def test_v2_shards_are_deterministic_and_unique(self):
        again = stream._generate_scenario_shard(self.v2, split="train", shard_index=0, arm="mixed")
        self.assertEqual(again.manifest_sha256, self.mixed[0].manifest_sha256)
        self.assertEqual(json_value(again.annotations), json_value(self.mixed[0].annotations))
        self.assertNotEqual(self.mixed[0].manifest_sha256,
                            stream.generate_scenario_shard(self.v1, split="train", shard_index=0).manifest_sha256)
        # Uniqueness is per shard, as in v1.
        for shard in self.mixed:
            within = [a["composition_sha256"] for a in shard.annotations.values()]
            self.assertEqual(len(within), len(set(within)))
        compositions = [a["composition_sha256"] for s in self.mixed for a in s.annotations.values()]
        self.assertEqual(len({s.manifest_sha256 for s in self.mixed}), 4)
        for shard in self.mixed:
            self.assertEqual(Counter(a["family"] for a in shard.annotations.values()),
                             {"run_like": 179, "lesson": 51, "unusual": 26})
        summary = stream.validate_coverage_cycle(self.mixed, self.v2)
        self.assertTrue(summary["broad_coverage"])
        selection = stream.generate_scenario_shard(self.v2, split="selection", shard_index=0)
        held_out = {a["composition_sha256"] for a in selection.annotations.values()}
        self.assertFalse(held_out & set(compositions))


if __name__ == "__main__":
    unittest.main()
