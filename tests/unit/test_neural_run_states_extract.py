"""Extraction of Gym-validated fight states: synthetic runs and fake validators only."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neural import run_states as ex
from spire_learn.neural import run_states as rs
from spire_learn.neural.entry_resources import load_entry_resources
from spire_learn.neural.standard_names import load_aliases

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests/fixtures/neural/runs"
NAMES = load_aliases(ROOT / "configs/neural/standard/name-aliases.json")
MAP = json.loads((ROOT / "configs/neural/standard/encounter-map.json").read_text())["encounters"]
ENCOUNTERS = json.loads((ROOT / "tests/fixtures/neural/gym-encounters.json").read_text())  # Gym 1.1.1 listing


REV = "e3c13836bd1458be8a87b93160f5c350f78fcd80"  # Gym 1.1.1
REV_1_2_0 = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"


def _base():
    return json.loads((RUNS / "ironclad-base.json").read_text())


def _runs(count=1):
    runs = []
    for index in range(count):
        run = _base()
        run["timestamp"] = 1000000001 + index
        runs.append(run)
    return runs


def _build(runs, validate=lambda catalog: None, encounter_map=MAP, encounters=ENCOUNTERS):
    return ex.build(runs, NAMES, encounters, encounter_map, validate, gym_revision=REV)


class ExtractTest(unittest.TestCase):
    def test_canonical_output_sha_stable(self):
        runs = _runs(3)
        first, _ = _build(runs)
        second, _ = _build(list(reversed(copy.deepcopy(runs))))
        self.assertEqual(ex.canonical_bytes(first), ex.canonical_bytes(second))
        self.assertEqual(hashlib.sha256(ex.canonical_bytes(first)).hexdigest(),
                         hashlib.sha256(ex.canonical_bytes(second)).hexdigest())
        keys = [(s["character"], s["run_id"], s["floor"]) for s in first["states"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(first["schema"], "spire-learn-run-states-v1")
        self.assertEqual((first["aliases_sha256"], first["gym_revision"]), (NAMES.sha256, REV))
        ids = [s["entry"]["id"] for s in first["states"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_report_counts_and_drop_rule_verdict(self):
        runs = _runs(1)
        document, report = _build(runs)
        ironclad = report["characters"]["ironclad"]
        kept = len(document["states"])
        self.assertEqual(ironclad["kept"], kept)
        self.assertEqual(ironclad["states"], ironclad["kept"] + ironclad["dropped"] + ironclad["scope_excluded"])
        self.assertEqual((ironclad["states"], kept, ironclad["dropped"]), (7, 7, 0))
        self.assertEqual((ironclad["drop_denominator"], ironclad["drop_rate"], ironclad["drop_rule"]), (7, 0.0, "pass"))
        reject_floor_7 = lambda catalog: _reject(catalog, lambda e: e["id"].endswith("-7"))
        document, report = _build(runs, reject_floor_7)
        ironclad = report["characters"]["ironclad"]
        self.assertEqual((ironclad["dropped"], ironclad["drop_denominator"], ironclad["drop_rate"]), (1, 7, round(1 / 7, 6)))
        self.assertEqual(ironclad["drop_rule"], "fail")
        self.assertEqual([r for r in ironclad["drop_reasons"] if r.startswith("gym_rejected")],
                         ["gym_rejected:boom"])
        self.assertEqual(ironclad["potions_assumed_empty"], 0)
        strata = {(s["character"], s["act"], s["kind"]): s for s in report["strata"]}
        self.assertEqual(strata[("ironclad", 1, "monster")]["dropped"], 1)
        self.assertEqual(strata[("ironclad", 1, "elite")]["dropped"], 0)
        self.assertEqual(sum(s["kept"] for s in report["strata"]), len(document["states"]))

    def test_scope_excluded_not_in_drop_denominator(self):
        run = _base()
        run["path_per_floor"][2] = "?"
        run["damage_taken"].append({"floor": 3, "enemies": "Mind Bloom Boss Battle", "damage": 0, "turns": 3})
        run["damage_taken"].sort(key=lambda row: row["floor"])
        _, report = _build([run])
        ironclad = report["characters"]["ironclad"]
        self.assertEqual(ironclad["scope_excluded"], 1)
        self.assertEqual(ironclad["scope_excluded_reasons"], {"scope_excluded:Mind Bloom Boss Battle": 1})
        self.assertEqual(ironclad["drop_denominator"], ironclad["kept"] + ironclad["dropped"])
        self.assertEqual(ironclad["drop_denominator"], ironclad["states"] - 1)
        strata = {(s["character"], s["act"], s["kind"]): s for s in report["strata"]}
        self.assertEqual(strata[("ironclad", 1, "event")]["scope_excluded"], 1)

    def test_encounter_row_act_and_kind_must_match(self):
        wrong = dict(MAP, **{"Jaw Worm": "champ", "Cultist": "lagavulin"})
        document, report = _build(_runs(), encounter_map=wrong)
        self.assertEqual(report["characters"]["ironclad"]["drop_reasons"]["encounter_mismatch:Jaw Worm"], 1)
        self.assertEqual(report["characters"]["ironclad"]["drop_reasons"]["encounter_mismatch:Cultist"], 1)
        self.assertNotIn("jaw_worm", {s["entry"]["encounter"] for s in document["states"]})
        _, report = _build(_runs(), encounter_map={k: v for k, v in MAP.items() if k != "Looter"})
        self.assertEqual(report["characters"]["ironclad"]["drop_reasons"], {"unmapped_encounter:Looter": 1})

    def test_burning_kind_deterministic(self):
        kinds = {ex.burning_kind("1000000001", floor) for floor in range(1, 60)}
        self.assertEqual(kinds, {"strength", "max_hp", "metallicize", "regeneration"})
        self.assertEqual(ex.burning_kind("1000000001", 10), ex.burning_kind("1000000001", 10))
        document, _ = _build(_runs())
        by_floor = {s["floor"]: s for s in document["states"]}
        burning = by_floor[10]
        self.assertEqual(burning["entry"]["burning"], ex.burning_kind(burning["run_id"], 10))
        self.assertIn("burning_kind_sampled", burning["tags"])
        self.assertNotIn("burning", by_floor[4]["entry"])
        self.assertNotIn("burning_kind_sampled", by_floor[4]["tags"])
        again, _ = _build(_runs())
        self.assertEqual(document, again)

    def test_entry_fields_and_potion_slots(self):
        document, _ = _build(_runs())
        entry = {s["floor"]: s for s in document["states"]}[10]["entry"]
        self.assertEqual((entry["character"], entry["ascension"], entry["encounter"]), ("ironclad", 20, "lagavulin"))
        self.assertEqual(entry["potions"], ["strength_potion", "fire_potion"])
        self.assertEqual(len(entry["potions"]), 2)
        for key in ("hp", "max_hp", "gold", "deck", "relics", "relic_counters"):
            self.assertIn(key, entry)
        self.assertNotIn("bottled_cards", entry)
        early = {s["floor"]: s for s in document["states"]}[1]["entry"]
        self.assertEqual(early["potions"], [None, None])
        run = _base()
        run["relics"].append("Potion Belt")
        run["relics_obtained"].append({"floor": 8, "key": "Potion Belt"})
        belt = {s["floor"]: s for s in _build([run])[0]["states"]}[10]["entry"]
        self.assertEqual(len(belt["potions"]), 4)
        state = rs.state_at(run, [f for f in rs.fight_floors(run) if f.floor == 10][0], NAMES)
        state = rs.FightState(**{**state.__dict__, "potions": ["fire_potion"] * 5})
        with self.assertRaises(ValueError):
            ex.to_standard_entry(state, "lagavulin")

    def test_one_rejected_entry_drops_only_itself(self):
        sizes = []

        def validate(catalog):
            sizes.append(len(catalog["entries"]))
            _reject(catalog, lambda e: e["id"] in bad)

        runs = _runs(60)  # 7 fights each: 420 entries, so two catalogs
        everything, _ = _build(runs)
        ids = [s["entry"]["id"] for s in everything["states"]]
        bad = {ids[3], ids[300]}
        document, report = _build(runs, validate)
        self.assertLessEqual(max(sizes), 256)
        self.assertGreater(len(sizes), 2)
        self.assertEqual({s["entry"]["id"] for s in everything["states"]} - {s["entry"]["id"] for s in document["states"]}, bad)
        self.assertEqual(report["characters"]["ironclad"]["drop_reasons"], {"gym_rejected:boom": 2})
        self.assertEqual(report["gym_rejections"], {"gym_rejected:boom": 2})

    def test_unexpected_validator_error_propagates_unretried(self):
        calls = []

        def validate(catalog):
            calls.append(len(catalog["entries"]))
            raise RuntimeError("native fault")

        with self.assertRaises(RuntimeError):
            _build(_runs(), validate)
        self.assertEqual(len(calls), 1)  # no bisection, no retry

    def test_event_fight_cannot_map_to_a_boss_row(self):
        run = _base()
        run["path_per_floor"][2] = "?"
        run["damage_taken"].append({"floor": 3, "enemies": "Cultist", "damage": 0, "turns": 3})
        run["damage_taken"].sort(key=lambda row: row["floor"])
        boss = dict(MAP, Cultist="slime_boss")
        _, report = _build([run], encounter_map=boss)
        reasons = report["characters"]["ironclad"]["drop_reasons"]
        self.assertEqual(reasons.get("encounter_mismatch:Cultist"), 2)  # floor 2 (monster) and floor 3 (event)
        _, report = _build([run])  # a monster row for the same event fight is not a mismatch
        self.assertNotIn("encounter_mismatch:Cultist", report["characters"]["ironclad"]["drop_reasons"])


class EntryResourcesExtractTest(unittest.TestCase):
    RESOURCES = load_entry_resources()

    def _with_pen_nib(self):
        run = _base()
        run["relics"].append("Pen Nib")
        run["relics_obtained"].append({"floor": 5, "key": "Pen Nib"})
        return run

    def _build_1_2_0(self, runs, resources=None):
        return ex.build(runs, NAMES, ENCOUNTERS, MAP, lambda catalog: None, gym_revision=REV_1_2_0,
                        entry_resources=self.RESOURCES if resources is None else resources)

    def test_extraction_tags_relic_counter_defaulted(self):
        run = self._with_pen_nib()
        for document, _ in (_build([run]), self._build_1_2_0([run])):
            by_floor = {s["floor"]: s for s in document["states"]}
            tagged = {f for f, s in by_floor.items() if "relic_counter_defaulted" in s["tags"]}
            self.assertTrue(tagged and all(f > 5 for f in tagged), tagged)
            self.assertTrue(all("pen_nib" in by_floor[f]["relics"] for f in tagged))
            self.assertNotIn("pen_nib", by_floor[max(tagged)]["entry"]["relic_counters"])

    def test_defaulted_relics_come_from_the_descriptor(self):
        run = self._with_pen_nib()
        counters = {r: rule for r, rule in self.RESOURCES.counters.items() if r != "pen_nib"}
        resources = dataclasses.replace(self.RESOURCES, counters=counters)
        # The pin is checked on the fingerprint: a descriptor with another one is refused outright.
        with self.assertRaisesRegex(ValueError, "pinned"):
            self._build_1_2_0([run], dataclasses.replace(resources, fingerprint="0" * 64))
        document, _ = self._build_1_2_0([run], resources)
        self.assertTrue(all("relic_counter_defaulted" not in s["tags"] for s in document["states"]))

    def test_table_records_the_descriptor_fingerprint_for_gym_1_2_0_only(self):
        document, _ = self._build_1_2_0(_runs(1))
        self.assertEqual(document["entry_resources_fingerprint"], self.RESOURCES.fingerprint)
        self.assertNotIn("entry_resources_fingerprint", _build(_runs(1))[0])

    def test_revision_and_resources_must_agree(self):
        with self.assertRaisesRegex(ValueError, "entry_resources"):
            ex.build(_runs(1), NAMES, ENCOUNTERS, MAP, lambda c: None, gym_revision=REV_1_2_0)
        with self.assertRaisesRegex(ValueError, "entry_resources"):
            ex.build(_runs(1), NAMES, ENCOUNTERS, MAP, lambda c: None, gym_revision=REV,
                     entry_resources=self.RESOURCES)
        with self.assertRaisesRegex(ValueError, "unsupported Gym revision"):
            ex.build(_runs(1), NAMES, ENCOUNTERS, MAP, lambda c: None, gym_revision="rev")

    def test_potion_slots_follow_the_descriptor(self):
        run = _base()
        run["relics"].append("Potion Belt")
        run["relics_obtained"].append({"floor": 8, "key": "Potion Belt"})
        document, _ = self._build_1_2_0([run])
        entry = {s["floor"]: s for s in document["states"]}[10]["entry"]
        self.assertEqual(len(entry["potions"]), 4)
        legacy, _ = _build([run])
        self.assertEqual({k: v for k, v in document.items() if k not in ("gym_revision", "entry_resources_fingerprint")},
                         {k: v for k, v in legacy.items() if k != "gym_revision"})

    def test_revision_comes_from_the_runtime_profile(self):
        self.assertEqual(ex.profile_revision(ROOT / "configs/neural/gym-1.1.1/runtime.json"), REV)
        self.assertEqual(ex.profile_revision(ROOT / "configs/neural/gym-1.2.0/runtime.json"), REV_1_2_0)
        self.assertEqual(ex.GYM_REVISION, REV)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps({"gym_revision": "0" * 40}))
            with self.assertRaises(ValueError):
                ex.profile_revision(path)

    def test_live_gym_must_match_the_profile_revision(self):
        live = lambda revision, extension="spire-python": (lambda: {"revision": revision, "extension": extension})  # noqa: E731
        ex.require_live_revision(REV_1_2_0, live(REV_1_2_0))
        with self.assertRaisesRegex(ValueError, "profile"):  # the v1.2.0 Gym with the default v1.1.1 profile
            ex.require_live_revision(REV, live(REV_1_2_0))
        with self.assertRaisesRegex(ValueError, "profile"):
            ex.require_live_revision(REV_1_2_0, live(REV))
        with self.assertRaisesRegex(ValueError, "profile"):
            ex.require_live_revision(REV, live(REV, "other"))


class CollectRunsTest(unittest.TestCase):
    def test_duplicates_kept_once_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a"), Path(tmp, "b")
            a.mkdir(), b.mkdir()
            data = (RUNS / "ironclad-base.json").read_bytes()
            (a / "1.json").write_bytes(data)
            (b / "1.json").write_bytes(data)
            self.assertEqual(len(ex.collect_runs([a, b])), 1)
            changed = json.loads(data)
            changed["floor_reached"] = 11
            (b / "1.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                ex.collect_runs([a, b])


def _reject(catalog, predicate):
    for entry in catalog["entries"]:
        if predicate(entry):
            raise ex.CatalogRejected("boom")


if __name__ == "__main__":
    unittest.main()
