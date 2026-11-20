"""Extraction of Gym-validated fight states: synthetic runs and fake validators only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neural import run_states as ex
from spire_learn.neural import run_states as rs
from spire_learn.neural.standard_names import load_aliases

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests/fixtures/neural/runs"
NAMES = load_aliases(ROOT / "configs/neural/standard/name-aliases.json")
MAP = json.loads((ROOT / "configs/neural/standard/encounter-map.json").read_text())["encounters"]
ENCOUNTERS = json.loads((ROOT / "tests/fixtures/neural/gym-encounters.json").read_text())  # Gym 1.1.1 listing


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
    return ex.build(runs, NAMES, encounters, encounter_map, validate, gym_revision="rev")


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
        self.assertEqual((first["aliases_sha256"], first["gym_revision"]), (NAMES.sha256, "rev"))
        ids = [s["entry"]["id"] for s in first["states"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_report_counts_and_drop_rule_verdict(self):
        runs = _runs(1)
        document, report = _build(runs)
        ironclad = report["characters"]["ironclad"]
        kept = len(document["states"])
        self.assertEqual(ironclad["kept"], kept)
        self.assertEqual(ironclad["states"], ironclad["kept"] + ironclad["dropped"] + ironclad["scope_excluded"])
        self.assertEqual(ironclad["drop_rate"], 0.0 if not ironclad["dropped"] else ironclad["dropped"] / ironclad["states"])
        self.assertEqual(ironclad["drop_rule"], "pass")
        reject_floor_7 = lambda catalog: _reject(catalog, lambda e: e["id"].endswith("-7"))
        document, report = _build(runs, reject_floor_7)
        ironclad = report["characters"]["ironclad"]
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

    def test_unexpected_validator_error_propagates_only_as_rejection(self):
        def validate(catalog):
            raise RuntimeError("fault")
        document, report = _build(_runs(), validate)
        self.assertEqual(document["states"], [])
        self.assertEqual(report["characters"]["ironclad"]["drop_rule"], "fail")


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
            raise ValueError("boom")


if __name__ == "__main__":
    unittest.main()
