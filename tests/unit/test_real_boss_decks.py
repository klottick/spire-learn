"""Real boss decks: the extraction script (on hand-authored synthetic runs) and the v2 generator table."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import urllib.error

from scripts.neural import real_boss_decks as rbd
from spire_learn.neural import scenario_stream as stream
from spire_learn.neural.scenario_support import (
    canonical_bytes, content_sha256, json_value, load_scenario_support,
)
from tests.unit.test_neural_scenario_stream_v2 import V2_RECIPE_SHA256, V2_TRAIN_SHARD_0

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests/fixtures/neural/real-boss-runs"
SUPPORT_PATH = ROOT / "configs/neural/curriculum-v1/support.json"
V2 = ROOT / "configs/neural/curriculum-v2/recipe.json"
REAL_ENCOUNTERS = ("champ", "collector", "bronze_automaton", "awakened_one", "time_eater", "donu_and_deca")


def _run(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def _rows(deck):
    return sorted((row["card"], row["upgrades"]) for row in deck)


class RebuildTests(unittest.TestCase):
    def test_split_card_reads_upgrade_counts(self):
        self.assertEqual(rbd.split_card("Searing Blow+3"), ("Searing Blow", 3))
        self.assertEqual(rbd.split_card("Bash+1"), ("Bash", 1))
        self.assertEqual(rbd.split_card("Strike_R"), ("Strike_R", 0))

    def test_deck_at_act2_boss_undoes_later_picks_purchases_purges_smiths_and_events(self):
        deck, misses = rbd.deck_at(_run("101"), 33)
        self.assertEqual(misses, 0)
        self.assertEqual(sorted(deck), sorted([
            "Strike_R", "Strike_R", "Strike_R", "Defend_R", "Defend_R+1", "Bash", "Ghostly",
            "Inflame", "Searing Blow+2", "AscendersBane", "Shrug It Off"]))

    def test_deck_at_act3_boss_restores_only_the_later_purge(self):
        deck, misses = rbd.deck_at(_run("101"), 50)
        self.assertEqual(misses, 0)
        self.assertEqual(sorted(deck), sorted([
            "Strike_R", "Strike_R", "Defend_R+1", "Bash+1", "Ghostly", "Inflame+1", "Searing Blow+3",
            "AscendersBane", "Shrug It Off+1", "Offering", "Feel No Pain", "Apparition", "Dual Wield+1"]))

    def test_relics_at_excludes_relics_obtained_on_or_after_the_floor(self):
        run = _run("101")
        self.assertEqual(sorted(rbd.relics_at(run, 33)), ["Anchor", "NeowsBlessing", "Pen Nib", "Runic Cube"])
        self.assertEqual(sorted(rbd.relics_at(run, 50)), sorted(run["relics"]))

    def test_name_mapping_includes_starter_and_ghostly_aliases(self):
        support = load_scenario_support(SUPPORT_PATH)
        cards, relics = rbd.name_maps(support)
        self.assertEqual(cards[rbd.normalize("Strike_R")], "strike")
        self.assertEqual(cards[rbd.normalize("Defend_R")], "defend")
        self.assertEqual(cards[rbd.normalize("Ghostly")], "ghostly_armor")
        self.assertEqual(cards[rbd.normalize("Shrug It Off")], "shrug_it_off")
        self.assertNotIn(rbd.normalize("AscendersBane"), cards)
        self.assertEqual(relics[rbd.normalize("Paper Frog")], "paper_phrog")
        self.assertEqual(relics[rbd.normalize("Burning Blood")], "burning_blood")
        self.assertNotIn(rbd.normalize("NeowsBlessing"), relics)


class ExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = load_scenario_support(SUPPORT_PATH)
        cls.document, cls.stats = rbd.extract(RUNS, cls.support)

    def deck(self, run_id, floor):
        (row,) = [d for d in self.document["decks"] if d["run_id"] == run_id and d["floor"] == floor]
        return row

    def test_document_rows_and_won_lost_detection(self):
        self.assertEqual(self.document["schema"], "spire-learn-real-boss-decks-v1")
        self.assertEqual(self.document["source"], "baalorlord.tv profile 0 (Slay the Spire)")
        keys = [(d["run_id"], d["floor"], d["encounter"], d["won"]) for d in self.document["decks"]]
        self.assertEqual(sorted(keys), sorted([
            ("101", 33, "champ", True), ("101", 50, "time_eater", True),
            ("101", 51, "awakened_one", False), ("102", 33, "collector", False),
            ("105", 50, "time_eater", True), ("105", 51, "donu_and_deca", True)]))
        self.assertEqual(self.stats["encounters"], {
            "awakened_one": [0, 1], "champ": [1, 1], "collector": [0, 1],
            "donu_and_deca": [1, 1], "time_eater": [2, 2]})

    def test_mapped_decks_relics_and_hp_fractions(self):
        champ = self.deck("101", 33)
        self.assertEqual(_rows(champ["deck"]), sorted([
            ("strike", 0), ("strike", 0), ("strike", 0), ("defend", 0), ("defend", 1), ("bash", 0),
            ("ghostly_armor", 0), ("inflame", 0), ("searing_blow", 2), ("shrug_it_off", 0)]))
        self.assertEqual(champ["dropped_cards"], 1)
        self.assertEqual(champ["relics"], ["anchor", "runic_cube"])
        self.assertEqual(champ["hp_fraction"], 0.5)
        self.assertEqual(champ["ascension"], 20)
        time_eater = self.deck("101", 50)
        self.assertEqual(_rows(time_eater["deck"]), sorted([
            ("strike", 0), ("strike", 0), ("defend", 1), ("bash", 1), ("ghostly_armor", 0),
            ("inflame", 1), ("searing_blow", 3), ("shrug_it_off", 1), ("offering", 0),
            ("feel_no_pain", 0), ("dual_wield", 1)]))
        self.assertEqual(time_eater["dropped_cards"], 2)
        self.assertEqual(time_eater["relics"],
                         ["anchor", "black_blood", "orichalcum", "paper_phrog", "runic_cube", "vajra"])
        self.assertEqual(time_eater["hp_fraction"], 0.75)
        # The second act 3 boss uses the deck at floor 50 and the HP entering floor 51.
        awakened = self.deck("101", 51)
        self.assertEqual(awakened["deck"], time_eater["deck"])
        self.assertEqual(awakened["hp_fraction"], 0.25)
        self.assertEqual(self.deck("102", 33)["hp_fraction"], 0.4)
        self.assertEqual(self.deck("102", 33)["relics"], ["burning_blood", "lantern"])
        self.assertEqual(self.deck("105", 51)["relics"], ["burning_blood"])

    def test_exclusions_and_unmatched_names_are_counted(self):
        self.assertEqual(self.stats["unmatched_boss_names"], {"33:Mystery Boss": 1})
        self.assertEqual(self.stats["excluded_invalid"], 1)          # 104: Bash+2
        self.assertEqual(self.stats["excluded_untracked_boss_relic"], 1)  # 105: Astrolabe at floor 34
        self.assertEqual(self.stats["skipped_other_character"], 1)   # 103
        self.assertEqual(self.stats["unmapped_cards"], {"AscendersBane": 3, "Apparition": 2})
        self.assertIn("Pen Nib", self.stats["unmapped_relics"])
        self.assertEqual(self.stats["rebuild_misses"], 0)

    def test_integral_float_numbers_are_accepted(self):
        # Some run files store every number as a float (floor 1.0, HP 30.0).
        def floats(value):
            if isinstance(value, dict):
                return {k: floats(v) for k, v in value.items()}
            if isinstance(value, list):
                return [floats(v) for v in value]
            return float(value) if type(value) is int else value

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("101", "102"):
                Path(tmp, f"{name}.json").write_text(json.dumps(floats(_run(name))))
            document, stats = rbd.extract(Path(tmp), self.support)
        self.assertEqual(stats["malformed_runs"], 0)
        expected = [d for d in self.document["decks"] if d["run_id"] in ("101", "102")]
        self.assertEqual(document["decks"], expected)
        with self.assertRaises(ValueError):
            rbd.as_int(1.5)

    def test_output_is_canonical_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp, "a.json"), Path(tmp, "b.json")
            sha_first = rbd.write_document(self.document, first)
            again, _ = rbd.extract(RUNS, self.support)
            sha_second = rbd.write_document(again, second)
            data = first.read_bytes()
            self.assertEqual(sha_first, sha_second)
            self.assertEqual(data, second.read_bytes())
            self.assertEqual(sha_first, hashlib.sha256(data).hexdigest())
            self.assertEqual(data, canonical_bytes(json.loads(data)))
            self.assertFalse(data.endswith(b"\n"))
        order = [(d["encounter"], d["run_id"], d["floor"]) for d in self.document["decks"]]
        self.assertEqual(order, sorted(order))
        for deck in self.document["decks"]:
            self.assertEqual(deck["deck"], sorted(deck["deck"], key=lambda r: (r["card"], r["upgrades"])))

    def test_every_extracted_deck_is_a_valid_entry_and_loads_as_a_table(self):
        for deck in self.document["decks"]:
            relics = list(deck["relics"])
            if not {"burning_blood", "black_blood"} & set(relics):
                relics.append("burning_blood")
            self.support.validate_entry({"id": "probe", "encounter": deck["encounter"], "hp": 1, "max_hp": 1,
                                         "deck": deck["deck"], "relics": relics, "potions": [None] * 3})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "decks.json")
            sha = rbd.write_document(self.document, path)
            v2 = stream.load_scenario_recipe(V2)
            # Lost-only encounters cannot be sampled; act 3 has wins for both encounters drawn below.
            with self.assertRaisesRegex(ValueError, "won deck"):
                replace(v2, real_boss_decks={"sha256": sha, "tiers": ["act3:boss"]},
                        real_boss_deck_table=json.loads(path.read_bytes()))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FetchTests(unittest.TestCase):
    def test_fetch_is_polite_skips_existing_and_validates_json(self):
        pages = {
            "https://baalorlord.tv/profile/0/runs/1":
                b'<a href="/runs/11" class="ironclad win"></a><a href="/runs/12" class="silent"></a>'
                b'<a href="/runs/13" class="ironclad loss"></a><a href="/runs/14" class="ironclad"></a>',
            "https://baalorlord.tv/profile/0/runs/2": b"<p>no more runs</p>",
            "https://baalorlord.tv/runs/11/raw": b'{"character_chosen": "IRONCLAD"}',
            "https://baalorlord.tv/runs/13/raw": b"not json",
        }
        requests, sleeps = [], []

        def urlopen(request, timeout):
            requests.append(request)
            self.assertIn("spire-learn", request.get_header("User-agent"))
            self.assertLessEqual(timeout, 60)
            if request.full_url not in pages:
                raise urllib.error.URLError("unexpected url")
            return _Response(pages[request.full_url])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "raw").mkdir()
            (out / "raw/14.json").write_text("{}")
            stats = rbd.fetch(out, urlopen=urlopen, sleep=sleeps.append, max_pages=10)
            self.assertEqual(json.loads((out / "raw/11.json").read_text()), {"character_chosen": "IRONCLAD"})
            self.assertFalse((out / "raw/13.json").exists())
            self.assertFalse((out / "raw/12.json").exists())
            self.assertEqual((out / "raw/14.json").read_text(), "{}")
            listing = json.loads((out / "listing.json").read_text())
            self.assertEqual(set(listing), {"11", "12", "13", "14"})
        urls = [r.full_url for r in requests]
        self.assertNotIn("https://baalorlord.tv/runs/14/raw", urls)
        self.assertNotIn("https://baalorlord.tv/runs/12/raw", urls)
        self.assertEqual(len(sleeps), len(requests))
        self.assertTrue(all(s >= 1.0 for s in sleeps))
        self.assertEqual(stats, {"listed": 4, "ironclad": 3, "fetched": 1, "skipped": 1, "failed": 1})


def _synthetic_table(support, seed=7, won_per_encounter=150):
    """Hand-built synthetic decks (no run data): random supported cards and relics."""
    rng = random.Random(seed)
    cards = sorted(support.cards)
    relics = [r for r in support.relics if r not in ("burning_blood", "black_blood", "neows_lament")]
    decks = []
    for encounter in REAL_ENCOUNTERS:
        floor = 33 if support.encounters[encounter]["act"] == 2 else 50
        for index in range(won_per_encounter + 3):
            deck = [{"card": rng.choice(cards), "upgrades": rng.randrange(2)} for _ in range(rng.randint(12, 30))]
            chosen = rng.sample(relics, rng.randrange(4))
            if index % 5 == 0:
                chosen.append("black_blood")
            decks.append({"run_id": f"{index:04d}", "encounter": encounter, "floor": floor,
                          "won": index < won_per_encounter, "ascension": 20,
                          "deck": sorted(deck, key=lambda r: (r["card"], r["upgrades"])),
                          "relics": sorted(chosen), "hp_fraction": rng.randint(1, 100) / 100,
                          "dropped_cards": rng.randrange(3)})
    decks.sort(key=lambda d: (d["encounter"], d["run_id"], d["floor"]))
    return {"schema": "spire-learn-real-boss-decks-v1", "source": "synthetic test table", "decks": decks}


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2 = stream.load_scenario_recipe(V2)
        cls.table = _synthetic_table(cls.v2.support)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.decks_path = cls.dir / "decks.json"
        cls.sha = rbd.write_document(cls.table, cls.decks_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        stream._SHARD_CACHE.clear()
        self.addCleanup(stream._SHARD_CACHE.clear)

    def recipe_path(self, tiers=("act2:boss", "act3:boss"), **changes):
        raw = json.loads(V2.read_text())
        raw["support_file"] = str(SUPPORT_PATH)
        raw["real_boss_decks"] = {"path": "decks.json", "sha256": self.sha, "tiers": list(tiers), **changes}
        path = self.dir / "recipe.json"
        path.write_text(json.dumps(raw))
        return path

    def real_matches(self, encounter, entry, targets):
        """The won table decks whose cards, HP and relics this entry carries."""
        found = []
        for deck in self.table["decks"]:
            if deck["encounter"] != encounter or not deck["won"]:
                continue
            rows = Counter((r["card"], r["upgrades"]) for r in deck["deck"])
            have = Counter((r["card"], r["upgrades"]) for r in entry["deck"])
            relics = set(deck["relics"])
            if not {"burning_blood", "black_blood"} & relics:
                relics.add("burning_blood")
            if (not rows - have and sum((have - rows).values()) <= len(targets) + 1
                    and entry["hp"] == max(1, round(deck["hp_fraction"] * entry["max_hp"]))
                    and sorted(relics) == list(entry["relics"])):
                found.append(deck)
        return found

    def shard(self, recipe, split="train", index=0):
        return stream._generate_scenario_shard(recipe, split=split, shard_index=index, arm="mixed")

    def test_boss_cases_in_listed_tiers_get_won_real_decks(self):
        recipe = stream.load_scenario_recipe(self.recipe_path())
        self.assertEqual(json_value(recipe.payload["real_boss_decks"]),
                         {"sha256": self.sha, "tiers": ["act2:boss", "act3:boss"]})
        counts = Counter()
        for split, index in (("train", 0), ("train", 1), ("selection", 0)):
            shard = self.shard(recipe, split, index)
            for case in shard.manifest["cases"]:
                entry, note = case["entry"], shard.annotations[case["case_id"]]
                recipe.support.validate_entry(entry)
                real = (note["family"] in ("run_like", "lesson")
                        and note["encounter_tier"] in ("act2:boss", "act3:boss"))
                matches = self.real_matches(entry["encounter"], entry, note["target_card_ids"])
                with self.subTest(split=split, case=case["case_id"]):
                    self.assertEqual(bool(matches), real)
                    if real:
                        have = Counter(r["card"] for r in entry["deck"])
                        self.assertFalse(Counter(note["target_card_ids"]) - have)
                counts[real] += 1
        self.assertGreater(counts[True], 20)

    def test_unlisted_tiers_keep_prepared_decks(self):
        recipe = stream.load_scenario_recipe(self.recipe_path(tiers=("act3:boss",)))
        shard = self.shard(recipe)
        seen = Counter()
        for case in shard.manifest["cases"]:
            note = shard.annotations[case["case_id"]]
            if note["family"] not in ("run_like", "lesson"):
                continue
            matches = self.real_matches(case["entry"]["encounter"], case["entry"], note["target_card_ids"])
            self.assertEqual(bool(matches), note["encounter_tier"] == "act3:boss", note)
            seen[note["encounter_tier"]] += 1
        self.assertGreater(seen["act2:boss"], 0)
        self.assertGreater(seen["act3:boss"], 0)

    def test_sha256_mismatch_fails_closed_naming_the_regenerate_command(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch.*real_boss_decks.py extract"):
            stream.load_scenario_recipe(self.recipe_path(sha256="0" * 64))

    def test_missing_file_fails_closed_naming_the_regenerate_command(self):
        with self.assertRaisesRegex(ValueError, "missing.*real_boss_decks.py extract"):
            stream.load_scenario_recipe(self.recipe_path(path="absent.json"))

    def test_field_and_table_validation(self):
        for tiers in ((), ("act1:boss",), ("act2:boss", "act2:boss"), ("act4:boss",)):
            with self.subTest(tiers=tiers), self.assertRaises(ValueError):
                stream.load_scenario_recipe(self.recipe_path(tiers=tiers))
        with self.assertRaises(ValueError):
            stream.load_scenario_recipe(self.recipe_path(extra=1))
        v1 = stream.load_scenario_recipe(ROOT / "configs/neural/curriculum-v1/recipe.json")
        with self.assertRaises(ValueError):
            replace(v1, real_boss_decks={"sha256": self.sha, "tiers": ["act2:boss"]},
                    real_boss_deck_table=self.table)
        with self.assertRaises(ValueError):
            replace(self.v2, real_boss_decks={"sha256": self.sha, "tiers": ["act2:boss"]})
        with self.assertRaises(ValueError):
            replace(self.v2, real_boss_deck_table=self.table)

        def bad(mutate):
            table = json.loads(json.dumps(self.table))
            mutate(table)
            return table

        def first(table):
            return table["decks"][0]

        bad_tables = [
            bad(lambda t: t.update(schema="other")),
            bad(lambda t: t.update(extra=1)),
            bad(lambda t: first(t).update(encounter="guardian")),
            bad(lambda t: first(t).update(won="yes")),
            bad(lambda t: first(t).update(hp_fraction=0)),
            bad(lambda t: first(t).update(hp_fraction=1.5)),
            bad(lambda t: first(t).update(deck=[])),
            bad(lambda t: first(t).update(deck=[{"card": "bash", "upgrades": 2}])),
            bad(lambda t: first(t).update(deck=[{"card": "unknown", "upgrades": 0}])),
            bad(lambda t: first(t).update(relics=["burning_blood", "black_blood"])),
            bad(lambda t: first(t).update(relics=["neows_lament"])),
            bad(lambda t: first(t).update(relics=["anchor", "anchor"])),
            bad(lambda t: first(t).update(dropped_cards=-1)),
            bad(lambda t: first(t).pop("floor")),
        ]
        for index, table in enumerate(bad_tables):
            with self.subTest(bad_table=index), self.assertRaises(ValueError):
                replace(self.v2, real_boss_decks={"sha256": content_sha256(table), "tiers": ["act2:boss"]},
                        real_boss_deck_table=table)
        # The table must match its pinned hash even when constructed directly.
        with self.assertRaisesRegex(ValueError, "SHA256"):
            replace(self.v2, real_boss_decks={"sha256": "0" * 64, "tiers": ["act2:boss"]},
                    real_boss_deck_table=self.table)

    def test_absent_table_keeps_v2_bytes_and_payload(self):
        self.assertIsNone(self.v2.real_boss_decks)
        self.assertNotIn("real_boss_decks", self.v2.payload)
        self.assertEqual(self.v2.sha256, V2_RECIPE_SHA256)
        self.assertEqual(self.shard(self.v2).manifest_sha256, V2_TRAIN_SHARD_0)
        with_table = stream.load_scenario_recipe(self.recipe_path())
        self.assertNotEqual(with_table.sha256, self.v2.sha256)
        self.assertEqual({k: v for k, v in json_value(with_table.payload).items() if k != "real_boss_decks"},
                         json_value(self.v2.payload))


if __name__ == "__main__":
    unittest.main()
