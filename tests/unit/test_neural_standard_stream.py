"""baalorlord-standard-v1 generator: synthetic state tables and fake Gym validators only."""
from __future__ import annotations

from collections import Counter
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.neural import standard_stream as ss
from spire_learn.neural.cases import _parse_case
from spire_learn.neural.scenario_support import canonical_bytes, content_sha256, json_value
from spire_learn.neural.standard_catalog import CatalogRejected

ROOT = Path(__file__).resolve().parents[2]
ALIASES = ROOT / "configs/neural/standard/name-aliases.json"
ALIASES_SHA = hashlib.sha256(ALIASES.read_bytes()).hexdigest()
SHIPPED = ROOT / "configs/neural/baalorlord-standard-v1/recipe.json"

EXTRAS = ("inflame", "shrug_it_off", "pommel_strike", "anger", "regret", "searing_blow", "ritual_dagger")
OTHER_RELICS = ("neows_lament", "vajra", "anchor", "lantern", "bag_of_marbles", "akabeko", "omamori", "potion_belt",
                "bottled_flame", "pen_nib", "kunai", "orichalcum", "red_skull", "shuriken", "ornamental_fan")
STRATA = ((1, "monster", "cultist"), (1, "elite", "lagavulin"), (1, "boss", "hexaghost"),
          (2, "monster", "chosen"), (2, "elite", "book_of_stabbing"), (3, "boss", "awakened_one"),
          (1, "event", "cultist"))
NO_FUZZ = lambda catalog: None  # noqa: E731 - the accepting fake Gym


def _row(card, upgrades=0):
    return {"card": card, "upgrades": upgrades, "misc": 15 if card == "ritual_dagger" else 0}


def _state(run_id, floor, act, kind, encounter, deck, relics, *, bottled=None, counters=None, potions=(),
           tags=()):
    deck = sorted(deck, key=lambda r: (r["card"], r["upgrades"]))
    slots = 2 + (2 if "potion_belt" in relics else 0)
    entry = {"id": f"{run_id}-{floor}", "character": "ironclad", "ascension": 20, "encounter": encounter,
             "hp": 50, "max_hp": 80, "gold": 99, "deck": deck, "relics": list(relics),
             "relic_counters": dict(counters or {}), "potions": list(potions) + [None] * (slots - len(potions))}
    if bottled:
        entry["bottled_cards"] = bottled
    tags = list(tags) + (["misc_defaulted"] if any(r["card"] == "ritual_dagger" for r in deck) else [])
    return {"run_id": run_id, "character": "ironclad", "floor": floor, "act": act, "room_kind": kind,
            "enemies": encounter, "deck": deck, "relics": list(relics), "relic_counters": dict(counters or {}),
            "bottled_cards": dict(bottled or {}), "potions": list(potions), "hp": 50, "max_hp": 80, "gold": 99,
            "burning": False, "won": True, "tags": sorted(set(tags)), "entry": entry}


def synthetic_states(runs=400, seed=7):
    rng = random.Random(seed)
    states = []
    for index in range(runs):
        run_id = str(1_600_000_000 + index * 977)
        for floor in range(1, 7):
            act, kind, encounter = STRATA[rng.randrange(len(STRATA))]
            deck = ([_row("strike", rng.randrange(2)) for _ in range(5)] + [_row("defend") for _ in range(4)]
                    + [_row("bash")])
            for _ in range(rng.randrange(8)):
                card = rng.choice(EXTRAS)
                upgrades = 0 if card == "regret" else rng.randrange(4) if card == "searing_blow" else rng.randrange(2)
                deck.append(_row(card, upgrades))
            relics = ["burning_blood"] + rng.sample(OTHER_RELICS, rng.randrange(5))
            bottled = {"bottled_flame": _row("bash")} if "bottled_flame" in relics else None
            counters = {r: {"omamori": 1, "neows_lament": 2}[r] for r in relics if r in ("omamori", "neows_lament")}
            states.append(_state(run_id, floor * 3, act, kind, encounter, deck, relics, bottled=bottled,
                                 counters=counters))
    # Another character's states never enter the Ironclad curriculum.
    states.append({**states[0], "character": "silent", "run_id": "999", "entry": {**states[0]["entry"],
                   "id": "999-3", "character": "silent"}})
    return states


def write_table(directory: Path, states=None, *, aliases_sha=ALIASES_SHA) -> tuple[Path, str]:
    document = {"aliases_sha256": aliases_sha, "gym_revision": "e3c13836bd1458be8a87b93160f5c350f78fcd80",
                "schema": "spire-learn-run-states-v1",
                "states": synthetic_states() if states is None else states}
    data = canonical_bytes(document)
    path = directory / "run-states.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def recipe_document(sha, **overrides):
    document = {
        "schema": "spire-learn-standard-recipe-v1", "generator_version": "baalorlord-standard-v1",
        "states_table": {"path": "run-states.json", "sha256": sha}, "character": "ironclad", "ascension": 20,
        "generator_seed": 20261120, "shard_size": 64, "split_fractions": [0.8, 0.1, 0.1],
        "fuzz_levels_max": 4,
        "fuzz_schedule": [{"from_shard": 0, "levels": [0]}, {"from_shard": 2, "levels": [0, 1]},
                          {"from_shard": 4, "levels": [0, 1, 2]}, {"from_shard": 6, "levels": [0, 1, 2, 4]}],
        "redraw_bound": 3, "excluded_strata": []}
    document.update(overrides)
    return document


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        _, cls.sha = write_table(cls.dir)
        cls.recipe = cls.load()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def load(cls, **overrides):
        path = cls.dir / "recipe.json"
        path.write_text(json.dumps(recipe_document(cls.sha, **overrides)))
        return ss.load_standard_recipe(path)

    def fuzz(self, state, level, seed=0):
        return ss.fuzz(state, level, self.recipe.pools, random.Random(seed))


class SplitTest(unittest.TestCase):
    def test_split_by_run_disjoint(self):
        ids = [str(1_500_000_000 + 7919 * i) for i in range(1000)]
        splits = {run_id: ss.run_split(run_id) for run_id in ids}
        self.assertTrue(all(ss.run_split(run_id) == split for run_id, split in splits.items()))
        counts = Counter(splits.values())
        for split, fraction in zip(("train", "selection", "final"), (0.8, 0.1, 0.1)):
            self.assertLessEqual(abs(counts[split] / 1000 - fraction), 0.03, counts)


class RecipeTest(_Fixture):
    def test_recipe_fields(self):
        recipe = self.recipe
        self.assertEqual(recipe.mode, "standard-standalone")
        self.assertEqual(recipe.payload["generator_version"], "baalorlord-standard-v1")
        self.assertEqual(recipe.sha256, content_sha256(recipe.payload))
        self.assertIn("level:4", recipe.category_keys)
        self.assertIn("room:1:boss", recipe.category_keys)
        self.assertNotIn("level:3", recipe.category_keys)
        self.assertNotIn("path", json.dumps(json_value(recipe.payload)))

    def test_recipe_rejects_bad_schedule(self):
        for schedule in ([{"from_shard": 1, "levels": [0]}],
                         [{"from_shard": 0, "levels": [0]}, {"from_shard": 0, "levels": [0, 1]}],
                         [{"from_shard": 0, "levels": [0]}, {"from_shard": 5, "levels": [0, 5]}],
                         [{"from_shard": 0, "levels": [0]}, {"from_shard": -1, "levels": [0]}],
                         [{"from_shard": 0, "levels": [1, 0]}]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                self.load(fuzz_schedule=schedule)
        for overrides in ({"split_fractions": [0.7, 0.2, 0.1]}, {"shard_size": 257},
                          {"excluded_strata": ["act1:boss"]}, {"character": "silent"}, {"ascension": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load(**overrides)

    def test_missing_or_mismatched_table_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = directory / "recipe.json"
            path.write_text(json.dumps(recipe_document("0" * 64)))
            with self.assertRaisesRegex(ValueError, "missing"):
                ss.load_standard_recipe(path)
            _, sha = write_table(directory)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                ss.load_standard_recipe(path)
            _, sha = write_table(directory, aliases_sha="1" * 64)
            path.write_text(json.dumps(recipe_document(sha)))
            with self.assertRaisesRegex(ValueError, "alias"):
                ss.load_standard_recipe(path)

    def test_shipped_recipe_matches_local_table(self):
        document = json.loads(SHIPPED.read_text())
        self.assertEqual(document["generator_version"], "baalorlord-standard-v1")
        self.assertEqual(document["states_table"]["sha256"],
                         "30e6e64441e3c5ec72d6c227982f583c47ed4535dcfdfe48a745cb99a2c4b44f")
        self.assertEqual(document["excluded_strata"], [])
        table = (SHIPPED.parent / document["states_table"]["path"]).resolve()
        if not table.is_file():
            self.skipTest("local run-states table is absent")
        recipe = ss.load_standard_recipe(SHIPPED)
        self.assertEqual(recipe.shard_size, 256)


class FuzzTest(_Fixture):
    BIG = [_row("strike")] * 5 + [_row("defend")] * 4 + [_row("bash"), _row("inflame", 1), _row("anger"),
                                                          _row("shrug_it_off"), _row("searing_blow", 3), _row("regret")]

    def test_level_zero_is_exact_state(self):
        state = self.recipe.states_for("train")[0]
        entry, record = self.fuzz(state, 0)
        self.assertEqual(entry, json_value(state["entry"]))
        self.assertEqual((record.cards_replaced, record.relics_replaced), (0, 0))

    def test_fuzz_level_k_changes_exactly_k_cards_and_k_relics(self):
        state = _state("1", 3, 1, "monster", "cultist", self.BIG,
                       ["burning_blood", "vajra", "anchor", "lantern", "akabeko", "pen_nib"])
        base = state["entry"]
        for level in (1, 2, 4):
            for seed in range(20):
                entry, record = self.fuzz(state, level, seed)
                cards = sum(a != b for a, b in zip(base["deck"], entry["deck"]))
                relics = sum(a != b for a, b in zip(base["relics"], entry["relics"]))
                self.assertEqual((cards, relics), (level, level))
                self.assertEqual((record.cards_replaced, record.relics_replaced), (level, level))
                self.assertEqual(len(set(entry["relics"])), len(entry["relics"]))
                for key in ("encounter", "hp", "max_hp", "gold", "character", "ascension", "id"):
                    self.assertEqual(entry[key], base[key])
                held = [p for p in entry["potions"] if p is not None]
                self.assertEqual(held, [p for p in base["potions"] if p is not None])
                self.assertEqual(len(entry["potions"]), 4 if "potion_belt" in entry["relics"] else 2)

    def test_fuzz_keeps_starter_relic(self):
        for starter in ("burning_blood", "black_blood"):
            state = _state("1", 3, 1, "monster", "cultist", self.BIG, [starter, "vajra"])
            for seed in range(10):
                entry, record = self.fuzz(state, 4, seed)
                self.assertEqual(entry["relics"][0], starter)
                self.assertEqual(record.relics_replaced, 1)
                self.assertNotIn("burning_blood" if starter == "black_blood" else "black_blood", entry["relics"])

    def test_fuzz_caps_on_small_deck_and_starter_only_relics(self):
        state = _state("1", 3, 1, "monster", "cultist", [_row("strike"), _row("bash"), _row("defend")],
                       ["burning_blood"])
        entry, record = self.fuzz(state, 4)
        self.assertEqual((record.cards_replaced, record.relics_replaced), (3, 0))
        self.assertEqual(entry["relics"], ["burning_blood"])
        self.assertTrue(all(a != b for a, b in zip(state["entry"]["deck"], entry["deck"])))

    def test_fuzz_pool_excludes_bottles(self):
        pools = self.recipe.pools
        for relic in ("bottled_flame", "bottled_lightning", "bottled_tornado", "burning_blood", "black_blood",
                      "neows_lament"):
            self.assertNotIn(relic, pools.relics)
        self.assertIn("vajra", pools.relics)
        self.assertIn("potion_belt", pools.relics)  # an ordinary non-starter relic (fix round 1 ruling)
        self.assertIn("regret", pools.cards)  # curses seen in his decks are in the card pool
        self.assertEqual(list(pools.cards), sorted(pools.cards))

    def test_pools_come_from_train_runs_only(self):
        only = "selection_only_run"
        while ss.run_split(only) != "selection":
            only += "x"
        states = synthetic_states(runs=50)
        states.append(_state(only, 3, 1, "monster", "cultist", [_row("parasite")] + self.BIG,
                             ["burning_blood", "toy_ornithopter"]))
        with tempfile.TemporaryDirectory() as raw:
            _, sha = write_table(Path(raw), states)
            path = Path(raw) / "recipe.json"
            path.write_text(json.dumps(recipe_document(sha)))
            pools = ss.load_standard_recipe(path).pools
        self.assertNotIn("parasite", pools.cards)
        self.assertNotIn("toy_ornithopter", pools.relics)

    def test_replacing_bottled_card_replaces_bottle(self):
        deck = [_row("bash"), _row("defend"), _row("defend"), _row("strike"), _row("inflame")]
        state = _state("1", 3, 1, "monster", "cultist", deck, ["burning_blood", "bottled_flame", "vajra"],
                       bottled={"bottled_flame": _row("bash")})
        seen = Counter()
        for seed in range(200):
            entry, record = self.fuzz(state, 1, seed)
            bash_kept = any(r == _row("bash") for r in entry["deck"])
            bottle_kept = "bottled_flame" in entry["relics"]
            self.assertEqual(record.relics_replaced, 1)
            self.assertEqual(bottle_kept, "bottled_flame" in entry.get("bottled_cards", {}))
            if not bash_kept:
                self.assertFalse(bottle_kept)  # a replaced bottled card takes its bottle with it
                seen["card"] += 1
            elif not bottle_kept:
                self.assertNotIn("bottled_cards", entry)  # a replaced bottle leaves its card unbottled
                seen["bottle"] += 1
        self.assertGreater(seen["card"], 0)
        self.assertGreater(seen["bottle"], 0)

    def test_replaced_relics_drop_counters_and_new_relics_take_defaults(self):
        state = _state("1", 3, 1, "monster", "cultist", self.BIG, ["burning_blood", "omamori"],
                       counters={"omamori": 1})
        entry, _ = self.fuzz(state, 1)
        self.assertNotIn("omamori", entry["relics"])
        self.assertEqual(entry["relic_counters"], {})

    def test_potion_belt_is_replaceable_with_at_most_two_potions(self):
        state = _state("1", 3, 1, "monster", "cultist", self.BIG, ["burning_blood", "potion_belt", "vajra"],
                       potions=("fire_potion", "block_potion"))
        for seed in range(20):
            entry, record = self.fuzz(state, 4, seed)
            self.assertEqual(record.relics_replaced, 2)  # the full non-starter cap
            self.assertNotIn("potion_belt", entry["relics"])
            self.assertEqual(entry["potions"], ["fire_potion", "block_potion"])  # 2 slots, potions kept

    def test_potion_belt_stays_when_more_than_two_potions_are_held(self):
        state = _state("1", 3, 1, "monster", "cultist", self.BIG, ["burning_blood", "potion_belt", "vajra"],
                       potions=("fire_potion", "block_potion", "swift_potion"))
        for seed in range(20):
            entry, record = self.fuzz(state, 4, seed)
            self.assertIn("potion_belt", entry["relics"])
            self.assertEqual(record.relics_replaced, 1)  # the cap shrinks by the ineligible belt
            self.assertEqual(entry["potions"], state["entry"]["potions"])

    def test_drawn_potion_belt_adds_empty_slots(self):
        state = _state("1", 3, 1, "monster", "cultist", self.BIG, ["burning_blood", "vajra"], potions=("fire_potion",))
        pools = dataclasses.replace(self.recipe.pools, relics=("potion_belt",))
        entry, record = ss.fuzz(state, 1, pools, random.Random(0))
        self.assertEqual(record.relics_replaced, 1)
        self.assertEqual(entry["relics"], ["burning_blood", "potion_belt"])
        self.assertEqual(entry["potions"], ["fire_potion", None, None, None])

    def test_bottle_the_pool_cannot_replace_keeps_its_card(self):
        deck = [_row("bash")]
        state = _state("1", 3, 1, "monster", "cultist", deck, ["burning_blood", "bottled_flame"],
                       bottled={"bottled_flame": _row("bash")})
        pools = dataclasses.replace(self.recipe.pools, relics=("bottled_flame",))  # nothing unheld to draw
        entry, record = ss.fuzz(state, 1, pools, random.Random(0))
        self.assertEqual((record.cards_replaced, record.relics_replaced), (0, 0))
        self.assertEqual(entry, json_value(state["entry"]))

    def test_replacements_keep_upgrades_within_new_card_limit(self):
        state = _state("1", 3, 1, "monster", "cultist", [_row("searing_blow", 3)] * 6, ["burning_blood"])
        names = self.recipe.aliases
        for seed in range(30):
            entry, _ = self.fuzz(state, 4, seed)
            for row in entry["deck"]:
                limit = names.max_upgrades(row["card"])
                self.assertEqual(row["upgrades"], 3 if limit is None else min(3, limit))
                self.assertEqual(row["misc"], {"ritual_dagger": 15, "genetic_algorithm": 1}.get(row["card"], 0))


class ShardTest(_Fixture):
    def test_train_shards_contain_only_train_runs(self):
        for index in (0, 3, 7):
            shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=index, validate=NO_FUZZ)
            runs = {a["run_id"] for a in shard.annotations.values()}
            self.assertTrue(runs)
            self.assertEqual({ss.run_split(r) for r in runs}, {"train"})
            self.assertEqual(len(shard.catalog["entries"]), 64)
            for case in json_value(shard.manifest)["cases"]:
                self.assertEqual(set(case), {"case_id", "lineage_id", "mode", "environment_seed", "cohort", "entry"})
                self.assertEqual(case["mode"], "standard-standalone")
                self.assertEqual(case["case_id"], case["entry"]["id"])
                _parse_case(case, split="train")
            annotation = next(iter(shard.annotations.values()))
            self.assertEqual(set(annotation), {"run_id", "floor", "act", "room_kind", "level", "fuzz", "tags"})

    def test_schedule_levels_by_shard_index(self):
        expect = {0: (0,), 1: (0,), 2: (0, 1), 5: (0, 1, 2), 6: (0, 1, 2, 4), 10_000: (0, 1, 2, 4)}
        for index, levels in expect.items():
            self.assertEqual(ss.levels_for_shard(self.recipe, index), levels)
        seen = {}
        for index in (0, 2, 4, 6):
            shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=index, validate=NO_FUZZ)
            seen[index] = {a["level"] for a in shard.annotations.values()}
        self.assertEqual(seen, {0: {0}, 2: {0, 1}, 4: {0, 1, 2}, 6: {0, 1, 2, 4}})

    def test_level_zero_shards_need_no_validator_but_fuzzed_shards_do(self):
        ss.generate_standard_shard(self.recipe, split="train", shard_index=1)
        with self.assertRaisesRegex(ValueError, "validator"):
            ss.generate_standard_shard(self.recipe, split="train", shard_index=2)

    def test_shard_deterministic(self):
        first = ss.generate_standard_shard(self.recipe, split="train", shard_index=6, validate=NO_FUZZ)
        ss.clear_shard_cache()
        second = ss.generate_standard_shard(self.load(), split="train", shard_index=6, validate=NO_FUZZ)
        self.assertIsNot(first, second)
        self.assertEqual(canonical_bytes(first.manifest), canonical_bytes(second.manifest))
        self.assertEqual(canonical_bytes(first.annotations), canonical_bytes(second.annotations))
        other = ss.generate_standard_shard(self.load(generator_seed=1), split="train", shard_index=6,
                                           validate=NO_FUZZ)
        self.assertNotEqual(first.manifest_sha256, other.manifest_sha256)

    def test_shard_bytes_pinned(self):
        # Panel identity (I1) and tag recomputation (M3) leave training shard bytes unchanged.
        for index, pinned in SHARD_PINS.items():
            ss.clear_shard_cache()
            shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=index, validate=NO_FUZZ)
            self.assertEqual(shard.manifest_sha256, pinned, index)
            self.assertEqual(shard.manifest_sha256, content_sha256(shard.manifest))

    def test_shipped_recipe_shard_zero_pinned(self):
        document = json.loads(SHIPPED.read_text())
        if not (SHIPPED.parent / document["states_table"]["path"]).resolve().is_file():
            self.skipTest("local run-states table is absent")
        recipe = ss.load_standard_recipe(SHIPPED)
        self.assertEqual(recipe.sha256, "d0e08c447919ff6133bb11f0d2c1184eb8612412512e9dcf15441daee7a51b5b")
        ss.clear_shard_cache()
        shard = ss.generate_standard_shard(recipe, split="train", shard_index=0)
        self.assertEqual(shard.manifest_sha256, "bcb11640c0f848e88699477b6df2108e5812b8e76ba65bb5d922ce694b15f1d9")

    def test_defaulting_tags_follow_the_fuzzed_entry(self):
        ss.clear_shard_cache()
        shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=6, validate=NO_FUZZ)
        seen = Counter()
        for entry in shard.catalog["entries"]:
            tags = shard.annotations[entry["id"]]["tags"]
            misc = any(row["card"] in ss.MISC_DEFAULTS for row in entry["deck"])
            relic = any(r in ss.DEFAULTED_COUNTER_RELICS for r in entry["relics"])
            self.assertEqual("misc_defaulted" in tags, misc)
            self.assertEqual("relic_counter_defaulted" in tags, relic)
            seen[misc, relic] += 1
        self.assertEqual(len({key[0] for key in seen}), 2)
        self.assertEqual(len({key[1] for key in seen}), 2)

    def test_misc_tag_dropped_with_its_card(self):
        deck = [_row("ritual_dagger"), _row("strike"), _row("defend"), _row("bash")]
        state = _state("1", 3, 1, "monster", "cultist", deck, ["burning_blood", "pen_nib", "vajra"])
        self.assertEqual(ss._tags(state, state["entry"]), ["misc_defaulted", "relic_counter_defaulted"])
        outcomes = Counter()
        for seed in range(60):
            entry, _ = self.fuzz(state, 2, seed)
            tags = ss._tags(state, entry)
            kept = any(row["card"] in ss.MISC_DEFAULTS for row in entry["deck"])
            self.assertEqual("misc_defaulted" in tags, kept)
            self.assertEqual("relic_counter_defaulted" in tags,
                             any(r in ss.DEFAULTED_COUNTER_RELICS for r in entry["relics"]))
            outcomes[kept] += 1
        self.assertEqual(set(outcomes), {True, False})
        # An added misc-defaulted card brings the tag in.
        plain = _state("2", 3, 1, "monster", "cultist", [_row("strike")] * 4, ["burning_blood"])
        added = json_value(plain["entry"])
        added["deck"][0] = _row("ritual_dagger")
        self.assertEqual(ss._tags(plain, added), ["misc_defaulted"])
        self.assertEqual(ss._tags(plain, plain["entry"]), [])

    def test_generation_ignores_table_row_order(self):
        shuffled = list(self.recipe.states)
        random.Random(3).shuffle(shuffled)
        recipes = [self.recipe, dataclasses.replace(self.recipe, states=tuple(reversed(self.recipe.states))),
                   dataclasses.replace(self.recipe, states=tuple(shuffled))]
        outputs = []
        for recipe in recipes:
            self.assertEqual(recipe.sha256, self.recipe.sha256)
            ss.clear_shard_cache()
            shard = ss.generate_standard_shard(recipe, split="train", shard_index=6, validate=NO_FUZZ)
            panel = ss.build_standard_panel(recipe, "selection", validate=NO_FUZZ)
            outputs.append((canonical_bytes(shard.manifest), canonical_bytes(shard.annotations),
                            content_sha256(panel)))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_duplicate_base_state_is_rejected(self):
        state = self.recipe.states[0]
        with self.assertRaisesRegex(ValueError, "duplicate base state"):
            dataclasses.replace(self.recipe, states=self.recipe.states + (dict(state),))

    def test_excluded_strata_never_sampled(self):
        recipe = self.load(excluded_strata=["1:boss", "1:event"])
        self.assertNotIn("room:1:boss", recipe.category_keys)
        for index in (0, 6):
            shard = ss.generate_standard_shard(recipe, split="train", shard_index=index, validate=NO_FUZZ)
            strata = {(a["act"], a["room_kind"]) for a in shard.annotations.values()}
            self.assertFalse(strata & {(1, "boss"), (1, "event")})

    def test_rejected_fuzz_is_redrawn_then_skipped(self):
        calls = Counter()

        def reject_vajra(document):
            calls["catalogs"] += 1
            if any("vajra" in e["relics"] for e in document["entries"]):
                raise CatalogRejected("vajra")

        shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=6, validate=reject_vajra)
        fuzzed = [e for e, a in zip(shard.catalog["entries"], shard.annotations.values()) if a["level"]]
        self.assertTrue(fuzzed)
        self.assertFalse(any("vajra" in e["relics"] for e in fuzzed))

        def reject_all(document):
            raise CatalogRejected("no")

        ss.clear_shard_cache()
        shard = ss.generate_standard_shard(self.recipe, split="train", shard_index=6, validate=reject_all)
        levels = Counter(a["level"] for a in shard.annotations.values())
        self.assertEqual(set(levels), {0})
        self.assertLess(len(shard.catalog["entries"]), 64)  # the skipped cases are counted as missing

    def test_gym_fault_propagates(self):
        def fault(document):
            raise RuntimeError("native fault")

        ss.clear_shard_cache()
        with self.assertRaisesRegex(RuntimeError, "native fault"):
            ss.generate_standard_shard(self.recipe, split="train", shard_index=6, validate=fault)


class PanelTest(_Fixture):
    def test_allocate_quotas(self):
        self.assertEqual(ss.allocate_quotas(10, {"a": 1, "b": 5, "c": 5}), {"a": 1, "b": 5, "c": 4})
        self.assertEqual(ss.allocate_quotas(10, {"a": 30, "b": 30, "c": 30}, caps={"a": 1, "b": 30, "c": 30}),
                         {"a": 1, "b": 5, "c": 4})
        self.assertEqual(sum(ss.allocate_quotas(64, {"x": 913, "y": 17, "z": 3}).values()), 64)
        with self.assertRaises(ValueError):
            ss.allocate_quotas(10, {"a": 4, "b": 5})

    def test_panel_quotas_and_distinct_bases(self):
        panel = json_value(ss.build_standard_panel(self.recipe, "selection", validate=NO_FUZZ))
        cases = panel["manifest"]["cases"]
        self.assertEqual(len(cases), 256)
        self.assertEqual(len({c["case_id"] for c in cases}), 256)
        self.assertEqual(len({c["environment_seed"] for c in cases}), 256)
        annotations = panel["annotations"]
        sizes = Counter(f"{s['act']}:{s['room_kind']}" for s in self.recipe.states_for("selection")
                        if s["room_kind"] != "event")
        expected = ss.allocate_quotas(64, dict(sizes))
        for level in (0, 1, 2, 4):
            rows = [annotations[c["case_id"]] for c in cases if annotations[c["case_id"]]["level"] == level]
            self.assertEqual(len(rows), 64)
            bases = [(r["run_id"], r["floor"]) for r in rows]
            self.assertEqual(len(set(bases)), 64)
            self.assertTrue(all(ss.run_split(r["run_id"]) == "selection" for r in rows))
            counts = Counter(f"{r['act']}:{r['room_kind']}" for r in rows)
            self.assertEqual(dict(counts), {k: v for k, v in expected.items() if v})
            self.assertEqual(panel["strata"][str(level)], {k: v for k, v in sorted(expected.items()) if v})
        for case in cases:
            _parse_case(case, split="selection")

    def test_panel_redistributes_exhausted_stratum(self):
        def reject_boss(document):
            if any(e["encounter"] == "awakened_one" for e in document["entries"]):
                raise CatalogRejected("boss")

        recipe = self.recipe
        panel = json_value(ss.build_standard_panel(recipe, "selection", validate=reject_boss))
        annotations = panel["annotations"]
        for level in (1, 2, 4):
            rows = [a for a in annotations.values() if a["level"] == level]
            self.assertEqual(len(rows), 64)
            self.assertFalse(any(r["act"] == 3 and r["room_kind"] == "boss" for r in rows))
        self.assertTrue(any(a["act"] == 3 and a["room_kind"] == "boss" and a["level"] == 0
                            for a in annotations.values()))
        self.assertGreater(sum(panel["skipped"].values()), 0)

    def test_panel_honours_excluded_strata(self):
        recipe = self.load(excluded_strata=["1:monster", "3:boss"])
        panel = json_value(ss.build_standard_panel(recipe, "selection", validate=NO_FUZZ))
        rows = list(panel["annotations"].values())
        self.assertEqual(len(rows), 256)
        self.assertFalse({(r["act"], r["room_kind"]) for r in rows} & {(1, "monster"), (3, "boss")})
        self.assertNotIn("1:monster", panel["available"])
        for level in ("0", "1", "2", "4"):
            self.assertEqual(sum(panel["strata"][level].values()), 64)

    def test_panel_excludes_event_fights(self):
        self.assertTrue(any(s["room_kind"] == "event" for s in self.recipe.states_for("selection")))
        for split in ("selection", "final"):
            panel = json_value(ss.build_standard_panel(self.recipe, split, validate=NO_FUZZ))
            self.assertFalse(any(a["room_kind"] == "event" for a in panel["annotations"].values()))
            self.assertFalse(any(key.endswith(":event") for key in panel["available"]))
            self.assertEqual(panel["panel_identity"]["panel_kinds"], ["monster", "elite", "boss"])

    def test_panel_ignores_training_only_recipe_fields(self):
        base = {split: canonical_bytes(ss.build_standard_panel(self.recipe, split, validate=NO_FUZZ))
                for split in ("selection", "final")}
        schedule = [{"from_shard": 0, "levels": [0]}, {"from_shard": 50, "levels": [0, 1, 2, 4]}]
        for overrides in ({"fuzz_schedule": schedule}, {"shard_size": 128}, {"generator_seed": 1},
                          {"fuzz_schedule": schedule, "shard_size": 7, "generator_seed": 99}):
            recipe = self.load(**overrides)
            self.assertNotEqual(recipe.sha256, self.recipe.sha256)
            self.assertEqual(ss.panel_sha256(recipe), ss.panel_sha256(self.recipe))
            for split in ("selection", "final"):
                with self.subTest(overrides=overrides, split=split):
                    self.assertEqual(canonical_bytes(ss.build_standard_panel(recipe, split, validate=NO_FUZZ)),
                                     base[split])

    def test_panel_fields_move_the_panel(self):
        base = {split: content_sha256(ss.build_standard_panel(self.recipe, split, validate=NO_FUZZ))
                for split in ("selection", "final")}
        panel = json_value(ss.build_standard_panel(self.recipe, "selection", validate=NO_FUZZ))
        self.assertEqual(panel["panel_sha256"], ss.panel_sha256(self.recipe))
        self.assertEqual(panel["panel_sha256"], content_sha256(panel["panel_identity"]))
        self.assertEqual(panel["panel_identity"]["states_table"], {"sha256": self.sha})
        self.assertNotIn("recipe_sha256", panel)
        changed = [self.load(redraw_bound=4), self.load(fuzz_levels_max=5), self.load(excluded_strata=["4:elite"]),
                   dataclasses.replace(self.recipe, table_sha256="f" * 64)]
        for recipe in changed:
            self.assertNotEqual(ss.panel_sha256(recipe), ss.panel_sha256(self.recipe))
            for split in ("selection", "final"):
                with self.subTest(identity=ss.panel_identity(recipe), split=split):
                    self.assertNotEqual(content_sha256(ss.build_standard_panel(recipe, split, validate=NO_FUZZ)),
                                        base[split])
        with patch.object(ss, "PANEL_VERSION", "baalorlord-standard-panel-test"):
            self.assertNotEqual(ss.panel_sha256(self.recipe), panel["panel_sha256"])

    def test_panel_deterministic_hash(self):
        first = ss.build_standard_panel(self.recipe, "selection", validate=NO_FUZZ)
        second = ss.build_standard_panel(self.load(), "selection", validate=NO_FUZZ)
        self.assertEqual(content_sha256(first), content_sha256(second))
        final = ss.build_standard_panel(self.recipe, "final", validate=NO_FUZZ)
        self.assertNotEqual(content_sha256(first), content_sha256(final))
        self.assertEqual(content_sha256(first), PINNED_SELECTION_PANEL)
        with self.assertRaises(ValueError):
            ss.build_standard_panel(self.recipe, "train", validate=NO_FUZZ)


SHARD_PINS = {0: "23d6abab4341df1e1c60815166b71ce73bb8642fac855b5a71563418312bd388",
              2: "ae19a9c3b4e29971286fc32a1dbda01d717b30272977cc7968cb3dea4661d2a8",
              6: "86dbaa45a1d4831fb753f8c954d4d3076831944ce60a96744186b4c81d3d0523"}
PINNED_SELECTION_PANEL = "8e6da5392c7da0ed79436887a2e6b5189021a9118e8bf47705cd1ce6b9b2f058"


if __name__ == "__main__":
    unittest.main()
