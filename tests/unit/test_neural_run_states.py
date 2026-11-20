"""Per-floor fight state reconstruction, on hand-written synthetic run files."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from spire_learn.neural import run_states as rs
from spire_learn.neural.standard_names import load_aliases

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests/fixtures/neural/runs"
NAMES = load_aliases(ROOT / "configs/neural/standard/name-aliases.json")
REAL_TABLE = ROOT / "artifacts/development/neural-combat/real-boss-decks/real-boss-decks.json"
REAL_RAW = ROOT / "artifacts/development/neural-combat/baalorlord-runs-20261119/raw"
REAL_TABLE_SHA = "fd28ab417875fb80955076d55e9cdb84583b2aadcaca5d615b7a1df8c7c7526a"


def _base():
    return json.loads((RUNS / "ironclad-base.json").read_text())


def _state(run, floor):
    fight = next(f for f in rs.fight_floors(run) if f.floor == floor)
    return rs.state_at(run, fight, NAMES)


def _add_relic(run, name, floor):
    run["relics"].append(name)
    run["relics_obtained"].append({"floor": floor, "key": name})


def _cards(state):
    return sorted((row["card"], row["upgrades"]) for row in state.deck)


class RunStatesTest(unittest.TestCase):
    def assertState(self, state):
        self.assertIsInstance(state, rs.FightState, state)
        return state

    def test_deck_before_floor_undoes_picks_purges_smiths(self):
        run = _base()
        # Floor 4: Neow and floors 1-3 applied (Anger pick, Upgrade Shrine on a Defend).
        f4 = self.assertState(_state(run, 4))
        self.assertEqual(_cards(f4), sorted([("ascenders_bane", 0), ("anger", 0), ("bash", 0), ("defend", 0),
                                             ("defend", 0), ("defend", 0), ("defend", 1)] + [("strike", 0)] * 5))
        # Floor 7: the floor-4 pick, floor-5 SMITH, floor-6 purchase and purge are in.
        f7 = self.assertState(_state(run, 7))
        self.assertEqual(_cards(f7), sorted([("ascenders_bane", 0), ("anger", 0), ("bash", 1), ("defend", 0),
                                             ("defend", 0), ("defend", 0), ("defend", 1), ("pommel_strike", 0),
                                             ("shrug_it_off", 0)] + [("strike", 0)] * 4))
        # Floor 12, the last: every recorded change is before it.
        f12 = self.assertState(_state(run, 12))
        self.assertEqual(_cards(f12), sorted((NAMES.card(c)) for c in run["master_deck"]))
        self.assertTrue(all(row["misc"] == 0 for row in f12.deck))
        self.assertEqual(f12.tags, [])

    def test_potions_used_on_fight_floor_were_held(self):
        f4 = self.assertState(_state(_base(), 4))
        self.assertEqual(f4.potions, ["block_potion"])
        f10 = self.assertState(_state(_base(), 10))
        self.assertEqual(f10.potions, ["strength_potion", "fire_potion"])

    def test_potion_obtained_on_floor_is_post_fight(self):
        self.assertEqual(self.assertState(_state(_base(), 2)).potions, [])
        self.assertNotIn("strength_potion", self.assertState(_state(_base(), 4)).potions)
        self.assertEqual(self.assertState(_state(_base(), 7)).potions, ["strength_potion", "fire_potion"])
        self.assertEqual(self.assertState(_state(_base(), 12)).potions, ["strength_potion"])

    def test_missing_potion_logs_keep_state_with_no_potions_tagged(self):
        run = _base()
        del run["potion_use_per_floor"], run["potion_discard_per_floor"]
        f7 = self.assertState(_state(run, 7))
        self.assertEqual((f7.potions, f7.tags), ([], ["potions_assumed_empty"]))
        # Before any potion was obtained, empty is exact.
        f2 = self.assertState(_state(run, 2))
        self.assertEqual((f2.potions, f2.tags), ([], []))
        run = copy.deepcopy(run)  # a fresh variant; the earlier states stay as they were
        run["potion_use_per_floor"] = []
        self.assertEqual(self.assertState(_state(run, 7)).tags, ["potions_assumed_empty"])

    def test_more_potions_than_slots_keeps_state_empty_tagged(self):
        run = _base()
        run["potions_obtained"].append({"floor": 7, "key": "Weak Potion"})  # a third potion in 2 slots
        for floor in (4, 7, 12):  # the gap's floor is unknown: no state of the run has exact potions
            state = self.assertState(_state(run, floor))
            self.assertEqual((state.potions, state.tags), ([], ["potions_assumed_empty"]))
        run = _base()
        run["potion_use_per_floor"][8] = ["Weak Potion"]  # used, never obtained
        self.assertEqual(self.assertState(_state(run, 4)).tags, ["potions_assumed_empty"])

    def test_potion_belt_allows_four_slots(self):
        run = _base()
        run["relics"].append("Potion Belt")
        run["relics_obtained"].append({"floor": 8, "key": "Potion Belt"})
        run["potions_obtained"] += [{"floor": 9, "key": "Weak Potion"}, {"floor": 9, "key": "FearPotion"}]
        f10 = self.assertState(_state(run, 10))
        self.assertEqual(f10.potions, ["strength_potion", "fire_potion", "weak_potion", "fear_potion"])
        self.assertEqual(f10.tags, [])
        self.assertEqual(f10.relics, ["burning_blood", "anchor", "potion_belt"])
        self.assertEqual(self.assertState(_state(run, 7)).relics, ["burning_blood"])
        run = copy.deepcopy(run)  # a fresh variant; the earlier states stay as they were
        run["potions_obtained"].append({"floor": 9, "key": "Fire Potion"})  # a fifth
        self.assertEqual(self.assertState(_state(run, 10)).tags, ["potions_assumed_empty"])

    def test_bottled_card_from_relic_stats(self):
        run = _base()
        run["relics"].append("Bottled Flame")
        run["relics_obtained"].append({"floor": 8, "key": "Bottled Flame"})
        run["relic_stats"] = {"Bottled Flame": "Pommel Strike"}
        f9 = self.assertState(_state(run, 9))
        self.assertEqual(f9.bottled_cards, {"bottled_flame": {"card": "pommel_strike", "upgrades": 0, "misc": 0}})
        self.assertEqual(f9.relics, ["burning_blood", "anchor", "bottled_flame"])
        self.assertEqual(self.assertState(_state(run, 7)).bottled_cards, {})

    def test_bottle_card_absent_drops(self):
        run = _base()
        run["relics"].append("Bottled Flame")
        run["relics_obtained"].append({"floor": 8, "key": "Bottled Flame"})
        self.assertEqual(_state(run, 9).reason, "bottle_card_missing")
        run = copy.deepcopy(run)  # a fresh variant; the earlier states stay as they were
        run["relic_stats"] = {"Bottled Flame": "Carnage"}  # not in the deck
        self.assertEqual(_state(run, 9).reason, "bottle_card_missing")
        self.assertIsInstance(_state(run, 7), rs.FightState)  # bottle not yet held

    def test_burning_only_on_green_key_floor(self):
        run = _base()
        self.assertTrue(self.assertState(_state(run, 10)).burning)
        self.assertFalse(self.assertState(_state(run, 4)).burning)  # another elite
        run = copy.deepcopy(run)  # a fresh variant; the earlier states stay as they were
        run["green_key_taken_log"] = None
        self.assertFalse(self.assertState(_state(run, 10)).burning)

    def test_hp_entering_floor(self):
        f4 = self.assertState(_state(_base(), 4))
        self.assertEqual((f4.hp, f4.max_hp, f4.gold), (66, 75, 230))
        # Floor 1 is entered from Neow: A20 start (68 of 75 HP, 99 gold) plus Neow's log,
        # checked against the floor-1 fight (68 - 5 + Burning Blood's 6 = 69).
        f1 = self.assertState(_state(_base(), 1))
        self.assertEqual((f1.hp, f1.max_hp, f1.gold), (68, 75, 199))
        run = _base()
        run["current_hp_per_floor"][0] = 70  # does not add up: not pinned
        self.assertEqual(_state(run, 1).reason, "hp_unrecorded")
        run = _base()
        del run["neow_bonus_log"]  # older runs: Neow not logged
        self.assertEqual(_state(run, 1).reason, "hp_unrecorded")

    def test_lost_fight_kept_with_won_false(self):
        run = _base()
        fights = rs.fight_floors(run)
        self.assertEqual([f.won for f in fights], [True] * 6 + [False])
        f12 = self.assertState(_state(run, 12))
        self.assertFalse(f12.won)
        self.assertEqual((f12.enemies, f12.room_kind, f12.act, f12.run_id), ("Blue Slaver", "monster", 1, "1000000001"))

    def test_unmapped_card_drops_with_name(self):
        run = _base()
        run["card_choices"][4]["picked"] = "Mystery Card"
        run["master_deck"][run["master_deck"].index("Clothesline")] = "Mystery Card"
        state = _state(run, 10)
        self.assertEqual((state.reason, state.detail), ("unmapped_card:Mystery Card", "Mystery Card"))
        self.assertIsInstance(_state(run, 9), rs.FightState)  # picked on floor 9: not held yet

    def test_master_deck_mismatch_drops_deck_unrecoverable(self):
        run = _base()
        run["master_deck"].append("Carnage")  # in the final deck, never recorded as obtained
        for fight in rs.fight_floors(run):
            self.assertEqual(rs.state_at(run, fight, NAMES).reason, "deck_unrecoverable")
        run = _base()
        run["master_deck"].remove("Anger")  # picked on floor 1, gone with no record
        self.assertEqual(_state(run, 4).reason, "deck_unrecoverable")
        self.assertEqual(_state(run, 12).reason, "deck_unrecoverable")

    def test_unrecorded_relic_change_is_put_back_from_history(self):
        run = _base()
        run["relics"].append("Empty Cage")
        run["relics_obtained"].append({"floor": 8, "key": "Empty Cage"})
        run["master_deck"].remove("Anger")
        run["master_deck"].remove("Strike_R")
        # Undoing the recorded history leaves exactly Empty Cage's two removals.
        f7 = self.assertState(_state(run, 7))
        self.assertEqual(_cards(f7).count(("strike", 0)), 4)
        self.assertIn(("anger", 0), _cards(f7))
        f9 = self.assertState(_state(run, 9))
        self.assertEqual(_cards(f9).count(("strike", 0)), 3)
        self.assertNotIn(("anger", 0), _cards(f9))

    def test_ambiguous_unrecorded_change_drops_states_before_it(self):
        run = _base()
        run["relics"].append("Empty Cage")
        run["relics_obtained"].append({"floor": 8, "key": "Empty Cage"})
        run["master_deck"].remove("Anger")  # picked on floor 1: only the Cage can have taken it
        run["master_deck"].remove("Strike_R")
        run["master_deck"].remove("Strike_R")
        del run["neow_bonus_log"]
        run["neow_bonus"] = "REMOVE_CARD"  # unlogged Neow removed one of the Strikes, the Cage the other
        self.assertEqual(_state(run, 4).reason, "untracked_deck_relic")
        self.assertEqual(_cards(self.assertState(_state(run, 9))).count(("strike", 0)), 2)
        run = copy.deepcopy(run)
        run["master_deck"].append("Strike_R")  # one removal short: an exact effect is incomplete
        self.assertEqual(_state(run, 9).reason, "deck_unrecoverable")

    def test_relics_follow_acquisition_floor_and_starter_upgrade(self):
        run = _base()
        run["relics"][0] = "Black Blood"
        run["boss_relics"] = [{"picked": "Black Blood", "not_picked": ["Sozu", "Ectoplasm"]}]
        run["path_per_floor"][10] = None  # a boss chest on floor 11
        self.assertEqual(self.assertState(_state(run, 10)).relics, ["burning_blood", "anchor"])
        self.assertEqual(self.assertState(_state(run, 12)).relics, ["black_blood", "anchor"])
        self.assertEqual(self.assertState(_state(run, 12)).act, 2)

    def test_genetic_algorithm_misc_defaulted(self):
        run = _base()
        run["card_choices"][3]["picked"] = "Genetic Algorithm"
        run["master_deck"][run["master_deck"].index("Inflame")] = "Genetic Algorithm"
        f9 = self.assertState(_state(run, 9))
        self.assertIn({"card": "genetic_algorithm", "upgrades": 0, "misc": 1}, f9.deck)
        self.assertIn("misc_defaulted", f9.tags)

    def test_neows_lament_counts_down_combats(self):
        run = _base()
        run["relics"].insert(1, "NeowsBlessing")
        run["neow_bonus"] = "THREE_ENEMY_KILL"
        run["neow_bonus_log"]["relicsObtained"] = ["NeowsBlessing"]
        run["neow_bonus_log"]["goldGained"] = 0
        run["gold_per_floor"][0] -= 100
        self.assertEqual(self.assertState(_state(run, 1)).relic_counters, {"neows_lament": 3})
        self.assertEqual(self.assertState(_state(run, 4)).relic_counters, {"neows_lament": 1})
        self.assertEqual(self.assertState(_state(run, 7)).relic_counters, {"neows_lament": -2})

    def test_fight_floors_kind_act_and_double_boss(self):
        path = (["M"] * 15 + ["B", None] + ["M"] * 15 + ["B", None] + ["?"] + ["M"] * 14 + ["B", "B", None]
                + ["R", "$", "E", "B"])
        rows = [(1, "Cultist"), (16, "Slime Boss"), (18, "Chosen"), (33, "Champ"), (35, "Mind Bloom Boss Battle"),
                (50, "Awakened One"), (51, "Time Eater"), (55, "Shield and Spear"), (56, "The Heart")]
        run = {"path_per_floor": path, "floor_reached": 57, "victory": True, "killed_by": None,
               "damage_taken": [{"floor": f, "enemies": e, "damage": 0} for f, e in rows]}
        self.assertEqual([(f.floor, f.room_kind, f.act) for f in rs.fight_floors(run)],
                         [(1, "monster", 1), (16, "boss", 1), (18, "monster", 2), (33, "boss", 2),
                          (35, "event", 3), (50, "boss", 3), (51, "boss", 3), (55, "elite", 4), (56, "boss", 4)])
        self.assertTrue(all(f.won for f in rs.fight_floors(run)))

    def test_event_fight_with_recorded_event_change_drops(self):
        run = _base()
        run["path_per_floor"][8] = "?"
        run["event_choices"].append({"floor": 9, "event_name": "Dead Adventurer", "player_choice": "Searched",
                                     "relics_obtained": [], "gold_gain": 0, "damage_taken": 0})
        self.assertEqual(self.assertState(_state(run, 9)).room_kind, "event")
        for key, value in (("max_hp_loss", 5), ("gold_gain", 30), ("relics_obtained", ["Vajra"])):
            changed = copy.deepcopy(run)
            changed["event_choices"][-1][key] = value  # a search reward may have come before the fight
            if key == "relics_obtained":
                changed["relics"].append("Vajra")
            self.assertEqual(_state(changed, 9).reason, "event_fight_reordered", key)

    def test_imports_without_gym(self):
        code = "import sys, spire_learn.neural.run_states; assert not any(m.startswith('spire_lab') for m in sys.modules)"
        subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); {code}"],
                       check=True)

    @unittest.skipUnless(REAL_TABLE.is_file() and REAL_RAW.is_dir(), "local real boss table and raw runs absent")
    def test_real_boss_table_regenerates_identically(self):
        from scripts.neural import real_boss_decks as rbd
        from spire_learn.neural.scenario_support import load_scenario_support
        support = load_scenario_support(ROOT / "configs/neural/curriculum-v1/support.json")
        document, _stats = rbd.extract(REAL_RAW, support)
        with tempfile.TemporaryDirectory() as tmp:
            sha = rbd.write_document(document, Path(tmp, "table.json"))
        self.assertEqual(sha, REAL_TABLE_SHA)
        self.assertEqual(hashlib.sha256(REAL_TABLE.read_bytes()).hexdigest(), REAL_TABLE_SHA)


class UnrecordedEffectsTest(unittest.TestCase):
    """Each unrecorded effect is put back from the run's history, or drops what it cannot pin."""

    def state(self, run, floor):
        state = _state(run, floor)
        self.assertIsInstance(state, rs.FightState, state)
        return state

    def drop(self, run, floor):
        state = _state(run, floor)
        self.assertIsInstance(state, rs.Drop, state)
        return state.reason

    def test_astrolabe(self):
        run = _base()
        _add_relic(run, "Astrolabe", 8)
        run["relic_stats"] = {"Astrolabe": ["Carnage", "Uppercut", "Rage"]}
        for card in ("Defend_R", "Defend_R", "Bash+1"):
            run["master_deck"].remove(card)
        run["master_deck"] += ["Carnage+1", "Uppercut+1", "Rage+1"]
        self.assertIn(("bash", 1), _cards(self.state(run, 7)))  # smithed on 5, then transformed
        self.assertNotIn(("carnage", 1), _cards(self.state(run, 7)))
        f9 = _cards(self.state(run, 9))
        self.assertTrue({("carnage", 1), ("uppercut", 1), ("rage", 1)} <= set(f9))
        self.assertNotIn(("bash", 1), f9)
        run = copy.deepcopy(run)
        run["master_deck"].append("Defend_R")  # only two cards transformed: incomplete effect
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")

    def test_pandoras_box(self):
        run = _base()
        _add_relic(run, "Pandora's Box", 8)
        basics = [c for c in run["master_deck"] if c.startswith(("Strike_R", "Defend_R"))]
        run["relic_stats"] = {"Pandora's Box": ["Cleave"] * len(basics)}
        run["master_deck"] = [c for c in run["master_deck"] if c not in basics] + ["Cleave"] * len(basics)
        self.assertEqual(_cards(self.state(run, 7)).count(("strike", 0)), 4)
        f9 = _cards(self.state(run, 9))
        self.assertEqual((f9.count(("cleave", 0)), f9.count(("strike", 0))), (8, 0))
        run = copy.deepcopy(run)
        run["master_deck"].append("Strike_R")  # a Strike left after Pandora's Box
        run["relic_stats"]["Pandora's Box"].append("Cleave")
        run["master_deck"].append("Cleave")
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")

    def test_tiny_house_and_whetstone(self):
        run = _base()
        _add_relic(run, "Tiny House", 8)
        run["master_deck"][run["master_deck"].index("Anger")] = "Anger+1"
        self.assertIn(("anger", 0), _cards(self.state(run, 7)))
        self.assertIn(("anger", 1), _cards(self.state(run, 9)))
        run = _base()
        _add_relic(run, "Whetstone", 8)
        run["relic_stats"] = {"Whetstone": ["Anger", "Pommel Strike"]}
        for card in ("Anger", "Pommel Strike"):
            run["master_deck"][run["master_deck"].index(card)] = card + "+1"
        self.assertTrue({("anger", 0), ("pommel_strike", 0)} <= set(_cards(self.state(run, 7))))
        self.assertTrue({("anger", 1), ("pommel_strike", 1)} <= set(_cards(self.state(run, 9))))

    def test_dollys_mirror_needs_an_original(self):
        run = _base()
        _add_relic(run, "DollysMirror", 8)
        run["master_deck"].append("Pommel Strike")
        self.assertEqual(_cards(self.state(run, 7)).count(("pommel_strike", 0)), 1)
        self.assertEqual(_cards(self.state(run, 9)).count(("pommel_strike", 0)), 2)
        run = _base()
        _add_relic(run, "DollysMirror", 8)
        run["master_deck"].append("Carnage")  # nothing to copy: Carnage was never in the deck
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")
        self.assertEqual(self.drop(run, 4), "deck_unrecoverable")

    def test_cursed_key_chests(self):
        run = _base()
        _add_relic(run, "Cursed Key", 2)
        run["relic_stats"] = {"Cursed Key": 1}
        run["master_deck"].append("Shame")  # from the floor-8 chest
        self.assertNotIn(("shame", 0), _cards(self.state(run, 7)))
        self.assertIn(("shame", 0), _cards(self.state(run, 9)))
        two_chests = copy.deepcopy(run)
        two_chests["path_per_floor"][10] = "T"  # chests on 8 and 11, a curse from each
        two_chests["relics"].append("Vajra")
        two_chests["relics_obtained"].append({"floor": 11, "key": "Vajra"})
        two_chests["relic_stats"]["Cursed Key"] = 2
        two_chests["master_deck"].append("Regret")
        self.assertEqual(self.drop(two_chests, 9), "untracked_deck_relic")  # which chest gave which
        self.assertFalse({("shame", 0), ("regret", 0)} & set(_cards(self.state(two_chests, 7))))
        self.assertTrue({("shame", 0), ("regret", 0)} <= set(_cards(self.state(two_chests, 12))))
        miscount = copy.deepcopy(run)
        miscount["relic_stats"]["Cursed Key"] = 2  # two curses from one chest
        self.assertEqual(self.drop(miscount, 9), "deck_unrecoverable")
        negated = copy.deepcopy(run)
        _add_relic(negated, "Omamori", 7)
        negated["master_deck"].remove("Shame")  # the chest's curse was negated
        negated["relic_stats"]["Omamori"] = 1  # Cursed Key's count includes the negated curse
        self.assertEqual(self.state(negated, 9).relic_counters, {"omamori": 1})
        self.assertNotIn(("shame", 0), _cards(self.state(negated, 9)))

    def test_cursed_key_unopened_chest_gives_no_curse(self):
        run = _base()
        _add_relic(run, "Cursed Key", 2)
        run["relics"].remove("Anchor")
        run["relics_obtained"] = [r for r in run["relics_obtained"] if r["key"] != "Anchor"]
        run["relic_stats"] = {"Cursed Key": 0}
        self.assertIsInstance(_state(run, 9), rs.FightState)

    def test_egg_upgrades_obtained_cards(self):
        run = _base()
        _add_relic(run, "Molten Egg 2", 2)
        run["master_deck"][run["master_deck"].index("Clothesline")] = "Clothesline+1"  # picked on 9
        self.assertIn(("clothesline", 1), _cards(self.state(run, 10)))
        run = _base()
        run["master_deck"][run["master_deck"].index("Clothesline")] = "Clothesline+1"  # no egg: unexplained
        self.assertEqual(self.drop(run, 10), "deck_unrecoverable")

    def test_upgrade_with_no_trace_drops_states_after_it(self):
        run = _base()
        _add_relic(run, "Tiny House", 8)  # its upgrade left no trace in the final deck
        self.assertIsInstance(_state(run, 7), rs.FightState)
        self.assertEqual(self.drop(run, 9), "untracked_deck_relic")

    def test_writhing_mass_parasite(self):
        run = _base()
        run["damage_taken"][4]["enemies"] = "Writhing Mass"  # floor 9
        run["master_deck"].append("Parasite")
        self.assertNotIn(("parasite", 0), _cards(self.state(run, 9)))
        self.assertIn(("parasite", 0), _cards(self.state(run, 10)))
        run["damage_taken"][5]["enemies"] = "Writhing Mass"  # floor 10 too: one Parasite, two fights
        run = copy.deepcopy(run)
        self.assertEqual(self.drop(run, 10), "deck_unrecoverable")
        self.assertIn(("parasite", 0), _cards(self.state(run, 12)))

    def test_vampires(self):
        run = _base()
        run["event_choices"][0] = {"floor": 3, "event_name": "Vampires", "player_choice": "Became a vampire",
                                   "cards_obtained": ["Bite"] * 5, "max_hp_loss": 0, "gold_gain": 0}
        run["items_purged"] = ["Defend_R"]
        run["master_deck"] = [c for c in run["master_deck"] if not c.startswith("Strike_R")]
        run["master_deck"][run["master_deck"].index("Defend_R+1")] = "Defend_R"
        run["master_deck"].remove("Defend_R")
        run["master_deck"] += ["Bite"] * 5
        self.assertEqual(_cards(self.state(run, 2)).count(("strike", 0)), 5)
        f4 = _cards(self.state(run, 4))
        self.assertEqual((f4.count(("strike", 0)), f4.count(("bite", 0))), (0, 5))
        run = copy.deepcopy(run)
        run["master_deck"].append("Strike_R")  # a Strike after Vampires
        self.assertEqual(self.drop(run, 4), "deck_unrecoverable")

    def test_lesson_learned(self):
        run = _base()
        run["card_choices"][2]["picked"] = "LessonLearned"  # floor 4, replacing Shrug It Off
        run["campfire_choices"][1]["data"] = "LessonLearned"
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "LessonLearned+1"
        logged = copy.deepcopy(run)
        run["master_deck"][run["master_deck"].index("Anger")] = "Anger+1"  # an unlogged upgrade
        self.assertIn(("anger", 0), _cards(self.state(run, 7)))
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")  # upgraded on floor 7, 9 or 10
        logged["master_deck"][logged["master_deck"].index("Anger")] = "Anger+1"
        logged["lesson_learned_per_floor"] = [[]] * 9 + [["Anger"]] + [[]] * 2  # floor 10
        self.assertIn(("anger", 0), _cards(self.state(logged, 10)))
        self.assertIn(("anger", 1), _cards(self.state(logged, 12)))

    def test_calling_bell_and_necronomicon_curses(self):
        for relic, curse in (("Calling Bell", "curse_of_the_bell"), ("Necronomicon", "necronomicurse")):
            run = _base()
            _add_relic(run, relic, 8)
            run["master_deck"].append({"curse_of_the_bell": "CurseOfTheBell"}.get(curse, "Necronomicurse"))
            self.assertNotIn((curse, 0), _cards(self.state(run, 7)))
            self.assertIn((curse, 0), _cards(self.state(run, 9)))

    def test_omamori_negates_only_curses_after_it(self):
        run = _base()
        _add_relic(run, "Omamori", 2)
        run["event_choices"][0]["cards_obtained"] = ["Regret"]  # floor 3, negated: not in the deck
        run["relic_stats"] = {"Omamori": 1}
        self.assertNotIn(("regret", 0), _cards(self.state(run, 4)))
        self.assertEqual(self.state(run, 4).relic_counters, {"omamori": 1})
        self.assertEqual(self.state(run, 2).relic_counters, {})  # obtained on floor 2
        self.assertEqual(self.state(run, 7).relic_counters, {"omamori": 1})
        late = _base()
        _add_relic(late, "Omamori", 8)  # too late to have negated the floor-3 Regret
        late["event_choices"][0]["cards_obtained"] = ["Regret"]
        self.assertEqual(self.drop(late, 4), "deck_unrecoverable")
        wrong = copy.deepcopy(run)
        wrong["relic_stats"] = {"Omamori": 0}
        self.assertEqual(self.drop(wrong, 4), "deck_unrecoverable")

    def test_omamori_and_writhing_mass_resolved_by_its_count(self):
        run = _base()
        _add_relic(run, "Omamori", 2)
        run["damage_taken"][4]["enemies"] = "Writhing Mass"  # floor 9
        run["relic_stats"] = {"Omamori": 1}  # one negation: the Parasite (no recorded curse gain)
        self.assertEqual(self.state(run, 10).relic_counters, {"omamori": 1})
        self.assertNotIn(("parasite", 0), _cards(self.state(run, 10)))
        unknown = copy.deepcopy(run)
        unknown["relic_stats"] = {}
        self.assertEqual(self.drop(unknown, 10), "deck_unrecoverable")
        both = copy.deepcopy(run)  # a Regret on floor 10 too: one negation fits either gain
        both["event_choices"].append({"floor": 11, "event_name": "Addict", "player_choice": "Stole",
                                      "cards_obtained": ["Regret"]})
        both["path_per_floor"][10] = "?"
        both["campfire_choices"].pop()
        both["master_deck"][both["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        both["relic_stats"] = {"Omamori": 2}  # both negated: the fight did add a Parasite
        f12 = self.state(both, 12)
        self.assertEqual((f12.relic_counters, ("regret", 0) in _cards(f12)), ({"omamori": 0}, False))
        both = copy.deepcopy(both)
        both["relic_stats"] = {"Omamori": 1}  # one negation: the Regret, so the fight added no Parasite
        self.assertEqual(self.state(both, 12).relic_counters, {"omamori": 1})

    def test_event_copy_named_without_upgrades(self):
        run = _base()
        run["event_choices"].append({"floor": 11, "event_name": "Duplicator", "player_choice": "Copied",
                                     "cards_obtained": ["Shrug It Off"]})
        run["path_per_floor"][10] = "?"
        run["campfire_choices"] = run["campfire_choices"][:1] + [{"floor": 8, "key": "SMITH", "data": "Shrug It Off"}]
        run["path_per_floor"][7] = "R"
        run["relics"].remove("Anchor")
        run["relics_obtained"] = []
        run["master_deck"].append("Shrug It Off+1")  # the copy of the smithed card
        self.assertEqual(_cards(self.state(run, 12)).count(("shrug_it_off", 1)), 2)
        self.assertEqual(_cards(self.state(run, 9)).count(("shrug_it_off", 1)), 1)

    def test_removed_card_named_without_upgrades(self):
        run = _base()
        run["campfire_choices"] = [{"floor": 5, "key": "SMITH", "data": "Strike_R"},
                                   {"floor": 11, "key": "PURGE", "data": "Strike_R"}]
        run["items_purged"] = ["Defend_R"]  # so only the floor-11 PURGE can have taken the smithed Strike
        run["master_deck"][run["master_deck"].index("Bash+1")] = "Bash"
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        run["master_deck"].remove("Defend_R")
        self.assertIn(("strike", 1), _cards(self.state(run, 7)))
        self.assertNotIn(("strike", 1), _cards(self.state(run, 4)))
        self.assertNotIn(("strike", 1), _cards(self.state(run, 12)))

    def test_which_removed_copy_was_upgraded_is_ambiguous(self):
        run = _base()
        run["campfire_choices"] = [{"floor": 5, "key": "SMITH", "data": "Strike_R"},
                                   {"floor": 11, "key": "PURGE", "data": "Strike_R"}]
        run["master_deck"][run["master_deck"].index("Bash+1")] = "Bash"
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        run["master_deck"].remove("Strike_R")  # purged on 6 and 11, both logged as plain Strikes
        # The smithed Strike left on floor 6 or on floor 11: states between them are unknown.
        self.assertEqual(self.drop(run, 7), "deck_unrecoverable")
        self.assertEqual(_cards(self.state(run, 4)).count(("strike", 0)), 5)
        f12 = _cards(self.state(run, 12))
        self.assertEqual((f12.count(("strike", 0)), f12.count(("strike", 1))), (3, 0))

    def test_upgraded_removal_is_not_upgraded_again(self):
        # Task 3 found Strike+2: a purge logged as "Strike_R+1" was fixed up one more level
        # for a second Shining Light upgrade that belonged to a plainly-logged removal.
        run = _base()
        run["event_choices"][0] = {"floor": 3, "event_name": "Shining Light", "player_choice": "Entered Light",
                                   "cards_upgraded": ["Strike_R", "Strike_R"]}
        run["master_deck"][run["master_deck"].index("Defend_R+1")] = "Defend_R"
        run["items_purged"] = ["Strike_R"]  # floor 6: logged plainly, yet one of the upgraded two
        run["campfire_choices"][1] = {"floor": 11, "key": "PURGE", "data": "Strike_R+1"}
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        run["master_deck"].remove("Strike_R")
        f4 = _cards(self.state(run, 4))
        self.assertEqual((f4.count(("strike", 0)), f4.count(("strike", 1))), (3, 2))
        f7 = _cards(self.state(run, 7))
        self.assertEqual((f7.count(("strike", 0)), f7.count(("strike", 1))), (3, 1))
        self.assertEqual(_cards(self.state(run, 12)).count(("strike", 0)), 3)
        self.assertFalse(any(u > 1 for c, u in f4))

    def test_two_upgrade_histories_drop_the_floors_they_disagree_on(self):
        # Two Searing Blows (floors 1 and 2), Upgrade Shrine on 3, SMITHs on 5 and 8, one
        # purged on 11 logged plainly; Searing Blow+1 survives. Entering floor 7 the pair is
        # +1/+1 or +0/+2 depending on which copy took which upgrade.
        run = _base()
        run["card_choices"][0]["picked"] = "Searing Blow"
        run["card_choices"][1]["picked"] = "Searing Blow"
        run["master_deck"].remove("Anger")
        run["master_deck"].append("Searing Blow+1")
        for card in ("Bash", "Shrug It Off", "Defend_R"):
            run["master_deck"][run["master_deck"].index(card + "+1")] = card
        run["event_choices"] = [{"floor": 3, "event_name": "Upgrade Shrine", "cards_upgraded": ["Searing Blow"]}]
        run["campfire_choices"] = [{"floor": 5, "key": "SMITH", "data": "Searing Blow"},
                                   {"floor": 8, "key": "SMITH", "data": "Searing Blow"},
                                   {"floor": 11, "key": "PURGE", "data": "Searing Blow"}]
        run["path_per_floor"][7] = "R"
        self.assertEqual(self.drop(run, 7), "deck_unrecoverable")
        searing = lambda floor: sorted(u for c, u in _cards(self.state(run, floor)) if c == "searing_blow")
        self.assertEqual(searing(2), [0])  # one copy so far: settled
        self.assertEqual(searing(12), [1])

    def test_upgrades_above_the_gym_limit_drop(self):
        run = _base()  # a consistent log can still claim an impossible card: Strike smithed twice
        run["campfire_choices"] = [{"floor": 5, "key": "SMITH", "data": "Strike_R"},
                                   {"floor": 11, "key": "SMITH", "data": "Strike_R"}]
        run["master_deck"][run["master_deck"].index("Bash+1")] = "Bash"
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        run["master_deck"][run["master_deck"].index("Strike_R")] = "Strike_R+2"
        self.assertEqual(self.drop(run, 12), "upgrades_exceed_max")
        self.assertIn(("strike", 1), _cards(self.state(run, 7)))

    def test_event_card_choice_counted_once(self):
        run = _base()
        run["event_choices"][0] = {"floor": 3, "event_name": "The Library", "player_choice": "Read",
                                   "cards_obtained": ["Carnage"]}
        run["card_choices"].append({"floor": 3, "picked": "Carnage", "not_picked": ["Rage"]})
        run["master_deck"][run["master_deck"].index("Defend_R+1")] = "Defend_R"
        run["master_deck"].append("Carnage")
        self.assertIn(("carnage", 0), _cards(self.state(run, 4)))


class CountersGoldPotionsTest(unittest.TestCase):
    def state(self, run, floor):
        state = _state(run, floor)
        self.assertIsInstance(state, rs.FightState, state)
        return state

    def test_lizard_tail_use_floor_unrecorded_drops(self):
        run = _base()
        _add_relic(run, "Lizard Tail", 2)
        run["relic_stats"] = {"Lizard Tail": 0}  # never fired
        self.assertEqual(self.state(run, 9).relic_counters, {})
        used = copy.deepcopy(run)
        used["relic_stats"] = {"Lizard Tail": 37}  # restored 37 HP on some fight: which one is not recorded
        self.assertEqual(self.state(used, 4).relic_counters, {})  # its first possible fight
        self.assertEqual(_state(used, 7).reason, "relic_counter_unrecorded")
        missing = copy.deepcopy(run)
        missing["relic_stats"] = {}
        self.assertEqual(_state(missing, 4).reason, "relic_counter_unrecorded")

    def test_maw_bank_entry_gold(self):
        run = _base()
        _add_relic(run, "MawBank", 2)
        run["relic_stats"] = {"MawBank": 48}  # floors 3-6; spent at the floor-6 shop
        self.assertEqual(self.state(run, 2).gold, 214)  # obtained on floor 2: not yet
        f4 = self.state(run, 4)
        self.assertEqual((f4.gold, f4.relic_counters), (230 + 12, {}))
        f7 = self.state(run, 7)
        self.assertEqual((f7.gold, f7.relic_counters), (100, {"maw_bank": 1}))
        run = copy.deepcopy(run)
        run["relic_stats"] = {"MawBank": 36}
        self.assertEqual(_state(run, 4).reason, "gold_unrecorded")

    def test_girya_and_tea_set(self):
        run = _base()
        _add_relic(run, "Girya", 2)
        _add_relic(run, "Ancient Tea Set", 2)
        run["campfire_choices"][1] = {"floor": 11, "key": "LIFT"}
        run["master_deck"][run["master_deck"].index("Shrug It Off+1")] = "Shrug It Off"
        self.assertEqual(self.state(run, 7).relic_counters, {"girya": 0, "ancient_tea_set": 1})  # rest on 5
        self.assertEqual(self.state(run, 9).relic_counters, {"girya": 0})
        self.assertEqual(self.state(run, 12).relic_counters, {"girya": 1, "ancient_tea_set": 1})

    def test_reward_potion_used_on_fight_floor_is_ambiguous(self):
        run = _base()
        run["potions_obtained"][0]["floor"] = 4  # the Block Potion used on floor 4 is its reward
        state = self.state(run, 4)
        self.assertEqual((state.potions, state.tags), ([], ["potions_assumed_empty"]))

    def test_entry_inventory_over_capacity_is_ambiguous(self):
        run = _base()
        run["potions_obtained"] = [{"floor": 0, "key": k} for k in ("Block Potion", "Fire Potion", "Weak Potion")]
        run["potion_use_per_floor"] = [["Weak Potion"]] + [[]] * 11
        run["items_purchased"] = ["Pommel Strike"]
        run["item_purchase_floors"] = [6]
        state = self.state(run, 1)
        self.assertEqual((state.potions, state.tags), ([], ["potions_assumed_empty"]))


class DropReasonsTest(unittest.TestCase):
    def test_unmapped_relic_and_potion(self):
        run = _base()
        _add_relic(run, "Mystery Relic", 8)
        self.assertEqual(_state(run, 9).reason, "unmapped_relic:Mystery Relic")
        run = _base()
        run["potions_obtained"][0]["key"] = "Mystery Potion"
        run["potion_use_per_floor"][3] = ["Mystery Potion"]
        self.assertEqual(_state(run, 4).reason, "unmapped_potion:Mystery Potion")

    def test_relic_with_no_source(self):
        run = _base()
        run["relics"].append("Vajra")
        self.assertEqual(_state(run, 4).reason, "relics_unrecoverable")

    def test_second_fight_on_a_floor(self):
        run = _base()
        run["path_per_floor"][8] = "?"
        run["damage_taken"].insert(5, {"floor": 9, "enemies": "Colosseum Nobs", "damage": 3})
        fights = [f for f in rs.fight_floors(run) if f.floor == 9]
        self.assertEqual([f.order for f in fights], [0, 1])
        self.assertIsInstance(rs.state_at(run, fights[0], NAMES), rs.FightState)
        self.assertEqual(rs.state_at(run, fights[1], NAMES).reason, "hp_unrecorded")

    def test_abandoned_run_last_fight_outcome_unrecorded(self):
        run = _base()
        run["killed_by"] = None
        self.assertEqual([f.won for f in rs.fight_floors(run)], [True] * 6 + [None])
        self.assertEqual(_state(run, 12).reason, "outcome_unrecorded")
        self.assertIsInstance(_state(run, 10), rs.FightState)

    def test_shared_history_gives_the_same_states(self):
        run = _base()
        history = rs.run_history(run, NAMES)
        for fight in rs.fight_floors(run):
            self.assertEqual(rs.state_at(run, fight, NAMES, history), rs.state_at(run, fight, NAMES))


class TypedUpgradesAndCountersTest(unittest.TestCase):
    """Round-2 regressions: effect card types, Egg tenure, Hungry Face evidence, outcomes, Mirror copies."""

    def state(self, run, floor):
        state = _state(run, floor)
        self.assertIsInstance(state, rs.FightState, state)
        return state

    def drop(self, run, floor):
        state = _state(run, floor)
        self.assertIsInstance(state, rs.Drop, state)
        return state.reason

    @staticmethod
    def _upgrade(run, card, times=1):
        for _ in range(times):
            run["master_deck"][run["master_deck"].index(card)] = card + "+1"

    def test_whetstone_without_stats_upgrades_only_attacks(self):
        run = _base()
        _add_relic(run, "Whetstone", 8)
        for card in ("Anger", "Pommel Strike"):
            self._upgrade(run, card)
        self.assertTrue({("anger", 1), ("pommel_strike", 1)} <= set(_cards(self.state(run, 9))))
        self.assertIn(("anger", 0), _cards(self.state(run, 7)))
        run = _base()
        _add_relic(run, "Whetstone", 8)
        self._upgrade(run, "Defend_R", 2)  # Whetstone cannot upgrade Skills
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")

    def test_whetstone_stats_listing_a_skill_drops(self):
        run = _base()
        _add_relic(run, "Whetstone", 8)
        run["relic_stats"] = {"Whetstone": ["Anger", "Defend_R"]}
        self._upgrade(run, "Anger")
        self._upgrade(run, "Defend_R")
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")

    def test_war_paint_with_and_without_stats(self):
        for stats in ({"War Paint": ["Defend_R", "Defend_R"]}, {}):
            run = _base()
            _add_relic(run, "War Paint", 8)
            run["relic_stats"] = stats
            self._upgrade(run, "Defend_R", 2)
            self.assertEqual(_cards(self.state(run, 7)).count(("defend", 1)), 1, stats)
            self.assertEqual(_cards(self.state(run, 9)).count(("defend", 1)), 3, stats)
        run = _base()
        _add_relic(run, "War Paint", 8)
        for card in ("Anger", "Pommel Strike"):  # War Paint cannot upgrade Attacks
            self._upgrade(run, card)
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")

    def test_egg_type_and_tenure(self):
        run = _base()
        _add_relic(run, "Toxic Egg 2", 2)  # Skills only: cannot explain an upgraded Attack
        self._upgrade(run, "Clothesline")
        self.assertEqual(self.drop(run, 10), "deck_unrecoverable")
        run = _base()
        _add_relic(run, "Molten Egg 2", 2)
        run["relics"].remove("Molten Egg 2")
        run["event_choices"].append({"floor": 6, "event_name": "N'loth", "player_choice": "Traded",
                                     "relics_lost": ["Molten Egg 2"]})
        self._upgrade(run, "Clothesline")  # picked on floor 9, after the Egg was lost
        self.assertEqual(self.drop(run, 10), "deck_unrecoverable")
        run = _base()
        _add_relic(run, "Molten Egg 2", 2)
        self._upgrade(run, "Clothesline")
        self.assertIn(("clothesline", 1), _cards(self.state(run, 10)))

    def test_hungry_face_needs_chest_opening_evidence(self):
        run = _base()
        _add_relic(run, "NlothsMask", 7)
        self.assertEqual(self.state(run, 7).relic_counters, {})
        self.assertEqual(self.state(run, 9).relic_counters, {"nloths_hungry_face": 1})  # Anchor from the chest
        skipped = _base()
        _add_relic(skipped, "NlothsMask", 7)
        skipped["relics"].remove("Anchor")
        skipped["relics_obtained"] = [r for r in skipped["relics_obtained"] if r["key"] != "Anchor"]
        self.assertEqual(self.drop(skipped, 9), "relic_counter_unrecorded")  # opened or skipped: unknown

    def test_pandoras_box_ambiguous_with_unlogged_neow(self):
        run = _base()
        del run["neow_bonus_log"]
        run["neow_bonus"] = "REMOVE_CARD"  # one more basic gone, Neow's or Pandora's
        _add_relic(run, "Pandora's Box", 8)
        basics = [c for c in run["master_deck"] if c.startswith(("Strike_R", "Defend_R"))]
        run["relic_stats"] = {"Pandora's Box": ["Cleave"] * (len(basics) - 1)}
        run["master_deck"] = [c for c in run["master_deck"] if c not in basics] + ["Cleave"] * (len(basics) - 1)
        self.assertEqual(self.drop(run, 4), "untracked_deck_relic")
        f9 = _cards(self.state(run, 9))
        self.assertEqual((f9.count(("cleave", 0)), f9.count(("strike", 0)), f9.count(("defend", 0))), (7, 0, 0))

    def test_unlogged_neow_alone_is_exact(self):
        run = _base()
        del run["neow_bonus_log"]
        run["neow_bonus"] = "REMOVE_TWO"
        run["master_deck"].remove("Strike_R")
        run["master_deck"].remove("Defend_R")
        f2 = _cards(self.state(run, 2))
        self.assertEqual((f2.count(("strike", 0)), f2.count(("defend", 0))), (4, 3))

    def test_killer_must_name_the_last_fight(self):
        run = _base()
        run["killed_by"] = "blue slaver"  # same name after normalization
        self.assertFalse(self.state(run, 12).won)
        run = _base()
        run["killed_by"] = "Red Slaver"  # someone else: how the last fight ended is unknown
        self.assertEqual([f.won for f in rs.fight_floors(run)], [True] * 6 + [None])
        self.assertEqual(self.drop(run, 12), "outcome_unrecorded")

    def test_mirror_copy_must_match_the_original_upgrades(self):
        run = _base()
        _add_relic(run, "DollysMirror", 8)
        run["master_deck"].append("Pommel Strike+1")  # the original was not upgraded when copied
        self.assertEqual(self.drop(run, 9), "deck_unrecoverable")


class TypedUpgradeListLengthTest(unittest.TestCase):
    """A relic_stats list shorter than the deck allowed leaves an upgrade unrecorded."""

    def _run(self, relic, listed, upgraded=()):
        run = _base()
        _add_relic(run, relic, 8)
        run["relic_stats"] = {relic: listed}
        for card in upgraded:
            run["master_deck"][run["master_deck"].index(card)] = card + "+1"
        return run

    def test_short_lists_drop_states_after_the_relic(self):
        for run in (self._run("Whetstone", []), self._run("Whetstone", ["Anger"], ["Anger"]),
                    self._run("War Paint", [])):
            self.assertIsInstance(_state(run, 7), rs.FightState)
            self.assertEqual(_state(run, 9).reason, "untracked_deck_relic")
        # One listed, a second Defend upgraded with no record: the run's history is unexplained.
        run = self._run("War Paint", ["Defend_R"], ["Defend_R", "Defend_R"])
        self.assertEqual(_state(run, 9).reason, "deck_unrecoverable")

    def test_one_entry_when_only_one_card_was_eligible(self):
        run = self._run("Whetstone", ["Anger"], ["Anger"])
        # Only Anger can be upgraded on floor 8: Pommel Strike was bought upgraded, Bash smithed.
        run["master_deck"][run["master_deck"].index("Pommel Strike")] = "Pommel Strike+1"
        run["items_purchased"][0] = "Pommel Strike+1"
        run["master_deck"] = [c for c in run["master_deck"] if not c.startswith("Strike_R")]
        run["master_deck"] += ["Defend_R"] * 4
        run["neow_bonus_log"]["cardsTransformed"] = ["Strike_R"] * 5
        run["neow_bonus_log"]["cardsObtained"] = ["Defend_R"] * 5
        run["items_purged"] = ["Defend_R"]
        state = _state(run, 9)
        self.assertIsInstance(state, rs.FightState, state)
        self.assertIn(("anger", 1), _cards(state))


if __name__ == "__main__":
    unittest.main()
