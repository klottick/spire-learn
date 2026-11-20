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
        run = copy.deepcopy(run)  # state_at reuses a run object's history
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
        run = copy.deepcopy(run)  # state_at reuses a run object's history
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
        run = copy.deepcopy(run)  # state_at reuses a run object's history
        run["relic_stats"] = {"Bottled Flame": "Carnage"}  # not in the deck
        self.assertEqual(_state(run, 9).reason, "bottle_card_missing")
        self.assertIsInstance(_state(run, 7), rs.FightState)  # bottle not yet held

    def test_burning_only_on_green_key_floor(self):
        run = _base()
        self.assertTrue(self.assertState(_state(run, 10)).burning)
        self.assertFalse(self.assertState(_state(run, 4)).burning)  # another elite
        run = copy.deepcopy(run)  # state_at reuses a run object's history
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
        run["master_deck"][run["master_deck"].index("Clothesline")] = "Mystery Card+1"
        state = _state(run, 10)
        self.assertEqual((state.reason, state.detail), ("unmapped_card:Mystery Card", "Mystery Card+1"))
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
        run["master_deck"].remove("Strike_R")
        del run["neow_bonus_log"]
        run["neow_bonus"] = "REMOVE_CARD"  # unlogged Neow could have removed that Strike too
        self.assertEqual(_state(run, 4).reason, "untracked_deck_relic")
        self.assertEqual(_cards(self.assertState(_state(run, 9))).count(("strike", 0)), 3)

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
                                     "relics_obtained": [], "gold_gain": 30, "damage_taken": 0})
        self.assertEqual(self.assertState(_state(run, 9)).room_kind, "event")  # rewards are post-fight
        run = copy.deepcopy(run)  # state_at reuses a run object's history
        run["event_choices"][-1]["max_hp_loss"] = 5
        self.assertEqual(_state(run, 9).reason, "event_fight_reordered")

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


if __name__ == "__main__":
    unittest.main()
