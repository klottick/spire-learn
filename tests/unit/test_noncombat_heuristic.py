"""noncombat-heuristic-v1 on synthetic Run boundaries built from the Standard Run descriptor (stdlib only)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from spire_learn.neural.noncombat_heuristic import (
    HEURISTIC_ID, ROOM_RULES, SELECTION_RULES, CardPriority, HeuristicError, NoncombatHeuristic, RoomFlags,
)

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = json.loads((ROOT / "tests/fixtures/neural/standard-run-1929e18.json").read_text())
PRIORITY_PATH = ROOT / "configs/neural/full-run/ironclad-priority-v1.json"
VOCAB = DESCRIPTOR["vocabularies"]


def _zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class Boundary:
    """A blank public Run observation (every tensor zero) with setters for the fields the heuristic reads."""

    TENSORS = {tensor["key"]: tensor for tensor in DESCRIPTOR["tensors"]}

    def __init__(self, room, *, hp=60, max_hp=80, gold=100, act=1, floor=3, event=None, slots=2):
        self.obs = {key: _zeros(tensor["shape"]) for key, tensor in self.TENSORS.items()}
        self.legal = []
        self.set("context", None, room_kind=VOCAB["room"].index(room), act=act, floor=floor)
        if event is not None:
            self.set("context", None, event_kind=VOCAB["event"].index(event))
        self.set("player", None, hp=hp, max_hp=max_hp, gold=gold)
        for slot in range(slots):
            self.obs["potions_known"][slot][0] = 1
        self._rows = {"offers": 0, "actions": 0, "permanent_cards": 0, "selection": 0, "map_nodes": 0}

    def _column(self, key, field):
        return self.TENSORS[key]["fields"].index(field)

    def set(self, key, row, **fields):
        for field, value in fields.items():
            column = self._column(key, field)
            if row is None:
                self.obs[key][column] = value
                self.obs[f"{key}_known"][column] = 1
            else:
                self.obs[key][row][column] = value
                self.obs[f"{key}_known"][row][column] = 1
        if row is not None and f"{key}_occupied" in self.obs:
            self.obs[f"{key}_occupied"][row] = 1

    def _next(self, key):
        row = self._rows[key]
        self._rows[key] += 1
        return row

    def offer(self, choice, *, status="offered", **fields):
        row = self._next("offers")
        values = {"choice_kind": VOCAB["choice"].index(choice), "status": VOCAB["choice_status"].index(status)}
        for field, value in fields.items():
            vocabulary = {"card_kind": "card", "relic_kind": "relic", "potion_kind": "potion",
                          "neow_reward": "neow_reward", "neow_drawback": "neow_drawback"}.get(field)
            values[field] = VOCAB[vocabulary].index(value) if vocabulary else value
        self.set("offers", row, **values)
        return row

    def action(self, verb, *, group="none", row=-1, choice=None, legal=True):
        index = self._next("actions")
        values = {"verb": VOCAB["verb"].index(verb), "source_group": VOCAB["row_group"].index(group),
                  "source_row": row}
        if choice is not None:
            values["choice_kind"] = VOCAB["choice"].index(choice)
        self.set("actions", index, **values)
        if legal:
            self.obs["action_mask"][index] = 1
            self.legal.append(index)
        return index

    def choose(self, choice, **fields):
        """An offer and its legal choose action; returns the action index."""
        return self.action("choose", group="offers", row=self.offer(choice, **fields), choice=choice)

    def card(self, card, upgrades=0):
        row = self._next("permanent_cards")
        self.set("permanent_cards", row, card_kind=VOCAB["card"].index(card), upgrades=upgrades, misc=0)
        return row

    def selection(self, operation, cards, *, required=1, selected=0, confirm=False, cancel=True):
        """A Run card selection over deck ``cards`` (name or (name, eligible)); returns the select action indices."""
        self.set("selection_counts", None, required=required, selected_count=selected, minimum=required,
                 maximum=required, confirm_available=int(confirm),
                 operation=VOCAB["selection_operation"].index(operation))
        actions = []
        for item in cards:
            name, eligible = (item, True) if isinstance(item, str) else item
            deck_row = self.card(name)
            row = self._next("selection")
            self.set("selection", row, source_group=VOCAB["row_group"].index("permanent_cards"),
                     source_row=deck_row, eligible=int(eligible), selected=0)
            actions.append(self.action("select", group="selection", row=row) if eligible else None)
        if confirm:
            actions.append(self.action("confirm"))
        if cancel:
            actions.append(self.action("cancel"))
        return actions

    def node(self, kind):
        """A reachable map node and its legal move; returns the action index."""
        row = self._next("map_nodes")
        self.set("map_nodes", row, kind=VOCAB["map_kind"].index(kind), floor=4)
        return self.action("move", group="map_nodes", row=row)

    def potion(self, slot, name):
        self.set("potions", slot, potion_kind=VOCAB["potion"].index(name))
        return self.action("discard_potion", group="potions", row=slot)


class HeuristicTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.priority = CardPriority.load(PRIORITY_PATH)
        cls.heuristic = NoncombatHeuristic(DESCRIPTOR, cls.priority)

    def pick(self, boundary, flags=None):
        return self.heuristic.choose(boundary.obs, boundary.legal, flags)


class TotalityTests(HeuristicTestCase):
    def test_heuristic_total_over_descriptor_enums(self):
        self.assertEqual(HEURISTIC_ID, "noncombat-heuristic-v1")
        self.assertEqual(set(ROOM_RULES), set(VOCAB["room"]))
        self.assertEqual(set(SELECTION_RULES), set(VOCAB["selection_operation"]))
        for name, vocabulary in (("room", "room"), ("operation", "selection_operation")):
            with self.subTest(extended=name):
                extended = copy.deepcopy(DESCRIPTOR)
                extended["vocabularies"][vocabulary].append("new_kind_from_a_later_gym")
                with self.assertRaisesRegex(HeuristicError, "new_kind_from_a_later_gym"):
                    NoncombatHeuristic(extended, self.priority)
        # Every room has a declared rule; each yields a legal choice on a boundary with one proceed.
        for room in VOCAB["room"]:
            with self.subTest(room=room):
                boundary = Boundary(room)
                proceed = boundary.choose("proceed")
                choice = self.pick(boundary, RoomFlags())
                self.assertEqual(choice.action, proceed)
                self.assertIn(choice.action, boundary.legal)

    def _legal_non_potion(self, boundary, choice):
        self.assertIn(choice.action, boundary.legal)
        verb = boundary.obs["actions"][choice.action][boundary._column("actions", "verb")]
        self.assertNotIn(VOCAB["verb"][verb], ("use_potion", "discard_potion"))
        self.assertEqual(choice.fallback, choice.rule.startswith("fallback:"))
        return VOCAB["verb"][verb]

    def test_choose_total_over_every_selection_operation(self):
        """Every operation on a structured selection: a legal non-potion move or a counted fallback, never a
        cancel while a select or confirm is legal, and no reselection loop while the selection advances."""
        cards = ["regret", "strike", "defend", "clash", ("bash", False), "offering", "anger"]
        for operation in VOCAB["selection_operation"]:
            for confirm in (False, True):
                with self.subTest(operation=operation, confirm=confirm):
                    boundary = Boundary("event", event="living_wall")
                    boundary.potion(0, "fire_potion")
                    actions = boundary.selection(operation, cards, required=2, confirm=confirm)
                    cancel = actions[-1]
                    family = SELECTION_RULES[operation]
                    seen = set()
                    for _ in range(len(cards) + 2):
                        verb_column = boundary._column("actions", "verb")
                        open_moves = any(VOCAB["verb"][boundary.obs["actions"][i][verb_column]] in ("select", "confirm")
                                         for i in boundary.legal)
                        choice = self.pick(boundary)
                        verb = self._legal_non_potion(boundary, choice)
                        if open_moves:
                            self.assertNotEqual(choice.action, cancel)
                            self.assertEqual(choice.fallback, family in ("fallback", "combat"))
                        if confirm and family not in ("fallback", "combat"):
                            self.assertEqual(verb, "confirm")
                        if verb != "select":
                            break
                        self.assertNotIn(choice.action, seen)
                        seen.add(choice.action)
                        # The engine marks the picked row selected; it is no longer a candidate.
                        row = boundary.obs["actions"][choice.action][boundary._column("actions", "source_row")]
                        boundary.set("selection", row, selected=1)
                        boundary.obs["action_mask"][choice.action] = 0
                        boundary.legal.remove(choice.action)
                    else:
                        self.fail(f"{operation} kept selecting")
        named = {"select_permanent": (True, "strike"), "event_choose": (False, "regret"),
                 "relic_equip": (False, "offering")}
        for operation, (fallback, card) in named.items():
            with self.subTest(named=operation):
                boundary = Boundary("event")
                boundary.potion(0, "fire_potion")
                actions = boundary.selection(operation, ["strike", "regret", "offering", "bash"])
                choice = self.pick(boundary)
                expected = ["strike", "regret", "offering", "bash"].index(card)
                self.assertEqual((choice.action, choice.fallback), (actions[expected], fallback))

    def test_choose_total_over_every_room(self):
        """Every room on a boundary with many legal kinds: a legal non-potion move or a counted fallback."""
        for room in VOCAB["room"]:
            with self.subTest(room=room):
                boundary = Boundary(room, event="golden_shrine", hp=50, max_hp=80)
                boundary.potion(0, "fire_potion")
                boundary.action("use_potion", group="potions", row=0)
                keys = {boundary.choose(key) for key in ("sapphire", "emerald")}
                for kind in ("monster", "elite"):
                    boundary.node(kind)
                boundary.card("strike")
                boundary.choose("neow_blessing", neow_reward="three_cards")
                boundary.choose("event_option", event_stage=0, event_option=0, param0=50)
                for choice in ("gold", "relic", "card_reward", "potion", "open_chest", "rest", "smith", "recall",
                               "purge", "leave", "skip", "act3_continue", "proceed"):
                    boundary.choose(choice, **({"amount": 10} if choice == "gold" else {}))
                boundary.choose("card", card_kind="offering", price=10)
                choice = self.pick(boundary, RoomFlags())
                self._legal_non_potion(boundary, choice)
                if not choice.fallback:
                    self.assertNotIn(choice.action, keys)
        for room in ("act3_ready", "act4_ready", "act2_ready", "act3_boss_complete", "checkpoint_complete"):
            with self.subTest(transition=room):
                boundary = Boundary(room)
                boundary.potion(0, "fire_potion")
                boundary.choose("sapphire")
                boundary.choose("ruby" if "ruby" in VOCAB["choice"] else "emerald")
                proceed = boundary.choose("proceed")
                choice = self.pick(boundary)
                self.assertEqual((choice.action, choice.rule, choice.fallback), (proceed, "transition", False))

    def test_act3_ending_skips_keys(self):
        """Without keys the Act 3 ending is act3_continue (Gym act3_end.rs:100-112); a lower-index key is skipped."""
        boundary = Boundary("act3_ending", act=3, floor=51)
        boundary.potion(0, "fire_potion")
        boundary.choose("sapphire")
        boundary.choose("emerald")
        cont = boundary.choose("act3_continue")
        choice = self.pick(boundary)
        self.assertEqual((choice.action, choice.rule, choice.fallback), (cont, "transition", False))

    def test_priority_list_validates_against_descriptor(self):
        self.assertEqual(self.priority.id, "ironclad-priority-v1")
        self.assertLess(self.priority.rank("offering"), self.priority.rank("strike"))
        self.assertIsNone(self.priority.rank("clash"))
        broken = copy.deepcopy(DESCRIPTOR)
        broken["vocabularies"]["card"] = [name for name in VOCAB["card"] if name != "offering"]
        with self.assertRaisesRegex(HeuristicError, "offering"):
            NoncombatHeuristic(broken, self.priority)


class RuleTests(HeuristicTestCase):
    def test_heuristic_rules_on_synthetic_boundaries(self):
        with self.subTest("rewards: gold, relic, card reward, potion with a free slot, proceed; keys skipped"):
            boundary = Boundary("rewards")
            boundary.choose("sapphire")
            proceed = boundary.choose("proceed")
            potion = boundary.choose("potion", potion_kind="fire_potion")
            card_reward = boundary.choose("card_reward")
            relic = boundary.choose("relic", relic_kind="anchor")
            gold = boundary.choose("gold", amount=20)
            self.assertEqual(self.pick(boundary).action, gold)
            for taken in (gold, relic, card_reward, potion):
                boundary.obs["action_mask"][taken] = 0
                boundary.legal.remove(taken)
                expected = {gold: relic, relic: card_reward, card_reward: potion, potion: proceed}[taken]
                self.assertEqual(self.pick(boundary).action, expected)
        with self.subTest("treasure opens its chest before proceeding"):
            boundary = Boundary("treasure")
            opened = boundary.choose("open_chest")
            boundary.choose("proceed")
            self.assertEqual(self.pick(boundary).action, opened)
        with self.subTest("card reward: highest listed card"):
            boundary = Boundary("card_reward")
            boundary.choose("card", card_kind="anger")
            best = boundary.choose("card", card_kind="shrug_it_off")
            boundary.choose("card", card_kind="clash")
            boundary.choose("skip")
            choice = self.pick(boundary)
            self.assertEqual((choice.action, choice.fallback), (best, False))
        with self.subTest("card reward: skip when no listed card is offered"):
            boundary = Boundary("card_reward")
            boundary.choose("card", card_kind="clash")
            boundary.choose("card", card_kind="panic_button")
            skip = boundary.choose("skip")
            self.assertEqual(self.pick(boundary).action, skip)
        with self.subTest("map: rest only below 55% HP, then treasure, monster, question, shop, elite, boss"):
            order = ["boss", "elite", "shop", "question", "monster", "treasure", "rest"]
            for hp, expected in ((40, "rest"), (50, "rest"), (60, "treasure"), (79, "treasure")):
                boundary = Boundary("map", hp=hp, max_hp=100)
                moves = {kind: boundary.node(kind) for kind in order}
                self.assertEqual(self.pick(boundary).action, moves[expected], (hp, expected))
            boundary = Boundary("map", hp=90, max_hp=100)
            moves = {kind: boundary.node(kind) for kind in ("boss", "elite", "shop", "rest")}
            self.assertEqual(self.pick(boundary).action, moves["shop"])
            boundary = Boundary("map", hp=90, max_hp=100)
            moves = {kind: boundary.node(kind) for kind in ("event", "monster")}
            self.assertEqual(self.pick(boundary).action, moves["monster"])
            boundary = Boundary("map", hp=90, max_hp=100)
            first, second = boundary.node("monster"), boundary.node("monster")
            self.assertEqual(self.pick(boundary).action, min(first, second))
        with self.subTest("campfire: rest below 50% HP, else smith"):
            for hp, expected in ((39, "rest"), (40, "smith")):
                boundary = Boundary("campfire", hp=hp, max_hp=80)
                boundary.card("bash")
                actions = {kind: boundary.choose(kind) for kind in ("recall", "rest", "smith", "lift")}
                self.assertEqual(self.pick(boundary).action, actions[expected])
        with self.subTest("campfire: recall, lift, toke and dig only when nothing else is legal"):
            boundary = Boundary("campfire", hp=20, max_hp=80)
            dig = boundary.choose("dig")
            boundary.choose("toke")
            self.assertEqual(self.pick(boundary).action, dig)
        with self.subTest("neow: the first option without an HP or max-HP cost"):
            boundary = Boundary("neow")
            boundary.choose("neow_blessing", neow_reward="max_hp_twice", neow_drawback="max_hp_loss")
            boundary.choose("neow_blessing", neow_reward="remove_two", neow_drawback="current_hp_loss")
            curse = boundary.choose("neow_blessing", neow_reward="rare_relic", neow_drawback="random_curse")
            boundary.choose("neow_blessing", neow_reward="three_cards")
            self.assertEqual(self.pick(boundary).action, curse)
        with self.subTest("boss relics: the first listed"):
            boundary = Boundary("boss_relic_choice")
            first = boundary.choose("relic", relic_kind="snecko_eye")
            boundary.choose("relic", relic_kind="cursed_key")
            boundary.choose("skip")
            self.assertEqual(self.pick(boundary).action, first)
        with self.subTest("events: skip HP-loss options below 50% HP and curses; take the first remaining"):
            boundary = Boundary("event", event="golden_idol", hp=30, max_hp=80)
            curse = boundary.choose("event_option", event_stage=1, event_option=0)       # injury: a curse
            boundary.choose("event_option", event_stage=1, event_option=1, param0=20)  # damage_taken
            boundary.choose("event_option", event_stage=1, event_option=2, param0=6)   # max_hp_lost
            choice = self.pick(boundary)
            # Everything is excluded: the smallest non-lethal HP plus max-HP loss (the curse loses none).
            self.assertEqual((choice.action, choice.rule), (curse, "event:least-loss"))
            boundary = Boundary("event", event="golden_idol", hp=30, max_hp=80)
            boundary.choose("event_option", event_stage=1, event_option=1, param0=20)
            maxhp = boundary.choose("event_option", event_stage=1, event_option=2, param0=6)
            self.assertEqual(self.pick(boundary).action, maxhp)
            boundary = Boundary("event", event="golden_idol", hp=70, max_hp=80)
            boundary.choose("event_option", event_stage=1, event_option=0)
            damage = boundary.choose("event_option", event_stage=1, event_option=1, param0=20)
            boundary.choose("event_option", event_stage=1, event_option=2, param0=6)
            self.assertEqual(self.pick(boundary).action, damage)
        with self.subTest("events: when every option would be lethal, the smallest loss anyway"):
            boundary = Boundary("event", event="golden_idol", hp=5, max_hp=80)
            smaller = boundary.choose("event_option", event_stage=1, event_option=1, param0=20)
            boundary.choose("event_option", event_stage=1, event_option=2, param0=90)
            self.assertEqual(self.pick(boundary).action, smaller)
        with self.subTest("removal: curse, then strike, then defend, then unlisted, then the lowest listed"):
            for cards, expected in ((["bash", "defend", "strike", "regret"], "regret"),
                                    (["bash", "defend", "strike"], "strike"),
                                    (["bash", "defend"], "defend"),
                                    (["offering", "clash", "bash"], "clash"),
                                    (["offering", "anger", "bash"], "bash")):
                for operation in ("shop_purge", "event_transform", "campfire_toke", "neow_remove", "event_choose"):
                    boundary = Boundary("shop")
                    selects = boundary.selection(operation, cards)
                    self.assertEqual(self.pick(boundary).action, selects[cards.index(expected)], (operation, cards))
        with self.subTest("upgrade, duplicate and bottle: the highest listed eligible card"):
            for operation in ("smith", "event_upgrade", "event_duplicate", "relic_equip", "neow_upgrade"):
                boundary = Boundary("campfire")
                selects = boundary.selection(operation, ["strike", ("offering", False), "anger", "bash"])
                self.assertEqual(self.pick(boundary).action, selects[2], operation)
        with self.subTest("confirm as soon as it is legal"):
            boundary = Boundary("event")
            *_, confirm, _ = boundary.selection("event_remove", ["strike", "defend"], confirm=True)
            self.assertEqual(self.pick(boundary).action, confirm)
        with self.subTest("ties break by the lowest legal index"):
            boundary = Boundary("card_reward")
            first = boundary.choose("card", card_kind="anger")
            boundary.choose("card", card_kind="anger")
            self.assertEqual(self.pick(boundary).action, first)
        with self.subTest("deterministic"):
            boundary = Boundary("map")
            for kind in ("elite", "monster", "monster", "question"):
                boundary.node(kind)
            self.assertEqual({self.pick(boundary) for _ in range(5)}, {self.pick(boundary)})

    def test_shop_terminates_with_courier(self):
        """With The Courier every bought item is restocked in place; the shop still ends in at most 3 shop steps."""
        flags = RoomFlags()
        shop_steps, selected = 0, []
        boundary = None
        for _ in range(6):
            boundary = Boundary("shop", gold=500, act=2, floor=22)
            for card in ("strike", "defend", "bash"):
                boundary.card(card)
            boundary.choose("card", card_kind="anger", price=50)
            boundary.choose("card", card_kind="offering", price=150)
            boundary.choose("relic", relic_kind="anchor", price=150)
            boundary.choose("purge", price=75)
            leave = boundary.choose("leave")
            choice = self.pick(boundary, flags)
            flags.record(self.heuristic.room_key(boundary.obs), choice)
            shop_steps += 1
            selected.append(choice.rule)
            if choice.action == leave:
                break
        self.assertEqual(selected, ["shop:purge", "shop:buy", "shop:leave"])
        self.assertLessEqual(shop_steps, 3)
        # Flags are per room: the next shop purges again.
        boundary = Boundary("shop", gold=500, act=2, floor=30)
        boundary.card("strike")
        purge = boundary.choose("purge", price=75)
        boundary.choose("leave")
        self.assertEqual(self.pick(boundary, flags).action, purge)

    def test_knowing_skull_leaves(self):
        for hp in (20, 70):
            with self.subTest(hp=hp):
                boundary = Boundary("event", event="knowing_skull", hp=hp, max_hp=80)
                for option in range(4):
                    boundary.choose("event_option", event_stage=0, event_option=option, param0=6)
                choice = self.pick(boundary)
                self.assertEqual(choice.action, 3)
                self.assertFalse(choice.fallback)
        with self.subTest("scrap ooze reaches in only from 50% HP"):
            for hp, expected in ((39, 1), (40, 0)):
                boundary = Boundary("event", event="scrap_ooze", hp=hp, max_hp=80)
                boundary.choose("event_option", event_stage=0, event_option=0, param0=3, param1=25)
                boundary.choose("event_option", event_stage=0, event_option=1)
                self.assertEqual(self.pick(boundary).action, expected)

    def test_unlisted_smith_never_cancels(self):
        boundary = Boundary("campfire")
        selects = boundary.selection("smith", ["clash", "wild_strike", "reckless_charge"], cancel=False)
        cancel = boundary.action("cancel")
        choice = self.pick(boundary)
        self.assertEqual(choice.action, selects[0])
        self.assertNotEqual(choice.action, cancel)
        self.assertFalse(choice.fallback)
        # A selection whose only legal action is cancel cancels (nothing else is possible).
        boundary = Boundary("campfire")
        boundary.selection("smith", [("clash", False)], cancel=False)
        only = boundary.action("cancel")
        self.assertEqual(self.pick(boundary).action, only)

    def test_no_noncombat_potion_actions_and_owned_slot_rule(self):
        def rewards(held, *, slots=2, unknown_rows=0):
            boundary = Boundary("rewards", slots=slots)
            for slot, name in enumerate(held):
                boundary.potion(slot, name)
            boundary.action("use_potion", group="potions", row=0)
            potion = boundary.choose("potion", potion_kind="block_potion")
            proceed = boundary.choose("proceed")
            return boundary, potion, proceed

        boundary, potion, proceed = rewards(["fire_potion", "fire_potion"])
        self.assertEqual(self.pick(boundary).action, proceed)  # both owned slots are full
        boundary, potion, _ = rewards(["fire_potion"])
        self.assertEqual(self.pick(boundary).action, potion)  # owned slot 1 is free
        boundary, potion, proceed = rewards([], slots=0)
        self.assertEqual(self.pick(boundary).action, proceed)  # unknown rows are not owned slots
        # Potion verbs are never chosen, even as the lowest legal index or by the fallback.
        boundary = Boundary("map")
        boundary.potion(0, "fire_potion")
        boundary.action("use_potion", group="potions", row=0)
        move = boundary.node("monster")
        self.assertEqual(self.pick(boundary).action, move)
        boundary = Boundary("checkpoint_complete")
        boundary.potion(0, "fire_potion")
        with self.assertRaisesRegex(HeuristicError, "no legal action other than a potion verb"):
            self.pick(boundary)
        boundary = Boundary("map")
        with self.assertRaisesRegex(HeuristicError, "no legal action"):
            self.pick(boundary)


class FallbackTests(HeuristicTestCase):
    def test_fallback_is_declared_on_the_choice(self):
        boundary = Boundary("act2_ready")
        boundary.choose("joust_owner")
        boundary.choose("joust_murderer")
        choice = self.pick(boundary)
        self.assertTrue(choice.fallback)
        self.assertEqual(choice.action, 0)
        self.assertEqual(choice.rule, "fallback:act2_ready")
        boundary = Boundary("rewards")
        boundary.potion(0, "fire_potion")
        boundary.selection("scry", ["strike", "defend"], cancel=False)
        choice = self.pick(boundary)
        self.assertTrue(choice.fallback)
        self.assertEqual(choice.rule, "fallback:selection:scry")
        self.assertEqual(choice.action, boundary.legal[1])


if __name__ == "__main__":
    unittest.main()
