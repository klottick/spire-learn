"""noncombat-heuristic-v1: the pre-registered, deterministic non-combat player of the full-run harness (stdlib).

It plays every non-combat boundary of a Standard Run from the public Run observation and legal
action set alone (no seed, RNG state, draw order or record). Every ``room`` and every
``selection_operation`` value of the Run descriptor has a declared rule (``ROOM_RULES``,
``SELECTION_RULES``); a descriptor with a value outside them is refused at construction. Ties go
to the lowest legal index, and ``use_potion`` and ``discard_potion`` are never played.

Rules (Task 4 brief, verbatim values):

- Card selections (any legal ``select``/``confirm``/``cancel``), by operation family: confirm as soon
  as it is legal. Remove, transform or toke: a curse, then strike, then defend, then the lowest-listed
  card (an unlisted card ranks below every listed one). Upgrade, duplicate or bottle: the highest-listed
  eligible card, or the lowest legal ``select`` if none is listed. Never cancel while a ``select`` or
  ``confirm`` is legal.
- Rewards, treasure and boss treasure: gold, then the relic, then the card reward, then a potion if an
  owned slot is free (``potions_known == 1`` and ``potions_occupied == 0``), then open a chest, then
  proceed. Keys are never taken.
- Map: rest (HP < 55%), treasure, monster, question or event, shop, elite, boss; an elite drops below
  shop when HP < 70%. Other kinds rank last.
- Card reward: the highest-listed card; otherwise Singing Bowl, skip or leave.
- Shop: at most one purge (when the deck holds a removable curse, a strike or a defend) and at most
  one affordable listed card per room, then leave; ``RoomFlags`` keeps both flags per (act, floor).
- Campfire: rest when HP < 50%, otherwise smith, otherwise rest; Recall, Lift, Toke and Dig only when
  nothing else is legal.
- Events: Knowing Skull options 0-2 are always excluded, Scrap Ooze option 0 when HP < 50%, every option
  that gains a curse (``event_constants``) and, while HP < 50%, every option with HP or max-HP loss
  (``param_semantics``). Take the first remaining option; if none remains, the non-lethal option with
  the smallest HP plus max-HP loss (ties: the highest ``event_option``); if all are lethal, the smallest
  loss anyway.
- Neow: the first blessing without an HP or max-HP cost. Boss relics: the first listed.
- Transition rooms: the single non-potion action, else ``proceed``, else ``act3_continue`` (the Act 3 ending
  without keys); a key is never taken.
- Keys are never taken: events and the fallback skip a key choice while anything else is legal.
- Anything else: the declared fallback, the lowest legal index that is not a potion verb, marked
  ``fallback`` on the ``Choice`` so the harness counts and reports it. No legal action, or only potion
  verbs, raises ``HeuristicError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

HEURISTIC_ID = "noncombat-heuristic-v1"
PRIORITY_SCHEMA = "spire-learn-card-priority-v1"

POTION_VERBS = ("use_potion", "discard_potion")
SELECTION_VERBS = ("select", "confirm", "cancel")
# Key choices (the Ruby key is Recall, a campfire choice): keys are skipped, so runs end at Act 3.
KEY_CHOICES = ("sapphire", "emerald", "ruby")

REST_HP = 0.55       # map: rest ranks first below this HP share
ELITE_HP = 0.70      # map: an elite drops below shop below this HP share
CAMPFIRE_REST_HP = 0.50
EVENT_HARM_HP = 0.50
SCRAP_OOZE_HP = 0.50

# Every Standard room value (descriptor vocabulary ``room``) and its rule.
ROOM_RULES = {
    "padding": "transition", "act2_prefix_complete": "transition", "act2_ready": "transition",
    "act3_boss_complete": "transition", "act3_ending": "transition", "act3_prefix_complete": "transition",
    "act3_ready": "transition", "act4_ready": "transition", "boss_relic_child": "rewards",
    "boss_relic_choice": "boss_relic", "boss_treasure": "rewards", "campfire": "campfire",
    "card_reward": "card_reward", "checkpoint_complete": "transition", "combat": "transition",
    "event": "event", "heart_victory": "transition", "map": "map", "neow": "neow",
    "proceed_without_relic": "transition", "rewards": "rewards", "shop": "shop", "treasure": "rewards",
    "true_victory": "transition",
}
# Every Standard selection operation (descriptor vocabulary ``selection_operation``) and its family.
# ``combat`` operations belong to combat selections; outside combat they take the counted fallback,
# as does ``select_permanent``, a Run selection without a named purpose.
SELECTION_RULES = {
    "padding": "fallback", "select_permanent": "fallback",
    "ancient_elegance": "remove", "campfire_toke": "remove", "empty_cage": "remove", "event_choose": "remove",
    "event_remove": "remove", "event_transform": "remove", "neow_remove": "remove", "neow_transform": "remove",
    "shop_purge": "remove", "shrine_remove": "remove",
    "event_duplicate": "upgrade", "event_upgrade": "upgrade", "neow_upgrade": "upgrade", "relic_equip": "upgrade",
    "shrine_upgrade": "upgrade", "smith": "upgrade",
    **dict.fromkeys((
        "bottomdeck", "copy_to_hand_1", "copy_to_hand_2", "copy_to_hand_3", "hologram_retrieve",
        "meditate_return", "move_discard", "move_draw_top", "move_exhaust", "move_hand", "nightmare_capture",
        "omniscience", "recycle", "retain", "retrieve_memories", "return_from_exhaust", "scry",
        "seek_retrieve", "setup", "topdeck", "upgrade",
    ), "combat"),
}
MAP_ORDER = ("rest", "treasure", "monster", "question", "shop", "elite", "boss")
REWARD_ORDER = ("gold", "relic", "card_reward", "potion", "open_chest", "proceed")
CARD_REWARD_PASS = ("singing_bowl", "skip", "leave")
CAMPFIRE_OTHER = ("recall", "lift", "toke", "dig")
# The Act 3 ending without keys is Proceed, then act3_continue stages 1-4 (Gym act3_end.rs:100-112); keys are never taken.
TRANSITION_CHOICES = ("proceed", "act3_continue")
HP_LOSS = frozenset({"damage_taken", "hp_lost"})
MAX_HP_LOSS = frozenset({"max_hp_lost"})
NEOW_HP_COSTS = frozenset({"max_hp_loss", "current_hp_loss"})
KNOWING_SKULL_EXCLUDED = frozenset({0, 1, 2})


class HeuristicError(ValueError):
    """The heuristic cannot act: an unknown descriptor kind or a boundary with no admissible action."""


@dataclass(frozen=True)
class Choice:
    """One move: the legal ``action``, the rule that chose it and whether it is the declared fallback."""

    action: int
    rule: str
    fallback: bool = False


@dataclass(frozen=True)
class CardPriority:
    """The committed card priority list (best first), its curses and its file sha256."""

    id: str
    character: str
    priority: tuple[str, ...]
    avoid: tuple[str, ...]
    curses: frozenset[str]
    unremovable_curses: frozenset[str]
    sha256: str
    _ranks: Mapping[str, int] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, path: Path | str) -> "CardPriority":
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        if document.get("schema") != PRIORITY_SCHEMA:
            raise HeuristicError(f"{path} is not a {PRIORITY_SCHEMA} card priority list")
        priority, avoid = tuple(document["priority"]), tuple(document["avoid"])
        curses, unremovable = frozenset(document["curses"]), frozenset(document["unremovable_curses"])
        if len(set(priority)) != len(priority) or set(priority) & set(avoid) or curses & set(priority):
            raise HeuristicError("the priority list repeats a card or lists an avoided card or a curse")
        if not unremovable <= curses:
            raise HeuristicError("unremovable curses must be curses")
        return cls(id=document["id"], character=document["character"], priority=priority, avoid=avoid,
                   curses=curses, unremovable_curses=unremovable, sha256=hashlib.sha256(raw).hexdigest(),
                   _ranks={card: rank for rank, card in enumerate(priority)})

    def rank(self, card: str | None) -> int | None:
        """0 for the best card; None when unlisted."""
        return self._ranks.get(card)

    def names(self) -> frozenset[str]:
        return frozenset(self.priority) | frozenset(self.avoid) | self.curses


class RoomFlags:
    """Shop flags per room, keyed by (act, floor): whether this shop has purged and whether it has bought."""

    def __init__(self):
        self._rooms: dict[tuple[int, int], dict[str, bool]] = {}

    def get(self, key: tuple[int, int]) -> dict[str, bool]:
        return dict(self._rooms.get(key, {"purged": False, "bought": False}))

    def record(self, key: tuple[int, int], choice: Choice) -> None:
        if choice.rule in ("shop:purge", "shop:buy"):
            flags = self._rooms.setdefault(key, {"purged": False, "bought": False})
            flags["purged" if choice.rule == "shop:purge" else "bought"] = True


def _rows(array) -> int:
    return len(array)


class NoncombatHeuristic:
    """``choose(observation, legal, flags)``: one ``Choice`` for a non-combat Run boundary."""

    def __init__(self, descriptor: Mapping, priority: CardPriority):
        tensors = {tensor["key"]: tensor for tensor in descriptor["tensors"]}
        self._fields = {key: list(tensor.get("fields") or ()) for key, tensor in tensors.items()}
        vocabularies = descriptor["vocabularies"]
        for name, table in (("room", ROOM_RULES), ("selection_operation", SELECTION_RULES)):
            unknown = [value for value in vocabularies[name] if value not in table]
            if unknown:
                raise HeuristicError(f"no {HEURISTIC_ID} rule for {name} values {unknown}")
        cards = set(vocabularies["card"])
        missing = sorted(priority.names() - cards)
        if missing:
            raise HeuristicError(f"priority list cards are not in the descriptor's card vocabulary: {missing}")
        self.priority = priority
        self.vocab = {name: tuple(values) for name, values in vocabularies.items()}
        semantics = descriptor.get("semantics", {})
        self._param_losses: dict[tuple[str, int, int], list[tuple[int, str]]] = {}
        for row in semantics.get("param_semantics", ()):
            if row["meaning"] in HP_LOSS | MAX_HP_LOSS:
                self._param_losses.setdefault((row["event"], row["stage"], row["option"]), []).append(
                    (row["index"], row["meaning"]))
        self._curse_options = frozenset(
            (row["event"], row["stage"], row["option"])
            for row in semantics.get("event_constants", {}).get("rows", ())
            if row.get("kind") == "card" and row.get("identity") in priority.curses
            and row.get("stage") is not None and row.get("option") is not None)

    # --- observation access --------------------------------------------------------------

    def _column(self, key: str, field_name: str) -> int:
        return self._fields[key].index(field_name)

    def _scalar(self, observation, key: str, field_name: str) -> int | None:
        column = self._column(key, field_name)
        return int(observation[key][column]) if observation[f"{key}_known"][column] else None

    def _cell(self, observation, key: str, row: int, field_name: str) -> int | None:
        column = self._column(key, field_name)
        return int(observation[key][row][column]) if observation[f"{key}_known"][row][column] else None

    def _name(self, vocabulary: str, value: int | None) -> str | None:
        return None if value is None else self.vocab[vocabulary][value]

    def room_key(self, observation) -> tuple[int, int]:
        return (self._scalar(observation, "context", "act") or 0, self._scalar(observation, "context", "floor") or 0)

    def _hp_share(self, observation) -> float:
        hp, max_hp = self._scalar(observation, "player", "hp"), self._scalar(observation, "player", "max_hp")
        if hp is None or not max_hp:
            raise HeuristicError("the Run boundary does not publish the player's HP and max HP")
        return hp / max_hp

    def free_potion_slot(self, observation) -> bool:
        """An owned slot with nothing in it: ``potions_known[:, 0] == 1`` and ``potions_occupied == 0``."""
        known, occupied = observation["potions_known"], observation["potions_occupied"]
        return any(known[row][0] == 1 and occupied[row] == 0 for row in range(_rows(occupied)))

    def _deck_card(self, observation, row: int) -> str | None:
        if row < 0 or not observation["permanent_cards_occupied"][row]:
            return None
        return self._name("card", self._cell(observation, "permanent_cards", row, "card_kind"))

    def _deck(self, observation) -> list[str | None]:
        occupied = observation["permanent_cards_occupied"]
        return [self._deck_card(observation, row) for row in range(_rows(occupied)) if occupied[row]]

    def _actions(self, observation, legal: Sequence[int]) -> list[dict]:
        actions = []
        for index in legal:
            verb = self._name("verb", self._cell(observation, "actions", index, "verb"))
            group = self._name("row_group", self._cell(observation, "actions", index, "source_group"))
            row = self._cell(observation, "actions", index, "source_row")
            item = {"index": int(index), "verb": verb, "group": group, "row": -1 if row is None else row}
            if verb == "choose" and group == "offers" and item["row"] >= 0:
                item["choice"] = self._name("choice", self._cell(observation, "offers", item["row"], "choice_kind"))
            else:
                item["choice"] = None
            actions.append(item)
        return actions

    # --- choice ---------------------------------------------------------------------------

    def choose(self, observation, legal: Sequence[int], flags: RoomFlags | None = None) -> Choice:
        legal = sorted(int(index) for index in legal)
        if not legal:
            raise HeuristicError("the non-combat boundary has no legal action")
        actions = [item for item in self._actions(observation, legal) if item["verb"] not in POTION_VERBS]
        if not actions:
            raise HeuristicError("the non-combat boundary has no legal action other than a potion verb")
        if any(item["verb"] in SELECTION_VERBS for item in actions):
            operation = self._name("selection_operation",
                                   self._scalar(observation, "selection_counts", "operation")) or "padding"
            return self._selection(observation, actions, operation)
        room = self._name("room", self._scalar(observation, "context", "room_kind")) or "padding"
        rule = ROOM_RULES[room]
        choice = getattr(self, f"_room_{rule}")(observation, actions, flags)
        return choice if choice is not None else self._fallback(actions, room)

    @staticmethod
    def _fallback(actions, where: str) -> Choice:
        """The lowest legal non-potion index; a key only when nothing else is legal."""
        keyless = [item for item in actions if item["choice"] not in KEY_CHOICES] or actions
        return Choice(keyless[0]["index"], f"fallback:{where}", fallback=True)

    @staticmethod
    def _first(actions, choice: str):
        for item in actions:
            if item["choice"] == choice:
                return item
        return None

    def _selection(self, observation, actions, operation: str) -> Choice:
        family = SELECTION_RULES[operation]
        if family in ("fallback", "combat"):
            return self._fallback(actions, f"selection:{operation}")
        for item in actions:
            if item["verb"] == "confirm":
                return Choice(item["index"], "selection:confirm")
        candidates = []
        for item in actions:
            if item["verb"] != "select" or item["group"] != "selection" or item["row"] < 0:
                continue
            row = item["row"]
            if self._cell(observation, "selection", row, "selected") == 1:
                continue
            if self._cell(observation, "selection", row, "eligible") == 0:
                continue
            source = self._name("row_group", self._cell(observation, "selection", row, "source_group"))
            card_row = self._cell(observation, "selection", row, "source_row")
            card = self._deck_card(observation, card_row) if source == "permanent_cards" and card_row is not None else None
            candidates.append((item["index"], card))
        if not candidates:
            selects = [item for item in actions if item["verb"] == "select"]
            if selects:
                return self._fallback(selects, f"selection:{operation}")
            return Choice(actions[0]["index"], "selection:only-cancel")
        if family == "remove":
            index, _ = min(candidates, key=lambda pair: (self._removal_key(pair[1]), pair[0]))
            return Choice(index, f"selection:{family}")
        listed = [(self.priority.rank(card), index) for index, card in candidates if self.priority.rank(card) is not None]
        index = min(listed)[1] if listed else candidates[0][0]
        return Choice(index, f"selection:{family}")

    def _removal_key(self, card: str | None) -> tuple[int, int]:
        """Curse, then strike, then defend, then unlisted, then the lowest-listed first."""
        if card in self.priority.curses:
            return (0, 0)
        if card == "strike":
            return (1, 0)
        if card == "defend":
            return (2, 0)
        rank = self.priority.rank(card)
        if rank is None:
            return (3, 0)
        return (4, len(self.priority.priority) - rank)

    def _room_transition(self, observation, actions, flags) -> Choice | None:
        if len(actions) == 1:
            return Choice(actions[0]["index"], "transition")
        for choice in TRANSITION_CHOICES:
            item = self._first(actions, choice)
            if item is not None:
                return Choice(item["index"], "transition")
        return None

    def _room_rewards(self, observation, actions, flags) -> Choice | None:
        free_slot = self.free_potion_slot(observation)
        for choice in REWARD_ORDER:
            if choice == "potion" and not free_slot:
                continue
            item = self._first(actions, choice)
            if item is not None:
                return Choice(item["index"], f"rewards:{choice}")
        return None

    def _room_card_reward(self, observation, actions, flags) -> Choice | None:
        listed = []
        for item in actions:
            if item["choice"] == "card":
                rank = self.priority.rank(self._name("card", self._cell(observation, "offers", item["row"], "card_kind")))
                if rank is not None:
                    listed.append((rank, item["index"]))
        if listed:
            return Choice(min(listed)[1], "card_reward:card")
        for choice in CARD_REWARD_PASS:
            item = self._first(actions, choice)
            if item is not None:
                return Choice(item["index"], f"card_reward:{choice}")
        return None

    def _room_map(self, observation, actions, flags) -> Choice | None:
        moves = [item for item in actions if item["verb"] == "move" and item["group"] == "map_nodes" and item["row"] >= 0]
        if not moves:
            return self._room_transition(observation, actions, flags)
        share = self._hp_share(observation)
        order = list(MAP_ORDER)
        if share >= REST_HP:
            order.remove("rest")
        if share < ELITE_HP:
            order.remove("elite")
            order.insert(order.index("shop") + 1, "elite")

        def score(item):
            kind = self._name("map_kind", self._cell(observation, "map_nodes", item["row"], "kind"))
            kind = "question" if kind == "event" else kind
            return (order.index(kind) if kind in order else len(order), item["index"])

        return Choice(min(moves, key=score)["index"], "map")

    def _room_campfire(self, observation, actions, flags) -> Choice | None:
        rest, smith = self._first(actions, "rest"), self._first(actions, "smith")
        if rest is not None and self._hp_share(observation) < CAMPFIRE_REST_HP:
            return Choice(rest["index"], "campfire:rest")
        if smith is not None:
            return Choice(smith["index"], "campfire:smith")
        if rest is not None:
            return Choice(rest["index"], "campfire:rest")
        others = [item for item in actions if item["choice"] in CAMPFIRE_OTHER]
        if others:
            return Choice(others[0]["index"], "campfire:only-other")
        return self._room_transition(observation, actions, flags)

    def _room_shop(self, observation, actions, flags) -> Choice | None:
        state = (flags or RoomFlags()).get(self.room_key(observation))
        gold = self._scalar(observation, "player", "gold") or 0

        def affordable(item):
            price = self._cell(observation, "offers", item["row"], "price")
            return price is None or price <= gold

        purge = self._first(actions, "purge")
        if not state["purged"] and purge is not None and affordable(purge):
            purgeable = self.priority.curses - self.priority.unremovable_curses | {"strike", "defend"}
            if any(card in purgeable for card in self._deck(observation)):
                return Choice(purge["index"], "shop:purge")
        if not state["bought"]:
            listed = []
            for item in actions:
                if item["choice"] == "card" and affordable(item):
                    rank = self.priority.rank(self._name("card", self._cell(observation, "offers", item["row"], "card_kind")))
                    if rank is not None:
                        listed.append((rank, item["index"]))
            if listed:
                return Choice(min(listed)[1], "shop:buy")
        leave = self._first(actions, "leave")
        return Choice(leave["index"], "shop:leave") if leave is not None else None

    def _room_neow(self, observation, actions, flags) -> Choice | None:
        blessings = [item for item in actions if item["choice"] == "neow_blessing"]
        for item in blessings:
            drawback = self._name("neow_drawback", self._cell(observation, "offers", item["row"], "neow_drawback"))
            if drawback not in NEOW_HP_COSTS:
                return Choice(item["index"], "neow")
        if blessings:
            return None
        return self._room_transition(observation, actions, flags)

    def _room_boss_relic(self, observation, actions, flags) -> Choice | None:
        relic = self._first(actions, "relic")
        if relic is not None:
            return Choice(relic["index"], "boss_relic")
        return self._room_transition(observation, actions, flags)

    def _option_cost(self, observation, event: str | None, item) -> tuple[int, int, bool, int]:
        """(HP loss, max-HP loss, gains a curse, event_option) of one event option."""
        row = item["row"]
        if item["verb"] != "choose" or item["group"] != "offers" or row < 0:
            return 0, 0, False, -1
        stage = self._cell(observation, "offers", row, "event_stage")
        option = self._cell(observation, "offers", row, "event_option")
        if stage is None or option is None:
            return 0, 0, False, -1
        hp_loss = max_hp_loss = 0
        for index, meaning in self._param_losses.get((event, stage, option), ()):
            value = self._cell(observation, "offers", row, f"param{index}")
            if value is not None and value > 0:
                if meaning in MAX_HP_LOSS:
                    max_hp_loss += value
                else:
                    hp_loss += value
        return hp_loss, max_hp_loss, (event, stage, option) in self._curse_options, option

    def _room_event(self, observation, actions, flags) -> Choice | None:
        event = self._name("event", self._scalar(observation, "context", "event_kind"))
        hp, max_hp = self._scalar(observation, "player", "hp") or 0, self._scalar(observation, "player", "max_hp") or 0
        share = self._hp_share(observation)
        options = []
        for item in [item for item in actions if item["choice"] not in KEY_CHOICES] or actions:
            hp_loss, max_hp_loss, curse, option = self._option_cost(observation, event, item)
            excluded = (curse
                        or (event == "knowing_skull" and option in KNOWING_SKULL_EXCLUDED)
                        or (event == "scrap_ooze" and option == 0 and share < SCRAP_OOZE_HP)
                        or (share < EVENT_HARM_HP and (hp_loss > 0 or max_hp_loss > 0)))
            if not excluded:
                return Choice(item["index"], "event:first")
            lethal = hp_loss >= hp or max_hp_loss >= max_hp
            options.append((lethal, hp_loss + max_hp_loss, -option, item["index"]))
        lethal, *_, index = min(options)
        return Choice(index, "event:least-loss-lethal" if lethal else "event:least-loss")


__all__ = [
    "CAMPFIRE_OTHER", "CardPriority", "Choice", "HEURISTIC_ID", "HeuristicError", "KEY_CHOICES", "MAP_ORDER",
    "NoncombatHeuristic", "POTION_VERBS", "REWARD_ORDER", "ROOM_RULES", "RoomFlags", "SELECTION_RULES",
]
