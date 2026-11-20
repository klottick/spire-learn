"""Run-file names to Spire Gym Standard ids.

The alias table (``configs/neural/standard/name-aliases.json``) holds name
pairs (run-file name to Gym id) plus the Gym's card ids, types and upgrade limits, copied
from the Gym card metadata named by ``gym_card_metadata_fingerprint``. Card names that equal a Gym id after
normalization need no entry; the table lists the ones that differ. Relic and
potion pairs are explicit per character. The module never imports the Gym at
import time; ``gym_card_metadata`` and ``gym_rejections`` import it on call.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA = "spire-learn-standard-name-aliases-v1"
CHARACTERS = ("ironclad", "silent", "defect", "watcher")
_STARTER_RELIC = {"ironclad": "burning_blood", "silent": "ring_of_the_snake",
                  "defect": "cracked_core", "watcher": "pure_water"}
# Boss relics that replace the starter relic. Their bottled/deck probes use one
# attack, one skill and one power per character.
_UPGRADED_STARTERS = frozenset({"black_blood", "ring_of_the_serpent", "frozen_core", "holy_water"})
_DECK = {"ironclad": ("strike", "defend", "inflame"), "silent": ("strike_g", "defend_g", "footwork"),
         "defect": ("strike_b", "defend_b", "defragment"), "watcher": ("strike_p", "defend_p", "mental_fortress")}
_CARD_NAME = re.compile(r"(.*?)(?:\+(\d+))?")


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def split_card(name: str) -> tuple[str, int]:
    """'Searing Blow+3' -> ('Searing Blow', 3)."""
    match = _CARD_NAME.fullmatch(name)
    return match.group(1), int(match.group(2) or 0)


@dataclass(frozen=True)
class NameAliases:
    """Lookup of run-file names; unknown names return None."""

    sha256: str
    card_ids: frozenset[str]
    card_types: Mapping[str, str]  # Gym card id -> attack/skill/power/status/curse (Gym card metadata)
    card_max_upgrades: Mapping[str, int | None]  # Gym card id -> max upgrades (None: unlimited, Searing Blow)
    relics_by_character: Mapping[str, Mapping[str, str]]
    potions_by_character: Mapping[str, Mapping[str, str]]
    _cards: Mapping[str, str]
    _relics: Mapping[str, str]
    _potions: Mapping[str, str]

    def card(self, run_name: str) -> tuple[str, int] | None:
        name, upgrades = split_card(run_name)
        card = self._cards.get(normalize(name))
        return None if card is None else (card, upgrades)

    def card_type(self, card_id: str) -> str | None:
        return self.card_types.get(card_id)

    def max_upgrades(self, card_id: str) -> int | None:
        """The Gym's upgrade limit for ``card_id``; None when unlimited or unknown."""
        return self.card_max_upgrades.get(card_id)

    def relic(self, run_name: str) -> str | None:
        return self._relics.get(normalize(run_name))

    def potion(self, run_name: str) -> str | None:
        return self._potions.get(normalize(run_name))


def _flatten(by_character: Mapping[str, Mapping[str, str]], kind: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    for character, pairs in by_character.items():
        for name, target in pairs.items():
            key = normalize(name)
            if merged.setdefault(key, target) != target:
                raise ValueError(f"{kind} {name!r} maps to two ids ({character})")
    return merged


def load_aliases(path: Path) -> NameAliases:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("schema") != SCHEMA:
        raise ValueError("unexpected name alias schema")
    card_ids = frozenset(document["gym_card_ids"])
    cards: dict[str, str] = {}
    for card in sorted(card_ids):
        if cards.setdefault(normalize(card), card) != card:
            raise ValueError(f"card id {card!r} collides after normalization")
    for name, target in document["cards"].items():
        if target not in card_ids:
            raise ValueError(f"alias target {target!r} is not a Gym card id")
        cards[normalize(name)] = target
    card_types = dict(document["gym_card_types"])
    card_max_upgrades = dict(document["gym_card_max_upgrades"])
    if set(card_types) != card_ids or set(card_max_upgrades) != card_ids:
        raise ValueError("gym_card_types and gym_card_max_upgrades must cover exactly the Gym card ids")
    for kind in ("relics", "potions"):
        if set(document[kind]) - set(CHARACTERS):
            raise ValueError(f"unknown character in {kind}")
    return NameAliases(
        sha256=hashlib.sha256(raw).hexdigest(), card_ids=card_ids, card_types=card_types,
        card_max_upgrades=card_max_upgrades,
        relics_by_character=document["relics"], potions_by_character=document["potions"],
        _cards=cards, _relics=_flatten(document["relics"], "relic"),
        _potions=_flatten(document["potions"], "potion"))


def gym_card_metadata() -> dict[str, Any]:
    """The Gym's static card metadata (needs the pinned runtime)."""
    from spire_lab import card_metadata
    return card_metadata.load()


def gym_rejections(aliases: NameAliases) -> list[list[str]]:
    """[character, kind, id, reason] for every relic/potion target a one-entry
    Standard combat catalog for that character rejects (needs the pinned runtime)."""
    from spire_lab.gymnasium import StandardCombatEnv

    def reason(character: str, relic: str | None, potion: str | None) -> str | None:
        starter = _STARTER_RELIC[character]
        if relic is None or relic == starter:
            relics = [starter]
        elif relic in _UPGRADED_STARTERS:
            relics = [relic]
        else:
            relics = [starter, relic]
        deck = [{"card": card, "upgrades": 0, "misc": 0} for card in _DECK[character]]
        entry: dict[str, Any] = {
            "id": "names", "character": character, "ascension": 0, "encounter": "cultist",
            "hp": 70, "max_hp": 80, "gold": 99, "deck": deck, "relics": relics, "relic_counters": {},
            "potions": ([potion] if potion else []) + [None] * 5}
        entry["potions"] = entry["potions"][:5 if "potion_belt" in relics else 3]
        if "neows_lament" in relics:
            entry["relic_counters"] = {"neows_lament": 3}
        bottles = {name: {"card": _DECK[character][index], "upgrades": 0, "misc": 0}
                   for index, name in enumerate(("bottled_flame", "bottled_lightning", "bottled_tornado"))
                   if name in relics}
        if bottles:
            entry["bottled_cards"] = bottles
        catalog = {"catalog_schema_id": "spire-gym-standard-combat-catalog-v1",
                   "profile_id": "standard-combat-v1", "entries": [entry]}
        try:
            StandardCombatEnv(catalog).close()
        except Exception as error:  # noqa: BLE001 - report the Gym's rejection
            return f"{type(error).__name__}: {error}"[:300]
        return None

    failures: list[list[str]] = []
    for character in CHARACTERS:
        for relic in sorted(set(aliases.relics_by_character.get(character, {}).values())):
            why = reason(character, relic, None)
            if why:
                failures.append([character, "relic", relic, why])
        for potion in sorted(set(aliases.potions_by_character.get(character, {}).values())):
            why = reason(character, None, potion)
            if why:
                failures.append([character, "potion", potion, why])
    return failures
