"""The Gym's combat entry-resource descriptor (``spire-lab-combat-entry-resources-v1``).

The descriptor lists, per character, the relics and potions a ``standard-combat-v1`` entry may own,
each relic's counter and companion (bottle card) rules, and the potion-slot counts. The committed
snapshot (``configs/neural/standard/entry-resources-gym-1.2.0.json``) is the Gym's own descriptor;
its fingerprint is re-derived on load and a Gym-gated test compares it with the live one. The
extraction and the fuzz generator read their per-relic rules here for Gym v1.2.0 tables; the
v1.1.1 series keeps its own constants. Importing this module never imports the Gym.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

SCHEMA_ID = "spire-lab-combat-entry-resources-v1"
FINGERPRINT_PREFIX = b"spire-lab/combat-entry-resources/v1\0"
PROFILE_ID = "standard-combat-v1"
CATALOG_SCHEMA_ID = "spire-gym-standard-combat-catalog-v1"
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[3] / "configs/neural/standard/entry-resources-gym-1.2.0.json"


def canonical_json(value: object) -> bytes:
    """UTF-8, recursively sorted keys, preserved array order, no extra whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def descriptor_fingerprint(descriptor: Mapping[str, Any]) -> str:
    """Lowercase hex SHA-256 over the prefix and the canonical JSON of every other field."""
    body = {key: value for key, value in descriptor.items() if key != "fingerprint"}
    return hashlib.sha256(FINGERPRINT_PREFIX + canonical_json(body)).hexdigest()


@dataclass(frozen=True)
class CounterRule:
    rule: str  # "optional" or "required"
    values: tuple[tuple[int, int], ...]  # inclusive [minimum, maximum] intervals

    def admits(self, value: int) -> bool:
        return any(low <= value <= high for low, high in self.values)

    def first_value(self) -> int:
        return self.values[0][0]


@dataclass(frozen=True)
class EntryResources:
    """The descriptor's admission, counter, companion and potion-slot rules."""

    fingerprint: str
    document: Mapping[str, Any]
    relics_by_character: Mapping[str, tuple[str, ...]]
    potions_by_character: Mapping[str, tuple[str, ...]]
    counters: Mapping[str, CounterRule]  # relic id -> rule; relics absent from it forbid a counter
    companions: Mapping[str, str]  # bottle relic id -> card type of the bottled card
    exclusive_with: Mapping[str, tuple[str, ...]]
    permanent_cards: frozenset[str]

    def admits_relic(self, character: str, relic: str) -> bool:
        return relic in self.relics_by_character.get(character, ())

    def admits_potion(self, character: str, potion: str) -> bool:
        return potion in self.potions_by_character.get(character, ())

    def counter_required_relics(self) -> tuple[str, ...]:
        """Relics whose counter key the entry must supply explicitly."""
        return tuple(sorted(r for r, rule in self.counters.items() if rule.rule == "required"))

    def bottle_relics(self) -> tuple[str, ...]:
        """Relics owning a companion bottled card."""
        return tuple(sorted(self.companions))

    def defaulted_counter_relics(self, recorded: Collection[str]) -> tuple[str, ...]:
        """Relics with an optional counter that ``recorded`` (those a run file lets us set) omits,
        so the Gym fills in its default."""
        return tuple(sorted(r for r, rule in self.counters.items()
                            if rule.rule == "optional" and r not in recorded))

    def potion_slots(self, relics: Sequence[str], ascension: int) -> int:
        """The slot count a fresh entry uses: the ascension's base count, plus the Belt's extra slots
        when the Belt is owned. It must be one of the counts the descriptor allows for that case."""
        slots = self.document["potion_slots"]
        row = slots["allowed"][ascension]
        if row["ascension"] != ascension:
            raise ValueError("potion_slots.allowed is not indexed by ascension")
        belt = slots["potion_belt"] in relics
        count = slots["base_by_ascension"][ascension] + (slots["belt_extra_slots"] if belt else 0)
        if count not in row["with_potion_belt" if belt else "without_potion_belt"]:
            raise ValueError("the descriptor's base slots and its allowed slots disagree")
        return count


def parse_entry_resources(document: Mapping[str, Any]) -> EntryResources:
    if document.get("schema_id") != SCHEMA_ID:
        raise ValueError("unexpected entry-resources schema")
    if document.get("profile_id") != PROFILE_ID or document.get("catalog_schema_id") != CATALOG_SCHEMA_ID:
        raise ValueError("entry-resources descriptor is not for standard-combat-v1")
    fingerprint = document.get("fingerprint")
    if fingerprint != descriptor_fingerprint(document):
        raise ValueError("entry-resources fingerprint does not match its contents")
    characters = {row["character"]: row for row in document["characters"]}
    relics = {row["id"]: row for row in document["relics"]}
    for name, row in characters.items():
        unknown = sorted(set(row["relics"]) - set(relics))
        if unknown:
            raise ValueError(f"{name} admits relics without a rule row: {unknown[:5]}")
    counters = {relic: CounterRule(row["counter"]["rule"],
                                   tuple((int(low), int(high)) for low, high in row["counter"]["values"]))
                for relic, row in relics.items() if row["counter"] is not None}
    companions = {relic: row["companion"]["card_type"]
                  for relic, row in relics.items() if row["companion"] is not None}
    return EntryResources(
        fingerprint=fingerprint, document=document,
        relics_by_character={c: tuple(row["relics"]) for c, row in characters.items()},
        potions_by_character={c: tuple(row["potions"]) for c, row in characters.items()},
        counters=counters, companions=companions,
        exclusive_with={r: tuple(row["exclusive_with"]) for r, row in relics.items() if row["exclusive_with"]},
        permanent_cards=frozenset(row["card"] for row in document["bounds"]["cards"]))


def load_entry_resources(path: Path = DEFAULT_SNAPSHOT) -> EntryResources:
    """Load a committed descriptor snapshot; a changed or mislabelled file fails closed."""
    return parse_entry_resources(json.loads(Path(path).read_bytes()))


def gym_entry_resources() -> dict[str, Any]:
    """The live descriptor (needs the pinned runtime)."""
    from spire_lab.combat_entry_resources import load
    return load()
