"""Source-bound finite Gym66 input support; no simulator or runtime imports.

Rarity/mechanic annotations are learner sampling heuristics. Source provenance
does not qualify an installed extension or establish original-game parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .cases import _freeze_json, _reject_json_constant, _unique_object, _validate_entry
from .config import ACCEPTED_GYM_REVISION


def json_value(value: object) -> object:
    """Return owned ordinary JSON containers from an immutable payload."""
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        # Reuse the existing finite-value/primitive check.
        return _freeze_json(value)
    raise ValueError("payload must be JSON-convertible")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object,
                         parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load scenario JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("scenario JSON root must be an object")
    return raw


def _id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value) is not None


@dataclass(frozen=True)
class ScenarioSupport:
    """An owned immutable support snapshot, authenticated to the pinned inventory."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        raw = json_value(self.payload)
        required = {"schema", "gym_revision", "profile_id", "content_id", "provenance",
                    "cards", "encounters", "relics", "potions"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("support has missing or unknown fields")
        if (raw["schema"] != "spire-learn-scenario-support-v1"
                or raw["gym_revision"] != ACCEPTED_GYM_REVISION
                or raw["profile_id"] != "ironclad-combat-v1"
                or raw["content_id"] != "spire-gym-ironclad-combat-content-v1"):
            raise ValueError("unsupported source-bound support identity")
        for name, count in (("cards", 75), ("encounters", 43), ("relics", 30), ("potions", 13)):
            rows = raw[name]
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError(f"support {name} must contain exactly {count} rows")
            ids = [r.get("id") if isinstance(r, dict) else r for r in rows]
            if not all(_id(i) for i in ids) or len(set(ids)) != count:
                raise ValueError(f"support {name} IDs must be valid and unique")
        for card in raw["cards"]:
            if (not isinstance(card, dict)
                    or set(card) != {"id", "maximum_upgrades", "rarity", "mechanic_tags"}
                    or type(card["maximum_upgrades"]) is not int
                    or card["maximum_upgrades"] != (10 if card["id"] == "searing_blow" else 1)
                    or card["rarity"] not in {"starter", "common", "uncommon", "rare"}
                    or not isinstance(card["mechanic_tags"], list) or not card["mechanic_tags"]
                    or not all(_id(tag) for tag in card["mechanic_tags"])):
                raise ValueError("invalid support card annotation or upgrade bound")
        for encounter in raw["encounters"]:
            if (not isinstance(encounter, dict) or set(encounter) != {"id", "act", "kind"}
                    or type(encounter["act"]) is not int or not 1 <= encounter["act"] <= 4
                    or encounter["kind"] not in {"monster", "elite", "boss"}):
                raise ValueError("invalid support encounter")
        provenance = raw["provenance"]
        if (not isinstance(provenance, dict) or set(provenance) != {
                "repository", "revision", "sources", "support_authority",
                "annotation_semantics", "qualification"}
                or provenance["revision"] != ACCEPTED_GYM_REVISION
                or provenance["repository"] != "https://github.com/klottick/spire-gym"
                or not isinstance(provenance["sources"], list)
                or any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                       or not isinstance(row["path"], str)
                       or re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None
                       for row in provenance["sources"])
                or not all(isinstance(provenance[k], str) and provenance[k] for k in
                           ("support_authority", "annotation_semantics", "qualification"))):
            raise ValueError("invalid support provenance")
        inventory = {"cards": [{k: c[k] for k in ("id", "maximum_upgrades")}
                               for c in raw["cards"]],
                     **{k: raw[k] for k in ("encounters", "relics", "potions")}}
        if content_sha256(inventory) != _INVENTORY_SHA256:
            raise ValueError("support inventory differs from accepted Gym66 public tables")
        if {r["path"]: r["sha256"] for r in provenance["sources"]} != _SOURCE_SHA256:
            raise ValueError("support provenance differs from accepted public sources")
        object.__setattr__(self, "payload", _freeze_json(raw))

    @property
    def cards(self) -> Mapping[str, Mapping[str, object]]:
        return _freeze_json({c["id"]: c for c in self.payload["cards"]})

    @property
    def encounters(self) -> Mapping[str, Mapping[str, object]]:
        return _freeze_json({e["id"]: e for e in self.payload["encounters"]})

    @property
    def relics(self) -> tuple[str, ...]:
        return self.payload["relics"]

    @property
    def potions(self) -> tuple[str, ...]:
        return self.payload["potions"]

    @property
    def sha256(self) -> str:
        return content_sha256(self.payload)

    def validate_entry(self, entry: Mapping[str, object]) -> None:
        """Check public construction inputs; active Lament is recipe-forbidden."""
        raw = json_value(entry)
        if not isinstance(raw, dict):
            raise ValueError("scenario entry must be a mapping")
        _validate_entry(raw, entry_id=raw.get("id"))
        cards, encounters = self.cards, self.encounters
        if raw["encounter"] not in encounters:
            raise ValueError("unsupported encounter")
        for row in raw["deck"]:
            if (row["card"] not in cards
                    or row["upgrades"] > cards[row["card"]]["maximum_upgrades"]):
                raise ValueError("unsupported card or upgrades")
        if any(r not in self.relics for r in raw["relics"]):
            raise ValueError("unsupported relic")
        if {"burning_blood", "black_blood"} <= set(raw["relics"]):
            raise ValueError("Burning Blood and Black Blood are mutually exclusive")
        if any(p is not None and p not in self.potions for p in raw["potions"]):
            raise ValueError("unsupported potion")
        if "neows_lament" in raw["relics"] and raw.get("relic_counters", {}).get("neows_lament") != -2:
            raise ValueError("active Neow's Lament is forbidden by this recipe")


def load_scenario_support(path: Path) -> ScenarioSupport:
    return ScenarioSupport(read_json(path))


# Hashes authenticate only the small public input inventory/source provenance.
# They are not native artifact or gameplay-qualification identities.
_INVENTORY_SHA256 = "dd9a87c27884fca4894e51750d569b0de98da1c849fcaeea9d63b59a3e6a333f"
_SOURCE_SHA256 = {
    "crates/spire-env/src/combat_training.rs": "86a27674646d50e32edab1df6fb14e932efa5aea4cdff6a6503b26025d4de288",
    "crates/spire-core/src/run/training.rs": "12f1eb5e8fa375a08895892bfe55445dabfd9ca9133e21b74479b69ae0dec2ae",
    "crates/spire-core/src/combat/enemies/mod.rs": "b7a744a91d5c624f6f2ac3cb511a85a12a832fe4448cf5d12a88c60b09a8cb54",
    "crates/spire-core/src/content/ids/cards.rs": "c0831dc3d28eeb4074afe4f2edcb6de1cf6a0a1b1451cbcd8ea8f5b2e509ab67",
    "crates/spire-core/src/content/ids/items.rs": "ac5bfcc7ab693aa926c0bcc1572069ac326d255b6d7c172c2fa9c244463d578a",
    "crates/spire-core/src/content/ids/encounters.rs": "b54dfa238b78aaf7eb9816ee0a9e1229446bf652a89f3ec9fb64943224214549",
    "crates/spire-core/src/content/cards.rs": "80e33c65170f22904b985028611edde2d71f33e7d6a89d8bbd93b3511701239b"
}

__all__ = ["ScenarioSupport", "load_scenario_support", "json_value", "canonical_bytes",
           "content_sha256"]
