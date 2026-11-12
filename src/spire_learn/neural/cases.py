"""Strict, authored dual-mode manifests and deterministic seed domains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*\Z")
_ENTRY_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_SPLITS = {"train", "selection", "final"}
_MODES = {"standalone", "run", "standard-standalone", "standard-run"}
_STANDARD_CHARACTERS = {"ironclad", "silent", "defect", "watcher"}
_RUN_PROFILE = "m10-ironclad-policy"

_TRAIN_START = 0
_TRAIN_STOP = 2**40
_SELECTION_START = 2**40
_SELECTION_STOP = 2**48
_FINAL_START = 2**48
_FINAL_STOP = 2**64
_TRAIN_REPLICA_STRIDE = 2**32


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("case metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("case metadata mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("case metadata must contain JSON-compatible values")


@dataclass(frozen=True)
class NeuralCase:
    case_id: str
    lineage_id: str
    mode: str
    environment_seed: int
    cohort: str
    entry_id: str | None
    entry: object | None
    profile_id: str | None
    character: str | None
    ascension: int | None
    creation: object | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "lineage_id", "cohort"):
            _identifier(getattr(self, name), f"NeuralCase {name}")
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            raise ValueError("NeuralCase mode must be standalone, run, standard-standalone, or standard-run")
        if type(self.environment_seed) is not int or self.environment_seed < 0:
            raise ValueError("NeuralCase environment_seed must be a non-negative integer")
        if self.mode in {"standalone", "standard-standalone"}:
            if self.entry_id is None or not isinstance(self.entry_id, str):
                raise ValueError("standalone cases require entry_id")
            if _ENTRY_ID.fullmatch(self.entry_id) is None:
                raise ValueError("entry_id must be 1..64 ASCII alphanumeric, underscore, or hyphen characters")
            if not isinstance(self.entry, Mapping):
                raise ValueError("standalone cases require an entry mapping")
            if self.entry.get("id") != self.entry_id:
                raise ValueError("entry_id must equal the native entry.id")
            if self.mode == "standalone":
                if any(value is not None for value in (self.profile_id, self.character, self.ascension)):
                    raise ValueError("standalone cases cannot declare Run profile/reset fields")
                if self.creation is not None:
                    raise ValueError("standalone cases cannot declare Run creation fields")
            else:
                _validate_standard_identity(self.character, self.ascension)
                if (self.entry.get("character") != self.character
                        or self.entry.get("ascension") != self.ascension):
                    raise ValueError("standard-standalone case identity must match its catalog entry")
                if self.profile_id is not None or self.creation is not None:
                    raise ValueError("standard-standalone cases cannot declare Run profile/creation fields")
                _freeze_json(self.entry)
        else:
            if self.entry_id is not None or self.entry is not None:
                raise ValueError("Run cases cannot declare catalog/resource reset entries")
            if self.mode == "run":
                if (self.profile_id != _RUN_PROFILE or self.character != "ironclad"
                        or type(self.ascension) is not int or self.ascension != 0):
                    raise ValueError("Run cases require the explicit Ironclad A0 m10-ironclad-policy profile")
                if self.creation is not None:
                    raise ValueError("legacy Run cases cannot declare Standard creation fields")
            else:
                _validate_standard_identity(self.character, self.ascension)
                if self.profile_id is not None:
                    raise ValueError("standard-run cases cannot declare a legacy Run profile")
                if self.creation is not None:
                    if not isinstance(self.creation, Mapping) or set(self.creation) != {"neow", "seen_bosses", "note"}:
                        raise ValueError("standard-run creation must contain exactly neow, seen_bosses, and note")
                    note = self.creation["note"]
                    if (not isinstance(note, Mapping) or set(note) != {"card", "upgrades"}
                            or not isinstance(note["card"], str) or not note["card"]
                            or type(note["upgrades"]) is not int or not 0 <= note["upgrades"] <= 73):
                        raise ValueError("standard-run creation note must contain card and upgrades")
                    object.__setattr__(self, "creation", _freeze_json(self.creation))
        if self.entry is not None:
            frozen = _freeze_json(self.entry)
            if not isinstance(frozen, MappingProxyType):
                raise ValueError("standalone entry must be a JSON object")
            object.__setattr__(self, "entry", frozen)


def _validate_standard_identity(character: object, ascension: object) -> None:
    if not isinstance(character, str) or character not in _STANDARD_CHARACTERS:
        raise ValueError("Standard cases require character ironclad, silent, defect, or watcher")
    if type(ascension) is not int or not 0 <= ascension <= 20:
        raise ValueError("Standard cases require ascension in [0, 20]")


@dataclass(frozen=True)
class NeuralCases:
    path: Path
    split: str
    partition: str
    cases: tuple[NeuralCase, ...]
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if (not isinstance(self.split, str) or self.split not in _SPLITS
                or self.partition != self.split):
            raise ValueError("NeuralCases split and partition must be the same supported value")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("NeuralCases requires an immutable non-empty cases tuple")
        if any(not isinstance(case, NeuralCase) for case in self.cases):
            raise ValueError("NeuralCases entries must be NeuralCase records")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("NeuralCases sha256 must be lowercase hexadecimal")

    def for_mode(self, mode: str) -> tuple[NeuralCase, ...]:
        if not isinstance(mode, str) or mode not in _MODES:
            raise ValueError("mode must be standalone or run")
        return tuple(case for case in self.cases if case.mode == mode)


def environment_seed_for(split: str, *, replica: int, ordinal: int) -> int:
    """Allocate one deterministic environment seed without crossing split domains."""
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("split must be train, selection, or final")
    if type(replica) is not int or replica < 0 or type(ordinal) is not int or ordinal < 0:
        raise ValueError("replica and ordinal must be non-negative integers")
    if split == "train":
        replicas = (_TRAIN_STOP - _TRAIN_START) // _TRAIN_REPLICA_STRIDE
        if replica >= replicas or ordinal >= _TRAIN_REPLICA_STRIDE:
            raise ValueError("train replica/ordinal exceeds its environment seed domain")
        return _TRAIN_START + replica * _TRAIN_REPLICA_STRIDE + ordinal
    if replica != 0:
        raise ValueError("selection/final seeds do not use training replica offsets")
    start, stop = ((_SELECTION_START, _SELECTION_STOP) if split == "selection"
                   else (_FINAL_START, _FINAL_STOP))
    if ordinal >= stop - start:
        raise ValueError(f"{split} ordinal exceeds its environment seed domain")
    return start + ordinal


def _validate_seed(split: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("environment_seed must be a non-negative integer")
    if split == "train":
        valid = _TRAIN_START <= value < _TRAIN_STOP
    elif split == "selection":
        valid = _SELECTION_START <= value < _SELECTION_STOP
    else:
        valid = _FINAL_START <= value < _FINAL_STOP
    if not valid:
        raise ValueError(f"environment_seed falls outside the {split} domain")
    return value


def _validate_entry(value: object, *, entry_id: object) -> dict[str, object]:
    required = {"id", "encounter", "hp", "max_hp", "deck", "relics", "potions"}
    optional = {"relic_counters"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - (required | optional)):
        raise ValueError("standalone entry has missing or unknown native fields")
    native_id = value["id"]
    if not isinstance(native_id, str) or _ENTRY_ID.fullmatch(native_id) is None:
        raise ValueError("entry.id must be 1..64 ASCII alphanumeric, underscore, or hyphen characters")
    if not isinstance(entry_id, str) or entry_id != native_id:
        raise ValueError("entry_id must equal the native entry.id")
    encounter = value["encounter"]
    if not isinstance(encounter, str) or not encounter:
        raise ValueError("entry.encounter must be a non-empty string ID")
    deck = value["deck"]
    if not isinstance(deck, list) or not 1 <= len(deck) <= 128:
        raise ValueError("entry.deck must contain 1..128 native card rows")
    for index, card_row in enumerate(deck):
        if not isinstance(card_row, dict) or set(card_row) != {"card", "upgrades"}:
            raise ValueError(f"entry.deck[{index}] must contain exactly card and upgrades")
        card = card_row["card"]
        upgrades = card_row["upgrades"]
        if not isinstance(card, str) or not card:
            raise ValueError(f"entry.deck[{index}].card must be a non-empty string ID")
        if type(upgrades) is not int or not 0 <= upgrades <= 10:
            raise ValueError(f"entry.deck[{index}].upgrades must be a primitive integer in [0, 10]")
    hp, max_hp = value["hp"], value["max_hp"]
    if (type(hp) is not int or type(max_hp) is not int
            or not 1 <= hp <= max_hp <= 999):
        raise ValueError("entry HP must satisfy 1 <= hp <= max_hp <= 999")
    relics = value["relics"]
    if (not isinstance(relics, list)
            or any(not isinstance(relic, str) or not relic for relic in relics)):
        raise ValueError("entry.relics must be a list of non-empty string IDs")
    if len(set(relics)) != len(relics):
        raise ValueError("entry.relics cannot contain duplicate IDs")
    potions = value["potions"]
    if (not isinstance(potions, list) or len(potions) != 3
            or any(potion is not None and (not isinstance(potion, str) or not potion)
                   for potion in potions)):
        raise ValueError("entry.potions must contain exactly three empty or non-empty string ID slots")
    if "relic_counters" in value:
        counters = value["relic_counters"]
        if not isinstance(counters, dict) or counters.keys() - {"neows_lament"}:
            raise ValueError("entry.relic_counters may contain only neows_lament")
        if "neows_lament" in counters and "neows_lament" not in relics:
            raise ValueError("entry.relic_counters requires neows_lament to be equipped")
        if any(type(counter) is not int or counter not in {-2, 1, 2, 3}
               for counter in counters.values()):
            raise ValueError("neows_lament counter must be one of -2, 1, 2, or 3")
    return value


def _validate_standard_entry(value: object) -> dict[str, object]:
    required = {"id", "character", "ascension", "encounter", "hp", "max_hp", "gold",
                "deck", "relics", "potions"}
    optional = {"relic_counters", "bottled_cards", "burning"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - (required | optional)):
        raise ValueError("Standard catalog entry has missing or unknown fields")
    _validate_standard_identity(value["character"], value["ascension"])
    if not isinstance(value["id"], str) or _ENTRY_ID.fullmatch(value["id"]) is None:
        raise ValueError("Standard catalog entry id must be 1..64 ASCII alphanumeric, underscore, or hyphen characters")
    if not isinstance(value["encounter"], str) or not value["encounter"]:
        raise ValueError("Standard catalog entry encounter must be a non-empty string")
    if type(value["hp"]) is not int or type(value["max_hp"]) is not int or not 1 <= value["hp"] <= value["max_hp"] <= 999:
        raise ValueError("Standard catalog entry HP must satisfy 1 <= hp <= max_hp <= 999")
    if type(value["gold"]) is not int or not 0 <= value["gold"] <= 2**32 - 1:
        raise ValueError("Standard catalog entry gold must be a uint32 integer")
    deck = value["deck"]
    if not isinstance(deck, list) or not 1 <= len(deck) <= 999:
        raise ValueError("Standard catalog entry deck must contain 1..999 cards")
    for index, row in enumerate(deck):
        if (not isinstance(row, dict) or set(row) - {"card", "upgrades", "misc"}
                or not {"card", "upgrades"} <= row.keys()):
            raise ValueError(f"Standard catalog entry deck[{index}] must contain card and upgrades")
        if (not isinstance(row["card"], str) or not row["card"]
                or type(row["upgrades"]) is not int or not 0 <= row["upgrades"] <= 73):
            raise ValueError(f"Standard catalog entry deck[{index}] has invalid card/upgrades")
        if "misc" in row and type(row["misc"]) is not int:
            raise ValueError(f"Standard catalog entry deck[{index}].misc must be an integer")
    for name in ("relics", "potions"):
        if not isinstance(value[name], list):
            raise ValueError(f"Standard catalog entry {name} must be a list")
    if len(value["potions"]) > 16 or any(item is not None and not isinstance(item, str) for item in value["potions"]):
        raise ValueError("Standard catalog entry potions must be strings or null")
    if any(not isinstance(item, str) or not item for item in value["relics"]):
        raise ValueError("Standard catalog entry relics must be non-empty strings")
    return value


def _parse_case(raw: object, *, split: str) -> NeuralCase:
    if not isinstance(raw, dict):
        raise ValueError("each neural case must be an object")
    mode = raw.get("mode")
    common = {"case_id", "lineage_id", "mode", "environment_seed", "cohort"}
    if mode == "standalone":
        if set(raw) != common | {"entry_id", "entry"}:
            raise ValueError("invalid standalone neural case fields")
    elif mode == "run":
        if set(raw) != common | {"profile_id", "character", "ascension"}:
            raise ValueError("invalid Run neural case fields; catalog/resource resets are forbidden")
    elif mode == "standard-standalone":
        if set(raw) != common | {"entry"}:
            raise ValueError("invalid Standard standalone neural case fields")
    elif mode == "standard-run":
        if set(raw) not in (common | {"character", "ascension"},
                            common | {"character", "ascension", "creation"}):
            raise ValueError("invalid Standard Run neural case fields")
    else:
        raise ValueError("neural case mode must be standalone, run, standard-standalone, or standard-run")
    case_id = _identifier(raw["case_id"], "case_id")
    lineage_id = _identifier(raw["lineage_id"], "lineage_id")
    cohort = _identifier(raw["cohort"], "cohort")
    seed = _validate_seed(split, raw["environment_seed"])
    if mode == "standalone":
        _validate_entry(raw["entry"], entry_id=raw["entry_id"])
        return NeuralCase(case_id, lineage_id, mode, seed, cohort, raw["entry_id"],
                          raw["entry"], None, None, None)
    if mode == "run":
        return NeuralCase(case_id, lineage_id, mode, seed, cohort, None, None,
                          raw["profile_id"], raw["character"], raw["ascension"])
    if mode == "standard-standalone":
        entry = _validate_standard_entry(raw["entry"])
        return NeuralCase(case_id, lineage_id, mode, seed, cohort, entry["id"], entry,
                          None, entry["character"], entry["ascension"])
    _validate_standard_identity(raw["character"], raw["ascension"])
    creation = raw.get("creation")
    return NeuralCase(case_id, lineage_id, mode, seed, cohort, None, None,
                      None, raw["character"], raw["ascension"], creation)


def load_neural_cases(path: Path) -> NeuralCases:
    """Load and strictly validate a versioned, explicit neural case manifest."""
    source = Path(path).resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            raw = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read neural cases {source}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema", "split", "partition", "cases"}:
        raise ValueError("neural cases root must contain only schema, split, partition, and cases")
    if raw["schema"] != "spire-learn-neural-cases-v1":
        raise ValueError("unsupported neural cases schema")
    split, partition = raw["split"], raw["partition"]
    if not isinstance(split, str) or split not in _SPLITS or partition != split:
        raise ValueError("neural case split and partition must match train/selection/final")
    if not isinstance(raw["cases"], list) or not raw["cases"]:
        raise ValueError("neural cases must be a non-empty array")
    cases = tuple(_parse_case(item, split=split) for item in raw["cases"])
    case_ids = [case.case_id for case in cases]
    lineages = [case.lineage_id for case in cases]
    seeds = [case.environment_seed for case in cases]
    entry_ids = [case.entry_id for case in cases if case.entry_id is not None]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate neural case_id")
    if len(set(lineages)) != len(lineages):
        raise ValueError("duplicate neural lineage_id within split")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate environment_seed within split manifest")
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("duplicate standalone entry_id within split manifest")
    return NeuralCases(source, split, partition, cases, _canonical_hash(raw))


def validate_neural_case_lineage(
    cases: Sequence[NeuralCases],
    *,
    training: bool = False,
) -> None:
    """Reject split overlap and prevent final cases from entering training."""
    if type(training) is not bool:
        raise ValueError("training must be a boolean")
    if not isinstance(cases, Sequence) or not cases or any(not isinstance(item, NeuralCases) for item in cases):
        raise ValueError("cases must be a non-empty sequence of NeuralCases")
    if training and any(item.split == "final" for item in cases):
        raise ValueError("final cases are evaluation-only and cannot be used for training")
    splits = [item.split for item in cases]
    if len(set(splits)) != len(splits):
        raise ValueError("duplicate neural split manifest")
    seen_lineages: dict[str, str] = {}
    seen_case_ids: dict[str, str] = {}
    seen_seeds: dict[int, str] = {}
    for manifest in cases:
        for case in manifest.cases:
            for value, seen, label in (
                (case.lineage_id, seen_lineages, "lineage"),
                (case.case_id, seen_case_ids, "case ID"),
                (case.environment_seed, seen_seeds, "environment seed"),
            ):
                previous = seen.get(value)
                if previous is not None:
                    raise ValueError(f"duplicate {label} across {previous} and {manifest.split} splits")
                seen[value] = manifest.split


__all__ = [
    "NeuralCase",
    "NeuralCases",
    "environment_seed_for",
    "load_neural_cases",
    "validate_neural_case_lineage",
]
