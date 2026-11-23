"""``baalorlord-standard-v1``: Standard combat cases from Baalorlord's reconstructed fight states.

A case is a Gym-validated fight state from the local run-states table (Task 3), at fuzz
level *k*: level 0 is his exact state; level *k* replaces *k* deck cards and *k*
non-starter relics with ones seen in his train-split runs (capped at what the state has).

Runs are split whole by a hash of ``run_id`` (80 / 10 / 10 train / selection / final);
training shards draw only train-split states, panels only selection or final ones, and
fuzz pools come from train-split states only. Every random choice comes from an RNG
seeded by the case's position and an identity hash, so a recipe fixes every shard and
panel byte for byte. Training shards are seeded from the full recipe hash. Panels are seeded
from ``panel_sha256``, a panel-only identity (states table, character, ascension, panel
levels and size, ``fuzz_levels_max``, ``excluded_strata``, ``redraw_bound`` and a fixed
panel version), so training-only fields (``fuzz_schedule``, ``shard_size``,
``generator_seed``) never move a panel. Panel strata are monster, elite and boss fights
only; event fights never enter a panel. The Gym validator is injected: fuzzed entries need it, and a
rejected draw is redrawn up to ``redraw_bound`` times and then skipped. Importing this
module never imports the Gym.

Relic-specific rules (fuzzing): the starter relic (Burning Blood or Black Blood) is never
replaced, and neither the starters, the bottles nor Neow's Lament (which the Gym accepts
only with an explicit counter) are drawn as replacements. Potion Belt is an ordinary
relic (controller ruling, fix round 1) except that a state holding more than 2 potions
cannot lose it, since potions stay unchanged; when the belt comes or goes, the held
potions keep their order and the empty slots are re-padded to the new slot count.
Replacing a bottle leaves its card unbottled; replacing the bottled card replaces its bottle too (that bottle counts as one
of the *k* relics; if the relic pool cannot supply that bottle's replacement, the card
replacement is undone and the actual counts record it). A bottle's card is the first deck row equal to its bottled card.
Removed relics lose their counters; replacement relics take the Gym's default counters.
Excluding bottles from the replacement pool and protecting the starter relic are generator
policy, not descriptor requirements (the descriptor lets a bottle be owned with its companion
card and lets a boss relic replace its starter); only Neow's Lament's required counter is a
descriptor rule here.
A replacement card keeps the replaced card's upgrades up to its own limit and takes the
Ruling A ``misc`` default (Genetic Algorithm 1, Ritual Dagger 15, otherwise 0).

Base-state order: the recipe sorts its states by ``(run_id, floor)`` (unique, else the
load fails), so shards and panels do not depend on the table's row order. Generation
identity still uses the raw table sha256 that authenticates the file: a re-serialized
table must be re-pinned and then draws new seeds. A semantic digest was not added because
the extractor writes the table canonically and sorted, so the raw hash is already stable.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields
import hashlib
import json
from pathlib import Path
import random
import re
import threading

from .cases import _freeze_json, _parse_case, environment_seed_for
from .scenario_stream import ScenarioShard
from .scenario_support import content_sha256, json_value, read_json
from .entry_resources import DEFAULT_SNAPSHOT, EXPECTED_FINGERPRINTS, EntryResources, load_entry_resources
from .gym_contracts import GYM_V1_1_1_CONTRACT, GYM_V1_2_0_CONTRACT
from .run_states import RECORDED_COUNTER_RELICS
from .standard_catalog import Validate, potion_slots, validate_entries
from .standard_names import NameAliases, load_aliases

GENERATOR = "baalorlord-standard-v1"
RECIPE_SCHEMA = "spire-learn-standard-recipe-v1"
TABLE_SCHEMA = "spire-learn-run-states-v1"
PANEL_SCHEMA = "spire-learn-standard-panel-v2"
PANEL_VERSION = "baalorlord-standard-panel-v1"  # fixed: part of every panel identity
MODE = "standard-standalone"
SPLITS = ("train", "selection", "final")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
KINDS = ("monster", "elite", "boss", "event")
STARTER_RELICS = {"ironclad": ("burning_blood", "black_blood")}
BOTTLES = ("bottled_flame", "bottled_lightning", "bottled_tornado")
POTION_BELT = "potion_belt"
NO_DEFAULT_COUNTER = ("neows_lament",)  # the Gym requires an explicit counter, so it cannot be a replacement
MISC_DEFAULTS = {"genetic_algorithm": 1, "ritual_dagger": 15}  # Ruling A
# Relics whose counter no run file records, so every state holds them at the Gym default
# (run_states._relic_counters); a state holding one is tagged ``relic_counter_defaulted``.
# This is the Gym 1.1.1 list, kept exactly for the v1.1.1 series; Gym 1.2.0 tables take their list
# from the entry-resource descriptor (``FuzzPools.defaulted_counters``).
DEFAULTED_COUNTER_RELICS = ("pen_nib", "nunchaku", "ink_bottle", "incense_burner", "happy_flower", "sundial",
                            "inserter", "wing_boots", "tiny_chest", "matryoshka")
DEFAULT_ENTRY_RESOURCES = DEFAULT_SNAPSHOT
_TABLE_REVISIONS = (GYM_V1_1_1_CONTRACT.source_revision, GYM_V1_2_0_CONTRACT.source_revision)
PANEL_KINDS = ("monster", "elite", "boss")  # event fights are never panel strata
PANEL_LEVELS = (0, 1, 2, 4)
PANEL_PER_LEVEL = 64
PANEL_SEED_OFFSET = 2**33  # panel ordinals sit above every shard ordinal (index * 256 + local < 2**32)
DEFAULT_ALIASES = Path(__file__).resolve().parents[3] / "configs/neural/standard/name-aliases.json"
_STRATUM = re.compile(r"[1-4]:(monster|elite|boss|event)\Z")
_REGENERATE = ("regenerate it with scripts/neural/run_states.py extract in the runtime container "
               "(docs/superpowers/plans/2026-11-20-standard-foundation.md, Task 3)")
_RECIPE_FIELDS = {"schema", "generator_version", "states_table", "character", "ascension", "generator_seed",
                  "shard_size", "split_fractions", "fuzz_levels_max", "fuzz_schedule", "redraw_bound"}


def _integer(value: object, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def run_split(run_id: str) -> str:
    """The run's split: a hash of ``run_id`` into 80 / 10 / 10 train / selection / final."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    digest = hashlib.sha256(f"baalorlord-standard-split-v1:{run_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "train" if bucket < 8 else "selection" if bucket == 8 else "final"


def _base_key(state: Mapping[str, object]) -> tuple[str, int]:
    return state["run_id"], state["floor"]


def stratum(state: Mapping[str, object]) -> str:
    return f"{state['act']}:{state['room_kind']}"


def _stratum_order(key: str) -> tuple[int, int]:
    act, kind = key.split(":")
    return int(act), KINDS.index(kind)


@dataclass(frozen=True)
class FuzzRecord:
    """What a fuzz draw actually replaced (``level`` capped by the state)."""

    cards_replaced: int
    relics_replaced: int

    def as_json(self) -> dict[str, int]:
        return {"cards_replaced": self.cards_replaced, "relics_replaced": self.relics_replaced}


@dataclass(frozen=True)
class FuzzPools:
    """Replacement pools, sorted: cards and relics seen in train-split states of the character.

    ``defaulted_counters`` and ``slots`` (potion slots without and with Potion Belt) are the table's
    per-revision rules: the Gym 1.1.1 constants, or the Gym 1.2.0 entry-resource descriptor's.
    """

    cards: tuple[str, ...]
    relics: tuple[str, ...]
    max_upgrades: Mapping[str, int | None]
    protected: tuple[str, ...]  # relics never replaced: the starters
    defaulted_counters: tuple[str, ...] = DEFAULTED_COUNTER_RELICS
    optional_counters: tuple[str, ...] | None = None  # Gym 1.2.0: every relic with an optional counter
    slots: tuple[int, int] = (potion_slots([]), potion_slots([POTION_BELT]))

    def potion_slots(self, relics: Sequence[str]) -> int:
        return self.slots[1] if POTION_BELT in relics else self.slots[0]


@dataclass(frozen=True)
class StandardRecipe:
    generator_seed: int
    shard_size: int
    fuzz_schedule: tuple[tuple[int, tuple[int, ...]], ...]
    redraw_bound: int
    excluded_strata: tuple[str, ...]
    table_sha256: str
    states: tuple[Mapping[str, object], ...]  # the character's states; never mutated
    aliases: NameAliases
    character: str = "ironclad"
    ascension: int = 20
    fuzz_levels_max: int = 4
    split_fractions: tuple[float, ...] = SPLIT_FRACTIONS
    generator_version: str = GENERATOR
    entry_resources: EntryResources | None = field(default=None, compare=False)  # set for Gym 1.2.0 tables
    gym_revision: str = GYM_V1_1_1_CONTRACT.source_revision  # the table's; configs must name the same one
    _by_split: Mapping[str, tuple[Mapping[str, object], ...]] = field(default=None, repr=False, compare=False)
    _pools: FuzzPools = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.generator_version != GENERATOR:
            raise ValueError(f"generator_version must be {GENERATOR}")
        if self.gym_revision not in _TABLE_REVISIONS:
            raise ValueError(f"unsupported Gym revision {self.gym_revision!r}")
        if (self.entry_resources is not None) != (self.gym_revision == GYM_V1_2_0_CONTRACT.source_revision):
            raise ValueError("entry_resources are required for Gym 1.2.0 recipes and only for them")
        if self.character not in STARTER_RELICS:
            raise ValueError("the curriculum character must be ironclad")
        if self.ascension != 20:
            raise ValueError("the curriculum ascension must be 20")
        if tuple(self.split_fractions) != SPLIT_FRACTIONS:
            raise ValueError("split_fractions must be [0.8, 0.1, 0.1] (run_split hard-codes them)")
        _integer(self.generator_seed, 0, 2**64 - 1, "generator_seed")
        _integer(self.shard_size, 1, 256, "shard_size")
        _integer(self.fuzz_levels_max, 0, 16, "fuzz_levels_max")
        _integer(self.redraw_bound, 0, 64, "redraw_bound")
        schedule = []
        for index, step in enumerate(self.fuzz_schedule):
            start, levels = step
            _integer(start, 0, 2**24 - 1, "from_shard")
            if (index == 0 and start != 0) or (index and start <= schedule[-1][0]):
                raise ValueError("fuzz_schedule from_shard values must start at 0 and strictly increase")
            levels = tuple(levels)
            if (not levels or any(type(k) is not int or not 0 <= k <= self.fuzz_levels_max for k in levels)
                    or list(levels) != sorted(set(levels))):
                raise ValueError("fuzz_schedule levels must be sorted unique integers in [0, fuzz_levels_max]")
            schedule.append((start, levels))
        if not schedule:
            raise ValueError("fuzz_schedule must not be empty")
        object.__setattr__(self, "fuzz_schedule", tuple(schedule))
        excluded = tuple(self.excluded_strata)
        if any(not isinstance(s, str) or not _STRATUM.fullmatch(s) for s in excluded) or len(set(excluded)) != len(excluded):
            raise ValueError('excluded_strata must be unique "<act>:<kind>" strings, e.g. "4:elite"')
        object.__setattr__(self, "excluded_strata", tuple(sorted(excluded, key=_stratum_order)))
        if not re.fullmatch(r"[0-9a-f]{64}", self.table_sha256):
            raise ValueError("table_sha256 must be 64 lowercase hex digits")
        # Canonical base order: generation must not depend on the table's row order.
        states = tuple(sorted(self.states, key=_base_key))
        keys = [_base_key(s) for s in states]
        duplicate = next((a for a, b in zip(keys, keys[1:]) if a == b), None)
        if duplicate is not None:
            raise ValueError(f"duplicate base state (run_id, floor) {duplicate}")
        object.__setattr__(self, "states", states)
        by_split: dict[str, list] = {split: [] for split in SPLITS}
        train_all = []
        for state in states:
            entry = state["entry"]
            if (state["character"] != self.character or entry["character"] != self.character
                    or entry["ascension"] != self.ascension or not _STRATUM.fullmatch(stratum(state))):
                raise ValueError(f"state {entry.get('id')!r} is not a {self.character} A{self.ascension} state")
            split = run_split(state["run_id"])
            if split == "train":
                train_all.append(state)
            if stratum(state) not in excluded:
                by_split[split].append(state)
        object.__setattr__(self, "_by_split", {k: tuple(v) for k, v in by_split.items()})
        object.__setattr__(self, "_pools", _pools(train_all, self.aliases, self.character,
                                                  self.entry_resources, self.ascension))

    @property
    def mode(self) -> str:
        return MODE

    @property
    def pools(self) -> FuzzPools:
        return self._pools

    def states_for(self, split: str) -> tuple[Mapping[str, object], ...]:
        """The split's states outside ``excluded_strata``, in table order."""
        if split not in SPLITS:
            raise ValueError("unsupported split")
        return self._by_split[split]

    @property
    def category_keys(self) -> tuple[str, ...]:
        strata = sorted({stratum(s) for s in self.states} - set(self.excluded_strata), key=_stratum_order)
        levels = sorted({k for _, levels in self.fuzz_schedule for k in levels})
        return tuple(f"room:{s}" for s in strata) + tuple(f"level:{k}" for k in levels)

    @property
    def payload(self) -> Mapping[str, object]:
        return _freeze_json({
            "schema": RECIPE_SCHEMA, "generator_version": self.generator_version,
            "states_table": {"sha256": self.table_sha256}, "character": self.character,
            "ascension": self.ascension, "generator_seed": self.generator_seed, "shard_size": self.shard_size,
            "split_fractions": list(self.split_fractions), "fuzz_levels_max": self.fuzz_levels_max,
            "fuzz_schedule": [{"from_shard": start, "levels": list(levels)} for start, levels in self.fuzz_schedule],
            "redraw_bound": self.redraw_bound, "excluded_strata": list(self.excluded_strata)})

    @property
    def sha256(self) -> str:
        return content_sha256(self.payload)


def _pools(train_states: Sequence[Mapping[str, object]], aliases: NameAliases, character: str,
           resources: EntryResources | None = None, ascension: int = 20) -> FuzzPools:
    cards = sorted({row["card"] for s in train_states for row in s["entry"]["deck"]})
    unknown = [c for c in cards if c not in aliases.card_ids]
    if unknown:
        raise ValueError(f"cards without Gym metadata in the states table: {unknown[:5]}")
    protected = STARTER_RELICS[character]
    if resources is None:  # Gym 1.1.1 constants
        unusable, rules = set(BOTTLES) | set(NO_DEFAULT_COUNTER), {}
    else:  # a relic needing a companion or a counter the replacement cannot supply is never drawn
        unusable = set(resources.bottle_relics()) | set(resources.counter_required_relics())
        rules = {"defaulted_counters": resources.defaulted_counter_relics(RECORDED_COUNTER_RELICS),
                 "optional_counters": tuple(sorted(r for r, c in resources.counters.items() if c.rule == "optional")),
                 "slots": (resources.potion_slots([], ascension), resources.potion_slots([POTION_BELT], ascension))}
    relics = sorted({r for s in train_states for r in s["entry"]["relics"]} - unusable - set(protected))
    return FuzzPools(tuple(cards), tuple(relics), _freeze_json({c: aliases.max_upgrades(c) for c in cards}),
                     protected, **rules)


def load_standard_recipe(path: Path, *, aliases_path: Path = DEFAULT_ALIASES,
                         entry_resources_path: Path = DEFAULT_ENTRY_RESOURCES) -> StandardRecipe:
    """Load a recipe and its states table; a missing or changed table fails closed."""
    raw = read_json(path)
    keys = set(raw) - {"excluded_strata"}  # Ruling B: optional, default []
    if keys != _RECIPE_FIELDS or raw["schema"] != RECIPE_SCHEMA:
        raise ValueError("recipe has missing, unknown or unsupported schema fields")
    spec = raw["states_table"]
    if (not isinstance(spec, dict) or set(spec) != {"path", "sha256"} or not isinstance(spec["path"], str)
            or not spec["path"] or not isinstance(spec["sha256"], str)):
        raise ValueError("recipe states_table must contain exactly path and sha256")
    table_path = Path(path).resolve().parent / spec["path"]
    if not table_path.is_file():
        raise ValueError(f"run-states table is missing: {table_path}; " + _REGENERATE)
    data = table_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"run-states table SHA256 mismatch: {table_path}; " + _REGENERATE)
    table = json.loads(data)  # authenticated by the pinned hash above
    if not isinstance(table, dict) or table.get("schema") != TABLE_SCHEMA or not isinstance(table.get("states"), list):
        raise ValueError(f"run-states table must have schema {TABLE_SCHEMA}")
    aliases = load_aliases(aliases_path)
    if aliases.sha256 != table.get("aliases_sha256"):
        raise ValueError("the name alias table differs from the one the run-states table was built with")
    # The table's Gym revision selects the per-relic rules: Gym 1.1.1 tables keep the constants above,
    # Gym 1.2.0 tables follow the entry-resource descriptor.
    revision = table.get("gym_revision")
    if revision not in _TABLE_REVISIONS:
        raise ValueError(f"run-states table has an unsupported Gym revision {revision!r}")
    resources = None
    recorded = table.get("entry_resources_fingerprint")
    if revision == GYM_V1_2_0_CONTRACT.source_revision:
        resources = load_entry_resources(entry_resources_path)
        # The descriptor is part of the table's identity (and so of the recipe's and the panels'):
        # the table records it, and only the fingerprint pinned for this revision is accepted.
        if resources.fingerprint != EXPECTED_FINGERPRINTS[revision] or recorded != resources.fingerprint:
            raise ValueError("the entry-resource descriptor differs from the one the run-states table "
                             "was built with (or from the one pinned for Gym 1.2.0)")
    elif recorded is not None:
        raise ValueError("a Gym 1.1.1 run-states table cannot carry an entry-resource fingerprint")
    schedule = raw["fuzz_schedule"]
    if (not isinstance(schedule, list)
            or any(not isinstance(s, dict) or set(s) != {"from_shard", "levels"} or not isinstance(s["levels"], list)
                   for s in schedule)):
        raise ValueError("fuzz_schedule must be a list of {from_shard, levels}")
    fractions = raw["split_fractions"]
    if not isinstance(fractions, list):
        raise ValueError("split_fractions must be a list")
    character = raw["character"]
    return StandardRecipe(
        generator_seed=raw["generator_seed"], shard_size=raw["shard_size"],
        fuzz_schedule=tuple((s["from_shard"], tuple(s["levels"])) for s in schedule),
        redraw_bound=raw["redraw_bound"], excluded_strata=tuple(raw.get("excluded_strata", [])),
        table_sha256=spec["sha256"], aliases=aliases,
        states=tuple(s for s in table["states"] if s.get("character") == character),
        character=character, ascension=raw["ascension"], fuzz_levels_max=raw["fuzz_levels_max"],
        split_fractions=tuple(fractions), generator_version=raw["generator_version"],
        entry_resources=resources, gym_revision=revision)


# ---------------------------------------------------------------- fuzzing

def _same_card(row: Mapping[str, object], card: Mapping[str, object]) -> bool:
    return (row["card"], row["upgrades"], row.get("misc", 0)) == (card["card"], card["upgrades"], card.get("misc", 0))


def fuzz(state: Mapping[str, object], level: int, pools: FuzzPools, rng: random.Random) -> tuple[dict, FuzzRecord]:
    """The state's entry with ``level`` cards and ``level`` non-starter relics replaced (capped).

    Encounter, HP, gold, potions and the entry id are unchanged. Level 0 is the exact entry.
    """
    _integer(level, 0, 16, "level")
    entry = json_value(state["entry"])
    if level == 0:
        return entry, FuzzRecord(0, 0)
    deck, relics = entry["deck"], entry["relics"]
    original = [dict(row) for row in deck]
    potions = [p for p in entry["potions"] if p is not None]
    counters = entry.get("relic_counters", {})
    bottled = entry.get("bottled_cards", {})
    bottle_rows: dict[str, int] = {}
    for bottle in sorted(bottled):
        row = next((i for i, row in enumerate(deck)
                    if i not in bottle_rows.values() and _same_card(row, bottled[bottle])), None)
        if row is None:
            raise ValueError(f"{entry['id']}: {bottle} holds a card that is not in the deck")
        bottle_rows[bottle] = row
    replaced_rows = []
    for i in sorted(rng.sample(range(len(deck)), min(level, len(deck)))):
        old = deck[i]
        choices = [c for c in pools.cards if c != old["card"]]
        if not choices:
            continue
        card = rng.choice(choices)
        limit = pools.max_upgrades[card]
        deck[i] = {"card": card, "upgrades": old["upgrades"] if limit is None else min(old["upgrades"], limit),
                   "misc": MISC_DEFAULTS.get(card, 0)}
        replaced_rows.append(i)
    forced = [b for b in sorted(bottle_rows) if bottle_rows[b] in replaced_rows and b in relics]
    # Losing Potion Belt leaves 2 slots, so it stays when more than 2 potions are held.
    replaceable = [r for r in relics if r not in pools.protected
                   and not (r == POTION_BELT and len(potions) > pools.potion_slots([]))]
    held = set(relics)
    count = min(level, len(replaceable), len([r for r in pools.relics if r not in held]))
    for bottle in forced[count:]:
        # The pool cannot replace this bottle, so its card stays: undo that card replacement.
        deck[bottle_rows[bottle]] = original[bottle_rows[bottle]]
        replaced_rows.remove(bottle_rows[bottle])
    forced = forced[:count]
    others = [r for r in replaceable if r not in forced]
    chosen = forced + rng.sample(others, count - len(forced))
    for relic in chosen:
        new = rng.choice([r for r in pools.relics if r not in held])
        held.add(new)
        relics[relics.index(relic)] = new
        counters.pop(relic, None)  # the new relic takes the Gym's default counter
        bottled.pop(relic, None)   # a replaced bottle leaves its card unbottled
    if "bottled_cards" in entry and not bottled:
        del entry["bottled_cards"]
    slots = pools.potion_slots(relics)
    if slots != len(entry["potions"]):  # Potion Belt came or went; held potions are unchanged
        entry["potions"] = potions + [None] * (slots - len(potions))
    return entry, FuzzRecord(len(replaced_rows), len(chosen))


def _tags(state: Mapping[str, object], entry: Mapping[str, object],
          defaulted: Collection[str] = DEFAULTED_COUNTER_RELICS,
          optional_counters: Collection[str] | None = None) -> list[str]:
    """The base state's tags, with the defaulting tags recomputed for the (fuzzed) entry:
    ``misc_defaulted`` iff a deck card takes a Ruling A misc default, and
    ``relic_counter_defaulted`` iff a held relic's counter is the Gym default."""
    tags = set(state["tags"]) - {"misc_defaulted", "relic_counter_defaulted"}
    if any(row["card"] in MISC_DEFAULTS for row in entry["deck"]):
        tags.add("misc_defaulted")
    if any(relic in defaulted for relic in entry["relics"]):
        tags.add("relic_counter_defaulted")
    # Gym 1.2.0 only (``optional_counters`` is None for v1.1.1, whose panel documents are pinned and
    # keep the list-only rule): a relic fuzzed in that has an optional counter takes the Gym default
    # unless its key was supplied, even when run files do record that relic's counter.
    if optional_counters is not None:
        base = set(state["entry"]["relics"])
        if any(r in optional_counters and r not in base and r not in entry.get("relic_counters", {})
               for r in entry["relics"]):
            tags.add("relic_counter_defaulted")
    return sorted(tags)


@dataclass
class _Draft:
    state: Mapping[str, object]
    level: int
    rng: random.Random
    identity: list
    prefix: str
    entry: dict | None = None
    record: FuzzRecord | None = None

    def draw(self, pools: FuzzPools) -> None:
        entry, self.record = fuzz(self.state, self.level, pools, self.rng)
        body = {k: v for k, v in entry.items() if k != "id"}
        entry["id"] = self.prefix + content_sha256([*self.identity, body])[:48]
        self.entry = entry


def _resolve(drafts: Sequence[_Draft], recipe: StandardRecipe, validate: Validate | None) -> int:
    """Draw every draft; Gym-validate fuzzed ones, redrawing rejections up to the bound.

    Level 0 entries are the table's own Gym-accepted entries and are not revalidated. A
    draft still rejected after ``redraw_bound`` redraws gets ``entry = None``; returns how
    many. Faults (anything but ``CatalogRejected``) propagate from ``validate_entries``.
    """
    for draft in drafts:
        draft.draw(recipe.pools)
    pending = [d for d in drafts if d.level]
    if pending and validate is None:
        raise ValueError("fuzzed cases need a Gym validator")
    for attempt in range(recipe.redraw_bound + 1):
        if not pending:
            break
        if attempt:
            for draft in pending:
                draft.draw(recipe.pools)
        _, rejected = validate_entries([d.entry for d in pending], validate)
        bad = {entry["id"] for entry, _ in rejected}
        pending = [d for d in pending if d.entry["id"] in bad]
    for draft in pending:
        draft.entry = draft.record = None
    return len(pending)


def _rng(recipe: StandardRecipe, *parts: object) -> random.Random:
    # Training shards: seeded from the recipe hash and the case's position only; no global or Gym RNG.
    return random.Random(int(content_sha256([recipe.sha256, *parts]), 16))


def _panel_rng(identity: str, *parts: object) -> random.Random:
    # Panels: seeded from the panel identity hash and the case's position only.
    return random.Random(int(content_sha256([identity, *parts]), 16))


def panel_identity(recipe: StandardRecipe) -> Mapping[str, object]:
    """Everything a panel depends on, and nothing training-only.

    The fuzz pools and ``max_upgrades`` follow from the states table (which pins the alias
    table and, for Gym 1.2.0, records the entry-resource descriptor fingerprint) and the
    character, so the table hash covers them.
    """
    if not isinstance(recipe, StandardRecipe):
        raise ValueError("recipe must be a StandardRecipe")
    return _freeze_json({
        "panel_version": PANEL_VERSION, "generator_version": recipe.generator_version,
        "states_table": {"sha256": recipe.table_sha256}, "character": recipe.character,
        "ascension": recipe.ascension, "panel_levels": list(PANEL_LEVELS), "panel_per_level": PANEL_PER_LEVEL,
        "panel_kinds": list(PANEL_KINDS), "fuzz_levels_max": recipe.fuzz_levels_max,
        "excluded_strata": list(recipe.excluded_strata), "redraw_bound": recipe.redraw_bound})


def panel_sha256(recipe: StandardRecipe) -> str:
    """The panel seed: the hash of ``panel_identity(recipe)``."""
    return content_sha256(panel_identity(recipe))


def _annotation(draft: _Draft, pools: FuzzPools) -> dict[str, object]:
    state = draft.state
    return {"run_id": state["run_id"], "floor": state["floor"], "act": state["act"],
            "room_kind": state["room_kind"], "level": draft.level, "fuzz": draft.record.as_json(),
            "tags": _tags(state, draft.entry, pools.defaulted_counters, pools.optional_counters)}


def _case(draft: _Draft, split: str, ordinal: int, lineage: str) -> dict[str, object]:
    case = {"case_id": draft.entry["id"], "lineage_id": lineage, "mode": MODE,
            "environment_seed": environment_seed_for(split, replica=0, ordinal=ordinal),
            "cohort": f"baalorlord_level_{draft.level}", "entry": draft.entry}
    _parse_case(case, split=split)
    return case


def _base_id(state: Mapping[str, object]) -> str:
    return f"{state['run_id']}-{state['floor']}"


# ---------------------------------------------------------------- training shards

def levels_for_shard(recipe: StandardRecipe, shard_index: int) -> tuple[int, ...]:
    """The fuzz levels enabled at ``shard_index``: the last schedule step at or before it."""
    _integer(shard_index, 0, 2**24 - 1, "shard_index")
    return [levels for start, levels in recipe.fuzz_schedule if start <= shard_index][-1]


_SHARD_CACHE_CAPACITY = 16
_SHARD_CACHE: OrderedDict[tuple, tuple[ScenarioShard, tuple[object, ...]]] = OrderedDict()
_SHARD_CACHE_LOCK = threading.Lock()


def _shard_fields(shard: ScenarioShard) -> tuple[object, ...]:
    return tuple(getattr(shard, f.name) for f in fields(shard))


def _descriptor_fingerprint(recipe: StandardRecipe) -> str | None:
    return None if recipe.entry_resources is None else recipe.entry_resources.fingerprint


def clear_shard_cache() -> None:
    with _SHARD_CACHE_LOCK:
        _SHARD_CACHE.clear()


def generate_standard_shard(recipe: StandardRecipe, *, split: str, shard_index: int, arm: str = "mixed",
                            validate: Validate | None = None) -> ScenarioShard:
    """Shard ``shard_index`` of ``split``: ``shard_size`` cases, each a uniform level from the
    schedule and a uniform base state of the split, minus fuzz draws skipped after the redraw bound.

    ``validate`` (the Gym catalog check) is required once the schedule enables a fuzz level.
    """
    if not isinstance(recipe, StandardRecipe):
        raise ValueError("recipe must be a StandardRecipe")
    if split not in SPLITS:
        raise ValueError("unsupported split")
    if arm != "mixed":
        raise ValueError("baalorlord-standard-v1 has only the mixed arm")
    levels = levels_for_shard(recipe, shard_index)
    if max(levels) and validate is None:
        raise ValueError(f"shard {shard_index} enables fuzz levels {levels}; pass the Gym validator")
    cacheable = type(recipe) is StandardRecipe
    key = (recipe.sha256, _descriptor_fingerprint(recipe), split, shard_index, arm, validate)
    if cacheable:
        with _SHARD_CACHE_LOCK:
            cached = _SHARD_CACHE.get(key)
            if cached is not None and all(a is b for a, b in zip(_shard_fields(cached[0]), cached[1])):
                _SHARD_CACHE.move_to_end(key)
                return cached[0]
    shard = _build_shard(recipe, split, shard_index, arm, levels, validate)
    if cacheable:
        with _SHARD_CACHE_LOCK:
            _SHARD_CACHE[key] = (shard, _shard_fields(shard))
            while len(_SHARD_CACHE) > _SHARD_CACHE_CAPACITY:
                _SHARD_CACHE.popitem(last=False)
    return shard


def _build_shard(recipe: StandardRecipe, split: str, shard_index: int, arm: str, levels: tuple[int, ...],
                 validate: Validate | None) -> ScenarioShard:
    states = recipe.states_for(split)
    if not states:
        raise ValueError(f"no {split} states to draw from")
    distribution = content_sha256({"recipe": recipe.payload, "split": split, "arm": arm, "generator": GENERATOR})
    drafts = []
    for local in range(recipe.shard_size):
        pick = _rng(recipe, "shard", split, arm, shard_index, local, "pick")
        level = pick.choice(levels)
        state = states[pick.randrange(len(states))]
        drafts.append(_Draft(state, level, _rng(recipe, "shard", split, arm, shard_index, local, "fuzz"),
                             [distribution, shard_index, local], "b-"))
    _resolve(drafts, recipe, validate)
    cases, annotations = [], {}
    for local, draft in enumerate(drafts):
        if draft.entry is None:
            continue  # skipped after the redraw bound; shard_size - len(cases) counts them
        lineage = f"baalorlord:{_base_id(draft.state)}:k{draft.level}:{shard_index}.{local}"
        cases.append(_case(draft, split, shard_index * 256 + local, lineage))
        annotations[draft.entry["id"]] = _annotation(draft, recipe.pools)
    manifest = {"schema": "spire-learn-neural-cases-v1", "split": split, "partition": split, "cases": cases}
    return ScenarioShard(manifest, {"entries": [c["entry"] for c in cases]}, annotations, shard_index, arm,
                         content_sha256(manifest), distribution)


# ---------------------------------------------------------------- panels

def allocate_quotas(total: int, sizes: Mapping[str, int], caps: Mapping[str, int] | None = None) -> dict[str, int]:
    """Split ``total`` over strata in proportion to ``sizes`` (largest remainder, ties by key),
    never above ``caps`` (default: the sizes); a capped stratum's shortfall goes to the others
    in proportion to their sizes."""
    caps = dict(sizes if caps is None else caps)
    if sum(caps.values()) < total:
        raise ValueError(f"{sum(caps.values())} available cases cannot fill {total}")
    result = {key: 0 for key in sizes}
    active = sorted(key for key in sizes if caps[key] > 0 and sizes[key] > 0)
    remaining = total
    while remaining:
        weight = sum(sizes[key] for key in active)
        share = {key: remaining * sizes[key] // weight for key in active}
        extra = remaining - sum(share.values())
        for key in sorted(active, key=lambda k: (-(remaining * sizes[k] % weight), k))[:extra]:
            share[key] += 1
        over = [key for key in active if share[key] > caps[key]]
        if not over:
            result.update(share)
            break
        for key in over:
            result[key] = caps[key]
            remaining -= caps[key]
            active.remove(key)
    return result


def build_standard_panel(recipe: StandardRecipe, split: str, *, validate: Validate) -> Mapping[str, object]:
    """The ``split`` panel: 64 cases at each of levels 0, 1, 2 and 4, with distinct base
    states within a level, stratified by act x room kind (monster, elite and boss only) in
    proportion to the available states, and seeded from ``panel_sha256(recipe)`` only. A stratum that runs out of Gym-accepted fuzz draws takes what it has and its
    shortfall goes to the level's other strata."""
    if not isinstance(recipe, StandardRecipe):
        raise ValueError("recipe must be a StandardRecipe")
    if split not in ("selection", "final"):
        raise ValueError("panels are built from selection or final runs only")
    if validate is None:
        raise ValueError("panels need the Gym validator")
    identity = panel_identity(recipe)
    seed = content_sha256(identity)
    by_stratum: dict[str, list] = {}
    for state in recipe.states_for(split):
        if state["room_kind"] in PANEL_KINDS:  # event fights are not panel strata (spec)
            by_stratum.setdefault(stratum(state), []).append(state)
    sizes = {key: len(rows) for key, rows in by_stratum.items()}
    cases, annotations, counts, skipped = [], {}, {}, {}
    for level in PANEL_LEVELS:
        orders = {}
        for key, rows in by_stratum.items():
            order = list(rows)
            _panel_rng(seed, "panel", split, level, key, "order").shuffle(order)
            orders[key] = order
        accepted = {key: [] for key in by_stratum}
        cursor = dict.fromkeys(by_stratum, 0)
        skipped[str(level)] = 0
        caps = dict(sizes)
        while True:
            quotas = allocate_quotas(PANEL_PER_LEVEL, sizes, caps)
            short = False
            for key in sorted(by_stratum, key=_stratum_order):
                while len(accepted[key]) < quotas[key] and cursor[key] < len(orders[key]):
                    start = cursor[key]
                    batch = orders[key][start:start + quotas[key] - len(accepted[key])]
                    cursor[key] += len(batch)
                    drafts = [_Draft(state, level, _panel_rng(seed, "panel", split, level, key, start + offset, "fuzz"),
                                     [seed, "panel", split, level, key, start + offset], "p-")
                              for offset, state in enumerate(batch)]
                    skipped[str(level)] += _resolve(drafts, recipe, validate)
                    accepted[key].extend(d for d in drafts if d.entry is not None)
                if len(accepted[key]) < quotas[key]:
                    caps[key] = len(accepted[key])
                    short = True
            if not short:
                break
        counts[str(level)] = {key: quotas[key] for key in sorted(quotas, key=_stratum_order) if quotas[key]}
        for key in sorted(by_stratum, key=_stratum_order):
            for draft in accepted[key][:quotas[key]]:
                case = _case(draft, split, PANEL_SEED_OFFSET + len(cases),
                             f"baalorlord:{_base_id(draft.state)}:k{level}")
                cases.append(case)
                annotations[case["case_id"]] = _annotation(draft, recipe.pools)
    manifest = {"schema": "spire-learn-neural-cases-v1", "split": split, "partition": split, "cases": cases}
    return _freeze_json({
        "schema": PANEL_SCHEMA, "generator_version": GENERATOR, "panel_identity": identity,
        "panel_sha256": seed, "split": split,
        "manifest": manifest, "manifest_sha256": content_sha256(manifest), "annotations": annotations,
        "available": {key: sizes[key] for key in sorted(sizes, key=_stratum_order)},
        "strata": counts, "skipped": skipped,
        "claim": "simulator panel of reconstructed public-run states; no original-game transfer claim"})


__all__ = ["FuzzPools", "FuzzRecord", "StandardRecipe", "allocate_quotas", "build_standard_panel",
           "clear_shard_cache", "fuzz", "generate_standard_shard", "levels_for_shard", "load_standard_recipe",
           "panel_identity", "panel_sha256", "run_split", "stratum"]
