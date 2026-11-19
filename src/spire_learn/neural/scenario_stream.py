"""Deterministic bounded scenario construction, without gameplay simulation.

Public resources reach the catalog; generator seeds/labels remain side metadata.
The recipe describes learner sampling heuristics, not a true Run distribution.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path
import random
import threading

from .cases import _freeze_json, _parse_case, environment_seed_for
from .scenario_support import (
    ScenarioSupport, canonical_bytes, content_sha256, json_value,
    load_scenario_support, read_json,
)

_SPLITS = ("train", "selection", "final")
_ARMS = ("fixed", "randomized", "mixed")
_STAGES = ("early", "middle", "developed")
_STARTER = ("strike",) * 5 + ("defend",) * 4 + ("bash",)
_FIXED_ENCOUNTERS = ("cultist", "gremlin_nob", "guardian", "hexaghost", "jaw_worm",
                     "lagavulin", "large_slime", "lots_of_slimes", "slime_boss",
                     "small_slimes", "three_louse", "three_sentries", "two_louse")
_ATTEMPTS = 4096
_V1 = "ironclad-curriculum-v1"
_V2 = "ironclad-curriculum-v2"
_PREPARED_ACTS = ("act1", "act2", "act3")
_PREPARED_KINDS = ("boss", "elite")
_RARITIES = ("starter", "common", "uncommon", "rare")
_REAL_TIERS = ("act2:boss", "act3:boss")
_REAL_SCHEMA = "spire-learn-real-boss-decks-v1"
_REAL_DECK_FIELDS = {"run_id", "encounter", "floor", "won", "ascension", "deck", "relics",
                     "hp_fraction", "dropped_cards"}
_REGENERATE = ("regenerate it with: PYTHONPATH=src python3.12 scripts/neural/real_boss_decks.py extract "
               "--raw <run files> --support configs/neural/curriculum-v1/support.json --output <path>")


def _integer(value: object, minimum: int, maximum: int, name: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class ScenarioRecipe:
    support: ScenarioSupport
    generator_seed: int
    shard_size: int
    maximum_catalog_bytes: int
    reserved_pairs: tuple[tuple[str, str], ...]
    lesson_pairs: tuple[tuple[str, str], ...]
    mechanic_groups: Mapping[str, tuple[str, ...]]
    stage_profiles: Mapping[str, Mapping[str, int]]
    rarity_weights: Mapping[str, int]
    generator_version: str = _V1
    # v2-only tables; both None for v1, both required for v2.
    prepared_profiles: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None
    tier_weights: Mapping[str, float] | None = None
    # Optional v2 table: {sha256, tiers} joins the payload; the loaded document does not
    # (its canonical content must hash to that sha256).
    real_boss_decks: Mapping[str, object] | None = None
    real_boss_deck_table: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.support, ScenarioSupport):
            raise ValueError("recipe requires validated finite support")
        if self.generator_version not in (_V1, _V2):
            raise ValueError("unsupported scenario generator version")
        v2_tables = (self.prepared_profiles, self.tier_weights)
        if self.generator_version == _V1 and v2_tables != (None, None):
            raise ValueError("v1 recipes do not accept prepared_profiles or tier_weights")
        if self.generator_version == _V2 and None in v2_tables:
            raise ValueError("v2 recipes require prepared_profiles and tier_weights")
        _integer(self.generator_seed, 0, 2**64 - 1, "generator_seed")
        _integer(self.shard_size, 1, 256, "shard_size")
        _integer(self.maximum_catalog_bytes, 1, 1_048_576, "maximum_catalog_bytes")
        cards = self.support.cards
        for name, bound in (("reserved_pairs", 16), ("lesson_pairs", 64)):
            pairs = getattr(self, name)
            if not isinstance(pairs, (list, tuple)) or len(pairs) > bound:
                raise ValueError(f"{name} must be a bounded pair list")
            normalized = []
            for pair in pairs:
                if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                        or not all(isinstance(c, str) and c in cards for c in pair)
                        or pair[0] == pair[1] or any(c in _STARTER for c in pair)):
                    raise ValueError("reserved/lesson pairs must contain distinct supported nonstarter cards")
                normalized.append(tuple(sorted(pair)))
            if len(set(normalized)) != len(normalized):
                raise ValueError("duplicate recipe pair")
            object.__setattr__(self, name, tuple(normalized))
        if set(self.lesson_pairs) & set(self.reserved_pairs):
            raise ValueError("a reserved pair cannot be a training lesson")
        groups = json_value(self.mechanic_groups)
        if not isinstance(groups, dict) or not 1 <= len(groups) <= 32:
            raise ValueError("mechanic groups must be a bounded mapping")
        for key, group in groups.items():
            if (not isinstance(key, str) or not key.isidentifier()
                    or not isinstance(group, list) or not 2 <= len(group) <= 5
                    or any(not isinstance(c, str) or c not in cards for c in group)
                    or len(set(group)) != len(group)
                    or any(set(pair) <= set(group) for pair in self.reserved_pairs)):
                raise ValueError("invalid or infeasible mechanic group")
        object.__setattr__(self, "mechanic_groups", _freeze_json(groups))
        profiles = json_value(self.stage_profiles)
        if not isinstance(profiles, dict) or set(profiles) != set(_STAGES):
            raise ValueError("recipe requires early/middle/developed profiles")
        for profile in profiles.values():
            if not isinstance(profile, dict) or set(profile) != {
                    "minimum_size", "maximum_size", "starter_removals", "upgrade_percent", "preferred_act"}:
                raise ValueError("invalid stage profile fields")
            _integer(profile["minimum_size"], 8, 35, "minimum_size")
            _integer(profile["maximum_size"], profile["minimum_size"], 35, "maximum_size")
            _integer(profile["starter_removals"], 0, 9, "starter_removals")
            _integer(profile["upgrade_percent"], 0, 100, "upgrade_percent")
            _integer(profile["preferred_act"], 1, 4, "preferred_act")
        object.__setattr__(self, "stage_profiles", _freeze_json(profiles))
        weights = json_value(self.rarity_weights)
        if not isinstance(weights, dict) or set(weights) != {"starter", "common", "uncommon", "rare"}:
            raise ValueError("invalid rarity weights")
        for weight in weights.values():
            _integer(weight, 1, 100, "rarity weight")
        if weights["common"] <= weights["rare"]:
            raise ValueError("run-like sampling must prefer common cards to rare cards")
        object.__setattr__(self, "rarity_weights", _freeze_json(weights))
        if len(self.category_keys) > 256:
            raise ValueError("lesson category universe exceeds 256")
        if self.generator_version == _V2:
            self._validate_v2_tables()
        if (self.real_boss_decks, self.real_boss_deck_table) != (None, None):
            self._validate_real_boss_decks()

    def _validate_v2_tables(self) -> None:
        cards = self.support.cards
        core_groups = [g for name, g in self.mechanic_groups.items() if name != "status"]
        if not core_groups:
            raise ValueError("v2 requires a non-status mechanic group")
        profiles = json_value(self.prepared_profiles)
        if (not isinstance(profiles, dict) or set(profiles) != set(_PREPARED_ACTS)
                or any(not isinstance(by_kind, dict) or set(by_kind) != set(_PREPARED_KINDS)
                       for by_kind in profiles.values())):
            raise ValueError("prepared profiles must be keyed act1..act3 by boss/elite")
        for act, by_kind in profiles.items():
            for profile in by_kind.values():
                base = {"core_cards", "starter_removals", "upgrade_percent",
                        "rarity_weights", "hp_floor_percent"}
                if not isinstance(profile, dict) or set(profile) not in (base, base | {"deck_size"}):
                    raise ValueError("invalid prepared profile fields")
                core = profile["core_cards"]
                if not isinstance(core, list) or len(core) != 2:
                    raise ValueError("core_cards must be [minimum, maximum]")
                _integer(core[0], 0, 12, "core_cards minimum")
                _integer(core[1], core[0], 12, "core_cards maximum")
                _integer(profile["starter_removals"], 0, 9, "starter_removals")
                _integer(profile["upgrade_percent"], 0, 100, "upgrade_percent")
                _integer(profile["hp_floor_percent"], 1, 100, "hp_floor_percent")
                weights = profile["rarity_weights"]
                if not isinstance(weights, dict) or set(weights) != set(_RARITIES):
                    raise ValueError("invalid prepared rarity weights")
                for weight in weights.values():
                    _integer(weight, 0, 100, "prepared rarity weight")
                # Every core group must be drawable under these weights.
                if any(sum(weights[cards[c]["rarity"]] for c in group) == 0 for group in core_groups):
                    raise ValueError("prepared rarity weights leave a core group undrawable")
                # Remaining starters, the most core cards and the coverage card must fit the stage.
                stage = self.stage_profiles[_STAGES[_PREPARED_ACTS.index(act)]]
                if len(_STARTER) - profile["starter_removals"] + core[1] + 1 > stage["maximum_size"]:
                    raise ValueError(f"prepared {act} profile cannot fit its stage maximum_size")
                if "deck_size" in profile:
                    size = profile["deck_size"]
                    if not isinstance(size, list) or len(size) != 2:
                        raise ValueError("deck_size must be [minimum, maximum]")
                    _integer(size[0], 8, stage["maximum_size"], "deck_size minimum")
                    _integer(size[1], size[0], stage["maximum_size"], "deck_size maximum")
                    if len(_STARTER) - profile["starter_removals"] + core[1] + 1 > size[1]:
                        raise ValueError(f"prepared {act} profile cannot fit its deck_size maximum")
        object.__setattr__(self, "prepared_profiles", _freeze_json(profiles))
        tiers = {f"act{e['act']}:{e['kind']}" for e in self.support.encounters.values()}
        tier_weights = json_value(self.tier_weights)
        if not isinstance(tier_weights, dict) or set(tier_weights) != tiers:
            raise ValueError("tier weights must cover exactly the support's encounter tiers")
        for weight in tier_weights.values():
            if (type(weight) not in (int, float) or not math.isfinite(weight)
                    or not 0 < weight <= 1):
                raise ValueError("tier weights must be finite in (0, 1]")
        # 1 and 1.0 are the same weight; give them the same payload and hash.
        tier_weights = {tier: float(weight) for tier, weight in tier_weights.items()}
        object.__setattr__(self, "tier_weights", _freeze_json(tier_weights))

    def _validate_real_boss_decks(self) -> None:
        if self.generator_version != _V2:
            raise ValueError("only v2 recipes accept real_boss_decks")
        if None in (self.real_boss_decks, self.real_boss_deck_table):
            raise ValueError("real_boss_decks requires its loaded deck table")
        spec = json_value(self.real_boss_decks)
        if not isinstance(spec, dict) or set(spec) != {"sha256", "tiers"}:
            raise ValueError("real_boss_decks must contain exactly sha256 and tiers")
        if not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in spec["sha256"]):
            raise ValueError("real_boss_decks sha256 must be 64 lowercase hex digits")
        tiers = spec["tiers"]
        if (not isinstance(tiers, list) or not tiers or len(set(tiers)) != len(tiers)
                or any(t not in _REAL_TIERS for t in tiers)):
            raise ValueError(f"real_boss_decks tiers must be a non-empty unique subset of {_REAL_TIERS}")
        table = json_value(self.real_boss_deck_table)
        if content_sha256(table) != spec["sha256"]:
            raise ValueError("real boss deck table SHA256 mismatch; " + _REGENERATE)
        if (not isinstance(table, dict) or set(table) != {"schema", "source", "decks"}
                or table["schema"] != _REAL_SCHEMA or not isinstance(table["source"], str)
                or not isinstance(table["decks"], list) or not 1 <= len(table["decks"]) <= 100_000):
            raise ValueError("invalid real boss deck table schema; " + _REGENERATE)
        cards, encounters = self.support.cards, self.support.encounters
        bosses = {e for e, row in encounters.items() if f"act{row['act']}:{row['kind']}" in _REAL_TIERS}
        for deck in table["decks"]:
            if (not isinstance(deck, dict) or set(deck) != _REAL_DECK_FIELDS
                    or not isinstance(deck["run_id"], str) or deck["encounter"] not in bosses
                    or type(deck["floor"]) is not int or type(deck["won"]) is not bool
                    or type(deck["ascension"]) is not int
                    or type(deck["dropped_cards"]) is not int or deck["dropped_cards"] < 0
                    or type(deck["hp_fraction"]) not in (int, float)
                    or not 0 < deck["hp_fraction"] <= 1
                    or not isinstance(deck["deck"], list) or not 1 <= len(deck["deck"]) <= 100
                    or any(not isinstance(row, dict) or set(row) != {"card", "upgrades"}
                           or row["card"] not in cards or type(row["upgrades"]) is not int
                           or not 0 <= row["upgrades"] <= cards[row["card"]]["maximum_upgrades"]
                           for row in deck["deck"])
                    or not isinstance(deck["relics"], list) or len(deck["relics"]) > 29
                    or len(set(deck["relics"])) != len(deck["relics"])
                    or any(r not in self.support.relics or r == "neows_lament" for r in deck["relics"])
                    or {"burning_blood", "black_blood"} <= set(deck["relics"])):
                raise ValueError("invalid real boss deck row; " + _REGENERATE)
        won = Counter(d["encounter"] for d in table["decks"] if d["won"])
        for encounter in sorted(bosses):
            row = encounters[encounter]
            if f"act{row['act']}:{row['kind']}" in tiers and not won[encounter]:
                raise ValueError(f"real boss deck table has no won deck for {encounter}")
        object.__setattr__(self, "real_boss_decks", _freeze_json(
            {"sha256": spec["sha256"], "tiers": [t for t in _REAL_TIERS if t in tiers]}))
        object.__setattr__(self, "real_boss_deck_table", _freeze_json(table))

    @property
    def category_keys(self) -> tuple[str, ...]:
        return (tuple(f"card:{c}" for c in sorted(self.support.cards))
                + tuple(f"pair:{a}+{b}" for a, b in self.lesson_pairs)
                + tuple(f"group:{g}" for g in sorted(self.mechanic_groups)))

    @property
    def payload(self) -> Mapping[str, object]:
        # v2 tables join the payload only for v2, so v1 hashes stay unchanged.
        v2 = ("prepared_profiles", "tier_weights") if self.generator_version == _V2 else ()
        # Only when present, so v2 recipes without real decks keep their bytes.
        v2 += ("real_boss_decks",) if self.real_boss_decks is not None else ()
        return _freeze_json({"schema": "spire-learn-scenario-recipe-v1",
            "generator_version": self.generator_version, "support_sha256": self.support.sha256,
            **{k: getattr(self, k) for k in ("generator_seed", "shard_size", "maximum_catalog_bytes",
               "reserved_pairs", "lesson_pairs", "mechanic_groups", "stage_profiles", "rarity_weights",
               *v2)}})

    @property
    def sha256(self) -> str:
        return content_sha256(self.payload)


def load_scenario_recipe(path: Path) -> ScenarioRecipe:
    raw = read_json(path)
    fields = {"schema", "generator_version", "support_file", "support_sha256", "generator_seed",
              "shard_size", "maximum_catalog_bytes", "reserved_pairs", "lesson_pairs",
              "mechanic_groups", "stage_profiles", "rarity_weights"}
    if raw.get("generator_version") == _V2:
        fields |= {"prepared_profiles", "tier_weights"}
        if "real_boss_decks" in raw:
            fields.add("real_boss_decks")
    if set(raw) != fields or raw["schema"] != "spire-learn-scenario-recipe-v1":
        raise ValueError("recipe has missing, unknown or unsupported schema fields")
    if not isinstance(raw["support_file"], str) or not raw["support_file"]:
        raise ValueError("recipe support_file must be a path")
    support = load_scenario_support(Path(path).resolve().parent / raw["support_file"])
    if raw["support_sha256"] != support.sha256:
        raise ValueError("recipe support SHA256 mismatch")
    real = {}
    if "real_boss_decks" in raw:
        spec = raw["real_boss_decks"]
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256", "tiers"}:
            raise ValueError("recipe real_boss_decks must contain exactly path, sha256 and tiers")
        if not isinstance(spec["path"], str) or not spec["path"]:
            raise ValueError("recipe real_boss_decks path must be a path")
        # Resolved like support_file; the table is a local, regenerable artifact.
        table_path = Path(path).resolve().parent / spec["path"]
        if not table_path.is_file():
            raise ValueError(f"real boss deck table is missing: {table_path}; " + _REGENERATE)
        data = table_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"real boss deck table SHA256 mismatch: {table_path}; " + _REGENERATE)
        real = {"real_boss_decks": {"sha256": spec["sha256"], "tiers": spec["tiers"]},
                "real_boss_deck_table": read_json(table_path)}
    return ScenarioRecipe(support=support, **real, **{k: v for k, v in raw.items()
                          if k not in {"schema", "support_file", "support_sha256", "real_boss_decks"}})


@dataclass(frozen=True)
class ScenarioShard:
    manifest: Mapping[str, object]
    catalog: Mapping[str, object]
    annotations: Mapping[str, Mapping[str, object]]
    shard_index: int
    arm: str
    manifest_sha256: str
    distribution_sha256: str

    def __post_init__(self) -> None:
        for name in ("manifest", "catalog", "annotations"):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))
        _integer(self.shard_index, 0, 2**24 - 1, "shard_index")
        if self.arm not in _ARMS or self.manifest_sha256 != content_sha256(self.manifest):
            raise ValueError("invalid shard identity")


def _rng(recipe: ScenarioRecipe, split: str, arm: str, index: int, purpose: str) -> random.Random:
    # No use of module-global, actor, optimizer or native/environment RNG state.
    seed = content_sha256([recipe.sha256, split, arm, index, purpose])
    return random.Random(int(seed, 16))


def _composition_split(deck: Sequence[Mapping[str, object]]) -> str:
    """Seed-independent composition assignment for novel-deck cohorts only.

    Familiar/new-seed or resource diagnostics may reuse decks in a separately
    labeled authored cohort. This stream makes no historical-pretraining claim.
    """
    key = sorted((row["card"], row["upgrades"]) for row in deck)
    bucket = int(content_sha256(["composition-partition-v1", key]), 16) % 10
    return "train" if bucket < 8 else "selection" if bucket == 8 else "final"


def _quotas(size: int, shard_index: int, arm: str) -> tuple[int, int, int]:
    if arm != "mixed":
        return size, 0, 0
    # Fractional carry in the four-shard cycle. Full shard is exactly 179/51/26.
    offset = (shard_index % 4) * size
    run = ((offset + size) * 179 // 256) - (offset * 179 // 256)
    lesson = ((offset + size) * 51 // 256) - (offset * 51 // 256)
    return run, lesson, size - run - lesson


def _build(recipe: ScenarioRecipe, split: str, shard_index: int, arm: str, size: int) -> ScenarioShard:
    cards = sorted(recipe.support.cards)
    card_metadata = recipe.support.cards
    encounters = tuple(recipe.support.encounters.values())
    weights = [recipe.rarity_weights[card_metadata[c]["rarity"]] for c in cards]
    n_run, n_lesson, _ = _quotas(size, shard_index, arm)
    single_count = (n_lesson + 1) // 2
    pair_count = n_lesson * 3 // 10
    cases, entries, annotations = [], [], {}
    seen_compositions = set()
    distribution = content_sha256({"recipe": recipe.payload, "split": split, "arm": arm,
                                    "composition_rule": "novel-deck-partition-v1"})
    for local in range(size):
        family = "run_like" if local < n_run else "lesson" if local < n_run + n_lesson else "unusual"
        stage = _STAGES[(local + shard_index) % 3]
        profile = recipe.stage_profiles[stage]
        lesson_index = local - n_run
        kind, category, targets = None, None, ()
        if family == "lesson":
            if lesson_index < single_count:
                kind = "single"
                targets = (cards[(shard_index * 26 + lesson_index) % len(cards)],)
                category = "card:" + targets[0]
            elif lesson_index < single_count + pair_count and recipe.lesson_pairs:
                kind = "pair"
                targets = recipe.lesson_pairs[(shard_index * 15 + lesson_index - single_count) % len(recipe.lesson_pairs)]
                category = "pair:" + "+".join(targets)
            else:
                kind = "group"
                group = sorted(recipe.mechanic_groups)[(shard_index * 10 + lesson_index) % len(recipe.mechanic_groups)]
                targets = recipe.mechanic_groups[group]
                category = "group:" + group
        # Spread the 150 mandatory base/ordinary-upgraded targets over 4 shards.
        # The first 38 entries of each shard cover distinct forms; incidental
        # cards add breadth but never substitute for explicit coverage checks.
        coverage_ordinal = (shard_index % 4) * 38 + local
        coverage_card = cards[(coverage_ordinal // 2) % len(cards)]
        coverage_upgrade = coverage_ordinal % 2
        encounter_rng = _rng(recipe, split, arm, shard_index, f"encounter:{local}")
        if arm == "fixed":
            encounter_id = _FIXED_ENCOUNTERS[(shard_index * size + local) % 13]
        elif local < 11:
            encounter_id = encounters[((shard_index % 4) * 11 + local) % 43]["id"]
        else:
            # Coarse stage affinity, with 20% difficult/off-stage coverage.
            pool = [e for e in encounters if e["act"] == profile["preferred_act"]]
            encounter_id = encounter_rng.choice(encounters if encounter_rng.randrange(5) == 0 else pool)["id"]
        encounter = recipe.support.encounters[encounter_id]
        accepted = None
        deck_rng = _rng(recipe, split, arm, shard_index, f"deck:{local}")
        resource_rng = _rng(recipe, split, arm, shard_index, f"resources:{local}")
        for _attempt in range(_ATTEMPTS):
            if arm == "fixed":
                deck = [{"card": c, "upgrades": 0} for c in _STARTER]
                template = (shard_index * size + local) % 3
                if template == 1:
                    deck += [{"card": "warcry", "upgrades": 0}, {"card": "whirlwind", "upgrades": 0}]
                elif template == 2:
                    deck += [{"card": "immolate", "upgrades": 1}, {"card": "armaments", "upgrades": 1}]
            else:
                if family == "unusual":
                    deck = []
                    desired_size = deck_rng.randint(5, 40)
                else:
                    starter = list(_STARTER)
                    for _ in range(deck_rng.randint(0, profile["starter_removals"])):
                        starter.pop(deck_rng.randrange(len(starter)))
                    deck = [{"card": c, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])}
                            for c in starter]
                    desired_size = (max(8, len(deck) + len(targets)) if family == "lesson"
                                    else deck_rng.randint(profile["minimum_size"], profile["maximum_size"]))
                deck += [{"card": c, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])}
                         for c in targets]
                if local < 38:
                    # Replace an existing matching form only if already present.
                    row = {"card": coverage_card, "upgrades": coverage_upgrade}
                    if row not in deck:
                        deck.append(row)
                # Occasionally prefer a coherent annotated group, without
                # implementing any card effects or inferring fight solvability.
                coherent = recipe.mechanic_groups[deck_rng.choice(sorted(recipe.mechanic_groups))]
                while len(deck) < desired_size:
                    card = (deck_rng.choice(coherent) if deck_rng.randrange(6) == 0
                            else deck_rng.choices(cards, weights=weights, k=1)[0])
                    deck.append({"card": card, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])})
                ids = {row["card"] for row in deck}
                if split == "train" and any(set(pair) <= ids for pair in recipe.reserved_pairs):
                    continue
                if _composition_split(deck) != split:
                    continue
            deck.sort(key=lambda row: (row["card"], row["upgrades"]))
            composition = content_sha256(deck)
            if arm != "fixed" and composition in seen_compositions:
                continue
            if arm == "fixed":
                hp, max_hp = (80 if local % 2 == 0 else 64), 80
                relics, potions = ["burning_blood"], [None, None, None]
            else:
                max_hp = resource_rng.randint(80, 120)
                hp = resource_rng.randint(24, max_hp)
                available = [r for r in recipe.support.relics if r not in
                             {"burning_blood", "black_blood", "neows_lament"}]
                relics = ["burning_blood"] + resource_rng.sample(available, resource_rng.randrange(4))
                potions = [resource_rng.choice(recipe.support.potions) if resource_rng.randrange(3) == 0 else None
                           for _ in range(3)]
            lineage = "deck-" + composition
            identity = content_sha256([distribution, shard_index, local, composition])
            entry_id = "s-" + identity[:48]
            entry = {"id": entry_id, "encounter": encounter_id, "hp": hp, "max_hp": max_hp,
                     "deck": deck, "relics": sorted(relics), "potions": potions, "relic_counters": {}}
            recipe.support.validate_entry(entry)
            case = {"case_id": entry_id, "lineage_id": lineage if arm != "fixed" else lineage + f"-{local}",
                    "mode": "standalone", "environment_seed": environment_seed_for(
                        split, replica=0, ordinal=shard_index * 256 + local),
                    "cohort": "fixed_control" if arm == "fixed" else "novel_composition",
                    "entry_id": entry_id, "entry": entry}
            _parse_case(case, split=split)
            accepted = case
            seen_compositions.add(composition)
            break
        if accepted is None:
            raise ValueError(f"scenario entry {local} exhausted 4096 bounded construction attempts")
        cases.append(accepted)
        entries.append(accepted["entry"])
        annotations[accepted["case_id"]] = {
            "family": "fixed" if arm == "fixed" else family, "deck_stage": stage,
            "lesson_kind": kind, "category": category, "target_card_ids": targets,
            "encounter_tier": f"act{encounter['act']}:{encounter['kind']}",
            "lineage_id": accepted["lineage_id"], "composition_sha256": composition,
            "composition_scope": "seen_composition_control" if arm == "fixed" else "novel_composition",
        }
    manifest = {"schema": "spire-learn-neural-cases-v1", "split": split, "partition": split, "cases": cases}
    return ScenarioShard(manifest, {"entries": entries}, annotations, shard_index, arm,
                         content_sha256(manifest), distribution)


def _prepared_deck(recipe: ScenarioRecipe, prepared: Mapping[str, object], stage: Mapping[str, int],
                   rng: random.Random, targets: Sequence[str],
                   coverage: Mapping[str, object] | None) -> list[dict[str, object]]:
    """A boss/elite deck built around one core mechanic group, replacing the random fill."""
    metadata = recipe.support.cards
    rarity = prepared["rarity_weights"]
    upgrade = lambda: int(rng.randrange(100) < prepared["upgrade_percent"])
    # _STARTER lists Strikes first, so slicing removes Strikes before other starters.
    deck = [{"card": c, "upgrades": upgrade()} for c in _STARTER[prepared["starter_removals"]:]]
    group = recipe.mechanic_groups[rng.choice(sorted(g for g in recipe.mechanic_groups if g != "status"))]
    core = rng.choices(group, weights=[rarity[metadata[c]["rarity"]] for c in group],
                       k=rng.randint(*prepared["core_cards"]))
    deck += [{"card": c, "upgrades": upgrade()} for c in (*core, *targets)]
    if coverage is not None and coverage not in deck:
        deck.append(dict(coverage))
    cards = sorted(metadata)
    fill_weights = [rarity[metadata[c]["rarity"]] for c in cards]
    desired_size = rng.randint(*prepared.get("deck_size", (stage["minimum_size"], stage["maximum_size"])))
    while len(deck) < desired_size:
        deck.append({"card": rng.choices(cards, weights=fill_weights, k=1)[0], "upgrades": upgrade()})
    return deck


def _real_deck(prepared: Mapping[str, object], real: Mapping[str, object], rng: random.Random,
               targets: Sequence[str], coverage: Mapping[str, object] | None) -> list[dict[str, object]]:
    """A recorded boss deck, plus lesson targets and the coverage card as the prepared path adds them."""
    deck = [dict(row) for row in real["deck"]]
    deck += [{"card": c, "upgrades": int(rng.randrange(100) < prepared["upgrade_percent"])} for c in targets]
    if coverage is not None and coverage not in deck:
        deck.append(dict(coverage))
    return deck


def _build_v2(recipe: ScenarioRecipe, split: str, shard_index: int, arm: str, size: int) -> ScenarioShard:
    """Curriculum v2: the encounter is drawn first and sets the stage; boss/elite
    run-like and lesson entries get prepared decks. Copies the v1 loop so that the
    v1 builder's RNG draw order, and so its bytes, stay untouched."""
    cards = sorted(recipe.support.cards)
    card_metadata = recipe.support.cards
    encounters = tuple(recipe.support.encounters.values())
    weights = [recipe.rarity_weights[card_metadata[c]["rarity"]] for c in cards]
    encounter_weights = [recipe.tier_weights[f"act{e['act']}:{e['kind']}"] for e in encounters]
    n_run, n_lesson, _ = _quotas(size, shard_index, arm)
    single_count = (n_lesson + 1) // 2
    pair_count = n_lesson * 3 // 10
    cases, entries, annotations = [], [], {}
    seen_compositions = set()
    distribution = content_sha256({"recipe": recipe.payload, "split": split, "arm": arm,
                                    "composition_rule": "novel-deck-partition-v1"})
    real_tiers = recipe.real_boss_decks["tiers"] if recipe.real_boss_decks is not None else ()
    real_pools: dict[str, list[Mapping[str, object]]] = {}
    for real_row in (recipe.real_boss_deck_table["decks"] if real_tiers else ()):
        if real_row["won"]:
            real_pools.setdefault(real_row["encounter"], []).append(real_row)
    for local in range(size):
        family = "run_like" if local < n_run else "lesson" if local < n_run + n_lesson else "unusual"
        lesson_index = local - n_run
        kind, category, targets = None, None, ()
        if family == "lesson":
            if lesson_index < single_count:
                kind = "single"
                targets = (cards[(shard_index * 26 + lesson_index) % len(cards)],)
                category = "card:" + targets[0]
            elif lesson_index < single_count + pair_count and recipe.lesson_pairs:
                kind = "pair"
                targets = recipe.lesson_pairs[(shard_index * 15 + lesson_index - single_count) % len(recipe.lesson_pairs)]
                category = "pair:" + "+".join(targets)
            else:
                kind = "group"
                group = sorted(recipe.mechanic_groups)[(shard_index * 10 + lesson_index) % len(recipe.mechanic_groups)]
                targets = recipe.mechanic_groups[group]
                category = "group:" + group
        coverage_ordinal = (shard_index % 4) * 38 + local
        coverage_card = cards[(coverage_ordinal // 2) % len(cards)]
        coverage_upgrade = coverage_ordinal % 2
        encounter_rng = _rng(recipe, split, arm, shard_index, f"v2-encounter:{local}")
        if arm == "fixed":
            encounter_id = _FIXED_ENCOUNTERS[(shard_index * size + local) % 13]
        elif local < 11:
            encounter_id = encounters[((shard_index % 4) * 11 + local) % 43]["id"]
        else:
            # v1's selection rule (stage-cycle act pool, 20% any act), with tier
            # weights multiplying each candidate, so v2's tier mix stays v1-like.
            cycle_act = recipe.stage_profiles[_STAGES[(local + shard_index) % 3]]["preferred_act"]
            pool = (list(range(len(encounters))) if encounter_rng.randrange(5) == 0
                    else [i for i, e in enumerate(encounters) if e["act"] == cycle_act])
            chosen = encounter_rng.choices(pool, weights=[encounter_weights[i] for i in pool], k=1)[0]
            encounter_id = encounters[chosen]["id"]
        encounter = recipe.support.encounters[encounter_id]
        # Stage follows the encounter: act 1 early, act 2 middle, acts 3-4 developed.
        act = min(encounter["act"], 3)
        stage = _STAGES[act - 1]
        profile = recipe.stage_profiles[stage]
        prepared = (recipe.prepared_profiles[f"act{act}"][encounter["kind"]]
                    if arm != "fixed" and family in ("run_like", "lesson")
                    and encounter["kind"] in _PREPARED_KINDS else None)
        # Boss cases in a listed tier draw a won recorded deck instead of a prepared one.
        real_pool = (real_pools[encounter_id] if prepared is not None
                     and f"act{encounter['act']}:{encounter['kind']}" in real_tiers else None)
        real = None
        accepted = None
        deck_rng = _rng(recipe, split, arm, shard_index, f"v2-deck:{local}")
        resource_rng = _rng(recipe, split, arm, shard_index, f"v2-resources:{local}")
        for _attempt in range(_ATTEMPTS):
            if arm == "fixed":
                deck = [{"card": c, "upgrades": 0} for c in _STARTER]
                template = (shard_index * size + local) % 3
                if template == 1:
                    deck += [{"card": "warcry", "upgrades": 0}, {"card": "whirlwind", "upgrades": 0}]
                elif template == 2:
                    deck += [{"card": "immolate", "upgrades": 1}, {"card": "armaments", "upgrades": 1}]
            else:
                if real_pool is not None:
                    real = deck_rng.choice(real_pool)
                    deck = _real_deck(prepared, real, deck_rng, targets,
                                      {"card": coverage_card, "upgrades": coverage_upgrade}
                                      if local < 38 else None)
                elif prepared is not None:
                    deck = _prepared_deck(recipe, prepared, profile, deck_rng, targets,
                                          {"card": coverage_card, "upgrades": coverage_upgrade}
                                          if local < 38 else None)
                    if len(deck) > profile["maximum_size"]:
                        # Lesson targets can push past the validated mandatory part; never emit it.
                        raise ValueError(f"prepared deck for scenario entry {local} ({encounter_id}, "
                                         f"{split} shard {shard_index}) has {len(deck)} cards, over "
                                         f"the {stage} maximum_size {profile['maximum_size']}")
                else:
                    if family == "unusual":
                        deck = []
                        desired_size = deck_rng.randint(5, 40)
                    else:
                        starter = list(_STARTER)
                        for _ in range(deck_rng.randint(0, profile["starter_removals"])):
                            starter.pop(deck_rng.randrange(len(starter)))
                        deck = [{"card": c, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])}
                                for c in starter]
                        desired_size = (max(8, len(deck) + len(targets)) if family == "lesson"
                                        else deck_rng.randint(profile["minimum_size"], profile["maximum_size"]))
                    deck += [{"card": c, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])}
                             for c in targets]
                    if local < 38:
                        row = {"card": coverage_card, "upgrades": coverage_upgrade}
                        if row not in deck:
                            deck.append(row)
                    coherent = recipe.mechanic_groups[deck_rng.choice(sorted(recipe.mechanic_groups))]
                    while len(deck) < desired_size:
                        card = (deck_rng.choice(coherent) if deck_rng.randrange(6) == 0
                                else deck_rng.choices(cards, weights=weights, k=1)[0])
                        deck.append({"card": card, "upgrades": int(deck_rng.randrange(100) < profile["upgrade_percent"])})
                ids = {row["card"] for row in deck}
                if split == "train" and any(set(pair) <= ids for pair in recipe.reserved_pairs):
                    continue
                if _composition_split(deck) != split:
                    continue
            deck.sort(key=lambda row: (row["card"], row["upgrades"]))
            composition = content_sha256(deck)
            if arm != "fixed" and composition in seen_compositions:
                continue
            if arm == "fixed":
                hp, max_hp = (80 if local % 2 == 0 else 64), 80
                relics, potions = ["burning_blood"], [None, None, None]
            elif real is not None:
                max_hp = resource_rng.randint(80, 120)
                hp = max(1, round(real["hp_fraction"] * max_hp))
                relics = list(real["relics"])
                if not {"burning_blood", "black_blood"} & set(relics):
                    relics.append("burning_blood")
                potions = [resource_rng.choice(recipe.support.potions) if resource_rng.randrange(3) == 0 else None
                           for _ in range(3)]
            else:
                max_hp = resource_rng.randint(80, 120)
                hp = resource_rng.randint(24, max_hp)
                if prepared is not None:
                    hp = max(hp, -(-max_hp * prepared["hp_floor_percent"] // 100))
                available = [r for r in recipe.support.relics if r not in
                             {"burning_blood", "black_blood", "neows_lament"}]
                relics = ["burning_blood"] + resource_rng.sample(available, resource_rng.randrange(4))
                potions = [resource_rng.choice(recipe.support.potions) if resource_rng.randrange(3) == 0 else None
                           for _ in range(3)]
            lineage = "deck-" + composition
            identity = content_sha256([distribution, shard_index, local, composition])
            entry_id = "s-" + identity[:48]
            entry = {"id": entry_id, "encounter": encounter_id, "hp": hp, "max_hp": max_hp,
                     "deck": deck, "relics": sorted(relics), "potions": potions, "relic_counters": {}}
            recipe.support.validate_entry(entry)
            case = {"case_id": entry_id, "lineage_id": lineage if arm != "fixed" else lineage + f"-{local}",
                    "mode": "standalone", "environment_seed": environment_seed_for(
                        split, replica=0, ordinal=shard_index * 256 + local),
                    "cohort": "fixed_control" if arm == "fixed" else "novel_composition",
                    "entry_id": entry_id, "entry": entry}
            _parse_case(case, split=split)
            accepted = case
            seen_compositions.add(composition)
            break
        if accepted is None:
            raise ValueError(f"scenario entry {local} exhausted 4096 bounded construction attempts")
        cases.append(accepted)
        entries.append(accepted["entry"])
        annotations[accepted["case_id"]] = {
            "family": "fixed" if arm == "fixed" else family, "deck_stage": stage,
            "lesson_kind": kind, "category": category, "target_card_ids": targets,
            "encounter_tier": f"act{encounter['act']}:{encounter['kind']}",
            "lineage_id": accepted["lineage_id"], "composition_sha256": composition,
            "composition_scope": "seen_composition_control" if arm == "fixed" else "novel_composition",
        }
    manifest = {"schema": "spire-learn-neural-cases-v1", "split": split, "partition": split, "cases": cases}
    return ScenarioShard(manifest, {"entries": entries}, annotations, shard_index, arm,
                         content_sha256(manifest), distribution)


# Shards are pure functions of the recipe content hash, split, index and arm, and
# are deeply frozen, so a generated shard can be reused within a process. Resume
# and checkpoint provenance checks regenerate the same few shards many times.
_SHARD_CACHE_CAPACITY = 16
_SHARD_CACHE: OrderedDict[tuple[str, str, int, str], tuple[ScenarioShard, tuple[object, ...]]] = OrderedDict()


def _shard_fields(shard: ScenarioShard) -> tuple[object, ...]:
    return tuple(getattr(shard, field.name) for field in fields(shard))
_SHARD_CACHE_LOCK = threading.Lock()


def generate_scenario_shard(recipe: ScenarioRecipe, *, split: str, shard_index: int,
                            arm: str = "mixed") -> ScenarioShard:
    if not isinstance(recipe, ScenarioRecipe):
        raise ValueError("recipe must be a ScenarioRecipe")
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("unsupported scenario split")
    if not isinstance(arm, str) or arm not in _ARMS:
        raise ValueError("unsupported scenario arm")
    _integer(shard_index, 0, 2**24 - 1, "shard_index")
    if type(recipe) is not ScenarioRecipe or type(recipe.support) is not ScenarioSupport:
        # Subclasses could change generation inputs without changing the content hash.
        return _generate_scenario_shard(recipe, split=split, shard_index=shard_index, arm=arm)
    # Keyed by the recipe's current content hash, so a changed recipe never hits.
    key = (recipe.sha256, split, shard_index, arm)
    with _SHARD_CACHE_LOCK:
        cached = _SHARD_CACHE.get(key)
        if cached is not None:
            shard, stored = cached
            # A frozen record can still be rebound with object.__setattr__; never reuse one that was.
            if all(now is then for now, then in zip(_shard_fields(shard), stored)):
                _SHARD_CACHE.move_to_end(key)
                return shard
            del _SHARD_CACHE[key]
    shard = _generate_scenario_shard(recipe, split=split, shard_index=shard_index, arm=arm)
    with _SHARD_CACHE_LOCK:
        _SHARD_CACHE[key] = (shard, _shard_fields(shard))
        while len(_SHARD_CACHE) > _SHARD_CACHE_CAPACITY:
            _SHARD_CACHE.popitem(last=False)
    return shard


def _generate_scenario_shard(recipe: ScenarioRecipe, *, split: str, shard_index: int,
                             arm: str) -> ScenarioShard:
    # A conservative public-input upper bound gives every shard the same
    # capacity, so byte reduction cannot reset the fractional family carry.
    # Generated decks have <=40 rows, <=4 relics and exactly three potion slots.
    longest = lambda values: max(values, key=lambda value: len(canonical_bytes(value)))
    deck_rows, relic_rows, upgrades = 40, 4, 1
    if recipe.real_boss_decks is not None:
        # Recorded decks can be longer and carry more relics and Searing Blow upgrades;
        # add the largest lesson target set and the coverage card.
        real = recipe.real_boss_deck_table["decks"]
        deck_rows = max(deck_rows, max(len(d["deck"]) for d in real)
                        + max(len(g) for g in recipe.mechanic_groups.values()) + 1)
        relic_rows = max(relic_rows, max(len(d["relics"]) for d in real) + 1)
        upgrades = 10
    worst = {"id": "s-" + "0" * 48, "encounter": longest(recipe.support.encounters),
             "hp": 120, "max_hp": 120,
             "deck": [{"card": longest(recipe.support.cards), "upgrades": upgrades}] * deck_rows,
             "relics": [longest(recipe.support.relics)] * relic_rows,
             "potions": [longest(recipe.support.potions)] * 3, "relic_counters": {}}
    capacity = max(0, (recipe.maximum_catalog_bytes - len(canonical_bytes({"entries": []})))
                   // (len(canonical_bytes(worst)) + 1))
    # The fixed control is the complete old 3 decks x 13 encounters x 2 HP
    # settings, uniformly once each; expanding to 256 would bias that control.
    size = min(recipe.shard_size, capacity, 78 if arm == "fixed" else 256)
    if not size:
        raise ValueError("no scenario entry can fit the catalog byte bound")
    build = _build_v2 if recipe.generator_version == _V2 else _build
    shard = build(recipe, split, shard_index, arm, size)
    if len(canonical_bytes(shard.catalog)) > recipe.maximum_catalog_bytes:
        raise ValueError("scenario catalog exceeds byte bound")
    return shard


def validate_coverage_cycle(shards: Sequence[ScenarioShard], recipe: ScenarioRecipe) -> Mapping[str, object]:
    if (not isinstance(shards, Sequence) or len(shards) != 4
            or any(not isinstance(s, ScenarioShard) for s in shards)):
        raise ValueError("coverage requires exactly four shards")
    indices = [s.shard_index for s in shards]
    if indices != list(range(indices[0], indices[0] + 4)) or indices[0] % 4:
        raise ValueError("coverage cycle must be aligned consecutive four shards")
    if len({s.distribution_sha256 for s in shards}) != 1 or len({s.arm for s in shards}) != 1:
        raise ValueError("coverage shards must share distribution and arm")
    split, arm = shards[0].manifest["split"], shards[0].arm
    entries = []
    for shard in shards:
        # Reject fabricated/tampered payloads, wrong recipe and stale identities.
        expected = generate_scenario_shard(recipe, split=split, shard_index=shard.shard_index, arm=arm)
        if (shard.manifest_sha256 != expected.manifest_sha256
                or canonical_bytes(shard.catalog) != canonical_bytes(expected.catalog)
                or canonical_bytes(shard.annotations) != canonical_bytes(expected.annotations)
                or shard.distribution_sha256 != expected.distribution_sha256):
            raise ValueError("coverage shard differs from frozen recipe")
        entries.extend(shard.catalog["entries"])
    forms = {(c["card"], c["upgrades"]) for e in entries for c in e["deck"]}
    encounter_ids = {e["encounter"] for e in entries}
    required = {(c, u) for c in recipe.support.cards for u in (0, 1)}
    broad = required <= forms and set(recipe.support.encounters) <= encounter_ids
    if arm != "fixed" and not broad:
        raise ValueError("coverage cycle lacks supported base/upgraded cards or encounters")
    return _freeze_json({"schema": "spire-learn-scenario-coverage-v1", "broad_coverage": broad,
        "arm": arm, "split": split, "exposure": "generated_only", "entry_count": len(entries),
        "card_count": len({c for c, _ in forms}), "base_upgraded_count": len(required & forms),
        "encounter_count": len(encounter_ids),
        "family_counts": dict(Counter(a["family"] for s in shards for a in s.annotations.values())),
        "pair_count": len({tuple(sorted((a, b))) for e in entries
            for a in {c["card"] for c in e["deck"]} for b in {c["card"] for c in e["deck"]} if a != b}),
        "composition_scope": "seen_composition_control" if arm == "fixed" else "novel_composition",
        "historical_pretraining_exposure": "unknown"})


__all__ = ["ScenarioRecipe", "ScenarioShard", "load_scenario_recipe", "generate_scenario_shard",
           "validate_coverage_cycle"]
