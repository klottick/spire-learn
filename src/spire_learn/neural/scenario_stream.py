"""Deterministic bounded scenario construction, without gameplay simulation.

Public resources reach the catalog; generator seeds/labels remain side metadata.
The recipe describes learner sampling heuristics, not a true Run distribution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import random

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
    generator_version: str = "ironclad-curriculum-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.support, ScenarioSupport):
            raise ValueError("recipe requires validated finite support")
        if self.generator_version != "ironclad-curriculum-v1":
            raise ValueError("unsupported scenario generator version")
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

    @property
    def category_keys(self) -> tuple[str, ...]:
        return (tuple(f"card:{c}" for c in sorted(self.support.cards))
                + tuple(f"pair:{a}+{b}" for a, b in self.lesson_pairs)
                + tuple(f"group:{g}" for g in sorted(self.mechanic_groups)))

    @property
    def payload(self) -> Mapping[str, object]:
        return _freeze_json({"schema": "spire-learn-scenario-recipe-v1",
            "generator_version": self.generator_version, "support_sha256": self.support.sha256,
            **{k: getattr(self, k) for k in ("generator_seed", "shard_size", "maximum_catalog_bytes",
               "reserved_pairs", "lesson_pairs", "mechanic_groups", "stage_profiles", "rarity_weights")}})

    @property
    def sha256(self) -> str:
        return content_sha256(self.payload)


def load_scenario_recipe(path: Path) -> ScenarioRecipe:
    raw = read_json(path)
    fields = {"schema", "generator_version", "support_file", "support_sha256", "generator_seed",
              "shard_size", "maximum_catalog_bytes", "reserved_pairs", "lesson_pairs",
              "mechanic_groups", "stage_profiles", "rarity_weights"}
    if set(raw) != fields or raw["schema"] != "spire-learn-scenario-recipe-v1":
        raise ValueError("recipe has missing, unknown or unsupported schema fields")
    if not isinstance(raw["support_file"], str) or not raw["support_file"]:
        raise ValueError("recipe support_file must be a path")
    support = load_scenario_support(Path(path).resolve().parent / raw["support_file"])
    if raw["support_sha256"] != support.sha256:
        raise ValueError("recipe support SHA256 mismatch")
    return ScenarioRecipe(support=support, **{k: v for k, v in raw.items()
                          if k not in {"schema", "support_file", "support_sha256"}})


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


def generate_scenario_shard(recipe: ScenarioRecipe, *, split: str, shard_index: int,
                            arm: str = "mixed") -> ScenarioShard:
    if not isinstance(recipe, ScenarioRecipe):
        raise ValueError("recipe must be a ScenarioRecipe")
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("unsupported scenario split")
    if not isinstance(arm, str) or arm not in _ARMS:
        raise ValueError("unsupported scenario arm")
    _integer(shard_index, 0, 2**24 - 1, "shard_index")
    # A conservative public-input upper bound gives every shard the same
    # capacity, so byte reduction cannot reset the fractional family carry.
    # Generated decks have <=40 rows, <=4 relics and exactly three potion slots.
    longest = lambda values: max(values, key=lambda value: len(canonical_bytes(value)))
    worst = {"id": "s-" + "0" * 48, "encounter": longest(recipe.support.encounters),
             "hp": 120, "max_hp": 120,
             "deck": [{"card": longest(recipe.support.cards), "upgrades": 1}] * 40,
             "relics": [longest(recipe.support.relics)] * 4,
             "potions": [longest(recipe.support.potions)] * 3, "relic_counters": {}}
    capacity = max(0, (recipe.maximum_catalog_bytes - len(canonical_bytes({"entries": []})))
                   // (len(canonical_bytes(worst)) + 1))
    # The fixed control is the complete old 3 decks x 13 encounters x 2 HP
    # settings, uniformly once each; expanding to 256 would bias that control.
    size = min(recipe.shard_size, capacity, 78 if arm == "fixed" else 256)
    if not size:
        raise ValueError("no scenario entry can fit the catalog byte bound")
    shard = _build(recipe, split, shard_index, arm, size)
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
