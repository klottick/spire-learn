"""Finite, checkpointable lesson allocation over the current public catalog.

Allocation counters measure choices, not accepted native resets. Collector's
reset delivery diagnostics remain the authority for actual exposure. Only
natural training settlements may be supplied to ``observe``. No trajectories,
policy tensors, RNG state, or historical case registry are retained here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import random
import re

from spire_learn.neural.cases import NeuralCase
from spire_learn.neural.scenario_support import content_sha256


_SCHEMA = "spire-learn-curriculum-v1"
_MAX_COUNT = 2**63 - 1
_FAMILIES = ("run_like", "lesson", "unusual", "fixed")
# Fixed cycles preserve the declared proportions without outcome-dependent drift.
_FAMILY_CYCLE = ("run_like", "run_like", "lesson", "run_like", "run_like",
                 "unusual", "run_like", "lesson", "run_like", "run_like")
_LESSON_CYCLE = ("single", "pair", "single", "group", "single",
                 "pair", "single", "group", "single", "pair")
_ANNOTATION_KEYS = {"family", "deck_stage", "lesson_kind", "category", "target_card_ids",
                    "encounter_tier", "lineage_id", "composition_sha256", "composition_scope"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_KEY = re.compile(r"(?:card|pair|group):[A-Za-z0-9_.:+-]{1,120}\Z")


def _count(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError("curriculum counts must be bounded non-negative primitive integers")
    return value


def _increment(value: int) -> int:
    return min(_MAX_COUNT, value + 1)


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _cycle_choice(cycle: tuple[str, ...], cursor: int, present: set[str]) -> tuple[str, int]:
    # Missing families/kinds consume their positions. Over a complete filtered
    # cycle this explicitly renormalizes weights to the represented choices.
    for _ in cycle:
        selected = cycle[cursor]
        cursor = (cursor + 1) % len(cycle)
        if selected in present:
            return selected, cursor
    raise ValueError("catalog has no represented curriculum family/kind")


class CurriculumSampler:
    """Allocate 70/20/10 families, then 50/30/20 lesson kinds, when present.

    Within a represented kind, half the probability targets the least allocated
    categories. After 32 global completions, 45% targets categories with at
    least 32 natural outcomes and 20–80% recent victory rate; otherwise that
    share also balances allocation. The remaining 5% is uniform across all
    represented categories. The rolling window is at most 128 outcomes.

    The finite declared categories and distribution hash bind state across new
    shards. Recreate with the new annotations and the saved snapshot, restoring
    the separately checkpointed curriculum RNG to reproduce future choices.
    All diagnostic counters saturate at 2**63 - 1; that value is a lower bound,
    not an exact exposure/completion count. Outcome windows continue to roll.
    """

    def __init__(self, annotations: Mapping[str, Mapping[str, object]], *,
                 distribution_sha256: str, categories: Sequence[str],
                 state: Mapping[str, object] | None = None) -> None:
        if not isinstance(distribution_sha256, str) or not _HASH.fullmatch(distribution_sha256):
            raise ValueError("distribution_sha256 must be a lowercase SHA256")
        if (not _sequence(categories) or len(categories) > 256
                or any(not isinstance(key, str) or not _KEY.fullmatch(key) for key in categories)
                or len(set(categories)) != len(categories)):
            raise ValueError("categories must be at most 256 unique declared lesson keys")
        self._distribution = distribution_sha256
        self._categories = tuple(categories)
        if not isinstance(annotations, Mapping) or not 1 <= len(annotations) <= 256:
            raise ValueError("annotations must describe 1..256 current catalog cases")
        self._annotations = {}
        for case_id, annotation in annotations.items():
            if (not isinstance(case_id, str) or not 1 <= len(case_id) <= 128
                    or not isinstance(annotation, Mapping) or set(annotation) != _ANNOTATION_KEYS):
                raise ValueError("invalid current catalog annotation")
            family = annotation["family"]
            kind, category = annotation["lesson_kind"], annotation["category"]
            targets = annotation["target_card_ids"]
            if (not isinstance(family, str) or family not in _FAMILIES
                    or annotation["deck_stage"] not in ("early", "middle", "developed")
                    or not isinstance(annotation["encounter_tier"], str)
                    or re.fullmatch(r"act[1-4]:(monster|elite|boss)", annotation["encounter_tier"]) is None
                    or not isinstance(annotation["lineage_id"], str)
                    or not 1 <= len(annotation["lineage_id"]) <= 128
                    or not isinstance(annotation["composition_sha256"], str)
                    or not _HASH.fullmatch(annotation["composition_sha256"])
                    or annotation["composition_scope"] not in ("novel_composition", "seen_composition_control")
                    or not _sequence(targets) or len(targets) > 75
                    or any(not isinstance(target, str) or not 1 <= len(target) <= 64 for target in targets)
                    or len(set(targets)) != len(targets)):
                raise ValueError("invalid current catalog annotation values")
            if family == "lesson":
                prefixes = {"single": "card:", "pair": "pair:", "group": "group:"}
                if (not isinstance(kind, str) or kind not in prefixes
                        or not isinstance(category, str) or category not in self._categories
                        or not category.startswith(prefixes[kind]) or not targets):
                    raise ValueError("lesson category/kind does not match the declared universe")
                if ((kind == "single" and (len(targets) != 1 or category != "card:" + targets[0]))
                        or (kind == "pair" and (len(targets) != 2 or category != "pair:" + "+".join(sorted(targets))))):
                    raise ValueError("lesson target IDs do not match the category")
            elif kind is not None or category is not None or targets:
                raise ValueError("nonlesson annotations cannot carry lesson metadata")
            self._annotations[case_id] = {**annotation, "target_card_ids": tuple(targets)}
        families = {annotation["family"] for annotation in self._annotations.values()}
        if "fixed" in families and len(families) != 1:
            raise ValueError("fixed control cannot be mixed with adaptive families")
        self._family_cursor = self._lesson_cursor = self._completed = 0
        self._family_allocated = dict.fromkeys(_FAMILIES, 0)
        self._stats = {key: {"allocated": 0, "completed": 0, "recent": []} for key in self._categories}
        if state is not None:
            self._restore(state)

    def _restore(self, state: Mapping[str, object]) -> None:
        keys = {"schema", "distribution_sha256", "categories", "family_cursor", "lesson_cursor",
                "family_allocated", "category_stats", "completed"}
        if (not isinstance(state, Mapping) or set(state) != keys or state["schema"] != _SCHEMA
                or state["distribution_sha256"] != self._distribution
                or not _sequence(state["categories"]) or tuple(state["categories"]) != self._categories):
            raise ValueError("curriculum state identity/schema mismatch")
        family_cursor, lesson_cursor = state["family_cursor"], state["lesson_cursor"]
        if (type(family_cursor) is not int or not 0 <= family_cursor < 10
                or type(lesson_cursor) is not int or not 0 <= lesson_cursor < 10):
            raise ValueError("curriculum cursors must be primitive integers in 0..9")
        completed = _count(state["completed"])
        allocated, stats = state["family_allocated"], state["category_stats"]
        if (not isinstance(allocated, Mapping) or set(allocated) != set(_FAMILIES)
                or not isinstance(stats, Mapping) or set(stats) != set(self._categories)):
            raise ValueError("curriculum state family/category mismatch")
        copied_allocated = {key: _count(allocated[key]) for key in _FAMILIES}
        copied_stats = {}
        for key in self._categories:
            item = stats[key]
            if not isinstance(item, Mapping) or set(item) != {"allocated", "completed", "recent"}:
                raise ValueError("invalid category statistics")
            recent = item["recent"]
            count = _count(item["completed"])
            if (count > completed or not _sequence(recent) or len(recent) != min(128, count)
                    or any(not isinstance(outcome, str) or outcome not in ("victory", "death", "escaped")
                           for outcome in recent)):
                raise ValueError("invalid bounded natural outcome window")
            copied_stats[key] = {"allocated": _count(item["allocated"]), "completed": count,
                                 "recent": list(recent)}
        if min(_MAX_COUNT, sum(item["allocated"] for item in copied_stats.values())) > copied_allocated["lesson"]:
            raise ValueError("category allocations exceed lesson allocations")
        if min(_MAX_COUNT, sum(item["completed"] for item in copied_stats.values())) > completed:
            raise ValueError("category completions exceed all natural completions")
        self._family_cursor, self._lesson_cursor = family_cursor, lesson_cursor
        self._completed, self._family_allocated, self._stats = completed, copied_allocated, copied_stats

    def validate_cases(self, cases: Sequence[NeuralCase]) -> None:
        """Reject catalog/annotation mismatches without consuming RNG or state."""
        if (not _sequence(cases) or len(cases) != len(self._annotations)
                or any(not isinstance(case, NeuralCase) for case in cases)
                or len({case.case_id for case in cases}) != len(cases)
                or {case.case_id for case in cases} != set(self._annotations)):
            raise ValueError("cases must exactly match the current annotation catalog")
        for case in cases:
            annotation = self._annotations[case.case_id]
            if (case.mode != "standalone" or case.lineage_id != annotation["lineage_id"]
                    or not isinstance(case.entry, Mapping)):
                raise ValueError("case mode/lineage disagrees with annotations")
            deck = case.entry.get("deck")
            if (not _sequence(deck) or not 1 <= len(deck) <= 128
                    or any(not isinstance(card, Mapping) or set(card) != {"card", "upgrades"}
                           or not isinstance(card["card"], str)
                           or type(card["upgrades"]) is not int or not 0 <= card["upgrades"] <= 10
                           for card in deck)
                    or content_sha256(sorted(deck, key=lambda card: (card["card"], card["upgrades"])))
                    != annotation["composition_sha256"]
                    or not set(annotation["target_card_ids"]).issubset({card["card"] for card in deck})):
                raise ValueError("case deck disagrees with annotated composition/targets")

    def choose_case(self, cases: Sequence[NeuralCase], *, rng: random.Random) -> NeuralCase:
        self.validate_cases(cases)
        if not isinstance(rng, random.Random):
            raise TypeError("curriculum requires an injected random.Random")
        families = {self._annotations[case.case_id]["family"] for case in cases}
        family, family_cursor = ("fixed", self._family_cursor) if families == {"fixed"} else _cycle_choice(
            _FAMILY_CYCLE, self._family_cursor, families)
        pool = [case for case in cases if self._annotations[case.case_id]["family"] == family]
        lesson_cursor, category = self._lesson_cursor, None
        if family == "lesson":
            kind, lesson_cursor = _cycle_choice(
                _LESSON_CYCLE, lesson_cursor,
                {self._annotations[case.case_id]["lesson_kind"] for case in pool})
            pool = [case for case in pool if self._annotations[case.case_id]["lesson_kind"] == kind]
            present = {self._annotations[case.case_id]["category"] for case in pool}
            categories = [key for key in self._categories if key in present]
            minimum = min(self._stats[key]["allocated"] for key in categories)
            least = [key for key in categories if self._stats[key]["allocated"] == minimum]
            intermediate = [key for key in categories if self._stats[key]["completed"] >= 32
                            and .2 <= self._stats[key]["recent"].count("victory") /
                            len(self._stats[key]["recent"]) <= .8] if self._completed >= 32 else []
            draw = rng.random()
            targets = categories if draw < .05 else intermediate if draw < .5 and intermediate else least
            category = targets[rng.randrange(len(targets))]
            pool = [case for case in pool if self._annotations[case.case_id]["category"] == category]
        selected = pool[rng.randrange(len(pool))]
        self._family_cursor, self._lesson_cursor = family_cursor, lesson_cursor
        self._family_allocated[family] = _increment(self._family_allocated[family])
        if category is not None:
            self._stats[category]["allocated"] = _increment(self._stats[category]["allocated"])
        return selected

    def observe(self, case_id: str, outcome: str) -> None:
        if not isinstance(case_id, str) or case_id not in self._annotations:
            raise ValueError("natural outcome must belong to the current catalog")
        if not isinstance(outcome, str) or outcome not in ("victory", "death", "escaped"):
            raise ValueError("only natural settlement outcomes may update curriculum")
        self._completed = _increment(self._completed)
        category = self._annotations[case_id]["category"]
        if category is not None:
            stats = self._stats[category]
            stats["completed"] = _increment(stats["completed"])
            stats["recent"] = (stats["recent"] + [outcome])[-128:]

    def snapshot(self) -> Mapping[str, object]:
        """Return an owned JSON value, bounded independently of shard history."""
        return {"schema": _SCHEMA, "distribution_sha256": self._distribution,
                "categories": list(self._categories), "family_cursor": self._family_cursor,
                "lesson_cursor": self._lesson_cursor, "family_allocated": dict(self._family_allocated),
                "category_stats": {key: {"allocated": item["allocated"], "completed": item["completed"],
                                         "recent": list(item["recent"])} for key, item in self._stats.items()},
                "completed": self._completed}


__all__ = ["CurriculumSampler"]
