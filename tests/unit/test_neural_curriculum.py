"""Bounded curriculum decisions use public labels, never trajectories or runtime."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import random
import unittest

from spire_learn.neural.cases import NeuralCase
from spire_learn.neural.scenario_support import content_sha256


CATEGORIES = ("card:a", "card:b", "card:c", "pair:a+b", "group:strength")
DISTRIBUTION = "a" * 64


def catalog(prefix="case"):
    cases, annotations = [], {}
    lessons = (("single", "card:a", ("a",)), ("single", "card:b", ("b",)),
               ("single", "card:c", ("c",)), ("pair", "pair:a+b", ("a", "b")),
               ("group", "group:strength", ("a", "c")))
    for index, (family, kind, category, targets) in enumerate(
        [("run_like", None, None, ()), ("unusual", None, None, ())]
        + [("lesson", *item) for item in lessons]
    ):
        case_id = f"{prefix}-{index}"
        entry = {"id": case_id, "deck": [{"card": card, "upgrades": 0}
                                         for card in (targets or ("a",))]}
        case = NeuralCase(case_id, f"lineage-{index}", "standalone", index,
                          "novel_composition", case_id, entry, None, None, None)
        cases.append(case)
        annotations[case_id] = {
            "family": family, "deck_stage": "early", "lesson_kind": kind,
            "category": category, "target_card_ids": targets, "encounter_tier": "act1:monster",
            "lineage_id": case.lineage_id, "composition_sha256": content_sha256(entry["deck"]),
            "composition_scope": "novel_composition",
        }
    return tuple(cases), annotations


class CurriculumTests(unittest.TestCase):
    def sampler(self, annotations=None, *, state=None, categories=CATEGORIES, distribution=DISTRIBUTION):
        from spire_learn.neural.curriculum import CurriculumSampler
        return CurriculumSampler(catalog()[1] if annotations is None else annotations,
                                 distribution_sha256=distribution, categories=categories, state=state)

    def test_fixed_family_and_lesson_kind_allocation_is_independent_of_adaptation(self):
        cases, annotations = catalog()
        sampler, rng = self.sampler(), random.Random(17)
        self.assertEqual(sampler.snapshot()["completed"], 0)
        for case in cases:
            for index in range(32):
                sampler.observe(case.case_id, "victory" if index % 2 else "death")
        families, kinds = Counter(), Counter()
        for index in range(1000):
            selected = sampler.choose_case(cases, rng=rng)
            sampler.observe(selected.case_id, "victory" if index % 3 else "death")
            annotation = annotations[selected.case_id]
            families[annotation["family"]] += 1
            if annotation["family"] == "lesson":
                kinds[annotation["lesson_kind"]] += 1
        self.assertEqual(families, {"run_like": 700, "lesson": 200, "unusual": 100})
        self.assertEqual(kinds, {"single": 100, "pair": 60, "group": 40})
        self.assertEqual(sum(sampler.snapshot()["family_allocated"].values()), 1000)
        self.assertEqual(sampler.snapshot()["completed"], 1224)

    def test_rng_and_bounded_state_restore_exact_choices_and_owned_snapshot(self):
        cases, _ = catalog()
        sampler, rng = self.sampler(), random.Random(19)
        for index in range(800):
            selected = sampler.choose_case(cases, rng=rng)
            sampler.observe(selected.case_id, "victory" if index % 3 else "death")
        state = json.loads(json.dumps(sampler.snapshot()))
        restored = self.sampler(state=state)
        restored_rng = random.Random()
        restored_rng.setstate(rng.getstate())
        state["category_stats"]["card:a"]["recent"].clear()
        self.assertEqual([sampler.choose_case(cases, rng=rng) for _ in range(400)],
                         [restored.choose_case(cases, rng=restored_rng) for _ in range(400)])
        self.assertEqual(sampler.snapshot(), restored.snapshot())
        self.assertTrue(all(len(value["recent"]) <= 128
                            for value in restored.snapshot()["category_stats"].values()))

    def test_less_allocated_categories_balance_before_outcome_warmup(self):
        cases, annotations = catalog()
        singles = tuple(case for case in cases if annotations[case.case_id]["lesson_kind"] == "single")
        sampler = self.sampler({case.case_id: annotations[case.case_id] for case in singles})
        counts = Counter(sampler.choose_case(singles, rng=random.Random(index)).case_id
                         for index in range(600))
        self.assertLess(max(counts.values()) - min(counts.values()), 8)

    def test_intermediate_success_preferred_after_32_fights_and_extremes_keep_floor(self):
        cases, annotations = catalog()
        singles = tuple(case for case in cases if annotations[case.case_id]["lesson_kind"] == "single")
        annotations = {case.case_id: annotations[case.case_id] for case in singles}
        before, after = self.sampler(annotations), self.sampler(annotations)
        for index in range(32):
            for case, outcomes in zip(singles, (("death",), ("victory", "death"), ("victory",)), strict=True):
                if index < 31:
                    before.observe(case.case_id, outcomes[index % len(outcomes)])
                after.observe(case.case_id, outcomes[index % len(outcomes)])
        # Equalize allocations: preference must come from completed outcomes, not frequency of cases.
        rng_a, rng_b = random.Random(29), random.Random(29)
        counts_a = Counter(before.choose_case(singles, rng=rng_a).case_id for _ in range(6000))
        counts_b = Counter(after.choose_case(singles, rng=rng_b).case_id for _ in range(6000))
        self.assertLess(max(counts_a.values()) - min(counts_a.values()), 12)
        self.assertGreater(counts_b[singles[1].case_id], counts_b[singles[0].case_id] * 1.5)
        self.assertGreater(counts_b[singles[0].case_id], 100)
        self.assertGreater(counts_b[singles[2].case_id], 100)

    def test_new_shard_preserves_declared_stats_without_growing_case_registry(self):
        cases, annotations = catalog()
        sampler = self.sampler()
        for _ in range(300):
            sampler.observe(cases[2].case_id, "victory")
        state = sampler.snapshot()
        new_cases, new_annotations = catalog("next")
        retained = (new_cases[0], new_cases[3])
        restored = self.sampler({case.case_id: new_annotations[case.case_id] for case in retained}, state=state)
        self.assertEqual(restored.snapshot(), state)
        self.assertEqual(state["category_stats"]["card:a"]["completed"], 300)
        self.assertEqual(len(state["category_stats"]["card:a"]["recent"]), 128)
        for index in range(40):
            self.assertIn(restored.choose_case(retained, rng=random.Random(index)), retained)
        self.assertEqual(set(restored.snapshot()["category_stats"]), set(CATEGORIES))
        self.assertNotIn("case-2", json.dumps(restored.snapshot()))

    def test_sparse_family_and_kind_cycles_renormalize_present_choices(self):
        cases, annotations = catalog()
        retained = (cases[0], cases[2])
        sampler = self.sampler({case.case_id: annotations[case.case_id] for case in retained})
        choices = Counter(annotations[sampler.choose_case(retained, rng=random.Random(index)).case_id]["family"]
                          for index in range(900))
        self.assertEqual(choices, {"run_like": 700, "lesson": 200})
        only = (cases[5],)
        sampler = self.sampler({only[0].case_id: annotations[only[0].case_id]})
        self.assertTrue(all(sampler.choose_case(only, rng=random.Random(index)) == only[0] for index in range(20)))

    def test_uniform_floor_reaches_high_allocation_categories_before_warmup(self):
        cases, annotations = catalog()
        singles = cases[2:5]
        annotations = {case.case_id: annotations[case.case_id] for case in singles}
        state = self.sampler(annotations).snapshot()
        state["category_stats"]["card:a"]["allocated"] = 10000
        state["category_stats"]["card:c"]["allocated"] = 10000
        state["family_allocated"]["lesson"] = 20000
        sampler = self.sampler(annotations, state=state)
        rng = random.Random(41)
        choices = Counter(sampler.choose_case(singles, rng=rng).case_id for _ in range(1000))
        self.assertGreater(choices[singles[0].case_id], 0)
        self.assertGreater(choices[singles[2].case_id], 0)
        self.assertGreater(choices[singles[1].case_id], 900)

    def test_inconsistent_allocation_and_malformed_deck_reject_as_value_errors(self):
        state = self.sampler().snapshot()
        state["category_stats"]["card:a"]["allocated"] = 1
        with self.assertRaises(ValueError):
            self.sampler(state=state)
        cases, _ = catalog()
        for deck in ([{"card": "a"}], [{"card": "a", "upgrades": True}],
                     [{"card": "a", "upgrades": -1}]):
            malformed = (replace(cases[0], entry={"id": cases[0].entry_id, "deck": deck}),) + cases[1:]
            with self.subTest(deck=deck), self.assertRaises(ValueError):
                self.sampler().validate_cases(malformed)

    def test_saturated_counters_stay_bounded_and_recent_window_continues(self):
        cases, annotations = catalog()
        only = (cases[2],)
        annotations = {only[0].case_id: annotations[only[0].case_id]}
        state = self.sampler(annotations).snapshot()
        state["completed"] = state["family_allocated"]["lesson"] = 2**63 - 1
        state["category_stats"]["card:a"] = {"allocated": 2**63 - 1,
            "completed": 2**63 - 1, "recent": ["victory"] * 128}
        sampler = self.sampler(annotations, state=state)
        sampler.choose_case(only, rng=random.Random(43))
        sampler.observe(only[0].case_id, "death")
        result = sampler.snapshot()
        self.assertEqual(result["completed"], 2**63 - 1)
        self.assertEqual(result["category_stats"]["card:a"]["allocated"], 2**63 - 1)
        self.assertEqual(result["category_stats"]["card:a"]["recent"], ["victory"] * 127 + ["death"])
        self.sampler(annotations, state=result)

    def test_state_rejects_cross_distribution_unknown_unbounded_bool_and_nonfinite(self):
        state = self.sampler().snapshot()
        mutations = [lambda s: s.update(distribution_sha256="b" * 64),
                     lambda s: s.update(unknown=0), lambda s: s.update(completed=True),
                     lambda s: s.update(completed=float("nan")), lambda s: s.update(completed=2**63),
                     lambda s: s.update(family_cursor=10), lambda s: s.update(lesson_cursor=-1),
                     lambda s: s["category_stats"].update({"card:unknown": {}}),
                     lambda s: s["category_stats"]["card:a"].update(recent=["death"] * 129),
                     lambda s: s["category_stats"]["card:a"].update(allocated=False),
                     lambda s: s["category_stats"]["card:a"].update(recent=["fault"]),
                     lambda s: s["family_allocated"].update(lesson=float("inf")),
                     lambda s: s["categories"].reverse()]
        for mutate in mutations:
            corrupted = deepcopy(state)
            mutate(corrupted)
            with self.subTest(state=corrupted), self.assertRaises(ValueError):
                self.sampler(state=corrupted)
        for categories in (("card:a", "card:a"), tuple(f"card:x{i}" for i in range(257)), (True,)):
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                self.sampler(categories=categories)

    def test_annotation_and_case_mismatch_reject_before_rng_or_allocation(self):
        cases, annotations = catalog()
        sampler = self.sampler()
        rng = random.Random(31)
        for wrong in (cases[:-1], cases + (cases[0],),
                      (replace(cases[0], lineage_id="wrong"),) + cases[1:],
                      (replace(cases[0], entry={"id": cases[0].entry_id,
                                               "deck": [{"card": "b", "upgrades": 0}]}),) + cases[1:]):
            original_state, original_rng = sampler.snapshot(), rng.getstate()
            with self.assertRaises(ValueError):
                sampler.choose_case(wrong, rng=rng)
            self.assertEqual(sampler.snapshot(), original_state)
            self.assertEqual(rng.getstate(), original_rng)
        tampered = deepcopy(annotations)
        tampered[cases[6].case_id]["target_card_ids"] = ("b",)
        other = self.sampler(tampered)
        original_state, original_rng = other.snapshot(), rng.getstate()
        with self.assertRaises(ValueError):
            other.choose_case(cases, rng=rng)
        self.assertEqual(other.snapshot(), original_state)
        self.assertEqual(rng.getstate(), original_rng)
        bad = deepcopy(annotations)
        bad[cases[2].case_id]["category"] = "card:missing"
        with self.assertRaises(ValueError):
            self.sampler(bad)
        for outcome in ("fault", "truncated", None, True):
            with self.assertRaises(ValueError):
                sampler.observe(cases[0].case_id, outcome)
        with self.assertRaises(ValueError):
            sampler.observe("unloaded-case", "death")


if __name__ == "__main__":
    unittest.main()
