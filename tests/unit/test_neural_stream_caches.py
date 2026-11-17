"""Per-process caches of deterministic stream work must never change a result."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from spire_learn.neural import curriculum, scenario_stream
from spire_learn.neural.cases import NeuralCase
from tests.unit.test_neural_curriculum import CATEGORIES, DISTRIBUTION, catalog

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/neural/curriculum-v1/recipe.json"


class ShardCacheTests(unittest.TestCase):
    def setUp(self):
        self.recipe = scenario_stream.load_scenario_recipe(RECIPE)
        scenario_stream._SHARD_CACHE.clear()
        self.addCleanup(scenario_stream._SHARD_CACHE.clear)

    def generate(self, recipe=None, **kwargs):
        return scenario_stream.generate_scenario_shard(
            recipe or self.recipe, split=kwargs.get("split", "train"),
            shard_index=kwargs.get("shard_index", 3), arm=kwargs.get("arm", "mixed"))

    def test_repeat_generation_reuses_an_identical_frozen_shard(self):
        with patch.object(scenario_stream, "_generate_scenario_shard",
                          wraps=scenario_stream._generate_scenario_shard) as build:
            first = self.generate()
            second = self.generate()
        self.assertIs(first, second)
        self.assertEqual(build.call_count, 1)
        scenario_stream._SHARD_CACHE.clear()
        fresh = self.generate()
        self.assertIsNot(fresh, first)
        self.assertEqual((fresh.manifest_sha256, fresh.distribution_sha256),
                         (first.manifest_sha256, first.distribution_sha256))
        with self.assertRaises(TypeError):
            first.manifest["cases"] = ()  # frozen: a shared shard cannot be altered

        def mutable(value):
            if isinstance(value, (dict, list, set)):
                return True
            if isinstance(value, Mapping):
                return any(mutable(item) for item in value.values())
            if isinstance(value, tuple):
                return any(mutable(item) for item in value)
            return False

        for name in ("manifest", "catalog", "annotations"):
            self.assertFalse(mutable(getattr(first, name)), name)

    def test_every_key_component_and_a_changed_recipe_miss(self):
        base = self.generate()
        others = [self.generate(shard_index=4), self.generate(split="selection"),
                  self.generate(arm="fixed"),
                  self.generate(replace(self.recipe, generator_seed=self.recipe.generator_seed + 1)),
                  self.generate(replace(self.recipe, shard_size=self.recipe.shard_size - 1))]
        for other in others:
            self.assertIsNot(other, base)
        self.assertNotEqual(others[-1].manifest_sha256, base.manifest_sha256)

    def test_invalid_arguments_still_raise_and_failures_are_not_cached(self):
        for kwargs in ({"split": "nope"}, {"arm": "nope"}, {"shard_index": -1}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        with patch.object(scenario_stream, "_generate_scenario_shard",
                          side_effect=ValueError("bounded construction")):
            with self.assertRaises(ValueError):
                self.generate(shard_index=5)
        self.assertNotIn((self.recipe.sha256, "train", 5, "mixed"), scenario_stream._SHARD_CACHE)

    def test_cache_is_bounded(self):
        with patch.object(scenario_stream, "_SHARD_CACHE_CAPACITY", 2):
            for index in range(3):
                self.generate(shard_index=index)
        self.assertEqual(len(scenario_stream._SHARD_CACHE), 2)
        self.assertNotIn((self.recipe.sha256, "train", 0, "mixed"), scenario_stream._SHARD_CACHE)


class CaseValidationMemoTests(unittest.TestCase):
    def sampler(self):
        return curriculum.CurriculumSampler(catalog()[1], distribution_sha256=DISTRIBUTION,
                                            categories=CATEGORIES)

    def test_same_frozen_tuple_is_validated_once(self):
        cases = catalog()[0]
        sampler = self.sampler()
        with patch.object(sampler, "_validate_cases", wraps=sampler._validate_cases) as check:
            for _ in range(5):
                sampler.choose_case(cases, rng=random.Random(1))
        self.assertEqual(check.call_count, 1)

    def test_lists_new_tuples_and_failures_are_always_validated(self):
        cases = catalog()[0]
        sampler = self.sampler()
        with patch.object(sampler, "_validate_cases", wraps=sampler._validate_cases) as check:
            sampler.validate_cases(list(cases))
            sampler.validate_cases(list(cases))
            sampler.validate_cases(tuple(list(cases)))
            sampler.validate_cases(tuple(list(cases)))  # equal tuples, but new objects
        self.assertEqual(check.call_count, 4)
        wrong = (*cases[:-1], replace(cases[-1], lineage_id="lineage-other"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                sampler.validate_cases(wrong)

    def test_a_tuple_with_a_case_subclass_is_not_remembered(self):
        class LooseCase(NeuralCase):
            pass

        cases = catalog()[0]
        loose = (*cases[:-1], LooseCase(*(getattr(cases[-1], field) for field in (
            "case_id", "lineage_id", "mode", "environment_seed", "cohort", "entry_id", "entry",
            "profile_id", "character", "ascension"))))
        sampler = self.sampler()
        with patch.object(sampler, "_validate_cases", wraps=sampler._validate_cases) as check:
            sampler.validate_cases(loose)
            sampler.validate_cases(loose)
        self.assertEqual(check.call_count, 2)

    def test_choices_are_unchanged_by_the_memo(self):
        cases = catalog()[0]
        remembered, fresh = self.sampler(), self.sampler()
        rng_a, rng_b = random.Random(5), random.Random(5)
        picks_a, picks_b = [], []
        for index in range(200):
            picks_a.append(remembered.choose_case(cases, rng=rng_a).case_id)
            picks_b.append(fresh.choose_case(tuple(list(cases)), rng=rng_b).case_id)  # new tuple: no memo hit
            for sampler, pick in ((remembered, picks_a[-1]), (fresh, picks_b[-1])):
                sampler.observe(pick, "victory" if index % 3 else "death")
        self.assertEqual(picks_a, picks_b)


if __name__ == "__main__":
    unittest.main()
