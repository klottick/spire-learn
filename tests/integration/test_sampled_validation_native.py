"""Compare sampled and fully validated collection on pinned Gym v1.2.0."""

from dataclasses import replace
import os
from pathlib import Path
import random
import unittest

from spire_learn.experiments.budget import Budget
from spire_learn.neural.collector import Collector
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.contracts import PolicyChoice
from spire_learn.neural.envs import make_driver
from spire_learn.neural.runner import _record_frame_hash


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class SampledValidationNativeTests(unittest.TestCase):
    def test_compact_sampled_collection_matches_full_validation(self):
        template = load_neural_config(ROOT / "configs/neural/baseline-standard-1.2.0/template.toml")
        identity = template.accepted_identities[0]

        class FirstLegalPolicy:
            model_id = identity.model_id
            features_id = identity.features_id
            action_id = identity.action_id
            objective_id = identity.objective_id

            def act(self, frames, *, generator, deterministic=False):
                return tuple(PolicyChoice(frame.tensors["action_mask"].data.index(b"\x01"), 0.0, 0.0, 0)
                             for frame in frames)

            def values(self, frames):
                return (0.0,) * len(frames)

        collectors = []
        for stride in (None, 16):
            config = replace(template, device="cpu", runtime_gpu_uuid=None, rollout_rows=2,
                             rollout_steps=20, budgets=replace(template.budgets, validation_stride=stride))
            driver = make_driver(config, config.cases)
            self.addCleanup(driver.close)
            collector = Collector(driver, FirstLegalPolicy(), config,
                                  controller_rng=random.Random(11), curriculum_rng=random.Random(17),
                                  budget=Budget(30.0, lambda: 0.0), policy_generator=object())
            self.addCleanup(collector.close)
            collectors.append(collector)

        for _ in range(2):
            full, sampled = [collector.collect(max_env_actions=80, max_learned_decisions=80,
                                               policy_version=0) for collector in collectors]
            self.assertEqual(len(sampled.records), 40)
            self.assertEqual(sampled.records, full.records)
            self.assertEqual(sampled.counters, full.counters)
            self.assertTrue(all(_record_frame_hash(row.source, "train") for row in sampled.records[:2]))
            self.assertTrue(any(_record_frame_hash(row.successor, "train") is None
                                for row in sampled.records))
