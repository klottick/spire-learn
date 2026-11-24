"""Gated Gym v1.2.0 compact Standard vector reset and collector checks."""

from dataclasses import replace
import os
from pathlib import Path
import random
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _play_card(legal):
    return next(action.index for action in legal if action.verb == "play_card")


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class StandardTurnCapIntegrationTests(unittest.TestCase):
    def setUp(self):
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.envs import make_driver

        template = load_neural_config(ROOT / "configs/neural/baseline-standard-1.2.0/template.toml")
        self.config = replace(template, device="cpu", runtime_gpu_uuid=None, rollout_rows=2,
                              rollout_steps=3, budgets=replace(template.budgets, max_turn_actions=2))
        # The factory verifies the pinned build and compact descriptor before handing
        # back the driver backed by Gym's real StandardCombatVectorEnv.
        self.driver = make_driver(self.config, self.config.cases)
        self.addCleanup(self.driver.close)

    def test_masked_reset_restarts_only_the_selected_active_fight(self):
        case, = self.config.cases.for_mode(self.config.mode)
        entry_id = case.entry["id"]
        opening = self.driver.reset(seeds=(11, 17), entry_ids=(entry_id, entry_id))
        for _ in range(2):
            self.driver.step(tuple(_play_card(row.candidates) for row in self.driver.rows))
        before = self.driver.rows
        self.assertTrue(all(row.lifecycle.active for row in before))
        for start, current in zip(opening, before, strict=True):
            self.assertNotEqual(start.frame, current.frame)
        after = self.driver.reset(seeds=(11, 17), entry_ids=(entry_id, entry_id),
                                  reset_mask=(True, False))
        self.assertEqual(after[0].frame, opening[0].frame)
        self.assertEqual(after[0].lifecycle, opening[0].lifecycle)
        self.assertEqual(after[1].frame, before[1].frame)
        self.assertEqual(after[1].lifecycle, before[1].lifecycle)
        self.driver.step(tuple(_play_card(row.candidates) if any(
            action.verb == "play_card" for action in row.candidates
        ) else row.candidates[0].index for row in after))

    def test_collector_caps_and_bootstraps_real_compact_training_rows(self):
        from spire_learn.experiments.budget import Budget
        from spire_learn.neural.collector import Collector
        from spire_learn.neural.contracts import PolicyChoice
        from spire_learn.neural.schema import candidates

        schema = self.driver.schema
        identity = self.config.accepted_identities[0]

        class PublicPlayCardPolicy:
            model_id = identity.model_id
            features_id = identity.features_id
            action_id = identity.action_id
            objective_id = identity.objective_id

            def act(self, frames, *, generator, deterministic=False):
                return tuple(PolicyChoice(_play_card(candidates(frame, schema)), 0.0, 0.0, 0)
                             for frame in frames)

            def values(self, frames):
                return tuple(0.25 for _ in frames)

        collector = Collector(self.driver, PublicPlayCardPolicy(), self.config,
                              controller_rng=random.Random(11), curriculum_rng=random.Random(17),
                              budget=Budget(30.0, time.monotonic), policy_generator=object())
        self.addCleanup(collector.close)
        capped = []
        for _ in range(3):
            rollout = collector.collect(max_env_actions=18, max_learned_decisions=18, policy_version=0)
            capped.extend(record for record in rollout.records if record.truncated)
        diagnostics = rollout.collection
        self.assertGreater(diagnostics.turn_action_budget_hits, 0)
        self.assertEqual(diagnostics.turn_action_budget_hits, len(capped))
        self.assertGreater(diagnostics.reset_slots, 2)
        self.assertEqual(diagnostics.delivered_environment_slots, 18)
        self.assertIsNone(diagnostics.fault)
        self.assertEqual(diagnostics.faults, 0)
        for record in capped:
            self.assertFalse(record.terminated or record.task_signal.task_terminal)
            self.assertEqual(record.task_signal.reward, 0.0)
            self.assertTrue(record.task_signal.bootstrap)
            self.assertFalse(record.task_signal.continuation)
            self.assertEqual(record.next_value, 0.25)
