"""Standard modes reach the collector and runner through their driver family."""

from __future__ import annotations

from dataclasses import replace
import random
import unittest

from spire_learn.contracts import ContractError
from spire_learn.experiments.budget import Budget
from spire_learn.neural.collector import Collector
from spire_learn.neural.config import mode_family
from spire_learn.neural.contracts import CollectionDiagnostics, PolicyChoice, ResourceCheckpoint
from spire_learn.neural.runner import _eval_complete, _fight_summary, _manifest_projection
from tests.unit import test_neural_envs_standard as standalone_env
from tests.unit import test_neural_envs_standard_run as run_env


class _Policy:
    def __init__(self, config):
        identity = next(
            identity for identity in config.accepted_identities
            if config.mode in identity.accepted_raw_mode_mappings
        )
        self.model_id = identity.model_id
        self.features_id = identity.features_id
        self.action_id = identity.action_id
        self.objective_id = identity.objective_id

    def act(self, frames, *, generator, deterministic=False):
        return tuple(PolicyChoice(0, -0.25, 0.5, 0) for _ in frames)

    def values(self, frames):
        return tuple(0.5 for _ in frames)


def _collector(driver, config):
    return Collector(
        driver, _Policy(config), config,
        controller_rng=random.Random(1), curriculum_rng=random.Random(2),
        budget=Budget(60.0, lambda: 0.0), policy_generator=random.Random(3),
    )


def _runtime(driver):
    return type("Runtime", (), {"collector": type("Collector", (), {"driver": driver})()})()


class ModeFamilyTests(unittest.TestCase):
    def test_mode_family(self):
        self.assertEqual(
            {mode: mode_family(mode) for mode in ("standalone", "standard-standalone", "run", "standard-run")},
            {"standalone": "standalone", "standard-standalone": "standalone",
             "run": "run", "standard-run": "run"},
        )
        with self.assertRaises(ValueError):
            mode_family("other")


class StandardStandaloneCollectorTests(unittest.TestCase):
    setUpClass = classmethod(standalone_env.StandardEnvTests.setUpClass.__func__)
    _runtime = standalone_env.StandardEnvTests._runtime
    _factory = standalone_env.StandardEnvTests._factory

    def test_collector_uses_standard_cases(self):
        driver, _fake = self._factory()
        config = replace(self.config, rollout_steps=1)
        rows = config.rollout_rows
        rollout = _collector(driver, config).collect(
            max_learned_decisions=rows, max_env_actions=rows, policy_version=0)
        self.assertIsNone(rollout.collection.fault)
        allocated = {allocation.case_id for allocation in rollout.collection.seed_allocations}
        standard = {case.case_id for case in self.cases.for_mode("standard-standalone")}
        self.assertTrue(allocated)
        self.assertLessEqual(allocated, standard)
        driver.close()

    def test_collector_rejects_cross_family_driver(self):
        driver, _fake = self._factory()
        with self.assertRaises(ContractError):
            _collector(driver, replace(self.config, mode="standard-run"))
        driver.close()

    def test_manifest_records_standard_combat_identity(self):
        driver, _fake = self._factory()
        manifest = _manifest_projection(
            self.config, _runtime(driver), policy_kind="learned", action_selection="sample",
            allow_final=False)
        self.assertEqual(manifest["mode"], "standard-standalone")
        self.assertEqual(manifest["profile_id"], "standard-combat-v1")
        self.assertEqual(manifest["content_id"], "spire-gym-standard-content-v1")
        self.assertEqual(manifest["rules_id"], "spire-gym-native-standard-rules-v3")
        self.assertEqual(manifest["reward_id"], "spire-gym-combat-outcome-reward-v1")
        self.assertEqual(manifest["standard_view_fingerprint"], self.config.standard_view_fingerprint)
        self.assertIsNone(manifest["run_noncombat_controller_id"])
        driver.close()

    def test_fight_summary_counts_standard_standalone(self):
        driver, _fake = self._factory()
        frame = driver.reset(seeds=[1] * driver.batch_size,
                             entry_ids=[None] * driver.batch_size)[0].frame
        checkpoint = ResourceCheckpoint(
            stream_id="stream-0", task_id="task-0", visit=0, outcome="victory", frame=frame,
            encounter="cultist", hp_before_healing=42, hp_after_cleanup=42)
        diagnostics = CollectionDiagnostics(tasks_started=1, task_terminals=1, tasks_completed=1,
                                            victories=1, resource_checkpoints=(checkpoint,))
        summary = _fight_summary(self.config, _runtime(driver), (), diagnostics)
        self.assertEqual(summary["standalone_fights"]["victories"], 1)
        self.assertEqual(summary["in_run_fights"]["victories"], 0)
        driver.close()


class StandardRunCollectorTests(unittest.TestCase):
    setUpClass = classmethod(run_env.StandardRunEnvTests.setUpClass.__func__)
    _runtime = run_env.StandardRunEnvTests._runtime
    _factory = run_env.StandardRunEnvTests._factory

    def test_collector_runs_standard_run_noncombat_and_combat(self):
        driver, _fake = self._factory()
        config = replace(self.config, rollout_steps=2)
        rollout = _collector(driver, config).collect(
            max_learned_decisions=1, max_env_actions=2, policy_version=0)
        self.assertIsNone(rollout.collection.fault)
        allocated = {allocation.case_id for allocation in rollout.collection.seed_allocations}
        self.assertEqual(allocated, {case.case_id for case in self.cases.for_mode("standard-run")})
        driver.close()

    def test_manifest_records_standard_run_identity(self):
        driver, _fake = self._factory()
        manifest = _manifest_projection(
            self.config, _runtime(driver), policy_kind="learned", action_selection="sample",
            allow_final=False)
        self.assertEqual(manifest["profile_id"], "standard")
        self.assertEqual(manifest["content_id"], "spire-gym-standard-content-v1")
        self.assertEqual(manifest["reward_id"], "spire-gym-floor-key-terminal-reward-v1")
        self.assertEqual(manifest["standard_run_fingerprint"], self.config.standard_run_fingerprint)
        self.assertEqual(manifest["run_noncombat_controller_id"], "legal-random-v1")
        driver.close()

    def test_run_family_completion(self):
        # Run-family evaluation completes on the Run terminal, never on case accounting.
        self.assertFalse(_eval_complete(self.config, (), None, None))


if __name__ == "__main__":
    unittest.main()
