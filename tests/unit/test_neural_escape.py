"""Standard combat escapes are terminal natural settlements with zero return."""

from __future__ import annotations

import json
from pathlib import Path
import random
import unittest
from dataclasses import replace

from spire_learn.contracts import ContractError
from spire_learn.experiments.budget import Budget
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.contracts import CollectionDiagnostics, PolicyChoice, ResourceCheckpoint
from spire_learn.neural.collector import Collector
from spire_learn.neural.envs import CombatDriver, _RawBoundary, _RawTransition
from spire_learn.neural.runner import _fight_summary
from spire_learn.neural.schema import parse_combat_descriptor, parse_lifecycle
from spire_learn.neural.returns import task_signal
from spire_learn.neural.views import COMMON_V1
from tests.neural._support import CandidateInput, authored_frame


ROOT = Path(__file__).resolve().parents[2]
REVISION = "1929e18b9f44372e020877bd62f2a5c9dee3e64e"


class _DType:
    def __init__(self, name: str):
        self.name = name
        self.kind = "i"
        self.itemsize = 1 if name == "int8" else 8
        self.isnative = True


class _Flags:
    c_contiguous = True


class _Array:
    def __init__(self, buffer):
        self.shape = buffer.shape
        self.ndim = len(buffer.shape)
        self.dtype = _DType(buffer.dtype)
        self.flags = _Flags()
        self._data = buffer.data

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError("expected C-order snapshot")
        return self._data


def _info(view, *, active, settled=0, outcome=None):
    exit_info = None
    if settled:
        exit_info = {
            "visit": 0,
            "encounter": "cultist",
            "outcome": outcome,
            "hp_before_healing": 42,
            "hp_after_cleanup": 42,
        }
    return {
        "combat_view_id": view.lifecycle_view_id,
        "active": active,
        "routing": "combat" if active else "noncombat",
        "selection_owner": None,
        "active_visit": 0 if active else None,
        "combats_entered": 1,
        "combats_settled": settled,
        "last_exit": exit_info,
    }


class _EscapeProvider:
    mode = "standalone"
    batch_size = 1
    input_errors = ()
    state_errors = ()

    def __init__(self, active, escaped):
        self.active = active
        self.escaped = escaped
        self.closed = False

    def reset(self, seeds, entry_ids, reset_mask):
        return (self.active,)

    def step(self, actions):
        return (_RawTransition(self.escaped, 0.0, True, False),)

    def close(self):
        self.closed = True


def _standard_schema():
    descriptor = json.loads(
        (ROOT / "tests/fixtures/neural/standard-combat-1929e18.json").read_text()
    )
    return parse_combat_descriptor(descriptor, revision=REVISION)


def _raw(schema, *, active, escaped=False):
    candidates = (CandidateInput(0, "end_turn"),) if active else ()
    frame = authored_frame(schema, candidates=candidates)
    return _RawBoundary(
        {key: _Array(buffer) for key, buffer in frame.tensors.items()},
        _info(schema.view, active=active, settled=0 if active else 1,
              outcome="escaped" if escaped else "victory"),
    )


class NeuralEscapeTests(unittest.TestCase):
    def test_escape_settles_without_fault(self):
        schema = _standard_schema()
        provider = _EscapeProvider(
            _raw(schema, active=True), _raw(schema, active=False, escaped=True)
        )
        driver = CombatDriver(provider, schema)
        source = driver.reset(seeds=(123,), entry_ids=(None,))[0]
        transition = driver.step((0,))[0]

        signal = task_signal(
            source.lifecycle,
            transition.successor.lifecycle,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )
        self.assertTrue(signal.task_terminal)
        self.assertEqual(signal.reward, 0.0)
        self.assertEqual(transition.successor.lifecycle.last_exit["outcome"], "escaped")
        self.assertEqual(driver.accepted_steps, 1)
        self.assertEqual(driver.reset(seeds=(124,), entry_ids=(None,))[0].lifecycle.active, True)
        driver.close()

    def test_escape_counted_separately(self):
        schema = _standard_schema()
        frame = authored_frame(schema, candidates=(CandidateInput(0, "end_turn"),))
        resources = tuple(
            ResourceCheckpoint(
                stream_id="stream-0", task_id=f"task-{index}", visit=0,
                outcome=outcome, frame=frame, encounter="cultist",
                hp_before_healing=42, hp_after_cleanup=42,
            )
            for index, outcome in enumerate(("victory", "escaped"))
        )
        diagnostics = CollectionDiagnostics(
            tasks_started=2, task_terminals=2, tasks_completed=2,
            victories=1, escaped=1, resource_checkpoints=resources,
        )
        config = load_neural_config(ROOT / "configs/neural/standalone.toml")
        runtime = type("Runtime", (), {"collector": type("Collector", (), {
            "driver": type("Driver", (), {"schema": schema})()
        })()})()

        summary = _fight_summary(config, runtime, (), diagnostics)["standalone_fights"]
        self.assertEqual(summary["victories"], 1)
        self.assertEqual(summary["deaths"], 0)
        self.assertEqual(summary["escaped"], 1)
        self.assertEqual(summary["win_rate"], 0.5)

    def test_collector_counts_escape_as_completed_but_not_win_or_death(self):
        schema = _standard_schema()
        provider = _EscapeProvider(
            _raw(schema, active=True), _raw(schema, active=False, escaped=True)
        )
        driver = CombatDriver(provider, schema)
        config = replace(
            load_neural_config(ROOT / "configs/neural/standalone.toml"),
            rollout_rows=1, rollout_steps=1,
        )

        class Policy:
            def __init__(self):
                identity = config.accepted_identities[0]
                self.model_id = identity.model_id
                self.features_id = identity.features_id
                self.action_id = identity.action_id
                self.objective_id = identity.objective_id

            def act(self, frames, *, generator, deterministic=False):
                return tuple(PolicyChoice(0, -0.25, 0.5, 0) for _ in frames)

            def values(self, frames):
                return tuple(0.5 for _ in frames)

        class CaseSampler:
            def __init__(self):
                self.observed = []

            def validate_cases(self, cases):
                self.cases = tuple(cases)

            def choose_case(self, cases, *, rng):
                return cases[0]

            def observe(self, case_id, outcome):
                self.observed.append((case_id, outcome))

            def snapshot(self):
                return {}

        sampler = CaseSampler()
        collector = Collector(
            driver, Policy(), config,
            controller_rng=random.Random(1), curriculum_rng=random.Random(2),
            budget=Budget(60.0, lambda: 0.0), policy_generator=random.Random(3),
            case_sampler=sampler,
        )
        rollout = collector.collect(
            max_learned_decisions=1, max_env_actions=1, policy_version=0,
        )

        self.assertIsNone(rollout.collection.fault)
        self.assertEqual(rollout.records[0].task_signal.reward, 0.0)
        self.assertTrue(rollout.records[0].task_signal.task_terminal)
        self.assertEqual(rollout.collection.tasks_completed, 1)
        self.assertEqual(rollout.collection.victories, 0)
        self.assertEqual(rollout.collection.deaths, 0)
        self.assertEqual(rollout.collection.escaped, 1)
        self.assertEqual(sampler.observed, [(rollout.collection.seed_allocations[0].case_id, "escaped")])

    def test_curriculum_keeps_escape_in_the_recent_win_rate_denominator(self):
        from spire_learn.neural.curriculum import CurriculumSampler
        from tests.unit.test_neural_curriculum import CATEGORIES, DISTRIBUTION, catalog

        cases, annotations = catalog()
        sampler = CurriculumSampler(
            annotations, distribution_sha256=DISTRIBUTION, categories=CATEGORIES,
        )
        case = cases[2]
        sampler.observe(case.case_id, "victory")
        sampler.observe(case.case_id, "escaped")

        stats = sampler.snapshot()["category_stats"]["card:a"]
        self.assertEqual(stats["recent"], ["victory", "escaped"])
        self.assertEqual(stats["completed"], 2)

    def test_common_view_still_rejects_escape(self):
        metadata = _info(COMMON_V1, active=False, settled=1, outcome="escaped")
        with self.assertRaises(ContractError):
            parse_lifecycle(metadata, view=COMMON_V1)


if __name__ == "__main__":
    unittest.main()
