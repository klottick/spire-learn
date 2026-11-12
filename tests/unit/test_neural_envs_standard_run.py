"""Scripted provider tests for combat learning inside Gym 1.0.0 Standard Runs."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import envs
from spire_learn.neural.cases import load_neural_cases
from spire_learn.neural.config import load_neural_config
from tests.unit.test_neural_envs import _DType, _Flags, _NumpyStub
from tests.unit.test_neural_envs_standard import FakeStandardVector, _Array
from tests.unit.test_neural_gym_v1 import PUBLISHED_BUILD, V1
from tests.unit.test_neural_schema import _api


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/neural"
FORBIDDEN = {"effective_seed", "diagnostic", "slot_fault", "fault", "reward_events", "creation"}
RUN_ACTIONS = 2048


def _fixture(name):
    return json.loads((FIXTURES / f"{name}-1929e18.json").read_text())


class _Mask:
    def __init__(self, legal):
        self.values = tuple(1 if index in legal else 0 for index in range(RUN_ACTIONS))
        self.shape = (RUN_ACTIONS,)
        self.dtype = _DType("int8", "i", 1)
        self.flags = _Flags()

    def __iter__(self):
        return iter(self.values)


class FakeRunInputError(ValueError):
    pass


class FakeRunStateError(RuntimeError):
    pass


def _exit(outcome, visit=0):
    return {"visit": visit, "encounter": "cultist", "outcome": outcome,
            "hp_before_healing": 50, "hp_after_cleanup": 56}


def _combat_info(*, active, settled=0, last_exit=None):
    entered = settled + (1 if active else 0)
    return {
        "combat_view_id": "spire-gym-standard-combat-view-v1",
        "active": active, "routing": "combat" if active else "noncombat",
        "selection_owner": None, "active_visit": entered - 1 if active else None,
        "combats_entered": entered, "combats_settled": settled, "last_exit": last_exit,
        "phase": "decision" if active else None, "selection_resume_phase": None,
        "visit": entered, "encounter": "cultist" if active else None, "outcome": None,
        "effective_seed": 987654321, "diagnostic": "private",
    }


class FakeStandardRun:
    """Scalar ``RunEnv(profile="standard")`` stand-in driven by a boundary script."""

    def __init__(self, character, ascension, *, profile="m9-ironclad-act3", partition="train",
                 boss_history=None, creation=None, capacities=None, record_replay=False):
        self.args = (character, ascension)
        self.kwargs = dict(profile=profile, partition=partition, boss_history=boss_history,
                           creation=creation, capacities=capacities, record_replay=record_replay)
        self.closed = False
        self.calls = []
        self.environment = _fixture("standard-run-environment")
        self.combat_descriptor = _fixture("standard-combat")
        self.schema = None
        # Each boundary: (combat_info, run_legal, reward, terminated)
        self.script = [
            (_combat_info(active=False), (5, 2047), 0.0, False),
            (_combat_info(active=True), (), 1.0, False),
        ]
        self.cursor = 0

    def observation_spec(self):
        return _fixture("standard-run")

    def environment_spec(self):
        return dict(self.environment)

    def combat_observation_spec(self):
        return self.combat_descriptor

    def _combat_obs(self, active):
        vector = FakeStandardVector.__new__(FakeStandardVector)
        vector.schema = self.schema
        obs = vector._obs()
        if not active:
            for name in ("action_mask", self.schema.tensors["actions"].occupied_key,
                         self.schema.tensors["actions"].known_key, "actions"):
                spec = self.schema.tensors[name]
                size = 1
                for dimension in spec.shape:
                    size *= dimension
                obs[name] = _Array([0] * size, spec.shape, spec.dtype)
        return obs

    def _boundary(self):
        info, legal, reward, terminated = self.script[min(self.cursor, len(self.script) - 1)]
        return info, {"action_mask": _Mask(legal)}, reward, terminated

    def reset(self, *, seed=None, options=None):
        self.calls.append(("reset", seed))
        self.cursor = 0
        info, run_obs, _reward, _terminated = self._boundary()
        return run_obs, {"schema_id": "spire-gym-standard-run-observation-v1", "effective_seed": 1}

    def step(self, action):
        self.calls.append(("step", action))
        self.cursor += 1
        info, run_obs, reward, terminated = self._boundary()
        return run_obs, reward, terminated, False, {"reward_events": [], "effective_seed": 1}

    def combat_view(self):
        info, _run_obs, _reward, _terminated = self._boundary()
        return {"obs": self._combat_obs(info["active"]), "token": object(), "info": info}

    def close(self):
        self.closed = True

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("privileged method must never be called")

    diagnostic = exact_checkpoint = exact_restore = exact_fork = _forbidden
    record = reproduction = public_search = from_record = _forbidden


class StandardRunEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_neural_config(ROOT / "configs/neural/gym-v1.0.0/standard-run.toml")
        cls.cases = load_neural_cases(ROOT / "configs/cases/neural/standard-run-train.json")
        cls.api = _api(cls)
        cls.schema = cls.api.schema.parse_combat_descriptor(_fixture("standard-combat"), revision=V1)

    def _runtime(self, run_class):
        native = SimpleNamespace(
            build_info=lambda: dict(PUBLISHED_BUILD),
            RunInputError=FakeRunInputError, RunStateError=FakeRunStateError,
        )
        return envs._NativeRuntime(_NumpyStub(), native, object, run_class)

    def _factory(self, mutate=None):
        made = []
        schema = self.schema

        class Run(FakeStandardRun):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                inner.schema = schema
                if mutate is not None:
                    mutate(inner)
                made.append(inner)

        with patch.object(envs, "_load_native_runtime", return_value=self._runtime(Run)):
            driver = envs.make_driver(self.config, self.cases)
        return driver, made[0]

    def test_factory_constructs_standard_run_without_reset(self):
        driver, fake = self._factory()
        case = self.cases.for_mode("standard-run")[0]
        self.assertEqual(fake.args, ("ironclad", 0))
        self.assertEqual(fake.kwargs["profile"], "standard")
        self.assertEqual(fake.kwargs["partition"], "train")
        self.assertFalse(fake.kwargs["record_replay"])
        self.assertIsNone(fake.kwargs["boss_history"])
        self.assertEqual(fake.kwargs["creation"], json.loads(json.dumps(
            envs.to_json_value(case.creation))))
        self.assertEqual(fake.calls, [])
        self.assertIs(driver.schema.view, self.schema.view)
        self.assertEqual(driver.schema.schema_fingerprint, self.config.standard_view_fingerprint)
        driver.close()
        self.assertTrue(fake.closed)

    def test_noncombat_boundary_routes_to_run_mask(self):
        driver, _fake = self._factory()
        row = driver.reset(seeds=[3], entry_ids=[None])[0]
        self.assertFalse(row.lifecycle.active)
        self.assertEqual(row.lifecycle.routing, "noncombat")
        self.assertEqual(row.candidates, ())
        self.assertEqual(row.noncombat_indices, (5, 2047))
        with self.assertRaises(envs.DriverInputError):
            driver.step([2048])
        with self.assertRaises(envs.DriverInputError):
            driver.step([6])
        transition = driver.step([2047])[0]
        self.assertTrue(transition.successor.lifecycle.active)
        driver.close()

    def test_combat_boundary_uses_view_candidates(self):
        driver, _fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        row = driver.step([5])[0].successor
        self.assertEqual(tuple(candidate.index for candidate in row.candidates), (0,))
        self.assertEqual(row.noncombat_indices, ())
        driver.close()

    def test_escape_in_run_continues(self):
        def script(fake):
            fake.script = [
                (_combat_info(active=True), (), 0.0, False),
                (_combat_info(active=False, settled=1, last_exit=_exit("escaped")), (7,), 0.0, False),
            ]

        driver, _fake = self._factory(script)
        driver.reset(seeds=[3], entry_ids=[None])
        transition = driver.step([0])[0]
        self.assertFalse(transition.terminated)
        self.assertEqual(transition.lifecycle.last_exit["outcome"], "escaped")
        self.assertEqual(transition.successor.noncombat_indices, (7,))
        driver.step([7])
        driver.close()

    def test_run_terminal_reasons(self):
        for outcome in ("death", "victory"):
            with self.subTest(outcome=outcome):
                def script(fake, outcome=outcome):
                    fake.script = [
                        (_combat_info(active=True), (), 0.0, False),
                        (_combat_info(active=False, settled=1, last_exit=_exit(outcome)),
                         (), -1.0 if outcome == "death" else 100.0, True),
                    ]

                driver, _fake = self._factory(script)
                driver.reset(seeds=[3], entry_ids=[None])
                transition = driver.step([0])[0]
                self.assertTrue(transition.terminated)
                row = driver.reset(seeds=[4], entry_ids=[None])[0]
                self.assertTrue(row.lifecycle.active)
                driver.close()

    def test_run_terminal_outside_combat(self):
        # heart_victory (after the Act 4 Proceed), empty_deck and act3_no_keys end the
        # Run on a non-combat decision: no new settlement and an empty terminal mask.
        def script(fake):
            settled = _combat_info(active=False, settled=1, last_exit=_exit("victory"))
            fake.script = [(settled, (3,), 0.0, False), (settled, (), 100.0, True)]

        driver, _fake = self._factory(script)
        driver.reset(seeds=[3], entry_ids=[None])
        transition = driver.step([3])[0]
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.successor.noncombat_indices, ())
        self.assertEqual(transition.lifecycle.combats_settled, 1)
        driver.reset(seeds=[4], entry_ids=[None])
        driver.close()

    def test_run_info_privacy(self):
        driver, _fake = self._factory()
        row = driver.reset(seeds=[3], entry_ids=[None])[0]
        transition = driver.step([5])[0]
        for lifecycle in (row.lifecycle, transition.lifecycle):
            self.assertFalse(FORBIDDEN.intersection(field.name for field in fields(lifecycle)))
            self.assertFalse(FORBIDDEN.intersection(vars(lifecycle)))
        for frame in (row.frame, transition.successor.frame):
            self.assertFalse(FORBIDDEN.intersection(frame.tensors))
        driver.close()

    def test_identity_mismatches_close_env(self):
        mutations = {
            "combat fingerprint": lambda fake: setattr(
                fake, "combat_descriptor", _fixture("common")),
            "rules id": lambda fake: fake.environment.update(rules_id="other"),
            "creation": lambda fake: fake.environment.update(creation={"neow": "short"}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                made = []

                def record(fake, mutate=mutate):
                    mutate(fake)
                    made.append(fake)

                with self.assertRaises(ContractError):
                    self._factory(record)
                self.assertTrue(made[0].closed)

    def test_rejects_multiple_workers_before_import(self):
        from dataclasses import replace
        with patch.object(envs, "_load_native_runtime", side_effect=AssertionError("import")):
            with self.assertRaises(ContractError):
                envs.make_driver(replace(self.config, workers=2), self.cases)


if __name__ == "__main__":
    unittest.main()
