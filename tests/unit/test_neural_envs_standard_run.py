"""Scripted provider tests for combat learning inside Gym 1.0.0 Standard Runs."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import envs
from spire_learn.neural.cases import load_neural_cases
from spire_learn.neural.config import load_neural_config
from tests.unit.test_neural_envs import _DType, _Flags, _NumpyStub
from tests.unit.test_neural_envs_standard import (
    FakeStandardVector,
    _Array,
    _AttributeHostileError,
    _CausePropertyError,
    _ForgedNameError,
    _InvalidNameError,
    _LearnAttributeHostileError,
    _LearnNativeTypePropertyError,
    _RecursiveNameError,
    _RaisingNameError,
    FakeNativeValueError,
    _HostileContractError,
    _secret_error,
    assert_driver_fault_redacted,
    assert_no_secret_error_chain,
)
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


class _BadMask(_Mask):
    def __iter__(self):
        raise _secret_error(TypeError, "Run mask iteration")


class FakeRunInputError(ValueError):
    pass


class FakeRunStateError(RuntimeError):
    pass


class FakeRunFaultError(RuntimeError):
    pass


class _HostileRunStateError(FakeRunStateError):
    def __str__(self):
        raise RuntimeError("SECRET Run state str")

    def __repr__(self):
        raise RuntimeError("SECRET Run state repr")


class _HostileRunReasonStateError(FakeRunStateError):
    @property
    def reason(self):
        raise RuntimeError("SECRET Run state reason")


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
        self.next_error = None
        self.next_close_error = None
        self.next_combat_view_error = None
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
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        self.calls.append(("reset", seed))
        self.cursor = 0
        info, run_obs, _reward, _terminated = self._boundary()
        return run_obs, {"schema_id": "spire-gym-standard-run-observation-v1", "effective_seed": 1}

    def step(self, action):
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        self.calls.append(("step", action))
        self.cursor += 1
        info, run_obs, reward, terminated = self._boundary()
        return run_obs, reward, terminated, False, {"reward_events": [], "effective_seed": 1}

    def combat_view(self):
        if self.next_combat_view_error is not None:
            error, self.next_combat_view_error = self.next_combat_view_error, None
            raise error
        info, _run_obs, _reward, _terminated = self._boundary()
        return {"obs": self._combat_obs(info["active"]), "token": object(), "info": info}

    def close(self):
        self.closed = True
        if self.next_close_error is not None:
            error, self.next_close_error = self.next_close_error, None
            raise error

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
            RunFaultError=FakeRunFaultError,
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

    def test_standard_run_fault_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(FakeRunFaultError, "step fault")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "FakeRunFaultError")
        redacted = caught.exception.__cause__
        self.assertIsInstance(redacted, envs._StandardNativeFault)
        self.assertIsInstance(redacted.__cause__, envs._StandardNativeError)
        self.assertEqual(redacted.__cause__.native_type, "RuntimeError")
        assert_no_secret_error_chain(self, redacted)

    def test_standard_run_reset_fault_is_redacted(self):
        driver, fake = self._factory()
        fake.next_error = _secret_error(FakeRunFaultError, "reset fault")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[3], entry_ids=[None])

        assert_driver_fault_redacted(self, caught, "FakeRunFaultError")

    def test_standard_run_learn_contract_error_keeps_message_and_sanitizes_native_cause(self):
        driver, fake = self._factory()
        info = _combat_info(active=False)
        fake._boundary = lambda: (
            info, {"action_mask": _BadMask((5, 2047))}, 0.0, False,
        )

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[3], entry_ids=[None])

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertEqual(
            caught.exception.evidence.message,
            "Run legal-action mask is not iterable",
        )
        learned = caught.exception.__cause__
        self.assertIsInstance(learned, ContractError)
        self.assertEqual(learned.args, ("Run legal-action mask is not iterable",))
        self.assertIsInstance(learned.__cause__, envs._StandardNativeError)
        self.assertEqual(learned.__cause__.native_type, "TypeError")

    def test_standard_run_snapshot_contract_error_sanitizes_native_tobytes_cause(self):
        driver, fake = self._factory()
        original_obs = fake._combat_obs

        def bad_combat_obs(active):
            observation = original_obs(active)
            first = next(iter(observation.values()))
            first.tobytes = lambda order="C": (_ for _ in ()).throw(
                _secret_error(RuntimeError, "Run snapshot tobytes")
            )
            return observation

        fake._combat_obs = bad_combat_obs
        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[3], entry_ids=[None])

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertIn("runtime tensor", caught.exception.evidence.message)
        self.assertIsInstance(caught.exception.__cause__, ContractError)
        self.assertIsInstance(caught.exception.__cause__.__cause__, envs._StandardNativeError)
        self.assertEqual(caught.exception.__cause__.__cause__.native_type, "RuntimeError")

    def test_standard_run_snapshot_does_not_expose_native_tensor_names(self):
        driver, fake = self._factory()
        original_obs = fake._combat_obs
        fake._combat_obs = lambda active: original_obs(active) | {
            "SECRET native tensor name": next(iter(original_obs(active).values()))
        }

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[3], entry_ids=[None])

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertNotIn("SECRET", caught.exception.evidence.message)
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_learn_message_redacts_native_text_interpolated_from_cause(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        native = RuntimeError("SECRET native text")
        learn = ContractError(f"Learn validation failed: {native}")
        learn.__cause__ = native
        fake.next_error = learn

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "ContractError")
        self.assertIn("Learn validation failed:", caught.exception.evidence.message)
        self.assertNotIn("SECRET", caught.exception.evidence.message)
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_combat_view_failures_after_reset_and_step_are_redacted(self):
        for phase in ("reset", "step"):
            with self.subTest(phase=phase):
                driver, fake = self._factory()
                if phase == "step":
                    driver.reset(seeds=[3], entry_ids=[None])
                fake.next_combat_view_error = _secret_error(
                    FakeNativeValueError, f"combat_view after {phase}",
                )

                with self.assertRaises(envs.DriverFault) as caught:
                    if phase == "reset":
                        driver.reset(seeds=[3], entry_ids=[None])
                    else:
                        driver.step([5])

                assert_driver_fault_redacted(
                    self, caught, "FakeNativeValueError", delivery="returned",
                )
                self.assertEqual(caught.exception.evidence.phase, f"{phase}-export")

    def test_standard_run_input_error_is_recoverable_and_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(FakeRunInputError, "input error")

        with self.assertRaises(envs._StandardNativeRecoverable) as caught:
            driver.step([5])

        self.assertEqual(caught.exception.native_type, "FakeRunInputError")
        assert_no_secret_error_chain(self, caught.exception)
        self.assertFalse(driver._failed)
        self.assertTrue(driver.step([5])[0].successor.lifecycle.active)
        driver.close()

    def test_standard_run_stale_state_error_is_recoverable_and_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(FakeRunStateError, "stale state")
        fake.next_error.args = ("stale token",)

        with self.assertRaises(envs._StandardNativeRecoverable) as caught:
            driver.step([5])

        self.assertEqual(caught.exception.native_type, "FakeRunStateError")
        assert_no_secret_error_chain(self, caught.exception)
        self.assertFalse(driver._failed)
        self.assertTrue(driver.step([5])[0].successor.lifecycle.active)
        driver.close()

    def test_standard_run_hostile_exception_inspection_is_redacted_at_all_boundaries(self):
        cases = (
            (_AttributeHostileError, "_AttributeHostileError"),
            (_CausePropertyError, "RuntimeError"),
            (_LearnAttributeHostileError, "ContractError"),
            (_LearnNativeTypePropertyError, "ContractError"),
            (_ForgedNameError, "_ForgedNameError"),
            (_RaisingNameError, "_RaisingNameError"),
            (_RecursiveNameError, "_RecursiveNameError"),
            (_InvalidNameError, "UnknownError"),
        )
        for error_type, safe_name in cases:
            for phase in ("reset", "step", "close", "cleanup"):
                with self.subTest(error=error_type, phase=phase):
                    driver, fake = self._factory()
                    if phase in {"step", "cleanup"}:
                        driver.reset(seeds=[3], entry_ids=[None])
                    error = error_type("SECRET payload")
                    if phase == "close":
                        fake.next_close_error = error
                        failure = driver.close
                    elif phase == "cleanup":
                        fake.next_error = RuntimeError("SECRET primary")
                        fake.next_close_error = error
                        failure = lambda: driver.step([5])
                    elif phase == "step":
                        fake.next_error = error
                        failure = lambda: driver.step([5])
                    else:
                        fake.next_error = error
                        failure = lambda: driver.reset(seeds=[3], entry_ids=[None])

                    with self.assertRaises(envs.DriverFault) as caught:
                        failure()

                    expected_name = "RuntimeError" if phase == "cleanup" else safe_name
                    assert_driver_fault_redacted(self, caught, expected_name)
                    if phase == "cleanup":
                        self.assertIn(safe_name, caught.exception.evidence.close_error)
                    self.assertTrue(driver._failed)
                    self.assertTrue(fake.closed)

    def test_standard_run_hostile_state_formatting_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(_HostileRunStateError, "hostile state")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "_HostileRunStateError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_state_classification_failure_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(_HostileRunReasonStateError, "hostile reason")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "RuntimeError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_hostile_contract_formatting_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(_HostileContractError, "hostile contract")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "_HostileContractError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_long_native_chain_is_redacted_and_truncated(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        error = RuntimeError("SECRET deep native error")
        for _ in range(1100):
            parent = RuntimeError("SECRET deep native error")
            parent.__cause__ = error
            error = parent
        fake.next_error = error

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "RuntimeError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)
        markers = []
        pending = [caught.exception]
        seen = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, envs._StandardNativeError):
                if current.native_type == "chain truncated":
                    markers.append(current)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        self.assertEqual(len(markers), 1)
        self.assertLessEqual(len(seen), 34)

    def test_standard_run_cyclic_native_chain_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        error = _secret_error(RuntimeError, "cyclic native error")
        error.__context__ = error
        fake.next_error = error

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "RuntimeError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_run_explicit_close_failure_is_redacted(self):
        driver, fake = self._factory()
        fake.next_close_error = _secret_error(FakeNativeValueError, "close failure")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.close()

        assert_driver_fault_redacted(self, caught, "FakeNativeValueError")

    def test_standard_run_cleanup_close_failure_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = _secret_error(FakeRunFaultError, "primary fault")
        fake.next_close_error = _secret_error(FakeNativeValueError, "cleanup failure")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        assert_driver_fault_redacted(self, caught, "FakeRunFaultError")
        self.assertIn("FakeNativeValueError", caught.exception.evidence.close_error)
        self.assertNotIn("SECRET", repr(caught.exception.evidence))

    def test_legacy_run_error_message_unchanged(self):
        _standard_driver, fake = self._factory()
        provider = envs._RunProvider(
            fake, self._runtime(type(fake)), RUN_ACTIONS, self.schema,
        )
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = FakeRunFaultError("native Run fault detail")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([5])

        self.assertIn("native Run fault detail", caught.exception.evidence.message)

    def test_legacy_run_stale_token_state_error_remains_recoverable(self):
        _standard_driver, fake = self._factory()
        provider = envs._RunProvider(
            fake, self._runtime(type(fake)), RUN_ACTIONS, self.schema,
        )
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[3], entry_ids=[None])
        fake.next_error = FakeRunStateError("stale token")

        with self.assertRaises(FakeRunStateError) as caught:
            driver.step([5])

        self.assertEqual(str(caught.exception), "stale token")
        self.assertFalse(driver._failed)
        self.assertFalse(fake.closed)
        self.assertTrue(driver.step([5])[0].successor.lifecycle.active)
        driver.close()

    def test_legacy_run_close_error_message_unchanged(self):
        _standard_driver, fake = self._factory()
        provider = envs._RunProvider(
            fake, self._runtime(type(fake)), RUN_ACTIONS, self.schema,
        )
        driver = envs.CombatDriver(provider, self.schema)
        fake.next_close_error = RuntimeError("native Run close detail")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.close()

        self.assertIn("native Run close detail", caught.exception.evidence.message)

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

    def test_config_rejects_multiple_standard_run_workers(self):
        source = ROOT / "configs/neural/gym-v1.0.0/standard-run.toml"
        raw = source.read_text()
        self.assertIn("workers = 1", raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standard-run.toml"
            path.write_text(raw.replace("workers = 1", "workers = 2", 1))
            with self.assertRaisesRegex(ValueError, "standard-run requires workers = 1"):
                load_neural_config(path)


if __name__ == "__main__":
    unittest.main()
