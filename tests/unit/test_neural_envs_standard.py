"""Scripted public-provider tests for the Gym 1.0.0 Standard combat adapter."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import struct
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.artifacts.codec import to_json_value
from spire_learn.neural import envs
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.cases import load_neural_cases
from spire_learn.neural.features import EncodedBatch
from tests.unit.test_neural_envs import _DType, _Flags, _NumpyStub
from tests.unit.test_neural_gym_v1 import PUBLISHED_BUILD, V1
from tests.unit.test_neural_schema import _api


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {"effective_seed", "diagnostic", "slot_fault", "fault"}


class _Array:
    def __init__(self, values, shape, dtype):
        self.values = tuple(values)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = _DType(dtype, "i", 1 if dtype == "int8" else 8)
        self.flags = _Flags()

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError("expected C-order array export")
        code = "b" if self.dtype.name == "int8" else "q"
        return struct.pack("=" + code * len(self.values), *self.values)


class _BatchArray:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.shape = (len(rows),) + rows[0].shape
        self.dtype = rows[0].dtype
        self.flags = _Flags()

    def __getitem__(self, slot):
        return self.rows[slot]


class _NumpyScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBatchFaultError(RuntimeError):
    def __init__(self, slots):
        super().__init__("native batch fault with sensitive native detail")
        self.slots = slots


class FakeInputError(ValueError):
    pass


class FakeCombatInputError(ValueError):
    pass


class FakeStateError(RuntimeError):
    pass


class FakeRunFaultError(RuntimeError):
    pass


class FakeNativeValueError(RuntimeError):
    pass


class _HostileStateError(FakeStateError):
    def __str__(self):
        raise RuntimeError("SECRET state str")

    def __repr__(self):
        raise RuntimeError("SECRET state repr")


class _HostileReasonStateError(FakeStateError):
    @property
    def reason(self):
        raise RuntimeError("SECRET state reason")


class _HostileContractError(ContractError):
    def __str__(self):
        raise RuntimeError("SECRET learn str")

    def __repr__(self):
        raise RuntimeError("SECRET learn repr")


class _AttributeHostileError(RuntimeError):
    def __getattribute__(self, name):
        if name in {"args", "__dict__", "__cause__", "__context__", "__class__"}:
            raise RuntimeError("SECRET native getattribute")
        return super().__getattribute__(name)


class _CausePropertyError(RuntimeError):
    @property
    def __cause__(self):
        raise RuntimeError("SECRET cause property")


class _LearnAttributeHostileError(ContractError):
    def __getattribute__(self, name):
        if name in {"__class__", "native_type"}:
            raise RuntimeError("SECRET Learn getattribute")
        return super().__getattribute__(name)


class _LearnNativeTypePropertyError(ContractError):
    @property
    def native_type(self):
        raise RuntimeError("SECRET Learn native_type property")


class _ForgedNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            return "SECRET forged class name"
        return super().__getattribute__(name)


class _ForgedNameError(RuntimeError, metaclass=_ForgedNameMeta):
    pass


class _RaisingNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise RuntimeError("SECRET name lookup")
        return super().__getattribute__(name)


class _RaisingNameError(RuntimeError, metaclass=_RaisingNameMeta):
    pass


class _RecursiveNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise _NameBomb("SECRET fallback name lookup")
        return super().__getattribute__(name)


class _NameBomb(RuntimeError, metaclass=_RecursiveNameMeta):
    pass


class _RecursiveNameError(RuntimeError, metaclass=_RecursiveNameMeta):
    pass


_InvalidNameError = type("SECRET invalid class name", (RuntimeError,), {})


def _secret_error(error_type, label):
    if error_type is FakeBatchFaultError:
        error = error_type([])
    else:
        error = error_type(f"SECRET {label} message")
    error.args = (f"SECRET {label} args",)
    error.private_detail = f"SECRET {label} attribute"
    error.add_note(f"SECRET {label} note")
    nested = RuntimeError(f"SECRET {label} nested cause")
    nested.args = (f"SECRET {label} nested args",)
    nested.private_detail = f"SECRET {label} nested attribute"
    nested.add_note(f"SECRET {label} nested note")
    error.__cause__ = nested
    return error


def assert_no_secret_error_chain(testcase, error):
    from spire_learn.neural.runner import _error_value

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        values = (
            str(current), repr(current), repr(getattr(current, "__dict__", {})),
            repr(getattr(current, "__notes__", ())), repr(current.args),
            repr(_error_value(current)),
        )
        for value in values:
            testcase.assertNotIn("SECRET", value)
        if isinstance(current, envs._StandardNativeError):
            # Only a fault may also carry the Gym's public bound: (name, int or short identifier) pairs.
            testcase.assertLessEqual(set(current.__dict__), {"native_type", "bound"})
            for name, value in current.__dict__.get("bound", ()):
                testcase.assertIn(name, ("counter", "limit", "attempted", "power_id"))
                testcase.assertIn(type(value), (int, str))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def assert_driver_fault_redacted(testcase, caught, native_type, *, delivery=None):
    error = caught.exception
    testcase.assertIsInstance(error, envs.DriverFault)
    testcase.assertEqual(error.evidence.exception_type, native_type)
    testcase.assertNotIn("SECRET", repr(error.evidence))
    if delivery is not None:
        testcase.assertEqual(error.evidence.delivery, delivery)
    assert_no_secret_error_chain(testcase, error)
    if error.__cause__ is not None:
        assert_no_secret_error_chain(testcase, error.__cause__)


class _RaiseFromItem:
    def __init__(self, error):
        self.error = error

    def item(self):
        raise self.error


class _BadIndex:
    def __getitem__(self, index):
        raise _secret_error(TypeError, "presence mask indexing")


class FakeStandardVector:
    def __init__(self, catalog, num_envs, *, records=None, partition="train", entry_ids=None,
                 workers=1, autoreset_mode="next_step", fault_policy="freeze",
                 profile_id="standard-combat-v1"):
        self.args = (catalog, num_envs)
        self.num_envs = num_envs
        self.kwargs = dict(records=records, partition=partition, entry_ids=entry_ids,
                           workers=workers, autoreset_mode=autoreset_mode,
                           fault_policy=fault_policy)
        self.closed = False
        self.profile_id = profile_id
        self.next_error = None
        self.next_close_error = None
        self.next_reward = None
        self.schema = None

    def observation_spec(self):
        return json.loads((ROOT / "tests/fixtures/neural/standard-combat-1929e18.json").read_text())

    def environment_spec(self):
        return json.loads((ROOT / "tests/fixtures/neural/standard-combat-environment-1929e18.json").read_text()) | {
            "profile_id": self.profile_id
        }

    def catalog_descriptor(self):
        return json.loads((ROOT / "tests/fixtures/neural/standard-combat-catalog-1929e18.json").read_text())

    def _obs(self):
        result = {}
        action_spec = self.schema.tensors["actions"]
        self._occupied = [1] + [0] * (action_spec.shape[0] - 1)
        self._action_known_key = action_spec.known_key
        self._action_verb_offset = next(
            i for i, col in enumerate(action_spec.columns) if col.field == "verb"
        )
        for name, spec in self.schema.tensors.items():
            size = 1
            for dimension in spec.shape:
                size *= dimension
            values = [0] * size
            if name in {"actions", "action_mask"}:
                action_spec = self.schema.tensors["actions"]
                if name == "action_mask":
                    values[0] = 1
                else:
                    values[self._action_verb_offset] = 1
            if name == "actions":
                for i, col in enumerate(self.schema.tensors["actions"].columns):
                    if col.reference is not None:
                        values[i] = -1
            if name == self._action_known_key:
                values[self._action_verb_offset] = 1
            if name == self.schema.tensors["actions"].occupied_key:
                values = self._occupied
            if name.endswith("_occupied") and name != self.schema.tensors["actions"].occupied_key:
                values = [0] * size
            if name == "potions_known":
                values = [1] * size
            result[name] = _Array(values, spec.shape, spec.dtype)
        return result

    def _info(self):
        values = {
            "combat_view_id": [_NumpyScalar("spire-gym-standard-combat-view-v1")]
                * self.num_envs,
            "active": [_NumpyScalar(True)] * self.num_envs,
            "routing": [_NumpyScalar("combat")] * self.num_envs,
            "selection_owner": [None] * self.num_envs,
            "active_visit": [_NumpyScalar(0)] * self.num_envs,
            "combats_entered": [_NumpyScalar(1)] * self.num_envs,
            "combats_settled": [_NumpyScalar(0)] * self.num_envs,
            "last_exit": [None] * self.num_envs,
            "effective_seed": [987654321] * self.num_envs,
            "diagnostic": ["private"] * self.num_envs,
            "slot_fault": [False] * self.num_envs, "fault": ["private"] * self.num_envs,
            "phase": [_NumpyScalar("decision")] * self.num_envs,
        }
        info = {}
        for key, row in values.items():
            info[key] = row
            presence = [True] * self.num_envs
            if key == "phase" and self.num_envs > 1:
                presence[1] = _NumpyScalar(False)
            else:
                presence = [_NumpyScalar(flag) for flag in presence]
            info["_" + key] = presence
        return info

    def reset(self, *, seed=None, options=None):
        if self.next_error:
            error, self.next_error = self.next_error, None
            raise error
        self.schema = self.schema or self._parsed_schema
        return {key: _BatchArray([value] * self.num_envs) for key, value in self._obs().items()}, self._info()

    def step(self, actions):
        if self.next_error:
            error, self.next_error = self.next_error, None
            raise error
        reward = 0.0 if self.next_reward is None else self.next_reward
        self.next_reward = None
        return ({key: _BatchArray([value] * self.num_envs) for key, value in self._obs().items()},
                [reward] * self.num_envs, [False] * self.num_envs,
                [False] * self.num_envs, self._info())

    def close(self):
        self.closed = True
        if self.next_close_error is not None:
            error, self.next_close_error = self.next_close_error, None
            raise error

    def diagnostic(self, *args, **kwargs):
        raise AssertionError("diagnostic must never be called")

    diagnostics = diagnostic
    take_completed_records = diagnostic
    take_fault_records = diagnostic
    exact_checkpoint = diagnostic
    public_search = diagnostic


class StandardEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_neural_config(ROOT / "configs/neural/gym-v1.0.0/standard-standalone.toml")
        cls.cases = load_neural_cases(ROOT / "configs/cases/neural/standard-standalone-train.json")
        cls.api = _api(cls)
        cls.descriptor = json.loads((ROOT / "tests/fixtures/neural/standard-combat-1929e18.json").read_text())
        cls.schema = cls.api.schema.parse_combat_descriptor(cls.descriptor, revision=V1)

    def _runtime(self, vector_class):
        native = SimpleNamespace(
            build_info=lambda: dict(PUBLISHED_BUILD),
            CombatInputError=FakeCombatInputError, CombatStateError=FakeStateError,
            RunInputError=FakeInputError, RunStateError=FakeStateError,
            RunFaultError=FakeRunFaultError, BatchFaultError=FakeBatchFaultError,
        )
        return envs._NativeRuntime(_NumpyStub(), native, object, object, vector_class)

    def _factory(self, *, profile_id="standard-combat-v1"):
        made = []

        class Vector(FakeStandardVector):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs, profile_id=profile_id)
                inner.schema = self.schema
                inner._parsed_schema = self.schema
                made.append(inner)

        runtime = self._runtime(Vector)
        with patch.object(envs, "_load_native_runtime", return_value=runtime):
            driver = envs.make_driver(self.config, self.cases)
        return driver, made[0]

    def test_standard_factory_constructs_with_disabled_autoreset(self):
        driver, fake = self._factory()
        self.assertEqual(fake.args[1], self.config.rollout_rows)
        self.assertEqual(fake.kwargs["partition"], "train")
        self.assertEqual(fake.kwargs["workers"], self.config.workers)
        self.assertEqual(fake.kwargs["autoreset_mode"], "disabled")
        self.assertEqual(fake.kwargs["fault_policy"], "freeze")
        self.assertEqual(fake.args[0]["catalog_schema_id"], "spire-gym-standard-combat-catalog-v1")
        driver.close()

    def test_standard_reset_and_step_produce_frames(self):
        driver, fake = self._factory()
        rows = driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        self.assertTrue(rows[0].frame.schema_fingerprint == self.schema.schema_fingerprint)
        self.assertEqual(tuple(candidate.index for candidate in rows[0].candidates), (0,))
        transitions = driver.step([0] * driver.batch_size)
        self.assertEqual(tuple(candidate.index for candidate in transitions[0].successor.candidates), (0,))
        driver.close()

    def test_masked_partial_reset_preserves_unselected_rows(self):
        driver, fake = self._factory()
        initial = driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        self.assertGreaterEqual(driver.batch_size, 2)
        reset = driver.reset(seeds=[2] * driver.batch_size, entry_ids=[None] * driver.batch_size,
                             reset_mask=[True] + [False] * (driver.batch_size - 1))
        self.assertIs(reset[1], initial[1])
        self.assertEqual(fake.kwargs["autoreset_mode"], "disabled")
        driver.close()

    def test_standard_identity_mismatch_closes_env(self):
        made = []

        class WrongProfile(FakeStandardVector):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs, profile_id="wrong")
                made.append(inner)

        with patch.object(envs, "_load_native_runtime", return_value=self._runtime(WrongProfile)):
            with self.assertRaises(ContractError):
                envs.make_driver(self.config, self.cases)
        self.assertTrue(made[0].closed)

    def test_standard_info_privacy(self):
        driver, fake = self._factory()
        row = driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)[0]
        for value in (row.lifecycle, row.frame, EncodedBatch):
            self.assertFalse(FORBIDDEN.intersection(vars(value)))
            self.assertFalse(FORBIDDEN.intersection(field.name for field in fields(value)))
        self.assertFalse(FORBIDDEN.intersection(row.frame.tensors))
        driver.close()

    def test_standard_numpy_info_scalars_and_presence_mask(self):
        driver, fake = self._factory()
        rows = driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        lifecycle = rows[0].lifecycle
        self.assertIs(type(lifecycle.active), bool)
        self.assertIs(type(lifecycle.combats_entered), int)
        self.assertIs(type(lifecycle.routing), str)
        self.assertEqual(lifecycle.phase, "decision")
        self.assertIsNone(rows[1].lifecycle.phase)
        self.assertNotIn("phase", envs._standard_info_slot(fake._info(), 1))
        driver.close()

    def test_batch_fault_freezes_without_retaining_native_records(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = FakeBatchFaultError([
            {"status": "unaffected", "stage": "step", "committed": False,
             "recoverable": False, "message": "private native detail"},
            {"status": "faulted", "stage": "step", "committed": True,
             "recoverable": False, "message": "private native detail"},
        ])
        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)
        self.assertEqual(caught.exception.evidence.fault_slots, ())
        self.assertEqual(caught.exception.evidence.fault_slot_statuses, ())
        self.assertNotIn("private native detail", repr(caught.exception.evidence))
        self.assertNotIn("sensitive native detail", caught.exception.evidence.message)
        self.assertNotIn("private native detail", caught.exception.evidence.message)

    def test_standard_native_fault_reports_native_exception_type_only(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(FakeBatchFaultError, "step fault")
        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)
        assert_driver_fault_redacted(self, caught, "FakeBatchFaultError")

    def test_standard_native_reset_fault_is_redacted(self):
        driver, fake = self._factory()
        fake.next_error = _secret_error(FakeBatchFaultError, "reset fault")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "FakeBatchFaultError")

    def test_standard_post_return_scalar_error_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_reward = _RaiseFromItem(_secret_error(FakeNativeValueError, "reward item"))

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(
            self, caught, "FakeNativeValueError", delivery="returned",
        )

    def test_standard_learn_contract_error_keeps_message_and_sanitizes_native_cause(self):
        driver, fake = self._factory()
        original_info = fake._info

        def bad_info():
            info = original_info()
            info["_active"] = _BadIndex()
            return info

        fake._info = bad_info
        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertEqual(
            caught.exception.evidence.message,
            "Standard vector info presence mask has the wrong batch width",
        )
        learned = caught.exception.__cause__
        self.assertIsInstance(learned, ContractError)
        self.assertEqual(learned.args, (
            "Standard vector info presence mask has the wrong batch width",
        ))
        self.assertIsInstance(learned.__cause__, envs._StandardNativeError)
        self.assertEqual(learned.__cause__.native_type, "TypeError")

    def test_standard_snapshot_contract_error_sanitizes_native_tobytes_cause(self):
        driver, fake = self._factory()
        original_obs = fake._obs

        def bad_obs():
            observation = original_obs()
            first = next(iter(observation.values()))
            first.tobytes = lambda order="C": (_ for _ in ()).throw(
                _secret_error(RuntimeError, "snapshot tobytes")
            )
            return observation

        fake._obs = bad_obs
        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertIn("runtime tensor", caught.exception.evidence.message)
        self.assertIsInstance(caught.exception.__cause__, ContractError)
        self.assertIsInstance(caught.exception.__cause__.__cause__, envs._StandardNativeError)
        self.assertEqual(caught.exception.__cause__.__cause__.native_type, "RuntimeError")

    def test_standard_snapshot_contract_error_does_not_expose_native_tensor_names(self):
        driver, fake = self._factory()
        original_obs = fake._obs
        fake._obs = lambda: original_obs() | {
            "SECRET native tensor name": next(iter(original_obs().values()))
        }

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "ContractError", delivery="returned")
        self.assertNotIn("SECRET", caught.exception.evidence.message)
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_learn_message_redacts_native_text_interpolated_from_cause(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        native = RuntimeError("SECRET native text")
        learn = ContractError(f"Learn validation failed: {native}")
        learn.__cause__ = native
        fake.next_error = learn

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "ContractError")
        self.assertIn("Learn validation failed:", caught.exception.evidence.message)
        self.assertNotIn("SECRET", caught.exception.evidence.message)
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_input_error_is_redacted_and_recoverable(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(FakeCombatInputError, "input error")

        with self.assertRaises(envs._StandardNativeRecoverable) as caught:
            driver.step([0] * driver.batch_size)

        self.assertEqual(caught.exception.native_type, "FakeCombatInputError")
        assert_no_secret_error_chain(self, caught.exception)
        self.assertFalse(driver._failed)
        self.assertEqual(driver.step([0] * driver.batch_size)[0].native_reward, 0.0)
        driver.close()

    def test_standard_stale_state_error_is_redacted_and_recoverable(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(FakeStateError, "stale state")
        fake.next_error.args = ("stale token",)

        with self.assertRaises(envs._StandardNativeRecoverable) as caught:
            driver.step([0] * driver.batch_size)

        self.assertEqual(caught.exception.native_type, "FakeStateError")
        assert_no_secret_error_chain(self, caught.exception)
        self.assertFalse(driver._failed)
        self.assertEqual(driver.step([0] * driver.batch_size)[0].native_reward, 0.0)
        driver.close()

    def test_standard_hostile_exception_inspection_is_redacted_at_all_boundaries(self):
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
                        driver.reset(
                            seeds=[1] * driver.batch_size,
                            entry_ids=[None] * driver.batch_size,
                        )
                    error = error_type("SECRET payload")
                    if phase == "close":
                        fake.next_close_error = error
                        failure = driver.close
                    elif phase == "cleanup":
                        fake.next_error = RuntimeError("SECRET primary")
                        fake.next_close_error = error
                        failure = lambda: driver.step([0] * driver.batch_size)
                    elif phase == "step":
                        fake.next_error = error
                        failure = lambda: driver.step([0] * driver.batch_size)
                    else:
                        fake.next_error = error
                        failure = lambda: driver.reset(
                            seeds=[1] * driver.batch_size,
                            entry_ids=[None] * driver.batch_size,
                        )

                    with self.assertRaises(envs.DriverFault) as caught:
                        failure()

                    expected_name = "RuntimeError" if phase == "cleanup" else safe_name
                    assert_driver_fault_redacted(self, caught, expected_name)
                    if phase == "cleanup":
                        self.assertIn(safe_name, caught.exception.evidence.close_error)
                    self.assertTrue(driver._failed)
                    self.assertTrue(fake.closed)

    def test_standard_hostile_state_formatting_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(_HostileStateError, "hostile state")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "_HostileStateError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_state_classification_failure_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(_HostileReasonStateError, "hostile reason")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "RuntimeError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_hostile_contract_formatting_is_redacted_and_fatal(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(_HostileContractError, "hostile contract")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "_HostileContractError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_long_native_chain_is_redacted_and_truncated(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        error = RuntimeError("SECRET deep native error")
        for _ in range(1100):
            parent = RuntimeError("SECRET deep native error")
            parent.__cause__ = error
            error = parent
        fake.next_error = error

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

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

    def test_standard_cyclic_native_chain_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        error = _secret_error(RuntimeError, "cyclic native error")
        error.__context__ = error
        fake.next_error = error

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "RuntimeError")
        self.assertTrue(driver._failed)
        self.assertTrue(fake.closed)

    def test_standard_explicit_close_failure_is_redacted(self):
        driver, fake = self._factory()
        fake.next_close_error = _secret_error(FakeNativeValueError, "close failure")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.close()

        assert_driver_fault_redacted(self, caught, "FakeNativeValueError")

    def test_standard_cleanup_close_failure_is_redacted(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(FakeBatchFaultError, "primary fault")
        fake.next_close_error = _secret_error(FakeNativeValueError, "cleanup failure")

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "FakeBatchFaultError")
        self.assertIn("FakeNativeValueError", caught.exception.evidence.close_error)
        self.assertNotIn("SECRET", repr(caught.exception.evidence))

    def test_standard_fault_has_no_native_context(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        native = _secret_error(FakeBatchFaultError, "step fault")
        native.slots = [{"status": "faulted", "message": "SECRET slot detail"}]
        fake.next_error = native

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([0] * driver.batch_size)

        assert_driver_fault_redacted(self, caught, "FakeBatchFaultError")
        redacted = caught.exception.__cause__
        self.assertIsInstance(redacted, envs._StandardNativeFault)
        self.assertIsInstance(redacted.__cause__, envs._StandardNativeError)
        self.assertEqual(redacted.__cause__.native_type, "RuntimeError")
        assert_no_secret_error_chain(self, redacted)

    def test_fault_slot_truncation_only_marks_omitted_fault_records(self):
        driver, _fake = self._factory()
        for slot, expected in ((3, False), (66, True)):
            records = [{"status": "unaffected"} for _ in range(70)]
            records[slot] = {"status": "faulted"}
            _slots, truncated, _statuses = driver._fault_slots(
                FakeBatchFaultError(records), tuple(range(70))
            )
            self.assertEqual(truncated, expected)
        driver.close()

    def test_standard_combat_input_error_on_step_is_recoverable(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        fake.next_error = _secret_error(FakeCombatInputError, "input error")
        with self.assertRaises(envs._StandardNativeRecoverable) as caught:
            driver.step([0] * driver.batch_size)
        self.assertEqual(caught.exception.native_type, "FakeCombatInputError")
        assert_no_secret_error_chain(self, caught.exception)
        self.assertEqual(driver.accepted_steps, 0)
        self.assertEqual(driver.step([0] * driver.batch_size)[0].native_reward, 0.0)
        self.assertEqual(driver.accepted_steps, 1)
        driver.close()

    def test_standard_catalog_preserves_relic_order_and_fingerprint(self):
        authored = to_json_value(self.cases.for_mode("standard-standalone")[0].entry)
        authored["relics"] = ["burning_blood", "anchor"]
        expected = json.loads(
            (ROOT / "tests/fixtures/neural/standard-combat-catalog-1929e18.json").read_text()
        )
        expected["entries"][0]["relics"] = ["burning_blood", "anchor"]
        self.assertEqual(
            envs._verify_standard_catalog(expected, [authored]),
            "0a13a076321511dfef2d7aeff644e2b6f8f37636a965a8daa9055a006880aa99",
        )

    def test_legacy_catalog_sorts_relics_using_literal_descriptor(self):
        entry = {
            "id": "legacy-order-check",
            "deck": [{"card": "strike", "upgrades": 0}],
            "relics": ["burning_blood", "anchor"],
        }
        descriptor = {
            "catalog_schema_id": "spire-gym-combat-entry-catalog-v1",
            "entries": [{
                "id": "legacy-order-check",
                "deck": [{"card": "strike", "upgrades": 0}],
                "relics": ["anchor", "burning_blood"],
                "relic_counters": {},
            }],
        }
        envs._verify_catalog(descriptor, [entry])

    def test_never_calls_privileged_methods(self):
        driver, fake = self._factory()
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        driver.step([0] * driver.batch_size)
        driver.close()
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()


class StandardTerminalSettlementTests(unittest.TestCase):
    """Gym ends a Standard standalone episode at the combat's own terminal, unsettled."""

    setUpClass = classmethod(StandardEnvTests.setUpClass.__func__)
    _runtime = StandardEnvTests._runtime

    def _factory(self, outcome, *, phase="terminal", ended=True, reward=None):
        made = []
        schema = self.schema

        class Terminal(FakeStandardVector):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                inner.schema = schema
                inner._parsed_schema = schema
                made.append(inner)

            def step(inner, actions):
                obs = inner._obs()
                for name in ("action_mask", schema.tensors["actions"].occupied_key,
                             schema.tensors["actions"].known_key, "actions"):
                    spec = schema.tensors[name]
                    size = 1
                    for dimension in spec.shape:
                        size *= dimension
                    obs[name] = _Array([0] * size, spec.shape, spec.dtype)
                info = inner._info()
                rows = inner.num_envs
                for key, value in (("phase", phase), ("terminal_reason", outcome),
                                   ("hp_before_healing", 29), ("hp_after_cleanup", 35),
                                   ("entry_id", inner.args[0]["entries"][0]["id"])):
                    info[key] = [value] * rows
                    info["_" + key] = [True] * rows
                native_reward = ({"victory": 1.0, "death": -1.0, "escaped": 0.0}[outcome]
                                 if reward is None else reward)
                return ({key: _BatchArray([value] * rows) for key, value in obs.items()},
                        [native_reward] * rows, [ended] * rows, [False] * rows, info)

        with patch.object(envs, "_load_native_runtime", return_value=self._runtime(Terminal)):
            driver = envs.make_driver(self.config, self.cases)
        return driver, made[0]

    def test_terminal_boundary_settles_the_visit(self):
        for outcome in ("victory", "death", "escaped"):
            with self.subTest(outcome=outcome):
                driver, fake = self._factory(outcome)
                driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
                transition = driver.step([0] * driver.batch_size)[0]
                lifecycle = transition.lifecycle
                self.assertTrue(transition.terminated)
                self.assertFalse(lifecycle.active)
                self.assertEqual(lifecycle.routing, "noncombat")
                self.assertEqual(lifecycle.combats_settled, lifecycle.combats_entered)
                self.assertEqual(dict(lifecycle.last_exit), {
                    "visit": 0, "outcome": outcome, "hp_before_healing": 29, "hp_after_cleanup": 35,
                    "encounter": fake.args[0]["entries"][0]["encounter"],
                })
                self.assertEqual(transition.successor.candidates, ())
                driver.close()

    def test_terminal_without_reason_is_a_contract_fault(self):
        driver, fake = self._factory("victory")
        original = fake.step

        def no_reason(actions):
            result = original(actions)
            del result[4]["terminal_reason"], result[4]["_terminal_reason"]
            return result

        fake.step = no_reason
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        with self.assertRaises(envs.DriverFault):
            driver.step([0] * driver.batch_size)

    def test_terminal_flag_must_match_raw_terminal_phase(self):
        driver, _fake = self._factory("victory", ended=False)
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        with self.assertRaises(envs.DriverFault):
            driver.step([0] * driver.batch_size)

    def test_terminal_reward_must_match_outcome(self):
        driver, _fake = self._factory("victory", reward=0.0)
        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        with self.assertRaises(envs.DriverFault):
            driver.step([0] * driver.batch_size)
