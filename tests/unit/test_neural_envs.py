"""Scripted public-provider tests for the neural combat driver boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural import envs
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.contracts import Lifecycle
from tests.unit.test_neural_schema import (
    _api,
    _descriptor,
    _frame,
    _native_fingerprint,
    _synthetic_schema,
)


class FakeInputError(ValueError):
    pass


class FakeStateError(RuntimeError):
    reason = "stale_token"


class FakeRunInputError(ValueError):
    pass


class FakeRunStateError(RuntimeError):
    pass


class FakeStructuredFault(RuntimeError):
    faults = [{"slot": 1, "stage": "dispatch", "message": "bounded public summary"}]


class _DType:
    def __init__(self, name: str, kind: str, itemsize: int):
        self.name = name
        self.kind = kind
        self.itemsize = itemsize
        self.isnative = True


class _Flags:
    c_contiguous = True


class _Array:
    def __init__(self, buffer):
        self.shape = buffer.shape
        self.ndim = len(buffer.shape)
        self.dtype = _DType(
            buffer.dtype,
            "i",
            1 if buffer.dtype == "int8" else 8,
        )
        self.flags = _Flags()
        self._data = buffer.data

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError("the scripted public array only supports C order")
        return self._data


def _observation(api, schema, *, legal=(2, 9), verb="use_potion"):
    frame = _frame(api, schema, legal=legal, verb=verb)
    return {key: _Array(value) for key, value in frame.tensors.items()}


def _lifecycle(*, active=True, entered=1, settled=0, outcome="victory"):
    if active:
        return {
            "combat_view_id": "spire-gym-combat-policy-view-v1",
            "active": True,
            "routing": "combat",
            "selection_owner": None,
            "active_visit": entered - 1,
            "combats_entered": entered,
            "combats_settled": settled,
            "last_exit": None,
        }
    exit_info = None
    if settled:
        exit_info = {
            "visit": settled - 1,
            "encounter": "cultist",
            "outcome": outcome,
            "hp_before_healing": 42,
            "hp_after_cleanup": 42,
        }
    return {
        "combat_view_id": "spire-gym-combat-policy-view-v1",
        "active": False,
        "routing": "noncombat",
        "selection_owner": None,
        "active_visit": None,
        "combats_entered": entered,
        "combats_settled": settled,
        "last_exit": exit_info,
    }


def _raw(api, schema, *, active=True, legal=(2, 9), noncombat=()):
    return envs._RawBoundary(
        _observation(api, schema, legal=legal),
        _lifecycle(active=active, entered=1, settled=0),
        tuple(noncombat),
    )


def _inactive_raw(api, schema, *, settled=1):
    return envs._RawBoundary(
        _observation(api, schema, legal=()),
        _lifecycle(active=False, entered=1, settled=settled),
    )


class _ScriptedProvider:
    mode = "standalone"
    input_errors = (FakeInputError,)
    state_errors = (FakeStateError,)

    def __init__(self, initial, scripts=None):
        self.batch_size = len(initial)
        self.rows = list(initial)
        self.scripts = list(scripts or ())
        self.reset_calls = []
        self.step_calls = []
        self.native_dispatches = 0
        self.tokens = [0] * self.batch_size
        self.close_calls = 0
        self.next_error = None
        self.error_after_dispatch = False

    def reset(self, seeds, entry_ids, reset_mask):
        self.reset_calls.append((seeds, entry_ids, reset_mask))
        result = []
        for slot, selected in enumerate(reset_mask):
            if selected:
                self.tokens[slot] += 1
                self.rows[slot] = _raw_for_entry.get(entry_ids[slot], self.rows[slot])
                result.append(self.rows[slot])
            else:
                result.append(self.rows[slot])
        return tuple(result)

    def step(self, actions):
        self.step_calls.append(actions)
        error = self.next_error
        self.next_error = None
        if error is not None and not self.error_after_dispatch:
            raise error
        self.native_dispatches += 1
        if error is not None:
            self.error_after_dispatch = False
            raise error
        if not self.scripts:
            raise AssertionError("scripted provider needs a transition for every accepted step")
        result = self.scripts.pop(0)
        self.rows = [transition.boundary for transition in result]
        self.tokens = [token + 1 for token in self.tokens]
        return result

    def close(self):
        self.close_calls += 1


class _ScriptedRunFacade:
    def __init__(
        self,
        boundary,
        *,
        sidecar_errors=(),
        reset_errors=(),
        step_errors=(),
        close_error=None,
    ):
        self.boundary = boundary
        self.sidecar_errors = list(sidecar_errors)
        self.reset_errors = list(reset_errors)
        self.step_errors = list(step_errors)
        self.close_error = close_error
        self.calls = []
        self.close_calls = 0

    def reset(self, *, seed):
        self.calls.append(("reset", seed))
        if self.reset_errors:
            error = self.reset_errors.pop(0)
            if error is not None:
                raise error
        return ({}, {})

    def combat_view(self):
        self.calls.append(("combat_view",))
        if self.sidecar_errors:
            error = self.sidecar_errors.pop(0)
            if error is not None:
                raise error
        return {
            "obs": self.boundary.observation,
            "info": self.boundary.info,
            "token": (9, 4),
        }

    def step(self, action):
        self.calls.append(("step", action))
        if self.step_errors:
            error = self.step_errors.pop(0)
            if error is not None:
                raise error
        return ({}, 0.0, False, False, {})

    def close(self):
        self.calls.append(("close",))
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


_raw_for_entry = {}


class _Packed:
    def __init__(self, values, dtype):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)


class _NumpyStub:
    @staticmethod
    def dtype(value):
        if value in {"bool", "?"}:
            return _DType("bool", "b", 1)
        if value in {"=i8", "int64"}:
            return _DType("int64", "i", 8)
        raise AssertionError(f"unexpected NumPy dtype request: {value}")

    @classmethod
    def asarray(cls, values, *, dtype):
        return _Packed(values, dtype)


class _BatchArray:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.shape = (len(rows),) + rows[0].shape
        self.dtype = rows[0].dtype
        self.flags = _Flags()

    def __getitem__(self, slot):
        return self.rows[slot]


class NeuralEnvTests(unittest.TestCase):
    def test_fixed_gym_factory_admission_still_requires_exact_loaded_revision(self):
        revision = "a86df922a9dc598645820cc6544df36aefb3d2cc"
        base = load_neural_config(Path("configs/neural/standalone.toml"))
        config = replace(base, environment_revision=revision)
        try:
            envs._validate_factory_inputs(config, config.cases)
        except ValueError as error:
            self.fail(f"the delivered fixed Gym revision must reach build validation: {error}")
        good = SimpleNamespace(build_info=lambda: {
            "extension": "spire-python", "revision": revision,
        })
        envs._verify_native_build(good, config.environment_revision)
        old = SimpleNamespace(build_info=lambda: {
            "extension": "spire-python", "revision": base.environment_revision,
        })
        with self.assertRaisesRegex(ValueError, "does not match"):
            envs._verify_native_build(old, config.environment_revision)
        with self.assertRaisesRegex(ValueError, "accepted Gym revision"):
            envs._validate_factory_inputs(replace(config, environment_revision="0" * 40), config.cases)

    def setUp(self):
        self.api = _api(self)
        self.schema_context = _synthetic_schema(self.api)
        self.schema = self.schema_context.__enter__()

    def tearDown(self):
        self.schema_context.__exit__(None, None, None)
        _raw_for_entry.clear()

    def test_constructor_does_not_reset_and_reset_accepts_u64_seed(self):
        provider = _ScriptedProvider([_raw(self.api, self.schema)])
        driver = envs.CombatDriver(provider, self.schema)
        self.assertEqual(provider.reset_calls, [])
        row = driver.reset(seeds=[2**64 - 1], entry_ids=["entry"])[0]
        self.assertEqual(provider.reset_calls[0], ((2**64 - 1,), ("entry",), (True,)))
        self.assertEqual(tuple(candidate.index for candidate in row.candidates), (2, 9))
        driver.close()

    def test_module_import_keeps_six_optional_runtime_prefixes_unloaded(self):
        prefixes = ("numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab")
        code = (
            "import sys; import spire_learn.neural.envs; "
            f"prefixes={prefixes!r}; "
            "loaded=sorted(name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes)); "
            "print(','.join(loaded)); "
            "raise SystemExit(bool(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.stdout, "\n")

    def test_conformance_configs_resolve_only_selection_cases_and_explicit_probes(self):
        standalone = load_neural_config(
            Path("configs/neural/task3-conformance-standalone.toml")
        )
        run = load_neural_config(Path("configs/neural/task3-conformance-run.toml"))
        self.assertEqual((standalone.split, standalone.partition), ("selection", "selection"))
        self.assertEqual((run.split, run.partition), ("selection", "selection"))
        self.assertEqual((standalone.operation, run.operation), ("evaluate", "evaluate"))
        self.assertEqual(standalone.rollout_rows, 2)
        self.assertEqual(run.rollout_rows, 1)
        self.assertEqual(
            {case.entry_id for case in standalone.cases.for_mode("standalone")},
            {"task3-warcry-elixir", "task3-mixed-winner", "task3-mixed-ongoing"},
        )
        self.assertEqual(len(run.cases.for_mode("run")), 1)
        self.assertEqual(run.cases.for_mode("run")[0].environment_seed, 1099511632779)

    def test_authored_catalog_validation_uses_configured_unreset_width_for_every_split(self):
        from tests.integration import test_neural_envs as native_tests

        created = []
        closed = []

        class CounterOnlyDriver:
            mode = "standalone"

            def __init__(inner, width):
                inner.batch_size = width
                inner.rows = (None,) * width

            def close(inner):
                closed.append(inner.batch_size)

        def factory(config, cases):
            created.append((config.split, config.partition, config.operation, config.rollout_rows))
            return CounterOnlyDriver(config.rollout_rows)

        class CounterOnlyBudget:
            def __init__(inner):
                inner.catalog_splits = []

            @staticmethod
            def note_close(driver):
                driver.close()

        budget = CounterOnlyBudget()
        with patch.object(native_tests, "_make_driver", side_effect=factory), patch.object(
            native_tests, "_BUDGET", budget
        ):
            native_tests._validate_authored_catalog_splits()

        self.assertEqual(
            created,
            [
                ("train", "train", "train", 4),
                ("selection", "selection", "evaluate", 4),
                ("final", "final", "evaluate", 4),
            ],
        )
        self.assertEqual(closed, [4, 4, 4])
        self.assertEqual(budget.catalog_splits, ["train", "selection", "final"])

    def test_combat_selection_controller_prefers_confirm_and_smallest_original_index(self):
        from tests.integration import test_neural_envs as native_tests

        lifecycle = SimpleNamespace(active=True, routing="combat_selection")

        def choose(candidates):
            row = SimpleNamespace(lifecycle=lifecycle, candidates=tuple(candidates))
            return native_tests._combat_action(row, None, prefer_attack=False)

        confirm_9 = SimpleNamespace(verb="confirm_selection", index=9)
        confirm_7 = SimpleNamespace(verb="confirm_selection", index=7)
        select_2 = SimpleNamespace(verb="combat_select_card", index=2)
        select_8 = SimpleNamespace(verb="combat_select_card", index=8)

        self.assertEqual(choose((confirm_9, confirm_7)), 7)
        self.assertEqual(choose((select_2, confirm_9, confirm_7)), 7)
        self.assertEqual(choose((select_8, select_2)), 2)
        with self.assertRaisesRegex(
            AssertionError,
            "public combat selector has no legal confirm_selection or combat_select_card candidate",
        ):
            choose(())

    def test_reset_rejects_bool_out_of_range_and_empty_masks_before_provider(self):
        provider = _ScriptedProvider([_raw(self.api, self.schema)])
        driver = envs.CombatDriver(provider, self.schema)
        bad_calls = (
            {"seeds": [True], "entry_ids": ["entry"]},
            {"seeds": [2**64], "entry_ids": ["entry"]},
            {"seeds": [1], "entry_ids": ["entry"], "reset_mask": [False]},
        )
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(envs.DriverInputError):
                driver.reset(**kwargs)
        self.assertEqual(provider.reset_calls, [])
        driver.close()

    def test_masked_reset_preserves_ongoing_row_snapshot_and_token(self):
        ongoing = _raw(self.api, self.schema)
        terminal_reset = _raw(self.api, self.schema)
        next0 = _raw(self.api, self.schema)
        next1 = _raw(self.api, self.schema)
        provider = _ScriptedProvider([ongoing, ongoing])
        _raw_for_entry["replace"] = terminal_reset
        provider.scripts.append((
            envs._RawTransition(next0, 0.0, False, False),
            envs._RawTransition(next1, 0.0, False, False),
        ))
        driver = envs.CombatDriver(provider, self.schema)
        rows = driver.reset(seeds=[1, 2], entry_ids=["keep", "keep"])
        retained = rows[1]
        retained_token = provider.tokens[1]
        reset_rows = driver.reset(
            seeds=[3, 4],
            entry_ids=["replace", "keep"],
            reset_mask=[True, False],
        )
        self.assertIs(reset_rows[1], retained)
        self.assertIs(driver.rows[1], retained)
        self.assertEqual(provider.tokens[1], retained_token)
        self.assertIs(provider.rows[1], ongoing)
        transition = driver.step([2, 2])
        self.assertIs(transition[1].source, retained)
        self.assertEqual(transition[1].source.frame, retained.frame)
        self.assertEqual(provider.native_dispatches, 1)
        driver.close()

    def test_masked_reset_freezes_when_unselected_public_frame_or_lifecycle_changes(self):
        for changed_field in ("frame", "lifecycle"):
            with self.subTest(changed_field=changed_field):
                ongoing = _raw(self.api, self.schema)
                changed = _raw(self.api, self.schema)
                if changed_field == "frame":
                    changed.observation["player"]._data = struct.pack("=q", 41)
                    expected_message = "unselected public frame changed"
                else:
                    changed = replace(
                        changed,
                        info=_lifecycle(active=True, entered=2, settled=0),
                    )
                    expected_message = "unselected public lifecycle changed"
                provider = _ScriptedProvider([ongoing, ongoing])
                driver = envs.CombatDriver(provider, self.schema)
                driver.reset(seeds=[1, 2], entry_ids=["a", "b"])
                provider.rows[1] = changed

                with self.assertRaisesRegex(envs.DriverFault, expected_message) as caught:
                    driver.reset(
                        seeds=[3, 4],
                        entry_ids=["a", "b"],
                        reset_mask=[True, False],
                    )

                evidence = caught.exception.evidence
                self.assertEqual(evidence.phase, "reset-export")
                self.assertEqual(evidence.delivery, "returned")
                self.assertEqual(evidence.known_current_delivery_slots, 1)
                self.assertEqual(evidence.uncertain_current_slots, 0)
                self.assertEqual(evidence.operation_slot_indices, (0,))
                self.assertEqual(provider.close_calls, 1)
                reset_calls = len(provider.reset_calls)
                with self.assertRaises(envs.DriverStateError):
                    driver.reset(seeds=[5, 6], entry_ids=["a", "b"])
                self.assertEqual(len(provider.reset_calls), reset_calls)

    def test_masked_reset_fails_when_standalone_provider_omits_unselected_public_boundary(self):
        ongoing = _raw(self.api, self.schema)

        class MissingBoundaryProvider(_ScriptedProvider):
            def reset(inner, seeds, entry_ids, reset_mask):
                rows = list(super().reset(seeds, entry_ids, reset_mask))
                if not reset_mask[1]:
                    rows[1] = None
                return tuple(rows)

        provider = MissingBoundaryProvider([ongoing, ongoing])
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1, 2], entry_ids=["a", "b"])

        with self.assertRaisesRegex(
            envs.DriverFault,
            "provider omitted the public unselected reset boundary",
        ) as caught:
            driver.reset(
                seeds=[3, 4],
                entry_ids=["a", "b"],
                reset_mask=[True, False],
            )

        evidence = caught.exception.evidence
        self.assertEqual(evidence.delivery, "returned")
        self.assertEqual(evidence.known_current_delivery_slots, 1)
        self.assertEqual(evidence.uncertain_current_slots, 0)
        self.assertEqual(provider.close_calls, 1)
        reset_calls = len(provider.reset_calls)
        with self.assertRaises(envs.DriverStateError):
            driver.reset(seeds=[5, 6], entry_ids=["a", "b"])
        self.assertEqual(len(provider.reset_calls), reset_calls)

    def test_terminal_row_rejects_the_whole_batch_before_dispatch(self):
        start = _raw(self.api, self.schema)
        end = _inactive_raw(self.api, self.schema)
        ongoing = _raw(self.api, self.schema)
        provider = _ScriptedProvider([start, start])
        provider.scripts.append((
            envs._RawTransition(end, -1.0, True, False),
            envs._RawTransition(ongoing, 0.0, False, False),
        ))
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1, 2], entry_ids=["a", "b"])
        result = driver.step([2, 2])
        self.assertTrue(result[0].terminated)
        self.assertFalse(result[1].terminated)
        dispatches = provider.native_dispatches
        with self.assertRaises(envs.DriverStateError):
            driver.step([2, 2])
        self.assertEqual(provider.native_dispatches, dispatches)
        driver.close()

    def test_input_and_stale_errors_recover_but_accepted_failure_freezes(self):
        start = _raw(self.api, self.schema)
        provider = _ScriptedProvider([start])
        provider.scripts.append((envs._RawTransition(start, 0.0, False, False),))
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1], entry_ids=["a"])
        for error in (FakeInputError("invalid action"), FakeStateError("stale token")):
            provider.next_error = error
            with self.assertRaises(type(error)):
                driver.step([2])
            self.assertEqual(driver.accepted_steps, 0)
            self.assertEqual(len(provider.step_calls), 1 + (1 if isinstance(error, FakeStateError) else 0))
        provider.next_error = RuntimeError("accepted operation delivery failed")
        provider.error_after_dispatch = True
        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([2])
        self.assertEqual(caught.exception.evidence.accepted_prefix, 0)
        self.assertEqual(caught.exception.evidence.delivery, "unknown")
        self.assertEqual(provider.close_calls, 1)
        calls = len(provider.step_calls)
        with self.assertRaises(envs.DriverStateError):
            driver.step([2])
        self.assertEqual(len(provider.step_calls), calls)

    def test_standalone_shared_session_input_and_stale_errors_are_recoverable(self):
        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeRunInputError,
            RunStateError=FakeRunStateError,
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        provider = envs._StandaloneProvider(object(), runtime, 1)
        driver = envs.CombatDriver(provider, self.schema)
        self.assertTrue(driver._is_recoverable(FakeRunInputError("invalid standalone input")))
        self.assertTrue(driver._is_recoverable(FakeRunStateError("stale Run decision token")))
        self.assertFalse(driver._is_recoverable(FakeRunStateError("Run is faulted")))

    def test_run_noncombat_action_uses_full_run_dispatch_capacity(self):
        raw = _raw(self.api, self.schema, active=False, legal=(), noncombat=(1500,))

        class RunScriptedProvider:
            mode = "run"
            batch_size = 1
            dispatch_action_count = 2048
            input_errors = (FakeRunInputError,)
            state_errors = (FakeRunStateError,)

            def __init__(inner):
                inner.dispatched = []

            def reset(inner, seeds, entry_ids, reset_mask):
                return (raw,)

            def step(inner, actions):
                inner.dispatched.append(actions)
                return (envs._RawTransition(raw, 0.0, False, False),)

            def close(inner):
                pass

        provider = RunScriptedProvider()
        driver = envs.CombatDriver(provider, self.schema)
        row = driver.reset(seeds=[3], entry_ids=[None])[0]
        self.assertEqual(row.noncombat_indices, (1500,))
        transition = driver.step([1500])[0]
        self.assertEqual(provider.dispatched, [(1500,)])
        self.assertEqual(transition.source.noncombat_indices, (1500,))

    def test_fatal_fault_evidence_separates_prior_delivery_and_uncertain_slots(self):
        start = _raw(self.api, self.schema)
        provider = _ScriptedProvider([start, start])
        provider.scripts.append((
            envs._RawTransition(start, 0.0, False, False),
            envs._RawTransition(start, 0.0, False, False),
        ))
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1, 2], entry_ids=["a", "b"])
        driver.step([2, 2])
        provider.next_error = FakeStructuredFault("native batch fault")
        provider.error_after_dispatch = True
        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([2, 2])
        evidence = caught.exception.evidence
        self.assertEqual(evidence.accepted_prefix, 1)
        self.assertEqual(evidence.accepted_prefix_operations, 2)
        self.assertEqual(evidence.failed_operation, 3)
        self.assertEqual(evidence.accepted_prefix_slot_actions, 2)
        self.assertEqual(evidence.operation_slots, 2)
        self.assertEqual(evidence.operation_slot_indices, (0, 1))
        self.assertEqual(evidence.known_current_delivery_slots, 0)
        self.assertEqual(evidence.uncertain_current_slots, 2)
        self.assertEqual(evidence.fault_slots, (1,))
        self.assertIn("1 step calls / 2 slot actions", str(caught.exception))

    def test_post_dispatch_snapshot_fault_closes_without_retry(self):
        malformed = envs._RawBoundary({}, _lifecycle())
        provider = _ScriptedProvider([_raw(self.api, self.schema)])
        provider.scripts.append((envs._RawTransition(malformed, 0.0, False, False),))
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1], entry_ids=["a"])
        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([2])
        self.assertEqual(caught.exception.evidence.phase, "step-export")
        self.assertEqual(caught.exception.evidence.delivery, "returned")
        self.assertEqual(caught.exception.evidence.operation_slot_indices, (0,))
        self.assertEqual(caught.exception.evidence.known_current_delivery_slots, 1)
        self.assertEqual(caught.exception.evidence.uncertain_current_slots, 0)
        self.assertEqual(provider.close_calls, 1)
        dispatches = provider.native_dispatches
        with self.assertRaises(envs.DriverStateError):
            driver.step([2])
        self.assertEqual(provider.native_dispatches, dispatches)

    def test_full_batch_dispatch_is_packed_as_native_int64(self):
        first = _raw(self.api, self.schema)
        second = _raw(self.api, self.schema)
        frame0 = _frame(self.api, self.schema)
        frame1 = _frame(self.api, self.schema)

        class FakeVector:
            def __init__(self):
                self.received = None

            def step(inner, actions):
                inner.received = actions
                observations = {
                    key: _BatchArray((_Array(frame0.tensors[key]), _Array(frame1.tensors[key])))
                    for key in frame0.tensors
                }
                return (
                    observations,
                    [0.0, 0.0],
                    [False, False],
                    [False, False],
                    {"slots": [first.info, second.info]},
                )

            def close(inner):
                pass

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeRunInputError,
            RunStateError=FakeRunStateError,
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        gym = FakeVector()
        provider = envs._StandaloneProvider(gym, runtime, 2)
        result = provider.step((2, 9))
        self.assertEqual(gym.received.values, (2, 9))
        self.assertEqual(gym.received.shape, (2,))
        self.assertEqual(gym.received.dtype.name, "int64")
        self.assertEqual(len(result), 2)

    def test_standalone_reset_passes_full_seed_entry_and_bool_mask_vectors(self):
        raw0 = _raw(self.api, self.schema)
        raw1 = _raw(self.api, self.schema)
        frame0 = _frame(self.api, self.schema)
        frame1 = _frame(self.api, self.schema)

        class FakeVector:
            def __init__(inner):
                inner.received = None

            def reset(inner, *, seed, options):
                inner.received = (seed, options)
                observations = {
                    key: _BatchArray((_Array(frame0.tensors[key]), _Array(frame1.tensors[key])))
                    for key in frame0.tensors
                }
                return observations, {"slots": [raw0.info, raw1.info]}

            def close(inner):
                pass

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeRunInputError,
            RunStateError=FakeRunStateError,
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        gym = FakeVector()
        provider = envs._StandaloneProvider(gym, runtime, 2)
        result = provider.reset((2**64 - 1, 17), ("first", "unselected"), (True, False))
        seeds, options = gym.received
        self.assertEqual(seeds, [2**64 - 1, 17])
        self.assertEqual(set(options), {"entry_ids", "reset_mask"})
        self.assertEqual(options["entry_ids"], ["first", "unselected"])
        self.assertEqual(options["reset_mask"].dtype.name, "bool")
        self.assertEqual(options["reset_mask"].shape, (2,))
        self.assertEqual(options["reset_mask"].values, (True, False))
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertEqual(result[1].info, raw1.info)
        self.assertEqual(
            result[1].observation["player"]._data,
            frame1.tensors["player"].data,
        )

    def test_run_provider_reads_sidecar_immediately_after_reset_and_step(self):
        active = _raw(self.api, self.schema)

        class FakeRun:
            def __init__(inner):
                inner.calls = []

            def reset(inner, *, seed):
                inner.calls.append(("reset", seed))
                return ({}, {})

            def combat_view(inner):
                inner.calls.append(("combat_view",))
                return {"obs": active.observation, "info": active.info, "token": (9, 4)}

            def step(inner, action):
                inner.calls.append(("step", action))
                return ({}, 0.0, False, False, {})

            def close(inner):
                inner.calls.append(("close",))

        native = SimpleNamespace(RunInputError=FakeInputError, RunStateError=FakeStateError)
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        facade = FakeRun()
        provider = envs._RunProvider(facade, runtime, 10)
        reset = provider.reset((7,), (None,), (True,))
        step = provider.step((2,))
        self.assertEqual(facade.calls, [("reset", 7), ("combat_view",), ("step", 2), ("combat_view",)])
        self.assertFalse(hasattr(reset[0], "token"))
        self.assertFalse(hasattr(step[0].boundary, "token"))

    def test_run_post_return_recoverable_class_errors_freeze_reset_and_step(self):
        native = SimpleNamespace(RunInputError=FakeRunInputError, RunStateError=FakeRunStateError)
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        active = _raw(self.api, self.schema)

        for operation in ("reset", "step"):
            for error_type in (FakeRunInputError, FakeRunStateError):
                with self.subTest(operation=operation, error_type=error_type.__name__):
                    error_text = f"post-return {operation} sidecar"
                    sidecar_errors = (
                        [error_type(error_text)]
                        if operation == "reset"
                        else [None, error_type(error_text)]
                    )
                    facade = _ScriptedRunFacade(active, sidecar_errors=sidecar_errors)
                    driver = envs.CombatDriver(envs._RunProvider(facade, runtime, 10), self.schema)
                    if operation == "step":
                        driver.reset(seeds=[7], entry_ids=[None])
                        call = lambda: driver.step([2])
                        expected_phase = "step-export"
                    else:
                        call = lambda: driver.reset(seeds=[7], entry_ids=[None])
                        expected_phase = "reset-export"

                    with self.assertRaises(envs.DriverFault) as caught:
                        call()

                    evidence = caught.exception.evidence
                    self.assertEqual(evidence.phase, expected_phase)
                    self.assertEqual(evidence.delivery, "returned")
                    self.assertEqual(evidence.exception_type, error_type.__name__)
                    self.assertEqual(evidence.message, error_text)
                    self.assertEqual(evidence.known_current_delivery_slots, 1)
                    self.assertEqual(evidence.uncertain_current_slots, 0)
                    self.assertEqual(facade.close_calls, 1)
                    calls_at_fault = tuple(facade.calls)
                    with self.assertRaises(envs.DriverStateError):
                        driver.step([2])
                    self.assertEqual(tuple(facade.calls), calls_at_fault)

    def test_run_facade_preflight_input_and_stale_errors_remain_recoverable(self):
        native = SimpleNamespace(RunInputError=FakeRunInputError, RunStateError=FakeRunStateError)
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        active = _raw(self.api, self.schema)
        facade = _ScriptedRunFacade(
            active,
            reset_errors=(FakeRunInputError("invalid seed"), None),
            step_errors=(FakeRunStateError("stale token"), None),
        )
        driver = envs.CombatDriver(envs._RunProvider(facade, runtime, 10), self.schema)

        with self.assertRaisesRegex(FakeRunInputError, "invalid seed"):
            driver.reset(seeds=[7], entry_ids=[None])
        self.assertEqual(facade.close_calls, 0)
        driver.reset(seeds=[7], entry_ids=[None])
        with self.assertRaisesRegex(FakeRunStateError, "stale token"):
            driver.step([2])
        self.assertEqual(driver.accepted_steps, 0)
        self.assertEqual(facade.close_calls, 0)
        transition = driver.step([2])
        self.assertEqual(len(transition), 1)
        self.assertEqual(driver.accepted_steps, 1)
        driver.close()
        self.assertEqual(facade.close_calls, 1)

    def test_standalone_post_return_recoverable_class_error_freezes_reset(self):
        frame = _frame(self.api, self.schema)

        class BrokenSlots(dict):
            def get(inner, key, default=None):
                raise FakeRunInputError("post-return standalone packaging")

        class FakeVector:
            def __init__(inner):
                inner.reset_calls = 0
                inner.close_calls = 0

            def reset(inner, *, seed, options):
                inner.reset_calls += 1
                observation = {
                    key: _BatchArray((_Array(value),))
                    for key, value in frame.tensors.items()
                }
                return observation, BrokenSlots()

            def close(inner):
                inner.close_calls += 1

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeRunInputError,
            RunStateError=FakeRunStateError,
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        facade = FakeVector()
        driver = envs.CombatDriver(
            envs._StandaloneProvider(facade, runtime, 1), self.schema
        )

        with self.assertRaises(envs.DriverFault) as caught:
            driver.reset(seeds=[7], entry_ids=["entry"])

        evidence = caught.exception.evidence
        self.assertEqual(evidence.phase, "reset-export")
        self.assertEqual(evidence.delivery, "returned")
        self.assertEqual(evidence.exception_type, "FakeRunInputError")
        self.assertEqual(evidence.message, "post-return standalone packaging")
        self.assertEqual(evidence.known_current_delivery_slots, 1)
        self.assertEqual(evidence.uncertain_current_slots, 0)
        self.assertEqual(facade.close_calls, 1)
        reset_calls = facade.reset_calls
        with self.assertRaises(envs.DriverStateError):
            driver.reset(seeds=[8], entry_ids=["entry"])
        self.assertEqual(facade.reset_calls, reset_calls)

    def test_standalone_post_return_recoverable_class_error_freezes_step(self):
        frame = _frame(self.api, self.schema)
        observation = {
            key: _BatchArray((_Array(value),))
            for key, value in frame.tensors.items()
        }

        class BrokenSlots(dict):
            def get(inner, key, default=None):
                raise FakeRunStateError("post-return stale standalone info")

        class FakeVector:
            def __init__(inner):
                inner.step_calls = 0
                inner.close_calls = 0

            def reset(inner, *, seed, options):
                return observation, {"slots": [_lifecycle()]}

            def step(inner, actions):
                inner.step_calls += 1
                return observation, [0.0], [False], [False], BrokenSlots()

            def close(inner):
                inner.close_calls += 1

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeRunInputError,
            RunStateError=FakeRunStateError,
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        facade = FakeVector()
        driver = envs.CombatDriver(
            envs._StandaloneProvider(facade, runtime, 1), self.schema
        )
        driver.reset(seeds=[7], entry_ids=["entry"])

        with self.assertRaises(envs.DriverFault) as caught:
            driver.step([2])

        evidence = caught.exception.evidence
        self.assertEqual(evidence.phase, "step-export")
        self.assertEqual(evidence.delivery, "returned")
        self.assertEqual(evidence.exception_type, "FakeRunStateError")
        self.assertEqual(evidence.known_current_delivery_slots, 1)
        self.assertEqual(evidence.uncertain_current_slots, 0)
        self.assertEqual(facade.close_calls, 1)
        step_calls = facade.step_calls
        with self.assertRaises(envs.DriverStateError):
            driver.step([2])
        self.assertEqual(facade.step_calls, step_calls)

    def test_close_process_control_after_post_return_failure_keeps_delivery_context(self):
        native = SimpleNamespace(RunInputError=FakeRunInputError, RunStateError=FakeRunStateError)
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        active = _raw(self.api, self.schema)

        for close_error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(close_error_type=close_error_type.__name__):
                facade = _ScriptedRunFacade(
                    active,
                    sidecar_errors=(None, RuntimeError("accepted sidecar packaging failed")),
                    close_error=close_error_type("close process control"),
                )
                driver = envs.CombatDriver(envs._RunProvider(facade, runtime, 10), self.schema)
                driver.reset(seeds=[7], entry_ids=[None])

                with self.assertRaises(close_error_type) as caught:
                    driver.step([2])

                self.assertEqual(facade.close_calls, 1)
                notes = "\n".join(getattr(caught.exception, "__notes__", ()))
                self.assertIn("accepted sidecar packaging failed", notes)
                self.assertIn("delivery=returned", notes)
                self.assertIn("known_current_delivery_slots=1", notes)
                calls_at_fault = tuple(facade.calls)
                with self.assertRaises(envs.DriverStateError):
                    driver.reset(seeds=[8], entry_ids=[None])
                self.assertEqual(tuple(facade.calls), calls_at_fault)

    def test_primary_process_control_wins_over_process_control_close_and_close_alone(self):
        active = _raw(self.api, self.schema)
        for close_error in (
            RuntimeError("ordinary close error"),
            KeyboardInterrupt("secondary interrupt"),
            SystemExit("secondary exit"),
        ):
            with self.subTest(close_error=type(close_error).__name__):
                provider = _ScriptedProvider([active])

                def close_with_error():
                    provider.close_calls += 1
                    raise close_error

                provider.close = close_with_error
                driver = envs.CombatDriver(provider, self.schema)
                driver.reset(seeds=[1], entry_ids=["entry"])
                provider.next_error = KeyboardInterrupt("primary interruption")

                with self.assertRaisesRegex(KeyboardInterrupt, "primary interruption") as caught:
                    driver.step([2])

                self.assertEqual(provider.close_calls, 1)
                self.assertTrue(any(
                    type(close_error).__name__ in note
                    for note in getattr(caught.exception, "__notes__", ())
                ))

        close_control = SystemExit("close-only exit")
        provider = _ScriptedProvider([active])

        def close_with_control():
            provider.close_calls += 1
            raise close_control

        provider.close = close_with_control
        driver = envs.CombatDriver(provider, self.schema)
        with self.assertRaisesRegex(SystemExit, "close-only exit"):
            driver.close()
        with self.assertRaises(envs.DriverStateError):
            driver.reset(seeds=[1], entry_ids=["entry"])
        driver.close()
        self.assertEqual(provider.close_calls, 1)

    def test_post_return_process_control_remains_primary_and_notes_close_failure(self):
        native = SimpleNamespace(RunInputError=FakeRunInputError, RunStateError=FakeRunStateError)
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, object)
        active = _raw(self.api, self.schema)
        facade = _ScriptedRunFacade(
            active,
            sidecar_errors=(None, KeyboardInterrupt("post-return interrupt")),
            close_error=RuntimeError("secondary close failure"),
        )
        driver = envs.CombatDriver(envs._RunProvider(facade, runtime, 10), self.schema)
        driver.reset(seeds=[7], entry_ids=[None])

        with self.assertRaisesRegex(KeyboardInterrupt, "post-return interrupt") as caught:
            driver.step([2])

        self.assertEqual(facade.close_calls, 1)
        notes = "\n".join(getattr(caught.exception, "__notes__", ()))
        self.assertIn("secondary close failure", notes)
        self.assertIn("delivery=returned", notes)
        self.assertIn("known_current_delivery_slots=1", notes)
        calls_at_fault = tuple(facade.calls)
        with self.assertRaises(envs.DriverStateError):
            driver.step([2])
        self.assertEqual(tuple(facade.calls), calls_at_fault)

    def test_process_control_exception_closes_then_propagates_primary_failure(self):
        class FailingCloseProvider(_ScriptedProvider):
            def close(inner):
                inner.close_calls += 1
                raise RuntimeError("secondary close failure")

        provider = FailingCloseProvider([_raw(self.api, self.schema)])
        driver = envs.CombatDriver(provider, self.schema)
        driver.reset(seeds=[1], entry_ids=["a"])
        provider.next_error = KeyboardInterrupt("stop now")
        with self.assertRaisesRegex(KeyboardInterrupt, "stop now") as caught:
            driver.step([2])
        self.assertEqual(provider.close_calls, 1)
        self.assertTrue(any("secondary close failure" in note for note in caught.exception.__notes__))
        with self.assertRaises(envs.DriverStateError):
            driver.step([2])

    def test_close_error_is_bounded_and_close_is_not_retried(self):
        class FailingCloseProvider(_ScriptedProvider):
            def close(inner):
                inner.close_calls += 1
                raise RuntimeError("public close failure" * 100)

        provider = FailingCloseProvider([_raw(self.api, self.schema)])
        driver = envs.CombatDriver(provider, self.schema)
        with self.assertRaises(envs.DriverFault) as caught:
            driver.close()
        self.assertEqual(caught.exception.evidence.phase, "close")
        self.assertLessEqual(len(caught.exception.evidence.message), 513)
        driver.close()
        self.assertEqual(provider.close_calls, 1)

    def test_factory_uses_exact_standalone_constructor_without_reset(self):
        base = load_neural_config(Path("configs/neural/standalone.toml"))
        cases = base.cases
        raw_descriptor = _descriptor()
        fingerprint = _native_fingerprint(raw_descriptor)
        raw_descriptor["fingerprint"] = fingerprint
        config = replace(base, common_view_fingerprint=fingerprint)
        normalized = [envs._normalized_entry(case.entry) for case in cases.for_mode("standalone")]
        calls = {}

        class FakeGym:
            def __init__(inner, *args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs
                inner.reset_calls = 0
                inner.step_calls = 0
                inner.close_calls = 0

            def observation_spec(inner):
                return raw_descriptor

            def environment_spec(inner):
                return {
                    "profile_id": "ironclad-combat-v1",
                    "schema_id": envs.COMMON_VIEW_ID,
                    "observation_fingerprint": fingerprint,
                    "content_id": "spire-gym-ironclad-combat-content-v1",
                    "rules_id": "spire-gym-native-m10-rules-v1",
                    "reward_id": "spire-gym-combat-outcome-reward-v1",
                    "mode": "standalone_combat",
                    "partition": "train",
                    "autoreset": "disabled",
                }

            def catalog_descriptor(inner):
                return {
                    "catalog_schema_id": "spire-gym-combat-entry-catalog-v1",
                    "entries": normalized,
                }

            def reset(inner, **kwargs):
                inner.reset_calls += 1
                raise AssertionError("factory must not reset")

            def step(inner, *args):
                inner.step_calls += 1
                raise AssertionError("factory must not step")

            def close(inner):
                inner.close_calls += 1

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeInputError,
            RunStateError=FakeStateError,
            build_info=lambda: {
                "extension": "spire-python",
                "revision": "533dad1c8aec86df624b3742b6b3ad1dbf1d0033",
            },
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, FakeGym, object)
        from spire_learn.neural.gym_contracts import LEGACY_CONTRACT
        synthetic_contract = SimpleNamespace(
            common_fingerprint=fingerprint,
            run_fingerprint=LEGACY_CONTRACT.run_fingerprint,
            raw_mode_mappings=LEGACY_CONTRACT.raw_mode_mappings,
            build=None,
        )
        with (
            patch.object(envs, "contract_for_revision", return_value=synthetic_contract),
            patch.object(envs, "_load_native_runtime", return_value=runtime),
            patch.object(self.api.schema, "COMMON_VIEW_FINGERPRINT", fingerprint),
            patch.object(self.api.neural_contracts, "_COMMON_VIEW_FINGERPRINT", fingerprint),
        ):
            driver = envs.make_driver(config, cases)
        self.assertEqual(calls["args"][0], base.rollout_rows)
        self.assertEqual(calls["kwargs"], {"partition": "train"})
        self.assertEqual(len(calls["args"][1]["entries"]), len(normalized))
        gym = driver._provider._env
        self.assertEqual((gym.reset_calls, gym.step_calls), (0, 0))
        driver.close()
        self.assertEqual(gym.close_calls, 1)

    def test_factory_uses_explicit_policy_v3_run_profile_without_reset(self):
        base = load_neural_config(Path("configs/neural/run.toml"))
        cases = base.cases
        raw_descriptor = _descriptor()
        fingerprint = _native_fingerprint(raw_descriptor)
        raw_descriptor["fingerprint"] = fingerprint
        config = replace(base, common_view_fingerprint=fingerprint)
        run_descriptor = {
            "schema_id": "spire-gym-run-observation-v3",
            "profile_id": "M10IroncladPolicy",
            "fingerprint": envs.RUN_OBSERVATION_FINGERPRINT,
            "action_space": {"type": "Discrete", "n": 2048, "start": 0},
            "tensors": [{"key": "action_mask", "dtype": "int8", "shape": [2048]}],
        }
        calls = {}

        class FakeRun:
            def __init__(inner, *args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs
                inner.reset_calls = 0
                inner.view_calls = 0
                inner.close_calls = 0

            def observation_spec(inner):
                return run_descriptor

            def environment_spec(inner):
                return {
                    "profile_id": "M10IroncladPolicy",
                    "schema_id": "spire-gym-run-observation-v3",
                    "observation_fingerprint": envs.RUN_OBSERVATION_FINGERPRINT,
                    "rules_id": "spire-gym-native-m10-rules-v1",
                    "reward_id": "spire-gym-floor-key-terminal-reward-v1",
                    "combat_view_id": "spire-gym-combat-policy-view-v1",
                    "mode": "normal_run",
                    "partition": "train",
                    "autoreset": "disabled",
                }

            def combat_observation_spec(inner):
                return raw_descriptor

            def reset(inner, **kwargs):
                inner.reset_calls += 1
                raise AssertionError("factory must not reset")

            def combat_view(inner):
                inner.view_calls += 1
                raise AssertionError("factory must not read a combat view")

            def step(inner, *args):
                raise AssertionError("factory must not step")

            def close(inner):
                inner.close_calls += 1

        native = SimpleNamespace(
            CombatInputError=FakeInputError,
            CombatStateError=FakeStateError,
            RunInputError=FakeInputError,
            RunStateError=FakeStateError,
            build_info=lambda: {
                "extension": "spire-python",
                "revision": "533dad1c8aec86df624b3742b6b3ad1dbf1d0033",
            },
        )
        runtime = envs._NativeRuntime(_NumpyStub(), native, object, FakeRun)
        from spire_learn.neural.gym_contracts import LEGACY_CONTRACT
        synthetic_contract = SimpleNamespace(
            common_fingerprint=fingerprint,
            run_fingerprint=LEGACY_CONTRACT.run_fingerprint,
            raw_mode_mappings=LEGACY_CONTRACT.raw_mode_mappings,
            build=None,
        )
        with (
            patch.object(envs, "contract_for_revision", return_value=synthetic_contract),
            patch.object(envs, "_load_native_runtime", return_value=runtime),
            patch.object(self.api.schema, "COMMON_VIEW_FINGERPRINT", fingerprint),
            patch.object(self.api.neural_contracts, "_COMMON_VIEW_FINGERPRINT", fingerprint),
        ):
            driver = envs.make_driver(config, cases)
        self.assertEqual(calls["args"], ("ironclad", 0))
        self.assertEqual(
            calls["kwargs"],
            {"profile": "m10-ironclad-policy", "partition": "train", "record_replay": False},
        )
        self.assertEqual(driver.batch_size, 1)
        self.assertEqual(driver._provider.dispatch_action_count, 2048)
        gym = driver._provider._env
        self.assertEqual((gym.reset_calls, gym.view_calls), (0, 0))
        driver.close()
        self.assertEqual(gym.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
