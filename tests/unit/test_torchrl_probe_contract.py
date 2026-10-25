"""Pure guard, snapshot-ownership, and budget contracts for the TorchRL probe."""

import unittest

from spire_learn.contracts import ContractError, SetupBlocked


def _torchrl():
    from spire_learn.probes import torchrl

    return torchrl


class _CounterProtocol:
    def __init__(self, *, step_error=None):
        self.reset_calls = []
        self.step_calls = []
        self.close_calls = 0
        self.step_error = step_error

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append((seed, options))
        return "reset-result"

    def step(self, actions):
        self.step_calls.append(tuple(actions))
        if self.step_error is not None:
            raise self.step_error
        return "step-result"

    def close(self):
        self.close_calls += 1


class _VectorProtocol(_CounterProtocol):
    def __init__(self, reset_result, step_result, *, metadata=None):
        super().__init__()
        self.reset_result = reset_result
        self.step_result = step_result
        self.metadata = metadata or {"autoreset_mode": "same_step"}
        self.unwrapped = self
        self.num_envs = 2
        self.action_space = object()
        self.observation_space = object()
        self.single_action_space = object()
        self.single_observation_space = object()

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append((seed, options))
        return self.reset_result

    def step(self, actions):
        self.step_calls.append(tuple(actions))
        return self.step_result


class _BatchedArray:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.dtype = self.rows[0].dtype
        self.shape = (len(self.rows), *self.rows[0].shape)

    def __getitem__(self, index):
        return self.rows[index]

    def tolist(self):
        return [row.tolist() for row in self.rows]


class _ProtocolArray:
    def __init__(self, values, dtype_name, shape):
        self.values = tuple(values)
        self.dtype = type(
            "DType", (), {"name": dtype_name, "isnative": True}
        )()
        self.shape = shape

    def tolist(self):
        if self.shape == (len(self.values), 1):
            return [[value] for value in self.values]
        return list(self.values)


class _ActionBatch:
    def __init__(self, dtype_name="int64", *, native=True, shape=(2,), values=(0, 1)):
        self.dtype = type("DType", (), {"name": dtype_name, "isnative": native})()
        self.shape = shape
        self.values = values

    def tolist(self):
        return list(self.values)

    def __iter__(self):
        return iter(self.values)


class _TensorLike:
    def __init__(self, array):
        self.array = array
        self.device = type("Device", (), {"type": "cpu"})()
        self.shape = array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _TensorDictLike(dict):
    def keys(self):
        return super().keys()


class _Spec(dict):
    def __init__(self, values):
        super().__init__(values)
        self.zero_result = object()

    def zero(self):
        return self.zero_result


class _Reader:
    def __init__(self, info_spec):
        self.info_spec = None
        self.info_spec_after_read = info_spec
        self.calls = []

    def __call__(self, info, tensordict):
        self.calls.append((info, tensordict))
        self.info_spec = self.info_spec_after_read


class _PassiveWrapperProtocol:
    def __init__(self, observation_spec):
        self.observation_spec = observation_spec
        self.info_dict_reader = []
        self.registered = []

    def set_info_dict_reader(self, *, info_dict_reader):
        self.registered.append(info_dict_reader)
        self.info_dict_reader.append(info_dict_reader)
        return self


class _AutoRegisterWrapper(_PassiveWrapperProtocol):
    def __init__(self, observation_spec):
        super().__init__(observation_spec)
        self.automatic_registration_calls = 0

    def auto_register_info_dict(self, *args, **kwargs):
        self.automatic_registration_calls += 1
        raise AssertionError("the rollout-capable registration method must not run")


def _fixture_schema():
    from spire_learn.schema import parse_descriptor
    from tests.support import tiny_descriptor

    return parse_descriptor(
        tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"})
    )


def _raw_row(hp):
    from tests.support import FakeArray

    return {
        "action_mask": FakeArray("int8", (3,), (1, 1, 0)),
        "action_kind": FakeArray("int32", (3,), (1, 2, 0)),
        "action_source": FakeArray("int32", (3,), (-1, 0, -1)),
        "action_target": FakeArray("int32", (3,), (-1, 0, -1)),
        "action_features": FakeArray(
            "int32", (3, 4), (0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0)
        ),
        "player": FakeArray("int64", (), (hp,)),
    }


def _raw_batch(rows):
    return {
        key: _BatchedArray([row[key] for row in rows])
        for key in rows[0]
    }


def _current_info(ids, steps):
    return {
        "episode_id": tuple(ids),
        "_episode_id": (True, True),
        "scenario_id": ("cultist", "jaw-worm"),
        "_scenario_id": (True, True),
        "episode_steps": tuple(steps),
        "_episode_steps": (True, True),
        "outcome": ("ongoing", "ongoing"),
        "_outcome": (True, True),
    }


def _final_info():
    return {
        "episode_id": ("old-0", ""),
        "_episode_id": (True, False),
        "scenario_id": ("cultist", ""),
        "_scenario_id": (True, False),
        "episode_steps": (5, 0),
        "_episode_steps": (True, False),
        "outcome": ("death", ""),
        "_outcome": (True, False),
    }


class FacadeCallGuardTests(unittest.TestCase):
    def test_constructor_denials_latch_before_any_protocol_forward(self):
        torchrl = _torchrl()

        reset_guard = torchrl._FacadeCallGuard(batch_size=2)
        reset_target = _CounterProtocol()
        with self.assertRaises(SetupBlocked):
            reset_guard.forward_reset(
                reset_target.reset,
                seed=[7, 11],
                options={"scenario_ids": ["c", "j"]},
            )
        with self.assertRaises(SetupBlocked):
            reset_guard.forward_step(reset_target.step, (0, 1))
        self.assertEqual(reset_target.reset_calls, [])
        self.assertEqual(reset_target.step_calls, [])

        step_guard = torchrl._FacadeCallGuard(batch_size=2)
        step_target = _CounterProtocol()
        with self.assertRaises(SetupBlocked):
            step_guard.forward_step(step_target.step, (0, 1))
        self.assertEqual(step_target.step_calls, [])

    def test_only_permitted_full_reset_and_steps_forward_once(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()
        seeds = [7, 11]
        options = {"scenario_ids": ["cultist", "jaw-worm"]}

        guard.construction_complete()
        guard.permit_initial_reset(seed=seeds, options=options)
        self.assertEqual(
            guard.forward_reset(target.reset, seed=seeds, options=options),
            "reset-result",
        )
        with self.assertRaises(SetupBlocked):
            guard.forward_reset(target.reset, seed=seeds, options=options)
        self.assertEqual(target.reset_calls, [(seeds, options)])

        step_guard = torchrl._FacadeCallGuard(batch_size=2)
        step_target = _CounterProtocol()
        step_guard.construction_complete()
        step_guard.permit_initial_reset(seed=seeds, options=options)
        step_guard.forward_reset(step_target.reset, seed=seeds, options=options)
        step_guard.permit_step((1, 0))
        self.assertEqual(step_guard.forward_step(step_target.step, (1, 0)), "step-result")
        with self.assertRaises(SetupBlocked):
            step_guard.forward_step(step_target.step, (1, 0))
        self.assertEqual(step_target.step_calls, [(1, 0)])

    def test_failure_latches_and_close_attempt_is_forwarded_at_most_once(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol(step_error=RuntimeError("accepted step failed"))
        guard.construction_complete()
        guard.permit_initial_reset(seed=[7, 11], options={"scenario_ids": ["c", "j"]})
        guard.forward_reset(
            target.reset, seed=[7, 11], options={"scenario_ids": ["c", "j"]}
        )
        guard.permit_step((0, 1))
        with self.assertRaisesRegex(RuntimeError, "accepted step failed"):
            guard.forward_step(target.step, (0, 1))
        with self.assertRaises(SetupBlocked):
            guard.permit_step((0, 1))
        with self.assertRaises(SetupBlocked):
            guard.forward_step(target.step, (0, 1))
        self.assertEqual(target.step_calls, [(0, 1)])

        guard.close_once(target.close)
        guard.close_once(target.close)
        self.assertEqual(target.close_calls, 1)

    def test_delegate_return_is_counted_when_raw_capture_or_wrapper_processing_raises(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        guard.forward_reset(target.reset, seed=[7, 11], options=options)
        guard.permit_step((0, 1))

        with self.assertRaisesRegex(RuntimeError, "raw capture failed"):
            guard.forward_step(
                target.step,
                (0, 1),
                capture=lambda _raw: (_ for _ in ()).throw(RuntimeError("raw capture failed")),
            )

        self.assertEqual(target.step_calls, [(0, 1)])
        self.assertEqual(guard.counts["step_forwarded"], 1)
        self.assertEqual(guard.counts["step_delivered"], 1)
        self.assertEqual(guard.counts["step_captured"], 0)
        with self.assertRaises(SetupBlocked):
            guard.permit_step((0, 1))

    def test_wrapper_catching_a_denied_extra_step_does_not_clear_the_failed_latch(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        guard.forward_reset(target.reset, seed=[7, 11], options=options)

        def wrapper_operation():
            guard.permit_step((0, 1))
            first = guard.forward_step(target.step, (0, 1))
            try:
                guard.forward_step(target.step, (0, 1))
            except SetupBlocked:
                pass
            return first

        self.assertEqual(wrapper_operation(), "step-result")
        self.assertTrue(guard.failed)
        self.assertEqual(guard.counts["step_delivered"], 1)
        self.assertEqual(target.step_calls, [(0, 1)])

    def test_successful_raw_capture_survives_later_wrapper_processing_error(self):
        torchrl = _torchrl()
        schema = _fixture_schema()
        raw_rows = [_raw_row(31), _raw_row(29)]
        target = _VectorProtocol(
            (_raw_batch([_raw_row(13), _raw_row(21)]), _current_info(("e0", "e1"), (0, 0))),
            (
                _raw_batch(raw_rows),
                (0.0, 0.0),
                (False, False),
                (False, False),
                _current_info(("e0", "e1"), (1, 1)),
            ),
        )
        guard = torchrl._FacadeCallGuard(batch_size=2)
        facade = torchrl._GuardedVectorFacade(target, guard, schema)
        guard.construction_complete()
        options = {"scenario_ids": ["cultist", "jaw-worm"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        facade.reset(seed=[7, 11], options=options)
        guard.permit_step((0, 1))
        raw_result = facade.step((0, 1))

        with self.assertRaisesRegex(RuntimeError, "wrapper processing failed"):
            raise RuntimeError("wrapper processing failed")

        self.assertIs(raw_result, target.step_result)
        self.assertEqual(facade.last_step.main_rows[0]["player"].values, (31,))
        self.assertEqual(guard.counts["step_delivered"], 1)
        self.assertEqual(guard.counts["step_captured"], 1)
        self.assertEqual(target.step_calls, [(0, 1)])

    def test_failed_close_is_not_retried_and_wrong_width_step_never_forwards(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        guard.forward_reset(target.reset, seed=[7, 11], options=options)
        guard.permit_step((0, 1))
        with self.assertRaises(SetupBlocked):
            guard.forward_step(target.step, (0,))
        with self.assertRaises(SetupBlocked):
            guard.forward_step(target.step, (0, 1))
        self.assertEqual(target.step_calls, [])

        close_calls = []

        def failing_close():
            close_calls.append("close")
            raise RuntimeError("close failed")

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            guard.close_once(failing_close)
        self.assertIsNone(guard.close_once(failing_close))
        self.assertEqual(close_calls, ["close"])

    def test_reset_guard_rejects_extra_options_and_forwarded_bool_seeds(self):
        torchrl = _torchrl()
        target = _CounterProtocol()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        guard.construction_complete()
        with self.assertRaises(SetupBlocked):
            guard.permit_initial_reset(
                seed=[7, 11],
                options={"scenario_ids": ["c", "j"], "reset_mask": [True, True]},
            )
        self.assertEqual(target.reset_calls, [])

        guard = torchrl._FacadeCallGuard(batch_size=2)
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[1, 2], options=options)
        with self.assertRaises(SetupBlocked):
            guard.forward_reset(target.reset, seed=[True, 2], options=options)
        self.assertEqual(target.reset_calls, [])

    def test_close_disables_all_later_permits_and_forwards_even_when_close_raises(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        guard.forward_reset(target.reset, seed=[7, 11], options=options)

        def failing_close():
            target.close()
            raise RuntimeError("close failed")

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            guard.close_once(failing_close)
        with self.assertRaises(SetupBlocked):
            guard.permit_step((0, 1))
        with self.assertRaises(SetupBlocked):
            guard.permit_initial_reset(seed=[7, 11], options=options)
        with self.assertRaises(SetupBlocked):
            guard.forward_step(target.step, (0, 1))
        self.assertEqual(target.step_calls, [])
        self.assertEqual(target.reset_calls, [([7, 11], options)])
        self.assertEqual(target.close_calls, 1)

        pending_guard = torchrl._FacadeCallGuard(batch_size=2)
        pending_target = _CounterProtocol()
        pending_guard.construction_complete()
        pending_guard.permit_initial_reset(seed=[7, 11], options=options)

        def pending_close():
            pending_target.close()
            raise RuntimeError("pending close failed")

        with self.assertRaisesRegex(RuntimeError, "pending close failed"):
            pending_guard.close_once(pending_close)
        with self.assertRaises(SetupBlocked):
            pending_guard.forward_reset(
                pending_target.reset, seed=[7, 11], options=options
            )
        self.assertEqual(pending_target.reset_calls, [])
        self.assertEqual(pending_target.close_calls, 1)

    def test_runtime_guard_rejects_non_int64_non_native_or_partial_actions_before_forward(self):
        torchrl = _torchrl()
        bad_batches = (
            _ActionBatch("int32"),
            _ActionBatch("int64", native=False),
            _ActionBatch("int64", shape=(1,)),
        )
        for actions in bad_batches:
            with self.subTest(actions=(actions.dtype.name, actions.dtype.isnative, actions.shape)):
                guard = torchrl._FacadeCallGuard(batch_size=2, require_native_actions=True)
                target = _CounterProtocol()
                guard.construction_complete()
                options = {"scenario_ids": ["c", "j"]}
                guard.permit_initial_reset(seed=[7, 11], options=options)
                guard.forward_reset(target.reset, seed=[7, 11], options=options)
                guard.permit_step((0, 1))
                with self.assertRaises(SetupBlocked):
                    guard.forward_step(target.step, actions)
                self.assertEqual(target.step_calls, [])


class GuardedFacadeTests(unittest.TestCase):
    def test_facade_denies_constructor_reset_then_captures_each_accepted_call(self):
        torchrl = _torchrl()
        schema = _fixture_schema()
        reset_rows = [_raw_row(13), _raw_row(21)]
        reset_result = (
            _raw_batch(reset_rows),
            _current_info(("episode-0", "episode-1"), (0, 3)),
        )
        main_rows = [_raw_row(31), _raw_row(29)]
        final_row = _raw_row(8)
        final_obs = [final_row, None]
        step_info = _current_info(("replacement-0", "episode-1"), (0, 4))
        step_info.update(
            {
                "final_obs": final_obs,
                "_final_obs": (True, False),
                "final_info": _final_info(),
                "_final_info": (True, False),
            }
        )
        step_result = (
            _raw_batch(main_rows),
            (1.0, 0.0),
            (True, False),
            (False, False),
            step_info,
        )
        target = _VectorProtocol(reset_result, step_result)
        denied_guard = torchrl._FacadeCallGuard(batch_size=2)
        denied_facade = torchrl._GuardedVectorFacade(target, denied_guard, schema)
        with self.assertRaises(SetupBlocked):
            denied_facade.reset(seed=[7, 11], options={"scenario_ids": ["c", "j"]})
        self.assertEqual(target.reset_calls, [])

        guard = torchrl._FacadeCallGuard(batch_size=2, require_native_actions=True)
        facade = torchrl._GuardedVectorFacade(target, guard, schema)
        self.assertEqual(facade.autoreset_mode, "same_step")
        self.assertIs(facade.action_space, target.action_space)
        options = {"scenario_ids": ["cultist", "jaw-worm"]}
        guard.construction_complete()
        guard.permit_initial_reset(seed=[7, 11], options=options)
        returned_reset = facade.reset(seed=[7, 11], options=options)
        returned_reset[0]["player"]._values = (90, 91)
        self.assertEqual(facade.last_reset.rows[0]["player"].values, (13,))

        guard.permit_step((0, 1))
        returned_step = facade.step(_ActionBatch(values=(0, 1)))
        returned_step[0]["player"].rows[0]._values = (99,)
        final_row["player"] = _raw_row(100)["player"]
        final_obs[1] = _raw_row(101)
        self.assertEqual(facade.last_step.main_rows[0]["player"].values, (31,))
        self.assertEqual(facade.last_step.final_rows[0]["player"].values, (8,))
        self.assertIsNone(facade.last_step.final_rows[1])
        self.assertEqual(facade.last_step.final_presence, (True, False))
        self.assertEqual(target.reset_calls, [([7, 11], options)])
        self.assertEqual(target.step_calls, [(0, 1)])


class WrapperStepErrorPriorityTests(unittest.TestCase):
    def _started_guard(self, torchrl):
        guard = torchrl._FacadeCallGuard(batch_size=2)
        guard.construction_complete()
        options = {"scenario_ids": ["c", "j"]}
        guard.permit_initial_reset(seed=[7, 11], options=options)
        guard.forward_reset(_CounterProtocol().reset, seed=[7, 11], options=options)
        return guard

    def test_delegate_fault_keeps_native_simulator_fault_classification(self):
        from spire_learn.envs.native import NativeErrorTypes, native_failure
        from tests.support import FakeNativeErrors

        torchrl = _torchrl()
        fault = FakeNativeErrors.FaultError("combat fault detail")
        target = _CounterProtocol(step_error=fault)
        guard = self._started_guard(torchrl)
        guard.permit_step((0, 1))
        denied_before = guard.counts["denied_calls"]

        wrapper_error = None
        try:
            guard.forward_step(target.step, (0, 1))
        except BaseException as error:
            wrapper_error = error

        self.assertTrue(guard.failed)
        self.assertEqual(guard.counts["step_delivered"], 0)
        self.assertEqual(guard.counts["denied_calls"], denied_before)
        with self.assertRaises(FakeNativeErrors.FaultError) as raised:
            torchrl._raise_step_error_or_denial(
                guard,
                denied_calls_before=denied_before,
                wrapper_error=wrapper_error,
            )
        self.assertIs(raised.exception, fault)
        error_types = NativeErrorTypes(
            FakeNativeErrors.InputError,
            FakeNativeErrors.StateError,
            FakeNativeErrors.SetupError,
            FakeNativeErrors.FaultError,
        )
        self.assertEqual(
            native_failure(raised.exception, error_types=error_types).kind,
            "simulator_fault",
        )

    def test_wrapper_return_after_catching_delegate_fault_preserves_original(self):
        from spire_learn.envs.native import NativeErrorTypes, native_failure
        from tests.support import FakeNativeErrors

        torchrl = _torchrl()
        fault = FakeNativeErrors.FaultError("caught combat fault")
        target = _CounterProtocol(step_error=fault)
        guard = self._started_guard(torchrl)
        guard.permit_step((0, 1))
        denied_before = guard.counts["denied_calls"]

        def wrapper_callback():
            try:
                guard.forward_step(target.step, (0, 1))
            except FakeNativeErrors.FaultError:
                return "wrapper swallowed delegate exception"

        self.assertEqual(wrapper_callback(), "wrapper swallowed delegate exception")
        self.assertTrue(guard.failed)
        self.assertEqual(guard.counts["denied_calls"], denied_before)
        with self.assertRaises(FakeNativeErrors.FaultError) as raised:
            torchrl._raise_step_error_or_denial(
                guard, denied_calls_before=denied_before, wrapper_error=None
            )
        self.assertIs(raised.exception, fault)
        error_types = NativeErrorTypes(
            FakeNativeErrors.InputError,
            FakeNativeErrors.StateError,
            FakeNativeErrors.SetupError,
            FakeNativeErrors.FaultError,
        )
        self.assertEqual(
            native_failure(raised.exception, error_types=error_types).kind,
            "simulator_fault",
        )

    def test_caught_guard_denial_is_reported_when_wrapper_returns(self):
        torchrl = _torchrl()
        target = _CounterProtocol()
        guard = self._started_guard(torchrl)
        guard.permit_step((0, 1))
        denied_before = guard.counts["denied_calls"]

        def wrapper_callback():
            try:
                return guard.forward_step(target.step, (1, 1))
            except SetupBlocked:
                return "wrapper caught denial"

        self.assertEqual(wrapper_callback(), "wrapper caught denial")
        with self.assertRaises(SetupBlocked):
            torchrl._raise_step_error_or_denial(
                guard, denied_calls_before=denied_before, wrapper_error=None
            )
        self.assertEqual(target.step_calls, [])

    def test_wrapper_error_after_raw_delivery_wins_over_denied_latch(self):
        torchrl = _torchrl()
        target = _CounterProtocol()
        guard = self._started_guard(torchrl)
        guard.permit_step((0, 1))
        guard.forward_step(target.step, (0, 1))
        denied_before = guard.counts["denied_calls"]
        try:
            guard.forward_step(target.step, (0, 1))
        except SetupBlocked:
            pass
        wrapper_error = RuntimeError("wrapper failed after native delivery")

        with self.assertRaises(RuntimeError) as raised:
            torchrl._raise_step_error_or_denial(
                guard,
                denied_calls_before=denied_before,
                wrapper_error=wrapper_error,
            )
        self.assertIs(raised.exception, wrapper_error)
        self.assertEqual(guard.counts["step_delivered"], 1)
        self.assertEqual(target.step_calls, [(0, 1)])


class PassiveRegistrationTests(unittest.TestCase):
    def test_bridge_initializes_reader_from_known_zero_spec_without_auto_rollout(self):
        register = _torchrl()._passive_register_info_reader
        observation_spec = _Spec({"observation": ("int32", (2, 3)), "action_mask": ("bool", (2, 4))})
        expected_final_spec = _Spec(dict(observation_spec))
        reader = _Reader({"final": expected_final_spec})
        wrapper = _PassiveWrapperProtocol(observation_spec)

        result = register(wrapper, reader)

        self.assertIs(result, wrapper)
        self.assertEqual(reader.calls, [({}, observation_spec.zero_result)])
        self.assertEqual(wrapper.registered, [reader])
        self.assertEqual(wrapper.info_dict_reader, [reader])

    def test_wrapper_subclass_routes_auto_registration_to_passive_bridge(self):
        torchrl = _torchrl()
        observation_spec = _Spec({"observation": ("int32", (2, 3))})
        reader = _Reader({"final": _Spec(dict(observation_spec))})
        wrapper_type = torchrl._passive_gym_wrapper_class(_AutoRegisterWrapper)
        wrapper = wrapper_type(observation_spec)

        result = wrapper.auto_register_info_dict(info_dict_reader=reader)

        self.assertIs(result, wrapper)
        self.assertEqual(wrapper.automatic_registration_calls, 0)
        self.assertEqual(wrapper.registered, [reader])

    def test_bridge_rejects_incomplete_spec_before_reader_registration(self):
        register = _torchrl()._passive_register_info_reader
        observation_spec = _Spec({"observation": ("int32", (2, 3)), "action_mask": ("bool", (2, 4))})
        reader = _Reader({"final": {"observation": ("int32", (2, 3))}})
        wrapper = _PassiveWrapperProtocol(observation_spec)

        with self.assertRaises(SetupBlocked):
            register(wrapper, reader)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(wrapper.registered, [])


class RawCaptureTests(unittest.TestCase):
    def test_present_final_rows_are_owned_and_none_rows_remain_absent(self):
        torchrl = _torchrl()
        raw_row = _raw_row(17)
        source = [raw_row, None]

        captured = torchrl._snapshot_final_rows(
            source, (True, False), _fixture_schema()
        )
        raw_row["player"] = _raw_row(99)["player"]
        source[1] = _raw_row(101)

        self.assertEqual(captured[0]["player"].values, (17,))
        self.assertIsNone(captured[1])

    def test_raw_reset_and_step_captures_own_same_call_data_before_wrapper_mutation(self):
        torchrl = _torchrl()
        schema = _fixture_schema()
        reset_rows = [_raw_row(13), _raw_row(21)]
        reset_info = _current_info(("episode-0", "episode-1"), (0, 3))
        reset_capture = torchrl._capture_reset_result(
            (_raw_batch(reset_rows), reset_info), schema, width=2
        )
        reset_rows[0]["player"] = _raw_row(90)["player"]
        self.assertEqual(reset_capture.rows[0]["player"].values, (13,))

        main_rows = [_raw_row(31), _raw_row(29)]
        final_row = _raw_row(8)
        final_rows = [final_row, None]
        info = _current_info(("episode-new", "episode-1"), (0, 4))
        info.update(
            {
                "final_obs": final_rows,
                "_final_obs": (True, False),
                "final_info": _final_info(),
                "_final_info": (True, False),
            }
        )
        step_capture = torchrl._capture_step_result(
            (_raw_batch(main_rows), (1.0, 0.0), (True, False), (False, False), info),
            schema,
            width=2,
        )

        # GymLikeEnv's terminal reader fills None placeholders and reads the
        # same mutable arrays only after the guard's capture callback returns.
        final_row["player"] = _raw_row(99)["player"]
        main_rows[0]["player"] = _raw_row(100)["player"]
        final_rows[1] = _raw_row(101)

        self.assertEqual(step_capture.final_rows[0]["player"].values, (8,))
        self.assertIsNone(step_capture.final_rows[1])
        self.assertEqual(step_capture.main_rows[0]["player"].values, (31,))
        self.assertEqual(step_capture.current_info["episode_id"], ("episode-new", "episode-1"))
        self.assertEqual(step_capture.final_presence, (True, False))
        self.assertEqual(step_capture.terminated, (True, False))
        self.assertEqual(step_capture.truncated, (False, False))
        self.assertEqual(step_capture.rewards, (1.0, 0.0))

    def test_final_row_capture_rejects_live_payloads_and_missing_ended_rows(self):
        snapshot_final_rows = _torchrl()._snapshot_final_rows
        schema = _fixture_schema()
        with self.assertRaises(ContractError):
            snapshot_final_rows([None, None], (True, False), schema)
        with self.assertRaises(ContractError):
            snapshot_final_rows([_raw_row(17), _raw_row(18)], (True, False), schema)


class ObservationMappingTests(unittest.TestCase):
    def test_mapped_tensor_rows_match_all_public_descriptor_tensors(self):
        rows = (_raw_row(13), _raw_row(21))
        tensor_dict = {
            key: _TensorLike(_BatchedArray([row[key] for row in rows]))
            for key in rows[0]
        }

        mapped = _torchrl()._mapped_observation_rows(
            tensor_dict, _fixture_schema(), width=2
        )

        self.assertEqual(mapped[0]["player"].values, (13,))
        self.assertEqual(mapped[1]["player"].values, (21,))
        self.assertEqual(mapped[0]["action_mask"].values, (1, 1, 0))

    def test_mapping_selects_descriptor_leaves_from_full_tensordict(self):
        rows = (_raw_row(13), _raw_row(21))
        tensor_dict = _TensorDictLike(
            {
                key: _TensorLike(_BatchedArray([row[key] for row in rows]))
                for key in rows[0]
            }
        )
        tensor_dict.update(
            {
                "action": object(),
                "reward": object(),
                "done": object(),
                "terminated": object(),
                "truncated": object(),
                "final": object(),
            }
        )

        mapped = _torchrl()._mapped_observation_rows(
            tensor_dict, _fixture_schema(), width=2
        )

        self.assertEqual(set(mapped[0]), set(rows[0]))
        self.assertEqual(mapped[1]["player"].values, (21,))

    def test_missing_or_wrong_width_mapped_tensor_is_a_named_mapping_failure(self):
        rows = (_raw_row(13), _raw_row(21))
        missing = {
            key: _TensorLike(_BatchedArray([row[key] for row in rows]))
            for key in rows[0]
            if key != "player"
        }
        with self.assertRaises(ContractError) as raised:
            _torchrl()._mapped_observation_rows(missing, _fixture_schema(), width=2)
        self.assertEqual(getattr(raised.exception, "check", None), "observation_mapping")

        wrong_width = dict(missing)
        wrong_width["player"] = _TensorLike(_BatchedArray([_raw_row(13)["player"]]))
        with self.assertRaises(ContractError) as raised:
            _torchrl()._mapped_observation_rows(wrong_width, _fixture_schema(), width=2)
        self.assertEqual(getattr(raised.exception, "check", None), "observation_mapping")


class TorchRLColumnTypeTests(unittest.TestCase):
    def test_flags_must_be_builtin_booleans_and_rewards_finite_non_boolean_numbers(self):
        torchrl = _torchrl()
        captured = torchrl._CapturedStep(
            main_rows=({}, {}),
            final_rows=(None, None),
            current_info={},
            final_info=None,
            final_presence=None,
            final_info_presence=None,
            final_info_masks=None,
            rewards=(1.0, 0.0),
            terminated=(True, False),
            truncated=(False, False),
            ended=(True, False),
        )

        def values(*, rewards=(1.0, 0.0), terminated=(True, False)):
            return _TensorDictLike(
                {
                    "reward": _TensorLike(
                        _ProtocolArray(rewards, "float32", (2, 1))
                    ),
                    "terminated": _TensorLike(
                        _ProtocolArray(terminated, "bool", (2,))
                    ),
                    "truncated": _TensorLike(
                        _ProtocolArray((False, False), "bool", (2,))
                    ),
                    "done": _TensorLike(
                        _ProtocolArray((True, False), "bool", (2,))
                    ),
                }
            )

        torchrl._captured_flags_and_rewards_match(
            values(), captured, width=2
        )
        with self.assertRaises(ContractError) as raised:
            torchrl._captured_flags_and_rewards_match(
                values(terminated=(1, 0)), captured, width=2
            )
        self.assertEqual(getattr(raised.exception, "check", None), "presence_masks")

        with self.assertRaises(ContractError) as raised:
            torchrl._captured_flags_and_rewards_match(
                values(rewards=(True, False)), captured, width=2
            )
        self.assertEqual(getattr(raised.exception, "check", None), "presence_masks")


class CloseHandlingTests(unittest.TestCase):
    def test_guarded_cleanup_collects_errors_without_double_forwarding_close(self):
        torchrl = _torchrl()
        guard = torchrl._FacadeCallGuard(batch_size=2)
        target = _CounterProtocol()

        def wrapper_close_fails_before_forward():
            raise ValueError("wrapper close failed")

        def facade_close_fails_after_forward():
            guard.close_once(target.close)
            raise RuntimeError("native close failed")

        errors = torchrl._close_guarded_resources(
            wrapper_close_fails_before_forward,
            facade_close_fails_after_forward,
            guard,
        )
        failures = []
        control, ordinary_failure = torchrl._record_close_errors(errors, failures)

        self.assertIsNone(control)
        self.assertTrue(ordinary_failure)
        self.assertEqual(target.close_calls, 1)
        self.assertEqual(guard.counts["close_forwarded"], 1)
        self.assertEqual(
            [(failure.reason, failure.phase) for failure in failures],
            [("wrapper close failed", "close"), ("native close failed", "close")],
        )

    def test_close_side_process_control_retains_prior_failures_and_primary(self):
        torchrl = _torchrl()
        prior = [ContractError("earlier contract failure")]
        primary = RuntimeError("writer failed")
        secondary = KeyboardInterrupt("close interrupted")

        control, ordinary_failure = torchrl._record_close_errors(
            (secondary,), prior, primary=primary
        )

        self.assertIs(control, secondary)
        self.assertFalse(ordinary_failure)
        self.assertTrue(any("writer failed" in note for note in secondary.__notes__))
        self.assertEqual(len(prior), 1)

        failures = []
        multiple, ordinary_failure = torchrl._record_close_errors(
            (ValueError("first close"), OSError("second close"), secondary),
            failures,
        )
        self.assertIs(multiple, secondary)
        self.assertTrue(ordinary_failure)
        self.assertEqual(
            [failure.reason for failure in failures], ["first close", "second close"]
        )
        torchrl.vector._note_prior_failures(multiple, failures)
        self.assertTrue(any("first close" in note for note in multiple.__notes__))

class DiagnosticSamplingBudgetTests(unittest.TestCase):
    def test_draw_budget_is_reserved_before_sampling_and_failure_does_not_refund(self):
        budget = _torchrl()._DiagnosticSampleBudget(max_samples=4)
        observations = []

        def fail_after_observing_reservation():
            observations.append(budget.samples_reserved)
            raise RuntimeError("sampler failed")

        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            budget.sample(rows=2, sampler=fail_after_observing_reservation)
        self.assertEqual(observations, [2])
        self.assertEqual(budget.samples_reserved, 2)

        called = []
        with self.assertRaises(ContractError):
            budget.sample(rows=3, sampler=lambda: called.append(True))
        self.assertEqual(called, [])
        self.assertEqual(budget.samples_reserved, 2)

    def test_mask_diagnostic_rejects_illegal_sample_and_nonfinite_log_prob(self):
        validate = _torchrl()._validate_mask_diagnostic
        masks = ((True, False, True), (False, True, True))

        validate(masks, (0, 1), (0.0, -0.5), width=2, action_count=3)
        with self.assertRaises(ContractError) as raised:
            validate(masks, (1, 2), (0.0, -0.5), width=2, action_count=3)
        self.assertEqual(getattr(raised.exception, "check", None), "mask_sampling")

        with self.assertRaises(ContractError) as raised:
            validate(masks, (0, 1), (0.0, float("inf")), width=2, action_count=3)
        self.assertEqual(getattr(raised.exception, "check", None), "mask_log_prob")

    def test_mask_diagnostic_requires_boolean_nonempty_full_width_mask(self):
        validate = _torchrl()._validate_mask_diagnostic
        invalid_masks = (
            ((True, False, True),),
            ((True, 0, True), (False, True, True)),
            ((False, False, False), (False, True, True)),
        )
        for masks in invalid_masks:
            with self.subTest(masks=masks), self.assertRaises(ContractError) as raised:
                validate(masks, (0, 1), (0.0, -0.5), width=2, action_count=3)
            self.assertEqual(getattr(raised.exception, "check", None), "mask_sampling")


class TorchRLVerdictTests(unittest.TestCase):
    def test_operational_and_adaptation_precedence_preserve_required_failures(self):
        torchrl = _torchrl()
        passed = {name: "pass" for name in torchrl.TORCHRL_CHECKS}

        self.assertEqual(
            torchrl._torchrl_probe_verdict(
                passed,
                runtime_available=True,
                operational_failure=True,
                adaptation_verified=False,
            ),
            "blocked",
        )
        self.assertEqual(
            torchrl._torchrl_probe_verdict(
                {**passed, "mask_sampling": "fail"},
                runtime_available=True,
                operational_failure=True,
                adaptation_verified=True,
            ),
            "incompatible",
        )
        self.assertEqual(
            torchrl._torchrl_probe_verdict(
                passed,
                runtime_available=True,
                operational_failure=False,
                adaptation_verified=True,
            ),
            "adaptable",
        )
        self.assertEqual(
            torchrl._torchrl_probe_verdict(
                passed,
                runtime_available=False,
                operational_failure=False,
                adaptation_verified=True,
            ),
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()
