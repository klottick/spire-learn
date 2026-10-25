"""Single-case collection over scripted public environment protocols."""

from importlib import import_module
import math
import unittest

from spire_learn.contracts import Choice
from spire_learn.envs.native import NativeErrorTypes, native_failure
from spire_learn.experiments.budget import Budget
from tests.support import (
    FakeClock,
    FakeNativeErrors,
    ScriptedAgent,
    ScriptedEnv,
    case_execution,
    fixture_contract,
    observation,
    tiny_descriptor,
)


def _run_case(test_case):
    try:
        module = import_module("spire_learn.collection.single")
    except ImportError as error:
        test_case.fail(f"single-case run_case API is missing: {error}")
    runner = getattr(module, "run_case", None)
    test_case.assertTrue(callable(runner), "single-case run_case API is missing")

    def invoke(*args, **kwargs):
        try:
            return runner(*args, **kwargs)
        except Exception as error:
            test_case.fail(f"run_case raised instead of returning a CaseResult: {error!r}")

    return invoke


class SingleCaseTests(unittest.TestCase):
    def test_terminal_transition_records_public_outcome(self):
        run_case = _run_case(self)

        clock = FakeClock()
        env = ScriptedEnv(
            (observation(), {}),
            [
                (
                    observation(mask=(0, 0, 0), kinds=(0, 0, 0)),
                    1.0,
                    True,
                    False,
                    {
                        "outcome": 1,
                        "episode_steps": 1,
                        "hp_before_healing": 8,
                        "hp_after_cleanup": 10,
                    },
                )
            ],
        )
        transitions = []

        result = run_case(
            case_execution(),
            ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env,
            budget=Budget(10.0, clock),
            max_decisions=4,
            on_transition=transitions.append,
            contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome, result.steps), ("terminal", "victory", 1))
        self.assertEqual(result.reward_sum, 1.0)
        self.assertEqual(
            dict(result.final_resources),
            {"hp_before_healing": 8, "hp_after_cleanup": 10},
        )
        self.assertEqual(env.reset_calls, [(17, {"scenario_id": "fixture-scenario"})])
        self.assertEqual(env.step_calls, [0])
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(
            (
                result.observed_descriptor["schema_id"],
                result.observed_descriptor["content_id"],
                result.observed_descriptor["fingerprint"],
            ),
            ("spire-learn-fixture-v1", "spire-learn-fixture-content-v1", "a" * 64),
        )
        self.assertIsNone(result.native_build_info)
        self.assertEqual(len(transitions), 1)
        transition = transitions[0]
        self.assertEqual((transition.action, transition.reward), (0, 1.0))
        self.assertTrue(transition.terminated)
        self.assertFalse(transition.truncated)
        self.assertIsNone(transition.replacement)

    def test_both_ending_flags_preserve_terminal_outcome(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [
                (
                    observation(mask=(0, 0, 0), kinds=(0, 0, 0)),
                    1.0,
                    True,
                    True,
                    {"outcome": 1},
                )
            ],
        )
        transitions = []

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, on_transition=transitions.append,
            contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome, result.steps), ("terminal", "victory", 1))
        self.assertEqual(len(transitions), 1)
        self.assertTrue(transitions[0].terminated)
        self.assertTrue(transitions[0].truncated)

    def test_truncation_is_distinct(self):
        run_case = _run_case(self)

        env = ScriptedEnv(
            (observation(), {}),
            [
                (observation(), 0.25, False, False, {"outcome": 0}),
                (observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 0.5, False, True, {"outcome": 0}),
            ],
        )
        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome), ("env_truncated", None))
        self.assertEqual((result.steps, result.reward_sum), (2, 0.75))
        self.assertEqual(len(env.step_calls), 2)

    def test_decision_cap_is_budget_stop(self):
        run_case = _run_case(self)

        env = ScriptedEnv(
            (observation(), {}),
            [(observation(), 0.0, False, False, {}) for _ in range(3)],
        )
        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(0), Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=2, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.stop_reason), ("budget_stopped", "decision_cap"))
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(env.step_calls), 2)

    def test_illegal_action_has_zero_steps(self):
        run_case = _run_case(self)
        for choice in (Choice(True), Choice(1)):
            with self.subTest(action=choice.action):
                env = ScriptedEnv((observation(mask=(1, 0, 0)), {}), [])
                result = run_case(
                    case_execution(), ScriptedAgent([choice]),
                    env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
                    max_decisions=4, contract=fixture_contract(),
                )

                self.assertEqual((result.status, result.steps), ("agent_error", 0))
                self.assertEqual(env.step_calls, [])
                self.assertEqual(env.close_calls, 1)

    def test_empty_nonterminal_mask_is_contract_error(self):
        run_case = _run_case(self)

        env = ScriptedEnv((observation(mask=(0, 0, 0), kinds=(0, 0, 0)), {}), [])
        policy = ScriptedAgent([Choice(0)])
        result = run_case(
            case_execution(), policy,
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps), ("contract_error", 0))
        self.assertEqual(env.step_calls, [])
        self.assertEqual(policy.choices, [Choice(0)])

    def test_descriptor_mismatch_closes_before_reset(self):
        run_case = _run_case(self)
        descriptor = tiny_descriptor()
        descriptor["content_id"] = "not-allowed"
        env = ScriptedEnv((observation(), {}), [], descriptor=descriptor)
        factory_calls = []

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: factory_calls.append(case) or env,
            budget=Budget(10.0, FakeClock()), max_decisions=4,
            contract=fixture_contract(),
        )

        self.assertEqual(result.status, "setup_error")
        self.assertFalse(result.started)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(env.reset_calls, [])
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(result.primary_failure.kind, "contract_error")

    def test_expected_fingerprint_mismatch_closes_before_reset(self):
        run_case = _run_case(self)
        env = ScriptedEnv((observation(), {}), [])
        factory_calls = []

        result = run_case(
            case_execution(expected_fingerprint="b" * 64), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: factory_calls.append(case) or env,
            budget=Budget(10.0, FakeClock()), max_decisions=4,
            contract=fixture_contract(),
        )

        self.assertEqual(result.status, "setup_error")
        self.assertEqual(len(factory_calls), 1)
        self.assertFalse(result.started)
        self.assertEqual(env.reset_calls, [])
        self.assertEqual(env.close_calls, 1)

    def test_factory_failure_has_no_environment_to_close(self):
        run_case = _run_case(self)
        error = RuntimeError("factory rejected case")

        def fail_factory(case):
            raise error

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=fail_factory, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.started), ("setup_error", False))
        self.assertEqual(result.primary_failure.kind, "setup_error")
        self.assertEqual(result.primary_failure.phase, "factory")

    def test_reset_failure_is_unstarted_and_preserves_typed_kind(self):
        run_case = _run_case(self)

        class ResetFailureEnv(ScriptedEnv):
            def reset(self, *, seed=None, options=None):
                self.reset_calls.append((seed, options))
                raise FakeNativeErrors.SetupError("reset setup failed")

        env = ResetFailureEnv((observation(), {}), [])
        errors = NativeErrorTypes(
            FakeNativeErrors.InputError,
            FakeNativeErrors.StateError,
            FakeNativeErrors.SetupError,
            FakeNativeErrors.FaultError,
        )
        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4,
            classify_failure=lambda error: native_failure(error, error_types=errors),
            contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.started), ("setup_error", False))
        self.assertEqual(result.primary_failure.kind, "setup_error")
        self.assertEqual(result.primary_failure.phase, "reset")
        self.assertEqual(env.close_calls, 1)

    def test_typed_fault_is_not_retried(self):
        run_case = _run_case(self)
        env = ScriptedEnv((observation(), {}), [FakeNativeErrors.FaultError("native fault")])
        errors = NativeErrorTypes(
            FakeNativeErrors.InputError,
            FakeNativeErrors.StateError,
            FakeNativeErrors.SetupError,
            FakeNativeErrors.FaultError,
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(1)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4,
            classify_failure=lambda error: native_failure(error, error_types=errors),
            contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps), ("simulator_fault", 0))
        self.assertEqual(len(env.step_calls), 1)
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(result.primary_failure.kind, "simulator_fault")

    def test_arbitrary_runtime_error_is_not_a_simulator_fault(self):
        run_case = _run_case(self)
        env = ScriptedEnv((observation(), {}), [RuntimeError("opaque step failure")])

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.primary_failure.kind), ("exec_error", "exec_error"))
        self.assertEqual(len(env.step_calls), 1)

    def test_terminal_step_delivered_after_deadline_is_preserved(self):
        run_case = _run_case(self)
        clock = FakeClock()
        transitions = []
        env = ScriptedEnv(
            (observation(), {}),
            [(observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": 1})],
            on_step=lambda: clock.advance(2.0),
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(1.0, clock),
            max_decisions=4, on_transition=transitions.append, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome, result.steps), ("terminal", "victory", 1))
        self.assertEqual(len(transitions), 1)
        self.assertEqual(len(env.step_calls), 1)
        self.assertEqual(env.close_calls, 1)

    def test_trace_and_close_errors_preserve_terminal_result_without_retry(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [(observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": 1})],
        )

        def fail_close():
            env.close_calls += 1
            raise RuntimeError("close failed")

        env.close = fail_close

        def fail_trace(transition):
            raise RuntimeError("trace failed")

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, on_transition=fail_trace, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome), ("terminal", "victory"))
        self.assertIsNone(result.primary_failure)
        self.assertEqual([failure.phase for failure in result.secondary_errors], ["trace", "close"])
        self.assertEqual(len(env.step_calls), 1)
        self.assertEqual(env.close_calls, 1)

    def test_ongoing_trace_failure_is_primary_exec_error(self):
        run_case = _run_case(self)
        env = ScriptedEnv((observation(), {}), [(observation(), 0.0, False, False, {})])

        def fail_close():
            env.close_calls += 1
            raise RuntimeError("close failed after trace")

        env.close = fail_close

        def fail_trace(transition):
            raise RuntimeError("trace failed")

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, on_transition=fail_trace, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps), ("exec_error", 1))
        self.assertEqual(result.primary_failure.phase, "trace")
        self.assertEqual([failure.phase for failure in result.secondary_errors], ["close"])
        self.assertEqual(env.close_calls, 1)

    def test_process_control_from_trace_propagates_after_close(self):
        run_case = _run_case(self)

        class StopRun(BaseException):
            pass

        env = ScriptedEnv((observation(), {}), [(observation(), 0.0, False, False, {})])
        close_error = RuntimeError("close also failed")

        def fail_close():
            env.close_calls += 1
            raise close_error

        env.close = fail_close
        stop = StopRun("stop now")

        def stop_trace(transition):
            raise stop

        with self.assertRaises(StopRun) as raised:
            run_case(
                case_execution(), ScriptedAgent([Choice(0)]),
                env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
                max_decisions=4, on_transition=stop_trace, contract=fixture_contract(),
            )

        self.assertIs(raised.exception, stop)
        self.assertEqual(env.close_calls, 1)
        self.assertTrue(any("close also failed" in note for note in getattr(stop, "__notes__", ())))

    def test_terminal_trace_failure_is_not_lost_when_close_raises_process_control(self):
        run_case = _run_case(self)

        class StopClose(KeyboardInterrupt):
            pass

        env = ScriptedEnv(
            (observation(), {}),
            [(observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": 1})],
        )
        stop = StopClose("close interrupted")

        def fail_close():
            env.close_calls += 1
            raise stop

        env.close = fail_close

        def fail_trace(transition):
            raise RuntimeError("terminal trace failed")

        with self.assertRaises(StopClose) as raised:
            run_case(
                case_execution(), ScriptedAgent([Choice(0)]),
                env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
                max_decisions=4, on_transition=fail_trace, contract=fixture_contract(),
            )

        self.assertIs(raised.exception, stop)
        self.assertEqual(env.close_calls, 1)
        notes = "\n".join(getattr(stop, "__notes__", ()))
        self.assertIn("trace", notes)
        self.assertIn("terminal", notes)
        self.assertIn("victory", notes)

    def test_unknown_terminal_outcome_is_not_inferred_from_reward(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [(observation(mask=(0, 0, 0), kinds=(0, 0, 0)), -1.0, True, False, {"outcome": 999})],
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome, result.reward_sum), ("terminal", None, -1.0))

    def test_public_decision_excludes_case_ids_seeds_and_opaque_state(self):
        run_case = _run_case(self)
        sentinel = object()

        class CapturePolicy:
            def __init__(self):
                self.seen = []

            def init_state(self):
                return sentinel

            def act(self, public, state):
                self.seen.append((public, state))
                return Choice(0, state)

        policy = CapturePolicy()
        env = ScriptedEnv(
            (observation(), {}),
            [
                (observation(), 0.0, False, False, {}),
                (observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": 1}),
            ],
        )
        transitions = []

        result = run_case(
            case_execution(), policy, env_factory=lambda case: env,
            budget=Budget(10.0, FakeClock()), max_decisions=3,
            on_transition=transitions.append, contract=fixture_contract(),
        )

        public, first_state = policy.seen[0]
        self.assertEqual(
            set(vars(public)),
            {"observation", "candidates", "mask", "schema_fingerprint", "action_mapping_id"},
        )
        self.assertNotIn("fixture-execution", repr(public))
        self.assertNotIn("fixture-case", repr(public))
        self.assertNotIn("17", repr(public))
        self.assertNotIn("29", repr(public))
        self.assertEqual(public.action_mapping_id, "spire-learn-native-current-index-v1")
        self.assertIs(first_state, sentinel)
        self.assertIs(policy.seen[1][1], sentinel)
        self.assertIs(transitions[0].state_after_source, sentinel)
        self.assertIs(transitions[1].state_after_source, sentinel)
        self.assertEqual(
            [(transition.route.episode_start, transition.route.decision_id) for transition in transitions],
            [(True, "fixture-execution:0"), (False, "fixture-execution:1")],
        )
        self.assertEqual((result.status, result.outcome), ("terminal", "victory"))

    def test_missing_resources_remain_absent_and_explicit_null_is_preserved(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [
                (
                    observation(mask=(0, 0, 0), kinds=(0, 0, 0)),
                    1.0,
                    True,
                    False,
                    {
                        "outcome": 1,
                        "hp_after_cleanup": None,
                        "final_resources": {"energy": 2, "invalid": True},
                        "private": {"hp": 99},
                    },
                )
            ],
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual(dict(result.final_resources), {"energy": 2, "hp_after_cleanup": None})
        self.assertNotIn("hp_before_healing", result.final_resources)
        self.assertNotIn("private", result.final_resources)

    def test_case_profile_validation_precedes_factory(self):
        run_case = _run_case(self)
        calls = []
        invalid_case = case_execution(split="debug")

        result = run_case(
            invalid_case, ScriptedAgent([Choice(0)]),
            env_factory=lambda case: calls.append(case),
            budget=Budget(10.0, FakeClock()), max_decisions=4,
            contract=fixture_contract(),
        )

        self.assertEqual(result.status, "setup_error")
        self.assertFalse(result.started)
        self.assertEqual(calls, [])
        self.assertEqual(result.primary_failure.phase, "preflight")

    def test_budget_before_factory_is_not_started(self):
        run_case = _run_case(self)
        clock = FakeClock()
        budget = Budget(1.0, clock)
        clock.advance(2.0)
        calls = []

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: calls.append(case), budget=budget,
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.started, result.stop_reason), ("not_started", False, "wall_time_budget"))
        self.assertEqual(calls, [])

    def test_deadline_before_reset_is_not_started_and_closes_factory_result(self):
        run_case = _run_case(self)
        clock = FakeClock()
        env = ScriptedEnv((observation(), {}), [])

        def create_env(case):
            clock.advance(2.0)
            return env

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=create_env, budget=Budget(1.0, clock),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.started, result.stop_reason), ("not_started", False, "wall_time_budget"))
        self.assertEqual(env.reset_calls, [])
        self.assertEqual(env.close_calls, 1)

    def test_deadline_after_ongoing_step_stops_before_next_decision(self):
        run_case = _run_case(self)
        clock = FakeClock()
        env = ScriptedEnv(
            (observation(), {}),
            [(observation(), 0.0, False, False, {}), (observation(), 0.0, False, False, {})],
            on_step=lambda: clock.advance(2.0),
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(0)]),
            env_factory=lambda case: env, budget=Budget(1.0, clock),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps, result.stop_reason), ("budget_stopped", 1, "wall_time_budget"))
        self.assertEqual(len(env.step_calls), 1)

    def test_deadline_expiring_during_policy_prevents_step(self):
        run_case = _run_case(self)
        clock = FakeClock()
        env = ScriptedEnv(
            (observation(), {}),
            [(observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1.0, True, False, {"outcome": 1})],
        )

        class SlowPolicy:
            def init_state(self):
                return None

            def act(self, public, state):
                clock.advance(2.0)
                return Choice(0)

        result = run_case(
            case_execution(), SlowPolicy(), env_factory=lambda case: env,
            budget=Budget(1.0, clock), max_decisions=4,
            contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps, result.stop_reason), ("budget_stopped", 0, "wall_time_budget"))
        self.assertEqual(env.step_calls, [])
        self.assertEqual(env.close_calls, 1)

    def test_reset_return_marks_started_before_snapshot_validation(self):
        run_case = _run_case(self)
        env = ScriptedEnv((object(), {}), [])

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.started), ("contract_error", True))
        self.assertEqual(result.primary_failure.phase, "reset")
        self.assertEqual(env.close_calls, 1)

    def test_malformed_delivered_step_keeps_step_and_reward_without_retry(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [(object(), 0.5, False, False, {})],
        )
        transitions = []

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, on_transition=transitions.append, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.steps, result.reward_sum), ("contract_error", 1, 0.5))
        self.assertEqual(len(env.step_calls), 1)
        self.assertEqual(transitions, [])

    def test_invalid_delivered_reward_is_contract_error(self):
        run_case = _run_case(self)
        for reward in (True, math.nan, 10**400):
            with self.subTest(reward=type(reward).__name__):
                env = ScriptedEnv((observation(), {}), [(observation(), reward, False, False, {})])
                result = run_case(
                    case_execution(), ScriptedAgent([Choice(0)]),
                    env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
                    max_decisions=4, contract=fixture_contract(),
                )

                self.assertEqual((result.status, result.steps), ("contract_error", 1))
                self.assertEqual(result.reward_sum, 0.0)
                self.assertEqual(len(env.step_calls), 1)

    def test_reward_sum_overflow_keeps_last_finite_diagnostic(self):
        run_case = _run_case(self)
        env = ScriptedEnv(
            (observation(), {}),
            [
                (observation(), 1e308, False, False, {}),
                (observation(mask=(0, 0, 0), kinds=(0, 0, 0)), 1e308, True, False, {"outcome": 1}),
                (observation(), 0.0, False, False, {}),
            ],
        )

        result = run_case(
            case_execution(), ScriptedAgent([Choice(0), Choice(0), Choice(0)]),
            env_factory=lambda case: env, budget=Budget(10.0, FakeClock()),
            max_decisions=4, contract=fixture_contract(),
        )

        self.assertEqual((result.status, result.outcome, result.steps), ("contract_error", "victory", 2))
        self.assertEqual(result.reward_sum, 1e308)
        self.assertTrue(math.isfinite(result.reward_sum))
        self.assertEqual(len(env.step_calls), 2)


if __name__ == "__main__":
    unittest.main()
