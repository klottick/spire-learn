import unittest

from tests.support import FakeArray, FakeClock, ScriptedAgent, ScriptedEnv, case_execution, case_result, public_decision
from spire_learn.contracts import Choice


class SupportTests(unittest.TestCase):
    def test_scripted_env_logs_calls_before_errors_and_only_ticks_delivered_results(self):
        ticks = []
        boom = RuntimeError("scripted")
        env = ScriptedEnv((("obs", {}), {}), [boom], on_step=lambda: ticks.append(True))
        self.assertEqual(env.reset(seed=4, options={"x": 1}), (("obs", {}), {}))
        with self.assertRaisesRegex(RuntimeError, "scripted"):
            env.step(2)
        self.assertEqual(env.reset_calls, [(4, {"x": 1})])
        self.assertEqual(env.step_calls, [2])
        self.assertEqual(ticks, [])
        env.close()
        self.assertEqual(env.close_calls, 1)

    def test_helpers_produce_declared_shapes_and_fixture_identity(self):
        arr = FakeArray("int8", (2,), (1, 0))
        self.assertEqual((arr.dtype.name, arr.shape, arr.tolist()), ("int8", (2,), [1, 0]))
        clock = FakeClock(2)
        clock.advance(3)
        self.assertEqual(clock(), 5)
        agent = ScriptedAgent([Choice(1)])
        self.assertEqual(agent.act(public_decision(), None), Choice(1))
        self.assertEqual(case_execution().expected_schema_id, "spire-learn-fixture-v1")
        self.assertEqual(case_result().outcome, "victory")
