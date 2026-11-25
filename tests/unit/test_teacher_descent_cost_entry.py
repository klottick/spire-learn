"""Host-only tests of the teacher descent measurement helpers."""

import random
import unittest
from unittest.mock import patch

from scripts.neural import teacher_descent_cost_entry as entry


class FakeView:
    def heuristic_choice(self, observation, rng):
        return rng.choice(observation["legal"])


class FakeBranch:
    def __init__(self, *, stop_at=None, prefix=()):
        self.actions = list(prefix)
        self.stop_at = stop_at
        self.closed = False

    def step(self, action, *, view):
        assert view == "full"
        self.actions.append(action)
        count = len(self.actions)
        return {"legal": [count]}, 0.0, count == self.stop_at, False, {}

    def close(self):
        self.closed = True


class TeacherDescentCostTests(unittest.TestCase):
    def test_prefix_uses_successive_public_observations_and_can_continue(self):
        branch = FakeBranch()
        prefix = []
        args = (branch, FakeView(), random.Random(7), prefix)
        observation, reason = entry.extend_prefix(*args, {"legal": [7]}, 2)
        self.assertIsNone(reason)
        self.assertEqual(prefix, [7, 1])
        observation, reason = entry.extend_prefix(*args, observation, 4)
        self.assertIsNone(reason)
        self.assertEqual(prefix, [7, 1, 2, 3])
        self.assertEqual(branch.actions, prefix)
        self.assertEqual(observation, {"legal": [4]})

    def test_terminal_prefix_stops_before_target_depth(self):
        branch = FakeBranch(stop_at=2)
        prefix = []
        observation, reason = entry.extend_prefix(
            branch, FakeView(), random.Random(7), prefix, {"legal": [7]}, 4,
        )
        self.assertIsNone(observation)
        self.assertEqual(reason, "prefix_terminated")
        self.assertEqual(prefix, [7, 1])

    def test_fork_and_replay_take_the_same_step_at_the_node(self):
        prefix = [7, 1, 2, 3]
        fork, replay = FakeBranch(prefix=prefix), FakeBranch()
        with (
            patch.object(entry, "process_time", side_effect=[1.0, 2.0, 3.0, 5.0]),
            patch.object(entry, "perf_counter", side_effect=[0.0, 2.0, 3.0, 6.0]),
        ):
            fork_result = entry.timed_descent(
                lambda: fork, (), FakeView(), random.Random(7), {"legal": [4]},
            )
            replay_result = entry.timed_descent(lambda: replay, prefix, FakeView(), random.Random(7))
        self.assertEqual(fork_result[0], (1.0, 2.0))
        self.assertEqual(replay_result[0], (2.0, 3.0))
        self.assertEqual(replay.actions, prefix + [4])
        self.assertEqual(fork.actions, replay.actions)
        self.assertEqual(fork_result[1:], replay_result[1:])
        self.assertTrue(fork.closed and replay.closed)

    def test_timed_descent_propagates_step_fault_and_closes_branch(self):
        branch = FakeBranch()
        fault = RuntimeError("fake simulator fault")
        with patch.object(branch, "step", side_effect=fault) as step:
            with self.assertRaises(RuntimeError) as raised:
                entry.timed_descent(lambda: branch, (), FakeView(), random.Random(7), {"legal": [4]})
        self.assertIs(raised.exception, fault)
        step.assert_called_once_with(4, view="full")
        self.assertTrue(branch.closed)

    def test_summary_keeps_clock_samples_separate(self):
        self.assertEqual(entry.timing_summary([(1, 8), (2, 4), (9, 6)]), {
            "samples": 3, "mean_cpu_seconds": 4, "median_cpu_seconds": 2,
            "mean_wall_seconds": 6, "median_wall_seconds": 6,
        })

    def test_empty_summary_uses_null_timings(self):
        self.assertEqual(entry.timing_summary([]), {
            "samples": 0, "mean_cpu_seconds": None, "median_cpu_seconds": None,
            "mean_wall_seconds": None, "median_wall_seconds": None,
        })
