import unittest

from tests.support import FakeClock
from spire_learn.experiments.budget import Budget


class BudgetTests(unittest.TestCase):
    def test_remaining_and_expiry(self):
        clock = FakeClock(10)
        budget = Budget(3.0, clock)
        self.assertEqual(budget.remaining(), 3.0)
        self.assertFalse(budget.exhausted())
        clock.advance(2.0)
        self.assertEqual(budget.remaining(), 1.0)
        clock.advance(1.0)
        self.assertTrue(budget.exhausted())
        self.assertEqual(budget.remaining(), 0.0)
        clock.advance(1.0)
        self.assertEqual(budget.remaining(), 0.0)

    def test_clock_reversal_clamps_elapsed(self):
        clock = FakeClock(5)
        budget = Budget(2.0, clock)
        clock.now = 4
        self.assertEqual(budget.remaining(), 2.0)

    def test_instances_capture_independent_start_times(self):
        clock = FakeClock(0)
        first = Budget(10.0, clock)
        clock.advance(4)
        second = Budget(10.0, clock)
        clock.advance(2)
        self.assertEqual(first.remaining(), 4.0)
        self.assertEqual(second.remaining(), 8.0)

    def test_rejects_invalid_durations(self):
        for value in (True, 0, -1, float("nan"), float("inf"), -float("inf"), "2"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Budget(value, FakeClock())

    def test_rejects_invalid_clock_and_clock_values(self):
        with self.assertRaises(TypeError):
            Budget(1.0, None)
        for value in (True, "1", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Budget(1.0, lambda value=value: value)


if __name__ == "__main__":
    unittest.main()
