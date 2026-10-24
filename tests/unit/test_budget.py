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

    def test_unrepresentable_integer_duration_uses_value_error(self):
        for value in (10**10000, -(10**10000)):
            with self.subTest(sign='positive' if value > 0 else 'negative'):
                with self.assertRaises(ValueError):
                    Budget(value, FakeClock())

    def test_rejects_invalid_clock_and_clock_values(self):
        with self.assertRaises(TypeError):
            Budget(1.0, None)
        for value in (True, "1", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Budget(1.0, lambda value=value: value)

    def test_unrepresentable_initial_and_later_clock_readings_use_value_error(self):
        with self.assertRaises(ValueError):
            Budget(1.0, lambda: 10**10000)

        readings = iter((0, -(10**10000)))
        budget = Budget(1.0, lambda: next(readings))
        with self.assertRaises(ValueError):
            budget.remaining()

    def test_clock_callable_exception_propagates_unchanged(self):
        error = RuntimeError('clock failed')

        def failing_clock():
            raise error

        with self.assertRaises(RuntimeError) as raised:
            Budget(1.0, failing_clock)
        self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
