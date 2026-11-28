import unittest

from scripts.neural import distill_tau_entry as tau


class DistillTauTests(unittest.TestCase):
    def test_top_two_distinct_gap(self):
        self.assertAlmostEqual(tau.top_two_gap([0.5, 0.2, 0.5, 0.1]), 0.3)
        self.assertIsNone(tau.top_two_gap([0.4, 0.4, 0.4]))
        self.assertAlmostEqual(tau.top_two_gap([-1.0, 0.25]), 1.25)

    def test_linear_percentiles_and_floor(self):
        gaps = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]  # zero excluded; 6 positive gaps
        out = tau.tau_candidates(gaps)
        # positive = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5]; ranks 1.25, 2.5, 3.75
        self.assertAlmostEqual(out["p25"], 0.1 + 0.25 * 0.1)
        self.assertAlmostEqual(out["p50"], 0.25)
        self.assertAlmostEqual(out["p75"], 0.3 + 0.75 * 0.1)
        self.assertEqual(tau.tau_candidates([0.001, 0.002, 0.003])["p50"], 0.01)

    def test_refuses_no_positive_gaps(self):
        with self.assertRaises(ValueError):
            tau.tau_candidates([0.0, 0.0])

    def test_gradient_steps(self):
        self.assertEqual(tau.gradient_steps(1_520_000), round(6 * 1_520_000 / 512))


if __name__ == "__main__":
    unittest.main()
