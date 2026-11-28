"""Distillation gate logic on tiny synthetic outcomes."""

from __future__ import annotations

import unittest

from scripts.neural import distill_gates as dg


def outcomes(wins: int, losses: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    m = {f"c{i}": "win" for i in range(wins)}
    m.update({f"c{wins + i}": "loss" for i in range(losses)})
    m.update(extra or {})
    return m


def pair(only_c: int, only_b: int, both: int = 0) -> tuple[dict, dict]:
    c, b = {}, {}
    for i in range(only_c):
        c[f"a{i}"], b[f"a{i}"] = "win", "loss"
    for i in range(only_b):
        c[f"b{i}"], b[f"b{i}"] = "loss", "win"
    for i in range(both):
        c[f"w{i}"] = b[f"w{i}"] = "win"
    return c, b


class DistillGatesTest(unittest.TestCase):
    def test_beats_needs_both_pairs_on_paired_wins_and_pooled_p(self) -> None:
        strong, weak = pair(30, 2), pair(25, 1)
        r = dg.beats([strong[0], weak[0]], [strong[1], weak[1]], 0.05)
        self.assertTrue(r["beats"])
        # one pair favours the baseline: pooled p tiny but not a beat
        loser = pair(1, 2)
        r = dg.beats([strong[0], loser[0]], [strong[1], loser[1]], 0.05)
        self.assertTrue(r["significant"])
        self.assertFalse(r["each_pair_favours_challenger"])
        self.assertFalse(r["beats"])
        # both favour but pooled is not significant
        s1, s2 = pair(3, 1), pair(3, 2)
        r = dg.beats([s1[0], s2[0]], [s1[1], s2[1]], 0.05)
        self.assertTrue(r["each_pair_favours_challenger"])
        self.assertFalse(r["significant"])
        self.assertFalse(r["beats"])
        self.assertEqual(r["pooled"]["only_challenger"], 6)
        self.assertEqual(r["pooled"]["only_baseline"], 3)

    def test_drop_censored_escaped_per_pair_and_report_counts(self) -> None:
        c, b = pair(2, 1, both=1)
        c.update({"e1": "escaped", "e2": "win", "i1": "incomplete"})
        b.update({"e1": "win", "e2": "incomplete", "i1": "escaped"})
        p = dg.pair_stats(c, b)
        self.assertEqual(p["paired"], 4)
        self.assertEqual(p["dropped_cases"], 3)
        self.assertEqual(p["dropped_escaped_sides"], 2)
        self.assertEqual(p["dropped_censored_sides"], 2)
        self.assertEqual((p["only_challenger"], p["only_baseline"]), (2, 1))
        # drops are per pair: a second pair keeps its own counts
        r = dg.beats([c, pair(1, 0)[0]], [b, pair(1, 0)[1]], 0.05)
        self.assertEqual([x["dropped_cases"] for x in r["pairs"]], [3, 0])
        self.assertEqual(r["dropped_cases"], 3)
        with self.assertRaises(dg.GateInputError):
            dg.pair_stats({"a": "win"}, {"b": "win"})

    def test_alpha_per_gate(self) -> None:
        self.assertAlmostEqual(dg.GATES["G2"][0], 0.05 / 3)
        self.assertEqual(dg.GATES["G3"][0], 0.025)
        self.assertEqual(dg.GATES["G4"][0], 0.05)
        self.assertEqual(len(dg.GATES["G2"][1]), 3)
        self.assertEqual(len(dg.GATES["G3"][1]), 2)
        self.assertEqual(len(dg.GATES["G4"][1]), 1)
        # pooled 9 vs 1: p = 0.0215, inside (0.0167, 0.025)
        a, b = pair(5, 1), pair(4, 0)
        chal, base = [a[0], b[0]], [a[1], b[1]]
        self.assertTrue(dg.beats(chal, base, dg.GATES["G3"][0])["beats"])
        self.assertFalse(dg.beats(chal, base, dg.GATES["G2"][0])["beats"])
        self.assertTrue(dg.beats(chal, base, dg.GATES["G4"][0])["beats"])

    def test_g2_any_of_three(self) -> None:
        strong = pair(30, 2)
        out = dg.evaluate({"ppo": [strong[1]] * 2, "s1": [strong[1]] * 2,
                           "s2": [strong[0]] * 2, "s3": [strong[1]] * 2})
        g2 = out["G2"]
        self.assertTrue(g2["pass"])
        self.assertTrue(g2["tests"]["s2_vs_ppo"]["beats"])
        self.assertFalse(g2["tests"]["s1_vs_ppo"]["beats"])
        self.assertFalse(g2["tests"]["s3_vs_ppo"]["beats"])
        self.assertEqual(out["G3"]["status"], "evaluated")
        self.assertTrue(out["G3"]["pass"])
        self.assertEqual(out["G4"]["status"], "skipped")
        none = dg.evaluate({k: [strong[1]] * 2 for k in ("ppo", "s1", "s2", "s3")})
        self.assertFalse(none["G2"]["pass"])


if __name__ == "__main__":
    unittest.main()
