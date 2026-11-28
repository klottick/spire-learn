"""Distillation gate logic on tiny synthetic outcomes."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neural import distill_gates as dg


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


    def test_p_is_two_sided(self) -> None:
        c, b = pair(1, 5)
        p = dg.pair_stats(c, b)["p"]
        self.assertAlmostEqual(p, 2 * 7 / 64)
        self.assertAlmostEqual(dg.pair_stats(b, c)["p"], p)

    def test_pairing_by_id_not_order(self) -> None:
        c, b = pair(3, 1, both=2)
        rev = dict(reversed(list(b.items())))
        self.assertEqual(dg.pair_stats(c, b), dg.pair_stats(c, rev))

    def test_subset_case_set_refused(self) -> None:
        c, b = pair(3, 1)
        sub = dict(list(b.items())[:-1])
        with self.assertRaises(dg.GateInputError):
            dg.pair_stats(c, sub)
        with self.assertRaises(dg.GateInputError):
            dg.pair_stats(sub, c)

    def test_unknown_outcome_refused(self) -> None:
        with self.assertRaises(dg.GateInputError):
            dg.pair_stats({"a": "win"}, {"a": "fault"})
        with self.assertRaises(dg.GateInputError):
            dg.evaluate({"ppo": [{"a": "wn"}, {"a": "win"}]})

    def test_seed_i_pairs_with_seed_i(self) -> None:
        ids = [f"x{i}" for i in range(5)]
        win = {i: "win" for i in ids}
        loss = {i: "loss" for i in ids}
        r = dg.beats([win, loss], [loss, win], 0.05)
        self.assertEqual([x["only_challenger"] for x in r["pairs"]], [5, 0])
        self.assertEqual([x["only_baseline"] for x in r["pairs"]], [0, 5])
        r2 = dg.beats([win, loss], [win, loss], 0.05)
        self.assertEqual([x["both_win"] for x in r2["pairs"]], [5, 0])

    def test_g3_s3_vs_s1_in_isolation_and_g4_end_to_end(self) -> None:
        strong, weak = pair(30, 2), pair(2, 2)
        arms = {"ppo": [strong[1]] * 2, "s1": [strong[1]] * 2,
                "s2": [strong[1]] * 2, "s3": [strong[0]] * 2,
                "distilled": [strong[0]] * 2}
        out = dg.evaluate(arms)
        g3 = out["G3"]
        self.assertTrue(g3["pass"])
        self.assertTrue(g3["tests"]["s3_vs_s1"]["beats"])
        self.assertFalse(g3["tests"]["s2_vs_s1"]["beats"])
        g4 = out["G4"]
        self.assertEqual(g4["status"], "evaluated")
        self.assertEqual(g4["alpha_per_test"], 0.05)
        self.assertTrue(g4["pass"])
        self.assertTrue(g4["tests"]["distilled_vs_ppo"]["beats"])

    def test_arm_guards(self) -> None:
        c, b = pair(3, 1)
        with self.assertRaises(dg.GateInputError):
            dg.evaluate({"ppo": [c]})                 # one seed
        with self.assertRaises(dg.GateInputError):
            dg.evaluate({"ppo": [c, c, c]})           # three seeds
        with self.assertRaises(dg.GateInputError):
            dg.evaluate({"nope": [c, c]})             # unknown arm
        with self.assertRaises(dg.GateInputError):    # different panels
            dg.evaluate({"ppo": [c, c], "s1": [b, {"z": "win"}]})
        with self.assertRaises(dg.GateInputError):    # requested gate lacks arms
            dg.evaluate({"ppo": [c, c], "s1": [b, b]}, ["G2"])
        out = dg.evaluate({"ppo": [c, c], "s1": [c, c]})
        self.assertEqual(out["G2"]["status"], "skipped")

    def _write(self, d: Path, name: str, outcomes: dict[str, str]) -> str:
        path = d / f"{name}.json"
        cases = {k: {"outcome": v} for k, v in outcomes.items()}
        path.write_text(json.dumps({"evaluations": {"final": {"cases": cases}}}))
        return str(path)

    def test_load_arm_and_main(self) -> None:
        c, b = pair(30, 2)
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ppo = ",".join(self._write(d, f"p{i}", b) for i in (1, 2))
            s1 = ",".join(self._write(d, f"s{i}", c) for i in (1, 2))
            self.assertEqual(dg.load_arm(ppo.split(",")), [b, b])
            out = d / "g.json"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(dg.main(["--arm", f"ppo={ppo}", "--arm", f"s1={s1}",
                                          "--gate", "G2", "--out", str(out)]), 2)
                with self.assertRaises(SystemExit):
                    dg.main(["--arm", f"ppo={ppo}", "--arm", f"ppo={ppo}"])
            self.assertFalse(out.exists())
            s2 = ",".join(self._write(d, f"t{i}", c) for i in (1, 2))
            s3 = ",".join(self._write(d, f"u{i}", c) for i in (1, 2))
            self.assertEqual(dg.main(["--arm", f"ppo={ppo}", "--arm", f"s1={s1}",
                                      "--arm", f"s2={s2}", "--arm", f"s3={s3}",
                                      "--gate", "G2", "--out", str(out)]), 0)
            g2 = json.loads(out.read_text())["G2"]
            self.assertTrue(g2["pass"])


if __name__ == "__main__":
    unittest.main()
