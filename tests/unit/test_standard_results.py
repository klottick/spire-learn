"""Standard-baseline results aggregation on tiny synthetic data."""

from __future__ import annotations

import unittest

from scripts.neural import standard_results as sr


def _summary(final: dict, fresh: dict) -> dict:
    wrap = lambda m: {"cases": {c: {"outcome": o} for c, o in m.items()}}
    evaluations = {label: wrap(final) for label in sr.LABELS}
    evaluations["fresh-initial"] = wrap(fresh)
    return {"evaluations": evaluations}


class StandardResultsTest(unittest.TestCase):
    def test_mcnemar_exact(self) -> None:
        self.assertEqual(sr.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(sr.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(sr.mcnemar_exact(0, 4), 2 / 16)
        self.assertAlmostEqual(sr.mcnemar_exact(1, 5), 2 * 7 / 64)

    def test_paired_excludes_escaped_and_censored(self) -> None:
        a = {"a": "win", "b": "win", "c": "loss", "d": "escaped", "e": "win", "f": "loss"}
        b = {"a": "win", "b": "loss", "c": "win", "d": "win", "e": "incomplete", "f": "loss"}
        p = sr.paired(a, b)
        self.assertEqual((p["both_win"], p["only_a"], p["only_b"], p["neither"], p["excluded"]),
                         (1, 1, 1, 1, 2))

    def test_stratify_joins_recorded_outcome_at_level_zero_only(self) -> None:
        panel = [
            {"case_id": "x", "act": 1, "room_kind": "elite", "level": 0, "run_id": "9",
             "floor": 7, "search": {"outcome": "victory"}},
            {"case_id": "y", "act": 1, "room_kind": "elite", "level": 1, "run_id": "9",
             "floor": 7, "search": {"outcome": "death"}},
        ]
        states = [{"character": "ironclad", "run_id": 9, "floor": 7, "won": False},
                  {"character": "silent", "run_id": 9, "floor": 7, "won": True}]
        runs = {"r1": {"x": "win", "y": "loss"}}
        (row,) = sr.stratify(panel, runs, sr.recorded_wins(states),
                             lambda c: (c["act"], c["room_kind"]))
        self.assertEqual((row["n"], row["r1"], row["search"], row["his_n"], row["his"]),
                         (2, 1, 1, 1, 0))

    def test_render_has_no_case_or_run_ids(self) -> None:
        panel = [{"case_id": "zz-secret", "act": 2, "room_kind": "monster", "level": 0,
                  "run_id": "424242", "floor": 3, "search": {"outcome": "victory"}}]
        states = [{"character": "ironclad", "run_id": "424242", "floor": 3, "won": True}]
        s = _summary({"zz-secret": "win"}, {"zz-secret": "loss"})
        text = sr.render(s, s, panel, states)
        self.assertNotIn("zz-secret", text)
        self.assertNotIn("424242", text)
        self.assertIn("| 2 monster | 1 | 1 | 1 | 1 |", text)


if __name__ == "__main__":
    unittest.main()
