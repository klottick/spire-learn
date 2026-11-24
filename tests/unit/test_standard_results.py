"""Standard-baseline results aggregation on tiny synthetic data."""

from __future__ import annotations

import unittest

from scripts.neural import standard_results as sr


def _summary(final: dict, fresh: dict, manifest: str = "a" * 64) -> dict:
    wrap = lambda m: {"cases": {c: {"outcome": o} for c, o in m.items()},
                      "summary": {"panel_manifest_sha256": manifest}}
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

    def test_render_reports_censor_reasons_and_keeps_legacy_totals(self) -> None:
        legacy = _summary({"a": "incomplete"}, {})
        self.assertIn("| final | 0 | 0 | 1 | 0 | 0 | 1 |", sr.render(legacy, legacy, [], []))
        self.assertNotIn("Censored cases by reason", sr.render(legacy, legacy, [], []))
        capped = _summary({"a": "incomplete", "b": "incomplete", "c": "win"}, {})
        cases = capped["evaluations"]["final"]["cases"]
        cases["a"]["censor_reason"] = "turn_action_budget"
        cases["b"]["censor_reason"] = "environment_action_budget"
        cases["c"]["censor_reason"] = "turn_action_budget"
        text = sr.render(capped, legacy, [], [])
        self.assertIn("Censored cases by reason", text)
        self.assertIn("| final | 1 | 1 | 0 | 0 | 0 | 1 |", text)

    def test_render_has_no_case_or_run_ids(self) -> None:
        panel = [{"case_id": "zz-secret", "act": 2, "room_kind": "monster", "level": 0,
                  "run_id": "424242", "floor": 3, "search": {"outcome": "victory"}}]
        states = [{"character": "ironclad", "run_id": "424242", "floor": 3, "won": True}]
        s = _summary({"zz-secret": "win"}, {"zz-secret": "loss"})
        text = sr.render(s, s, panel, states)
        self.assertNotIn("zz-secret", text)
        self.assertNotIn("424242", text)
        self.assertIn("| 2 monster | 1 | 1 | 1 | 1 |", text)


class SearchAndRootsTest(unittest.TestCase):
    def test_search_only_tables_join_his_level_zero_and_hide_ids(self) -> None:
        panel = [{"case_id": "zz-secret", "act": 2, "room_kind": "monster", "level": 0, "run_id": "42", "floor": 3,
                  "search": {"outcome": "victory"}},
                 {"case_id": "yy", "act": 2, "room_kind": "monster", "level": 1, "run_id": "42", "floor": 3,
                  "search": {"outcome": "death"}}]
        states = [{"character": "ironclad", "run_id": "42", "floor": 3, "won": False}]
        text = sr.render_search(panel, states)
        self.assertIn("Search wins 1 of 2.", text)
        self.assertIn("| 2 monster | 2 | 1 | 1 | 0 |", text)
        self.assertNotIn("zz-secret", text)
        self.assertNotIn("42", text)

    def test_root_availability_tables_split_by_class_kind_act_and_dome(self) -> None:
        data = {"characters": {"ironclad": {
            "fights": 3, "corpus_states": 9, "decisions": 8, "dome_fights": 1, "fight_outcomes": {},
            "cells": {"1:monster:no_dome": {"root": 3, "draw knowledge exceeds the exact model": 1},
                      "2:elite:dome": {"root": 1, "hidden intents are not modelled": 3}}}}}
        text = sr.render_root_availability(data)
        self.assertIn("Runic Dome in 1 fights (33.3%)", text)
        self.assertIn("| 1 monster | 4 | 75.0% | 0.0% | 25.0% |", text)
        dome = text.split("ironclad, Runic Dome holders:")[1].split("ironclad, no Runic Dome")[0]
        self.assertIn("| 2 elite | 4 | 25.0% | 75.0% |", dome)
        self.assertNotIn("1 monster", dome)


class InputCheckTest(unittest.TestCase):
    PANEL = [{"case_id": "x", "act": 1, "room_kind": "elite", "level": 0, "run_id": "9", "floor": 7,
              "search": {"outcome": "victory"}}]
    RECIPE = {"states_table": {"path": "t.json", "sha256": "b" * 64}}

    def check(self, r1=None, r2=None, search=None, panel=None, table="b" * 64):
        r1 = r1 or _summary({"x": "win"}, {"x": "loss"})
        r2 = r2 or _summary({"x": "win"}, {"x": "loss"})
        search = {"panel_manifest_sha256": "a" * 64} if search is None else search
        return sr.check_inputs(r1, r2, search, self.PANEL if panel is None else panel, table, self.RECIPE)

    def test_agreeing_inputs_pass(self) -> None:
        self.assertEqual(self.check(), "a" * 64)

    def test_panel_manifest_must_agree_everywhere(self) -> None:
        stale = _summary({"x": "win"}, {"x": "loss"})
        stale["evaluations"]["near-405504"]["summary"]["panel_manifest_sha256"] = "c" * 64
        for kwargs in ({"r1": stale}, {"r2": stale}, {"search": {"panel_manifest_sha256": "c" * 64}},
                       {"search": {}}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaisesRegex(sr.InputMismatch, "panel manifests"):
                self.check(**kwargs)
        missing = {"panel_manifest_sha256": None}
        none = _summary({"x": "win"}, {"x": "loss"}, manifest=None)
        with self.assertRaisesRegex(sr.InputMismatch, "do not record"):
            self.check(r1=none, r2=none, search=missing)

    def test_states_table_must_match_the_recipe_pin(self) -> None:
        with self.assertRaisesRegex(sr.InputMismatch, "recipe pin"):
            self.check(table="d" * 64)

    def test_unexpected_room_kind_is_rejected_and_render_does_not_crash(self) -> None:
        panel = [dict(self.PANEL[0], room_kind="event")]
        with self.assertRaisesRegex(sr.InputMismatch, "event"):
            self.check(panel=panel)
        s = _summary({"x": "win"}, {"x": "loss"})
        states = [{"character": "ironclad", "run_id": "9", "floor": 7, "won": True}]
        self.assertIn("1 event", sr.render(s, s, panel, states))


if __name__ == "__main__":
    unittest.main()
