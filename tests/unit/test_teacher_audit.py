"""Teacher audit and gate G1 on synthetic run outputs (no Gym runtime)."""

from collections import Counter
from contextlib import contextmanager, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.neural import teacher_audit as audit
from scripts.neural.standard_results import mcnemar_exact

STATES_SHA = "states-table-sha"


def panel_doc(kind, cases):
    """cases: (case_id, act, room_kind, level, max_hp) tuples."""
    return {
        "panel_sha256": f"{kind}-panel-sha", "manifest_sha256": f"{kind}-manifest-sha",
        "panel_identity": {"states_table": {"sha256": STATES_SHA}},
        "manifest": {"split": "selection", "cases": [
            {"case_id": cid, "entry": {"id": cid, "max_hp": hp}, "environment_seed": index}
            for index, (cid, _, _, _, hp) in enumerate(cases)]},
        "annotations": {cid: {"act": act, "room_kind": kind_, "level": level,
                              "run_id": f"run-{index}", "floor": 3}
                        for index, (cid, act, kind_, level, _) in enumerate(cases)},
    }


def decision(case_id, step, root, choice=0, cpu=0.5):
    values = {"0": [0.5, 8], "1": [0.25, 8]} if root == "searched" else None
    return {"case_id": case_id, "step": step, "action_prefix_length": step, "root": root,
            "values": values, "choice": choice, "cpu_seconds": cpu if root == "searched" else None,
            "research_seed": 1000 + step, "branch_refusals": 0,
            "simulations": 16 if root == "searched" else None}


def write_run(directory, teacher, panel_kind, panel, outcomes, roots=None, *, hp=10, cpu=0.5,
              provenance=None):
    """outcomes: case_id -> entry outcome; roots: case_id -> list of root classes."""
    directory.mkdir(parents=True)
    fights, rows = [], []
    for cid, outcome in outcomes.items():
        case_roots = (roots or {}).get(cid, ["searched", "single_legal"])
        rows.extend(decision(cid, step, root, cpu=cpu) for step, root in enumerate(case_roots))
        fights.append({"case_id": cid, "outcome": outcome,
                       "censor_reason": "turn_cap" if outcome == "censored" else None,
                       "steps": len(case_roots), "cpu_seconds": cpu * len(case_roots) + 0.1,
                       "wall_seconds": 1.0,
                       "hp_after_cleanup": None if outcome == "censored" else (0 if outcome == "death" else hp)})
    (directory / "fights.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fights))
    (directory / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    searched = [r for r in rows if r["root"] == "searched"]
    summary = {
        "schema": "spire-learn-search-teacher-entry-v1", "cases": len(fights),
        "wins": sum(f["outcome"] == "victory" for f in fights),
        "outcomes": dict(Counter(f["outcome"] for f in fights)),
        "roots": dict(Counter(r["root"] for r in rows)),
        "searched_decisions": len(searched),
        "cpu_seconds_per_searched_decision": cpu if searched else None,
        "provenance": {
            "teacher": teacher, "panel": panel_kind, "split": "selection", "calibrated": True,
            "config_sha256": f"{teacher}-config", "panel_sha256": panel["panel_sha256"],
            "manifest_sha256": panel["manifest_sha256"], "case_manifest_sha256": "cases",
            "gym_revision": "rev", "gym_build": {"revision": "rev", "dirty": False},
            "rules_epoch": "gym-1.2.4", "recipe_sha256": "recipe", "states_table_sha256": STATES_SHA,
            "observation_fingerprint": "fp", "max_steps": 1024, "max_turn_actions": 500, "seed": 7,
            **(provenance or {}),
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def write_ppo(path, panel, outcomes):
    path.write_text(json.dumps({"evaluations": {"final": {
        "cases": {cid: {"outcome": outcome} for cid, outcome in outcomes.items()},
        "summary": {"panel_sha256": panel["panel_sha256"],
                    "panel_manifest_sha256": panel["manifest_sha256"], "failures": [],
                    "total": {"operational_failures": 0, "mean_hp_retention": 0.25}},
    }}}))
    return path


class Fixture:
    """Three-teacher selection panel and a boss panel in a temporary artifacts tree."""

    def __init__(self, root, selection=None, boss=None, cpu=None):
        self.root = Path(root)
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        self.selection = panel_doc("selection", [
            (f"p-sel-{i:02d}", 1 + i % 3, ("monster", "elite", "boss")[i % 3], (0, 1, 2, 4)[i % 4], 80)
            for i in range(12)])
        self.boss = panel_doc("boss", [(f"p-boss-{i}", 1 + i % 3, ("elite", "boss")[i % 2], 0, 70)
                                       for i in range(4)])
        self.paths = {}
        for kind, doc in (("selection", self.selection), ("boss", self.boss)):
            path = artifacts / f"{kind}-panel.json"
            path.write_text(json.dumps(doc))
            self.paths[kind] = path
        ids = [c["case_id"] for c in self.selection["manifest"]["cases"]]
        selection = selection or {}
        cpu = cpu or {}
        self.runs = {}
        for teacher in audit.TEACHERS:
            outcomes = selection.get(teacher, {cid: "victory" for cid in ids})
            self.runs[teacher, "selection"] = write_run(
                artifacts / f"{teacher}-selection", teacher, "selection", self.selection, outcomes,
                cpu=cpu.get(teacher, 0.5))
        boss_ids = [c["case_id"] for c in self.boss["manifest"]["cases"]]
        for teacher in ("current", "tree-3x"):
            self.runs[teacher, "boss"] = write_run(
                artifacts / f"{teacher}-boss", teacher, "boss", self.boss,
                (boss or {}).get(teacher, {cid: "death" for cid in boss_ids}))
        self.ppo = write_ppo(artifacts / "ppo-boss.json", self.boss, {cid: "loss" for cid in boss_ids})

    def argv(self, output="artifacts/report.json"):
        argv = ["report", "--output", str(self.root / output),
                "--panel", f"selection:{self.paths['selection']}", "--panel", f"boss:{self.paths['boss']}",
                "--ppo", f"ppo-r1:boss:{self.ppo}"]
        for (teacher, panel), path in self.runs.items():
            argv += ["--run", f"{teacher}:{panel}:{path}"]
        return argv

    def report(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertion(run_main(self.root, self.argv()))
        return json.loads((self.root / "artifacts/report.json").read_text()), stdout.getvalue()

    @staticmethod
    def assertion(code):
        if code != 0:
            raise AssertionError(f"report exited {code}")


def run_main(root, argv):
    with patch.object(audit, "_ROOT", Path(root)), patch.dict("os.environ", {"SPIRE_LEARN_RUNNER_OUTPUT": ""}):
        return audit.main(argv)


def outcomes_from(pattern):
    """pattern: one character per selection case; W victory, L death, E escaped, C censored."""
    names = {"W": "victory", "L": "death", "E": "escaped", "C": "censored"}
    return {f"p-sel-{i:02d}": names[ch] for i, ch in enumerate(pattern)}


class GateTests(unittest.TestCase):
    def test_g1_uses_mcnemar_with_censored_and_escaped_dropped(self):
        current = outcomes_from("LLLLLWWWWCEW")
        tree = outcomes_from("WWWWWLWWCWWE")
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, selection={"current": current, "tree-1x": tree,
                                                    "tree-3x": current})
            report, _ = fixture.report()
        g1 = report["g1"]["tests"]["tree-1x"]
        # Cases 8-11 are censored or escaped on one side and leave the pair.
        self.assertEqual((g1["excluded"], g1["paired"]), (4, 8))
        self.assertEqual((g1["only_a"], g1["only_b"], g1["both_win"], g1["neither"]), (5, 1, 2, 0))
        self.assertEqual(g1["p"], mcnemar_exact(5, 1))
        self.assertEqual((g1["wins_challenger"], g1["wins_current"]), (7, 3))
        self.assertFalse(g1["beats"])
        same = report["g1"]["tests"]["tree-3x"]
        self.assertEqual((same["only_a"], same["only_b"], same["p"], same["beats"]), (0, 0, 1.0, False))
        self.assertEqual(report["g1"]["verdict"], "FAIL")
        self.assertIn("tree-3x", report["g1"]["boss_panel"])

    def test_beats_requires_more_wins_and_p_below_0_025(self):
        # More wins, p = 0.0386: not significant at the Bonferroni level.
        self.assertGreater(mcnemar_exact(10, 2), 0.025)
        self.assertFalse(audit.beats(10, 2))
        # Significant but on the wrong side.
        self.assertLess(mcnemar_exact(2, 12), 0.025)
        self.assertFalse(audit.beats(2, 12))
        self.assertTrue(audit.beats(12, 2))
        self.assertFalse(audit.beats(0, 0))

        with tempfile.TemporaryDirectory() as directory:
            current = {f"p-sel-{i:02d}": "death" for i in range(12)}
            tree = {f"p-sel-{i:02d}": "victory" for i in range(12)}
            fixture = Fixture(directory, selection={"current": current, "tree-1x": current,
                                                    "tree-3x": tree})
            report, _ = fixture.report()
        test = report["g1"]["tests"]["tree-3x"]
        self.assertEqual((test["only_a"], test["only_b"]), (12, 0))
        self.assertTrue(test["beats"])
        self.assertEqual(report["g1"]["verdict"], "PASS")

    def test_teacher_choice_ties_go_to_cheaper(self):
        strength = {
            "current": {"wins": 200, "cpu_seconds_per_searched_decision": 0.30},
            "tree-1x": {"wins": 210, "cpu_seconds_per_searched_decision": 0.29},
            "tree-3x": {"wins": 210, "cpu_seconds_per_searched_decision": 0.95},
            "tree-10x": {"wins": 240, "cpu_seconds_per_searched_decision": 3.1},
        }
        self.assertEqual(audit.choose_teacher(strength)["chosen"], "tree-1x")
        # A tree-1x / tree-3x tie goes to the 1x budget even when tree-3x measured cheaper.
        strength["tree-1x"]["cpu_seconds_per_searched_decision"] = 0.96
        self.assertEqual(audit.choose_teacher(strength)["chosen"], "tree-1x")
        # current and tree-1x share the 1x budget; measured CPU breaks that tie.
        strength["current"]["wins"] = 210
        self.assertEqual(audit.choose_teacher(strength)["chosen"], "current")
        strength["tree-3x"]["wins"] = 211
        self.assertEqual(audit.choose_teacher(strength)["chosen"], "tree-3x")
        # tree-10x is on the strength curve only.
        self.assertNotIn("tree-10x", audit.choose_teacher(strength)["candidates"])

        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory, cpu={"current": 0.4, "tree-1x": 0.3, "tree-3x": 0.9})
            report, _ = fixture.report()
        self.assertEqual(report["teacher_choice"]["chosen"], "tree-1x")

    def test_stop_rule_triggers_below_half_boss_roots(self):
        below = Counter({"searched": 49, "draw knowledge exceeds the exact model": 51, "single_legal": 400})
        at = Counter({"searched": 50, "a card read a hidden intent": 50, "single_legal": 1})
        self.assertEqual(audit.stop_rule(below)["verdict"], "STOP")
        self.assertEqual(audit.stop_rule(below)["boss_root_share"], 0.49)
        self.assertEqual(audit.stop_rule(at)["verdict"], "CONTINUE")
        with self.assertRaises(audit.AuditInputError):
            audit.stop_rule(Counter({"single_legal": 3}))

        # In a report, only the chosen teacher's own boss fights count, across both panels.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            artifacts = fixture.root / "artifacts"
            boss_cases = [cid for cid, a in fixture.selection["annotations"].items() if a["room_kind"] == "boss"]
            refused = {cid: ["searched", "a card read a hidden intent", "a card read a hidden intent"]
                       for cid in boss_cases}
            ids = list(fixture.selection["annotations"])
            for teacher in ("current",):
                path = artifacts / f"{teacher}-selection-refused"
                fixture.runs[teacher, "selection"] = write_run(
                    path, teacher, "selection", fixture.selection, {cid: "victory" for cid in ids}, refused,
                    cpu=0.1)
            report, _ = fixture.report()
        self.assertEqual(report["teacher_choice"]["chosen"], "current")
        rule = report["stop_rule"]
        self.assertEqual(rule["teacher"], "current")
        # Selection boss cases: 4 x (1 of 3); boss-panel boss cases: 2 x (1 of 1).
        self.assertEqual((rule["searched"], rule["decisions"]), (6, 14))
        self.assertEqual(rule["verdict"], "STOP")


class ReportTests(unittest.TestCase):
    def test_report_is_aggregate_only_and_records_strength_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            report, stdout = fixture.report()
            text = (fixture.root / "artifacts/report.json").read_text()
        for cid in [*fixture.selection["annotations"], *fixture.boss["annotations"]]:
            self.assertNotIn(cid, text)
            self.assertNotIn(cid, stdout)
        strength = report["strength"]["selection"]["current"]
        self.assertEqual((strength["cases"], strength["wins"], strength["decisions"],
                          strength["searched_decisions"]), (12, 12, 24, 12))
        self.assertEqual(strength["mean_hp_kept"], 10 / 80)
        self.assertEqual(strength["cpu_seconds_per_searched_decision"], 0.5)
        self.assertEqual(report["strength"]["boss"]["ppo-r1"]["wins"], 0)
        self.assertIn("tree-10x", report["strength"]["selection"])
        coverage = report["roots"]["selection"]["current"]["1:monster"]
        self.assertEqual(coverage["counts"], {"searched": 4, "single_legal": 4})
        self.assertEqual((coverage["root_share"], coverage["labelled_share"]), (1.0, 0.5))
        self.assertEqual(report["provenance"]["rules_epoch"], "gym-1.2.4")
        self.assertIn("G1", stdout)

    def test_report_rejects_incomplete_runs_and_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            (fixture.runs["tree-3x", "selection"] / "summary.json").unlink()
            with self.assertRaisesRegex(audit.AuditInputError, "no summary.json"):
                run_main(fixture.root, fixture.argv())
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            path = fixture.runs["tree-1x", "selection"]
            write_run(Path(directory) / "artifacts/other", "tree-1x", "selection", fixture.selection,
                      outcomes_from("W" * 12), provenance={"gym_revision": "other"})
            fixture.runs["tree-1x", "selection"] = Path(directory) / "artifacts/other"
            with self.assertRaisesRegex(audit.AuditInputError, "gym_revision"):
                run_main(fixture.root, fixture.argv())
            fixture.runs["tree-1x", "selection"] = path
            fights = (path / "fights.jsonl").read_text().replace('"victory"', '"fault"', 1)
            (path / "fights.jsonl").write_text(fights)
            with self.assertRaisesRegex(audit.AuditInputError, "outcome"):
                run_main(fixture.root, fixture.argv())

    def test_report_requires_selection_runs_of_every_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            del fixture.runs["tree-1x", "selection"]
            with self.assertRaisesRegex(audit.AuditInputError, "tree-1x"):
                run_main(fixture.root, fixture.argv())

    def test_run_spec_parses_teacher_panel_and_path(self):
        self.assertEqual(audit._run_spec("tree-3x:boss:/a:b/c"), ("tree-3x", "boss", Path("/a:b/c")))
        for bad in ("tree-2x:boss:/x", "tree-3x:final:/x", "tree-3x:/x"):
            with self.assertRaises(Exception):
                audit._run_spec(bad)

    def test_report_records_baalorlord_level0_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            states = {"states": [
                {"run_id": f"run-{i}", "floor": 3, "character": "ironclad", "won": i % 2 == 0}
                for i in range(12)]}
            data = json.dumps(states).encode()
            path = fixture.root / "artifacts/run-states.json"
            path.write_bytes(data)
            for doc, kind in ((fixture.selection, "selection"), (fixture.boss, "boss")):
                doc["panel_identity"]["states_table"]["sha256"] = sha256(data).hexdigest()
                fixture.paths[kind].write_text(json.dumps(doc))
            argv = fixture.argv() + ["--run-states", str(path)]
            with redirect_stdout(io.StringIO()):
                run_main(fixture.root, argv)
            report = json.loads((fixture.root / "artifacts/report.json").read_text())
        # Selection level-0 cases are 0, 4 and 8 (all recorded wins); boss panel is all level 0.
        self.assertEqual(report["baalorlord_level0"]["selection"], {"cases": 3, "wins": 3})
        self.assertEqual(report["baalorlord_level0"]["boss"], {"cases": 4, "wins": 2})


class FakeFight:
    def __init__(self, legal_by_step, finish):
        self.legal_by_step = legal_by_step
        self.finish = finish
        self.actions = []
        self.resets = []

    def reset(self, case):
        self.resets.append(case["case_id"])
        return {"step": 0}, {"phase": "combat"}

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) == self.finish
        return ({"step": len(self.actions)}, float(done), done, False,
                {"phase": "terminal" if done else "combat"})


class FakeView:
    def __init__(self, fight):
        self.fight = fight

    def legal(self, observation):
        return list(self.fight.legal_by_step[observation["step"]])


CASE = {"case_id": "c1", "entry": {"id": "c1"}, "environment_seed": 3}


def searched(step, values, choice):
    return {"case_id": "c1", "step": step, "action_prefix_length": step, "root": "searched",
            "values": {str(a): [v, 8] for a, v in values.items()}, "choice": choice,
            "cpu_seconds": 0.1, "research_seed": 500 + step, "branch_refusals": 0, "simulations": 16}


class ConsistencyTests(unittest.TestCase):
    def rows(self):
        return [
            {**searched(0, {}, 0), "root": "single_legal", "values": None},
            searched(1, {0: 0.2, 1: 0.6}, 1),
            {**searched(2, {}, 2), "root": "a card read a hidden intent", "values": None},
            searched(3, {0: 0.1, 1: 0.4, 2: 0.3}, 1),
        ]

    def run_case(self, decide, sampled=(1, 3), legal=None):
        legal = legal or {0: [0], 1: [0, 1], 2: [1, 2], 3: [0, 1, 2], 4: []}
        fight = FakeFight(legal, finish=4)
        with patch.object(audit.entry, "_decision", side_effect=decide) as mock:
            results = audit.audit_case(fight, CASE, FakeView(fight), "teacher", self.rows(), sampled)
        return fight, mock, results

    def test_consistency_value_lost_is_zero_when_choices_agree(self):
        recorded = searched(1, {0: 0.2, 1: 0.6}, 1)
        agree = audit.compare(recorded, 1, {"0": [0.5, 8], "1": [0.9, 8]})
        self.assertEqual((agree["agree"], agree["lost_a"], agree["lost_b"]), (True, 0.0, 0.0))
        with self.assertRaises(audit.ReplayMismatch):
            audit.compare(recorded, 2, {"0": [0.5, 8], "1": [0.9, 8]})
        with self.assertRaises(audit.ReplayMismatch):
            audit.compare({**recorded, "choice": 5}, 1, {"0": [0.5, 8], "1": [0.9, 8]})
        differ = audit.compare(recorded, 0, {"0": [0.9, 8], "1": [0.5, 8]})
        self.assertFalse(differ["agree"])
        self.assertAlmostEqual(differ["lost_a"], 0.4)
        self.assertAlmostEqual(differ["lost_b"], 0.4)

        def same_choice(env, observation, legal, view, teacher, rng, seed):
            choice = {1: 1, 3: 1}[observation["step"]]
            return "searched", choice, {str(a): [0.0 if a != choice else 1.0, 8] for a in legal}, 0.2, 0, 16

        _, _, results = self.run_case(same_choice)
        summary = audit.summarize_consistency(results)
        self.assertEqual((summary["decisions"], summary["agreements"], summary["top_choice_agreement"]), (2, 2, 1.0))
        self.assertEqual((summary["mean_value_lost_run_a"], summary["mean_value_lost_run_b"]), (0.0, 0.0))

    def test_replay_follows_recorded_prefix_and_searches_with_next_seed(self):
        def decide(env, observation, legal, view, teacher, rng, seed):
            return "searched", legal[0], {str(a): [0.5, 8] for a in legal}, 0.2, 0, 16

        fight, mock, results = self.run_case(decide)
        self.assertEqual(fight.resets, ["c1"])
        # Search B at step 3 happens after the recorded steps 0-2 are replayed.
        self.assertEqual(fight.actions, [0, 1, 2, 1])
        self.assertEqual([call.args[6] for call in mock.call_args_list], [502, 504])
        self.assertEqual([call.args[4] for call in mock.call_args_list], ["teacher", "teacher"])
        self.assertEqual([r["agree"] for r in results], [False, False])
        self.assertAlmostEqual(results[0]["lost_a"], 0.4)
        self.assertAlmostEqual(results[1]["lost_a"], 0.3)

    def test_replay_divergence_is_an_error_never_skipped(self):
        decide = lambda *args: ("searched", 0, {"0": [0.5, 8], "1": [0.5, 8]}, 0.1, 0, 16)
        for name, legal in (
            ("recorded legal set differs", {0: [0], 1: [0, 1, 2], 2: [1, 2], 3: [0, 1, 2]}),
            ("single legal differs", {0: [0, 1], 1: [0, 1], 2: [1, 2], 3: [0, 1, 2]}),
            ("recorded choice is illegal", {0: [0], 1: [0, 1], 2: [0, 1], 3: [0, 1, 2]}),
        ):
            with self.subTest(name), self.assertRaises(audit.ReplayMismatch):
                self.run_case(decide, legal=legal)

        def refused(*args):
            return "draw knowledge exceeds the exact model", 0, None, None, 0, None

        with self.assertRaises(audit.ReplayMismatch):
            self.run_case(refused, sampled=(1,))

        fight = FakeFight({0: [0], 1: [0, 1], 2: [1, 2], 3: [0, 1, 2]}, finish=2)
        with patch.object(audit.entry, "_decision", side_effect=decide), self.assertRaises(audit.ReplayMismatch):
            audit.audit_case(fight, CASE, FakeView(fight), "teacher", self.rows(), (3,))

        with self.assertRaises(audit.AuditInputError):
            audit.case_decisions([searched(0, {0: 0.1}, 0), searched(2, {0: 0.1}, 0)])
        with self.assertRaises(audit.AuditInputError):
            audit.case_decisions([{**searched(0, {0: 0.1}, 0), "action_prefix_length": 1}])

    def test_audit_sample_is_selection_split_and_deterministic(self):
        rows = [decision(f"c{c}", step, "searched" if step % 3 else "single_legal")
                for c in range(10) for step in range(9)]
        summary = {"schema": "spire-learn-search-teacher-entry-v1",
                   "provenance": {"split": "selection", "panel": "selection"}}
        sample = audit.sample_decisions(summary, rows, 20, 5)
        self.assertEqual(len(sample), 20)
        self.assertEqual(len({(r["case_id"], r["step"]) for r in sample}), 20)
        self.assertTrue(all(r["root"] == "searched" for r in sample))
        self.assertEqual(sample, audit.sample_decisions(summary, list(reversed(rows)), 20, 5))
        self.assertNotEqual(sample, audit.sample_decisions(summary, rows, 20, 6))
        for split in ("train", "final", None):
            with self.subTest(split=split), self.assertRaisesRegex(audit.AuditInputError, "selection"):
                audit.sample_decisions({**summary, "provenance": {"split": split, "panel": None}}, rows, 20, 5)
        with self.assertRaisesRegex(audit.AuditInputError, "searched"):
            audit.sample_decisions(summary, rows, 61, 5)

    def test_consistency_cli_refuses_a_training_run_before_any_gym_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts/run"
            run.mkdir(parents=True)
            (run / "decisions.jsonl").write_text(json.dumps(decision("c0", 0, "searched")) + "\n")
            (run / "summary.json").write_text(json.dumps({
                "schema": "spire-learn-search-teacher-entry-v1",
                "provenance": {"split": "train", "panel": None, "teacher": "tree-3x"}}))
            argv = ["consistency", "--decisions", str(run / "decisions.jsonl"), "--sample", "1",
                    "--seed", "1", "--teacher", "tree-3x", "--teachers", "teachers.json",
                    "--recipe", "recipe.json", "--output", str(root / "artifacts/out.json")]
            with patch.object(audit.entry, "build_panel", side_effect=AssertionError("gym used")), \
                 self.assertRaisesRegex(audit.AuditInputError, "selection"):
                run_main(root, argv)
            self.assertFalse((root / "artifacts/out.json").exists())


class ConsistencyRunTests(unittest.TestCase):
    """run_consistency end to end with fakes for the teacher, panel, Gym build and replay seams."""

    LEGAL = {"p-case-alpha": {0: [0, 1], 1: [0, 1]}, "p-case-beta": {0: [0], 1: [0, 1, 2]}}
    BUILD = {"revision": "rev", "dirty": False, "source_digest": "src"}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run = self.root / "artifacts/run"
        self.run.mkdir(parents=True)
        self.recipe = self.root / "recipe.json"
        self.recipe.write_text("{}")
        self.cases = [{"case_id": cid, "entry": {"id": cid}, "environment_seed": index}
                      for index, cid in enumerate(self.LEGAL)]
        self.rows = [
            {**searched(0, {0: 0.2, 1: 0.6}, 1), "case_id": "p-case-alpha"},
            {**searched(1, {0: 0.5, 1: 0.1}, 0), "case_id": "p-case-alpha"},
            {**searched(0, {}, 0), "case_id": "p-case-beta", "root": "single_legal", "values": None},
            {**searched(1, {0: 0.3, 1: 0.9, 2: 0.1}, 1), "case_id": "p-case-beta"},
        ]
        self.fights = [{"case_id": cid, "outcome": "victory", "censor_reason": None, "steps": 2,
                        "hp_after_cleanup": 5, "cpu_seconds": 1.0, "wall_seconds": 1.0} for cid in self.LEGAL]
        self.provenance = {
            "split": "selection", "panel": "selection", "teacher": "tree-3x", "config_sha256": "cfg",
            "recipe_sha256": sha256(b"{}").hexdigest(), "gym_revision": "rev", "gym_build": dict(self.BUILD),
            "case_manifest_sha256": audit.entry.content_sha256(self.cases), "seed": 9,
        }
        self.build = dict(self.BUILD)
        self.panel_cases = self.cases

    def write(self):
        (self.run / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        (self.run / "fights.jsonl").write_text("".join(json.dumps(f) + "\n" for f in self.fights))
        (self.run / "summary.json").write_text(json.dumps({
            "schema": "spire-learn-search-teacher-entry-v1", "cases": len(self.fights),
            "roots": dict(Counter(r["root"] for r in self.rows)), "provenance": self.provenance}))

    def consistency(self, sample=3):
        self.write()
        fights = {}

        @contextmanager
        def fight_env(case, split):
            self.assertEqual(split, "selection")
            fight = fights[case["case_id"]] = FakeFight(self.LEGAL[case["case_id"]], finish=2)
            yield fight, fight

        def decide(env, observation, legal, view, teacher, rng, seed):
            self.assertEqual(teacher, "teacher")
            return "searched", 1, {str(a): [0.7 if a == 1 else 0.5, 8] for a in legal}, 0.25, 0, 16

        gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: self.build))
        argv = ["consistency", "--decisions", str(self.run / "decisions.jsonl"), "--sample", str(sample), "--seed", "4",
                "--teacher", "tree-3x", "--teachers", "teachers.json", "--recipe", str(self.recipe),
                "--output", str(self.root / "artifacts/consistency.json")]
        with patch.dict(sys.modules, {"spire_lab": gym}), \
             patch.object(audit.entry, "_teacher", return_value=("teacher", "cfg", True)), \
             patch.object(audit.entry, "load_recipe", return_value=SimpleNamespace(gym_revision="rev")), \
             patch.object(audit.entry, "build_panel", return_value={"manifest": {"cases": self.panel_cases}}), \
             patch.object(audit.entry, "preflight") as preflight, \
             patch.object(audit.entry, "_fight_env", fight_env), \
             patch.object(audit.entry, "View", FakeView), \
             patch.object(audit.entry, "_decision", side_effect=decide), \
             redirect_stdout(io.StringIO()) as stdout:
            run_main(self.root, argv)
        preflight.assert_called_once()
        self.assertEqual({cid: f.actions for cid, f in fights.items()},
                         {"p-case-alpha": [1, 0], "p-case-beta": [0, 1]})
        return (self.root / "artifacts/consistency.json").read_text(), stdout.getvalue()

    def test_consistency_run_writes_aggregate_report(self):
        text, stdout = self.consistency()
        self.assertNotIn("p-case-", text)
        self.assertNotIn("p-case-", stdout)
        report = json.loads(text)
        self.assertEqual(report["schema"], "spire-learn-teacher-consistency-v1")
        self.assertEqual((report["decisions"], report["agreements"]), (3, 2))
        self.assertAlmostEqual(report["top_choice_agreement"], 2 / 3)
        # Only alpha step 1 disagrees: A loses 0.5 - 0.1, B loses 0.7 - 0.5.
        self.assertAlmostEqual(report["mean_value_lost_run_a"], 0.4 / 3)
        self.assertAlmostEqual(report["mean_value_lost_run_b"], 0.2 / 3)
        self.assertAlmostEqual(report["mean_value_lost_run_a_when_disagreeing"], 0.4)
        self.assertEqual(report["provenance"]["sample_sha256"], audit.entry.content_sha256(
            [["p-case-alpha", 0], ["p-case-alpha", 1], ["p-case-beta", 1]]))
        self.assertEqual((report["provenance"]["split"], report["provenance"]["audit_only"]), ("selection", True))

    def test_consistency_run_refuses_records_that_do_not_cover_the_fights(self):
        # A decisions file truncated after alpha's first step; the summary roots agree with it.
        self.rows = [self.rows[0], *self.rows[2:]]
        with self.assertRaisesRegex(audit.AuditInputError, "do not match its fights"):
            self.consistency(sample=2)
        # A fight with no decision rows at all.
        self.rows = self.rows[1:]
        self.fights = self.fights[1:]
        self.fights.append({**self.fights[0], "case_id": "p-case-gamma"})
        with self.assertRaisesRegex(audit.AuditInputError, "do not match its fights"):
            self.consistency(sample=1)

    def test_consistency_run_refuses_a_case_missing_from_the_rebuilt_panel(self):
        self.panel_cases = self.cases[:1]
        self.provenance["case_manifest_sha256"] = audit.entry.content_sha256(self.panel_cases)
        with self.assertRaisesRegex(audit.AuditInputError, "not in the rebuilt selection panel"):
            self.consistency()

    def test_consistency_run_refuses_a_different_gym_build(self):
        for key, value in (("dirty", True), ("source_digest", "other")):
            with self.subTest(key=key):
                self.build = {**self.BUILD, key: value}
                with self.assertRaisesRegex(audit.AuditInputError, key):
                    self.consistency()


if __name__ == "__main__":
    unittest.main()
