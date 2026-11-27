import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import random
from unittest.mock import Mock, patch

from scripts.neural import search_teacher_entry as entry
from scripts.neural.boss_winnability import _is_train
from spire_learn.neural.gym_contracts import GYM_V1_2_0_CONTRACT, GYM_V1_2_4_CONTRACT
from spire_learn.neural.search_one_move import ActionStat
from spire_learn.neural.search_teacher import TeacherConfig, TeacherResult
from spire_learn.neural.search_value import RootContext, ValueRule
from spire_learn.neural.search_view import View
from spire_learn.neural.standard_stream import run_split
from tests.unit import _search_fakes as fakes


class Unsupported(Exception):
    pass


class StateError(Exception):
    pass


class Fault(Exception):
    pass


CASE = {"case_id": "fixture", "entry": {"id": "fixture"}, "environment_seed": 42}
CURRENT = {"samples": 32, "playout_steps": 80}
V2 = ValueRule("resource-v2")
POLICY = "no-potions-v2"
CURRENT_V2 = {**CURRENT, "value_rule": V2, "playout_policy": POLICY}
TEACHERS = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
# The fakes' player columns, then the public turn.
TURN = len(fakes.PLAYER)


def public(turn, **fields):
    """A Standard-shaped public observation (``_search_fakes``) with the public turn appended."""
    row = fakes.observation(**fields)
    row["player"] = fakes.Arr([*row["player"], turn])
    row["player_known"] = fakes.Arr([*row["player_known"], 1])
    return row


def resource_view(legal=(0,), choice=None):
    """A real ``View`` on the fake descriptor whose legal actions and heuristic move are fixed."""
    view = Mock(wraps=View(fakes.descriptor()))
    view.legal.return_value = list(legal)
    view.heuristic_choice.return_value = choice
    return view


class Fight:
    def __init__(self, *, legal=(0,), turn_stride=1, terminal=False, finish=None, start=None, after=None,
                 verbs=None):
        self.legal = list(legal)
        self.turn_stride = turn_stride
        self.finish = finish
        self.terminal = terminal
        self.start = start or {}
        self.after = self.start if after is None else after
        self.verbs = tuple("play_card" for _ in self.legal) if verbs is None else verbs
        self.actions = []
        self.public_search = Mock(side_effect=Unsupported("runic_dome"))

    def observation(self):
        fields = self.after if self.actions else self.start
        row = public(len(self.actions) // self.turn_stride, verbs=self.verbs, **fields)
        row["hidden_rng"] = "private"
        return row

    def reset(self, case):
        return self.observation(), {
            "phase": "terminal" if self.terminal else "combat",
            "terminal_reason": "victory" if self.terminal else None,
            "hp_after_cleanup": 3, "hidden_seed": "private",
        }

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) == self.finish
        return self.observation(), float(done), done, False, {"hp_after_cleanup": 3, "terminal_reason": "victory" if done else None}


def process_job(job):
    case = {**CASE, "case_id": f"fixture-{job}"}
    fight, decisions = entry.play_fight(
        Fight(finish=job + 1, after={"hp": 40 - job, "potions": ("fire_potion",)}), case, resource_view(),
        TURN, CURRENT, research_seed=job,
    )
    return os.getpid(), fight, decisions


class TeacherEntryTests(unittest.TestCase):
    def play(self, fight, teacher=CURRENT, view=None, **kwargs):
        if view is None:
            view = resource_view(fight.legal, fight.legal[-1] if fight.legal else None)
        with patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)):
            return entry.play_fight(fight, CASE, view, TURN, teacher, research_seed=11, **kwargs)

    def test_terminal_at_reset_fight_has_zero_decisions(self):
        fight = Fight(terminal=True, legal=())
        result, records = self.play(fight)
        self.assertEqual((result["outcome"], result["steps"], records), ("victory", 0, []))
        fight.public_search.assert_not_called()

    def test_single_legal_decision_has_no_search_and_no_label(self):
        fight = Fight(finish=1)
        _, records = self.play(fight)
        self.assertEqual(fight.actions, [0])
        self.assertEqual(records[0]["root"], "single_legal")
        self.assertIsNone(records[0]["values"])
        fight.public_search.assert_not_called()

    def test_refused_root_records_class_plays_heuristic_and_has_no_label(self):
        fight = Fight(legal=(0, 1), finish=1)
        _, records = self.play(fight)
        self.assertEqual(fight.actions, [1])
        self.assertEqual(records[0]["root"], "runic_dome")
        self.assertIsNone(records[0]["values"])

    def test_turn_cap_censors_at_500_and_step_cap_at_1024(self):
        for stride, cap, reason in ((2000, 500, "turn_cap"), (2, 1024, "step_cap")):
            with self.subTest(stride=stride):
                result, records = self.play(Fight(turn_stride=stride))
                self.assertEqual((result["outcome"], result["censor_reason"]), ("censored", reason))
                self.assertEqual((result["steps"], len(records)), (cap, cap))

    def test_fault_is_an_error_not_a_loss(self):
        fight = Fight(legal=(0, 1))
        fight.public_search.side_effect = Fault("branch fault")
        with self.assertRaises(Fault):
            self.play(fight)
        self.assertEqual(fight.actions, [])

    def test_train_sample_uses_train_split_only(self):
        states = [{"run_id": f"run-{i}", "floor": 1, "character": "ironclad",
                   "entry": {"id": f"entry-{i}", "ascension": 20}} for i in range(100)]
        self.assertEqual({run_split(s["run_id"]) for s in states}, {"train", "selection", "final"})
        cases = entry.sample_train_cases(states, 20, 103)
        self.assertEqual(cases, entry.sample_train_cases(list(reversed(states)), 20, 103))
        self.assertNotEqual(cases, entry.sample_train_cases(states, 20, 104))
        by_id = {s["entry"]["id"]: s["run_id"] for s in states}
        self.assertEqual(len({c["case_id"] for c in cases}), 20)
        self.assertTrue(all(_is_train(by_id[c["entry"]["id"]]) for c in cases))

    def test_outputs_have_no_hidden_state(self):
        result, records = self.play(Fight(finish=1))
        self.assertEqual(set(result), {"case_id", "outcome", "censor_reason", "steps",
                                       "hp_after_cleanup", "cpu_seconds", "wall_seconds",
                                       "hp_start", "max_hp_start", "max_hp_end", "potions_start",
                                       "potions_end", "potion_hp_spent"})
        self.assertEqual(set(records[0]), {"case_id", "step", "action_prefix_length", "root",
                                           "values", "choice", "cpu_seconds", "research_seed",
                                           "branch_refusals", "simulations"})

    def test_compact_preflight_fails_before_fights_on_capacity_fault(self):
        vector = Mock()
        vector.reset.side_effect = Fault("Run capacity actions: 1025 > 1024")
        factory = Mock(return_value=vector)
        numpy = Mock()
        mask = numpy.ones.return_value
        with (patch.dict("sys.modules", {"numpy": numpy}),
              patch.object(entry, "_vector_env", factory), self.assertRaises(Fault)):
            entry.preflight([CASE], "selection")
        vector.reset.assert_called_once_with(seed=[42], options={
            "reset_mask": mask, "entry_ids": ["fixture"],
        })
        factory.assert_called_once_with(CASE, "selection")
        vector.close.assert_called_once()

    def test_searched_root_records_values_time_and_refusals(self):
        fight = Fight(legal=(0, 1), finish=1)
        arena = Mock()
        fight.public_search = Mock(return_value=arena)
        with (patch.object(entry, "View"),
              patch.object(entry, "one_move_values", return_value={
                  0: ActionStat(0.5, 32, 2), 1: ActionStat(1.0, 32, 1),
              }) as search):
            _, records = self.play(fight)
        self.assertEqual(fight.actions, [1])
        row, = records
        self.assertEqual(row["root"], "searched")
        self.assertEqual(row["values"], {"0": [0.5, 32], "1": [1.0, 32]})
        self.assertGreater(row["cpu_seconds"], 0)
        self.assertEqual(row["branch_refusals"], 3)
        self.assertEqual(row["simulations"], 64)
        self.assertEqual(search.call_args.kwargs["research_seed"], row["research_seed"])
        self.assertEqual(fight.public_search.call_args.kwargs["workers"], 1)
        self.assertEqual(fight.public_search.call_args.kwargs["capacity_preset"],
                         "standard-combat-compact-v1")
        arena.close.assert_called_once()

    def test_uncalibrated_tree_is_allowed_only_for_training_sample(self):
        real = entry._ROOT / "configs/neural/search-teacher-gym-1.2.0/teachers.json"
        data = json.loads(real.read_text())
        data["teachers"]["tree-1x"]["calibrated"] = False
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory) / "teachers.json"
            configs.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "uncalibrated"):
                entry._teacher(configs, "tree-1x", "selection")
            teacher, digest, calibrated = entry._teacher(configs, "tree-1x", None)
        self.assertFalse(calibrated)
        self.assertEqual(digest, entry.config_sha256(teacher))

    def test_committed_tree_teachers_are_calibrated_with_spec_root_visits(self):
        configs = entry._ROOT / "configs/neural/search-teacher-gym-1.2.0/teachers.json"
        for name in ("tree-1x", "tree-3x", "tree-10x"):
            teacher, _, calibrated = entry._teacher(configs, name, "selection")
            self.assertTrue(calibrated)
            self.assertEqual(teacher.min_root_visits, 8)

    def test_parallel_workers_emit_identical_rows_in_case_order(self):
        jobs = list(range(8))
        with patch.object(entry, "_run_case", process_job):
            serial = list(entry._results(jobs, 1))
            parallel = list(entry._results(jobs, 2))
        self.assertTrue(all(pid == os.getpid() for pid, _, _ in serial))
        self.assertTrue(all(pid != os.getpid() for pid, _, _ in parallel))
        self.assertEqual([fight["case_id"] for _, fight, _ in parallel],
                         [f"fixture-{job}" for job in jobs])

        def without_timings(row):
            return {key: value for key, value in row.items()
                    if key not in {"cpu_seconds", "wall_seconds"}}

        def payload(results):
            return [(without_timings(fight), [without_timings(row) for row in decisions])
                    for _, fight, decisions in results]

        self.assertEqual(payload(serial), payload(parallel))


class SearchFightParityTests(unittest.TestCase):
    def setUp(self):
        self.vector = Mock()
        self.standalone = Mock()
        self.standalone_factory = Mock(return_value=self.standalone)
        gymnasium = SimpleNamespace(StandardCombatEnv=self.standalone_factory)
        numpy = SimpleNamespace(bool_=bool, ones=lambda size, **_: [True] * size,
                                asarray=lambda values, **_: values,
                                array_equal=lambda left, right: left == right)
        self.enterContext(patch.dict("sys.modules", {
            "numpy": numpy, "spire_lab.gymnasium": gymnasium,
        }))
        self.enterContext(patch.object(entry, "_vector_env", return_value=self.vector))

        reset = public(1, verbs=("play_card",))
        successor = public(2, verbs=("play_card", "end_turn"))
        info = {"phase": "combat", "terminal_reason": None, "hp_after_cleanup": None}
        self.vector.reset.return_value = self.batch(reset), self.info_batch(info)
        self.vector.step.return_value = (
            self.batch(successor), [0.0], [False], [False], self.info_batch(info),
        )
        self.standalone.reset.return_value = reset, info
        self.standalone.step.return_value = successor, 0.0, False, False, info

    @staticmethod
    def batch(observation):
        return {key: [value] for key, value in observation.items()}

    @staticmethod
    def info_batch(info):
        return {key: [value] for field, item in info.items()
                for key, value in ((field, item), ("_" + field, True))}

    def assert_fight_rejects(self, message, *, after_step):
        view = View(fakes.descriptor())
        with (patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)),
              self.assertRaisesRegex(RuntimeError, message)):
            with entry._fight_env(CASE, "train") as (fight, _):
                entry.play_fight(fight, CASE, view, TURN, CURRENT, research_seed=11)
        self.standalone.public_search.assert_not_called()
        self.vector.close.assert_called_once_with()
        self.standalone.close.assert_called_once_with()
        if after_step:
            self.vector.step.assert_called_once_with([0])
            self.standalone.step.assert_called_once_with(0)
        else:
            self.vector.step.assert_not_called()
            self.standalone.step.assert_not_called()

    def test_rejects_observation_divergence_at_reset(self):
        observation, info = self.standalone.reset.return_value
        self.standalone.reset.return_value = {**observation, "player": [3]}, info
        self.assert_fight_rejects("public observations diverged", after_step=False)

    def test_rejects_observation_divergence_after_step(self):
        observation, *rest = self.standalone.step.return_value
        self.standalone.step.return_value = ({**observation, "player": [3]}, *rest)
        self.assert_fight_rejects("public observations diverged", after_step=True)

    def test_rejects_legal_mask_divergence(self):
        observation, *rest = self.standalone.step.return_value
        self.standalone.step.return_value = ({**observation, "action_mask": [0, 1]}, *rest)
        self.assert_fight_rejects("public observations diverged", after_step=True)

    def test_rejects_transition_outcome_divergence(self):
        original = self.standalone.step.return_value
        for index, value in ((1, 1.0), (2, True), (3, True)):
            with self.subTest(field=("reward", "terminated", "truncated")[index - 1]):
                modified = list(original)
                modified[index] = value
                self.standalone.step.return_value = tuple(modified)
                self.assert_fight_rejects("transition outcomes diverged", after_step=True)
                self.vector.reset_mock()
                self.standalone.reset_mock()

    def test_rejects_boundary_info_divergence(self):
        *transition, info = self.standalone.step.return_value
        for field, value in (("phase", "terminal"), ("terminal_reason", "victory"), ("hp_after_cleanup", 4)):
            with self.subTest(field=field):
                self.standalone.step.return_value = (*transition, {**info, field: value})
                self.assert_fight_rejects("public boundary info diverged", after_step=True)
                self.vector.reset_mock()
                self.standalone.reset_mock()

    def test_closes_vector_when_standalone_construction_fails(self):
        self.standalone_factory.side_effect = Fault("standalone construction failed")
        with self.assertRaisesRegex(Fault, "standalone construction failed"):
            with entry._fight_env(CASE, "train"):
                self.fail("Fight yielded despite a standalone construction fault")
        self.standalone_factory.assert_called_once()
        self.vector.close.assert_called_once_with()
        self.vector.reset.assert_not_called()
        self.standalone.public_search.assert_not_called()


class BossPanelEntryTest(unittest.TestCase):
    def test_boss_cli_uses_selection_split_boss_panel(self):
        revision = GYM_V1_2_0_CONTRACT.source_revision
        recipe = Mock(spec=entry.StandardRecipe, gym_revision=revision)
        gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: {"revision": revision}))
        stop = RuntimeError("dispatched panel")
        output = entry._ROOT / "artifacts" / "unused-boss-test-output"
        with patch.dict(sys.modules, {"spire_lab": gym}), patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": ""}), \
             patch.object(entry, "_teacher", return_value=(None, None, True)), \
             patch.object(entry, "load_recipe", return_value=recipe), \
             patch.object(entry, "build_panel", side_effect=stop) as build:
            with self.assertRaisesRegex(RuntimeError, "dispatched panel"):
                entry.main(["--recipe", "recipe.json", "--panel", "boss", "--teacher", "current",
                            "--teachers", "teachers.json", "--seed", "1", "--output", str(output)])
        build.assert_called_once_with(recipe, "selection", kind="boss")


class PerActionTeacherEntryTests(unittest.TestCase):
    def test_gym_1_2_4_tree_teachers_use_per_action_budgets(self):
        configs = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
        for name, k in (("tree-1x", 22), ("tree-3x", 66), ("tree-10x", 220)):
            teacher, _, calibrated = entry._teacher(configs, name, None)
            self.assertTrue(calibrated)
            self.assertEqual(teacher.min_root_visits, 8)
            self.assertEqual((teacher.simulations, teacher.simulations_per_action), (None, k))
            self.assertEqual(teacher.decision_simulations(5), 5 * k)


CURRENT_SHA = "2f29eef07620a022843bff0b634aad167d70902f06331ee6a5ef4c3284f1cdb4"
SQRT_15 = {"name": "resource-v2", "potion_hp": 15, "hp_curve": "sqrt"}


def v2_teachers(directory, **rows):
    """The committed gym-1.2.4 teachers plus uncalibrated v2 rows; ``rows`` override or add teachers."""
    data = json.loads(TEACHERS.read_text())
    tree = data["teachers"]["tree-3x"]
    data["teachers"].update({
        "current-v2": {"kind": "one-move", "calibrated": False, "samples": 32, "playout_steps": 80,
                       "playout_backend": "python", "value_rule": {"name": "resource-v2"},
                       "playout_policy": POLICY},
        "tree-3x-v2": {"calibrated": False, "target_cost_multiple": 3,
                       "config": {**tree["config"], "playout_backend": "python",
                                  "value_rule": {"name": "resource-v2"}, "playout_policy": POLICY}},
        **rows,
    })
    path = Path(directory) / "teachers.json"
    path.write_text(json.dumps(data))
    return path


def run_entry(directory, teachers, name, *, allow_uncalibrated=False,
              revision=GYM_V1_2_4_CONTRACT.source_revision):
    """``_run`` on one selection fight, with the Gym, recipe and panel replaced by fakes.

    The recipe is for Gym v1.2.4; ``revision`` is the live Gym's (v1.2.4 or a drop-in in its rules epoch).
    """
    recipe = Mock(spec=entry.StandardRecipe, gym_revision=GYM_V1_2_4_CONTRACT.source_revision,
                  table_sha256="table", character="ironclad", ascension=20)
    gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: {"revision": revision}))
    panel = {"manifest": {"cases": [CASE]}, "panel_identity": "panel", "panel_sha256": "p", "manifest_sha256": "m"}
    descriptor = {"profile_id": "profile", "schema_id": "schema", "fingerprint": "fingerprint"}
    fight = {"case_id": CASE["case_id"], "outcome": "victory", "cpu_seconds": 0.0}
    recipe_path = Path(directory) / "recipe.json"
    recipe_path.write_text("{}")
    args = SimpleNamespace(teachers=teachers, teacher=name, panel="selection", recipe=recipe_path,
                           train_sample=None, seed=1, limit=None, max_steps=1024, max_turn_actions=500,
                           parallel=1, allow_uncalibrated=allow_uncalibrated)
    with (patch.dict(sys.modules, {"spire_lab": gym}),
          patch.object(entry, "load_recipe", return_value=recipe),
          patch.object(entry, "build_panel", return_value=panel),
          patch.object(entry, "preflight", return_value=descriptor),
          patch.object(entry, "_results", return_value=[(fight, [])]) as results):
        summary = entry._run(args, Path(directory) / f"output-{name}")
    return summary, results.call_args.args[0]


class NamedTeacherTests(unittest.TestCase):
    def test_v2_teacher_names_load_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = v2_teachers(directory)
            one_move, one_sha, one_calibrated = entry._teacher(configs, "current-v2", None)
            tree, tree_sha, tree_calibrated = entry._teacher(configs, "tree-3x-v2", None)
            with self.assertRaisesRegex(ValueError, "unknown teacher 'tree-99x'.*current, current-v2, tree-10x"):
                entry._teacher(configs, "tree-99x", None)
            with (patch.object(entry, "_run", return_value={}) as run,
                  patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": ""})):
                entry.main(["--recipe", "recipe.json", "--train-sample", "1", "--teacher", "tree-3x-v2",
                            "--teachers", str(configs), "--seed", "1",
                            "--output", str(entry._ROOT / "artifacts" / "unused-v2-name-test")])
        self.assertEqual(run.call_args.args[0].teacher, "tree-3x-v2")
        self.assertFalse(run.call_args.args[0].allow_uncalibrated)
        self.assertEqual(one_move, CURRENT_V2)
        self.assertIsInstance(tree, TeacherConfig)
        self.assertEqual((tree.value_rule, tree.playout_policy, tree.playout_backend), (V2, POLICY, "python"))
        self.assertEqual(tree.simulations_per_action, 66)
        self.assertEqual(tree_sha, entry.config_sha256(tree))
        self.assertFalse(one_calibrated or tree_calibrated)
        v1_tree = entry._teacher(TEACHERS, "tree-3x", None)[1]
        self.assertEqual(len({one_sha, tree_sha, CURRENT_SHA, v1_tree}), 4)

    def test_v2_rows_reject_invalid_value_options(self):
        tree = json.loads(TEACHERS.read_text())["teachers"]["tree-3x"]
        # Native v2 rows are valid offline (the live Gym's tiered rule is checked in ``_run``), but
        # resource-v2 still needs no-potions-v2 on either backend.
        bad = {
            "native-uniform-v2": {"kind": "one-move", "samples": 32, "playout_steps": 80,
                                  "playout_backend": "native", "value_rule": {"name": "resource-v2"}},
            "uniform-v2": {"kind": "one-move", "samples": 32, "playout_steps": 80,
                           "value_rule": {"name": "resource-v2"}},
            "extra-field": {"kind": "one-move", "samples": 32, "playout_steps": 80,
                            "value_rule": {"name": "resource-v2", "scale": 2}, "playout_policy": POLICY},
            "tree-native-uniform-v2": {"calibrated": False, "config": {**tree["config"],
                                                                       "value_rule": {"name": "resource-v2"}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            configs = v2_teachers(directory, **bad)
            for name in bad:
                with self.subTest(teacher=name), self.assertRaises(ValueError):
                    entry._teacher(configs, name, None)

    def test_current_hash_unchanged_after_dispatch_change(self):
        teacher, digest, calibrated = entry._teacher(TEACHERS, "current", "selection")
        self.assertEqual(digest, CURRENT_SHA)
        self.assertEqual(teacher, {"samples": 32, "playout_steps": 80, "playout_backend": "native"})
        self.assertTrue(calibrated)
        # Dispatch follows the row's kind, not its name: the same row under another name hashes the same.
        row = json.loads(TEACHERS.read_text())["teachers"]["current"]
        with tempfile.TemporaryDirectory() as directory:
            configs = v2_teachers(directory, renamed=row, **{"current-explicit": {
                **row, "value_rule": {"name": "flat-v1"}, "playout_policy": "uniform-v1"}})
            self.assertEqual(entry._teacher(configs, "renamed", "selection")[:2], (teacher, CURRENT_SHA))
            self.assertEqual(entry._teacher(configs, "current-explicit", "selection")[1], CURRENT_SHA)
            summary, _ = run_entry(directory, TEACHERS, "current")
        provenance = summary["provenance"]
        self.assertEqual(provenance["config_sha256"], CURRENT_SHA)
        self.assertEqual(provenance["teacher_config"], teacher)
        self.assertNotIn("uncalibrated_allowed", provenance)

    def test_uncalibrated_panel_run_needs_flag_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = v2_teachers(directory)
            for name in ("current-v2", "tree-3x-v2"):
                with self.subTest(teacher=name):
                    with self.assertRaisesRegex(ValueError, "uncalibrated"):
                        entry._teacher(configs, name, "selection")
                    with self.assertRaisesRegex(ValueError, "uncalibrated"):
                        run_entry(directory, configs, name)
                    _, _, calibrated = entry._teacher(configs, name, "selection", allow_uncalibrated=True)
                    self.assertFalse(calibrated)
                    summary, _ = run_entry(directory, configs, name, allow_uncalibrated=True)
                    provenance = summary["provenance"]
                    self.assertIs(provenance["uncalibrated_allowed"], True)
                    self.assertIs(provenance["calibrated"], False)
                    self.assertEqual(provenance["panel"], "selection")
            with patch.object(entry, "_run", return_value={}) as run, \
                 patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": ""}):
                entry.main(["--recipe", "recipe.json", "--panel", "selection", "--teacher", "current-v2",
                            "--teachers", str(configs), "--seed", "1", "--allow-uncalibrated",
                            "--output", str(entry._ROOT / "artifacts" / "unused-v2-flag-test")])
        self.assertTrue(run.call_args.args[0].allow_uncalibrated)

    def test_value_rule_round_trips_into_provenance(self):
        one_move = {"kind": "one-move", "calibrated": True, "samples": 32, "playout_steps": 80,
                    "value_rule": SQRT_15, "playout_policy": POLICY}
        tree = json.loads(TEACHERS.read_text())["teachers"]["tree-1x"]
        tree = {**tree, "config": {**tree["config"], "playout_backend": "python", "value_rule": SQRT_15,
                                   "playout_policy": POLICY}}
        rule = ValueRule.from_identity(SQRT_15)
        with tempfile.TemporaryDirectory() as directory:
            configs = v2_teachers(directory, **{"one-move-sqrt": one_move, "tree-sqrt": tree})
            for name in ("one-move-sqrt", "tree-sqrt"):
                with self.subTest(teacher=name):
                    teacher, digest, _ = entry._teacher(configs, name, "selection")
                    summary, jobs = run_entry(directory, configs, name)
                    provenance = summary["provenance"]
                    config = provenance["teacher_config"]
                    self.assertEqual(config["value_rule"], rule.as_identity())
                    self.assertEqual(config["value_rule"]["potion_overrides"],
                                     {"fairy_potion": 30.0, "fruit_juice": 7.5})
                    self.assertEqual(ValueRule.from_identity(config["value_rule"]), rule)
                    self.assertEqual(config["playout_policy"], POLICY)
                    self.assertEqual(provenance["config_sha256"], digest)
                    written = json.loads((Path(directory) / f"output-{name}" / "summary.json").read_text())
                    self.assertEqual(written["provenance"]["teacher_config"], json.loads(json.dumps(config)))
                    self.assertEqual(jobs[0][2], teacher)
                    self.assertNotIn("playout_backend", provenance)
            self.assertEqual(entry._teacher(configs, "tree-sqrt", None)[0].value_rule, rule)
            one, _, _ = entry._teacher(configs, "one-move-sqrt", None)
        self.assertEqual(one, {**CURRENT, "value_rule": rule, "playout_policy": POLICY})


class V2DecisionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(entry.search_view, "_native_errors",
                                       return_value=(Unsupported, StateError, Fault)))

    def tree(self, **overrides):
        return TeacherConfig(simulations_per_action=8, worlds=2, ucb_c=1.0, max_depth=2, widening=None,
                             **overrides)

    def refused(self, teacher):
        # Real heuristic_choice prefers the potion (index 1); no-potions-v2 ends the turn (index 0).
        observation = public(0, verbs=("end_turn", "use_potion"), potions=("fire_potion",))
        view = Mock(wraps=View(fakes.descriptor()))
        env = Mock(public_search=Mock(side_effect=Unsupported("runic_dome")))
        rng = random.Random(5)
        result = entry._decision(env, observation, [0, 1], view, teacher, rng, 7)
        return result, view, rng, observation

    def test_refused_root_uses_policy_for_v2(self):
        flat_control = {**CURRENT, "playout_policy": POLICY}
        for name, teacher in (("one-move-v2", CURRENT_V2), ("one-move-control", flat_control),
                              ("tree-v2", self.tree(value_rule=V2, playout_policy=POLICY)),
                              ("tree-control", self.tree(playout_policy=POLICY))):
            with self.subTest(teacher=name):
                (root, choice, values, *_), view, rng, observation = self.refused(teacher)
                self.assertEqual((root, choice, values), ("runic_dome", 0, None))
                view.playout_choice.assert_called_once_with(observation, rng, POLICY)
                view.heuristic_choice.assert_not_called()
        for name, teacher in (("one-move-v1", CURRENT), ("tree-v1", self.tree())):
            with self.subTest(teacher=name):
                (root, choice, values, *_), view, rng, observation = self.refused(teacher)
                reference = random.Random(5)
                expected = View(fakes.descriptor()).heuristic_choice(observation, reference)
                self.assertEqual((root, choice, values), ("runic_dome", expected, None))
                self.assertEqual(expected, 1)
                self.assertEqual(rng.getstate(), reference.getstate())
                view.heuristic_choice.assert_called_once_with(observation, rng)
                view.playout_choice.assert_not_called()

    def test_searched_v2_root_passes_the_decision_root_context(self):
        observation = public(0, hp=40, max_hp=70, potions=("fairy_potion",), verbs=("play_card", "end_turn"))
        view = View(fakes.descriptor())
        expected = RootContext.from_observation(view, observation, V2)
        stats = {0: ActionStat(0.5, 8, 0), 1: ActionStat(0.25, 8, 0)}
        cases = (("one-move-v2", CURRENT_V2, expected), ("one-move-v1", CURRENT, None),
                 ("tree-v2", self.tree(value_rule=V2, playout_policy=POLICY), expected),
                 ("tree-v1", self.tree(), None))
        for name, teacher, root in cases:
            with self.subTest(teacher=name):
                env = Mock()
                with (patch.object(entry, "View"),
                      patch.object(entry, "one_move_values", return_value=stats) as one_move,
                      patch.object(entry, "teach", return_value=TeacherResult(0, stats, 16, 0, 0.0)) as tree):
                    result = entry._decision(env, observation, [0, 1], view, teacher, random.Random(3), 9)
                self.assertEqual(result[:2], ("searched", 0))
                call = (tree if isinstance(teacher, TeacherConfig) else one_move).call_args
                if root is None:
                    self.assertNotIn("root", call.kwargs)
                else:
                    self.assertEqual(call.kwargs["root"], root)
                env.public_search.return_value.close.assert_called_once()


class FightResourceTests(unittest.TestCase):
    START = {"hp": 50, "max_hp": 80, "potions": ("fire_potion", "fairy_potion", "block_potion")}
    END = {"hp": 30, "max_hp": 85, "potions": ("block_potion", "fruit_juice")}

    def play(self, fight, teacher=CURRENT):
        view = resource_view(fight.legal, fight.legal[-1] if fight.legal else None)
        with patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)):
            return entry.play_fight(fight, CASE, view, TURN, teacher, research_seed=11)[0]

    def resources(self, row):
        keys = ("hp_start", "max_hp_start", "max_hp_end", "potions_start", "potions_end", "potion_hp_spent")
        for key in keys[:-1]:
            self.assertIs(type(row[key]), int, key)
        self.assertIs(type(row["potion_hp_spent"]), float)
        return tuple(row[key] for key in keys)

    def test_fight_row_records_resources(self):
        # Spent fire (12) and fairy (30), gained fruit juice (7.5): 34.5 HP on the fixed reference prices.
        expected = (50, 80, 85, 3, 2, 34.5)
        dear = {**CURRENT_V2, "value_rule": ValueRule("resource-v2", potion_hp=15.0,
                                                      potion_overrides=(("fairy_potion", 50.0),))}
        for name, teacher in (("v1", CURRENT), ("v2-own-prices", dear)):
            with self.subTest(teacher=name):
                row = self.play(Fight(legal=(0, 1), finish=1, start=self.START, after=self.END), teacher)
                self.assertEqual((row["outcome"], row["hp_after_cleanup"]), ("victory", 3))
                self.assertEqual(self.resources(row), expected)

        censored = self.play(Fight(turn_stride=2000, start=self.START, after=self.END))
        self.assertEqual((censored["outcome"], censored["hp_after_cleanup"]), ("censored", None))
        self.assertEqual(self.resources(censored), expected)

        at_reset = self.play(Fight(terminal=True, legal=(), start=self.START, after=self.END))
        self.assertEqual((at_reset["outcome"], at_reset["steps"], at_reset["hp_after_cleanup"]), ("victory", 0, 3))
        self.assertEqual(self.resources(at_reset), (50, 80, 80, 3, 3, 0.0))


class PlayFightHookTests(unittest.TestCase):
    """The chooser, observer and finisher hooks; the defaults leave play_fight unchanged."""

    STATS = {0: ActionStat(0.5, 32, 0), 1: ActionStat(1.0, 32, 1)}
    TIED = {0: ActionStat(0.5, 32, 0), 1: ActionStat(0.5, 32, 0)}

    def play(self, fight, stats, **hooks):
        view = resource_view(fight.legal, fight.legal[-1])
        fight.public_search = Mock(return_value=Mock())
        with (patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)),
              patch.object(entry, "View"), patch.object(entry, "one_move_values", return_value=stats)):
            result, rows = entry.play_fight(fight, CASE, view, TURN, CURRENT, research_seed=11, **hooks)
        for row in [result, *rows]:
            row.pop("cpu_seconds")
            row.pop("wall_seconds", None)
        return result, rows

    def test_default_hooks_unchanged(self):
        def row(step, choice, values, refusals=0):
            return {"case_id": "fixture", "step": step, "action_prefix_length": step, "root": "searched",
                    "values": values, "choice": choice, "research_seed": 11 * 1024 + step,
                    "branch_refusals": refusals, "simulations": 64}

        fight_row = {"case_id": "fixture", "outcome": "victory", "censor_reason": None, "steps": 2,
                     "hp_after_cleanup": 3, "hp_start": 50, "max_hp_start": 80, "max_hp_end": 80,
                     "potions_start": 0, "potions_end": 0, "potion_hp_spent": 0.0}
        best = {"0": [0.5, 32], "1": [1.0, 32]}
        # Pinned: today's rows and fight row for a fixed searched fight.
        self.assertEqual(self.play(Fight(legal=(0, 1), finish=2), self.STATS),
                         (fight_row, [row(0, 1, best, 1), row(1, 1, best, 1)]))
        # A tie is broken by the fight's own Random(research_seed); the hooks must not perturb it.
        reference = random.Random(11)
        ties = [reference.choice([0, 1]) for _ in range(3)]
        tied = {"0": [0.5, 32], "1": [0.5, 32]}
        baseline = self.play(Fight(legal=(0, 1), finish=3), self.TIED)
        self.assertEqual(baseline, ({**fight_row, "steps": 3}, [row(i, ties[i], tied) for i in range(3)]))

        seen = []
        hooked = self.play(
            Fight(legal=(0, 1), finish=3), self.TIED,
            chooser=lambda observation, legal, teacher, rng: teacher[1],
            observer=lambda observation, decision, turn: seen.append((decision["step"], turn)),
            finisher=lambda *args: seen.append("end"))
        self.assertEqual(hooked, baseline)
        self.assertEqual(seen, [(0, 0), (1, 1), (2, 2), "end"])

    def test_chooser_plays_its_move_and_must_stay_legal(self):
        fight = Fight(legal=(0, 1), finish=2)
        seen = []

        def chooser(observation, legal, teacher, rng):
            seen.append((legal, teacher[:3], isinstance(rng, random.Random)))
            return 0

        _, rows = self.play(fight, self.STATS, chooser=chooser)
        self.assertEqual(fight.actions, [0, 0])
        self.assertEqual([row["choice"] for row in rows], [0, 0])  # rows record the move played
        best = {"0": [0.5, 32], "1": [1.0, 32]}
        self.assertEqual(seen, [([0, 1], ("searched", 1, best), True)] * 2)
        with self.assertRaisesRegex(RuntimeError, "legal"):
            self.play(Fight(legal=(0, 1), finish=2), self.STATS, chooser=lambda *args: 5)

    def test_observer_sees_the_decision_observation_before_the_step(self):
        fight = Fight(legal=(0, 1), finish=2, start={"hp": 50}, after={"hp": 40})
        seen = []
        self.play(fight, self.STATS, observer=lambda observation, decision, turn: seen.append(
            (observation["player"][0], decision["step"], len(fight.actions), turn)))
        self.assertEqual(seen, [(50, 0, 0, 0), (40, 1, 1, 1)])

    def test_finisher_receives_end_state(self):
        start = {"hp": 50, "max_hp": 80, "potions": ("fire_potion", "fairy_potion")}
        end = {"hp": 30, "max_hp": 85, "cleanup": 31, "potions": ("fruit_juice",)}
        view = View(fakes.descriptor())
        calls = []

        def finisher(final_observation, info, outcome, censor_reason):
            calls.append((view.potion_kinds(final_observation), view.player_max_hp(final_observation),
                          view.cleanup_hp(final_observation), info.get("terminal_reason"),
                          info.get("hp_after_cleanup"), outcome, censor_reason))

        juice = fakes.POTIONS.index("fruit_juice")
        result, _ = self.play(Fight(legal=(0, 1), finish=2, start=start, after=end), self.STATS, finisher=finisher)
        self.assertEqual(calls, [((juice,), 85, 31, "victory", 3, "victory", None)])
        self.assertEqual(result["max_hp_end"], 85)

        calls.clear()
        self.play(Fight(legal=(0, 1), turn_stride=2000, start=start, after=end), self.STATS, finisher=finisher)
        self.assertEqual(calls, [((juice,), 85, 31, None, 3, "censored", "turn_cap")])

        calls.clear()
        at_reset = {**start, "cleanup": 44}
        fight = Fight(legal=(0, 1), terminal=True, start=at_reset, after=at_reset)
        self.play(fight, self.STATS, finisher=finisher)
        fire, fairy = (fakes.POTIONS.index(name) for name in ("fire_potion", "fairy_potion"))
        self.assertEqual(calls, [(tuple(sorted((fire, fairy))), 80, 44, "victory", 3, "victory", None)])
