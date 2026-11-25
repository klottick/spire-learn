import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.neural import search_teacher_entry as entry
from scripts.neural.boss_winnability import _is_train
from spire_learn.neural.gym_contracts import GYM_V1_2_0_CONTRACT
from spire_learn.neural.search_one_move import ActionStat
from spire_learn.neural.standard_stream import run_split


class Unsupported(Exception):
    pass


class StateError(Exception):
    pass


class Fault(Exception):
    pass


CASE = {"case_id": "fixture", "entry": {"id": "fixture"}, "environment_seed": 42}
CURRENT = {"samples": 32, "playout_steps": 80}


class Fight:
    def __init__(self, *, legal=(0,), turn_stride=1, terminal=False, finish=None):
        self.legal = list(legal)
        self.turn_stride = turn_stride
        self.finish = finish
        self.terminal = terminal
        self.actions = []
        self.public_search = Mock(side_effect=Unsupported("runic_dome"))

    def observation(self):
        return {"player": [len(self.actions) // self.turn_stride], "player_known": [1],
                "hidden_rng": "private"}

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
    view = Mock()
    view.legal.return_value = [0]
    fight, decisions = entry.play_fight(
        Fight(finish=job + 1), case, view, 0, CURRENT, research_seed=job,
    )
    return os.getpid(), fight, decisions


class TeacherEntryTests(unittest.TestCase):
    def play(self, fight, **kwargs):
        view = Mock()
        view.legal.return_value = fight.legal
        view.heuristic_choice.return_value = fight.legal[-1] if fight.legal else None
        with patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)):
            return entry.play_fight(fight, CASE, view, 0, CURRENT, research_seed=11, **kwargs)

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
                                       "hp_after_cleanup", "cpu_seconds", "wall_seconds"})
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

        reset = {"player": [1], "player_known": [1], "action_mask": [1, 0]}
        successor = {"player": [2], "player_known": [1], "action_mask": [1, 1]}
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
        view = SimpleNamespace(legal=lambda row: [
            index for index, legal in enumerate(row["action_mask"]) if legal
        ])
        with (patch.object(entry.search_view, "_native_errors", return_value=(Unsupported, StateError, Fault)),
              self.assertRaisesRegex(RuntimeError, message)):
            with entry._fight_env(CASE, "train") as (fight, _):
                entry.play_fight(fight, CASE, view, 0, CURRENT, research_seed=11)
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
