import os
import unittest
from unittest.mock import Mock, patch

from scripts.neural import search_teacher_entry as entry
from scripts.neural.boss_winnability import _is_train
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
    return os.getpid(), job


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
                                           "branch_refusals"})

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
        self.assertEqual(search.call_args.kwargs["research_seed"], row["research_seed"])
        self.assertEqual(fight.public_search.call_args.kwargs["workers"], 1)
        self.assertEqual(fight.public_search.call_args.kwargs["capacity_preset"],
                         "standard-combat-compact-v1")
        arena.close.assert_called_once()

    def test_uncalibrated_tree_is_allowed_only_for_training_sample(self):
        configs = entry._ROOT / "configs/neural/search-teacher-gym-1.2.0/teachers.json"
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            entry._teacher(configs, "tree-1x", "selection")
        teacher, digest, calibrated = entry._teacher(configs, "tree-1x", None)
        self.assertFalse(calibrated)
        self.assertEqual(digest, entry.config_sha256(teacher))

    def test_parallel_workers_emit_results_in_case_order(self):
        jobs = list(range(8))
        with patch.object(entry, "_run_case", process_job):
            serial = list(entry._results(jobs, 1))
            parallel = list(entry._results(jobs, 2))
        self.assertTrue(all(pid == os.getpid() for pid, _ in serial))
        self.assertTrue(all(pid != os.getpid() for pid, _ in parallel))
        self.assertEqual([job for _, job in parallel], jobs)
