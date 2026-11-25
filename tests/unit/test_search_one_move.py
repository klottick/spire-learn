"""Lock-step search tests using scripted worlds and a scalar reference."""

from collections import Counter
import random
import unittest
from unittest.mock import Mock, patch

from scripts.neural import boss_winnability as bw
from spire_learn.neural.search_one_move import ActionStat, one_move_values
from tests.unit.test_boss_winnability import _Env, _Errors, _Search, _playout


class _View:
    def legal(self, observation):
        return observation["legal"]

    def heuristic_choice(self, observation, rng):
        return rng.choice(observation["legal"])

    def balance(self, observation):
        return observation["balance"]


def sequential_values(search, legal, view, *, samples, playout_steps, research_seed, unsupported):
    """Reference order and scoring from the original one-move player."""
    values = {}
    for action in legal:
        total = 0.0
        refused = 0
        for sample in range(samples):
            branch = search.branch(sample)
            rng = random.Random(research_seed * 1_000_003 + sample)
            try:
                total += _playout(branch, action, view, playout_steps, rng)
            except unsupported:
                total -= 1.0
                refused += 1
            finally:
                branch.close()
        values[action] = ActionStat(total / samples, samples, refused)
    return values


class _ScriptedSearch(_Search):
    def branch(self, sample=None):
        branch = super().branch(sample)
        history = []
        scripted = branch.step

        def step(action, view="full"):
            if self.outcomes:
                return scripted(action, view=view)
            history.append(action)
            moves = len(history)
            balance = (sum(history) - sample) / 37
            terminal = moves == 2 + (history[0] + sample) % 6
            truncated = not terminal and sample == 1 and moves == 2
            return {"legal": [2, 3, 5], "balance": balance}, (sum(history) % 3) - 1, terminal, truncated, {}

        branch.step = step
        return branch


class OneMoveTests(unittest.TestCase):
    def setUp(self):
        self.view = _View()
        self.legal = [0, 1]
        self.params = dict(samples=3, playout_steps=4, research_seed=17)
        patcher = patch("spire_learn.neural.search_view._native_errors",
                        return_value=(_Errors.Unsupported, _Errors.State, _Errors.Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    def values(self, search, **kwargs):
        return one_move_values(search, self.legal, self.view, **self.params, **kwargs)

    def test_native_parity_candidates_are_train_only_and_fit_catalogs(self):
        from tests.integration.test_search_one_move_native import CHARACTERS, _training_batches

        states = [
            {"character": character, "run_id": split, "floor": floor,
             "entry": {"ascension": ascension}}
            for character in ("ironclad", "silent", "defect", "watcher")
            for split, ascension, count in (("train", 20, 405), ("selection", 20, 2), ("train", 0, 2))
            for floor in range(count)
        ]
        self.assertEqual(CHARACTERS, ("ironclad", "silent", "defect", "watcher"))
        with patch.object(bw, "_is_train", side_effect=lambda run: run == "train"):
            for character in CHARACTERS:
                batches = list(_training_batches(states, character))
                self.assertTrue(all(1 <= len(batch) <= 256 for batch in batches))
                expected = [state for state in states if state["character"] == character
                            and state["run_id"] == "train" and state["entry"]["ascension"] == 20]
                self.assertEqual([state for batch in batches for state in batch], expected)

    def test_batched_values_equal_sequential_values(self):
        expected = sequential_values(_ScriptedSearch({}), self.legal, self.view,
                                     **self.params, unsupported=_Errors.Unsupported)
        for workers in (1, 4):
            with self.subTest(workers=workers):
                search = _ScriptedSearch({})
                actual = self.values(search, workers=workers)
                self.assertEqual(actual, expected)
                self.assertEqual([item.mean.hex() for item in actual.values()],
                                 [item.mean.hex() for item in expected.values()])
                self.assertEqual(search.batch_sizes[0], 6)
                self.assertEqual(len(search.batch_sizes), 5)
                self.assertLess(search.batch_sizes[-1], 6)
                self.assertTrue(all(branch.closed for branch in search.branches))

    def terminal_search(self, outcome):
        outcomes = {(action, sample): 0.0 for action in self.legal for sample in range(3)}
        outcomes[1, 0] = outcome
        return _ScriptedSearch(outcomes)

    def test_branch_refusal_counts_minus_one_and_is_reported(self):
        search = self.terminal_search(_Errors.Unsupported())
        self.assertEqual(self.values(search, workers=1),
                         {0: ActionStat(0.0, 3, 0), 1: ActionStat(-1 / 3, 3, 1)})

    def test_fault_in_one_branch_raises_and_closes_all_handles(self):
        search = self.terminal_search(_Errors.Fault("scripted fault"))
        stats = Counter()
        step_many = search.step_many
        unreadable = Mock()
        unreadable.items.side_effect = AssertionError("read a zero-filled fault row")

        def batch(*args, **kwargs):
            _, *result = step_many(*args, **kwargs)
            return unreadable, *result

        search.step_many = batch
        with self.assertRaisesRegex(_Errors.Fault, "scripted fault"):
            bw.search_choice(_Env(search), self.view, {"legal": self.legal},
                             rng=random.Random(9), stats=stats, **self.params)
        self.assertEqual(stats, {"branch_fault": 1})
        self.assertEqual(len(search.branches), 6)
        self.assertTrue(all(branch.closed for branch in search.branches))
        self.assertTrue(search.closed)
        unreadable.items.assert_not_called()

    def test_whole_call_unsupported_refusal_is_attributed_per_entry(self):
        search = self.terminal_search(_Errors.Unsupported("HIDDEN_ORDER_PICK"))
        self.assertEqual(self.values(search, workers=4)[1], ActionStat(-1 / 3, 3, 1))
        self.assertEqual(search.batch_sizes, [6])
        self.assertEqual(search.steps, [(sample, action) for action in self.legal for sample in range(3)])

    def test_whole_call_input_state_or_budget_error_raises(self):
        for error_type in (_Errors.Input, _Errors.State, _Errors.Budget):
            with self.subTest(error=error_type):
                search = self.terminal_search(0.0)
                search.step_many = Mock(side_effect=error_type())
                with self.assertRaises(error_type):
                    self.values(search, workers=1)
                self.assertEqual(search.steps, [])
                self.assertTrue(all(branch.closed for branch in search.branches))

    def test_search_choice_profiles_compact_workers_without_changing_stats(self):
        search = self.terminal_search(1.0)
        search.observation_spec = Mock(return_value={"layout": "compact"})
        env = _Env(search)
        timing, stats = {}, Counter()
        with patch.object(bw, "View", return_value=self.view) as layout:
            self.assertEqual(bw.search_choice(env, self.view, {"legal": self.legal},
                                              rng=random.Random(0), stats=stats, timing=timing,
                                              branch_layout="compact", workers=4, **self.params), 1)
        layout.assert_called_once_with({"layout": "compact"})
        self.assertEqual(env.search_options, {"workers": 4, "observation": "compact",
                                              "capacity_preset": "standard-combat-compact-v1"})
        self.assertEqual(timing["step_many_calls"], 1)
        self.assertGreaterEqual(timing["step_many_wall_seconds"], 0.0)
        self.assertGreaterEqual(timing["step_many_cpu_seconds"], 0.0)
        self.assertEqual(stats, {"searched_decisions": 1})

    def test_search_choice_unchanged_after_refactor(self):
        for outcome in (1.0, 0.0, _Errors.Unsupported()):
            expected = sequential_values(self.terminal_search(outcome), self.legal, self.view,
                                         **self.params, unsupported=_Errors.Unsupported)
            best = max(stat.mean for stat in expected.values())
            ties = [action for action, stat in expected.items() if stat.mean == best]
            expected_stats = Counter(searched_decisions=1)
            refused = sum(stat.refused for stat in expected.values())
            if refused:
                expected_stats["branch_refused"] = refused
            stats = Counter()
            self.assertEqual(bw.search_choice(_Env(self.terminal_search(outcome)), self.view,
                                              {"legal": self.legal}, rng=random.Random(9), stats=stats,
                                              **self.params), random.Random(9).choice(ties))
            self.assertEqual(stats, expected_stats)
