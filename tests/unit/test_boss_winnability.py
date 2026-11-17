"""Pure parts of the boss winnability diagnostic, with fake Gym environments and searches."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from scripts.neural import boss_winnability as bw

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = json.loads((ROOT / "tests/fixtures/neural/standard-combat-1929e18.json").read_text())


class _Array(list):
    """Just enough of numpy for the policies: row/column indexing and nonzero()."""

    def nonzero(self):
        return ([index for index, value in enumerate(self) if value],)


class _Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        rows, column = key
        return [self.rows[row][column] for row in rows]


def _observation(view, verbs_by_index, mask):
    width = len(DESCRIPTOR["tensors"][[t["key"] for t in DESCRIPTOR["tensors"]].index("actions")]["columns"])
    rows = [[0] * width for _ in range(len(mask))]
    for index, verb in verbs_by_index.items():
        rows[index][view.verb] = DESCRIPTOR["vocabularies"]["verb"].index(verb)
    return {"action_mask": _Array(mask), "actions": _Matrix(rows)}


class _Errors:
    class Unsupported(Exception):
        pass

    class State(Exception):
        pass

    class Fault(Exception):
        pass


class _Branch:
    def __init__(self, search, sample):
        self.search, self.sample, self.closed = search, sample, False

    def step(self, action, view="full"):
        self.search.steps.append((self.sample, action))
        outcome = self.search.outcomes[(action, self.sample)]
        if isinstance(outcome, Exception):
            raise outcome
        return {}, outcome, True, False, {}

    def close(self):
        self.closed = True


class _Search:
    def __init__(self, outcomes):
        self.outcomes, self.steps, self.branches, self.closed = outcomes, [], [], False

    def branch(self, sample=None):
        branch = _Branch(self, sample)
        self.branches.append(branch)
        return branch

    def close(self):
        self.closed = True


class _Env:
    def __init__(self, search=None, error=None):
        self.search, self.error = search, error

    def public_search(self, seed, *, limits):
        if self.error is not None:
            raise self.error
        return self.search


class BossWinnabilityTests(unittest.TestCase):
    def setUp(self):
        self.view = bw.View(DESCRIPTOR)
        patcher = patch.object(bw, "_native_errors", return_value=(_Errors.Unsupported, _Errors.State, _Errors.Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_standard_entry_adds_only_owned_standard_fields(self):
        entry = {"id": "s-1", "encounter": "time_eater", "hp": 62, "max_hp": 98,
                 "deck": [{"card": "bash", "upgrades": 1}], "relics": ["burning_blood"],
                 "relic_counters": {}, "potions": [None, None, "block_potion"]}
        converted = bw.standard_entry(entry)
        self.assertEqual(converted["character"], "ironclad")
        self.assertEqual(converted["ascension"], 0)
        self.assertEqual(converted["deck"], [{"card": "bash", "upgrades": 1, "misc": 0}])
        for key in ("id", "encounter", "hp", "max_hp", "relics", "potions"):
            self.assertEqual(converted[key], entry[key])
        self.assertNotIn("seed", converted)

    def test_scenarios_are_bounded_per_tier_and_deterministic(self):
        recipe = ROOT / "configs/neural/curriculum-v1/recipe.json"
        first = bw.select_scenarios(recipe, shards=range(0, 2), tiers={"act3:boss", "act1:elite"},
                                    per_tier=3, seed=5)
        again = bw.select_scenarios(recipe, shards=range(0, 2), tiers={"act3:boss", "act1:elite"},
                                    per_tier=3, seed=5)
        self.assertEqual(first, again)
        self.assertEqual(Counter(case["tier"] for case in first), {"act3:boss": 3, "act1:elite": 3})
        json.dumps(first)  # plain JSON, not frozen mappings

    def test_heuristic_plays_before_ending_the_turn_and_avoids_discards(self):
        observation = _observation(self.view, {0: "end_turn", 1: "discard_potion", 2: "play_card"}, [1, 1, 1])
        picks = {self.view.heuristic_choice(observation, random.Random(seed)) for seed in range(20)}
        self.assertEqual(picks, {2})
        only_end = _observation(self.view, {0: "end_turn"}, [1, 0, 0])
        self.assertEqual(self.view.heuristic_choice(only_end, random.Random(0)), 0)

    def test_search_picks_the_best_mean_over_shared_samples_and_closes_everything(self):
        observation = _observation(self.view, {0: "end_turn", 1: "play_card"}, [1, 1])
        outcomes = {(0, 0): -1.0, (0, 1): -1.0, (1, 0): 1.0, (1, 1): -1.0}
        search = _Search(outcomes)
        stats = Counter()
        choice = bw.search_choice(_Env(search), self.view, observation, samples=2, playout_steps=5,
                                  rng=random.Random(0), research_seed=1, stats=stats)
        self.assertEqual(choice, 1)
        self.assertEqual(sorted(sample for sample, _ in search.steps), [0, 0, 1, 1])  # same worlds per action
        self.assertTrue(all(branch.closed for branch in search.branches))
        self.assertTrue(search.closed)
        self.assertEqual(stats["searched_decisions"], 1)

    def test_unavailable_root_falls_back_to_the_heuristic_and_is_counted(self):
        observation = _observation(self.view, {0: "end_turn", 1: "play_card"}, [1, 1])
        stats = Counter()
        choice = bw.search_choice(_Env(error=_Errors.Unsupported()), self.view, observation, samples=2,
                                  playout_steps=5, rng=random.Random(0), research_seed=1, stats=stats)
        self.assertEqual(choice, 1)
        self.assertEqual(stats["root_unavailable"], 1)

    def test_refused_or_faulted_branches_score_as_losses(self):
        observation = _observation(self.view, {0: "end_turn", 1: "play_card"}, [1, 1])
        outcomes = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): _Errors.Unsupported(), (1, 1): _Errors.Fault()}
        search = _Search(outcomes)
        stats = Counter()
        choice = bw.search_choice(_Env(search), self.view, observation, samples=2, playout_steps=5,
                                  rng=random.Random(0), research_seed=1, stats=stats)
        self.assertEqual(choice, 0)
        self.assertEqual((stats["branch_refused"], stats["branch_fault"]), (1, 1))
        self.assertTrue(all(branch.closed for branch in search.branches))

    def test_balance_ignores_padded_enemy_rows(self):
        width_player = len(DESCRIPTOR["tensors"][[t["key"] for t in DESCRIPTOR["tensors"]].index("player")]["columns"])
        width_enemy = len(DESCRIPTOR["tensors"][[t["key"] for t in DESCRIPTOR["tensors"]].index("enemies")]["columns"])
        player = [0] * width_player
        player[self.view.player_hp], player[self.view.player_max] = 30, 60
        enemy = [0] * width_enemy
        enemy[self.view.enemy_hp], enemy[self.view.enemy_max] = 100, 400
        observation = {"player": _NumpyRow(player), "enemies": _NumpyRows([enemy, [0] * width_enemy])}
        self.assertAlmostEqual(self.view.balance(observation), 0.5 - 0.25)

    def test_multi_step_playout_reaches_the_cap_and_scores_balance(self):
        view = self.view

        class Branch:
            def __init__(self):
                self.steps = 0

            def step(self, action, view="full"):
                self.steps += 1
                return {"marker": self.steps}, 0.0, False, False, {}

        branch = Branch()
        with patch.object(bw.View, "heuristic_choice", return_value=0), \
             patch.object(bw.View, "balance", return_value=0.4):
            value = bw._playout(branch, 1, view, 5, random.Random(0))
        self.assertEqual(branch.steps, 6)  # the searched action, then 5 playout steps
        self.assertAlmostEqual(value, 0.2)

    def test_ties_choose_only_among_the_best_actions(self):
        observation = _observation(self.view, {0: "end_turn", 1: "play_card", 2: "play_card"}, [1, 1, 1])
        outcomes = {(0, 0): -1.0, (1, 0): 1.0, (2, 0): 1.0}
        picks = set()
        for seed in range(20):
            picks.add(bw.search_choice(_Env(_Search(outcomes)), self.view, observation, samples=1,
                                       playout_steps=5, rng=random.Random(seed), research_seed=1,
                                       stats=Counter()))
        self.assertEqual(picks, {1, 2})

    def test_root_fallback_never_opens_a_search(self):
        observation = _observation(self.view, {0: "end_turn", 1: "play_card"}, [1, 1])
        env = _Env(error=_Errors.State("not in combat"))
        stats = Counter()
        bw.search_choice(env, self.view, observation, samples=2, playout_steps=5,
                         rng=random.Random(0), research_seed=1, stats=stats)
        self.assertEqual(stats["root_unavailable"], 1)
        self.assertEqual(stats["root_unavailable:State:not in combat"], 1)


class _NumpyRow(list):
    def __getitem__(self, index):
        return list.__getitem__(self, index)


class _NumpyRows:
    """Rows supporting enemies[:, column], enemies[mask, column] and clip/sum chains."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        selector, column = key
        rows = self.rows if isinstance(selector, slice) else [r for r, keep in zip(self.rows, selector) if keep]
        return _Column([row[column] for row in rows])


class _Column(list):
    def __gt__(self, other):
        return [value > other for value in self]

    def clip(self, min=None):
        return _Column([max(value, min) for value in self])

    def sum(self):
        return sum(self)


if __name__ == "__main__":
    unittest.main()
