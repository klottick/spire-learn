"""Scripted public worlds for the tree teacher."""
from array import array
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from spire_learn.neural import search_teacher as teacher
from spire_learn.neural.search_one_move import ActionStat, one_move_values


class Unsupported(Exception):
    pass


class Fault(Exception):
    pass


class Budget(Exception):
    pass


class WorldView:
    def legal(self, row):
        return [action for action, allowed in enumerate(row["action_mask"]) if allowed]

    def heuristic_choice(self, row, rng):
        return rng.choice(self.legal(row))

    def balance(self, row):
        return row["public"][0] / 10


def observation(value, mask=(1,)):
    return {"public": array("i", [value]), "action_mask": array("B", mask)}


class Branch:
    def __init__(self, search, world, path):
        self.search, self.world, self.path = search, world, path
        self.closed = False

    def decision(self):
        return self.search.root, {}

    def step(self, action, *, view):
        assert not self.closed and view == "full"
        self.path += (action,)
        row, reward, done = self.search.script(self.world, self.path)
        return row, reward, done, False, {}

    def fork(self):
        assert not self.closed
        self.search.forks += 1
        return self.search.open(self.world, self.path)

    def close(self):
        assert not self.closed
        self.closed = True


class Search:
    def __init__(self, script, actions=1, live_limit=64):
        self.script = script
        self.root = observation(0, (1,) * actions)
        self.handles = []
        self.worlds = []
        self.forks = 0
        self.live_limit = live_limit

    def __getattr__(self, name):
        if name in {"checkpoint", "restore", "fork_session", "seed", "record", "snapshot"}:
            raise AssertionError(name)
        raise AttributeError(name)

    def limits(self):
        return {"max_live_branches": self.live_limit}

    def counters(self):
        return {"live_branches": sum(not branch.closed for branch in self.handles)}

    def open(self, world, path):
        if self.counters()["live_branches"] >= self.live_limit:
            raise Budget("live branches")
        branch = Branch(self, world, path)
        self.handles.append(branch)
        return branch

    def branch(self, world):
        self.worlds.append(world)
        return self.open(world, ())

    def step_many(self, branches, actions, *, view):
        raise Unsupported("fake batch preflight refusal")


def config(**overrides):
    return teacher.TeacherConfig(**{
        "simulations": 8, "worlds": 2, "min_root_visits": 1, "ucb_c": 1.0,
        "max_depth": 2, "widening": None, "playout_steps": 0, **overrides,
    })


def finish(world, path):
    return observation(9, (0,)), (path[0] - 1) * 0.5, True


class SearchTeacherTests(unittest.TestCase):
    def setUp(self):
        self.view = WorldView()
        patcher = patch.object(teacher.search_view, "_native_errors", return_value=(Unsupported, ValueError, Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    def teach(self, search, cfg, research_seed=17):
        result = teacher.teach(search, self.view.legal(search.root), self.view, cfg,
                               research_seed=research_seed)
        self.assertEqual(search.counters()["live_branches"], 0)
        return result

    def child_visits(self, same):
        def script(world, path):
            if len(path) == 1:
                return observation(1 if same else 1 + world), 0.0, False
            return observation(9, (0,)), float(world), True

        search = Search(script)
        node_class = teacher._Node
        allocated = []

        def make_node(*args):
            node = node_class(*args)
            allocated.append(node)
            return node

        with patch.object(teacher, "_Node", side_effect=make_node):
            self.teach(search, config())
        self.assertGreater(search.forks, 0)
        return sorted(next(iter(child.actions.values())).visits for child in allocated[0].children.values())

    def test_same_public_view_from_two_worlds_shares_a_node(self):
        self.assertEqual(self.child_visits(same=True), [7])

    def test_different_public_views_never_share_a_node(self):
        self.assertEqual(self.child_visits(same=False), [3, 3])

    def test_return_to_root_shares_node_and_counts_one_root_choice(self):
        def script(world, path):
            if len(path) == 1:
                return observation(0), 0.0, False
            return observation(9, (0,)), 1.0, True

        result = self.teach(Search(script), config())
        self.assertEqual(result.stats[0], ActionStat(1.0, 8, 0))

    def test_every_root_action_gets_min_visits(self):
        result = self.teach(Search(finish, 3), config(simulations=40, min_root_visits=8))
        self.assertTrue(all(stat.visits >= 8 for stat in result.stats.values()))
        self.assertEqual(sum(stat.visits for stat in result.stats.values()), 40)

    def test_choice_is_highest_mean_and_deterministic(self):
        cfg = config()
        results = [self.teach(Search(finish, 3), cfg, s) for s in (17, 17, 19)]
        self.assertEqual([result.choice for result in results], [2, 2, 2])
        self.assertEqual(results[0].stats, results[1].stats)
        self.assertEqual(results[0].stats.keys(), results[2].stats.keys())
        tied = self.teach(Search(lambda w, p: (observation(9, (0,)), 0, True), 2), cfg)
        self.assertEqual(tied.choice, 0)

    def test_branch_refusal_counts_against_that_simulation(self):
        def script(world, path):
            if world == 1:
                raise Unsupported("hidden order")
            return finish(world, path)
        result = self.teach(Search(script), config())
        self.assertEqual(result.refused, 4)
        self.assertEqual(result.stats[0], ActionStat(-0.75, 8, 4))

    def test_fault_aborts_search_and_closes_handles(self):
        terminal_steps = 0

        def script(world, path):
            nonlocal terminal_steps
            if len(path) == 1:
                return observation(1), 0, False
            terminal_steps += 1
            if terminal_steps == 5:
                raise Fault("simulation five")
            return finish(world, path)

        search = Search(script)
        with self.assertRaisesRegex(Fault, "simulation five"):
            self.teach(search, config(playout_steps=1))
        self.assertGreater(search.forks, 0)
        self.assertEqual(search.counters()["live_branches"], 0)

    def test_budget_error_raises(self):
        def script(world, path):
            raise Budget("steps")
        search = Search(script)
        with self.assertRaisesRegex(Budget, "steps"):
            self.teach(search, config())
        self.assertEqual(search.counters()["live_branches"], 0)

    def test_no_privileged_surface_is_called(self):
        self.teach(Search(finish), config())
        source = Path(teacher.__file__).read_text()
        self.assertIsNone(re.search(r"\b(checkpoint|restore|fork_session|seed|record|snapshot)\b", source))

    def test_teacher_is_one_move_search_when_depth_1_and_uniform(self):
        def script(world, path):
            if len(path) == 1:
                return observation(1, (1, 1, 1)), 0, False
            if world == 1 and path[0] == 2:
                raise Unsupported("hidden order")
            return observation(9, (0,)), (world + path[0] + path[1]) / 7, True

        legal = (0, 1, 2)
        search = Search(script, 3)
        cfg = config(max_depth=1, simulations=12, worlds=4, playout_steps=2)
        actual = self.teach(search, cfg)
        expected = one_move_values(Search(script, 3), legal, self.view,
                                   samples=4, playout_steps=2, research_seed=17)
        self.assertEqual(actual.stats, expected)
        self.assertEqual(search.worlds, [0, 1, 2, 3] * 3)

    def test_tight_live_budget_evicts_points_and_preserves_results(self):
        def script(world, path):
            if len(path) < 3:
                return observation(len(path), (1, 1)), 0, False
            return finish(world, path)

        cfg = config(max_depth=3, simulations=24, worlds=3)
        wide = self.teach(Search(script, 2), cfg)
        for limit in (1, 2):
            with self.subTest(limit=limit):
                tight = self.teach(Search(script, 2, live_limit=limit), cfg)
                self.assertEqual(tight.stats, wide.stats)

    def test_arena_options_follow_config(self):
        cfg = config()
        options = teacher.search_options(cfg)
        self.assertEqual(options["observation"], "compact")
        self.assertEqual(options["capacity_preset"], "standard-combat-compact-v1")
        self.assertEqual(options["workers"], 1)
        self.assertEqual(options["limits"]["max_total_steps"], 18)
        self.assertNotEqual(teacher.config_sha256(cfg), teacher.config_sha256(config(workers=2)))
