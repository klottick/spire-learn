"""Scripted public worlds for the tree teacher."""
from array import array
from collections import OrderedDict
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

    def test_repeat_playouts_in_one_world_vary_and_remain_deterministic(self):
        values = []

        def script(world, path):
            if len(path) == 1:
                return observation(1), 0.0, False
            if len(path) == 2:
                return observation(2, (1,) * 32), 0.0, False
            value = path[-1] / 32
            values.append(value)
            return observation(9, (0,)), value, True

        cfg = config(simulations=6, worlds=1, playout_steps=1)
        with patch.object(teacher, "process_time", return_value=0.0):
            first = self.teach(Search(script), cfg)
            first_values = list(values)
            values.clear()
            second = self.teach(Search(script), cfg)
        self.assertEqual(first, second)
        self.assertEqual(values, first_values)
        self.assertEqual(len(values), 5)
        self.assertGreater(len(set(values)), 1)
        self.assertAlmostEqual(first.stats[0].mean, (0.1 + sum(values)) / 6)

    def test_nonterminal_child_with_empty_legal_mask_raises_contract_error(self):
        search = Search(lambda world, path: (observation(1, (0,)), 0.0, False))
        with self.assertRaisesRegex(RuntimeError, "nonterminal.*no legal actions"):
            self.teach(search, config())
        self.assertEqual(search.counters()["live_branches"], 0)

    def test_progressive_widening_admits_sorted_actions_by_visit_count(self):
        cfg = config(widening=(0.5, 0.5))
        node = teacher._Node((2, 4, 6, 8))
        for visits, admitted in ((0, (2,)), (4, (2,)), (5, (2, 4)),
                                 (16, (2, 4)), (17, (2, 4, 6)), (37, (2, 4, 6, 8))):
            with self.subTest(visits=visits):
                node.visits = visits
                selected = []
                for stats in node.actions.values():
                    stats.visits = 0
                for _ in admitted:
                    action = teacher._select(node, cfg)
                    selected.append(action)
                    node.actions[action].visits += 1
                self.assertEqual(tuple(selected), admitted)
                self.assertIn(teacher._select(node, cfg), admitted)

    def test_retained_fork_points_cap_at_127_in_a_200_slot_arena(self):
        class MeasuredPoints(OrderedDict):
            peak = 0

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                self.peak = max(self.peak, len(self))

        def script(world, path):
            if len(path) == 1:
                return observation(1), 0.0, False
            return observation(9, (0,)), float(world % 2), True

        cfg = config(simulations=260, worlds=200)
        capped_points, uncapped_points = MeasuredPoints(), MeasuredPoints()
        with patch.object(teacher, "OrderedDict", return_value=capped_points):
            capped = self.teach(Search(script, live_limit=200), cfg)
        with (patch.object(teacher, "OrderedDict", return_value=uncapped_points),
              patch.object(teacher, "_MAX_LIVE_BRANCHES", 200)):
            uncapped = self.teach(Search(script, live_limit=200), cfg)
        self.assertEqual(capped_points.peak, 127)
        self.assertEqual(uncapped_points.peak, 199)
        self.assertEqual(capped.stats, uncapped.stats)

    def test_arena_options_follow_config(self):
        cfg = config()
        options = teacher.search_options(cfg)
        self.assertEqual(options["observation"], "compact")
        self.assertEqual(options["capacity_preset"], "standard-combat-compact-v1")
        self.assertEqual(options["workers"], 1)
        self.assertEqual(options["limits"]["max_total_steps"], 18)
        self.assertNotEqual(teacher.config_sha256(cfg), teacher.config_sha256(config(workers=2)))


class PerActionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.view = WorldView()
        patcher = patch.object(teacher.search_view, "_native_errors", return_value=(Unsupported, ValueError, Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def per_action(**overrides):
        return config(**{"simulations": None, "simulations_per_action": 4, **overrides})

    def test_simulations_equal_k_times_legal(self):
        for width in (2, 5):
            search = Search(finish, width)
            result = teacher.teach(search, self.view.legal(search.root), self.view, self.per_action(),
                                   research_seed=17)
            self.assertEqual(result.simulations, 4 * width)
            self.assertEqual(sum(stat.visits for stat in result.stats.values()), 4 * width)

    def test_root_quota_holds_at_k8_with_wide_decision(self):
        search = Search(finish, 20)
        cfg = self.per_action(min_root_visits=8, simulations_per_action=8)
        result = teacher.teach(search, self.view.legal(search.root), self.view, cfg, research_seed=3)
        self.assertEqual([stat.visits for stat in result.stats.values()], [8] * 20)
        self.assertEqual(result.simulations, 160)

    def test_absent_field_keeps_hashes_and_identity(self):
        cfg = teacher.TeacherConfig(
            simulations=1237, min_root_visits=8, worlds=8, ucb_c=1.0, max_depth=4, widening=(2.0, 0.5),
            playout_steps=80, branch_layout="compact", workers=1)
        self.assertEqual(teacher.config_sha256(cfg),
                         "16fc490aee64513ecc8937ac3b1511ac024fe0fdf0cd3f6b20533ce7add57de6")
        self.assertNotIn("simulations_per_action", teacher.config_identity(cfg))
        identity = teacher.config_identity(self.per_action())
        self.assertEqual(identity["simulations_per_action"], 4)
        self.assertNotIn("simulations", identity)
        self.assertNotEqual(teacher.config_sha256(self.per_action()),
                            teacher.config_sha256(self.per_action(simulations_per_action=5)))

    def test_invalid_budget_fields_are_refused(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            config(simulations_per_action=4)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            config(simulations=None)
        with self.assertRaisesRegex(ValueError, "root visits"):
            self.per_action(min_root_visits=8)
        with self.assertRaises(ValueError):
            self.per_action(simulations_per_action=0)

    def test_search_options_size_limits_from_decision_simulations(self):
        cfg = self.per_action()
        with self.assertRaises(ValueError):
            teacher.search_options(cfg)
        fixed = teacher.search_options(config(simulations=12))
        self.assertEqual(teacher.search_options(cfg, 3), fixed)
        self.assertGreater(teacher.search_options(cfg, 6)["limits"]["max_total_steps"],
                           fixed["limits"]["max_total_steps"])
