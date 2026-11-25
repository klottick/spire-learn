"""Gated check of native playouts against stepping, on real Standard training roots.

Entries come from the v1.2.0 run-states table: catalog entries are independent of the
Gym release, so the table's revision is not checked. The live runtime must be Gym v1.2.4.
"""

from contextlib import closing
import math
import os
import unittest
from unittest.mock import patch

from scripts.neural import boss_winnability as bw
from scripts.neural import teacher_descent_cost_entry as benchmark
from spire_learn.neural.gym_contracts import GYM_V1_2_4_CONTRACT
from spire_learn.neural.search_one_move import PLAYOUT_EXCLUDE_VERBS, one_move_values
from spire_learn.neural.search_teacher import TeacherConfig, search_options, teach
from spire_learn.neural.search_view import summary_balance

ROOTS = 20
STEPS = 40
TERMINAL = {"victory", "death", "escaped"}


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class NativePlayoutTests(unittest.TestCase):
    def test_playout_many_equals_stepping_and_one_move_is_well_formed(self):
        from spire_lab import _native

        _, states = bw._table_states(benchmark.RECIPE)
        bw._check_live_revision(GYM_V1_2_4_CONTRACT.source_revision)
        candidates = benchmark.training_roots(states)
        config = TeacherConfig(simulations=20, worlds=2, ucb_c=1.0, max_depth=2, widening=None,
                               playout_steps=STEPS, playout_backend="native")
        compared = 0
        self.pairs = self.reached = 0
        with closing(bw._combat_env(bw._catalog(candidates, native=True))) as env:
            for index, case in enumerate(candidates):
                seed = benchmark.SEED + index
                observation, _ = env.reset(seed=seed, options={"entry_id": case["entry"]["id"]})
                view = bw.View(env.observation_spec())
                legal = view.legal(observation)
                if len(legal) < 2:
                    continue
                research_seed = seed * 1009
                try:
                    arena = env.public_search(research_seed, **search_options(config))
                except _native.RunUnsupportedBoundaryError:
                    continue
                with closing(arena):
                    arena_view = bw.View(arena.observation_spec())
                    before = self.reached
                    self.assert_playout_equals_stepping(arena, arena_view, legal[0], _native)
                    self.assertGreaterEqual(self.reached - before, 1, "no playout compared on this root")
                    self.assert_closed(arena)
                self.assert_one_move(env, legal, research_seed, _native)
                compared += 1
                if compared == ROOTS:
                    break
        self.assertEqual(compared, ROOTS)
        self.assertGreaterEqual(2 * self.reached, self.pairs)

    def test_native_tree_teacher_on_real_roots(self):
        from spire_lab import _native
        from spire_lab.gymnasium.public_search import PublicBranch

        _, states = bw._table_states(benchmark.RECIPE)
        bw._check_live_revision(GYM_V1_2_4_CONTRACT.source_revision)
        candidates = benchmark.training_roots(states)
        compared = 0
        real_fork = PublicBranch.fork
        calls = {"fork": 0, "playout": 0}

        def spy_fork(branch):
            calls["fork"] += 1
            return real_fork(branch)

        with closing(bw._combat_env(bw._catalog(candidates, native=True))) as env, \
                patch.object(PublicBranch, "fork", spy_fork):
            for index, case in enumerate(candidates):
                seed = benchmark.SEED + index
                observation, _ = env.reset(seed=seed, options={"entry_id": case["entry"]["id"]})
                view = bw.View(env.observation_spec())
                legal = view.legal(observation)
                if len(legal) < 2:
                    continue
                config = TeacherConfig(simulations=3 * len(legal) + 8, min_root_visits=2, worlds=4,
                                       ucb_c=1.0, max_depth=3, widening=None, playout_steps=40,
                                       branch_layout="compact", playout_backend="native")
                research_seed = seed * 1009
                results = []
                try:
                    for _ in range(2):
                        arena = env.public_search(research_seed, **search_options(config))
                        real_playout = arena.playout_many

                        def spy_playout(*args, _real=real_playout, **kwargs):
                            calls["playout"] += 1
                            return _real(*args, **kwargs)

                        arena.playout_many = spy_playout
                        with closing(arena):
                            result = teach(arena, legal, bw.View(arena.observation_spec()), config,
                                           research_seed=research_seed)
                            counts, limits = arena.counters(), arena.limits()
                            self.assertEqual(counts["live_branches"], 0)
                            self.assertEqual(counts["branches_created"], counts["branches_closed"])
                            self.assertLessEqual(counts["branches_created"], limits["max_branches"])
                            self.assertLessEqual(counts["total_steps"], limits["max_total_steps"])
                        with self.assertRaises(_native.RunStateError):
                            arena.counters()
                        results.append(result)
                except _native.RunUnsupportedBoundaryError:
                    continue
                first, second = results
                self.assertEqual(first.stats, second.stats)
                self.assertEqual(first.choice, second.choice)
                self.assertEqual(first.refused, second.refused)
                self.assertEqual(set(first.stats), set(legal))
                self.assertEqual(sum(stat.visits for stat in first.stats.values()), config.simulations)
                for stat in first.stats.values():
                    self.assertGreaterEqual(stat.visits, config.min_root_visits)
                    self.assertTrue(-1 <= stat.mean <= 1)
                    self.assertTrue(0 <= stat.refused <= stat.visits)
                compared += 1
                if compared == 10:
                    break
        self.assertEqual(compared, 10)
        self.assertGreater(calls["fork"], 0)
        self.assertGreater(calls["playout"], 0)

    def assert_playout_equals_stepping(self, arena, view, action, native):
        options = {"rule": "uniform_exclude_verbs", "exclude_verbs": PLAYOUT_EXCLUDE_VERBS}
        for sample in (0, 1):
            for playout_seed in (0, 7):
                self.pairs += 1
                with self.subTest(sample=sample, playout_seed=playout_seed):
                    whole, stepped = arena.branch(sample), arena.branch(sample)
                    try:
                        refused_root = False
                        for branch in (whole, stepped):
                            try:
                                _, reward, done, cut, _ = branch.step(action, view="mask")
                            except native.RunUnsupportedBoundaryError:
                                refused_root = True
                        if refused_root or done or cut:
                            continue
                        self.reached += 1
                        result = arena.playout_many([whole], max_steps=STEPS, playout_seed=playout_seed, **options)
                        self.assertIsNone(result["fault"][0])
                        self.assertEqual(result["rule_id"], "spire-gym-public-playout-v1")
                        steps, reason, reward = self.step_playout(stepped, playout_seed, options, native)
                        if reason == "terminal":
                            self.assertIn(result["end_reason"][0], TERMINAL)
                        else:
                            self.assertEqual(result["end_reason"][0], reason)
                        self.assertEqual(int(result["steps"][0]), steps)
                        self.assertEqual(bool(result["terminated"][0]), reason == "terminal")
                        self.assertEqual(bool(result["truncated"][0]), reason == "step_limit")
                        if reason == "terminal":
                            self.assertEqual(int(result["rewards"][0]), reward)
                        else:
                            self.assertEqual(int(result["rewards"][0]), 0)
                            observation, _ = stepped.decision()
                            self.assertEqual(summary_balance(result, 0), view.balance(observation))
                            self.assertEqual(int(result["player_hp"][0]), int(observation["player"][view.player_hp]))
                    finally:
                        whole.close()
                        stepped.close()

    @staticmethod
    def step_playout(branch, playout_seed, options, native):
        steps = 0
        while steps < STEPS:
            try:
                pick = branch.playout_action(playout_seed=playout_seed, **options)
            except native.RunUnsupportedBoundaryError:
                return steps, "refused", 0
            _, reward, done, cut, _ = branch.step(pick, view="mask")
            steps += 1
            if done:
                return steps, "terminal", int(reward)
            if cut:
                return steps, "step_limit", 0
        return steps, "max_steps", 0

    def assert_one_move(self, env, legal, research_seed, native):
        samples, steps = 4, 20
        width, depth = samples * len(legal), steps + 2
        options = search_options(TeacherConfig(simulations=1, worlds=1, ucb_c=0.0, max_depth=1, widening=None))
        options["limits"] = {"max_live_branches": width + 8, "max_branches": width + 8,
                             "max_steps_per_branch": depth, "max_total_steps": width * depth}
        with closing(env.public_search(research_seed, **options)) as arena:
            stats = one_move_values(arena, legal, bw.View(arena.observation_spec()), samples=samples,
                                    playout_steps=steps, research_seed=research_seed,
                                    playout_backend="native")
            self.assertEqual(set(stats), set(legal))
            for stat in stats.values():
                self.assertEqual(stat.visits, samples)
                self.assertTrue(math.isfinite(stat.mean) and -1 <= stat.mean <= 1)
                self.assertTrue(0 <= stat.refused <= samples)
            self.assert_closed(arena)

    def assert_closed(self, arena):
        counts = arena.counters()
        self.assertEqual(counts["live_branches"], 0)
        self.assertEqual(counts["branches_created"], counts["branches_closed"])
