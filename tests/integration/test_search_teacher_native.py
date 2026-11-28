"""Gated qualification of the tree teacher on real Standard training roots."""

from contextlib import closing
from dataclasses import replace
import math
import os
from pathlib import Path
import unittest

from scripts.neural import boss_winnability as bw
from scripts.neural import teacher_descent_cost_entry as benchmark
from spire_learn.neural.gym_contracts import GYM_V1_2_4_CONTRACT
from spire_learn.neural.search_one_move import one_move_values
from spire_learn.neural.search_teacher import TeacherConfig, search_options, teach

# The descent-cost entry still pins its v1.2.0 table; this qualification runs on the current epoch.
RECIPE = Path(__file__).resolve().parents[2] / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class SearchTeacherNativeTests(unittest.TestCase):
    def test_tree_and_uniform_reference_on_twenty_training_roots(self):
        from spire_lab import _native

        revision, states = bw._table_states(RECIPE)
        self.assertEqual(revision, GYM_V1_2_4_CONTRACT.source_revision)
        bw._check_live_revision(revision)
        candidates = benchmark.training_roots(states)
        compared = 0
        with closing(bw._combat_env(bw._catalog(candidates, native=True))) as env:
            live_view = bw.View(env.observation_spec())
            for index, case in enumerate(candidates):
                seed = benchmark.SEED + index
                entry = case["entry"]
                observation, _ = env.reset(seed=seed, options={"entry_id": entry["id"]})
                legal = live_view.legal(observation)
                if not legal:
                    continue
                config = TeacherConfig(simulations=2 * len(legal) + 8, min_root_visits=2,
                                       worlds=2, ucb_c=1.0, max_depth=3,
                                       widening=None, playout_steps=4)
                research_seed = seed * 1009
                try:
                    arena = env.public_search(research_seed, **search_options(config))
                except _native.RunUnsupportedBoundaryError:
                    continue
                with closing(arena):
                    view = bw.View(arena.observation_spec())
                    result = teach(arena, legal, view, config, research_seed=research_seed)
                    self.assert_stats(result, legal, config)
                    self.assert_branches_closed(arena)
                self.assert_arena_closed(arena, _native)

                uniform = replace(config, max_depth=1, simulations=config.worlds * len(legal))
                with closing(env.public_search(research_seed, **search_options(uniform))) as arena:
                    actual = teach(arena, legal, view, uniform, research_seed=research_seed)
                    self.assert_stats(actual, legal, uniform)
                    self.assert_branches_closed(arena)
                self.assert_arena_closed(arena, _native)

                options = search_options(uniform)
                options["limits"]["max_live_branches"] = uniform.simulations
                with closing(env.public_search(research_seed, **options)) as arena:
                    expected = one_move_values(arena, legal, view, samples=uniform.worlds,
                                               playout_steps=uniform.playout_steps,
                                               research_seed=research_seed)
                    self.assert_branches_closed(arena)
                self.assert_arena_closed(arena, _native)

                self.assertEqual(actual.stats, expected)
                self.assertEqual([stat.mean.hex() for stat in actual.stats.values()],
                                 [expected[action].mean.hex() for action in actual.stats])
                compared += 1
                if compared == 20:
                    break
        self.assertEqual(compared, 20)

    def assert_stats(self, result, legal, config):
        self.assertEqual(set(result.stats), set(legal))
        self.assertEqual(result.simulations, config.simulations)
        self.assertEqual(sum(stat.visits for stat in result.stats.values()), config.simulations)
        self.assertEqual(sum(stat.refused for stat in result.stats.values()), result.refused)
        self.assertTrue(math.isfinite(result.cpu_seconds) and result.cpu_seconds >= 0)
        for stat in result.stats.values():
            self.assertGreaterEqual(stat.visits, config.min_root_visits)
            self.assertTrue(-1 <= stat.mean <= 1)
            self.assertTrue(0 <= stat.refused <= stat.visits)
        self.assertEqual(result.choice, max(sorted(legal), key=lambda action: result.stats[action].mean))

    def assert_branches_closed(self, arena):
        counts = arena.counters()
        self.assertEqual(counts["live_branches"], 0)
        self.assertEqual(counts["branches_created"], counts["branches_closed"])

    def assert_arena_closed(self, arena, native):
        with self.assertRaises(native.RunStateError):
            arena.counters()
