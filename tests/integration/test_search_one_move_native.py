"""Qualify batched one-move values on real, train-split Standard combat roots."""

from contextlib import ExitStack
import os
from pathlib import Path
import random
import unittest

from scripts.neural import boss_winnability as bw
from spire_learn.neural.search_one_move import one_move_values
from tests.unit.test_search_one_move import sequential_values

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/neural/baalorlord-standard-gym-1.2.0/recipe.json"
REVISION = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class SearchOneMoveNativeTests(unittest.TestCase):
    def test_one_move_batched_matches_sequential_on_real_roots(self):
        from spire_lab import _native
        import numpy as np

        revision, states = bw._table_states(RECIPE)
        self.assertEqual(revision, REVISION)
        bw._check_live_revision(REVISION)
        train = sorted(
            (state for state in states if state["character"] == "ironclad"
             and state["entry"]["ascension"] == 20 and bw._is_train(state["run_id"])),
            key=lambda state: (state["run_id"], state["floor"]),
        )
        roots, samples, playout_steps = 200, 32, 80
        self.assertGreaterEqual(len(train), roots)
        # Spread cases across the corpus and examine both opening and later decisions.
        candidates = train[::max(1, len(train) // (2 * roots))]
        catalog = bw._catalog(candidates, native=True)
        env = bw._combat_env(catalog)
        self.addCleanup(env.close)
        view = bw.View(env.observation_spec())
        compared = 0
        for index, state in enumerate(candidates):
            seed = 20261117 + index
            rng = random.Random(seed)
            observation, _ = env.reset(seed=seed, options={"entry_id": state["entry"]["id"]})
            for step in range(24):
                legal = view.legal(observation)
                if not legal:
                    break
                if step >= index % 12 and len(legal) > 1:
                    research_seed = seed * 1009 + step
                    try:
                        scalar = env.public_search(research_seed)
                    except _native.RunUnsupportedBoundaryError:
                        pass
                    else:
                        with ExitStack() as stack:
                            stack.callback(scalar.close)
                            params = dict(samples=samples, playout_steps=playout_steps,
                                          research_seed=research_seed)
                            expected = sequential_values(scalar, legal, view, **params,
                                                         unsupported=_native.RunUnsupportedBoundaryError)
                            for layout, workers in (("dense", 1), ("dense", 4), ("compact", 1), ("compact", 4)):
                                options = {"observation": layout, "workers": workers}
                                if layout == "compact":
                                    options["capacity_preset"] = "standard-combat-compact-v1"
                                batched = env.public_search(research_seed, **options)
                                try:
                                    self.assertEqual(scalar.spec()["root_digest"], batched.spec()["root_digest"])
                                    self.assert_action_alignment(scalar, batched, legal, np)
                                    actual = one_move_values(batched, legal, bw.View(batched.observation_spec()),
                                                             **params, workers=workers)
                                    self.assertEqual(actual, expected, (index, step, layout, workers))
                                    self.assertEqual([stat.mean.hex() for stat in actual.values()],
                                                     [stat.mean.hex() for stat in expected.values()])
                                finally:
                                    batched.close()
                        compared += 1
                        break
                observation, _, terminated, truncated, _ = env.step(view.heuristic_choice(observation, rng))
                if terminated or truncated:
                    break
            if compared == roots:
                break
        self.assertEqual(compared, roots)

    def assert_action_alignment(self, scalar, batched, legal, np):
        with ExitStack() as stack:
            branches = []
            for search in (scalar, batched):
                branch = search.branch(0)
                stack.callback(branch.close)
                branches.append(branch)
            expected, actual = (branch.decision()[0] for branch in branches)
            self.assertEqual(legal, actual["action_mask"].nonzero()[0].tolist())
            self.assertEqual(expected.keys(), actual.keys())
            for name, dense in expected.items():
                compact = actual[name]
                if compact.shape == dense.shape:
                    np.testing.assert_array_equal(compact, dense, err_msg=name)
                else:
                    self.assertEqual(dense.ndim, compact.ndim, name)
                    self.assertEqual(dense.shape[1:], compact.shape[1:], name)
                    self.assertLess(compact.shape[0], dense.shape[0], name)
                    np.testing.assert_array_equal(compact, dense[:len(compact)], err_msg=name)
                    np.testing.assert_array_equal(dense[len(compact):], 0, err_msg=name)
