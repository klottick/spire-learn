"""Qualify batched one-move values on real, train-split Standard combat roots."""

from contextlib import ExitStack, closing
from itertools import batched
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


CHARACTERS = ("ironclad", "silent", "defect", "watcher")
ROOTS_PER_CHARACTER = 50


def _training_batches(states, character):
    train = sorted(
        (state for state in states if state["character"] == character
         and state["entry"]["ascension"] == 20 and bw._is_train(state["run_id"])),
        key=lambda state: (state["run_id"], state["floor"]),
    )
    # Spread candidates across the corpus with room for root refusals. Gym
    # caps a catalog at 256 entries. Use the root diagnostic's 128-entry batches.
    stride = max(1, len(train) // (2 * ROOTS_PER_CHARACTER * len(CHARACTERS)))
    return batched(train[::stride], 128)


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class SearchOneMoveNativeTests(unittest.TestCase):
    def test_one_move_batched_matches_sequential_on_real_roots(self):
        from spire_lab import _native
        import numpy as np

        revision, states = bw._table_states(RECIPE)
        self.assertEqual(revision, REVISION)
        bw._check_live_revision(REVISION)
        for character_index, character in enumerate(CHARACTERS):
            with self.subTest(character=character):
                compared = 0
                index = 0
                for batch in _training_batches(states, character):
                    catalog = bw._catalog(batch, native=True)
                    with closing(bw._combat_env(catalog)) as env:
                        view = bw.View(env.observation_spec())
                        for state in batch:
                            seed = 20261117 + character_index * 1_000_000 + index
                            compared += self._compare_candidate(env, state, view, seed, index, _native, np)
                            index += 1
                            if compared == ROOTS_PER_CHARACTER:
                                break
                    if compared == ROOTS_PER_CHARACTER:
                        break
                self.assertEqual(compared, ROOTS_PER_CHARACTER)

    def _compare_candidate(self, env, state, view, seed, index, native, np):
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
                except native.RunUnsupportedBoundaryError:
                    pass
                else:
                    with closing(scalar):
                        self._compare_layouts(env, scalar, legal, view, research_seed, native, np)
                    return True
            observation, _, terminated, truncated, _ = env.step(view.heuristic_choice(observation, rng))
            if terminated or truncated:
                break
        return False

    def _compare_layouts(self, env, scalar, legal, view, research_seed, native, np):
        params = dict(samples=32, playout_steps=80, research_seed=research_seed)
        expected = sequential_values(scalar, legal, view, **params,
                                     unsupported=native.RunUnsupportedBoundaryError)
        for layout, workers in (("dense", 1), ("dense", 4), ("compact", 1), ("compact", 4)):
            options = {"observation": layout, "workers": workers}
            if layout == "compact":
                options["capacity_preset"] = "standard-combat-compact-v1"
            with closing(env.public_search(research_seed, **options)) as batched:
                self.assertEqual(scalar.spec()["root_digest"], batched.spec()["root_digest"])
                self.assert_action_alignment(scalar, batched, legal, np)
                actual = one_move_values(batched, legal, bw.View(batched.observation_spec()),
                                         **params)
                self.assertEqual(actual, expected, (research_seed, layout, workers))
                self.assertEqual([stat.mean.hex() for stat in actual.values()],
                                 [stat.mean.hex() for stat in expected.values()])

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
