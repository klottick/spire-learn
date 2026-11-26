"""Playout policies: v1 heuristic parity and the no-potions rule."""

import random
import unittest

from spire_learn.neural.search_view import PLAYOUT_POLICIES, View
from tests.unit._search_fakes import descriptor, observation

POLICY = 'no-potions-v2'


class _Rng:
    """Counts draws; picks the last option."""

    def __init__(self):
        self.calls = 0

    def choice(self, options):
        self.calls += 1
        return options[-1]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.view = View(descriptor())

    def test_policies_and_unknown_policy(self):
        self.assertEqual(PLAYOUT_POLICIES, ('uniform-v1', 'no-potions-v2'))
        with self.assertRaises(ValueError):
            self.view.playout_choice(observation(verbs=['end_turn']), random.Random(0), 'other')

    def test_uniform_v1_is_heuristic_choice(self):
        obs = observation(verbs=['play_card', 'use_potion', 'end_turn', 'discard_potion', 'play_card'])
        a, b = random.Random(7), random.Random(7)
        for _ in range(200):
            self.assertEqual(self.view.playout_choice(obs, a, 'uniform-v1'), self.view.heuristic_choice(obs, b))

    def test_policy_never_picks_potions_when_cards_playable(self):
        obs = observation(verbs=['use_potion', 'play_card', 'end_turn', 'discard_potion', 'play_card'])
        rng = random.Random(3)
        picks = {self.view.playout_choice(obs, rng, POLICY) for _ in range(300)}
        self.assertEqual(picks, {1, 4})

    def test_policy_ends_turn_before_any_potion(self):
        obs = observation(verbs=['use_potion', 'discard_potion', 'end_turn', 'use_potion', 'discard_potion'])
        rng = random.Random(3)
        self.assertEqual({self.view.playout_choice(obs, rng, POLICY) for _ in range(100)}, {2})

    def test_policy_falls_back_when_only_potions_legal(self):
        obs = observation(verbs=['use_potion', 'discard_potion', 'use_potion'])
        rng = random.Random(3)
        self.assertEqual({self.view.playout_choice(obs, rng, POLICY) for _ in range(200)}, {0, 1, 2})

    def test_policy_uses_only_verbs_and_rng(self):
        verbs = ['play_card', 'use_potion', 'play_card']
        plain = observation(verbs=verbs)
        other = observation(verbs=verbs, hp=1, max_hp=9, potions=['fairy_potion'], enemies=((5, 5), (9, 9)))
        self.assertEqual(self.view.playout_choice(plain, _Rng(), POLICY), 2)
        self.assertEqual(self.view.playout_choice(other, _Rng(), POLICY), 2)
        rng = _Rng()
        self.view.playout_choice(observation(verbs=['use_potion', 'end_turn']), rng, POLICY)
        self.assertEqual(rng.calls, 0)

    def test_use_potion_index_resolves(self):
        self.assertEqual(self.view.use_potion, descriptor()['vocabularies']['verb'].index('use_potion'))


if __name__ == '__main__':
    unittest.main()
