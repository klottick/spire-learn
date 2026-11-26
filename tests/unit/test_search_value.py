"""Resource-aware search value rule: arithmetic, guards and ordering."""

import itertools
import math
import unittest

from spire_learn.neural.search_value import (
    REFUSED_VALUE, RootContext, ValueRule, cutoff_value, terminal_value)
from spire_learn.neural.search_view import View
from tests.unit._search_fakes import descriptor, observation

V2 = ValueRule('resource-v2')
FLAT = ValueRule('flat-v1')
SCALE = 0.4


class ValueTests(unittest.TestCase):
    def setUp(self):
        self.view = View(descriptor())

    def root(self, rule=V2, **kw):
        return RootContext.from_observation(self.view, observation(**kw), rule)

    def victory(self, root, rule=V2, **kw):
        kw.setdefault('cleanup', kw.get('hp', 50))
        return terminal_value(rule, root, self.view, observation(**kw), 1.0, 'victory')

    def test_flat_v1_returns_reward_and_half_balance(self):
        root = self.root(FLAT, max_hp=80)
        end = observation(hp=40, max_hp=80, enemies=((10, 40),))
        self.assertEqual(terminal_value(FLAT, root, self.view, end, 0.25, 'victory'), 0.25)
        self.assertEqual(terminal_value(FLAT, root, self.view, end, -1.0, None), -1.0)
        self.assertEqual(cutoff_value(FLAT, root, self.view, end), 0.5 * self.view.balance(end))
        self.assertEqual(REFUSED_VALUE, -1.0)

    def test_v2_victory_full_hp_is_0_95(self):
        root = self.root(hp=80, max_hp=80)
        self.assertAlmostEqual(self.victory(root, hp=80, max_hp=80), 0.95)

    def test_v2_victory_same_end_hp_scores_same_regardless_of_start(self):
        low, high = self.root(hp=1, max_hp=80), self.root(hp=80, max_hp=80)
        self.assertEqual(self.victory(low, hp=1, max_hp=80), self.victory(high, hp=1, max_hp=80))

    def test_v2_potion_price_in_hp(self):
        root = self.root(hp=75, max_hp=75, potions=['block_potion'])
        for hp in (75, 40):
            spent = self.victory(root, hp=hp, max_hp=75)
            kept = self.victory(root, hp=hp, max_hp=75, potions=['block_potion'])
            self.assertAlmostEqual(kept - spent, SCALE * 12 / 75)

    def test_v2_fairy_and_fruit_juice_overrides(self):
        root = self.root(hp=60, max_hp=80, potions=['fruit_juice', 'fairy_potion'])
        kept = self.victory(root, hp=60, max_hp=80, potions=['fruit_juice', 'fairy_potion'])
        juice = self.victory(root, hp=60, max_hp=85, potions=['fairy_potion'])
        fairy = self.victory(root, hp=60, max_hp=80, potions=['fruit_juice'])
        self.assertAlmostEqual(kept, juice)
        self.assertAlmostEqual(kept - fairy, SCALE * 30 / 80)

    def test_v2_feed_headroom_at_full_hp(self):
        root = self.root(hp=80, max_hp=80)
        gained = self.victory(root, hp=83, max_hp=83)
        self.assertAlmostEqual(gained, min(1.0, 0.95 + SCALE * 4.5 / 80))

    def test_v2_hp_and_potion_price_trade_one_to_one(self):
        root = self.root(hp=40, max_hp=80, potions=['block_potion'])
        drank = self.victory(root, hp=40, max_hp=80)
        saved = self.victory(root, hp=28, max_hp=80, potions=['block_potion'])
        self.assertAlmostEqual(drank, saved)

    def test_v2_sqrt_curve(self):
        rule = ValueRule('resource-v2', hp_curve='sqrt')
        root = self.root(rule, hp=20, max_hp=80)
        self.assertAlmostEqual(self.victory(root, rule, hp=20, max_hp=80), 0.55 + SCALE * 0.5)

    def test_v2_ordering_holds_on_a_grid(self):
        spent = [[], ['fairy_potion'] * 1, ['block_potion'] * 2, ['fairy_potion', 'block_potion', 'fire_potion'],
                 ['fairy_potion', 'block_potion', 'fire_potion', 'swift_potion', 'fruit_juice']]
        root = RootContext(80, tuple(sorted(self.view.potion_kinds(observation(
            potions=['block_potion', 'fairy_potion', 'fire_potion', 'swift_potion', 'fruit_juice'])))),
            self.view.potion_price_table(V2))
        held = ['block_potion', 'fairy_potion', 'fire_potion', 'swift_potion', 'fruit_juice']
        for curve in ('linear', 'sqrt'):
            rule = ValueRule('resource-v2', hp_curve=curve)
            root = RootContext(80, root.potions, self.view.potion_price_table(rule))
            for hp, max_end, gone, enemy in itertools.product(
                    (0, 1, 40, 80), (70, 80, 90), spent, ((0, 40), (20, 40), (40, 40))):
                end = [p for p in held if p not in gone] if gone else held
                kw = dict(hp=hp, max_hp=max_end, potions=end, enemies=(enemy,))
                win = terminal_value(rule, root, self.view, observation(cleanup=hp, **kw), 1.0, 'victory')
                escape = terminal_value(rule, root, self.view, observation(cleanup=hp, **kw), 0.0, 'escaped')
                cut = cutoff_value(rule, root, self.view, observation(**kw))
                dead = terminal_value(rule, root, self.view, observation(**kw), -1.0, 'death')
                self.assertGreaterEqual(win, 0.41)
                self.assertLessEqual(win, 1.0)
                self.assertGreater(0.41, max(escape, cut))
                self.assertGreaterEqual(min(escape, cut), -0.99)
                self.assertEqual(dead, -1.0)

    def test_v2_escape_uses_cleanup_hp_and_charges_smoke_bomb(self):
        root = self.root(hp=50, max_hp=80, potions=['block_potion'])
        end = observation(hp=10, max_hp=80, cleanup=40)
        self.assertAlmostEqual(terminal_value(V2, root, self.view, end, 0.0, 'escaped'),
                               -0.2 + SCALE * 40 / 80 - SCALE * 12 / 80)

    def test_flat_v1_rejects_v2_parameters(self):
        for kw in ({'potion_hp': 10.0}, {'hp_curve': 'sqrt'}, {'potion_overrides': ()}):
            with self.assertRaises(ValueError):
                ValueRule('flat-v1', **kw)

    def test_value_rule_validation(self):
        for kw in ({'potion_hp': -1.0}, {'potion_hp': math.inf}, {'potion_hp': math.nan},
                   {'potion_overrides': (('a', 1.0), ('a', 2.0))}, {'potion_overrides': (('padding', 1.0),)},
                   {'potion_overrides': (('a', -1.0),)}, {'hp_curve': 'cube'}):
            with self.assertRaises(ValueError):
                ValueRule('resource-v2', **kw)
        with self.assertRaises(ValueError):
            ValueRule('other')
        self.assertEqual(ValueRule('resource-v2', potion_overrides=(('b', 1), ('a', 2))).potion_overrides,
                         (('a', 2.0), ('b', 1.0)))

    def test_value_rule_identity_round_trip(self):
        for rule in (FLAT, V2, ValueRule('resource-v2', potion_hp=10.0, hp_curve='sqrt')):
            self.assertEqual(ValueRule.from_identity(rule.as_identity()), rule)
        self.assertEqual(V2.as_identity()['potion_overrides'], {'fairy_potion': 30.0, 'fruit_juice': 7.5})

    def test_terminal_without_cleanup_hp_raises(self):
        root = self.root()
        for reason in ('victory', 'escaped'):
            with self.assertRaises(ValueError):
                terminal_value(V2, root, self.view, observation(hp=30, cleanup=None), 1.0, reason)
        self.assertEqual(terminal_value(V2, root, self.view, observation(hp=0, cleanup=None), -1.0, 'death'), -1.0)

    def test_unknown_reason_and_unknown_live_hp_raise(self):
        root = self.root()
        with self.assertRaises(ValueError):
            terminal_value(V2, root, self.view, observation(cleanup=5), 1.0, 'timeout')
        with self.assertRaises(ValueError):
            cutoff_value(V2, root, self.view, observation(known=False))

    def test_root_context_rejects_nonpositive_or_unknown_max_hp(self):
        for kw in ({'max_hp': 0}, {'known': False}):
            with self.assertRaises(ValueError):
                RootContext.from_observation(self.view, observation(**kw), V2)

    def test_gained_potion_credit_is_capped(self):
        root = self.root(hp=80, max_hp=80)
        gained = ['fairy_potion', 'block_potion', 'fire_potion']
        win = self.victory(root, hp=80, max_hp=80, potions=gained)
        self.assertEqual(win, 1.0)
        end = observation(hp=80, max_hp=80, cleanup=80, potions=gained)
        self.assertEqual(terminal_value(V2, root, self.view, end, 0.0, 'escaped'), 0.4)
        self.assertEqual(cutoff_value(V2, root, self.view, observation(hp=80, max_hp=80, potions=gained,
                                                                         enemies=((0, 40),))), 0.4)

    def test_potion_price_table_resolves_ids_and_rejects_unknown(self):
        table = self.view.potion_price_table(V2)
        self.assertEqual((table[1], table[2], table[4]), (12.0, 30.0, 7.5))
        self.assertNotIn(0, table)
        with self.assertRaises(ValueError):
            self.view.potion_price_table(ValueRule('resource-v2', potion_overrides=(('nope', 1.0),)))

    def test_potion_kinds_fail_closed_on_unknown_kind(self):
        obs = observation(potions=['block_potion'], potions_unknown=True)
        with self.assertRaises(ValueError):
            self.view.potion_kinds(obs)
        self.assertEqual(self.view.potion_kinds(observation(potions_unknown=True)), ())

    def test_from_identity_rejects_non_mapping_overrides(self):
        for bad in ([('fairy_potion', 30.0)], None, 'x'):
            with self.assertRaises(ValueError):
                ValueRule.from_identity({'name': 'resource-v2', 'potion_overrides': bad})

    def test_real_fixture_descriptor_resolves(self):
        import json
        from pathlib import Path
        path = Path(__file__).parents[1] / 'fixtures/neural/standard-combat-compact-4d19806.json'
        real = View(json.loads(path.read_text()))
        vocabulary = json.loads(path.read_text())['vocabularies']
        table = real.potion_price_table(V2)
        self.assertEqual(table[vocabulary['potion'].index('fairy_potion')], 30.0)
        self.assertEqual(table[vocabulary['potion'].index('fruit_juice')], 7.5)
        self.assertIsNotNone(real._cleanup_hp_column)
        self.assertIsNotNone(real._potion_kind_column)
        self.assertEqual(real.use_potion, vocabulary['verb'].index('use_potion'))

    def test_potion_multiset_charge(self):
        root = self.root(hp=80, max_hp=80, potions=['block_potion', 'block_potion'])
        base = self.victory(root, hp=40, max_hp=80, potions=['block_potion', 'block_potion'])
        one_spent = self.victory(root, hp=40, max_hp=80, potions=['block_potion'])
        self.assertAlmostEqual(base - one_spent, SCALE * 12 / 80)
        root1 = self.root(hp=80, max_hp=80, potions=['block_potion'])
        gained = self.victory(root1, hp=40, max_hp=80, potions=['block_potion', 'block_potion'])
        same = self.victory(root1, hp=40, max_hp=80, potions=['block_potion'])
        self.assertAlmostEqual(gained - same, SCALE * 12 / 80)

    def test_v2_cutoff_exact_values(self):
        # M 80; hp 40 -> h 0.5; enemies 10/40 -> e 0.25; max +4 -> 6 HP; one block_potion spent -> 12 HP.
        kw = dict(hp=40, max_hp=84, enemies=((10, 40),))
        root = self.root(hp=80, max_hp=80, potions=['block_potion'])
        bonus = SCALE * (6 - 12) / 80
        self.assertAlmostEqual(cutoff_value(V2, root, self.view, observation(**kw)),
                               SCALE * (0.5 - 0.25) + bonus)
        rule = ValueRule('resource-v2', hp_curve='sqrt')
        root = self.root(rule, hp=80, max_hp=80, potions=['block_potion'])
        self.assertAlmostEqual(cutoff_value(rule, root, self.view, observation(**kw)),
                               SCALE * (math.sqrt(0.5) - 0.25) + bonus)


if __name__ == '__main__':
    unittest.main()
