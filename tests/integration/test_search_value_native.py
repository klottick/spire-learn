"""Gated parity of native resource-v2 scoring with full-view replay, on Gym v1.2.5's tiered playout rule.

Entries come from the gym-1.2.4 run-states table. The live Gym must be in that rules epoch and must
declare the tiered rule (v1.2.5). Each native playout's ``actions()`` are replayed on a fresh branch of
the same sample with full-view stepping: every playout pick must be the rule's pick and lie in the tier
``View.playout_choice(..., "no-potions-v2")`` draws from, and the summary scorers must equal the
observation scorers exactly. Two targeted training entries add a Smoke Bomb (escapes, replayed under a
potion probe rule) and Alchemize (potions gained during production playouts).
"""

from collections import Counter
from contextlib import closing
import os
from pathlib import Path
import random
import unittest

from scripts.neural import boss_winnability as bw
from scripts.neural import teacher_descent_cost_entry as benchmark
from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET, contract_for_revision
from spire_learn.neural.search_one_move import (
    PLAYOUT_LAST_RESORT_VERBS, PLAYOUT_NEVER_VERBS, TIERED_PLAYOUT_RULE_ID, playout_branches,
)
from spire_learn.neural.search_value import (
    RootContext, ValueRule, cutoff_value, summary_cutoff_value, summary_terminal_value, terminal_value,
)
from spire_learn.neural.search_view import View

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
POLICY = "no-potions-v2"
ROOTS = 20
STEPS = 80
ACTIONS = 4
SAMPLES = (0, 1)
SEEDS = (0, 7)
TARGET_SEEDS = (0, 7, 11, 19, 23, 29)
CANDIDATES = 8
RULES = (ValueRule("resource-v2"), ValueRule("resource-v2", hp_curve="sqrt"))
TERMINAL = {"victory": 1, "death": -1, "escaped": 0}
PRODUCTION = {"rule": "tiered_verbs", "never_verbs": PLAYOUT_NEVER_VERBS,
              "last_resort_verbs": PLAYOUT_LAST_RESORT_VERBS}
# Scorer parity only, not a teacher rule: potions join the first tier, so Smoke Bomb escapes and spends occur.
PROBE = {"rule": "tiered_verbs", "never_verbs": ("discard_potion",), "last_resort_verbs": ("end_turn",)}


def targeted_entries(states, exclude):
    """Training A20 combats (not boss) holding a Smoke Bomb, and with Alchemize and a free potion slot."""
    pool = sorted((state for state in states if state["entry"]["ascension"] == 20 and bw._is_train(state["run_id"])
                   and state["room_kind"] in ("monster", "elite") and state["entry"]["id"] not in exclude),
                  key=lambda state: (state["run_id"], state["floor"]))
    smoke = [state for state in pool if "smoke_bomb" in state["entry"]["potions"]]
    alchemize = [state for state in pool if None in state["entry"]["potions"]
                 and any(card["card"] == "alchemize" for card in state["entry"]["deck"])]
    return smoke[:CANDIDATES], alchemize[:CANDIDATES]


def tier(view, observation, rule):
    """The legal actions a tiered rule draws from: outside both sets, else the last-resort ones, else all."""
    ids = {"use_potion": view.use_potion, "discard_potion": view.discard, "end_turn": view.end_turn}
    never = {ids[name] for name in rule["never_verbs"]}
    last = {ids[name] for name in rule["last_resort_verbs"]}
    legal = view.legal(observation)
    verbs = {action: int(observation["actions"][action, view.verb]) for action in legal}
    return ([action for action in legal if verbs[action] not in never | last]
            or [action for action in legal if verbs[action] in last] or legal)


def _summary_kinds(result):
    return tuple(sorted(int(kind) for kind, present in zip(result["potion_kind"][0], result["potion_present"][0])
                        if present))


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class SearchValueNativeTests(unittest.TestCase):
    def test_summary_scoring_equals_full_view_replay(self):
        from spire_lab import _native

        revision, states = bw._table_states(RECIPE)
        bw._check_live_revision(revision)
        self.assertTrue(contract_for_revision(_native.build_info()["revision"]).tiered_playouts,
                        f"the live Gym does not declare {TIERED_PLAYOUT_RULE_ID}")
        roots = benchmark.training_roots(states)
        smoke, alchemize = targeted_entries(states, {state["entry"]["id"] for state in roots})
        self.counts = Counter()
        self.assertEqual(self.compare_roots(roots, ROOTS, SEEDS, _native), ROOTS)
        for name, candidates, probe in (("smoke_bomb", smoke, True), ("alchemize", alchemize, False)):
            with self.subTest(target=name):
                self.assertTrue(candidates, f"the table has no training entry for {name}")
                self.assertEqual(self.compare_roots(candidates, 1, TARGET_SEEDS, _native, probe=probe), 1)
        print("search-value-native coverage:", dict(sorted(self.counts.items())))
        production = lambda key: self.counts[f"production:{key}"]
        self.assertGreater(production("victory") + production("death"), 0, self.counts)
        self.assertGreater(production("step_limit") + production("max_steps"), 0, self.counts)
        self.assertGreater(production("potion_gained"), 0, self.counts)
        self.assertGreater(self.counts["probe:escaped"], 0, self.counts)
        self.assertGreater(self.counts["probe:potion_spent"], 0, self.counts)

    def compare_roots(self, states, wanted, seeds, native, *, probe=False):
        compared = 0
        rules = (("production", PRODUCTION),) + ((("probe", PROBE),) if probe else ())
        branches = ACTIONS * len(SAMPLES) * len(seeds) * len(rules) * 3
        options = {"observation": "compact", "capacity_preset": STANDARD_COMPACT_PRESET, "workers": 1,
                   "limits": {"max_live_branches": 8, "max_branches": branches + 8,
                              "max_steps_per_branch": STEPS + 2, "max_total_steps": (branches + 8) * (STEPS + 2)}}
        with closing(bw._combat_env(bw._catalog(states, native=True))) as env:
            for index, state in enumerate(states):
                seed = benchmark.SEED + index
                observation, _ = env.reset(seed=seed, options={"entry_id": state["entry"]["id"]})
                view = View(env.observation_spec())
                legal = view.legal(observation)
                if len(legal) < 2:
                    continue
                try:
                    arena = env.public_search(seed * 1009, **options)
                except native.RunUnsupportedBoundaryError:
                    continue
                contexts = [RootContext.from_observation(view, observation, rule) for rule in RULES]
                with closing(arena):
                    arena_view = View(arena.observation_spec())
                    before = self.counts["production:compared"]
                    for action in legal[:ACTIONS]:
                        for sample in SAMPLES:
                            for playout_seed in seeds:
                                for label, rule in rules:
                                    with self.subTest(entry=state["entry"]["id"], action=action, sample=sample,
                                                      playout_seed=playout_seed, rule=label):
                                        self.compare(arena, arena_view, contexts, action, sample, playout_seed,
                                                     label, rule, native)
                    counts = arena.counters()
                    self.assertEqual(counts["live_branches"], 0)
                    self.assertEqual(counts["branches_created"], counts["branches_closed"])
                if self.counts["production:compared"] == before:
                    continue
                compared += 1
                if compared == wanted:
                    break
        return compared

    def compare(self, arena, view, contexts, action, sample, playout_seed, label, rule, native):
        raw, scored, replay = (arena.branch(sample) for _ in range(3))
        try:
            for branch in (raw, scored):
                try:
                    _, _, done, cut, _ = branch.step(action, view="full")
                except native.RunUnsupportedBoundaryError:
                    self.counts[f"{label}:root_refused"] += 1
                    return
            if done or cut:
                self.counts[f"{label}:root_ended"] += 1
                return
            result = arena.playout_many([raw], max_steps=STEPS, playout_seed=playout_seed, **rule)
            self.assertIsNone(result["fault"][0])
            self.assertEqual(result["rule_id"], TIERED_PLAYOUT_RULE_ID)
            reason = result["end_reason"][0]
            self.counts[f"{label}:{reason}"] += 1
            if reason == "refused":
                return
            actions = raw.actions()
            steps = int(result["steps"][0])
            self.assertEqual(len(actions), 1 + steps)
            self.assertEqual(actions[0], action)
            observation, reward, done, cut, info = self.replay(replay, view, actions, playout_seed, rule)
            terminated = bool(result["terminated"][0])
            self.assertEqual(bool(done), terminated)
            self.assertEqual(bool(cut), reason == "step_limit")
            if reason == "max_steps":
                self.assertEqual(steps, STEPS)
            if terminated:
                self.assertIn(reason, TERMINAL)
                self.assertEqual(info["terminal_reason"], reason)
                self.assertEqual(int(result["rewards"][0]), int(reward))
                self.assertEqual(int(result["player_hp"][0]), view.cleanup_hp(observation))
            else:
                self.assertEqual(int(result["player_hp"][0]), view.player_hp_now(observation))
            self.assertEqual(int(result["player_max_hp"][0]), view.player_max_hp(observation))
            kinds = _summary_kinds(result)
            self.assertEqual(kinds, view.potion_kinds(observation))
            held, root = Counter(kinds), Counter(contexts[0].potions)
            self.counts[f"{label}:potion_gained"] += bool(held - root)
            self.counts[f"{label}:potion_spent"] += bool(root - held)
            values = []
            for value_rule, context in zip(RULES, contexts, strict=True):
                if terminated:
                    expected = terminal_value(value_rule, context, view, observation, reward, info["terminal_reason"])
                    values.append(summary_terminal_value(value_rule, context, result, 0))
                else:
                    expected = cutoff_value(value_rule, context, view, observation)
                    values.append(summary_cutoff_value(value_rule, context, result, 0))
                self.assertEqual(values[-1], expected)
            if label == "production":
                # The teacher's own path: the same sample and stream give the same playout and value.
                outcome, = playout_branches(arena, [scored], max_steps=STEPS, stream=playout_seed,
                                            value_rule=RULES[0], playout_policy=POLICY, root=contexts[0])
                self.assertFalse(outcome.refused)
                self.assertEqual(outcome.value, values[0])
            self.counts[f"{label}:compared"] += 1
        finally:
            for branch in (raw, scored, replay):
                branch.close()

    def replay(self, branch, view, actions, playout_seed, rule):
        observation, reward, done, cut, info = branch.step(actions[0], view="full")
        for n, action in enumerate(actions[1:]):
            self.assertFalse(done or cut)
            self.assertEqual(branch.playout_action(playout_seed=playout_seed, **rule), action)
            pool = tier(view, observation, rule)
            self.assertIn(action, pool)
            if rule is PRODUCTION:
                # The test's tier is the one the Python policy draws from.
                for draw in range(3):
                    self.assertIn(view.playout_choice(observation, random.Random(n * 3 + draw), POLICY), pool)
            observation, reward, done, cut, info = branch.step(action, view="full")
        return observation, reward, done, cut, info


if __name__ == "__main__":
    unittest.main()
