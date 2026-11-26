"""Native playout backend with scripted playout_many results (no Gym runtime)."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural import search_teacher as teacher
from spire_learn.neural import search_view
from spire_learn.neural.gym_contracts import GYM_V1_2_4_CONTRACT, GYM_V1_2_5_CONTRACT
from spire_learn.neural.scenario_support import content_sha256
from spire_learn.neural.search_one_move import (
    PLAYOUT_EXCLUDE_VERBS, PLAYOUT_RULE_ID, TIERED_PLAYOUT_RULE_ID, ActionStat, PlayoutOutcome, one_move_values,
    playout_branches, playout_rule_id, playout_seed, require_tiered_playouts,
)
from spire_learn.neural.search_value import (
    REFUSED_VALUE, RootContext, ValueRule, cutoff_value, summary_cutoff_value, summary_terminal_value,
    terminal_value,
)
from spire_learn.neural.search_view import View, summary_balance
from tests.unit import _search_fakes as fakes
from tests.unit.test_search_one_move import _View
from tests.unit import test_search_teacher as tt

POTION_SLOTS = 5


class Unsupported(Exception):
    pass


class Fault(Exception):
    pass


def summary(player=(50, 100), enemies=((10, 20),), end_reason="step_limit", reward=0, terminated=False,
            potions=()):
    """One summary row; ``potions`` are the held potion names (``_search_fakes.POTIONS``)."""
    return {"end_reason": end_reason, "reward": reward, "terminated": terminated,
            "player": player, "enemies": enemies, "potions": potions}


def playout_result(rows, faults=None, rule_id=PLAYOUT_RULE_ID):
    """A ``playout_many`` result shaped like the documented summary (list-backed, no numpy)."""
    width = max(len(row["enemies"]) for row in rows) or 1
    pad = lambda values, fill: list(values) + [fill] * (width - len(values))
    slots = lambda values, fill: list(values) + [fill] * (POTION_SLOTS - len(values))
    return {
        "rule_id": rule_id,
        "player_gold": [0] * len(rows),
        "potion_kind": [slots([fakes.POTIONS.index(name) for name in row.get("potions", ())], 0) for row in rows],
        "potion_present": [slots([1] * len(row.get("potions", ())), 0) for row in rows],
        "end_reason": [row["end_reason"] for row in rows],
        "refusal": [None] * len(rows),
        "fault": faults or [None] * len(rows),
        "terminated": [row["terminated"] for row in rows],
        "rewards": [row["reward"] for row in rows],
        "player_hp": [row["player"][0] for row in rows],
        "player_max_hp": [row["player"][1] for row in rows],
        "enemy_hp": [pad([e[0] for e in row["enemies"]], 0) for row in rows],
        "enemy_max_hp": [pad([e[1] for e in row["enemies"]], 0) for row in rows],
        "enemy_present": [pad([1] * len(row["enemies"]), 0) for row in rows],
    }


def _rule_id(options):
    return TIERED_PLAYOUT_RULE_ID if options.get("rule") == "tiered_verbs" else PLAYOUT_RULE_ID


class Branch:
    def __init__(self, sample):
        self.sample, self.closed = sample, False

    def close(self):
        self.closed = True


class NativeSearch:
    """step_many leaves every branch live and nonterminal; playout_many is scripted by (sample)."""

    def __init__(self, script, whole_call_refusal=False):
        self.script, self.branches, self.calls = script, [], []
        self.whole_call_refusal = whole_call_refusal

    def branch(self, sample):
        branch = Branch(sample)
        self.branches.append(branch)
        return branch

    def step_many(self, branches, actions, *, view):
        rows = {"balance": [0.0] * len(branches), "legal": [[0, 1]] * len(branches)}
        n = len(branches)
        return rows, [0.0] * n, [False] * n, [False] * n, {"fault": [None] * n}

    def playout_many(self, branches, **options):
        self.calls.append((list(branches), options))
        if self.whole_call_refusal and len(branches) > 1:
            raise Unsupported("preflight")
        return playout_result([self.script(branch) for branch in branches], rule_id=_rule_id(options))


class SummaryBalanceTests(unittest.TestCase):
    def test_matches_view_balance_definition(self):
        # Alive rows only, enemy HP clipped at zero, player HP clipped, max clipped to 1.
        result = playout_result([summary(player=(50, 100), enemies=((10, 20), (-3, 10)))])
        self.assertEqual(summary_balance(result, 0), 0.5 - 10 / 30)
        result = playout_result([summary(player=(-4, 0), enemies=())])
        self.assertEqual(summary_balance(result, 0), 0.0)
        result = playout_result([summary(player=(0, 80), enemies=((0, 20),))])
        result["enemy_present"] = [[0]]
        self.assertEqual(summary_balance(result, 0), 0.0)


class OneMoveNativeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("spire_learn.neural.search_view._native_errors",
                        return_value=(Unsupported, ValueError, Fault))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.params = dict(samples=3, playout_steps=5, research_seed=17, playout_backend="native")

    def values(self, search):
        return one_move_values(search, [0, 1], _View(), **self.params)

    def test_means_visits_and_one_call_over_all_live_branches(self):
        def script(branch):
            if branch.sample == 0:
                return summary(end_reason="victory", reward=1, terminated=True)
            if branch.sample == 1:
                return summary(end_reason="death", reward=-1, terminated=True)
            return summary(player=(50, 100), enemies=((10, 20),))  # balance 0, half 0

        search = NativeSearch(script)
        stats = self.values(search)
        self.assertEqual(stats, {0: ActionStat(0.0, 3, 0), 1: ActionStat(0.0, 3, 0)})
        self.assertEqual(len(search.calls), 1)
        branches, options = search.calls[0]
        self.assertEqual(len(branches), 6)
        self.assertEqual(options, {"rule": "uniform_exclude_verbs", "exclude_verbs": PLAYOUT_EXCLUDE_VERBS,
                                   "max_steps": 5, "playout_seed": 0})
        self.assertEqual(PLAYOUT_EXCLUDE_VERBS, ("end_turn", "discard_potion"))
        self.assertTrue(all(branch.closed for branch in search.branches))

    def test_step_limit_scores_half_summary_balance(self):
        search = NativeSearch(lambda branch: summary(player=(100, 100), enemies=((0, 40),)))
        self.assertEqual(self.values(search)[0].mean, 0.5)

    def test_refused_playout_counts_minus_one(self):
        search = NativeSearch(lambda b: summary(end_reason="refused") if b.sample == 1 else summary(
            player=(50, 100), enemies=((10, 20),)))
        stats = self.values(search)
        self.assertEqual(stats[0], ActionStat(-1 / 3, 3, 1))
        self.assertEqual(stats[1], ActionStat(-1 / 3, 3, 1))

    def test_whole_call_refusal_is_attributed_per_branch(self):
        search = NativeSearch(lambda b: summary(end_reason="refused") if b.sample == 0 else summary(
            player=(50, 100), enemies=((10, 20),)), whole_call_refusal=True)
        self.assertEqual(self.values(search)[0], ActionStat(-1 / 3, 3, 1))

    def test_fault_raises_and_closes_all_handles(self):
        class Faulting(NativeSearch):
            def playout_many(self, branches, **options):
                result = playout_result([summary()] * len(branches))
                result["fault"][2] = {"stage": "engine"}
                return result

        search = Faulting(lambda b: summary())
        with self.assertRaisesRegex(Fault, "branch 2"):
            self.values(search)
        self.assertTrue(search.branches and all(branch.closed for branch in search.branches))

    def test_zero_playout_steps_uses_observation_balance_without_native_call(self):
        search = NativeSearch(lambda b: summary())
        self.params["playout_steps"] = 0
        self.assertEqual(self.values(search)[0].mean, 0.0)
        self.assertEqual(search.calls, [])

    def test_unknown_backend_is_rejected(self):
        self.params["playout_backend"] = "gpu"
        with self.assertRaises(ValueError):
            self.values(NativeSearch(lambda b: summary()))


class NativeTeacherSearch(tt.Search):
    def __init__(self, script, results):
        super().__init__(script)
        self.results, self.playouts = results, []

    def playout_many(self, branches, **options):
        self.playouts.append((branches[0].world, options))
        return playout_result([self.results(branches[0].world, len(self.playouts) - 1)] * len(branches),
                              rule_id=_rule_id(options))


class TeacherNativeTests(unittest.TestCase):
    def setUp(self):
        self.view = tt.WorldView()
        patcher = patch.object(teacher.search_view, "_native_errors",
                               return_value=(tt.Unsupported, ValueError, tt.Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def script(world, path):
        return tt.observation(len(path), (1,)), 0.0, False

    def teach(self, results, **overrides):
        search = NativeTeacherSearch(self.script, results)
        cfg = tt.config(playout_steps=4, playout_backend="native", **overrides)
        result = teacher.teach(search, [0], self.view, cfg, research_seed=17)
        self.assertEqual(search.counters()["live_branches"], 0)
        return result, search

    def test_values_come_from_native_results_and_seeds_are_deterministic(self):
        results = lambda world, call: summary(end_reason="victory", reward=1, terminated=True) \
            if world == 0 else summary(player=(100, 100), enemies=((0, 10),))
        first, search = self.teach(results)
        second, again = self.teach(results)
        self.assertEqual(first.stats[0].mean, 0.75)
        self.assertEqual(first.stats, second.stats)
        seeds = [options["playout_seed"] for _, options in search.playouts]
        self.assertEqual(seeds, [options["playout_seed"] for _, options in again.playouts])
        self.assertTrue(all(options["max_steps"] == 4 and options["rule"] == "uniform_exclude_verbs"
                            and options["exclude_verbs"] == PLAYOUT_EXCLUDE_VERBS
                            for _, options in search.playouts))
        self.assertEqual(seeds[:2], [0, 0])  # first visit per world is the reference stream

    def test_refusal_counts_like_branch_refusal(self):
        results = lambda world, call: summary(end_reason="refused") if world == 1 else summary(
            end_reason="victory", reward=1, terminated=True)
        result, _ = self.teach(results)
        self.assertEqual(result.refused, result.stats[0].refused)
        self.assertGreater(result.refused, 0)
        self.assertEqual(result.stats[0].mean, (result.stats[0].visits - 2 * result.refused) / result.stats[0].visits)

    def test_fault_raises_and_closes_handles(self):
        class Faulting(NativeTeacherSearch):
            def playout_many(self, branches, **options):
                result = playout_result([summary()])
                result["fault"] = [{"stage": "engine"}]
                return result

        search = Faulting(self.script, None)
        with self.assertRaises(tt.Fault):
            teacher.teach(search, [0], self.view, tt.config(playout_steps=4, playout_backend="native"),
                          research_seed=17)
        self.assertEqual(search.counters()["live_branches"], 0)


class IdentityTests(unittest.TestCase):
    def test_python_hash_is_unchanged_and_native_is_distinct(self):
        python, native = tt.config(), tt.config(playout_backend="native")
        fields = asdict(python)
        del fields["playout_backend"], fields["simulations_per_action"]
        del fields["value_rule"], fields["playout_policy"]
        legacy = sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.assertEqual(teacher.config_sha256(python), legacy.hexdigest())
        self.assertNotEqual(teacher.config_sha256(python), teacher.config_sha256(native))
        self.assertEqual(teacher.config_identity(native)["playout_rule_id"], PLAYOUT_RULE_ID)
        self.assertNotIn("playout_backend", teacher.config_identity(python))

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            tt.config(playout_backend="rust")

    def test_playout_seed_is_deterministic(self):
        self.assertEqual(playout_seed(5, 3, 0), 0)
        self.assertEqual(playout_seed(5, 3, 2), playout_seed(5, 3, 2))
        self.assertNotEqual(playout_seed(5, 3, 1), playout_seed(5, 3, 2))
        self.assertNotEqual(playout_seed(5, 3, 1), playout_seed(5, 4, 1))
        self.assertTrue(0 < playout_seed(2**64 - 1, 2**64 - 1, 99) < 2**64)


class EntryConfigTests(unittest.TestCase):
    def test_gym_1_2_4_teachers_are_native_and_calibrated(self):
        from scripts.neural import search_teacher_entry as entry

        new = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
        old = entry._ROOT / "configs/neural/search-teacher-gym-1.2.0/teachers.json"
        for name in ("current", "tree-1x", "tree-3x", "tree-10x"):
            config, digest, calibrated = entry._teacher(new, name, None)
            _, old_digest, _ = entry._teacher(old, name, None)
            self.assertNotEqual(digest, old_digest)
            backend = config["playout_backend"] if name == "current" else config.playout_backend
            self.assertEqual(backend, "native")
            self.assertTrue(calibrated)
            self.assertTrue(entry._native_backend(config))
            self.assertEqual(entry._teacher(new, name, "selection")[1], digest)
        for name in ("current", "tree-1x"):
            self.assertFalse(entry._native_backend(entry._teacher(old, name, None)[0]))

    def test_current_calibration_flag_and_explicit_python_backend(self):
        import tempfile
        from pathlib import Path
        from scripts.neural import search_teacher_entry as entry

        def load(row):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "teachers.json"
                path.write_text(json.dumps({"teachers": {"current": row}}))
                return entry._teacher(path, "current", "selection")

        base = {"kind": "one-move", "samples": 32, "playout_steps": 80}
        implicit = load(base)
        self.assertTrue(implicit[2])
        explicit = load({**base, "playout_backend": "python"})
        self.assertEqual(explicit, implicit)
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            load({**base, "calibrated": False})
        self.assertTrue(load({**base, "calibrated": True})[2])
        with self.assertRaises(ValueError):
            load({**base, "playout_backend": "gpu"})

    def test_v2_file_teachers_load_and_differ_from_v1_only_in_value_policy_calibration_and_ucb(self):
        from dataclasses import replace
        from scripts.neural import search_teacher_entry as entry
        from spire_learn.neural.search_one_move import TIERED_PLAYOUT_RULE_ID

        v1 = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
        v2 = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers-v2.json"
        names = ["current-v1-nopot", "current-v2", "current-v2-sqrt", "tree-1x-v2",
                 "current-v2-p10", "current-v2-p15"]
        self.assertEqual(sorted(json.loads(v2.read_text())["teachers"]), sorted(names))
        overrides = {"fairy_potion": 30, "fruit_juice": 7.5}
        expected = {"current-v1-nopot": ("flat-v1", 12.0, "linear"), "current-v2": ("resource-v2", 12, "linear"),
                    "current-v2-sqrt": ("resource-v2", 12, "sqrt"), "tree-1x-v2": ("resource-v2", 12, "linear"),
                    "current-v2-p10": ("resource-v2", 10, "linear"), "current-v2-p15": ("resource-v2", 15, "linear")}
        v1_hashes = {entry._teacher(v1, n, None)[1] for n in ("current", "tree-1x", "tree-3x", "tree-10x")}
        base, _, _ = entry._teacher(v1, "current", None)
        tree_base, _, _ = entry._teacher(v1, "tree-1x", None)
        hashes = set()
        for name in names:
            config, digest, calibrated = entry._teacher(v2, name, "selection", allow_uncalibrated=True)
            with self.assertRaisesRegex(ValueError, "uncalibrated"):
                entry._teacher(v2, name, "selection")
            self.assertFalse(calibrated)
            self.assertTrue(entry._native_backend(config))
            self.assertNotIn(digest, v1_hashes)
            hashes.add(digest)
            kind, potion_hp, curve = expected[name]
            if name == "tree-1x-v2":
                rule, policy = config.value_rule, config.playout_policy
                self.assertEqual(config.playout_rule_id if hasattr(config, "playout_rule_id") else
                                 entry.config_identity(config)["playout_rule_id"], TIERED_PLAYOUT_RULE_ID)
                neutral = {"value_rule": tree_base.value_rule, "playout_policy": tree_base.playout_policy}
                self.assertEqual(asdict(replace(config, **neutral)), asdict(tree_base))
            else:
                rule, policy = config["value_rule"] if kind != "flat-v1" else config.get("value_rule"), config["playout_policy"]
                self.assertEqual(entry.playout_rule_id(policy), TIERED_PLAYOUT_RULE_ID)
                self.assertEqual({k: v for k, v in config.items() if k not in ("value_rule", "playout_policy")},
                                 base)
            self.assertEqual(policy, "no-potions-v2")
            if kind == "flat-v1":
                self.assertIsNone(rule)
            else:
                self.assertEqual(rule.as_identity(), {"name": "resource-v2", "potion_hp": potion_hp,
                                                      "potion_overrides": overrides, "hp_curve": curve})
        self.assertEqual(len(hashes), len(names))


class NativePerActionTests(unittest.TestCase):
    script = staticmethod(TeacherNativeTests.script)

    def setUp(self):
        self.view = tt.WorldView()
        patcher = patch.object(teacher.search_view, "_native_errors",
                               return_value=(tt.Unsupported, ValueError, tt.Fault))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_native_backend_scales_simulations_with_legal_count(self):
        results = lambda world, call: summary(end_reason="victory", reward=1, terminated=True)
        search = NativeTeacherSearch(self.script, results)
        cfg = tt.config(simulations=None, simulations_per_action=8, playout_steps=4, playout_backend="native")
        result = teacher.teach(search, [0, 1, 2], self.view, cfg, research_seed=17)
        self.assertEqual(result.simulations, 24)
        self.assertEqual([stat.visits for stat in result.stats.values()], [8, 8, 8])
        self.assertEqual(len(search.playouts), 24)


V2 = ValueRule("resource-v2")
FLAT = ValueRule("flat-v1")
POLICY = "no-potions-v2"
UNIFORM_OPTIONS = {"rule": "uniform_exclude_verbs", "exclude_verbs": ("end_turn", "discard_potion")}
TIERED_OPTIONS = {"rule": "tiered_verbs", "never_verbs": ("use_potion", "discard_potion"),
                  "last_resort_verbs": ("end_turn",)}
# config_sha256 of tt.config(playout_backend="native") and of the committed gym-1.2.4 tree-3x, before v2.
NATIVE_TEST_SHA = "904333d5dc1b831dfa7a970d75bd6b65e9923b098138c5cfe733ec1e58e5ebc8"
TREE_3X_SHA = "7db31be63376a5a16745e77a8cfb4d83e7cc9ed408a3370a774696b556558eef"
ROOT_POTIONS = ("fairy_potion", "smoke_bomb")


def _stack(rows):
    """step_many's stacked output for per-branch (observation, reward, terminated, truncated, info) rows."""
    n = len(rows)
    observations = {key: [row[0][key] for row in rows] for key in rows[0][0]}
    info = {"fault": [None] * n}
    for key in sorted({key for row in rows for key in row[4]}):
        info[key] = [row[4].get(key) for row in rows]
        info["_" + key] = [key in row[4] for row in rows]
    return observations, [row[1] for row in rows], [row[2] for row in rows], [row[3] for row in rows], info


class StandardNativeSearch(NativeSearch):
    """Standard-shaped root steps from ``root_step(sample, action)``; playouts scripted by sample."""

    def __init__(self, script, root_step):
        super().__init__(script)
        self.root_step, self.views = root_step, []

    def step_many(self, branches, actions, *, view):
        self.views.append(view)
        return _stack([self.root_step(branch.sample, action) for branch, action in zip(branches, actions)])


class _FullBranch(tt.Branch):
    def step(self, action, *, view):
        assert not self.closed and view == "full"
        self.path += (action,)
        return self.search.script(self.world, self.path)


class StandardTreeSearch(tt.Search):
    """Tree fake over Standard-shaped observations; ``results(world)`` scripts each native playout."""

    def __init__(self, script, root, results):
        super().__init__(script)
        self.root, self.results, self.playouts = root, results, []

    def open(self, world, path):
        branch = _FullBranch(self, world, path)
        self.handles.append(branch)
        return branch

    def playout_many(self, branches, **options):
        self.playouts.append((branches[0].world, options))
        return playout_result([self.results(branches[0].world)] * len(branches), rule_id=_rule_id(options))


def _native_v2_rows(tree):
    """Uncalibrated native v2 teachers: a one-move row and the tree-3x config."""
    return {
        "current-native-v2": {"kind": "one-move", "calibrated": False, "samples": 32, "playout_steps": 80,
                              "playout_backend": "native", "value_rule": {"name": "resource-v2"},
                              "playout_policy": POLICY},
        "tree-native-v2": {"calibrated": False, "config": {**tree["config"], "value_rule": {"name": "resource-v2"},
                                                           "playout_policy": POLICY}},
    }


def _key(observation):
    return repr({name: list(tensor) for name, tensor in sorted(observation.items())}).encode()


class NativeV2Tests(unittest.TestCase):
    """Native v2 on Gym v1.2.5's tiered rule, with scripted playout_many summaries."""

    def setUp(self):
        self.view = View(fakes.descriptor())
        for patcher in (patch.object(search_view, "_native_errors", return_value=(Unsupported, ValueError, Fault)),
                        patch.object(teacher, "_public_key", _key)):
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(search_view, "_native_contract", return_value=GYM_V1_2_5_CONTRACT)
        self.contract = patcher.start()
        self.addCleanup(patcher.stop)
        self.root = fakes.observation(hp=60, max_hp=80, potions=ROOT_POTIONS, verbs=("play_card", "play_card"))
        self.context = RootContext.from_observation(self.view, self.root, V2)

    def one_move(self, search, legal=(0, 1), **options):
        stats = one_move_values(search, list(legal), self.view, samples=3, playout_steps=5, research_seed=17,
                                playout_backend="native", **options)
        self.assertTrue(search.branches and all(branch.closed for branch in search.branches))
        return stats

    def live(self, sample, action):
        return fakes.observation(hp=60, max_hp=80, potions=ROOT_POTIONS, verbs=("play_card",)), 0.0, False, False, {}

    def test_native_v1_call_and_hash_unchanged(self):
        # v1 never consults the live contract, keeps today's exact call and scores.
        self.contract.side_effect = AssertionError("v1 consulted the live contract")
        script = lambda branch: summary(end_reason="victory", reward=1, terminated=True) if branch.sample == 0 \
            else summary(player=(50, 100), enemies=((10, 25),))
        search = StandardNativeSearch(script, self.live)
        stats = self.one_move(search)
        self.assertEqual([options for _, options in search.calls],
                         [{**UNIFORM_OPTIONS, "max_steps": 5, "playout_seed": 0}])
        half = 0.5 * (0.5 - 10 / 25)
        mean = (1.0 + half + half) / 3
        self.assertEqual(stats, {0: ActionStat(mean, 3, 0), 1: ActionStat(mean, 3, 0)})

        results = lambda world, call: summary(end_reason="victory", reward=1, terminated=True) \
            if world == 0 else summary(player=(100, 100), enemies=((0, 10),))
        tree = NativeTeacherSearch(TeacherNativeTests.script, results)
        result = teacher.teach(tree, [0], tt.WorldView(), tt.config(playout_steps=4, playout_backend="native"),
                               research_seed=17)
        self.assertEqual(result.stats[0].mean, 0.75)
        self.assertTrue(tree.playouts)
        for _, options in tree.playouts:
            self.assertEqual({key: value for key, value in options.items() if key not in ("max_steps", "playout_seed")},
                             UNIFORM_OPTIONS)
        self.contract.assert_not_called()

        self.assertEqual(playout_rule_id("uniform-v1"), PLAYOUT_RULE_ID)
        self.assertEqual(PLAYOUT_RULE_ID, "spire-gym-public-playout-v1")
        native = tt.config(playout_backend="native")
        self.assertEqual(teacher.config_sha256(native), NATIVE_TEST_SHA)
        self.assertEqual(teacher.config_identity(native)["playout_rule_id"], PLAYOUT_RULE_ID)
        from tests.unit import test_search_value_wiring as wiring
        self.assertEqual(teacher.config_sha256(wiring._tree_config("tree-3x")), TREE_3X_SHA)

    def test_native_no_potions_uses_tiered_rule_with_explicit_sets(self):
        held = dict(potions=ROOT_POTIONS)
        script = lambda branch: summary(end_reason="victory", reward=1, terminated=True, player=(70, 80), **held) \
            if branch.sample == 0 else summary(player=(40, 80), enemies=((10, 40),), **held)
        cases = (("flat-control", {"playout_policy": POLICY}, 1.0, 0.5 * (0.5 - 0.25)),
                 ("resource-v2", {"value_rule": V2, "playout_policy": POLICY, "root": self.context},
                  0.55 + 0.4 * 70 / 80, 0.4 * (0.5 - 0.25)))
        for name, options, win, cut in cases:
            with self.subTest(teacher=name):
                search = StandardNativeSearch(script, self.live)
                stats = self.one_move(search, **options)
                self.assertEqual(len(search.calls), 1)
                branches, call = search.calls[0]
                self.assertEqual(len(branches), 6)
                self.assertEqual(call, {**TIERED_OPTIONS, "max_steps": 5, "playout_seed": 0})
                self.assertNotIn("exclude_verbs", call)
                for action in (0, 1):
                    self.assertAlmostEqual(stats[action].mean, (win + cut + cut) / 3)
                    self.assertEqual(stats[action].refused, 0)
        self.assertEqual(self.contract.call_count, 2)

        # Tree teachers: the flat control on the v1 fakes, resource-v2 on Standard-shaped ones.
        results = lambda world, call: summary(end_reason="victory", reward=1, terminated=True) \
            if world == 0 else summary(player=(100, 100), enemies=((0, 10),))
        control = NativeTeacherSearch(TeacherNativeTests.script, results)
        result = teacher.teach(control, [0], tt.WorldView(),
                               tt.config(playout_steps=4, playout_backend="native", playout_policy=POLICY),
                               research_seed=17)
        self.assertEqual(result.stats[0].mean, 0.75)
        step = lambda world, path: self.live(world, path[-1])
        v2 = StandardTreeSearch(step, self.root, lambda world: script(Branch(world)))
        cfg = tt.config(playout_steps=4, playout_backend="native", value_rule=V2, playout_policy=POLICY, max_depth=1)
        result = teacher.teach(v2, [0], self.view, cfg, research_seed=17, root=self.context)
        self.assertEqual(result.stats[0].visits, 8)
        self.assertAlmostEqual(result.stats[0].mean, (0.55 + 0.4 * 70 / 80 + 0.4 * 0.25) / 2)
        self.assertEqual(v2.counters()["live_branches"], 0)
        for search in (control, v2):
            self.assertEqual(len(search.playouts), 8)
            for _, options in search.playouts:
                self.assertEqual({key: value for key, value in options.items()
                                  if key not in ("max_steps", "playout_seed")}, TIERED_OPTIONS)
                self.assertEqual(options["max_steps"], 4)

        # A result under another rule is an error, not a score.
        class WrongRule(NativeSearch):
            def playout_many(self, branches, **options):
                return playout_result([summary()] * len(branches), rule_id=PLAYOUT_RULE_ID)

        with self.assertRaisesRegex(RuntimeError, TIERED_PLAYOUT_RULE_ID):
            playout_branches(WrongRule(None), [Branch(0)], max_steps=5, playout_policy=POLICY)

    def test_playout_rule_id_in_identity_and_provenance(self):
        from scripts.neural import search_teacher_entry as entry
        from tests.unit import test_search_teacher_entry as te

        self.assertEqual(playout_rule_id(POLICY), TIERED_PLAYOUT_RULE_ID)
        self.assertEqual(TIERED_PLAYOUT_RULE_ID, "spire-gym-public-playout-tiered-v1")
        with self.assertRaises(ValueError):
            playout_rule_id("other")
        for overrides in ({"playout_policy": POLICY}, {"value_rule": V2, "playout_policy": POLICY}):
            with self.subTest(overrides=overrides):
                native = teacher.config_identity(tt.config(playout_backend="native", **overrides))
                self.assertEqual(native["playout_rule_id"], TIERED_PLAYOUT_RULE_ID)
                self.assertNotIn("playout_rule_id", teacher.config_identity(tt.config(**overrides)))

        rows = _native_v2_rows(json.loads(te.TEACHERS.read_text())["teachers"]["tree-3x"])
        live = GYM_V1_2_5_CONTRACT.source_revision
        with tempfile.TemporaryDirectory() as directory:
            configs = te.v2_teachers(directory, **rows)
            one, one_sha, _ = entry._teacher(configs, "current-native-v2", None)
            self.assertEqual(one, {"samples": 32, "playout_steps": 80, "playout_backend": "native",
                                   "value_rule": V2, "playout_policy": POLICY})
            self.assertEqual(one_sha, content_sha256({
                "kind": "one-move", "samples": 32, "playout_steps": 80, "playout_backend": "native",
                "branch_layout": "compact", "workers": 1, "playout_rule_id": TIERED_PLAYOUT_RULE_ID,
                "value_rule": V2.as_identity(), "playout_policy": POLICY}))
            tree, tree_sha, _ = entry._teacher(configs, "tree-native-v2", None)
            self.assertEqual(teacher.config_identity(tree)["playout_rule_id"], TIERED_PLAYOUT_RULE_ID)
            self.assertEqual(tree_sha, teacher.config_sha256(tree))
            for name, digest in (("current-native-v2", one_sha), ("tree-native-v2", tree_sha)):
                with self.subTest(teacher=name):
                    summary_, _ = te.run_entry(directory, configs, name, allow_uncalibrated=True, revision=live)
                    provenance = summary_["provenance"]
                    self.assertEqual((provenance["playout_backend"], provenance["playout_rule_id"]),
                                     ("native", TIERED_PLAYOUT_RULE_ID))
                    self.assertEqual(provenance["config_sha256"], digest)
                    self.assertEqual(provenance["gym_revision"], live)
            # v1 on the v1.2.5 Gym keeps its rule id and hash.
            summary_, _ = te.run_entry(directory, te.TEACHERS, "current", revision=live)
            provenance = summary_["provenance"]
            self.assertEqual((provenance["playout_rule_id"], provenance["config_sha256"]),
                             (PLAYOUT_RULE_ID, te.CURRENT_SHA))
        self.contract.assert_not_called()

    def test_native_v2_root_step_terminal_scored_v2(self):
        root = fakes.observation(hp=50, max_hp=80, potions=("smoke_bomb", "block_potion"),
                                 verbs=("play_card", "use_potion"))
        context = RootContext.from_observation(self.view, root, V2)
        # A lethal Feed (+3 max HP; cleanup HP after Burning Blood) and a Smoke Bomb escape.
        fed = fakes.observation(hp=47, max_hp=83, cleanup=53, potions=("smoke_bomb", "block_potion"), enemies=())
        fled = fakes.observation(hp=50, max_hp=80, cleanup=50, potions=("block_potion",))
        steps = {0: (fed, 1.0, True, False, {"terminal_reason": "victory"}),
                 1: (fled, 0.0, True, False, {"terminal_reason": "escaped"})}
        no_playout = lambda branch: self.fail("a terminal root step needs no playout")
        search = StandardNativeSearch(no_playout, lambda sample, action: steps[action])
        stats = self.one_move(search, value_rule=V2, playout_policy=POLICY, root=context)
        win = terminal_value(V2, context, self.view, fed, 1.0, "victory")
        escape = terminal_value(V2, context, self.view, fled, 0.0, "escaped")
        self.assertAlmostEqual(win, 0.55 + 0.4 * 53 / 80 + 0.4 * 1.5 * 3 / 80)
        self.assertAlmostEqual(escape, -0.2 + 0.4 * 50 / 80 - 0.4 * 12 / 80)
        self.assertEqual(stats, {0: ActionStat((win + win + win) / 3, 3, 0),
                                 1: ActionStat((escape + escape + escape) / 3, 3, 0)})
        self.assertEqual((search.calls, search.views), ([], ["full"]))
        # flat-v1 keeps the reward.
        flat = self.one_move(StandardNativeSearch(no_playout, lambda sample, action: steps[action]),
                             playout_policy=POLICY)
        self.assertEqual((flat[0].mean, flat[1].mean), (1.0, 0.0))
        # A root cut-off scores the v2 cut-off value; a terminal without a reason raises.
        cut = fakes.observation(hp=40, max_hp=80, potions=("block_potion",), enemies=((10, 40),),
                                verbs=("play_card",))
        truncated = self.one_move(StandardNativeSearch(no_playout, lambda s, a: (cut, 0.0, False, True, {})),
                                  value_rule=V2, playout_policy=POLICY, root=context)
        expected = cutoff_value(V2, context, self.view, cut)
        self.assertEqual(truncated[0], ActionStat((expected + expected + expected) / 3, 3, 0))
        unexplained = StandardNativeSearch(no_playout, lambda s, a: (fed, 1.0, True, False, {}))
        with self.assertRaisesRegex(ValueError, "terminal_reason"):
            self.one_move(unexplained, value_rule=V2, playout_policy=POLICY, root=context)

    def test_native_v2_refused_without_tiered_contract(self):
        self.contract.return_value = GYM_V1_2_4_CONTRACT
        for options in ({"playout_policy": POLICY}, {"value_rule": V2, "playout_policy": POLICY, "root": self.context}):
            with self.subTest(options=options):
                search = StandardNativeSearch(lambda branch: summary(), self.live)
                with self.assertRaisesRegex(ValueError, TIERED_PLAYOUT_RULE_ID):
                    self.one_move(search, **options)
                self.assertEqual(search.calls, [])
                self.assertTrue(all(branch.closed for branch in search.branches))
        tree = NativeTeacherSearch(TeacherNativeTests.script, lambda world, call: summary())
        with self.assertRaisesRegex(ValueError, TIERED_PLAYOUT_RULE_ID):
            teacher.teach(tree, [0], tt.WorldView(),
                          tt.config(playout_steps=4, playout_backend="native", playout_policy=POLICY),
                          research_seed=17)
        self.assertEqual((tree.playouts, tree.counters()["live_branches"]), ([], 0))

        with self.assertRaisesRegex(ValueError, TIERED_PLAYOUT_RULE_ID):
            require_tiered_playouts(GYM_V1_2_4_CONTRACT)
        self.assertIsNone(require_tiered_playouts(GYM_V1_2_5_CONTRACT))
        # Config construction and the pure guards stay offline.
        with patch.dict(sys.modules, {"spire_lab": None}):
            for overrides in ({"playout_policy": POLICY}, {"value_rule": V2, "playout_policy": POLICY}):
                self.assertEqual(tt.config(playout_backend="native", **overrides).playout_backend, "native")

        # The entry refuses native v2 on a live Gym without the rule, before preflight and any output.
        from tests.unit import test_search_teacher_entry as te
        rows = _native_v2_rows(json.loads(te.TEACHERS.read_text())["teachers"]["tree-3x"])
        with tempfile.TemporaryDirectory() as directory:
            configs = te.v2_teachers(directory, **rows)
            for name in rows:
                with self.subTest(teacher=name):
                    with self.assertRaisesRegex(ValueError, TIERED_PLAYOUT_RULE_ID):
                        te.run_entry(directory, configs, name, allow_uncalibrated=True,
                                     revision=GYM_V1_2_4_CONTRACT.source_revision)
                    self.assertFalse((Path(directory) / f"output-{name}").exists())


class LiveContractHelperTests(unittest.TestCase):
    def test_helper_reads_the_live_build_revision(self):
        for contract in (GYM_V1_2_4_CONTRACT, GYM_V1_2_5_CONTRACT):
            gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: {"revision": contract.source_revision}))
            with patch.dict(sys.modules, {"spire_lab": gym}):
                # The uncached function: a process reads its one live Gym once.
                self.assertIs(search_view._native_contract.__wrapped__(), contract)


class SummaryScorerTests(unittest.TestCase):
    """Summary-row scoring equals observation scoring of the same public state."""

    def setUp(self):
        self.view = View(fakes.descriptor())
        self.root = fakes.observation(hp=60, max_hp=80, potions=ROOT_POTIONS)

    def test_summary_scoring_matches_observation_scoring(self):
        # (end reason, HP at the end (after cleanup when terminal), max HP, potions, enemies)
        ends = (
            ("victory", 70, 83, ("fairy_potion", "fruit_juice", "smoke_bomb"), ()),  # Feed, a potion gained
            ("victory", 70, 83, ("fruit_juice",), ()),                                # both spent, one gained
            ("victory", 80, 80, ROOT_POTIONS + ("swift_potion", "block_potion"), ()),  # gains capped at 1.0
            ("victory", 1, 80, (), ()),                                               # the 0.41 floor binds
            ("escaped", 40, 80, ("fairy_potion",), ((30, 50),)),                     # Smoke Bomb spent
            ("escaped", 2, 80, (), ((30, 50),)),
            ("death", 0, 80, ROOT_POTIONS, ((5, 50),)),
            ("step_limit", 30, 80, ROOT_POTIONS, ((10, 40), (0, 20))),
            ("max_steps", 95, 85, ("smoke_bomb", "fire_potion"), ((40, 40), (-3, 10))),
            ("max_steps", 20, 80, (), ()),
        )
        rewards = {"victory": 1, "death": -1, "escaped": 0}
        rules = (V2, ValueRule("resource-v2", hp_curve="sqrt"), ValueRule("resource-v2", potion_hp=15.0),
                 ValueRule("resource-v2", potion_hp=10.0, potion_overrides=(("fairy_potion", 30.0),)))
        for rule in rules:
            context = RootContext.from_observation(self.view, self.root, rule)
            for reason, hp, max_hp, potions, enemies in ends:
                with self.subTest(rule=rule, reason=reason, hp=hp, potions=potions):
                    terminal = reason in rewards
                    row = summary(player=(hp, max_hp), enemies=enemies, end_reason=reason,
                                  reward=rewards.get(reason, 0), terminated=terminal, potions=potions)
                    result = playout_result([summary(), row])
                    if terminal:
                        # The live hp column predates combat-end hooks; the summary reports HP after cleanup.
                        observation = fakes.observation(hp=max(hp - 6, 0), max_hp=max_hp, cleanup=hp,
                                                        potions=potions, enemies=enemies)
                        expected = terminal_value(rule, context, self.view, observation, rewards[reason], reason)
                        self.assertEqual(summary_terminal_value(rule, context, result, 1), expected)
                    else:
                        observation = fakes.observation(hp=hp, max_hp=max_hp, potions=potions, enemies=enemies,
                                                        verbs=("play_card",))
                        expected = cutoff_value(rule, context, self.view, observation)
                        self.assertEqual(summary_cutoff_value(rule, context, result, 1), expected)
        context = RootContext.from_observation(self.view, self.root, V2)
        result = playout_result([summary(player=(70, 83), end_reason="victory", reward=1, terminated=True,
                                         potions=("fruit_juice",))])
        # 0.55 + 0.4 * 70/80 + 0.4 * (1.5 * 3 - (30 + 12 - 7.5)) / 80
        self.assertAlmostEqual(summary_terminal_value(V2, context, result, 0), 0.55 + 0.35 - 0.4 * 30 / 80)

    def test_summary_reason_validation(self):
        context = RootContext.from_observation(self.view, self.root, V2)
        row = lambda **fields: playout_result([summary(**fields)])
        for fields in (dict(end_reason="victory", reward=1, terminated=True),
                       dict(end_reason="death", reward=-1, terminated=True, player=(0, 80)),
                       dict(end_reason="escaped", reward=0, terminated=True)):
            self.assertIsInstance(summary_terminal_value(V2, context, row(**fields), 0), float)
        for reason in ("step_limit", "max_steps"):
            self.assertIsInstance(summary_cutoff_value(V2, context, row(end_reason=reason), 0), float)
        bad_terminal = (
            dict(end_reason="step_limit", terminated=True), dict(end_reason="max_steps", terminated=True),
            dict(end_reason="victory", reward=0, terminated=True), dict(end_reason="death", reward=1, terminated=True),
            dict(end_reason="escaped", reward=-1, terminated=True), dict(end_reason="refused", terminated=True),
            dict(end_reason="fault", terminated=True), dict(end_reason="bogus", terminated=True),
            dict(end_reason="victory", reward=1, terminated=False),
        )
        for fields in bad_terminal:
            with self.subTest(scorer="terminal", **fields), self.assertRaises(ValueError):
                summary_terminal_value(V2, context, row(**fields), 0)
        bad_live = (
            dict(end_reason="victory", reward=1), dict(end_reason="death", reward=-1), dict(end_reason="escaped"),
            dict(end_reason="refused"), dict(end_reason="fault"), dict(end_reason="bogus"),
            dict(end_reason="step_limit", terminated=True),
        )
        for fields in bad_live:
            with self.subTest(scorer="cutoff", **fields), self.assertRaises(ValueError):
                summary_cutoff_value(V2, context, row(**fields), 0)
        # flat-v1 scores native summaries on its v1 path, never here.
        with self.assertRaises(ValueError):
            summary_terminal_value(FLAT, context, row(end_reason="victory", reward=1, terminated=True), 0)
        with self.assertRaises(ValueError):
            summary_cutoff_value(FLAT, context, row(), 0)

        # Through playout_branches: a malformed row raises; a refusal scores -1, flagged.
        with patch.object(search_view, "_native_errors", return_value=(Unsupported, ValueError, Fault)), \
             patch.object(search_view, "_native_contract", return_value=GYM_V1_2_5_CONTRACT):
            options = dict(max_steps=5, value_rule=V2, playout_policy=POLICY, root=context)
            malformed = NativeSearch(lambda branch: summary(end_reason="max_steps", terminated=True))
            with self.assertRaises(ValueError):
                playout_branches(malformed, [Branch(0)], **options)
            refused = NativeSearch(lambda branch: summary(end_reason="refused"))
            self.assertEqual(playout_branches(refused, [Branch(0)], **options), [PlayoutOutcome(REFUSED_VALUE, True)])
            with self.assertRaisesRegex(ValueError, "root context"):
                playout_branches(refused, [Branch(0)], **{**options, "root": None})


if __name__ == "__main__":
    unittest.main()
