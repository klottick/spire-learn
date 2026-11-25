"""Native playout backend with scripted playout_many results (no Gym runtime)."""

from dataclasses import asdict
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from spire_learn.neural import search_teacher as teacher
from spire_learn.neural.search_one_move import (
    PLAYOUT_EXCLUDE_VERBS, PLAYOUT_RULE_ID, ActionStat, one_move_values, playout_seed,
)
from spire_learn.neural.search_view import summary_balance
from tests.unit.test_search_one_move import _View
from tests.unit import test_search_teacher as tt


class Unsupported(Exception):
    pass


class Fault(Exception):
    pass


def summary(player=(50, 100), enemies=((10, 20),), end_reason="step_limit", reward=0, terminated=False):
    return {"end_reason": end_reason, "reward": reward, "terminated": terminated,
            "player": player, "enemies": enemies}


def playout_result(rows, faults=None):
    width = max(len(row["enemies"]) for row in rows) or 1
    pad = lambda values, fill: list(values) + [fill] * (width - len(values))
    return {
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
        return playout_result([self.script(branch) for branch in branches])


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
        return playout_result([self.results(branches[0].world, len(self.playouts) - 1)] * len(branches))


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
    def test_gym_1_2_4_teachers_are_native_and_uncalibrated(self):
        from scripts.neural import search_teacher_entry as entry

        new = entry._ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json"
        old = entry._ROOT / "configs/neural/search-teacher-gym-1.2.0/teachers.json"
        for name in ("current", "tree-1x", "tree-3x", "tree-10x"):
            config, digest, calibrated = entry._teacher(new, name, None)
            _, old_digest, _ = entry._teacher(old, name, None)
            self.assertNotEqual(digest, old_digest)
            backend = config["playout_backend"] if name == "current" else config.playout_backend
            self.assertEqual(backend, "native")
            self.assertFalse(calibrated)
            self.assertTrue(entry._native_backend(config))
            with self.assertRaisesRegex(ValueError, "uncalibrated"):
                entry._teacher(new, name, "selection")
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


if __name__ == "__main__":
    unittest.main()
