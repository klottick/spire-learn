"""Value rule and playout policy wiring in the tree and one-move teachers."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from spire_learn.neural import search_teacher as teacher
from spire_learn.neural.search_one_move import ActionStat, check_value_options, one_move_values
from spire_learn.neural.search_value import RootContext, ValueRule
from spire_learn.neural.search_view import View
from tests.unit import _search_fakes as fakes
from tests.unit import test_search_native_playouts as nt
from tests.unit import test_search_one_move as om
from tests.unit import test_search_teacher as tt

V2 = ValueRule("resource-v2")
FLAT = ValueRule("flat-v1")
POLICY = "no-potions-v2"


def _stats(result):
    return {action: (stat.mean.hex(), stat.visits, stat.refused) for action, stat in result.items()}


def _tree(result):
    return result.choice, _stats(result.stats), result.simulations, result.refused


def _v1_scenarios(explicit=False):
    """v1 teachers on the existing fakes; ``explicit`` passes the v1 defaults by name."""
    view = tt.WorldView()
    out = {}
    extra = dict(value_rule=FLAT, playout_policy="uniform-v1") if explicit else {}
    options = dict(extra, root=None) if explicit else {}

    def config(**overrides):
        return tt.config(**overrides, **extra)

    def teach(search, cfg, legal=None, research_seed=17):
        legal = view.legal(search.root) if legal is None else legal
        root = {"root": None} if explicit else {}
        return _tree(teacher.teach(search, legal, view, cfg, research_seed=research_seed, **root))

    errors = (tt.Unsupported, ValueError, tt.Fault)
    with patch.object(teacher.search_view, "_native_errors", return_value=errors):
        out["finish"] = teach(tt.Search(tt.finish, 3), config())
        out["finish-19"] = teach(tt.Search(tt.finish, 3), config(), research_seed=19)

        def refusing(world, path):
            if world == 1:
                raise tt.Unsupported("hidden order")
            return tt.finish(world, path)
        out["refusal"] = teach(tt.Search(refusing), config())

        def repeat(world, path):
            if len(path) == 1:
                return tt.observation(1), 0.0, False
            if len(path) == 2:
                return tt.observation(2, (1,) * 32), 0.0, False
            return tt.observation(9, (0,)), path[-1] / 32, True
        out["repeat"] = teach(tt.Search(repeat), config(simulations=6, worlds=1, playout_steps=1))

        def deep(world, path):
            if len(path) < 3:
                return tt.observation(len(path), (1, 1)), 0, False
            return tt.finish(world, path)
        out["deep"] = teach(tt.Search(deep, 2), config(max_depth=3, simulations=24, worlds=3))

        def uniform(world, path):
            if len(path) == 1:
                return tt.observation(1, (1, 1, 1)), 0, False
            if world == 1 and path[0] == 2:
                raise tt.Unsupported("hidden order")
            return tt.observation(9, (0,)), (world + path[0] + path[1]) / 7, True
        out["uniform"] = teach(tt.Search(uniform, 3), config(max_depth=1, simulations=12, worlds=4,
                                                              playout_steps=2))
        out["per-action"] = teach(tt.Search(tt.finish, 5),
                                  config(simulations=None, simulations_per_action=4))

        results = lambda world, call: nt.summary(end_reason="victory", reward=1, terminated=True) \
            if world == 0 else nt.summary(player=(100, 100), enemies=((0, 10),))
        native = nt.NativeTeacherSearch(nt.TeacherNativeTests.script, results)
        out["native"] = teach(native, config(playout_steps=4, playout_backend="native"), legal=[0])
        out["native-calls"] = [(world, opts["playout_seed"]) for world, opts in native.playouts]

    errors = (om._Errors.Unsupported, om._Errors.State, om._Errors.Fault)
    with patch("spire_learn.neural.search_view._native_errors", return_value=errors):
        params = dict(samples=3, playout_steps=4, research_seed=17, **options)
        out["one-move"] = _stats(one_move_values(om._ScriptedSearch({}), [0, 1], om._View(), **params))
        outcomes = {(action, sample): 0.0 for action in (0, 1) for sample in range(3)}
        outcomes[1, 0] = om._Errors.Unsupported()
        out["one-move-refusal"] = _stats(one_move_values(om._ScriptedSearch(outcomes), [0, 1], om._View(),
                                                         **params))

    with patch("spire_learn.neural.search_view._native_errors",
               return_value=(nt.Unsupported, ValueError, nt.Fault)):
        def script(branch):
            if branch.sample == 0:
                return nt.summary(end_reason="victory", reward=1, terminated=True)
            if branch.sample == 1:
                return nt.summary(end_reason="death", reward=-1, terminated=True)
            return nt.summary(player=(50, 100), enemies=((10, 25),))
        search = nt.NativeSearch(script)
        out["one-move-native"] = _stats(one_move_values(search, [0, 1], om._View(), samples=3, playout_steps=5,
                                                        research_seed=17, playout_backend="native", **options))
        out["one-move-native-calls"] = [opts["playout_seed"] for _, opts in search.calls]
    return out


# Captured at 12bd205, before the value rule and playout policy reached the teachers.
_V1_GOLDEN = {
    "finish": (2, {0: ("-0x1.0000000000000p-1", 1, 0), 1: ("0x0.0p+0", 2, 0),
                   2: ("0x1.0000000000000p-1", 5, 0)}, 8, 0),
    "finish-19": (2, {0: ("-0x1.0000000000000p-1", 1, 0), 1: ("0x0.0p+0", 2, 0),
                      2: ("0x1.0000000000000p-1", 5, 0)}, 8, 0),
    "refusal": (0, {0: ("-0x1.8000000000000p-1", 8, 4)}, 8, 4),
    "repeat": (0, {0: ("0x1.05ddddddddddep-1", 6, 0)}, 6, 0),
    "deep": (1, {0: ("-0x1.8f5c28f5c28f6p-2", 5, 0), 1: ("0x1.58ed2308158edp-8", 19, 0)}, 24, 0),
    "uniform": (1, {0: ("0x1.b6db6db6db6dcp-2", 4, 0), 1: ("0x1.2492492492492p-1", 4, 0),
                    2: ("0x1.6db6db6db6db6p-2", 4, 1)}, 12, 1),
    "per-action": (4, {0: ("-0x1.0000000000000p-1", 1, 0), 1: ("0x0.0p+0", 1, 0),
                       2: ("0x1.0000000000000p-1", 2, 0), 3: ("0x1.0000000000000p+0", 3, 0),
                       4: ("0x1.8000000000000p+0", 13, 0)}, 20, 0),
    "native": (0, {0: ("0x1.8000000000000p-1", 8, 0)}, 8, 0),
    "native-calls": [(0, 0), (1, 0), (0, 0), (1, 9935881281234371769), (0, 8836915395921057621),
                     (1, 5569434317643868827), (0, 5517080055182042600), (1, 2434911948756495634)],
    "one-move": {0: ("0x1.27350b8812755p-8", 3, 0), 1: ("-0x1.4c1bacf914c1cp-2", 3, 0)},
    "one-move-refusal": {0: ("0x0.0p+0", 3, 0), 1: ("-0x1.5555555555555p-2", 3, 1)},
    "one-move-native": {0: ("0x1.1111111111110p-6", 3, 0), 1: ("0x1.1111111111110p-6", 3, 0)},
    "one-move-native-calls": [0],
}

# config_sha256 at 12bd205 of the gym-1.2.4 tree teachers and of the test configs.
_V1_HASHES = {
    "tree-1x": "602c1b5db5d5f726c076dccf22f86f43e46b239e0892386e87e76dcf5aae476f",
    "tree-3x": "7db31be63376a5a16745e77a8cfb4d83e7cc9ed408a3370a774696b556558eef",
    "tree-10x": "2faa984d179a7b12772619757e40c46bdb518eabd817e084a8c88bd624c5cfe8",
}
_V1_TEST_HASHES = {
    "python": "86d67ce9b5e30a1dbb7b090f65bfa5451466a44c41f93784c11dc2517e780846",
    "native": "904333d5dc1b831dfa7a970d75bd6b65e9923b098138c5cfe733ec1e58e5ebc8",
    "per-action": "308516d8e1ff63bce01b41f2e9d013d56ef3430b6f7ad3f0441f83bbd48e8e19",
}
_TEACHERS = Path(teacher.__file__).parents[3] / "configs/neural/search-teacher-gym-1.2.4/teachers.json"


def _tree_config(name):
    config = json.loads(_TEACHERS.read_text())["teachers"][name]["config"]
    return teacher.TeacherConfig(**{**config, "widening": tuple(config["widening"])})


class _Branch(tt.Branch):
    def step(self, action, *, view):
        assert not self.closed and view == "full"
        self.path += (action,)
        return self.search.script(self.world, self.path)


class _Search(tt.Search):
    """Scripted worlds over Standard-shaped observations; the script returns the full step tuple."""

    def __init__(self, script, root):
        super().__init__(script)
        self.root = root

    def open(self, world, path):
        branch = _Branch(self, world, path)
        self.handles.append(branch)
        return branch


def _key(observation):
    return repr({name: list(tensor) for name, tensor in sorted(observation.items())}).encode()


class V1PreservationTests(unittest.TestCase):
    def test_v1_teachers_are_bit_identical(self):
        for name, digest in _V1_HASHES.items():
            with self.subTest(teacher=name):
                self.assertEqual(teacher.config_sha256(_tree_config(name)), digest)
        configs = {"python": tt.config(), "native": tt.config(playout_backend="native"),
                   "per-action": tt.config(simulations=None, simulations_per_action=4)}
        for name, cfg in configs.items():
            with self.subTest(config=name):
                self.assertEqual(teacher.config_sha256(cfg), _V1_TEST_HASHES[name])
                explicit = teacher.TeacherConfig(**{**asdict(cfg), "value_rule": FLAT,
                                                    "playout_policy": "uniform-v1"})
                self.assertEqual(teacher.config_sha256(explicit), _V1_TEST_HASHES[name])
                identity = teacher.config_identity(cfg)
                self.assertNotIn("value_rule", identity)
                self.assertNotIn("playout_policy", identity)
        self.assertEqual(_v1_scenarios(), _V1_GOLDEN)
        self.assertEqual(_v1_scenarios(explicit=True), _V1_GOLDEN)

    def test_v2_identity_adds_rule_and_policy(self):
        v2 = tt.config(value_rule=V2, playout_policy=POLICY)
        identity = teacher.config_identity(v2)
        self.assertEqual(identity["value_rule"], V2.as_identity())
        self.assertEqual(identity["playout_policy"], POLICY)
        control = teacher.config_identity(tt.config(playout_policy=POLICY))
        self.assertNotIn("value_rule", control)
        self.assertEqual(control["playout_policy"], POLICY)
        sqrt = tt.config(value_rule=ValueRule("resource-v2", hp_curve="sqrt"), playout_policy=POLICY)
        hashes = {teacher.config_sha256(cfg) for cfg in (tt.config(), tt.config(playout_policy=POLICY), v2, sqrt)}
        self.assertEqual(len(hashes), 4)


class V2WiringTests(unittest.TestCase):
    def setUp(self):
        self.view = View(fakes.descriptor())
        for patcher in (patch.object(teacher.search_view, "_native_errors",
                                     return_value=(tt.Unsupported, ValueError, tt.Fault)),
                        patch.object(teacher, "_public_key", _key)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def context(self, observation, rule=V2):
        return RootContext.from_observation(self.view, observation, rule)

    def teach(self, script, root_observation, cfg, root=None):
        search = _Search(script, root_observation)
        result = teacher.teach(search, self.view.legal(root_observation), self.view, cfg,
                               research_seed=17, root=root)
        self.assertEqual(search.counters()["live_branches"], 0)
        return result

    def one_move(self, script, root_observation, **options):
        search = _Search(script, root_observation)
        values = one_move_values(search, self.view.legal(root_observation), self.view, samples=3,
                                 playout_steps=3, research_seed=17, **options)
        self.assertTrue(all(branch.closed for branch in search.handles))
        return values

    def test_v2_guards(self):
        native = dict(playout_backend="native")
        # Native v2 configs are valid offline: the tiered rule is checked against a live Gym, never here.
        with patch.dict(sys.modules, {"spire_lab": None}):
            for overrides in ({"value_rule": V2, "playout_policy": POLICY, **native},
                              {"playout_policy": POLICY, **native}):
                with self.subTest(overrides=overrides):
                    self.assertEqual(tt.config(**overrides).playout_backend, "native")
        for overrides in ({"value_rule": V2}, {"value_rule": V2, **native}, {"playout_policy": "other"},
                          {"value_rule": "resource-v2"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                tt.config(**overrides)
        # resource-v2 needs the full branch view on both backends; the policy alone, on Python only.
        for backend in ("python", "native"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "full"):
                check_value_options(V2, POLICY, backend, "mask")
        with self.assertRaisesRegex(ValueError, "full"):
            check_value_options(FLAT, POLICY, "python", "mask")
        self.assertIsNone(check_value_options(FLAT, POLICY, "native", "mask"))

        root = fakes.observation(verbs=("play_card",))
        script = lambda world, path: (fakes.observation(), 1.0, True, False, {"terminal_reason": "victory"})
        with self.assertRaisesRegex(ValueError, "root context"):
            self.teach(script, root, tt.config(value_rule=V2, playout_policy=POLICY))

        context = self.context(root)
        cases = (
            ({"value_rule": V2, "playout_policy": POLICY, "root": context, "branch_view": "mask", **native}, "full"),
            ({"value_rule": V2, "playout_policy": POLICY, **native}, "root context"),
            ({"value_rule": V2, "root": context, **native}, POLICY),
            ({"playout_policy": POLICY, "branch_view": "mask"}, "full"),
            ({"value_rule": V2, "playout_policy": POLICY}, "root context"),
            ({"value_rule": V2, "root": context}, POLICY),
            ({"playout_policy": "other"}, "playout_policy"),
        )
        for options, message in cases:
            with self.subTest(options=options):
                search = _Search(script, root)
                with self.assertRaisesRegex(ValueError, message):
                    one_move_values(search, [0], self.view, samples=2, playout_steps=3, research_seed=17,
                                    **options)
                self.assertEqual(search.handles, [])

    def test_v2_charges_potion_used_on_search_path(self):
        max_hp = 80
        root = fakes.observation(hp=max_hp, max_hp=max_hp, potions=["block_potion"], verbs=("use_potion",))
        drunk = fakes.observation(hp=max_hp, max_hp=max_hp, verbs=("play_card", "use_potion", "end_turn"))

        def script(world, path):
            if len(path) == 1:
                return drunk, 0.0, False, False, {}
            if path[-1] != 0:
                return fakes.observation(hp=0, max_hp=max_hp), -1.0, True, False, {"terminal_reason": "death"}
            won = fakes.observation(hp=max_hp, max_hp=max_hp, cleanup=max_hp, enemies=())
            return won, 1.0, True, False, {"terminal_reason": "victory"}

        expected = 0.95 - 0.4 * 12 / max_hp
        # Depth one, so every visit plays out under the policy.
        cfg = tt.config(value_rule=V2, playout_policy=POLICY, playout_steps=3, max_depth=1)
        result = self.teach(script, root, cfg, root=self.context(root))
        self.assertEqual(result.stats[0].visits, 8)
        self.assertAlmostEqual(result.stats[0].mean, expected)
        values = self.one_move(script, root, value_rule=V2, playout_policy=POLICY, root=self.context(root))
        self.assertAlmostEqual(values[0].mean, expected)
        self.assertEqual(values[0].refused, 0)

    def test_v2_terminal_reason_from_info(self):
        root = fakes.observation(hp=80, max_hp=80, verbs=("play_card",))
        ends = {
            "escaped": (fakes.observation(hp=60, max_hp=80, cleanup=60), 0.0, True, False,
                        {"terminal_reason": "escaped"}, -0.2 + 0.4 * 0.75),
            # The reason, not the reward, decides the band.
            "victory": (fakes.observation(hp=60, max_hp=80, cleanup=60), 0.0, True, False,
                        {"terminal_reason": "victory"}, 0.55 + 0.4 * 0.75),
            "death": (fakes.observation(hp=0, max_hp=80), 0.0, True, False, {"terminal_reason": "death"}, -1.0),
            "cut-off": (fakes.observation(hp=40, max_hp=80, enemies=((10, 40),), verbs=("play_card",)), 0.0,
                        False, True, {}, 0.4 * (0.5 - 0.25)),
        }
        cfg = tt.config(value_rule=V2, playout_policy=POLICY, max_depth=1, simulations=2)
        for name, (*step, expected) in ends.items():
            with self.subTest(end=name):
                script = lambda world, path, step=tuple(step): step
                result = self.teach(script, root, cfg, root=self.context(root))
                self.assertAlmostEqual(result.stats[0].mean, expected)
                values = self.one_move(script, root, value_rule=V2, playout_policy=POLICY,
                                       root=self.context(root))
                self.assertAlmostEqual(values[0].mean, expected)

        missing = lambda world, path: (fakes.observation(cleanup=50), 1.0, True, False, {})
        with self.assertRaisesRegex(ValueError, "terminal_reason"):
            self.teach(missing, root, cfg, root=self.context(root))
        with self.assertRaisesRegex(ValueError, "terminal_reason"):
            self.one_move(missing, root, value_rule=V2, playout_policy=POLICY, root=self.context(root))
        # v1 scoring needs no reason.
        self.assertEqual(self.teach(missing, root, tt.config(max_depth=1, simulations=2)).stats[0].mean, 1.0)

    def test_flat_v1_with_no_potions_policy(self):
        root = fakes.observation(verbs=("play_card",))
        live = fakes.observation(verbs=("use_potion", "play_card", "discard_potion"))

        def script(world, path):
            if len(path) == 1:
                return live, 0.0, False, False, {}
            if path[-1] == 1:
                return fakes.observation(enemies=()), 1.0, True, False, {}
            return fakes.observation(hp=0), -1.0, True, False, {}

        control = tt.config(playout_policy=POLICY, playout_steps=3, max_depth=1)
        self.assertEqual(self.teach(script, root, control).stats[0], ActionStat(1.0, 8, 0))
        self.assertEqual(self.one_move(script, root, playout_policy=POLICY)[0], ActionStat(1.0, 3, 0))
        # The uniform-v1 heuristic drinks the potion in some playouts.
        uniform = tt.config(playout_steps=3, max_depth=1)
        self.assertLess(self.teach(script, root, uniform).stats[0].mean, 1.0)
        # A cut-off keeps the flat half balance.
        balance = 0.5 * self.view.balance(live)
        cut = tt.config(playout_policy=POLICY, playout_steps=0, max_depth=1, simulations=2)
        self.assertEqual(self.teach(script, root, cut).stats[0].mean, balance)


if __name__ == "__main__":
    unittest.main()
