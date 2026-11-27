"""The full-run harness's stdlib parts: the run panel, the priority list pin, Ruling J and counted fallbacks."""

from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import full_run_entry as full
from spire_learn.neural import search_view
from spire_learn.neural.cases import environment_seed_for
from spire_learn.neural.noncombat_heuristic import CardPriority, NoncombatHeuristic, RoomFlags
from spire_learn.neural.standard_stream import PANEL_SEED_OFFSET
from tests.unit.test_noncombat_heuristic import DESCRIPTOR, Boundary

ROOT = Path(__file__).resolve().parents[2]


class RunPanelTests(unittest.TestCase):
    def test_run_panel_seeds_stable(self):
        seeds = full.panel_seeds()
        self.assertEqual(len(seeds), full.RUN_PANEL_SIZE)
        self.assertEqual((full.RUN_PANEL_SIZE, full.RUN_PANEL_SEED_OFFSET), (64, 2 ** 34))
        self.assertEqual((full.TURN_CAP, full.FIGHT_CAP, full.RUN_CAP), (500, 1024, 10_000))
        self.assertEqual(seeds[0], 2 ** 40 + 2 ** 34)
        self.assertEqual(seeds, [environment_seed_for("selection", replica=0, ordinal=2 ** 34 + index)
                                 for index in range(64)])
        self.assertEqual(full.panel_identity(), "098eb73c9968a36db773b76e88027ba2400c446b94c32bf0ea9e92fda35a2b58")
        self.assertEqual(full.run_id(3), "run-03")

    def test_run_panel_seeds_disjoint_from_panel_ordinals(self):
        ordinals = [full.run_ordinal(index) for index in range(full.RUN_PANEL_SIZE)]
        smoke = [full.run_ordinal(index, smoke=True) for index in range(full.CHUNK_RUNS)]
        # Shard ordinals (index * 256 + local) stay below 2**32, the M10 panel sits at 2**32 + n and the
        # combat panels at 2**33 + n, each with far fewer than 2**32 cases.
        self.assertGreaterEqual(min(ordinals), 2 ** 33 + 2 ** 32)
        self.assertEqual(PANEL_SEED_OFFSET, 2 ** 33)
        self.assertGreater(min(ordinals), PANEL_SEED_OFFSET + 2 ** 32)
        self.assertFalse(set(ordinals) & set(smoke))
        seeds = full.panel_seeds() + [full.run_seed(index, smoke=True) for index in range(full.CHUNK_RUNS)]
        self.assertEqual(len(set(seeds)), len(seeds))
        selection_start, final_start = 2 ** 40, 2 ** 48
        self.assertTrue(all(selection_start <= seed < final_start for seed in seeds))
        combat_panel = {environment_seed_for("selection", replica=0, ordinal=PANEL_SEED_OFFSET + n) for n in range(4096)}
        m10_panel = {environment_seed_for("selection", replica=0, ordinal=2 ** 32 + n) for n in range(4096)}
        self.assertFalse(set(seeds) & (combat_panel | m10_panel))


class PriorityTests(unittest.TestCase):
    def test_priority_list_sha256_pinned(self):
        path = ROOT / full.PRIORITY_PATH
        # The pre-registered list for gate R1: the file never changes once committed, and the entry pins it.
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), full.PRIORITY_SHA256)
        self.assertEqual(CardPriority.load(path).sha256, full.PRIORITY_SHA256)


class FakeUnsupported(Exception):
    pass


class FakeState(Exception):
    pass


class FakeFault(Exception):
    pass


class FakeRun:
    def __init__(self, error):
        self.error = error

    def public_search(self, research_seed, **options):
        raise self.error


class RulingJTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(search_view, "_native_errors", return_value=(FakeUnsupported, FakeState, FakeFault))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_documented_root_refusal_stays_a_refusal(self):
        refusal = "hidden intents are not modelled"
        self.assertIn(refusal, search_view.ROOT_REFUSAL_CLASSES)
        with self.assertRaises(FakeUnsupported):
            full.RunSearchRoot(FakeRun(FakeUnsupported(refusal))).public_search(1)

    def test_state_error_and_unknown_refusal_are_faults(self):
        for error in (FakeState("the Run session is terminal"), FakeUnsupported("a brand new refusal class")):
            with self.subTest(error=type(error).__name__), self.assertRaises(full.RunFault):
                full.RunSearchRoot(FakeRun(error)).public_search(1)

    def test_decision_plays_the_fallback_only_on_a_documented_refusal(self):
        """``_decision`` through the Run root: a documented refusal plays ``playout_choice``; a state error raises."""
        from scripts.neural import search_teacher_entry as entry

        class View:
            def playout_choice(self, observation, rng, policy):
                return 7

        teacher = {"samples": 32, "playout_steps": 80, "playout_policy": "no-potions-v2"}
        refused = entry._decision(full.RunSearchRoot(FakeRun(FakeUnsupported("the combat has ended"))),
                                  None, [3, 7], View(), teacher, None, 1)
        self.assertEqual(refused[:2], ("the combat has ended", 7))
        with self.assertRaises(full.RunFault):
            entry._decision(full.RunSearchRoot(FakeRun(FakeState("stale token"))), None, [3, 7], View(), teacher,
                            None, 1)


class FallbackCountTests(unittest.TestCase):
    def test_fallback_counted_not_silent(self):
        heuristic = NoncombatHeuristic(DESCRIPTOR, CardPriority.load(ROOT / full.PRIORITY_PATH))
        rules, fallbacks, flags = Counter(), Counter(), RoomFlags()
        boundary = Boundary("act2_ready")
        boundary.choose("joust_owner")
        boundary.choose("joust_murderer")
        action = full.noncombat_move(heuristic, boundary.obs, boundary.legal, flags, rules, fallbacks)
        self.assertEqual(action, 0)
        self.assertEqual(fallbacks, Counter({"fallback:act2_ready": 1}))
        self.assertEqual(rules, Counter({"fallback:act2_ready": 1}))
        boundary = Boundary("rewards")
        gold = boundary.choose("gold", amount=10)
        self.assertEqual(full.noncombat_move(heuristic, boundary.obs, boundary.legal, flags, rules, fallbacks), gold)
        self.assertEqual(sum(fallbacks.values()), 1)
        self.assertEqual(rules["rewards:gold"], 1)
        # The run rows' fallback counts reach the summary's aggregates.
        run = {"floors_reached": 7, "victory": False, "terminal_reason": "death", "death": {"act": 1, "floor": 7},
               "censored": False, "censor_reason": None, "decisions": 9, "combat_decisions": 5, "fights": 1,
               "refused_roots": 0, "branch_refusals": 0, "fallback_total": 1, "potions_used": 0,
               "refusal_classes": {}, "fallbacks": dict(fallbacks)}
        aggregate = full._aggregate([run])
        self.assertEqual(aggregate["totals"]["fallbacks"], 1)
        self.assertEqual(aggregate["fallback_rules"], {"fallback:act2_ready": 1})


def _lifecycle(*, active, visit=None, entered, settled, last_exit=None):
    return {"combat_view_id": "spire-gym-standard-combat-view-v1", "active": active,
            "routing": "combat" if active else "noncombat", "selection_owner": None,
            "active_visit": visit, "combats_entered": entered, "combats_settled": settled,
            "last_exit": last_exit, "phase": "decision" if active else None, "selection_resume_phase": None}


def _exit(visit, hp):
    return {"visit": visit, "encounter": "cultist", "outcome": "victory", "hp_after_cleanup": hp,
            "hp_before_healing": hp}


def _combat_frame(hp, max_hp, potions=(), turn=1):
    """A minimal fake compact combat frame: legal actions 0 and 1, player HP, max HP and turn, potion kinds."""
    player, known = [0] * 13, [0] * 13
    for column, value in ((0, hp), (1, max_hp), (6, turn)):
        player[column], known[column] = value, 1
    return {"action_mask": [1, 1, 0], "player": player, "player_known": known, "potions": list(potions),
            "actions": [[8], [10], [0]]}


class FakeView:
    verb, use_potion, discard = 0, 6, 7

    def legal(self, observation):
        return [i for i, flag in enumerate(observation["action_mask"]) if flag]

    def player_hp_now(self, observation):
        return observation["player"][0]

    def player_max_hp(self, observation):
        return observation["player"][1]

    def potion_kinds(self, observation):
        return tuple(sorted(observation["potions"]))

    def potion_price_table(self, rule):
        return {1: 12.0, 2: 12.0}


class FakeBridge:
    def arrays(self, dense):
        return dense

    def __call__(self, dense):
        return dense, None


class FakePlayer:
    def act(self, frame, legal):
        return legal[0]


class ScriptedRun:
    """A fake Run: (Run observation, combat view, step outcome) per boundary, played in order."""

    def __init__(self, boundaries, terminal):
        self.boundaries, self.terminal, self.at = boundaries, terminal, 0
        self.actions = []

    def reset(self, seed):
        self.at = 0
        return self.boundaries[0][0], {"creation": {"neow": "full", "seen_bosses": [[1, 2, 3]] * 3}}

    def combat_view(self):
        return self.boundaries[self.at][1]

    def step(self, action):
        self.actions.append(action)
        self.at += 1
        if self.at == len(self.boundaries):
            self.at -= 1
            return self.terminal[0], 0.0, True, False, {"terminal_reason": self.terminal[1]}
        return self.boundaries[self.at][0], 0.0, False, False, {}


class FightEndStateTests(unittest.TestCase):
    def test_combat_followed_directly_by_combat_keeps_its_own_end_state(self):
        heuristic = NoncombatHeuristic(DESCRIPTOR, CardPriority.load(ROOT / full.PRIORITY_PATH))

        def combat_obs(floor):
            boundary = Boundary("combat", floor=floor)
            for index in (0, 1):
                boundary.obs["action_mask"][index] = 1
            return boundary.obs

        rewards = Boundary("rewards", floor=2)
        rewards.choose("proceed")
        boundaries = [
            (combat_obs(1), {"obs": _combat_frame(50, 80, [1]), "info": _lifecycle(active=True, visit=0, entered=1,
                                                                                 settled=0)}),
            # The next combat's first boundary, straight after the first fight settled (no room between).
            (combat_obs(1), {"obs": _combat_frame(48, 80, []),
                             "info": _lifecycle(active=True, visit=1, entered=2, settled=1, last_exit=_exit(0, 47))}),
            (rewards.obs, {"obs": _combat_frame(40, 82, []),
                           "info": _lifecycle(active=False, entered=2, settled=2, last_exit=_exit(1, 40))}),
        ]
        run = ScriptedRun(boundaries, (Boundary("map", floor=2).obs, "death"))
        context = full.PlayContext(player_kind="checkpoint", teacher=None, checkpoint="fake", mapping=None,
                                   priority_path=str(ROOT / full.PRIORITY_PATH), gym_revision="fake")
        job = full.RunJob(index=0, seed=1, run_id="run-00", research_seed=1)
        with patch.object(full, "_resources", return_value=(FakeBridge(), FakeView(), heuristic, FakePlayer(), 6)):
            result = full.play_run(run, job, context)
        first, second = result["fights"]
        self.assertEqual((first["outcome"], first["hp_after_cleanup"], first["end_state"]), ("victory", 47, "unavailable"))
        self.assertEqual((first["max_hp_end"], first["potions_end"], first["potion_hp_spent"]), (None, None, None))
        self.assertEqual((first["hp_start"], first["max_hp_start"], first["potions_start"]), (50, 80, 1))
        self.assertEqual((second["hp_after_cleanup"], second["end_state"]), (40, "settled"))
        self.assertEqual((second["max_hp_end"], second["potions_end"], second["potion_hp_spent"]), (82, 0, 0.0))
        self.assertEqual(result["run"]["fights"], 2)
        self.assertEqual(result["run"]["roots"], {"policy": 2})
        self.assertEqual(result["run"]["terminal_reason"], "death")
        self.assertEqual(run.actions[:2], [0, 0])


class CommandLineTests(unittest.TestCase):
    TEACHER = ("--combat-player", "teacher:current-v2", "--teachers",
               "configs/neural/search-teacher-gym-1.2.4/teachers-v2.json", "--seed", "1")

    def setUp(self):
        # A temporary output root, never the repository's artifacts tree.
        temp = tempfile.TemporaryDirectory(prefix="full-run-root-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target, value in ((full, "OUTPUT_ROOT"), ):
            patcher = patch.object(target, value, self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Anything past argument checks would load the teacher or the Gym: fail loudly if reached.
        for name in ("run_chunk", "_check_gym"):
            patcher = patch.object(full, name, side_effect=AssertionError(f"{name} reached"))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(full.entry, "_teacher", side_effect=AssertionError("teacher loaded"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _main(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            full.main(list(argv))
        return stderr.getvalue()

    def test_output_root_and_fresh_directory(self):
        self.assertIn("must resolve under", self._main(*self.TEACHER, "--runs", "0:2", "--output",
                                                       str(self.root.parent / "elsewhere")))
        existing = self.root / "existing"
        existing.mkdir()
        self.assertIn("already exists", self._main(*self.TEACHER, "--runs", "0:2", "--output", str(existing)))
        (self.root / "link").symlink_to(existing)
        self.assertIn("already exists", self._main(*self.TEACHER, "--runs", "0:2", "--output", str(self.root / "link")))
        self.assertIn("COUNT", self._main(*self.TEACHER, "--runs", "0:9", "--output", str(self.root / "x")))
        self.assertIn("teacher:<name> or checkpoint:<path>",
                      self._main("--combat-player", "random", "--runs", "0:1", "--seed", "1", "--output", "x"))

    def test_runs_outside_the_panel_need_smoke(self):
        for runs in ("60:5", "64:1", "100:2"):
            with self.subTest(runs=runs):
                self.assertIn("inside the 64-run panel",
                              self._main(*self.TEACHER, "--runs", runs, "--output", str(self.root / "out")))

    def test_priority_must_be_the_preregistered_list(self):
        other = self.root / "other-priority.json"
        other.write_bytes((ROOT / full.PRIORITY_PATH).read_bytes() + b"\n")
        self.assertIn("not the pre-registered", self._main(*self.TEACHER, "--runs", "0:1", "--priority", str(other),
                                                           "--output", str(self.root / "out")))
        # --smoke may play another list; it then reaches the chunk (patched here to stop).
        with self.assertRaisesRegex(AssertionError, "run_chunk reached"):
            full.main([*self.TEACHER, "--runs", "0:1", "--smoke", "--priority", str(other),
                       "--output", str(self.root / "smoke")])
        with self.assertRaisesRegex(AssertionError, "run_chunk reached"):
            full.main([*self.TEACHER, "--runs", "0:1", "--output", str(self.root / "panel")])


class RootRefusalTextTests(unittest.TestCase):
    def test_root_refusal_is_the_exact_gym_text(self):
        """Gym v1.2.5 raises ``RunUnsupportedBoundaryError(u.reason())`` with a bare static class string."""
        for text in sorted(search_view.ROOT_REFUSAL_CLASSES):
            self.assertEqual(full.root_refusal(FakeUnsupported(text)), text)
            self.assertLess(len(text), 80)
        with patch.object(search_view, "_native_errors", return_value=(FakeUnsupported, FakeState, FakeFault)):
            with self.assertRaises(FakeUnsupported):
                full.RunSearchRoot(FakeRun(FakeUnsupported("a hidden pre-rolled value was shown by damage"))).public_search(1)
            for text in ("hidden intents are not modelled: enemy 2", "Hidden intents are not modelled",
                         "public roots exist only on the Standard surface"):
                with self.subTest(text=text), self.assertRaises(full.RunFault):
                    full.RunSearchRoot(FakeRun(FakeUnsupported(text))).public_search(1)


if __name__ == "__main__":
    unittest.main()
