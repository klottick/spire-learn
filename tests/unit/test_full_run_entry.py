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
# Pre-registered for gate R1: this file never changes once committed.
PRIORITY_SHA256 = "9dd28b208c16e6fb05daced54ca264d287c9437c95d8aad4500e861448f36548"


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
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PRIORITY_SHA256)
        self.assertEqual(CardPriority.load(path).sha256, PRIORITY_SHA256)


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
        boundary = Boundary("act3_ending")
        boundary.choose("act3_continue")
        boundary.choose("sapphire")
        action = full.noncombat_move(heuristic, boundary.obs, boundary.legal, flags, rules, fallbacks)
        self.assertEqual(action, 0)
        self.assertEqual(fallbacks, Counter({"fallback:act3_ending": 1}))
        self.assertEqual(rules, Counter({"fallback:act3_ending": 1}))
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
        self.assertEqual(aggregate["fallback_rules"], {"fallback:act3_ending": 1})


class CommandLineTests(unittest.TestCase):
    def _main(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            full.main(list(argv))
        return stderr.getvalue()

    def test_output_root_and_fresh_directory(self):
        common = ("--combat-player", "teacher:current-v2", "--teachers",
                  "configs/neural/search-teacher-gym-1.2.4/teachers-v2.json", "--runs", "0:2", "--seed", "1")
        self.assertIn("must resolve under", self._main(*common, "--output", "artifacts/development/neural-combat/x"))
        root = ROOT / full.OUTPUT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as existing:
            self.assertIn("already exists", self._main(*common, "--output", existing))
        self.assertIn("COUNT", self._main(*common[:-4], "--runs", "0:9", "--seed", "1", "--output", "x"))
        self.assertIn("teacher:<name> or checkpoint:<path>",
                      self._main("--combat-player", "random", "--runs", "0:1", "--seed", "1", "--output", "x"))


if __name__ == "__main__":
    unittest.main()
