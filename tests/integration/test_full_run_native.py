"""The full-run harness on the pinned Gym (``--full-run-native`` route; runtime container only).

- At real Run combat boundaries the bridged frame equals the root view of
  ``RunEnv.public_search(observation="compact")`` (every group, both masks and the action mask), and the
  root's legal set is the Run's legal indices.
- The curse list of the committed priority file matches the Gym's card metadata.
- A two-run smoke (``--smoke``: ordinals past the panel, never panel runs) with teacher ``current-v2``,
  and a one-run smoke with a fixture checkpoint player, write complete, consistent outputs.

numpy, torch and the Gym are imported lazily, never at discovery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEACHERS = ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers-v2.json"


def _modules():
    try:
        import numpy
    except ImportError:
        raise unittest.SkipTest("numpy is only importable in the runtime container")
    from scripts.neural import full_run_entry
    return numpy, full_run_entry


def _output_parent():
    runner = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if not runner:
        raise unittest.SkipTest("the full-run smoke writes under SPIRE_LEARN_RUNNER_OUTPUT (runtime container)")
    return Path(runner)


class RunRootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.full = _modules()

    def test_run_root_legal_set_matches(self):
        np, full = self.np, self.full
        from spire_lab.gymnasium import RunEnv
        from spire_learn.neural import search_view
        from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET
        from spire_learn.neural.schema import parse_lifecycle
        from spire_learn.neural.views import STANDARD_COMBAT_V1

        _, revision, _, _ = full._check_gym(None)
        context = full.PlayContext(player_kind="teacher", teacher=None, checkpoint=None, mapping=None,
                                   priority_path=str(ROOT / full.PRIORITY_PATH), gym_revision=revision)
        unsupported, _, _ = search_view._native_errors()
        compared = refused = 0
        for index in range(2):
            env = RunEnv(full.CHARACTER, full.ASCENSION, profile="standard", partition=full.PARTITION)
            try:
                bridge, view, heuristic, _, _ = full._resources(context, env)
                flags, rng = full.RoomFlags(), random.Random(index)
                observation, _ = env.reset(seed=full.run_seed(index, smoke=True))
                for step in range(600):
                    combat = env.combat_view()
                    lifecycle = parse_lifecycle(combat["info"], view=STANDARD_COMBAT_V1)
                    legal = [int(i) for i in np.flatnonzero(observation["action_mask"])]
                    if not lifecycle.active:
                        action = heuristic.choose(observation, legal, flags).action
                    else:
                        arrays, frame = bridge(combat["obs"])
                        self.assertEqual(view.legal(arrays), legal)
                        try:
                            search = env.public_search(step, observation="compact",
                                                       capacity_preset=STANDARD_COMPACT_PRESET,
                                                       limits={"max_live_branches": 1, "max_branches": 1,
                                                               "max_steps_per_branch": 1, "max_total_steps": 1})
                        except unsupported as error:
                            self.assertIn(search_view.refusal_class(error), search_view.ROOT_REFUSAL_CLASSES)
                            refused += 1
                        else:
                            try:
                                self.assertEqual(search.observation_spec()["fingerprint"], bridge.compact_fingerprint)
                                root, _ = search.branch(0).decision()
                            finally:
                                search.close()
                            self.assertEqual(set(root), set(arrays))
                            for key, array in arrays.items():
                                self.assertEqual((root[key].dtype, root[key].shape), (array.dtype, array.shape), key)
                                self.assertTrue(np.array_equal(root[key], array), f"{key} at step {step}")
                            self.assertEqual([int(i) for i in np.flatnonzero(root["action_mask"])], legal)
                            compared += 1
                        action = view.playout_choice(arrays, rng, "no-potions-v2")
                    observation, _, terminated, _, _ = env.step(action)
                    if terminated:
                        break
            finally:
                env.close()
        self.assertGreaterEqual(compared, 40, f"{compared} compared roots, {refused} refused")

    def test_curse_list_matches_card_metadata(self):
        from spire_lab import _native
        from spire_learn.neural.noncombat_heuristic import CardPriority

        priority = CardPriority.load(ROOT / self.full.PRIORITY_PATH)
        cards = {card["id"]: card for card in _native.card_metadata()["cards"]}
        self.assertEqual(priority.curses, {name for name, card in cards.items() if card["type"] == "curse"})
        listed = set(priority.priority) | set(priority.avoid)
        self.assertEqual({name for name, card in cards.items() if card["color"] == "red"} - listed, set())
        for name in listed:
            self.assertIn(cards[name]["color"], ("red", "colorless"), name)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.full = _modules()
        cls.parent = Path(tempfile.mkdtemp(prefix="full-run-smoke-", dir=_output_parent()))

    def _check_outputs(self, output, runs, combat_player):
        full = self.full
        rows = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
        fights = [json.loads(line) for line in (output / "fights.jsonl").read_text().splitlines()]
        summary = json.loads((output / "summary.json").read_text())
        report = json.loads((output / "report.json").read_text())
        self.assertEqual([row["index"] for row in rows], list(range(runs)))
        self.assertEqual(len(fights), sum(row["fights"] for row in rows))
        self.assertEqual(len(report["timings"]), runs)
        for row in rows:
            self.assertEqual(row["seed"], full.run_seed(row["index"], smoke=True))
            self.assertTrue(row["censored"] or row["terminal_reason"] in ("death", "act3_no_keys", "heart_victory"))
            self.assertGreaterEqual(row["floors_reached"], 1)
            self.assertEqual(row["decisions"], row["combat_decisions"] + row["noncombat_decisions"])
            self.assertEqual(sum(row["roots"].values()), row["combat_decisions"])
            self.assertEqual(row["fallback_total"], sum(row["fallbacks"].values()))
            if row["terminal_reason"] == "death":
                self.assertEqual(row["death"]["floor"], row["floors_reached"])
        for fight in fights:
            self.assertIn(fight["outcome"], ("victory", "death", "escaped", "censored"))
            for key in ("case_id", "steps", "hp_start", "max_hp_start", "potions_start", "potion_hp_spent",
                        "cpu_seconds", "wall_seconds"):
                self.assertIn(key, fight)
        provenance = summary["provenance"]
        self.assertEqual(provenance["combat_player"], combat_player)
        self.assertTrue(provenance["smoke"])
        self.assertIsNone(provenance["panel"])
        self.assertEqual(provenance["noncombat_player"], "noncombat-heuristic-v1")
        self.assertEqual(provenance["priority"]["sha256"],
                         full.CardPriority.load(ROOT / full.PRIORITY_PATH).sha256)
        self.assertEqual((provenance["turn_cap"], provenance["fight_cap"], provenance["run_cap"]), (500, 1024, 10_000))
        self.assertEqual(summary["aggregates"]["runs"], runs)
        self.assertNotIn("cpu_seconds", json.dumps(summary["aggregates"]))
        return rows, summary

    def test_full_run_smoke_two_seeds(self):
        from spire_learn.neural.search_view import ROOT_REFUSAL_CLASSES

        output = self.parent / "teacher-current-v2"
        argv = ["--combat-player", "teacher:current-v2", "--teachers", str(TEACHERS), "--smoke", "--runs", "0:2",
                "--seed", "2026112701", "--parallel", "2", "--output", str(output)]
        self.assertEqual(self.full.main(argv), 0)
        rows, summary = self._check_outputs(output, 2, "teacher:current-v2")
        provenance = summary["provenance"]
        self.assertTrue(provenance["uncalibrated_allowed"])
        self.assertIs(provenance["calibrated"], False)
        self.assertEqual(provenance["teacher"], "current-v2")
        for row in rows:
            self.assertGreater(row["roots"].get("searched", 0), 0)
            self.assertTrue(set(row["refusal_classes"]) <= ROOT_REFUSAL_CLASSES)
        print(json.dumps({"smoke_runs": rows}, sort_keys=True))

    def test_checkpoint_player_run(self):
        full = self.full
        from spire_learn.neural.policy_player import standard_ff_identity
        from spire_learn.neural.schema import parse_combat_descriptor
        from tests.neural.test_policy_player import save_fixture_checkpoint

        _, revision, _, mapping = full._check_gym(None)
        schema = parse_combat_descriptor(full.compact_descriptor(), revision=revision)
        checkpoint = save_fixture_checkpoint(
            self.parent / "fixture-checkpoint", schema, full.label.MIXED_MODEL_CONFIG,
            standard_ff_identity(full.label.MIXED_MODEL_CONFIG, mapping))
        output = self.parent / "checkpoint"
        argv = ["--combat-player", f"checkpoint:{checkpoint}", "--smoke", "--runs", "0:1", "--seed", "3",
                "--output", str(output)]
        self.assertEqual(full.main(argv), 0)
        rows, summary = self._check_outputs(output, 1, "checkpoint")
        self.assertEqual(set(rows[0]["roots"]), {"policy"})
        self.assertEqual(summary["provenance"]["checkpoint"]["sha256"],
                         full.label.checkpoint_sha256(checkpoint)["sha256"])


if __name__ == "__main__":
    unittest.main()
