"""The distillation labelling entry (``--distill-native`` route; runtime container only).

``LabelRuleTests`` play scripted fights over real compact frames with the teacher's decision
faked, so every label rule is pinned exactly. ``LabelChunkNativeTests`` label a few real training
fights on the pinned Gym. numpy, torch and the Gym are imported lazily, never at discovery.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural.search_value import ValueRule

ROOT = Path(__file__).resolve().parents[2]
TEACHER = {"samples": 32, "playout_steps": 80, "playout_backend": "native",
           "value_rule": ValueRule("resource-v2"), "playout_policy": "no-potions-v2"}
REWARDS = {"victory": 1.0, "death": -1.0, "escaped": 0.0}


def _numpy_modules() -> dict:
    try:
        import numpy
    except ImportError:
        raise unittest.SkipTest("numpy is only importable in the runtime container")
    from scripts.neural import distill_label_entry as label
    from spire_learn.neural import distill_data
    from tests.neural import test_distill_data as shards

    shards._load_numpy_modules()
    return {"np": numpy, "label": label, "data": distill_data, "shards": shards}


def _runs(split, count):
    from spire_learn.neural.standard_stream import run_split

    found = [f"run-{index}" for index in range(400) if run_split(f"run-{index}") == split]
    return found[:count]


class Observations:
    """Compact observations from the test frame builder, with chosen public player and potion fields."""

    def __init__(self, np, shards, schema, descriptor):
        from spire_learn.neural.search_view import View

        self.np, self.shards, self.schema = np, shards, schema
        self.view = View(descriptor)
        player = next(t for t in descriptor["tensors"] if t["key"] == "player")
        self.fields = [column["field"] for column in player["columns"]]
        self.turn_column = self.fields.index("turn")
        self.potions = list(descriptor["vocabularies"]["potion"])
        self._base: dict = {}

    def make(self, *, turn, legal=(0, 1), hp=50, max_hp=80, cleanup=None, potions=()):
        key = (tuple(legal), len(potions))
        if key not in self._base:
            frame = self.shards.build_frame(self.schema, 7, {"actions": list(legal),
                                                             "potions": list(range(len(potions)))}, {})
            self._base[key] = {name: self.np.frombuffer(item.data, dtype=item.dtype).reshape(item.shape).copy()
                               for name, item in frame.tensors.items()}
        raw = {name: array.copy() for name, array in self._base[key].items()}
        for field, value in (("hp", hp), ("max_hp", max_hp), ("turn", turn), ("hp_after_cleanup", cleanup)):
            column = self.fields.index(field)
            raw["player"][column] = 0 if value is None else value
            raw["player_known"][column] = 0 if value is None else 1
        for row, name in enumerate(potions):
            raw["potions"][row, 0] = self.potions.index(name)
            raw["potions_known"][row, 0] = 1
        return raw


class FakeFight:
    """A scripted fight: ``at(i)`` is the observation of decision i; it ends after ``len`` decisions."""

    def __init__(self, at, *, length=None, final=None, outcome="victory", cleanup=None):
        self.at, self.length, self.final, self.outcome, self.cleanup = at, length, final, outcome, cleanup
        self.actions = []

    def reset(self, case):
        return self.at(0), {"phase": "combat"}

    def step(self, action):
        self.actions.append(action)
        index = len(self.actions)
        if index == self.length:
            info = {"phase": "terminal", "terminal_reason": self.outcome}
            if self.cleanup is not None:
                info["hp_after_cleanup"] = self.cleanup
            return self.final, REWARDS[self.outcome], True, False, info
        return self.at(index), 0.0, False, False, {"phase": "combat"}


def searched(values, choice=None):
    stats = {str(action): [value, 32] for action, value in values.items()}
    best = max(values.values())
    pick = choice if choice is not None else max(a for a, v in values.items() if v == best)
    return "searched", pick, stats, 0.1, 0, 32 * len(values)


class FakePlayer:
    """Always plays the lowest legal action; records its reseeds."""

    def __init__(self):
        self.seeds, self.calls = [], 0

    def reseed(self, seed):
        self.seeds.append(seed)

    def act(self, frame, legal):
        self.calls += 1
        return min(legal)


class LabelRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = _numpy_modules()
        cls.np, cls.label, cls.data, cls.shards = (modules[k] for k in ("np", "label", "data", "shards"))
        cls.descriptor = json.loads((ROOT / "tests/fixtures/neural/standard-combat-compact-4d19806.json").read_text())
        cls.schema = cls.shards._schema()
        cls.obs = Observations(cls.np, cls.shards, cls.schema, cls.descriptor)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT"))
        self.addCleanup(self.temp.cleanup)

    def job(self, *, local=0, mixed=False, series="train", run_id="run-a"):
        label = self.label
        return label.FightJob(
            case={"case_id": f"c{local}", "entry": {"id": f"c{local}"}, "environment_seed": 1}, run_id=run_id,
            shard=1000, local=local, series=series, mixed=mixed,
            research_seed=label.research_seed(9, 1000, local), mixed_seed=label.mixed_seed(9, 1000, local))

    def run_fight(self, fight, decide, *, job=None, player=None, roundtrip_every=50):
        script = iter(decide) if not callable(decide) else None

        def fake_decision(env, observation, legal, view, teacher, rng, seed):
            return next(script) if script is not None else decide(legal, rng)

        with patch.object(self.label.entry, "_decision", side_effect=fake_decision):
            return self.label.label_fight(fight, job or self.job(), self.obs.view, self.schema, self.obs.turn_column,
                                          TEACHER, player=player, roundtrip_every=roundtrip_every)

    def write(self, results, name="s"):
        writer = self.data.ShardWriter(Path(self.temp.name) / name, self.schema, {"test": name})
        for result in results:
            for arrays, labels in result["rows"]:
                writer.add(self.data.decode_decision(arrays, self.schema), labels)
        writer.close()
        reader = self.data.ShardReader(Path(self.temp.name) / name, self.schema)
        reader.verify()
        return reader

    @staticmethod
    def default(legal, rng):
        return searched({action: 0.1 * (index + 1) for index, action in enumerate(legal)})

    def test_only_train_cases_and_holdout_routed(self):
        from spire_learn.neural.cases import environment_seed_for

        label = self.label
        (train_a, train_b), (selection,) = _runs("train", 2), _runs("selection", 1)

        def case(local, run_id, *, seed=None):
            cid = f"b-{local}"
            return ({"case_id": cid, "lineage_id": f"baalorlord:{run_id}-3:k0:1000.{local}", "entry": {"id": cid},
                     "environment_seed": environment_seed_for("train", replica=0, ordinal=1000 * 256 + local)
                     if seed is None else seed}, {"run_id": run_id})

        def shard(*cases, split="train"):
            return SimpleNamespace(shard_index=1000, manifest={"split": split, "partition": split,
                                                                "cases": [c for c, _ in cases]},
                                   annotations={c["entry"]["id"]: a for c, a in cases})

        jobs = label.plan_jobs([shard(case(0, train_a), case(3, train_b))], {train_b}, seed=9, mixed_every=2)
        self.assertEqual([(j.local, j.run_id, j.series) for j in jobs],
                         [(0, train_a, "train"), (3, train_b, "validation")])
        self.assertEqual([j.research_seed for j in jobs], [label.research_seed(9, 1000, 0), label.research_seed(9, 1000, 3)])
        self.assertEqual([j.mixed for j in jobs], [False, True])  # 256000 + 3 = 1 mod 2
        self.assertEqual(len(label.plan_jobs([shard(case(0, train_a), case(3, train_b))], set(), seed=9,
                                             mixed_every=2, case_limit=1)), 1)
        with self.assertRaisesRegex(ValueError, "selection-split run"):
            label.plan_jobs([shard(case(0, train_a), case(1, selection))], set(), seed=9, mixed_every=5)
        with self.assertRaisesRegex(ValueError, "train-split shard"):
            label.plan_jobs([shard(case(0, train_a), split="selection")], set(), seed=9, mixed_every=5)
        with self.assertRaisesRegex(ValueError, "environment seed"):
            label.plan_jobs([shard(case(0, train_a, seed=5))], set(), seed=9, mixed_every=5)

    def test_mixed_schedule_played_vs_choice(self):
        label = self.label
        self.assertEqual([local for local in range(10) if label.is_mixed(1000, local, 5)], [4, 9])
        self.assertEqual([local for local in range(4) if label.is_mixed(1000, local, 0)], [])
        self.assertTrue(all(label.is_mixed(1001, local, 1) for local in range(4)))

        player = FakePlayer()
        fight = FakeFight(lambda i: self.obs.make(turn=1 + i // 3, legal=(0, 1, 2)), length=8,
                          final=self.obs.make(turn=3, cleanup=30), cleanup=30)
        result = self.run_fight(fight, self.default, job=self.job(local=4, mixed=True), player=player)
        coin = random.Random(label.mixed_seed(9, 1000, 4))
        self.assertEqual(player.seeds, [coin.getrandbits(63)])
        by_player = [coin.random() < 0.5 for _ in range(8)]
        self.assertTrue(any(by_player) and not all(by_player))
        played = [0 if flag else 2 for flag in by_player]
        self.assertEqual(fight.actions, played)
        labels = [labels for _, labels in result["rows"]]
        self.assertEqual([l.played for l in labels], played)
        self.assertEqual([l.choice for l in labels], [2] * 8)  # the teacher labels every decision
        self.assertTrue(all(l.policy_ok for l in labels))
        self.assertEqual((result["fight_type"], result["player_moves"], player.calls), ("mixed", sum(by_player),
                                                                                       sum(by_player)))

    def test_mixed_has_no_value_targets(self):
        fight = FakeFight(lambda i: self.obs.make(turn=1), length=3, final=self.obs.make(turn=1, cleanup=30),
                          cleanup=30)
        result = self.run_fight(fight, self.default, job=self.job(local=4, mixed=True), player=FakePlayer())
        labels = [labels for _, labels in result["rows"]]
        self.assertEqual(len(labels), 3)
        self.assertTrue(all(not l.value_ok and math.isnan(l.hp_kept) and math.isnan(l.realised_v2)
                            and l.outcome == "win" for l in labels))
        reader = self.write([result])
        self.assertFalse(self.np.asarray(reader.array("labels.value_ok")).any())
        with self.assertRaisesRegex(ValueError, "player"):
            self.run_fight(FakeFight(lambda i: self.obs.make(turn=1), length=1, final=self.obs.make(turn=1)),
                           self.default, job=self.job(local=4, mixed=True))

    def test_turn_cap_keeps_bounded_prefix_policy_only(self):
        label = self.label
        self.assertEqual((label.TURN_CAP, label.FIGHT_CAP, label.KEPT_PREFIX_MAX, label.MIXED_EVERY,
                          label.ROUNDTRIP_EVERY, label.LABEL_SHARD_START), (500, 1024, 256, 5, 50, 1000))
        turns = lambda i: self.obs.make(turn=1 if i < 3 else 2)  # noqa: E731 - three decisions, then a stall
        with patch.object(label, "TURN_CAP", 4), patch.object(label, "KEPT_PREFIX_MAX", 3):
            result = self.run_fight(FakeFight(turns), self.default)
        self.assertEqual((result["censor_reason"], result["decisions"], result["dropped"]), ("turn_cap", 7, 4))
        labels = [labels for _, labels in result["rows"]]
        self.assertEqual([l.turn for l in labels], [1, 1, 1])
        self.assertTrue(all(l.policy_ok and not l.value_ok and l.outcome == "none" and math.isnan(l.hp_kept)
                            for l in labels))
        self.write([result])
        with patch.object(label, "TURN_CAP", 4), patch.object(label, "KEPT_PREFIX_MAX", 2):
            result = self.run_fight(FakeFight(turns), self.default)
        self.assertEqual((len(result["rows"]), result["dropped"]), (0, 7))

    def test_fight_cap_keeps_nothing(self):
        label = self.label
        with patch.object(label, "TURN_CAP", 4), patch.object(label, "FIGHT_CAP", 6):
            result = self.run_fight(FakeFight(lambda i: self.obs.make(turn=1 + i // 2)), self.default)
        self.assertEqual((result["censor_reason"], result["decisions"], result["dropped"], result["rows"]),
                         ("step_cap", 6, 6, []))

    def test_turn_cap_on_turn_one_writes_cleanly(self):
        label = self.label
        with patch.object(label, "TURN_CAP", 4):
            capped = self.run_fight(FakeFight(lambda i: self.obs.make(turn=1)), self.default)
        self.assertEqual((capped["censor_reason"], capped["decisions"], capped["rows"]), ("turn_cap", 4, []))
        self.assertEqual(len(self.write([capped], "alone")), 0)
        won = self.run_fight(FakeFight(lambda i: self.obs.make(turn=1), length=2,
                                       final=self.obs.make(turn=1, cleanup=30), cleanup=30),
                             self.default, job=self.job(local=1))
        reader = self.write([capped, won], "mixed-in")
        self.assertEqual(len(reader), 2)
        self.assertEqual(reader.labels(0).case, (1000, 1))

    def test_realised_v2_matches_search_value_formula(self):
        from spire_learn.neural.search_value import RootContext, terminal_value

        view, rule = self.obs.view, TEACHER["value_rule"]
        first = self.obs.make(turn=1, hp=50, potions=("fire_potion", "fairy_potion"))
        second = self.obs.make(turn=1, hp=45, potions=("fairy_potion",))

        def play(outcome, cleanup, info_cleanup=None):
            final = self.obs.make(turn=2, hp=cleanup or 0, max_hp=85, cleanup=cleanup)
            fight = FakeFight(lambda i: (first, second)[i], length=2, final=final, outcome=outcome,
                              cleanup=info_cleanup if info_cleanup is not None else cleanup)
            return self.run_fight(fight, self.default), final

        result, final = play("victory", 40)
        labels = [labels for _, labels in result["rows"]]
        # 0.55 + 0.4 * 40/80 + 0.4 * (1.5 * (85 - 80) - spent) / 80; spent 12 + 30, then 30.
        for got, expected in zip([l.realised_v2 for l in labels], [0.5775, 0.6375]):
            self.assertAlmostEqual(got, expected, places=12)
        for observation, got in zip((first, second), labels):
            root = RootContext.from_observation(view, observation, rule)
            self.assertEqual(got.realised_v2, terminal_value(rule, root, view, final, 1, "victory"))
        self.assertEqual([(l.outcome, l.hp_kept, l.value_ok) for l in labels], [("win", 0.5, True)] * 2)

        died, _ = play("death", None)
        self.assertEqual([(l.outcome, l.hp_kept, l.realised_v2) for _, l in died["rows"]], [("death", 0.0, -1.0)] * 2)
        escaped, _ = play("escaped", 40)
        self.assertAlmostEqual(escaped["rows"][0][1].realised_v2, -0.1725, places=12)
        self.assertEqual(escaped["rows"][0][1].outcome, "escaped")
        with self.assertRaisesRegex(ValueError, "hp_after_cleanup"):
            play("victory", None)  # a terminal end with no cleanup HP raises
        with self.assertRaisesRegex(RuntimeError, "differs"):
            play("victory", 40, info_cleanup=41)

    def test_single_legal_refused_and_all_tie_rows(self):
        observations = [self.obs.make(turn=1, legal=(2,)), self.obs.make(turn=1, legal=(0, 1)),
                        self.obs.make(turn=1, legal=(0, 1)), self.obs.make(turn=2, legal=(0, 1))]
        script = [("single_legal", 2, None, None, 0, None), ("runic_dome", 0, None, None, 0, None),
                  searched({0: 0.5, 1: 0.5}, choice=1), searched({0: 0.25, 1: 0.75})]
        fight = FakeFight(lambda i: observations[i], length=4, final=self.obs.make(turn=2, cleanup=30), cleanup=30)
        result = self.run_fight(fight, script)
        self.assertEqual(fight.actions, [2, 0, 1, 1])
        self.assertEqual((result["roots"], result["refusals"], result["all_tie"]),
                         ({"single_legal": 1, "refused": 1, "searched": 2}, {"runic_dome": 1}, 1))
        reader = self.write([result])
        rows = [reader.labels(index) for index in range(4)]
        self.assertEqual([(r.root, r.choice, r.played, r.policy_ok) for r in rows],
                         [("single_legal", 2, 2, False), ("refused", -1, 0, False), ("searched", 1, 1, False),
                          ("searched", 1, 1, True)])
        self.assertTrue(all(self.np.isnan(rows[i].values).all() for i in (0, 1)))
        self.assertEqual(list(rows[2].values), [0.5, 0.5])
        self.assertEqual(list(rows[3].values), [0.25, 0.75])
        # Every row of a finished teacher fight gets value targets, whatever its root.
        self.assertTrue(all(r.value_ok and math.isfinite(r.realised_v2) for r in rows))
        self.assertEqual([r.samples for r in rows], [0, 0, 32, 32])

    def test_roundtrip_assertion_fires_on_mismatch(self):
        label = self.label
        fight = FakeFight(lambda i: self.obs.make(turn=1, hp=50 - i), length=3, final=self.obs.make(turn=1, cleanup=30),
                          cleanup=30)
        result = self.run_fight(fight, self.default, roundtrip_every=1)
        self.assertEqual([row for row, *_ in result["roundtrip"]], [0, 1, 2])
        reader = self.write([result])
        self.assertEqual(label.check_roundtrip(reader, result["roundtrip"]), 3)
        _, fingerprint, other = result["roundtrip"][1]
        with self.assertRaises(label.RoundTripError):
            label.check_roundtrip(reader, [(0, fingerprint, other)])
        # The snapshot kept for the check is exactly the Gym arrays.
        tensors = result["roundtrip"][0][2]
        raw = self.obs.make(turn=1, hp=50)
        self.assertEqual(set(tensors), set(raw))
        self.assertTrue(all(tensors[k].data == self.np.ascontiguousarray(raw[k]).tobytes() for k in raw))


def _native_ready():
    return os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"


@unittest.skipUnless(_native_ready(), "requires the pinned Gym native runtime")
class LabelChunkNativeTests(unittest.TestCase):
    RECIPE = ROOT / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
    TEACHERS = ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers-v2.json"

    @classmethod
    def setUpClass(cls):
        modules = _numpy_modules()
        cls.np, cls.label, cls.data = modules["np"], modules["label"], modules["data"]
        from spire_learn.neural.recipes import gym_validator, load_recipe
        from spire_learn.neural.scenario_support import json_value
        from spire_learn.neural.schema import parse_combat_descriptor
        from spire_learn.neural.standard_stream import generate_standard_shard
        from tests.neural.test_policy_player import save_fixture_checkpoint
        from spire_learn.neural.policy_player import standard_ff_identity

        label = cls.label
        cls.parent = tempfile.TemporaryDirectory(prefix="distill-label-", dir=os.environ.get(
            "SPIRE_LEARN_RUNNER_OUTPUT", ROOT / "artifacts"))
        cls.root = Path(cls.parent.name)
        recipe = load_recipe(cls.RECIPE)
        teacher, _, _ = label.entry._teacher(cls.TEACHERS, "current-v2", None)
        _, revision, _, mapping = label._check_gym(recipe, teacher)
        shard = generate_standard_shard(recipe, split="train", shard_index=1000, validate=gym_validator(recipe=recipe))
        descriptor = label.entry.preflight([json_value(shard.manifest["cases"][0])], "train")
        cls.schema = parse_combat_descriptor(descriptor, revision=revision)
        cls.checkpoint = save_fixture_checkpoint(
            cls.root / "mixed-checkpoint", cls.schema, label.MIXED_MODEL_CONFIG,
            standard_ff_identity(label.MIXED_MODEL_CONFIG, mapping))

    @classmethod
    def tearDownClass(cls):
        cls.parent.cleanup()

    def label_chunk(self, name, shards, *, parallel=1, mixed_every=2, roundtrip_every=1, limit=2):
        output = self.root / name
        argv = ["--recipe", str(self.RECIPE), "--teacher", "current-v2", "--teachers", str(self.TEACHERS),
                "--allow-uncalibrated", "--shards", shards, "--seed", "20261127", "--mixed-every", str(mixed_every),
                "--output", str(output), "--parallel", str(parallel), "--roundtrip-every", str(roundtrip_every),
                "--case-limit", str(limit)]
        if mixed_every:
            argv += ["--mixed-checkpoint", str(self.checkpoint)]
        self.assertEqual(self.label.main(argv), 0)
        return output

    @staticmethod
    def shard_files(path):
        return {child.name: child.read_bytes() for child in sorted(path.iterdir())}

    def test_seed_independent_of_chunking(self):
        label = self.label
        whole = self.label_chunk("whole", "1000:2", parallel=2)
        parts = [self.label_chunk("part-0", "1000:1"), self.label_chunk("part-1", "1001:1")]
        for series in label.SERIES:
            for index, part in zip((1000, 1001), parts):
                name = label.shard_name(index)
                self.assertEqual(self.shard_files(whole / series / name), self.shard_files(part / series / name))

        manifest = label.verify_chunk(whole, first=1000, count=2)
        counts = manifest["counts"]
        self.assertEqual(sum(counts["fights"].values()), 4)
        self.assertGreaterEqual(counts["fights"]["mixed"], 1)
        rows = sum(counts["rows"].values())
        self.assertGreater(rows, 0)
        self.assertEqual(sum(counts["roots"].values()), rows)
        self.assertEqual(counts["roundtrips"], rows)  # --roundtrip-every 1 re-reads every stored row
        provenance = manifest["provenance"]
        self.assertEqual((provenance["teacher"], provenance["uncalibrated_allowed"], provenance["mixed_every"]),
                         ("current-v2", True, 2))
        self.assertEqual(provenance["mixed_checkpoint_sha256"], label.checkpoint_sha256(self.checkpoint)["sha256"])
        self.assertIn("revision", provenance["learner"])

        for index in (1000, 1001):
            with self.data.ShardReader(whole / "train" / label.shard_name(index), self.schema) as reader:
                reader.verify()
                for row in range(len(reader)):
                    labels = reader.labels(row)
                    self.assertIn(labels.played, list(labels.legal))
                    mixed = label.is_mixed(*labels.case, 2)
                    if mixed:
                        self.assertFalse(labels.value_ok)
                    if labels.value_ok:
                        self.assertTrue(0.0 <= labels.hp_kept and -1.0 <= labels.realised_v2 <= 1.0)
                    if labels.root == "searched":
                        self.assertTrue(self.np.isfinite(labels.values).all())

        victim = next((whole / "train" / label.shard_name(1000)).glob("labels.*.npy"))
        victim.write_bytes(victim.read_bytes() + b"\0")
        with self.assertRaises(label.ChunkError):
            label.verify_chunk(whole)

    def test_roundtrip_mismatch_stops_the_chunk(self):
        from spire_learn.neural import distill_data
        from spire_learn.neural.contracts import BufferArray, CombatFrame

        add = distill_data.ShardWriter.add

        def corrupt(writer, frame, labels):
            player = bytearray(frame.tensors["player"].data)
            player[0] = (player[0] + 1) % 256  # the low byte of hp, which stays a valid value
            tensors = dict(frame.tensors)
            tensors["player"] = BufferArray(frame.tensors["player"].dtype, frame.tensors["player"].shape, bytes(player))
            return add(writer, CombatFrame(tensors, frame.schema_fingerprint), labels)

        with patch.object(distill_data.ShardWriter, "add", corrupt), self.assertRaises(self.label.RoundTripError):
            self.label_chunk("corrupt", "1000:1", mixed_every=0, limit=1)
        self.assertFalse((self.root / "corrupt" / self.label.CHUNK_MANIFEST).exists())

    def test_cli_refuses_unsafe_arguments(self):
        label = self.label
        base = ["--recipe", str(self.RECIPE), "--teacher", "current-v2", "--teachers", str(self.TEACHERS),
                "--allow-uncalibrated", "--seed", "1", "--output", str(self.root / "never")]
        for extra in (["--shards", "999:1", "--mixed-every", "0"],
                      ["--shards", "1000:1"],  # mixed fights need the checkpoint
                      ["--shards", "1000:1", "--mixed-every", "0", "--mixed-checkpoint", str(self.checkpoint)]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                label.main(base + extra)
        self.assertFalse((self.root / "never").exists())


if __name__ == "__main__":
    unittest.main()
