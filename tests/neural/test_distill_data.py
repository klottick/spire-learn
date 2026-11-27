"""Sparse distillation shards: exact observation round trip, label separation, determinism.

Needs numpy (runtime container only); the module skips cleanly on the host. Run it in the
container with ``gated_tests_entry.py --distill-data``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from unittest import mock
import unittest

from spire_learn.neural.contracts import BufferArray, CombatFrame
from spire_learn.neural.schema import parse_combat_descriptor, validate_frame
from spire_learn.neural.standard_stream import run_split

ROOT = Path(__file__).resolve().parents[2]
GYM_1_2_0 = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"
RUN_STATES = ROOT / "artifacts/development/neural-combat/run-states/run-states-gym-1.2.4.json"


def _load_numpy_modules() -> None:
    """Import numpy and the shard module on demand, never at discovery (the host has no numpy)."""
    try:
        import numpy
    except ImportError:
        raise unittest.SkipTest("numpy is not importable on the host; run in the runtime container")
    from spire_learn.neural import distill_data

    namespace = globals()
    namespace["np"] = numpy
    for name in ("OUTCOMES", "ROOT_KINDS", "DecisionLabels", "ShardError", "ShardReader", "ShardWriter",
                 "decode_decision", "encode_decision", "expand_decision", "run_hash", "validation_runs"):
        namespace[name] = getattr(distill_data, name)


def _schema():
    descriptor = json.loads(
        (ROOT / "tests/fixtures/neural/standard-combat-compact-4d19806.json").read_text())
    return parse_combat_descriptor(descriptor, revision=GYM_1_2_0)


def build_frame(schema, seed: int, rows: dict[str, list[int]], empty_known: dict[str, list[int]]):
    """A valid compact frame with the given occupied rows (any order of gaps) per row group."""
    rng = np.random.default_rng(seed)
    view = schema.view
    raw: dict[str, np.ndarray] = {}
    for key, spec in schema.tensors.items():
        raw[key] = np.zeros(spec.shape, dtype=np.dtype(spec.dtype))

    def fill(spec, values, known):
        for column_index, column in enumerate(spec.columns):
            if column.reference is not None:
                values[..., column_index] = column.unavailable  # unknown reference
                known[..., column_index] = 0
                continue
            if column.vocabulary is not None:
                size = len(schema.vocabularies[column.vocabulary])
                values[..., column_index] = rng.integers(1, size)
            else:
                low, high = max(column.low, 0), min(column.high, 1000)
                values[..., column_index] = rng.integers(low, high + 1)
            known[..., column_index] = 1

    for key, spec in schema.tensors.items():
        if not spec.columns:
            continue
        values, known = raw[key], raw[spec.known_key]
        if spec.occupied_key is None:
            fill(spec, values, known)
            continue
        occupied = raw[spec.occupied_key]
        for row in rows.get(key, []):
            occupied[row] = 1
            fill(spec, values[row], known[row])
        if key in view.optional_known_empty_groups:
            for row in range(spec.shape[0]):
                if not occupied[row]:
                    known[row] = 0
            for row in empty_known.get(key, []):
                known[row] = 1
    raw["action_mask"][:] = raw["actions_occupied"]
    frame = CombatFrame(
        {key: BufferArray(spec.dtype, spec.shape, raw[key].tobytes())
         for key, spec in schema.tensors.items()},
        schema.schema_fingerprint)
    validate_frame(frame, schema)
    return frame


def frames(schema):
    return [
        build_frame(schema, 1, {"permanent_cards": [0, 1, 2], "actions": [0, 3, 4, 9],
                                "potions": [1, 3], "orbs": [0, 2, 3]},
                    {"potions": [0, 4], "orbs": [1, 5, 9]}),
        build_frame(schema, 2, {"combat_cards": [5, 6, 40], "enemies": [1, 4], "actions": [2],
                                "relics": [0, 7], "potions": [], "orbs": []},
                    {"potions": [2], "orbs": [0, 9]}),
        build_frame(schema, 3, {"actions": [1000, 1023], "powers": [3, 100], "known_draw": [9]},
                    {}),
    ]


def labels_for(frame, index, *, value_ok=True, run_id="run-a"):
    mask = np.frombuffer(frame.tensors["action_mask"].data, dtype=np.int8)
    legal = [int(i) for i in np.flatnonzero(mask)]
    return DecisionLabels(
        legal=legal, values=[0.25 * (i + 1) for i in range(len(legal))], samples=32,
        choice=legal[-1], played=legal[0], root="searched", outcome="win",
        hp_kept=0.5 if value_ok else math.nan, realised_v2=0.7 if value_ok else math.nan,
        value_ok=value_ok, policy_ok=True, research_seed=2 ** 40 + index, case=(1000, index),
        run_id=run_id, turn=3 + index)


def write_shard(path, schema, items, provenance=None):
    writer = ShardWriter(path, schema, provenance or {"teacher": "current-v2", "seed": 7})
    for frame, labels in items:
        writer.add(frame, labels)
    return writer.close()


def same(a: CombatFrame, b: CombatFrame) -> bool:
    return (a.schema_fingerprint == b.schema_fingerprint and set(a.tensors) == set(b.tensors)
            and all(a.tensors[k] == b.tensors[k] for k in a.tensors))


class DistillShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _load_numpy_modules()
        cls.schema = _schema()
        cls.frames = frames(cls.schema)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def items(self):
        return [(frame, labels_for(frame, i, value_ok=i != 1)) for i, frame in enumerate(self.frames)]

    def test_round_trip_exact_on_compact_fixture(self):
        optional = self.schema.view.optional_known_empty_groups
        self.assertEqual(optional, {"potions", "orbs"})
        # The builders really exercise non-prefix rows and non-uniform empty-slot known flags.
        first = self.frames[0]
        occupied = np.frombuffer(first.tensors["potions_occupied"].data, dtype=np.int8)
        known = np.frombuffer(first.tensors["potions_known"].data, dtype=np.int8).reshape(5, 1)
        self.assertEqual(list(np.flatnonzero(occupied)), [1, 3])
        self.assertEqual(sorted(set(known[occupied == 0].ravel())), [0, 1])
        for frame in self.frames:
            self.assertTrue(same(decode_decision(encode_decision(frame, self.schema), self.schema), frame))
        write_shard(self.root / "s", self.schema, self.items())
        reader = ShardReader(self.root / "s")
        reader.verify()
        for index, frame in enumerate(self.frames):
            expanded = expand_decision(reader, index, self.schema)
            self.assertTrue(same(expanded, frame))
            self.assertTrue(same(reader.frame(index, self.schema), frame))
            for key, buffer in frame.tensors.items():
                self.assertEqual(expanded.tensors[key].data, buffer.data, key)
        self.assertIsInstance(reader.array("potions.known_full"), np.memmap)
        self.assertEqual(reader.array("player.values").shape, (3, 13))

    def test_label_files_disjoint_from_observation_files(self):
        write_shard(self.root / "s", self.schema, self.items())
        reader = ShardReader(self.root / "s")
        observation, labels = set(reader.observation_files), set(reader.label_files)
        self.assertFalse(observation & labels)
        self.assertTrue(all(name.startswith("labels.") for name in labels))
        self.assertFalse(any(name.startswith("labels.") for name in observation))
        self.assertEqual(
            {name[len("labels."):] for name in labels},
            {"legal_offsets", "legal", "values", "samples", "choice", "played", "root", "outcome",
             "hp_kept", "realised_v2", "value_ok", "policy_ok", "research_seed", "case", "run_hash",
             "turn"})
        self.assertEqual(
            reader.array("labels.run_hash")[0],
            int.from_bytes(hashlib.sha256(b"run-a").digest()[:8], "big"))
        # No run id or hash in any observation file: only labels.* carries run identity.
        stored = {path.name for path in (self.root / "s").glob("*.npy")}
        self.assertEqual(stored, {f"{name}.npy" for name in observation | labels})
        # Frames expand from observation files alone.
        for name in labels:
            (self.root / "s" / f"{name}.npy").rename(self.root / "s" / f"{name}.hidden")
        decoded = decode_decision(reader.decision_arrays(0), self.schema)
        self.assertTrue(same(decoded, self.frames[0]))

    def test_labels_round_trip_with_nan_and_enums(self):
        write_shard(self.root / "s", self.schema, self.items())
        reader = ShardReader(self.root / "s")
        stored = reader.labels(1)
        self.assertFalse(stored.value_ok)
        self.assertTrue(math.isnan(stored.hp_kept) and math.isnan(stored.realised_v2))
        self.assertEqual((stored.root, stored.outcome, stored.case, stored.turn),
                         ("searched", "win", (1000, 1), 4))
        self.assertEqual(stored.research_seed, 2 ** 40 + 1)
        first = reader.labels(0)
        self.assertEqual(list(first.legal), [0, 3, 4, 9])
        self.assertEqual(list(first.values), [0.25, 0.5, 0.75, 1.0])
        self.assertEqual((first.choice, first.played, first.samples), (9, 0, 32))
        self.assertEqual(reader.labels(2).run_hash, run_hash("run-a"))
        self.assertEqual(ROOT_KINDS, ("searched", "single_legal", "refused"))
        self.assertEqual(OUTCOMES, ("win", "death", "escaped", "none"))

    def test_unsearched_root_stores_nan_values_per_legal_action(self):
        frame = self.frames[0]
        labels = labels_for(frame, 0)
        unsearched = DecisionLabels(**{**labels.__dict__, "values": None, "root": "refused",
                                       "policy_ok": False})
        write_shard(self.root / "s", self.schema, [(frame, unsearched)])
        stored = ShardReader(self.root / "s").labels(0)
        self.assertEqual(len(stored.values), 4)
        self.assertTrue(np.isnan(stored.values).all())
        self.assertEqual(stored.root, "refused")

    def test_writer_refuses_inconsistent_labels(self):
        frame = self.frames[0]
        good = labels_for(frame, 0)
        for change in (
            {"legal": [0, 3]},
            {"choice": 5},
            {"value_ok": False},  # finite hp_kept with value_ok false
            {"hp_kept": math.nan},  # value_ok true needs finite
            {"root": "forced"},
            {"values": [1.0]},
        ):
            with self.subTest(change=change), self.assertRaises(ShardError):
                writer = ShardWriter(self.root / f"x{len(list(self.root.iterdir()))}", self.schema, {})
                writer.add(frame, DecisionLabels(**{**good.__dict__, **change}))

    def test_shard_files_deterministic(self):
        first = write_shard(self.root / "a", self.schema, self.items())
        second = write_shard(self.root / "b", self.schema, self.items())
        self.assertEqual(first, second)
        names = sorted(p.name for p in (self.root / "a").iterdir())
        self.assertEqual(names, sorted(p.name for p in (self.root / "b").iterdir()))
        for name in names:
            self.assertEqual((self.root / "a" / name).read_bytes(), (self.root / "b" / name).read_bytes(), name)
        text = (self.root / "a" / "manifest.json").read_text()
        for word in ("time", "date", "host"):
            self.assertNotIn(word, text.lower())
        self.assertEqual(first["provenance"], {"seed": 7, "teacher": "current-v2"})
        self.assertFalse(any(name.endswith(".npz") for name in names))

    def _rewrite(self, name, mutate):
        """Replace one file and keep the manifest hash honest, to reach the structural checks."""
        directory = self.root / "s"
        array = np.array(np.load(directory / f"{name}.npy"))
        array = mutate(array)
        np.save(directory / f"{name}.npy", array, allow_pickle=False)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["files"][name].update(
            sha256=hashlib.sha256((directory / f"{name}.npy").read_bytes()).hexdigest(),
            dtype=array.dtype.str, shape=list(array.shape))
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def test_verify_detects_tamper(self):
        write_shard(self.root / "s", self.schema, self.items())
        ShardReader(self.root / "s", self.schema).verify()
        for name in ("labels.turn", "player.values", "actions.rows"):
            target = self.root / "s" / f"{name}.npy"
            original = target.read_bytes()
            flipped = bytearray(original)
            flipped[-1] ^= 1
            target.write_bytes(bytes(flipped))
            with self.subTest(name=name), self.assertRaisesRegex(ShardError, "sha256"):
                ShardReader(self.root / "s").verify()
            target.write_bytes(original)
        ShardReader(self.root / "s").verify()
        (self.root / "s" / "extra.npy").write_bytes(original)
        with self.assertRaisesRegex(ShardError, "differ"):
            ShardReader(self.root / "s").verify()
        (self.root / "s" / "extra.npy").unlink()
        (self.root / "s" / "labels.case.npy").unlink()
        with self.assertRaises(ShardError):
            ShardReader(self.root / "s").verify()
        with self.assertRaises(ShardError):
            ShardReader(self.root / "s").array("labels.case")

    def test_verify_detects_structural_corruption(self):
        def corrupt(name, mutate, message):
            directory = self.root / "s"
            if directory.exists():
                shutil.rmtree(directory)
            write_shard(directory, self.schema, self.items())
            self._rewrite(name, mutate)
            with self.subTest(name=name), self.assertRaisesRegex(ShardError, message):
                ShardReader(directory, self.schema).verify()

        def bump_last(array):
            array[-1] += 1
            return array

        def swap_middle(array):
            array[1], array[2] = array[2] + 5, array[1]
            return array

        corrupt("actions.offsets", bump_last, "length differs")
        corrupt("actions.offsets", swap_middle, "offsets")
        corrupt("labels.legal_offsets", bump_last, "legal_offsets")
        corrupt("labels.root", lambda a: np.full_like(a, 9), "enum")
        corrupt("labels.outcome", lambda a: np.full_like(a, -1), "enum")
        corrupt("labels.played", lambda a: np.full_like(a, 1023), "legal")
        corrupt("labels.choice", lambda a: np.full_like(a, 1023), "legal")
        corrupt("labels.choice", lambda a: np.full_like(a, -1), "choice -1")
        corrupt("labels.turn", lambda a: a[:-1], "one entry")
        corrupt("action_mask", lambda a: a[:-1], "one entry")
        corrupt("potions.known_full", lambda a: a[:-1], "one entry")
        corrupt("player.values", lambda a: a[:-1], "one entry")

    def test_verify_reports_missing_required_files(self):
        write_shard(self.root / "s", self.schema, self.items())
        directory = self.root / "s"
        manifest = json.loads((directory / "manifest.json").read_text())
        del manifest["files"]["relics.known"]
        (directory / "relics.known.npy").unlink()
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ShardError, "required"):
            ShardReader(directory, self.schema).verify()
        with self.assertRaises(ShardError):  # without a schema, the CSR group is incomplete
            ShardReader(directory).verify()
        (directory / "manifest.json").write_text("{not json")
        with self.assertRaises(ShardError):
            ShardReader(directory)

    def test_wrong_schema_is_refused(self):
        write_shard(self.root / "s", self.schema, self.items())
        reader = ShardReader(self.root / "s")
        manifest = json.loads((self.root / "s" / "manifest.json").read_text())
        manifest["observation"]["schema_fingerprint"] = "0" * 64
        reader.manifest = manifest
        with self.assertRaises(ShardError):
            expand_decision(reader, 0, self.schema)
        with self.assertRaises(ShardError):
            ShardReader(self.root / "s").frame(0)  # no schema given

    def test_round_trip_on_frames_from_the_compact_test_builders(self):
        from tests.unit.test_neural_standard_compact import FakeCompactVector
        from spire_learn.neural.snapshot import snapshot_common

        fake = FakeCompactVector(None, 1)
        fake.schema = self.schema
        frame = snapshot_common(fake._obs(), self.schema)
        self.assertTrue(same(decode_decision(encode_decision(frame, self.schema), self.schema), frame))
        write_shard(self.root / "s", self.schema, [(frame, labels_for(frame, 0))])
        reader = ShardReader(self.root / "s", self.schema)
        reader.verify()
        self.assertTrue(same(reader.frame(0), frame))
        occupied = np.frombuffer(frame.tensors["potions_occupied"].data, dtype=np.int8)
        self.assertEqual(int(occupied.sum()), 0)  # every potion slot is empty, all known

    def test_close_is_atomic_and_retryable(self):
        final = self.root / "s"
        writer = ShardWriter(final, self.schema, {"a": 1})
        for frame, labels in self.items():
            writer.add(frame, labels)
        real_save, calls = np.save, []

        def failing(*args, **kwargs):
            calls.append(1)
            if len(calls) == 5:
                raise OSError("disk full")
            return real_save(*args, **kwargs)

        with mock.patch.object(np, "save", side_effect=failing), self.assertRaises(OSError):
            writer.close()
        self.assertFalse(final.exists())
        self.assertEqual([p.name for p in self.root.iterdir()], [])  # no temp dir left behind
        manifest = writer.close()  # retry succeeds
        self.assertEqual(manifest["rows"], 3)
        ShardReader(final, self.schema).verify()
        self.assertEqual([p.name for p in self.root.iterdir()], ["s"])

    def test_reader_maps_lazily_and_closes(self):
        write_shard(self.root / "s", self.schema, self.items())
        with ShardReader(self.root / "s", self.schema) as reader:
            self.assertEqual(reader.open_files, 0)
            reader.labels(0)
            self.assertGreater(reader.open_files, 0)
            self.assertLess(reader.open_files, len(reader.manifest["files"]))
            reader.frame(0)
        self.assertEqual(reader.open_files, 0)
        self.assertEqual(reader.labels(1).case, (1000, 1))  # re-opens on demand
        reader.close()

    def test_label_rules(self):
        frame = self.frames[0]
        good = labels_for(frame, 0)
        inf, nan = math.inf, math.nan

        def refuses(**change):
            with self.assertRaises(ShardError, msg=str(change)):
                ShardWriter(self.root / "x", self.schema, {}).add(
                    frame, DecisionLabels(**{**good.__dict__, **change}))

        refuses(hp_kept=inf)
        refuses(realised_v2=-inf)
        refuses(value_ok=False, hp_kept=inf, realised_v2=nan)
        refuses(values=[1.0, nan, 1.0, 1.0])  # searched needs every value finite
        refuses(values=[1.0, inf, 1.0, 1.0])
        refuses(root="single_legal", values=[1.0] * 4, policy_ok=False)  # unsearched keeps NaN
        refuses(root="refused", values=None, policy_ok=True)
        refuses(root="searched", values=None)
        refuses(samples=-1)
        refuses(samples=2 ** 31)
        refuses(turn=-1)
        refuses(turn=2 ** 31)
        refuses(played=5)
        refuses(choice=-1)  # the sentinel is for refused roots only
        refuses(root="refused", values=None, policy_ok=False, played=5)
        refuses(root="refused", values=None, policy_ok=True, choice=-1)
        # The refused-root sentinel is accepted, and every NaN is the canonical quiet NaN.
        odd_nan = np.frombuffer(np.uint64(0x7ff8000000000123).tobytes(), dtype=np.float64)[0]
        refused = DecisionLabels(**{**good.__dict__, "root": "refused", "values": [odd_nan] * 4,
                                    "policy_ok": False, "choice": -1, "value_ok": False,
                                    "hp_kept": float(odd_nan), "realised_v2": float(odd_nan)})
        write_shard(self.root / "ok", self.schema, [(frame, refused)])
        reader = ShardReader(self.root / "ok", self.schema)
        reader.verify()
        self.assertEqual(reader.labels(0).choice, -1)
        for name in ("labels.values", "labels.hp_kept", "labels.realised_v2"):
            self.assertTrue((np.asarray(reader.array(name)).view(np.uint64) == 0x7ff8000000000000).all(), name)

    def test_uint64_identity_fields_round_trip(self):
        frame = self.frames[0]
        run_id = next(f"run-{i}" for i in range(1000) if run_hash(f"run-{i}") >= 2 ** 63)
        labels = DecisionLabels(**{**labels_for(frame, 0).__dict__, "research_seed": 2 ** 64 - 1,
                                   "run_id": run_id})
        write_shard(self.root / "s", self.schema, [(frame, labels)])
        stored = ShardReader(self.root / "s", self.schema).labels(0)
        self.assertEqual(stored.research_seed, 2 ** 64 - 1)
        self.assertEqual(stored.run_hash, run_hash(run_id))
        self.assertGreaterEqual(stored.run_hash, 2 ** 63)

    def test_group_empty_in_every_decision(self):
        frames_ = [build_frame(self.schema, seed, {"actions": [0, 2], "relics": []}, {})
                   for seed in (11, 12)]
        write_shard(self.root / "s", self.schema,
                    [(f, labels_for(f, i)) for i, f in enumerate(frames_)])
        reader = ShardReader(self.root / "s", self.schema)
        reader.verify()
        self.assertEqual(list(reader.array("relics.offsets")), [0, 0, 0])
        self.assertEqual(reader.array("relics.values").shape, (0, 2))
        for index, frame in enumerate(frames_):
            self.assertTrue(same(reader.frame(index), frame))

    def test_zero_row_shard(self):
        manifest = write_shard(self.root / "z", self.schema, [], {"note": "empty"})
        self.assertEqual(manifest["rows"], 0)
        self.assertEqual(manifest["files"], {})
        self.assertEqual([p.name for p in (self.root / "z").iterdir()], ["manifest.json"])
        reader = ShardReader(self.root / "z", self.schema)
        reader.verify()
        self.assertEqual(len(reader), 0)
        with self.assertRaises(IndexError):
            reader.frame(0)
        with self.assertRaises(IndexError):
            reader.labels(0)

    def test_writer_does_not_overwrite_and_closes_once(self):
        write_shard(self.root / "s", self.schema, self.items()[:1])
        with self.assertRaises(FileExistsError):
            write_shard(self.root / "s", self.schema, self.items()[:1])
        writer = ShardWriter(self.root / "t", self.schema, {})
        writer.close()
        with self.assertRaises(ShardError):
            writer.close()


def _real_ironclad_train_runs():
    if not RUN_STATES.is_file():
        return None
    states = json.loads(RUN_STATES.read_text())["states"]
    return sorted({s["run_id"] for s in states
                   if s["character"] == "ironclad" and run_split(s["run_id"]) == "train"})


class ValidationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _load_numpy_modules()

    def test_validation_runs_exact_fraction(self):
        real = _real_ironclad_train_runs()
        ids = real if real is not None else [f"synthetic-run-{i:04d}" for i in range(505)]
        self.assertEqual(len(ids), 505)
        holdout = validation_runs(ids)
        self.assertEqual(len(holdout), 25)
        self.assertLessEqual(holdout, set(ids))
        ranked = sorted(ids, key=lambda r: int.from_bytes(
            hashlib.sha256(b"spire-learn-distill-validation-v1:" + r.encode()).digest(), "big"))
        self.assertEqual(holdout, frozenset(ranked[:25]))
        # Independent of input order and duplicates.
        self.assertEqual(validation_runs(reversed(ids + ids[:5])), holdout)

    @unittest.skipUnless(RUN_STATES.is_file(), "the local run-states table is not mounted")
    def test_validation_runs_golden_on_the_real_table(self):
        holdout = validation_runs(_real_ironclad_train_runs())
        self.assertEqual(len(holdout), 25)
        self.assertEqual(sorted(holdout)[:5],
                         ["1611086796", "1626125194", "1629837138", "1637095905", "1655315178"])

    def test_validation_runs_rounding_and_empty(self):
        self.assertEqual(validation_runs([]), frozenset())
        self.assertEqual(len(validation_runs([f"r{i}" for i in range(30)])), round(0.05 * 30))
        self.assertEqual(len(validation_runs([f"r{i}" for i in range(100)])), 5)
        with self.assertRaises(ValueError):
            validation_runs([""])


if __name__ == "__main__":
    unittest.main()
