import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from spire_learn.artifacts.codec import canonical_fingerprint, to_json_value, transition_to_json
from spire_learn.artifacts.writer import RunWriter
from spire_learn.contracts import ArrayValue, Route, Transition
from tests.support import case_result, observation


class CodecTests(unittest.TestCase):
    def test_array_json_preserves_dtype_shape_values(self):
        self.assertEqual(to_json_value(ArrayValue("int32", (2,), (3, 4))),
                         {"dtype": "int32", "shape": [2], "values": [3, 4]})

    def test_explicit_records_and_immutable_mappings_convert_to_json(self):
        result = case_result()
        value = to_json_value(result)
        self.assertEqual(value["execution"]["environment_seed"], 17)
        self.assertEqual(value["final_resources"], {})
        json.dumps(value, allow_nan=False)
        with self.assertRaises(TypeError):
            result.final_resources["hp"] = 2

    def test_opaque_records_boolean_numbers_and_nonfinite_values_are_rejected(self):
        @dataclasses.dataclass
        class Unapproved:
            value: int

        for value in (Unapproved(1), object(), math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                to_json_value(value)

    def test_canonical_fingerprint_is_sorted_compact_and_stable(self):
        expected = hashlib.sha256(b'{"a":[2],"b":1}').hexdigest()
        self.assertEqual(canonical_fingerprint({"b": 1, "a": (2,)}), expected)

    def test_transition_excludes_opaque_state_and_preserves_public_values(self):
        secret = object()
        transition = Transition(
            Route("exec", 0, "episode", "decision", True), observation(), 1, 0.5,
            False, False, observation(hp=9), None, secret,
        )
        encoded = transition_to_json(transition)
        self.assertNotIn("state_after_source", encoded)
        self.assertNotIn(repr(secret), json.dumps(encoded))
        self.assertEqual(encoded["action"], 1)


class WriterTests(unittest.TestCase):
    def test_create_refuses_existing_directory_and_malformed_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "already-exists").mkdir()
            with self.assertRaises(FileExistsError):
                RunWriter.create(root, "already-exists")
            with self.assertRaises(ValueError):
                RunWriter.create(root, "../escape")

    def test_refused_create_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "run"
            existing.mkdir()
            config = existing / "config.json"
            config.write_bytes(b"user bytes")
            with self.assertRaises(FileExistsError):
                RunWriter.create(root, "run")
            self.assertEqual(config.read_bytes(), b"user bytes")

    def test_writes_versioned_files_and_keeps_manifest_discovery_append_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = RunWriter.create(Path(temporary), "run-1")
            writer.write_start({"agent": "random"}, {"cases": []})
            writer.write_case(case_result())
            writer.write_metric({"schema_id": "spire-learn-probe-v1", "ok": True})
            writer.update_manifest({"discovered": "schema-x"})
            initial = json.loads((writer.run_dir / "manifest.json").read_text())
            writer.update_manifest({"discovered": "schema-y"})
            self.assertEqual(json.loads((writer.run_dir / "manifest.json").read_text()), initial)
            writer.finalize({"scheduled": 1})
            self.assertEqual(json.loads((writer.run_dir / "config.json").read_text())["schema_id"], "spire-learn-config-v1")
            self.assertEqual(json.loads((writer.run_dir / "summary.json").read_text())["schema_id"], "spire-learn-evaluation-v1")
            self.assertEqual(len((writer.run_dir / "episodes.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((writer.run_dir / "metrics.jsonl").read_text().splitlines()), 1)
            writer.close()


if __name__ == "__main__":
    unittest.main()
