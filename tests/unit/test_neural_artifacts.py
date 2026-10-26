"""Pure JSON and filesystem tests for the neural run writer."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.artifacts.codec import canonical_fingerprint
from spire_learn.neural.artifacts import NeuralWriter, RUN_SCHEMA_ID


class NeuralWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_run_ids_are_safe_and_created_exclusively(self):
        for run_id in ("../escape", ".hidden", "two words", "é", "", "a" * 65):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    NeuralWriter.create(self.root, run_id)
        writer = NeuralWriter.create(self.root, "run-01.good")
        writer.close()
        with self.assertRaises(FileExistsError):
            NeuralWriter.create(self.root, "run-01.good")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["run-01.good"])

    def test_start_uses_versioned_envelopes_and_canonical_hashes(self):
        config = {"device": "cpu", "seeds": {"policy": 17}}
        manifest = {
            "mode": "standalone",
            "partition": "train",
            "environment_revision": "533dad1c8aec86df624b3742b6b3ad1dbf1d0033",
            "schema_fingerprint": "a" * 64,
            "rules_id": "rules-v1",
            "content_id": "content-v1",
            "reward_id": "combat-winloss-v1",
            "objective_id": "objective-v1",
            "controller_id": "controller-v1",
            "device": "cpu",
            "budgets": {"environment_actions": 32},
        }
        writer = NeuralWriter.create(self.root, "run-start")
        writer.start(config, manifest)
        writer.close()

        run_dir = self.root / "run-start"
        config_record = json.loads((run_dir / "config.json").read_text())
        manifest_record = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(config_record["schema_id"], RUN_SCHEMA_ID)
        self.assertEqual(manifest_record["schema_id"], RUN_SCHEMA_ID)
        self.assertEqual(config_record["record_type"], "config")
        self.assertEqual(manifest_record["record_type"], "manifest")
        self.assertEqual(config_record["config_sha256"], canonical_fingerprint(config))
        self.assertEqual(manifest_record["manifest_sha256"], canonical_fingerprint(manifest))
        self.assertEqual(manifest_record["manifest"], manifest)
        self.assertTrue(manifest_record["code_sha256"])
        self.assertIn("learner", manifest_record["provenance"])

    def test_provenance_keeps_unavailable_and_dirty_states_distinct(self):
        cases = (
            ({"learner": {"revision": None, "head_state": "unavailable", "dirty": None}},
             (None, "unavailable", None)),
            ({"learner": {"revision": "abc123", "head_state": "present", "dirty": True}},
             ("abc123", "present", True)),
        )
        for index, (provenance, expected) in enumerate(cases):
            with self.subTest(provenance=provenance):
                with patch("spire_learn.neural.artifacts.collect_provenance", return_value=provenance):
                    writer = NeuralWriter.create(self.root, f"provenance-{index}")
                    writer.start({}, {})
                    writer.close()
                envelope = json.loads(
                    (self.root / f"provenance-{index}" / "manifest.json").read_text()
                )
                learner = envelope["provenance"]["learner"]
                self.assertEqual(
                    (learner["revision"], learner["head_state"], learner["dirty"]),
                    expected,
                )

    def test_json_projection_rejects_nonfinite_and_unapproved_objects(self):
        @dataclass
        class UnapprovedRecord:
            value: int

        writer = NeuralWriter.create(self.root, "strict-json")
        writer.start({}, {})
        with self.assertRaisesRegex(ValueError, "finite"):
            writer.event({"metric": float("nan")})
        with self.assertRaisesRegex(TypeError, "unsupported JSON value"):
            writer.event({"metric": UnapprovedRecord(1)})
        with self.assertRaisesRegex(TypeError, "unsupported JSON value"):
            writer.event({"payload": b"raw bytes"})
        with self.assertRaisesRegex(TypeError, "keys must be strings"):
            writer.event({1: "bad key"})
        writer.close()

    def test_events_reject_raw_observation_fields_and_are_size_bounded(self):
        writer = NeuralWriter.create(self.root, "bounded-events")
        writer.start({}, {})
        with self.assertRaisesRegex(ValueError, "raw observation"):
            writer.event({"action": 2, "observation": {"hp": [1, 2]}})
        with self.assertRaisesRegex(ValueError, "exceeds"):
            writer.event({"metric": "x" * (70 * 1024)})
        writer.event({"action": 2, "lifecycle": "combat", "decision_count": 4})
        writer.close()
        rows = (self.root / "bounded-events" / "events.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["record"]["action"], 2)

    def test_start_event_finalize_order_is_checked(self):
        writer = NeuralWriter.create(self.root, "state-machine")
        with self.assertRaisesRegex(RuntimeError, "start"):
            writer.event({"action": 1})
        writer.start({}, {})
        writer.finalize({"status": "complete"})
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            writer.event({"action": 2})
        with self.assertRaisesRegex(RuntimeError, "already started"):
            writer.start({}, {})
        writer.close()

    def test_summary_is_published_atomically_and_never_overwrites_existing_data(self):
        writer = NeuralWriter.create(self.root, "atomic-summary")
        writer.start({}, {})
        writer.finalize({"status": "complete", "updates": 3})
        summary_path = self.root / "atomic-summary" / "summary.json"
        first = summary_path.read_bytes()
        self.assertEqual(json.loads(first)["summary"]["updates"], 3)
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            writer.finalize({"status": "different"})
        self.assertEqual(summary_path.read_bytes(), first)
        self.assertFalse(list((self.root / "atomic-summary").glob(".summary-*.tmp")))
        writer.close()

    def test_failed_event_append_poison_writer_without_retrying(self):
        class BrokenStream:
            def write(self, _value):
                raise OSError(errno.ENOSPC, "full")

            def flush(self):
                return None

            def fileno(self):
                return 1

            def close(self):
                return None

        writer = NeuralWriter.create(self.root, "append-failure")
        writer.start({}, {})
        writer._events.close()
        writer._events = BrokenStream()
        with self.assertRaisesRegex(OSError, "full"):
            writer.event({"action": 1})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            writer.event({"action": 1})
        with patch("spire_learn.neural.artifacts.os.fsync"):
            writer.close()

    def test_failed_finalization_preserves_existing_summary_and_marks_writer_failed(self):
        writer = NeuralWriter.create(self.root, "finalize-failure")
        writer.start({}, {})
        summary_path = writer.run_dir / "summary.json"
        summary_path.write_text("keep-me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            writer.finalize({"status": "complete"})
        self.assertEqual(summary_path.read_text(), "keep-me")
        with self.assertRaisesRegex(RuntimeError, "failed"):
            writer.finalize({"status": "again"})
        writer.close()

    def test_failed_close_propagates_first_error_after_attempting_close(self):
        class BrokenClose:
            attempted = False

            def flush(self):
                return None

            def fileno(self):
                return 1

            def close(self):
                self.attempted = True
                raise OSError(errno.EIO, "close failed")

        writer = NeuralWriter.create(self.root, "close-failure")
        writer._events.close()
        broken = BrokenClose()
        writer._events = broken
        with patch("spire_learn.neural.artifacts.os.fsync"):
            with self.assertRaisesRegex(OSError, "close failed"):
                writer.close()
        self.assertTrue(broken.attempted)
        self.assertTrue(writer._closed)

    def test_close_prefers_process_control_and_attaches_secondary_failures(self):
        class FaultyStream:
            def __init__(self, flush_error=None, close_error=None):
                self.flush_error = flush_error
                self.close_error = close_error
                self.close_calls = 0

            def flush(self):
                if self.flush_error is not None:
                    raise self.flush_error

            def fileno(self):
                return 1

            def close(self):
                self.close_calls += 1
                if self.close_error is not None:
                    raise self.close_error

        cases = (
            (OSError(errno.EIO, "flush io"), KeyboardInterrupt("close interrupt"),
             KeyboardInterrupt, "close interrupt", "flush io"),
            (KeyboardInterrupt("flush interrupt"), OSError(errno.EIO, "close io"),
             KeyboardInterrupt, "flush interrupt", "close io"),
            (OSError(errno.ENOSPC, "first io"), OSError(errno.EIO, "second io"),
             OSError, "first io", "second io"),
        )
        for index, (flush_error, close_error, error_type, primary_text, note_text) in enumerate(cases):
            with self.subTest(index=index):
                writer = NeuralWriter.create(self.root, f"close-precedence-{index}")
                writer._events.close()
                stream = FaultyStream(flush_error, close_error)
                writer._events = stream
                with patch("spire_learn.neural.artifacts.os.fsync"):
                    with self.assertRaises(error_type) as raised:
                        writer.close()
                self.assertIs(raised.exception, close_error if index == 0 else flush_error)
                self.assertIn(primary_text, str(raised.exception))
                self.assertTrue(any(note_text in note for note in getattr(raised.exception, "__notes__", ())))
                self.assertEqual(stream.close_calls, 1)
                self.assertTrue(writer._closed)


if __name__ == "__main__":
    unittest.main()
