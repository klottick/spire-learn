"""Dependency-free checkpoint compatibility and storage preflight tests."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from spire_learn.neural.checkpoint import (
    CHECKPOINT_SCHEMA_ID,
    CheckpointCompatibilityError,
    LoadedCheckpoint,
    MAX_CHECKPOINT_BYTES,
    identity_payload,
    preflight_checkpoint,
    validate_checkpoint_metadata,
)
from spire_learn.neural.config import RUN_RAW_MAPPING_ID, STANDALONE_RAW_MAPPING_ID
from spire_learn.neural.contracts import NeuralIdentity


def _identity(*, modes=("standalone", "run"), model_hash="a" * 64, ppo_hash="b" * 64):
    mapping_ids = {"standalone": STANDALONE_RAW_MAPPING_ID, "run": RUN_RAW_MAPPING_ID}
    return NeuralIdentity(
        model_id="spire-learn-combat-ff-v1",
        features_id="spire-learn-combat-features-v1",
        action_id="spire-learn-combat-dispatch-v1",
        objective_id="spire-learn-combat-winloss-v1",
        accepted_raw_mode_mappings={mode: mapping_ids[mode] for mode in modes},
        config_hashes={"model": model_hash, "ppo": ppo_hash},
    )


def _metadata(*, identity=None, training=False):
    payloads = {"model.pt": {"size_bytes": 128, "sha256": "c" * 64}}
    training_state = None
    if training:
        payloads["training.pt"] = {"size_bytes": 256, "sha256": "d" * 64}
        training_state = {
            "optimizer_step": 9,
            "hyperparameter_id": "b" * 64,
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": {"environment_actions": 20, "learned_decisions": 8},
            "update_id": "update-9",
            "behavior_policy_version": 9,
            "next_environment_seed_ordinal": 5,
            "next_decision_sequence": 40,
            "saved_inflight_record_count": 3,
        }
    return {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "identity": identity_payload(identity or _identity()),
        "payload_inventory": payloads,
        "model_tensors": [
            {"name": "layer.weight", "dtype": "float32", "shape": [2, 3], "sha256": "e" * 64},
            {"name": "layer.bias", "dtype": "float32", "shape": [2], "sha256": "f" * 64},
        ],
        "training_state": training_state,
        "run_metadata": {
            "mode": "standalone",
            "environment_seed": 23,
            "partition": "train",
            "native_build_id": "build-public-id",
        },
        "parent_checkpoint": None,
    }


def _write_archive(path: Path, *, member_name="model/data.bin", content=b"named state"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, content)


class NeuralCheckpointMetadataTests(unittest.TestCase):
    def test_loaded_checkpoint_contract_always_drops_inflight_records(self):
        loaded = LoadedCheckpoint(
            metadata={},
            model_tensors={"layer.weight": object()},
            optimizer_state=None,
            rng_state=None,
            optimizer_step=0,
            inflight_record_count=0,
        )
        self.assertEqual(loaded.inflight_record_count, 0)
        with self.assertRaisesRegex(ValueError, "drop in-flight"):
            LoadedCheckpoint(
                metadata={},
                model_tensors={"layer.weight": object()},
                optimizer_state=None,
                rng_state=None,
                optimizer_step=0,
                inflight_record_count=1,
            )
        with self.assertRaisesRegex(ValueError, "present or absent together"):
            LoadedCheckpoint(
                metadata={},
                model_tensors={"layer.weight": object()},
                optimizer_state={},
                rng_state=None,
                optimizer_step=0,
                inflight_record_count=0,
            )

    def test_mode_seed_and_budget_changes_are_transfer_context(self):
        saved = _identity(modes=("standalone", "run"))
        expected = _identity(modes=("run",))
        metadata = _metadata(identity=saved)
        metadata["run_metadata"] = {
            "mode": "standalone",
            "seed": 999,
            "budgets": {"actions": 100},
        }
        checked = validate_checkpoint_metadata(metadata, expected, purpose="warm_start")
        self.assertEqual(checked["run_metadata"]["seed"], 999)

    def test_encoder_action_objective_and_raw_mapping_mismatches_fail_preflight(self):
        metadata = _metadata()
        cases = (
            (_identity_modes_features("other-features"), "features_id"),
            (_identity_modes_action("other-action"), "action_id"),
            (_identity_modes_objective("other-objective"), "objective_id"),
            (_identity_modes_mapping("run", "wrong-run-mapping-v1"), "run raw-mode mapping"),
        )
        for expected, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CheckpointCompatibilityError, message):
                    validate_checkpoint_metadata(metadata, expected, purpose="evaluate")

    def test_model_hash_mismatch_is_rejected_but_ppo_hash_is_transfer_context(self):
        metadata = _metadata()
        expected = _identity(model_hash="a" * 64, ppo_hash="9" * 64)
        validate_checkpoint_metadata(metadata, expected, purpose="evaluate")
        validate_checkpoint_metadata(metadata, expected, purpose="warm_start")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "model config hash"):
            validate_checkpoint_metadata(
                metadata,
                _identity(model_hash="8" * 64),
                purpose="warm_start",
            )

    def test_resume_requires_training_state_and_exact_learner_config_hashes(self):
        with self.assertRaisesRegex(CheckpointCompatibilityError, "requires optimizer"):
            validate_checkpoint_metadata(_metadata(), _identity(), purpose="resume")
        metadata = _metadata(training=True)
        validate_checkpoint_metadata(metadata, _identity(), purpose="resume")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "matching learner config hashes"):
            validate_checkpoint_metadata(
                metadata,
                _identity(ppo_hash="9" * 64),
                purpose="resume",
            )

    def test_invalid_purpose_and_non_strict_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "purpose"):
            validate_checkpoint_metadata(_metadata(), _identity(), purpose="restore")
        metadata = _metadata()
        metadata["unreviewed_object_state"] = {}
        with self.assertRaisesRegex(ValueError, "top-level field set"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

    def test_invalid_named_tensor_inventory_and_oversized_payload_metadata_fail(self):
        metadata = _metadata()
        metadata["model_tensors"] = [
            {"name": "same", "dtype": "float32", "shape": [1], "sha256": "a" * 64},
            {"name": "same", "dtype": "float32", "shape": [1], "sha256": "b" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")
        metadata = _metadata()
        metadata["payload_inventory"]["model.pt"]["size_bytes"] = MAX_CHECKPOINT_BYTES + 1
        with self.assertRaisesRegex(ValueError, "checkpoint limit"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

    def test_named_tensor_logical_geometry_is_bounded_before_loading(self):
        metadata = _metadata()
        metadata["model_tensors"] = [
            {"name": "scalar", "dtype": "float32", "shape": [], "sha256": "a" * 64},
        ]
        validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        metadata["model_tensors"] = [
            {"name": "zero-sized", "dtype": "float32", "shape": [0, 2], "sha256": "a" * 64},
        ]
        validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        # A single logical tensor may use the entire independent 256 MiB cap.
        metadata["model_tensors"] = [
            {
                "name": "boundary",
                "dtype": "float32",
                "shape": [MAX_CHECKPOINT_BYTES // 4],
                "sha256": "a" * 64,
            },
        ]
        validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        metadata["model_tensors"] = [
            {
                "name": "oversized",
                "dtype": "float32",
                "shape": [MAX_CHECKPOINT_BYTES // 4 + 1],
                "sha256": "a" * 64,
            },
        ]
        with self.assertRaisesRegex(ValueError, "logical tensor bytes"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        # Multiple individually valid entries still share one aggregate cap.
        metadata["model_tensors"] = [
            {
                "name": "first-half",
                "dtype": "float32",
                "shape": [MAX_CHECKPOINT_BYTES // 8],
                "sha256": "a" * 64,
            },
            {
                "name": "second-half",
                "dtype": "float32",
                "shape": [MAX_CHECKPOINT_BYTES // 8],
                "sha256": "b" * 64,
            },
            {"name": "one-more", "dtype": "float32", "shape": [1], "sha256": "c" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "aggregate logical tensor bytes"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

    def test_named_tensor_dtype_rank_and_each_dimension_are_explicitly_bounded(self):
        metadata = _metadata()
        metadata["model_tensors"] = [
            {"name": "unknown", "dtype": "float128", "shape": [1], "sha256": "a" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "unsupported model tensor dtype"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        metadata["model_tensors"] = [
            {"name": "huge-zero", "dtype": "float32", "shape": [0, MAX_CHECKPOINT_BYTES + 1], "sha256": "a" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

        metadata["model_tensors"] = [
            {"name": "too-many-dimensions", "dtype": "float32", "shape": [0] * 65, "sha256": "a" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "rank"):
            validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

    def test_incompatible_metadata_is_rejected_without_optional_imports(self):
        blocked = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
        original_import = builtins.__import__

        def reject_optional(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked:
                raise AssertionError(f"optional import attempted during metadata preflight: {name}")
            return original_import(name, globals, locals, fromlist, level)

        metadata = _metadata()
        metadata["identity"]["features_id"] = "wrong-features-v1"
        with patch.object(builtins, "__import__", new=reject_optional):
            with self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_metadata(metadata, _identity(), purpose="evaluate")

    def test_file_hash_and_exact_directory_inventory_are_verified_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            model_path = root / "model.pt"
            _write_archive(model_path)
            raw_metadata = _metadata()
            raw = model_path.read_bytes()
            raw_metadata["payload_inventory"]["model.pt"] = {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            (root / "metadata.json").write_text(json.dumps(raw_metadata), encoding="utf-8")
            checked = preflight_checkpoint(root, _identity(), purpose="evaluate")
            self.assertEqual(checked.model_path, model_path)
            self.assertIsNone(checked.training_path)
            self.assertGreater(checked.uncompressed_bytes, 0)
            self.assertEqual(checked.logical_tensor_bytes, 32)

            (root / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagree"):
                preflight_checkpoint(root, _identity(), purpose="evaluate")

    def test_payload_hash_mismatch_and_unsafe_zip_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            model_path = root / "model.pt"
            _write_archive(model_path)
            raw = model_path.read_bytes()
            metadata = _metadata()
            metadata["payload_inventory"]["model.pt"] = {
                "size_bytes": len(raw),
                "sha256": "0" * 64,
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                preflight_checkpoint(root, _identity(), purpose="evaluate")

            _write_archive(model_path, member_name="../outside.bin")
            raw = model_path.read_bytes()
            metadata["payload_inventory"]["model.pt"] = {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe member"):
                preflight_checkpoint(root, _identity(), purpose="evaluate")

    def test_uncompressed_size_limit_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            model_path = root / "model.pt"
            _write_archive(model_path)
            raw = model_path.read_bytes()
            metadata = _metadata()
            metadata["payload_inventory"]["model.pt"] = {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch(
                "spire_learn.neural.checkpoint._archive_uncompressed_bytes",
                return_value=MAX_CHECKPOINT_BYTES,
            ):
                with self.assertRaisesRegex(ValueError, "aggregate"):
                    preflight_checkpoint(root, _identity(), purpose="evaluate")

    def test_duplicate_json_keys_and_symlink_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            (root / "metadata.json").write_text('{"schema_id":"x","schema_id":"y"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate checkpoint metadata key"):
                preflight_checkpoint(root, _identity(), purpose="evaluate")

            other = Path(directory) / "other.pt"
            _write_archive(other)
            (root / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
            try:
                (root / "model.pt").symlink_to(other)
            except OSError:
                self.skipTest("symlinks are unavailable on this filesystem")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                preflight_checkpoint(root, _identity(), purpose="evaluate")


def _identity_modes_features(features_id):
    value = _identity()
    return NeuralIdentity(
        model_id=value.model_id,
        features_id=features_id,
        action_id=value.action_id,
        objective_id=value.objective_id,
        accepted_raw_mode_mappings=value.accepted_raw_mode_mappings,
        config_hashes=value.config_hashes,
    )


def _identity_modes_action(action_id):
    value = _identity()
    return NeuralIdentity(
        model_id=value.model_id,
        features_id=value.features_id,
        action_id=action_id,
        objective_id=value.objective_id,
        accepted_raw_mode_mappings=value.accepted_raw_mode_mappings,
        config_hashes=value.config_hashes,
    )


def _identity_modes_objective(objective_id):
    value = _identity()
    return NeuralIdentity(
        model_id=value.model_id,
        features_id=value.features_id,
        action_id=value.action_id,
        objective_id=objective_id,
        accepted_raw_mode_mappings=value.accepted_raw_mode_mappings,
        config_hashes=value.config_hashes,
    )


def _identity_modes_mapping(mode, mapping_id):
    value = _identity()
    mappings = dict(value.accepted_raw_mode_mappings)
    mappings[mode] = mapping_id
    return NeuralIdentity(
        model_id=value.model_id,
        features_id=value.features_id,
        action_id=value.action_id,
        objective_id=value.objective_id,
        accepted_raw_mode_mappings=mappings,
        config_hashes=value.config_hashes,
    )


if __name__ == "__main__":
    unittest.main()
