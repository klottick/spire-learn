"""Task 13 export: an S1 distillation student becomes a PPO warm-start checkpoint (delta 18).

``ExportRefusalTests`` is stdlib only (fake checkpoints with real metadata and zip payloads) and runs on
the host: every refusal happens before torch is imported. ``DistillExportTests`` needs numpy and torch
(runtime container only); run it with ``gated_tests_entry.py --distill-export``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from scripts.neural import distill_export
from scripts.neural.baseline_entry import checkpoint_sha256
from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, STANDARD_LEARNER_IDENTITY_IDS, load_neural_config, model_config_hash,
)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = (ROOT / "configs/neural/baseline-standard-1.2.5-distill/template.toml",
             ROOT / "configs/neural/baseline-standard-1.2.5-distill-r2/template.toml")
DISTILL_HASH = "d" * 64
DATASET_SHA256 = "e" * 64
GAMMA = 0.99


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distill_state(step: int = 7) -> dict:
    return {"optimizer_step": step, "hyperparameter_id": DISTILL_HASH,
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": {"optimizer_steps": step, "rows_seen": 8 * step, "policy_rows": 8 * step,
                         "value_rows": 4 * step},
            "update_id": f"step-{step}", "behavior_policy_version": step, "next_environment_seed_ordinal": 0,
            "next_decision_sequence": 0, "saved_inflight_record_count": 0}


def _run_metadata() -> dict:
    return {"operation": "distill-train", "step": 7, "train_shards": [1000, 1001], "dataset_sha256": DATASET_SHA256}


def _fake_checkpoint(path: Path, identity: dict, *, training_state=None) -> Path:
    """A checkpoint that passes preflight without torch: real metadata, a zip ``model.pt``."""
    path.mkdir()
    with zipfile.ZipFile(path / "model.pt", "w") as archive:
        archive.writestr("archive/data.pkl", b"weights")
    metadata = {
        "schema_id": "spire-learn-neural-checkpoint-v1", "identity": identity,
        "payload_inventory": {"model.pt": {"size_bytes": (path / "model.pt").stat().st_size,
                                           "sha256": _sha256(path / "model.pt")}},
        "model_tensors": [{"name": "value_head.weight", "dtype": "float32", "shape": [1, 64], "sha256": "0" * 64}],
        "training_state": _distill_state() if training_state is None else training_state,
        "run_metadata": _run_metadata(), "parent_checkpoint": None}
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def _identity(config, **changes) -> dict:
    template = config.accepted_identities[0]
    identity = {"model_id": template.model_id, "features_id": template.features_id,
                "action_id": template.action_id, "objective_id": DISTILL_OBJECTIVE_ID,
                "accepted_raw_mode_mappings": dict(template.accepted_raw_mode_mappings),
                "config_hashes": {"model": model_config_hash(replace(config.model, value_heads=3)),
                                  "distill": DISTILL_HASH}}
    identity.update(changes)
    return identity


def _record(c: float, **changes) -> dict:
    return {"critic_scale": c, "gamma": GAMMA, "train_shards": [1000, 1001], "dataset_sha256": DATASET_SHA256,
            **changes}


class ExportRefusalTests(unittest.TestCase):
    """Only an S1 distillation checkpoint of the template's FF architecture exports (stdlib)."""

    def setUp(self):
        self.config = load_neural_config(TEMPLATES[0])
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _export(self, source: Path, c: float = 0.5, record=None, schema=None):
        schema = schema or type("Schema", (), {"schema_fingerprint": self.config.standard_view_fingerprint})()
        return distill_export.export_ppo_checkpoint(source, self.config, c, output=self.root / "out", schema=schema,
                                                    critic_record=_record(c) if record is None else record)

    def test_export_refuses_non_s1(self):
        template = self.config.accepted_identities[0]
        ppo = {**_identity(self.config), "objective_id": template.objective_id,
               "config_hashes": dict(template.config_hashes)}
        bigger = replace(self.config.model, hidden_width=128, value_heads=3)
        cases = {
            "a PPO checkpoint": (ppo, None, "only a distillation checkpoint"),
            "another FF architecture": (_identity(self.config, config_hashes={
                "model": model_config_hash(bigger), "distill": DISTILL_HASH}), None, "model config hash"),
            "one value head": (_identity(self.config, config_hashes={
                "model": model_config_hash(self.config.model), "distill": DISTILL_HASH}), None, "model config hash"),
            "another model family": (_identity(self.config, model_id="spire-learn-combat-transformer-standard-v1"),
                                     None, "model_id"),
            "other features": (_identity(self.config, features_id="spire-learn-combat-features-standard-s2-v1"),
                               None, "features_id"),
            "another raw-mode mapping": (_identity(self.config, accepted_raw_mode_mappings={
                "standard-standalone": "spire-learn-standard-combat-compact-v1@" + "0" * 64 + "+gym-1.2.4"}),
                None, "raw-mode mapping"),
            "a PPO hash beside distill": (_identity(self.config, config_hashes={
                "model": model_config_hash(replace(self.config.model, value_heads=3)), "distill": DISTILL_HASH,
                "ppo": "f" * 64}), None, "exactly the model and distill hashes"),
            "no training state": (_identity(self.config), False, "training state"),
        }
        for name, (identity, state, error) in cases.items():
            with self.subTest(source=name):
                source = _fake_checkpoint(self.root / name.replace(" ", "-"), identity)
                if state is False:
                    metadata = json.loads((source / "metadata.json").read_text())
                    metadata["training_state"] = None
                    (source / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(distill_export.ExportError, error):
                    self._export(source)
                self.assertFalse((self.root / "out").exists())
        with self.assertRaisesRegex(distill_export.ExportError, "not a checkpoint directory"):
            self._export(self.root / "missing")
        # The S1 shape passes the identity check; the export's other inputs are checked before torch loads.
        source = _fake_checkpoint(self.root / "s1", _identity(self.config))
        preflight, expected = distill_export.check_s1(source, self.config)
        self.assertEqual(expected.objective_id, DISTILL_OBJECTIVE_ID)
        self.assertEqual(preflight.metadata["identity"]["config_hashes"]["distill"], DISTILL_HASH)
        for name, kwargs, error in (
                ("c above one", {"c": 1.5}, "critic scale c"),
                ("c not finite", {"c": math.nan}, "critic scale c"),
                ("c an int", {"c": 1}, "critic scale c"),
                ("c not the record's", {"record": _record(0.4)}, "disagrees with its critic-scale record"),
                ("another gamma", {"record": _record(0.5, gamma=0.9)}, "different gamma"),
                ("other shards", {"record": _record(0.5, train_shards=[1000])}, "other shards"),
                ("another dataset", {"record": _record(0.5, dataset_sha256="a" * 64)}, "other shards"),
                ("another schema", {"schema": type("Schema", (), {"schema_fingerprint": "0" * 64})()},
                 "fingerprint")):
            with self.subTest(refused=name), self.assertRaisesRegex(distill_export.ExportError, error):
                self._export(source, **kwargs)
        # A non-PPO template is refused too.
        with self.assertRaisesRegex(distill_export.ExportError, "PPO config"):
            distill_export.check_s1(source, replace(self.config, ppo=None))


class DistillExportTests(unittest.TestCase):
    """The export itself and the critic scale (``--distill-export``; numpy and torch, no skips)."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            import torch
        except ImportError:
            raise unittest.SkipTest("numpy and torch are only importable in the runtime container")
        from tests.neural import test_distill_data as shards

        shards._load_numpy_modules()
        cls.torch, cls.shards = torch, shards
        cls.schema = shards._schema()
        cls.frames = shards.frames(cls.schema)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _student(self, config, seed=11):
        from spire_learn.neural.model import build_policy

        model_id, features_id, _ = STANDARD_LEARNER_IDENTITY_IDS
        student = build_policy(self.schema, replace(config.model, value_heads=3), seed=seed, device="cpu",
                               objective_id=DISTILL_OBJECTIVE_ID, model_id=model_id, features_id=features_id)
        with self.torch.no_grad():
            student.value_head.bias.fill_(0.375)  # a non-zero bias, so its scaling is visible
        return student

    def _save_student(self, student, config, path: Path) -> Path:
        import random

        from spire_learn.neural.checkpoint import save_checkpoint

        torch = self.torch
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        generators = [torch.Generator(device="cpu") for _ in range(2)]
        save_checkpoint(path, policy=student, optimizer=optimizer, rng_state={
            "policy": {"torch": generators[0].get_state()}, "minibatch": {"torch": generators[1].get_state()},
            "curriculum": {"python": random.Random(1).getstate()},
            "controller": {"python": random.Random(2).getstate()}},
            metadata={"identity": _identity(config), "run_metadata": _run_metadata(),
                      "training_state": _distill_state(), "parent_checkpoint": None})
        return path

    def test_export_logits_equal_and_value_scaled_by_c(self):
        import random

        from spire_learn.neural.checkpoint import identity_payload, load_checkpoint, preflight_checkpoint
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model import build_policy

        torch = self.torch
        c = 0.625
        model_id, features_id, _ = STANDARD_LEARNER_IDENTITY_IDS
        batch = encode_frames(self.frames, self.schema, device="cpu")
        for template_path in TEMPLATES:
            config = load_neural_config(template_path)
            identity = config.accepted_identities[0]
            with self.subTest(template=template_path.parent.name):
                student = self._student(config)
                source = self._save_student(student, config, self.root / f"s1-{template_path.parent.name}")
                output = self.root / f"ppo-{template_path.parent.name}"
                result = distill_export.export_ppo_checkpoint(source, config, c, output=output, schema=self.schema,
                                                              critic_record=_record(c))
                self.assertEqual(result, output)
                for purpose in ("warm_start", "evaluate"):
                    preflight_checkpoint(output, identity, purpose=purpose)
                loaded = load_checkpoint(output, identity, purpose="resume", device="cpu")
                metadata = loaded.metadata
                # The template's identity and hashes; a fresh optimizer; the template's fresh RNG streams.
                self.assertEqual(json.loads(json.dumps(metadata["identity"], default=dict)),
                                 json.loads(json.dumps(identity_payload(identity))))
                state = metadata["training_state"]
                self.assertEqual((state["optimizer_step"], state["behavior_policy_version"],
                                  state["hyperparameter_id"]), (0, 0, identity.config_hashes["ppo"]))
                self.assertTrue(all(value == 0 for value in state["counters"].values()))
                self.assertEqual(loaded.optimizer_state["state"], {})
                self.assertEqual(loaded.optimizer_state["param_groups"][0]["lr"], config.ppo.learning_rate)
                fresh = distill_export.fresh_rng_state(config, torch)
                for domain in ("policy", "minibatch"):
                    self.assertTrue(torch.equal(loaded.rng_state[domain], fresh[domain]))
                self.assertEqual(loaded.rng_state["curriculum"], random.Random(config.seed_streams.curriculum).getstate())
                self.assertEqual(loaded.rng_state["controller"], random.Random(config.seed_streams.noncombat).getstate())
                self.assertEqual(dict(metadata["parent_checkpoint"]),
                                 {"path": str(source), "sha256": checkpoint_sha256(source)})
                run = metadata["run_metadata"]
                self.assertEqual((run["critic_scale"], run["gamma"]), (c, GAMMA))
                self.assertEqual(list(run["dropped_tensors"]), ["aux_value_head.bias", "aux_value_head.weight"])
                # Trunk and policy head copied, outcome head scaled by c, auxiliary heads dropped.
                original = student.state_dict()
                tensors = loaded.model_tensors
                self.assertEqual(set(tensors), {name for name in original if not name.startswith("aux_value_head.")})
                for name, tensor in tensors.items():
                    expected = original[name] * c if name in distill_export.SCALED_TENSORS else original[name]
                    self.assertTrue(torch.equal(tensor, expected), name)
                ppo = build_policy(self.schema, config.model, seed=99, device="cpu", model_id=model_id,
                                   features_id=features_id)
                ppo.load_state_dict(dict(tensors), strict=True)
                student.eval()
                ppo.eval()
                with torch.no_grad():
                    taught, exported = student(batch), ppo(batch)
                self.assertTrue(torch.equal(taught.logits, exported.logits))
                self.assertTrue(torch.allclose(exported.values, c * taught.values, rtol=1e-6, atol=1e-7))
                self.assertFalse(torch.allclose(exported.values, taught.values))
                self.assertIsNone(exported.aux_values)
                # The export is deterministic, and its own output (a PPO checkpoint) never exports.
                again = distill_export.export_ppo_checkpoint(source, config, c, output=output.with_name("again"),
                                                             schema=self.schema, critic_record=_record(c))
                self.assertEqual(json.loads((again / "metadata.json").read_text())["model_tensors"],
                                 json.loads((output / "metadata.json").read_text())["model_tensors"])
                with self.assertRaisesRegex(distill_export.ExportError, "only a distillation checkpoint"):
                    distill_export.export_ppo_checkpoint(output, config, c, output=output.with_name("twice"),
                                                         schema=self.schema, critic_record=_record(c))
                with self.assertRaisesRegex(distill_export.ExportError, "already exists"):
                    distill_export.export_ppo_checkpoint(source, config, c, output=output, schema=self.schema,
                                                         critic_record=_record(c))
                shutil.rmtree(again)

    def _shard(self, name: str, fights: list[tuple[tuple[int, int], int, bool]]) -> Path:
        """One shard with ``rows`` decisions per (case, rows, value_ok) fight, in order."""
        items, index = [], 0
        for case, rows, value_ok in fights:
            for _ in range(rows):
                frame = self.frames[index % len(self.frames)]
                labels = replace(self.shards.labels_for(frame, index, value_ok=value_ok), case=case)
                items.append((frame, labels))
                index += 1
        path = self.root / name
        self.shards.write_shard(path, self.schema, items)
        return path

    def test_critic_scale_from_fight_rows(self):
        first = self._shard("a", [((1000, 0), 3, True), ((1000, 1), 2, False), ((1000, 2), 1, True)])
        second = self._shard("b", [((1001, 0), 2, True)])
        # remaining = (fight_rows - 1) - row_in_fight, over value rows only (the second fight has none).
        terms = [GAMMA ** 2, GAMMA ** 1, GAMMA ** 0, GAMMA ** 0, GAMMA ** 1, GAMMA ** 0]
        empty = self._shard("empty", [])  # a zero-row shard lists no files and adds nothing
        details = distill_export.critic_scale_details([first, empty, second], gamma=GAMMA)
        self.assertAlmostEqual(details["critic_scale"], math.fsum(terms) / len(terms), places=15)
        self.assertEqual((details["value_rows"], details["fights"], details["value_fights"]), (6, 4, 3))
        self.assertEqual(distill_export.compute_critic_scale([first, second], gamma=GAMMA), details["critic_scale"])
        self.assertEqual(distill_export.compute_critic_scale([first], gamma=1.0), 1.0)
        for name, fights, error in (
                ("split", [((1000, 0), 1, True), ((1000, 1), 1, True), ((1000, 0), 1, True)], "contiguous"),
                ("none", [((1000, 0), 2, False)], "no value rows")):
            with self.subTest(refused=name), self.assertRaisesRegex(distill_export.ExportError, error):
                distill_export.compute_critic_scale([self._shard(name, fights)], gamma=GAMMA)
        with self.assertRaisesRegex(distill_export.ExportError, "contiguous"):
            distill_export.compute_critic_scale([first, self._shard("again", [((1000, 2), 1, True)])], gamma=GAMMA)
        mixed = self._shard("mixed", [((1000, 0), 1, True)])
        items = [(self.frames[0], replace(self.shards.labels_for(self.frames[0], 0, value_ok=True), case=(1, 1))),
                 (self.frames[1], replace(self.shards.labels_for(self.frames[1], 1, value_ok=False), case=(1, 1)))]
        self.shards.write_shard(self.root / "uneven", self.schema, items)
        with self.assertRaisesRegex(distill_export.ExportError, "same for every row"):
            distill_export.compute_critic_scale([mixed, self.root / "uneven"], gamma=GAMMA)
        with self.assertRaisesRegex(distill_export.ExportError, "gamma"):
            distill_export.compute_critic_scale([first], gamma=1)
        import numpy as np

        np.save(first / "labels.value_ok.npy", np.zeros(6, dtype=np.bool_), allow_pickle=False)
        with self.assertRaisesRegex(distill_export.ExportError, "fails its sha256"):
            distill_export.compute_critic_scale([first], gamma=GAMMA)


if __name__ == "__main__":
    unittest.main()
