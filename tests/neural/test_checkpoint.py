"""Opt-in CPU checkpoint round-trip and recovery tests.

The class skips before importing Torch unless the protected Task 6 runner has
explicitly enabled its gate. These tests do not sample actions or touch Gym.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.neural.config import RUN_RAW_MAPPING_ID, STANDALONE_RAW_MAPPING_ID
from spire_learn.neural.contracts import NeuralIdentity


def _identity(*, modes=("standalone", "run"), features_id="spire-learn-combat-features-v1"):
    mapping_ids = {"standalone": STANDALONE_RAW_MAPPING_ID, "run": RUN_RAW_MAPPING_ID}
    return NeuralIdentity(
        model_id="spire-learn-combat-ff-v1",
        features_id=features_id,
        action_id="spire-learn-combat-dispatch-v1",
        objective_id="spire-learn-combat-winloss-v1",
        accepted_raw_mode_mappings={mode: mapping_ids[mode] for mode in modes},
        config_hashes={"model": "a" * 64, "ppo": "b" * 64},
    )


def _training_state(torch):
    policy_rng = torch.Generator(device="cpu").manual_seed(101)
    minibatch_rng = torch.Generator(device="cpu").manual_seed(202)
    curriculum_rng = random.Random(303)
    controller_rng = random.Random(404)
    return {
        "policy": {"python": random.Random(101).getstate(), "torch": policy_rng.get_state()},
        "minibatch": {"torch": minibatch_rng.get_state()},
        "curriculum": {"python": curriculum_rng.getstate()},
        "controller": {"python": controller_rng.getstate()},
    }


def _metadata(identity):
    from spire_learn.neural.checkpoint import identity_payload

    return {
        "identity": identity_payload(identity),
        "run_metadata": {
            "mode": "standalone",
            "seed_streams": {"policy": 11, "optimizer": 13, "curriculum": 17, "noncombat": 19},
            "partition": "train",
            "budgets": {"environment_actions": 128, "learned_decisions": 64},
            "native_build_id": "accepted-public-build",
        },
        "training_state": {
            "optimizer_step": 1,
            "hyperparameter_id": "b" * 64,
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": {"environment_actions": 20, "learned_decisions": 8, "updates": 1},
            "update_id": "update-1",
            "behavior_policy_version": 1,
            "next_environment_seed_ordinal": 5,
            "next_decision_sequence": 40,
            "saved_inflight_record_count": 3,
        },
        "parent_checkpoint": None,
    }


def _state_hash(state, torch):
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode("utf-8"))

    add(state)
    return digest.hexdigest()


class NeuralCheckpointTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (os.environ.get("SPIRE_LEARN_RUN_NEURAL_TENSOR") != "1"
                or os.environ.get("SPIRE_LEARN_TENSOR_TASK") != "task6"):
            raise unittest.SkipTest("Task 6 checkpoint tests require the protected tensor gate")
        import torch
        from tests.neural._sample_guard import activate_from_environment

        cls.torch = torch
        cls.sample_guard = activate_from_environment()
        from importlib import import_module

        checkpoint_module = import_module("spire_learn.neural.checkpoint")
        required = ("load_checkpoint", "save_checkpoint")
        missing = sorted(name for name in required if not callable(getattr(checkpoint_module, name, None)))
        if missing:
            raise RuntimeError(
                "TASK6_CHECKPOINT_API_MISSING module=spire_learn.neural.checkpoint names="
                + ",".join(missing)
            )
        cls.load_checkpoint = staticmethod(checkpoint_module.load_checkpoint)
        cls.save_checkpoint = staticmethod(checkpoint_module.save_checkpoint)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        torch = self.torch
        torch.manual_seed(77)
        self.policy = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        loss = self.policy(torch.tensor([[1.0, 2.0, 3.0]])).square().mean()
        loss.backward()
        self.optimizer.step()
        self.identity = _identity()
        self.rng_state = _training_state(torch)
        self.source_model_hash = _state_hash(self.policy.state_dict(), torch)
        self.source_optimizer_hash = _state_hash(self.optimizer.state_dict(), torch)
        self.source_rng_hash = _state_hash(self.rng_state, torch)
        self.checkpoint = Path(self.temp.name) / "model-checkpoint"
        self.save_checkpoint(
            self.checkpoint,
            policy=self.policy,
            optimizer=self.optimizer,
            rng_state=self.rng_state,
            metadata=_metadata(self.identity),
        )

    def test_resume_round_trips_named_weights_optimizer_rng_and_counters(self):
        loaded = self.load_checkpoint(
            self.checkpoint, self.identity, purpose="resume", device="cpu"
        )
        self.assertEqual(_state_hash(self.policy.state_dict(), self.torch), self.source_model_hash)
        self.assertEqual(_state_hash(self.optimizer.state_dict(), self.torch), self.source_optimizer_hash)
        self.assertEqual(_state_hash(self.rng_state, self.torch), self.source_rng_hash)
        self.assertEqual(_state_hash(loaded.model_tensors, self.torch), self.source_model_hash)
        self.assertEqual(_state_hash(loaded.optimizer_state, self.torch), self.source_optimizer_hash)
        self.assertEqual(_state_hash(loaded.rng_state, self.torch), self.source_rng_hash)
        self.assertEqual(set(loaded.model_tensors), set(self.policy.state_dict()))
        for name, tensor in self.policy.state_dict().items():
            self.assertTrue(self.torch.equal(loaded.model_tensors[name], tensor.cpu()))
        self.assertIsNotNone(loaded.optimizer_state)
        self.assertIsNotNone(loaded.rng_state)
        self.assertEqual(loaded.optimizer_step, 1)
        self.assertEqual(loaded.inflight_record_count, 0)
        self.assertEqual(loaded.metadata["recovery"]["dropped_inflight_record_count"], 3)
        training_metadata = loaded.metadata["training_state"]
        self.assertEqual(
            training_metadata["counters"],
            {"environment_actions": 20, "learned_decisions": 8, "updates": 1},
        )
        self.assertEqual(training_metadata["next_environment_seed_ordinal"], 5)
        self.assertEqual(training_metadata["next_decision_sequence"], 40)
        self.assertEqual(training_metadata["update_id"], "update-1")
        self.assertEqual(training_metadata["behavior_policy_version"], 1)
        self.assertEqual(
            set(loaded.rng_state), {"policy", "minibatch", "curriculum", "controller"}
        )

        existing = {
            child.name: hashlib.sha256(child.read_bytes()).hexdigest()
            for child in self.checkpoint.iterdir()
        }
        with self.assertRaises(FileExistsError):
            self.save_checkpoint(
                self.checkpoint,
                policy=self.policy,
                optimizer=self.optimizer,
                rng_state=self.rng_state,
                metadata=_metadata(self.identity),
            )
        self.assertEqual(
            {
                child.name: hashlib.sha256(child.read_bytes()).hexdigest()
                for child in self.checkpoint.iterdir()
            },
            existing,
        )

        failed_path = self.checkpoint.parent / "failed-write-checkpoint"
        original_torch_save = self.torch.save
        save_calls = 0

        def fail_training_payload(value, destination, *args, **kwargs):
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                raise OSError("injected training payload write failure")
            return original_torch_save(value, destination, *args, **kwargs)

        with patch.object(self.torch, "save", side_effect=fail_training_payload):
            with self.assertRaisesRegex(OSError, "injected training payload write failure"):
                self.save_checkpoint(
                    failed_path,
                    policy=self.policy,
                    optimizer=self.optimizer,
                    rng_state=self.rng_state,
                    metadata=_metadata(self.identity),
                )
        self.assertFalse(failed_path.exists())
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [self.checkpoint])

        invalid_policy = copy.deepcopy(self.policy)
        with self.torch.no_grad():
            invalid_policy.weight[0, 0] = float("nan")
        nonfinite_path = self.checkpoint.parent / "nonfinite-checkpoint"
        with self.assertRaisesRegex(ValueError, "finite"):
            self.save_checkpoint(
                nonfinite_path,
                policy=invalid_policy,
                optimizer=self.optimizer,
                rng_state=self.rng_state,
                metadata=_metadata(self.identity),
            )
        self.assertFalse(nonfinite_path.exists())
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [self.checkpoint])

    def test_evaluation_load_is_frozen_and_omits_optimizer_and_rng(self):
        before_hash = _state_hash(self.policy.state_dict(), self.torch)
        before_optimizer = _state_hash(self.optimizer.state_dict(), self.torch)
        matrix = self.torch.arange(6, dtype=self.torch.float32).reshape(2, 3)
        transposed = matrix.t()
        self.assertFalse(transposed.is_contiguous())
        self.assertEqual(
            _state_hash({"layer.weight": transposed}, self.torch),
            _state_hash({"layer.weight": transposed.contiguous()}, self.torch),
        )
        self.assertNotEqual(
            _state_hash({"layer.weight": matrix}, self.torch),
            _state_hash({"layer.bias": matrix}, self.torch),
        )
        scalar_hash = _state_hash({"step": self.torch.tensor(1.0)}, self.torch)
        self.assertRegex(scalar_hash, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            _state_hash({"layer.weight": matrix}, self.torch),
            _state_hash({"layer.weight": matrix.to(dtype=self.torch.float64)}, self.torch),
        )
        loaded = self.load_checkpoint(
            self.checkpoint, self.identity, purpose="evaluate", device="cpu"
        )
        self.assertEqual(_state_hash(loaded.model_tensors, self.torch), before_hash)
        self.assertEqual(_state_hash(self.policy.state_dict(), self.torch), before_hash)
        self.assertEqual(_state_hash(self.optimizer.state_dict(), self.torch), before_optimizer)
        self.assertIsNone(loaded.optimizer_state)
        self.assertIsNone(loaded.rng_state)
        self.assertEqual(loaded.optimizer_step, 0)
        self.assertEqual(loaded.inflight_record_count, 0)

    def test_warm_start_transfers_weights_but_drops_optimizer_rng_and_counters(self):
        loaded = self.load_checkpoint(
            self.checkpoint, self.identity, purpose="warm_start", device="cpu"
        )
        self.assertEqual(set(loaded.model_tensors), set(self.policy.state_dict()))
        self.assertEqual(_state_hash(loaded.model_tensors, self.torch), self.source_model_hash)
        self.assertIsNone(loaded.optimizer_state)
        self.assertIsNone(loaded.rng_state)
        self.assertEqual(loaded.optimizer_step, 0)
        self.assertEqual(loaded.inflight_record_count, 0)
        self.assertEqual(loaded.metadata["recovery"]["parent_checkpoint"], str(self.checkpoint))

    def test_transfer_allows_mode_and_seed_change_but_rejects_encoder_change(self):
        run_identity = _identity(modes=("run",))
        transferred = self.load_checkpoint(
            self.checkpoint, run_identity, purpose="warm_start", device="cpu"
        )
        self.assertEqual(transferred.metadata["run_metadata"]["mode"], "standalone")
        incompatible = _identity(modes=("run",), features_id="spire-learn-other-features-v1")
        with self.assertRaisesRegex(ValueError, "features_id"):
            self.load_checkpoint(
                self.checkpoint, incompatible, purpose="warm_start", device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
