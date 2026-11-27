"""CheckpointPlayer on a tiny FF policy checkpoint built in a temp dir (torch; runtime container only).

Runs through the ``--policy-player`` native route of ``gated_tests_entry.py``. numpy and torch are
imported lazily so host discovery never needs them.
"""

from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest

from spire_learn.neural.contracts import ModelConfig

TINY = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000)
MAPPING = "spire-learn-standard-combat-compact-v1@fixture+test"


def _load_modules() -> dict:
    try:
        import numpy  # noqa: F401
        import torch
    except ImportError:
        raise unittest.SkipTest("numpy and torch are only importable in the runtime container")
    from tests.neural import test_distill_data as shards

    shards._load_numpy_modules()
    return {"torch": torch, "shards": shards}


def save_fixture_checkpoint(path: Path, schema, model_config: ModelConfig, identity, *, seed: int = 5) -> Path:
    """Save a freshly initialised FF policy as a checkpoint (the trainer's own save helper)."""
    import torch
    from spire_learn.neural.checkpoint import identity_payload, save_checkpoint
    from spire_learn.neural.model import build_policy

    policy = build_policy(schema, model_config, seed=seed, device="cpu", objective_id=identity.objective_id,
                          model_id=identity.model_id, features_id=identity.features_id)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    # Like a trained checkpoint, the saved identity also carries the PPO hash (evaluation ignores it).
    saved = identity_payload(identity)
    saved["config_hashes"] = {**saved["config_hashes"], "ppo": "b" * 64}
    save_checkpoint(
        Path(path), policy=policy, optimizer=optimizer,
        rng_state={"policy": {"torch": torch.Generator(device="cpu").manual_seed(seed).get_state()},
                   "minibatch": {"torch": torch.Generator(device="cpu").manual_seed(seed + 1).get_state()},
                   "curriculum": {"python": random.Random(seed).getstate()},
                   "controller": {"python": random.Random(seed + 1).getstate()}},
        metadata={
            "identity": saved,
            "run_metadata": {"fixture": "tests.neural.test_policy_player"},
            "training_state": {
                "optimizer_step": 0, "hyperparameter_id": "b" * 64, "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
                "counters": {"environment_actions": 0, "learned_decisions": 0, "updates": 0},
                "update_id": "fixture", "behavior_policy_version": 0,
                "next_environment_seed_ordinal": 0, "next_decision_sequence": 0,
                "saved_inflight_record_count": 0,
            },
            "parent_checkpoint": None,
        })
    return Path(path)


class CheckpointPlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = _load_modules()
        cls.torch, cls.shards = modules["torch"], modules["shards"]
        from spire_learn.neural import policy_player

        cls.module = policy_player
        cls.schema = cls.shards._schema()
        cls.temp = tempfile.TemporaryDirectory()
        cls.identity = policy_player.standard_ff_identity(TINY, MAPPING)
        cls.path = save_fixture_checkpoint(Path(cls.temp.name) / "checkpoint", cls.schema, TINY, cls.identity)
        cls.frame = cls.shards.build_frame(
            cls.schema, 1, {"permanent_cards": [0, 1, 2], "actions": [0, 3, 4, 9], "potions": [1, 3],
                            "orbs": [0, 2, 3]}, {"potions": [0, 4], "orbs": [1, 5, 9]})
        cls.single = cls.shards.build_frame(cls.schema, 2, {"enemies": [1], "actions": [2]}, {})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def player(self, **kwargs):
        options = {"seed": 7, "expected_identity": self.identity, "model_config": TINY, **kwargs}
        return self.module.CheckpointPlayer(self.path, self.schema, **options)

    def test_deterministic_act_is_the_legal_argmax(self):
        from spire_learn.neural.features import encode_frames

        player = self.player(deterministic=True)
        with self.torch.no_grad():
            logits = player._policy(encode_frames([self.frame], self.schema, device="cpu")).logits[0]
        legal = [0, 3, 4, 9]
        expected = max(legal, key=lambda action: float(logits[action]))
        self.assertEqual([player.act(self.frame, legal) for _ in range(3)], [expected] * 3)
        self.assertEqual(self.player(deterministic=True, seed=99).act(self.frame, legal), expected)

    def test_seeded_sampling_is_reproducible(self):
        legal = [0, 3, 4, 9]
        first, second = self.player(seed=11), self.player(seed=11)
        draws = [first.act(self.frame, legal) for _ in range(6)]
        self.assertEqual(draws, [second.act(self.frame, legal) for _ in range(6)])
        self.assertTrue(set(draws) <= set(legal))
        first.reseed(11)
        self.assertEqual(draws, [first.act(self.frame, legal) for _ in range(6)])

    def test_single_legal_plays_without_sampling(self):
        player = self.player()
        state = player._generator.get_state()
        player._policy.act = None  # any policy call would fail
        self.assertEqual(player.act(self.single, [2]), 2)
        self.assertTrue(self.torch.equal(player._generator.get_state(), state))

    def test_legal_set_must_match_the_frame_mask(self):
        player = self.player(deterministic=True)
        from spire_learn.contracts import ContractError

        with self.assertRaises(ContractError):
            player.act(self.frame, [0, 3, 4])
        with self.assertRaises(ContractError):
            player.act(self.single, [])

    def test_refuses_a_mismatched_checkpoint(self):
        from dataclasses import replace

        from spire_learn.contracts import ContractError
        from spire_learn.neural.checkpoint import CheckpointCompatibilityError

        wider = ModelConfig(embedding_width=4, hidden_width=16, max_parameters=10_000_000)
        with self.assertRaises(CheckpointCompatibilityError):
            self.player(expected_identity=self.module.standard_ff_identity(wider, MAPPING), model_config=wider)
        with self.assertRaises(CheckpointCompatibilityError):
            self.player(expected_identity=self.module.standard_ff_identity(TINY, MAPPING + "-other"))
        with self.assertRaises(CheckpointCompatibilityError):
            self.player(expected_identity=replace(self.identity, objective_id="spire-learn-combat-winloss-hpshaped-v1"))
        with self.assertRaisesRegex(ContractError, "model config"):
            self.player(model_config=wider)
        with self.assertRaisesRegex(ContractError, "FF"):
            self.player(expected_identity=replace(self.identity, model_id="spire-learn-combat-transformer-v1"))

    def test_torch_is_pinned_to_one_thread(self):
        self.player()
        self.assertEqual((self.torch.get_num_threads(), self.torch.get_num_interop_threads()), (1, 1))


if __name__ == "__main__":
    unittest.main()
