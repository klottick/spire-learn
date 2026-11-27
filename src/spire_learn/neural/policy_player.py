"""Play combat moves from a saved feed-forward policy checkpoint (torch; runtime container only).

``CheckpointPlayer`` loads a checkpoint for evaluation on CPU, refuses one whose identity or model
config differs from the expected one, and picks one legal action per public compact frame: a sample
from its own seeded generator, or the argmax with ``deterministic=True``. A single legal action is
played without consulting the policy. It never builds a training runtime (no ``create_runtime``).

Only the FF family is handled here; Task 8 dispatches through the ``ModelFamily`` registry.
Importing this module imports neither torch nor numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from spire_learn.contracts import ContractError
from spire_learn.neural.config import LEARNER_IDENTITY_IDS, STANDARD_LEARNER_IDENTITY_IDS, WINLOSS_OBJECTIVE_ID
from spire_learn.neural.config import model_config_hash as _model_config_hash
from spire_learn.neural.contracts import CombatFrame, ModelConfig, NeuralIdentity
from spire_learn.neural.schema import CombatSchema

FF_MODEL_IDS = (LEARNER_IDENTITY_IDS[0], STANDARD_LEARNER_IDENTITY_IDS[0])
_SEED_LIMIT = 2 ** 63


def model_config_hash(model_config: ModelConfig) -> str:
    """The identity's ``config_hashes["model"]``: the canonical hash of ``model_config_hash_payload``."""
    return _model_config_hash(model_config)


def standard_ff_identity(model_config: ModelConfig, raw_mode_mapping: str, *,
                         objective_id: str = WINLOSS_OBJECTIVE_ID) -> NeuralIdentity:
    """The Standard FF learner identity a ``standard-standalone`` checkpoint must declare to be evaluated."""
    model_id, features_id, action_id = STANDARD_LEARNER_IDENTITY_IDS
    return NeuralIdentity(
        model_id=model_id, features_id=features_id, action_id=action_id, objective_id=objective_id,
        accepted_raw_mode_mappings={"standard-standalone": raw_mode_mapping},
        config_hashes={"model": model_config_hash(model_config)},
    )


def pin_torch_threads(torch) -> None:
    """One intra-op and one inter-op thread per process (labelling workers share the host's cores)."""
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)  # raises once parallel work has started: an error, never ignored


def _mask_legal(frame: CombatFrame) -> list[int]:
    mask = frame.tensors["action_mask"]
    if mask.dtype != "int8":
        raise ContractError("action_mask must be int8")
    return [index for index, flag in enumerate(mask.data) if flag]


class CheckpointPlayer:
    """A reflex combat player over one FF policy checkpoint."""

    def __init__(self, path: Path | str, schema: CombatSchema, *, seed: int, expected_identity: NeuralIdentity,
                 model_config: ModelConfig, deterministic: bool = False):
        if not isinstance(expected_identity, NeuralIdentity):
            raise TypeError("expected_identity must be a NeuralIdentity")
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig")
        if expected_identity.model_id not in FF_MODEL_IDS:
            raise ContractError(f"CheckpointPlayer plays the FF family only, not {expected_identity.model_id}")
        if expected_identity.config_hashes.get("model") != model_config_hash(model_config):
            raise ContractError("the model config does not match the expected identity's model config hash")
        if type(deterministic) is not bool:
            raise ContractError("deterministic must be a boolean")
        import torch

        from spire_learn.neural.checkpoint import load_checkpoint
        from spire_learn.neural.model import build_policy

        pin_torch_threads(torch)
        self._torch = torch
        loaded = load_checkpoint(Path(path), expected_identity, purpose="evaluate", device="cpu")
        policy = build_policy(schema, model_config, seed=0, device="cpu",
                              objective_id=expected_identity.objective_id, model_id=expected_identity.model_id,
                              features_id=expected_identity.features_id)
        policy.load_state_dict(dict(loaded.model_tensors), strict=True)
        policy.eval()
        self._policy = policy
        self.schema = schema
        self.deterministic = deterministic
        self.identity = dict(loaded.metadata["identity"])
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Restart the sampling generator (one stream per fight keeps play independent of scheduling)."""
        if type(seed) is not int or not 0 <= seed < _SEED_LIMIT:
            raise ContractError("player seed must be an integer in [0, 2**63)")
        self._generator = self._torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def act(self, frame: CombatFrame, legal: Sequence[int]) -> int:
        """One legal action for ``frame``; ``legal`` must be the frame's action-mask legal set."""
        legal = [int(action) for action in legal]
        if legal != _mask_legal(frame):
            raise ContractError("legal actions differ from the frame's action mask")
        if not legal:
            raise ContractError("the player has no legal action to play")
        if len(legal) == 1:
            return legal[0]
        choice, = self._policy.act([frame], generator=self._generator, deterministic=self.deterministic)
        action = choice.original_action_index
        if action not in legal:
            raise ContractError(f"the policy chose illegal action {action}")
        return action


__all__ = ["CheckpointPlayer", "FF_MODEL_IDS", "model_config_hash", "pin_torch_threads", "standard_ff_identity"]
