"""Pure neural learner contracts and configuration; optional runtimes stay lazy."""

from spire_learn.neural.contracts import (
    BufferArray,
    CombatFrame,
    ContractError,
    CumulativeCounters,
    Lifecycle,
    ModelConfig,
    NeuralIdentity,
    PPOConfig,
    PolicyChoice,
    ReturnBatch,
    Rollout,
    RolloutCounters,
    StepRecord,
    TaskSignal,
)

__all__ = [
    "BufferArray",
    "CombatFrame",
    "ContractError",
    "CumulativeCounters",
    "Lifecycle",
    "ModelConfig",
    "NeuralIdentity",
    "PPOConfig",
    "PolicyChoice",
    "ReturnBatch",
    "Rollout",
    "RolloutCounters",
    "StepRecord",
    "TaskSignal",
]
