"""Immutable, dependency-free records shared by learner components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TypeAlias


class ContractError(ValueError):
    """A public decision or record violates its declared contract."""


class SetupBlocked(RuntimeError):
    """A requested setup is not supported by the selected contract."""


@dataclass(frozen=True)
class ArrayValue:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[int | float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ContractError("dtype must be a non-empty string")
        if not isinstance(self.shape, tuple) or any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ContractError("shape must contain non-negative integer dimensions")
        if not isinstance(self.values, tuple):
            raise ContractError("values must be a tuple")
        count = math.prod(self.shape)
        if len(self.values) != count:
            raise ContractError(f"shape requires {count} values, got {len(self.values)}")
        for value in self.values:
            if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
                raise ContractError("array values must be finite int or float primitives")


Observation: TypeAlias = Mapping[str, ArrayValue]


def frozen_mapping(values: Mapping[str, ArrayValue]) -> Observation:
    copied = dict(values)
    if any(not isinstance(key, str) or not isinstance(value, ArrayValue) for key, value in copied.items()):
        raise ContractError("observations map string keys to ArrayValue records")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class Candidate:
    index: int
    kind: str
    source: int | None
    target: int | None
    features: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PublicDecision:
    observation: Observation
    candidates: tuple[Candidate, ...]
    mask: tuple[bool, ...]
    schema_fingerprint: str
    action_mapping_id: str


@dataclass(frozen=True)
class Route:
    execution_id: str
    slot: int
    episode_id: str
    decision_id: str
    episode_start: bool


@dataclass(frozen=True)
class DecisionRequest:
    public: PublicDecision
    route: Route


@dataclass(frozen=True)
class Choice:
    action: int
    next_state: object | None = None


@dataclass(frozen=True)
class DecisionResult:
    decision_id: str
    choice: Choice


@dataclass(frozen=True)
class Transition:
    route: Route
    source: Observation
    action: int
    reward: float
    terminated: bool
    truncated: bool
    successor: Observation
    replacement: Observation | None
    state_after_source: object | None


@dataclass(frozen=True)
class CaseExecution:
    execution_id: str
    case_id: str
    split: str
    partition: str
    scenario_id: str
    environment_seed: int
    policy_seed: int
    expected_schema_id: str | None = None
    expected_fingerprint: str | None = None


@dataclass(frozen=True)
class Failure:
    kind: str
    reason: str
    phase: str
    exception_type: str | None = None


@dataclass(frozen=True)
class CaseResult:
    execution: CaseExecution
    status: str
    started: bool
    outcome: str | None
    steps: int
    reward_sum: float
    stop_reason: str
    final_resources: Mapping[str, int | None]
    primary_failure: Failure | None = None
    secondary_errors: tuple[Failure, ...] = ()
    observed_descriptor: Mapping[str, object] | None = None
    native_build_info: Mapping[str, object] | None = None
