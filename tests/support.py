"""Small authored protocol fixtures; these do not simulate game behavior."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from spire_learn.contracts import (
    ArrayValue, CaseExecution, CaseResult, Choice, Observation, PublicDecision,
    frozen_mapping,
)
from spire_learn.schema import candidates_for, parse_descriptor


def tiny_descriptor(*, action_count: int = 3) -> dict[str, object]:
    return {
        "schema_id": "spire-learn-fixture-v1",
        "content_id": "spire-learn-fixture-content-v1",
        "fingerprint": "a" * 64,
        "action_space": {"n": action_count},
        "tensors": [
            {"key": "action_mask", "dtype": "int8", "shape": [action_count], "fields": [], "axes": ["candidate"], "category": None, "mask": None},
            {"key": "action_kind", "dtype": "int64", "shape": [action_count], "fields": [], "axes": ["candidate"], "category": "action", "mask": "action_mask"},
            {"key": "action_features", "dtype": "int64", "shape": [action_count, 1], "fields": ["hp"], "axes": ["candidate", "field"], "category": None, "mask": None},
            {"key": "player", "dtype": "int64", "shape": [], "fields": ["hp"], "axes": [], "category": None, "mask": None},
        ],
        "vocabularies": {"action": ["padding", "end_turn", "play_card"], "outcome": ["unknown", "victory", "death"]},
    }


def observation(*, mask: tuple[int, ...] = (1, 1, 0), kinds: tuple[int, ...] = (1, 2, 0), hp: int = 10) -> Observation:
    if len(mask) != len(kinds):
        raise ValueError("mask and kinds widths must match")
    return frozen_mapping({
        "action_mask": ArrayValue("int8", (len(mask),), mask),
        "action_kind": ArrayValue("int64", (len(kinds),), kinds),
        "action_features": ArrayValue("int64", (len(mask), 1), (hp,) * len(mask)),
        "player": ArrayValue("int64", (), (hp,)),
    })


@dataclass(frozen=True)
class _DType:
    name: str
    isnative: bool


class FakeArray:
    def __init__(self, dtype_name: str, shape: tuple[int, ...], values: tuple[int | float, ...], *, isnative: bool = True):
        self.dtype = _DType(dtype_name, isnative)
        self.shape = shape
        self._values = values

    def tolist(self) -> list[int | float]:
        return list(self._values)


class ScriptedEnv:
    def __init__(self, reset_result: tuple[object, dict], steps: Sequence[tuple[object, float, bool, bool, dict] | Exception], *, descriptor: Mapping[str, object] | None = None, on_step: Callable[[], None] | None = None):
        self.reset_result = reset_result
        self.steps = list(steps)
        self.descriptor = descriptor if descriptor is not None else tiny_descriptor()
        self.on_step = on_step
        self.reset_calls: list[tuple[object, object]] = []
        self.step_calls: list[object] = []
        self.close_calls = 0

    def reset(self, *, seed: object = None, options: object = None) -> tuple[object, dict]:
        self.reset_calls.append((seed, options))
        return self.reset_result

    def step(self, action: object) -> tuple[object, float, bool, bool, dict]:
        self.step_calls.append(action)
        if not self.steps:
            raise RuntimeError("scripted steps exhausted")
        result = self.steps.pop(0)
        if isinstance(result, Exception):
            raise result
        if self.on_step is not None:
            self.on_step()
        return result

    def close(self) -> None:
        self.close_calls += 1

    def observation_spec(self) -> Mapping[str, object]:
        return self.descriptor


class FakeClock:
    def __init__(self, start: float = 0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


class ScriptedAgent:
    def __init__(self, choices: Sequence[Choice | Exception]):
        self.choices = list(choices)

    def init_state(self) -> object | None:
        return None

    def act(self, public: PublicDecision, state: object | None) -> Choice:
        if not self.choices:
            raise RuntimeError("scripted choices exhausted")
        choice = self.choices.pop(0)
        if isinstance(choice, Exception):
            raise choice
        return choice


class FakeNativeErrors:
    class InputError(Exception): pass
    class StateError(Exception): pass
    class SetupError(Exception): pass
    class FaultError(Exception): pass


def case_execution(**overrides: object) -> CaseExecution:
    values: dict[str, object] = {
        "execution_id": "fixture-execution", "case_id": "fixture-case", "split": "selection",
        "partition": "fixture", "scenario_id": "fixture-scenario", "environment_seed": 17,
        "policy_seed": 29, "expected_schema_id": "spire-learn-fixture-v1", "expected_fingerprint": None,
    }
    values.update(overrides)
    return CaseExecution(**values)  # type: ignore[arg-type]


def case_result(**overrides: object) -> CaseResult:
    values: dict[str, object] = {
        "execution": case_execution(), "status": "terminal", "started": True, "outcome": "victory",
        "steps": 1, "reward_sum": 1.0, "stop_reason": "terminated", "final_resources": {},
    }
    values.update(overrides)
    return CaseResult(**values)  # type: ignore[arg-type]


def public_decision(*, mask: tuple[int, ...] = (1, 1, 0), kinds: tuple[int, ...] = (1, 2, 0)) -> PublicDecision:
    obs = observation(mask=mask, kinds=kinds)
    schema = parse_descriptor(tiny_descriptor(action_count=len(mask)), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
    return PublicDecision(obs, candidates_for(obs, schema), tuple(bool(v) for v in mask), "a" * 64, "spire-learn-fixture-actions-v1")
