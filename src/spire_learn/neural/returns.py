"""Pure combat-task signaling and grouped generalized-advantage estimation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

from spire_learn.contracts import ContractError
from spire_learn.neural.contracts import (
    BufferArray,
    CombatFrame,
    Lifecycle,
    ReturnBatch,
    StepRecord,
    TaskSignal,
)
from spire_learn.neural.schema import TensorSpec, _unpack


_NATURAL_OUTCOMES = {"victory": 1.0, "death": -1.0, "escaped": 0.0}


def task_signal(
    previous: Lifecycle,
    current: Lifecycle,
    *,
    terminated: bool,
    truncated: bool,
    supports_truncation: bool = False,
) -> TaskSignal:
    """Derive combat-local reward without rewriting Gym termination semantics.

    A settled-count increase identifies a new natural combat task end. The sticky
    exit outcome is read only for that increase, so the same public exit cannot
    pay out again on subsequent noncombat rows.
    """
    if not isinstance(previous, Lifecycle) or not isinstance(current, Lifecycle):
        raise ContractError("task_signal requires previous/current Lifecycle records")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ContractError("native termination flags must be booleans")
    if type(supports_truncation) is not bool:
        raise ContractError("supports_truncation must be a boolean")
    if terminated and truncated:
        raise ContractError("native transition cannot be both terminated and truncated")
    if truncated and not supports_truncation:
        raise ValueError("truncation is unsupported by the selected environment contract")
    if current.combats_entered < previous.combats_entered:
        raise ContractError("combats_entered cannot move backwards")
    if current.combats_settled < previous.combats_settled:
        raise ContractError("combats_settled cannot move backwards")
    settled_delta = current.combats_settled - previous.combats_settled
    if settled_delta > 1:
        raise ContractError("one public transition cannot settle multiple combat tasks")

    reward = 0.0
    natural_task_end = settled_delta == 1
    if natural_task_end:
        if current.last_exit is None:
            raise ContractError("a newly settled combat requires its public last_exit outcome")
        outcome = current.last_exit.get("outcome")
        if not isinstance(outcome, str) or outcome not in _NATURAL_OUTCOMES:
            raise ContractError(f"unknown natural combat outcome: {outcome!r}")
        visit = current.last_exit.get("visit")
        expected_visit = current.combats_settled - 1
        if type(visit) is not int or visit != expected_visit:
            raise ContractError("natural combat exit visit does not match the settled counter")
        if visit >= current.combats_entered:
            raise ContractError("natural combat exit visit is outside entered combats")
        if previous.active_visit is not None and visit != previous.active_visit:
            raise ContractError("natural combat exit does not match the previously active visit")
        reward = _NATURAL_OUTCOMES[outcome]

    if natural_task_end or terminated:
        return TaskSignal(reward, True, False, False)
    if truncated:
        return TaskSignal(0.0, False, True, False)
    return TaskSignal(0.0, False, True, True)


def _enemy_tensor(frame: CombatFrame, key: str | None, spec_shape: tuple[int, ...],
                  dtype: str) -> tuple[int, ...]:
    array = frame.tensors.get(key) if key is not None else None
    if not isinstance(array, BufferArray) or array.dtype != dtype or array.shape != spec_shape:
        raise ContractError(f"hp shaping requires the public {key} tensor")
    return _unpack(array)


def hp_potential(frame: CombatFrame, schema: object) -> float:
    """Return minus the summed enemy HP fraction of a public combat frame.

    Only occupied ``enemies`` rows whose ``hp`` and ``max_hp`` cells are both
    known and whose ``max_hp`` is positive count. Columns are located by their
    descriptor field names. With no qualifying row the potential is zero.
    """
    if not isinstance(frame, CombatFrame):
        raise ContractError("hp_potential requires a public CombatFrame")
    tensors = getattr(schema, "tensors", None)
    spec = tensors.get("enemies") if hasattr(tensors, "get") else None
    if (not isinstance(spec, TensorSpec) or len(spec.shape) != 2 or spec.known_key is None
            or spec.occupied_key is None):
        raise ContractError("hp shaping requires the descriptor enemies row tensor")
    if frame.schema_fingerprint != getattr(schema, "schema_fingerprint", frame.schema_fingerprint):
        raise ContractError("frame schema fingerprint does not match the shaping schema")
    fields = tuple(column.field for column in spec.columns)
    if "hp" not in fields or "max_hp" not in fields:
        raise ContractError("hp shaping requires enemies hp and max_hp columns")
    hp_index, max_index = fields.index("hp"), fields.index("max_hp")
    rows, columns = spec.shape
    values = _enemy_tensor(frame, "enemies", spec.shape, spec.dtype)
    known = _enemy_tensor(frame, spec.known_key, spec.shape, "int8")
    occupied = _enemy_tensor(frame, spec.occupied_key, (rows,), "int8")
    hp_total = 0
    max_total = 0
    for row in range(rows):
        if occupied[row] != 1:
            continue
        hp_cell, max_cell = row * columns + hp_index, row * columns + max_index
        if known[hp_cell] != 1 or known[max_cell] != 1:
            continue
        maximum = values[max_cell]
        if maximum <= 0:
            continue
        hp_total += values[hp_cell]
        max_total += maximum
    if max_total == 0:
        return 0.0
    return -(hp_total / max_total)


def shaped_reward(
    reward: float,
    source_frame: CombatFrame,
    successor_frame: CombatFrame,
    *,
    terminal: bool,
    gamma: float,
    beta: float,
    schema: object,
) -> float:
    """Add ``beta * (gamma * phi(successor) - phi(source))`` to a task reward.

    The successor potential is zero at a task terminal. Source and successor
    are one transition's own frames, so shaping never spans a reset.
    """
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("reward must be a finite number")
    if type(terminal) is not bool:
        raise ValueError("terminal must be a boolean")
    gamma_value = _parameter(gamma, "gamma")
    if type(beta) not in (int, float) or not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must be a finite number in (0, 1]")
    successor = 0.0 if terminal else hp_potential(successor_frame, schema)
    source = hp_potential(source_frame, schema)
    return float(reward) + float(beta) * (gamma_value * successor - source)


def _parameter(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def compute_gae(
    records: Sequence[StepRecord],
    *,
    gamma: float,
    lam: float,
) -> ReturnBatch:
    """Compute aligned returns, recursing only within the same stream/task.

    A row's stored ``next_value`` was captured from its actual successor before
    any reset. It supplies bootstrap at a cut or supported truncation. A later
    row contributes the lambda recursion only when it is the next recorded row
    in the same stream/task and its source equals the preceding successor.
    Fixed-controller rows remain aligned but receive zero placeholders and a
    false train mask; their policy/value statistics are never fabricated.
    """
    gamma_value = _parameter(gamma, "gamma")
    lambda_value = _parameter(lam, "lam")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of StepRecord values")
    rows = tuple(records)
    if any(not isinstance(row, StepRecord) for row in rows):
        raise TypeError("records must contain only StepRecord values")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    seen_decisions: set[tuple[str, str]] = set()
    current_task_by_stream: dict[str, str] = {}
    closed_tasks: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        decision_key = (row.stream_id, row.decision_id)
        if decision_key in seen_decisions:
            raise ContractError("duplicate decision_id within a stream")
        seen_decisions.add(decision_key)
        previous_task = current_task_by_stream.get(row.stream_id)
        if previous_task is not None and previous_task != row.task_id:
            closed_tasks.add((row.stream_id, previous_task))
        task_key = (row.stream_id, row.task_id)
        if task_key in closed_tasks:
            raise ContractError("records cannot resume a closed task within one stream")
        current_task_by_stream[row.stream_id] = row.task_id
        grouped[(row.stream_id, row.task_id)].append(index)
        if row.terminated or row.task_signal.task_terminal:
            closed_tasks.add(task_key)

    advantages = [0.0] * len(rows)
    targets = [0.0] * len(rows)
    train_mask = [row.actor_training_mask for row in rows]
    for indices in grouped.values():
        gae_by_index: dict[int, float] = {}
        for position in range(len(indices) - 1, -1, -1):
            index = indices[position]
            row = rows[index]
            if not row.actor_training_mask:
                continue
            value = row.choice.value
            next_value = row.next_value
            assert value is not None and next_value is not None
            signal = row.task_signal
            terminal_boundary = row.terminated or signal.task_terminal
            bootstrap = 1.0 if signal.bootstrap and not terminal_boundary else 0.0
            continuation = (
                signal.continuation and not terminal_boundary and not row.truncated
            )
            delta = signal.reward + gamma_value * bootstrap * next_value - value

            next_gae = 0.0
            if continuation and position + 1 < len(indices):
                next_index = indices[position + 1]
                next_row = rows[next_index]
                if next_row.actor_training_mask:
                    if row.successor != next_row.source:
                        raise ContractError(
                            "adjacent same-task records do not join at the actual successor frame"
                        )
                    next_gae = gae_by_index.get(next_index, 0.0)
            advantage = delta + gamma_value * lambda_value * (
                1.0 if continuation else 0.0
            ) * next_gae
            gae_by_index[index] = advantage
            advantages[index] = advantage
            targets[index] = value + advantage

    return ReturnBatch(tuple(advantages), tuple(targets), tuple(train_mask),
                       tuple(row.decision_id for row in rows))


__all__ = ["compute_gae", "hp_potential", "shaped_reward", "task_signal"]
