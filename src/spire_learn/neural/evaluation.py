"""Frozen evaluation helpers over the accepted public combat view."""

from __future__ import annotations

import random
import struct
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.contracts import CombatFrame
from spire_learn.neural.schema import CombatSchema


REFERENCE_POLICIES = frozenset({"legal-random", "end-turn-last-v1"})


def _action_mask(frame: CombatFrame, schema: CombatSchema) -> tuple[int, ...]:
    buffer = frame.tensors.get("action_mask")
    if buffer is None or buffer.dtype != "int8" or buffer.shape != (schema.action_space.n,):
        raise ContractError("reference policy requires the descriptor action_mask")
    if frame.schema_fingerprint != schema.schema_fingerprint:
        raise ContractError("reference frame schema fingerprint does not match the descriptor")
    values = tuple(buffer.data)
    if len(values) != schema.action_space.n or any(value not in (0, 1) for value in values):
        raise ContractError("reference action_mask must contain one binary value per action")
    legal = tuple(index for index, value in enumerate(values) if value == 1)
    if not legal:
        raise ContractError("reference policy cannot choose from an empty legal set")
    return legal


def _candidate_verbs(frame: CombatFrame, schema: CombatSchema) -> tuple[str, ...]:
    spec = schema.tensors.get("actions")
    if spec is None or len(spec.shape) != 2 or spec.shape[0] != schema.action_space.n:
        raise ContractError("reference policy requires descriptor-defined action candidates")
    try:
        verb_index = next(index for index, column in enumerate(spec.columns) if column.field == "verb")
        verb_column = spec.columns[verb_index]
    except (StopIteration, IndexError) as error:
        raise ContractError("action candidates do not declare a verb field") from error
    if verb_column.vocabulary is None:
        raise ContractError("action candidate verb must use a public vocabulary")
    vocabulary = schema.vocabularies.get(verb_column.vocabulary)
    values = frame.tensors.get("actions")
    known = frame.tensors.get(spec.known_key) if spec.known_key is not None else None
    if (vocabulary is None or values is None or values.dtype != "int64"
            or values.shape != spec.shape or known is None or known.dtype != "int8"
            or known.shape != spec.shape):
        raise ContractError("public action candidate tensors do not match the descriptor")
    columns = spec.shape[1]
    result: list[str] = []
    for row in range(schema.action_space.n):
        offset = (row * columns + verb_index) * 8
        category = struct.unpack_from("=q", values.data, offset)[0]
        is_known = known.data[row * columns + verb_index]
        if is_known not in (0, 1):
            raise ContractError("action candidate known bits must be binary")
        if is_known:
            if not 0 <= category < len(vocabulary):
                raise ContractError("action candidate verb is outside its public vocabulary")
            result.append(vocabulary[category])
        else:
            result.append("")
    return tuple(result)


def choose_reference(
    frame: CombatFrame,
    schema: CombatSchema,
    *,
    kind: str,
    rng: random.Random,
) -> int:
    """Choose one legal original action index without using hidden state."""
    if not isinstance(frame, CombatFrame) or not isinstance(schema, CombatSchema):
        raise TypeError("choose_reference requires a public CombatFrame and CombatSchema")
    if kind not in REFERENCE_POLICIES:
        raise ValueError(f"unsupported reference policy: {kind!r}")
    legal = _action_mask(frame, schema)
    if not callable(getattr(rng, "choice", None)):
        raise TypeError("reference policy rng must expose choice")
    if kind == "legal-random":
        action = rng.choice(legal)
        if type(action) is not int or action not in legal:
            raise ContractError("reference RNG selected an action outside the legal mask")
        return action

    verbs = _candidate_verbs(frame, schema)
    if any(not verbs[index] for index in legal):
        raise ContractError("every legal candidate must expose its public verb")
    non_end_turn = tuple(index for index in legal if verbs[index] != "end_turn")
    return min(non_end_turn if non_end_turn else legal)


__all__ = ["REFERENCE_POLICIES", "choose_reference"]
