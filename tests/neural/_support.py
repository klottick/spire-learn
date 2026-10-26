"""Learner-authored combat descriptors and frames for gated tensor checks.

This module uses only the standard library at import time. Native schema access
and all tensor/framework imports are delayed until explicitly called by a gated
test.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterator, Mapping, Sequence


def _column(field: str, unit: str, low: int, high: int, *,
            vocabulary: str | None = None, reference: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "field": field, "unit": unit, "low": low, "high": high,
    }
    if vocabulary is not None:
        result["vocabulary"] = vocabulary
    if reference is not None:
        result["reference"] = reference
        result["unavailable"] = -1
    return result


def synthetic_descriptor() -> dict[str, object]:
    """A small public structural schema authored for these geometry tests."""
    vocabularies = {
        "card": ["padding", "strike", "defend"],
        "enemy": ["padding", "cultist", "jaw_worm"],
        "potion": ["padding", "fire_potion", "block_potion"],
        "power": ["padding", "strength", "vulnerable"],
        "verb": ["padding", "use_potion", "play_card", "end_turn", "choose"],
        "row_group": ["none", "potions", "combat_cards", "enemies", "player"],
        "choice": ["padding", "upgrade", "card_reward"],
    }
    tensors: list[dict[str, object]] = []

    def binary(key: str, shape: list[int]) -> None:
        tensors.append({
            "key": key, "dtype": "int8", "shape": shape,
            "low": 0, "high": 1, "space": "MultiBinary", "padding": 0,
        })

    def values(key: str, shape: list[int], columns: list[dict[str, object]]) -> None:
        tensors.append({
            "key": key,
            "dtype": "int64",
            "shape": shape,
            "low": [column["low"] for column in columns],
            "high": [column["high"] for column in columns],
            "space": "Box",
            "fields": [column["field"] for column in columns],
            "units": [column["unit"] for column in columns],
            "columns": columns,
            "padding": 0,
            "known": f"{key}_known",
            "reference_columns": [
                index for index, column in enumerate(columns) if "reference" in column
            ],
        })
        binary(f"{key}_known", shape)
        if len(shape) == 2:
            binary(f"{key}_occupied", [shape[0]])

    values("player", [1], [_column("hp", "hp", 0, 100)])
    values("combat_cards", [2, 2], [
        _column("card_kind", "category", 0, 2, vocabulary="card"),
        _column("zone_position", "visible_position", 0, 31),
    ])
    values("enemies", [2, 2], [
        _column("enemy_kind", "category", 0, 2, vocabulary="enemy"),
        _column("logical_slot", "visible_position", 0, 31),
    ])
    values("powers", [2, 4], [
        _column("power_kind", "category", 0, 2, vocabulary="power"),
        _column("amount", "count", 0, 99),
        _column("owner_group", "category", 0, 4, vocabulary="row_group"),
        _column("owner_row", "owner_row", -1, 1, reference="owner_group"),
    ])
    values("potions", [2, 1], [
        _column("potion_kind", "category", 0, 2, vocabulary="potion"),
    ])
    values("known_draw", [2, 2], [
        _column("card_kind", "category", 0, 2, vocabulary="card"),
        _column("offset", "visible_position", 0, 31),
    ])
    action_columns = [
        _column("verb", "category", 0, 4, vocabulary="verb"),
        _column("source_group", "category", 0, 4, vocabulary="row_group"),
        _column("source_row", "source_row", -1, 1, reference="source_group"),
        _column("target_group", "category", 0, 4, vocabulary="row_group"),
        _column("target_row", "target_row", -1, 1, reference="target_group"),
        _column("choice_kind", "category", 0, 2, vocabulary="choice"),
        _column("scalar0", "verb_specific", 0, 31),
        _column("scalar1", "verb_specific", 0, 31),
    ]
    values("actions", [10, len(action_columns)], action_columns)
    binary("action_mask", [10])
    return {
        "schema_id": "spire-gym-combat-policy-view-v1",
        "profile_id": "IroncladCombatPolicy",
        "array_layout": "owned_c_contiguous_native_endian_row_major",
        "action_space": {"type": "Discrete", "n": 10, "start": 0},
        "capacities": {
            "combat_cards": 2, "enemies": 2, "powers": 2,
            "potions": 2, "known_draw": 2, "actions": 10,
        },
        "tensors": tensors,
        "vocabularies": vocabularies,
    }


def _descriptor_fingerprint(descriptor: Mapping[str, object]) -> str:
    body = {key: value for key, value in descriptor.items() if key != "fingerprint"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"spire-gym/combat-policy-view/v1\0" + encoded).hexdigest()


@contextmanager
def synthetic_schema() -> Iterator[object]:
    """Parse a toy descriptor through the real authenticated parser in-test."""
    from spire_learn.neural import contracts as neural_contracts
    from spire_learn.neural import schema as neural_schema

    descriptor = synthetic_descriptor()
    fingerprint = _descriptor_fingerprint(descriptor)
    descriptor["fingerprint"] = fingerprint
    accepted = neural_schema.COMMON_VIEW_FINGERPRINT
    with _patch_attribute(neural_schema, "COMMON_VIEW_FINGERPRINT", fingerprint), \
            _patch_attribute(neural_contracts, "_COMMON_VIEW_FINGERPRINT", fingerprint):
        schema = neural_schema.parse_common_descriptor(descriptor)
        yield schema
    if neural_schema.COMMON_VIEW_FINGERPRINT != accepted:
        raise AssertionError("synthetic schema patch leaked beyond its context")


@contextmanager
def _patch_attribute(module: object, name: str, value: object):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


@dataclass(frozen=True)
class CandidateInput:
    index: int
    verb: str
    source_group: str | None = None
    source_row: int | None = None
    target_group: str | None = None
    target_row: int | None = None
    choice_kind: str | None = None
    scalar0: int | None = None
    scalar1: int | None = None


def _category_id(schema: object, vocabulary: str, value: str) -> int:
    values = schema.vocabularies[vocabulary]
    try:
        return values.index(value)
    except ValueError:
        raise AssertionError(f"test vocabulary {vocabulary!r} lacks {value!r}") from None


def _category_choice(schema: object, column: object, *, preferred: Sequence[str] = ()) -> int:
    vocabulary = schema.vocabularies[column.vocabulary]
    for candidate in preferred:
        if candidate in vocabulary:
            return vocabulary.index(candidate)
    for index, candidate in enumerate(vocabulary):
        if candidate not in {"padding", "none", "unknown"}:
            return index
    return 0


def _selector_groups(schema: object, spec: object, selector: object) -> tuple[str, ...]:
    if selector.vocabulary is None:
        return ()
    return tuple(
        value
        for value in schema.vocabularies[selector.vocabulary]
        if value in schema.tensors and schema.tensors[value].columns
    )


def _capacity(spec: object) -> int:
    return spec.shape[0] if len(spec.shape) == 2 else 1


def _cell_index(spec: object, row: int, field_index: int) -> int:
    columns = len(spec.columns)
    return row * columns + field_index if len(spec.shape) == 2 else field_index


def authored_frame(
    schema: object,
    *,
    candidates: Sequence[CandidateInput] | None = None,
    power_owners: tuple[int, int] = (0, 1),
    populate_relations: bool = True,
    identical_enemy_rows: bool = False,
) -> object:
    """Build a valid, learner-authored actor frame; no simulator calls occur."""
    from spire_learn.neural.contracts import BufferArray, CombatFrame

    tensors: dict[str, BufferArray] = {}
    mutable: dict[str, list[int]] = {}
    for key, spec in schema.tensors.items():
        size = math.prod(spec.shape)
        mutable[key] = [spec.padding] * size
        if spec.dtype == "int8":
            continue
        if len(spec.shape) == 1:
            for row in range(_capacity(spec)):
                for column_index, column in enumerate(spec.columns):
                    if not column.low <= spec.padding <= column.high:
                        cell = _cell_index(spec, row, column_index)
                        mutable[key][cell] = column.low
                        if spec.known_key is not None:
                            mutable[spec.known_key][cell] = 1
        else:
            for row in range(_capacity(spec)):
                for column_index, column in enumerate(spec.columns):
                    if column.reference is not None:
                        continue
                    if not column.low <= spec.padding <= column.high:
                        # Unoccupied rows retain the tensor group's exact padding.
                        continue
                    mutable[key][_cell_index(spec, row, column_index)] = spec.padding
        if spec.known_key is not None and key == "potions":
            mutable[spec.known_key] = [1] * size

    # Empty potion belt slots are known zeroes in the public view, unlike
    # unoccupied card/enemy padding rows whose known bits remain zero.
    potion_spec = schema.tensors.get("potions")
    if potion_spec is not None and potion_spec.known_key is not None:
        mutable[potion_spec.known_key] = [1] * math.prod(potion_spec.shape)

    def set_field(group: str, row: int, field: str, value: int, *, known: bool = True) -> None:
        spec = schema.tensors[group]
        index = next((i for i, column in enumerate(spec.columns) if column.field == field), None)
        if index is None:
            return
        cell = _cell_index(spec, row, index)
        mutable[group][cell] = value
        if spec.known_key is not None:
            mutable[spec.known_key][cell] = int(known)

    def set_occupied(group: str, row: int) -> None:
        spec = schema.tensors[group]
        if spec.occupied_key is not None:
            mutable[spec.occupied_key][row] = 1

    def set_binary(group: str, index: int, value: int) -> None:
        mutable[group][index] = value

    def row_is_occupied(group: str, row: int) -> bool:
        spec = schema.tensors[group]
        if not 0 <= row < _capacity(spec):
            return False
        return spec.occupied_key is None or bool(mutable[spec.occupied_key][row])

    def selector_target(group: str, selector: object) -> str | None:
        options = _selector_groups(schema, schema.tensors[group], selector)
        if "enemies" in options:
            return "enemies"
        return next((name for name in options if row_is_occupied(name, 0)), None)

    def set_reference(group: str, row: int, column: object, *, preferred_row: int) -> None:
        spec = schema.tensors[group]
        selector = next((item for item in spec.columns if item.field == column.reference), None)
        if selector is not None:
            target = selector_target(group, selector)
            if target is None:
                if "none" in schema.vocabularies[selector.vocabulary]:
                    set_field(group, row, selector.field,
                              schema.vocabularies[selector.vocabulary].index("none"), known=True)
                set_field(group, row, column.field, -1, known=False)
                return
            vocabulary = schema.vocabularies[selector.vocabulary]
            set_field(group, row, selector.field, vocabulary.index(target), known=True)
        else:
            target = column.reference
        target_spec = schema.tensors.get(target)
        chosen_row = preferred_row
        if (target_spec is None or not target_spec.columns
                or not row_is_occupied(target, chosen_row)):
            set_field(group, row, column.field, -1, known=False)
            return
        set_field(group, row, column.field, chosen_row, known=True)

    # Singleton public vectors describe the current player/context row even
    # when every optional row group in the authored frame is empty.
    for group, spec in schema.tensors.items():
        if not spec.columns or len(spec.shape) != 1:
            continue
        for column in spec.columns:
            if column.reference is not None:
                set_reference(group, 0, column, preferred_row=0)
            elif column.vocabulary is not None:
                set_field(group, 0, column.field, _category_choice(schema, column), known=True)
            else:
                value = 0 if column.low <= 0 <= column.high else column.low
                set_field(group, 0, column.field, value, known=True)

    # Two indistinguishable enemies and one public source card make the typed
    # power-owner and source/target permutation checks possible.
    if populate_relations:
        for group, row_count in (("combat_cards", 1), ("enemies", 2)):
            spec = schema.tensors.get(group)
            if spec is None or spec.occupied_key is None:
                continue
            for row in range(min(row_count, _capacity(spec))):
                set_occupied(group, row)
                for column in spec.columns:
                    if column.reference is not None:
                        set_reference(group, row, column, preferred_row=0)
                        continue
                    if column.vocabulary is not None:
                        if group == "combat_cards":
                            preferred = ("strike", "defend")
                        elif group == "enemies" and identical_enemy_rows:
                            preferred = ("cultist", "jaw_worm")
                        elif row == 0:
                            preferred = ("cultist", "jaw_worm")
                        else:
                            preferred = ("jaw_worm", "cultist")
                        value = _category_choice(schema, column, preferred=preferred)
                    else:
                        if (group == "enemies" and column.field == "logical_slot"
                                and identical_enemy_rows):
                            value = 0 if column.low <= 0 <= column.high else column.low
                        elif (group == "enemies" and column.field == "logical_slot"
                                and column.low <= row <= column.high):
                            value = row
                        else:
                            value = 0 if column.low <= 0 <= column.high else column.low
                    set_field(group, row, column.field, value, known=True)

        power_spec = schema.tensors.get("powers")
        if power_spec is not None and power_spec.occupied_key is not None:
            for row in range(min(2, _capacity(power_spec))):
                set_occupied("powers", row)
                for column in power_spec.columns:
                    if column.reference is not None:
                        set_reference(
                            "powers", row, column,
                            preferred_row=power_owners[row],
                        )
                        continue
                    if column.vocabulary is not None:
                        selector_used = any(item.reference == column.field for item in power_spec.columns)
                        if selector_used:
                            options = _selector_groups(schema, power_spec, column)
                            selected = "enemies" if "enemies" in options else None
                            if selected is None:
                                candidates = tuple(
                                    item for item in options if row_is_occupied(item, 0)
                                )
                                selected = candidates[0] if candidates else None
                            value = (schema.vocabularies[column.vocabulary].index(selected)
                                     if selected is not None else _category_choice(
                                         schema, column, preferred=("none",)
                                     ))
                        else:
                            preferred = (
                                ("strength", "vulnerable") if column.field == "power_kind"
                                else ("enemies",) if column.field in {"owner_group", "owner_kind"}
                                else ()
                            )
                            value = _category_choice(schema, column, preferred=preferred)
                        set_field("powers", row, column.field, value, known=True)
                    else:
                        value = (2 if row == 0 else 3) if column.field == "amount" else (
                            0 if column.low <= 0 <= column.high else column.low
                        )
                        set_field("powers", row, column.field, value, known=True)

    action_spec = schema.tensors["actions"]
    action_vocab_fields = {
        column.field: column.vocabulary for column in action_spec.columns
        if column.vocabulary is not None
    }
    inputs = tuple(candidates) if candidates is not None else (
        CandidateInput(
            index=schema.action_space.n - 1,
            verb="end_turn",
        ),
    )
    for candidate in inputs:
        if not 0 <= candidate.index < schema.action_space.n:
            raise AssertionError("test action index is outside the descriptor action space")
        set_occupied("actions", candidate.index)
        set_binary("action_mask", candidate.index, 1)
        for column_index, column in enumerate(action_spec.columns):
            field = column.field
            value = 0
            known = False
            if field == "verb":
                vocabulary = action_vocab_fields[field]
                value = _category_id(schema, vocabulary, candidate.verb)
                known = True
            elif field in {"source_group", "target_group"}:
                group = candidate.source_group if field == "source_group" else candidate.target_group
                vocabulary = action_vocab_fields[field]
                category = "none" if group is None else group
                value = _category_id(schema, vocabulary, category)
                known = True
            elif column.reference is not None:
                row = candidate.source_row if field == "source_row" else candidate.target_row
                value = -1 if row is None else row
                known = row is not None
            elif field == "choice_kind":
                if candidate.choice_kind is not None:
                    value = _category_id(schema, action_vocab_fields[field], candidate.choice_kind)
                    known = True
            elif field == "scalar0" and candidate.scalar0 is not None:
                value, known = candidate.scalar0, True
            elif field == "scalar1" and candidate.scalar1 is not None:
                value, known = candidate.scalar1, True
            cell = _cell_index(action_spec, candidate.index, column_index)
            mutable["actions"][cell] = value
            mutable["actions_known"][cell] = int(known)

    for key, spec in schema.tensors.items():
        values = mutable[key]
        if spec.dtype == "int8":
            data = bytes(values)
        else:
            data = struct.pack("=" + "q" * len(values), *values)
        tensors[key] = BufferArray(spec.dtype, spec.shape, data)
    return CombatFrame(tensors, schema.schema_fingerprint)


def accepted_schema_and_config() -> tuple[object, object]:
    """Construct/close the accepted public facade without reset or step."""
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.envs import make_driver

    root = Path(__file__).resolve().parents[2]
    config = load_neural_config(root / "configs/neural/run.toml")
    return schema_from_driver_factory(config, make_driver), config


def schema_from_driver_factory(config: object, driver_factory) -> object:
    """Construct and close a driver using the parsed config's accepted cases."""
    driver = driver_factory(config, config.cases)
    try:
        return driver.schema
    finally:
        driver.close()


def frame_bytes(frame: object) -> int:
    return sum(len(buffer.data) for buffer in frame.tensors.values())


def tensor_leaves(value: object, torch_module: object) -> list[object]:
    """Collect tensors from an EncodedBatch without importing Torch here."""
    from collections.abc import Mapping as MappingABC
    from dataclasses import fields, is_dataclass

    if torch_module.is_tensor(value):
        return [value]
    if isinstance(value, MappingABC):
        return [leaf for item in value.values() for leaf in tensor_leaves(item, torch_module)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in tensor_leaves(item, torch_module)]
    if is_dataclass(value):
        return [
            leaf for field in fields(value)
            for leaf in tensor_leaves(getattr(value, field.name), torch_module)
        ]
    return []
