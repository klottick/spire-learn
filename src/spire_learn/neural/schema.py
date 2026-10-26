"""Strict, dependency-free parsing and validation for the accepted common view."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass
import hashlib
import json
import re
import struct
from types import MappingProxyType

from spire_learn.contracts import ContractError
from spire_learn.neural.config import (
    ACCEPTED_GYM_REVISION,
    COMMON_VIEW_FINGERPRINT,
    COMMON_VIEW_ID,
)
from spire_learn.neural.contracts import BufferArray, CombatFrame, Lifecycle


_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_COMBAT_VIEW_ID = "spire-gym-combat-policy-view-v1"
_PROFILE_ID = "IroncladCombatPolicy"
_ARRAY_LAYOUT = "owned_c_contiguous_native_endian_row_major"
_FINGERPRINT_PREFIX = b"spire-gym/combat-policy-view/v1\0"
_PARSER_AUTHORITY = object()
_CHOICE_VERBS = frozenset({"choose"})
_POTION_VERBS = frozenset({"use_potion", "discard_potion"})


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < -(2**63) or value > 2**64 - 1:
        raise ContractError(f"{name} must be an integer JSON value")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ColumnSpec:
    """One descriptor-defined value column and its transport interpretation."""

    field: str
    unit: str
    low: int
    high: int
    vocabulary: str | None = None
    reference: str | None = None
    unavailable: int | None = None

    def __post_init__(self) -> None:
        _string(self.field, "column field")
        _string(self.unit, "column unit")
        _integer(self.low, "column low")
        _integer(self.high, "column high")
        if self.low > self.high:
            raise ContractError("column low must not exceed high")
        if self.vocabulary is not None:
            _string(self.vocabulary, "column vocabulary")
        if self.reference is not None:
            _string(self.reference, "column reference")
            if self.unavailable != -1 or self.low != -1:
                raise ContractError("row references must declare typed -1 unknown values")
        elif self.unavailable is not None:
            raise ContractError("unavailable is valid only for a reference column")
        if self.vocabulary is not None and self.reference is not None:
            raise ContractError("a column cannot be both categorical and a row reference")


@dataclass(frozen=True)
class TensorSpec:
    """Accepted shape, dtype, companion masks and value columns for a tensor."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    padding: int = 0
    columns: tuple[ColumnSpec, ...] = ()
    known_key: str | None = None
    occupied_key: str | None = None
    reference_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _string(self.key, "tensor key")
        if self.dtype not in {"int8", "int64"}:
            raise ContractError("common tensors must use int8 or int64")
        if not isinstance(self.shape, tuple) or not self.shape or any(
            type(dim) is not int or dim < 1 for dim in self.shape
        ):
            raise ContractError("tensor shape must contain positive integer dimensions")
        _integer(self.padding, "tensor padding")
        if not isinstance(self.columns, tuple) or any(
            not isinstance(column, ColumnSpec) for column in self.columns
        ):
            raise ContractError("tensor columns must be ColumnSpec records")
        if self.columns:
            if self.dtype != "int64" or len(self.shape) not in {1, 2}:
                raise ContractError("value tensors must be int64 vectors or row matrices")
            if len(self.columns) != self.shape[-1]:
                raise ContractError("value tensor columns must match its final dimension")
            if len({column.field for column in self.columns}) != len(self.columns):
                raise ContractError("value tensor column names must be unique")
            if self.known_key is None:
                raise ContractError("value tensors must name their declared known companion")
            if len(self.shape) == 2 and self.occupied_key is None:
                raise ContractError("row tensors must name their occupied companion")
            if len(self.shape) == 1 and self.occupied_key is not None:
                raise ContractError("singleton vectors do not have occupied companions")
        elif self.known_key is not None or self.occupied_key is not None:
            raise ContractError("mask tensors cannot declare companion masks")
        if self.known_key is not None:
            _string(self.known_key, "known tensor key")
        if self.occupied_key is not None:
            _string(self.occupied_key, "occupied tensor key")
        if not isinstance(self.reference_columns, tuple) or any(
            type(index) is not int or index < 0 for index in self.reference_columns
        ):
            raise ContractError("reference_columns must be non-negative integer indices")
        if len(set(self.reference_columns)) != len(self.reference_columns):
            raise ContractError("reference_columns must not contain duplicates")
        actual = tuple(i for i, column in enumerate(self.columns) if column.reference is not None)
        if self.reference_columns != actual:
            raise ContractError("reference_columns must exactly identify reference columns")


@dataclass(frozen=True)
class ActionSpace:
    """The descriptor's finite discrete dispatch space."""

    kind: str
    n: int
    start: int

    def __post_init__(self) -> None:
        if self.kind != "Discrete":
            raise ContractError("common action space must be Discrete")
        if type(self.n) is not int or self.n < 1:
            raise ContractError("action-space n must be positive")
        if type(self.start) is not int or self.start != 0:
            raise ContractError("common dispatch action-space start must be zero")


@dataclass(frozen=True)
class CombatSchema:
    """Frozen identity and tensor contract obtained from the accepted descriptor."""

    gym_revision: str
    schema_id: str
    profile_id: str
    schema_fingerprint: str
    descriptor_digest: str
    array_layout: str
    action_space: ActionSpace
    capacities: Mapping[str, int]
    vocabularies: Mapping[str, tuple[str, ...]]
    tensors: Mapping[str, TensorSpec]
    _parser_authority: InitVar[object]

    def __post_init__(self, _parser_authority: object) -> None:
        if _parser_authority is not _PARSER_AUTHORITY:
            raise ContractError("CombatSchema instances must come from the authenticated parser")
        if self.gym_revision != ACCEPTED_GYM_REVISION:
            raise ContractError("CombatSchema Gym revision is not accepted")
        if self.schema_id != COMMON_VIEW_ID:
            raise ContractError("CombatSchema common-view ID is not accepted")
        if self.profile_id != _PROFILE_ID:
            raise ContractError("CombatSchema profile ID is not accepted")
        if self.schema_fingerprint != COMMON_VIEW_FINGERPRINT:
            raise ContractError("CombatSchema common-view fingerprint is not accepted")
        if not isinstance(self.descriptor_digest, str) or _FINGERPRINT.fullmatch(
            self.descriptor_digest
        ) is None:
            raise ContractError("CombatSchema descriptor digest must be SHA-256 hex")
        if self.array_layout != _ARRAY_LAYOUT:
            raise ContractError("CombatSchema array layout is not accepted")
        if not isinstance(self.action_space, ActionSpace):
            raise ContractError("CombatSchema requires an ActionSpace")

        if not isinstance(self.capacities, Mapping):
            raise ContractError("capacities must be a mapping")
        capacities = dict(self.capacities)
        if any(not isinstance(key, str) or not key or type(value) is not int or value < 1
               for key, value in capacities.items()):
            raise ContractError("capacities must map names to positive integers")
        if capacities.get("actions") != self.action_space.n:
            raise ContractError("action-space size must equal the declared actions capacity")
        object.__setattr__(self, "capacities", MappingProxyType(capacities))

        if not isinstance(self.vocabularies, Mapping):
            raise ContractError("vocabularies must be a mapping")
        vocabularies: dict[str, tuple[str, ...]] = {}
        for key, values in self.vocabularies.items():
            if not isinstance(key, str) or not key or not isinstance(values, tuple):
                raise ContractError("vocabularies must map names to string tuples")
            if not values or any(not isinstance(value, str) for value in values):
                raise ContractError("vocabulary values must be non-empty string sequences")
            if len(set(values)) != len(values):
                raise ContractError("vocabulary values must be unique")
            vocabularies[key] = tuple(values)
        object.__setattr__(self, "vocabularies", MappingProxyType(vocabularies))

        if not isinstance(self.tensors, Mapping):
            raise ContractError("tensors must be a mapping")
        tensors = dict(self.tensors)
        if not tensors or any(key != spec.key or not isinstance(spec, TensorSpec)
                              for key, spec in tensors.items()):
            raise ContractError("tensors must map their exact keys to TensorSpec records")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))


@dataclass(frozen=True)
class CombatReference:
    """Descriptor-group row reference; unknown applicable rows keep row=None."""

    group: str
    row: int | None
    known: bool

    def __post_init__(self) -> None:
        _string(self.group, "reference group")
        if self.row is not None and (type(self.row) is not int or self.row < 0):
            raise ContractError("reference row must be non-negative or None")
        if type(self.known) is not bool or self.known != (self.row is not None):
            raise ContractError("reference known bit must agree with its typed row")


@dataclass(frozen=True)
class CombatCandidate:
    """One legal dispatch row with original index and typed public references."""

    index: int
    verb: str
    source: CombatReference | None
    target: CombatReference | None
    choice: str | None
    choice_known: bool
    choice_applicable: bool
    scalar0: int | None
    scalar0_known: bool
    scalar0_applicable: bool
    scalar1: int | None
    scalar1_known: bool
    scalar1_applicable: bool

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ContractError("candidate index must be a non-negative dispatch row")
        _string(self.verb, "candidate verb")
        if self.source is not None and not isinstance(self.source, CombatReference):
            raise ContractError("candidate source must be a CombatReference or None")
        if self.target is not None and not isinstance(self.target, CombatReference):
            raise ContractError("candidate target must be a CombatReference or None")
        if self.choice is not None and not isinstance(self.choice, str):
            raise ContractError("candidate choice must be a string or None")
        for name in ("choice_known", "choice_applicable", "scalar0_known",
                     "scalar0_applicable", "scalar1_known", "scalar1_applicable"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"candidate {name} must be a boolean")
        if self.choice_applicable != (self.verb in _CHOICE_VERBS):
            raise ContractError("candidate choice applicability must follow its decoded verb")
        if self.scalar0_applicable != (self.verb in _POTION_VERBS):
            raise ContractError("candidate scalar0 applicability must follow its decoded verb")
        if self.scalar1_applicable:
            raise ContractError("candidate scalar1 is reserved and inapplicable")
        if self.choice_known and (not self.choice_applicable or self.choice is None):
            raise ContractError("known choice values must be present and applicable")
        if not self.choice_known and self.choice is not None:
            raise ContractError("unknown choice values must be None")
        for name in ("scalar0", "scalar1"):
            value = getattr(self, name)
            known = getattr(self, f"{name}_known")
            applicable = getattr(self, f"{name}_applicable")
            if value is not None and type(value) is not int:
                raise ContractError(f"candidate {name} must be an integer or None")
            if known and (not applicable or value is None):
                raise ContractError(f"known {name} values must be present and applicable")
            if not known and value is not None:
                raise ContractError(f"unknown {name} values must be None")

    @property
    def source_applicable(self) -> bool:
        return self.source is not None

    @property
    def source_known(self) -> bool:
        return self.source is not None and self.source.known

    @property
    def target_applicable(self) -> bool:
        return self.target is not None

    @property
    def target_known(self) -> bool:
        return self.target is not None and self.target.known


def _canonical_value(value: object, path: str = "descriptor") -> object:
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        return _integer(value, path)
    if type(value) is float:
        raise ContractError(f"{path} must not contain floating-point numbers")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} mapping keys must be strings")
            result[key] = _canonical_value(item, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [_canonical_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ContractError(f"{path} must contain JSON-compatible values")


def _canonical_json(value: object) -> bytes:
    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return value


def _required_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a list")
    return value


def _descriptor_tensor(raw: Mapping[str, object], name: str) -> tuple[str, str, tuple[int, ...]]:
    key = _string(raw.get("key"), f"{name}.key")
    dtype = _string(raw.get("dtype"), f"{name}.dtype")
    shape_raw = _required_list(raw.get("shape"), f"{name}.shape")
    shape = tuple(_integer(dim, f"{name}.shape dimension") for dim in shape_raw)
    if dtype not in {"int8", "int64"} or not shape or any(dim < 1 for dim in shape):
        raise ContractError(f"{name} has unsupported dtype or shape")
    return key, dtype, shape


def _parse_descriptor_columns(
    raw: Mapping[str, object], key: str, shape: tuple[int, ...],
    vocabularies: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[ColumnSpec, ...], tuple[int, ...]]:
    fields = _required_list(raw.get("fields"), f"{key}.fields")
    units = _required_list(raw.get("units"), f"{key}.units")
    raw_columns = _required_list(raw.get("columns"), f"{key}.columns")
    if not (len(fields) == len(units) == len(raw_columns) == shape[-1]):
        raise ContractError(f"{key} fields, units, columns and final shape dimension disagree")
    low_values = _required_list(raw.get("low"), f"{key}.low")
    high_values = _required_list(raw.get("high"), f"{key}.high")
    if len(low_values) != len(fields) or len(high_values) != len(fields):
        raise ContractError(f"{key} low/high domains must align with columns")

    columns: list[ColumnSpec] = []
    for index, raw_column_value in enumerate(raw_columns):
        raw_column = _required_mapping(raw_column_value, f"{key}.columns[{index}]")
        field = _string(raw_column.get("field"), f"{key}.columns[{index}].field")
        unit = _string(raw_column.get("unit"), f"{key}.columns[{index}].unit")
        if fields[index] != field or units[index] != unit:
            raise ContractError(f"{key} field/unit arrays disagree with column metadata")
        low = _integer(raw_column.get("low"), f"{key}.{field}.low")
        high = _integer(raw_column.get("high"), f"{key}.{field}.high")
        if low_values[index] != low or high_values[index] != high:
            raise ContractError(f"{key}.{field} root and column domains disagree")

        vocabulary = raw_column.get("vocabulary")
        reference = raw_column.get("reference")
        unavailable = raw_column.get("unavailable")
        if vocabulary is not None:
            vocabulary = _string(vocabulary, f"{key}.{field}.vocabulary")
            values = vocabularies.get(vocabulary)
            if values is None or low != 0 or high != len(values) - 1:
                raise ContractError(f"{key}.{field} category does not match its vocabulary")
        if reference is not None:
            reference = _string(reference, f"{key}.{field}.reference")
            unavailable = _integer(unavailable, f"{key}.{field}.unavailable")
            if low != -1 or unavailable != -1:
                raise ContractError(f"{key}.{field} reference must use typed -1 unknown")
        elif unavailable is not None:
            raise ContractError(f"{key}.{field} has unavailable without a reference")
        columns.append(ColumnSpec(field, unit, low, high, vocabulary, reference, unavailable))

    if len({column.field for column in columns}) != len(columns):
        raise ContractError(f"{key} column field names must be unique")
    reference_columns_raw = _required_list(raw.get("reference_columns"), f"{key}.reference_columns")
    reference_columns = tuple(
        _integer(index, f"{key}.reference_columns index") for index in reference_columns_raw
    )
    expected = tuple(i for i, column in enumerate(columns) if column.reference is not None)
    if reference_columns != expected:
        raise ContractError(f"{key}.reference_columns disagrees with column metadata")
    return tuple(columns), reference_columns


def parse_common_descriptor(raw: Mapping[str, object]) -> CombatSchema:
    """Parse and authenticate the accepted Gym common combat descriptor."""
    descriptor = _required_mapping(raw, "descriptor")
    normalized = _canonical_value(descriptor)
    if not isinstance(normalized, dict):
        raise ContractError("descriptor root must be an object")
    supplied = normalized.get("fingerprint")
    if not isinstance(supplied, str) or _FINGERPRINT.fullmatch(supplied) is None:
        raise ContractError("descriptor fingerprint must be lowercase SHA-256 hex")
    body = dict(normalized)
    del body["fingerprint"]
    encoded = _canonical_json(body)
    native_fingerprint = hashlib.sha256(_FINGERPRINT_PREFIX + encoded).hexdigest()
    if supplied != native_fingerprint or supplied != COMMON_VIEW_FINGERPRINT:
        raise ContractError("descriptor body does not match the supplied and accepted fingerprint")

    if body.get("schema_id") != COMMON_VIEW_ID:
        raise ContractError("descriptor common-view ID is not accepted")
    if body.get("profile_id") != _PROFILE_ID:
        raise ContractError("descriptor profile ID is not accepted")
    if body.get("array_layout") != _ARRAY_LAYOUT:
        raise ContractError("descriptor array layout is not accepted")

    capacities_raw = _required_mapping(body.get("capacities"), "descriptor.capacities")
    capacities: dict[str, int] = {}
    for key, value in capacities_raw.items():
        capacities[key] = _integer(value, f"descriptor.capacities.{key}")
        if capacities[key] < 1:
            raise ContractError(f"descriptor capacity {key} must be positive")
    if "actions" not in capacities:
        raise ContractError("descriptor capacities must include actions")

    action_raw = _required_mapping(body.get("action_space"), "descriptor.action_space")
    action_space = ActionSpace(
        _string(action_raw.get("type"), "descriptor.action_space.type"),
        _integer(action_raw.get("n"), "descriptor.action_space.n"),
        _integer(action_raw.get("start"), "descriptor.action_space.start"),
    )
    if action_space.n != capacities["actions"]:
        raise ContractError("descriptor action space and actions capacity disagree")

    vocabularies_raw = _required_mapping(body.get("vocabularies"), "descriptor.vocabularies")
    vocabularies: dict[str, tuple[str, ...]] = {}
    for key, raw_values in vocabularies_raw.items():
        values = _required_list(raw_values, f"descriptor.vocabularies.{key}")
        vocabularies[key] = tuple(
            _string(value, f"descriptor.vocabularies.{key} entry") for value in values
        )
        if not vocabularies[key] or len(set(vocabularies[key])) != len(vocabularies[key]):
            raise ContractError(f"descriptor vocabulary {key} must be non-empty and unique")

    tensor_entries = _required_list(body.get("tensors"), "descriptor.tensors")
    raw_tensors: dict[str, Mapping[str, object]] = {}
    for index, entry in enumerate(tensor_entries):
        item = _required_mapping(entry, f"descriptor.tensors[{index}]")
        key, _, _ = _descriptor_tensor(item, f"descriptor.tensors[{index}]")
        if key in raw_tensors:
            raise ContractError(f"descriptor tensor key {key} is duplicated")
        raw_tensors[key] = item
    if not raw_tensors:
        raise ContractError("descriptor must declare tensors")

    tensors: dict[str, TensorSpec] = {}
    for key, item in raw_tensors.items():
        _, dtype, shape = _descriptor_tensor(item, f"descriptor.tensors.{key}")
        padding = _integer(item.get("padding"), f"descriptor.tensors.{key}.padding")
        if "columns" in item:
            if dtype != "int64":
                raise ContractError(f"value tensor {key} must use int64")
            if len(shape) == 2 and key in capacities and capacities[key] != shape[0]:
                raise ContractError(f"row tensor {key} shape must match its named capacity")
            columns, references = _parse_descriptor_columns(
                item, key, shape, vocabularies
            )
            known_key = _string(item.get("known"), f"{key}.known")
            occupied_key = f"{key}_occupied" if len(shape) == 2 else None
            tensors[key] = TensorSpec(
                key, dtype, shape, padding, columns, known_key, occupied_key, references
            )
        else:
            tensors[key] = TensorSpec(key, dtype, shape, padding)

    # The public guide declares row occupancy as a separate companion tensor.
    for key, spec in tuple(tensors.items()):
        if not spec.columns:
            continue
        known = tensors.get(spec.known_key or "")
        if known is None or known.dtype != "int8" or known.shape != spec.shape:
            raise ContractError(f"{key} known companion is missing or has the wrong shape/dtype")
        if spec.occupied_key is not None:
            occupied = tensors.get(spec.occupied_key)
            if occupied is None or occupied.dtype != "int8" or occupied.shape != (spec.shape[0],):
                raise ContractError(f"{key} occupied companion is missing or has the wrong shape/dtype")
        for column in spec.columns:
            if column.vocabulary is None and column.reference is None:
                continue
            if column.reference is None:
                continue
            selector = next((item for item in spec.columns if item.field == column.reference), None)
            if selector is not None:
                if selector.vocabulary is None:
                    raise ContractError(f"{key}.{column.field} selector is not categorical")
                continue
            if column.reference not in tensors:
                raise ContractError(f"{key}.{column.field} names an unavailable fixed group")
            group = tensors[column.reference]
            if not group.columns:
                raise ContractError(f"{key}.{column.field} fixed reference lacks value columns")

    if "action_mask" not in tensors:
        raise ContractError("descriptor must declare action_mask")
    mask = tensors["action_mask"]
    if mask.dtype != "int8" or mask.shape != (action_space.n,) or mask.columns:
        raise ContractError("action_mask must be int8 with the exact discrete action shape")

    return CombatSchema(
        gym_revision=ACCEPTED_GYM_REVISION,
        schema_id=COMMON_VIEW_ID,
        profile_id=_PROFILE_ID,
        schema_fingerprint=supplied,
        descriptor_digest=hashlib.sha256(encoded).hexdigest(),
        array_layout=_ARRAY_LAYOUT,
        action_space=action_space,
        capacities=capacities,
        vocabularies=vocabularies,
        tensors=tensors,
        _parser_authority=_PARSER_AUTHORITY,
    )


def _unpack(buffer: BufferArray) -> tuple[int, ...]:
    if buffer.dtype == "int8":
        return tuple(value[0] for value in struct.iter_unpack("=b", buffer.data))
    return tuple(value[0] for value in struct.iter_unpack("=q", buffer.data))


def _tensor_values(frame: CombatFrame, key: str) -> tuple[int, ...]:
    return _unpack(frame.tensors[key])


def _row_capacity(spec: TensorSpec) -> int:
    return spec.shape[0] if len(spec.shape) == 2 else 1


def _selector_group(
    schema: CombatSchema, spec: TensorSpec, column: ColumnSpec, row: int | None,
    values: tuple[int, ...], known: tuple[int, ...], columns: int,
) -> str | None:
    """Resolve a literal group or descriptor-named selector column."""
    if column.reference is None:
        raise ContractError("column is not a reference")
    selector_index = next(
        (index for index, candidate in enumerate(spec.columns)
         if candidate.field == column.reference),
        None,
    )
    if selector_index is None:
        return column.reference
    cell = selector_index if row is None else row * columns + selector_index
    if not known[cell]:
        return None
    selector = spec.columns[selector_index]
    if selector.vocabulary is None:
        raise ContractError("reference selector must be a vocabulary column")
    vocabulary = schema.vocabularies[selector.vocabulary]
    selected = vocabulary[values[cell]]
    return None if selected == "none" else selected


def _resolve_reference(
    schema: CombatSchema, group: str, row: int,
    occupancy: Mapping[str, tuple[int, ...]],
) -> None:
    target = schema.tensors.get(group)
    if target is None or not target.columns:
        raise ContractError(f"known reference names unavailable group {group}")
    capacity = _row_capacity(target)
    if row >= capacity:
        raise ContractError(f"known reference row is outside {group} capacity")
    if target.occupied_key is not None and not occupancy[target.occupied_key][row]:
        raise ContractError(f"known reference points to an unoccupied {group} row")


def validate_frame(frame: CombatFrame, schema: CombatSchema) -> None:
    """Validate every actor tensor against its exact accepted descriptor contract."""
    if not isinstance(schema, CombatSchema):
        raise ContractError("schema must be a CombatSchema")
    if not isinstance(frame, CombatFrame):
        raise ContractError("frame must be a CombatFrame")
    if frame.schema_fingerprint != schema.schema_fingerprint:
        raise ContractError("frame schema fingerprint does not match CombatSchema")
    if set(frame.tensors) != set(schema.tensors):
        missing = set(schema.tensors) - set(frame.tensors)
        unexpected = set(frame.tensors) - set(schema.tensors)
        raise ContractError(f"frame tensor keys mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    values_by_key: dict[str, tuple[int, ...]] = {}
    for key, spec in schema.tensors.items():
        array = frame.tensors[key]
        if not isinstance(array, BufferArray):
            raise ContractError(f"frame tensor {key} must be a BufferArray")
        if array.dtype != spec.dtype or array.shape != spec.shape:
            raise ContractError(f"frame tensor {key} dtype/shape does not match descriptor")
        values = _unpack(array)
        if spec.dtype == "int8" and any(value not in {0, 1} for value in values):
            raise ContractError(f"frame tensor {key} must contain only binary values")
        values_by_key[key] = values

    occupancy = {
        spec.occupied_key: values_by_key[spec.occupied_key]
        for spec in schema.tensors.values() if spec.occupied_key is not None
    }

    action_spec = schema.tensors.get("actions")
    action_mask_spec = schema.tensors.get("action_mask")
    if action_spec is not None and action_spec.occupied_key is not None and action_mask_spec is not None:
        legal = values_by_key["action_mask"]
        action_rows = occupancy[action_spec.occupied_key]
        if len(legal) != len(action_rows) or any(
            bool(is_legal) != bool(is_occupied)
            for is_legal, is_occupied in zip(legal, action_rows)
        ):
            raise ContractError("action_mask must identify exactly the occupied dispatch rows")

    for key, spec in schema.tensors.items():
        if not spec.columns:
            continue
        assert spec.known_key is not None
        values = values_by_key[key]
        known = values_by_key[spec.known_key]
        columns = len(spec.columns)
        rows = _row_capacity(spec)
        occupied = occupancy.get(spec.occupied_key) if spec.occupied_key is not None else None
        for row_index in range(rows):
            row = row_index if len(spec.shape) == 2 else None
            row_occupied = True if occupied is None else bool(occupied[row_index])
            for column_index, column in enumerate(spec.columns):
                flat_index = row_index * columns + column_index
                value = values[flat_index]
                is_known = bool(known[flat_index])
                if not row_occupied:
                    if key == "potions":
                        if value != 0 or not is_known:
                            raise ContractError("empty potion slots must be known zeroes with occupied=0")
                    elif value != spec.padding or is_known:
                        raise ContractError(f"unoccupied {key} rows must use zero/known0 padding")
                    continue

                if value < column.low or value > column.high:
                    raise ContractError(f"{key}.{column.field} is outside its registered domain")
                if not is_known:
                    expected = column.unavailable if column.reference is not None else spec.padding
                    if value != expected:
                        raise ContractError(f"unknown {key}.{column.field} has the wrong typed padding")
                    continue
                if column.vocabulary is not None:
                    vocabulary = schema.vocabularies.get(column.vocabulary)
                    if vocabulary is None or value >= len(vocabulary):
                        raise ContractError(f"{key}.{column.field} is outside its declared vocabulary")
                if column.reference is not None:
                    group = _selector_group(schema, spec, column, row, values, known, columns)
                    if group is None:
                        raise ContractError(f"known {key}.{column.field} lacks a known reference group")
                    if value < 0:
                        raise ContractError(f"known {key}.{column.field} cannot use the -1 unknown row")
                    _resolve_reference(schema, group, value, occupancy)


def _field_index(spec: TensorSpec, field: str, *, required: bool = False) -> int | None:
    found = next((index for index, column in enumerate(spec.columns)
                  if column.field == field), None)
    if found is None and required:
        raise ContractError(f"actions tensor is missing required {field} column")
    return found


def _candidate_reference(
    schema: CombatSchema, spec: TensorSpec, row: int, values: tuple[int, ...],
    known: tuple[int, ...], column_index: int | None,
) -> CombatReference | None:
    if column_index is None:
        return None
    column = spec.columns[column_index]
    if column.reference is None:
        raise ContractError(f"actions.{column.field} is not a declared row reference")
    columns = len(spec.columns)
    cell = row * columns + column_index
    group = _selector_group(schema, spec, column, row, values, known, columns)
    if group is None:
        return None
    if group == "none":
        return None
    if known[cell]:
        return CombatReference(group, values[cell], True)
    return CombatReference(group, None, False)


def candidates(frame: CombatFrame, schema: CombatSchema) -> tuple[CombatCandidate, ...]:
    """Decode legal rows without changing their original native dispatch indices."""
    validate_frame(frame, schema)
    mask = _tensor_values(frame, "action_mask")
    indexes = tuple(index for index, legal in enumerate(mask) if legal == 1)
    if not indexes:
        raise ContractError("empty legal-action mask cannot be sent to an actor")
    spec = schema.tensors.get("actions")
    if spec is None or not spec.columns or len(spec.shape) != 2:
        raise ContractError("accepted descriptor lacks a row-based actions tensor")
    actions = _tensor_values(frame, "actions")
    known = _tensor_values(frame, spec.known_key or "")
    columns = len(spec.columns)
    indexes_by_name = {
        name: _field_index(spec, name, required=True)
        for name in ("verb", "source_group", "source_row", "target_group",
                     "target_row", "choice_kind", "scalar0", "scalar1")
    }

    result: list[CombatCandidate] = []
    for index in indexes:
        base = index * columns

        verb_index = indexes_by_name["verb"]
        assert verb_index is not None
        if not known[base + verb_index]:
            raise ContractError("legal action row has an unknown verb")
        verb_column = spec.columns[verb_index]
        if verb_column.vocabulary is None:
            raise ContractError("actions.verb must declare a vocabulary")
        verbs = schema.vocabularies[verb_column.vocabulary]
        verb_id = actions[base + verb_index]
        verb = verbs[verb_id]
        if verb_id == 0:
            raise ContractError("legal action row cannot use the padding verb")

        source = _candidate_reference(
            schema, spec, index, actions, known, indexes_by_name["source_row"]
        )
        target = _candidate_reference(
            schema, spec, index, actions, known, indexes_by_name["target_row"]
        )
        choice_index = indexes_by_name["choice_kind"]
        choice: str | None = None
        choice_known = bool(known[base + choice_index])
        choice_applicable = verb in _CHOICE_VERBS
        if choice_known and not choice_applicable:
            raise ContractError("choice_kind is known on a verb where it is inapplicable")
        if choice_known:
            choice_column = spec.columns[choice_index]
            if choice_column.vocabulary is None:
                raise ContractError("actions.choice_kind must declare a vocabulary")
            choice = schema.vocabularies[choice_column.vocabulary][actions[base + choice_index]]

        scalars: list[int | None] = []
        scalar_known: list[bool] = []
        for field in ("scalar0", "scalar1"):
            scalar_index = indexes_by_name[field]
            if scalar_index is None or not known[base + scalar_index]:
                scalars.append(None)
                scalar_known.append(False)
            else:
                scalars.append(actions[base + scalar_index])
                scalar_known.append(True)

        scalar0_applicable = verb in _POTION_VERBS
        if scalar_known[0] and not scalar0_applicable:
            raise ContractError("scalar0 is known on a verb where it is inapplicable")
        if scalar_known[1]:
            raise ContractError("scalar1 is reserved and must remain unknown")

        if verb in _POTION_VERBS:
            if source is None or source.group != "potions" or not source.known:
                raise ContractError("potion action must name a known occupied potion source slot")
            if scalar_known[0] and scalars[0] != source.row:
                raise ContractError("potion scalar0 must match the public potion source slot")

        result.append(CombatCandidate(
            index=index,
            verb=verb,
            source=source,
            target=target,
            choice=choice,
            choice_known=choice_known,
            choice_applicable=choice_applicable,
            scalar0=scalars[0],
            scalar0_known=scalar_known[0],
            scalar0_applicable=scalar0_applicable,
            scalar1=scalars[1],
            scalar1_known=scalar_known[1],
            scalar1_applicable=False,
        ))
    return tuple(result)


def parse_lifecycle(info: Mapping[str, object]) -> Lifecycle:
    """Copy only the documented lifecycle fields from boundary metadata."""
    metadata = _required_mapping(info, "lifecycle info")
    if metadata.get("combat_view_id") != _COMBAT_VIEW_ID:
        raise ContractError("lifecycle combat_view_id is not the accepted common view")
    required = (
        "active", "routing", "selection_owner", "active_visit", "combats_entered",
        "combats_settled", "last_exit",
    )
    missing = tuple(key for key in required if key not in metadata)
    if missing:
        raise ContractError(f"lifecycle info is missing fields: {', '.join(missing)}")
    return Lifecycle(
        active=metadata["active"],
        routing=metadata["routing"],
        selection_owner=metadata["selection_owner"],
        active_visit=metadata["active_visit"],
        combats_entered=metadata["combats_entered"],
        combats_settled=metadata["combats_settled"],
        last_exit=metadata["last_exit"],
    )


__all__ = [
    "ActionSpace",
    "ColumnSpec",
    "CombatCandidate",
    "CombatReference",
    "CombatSchema",
    "ContractError",
    "TensorSpec",
    "candidates",
    "parse_common_descriptor",
    "parse_lifecycle",
    "validate_frame",
]
