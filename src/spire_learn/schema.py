"""Reader for public observation descriptors and candidate tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import TypeAlias

from .contracts import ArrayValue, Candidate, ContractError, Observation


_NUMERIC_DTYPES = frozenset({
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
})
_SIGNED_INTEGER_DTYPES = frozenset({"int8", "int16", "int32", "int64"})
_ACTION_TENSOR_KEYS = ("action_mask", "action_kind", "action_source", "action_target", "action_features")
CategorySpec: TypeAlias = str | tuple[str | None, ...] | None


@dataclass(frozen=True)
class TensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]
    fields: tuple[str, ...]
    axes: tuple[str, ...]
    category: CategorySpec
    mask: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "axes", tuple(self.axes))
        if isinstance(self.category, list):
            object.__setattr__(self, "category", tuple(self.category))
        if isinstance(self.category, tuple) and any(value is not None and (not isinstance(value, str) or not value) for value in self.category):
            raise ContractError("per-field category entries must be vocabulary names or None")
        if self.category is not None and not isinstance(self.category, (str, tuple)):
            raise ContractError("category must be a vocabulary name, per-field category sequence, or None")


@dataclass(frozen=True)
class ObservationSchema:
    schema_id: str
    content_id: str
    fingerprint: str
    action_count: int
    tensors: tuple[TensorSpec, ...]
    vocabularies: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "vocabularies", MappingProxyType({key: tuple(values) for key, values in self.vocabularies.items()}))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _parse_category(
    raw: object,
    key: str,
    fields: tuple[str, ...],
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    vocabularies: Mapping[str, tuple[str, ...]],
) -> CategorySpec:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw not in vocabularies:
            raise ContractError(f"{key}.category must name a declared vocabulary")
        return raw
    if not isinstance(raw, (list, tuple)):
        raise ContractError(f"{key}.category must be a vocabulary name or per-field sequence")
    if "field" not in axes:
        raise ContractError(f"{key}.category sequences require a named field axis")
    field_index = axes.index("field")
    if len(raw) != len(fields) or len(raw) != shape[field_index]:
        raise ContractError(f"{key}.category sequence must match its named fields")
    categories = tuple(raw)
    for category in categories:
        if category is not None and (not isinstance(category, str) or category not in vocabularies):
            raise ContractError(f"{key}.category entries must name declared vocabularies or be None")
    return categories


def parse_descriptor(raw: Mapping[str, object], *, allowed_schema_ids: frozenset[str]) -> ObservationSchema:
    root = _mapping(raw, "descriptor")
    schema_id = _string(root.get("schema_id"), "schema_id")
    content_id = _string(root.get("content_id"), "content_id")
    if schema_id not in allowed_schema_ids:
        raise ContractError(f"unsupported schema_id: {schema_id}")
    fingerprint = _string(root.get("fingerprint"), "fingerprint")
    if re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
        raise ContractError("fingerprint must be a 64-character hexadecimal value")
    action_space = _mapping(root.get("action_space"), "action_space")
    action_count = action_space.get("n")
    if type(action_count) is not int or action_count <= 0:
        raise ContractError("action_space.n must be a positive integer")
    if action_space.get("type") != "Discrete":
        raise ContractError("action_space.type must be Discrete")
    if type(action_space.get("start")) is not int or action_space["start"] != 0:
        raise ContractError("action_space.start must be the integer zero")
    raw_vocabularies = _mapping(root.get("vocabularies"), "vocabularies")
    vocabularies: dict[str, tuple[str, ...]] = {}
    for key, values in raw_vocabularies.items():
        if not isinstance(key, str) or not isinstance(values, (list, tuple)) or not values:
            raise ContractError("vocabularies must map names to non-empty label lists")
        labels = tuple(_string(label, f"vocabulary {key}") for label in values)
        if len(set(labels)) != len(labels):
            raise ContractError(f"duplicate label in vocabulary {key}")
        vocabularies[key] = labels
    raw_tensors = root.get("tensors")
    if not isinstance(raw_tensors, (list, tuple)) or not raw_tensors:
        raise ContractError("tensors must be a non-empty list")
    tensors: list[TensorSpec] = []
    keys: set[str] = set()
    for raw_tensor in raw_tensors:
        item = _mapping(raw_tensor, "tensor")
        required = {"key", "dtype", "shape", "fields", "axes", "category", "mask"}
        if not required.issubset(item):
            raise ContractError("tensor descriptor is missing required fields")
        key = _string(item.get("key"), "tensor key")
        dtype = _string(item.get("dtype"), f"{key}.dtype")
        if dtype not in _NUMERIC_DTYPES:
            raise ContractError(f"{key}.dtype must be a supported numeric dtype")
        shape_raw, fields_raw, axes_raw = item.get("shape"), item.get("fields"), item.get("axes")
        if not isinstance(shape_raw, (list, tuple)) or any(type(d) is not int or d < 0 for d in shape_raw):
            raise ContractError(f"{key}.shape must contain non-negative integer dimensions")
        if not isinstance(fields_raw, (list, tuple)) or any(not isinstance(f, str) or not f for f in fields_raw):
            raise ContractError(f"{key}.fields must contain strings")
        if not isinstance(axes_raw, (list, tuple)):
            raise ContractError(f"{key}.axes must be a list or tuple")
        fields, axes, shape = tuple(fields_raw), tuple(axes_raw), tuple(shape_raw)
        if len(set(fields)) != len(fields):
            raise ContractError(f"duplicate field name in tensor {key}")
        if len(axes) != len(shape) or any(not isinstance(axis, str) or not axis for axis in axes):
            raise ContractError(f"{key}.axes must name every tensor dimension")
        if len(set(axes)) != len(axes):
            raise ContractError(f"{key}.axes must be unique")
        category = _parse_category(item.get("category"), key, fields, axes, shape, vocabularies)
        mask = item.get("mask")
        if mask is not None and (not isinstance(mask, str) or not mask):
            raise ContractError(f"{key}.mask must be a descriptor reference or None")
        if "field" in axes:
            axis_index = axes.index("field")
            if axis_index >= len(shape) or shape[axis_index] != len(fields):
                raise ContractError(f"{key} field-axis count does not match fields")
        if "candidate" in axes and shape[axes.index("candidate")] != action_count:
            raise ContractError(f"{key} candidate-axis count does not match action_space.n")
        if key in keys:
            raise ContractError(f"duplicate tensor key: {key}")
        keys.add(key)
        tensors.append(TensorSpec(key, dtype, shape, fields, axes, category, mask))
    for tensor in tensors:
        if tensor.mask is None:
            continue
        parts = tensor.mask.split(":")
        if len(parts) > 2 or not parts[0] or (len(parts) == 2 and not parts[1]):
            raise ContractError(f"malformed mask reference on {tensor.key}")
        if parts[0] not in keys:
            raise ContractError(f"mask reference on {tensor.key} names an unknown tensor")
        if len(parts) == 2 and parts[1] not in tensor.fields:
            raise ContractError(f"mask qualifier on {tensor.key} must name one of its fields")
    _validate_action_descriptors({tensor.key: tensor for tensor in tensors}, action_count)
    return ObservationSchema(schema_id, content_id, fingerprint, action_count, tuple(tensors), MappingProxyType(vocabularies))


def _validate_action_descriptors(tensors: Mapping[str, TensorSpec], action_count: int) -> None:
    missing = set(_ACTION_TENSOR_KEYS).difference(tensors)
    if missing:
        raise ContractError(f"descriptor is missing required action tensors: {', '.join(sorted(missing))}")
    mask, kind, source, target, features = (tensors[key] for key in _ACTION_TENSOR_KEYS)
    simple = (
        (mask, "int8", ("candidate",), ("legal",), None),
        (kind, _SIGNED_INTEGER_DTYPES, ("candidate",), ("kind",), "action"),
        (source, _SIGNED_INTEGER_DTYPES, ("candidate",), ("hand_row",), None),
        (target, _SIGNED_INTEGER_DTYPES, ("candidate",), ("enemy_row",), None),
    )
    for spec, dtype, axes, fields, category in simple:
        dtype_matches = spec.dtype == dtype if isinstance(dtype, str) else spec.dtype in dtype
        if not dtype_matches or spec.shape != (action_count,) or spec.axes != axes or spec.fields != fields or spec.category != category:
            raise ContractError(f"{spec.key} does not match the required public action tensor contract")
    if mask.mask is not None:
        raise ContractError("action_mask must not reference another mask")
    if any(spec.mask != "action_mask" for spec in (kind, source, target, features)):
        raise ContractError("action kind/source/target/features tensors must reference action_mask")
    if features.dtype not in _SIGNED_INTEGER_DTYPES or set(features.axes) != {"candidate", "field"} or len(features.axes) != 2:
        raise ContractError("action_features must use a supported signed integer dtype and candidate/field axes only")
    if not features.fields:
        raise ContractError("action_features must declare named feature fields")
    if features.shape[features.axes.index("candidate")] != action_count or features.shape[features.axes.index("field")] != len(features.fields):
        raise ContractError("action_features shape must match candidate count and named fields")
    if features.category is not None:
        raise ContractError("action_features must not declare a category vocabulary")


def candidates_for(observation: Observation, schema: ObservationSchema) -> tuple[Candidate, ...]:
    specs = {tensor.key: tensor for tensor in schema.tensors}
    _validate_action_descriptors(specs, schema.action_count)
    values: dict[str, ArrayValue] = {}
    for key in _ACTION_TENSOR_KEYS:
        spec = specs[key]
        value = observation.get(key)
        if not isinstance(value, ArrayValue) or value.shape != spec.shape or value.dtype != spec.dtype:
            raise ContractError(f"observation tensor {key} must match descriptor dtype and shape")
        values[key] = value
    mask_value = values["action_mask"]
    if len(mask_value.values) != schema.action_count:
        raise ContractError("action mask width must match action_space.n")
    mask: list[bool] = []
    for value in mask_value.values:
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ContractError("action mask values must be binary")
        mask.append(bool(value))
    kind_spec, source_spec, target_spec, feature_spec = (specs[key] for key in _ACTION_TENSOR_KEYS[1:])
    kind_values = values["action_kind"].values
    source_values = values["action_source"].values
    target_values = values["action_target"].values
    feature_values = values["action_features"].values
    action_vocabulary = schema.vocabularies[kind_spec.category]
    candidates: list[Candidate] = []
    for index, legal in enumerate(mask):
        if not legal:
            continue
        code = kind_values[index]
        if type(code) is not int or not 0 <= code < len(action_vocabulary):
            raise ContractError("action kind code is outside the action vocabulary")
        kind = action_vocabulary[code]
        if kind == "padding":
            raise ContractError("a legal candidate cannot have padding kind")
        source = _optional_row(source_values[index], source_spec.key)
        target = _optional_row(target_values[index], target_spec.key)
        features = tuple(
            (field, _feature_value(feature_values, feature_spec, index, field_index))
            for field_index, field in enumerate(feature_spec.fields)
        )
        candidates.append(Candidate(index, kind, source, target, features))
    return tuple(candidates)


def _optional_row(value: int | float, key: str) -> int | None:
    if type(value) is not int or value < -1:
        raise ContractError(f"{key} row index must be -1 or a non-negative integer")
    return None if value == -1 else value


def _feature_value(values: tuple[int | float, ...], spec: TensorSpec, candidate: int, field: int) -> int:
    raw = values[_flat_index(spec.shape, spec.axes, candidate, field)]
    if type(raw) is not int:
        raise ContractError(f"action feature {spec.fields[field]} must be an integer")
    return raw


def _flat_index(shape: tuple[int, ...], axes: tuple[str, ...], candidate: int, field: int) -> int:
    """Return the row-major offset for named candidate/field coordinates."""
    if set(axes) != {"candidate", "field"} or len(axes) != 2:
        raise ContractError("cannot interpret unsupported candidate tensor axes")
    coordinates = [0] * len(shape)
    coordinates[axes.index("candidate")] = candidate
    coordinates[axes.index("field")] = field
    offset = 0
    for size, coordinate in zip(shape, coordinates):
        offset = offset * size + coordinate
    return offset


@dataclass(frozen=True)
class EnvironmentContract:
    schema_ids: frozenset[str]
    content_ids: frozenset[str]
    reward_contract_id: str
    reward_evidence: str
