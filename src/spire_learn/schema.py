"""Reader for public observation descriptors and candidate tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from .contracts import ArrayValue, Candidate, ContractError, Observation


@dataclass(frozen=True)
class TensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]
    fields: tuple[str, ...]
    axes: tuple[str, ...]
    category: str | None
    mask: str | None


@dataclass(frozen=True)
class ObservationSchema:
    schema_id: str
    content_id: str
    fingerprint: str
    action_count: int
    tensors: tuple[TensorSpec, ...]
    vocabularies: Mapping[str, tuple[str, ...]]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


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
        shape_raw, fields_raw, axes_raw = item.get("shape"), item.get("fields"), item.get("axes")
        if not isinstance(shape_raw, (list, tuple)) or any(type(d) is not int or d < 0 for d in shape_raw):
            raise ContractError(f"{key}.shape must contain non-negative integer dimensions")
        if not isinstance(fields_raw, (list, tuple)) or any(not isinstance(f, str) or not f for f in fields_raw):
            raise ContractError(f"{key}.fields must contain strings")
        fields, axes, shape = tuple(fields_raw), tuple(axes_raw) if isinstance(axes_raw, (list, tuple)) else (), tuple(shape_raw)
        if len(set(fields)) != len(fields):
            raise ContractError(f"duplicate field name in tensor {key}")
        if len(axes) != len(shape) or any(not isinstance(axis, str) or not axis for axis in axes):
            raise ContractError(f"{key}.axes must name every tensor dimension")
        if len(set(axes)) != len(axes):
            raise ContractError(f"{key}.axes must be unique")
        category, mask = item.get("category"), item.get("mask")
        if category is not None and (not isinstance(category, str) or category not in vocabularies):
            raise ContractError(f"{key}.category must name a declared vocabulary")
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
    if any(t.mask is not None and t.mask not in keys for t in tensors):
        raise ContractError("tensor mask references must name declared tensors")
    return ObservationSchema(schema_id, content_id, fingerprint, action_count, tuple(tensors), MappingProxyType(vocabularies))


def candidates_for(observation: Observation, schema: ObservationSchema) -> tuple[Candidate, ...]:
    mask_spec = next((tensor for tensor in schema.tensors if tensor.key == "action_mask"), None)
    if mask_spec is None:
        raise ContractError("descriptor has no action_mask tensor")
    mask_value = observation.get(mask_spec.key)
    if not isinstance(mask_value, ArrayValue) or len(mask_value.values) != schema.action_count:
        raise ContractError("action mask width must match action_space.n")
    if mask_value.shape != mask_spec.shape:
        raise ContractError("action mask shape differs from descriptor")
    mask: list[bool] = []
    for value in mask_value.values:
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ContractError("action mask values must be binary")
        mask.append(bool(value))
    tensors: dict[str, ArrayValue] = {}
    for spec in schema.tensors:
        value = observation.get(spec.key)
        if value is None:
            continue
        if not isinstance(value, ArrayValue) or value.shape != spec.shape:
            raise ContractError(f"observation tensor {spec.key} does not match descriptor shape")
        tensors[spec.key] = value
    candidates: list[Candidate] = []
    for spec in schema.tensors:
        if spec.category is None or "candidate" not in spec.axes or spec.key not in tensors:
            continue
        vocab = schema.vocabularies[spec.category]
        values = tensors[spec.key].values
        if len(values) != schema.action_count:
            raise ContractError(f"categorical tensor {spec.key} width must match action count")
        for index, legal in enumerate(mask):
            if not legal:
                continue
            code = values[index]
            if type(code) is not int or not 0 <= code < len(vocab):
                raise ContractError(f"category code out of range in {spec.key}")
        for index, legal in enumerate(mask):
            if not legal:
                continue
            code = values[index]
            features: list[tuple[str, int]] = []
            source = target = None
            for feature_spec in schema.tensors:
                feature_value = tensors.get(feature_spec.key)
                if feature_value is None or "candidate" not in feature_spec.axes:
                    continue
                candidate_axis = feature_spec.axes.index("candidate")
                width = feature_spec.shape[candidate_axis]
                if width != schema.action_count:
                    raise ContractError(f"candidate-axis count differs from action count in {feature_spec.key}")
                for field in feature_spec.fields:
                    if field in ("source", "target"):
                        field_index = feature_spec.fields.index(field)
                        if "field" in feature_spec.axes and "candidate" in feature_spec.axes:
                            offset = _flat_index(feature_spec.shape, feature_spec.axes, index, field_index)
                            raw = feature_value.values[offset]
                            if type(raw) is not int:
                                raise ContractError(f"{field} must be an integer")
                            if field == "source": source = raw
                            else: target = raw
                    elif field not in ("kind",):
                        field_index = feature_spec.fields.index(field)
                        if "field" in feature_spec.axes and "candidate" in feature_spec.axes and feature_spec.shape[feature_spec.axes.index("field")] == len(feature_spec.fields):
                            raw = feature_value.values[_flat_index(feature_spec.shape, feature_spec.axes, index, field_index)]
                            if type(raw) is not int:
                                raise ContractError(f"feature {field} must be an integer")
                            features.append((field, raw))
            candidates.append(Candidate(index, vocab[code], source, target, tuple(features)))
    if not any(t.category is not None and "candidate" in t.axes for t in schema.tensors):
        raise ContractError("descriptor has no categorical candidate tensor")
    return tuple(candidates)


def _flat_index(shape: tuple[int, ...], axes: tuple[str, ...], candidate: int, field: int) -> int:
    """Return the row-major offset for named candidate/field coordinates."""
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


def fixture_contract() -> EnvironmentContract:
    return EnvironmentContract(
        frozenset({"spire-learn-fixture-v1"}),
        frozenset({"spire-learn-fixture-content-v1"}),
        "fixture-protocol-v1",
        "Authored protocol fixtures only; not evidence of game reward semantics.",
    )
