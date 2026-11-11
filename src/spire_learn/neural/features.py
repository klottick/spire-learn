"""Descriptor-typed public feature plans and owned tensorization."""

from __future__ import annotations

from spire_learn.neural.profiling import observed

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.contracts import CombatFrame
from spire_learn.neural.schema import CombatSchema, validate_frame


FEATURES_ID = "spire-learn-combat-features-v1"


@dataclass(frozen=True)
class FeatureField:
    """One descriptor column's semantic encoding role."""

    field: str
    kind: str
    vocabulary: str | None
    reference: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ContractError("FeatureField field must be a non-empty string")
        if self.kind not in {"categorical", "numeric", "reference"}:
            raise ContractError("FeatureField kind must be categorical, numeric, or reference")
        if self.kind == "categorical":
            if not isinstance(self.vocabulary, str) or not self.vocabulary or self.reference is not None:
                raise ContractError("categorical FeatureField requires only a vocabulary")
        elif self.kind == "reference":
            if not isinstance(self.reference, str) or not self.reference or self.vocabulary is not None:
                raise ContractError("reference FeatureField requires only a selector/target")
        elif self.vocabulary is not None or self.reference is not None:
            raise ContractError("numeric FeatureField cannot name a vocabulary or reference")


@dataclass(frozen=True)
class FeatureContract:
    """Immutable field plan derived from the authenticated public descriptor."""

    features_id: str
    schema_fingerprint: str
    groups: Mapping[str, tuple[FeatureField, ...]]

    def __post_init__(self) -> None:
        if self.features_id != FEATURES_ID:
            raise ContractError("FeatureContract has an unsupported features_id")
        if not isinstance(self.schema_fingerprint, str) or not self.schema_fingerprint:
            raise ContractError("FeatureContract requires a schema fingerprint")
        if not isinstance(self.groups, Mapping):
            raise ContractError("FeatureContract groups must be a mapping")
        frozen: dict[str, tuple[FeatureField, ...]] = {}
        for name, fields in self.groups.items():
            if not isinstance(name, str) or not name:
                raise ContractError("FeatureContract group names must be non-empty strings")
            if not isinstance(fields, tuple) or any(not isinstance(item, FeatureField) for item in fields):
                raise ContractError("FeatureContract groups must contain FeatureField tuples")
            frozen[name] = fields
        object.__setattr__(self, "groups", MappingProxyType(frozen))


def feature_contract(schema: CombatSchema) -> FeatureContract:
    """Return an immutable, stdlib-only feature plan in descriptor column order."""
    if not isinstance(schema, CombatSchema):
        raise ContractError("feature_contract requires an authenticated CombatSchema")
    groups: dict[str, tuple[FeatureField, ...]] = {}
    for group, spec in schema.tensors.items():
        if not spec.columns:
            continue
        fields: list[FeatureField] = []
        for column in spec.columns:
            if column.reference is not None:
                item = FeatureField(column.field, "reference", None, column.reference)
            elif column.vocabulary is not None:
                item = FeatureField(column.field, "categorical", column.vocabulary, None)
            else:
                item = FeatureField(column.field, "numeric", None, None)
            fields.append(item)
        groups[group] = tuple(fields)
    return FeatureContract(FEATURES_ID, schema.schema_fingerprint, groups)


@dataclass(frozen=True)
class EncodedBatch:
    """Owned raw public values, masks, candidate legality and known coverage."""

    schema_fingerprint: str
    values: Mapping[str, Any]
    known: Mapping[str, Any]
    occupied: Mapping[str, Any]
    action_mask: Any
    known_coverage: Mapping[str, Any]
    _storage: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_fingerprint, str) or not self.schema_fingerprint:
            raise ContractError("EncodedBatch requires a schema fingerprint")
        for name in ("values", "known", "occupied", "known_coverage"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ContractError(f"EncodedBatch {name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        if not isinstance(self._storage, tuple):
            raise ContractError("EncodedBatch storage owners must be an immutable tuple")

    @property
    def batch_size(self) -> int:
        return int(self.action_mask.shape[0])

    def select_rows(self, indices: Any) -> EncodedBatch:
        """Copy ordered rows after checking bounds on CPU before device indexing.

        Indices may come from any device supporting a CPU copy. The checked
        vector is transferred once to the batch device; index_select allocates
        independent writable storage, including for duplicate selected rows.
        """
        import torch

        if (not isinstance(indices, torch.Tensor)
                or indices.dtype != torch.int64
                or indices.ndim != 1
                or indices.numel() == 0
                or indices.layout != torch.strided):
            raise ContractError("row indices must be a nonempty 1-D int64 tensor")
        try:
            cpu_indices = indices.detach().cpu()
            invalid = ((cpu_indices < 0) | (cpu_indices >= self.batch_size)).any().item()
        except (TypeError, RuntimeError, ValueError) as error:
            raise ContractError("row indices must support CPU bounds validation") from error
        if invalid:
            raise ContractError("row indices are outside the encoded batch")
        device_indices = cpu_indices.to(device=self.action_mask.device)

        def select_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
            return {key: tensor.index_select(0, device_indices) for key, tensor in values.items()}

        return EncodedBatch(
            schema_fingerprint=self.schema_fingerprint,
            values=select_mapping(self.values),
            known=select_mapping(self.known),
            occupied=select_mapping(self.occupied),
            action_mask=self.action_mask.index_select(0, device_indices),
            known_coverage=select_mapping(self.known_coverage),
        )


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "int8":
        return torch.int8
    if dtype == "int64":
        return torch.int64
    raise ContractError(f"unsupported descriptor tensor dtype {dtype!r}")


def _pack_tensor(torch: Any, frames: Sequence[CombatFrame], key: str, device: str) -> Any:
    """Pack one public tensor key into fresh writable batch storage on CPU."""
    buffer = frames[0].tensors[key]
    writable = bytearray().join(frame.tensors[key].data for frame in frames)
    try:
        view = torch.frombuffer(writable, dtype=_torch_dtype(torch, buffer.dtype))
        # frombuffer retains its writable owner. CPU views keep that storage
        # alive; a synchronous device copy needs no retained host duplicate.
        return view.reshape(len(frames), *buffer.shape).to(device=device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ContractError(f"could not tensorize public frame buffer {key}") from error


@observed("features_transfer")
def encode_frames(
    frames: Sequence[CombatFrame],
    schema: CombatSchema,
    *,
    device: str,
) -> EncodedBatch:
    """Copy and encode validated frames without exposing lifecycle metadata."""
    import torch

    if not isinstance(schema, CombatSchema):
        raise ContractError("encode_frames requires a CombatSchema")
    if not isinstance(device, str) or not device:
        raise ContractError("device must be a non-empty device string")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ContractError("frames must be a sequence of CombatFrame values")
    selected = tuple(frames)
    if not selected:
        raise ContractError("encode_frames requires at least one frame")
    raw_by_key: dict[str, Any] = {}
    for frame in selected:
        if not isinstance(frame, CombatFrame):
            raise ContractError("frames must contain only CombatFrame values")
        validate_frame(frame, schema)
    for key in schema.tensors:
        raw_by_key[key] = _pack_tensor(torch, selected, key, device)

    value_tensors: dict[str, Any] = {}
    known_tensors: dict[str, Any] = {}
    occupied_tensors: dict[str, Any] = {}
    coverage_tensors: dict[str, Any] = {}
    for group, spec in schema.tensors.items():
        if not spec.columns:
            continue
        columns = len(spec.columns)
        rows = spec.shape[0] if len(spec.shape) == 2 else 1
        values = raw_by_key[group].reshape(len(selected), rows, columns)
        known = raw_by_key[spec.known_key].reshape(len(selected), rows, columns).to(torch.bool)
        value_tensors[group] = values
        known_tensors[group] = known
        if spec.occupied_key is not None:
            occupied_tensors[group] = raw_by_key[spec.occupied_key].to(torch.bool)
            coverage_tensors[group] = known.to(torch.float32).mean(dim=1)
        else:
            coverage_tensors[group] = known[:, 0, :].to(torch.float32)

    action_mask = raw_by_key["action_mask"].to(torch.bool)
    return EncodedBatch(
        schema_fingerprint=schema.schema_fingerprint,
        values=value_tensors,
        known=known_tensors,
        occupied=occupied_tensors,
        action_mask=action_mask,
        known_coverage=coverage_tensors,
    )


__all__ = [
    "FEATURES_ID",
    "EncodedBatch",
    "FeatureContract",
    "FeatureField",
    "encode_frames",
    "feature_contract",
]
