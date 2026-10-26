"""Owned, dependency-free snapshots of public common-view arrays."""

from __future__ import annotations

from collections.abc import Mapping

from spire_learn.contracts import ContractError
from spire_learn.neural.contracts import BufferArray, CombatFrame
from spire_learn.neural.schema import CombatSchema, validate_frame


def _snapshot_array(value: object, *, key: str, dtype: str, shape: tuple[int, ...]) -> BufferArray:
    """Copy one C-contiguous native integer array via its checked array protocol."""
    actual_shape = getattr(value, "shape", None)
    if not isinstance(actual_shape, tuple) or actual_shape != shape:
        raise ContractError(f"runtime tensor {key} has the wrong shape")
    ndim = getattr(value, "ndim", None)
    if ndim is not None and (type(ndim) is not int or ndim != len(shape)):
        raise ContractError(f"runtime tensor {key} has the wrong rank")

    actual_dtype = getattr(value, "dtype", None)
    if actual_dtype is None or getattr(actual_dtype, "name", None) != dtype:
        raise ContractError(f"runtime tensor {key} has the wrong dtype")
    expected_kind, expected_size = ("i", 1) if dtype == "int8" else ("i", 8)
    if getattr(actual_dtype, "kind", None) != expected_kind or getattr(
        actual_dtype, "itemsize", None
    ) != expected_size:
        raise ContractError(f"runtime tensor {key} has an unsupported integer representation")
    is_native = getattr(actual_dtype, "isnative", None)
    if is_native is None or not bool(is_native):
        raise ContractError(f"runtime tensor {key} must use native-endian data")

    flags = getattr(value, "flags", None)
    c_contiguous = None if flags is None else getattr(flags, "c_contiguous", None)
    if c_contiguous is None or not bool(c_contiguous):
        raise ContractError(f"runtime tensor {key} must be C-contiguous")

    tobytes = getattr(value, "tobytes", None)
    if not callable(tobytes):
        raise ContractError(f"runtime tensor {key} must expose a byte-copy operation")
    try:
        owned_bytes = tobytes(order="C")
    except Exception as error:
        raise ContractError(f"runtime tensor {key} could not be copied in C order") from error
    if type(owned_bytes) is not bytes:
        raise ContractError(f"runtime tensor {key} byte-copy operation must return bytes")
    try:
        return BufferArray(dtype, shape, owned_bytes)
    except ContractError as error:
        raise ContractError(f"runtime tensor {key} byte length does not match its shape") from error


def snapshot_common(raw: Mapping[str, object], schema: CombatSchema) -> CombatFrame:
    """Copy and validate exactly the descriptor tensors into immutable learner bytes."""
    if not isinstance(schema, CombatSchema):
        raise ContractError("schema must be a CombatSchema")
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ContractError("runtime observation must map tensor names to arrays")
    expected = set(schema.tensors)
    actual = set(raw)
    if actual != expected:
        missing = expected - actual
        unexpected = actual - expected
        raise ContractError(
            f"runtime tensor keys mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    owned = {
        key: _snapshot_array(
            raw[key], key=key, dtype=spec.dtype, shape=spec.shape
        )
        for key, spec in schema.tensors.items()
    }
    frame = CombatFrame(owned, schema.schema_fingerprint)
    validate_frame(frame, schema)
    return frame


__all__ = ["snapshot_common"]
