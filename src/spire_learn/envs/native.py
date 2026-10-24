"""Pure-core adapter for the public Spire Gymnasium contract.

Optional bindings are intentionally not imported here. Runtime construction is
held behind a later authority gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..contracts import ArrayValue, ContractError, Failure, Observation, frozen_mapping
from ..schema import EnvironmentContract, ObservationSchema


NATIVE_CONTRACT = EnvironmentContract(
    frozenset({"spire-lab-combat-observation-v1"}),
    frozenset({"spire-lab-combat-reference-v2"}),
    "spire-lab-combat-winloss-v1",
    "Documented Spire Gym combat win/loss reward profile; see public combat environment guide.",
)


@dataclass(frozen=True)
class NativeErrorTypes:
    input: type[Exception]
    state: type[Exception]
    setup: type[Exception]
    fault: type[Exception]


def validate_actions(actions: Sequence[int]) -> tuple[int, ...]:
    result = tuple(actions)
    if any(type(action) is not int or action < 0 for action in result):
        raise ContractError("actions must be non-negative integers")
    return result


def native_snapshot(raw: Mapping[str, object], schema: ObservationSchema) -> Observation:
    specs = {item.key: item for item in schema.tensors}
    if set(raw) != set(specs):
        raise ContractError("observation keys must exactly match descriptor")
    values: dict[str, ArrayValue] = {}
    for key, spec in specs.items():
        source = raw[key]
        if isinstance(source, ArrayValue):
            dtype, shape, flat = source.dtype, source.shape, source.values
        else:
            try:
                dtype = source.dtype.name
                native = source.dtype.isnative
                shape = tuple(source.shape)
                listed = source.tolist()
            except (AttributeError, TypeError) as exc:
                raise ContractError(f"{key} is not a public numeric array value") from exc
            if native is not True:
                raise ContractError(f"{key} must use native-endian values")
            flat = _flatten(listed, shape)
        if dtype != spec.dtype or tuple(shape) != spec.shape:
            raise ContractError(f"{key} dtype or shape does not match descriptor")
        if dtype not in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"}:
            raise ContractError(f"{key} has unsupported dtype")
        _validate_dtype_values(dtype, flat, key)
        values[key] = ArrayValue(dtype, tuple(shape), tuple(flat))
    return frozen_mapping(values)


def _flatten(value: object, shape: tuple[int, ...]) -> tuple[int | float, ...]:
    if not shape:
        return (value,) if type(value) in (int, float) else ()
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        raise ContractError("array nested shape does not match descriptor")
    if len(shape) == 1:
        if any(type(v) not in (int, float) for v in value):
            raise ContractError("array values must be numeric primitives")
        return tuple(value)
    return tuple(item for row in value for item in _flatten(row, shape[1:]))


def _validate_dtype_values(dtype: str, values: tuple[int | float, ...], key: str) -> None:
    integer_bits = {"int8": 8, "int16": 16, "int32": 32, "int64": 64,
                    "uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64}
    if dtype in integer_bits:
        bits = integer_bits[dtype]
        lower, upper = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if dtype.startswith("int") else (0, (1 << bits) - 1))
        if any(type(value) is not int or not lower <= value <= upper for value in values):
            raise ContractError(f"{key} values do not fit {dtype}")
        return
    max_finite = {"float16": 65504.0, "float32": 3.4028234663852886e38, "float64": 1.7976931348623157e308}[dtype]
    for value in values:
        if type(value) not in (int, float):
            raise ContractError(f"{key} values must be numeric primitives")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ContractError(f"{key} value is outside {dtype} range") from exc
        if not math.isfinite(numeric) or abs(numeric) > max_finite:
            raise ContractError(f"{key} value is outside {dtype} range")


def normalize_public_info(info: Mapping[str, object], schema: ObservationSchema) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for key in ("outcome", "episode_steps", "steps", "final_resources"):
        if key not in info:
            continue
        value = info[key]
        if key == "outcome":
            vocab = schema.vocabularies.get("outcome", ())
            if type(value) is int and 0 <= value < len(vocab):
                result[key] = vocab[value]
            elif isinstance(value, str) and value in vocab:
                result[key] = value
        elif key in ("episode_steps", "steps") and type(value) is int and value >= 0:
            result["steps"] = value
        elif key == "final_resources" and isinstance(value, Mapping):
            result[key] = MappingProxyType({k: v for k, v in value.items() if isinstance(k, str) and (v is None or type(v) is int)})
    return MappingProxyType(result)


def native_failure(error: BaseException, *, error_types: NativeErrorTypes | None = None) -> Failure | None:
    if error_types is None:
        import sys

        module = sys.modules.get("spire_lab._native")
        if module is None:
            return None
        names = ("CombatInputError", "CombatStateError", "CombatSetupError", "CombatFaultError")
        resolved = tuple(getattr(module, name, None) for name in names)
        if any(not isinstance(value, type) or not issubclass(value, Exception) for value in resolved):
            return None
        error_types = NativeErrorTypes(*resolved)
    if any(not isinstance(value, type) or not issubclass(value, Exception) for value in (error_types.input, error_types.state, error_types.setup, error_types.fault)):
        return None
    kinds = ((error_types.input, "invalid_action"), (error_types.state, "environment_state"), (error_types.setup, "setup_error"), (error_types.fault, "simulator_fault"))
    for error_type, kind in kinds:
        if isinstance(error, error_type):
            return Failure(kind, str(error), "environment", type(error).__name__)
    return None


def pack_actions(actions: Sequence[int]) -> object:
    validated = validate_actions(actions)
    import numpy

    return numpy.asarray(validated, dtype=numpy.dtype("=i8"))


def make_single(case: object) -> object:
    from spire_lab.gymnasium import CombatEnv

    return CombatEnv(catalog=None, partition=case.partition, record_replay=False)


def make_vector(num_envs: int, partition: str, workers: int = 1) -> object:
    from spire_lab.gymnasium import CombatVectorEnv

    return CombatVectorEnv(num_envs, catalog=None, partition=partition, workers=workers, record_replay=False)


def public_build_info() -> Mapping[str, object] | None:
    import sys

    module = sys.modules.get("spire_lab._native")
    getter = getattr(module, "build_info", None) if module is not None else None
    if not callable(getter):
        return None
    value = getter()
    return _freeze_public_metadata(value) if isinstance(value, Mapping) else None


def _freeze_public_metadata(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ContractError("build metadata floats must be finite")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("build metadata keys must be strings")
        return MappingProxyType({key: _freeze_public_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_public_metadata(item) for item in value)
    raise ContractError("build metadata must contain JSON-compatible values")
