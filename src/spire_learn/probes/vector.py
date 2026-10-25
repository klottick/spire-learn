"""Dependency-free native vector probe report contracts."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import math
import platform
import re
from types import MappingProxyType

from ..artifacts.codec import to_json_value
from ..artifacts.writer import RunWriter
from ..agents.baselines import RandomPolicy
from ..agents.base import resolve_decision
from ..collection.boundaries import route_vector_step
from ..contracts import (
    CaseExecution,
    Choice,
    ContractError,
    DecisionRequest,
    Failure,
    PublicDecision,
    Route,
)
from ..envs.native import (
    NATIVE_CONTRACT,
    make_vector,
    native_failure,
    native_snapshot,
    normalize_public_info,
    pack_actions,
    public_build_info,
)
from ..experiments.budget import Budget
from ..experiments.config import ProbeConfig
from ..schema import candidates_for, parse_descriptor


NATIVE_CHECKS = frozenset(
    {"mixed_rows", "old_final", "replacement_reset", "presence_masks", "old_context"}
)
_VERDICTS = frozenset({"compatible", "adaptable", "incompatible", "blocked"})


@dataclass(frozen=True)
class ProbeReport:
    verdict: str
    checks: Mapping[str, str]
    versions: Mapping[str, str | None]
    batch_steps: int
    slot_decisions: int
    replacement_decisions: int
    failures: tuple[Failure, ...]

    def __post_init__(self) -> None:
        if type(self.verdict) is not str:
            raise ContractError("verdict must be a built-in string")
        if self.verdict not in _VERDICTS:
            raise ContractError("verdict must be compatible, adaptable, incompatible, or blocked")
        object.__setattr__(self, "checks", _copy_checks(self.checks))
        object.__setattr__(self, "versions", _copy_versions(self.versions))

        for name in ("batch_steps", "slot_decisions", "replacement_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if self.replacement_decisions > self.slot_decisions:
            raise ContractError("replacement_decisions cannot exceed slot_decisions")

        try:
            failures = tuple(self.failures)
        except TypeError as exc:
            raise ContractError("failures must contain Failure records") from exc
        owned_failures = []
        for failure in failures:
            if not isinstance(failure, Failure):
                raise ContractError("failures must contain Failure records")
            if (
                type(failure.kind) is not str
                or type(failure.reason) is not str
                or type(failure.phase) is not str
                or (failure.exception_type is not None and type(failure.exception_type) is not str)
            ):
                raise ContractError("Failure fields must be built-in strings or None for exception_type")
            owned_failures.append(
                Failure(failure.kind, failure.reason, failure.phase, failure.exception_type)
            )
        object.__setattr__(self, "failures", tuple(owned_failures))


def _copy_checks(values: object) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ContractError("checks must be a mapping")
    copied: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or type(value) is not str:
            raise ContractError("checks must map string keys to built-in string values")
        copied[name] = value
    return MappingProxyType(copied)


def _copy_versions(values: object) -> Mapping[str, str | None]:
    if not isinstance(values, Mapping):
        raise ContractError("versions must be a mapping")
    copied: dict[str, str | None] = {}
    for name, value in values.items():
        if not isinstance(name, str) or (value is not None and type(value) is not str):
            raise ContractError("versions must map string keys to built-in strings or None")
        copied[name] = value
    return MappingProxyType(copied)


def probe_verdict(
    checks: Mapping[str, str],
    *,
    required: frozenset[str],
    runtime_available: bool,
    adaptation_verified: bool = False,
) -> str:
    if not isinstance(checks, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in checks.items()
    ):
        raise ContractError("checks must map strings to strings")
    if not isinstance(required, frozenset) or any(
        not isinstance(name, str) for name in required
    ):
        raise ContractError("required must be a frozenset of strings")
    if not required:
        raise ContractError("required must not be empty")
    if type(runtime_available) is not bool:
        raise ContractError("runtime_available must be a boolean")
    if type(adaptation_verified) is not bool:
        raise ContractError("adaptation_verified must be a boolean")

    if not runtime_available:
        return "blocked"
    if any(checks.get(name) == "fail" for name in required):
        return "incompatible"
    if any(checks.get(name) != "pass" for name in required):
        return "blocked"
    return "adaptable" if adaptation_verified else "compatible"


def vector_probe_verdict(
    checks: Mapping[str, str], *, operational_failure: bool, runtime_available: bool
) -> str:
    """Preserve required contract failures while blocking operational failures."""
    if type(operational_failure) is not bool:
        raise ContractError("operational_failure must be a boolean")
    verdict = probe_verdict(
        checks,
        required=NATIVE_CHECKS,
        runtime_available=runtime_available,
    )
    if operational_failure and not any(checks.get(name) == "fail" for name in NATIVE_CHECKS):
        return "blocked"
    return verdict


def probe_report_to_json(report: ProbeReport) -> dict[str, object]:
    if not isinstance(report, ProbeReport):
        raise ContractError("report must be a ProbeReport")
    failures: list[dict[str, object]] = []
    for failure in report.failures:
        encoded = to_json_value(failure)
        if not isinstance(encoded, dict):
            raise ContractError("Failure did not encode as a JSON object")
        failures.append(encoded)
    return {
        "schema_id": "spire-learn-probe-v1",
        "verdict": report.verdict,
        "checks": dict(report.checks),
        "versions": dict(report.versions),
        "batch_steps": report.batch_steps,
        "slot_decisions": report.slot_decisions,
        "replacement_decisions": report.replacement_decisions,
        "failures": failures,
    }


class _VectorContractViolation(ContractError):
    """A public boundary failure attributable to one named probe check."""

    def __init__(self, check: str, message: str):
        self.check = check
        super().__init__(message)


def _sequence_values(
    value: object,
    *,
    width: int,
    label: str,
    boolean: bool = False,
    object_vector: bool = False,
) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise _VectorContractViolation("presence_masks", f"{label} must be a width-{width} vector")
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            actual_shape = tuple(shape)
        except TypeError as exc:
            raise _VectorContractViolation("presence_masks", f"{label} has an invalid shape") from exc
        if actual_shape != (width,):
            raise _VectorContractViolation("presence_masks", f"{label} must have shape ({width},)")
    dtype = getattr(value, "dtype", None)
    if boolean and dtype is not None and (
        getattr(dtype, "name", None) != "bool" or getattr(dtype, "isnative", None) is not True
    ):
        raise _VectorContractViolation("presence_masks", f"{label} must use native bool dtype")
    if object_vector and dtype is not None and getattr(dtype, "kind", None) != "O":
        raise _VectorContractViolation("presence_masks", f"{label} must be an object vector")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            values = tolist()
        except (TypeError, ValueError) as exc:
            raise _VectorContractViolation("presence_masks", f"{label} cannot be read as a vector") from exc
    elif isinstance(value, Sequence):
        values = value
    else:
        raise _VectorContractViolation("presence_masks", f"{label} must be a sized vector")
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise _VectorContractViolation("presence_masks", f"{label} must contain exactly {width} rows")
    result = tuple(values)
    if boolean and any(type(item) is not bool for item in result):
        raise _VectorContractViolation("presence_masks", f"{label} must contain only booleans")
    return result


def validate_vector_batch_shape(
    batch_shape: Sequence[int], row_shape: Sequence[int], *, width: int
) -> None:
    """Require one exact leading batch axis followed by the descriptor shape."""
    try:
        actual = tuple(batch_shape)
        row = tuple(row_shape)
    except TypeError as exc:
        raise _VectorContractViolation("native_contract", "tensor shapes must be sized integer sequences") from exc
    if (
        type(width) is not int
        or width <= 0
        or any(type(dimension) is not int or dimension < 0 for dimension in actual + row)
        or len(actual) != len(row) + 1
        or actual != (width, *row)
    ):
        raise _VectorContractViolation(
            "native_contract", f"batched tensor shape {actual} must equal ({width}, *{row})"
        )


def validate_probe_config(config: ProbeConfig) -> None:
    """Reject malformed direct probe requests before any runtime construction."""
    if not isinstance(config, ProbeConfig):
        raise ContractError("config must be a ProbeConfig")
    for name in ("num_envs", "workers", "max_batch_steps"):
        value = getattr(config, name)
        minimum = 2 if name == "num_envs" else 1
        if type(value) is not int or value < minimum:
            requirement = "at least two" if name == "num_envs" else "a positive integer"
            raise ContractError(f"{name} must be {requirement}")
    if type(config.partition) is not str or not config.partition:
        raise ContractError("partition must be a non-empty built-in string")
    if type(config.max_wall_seconds) not in (int, float):
        raise ContractError("max_wall_seconds must be a positive finite built-in number")
    try:
        wall_seconds = float(config.max_wall_seconds)
    except OverflowError as exc:
        raise ContractError("max_wall_seconds must be representable as a finite float") from exc
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ContractError("max_wall_seconds must be positive and finite")
    if type(config.executions) is not tuple or len(config.executions) != config.num_envs:
        raise ContractError("executions must be a tuple containing exactly num_envs cases")

    for execution in config.executions:
        if not isinstance(execution, CaseExecution):
            raise ContractError("every execution must be a CaseExecution")
        if any(
            type(value) is not str or not value
            for value in (execution.execution_id, execution.case_id, execution.scenario_id)
        ):
            raise ContractError("execution, case, and scenario IDs must be non-empty built-in strings")
        if execution.split != "selection":
            raise ContractError("all probe executions must use the selection split")
        if execution.partition != config.partition:
            raise ContractError("all probe executions must use the configured partition")
        if any(
            type(seed) is not int or seed < 0
            for seed in (execution.environment_seed, execution.policy_seed)
        ):
            raise ContractError("environment and policy seeds must be non-negative built-in integers")
        if execution.expected_schema_id is not None and (
            type(execution.expected_schema_id) is not str or not execution.expected_schema_id
        ):
            raise ContractError("expected_schema_id must be a non-empty built-in string or None")
        fingerprint = execution.expected_fingerprint
        if fingerprint is not None and (
            type(fingerprint) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None
        ):
            raise ContractError("expected_fingerprint must be a 64-character hexadecimal string or None")


def validate_current_info_presence(info: Mapping[str, object], *, width: int) -> Mapping[str, tuple[object, ...]]:
    """Validate full-width current/reset public columns and their presence masks."""
    if not isinstance(info, Mapping) or type(width) is not int or width <= 0:
        raise _VectorContractViolation("presence_masks", "current info and positive batch width are required")
    required = ("episode_id", "scenario_id", "episode_steps", "outcome")
    optional = ("hp_before_healing", "hp_after_cleanup")
    values: dict[str, tuple[object, ...]] = {}
    for key in required:
        if key not in info or "_" + key not in info:
            raise _VectorContractViolation("presence_masks", f"current info is missing {key} or its mask")
        column = _sequence_values(info[key], width=width, label=key)
        mask = _sequence_values(info["_" + key], width=width, label="_" + key, boolean=True)
        if mask != (True,) * width:
            raise _VectorContractViolation("presence_masks", f"current {key} must be present for every row")
        values[key] = column
    if any(
        type(value) not in (str, int) or (type(value) is str and not value)
        for value in values["episode_id"]
    ):
        raise _VectorContractViolation("presence_masks", "current episode_id values must be public strings or integers")
    if any(type(value) is not str or not value for value in values["scenario_id"]):
        raise _VectorContractViolation("presence_masks", "current scenario_id values must be non-empty strings")
    if any(type(value) is not int or value < 0 for value in values["episode_steps"]):
        raise _VectorContractViolation("presence_masks", "current episode_steps must be non-negative integers")
    if any(type(value) not in (str, int) for value in values["outcome"]):
        raise _VectorContractViolation("presence_masks", "current outcomes must be public strings or integers")
    for key in optional:
        has_value, has_mask = key in info, "_" + key in info
        if has_value != has_mask:
            raise _VectorContractViolation("presence_masks", f"current {key} value and mask must appear together")
        if not has_value:
            continue
        column = _sequence_values(info[key], width=width, label=key)
        mask = _sequence_values(info["_" + key], width=width, label="_" + key, boolean=True)
        if not any(mask):
            raise _VectorContractViolation("presence_masks", f"current {key} must be absent when no row has a value")
        if any(present and type(value) is not int for value, present in zip(column, mask, strict=True)):
            raise _VectorContractViolation("presence_masks", f"present current {key} values must be integers")
        values[key] = column
    return MappingProxyType(values)


def validate_final_batch_presence(
    info: Mapping[str, object], ended: Sequence[bool]
) -> tuple[tuple[object, ...] | None, Mapping[str, tuple[object, ...]] | None]:
    """Validate SAME_STEP final keys and nested full-width presence masks.

    Payloads in rows whose per-key mask is false are placeholders and are not
    interpreted as evidence. The masks, rather than zero/empty fill values,
    define which public values are present.
    """
    try:
        ended_rows = tuple(ended)
    except TypeError as exc:
        raise _VectorContractViolation("presence_masks", "ended rows must be a sized boolean sequence") from exc
    width = len(ended_rows)
    if width == 0 or any(type(value) is not bool for value in ended_rows):
        raise _VectorContractViolation("presence_masks", "ended rows must be non-empty built-in booleans")
    if not isinstance(info, Mapping):
        raise _VectorContractViolation("presence_masks", "step info must be a mapping")

    outer_keys = ("final_obs", "_final_obs", "final_info", "_final_info")
    present_keys = tuple(key in info for key in outer_keys)
    if not any(present_keys):
        if any(ended_rows):
            raise _VectorContractViolation("presence_masks", "ended rows require final observation and info")
        return None, None
    if not all(present_keys):
        raise _VectorContractViolation("presence_masks", "SAME_STEP final keys must be all present or all absent")

    final_observations = _sequence_values(
        info["final_obs"], width=width, label="final_obs", object_vector=True
    )
    final_observation_mask = _sequence_values(
        info["_final_obs"], width=width, label="_final_obs", boolean=True
    )
    final_info_mask = _sequence_values(
        info["_final_info"], width=width, label="_final_info", boolean=True
    )
    if final_observation_mask != ended_rows or final_info_mask != ended_rows:
        raise _VectorContractViolation("presence_masks", "outer final masks must exactly match ended rows")
    for index, did_end in enumerate(ended_rows):
        if did_end and not isinstance(final_observations[index], Mapping):
            raise _VectorContractViolation("presence_masks", "every ended row requires a final observation mapping")
        if not did_end and final_observations[index] is not None:
            raise _VectorContractViolation("presence_masks", "live rows cannot contain final observations")

    final_info = info["final_info"]
    if not isinstance(final_info, Mapping):
        raise _VectorContractViolation("presence_masks", "final_info must be a nested mapping")
    required = ("episode_id", "scenario_id", "episode_steps", "outcome")
    optional = ("hp_before_healing", "hp_after_cleanup")
    allowed_keys = {key for name in (*required, *optional) for key in (name, "_" + name)}
    if set(final_info) - allowed_keys:
        raise _VectorContractViolation("presence_masks", "final_info contains an unsupported public column")
    if any(key not in final_info or "_" + key not in final_info for key in required):
        raise _VectorContractViolation("presence_masks", "final_info is missing a required public column or mask")

    values: dict[str, tuple[object, ...]] = {}
    for key in required:
        column = _sequence_values(final_info[key], width=width, label="final_info." + key)
        mask = _sequence_values(
            final_info["_" + key], width=width, label="final_info._" + key, boolean=True
        )
        if mask != ended_rows:
            raise _VectorContractViolation("presence_masks", f"final_info {key} mask must exactly match ended rows")
        for index, (value, present) in enumerate(zip(column, mask, strict=True)):
            if not present:
                continue
            if key == "episode_id" and (
                type(value) not in (str, int) or (type(value) is str and not value)
            ):
                raise _VectorContractViolation("presence_masks", f"final_info episode_id row {index} is invalid")
            if key == "scenario_id" and (type(value) is not str or not value):
                raise _VectorContractViolation("presence_masks", f"final_info scenario_id row {index} is invalid")
            if key == "episode_steps" and (type(value) is not int or value < 0):
                raise _VectorContractViolation("presence_masks", f"final_info episode_steps row {index} is invalid")
            if key == "outcome" and type(value) not in (str, int):
                raise _VectorContractViolation("presence_masks", f"final_info outcome row {index} is invalid")
        values[key] = column
    for key in optional:
        has_value, has_mask = key in final_info, "_" + key in final_info
        if has_value != has_mask:
            raise _VectorContractViolation("presence_masks", f"final_info {key} value and mask must appear together")
        if not has_value:
            continue
        column = _sequence_values(final_info[key], width=width, label="final_info." + key)
        mask = _sequence_values(
            final_info["_" + key], width=width, label="final_info._" + key, boolean=True
        )
        if not any(mask) or any(present and not ended_rows[index] for index, present in enumerate(mask)):
            raise _VectorContractViolation("presence_masks", f"final_info {key} mask must cover ended rows only")
        if any(present and type(value) is not int for value, present in zip(column, mask, strict=True)):
            raise _VectorContractViolation("presence_masks", f"present final_info {key} values must be integers")
        values[key] = column
    return final_observations, MappingProxyType(values)


def validate_episode_identity(
    *,
    source_ids: Sequence[object],
    source_steps: Sequence[object],
    current_ids: Sequence[object],
    current_steps: Sequence[object],
    ended: Sequence[bool],
    final_info: Mapping[str, Sequence[object]] | None,
) -> None:
    """Check old-final continuity and distinct SAME_STEP replacement identity."""
    ended_rows = tuple(ended)
    width = len(ended_rows)
    if width == 0 or any(type(value) is not bool for value in ended_rows):
        raise ContractError("ended must be a non-empty boolean sequence")
    columns = {
        "source_ids": tuple(source_ids),
        "source_steps": tuple(source_steps),
        "current_ids": tuple(current_ids),
        "current_steps": tuple(current_steps),
    }
    if any(len(values) != width for values in columns.values()):
        raise ContractError("episode identity columns must match the batch width")
    for key in ("source_steps", "current_steps"):
        if any(type(value) is not int or value < 0 for value in columns[key]):
            raise ContractError(f"{key} must contain non-negative built-in integers")
    for key in ("source_ids", "current_ids"):
        if any(type(value) not in (str, int) or (type(value) is str and not value) for value in columns[key]):
            raise ContractError(f"{key} must contain non-empty strings or integers")

    final_ids: tuple[object, ...] | None = None
    final_steps: tuple[object, ...] | None = None
    if any(ended_rows):
        if not isinstance(final_info, Mapping) or not {"episode_id", "episode_steps"} <= set(final_info):
            raise _VectorContractViolation("old_final", "ended rows require old-final identity metadata")
        final_ids = tuple(final_info["episode_id"])
        final_steps = tuple(final_info["episode_steps"])
        if len(final_ids) != width or len(final_steps) != width:
            raise _VectorContractViolation("old_final", "old-final identity columns must match batch width")

    for index, did_end in enumerate(ended_rows):
        source_id = columns["source_ids"][index]
        source_step = columns["source_steps"][index]
        current_id = columns["current_ids"][index]
        current_step = columns["current_steps"][index]
        if did_end:
            if (
                final_ids[index] != source_id
                or type(final_steps[index]) is not int
                or final_steps[index] != source_step + 1
            ):
                raise _VectorContractViolation("old_final", f"ended row {index} does not preserve its old episode")
            if current_id == source_id or current_step != 0:
                raise _VectorContractViolation("replacement_reset", f"ended row {index} is not a distinct reset episode")
        elif current_id != source_id or current_step != source_step + 1:
            raise _VectorContractViolation("old_final", f"live row {index} does not continue its source episode")


@dataclass(frozen=True, eq=False)
class _SentinelState:
    identity: object
    decisions: int


class _SentinelPolicy:
    """Probe-owned recurrence sentinel that delegates action choice to random."""

    def __init__(self, seed: int):
        self._random = RandomPolicy(seed)

    def init_state(self) -> _SentinelState:
        return _SentinelState(object(), 0)

    def act(self, public: PublicDecision, state: object | None) -> Choice:
        if not isinstance(state, _SentinelState):
            raise ContractError("vector probe recurrence state was not initialized")
        choice = self._random.act(public, state)
        return Choice(choice.action, _SentinelState(state.identity, state.decisions + 1))


class _ProbeSetupFailure(Exception):
    pass


class _WriterFailure(Exception):
    def __init__(self, original: BaseException):
        self.original = original
        super().__init__(str(original))


def _writer_call(writer: RunWriter, method: str, *args: object) -> None:
    try:
        getattr(writer, method)(*args)
    except BaseException as error:
        raise _WriterFailure(error) from error


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _snapshot_one(raw: object, schema: object, *, check: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise _VectorContractViolation(check, "an observation row must be a public mapping")
    try:
        return native_snapshot(raw, schema)  # type: ignore[arg-type]
    except ContractError as error:
        raise _VectorContractViolation(check, str(error)) from error


def _snapshot_batch(raw: object, schema: object, *, width: int, check: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(raw, Mapping):
        raise _VectorContractViolation(check, "batched observations must be a public mapping")
    specs = {tensor.key: tensor for tensor in schema.tensors}  # type: ignore[union-attr]
    if set(raw) != set(specs):
        raise _VectorContractViolation(check, "batched observation keys must exactly match the descriptor")
    for key, spec in specs.items():
        values = raw[key]
        shape = getattr(values, "shape", None)
        dtype = getattr(values, "dtype", None)
        if shape is None or dtype is None:
            raise _VectorContractViolation(check, f"batched tensor {key} lacks public shape/dtype metadata")
        validate_vector_batch_shape(shape, spec.shape, width=width)
        if getattr(dtype, "name", None) != spec.dtype or getattr(dtype, "isnative", None) is not True:
            raise _VectorContractViolation(check, f"batched tensor {key} dtype/endian disagrees with descriptor")
    rows: list[Mapping[str, object]] = []
    for index in range(width):
        row: dict[str, object] = {}
        for key, values in raw.items():
            if not isinstance(key, str):
                raise _VectorContractViolation(check, "observation keys must be strings")
            try:
                row[key] = values[index]  # type: ignore[index]
            except (IndexError, KeyError, TypeError) as error:
                raise _VectorContractViolation(check, f"observation batch is missing row {index}") from error
        rows.append(_snapshot_one(row, schema, check=check))
    return tuple(rows)


def _active_mask_and_candidates(
    observation: Mapping[str, object], schema: object, *, check: str, slot: int
) -> tuple[tuple[bool, ...], tuple[object, ...]]:
    try:
        mask_values = observation["action_mask"].values  # type: ignore[index,union-attr]
        if any(type(value) is not int or value not in (0, 1) for value in mask_values):
            raise ContractError("current action mask must contain integer zero/one values")
        mask = tuple(value == 1 for value in mask_values)
        candidates = candidates_for(observation, schema)  # type: ignore[arg-type]
    except (KeyError, AttributeError, ContractError) as error:
        raise _VectorContractViolation(check, f"slot {slot} has an invalid public action mask: {error}") from error
    if not any(mask):
        raise _VectorContractViolation(check, f"active slot {slot} has no legal action candidates")
    return mask, candidates


def _decode_outcome(value: object, schema: object) -> str:
    normalized = normalize_public_info({"outcome": value}, schema)  # type: ignore[arg-type]
    outcome = normalized.get("outcome")
    if type(outcome) is not str:
        raise _VectorContractViolation("old_final", "public outcome is absent from the descriptor vocabulary")
    return outcome


def _read_rewards(value: object, width: int) -> tuple[float, ...]:
    rewards = _sequence_values(value, width=width, label="rewards")
    result: list[float] = []
    for reward in rewards:
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise _VectorContractViolation("old_final", "rewards must be finite numeric primitives")
        result.append(float(reward))
    return tuple(result)


def _route_episode_id(value: object) -> str:
    if type(value) in (str, int):
        return str(value)
    raise _VectorContractViolation("old_final", "episode_id must be a public string or integer")


def _check_packed_actions(actions: object, width: int) -> dict[str, object]:
    dtype = getattr(actions, "dtype", None)
    shape = getattr(actions, "shape", None)
    if (
        dtype is None
        or getattr(dtype, "name", None) != "int64"
        or getattr(dtype, "isnative", None) is not True
        or shape is None
        or tuple(shape) != (width,)
    ):
        raise _ProbeSetupFailure("pack_actions must produce native-endian int64[B]")
    return {"dtype": "int64", "native_endian": True, "shape": [width]}


def _exception_failure(error: BaseException, *, phase: str) -> Failure:
    classified = native_failure(error)
    if classified is not None:
        return classified
    kind = "setup_error" if phase in {"construction", "descriptor", "build_info", "reset"} else "operational_error"
    return Failure(kind, str(error), phase, type(error).__name__)


def _note_prior_failures(error: BaseException, failures: Sequence[Failure]) -> None:
    if not failures:
        return
    details = "; ".join(
        f"{failure.kind}@{failure.phase}: {failure.reason}" for failure in failures
    )
    try:
        error.add_note("probe failures captured before fatal exception: " + details)
    except AttributeError:
        pass


def _attempt_close(
    close: Callable[[], object],
    *,
    primary: BaseException | None = None,
    failures: Sequence[Failure] = (),
) -> BaseException | None:
    """Close once, retaining ordinary failures and never swallowing process control."""
    close_failure = None
    try:
        close()
    except BaseException as error:
        if primary is not None:
            try:
                primary.add_note(
                    f"environment close also failed: {type(error).__name__}: {error}"
                )
            except AttributeError:
                pass
        elif not isinstance(error, Exception):
            _note_prior_failures(error, failures)
            raise
        close_failure = error
    if primary is not None and not isinstance(primary, Exception):
        _note_prior_failures(primary, failures)
        raise primary.with_traceback(primary.__traceback__)
    return close_failure


_ACTION_MAPPING_ID = "spire-learn-native-current-index-v1"
_RNG_001 = (
    "Project-owned reproducible randomness is not seed-for-seed or bit-for-bit "
    "original-game RNG parity; no original-runtime transfer is established."
)
_TIME_001 = (
    "The full-run Secret Portal elapsed-time assumption is retained; this combat "
    "vector probe does not validate it."
)


def run_vector_probe(
    config: ProbeConfig,
    writer: RunWriter,
    *,
    clock: Callable[[], float],
) -> ProbeReport:
    """Collect bounded public SAME_STEP vector-boundary evidence.

    The environment is constructed and closed here, but writer initialization
    and closure remain the caller's responsibility. Native dependencies are
    reached only when this function invokes the lazy vector factory.
    """
    if not isinstance(config, ProbeConfig):
        raise ContractError("config must be a ProbeConfig")
    budget = Budget(config.max_wall_seconds, clock)
    validate_probe_config(config)

    checks: dict[str, str] = {name: "blocked" for name in NATIVE_CHECKS}
    failures: list[Failure] = []
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "spire_lab": _distribution_version("spire-lab"),
        "gymnasium": _distribution_version("gymnasium"),
        "numpy": _distribution_version("numpy"),
    }
    env = None
    schema = None
    descriptor_values: dict[str, object] = {}
    build_info: Mapping[str, object] | None = None
    action_batch_contract: dict[str, object] | None = None
    attempted_batch_steps = 0
    attempted_slot_decisions = 0
    attempted_replacement_decisions = 0
    batch_steps = 0
    slot_decisions = 0
    replacement_decisions = 0
    stop_reason = "not_started"
    operational_failure = False
    writer_failure: BaseException | None = None
    writer_traceback = None
    process_control_failure: BaseException | None = None
    process_control_traceback = None
    close_failure: BaseException | None = None
    remaining_wall_seconds: float | None = None
    elapsed_wall_seconds: float | None = None
    phase = "construction"

    try:
        if budget.exhausted():
            stop_reason = "wall_budget_before_construction"
        else:
            phase = "construction"
            env = make_vector(config.num_envs, config.partition, config.workers)

            phase = "descriptor"
            descriptor = env.observation_spec()
            schema = parse_descriptor(descriptor, allowed_schema_ids=NATIVE_CONTRACT.schema_ids)
            if schema.content_id not in NATIVE_CONTRACT.content_ids:
                raise _ProbeSetupFailure(f"unsupported native content_id: {schema.content_id}")
            descriptor_values = {
                "schema_id": schema.schema_id,
                "content_id": schema.content_id,
                "fingerprint": schema.fingerprint,
            }
            versions.update({key: value for key, value in descriptor_values.items()})
            for execution in config.executions:
                if (
                    execution.expected_schema_id is not None
                    and execution.expected_schema_id != schema.schema_id
                ):
                    raise _ProbeSetupFailure(
                        f"{execution.execution_id} expected schema {execution.expected_schema_id}, "
                        f"observed {schema.schema_id}"
                    )
                if (
                    execution.expected_fingerprint is not None
                    and execution.expected_fingerprint != schema.fingerprint
                ):
                    raise _ProbeSetupFailure(
                        f"{execution.execution_id} expected fingerprint {execution.expected_fingerprint}, "
                        f"observed {schema.fingerprint}"
                    )
            checks["schema_expectations"] = "pass"

            action_space = env.single_action_space
            if getattr(action_space, "n", None) != schema.action_count or getattr(action_space, "start", None) != 0:
                raise _ProbeSetupFailure("vector action space disagrees with the public observation descriptor")

            phase = "build_info"
            build_info = public_build_info()
            if build_info is None:
                raise _ProbeSetupFailure("the already-loaded native binding did not provide build_info")
            checks["native_build"] = "pass"
            _writer_call(
                writer,
                "update_manifest",
                {
                    "event": "vector_probe_discovery",
                    "schema": descriptor_values,
                    "reward": {
                        "contract_id": NATIVE_CONTRACT.reward_contract_id,
                        "evidence": NATIVE_CONTRACT.reward_evidence,
                    },
                    "python_version": versions["python"],
                    "distribution_versions": {
                        name: versions[name] for name in ("spire_lab", "gymnasium", "numpy")
                    },
                    "native_build_info": build_info,
                },
            )

            if budget.exhausted():
                stop_reason = "wall_budget_before_reset"
            else:
                phase = "reset"
                initial_observation, initial_info = env.reset(
                    seed=[execution.environment_seed for execution in config.executions],
                    options={"scenario_ids": [execution.scenario_id for execution in config.executions]},
                )
                current_rows = _snapshot_batch(
                    initial_observation,
                    schema,
                    width=config.num_envs,
                    check="native_contract",
                )
                current_info = validate_current_info_presence(initial_info, width=config.num_envs)
                current_ids = current_info["episode_id"]
                current_scenarios = current_info["scenario_id"]
                current_steps = current_info["episode_steps"]
                for index, execution in enumerate(config.executions):
                    if current_scenarios[index] != execution.scenario_id or current_steps[index] != 0:
                        raise _ProbeSetupFailure(
                            f"initial full reset did not start the declared scenario in slot {index}"
                        )
                    if _decode_outcome(current_info["outcome"][index], schema) != "ongoing":
                        raise _ProbeSetupFailure(f"initial full reset did not return an ongoing slot {index}")
                    _active_mask_and_candidates(
                        current_rows[index], schema, check="presence_masks", slot=index
                    )

                policies = tuple(_SentinelPolicy(execution.policy_seed) for execution in config.executions)
                states: list[object | None] = [policy.init_state() for policy in policies]
                generations = [0] * config.num_envs
                decision_counts = [0] * config.num_envs
                fresh_state_pending: list[_SentinelState | None] = [None] * config.num_envs
                ended_state_identity: list[object | None] = [None] * config.num_envs
                stop_reason = "batch_step_cap"

                while attempted_batch_steps < config.max_batch_steps:
                    if budget.exhausted():
                        stop_reason = "wall_budget_before_step"
                        break

                    routes: list[Route] = []
                    actions: list[int] = []
                    states_after_source: list[object | None] = []
                    source_states = list(states)
                    source_generations = list(generations)
                    fresh_source = [False] * config.num_envs
                    for index, execution in enumerate(config.executions):
                        if fresh_state_pending[index] is not None:
                            expected_state = fresh_state_pending[index]
                            if source_states[index] is not expected_state:
                                raise _VectorContractViolation(
                                    "old_context", f"replacement slot {index} did not receive its fresh state"
                                )
                            if source_states[index].identity is ended_state_identity[index]:
                                raise _VectorContractViolation(
                                    "old_context", f"replacement slot {index} reused its old episode state"
                                )
                            fresh_source[index] = True

                        mask, candidates = _active_mask_and_candidates(
                            current_rows[index], schema, check="old_final", slot=index
                        )
                        public = PublicDecision(
                            current_rows[index],
                            candidates,
                            mask,
                            schema.fingerprint,
                            _ACTION_MAPPING_ID,
                        )
                        route = Route(
                            execution.execution_id,
                            index,
                            _route_episode_id(current_ids[index]),
                            f"{execution.execution_id}:{index}:{decision_counts[index]}",
                            current_steps[index] == 0,
                        )
                        result = resolve_decision(
                            DecisionRequest(public, route), policies[index], source_states[index]
                        )
                        if type(result.choice.action) is not int:
                            raise _VectorContractViolation(
                                "presence_masks", f"policy returned a non-built-in action in slot {index}"
                            )
                        routes.append(route)
                        actions.append(result.choice.action)
                        states_after_source.append(result.choice.next_state)
                        decision_counts[index] += 1

                    packed_actions = pack_actions(actions)
                    action_batch_contract = _check_packed_actions(packed_actions, config.num_envs)
                    if budget.exhausted():
                        stop_reason = "wall_budget_before_step"
                        break

                    phase = "step"
                    attempted_batch_steps += 1
                    attempted_slot_decisions += config.num_envs
                    attempted_replacement_decisions += sum(
                        generation > 0 for generation in source_generations
                    )
                    (
                        next_observation,
                        raw_rewards,
                        raw_terminated,
                        raw_truncated,
                        step_info,
                    ) = env.step(packed_actions)
                    # A successfully returned batch is delivered even if its own
                    # execution crossed the soft wall-clock budget.
                    batch_steps += 1
                    slot_decisions += config.num_envs
                    replacement_delivered = sum(generation > 0 for generation in source_generations)
                    replacement_decisions += replacement_delivered

                    terminated = _sequence_values(
                        raw_terminated, width=config.num_envs, label="terminated", boolean=True
                    )
                    truncated = _sequence_values(
                        raw_truncated, width=config.num_envs, label="truncated", boolean=True
                    )
                    ended = tuple(term or trunc for term, trunc in zip(terminated, truncated, strict=True))
                    rewards = _read_rewards(raw_rewards, config.num_envs)
                    final_observations, final_info = validate_final_batch_presence(step_info, ended)
                    next_info = validate_current_info_presence(step_info, width=config.num_envs)
                    next_ids = next_info["episode_id"]
                    next_scenarios = next_info["scenario_id"]
                    next_steps = next_info["episode_steps"]
                    main_rows = _snapshot_batch(
                        next_observation,
                        schema,
                        width=config.num_envs,
                        check="native_contract",
                    )
                    for index, did_end in enumerate(ended):
                        _active_mask_and_candidates(
                            main_rows[index],
                            schema,
                            check="replacement_reset" if did_end else "old_final",
                            slot=index,
                        )
                    final_rows: list[Mapping[str, object] | None] = [None] * config.num_envs
                    if final_observations is not None:
                        for index, did_end in enumerate(ended):
                            if did_end:
                                final_rows[index] = _snapshot_one(
                                    final_observations[index], schema, check="old_final"
                                )

                    if any(ended):
                        if final_info is None:
                            raise _VectorContractViolation(
                                "presence_masks", "ended rows lack nested final_info"
                            )
                        validate_episode_identity(
                            source_ids=current_ids,
                            source_steps=current_steps,
                            current_ids=next_ids,
                            current_steps=next_steps,
                            ended=ended,
                            final_info=final_info,
                        )
                    else:
                        validate_episode_identity(
                            source_ids=current_ids,
                            source_steps=current_steps,
                            current_ids=next_ids,
                            current_steps=next_steps,
                            ended=ended,
                            final_info=final_info,
                        )

                    for index, execution in enumerate(config.executions):
                        expected_scenario = execution.scenario_id
                        if next_scenarios[index] != expected_scenario:
                            check = "replacement_reset" if ended[index] else "old_final"
                            raise _VectorContractViolation(
                                check, f"slot {index} changed its declared public scenario_id"
                            )
                        if _decode_outcome(next_info["outcome"][index], schema) != "ongoing":
                            check = "replacement_reset" if ended[index] else "old_final"
                            raise _VectorContractViolation(
                                check, f"slot {index} current/main row is not an ongoing episode"
                            )
                        if ended[index]:
                            if final_info is None or final_rows[index] is None:
                                raise _VectorContractViolation(
                                    "presence_masks", f"ended slot {index} lacks final data"
                                )
                            if final_info["scenario_id"][index] != expected_scenario:
                                raise _VectorContractViolation(
                                    "old_final", f"old final scenario does not match slot {index}"
                                )
                            _decode_outcome(final_info["outcome"][index], schema)

                    boundary_results = route_vector_step(
                        routes=routes,
                        sources=current_rows,
                        actions=actions,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        main_observations=main_rows,
                        final_observations=final_rows,
                        final_presence=ended,
                        final_info_presence=ended,
                        states_after_source=states_after_source,
                        reset_state=lambda index: policies[index].init_state(),
                    )
                    next_states: list[object | None] = []
                    for index, (boundary, did_end) in enumerate(zip(boundary_results, ended, strict=True)):
                        expected_successor = final_rows[index] if did_end else main_rows[index]
                        if boundary.transition.successor != expected_successor:
                            raise _VectorContractViolation(
                                "old_final", f"routed successor does not preserve slot {index} delivered data"
                            )
                        expected_replacement = main_rows[index] if did_end else None
                        if boundary.transition.replacement != expected_replacement:
                            raise _VectorContractViolation(
                                "replacement_reset", f"routed replacement does not match slot {index} main row"
                            )
                        if boundary.transition.state_after_source is not states_after_source[index]:
                            raise _VectorContractViolation(
                                "old_context", f"transition did not retain slot {index} source state"
                            )
                        if did_end:
                            active_state = boundary.active_state
                            old_state = states_after_source[index]
                            if (
                                not isinstance(active_state, _SentinelState)
                                or active_state is old_state
                                or active_state.identity is getattr(old_state, "identity", None)
                            ):
                                raise _VectorContractViolation(
                                    "old_context", f"ended slot {index} did not receive fresh active state"
                                )
                        elif boundary.active_state is not states_after_source[index]:
                            raise _VectorContractViolation(
                                "old_context", f"continuing slot {index} did not retain its source state"
                            )
                        next_states.append(boundary.active_state)

                    if any(ended) and not all(ended):
                        checks["mixed_rows"] = "pass"
                    if any(ended):
                        checks["presence_masks"] = "pass"
                        checks["old_final"] = "pass"
                        checks["replacement_reset"] = "pass"
                    if any(fresh_source) and replacement_delivered > 0:
                        checks["old_context"] = "pass"

                    for index, (did_end, boundary) in enumerate(
                        zip(ended, boundary_results, strict=True)
                    ):
                        if did_end:
                            generations[index] += 1
                            fresh_state_pending[index] = boundary.active_state  # type: ignore[assignment]
                            ended_state_identity[index] = getattr(
                                states_after_source[index], "identity", None
                            )
                        elif fresh_source[index]:
                            fresh_state_pending[index] = None
                            ended_state_identity[index] = None

                    for boundary in boundary_results:
                        _writer_call(writer, "write_transition", boundary.transition)
                    current_rows = main_rows
                    current_info = next_info
                    current_ids = next_ids
                    current_scenarios = next_scenarios
                    current_steps = next_steps
                    states = next_states

                    if all(checks.get(name) == "pass" for name in NATIVE_CHECKS) and replacement_decisions > 0:
                        stop_reason = "required_evidence_complete"
                        break
                else:
                    stop_reason = "batch_step_cap"
    except _WriterFailure as error:
        writer_failure = error.original
        writer_traceback = error.original.__traceback__
        stop_reason = "writer_io_error"
    except _VectorContractViolation as error:
        checks[error.check] = "fail"
        failures.append(Failure("contract_violation", str(error), phase, type(error).__name__))
        if error.check not in NATIVE_CHECKS:
            operational_failure = True
            checks["operational"] = "fail"
        stop_reason = "contract_violation"
    except _ProbeSetupFailure as error:
        operational_failure = True
        checks["operational"] = "fail"
        checks["setup"] = "fail"
        failures.append(Failure("setup_error", str(error), phase, type(error).__name__))
        stop_reason = "setup_error"
    except Exception as error:
        failure = _exception_failure(error, phase=phase)
        failures.append(failure)
        checks["operational"] = "fail"
        operational_failure = True
        if failure.kind in {"invalid_action", "environment_state", "setup_error", "simulator_fault"}:
            checks["native_failure"] = "fail"
        stop_reason = "native_failure" if failure.kind == "simulator_fault" else "operational_error"
    except BaseException as error:
        process_control_failure = error
        process_control_traceback = error.__traceback__
    finally:
        if env is not None:
            primary_failure = (
                process_control_failure
                if process_control_failure is not None
                else writer_failure
            )
            close_failure = _attempt_close(
                env.close,
                primary=primary_failure,
                failures=failures,
            )
            if close_failure is not None:
                operational_failure = True
                checks["operational"] = "fail"
                failures.append(
                    Failure("operational_error", str(close_failure), "close", type(close_failure).__name__)
                )

    if process_control_failure is not None:
        raise process_control_failure.with_traceback(process_control_traceback)

    if writer_failure is not None:
        _note_prior_failures(writer_failure, failures)
        raise writer_failure.with_traceback(writer_traceback)

    try:
        elapsed_wall_seconds = budget.elapsed()
        remaining_wall_seconds = max(
            0.0, float(config.max_wall_seconds) - elapsed_wall_seconds
        )
    except Exception as error:
        operational_failure = True
        checks["operational"] = "fail"
        failures.append(Failure("operational_error", str(error), "budget", type(error).__name__))
        if stop_reason == "not_started":
            stop_reason = "budget_error"

    checks.setdefault("operational", "pass")
    verdict = vector_probe_verdict(
        checks,
        operational_failure=operational_failure,
        runtime_available=env is not None,
    )
    report = ProbeReport(
        verdict=verdict,
        checks=checks,
        versions=versions,
        batch_steps=batch_steps,
        slot_decisions=slot_decisions,
        replacement_decisions=replacement_decisions,
        failures=tuple(failures),
    )
    encoded_report = probe_report_to_json(report)
    summary = {
        "schema_id": "spire-learn-vector-probe-summary-v1",
        "report": encoded_report,
        "stop_reason": stop_reason,
        "budgets": {
            "max_batch_steps": config.max_batch_steps,
            "max_wall_seconds": float(config.max_wall_seconds),
            "remaining_wall_seconds": remaining_wall_seconds,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "overshoot_wall_seconds": (
                None
                if elapsed_wall_seconds is None
                else max(0.0, elapsed_wall_seconds - float(config.max_wall_seconds))
            ),
        },
        "attempted": {
            "batch_steps": attempted_batch_steps,
            "slot_decisions": attempted_slot_decisions,
            "replacement_decisions": attempted_replacement_decisions,
        },
        "delivered": {
            "batch_steps": batch_steps,
            "slot_decisions": slot_decisions,
            "replacement_decisions": replacement_decisions,
        },
        "schema": descriptor_values,
        "native_build_info": build_info,
        "action_batch_contract": action_batch_contract,
        "reward": {
            "contract_id": NATIVE_CONTRACT.reward_contract_id,
            "evidence": NATIVE_CONTRACT.reward_evidence,
        },
        "rng_001": _RNG_001,
        "time_001": _TIME_001,
    }
    try:
        _writer_call(
            writer,
            "update_manifest",
            {
                "event": "vector_probe_finished",
                "stop_reason": stop_reason,
                "attempted": summary["attempted"],
                "delivered": summary["delivered"],
            },
        )
        _writer_call(writer, "write_metric", encoded_report)
        _writer_call(writer, "finalize", summary)
    except _WriterFailure as error:
        _note_prior_failures(error.original, failures)
        raise error.original.with_traceback(error.original.__traceback__)
    return report
