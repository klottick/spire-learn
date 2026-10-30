"""Lazy public adapters for the accepted Spire Gym combat interfaces.

The driver owns validation, snapshots and episode state. Optional Gymnasium,
NumPy and ``spire_lab`` imports are confined to :func:`make_driver` and the
native provider methods created there.
"""

from __future__ import annotations

from spire_learn.neural.profiling import observed, span

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import struct
from typing import Protocol

from spire_learn.artifacts.codec import to_json_value
from spire_learn.contracts import ContractError
from spire_learn.neural.cases import NeuralCases
from spire_learn.neural.config import (
    ACCEPTED_GYM_REVISION,
    COMMON_VIEW_FINGERPRINT,
    COMMON_VIEW_ID,
    RUN_OBSERVATION_FINGERPRINT,
    RUN_PROFILE_ID,
    NeuralConfig,
)
from spire_learn.neural.contracts import CombatFrame, Lifecycle
from spire_learn.neural.schema import CombatCandidate, CombatSchema, candidates
from spire_learn.neural.schema import parse_common_descriptor, parse_lifecycle, validate_frame
from spire_learn.neural.snapshot import snapshot_common


_MAX_ERROR_TEXT = 512
_MAX_FAULT_PREFIX = 1_000_000


class DriverInputError(ValueError):
    """A malformed driver call rejected before it reaches Spire Gym."""


class DriverStateError(RuntimeError):
    """A call is incompatible with the driver's current lifecycle."""


@dataclass(frozen=True)
class DriverFaultEvidence:
    """Bounded public delivery evidence retained when a driver freezes.

    ``accepted_prefix`` counts fully delivered driver step calls;
    ``accepted_prefix_operations`` also counts successful reset calls. The
    current operation reports submitted slot indices, known delivery and
    uncertain slots separately. Fault summaries retain slot IDs only.
    """

    phase: str
    accepted_prefix: int
    failed_operation: int
    delivery: str
    exception_type: str
    message: str
    accepted_prefix_operations: int = 0
    accepted_prefix_slot_actions: int = 0
    operation_slots: int = 0
    operation_slot_indices: tuple[int, ...] = ()
    operation_slot_indices_truncated: bool = False
    known_current_delivery_slots: int = 0
    uncertain_current_slots: int = 0
    fault_slots: tuple[int, ...] = ()
    fault_slots_truncated: bool = False
    close_error: str | None = None


class DriverFault(RuntimeError):
    """An accepted operation or its public export failed; the driver is frozen."""

    def __init__(self, evidence: DriverFaultEvidence):
        self.evidence = evidence
        super().__init__(
            f"driver frozen during {evidence.phase}; accepted prefix "
            f"{evidence.accepted_prefix} step calls / "
            f"{evidence.accepted_prefix_slot_actions} slot actions; current operation "
            f"{evidence.known_current_delivery_slots} slots delivered, "
            f"{evidence.uncertain_current_slots} uncertain of {evidence.operation_slots}; "
            f"delivery {evidence.delivery}: "
            f"{evidence.message}"
        )


@dataclass(frozen=True)
class DriverRow:
    """One owned common-view row and its separately held public lifecycle."""

    frame: CombatFrame
    lifecycle: Lifecycle
    candidates: tuple[CombatCandidate, ...]
    noncombat_indices: tuple[int, ...]


@dataclass(frozen=True)
class DriverTransition:
    """One successfully delivered native action with owned source/successor."""

    source: DriverRow
    successor: DriverRow
    native_reward: float
    terminated: bool
    truncated: bool

    @property
    def lifecycle(self) -> Lifecycle:
        """The immediate public lifecycle returned by the accepted operation."""
        return self.successor.lifecycle


@dataclass(frozen=True)
class _RawBoundary:
    observation: Mapping[str, object]
    info: Mapping[str, object]
    noncombat_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RawTransition:
    boundary: _RawBoundary
    reward: float
    terminated: bool
    truncated: bool


class _PostReturnFailure(Exception):
    """Internal marker for errors raised after a public native call returned."""

    def __init__(self, error: BaseException):
        self.error = error
        super().__init__("public result export failed after native return")


class _Provider(Protocol):
    mode: str
    batch_size: int
    input_errors: tuple[type[Exception], ...]
    state_errors: tuple[type[Exception], ...]

    def reset(
        self,
        seeds: tuple[int, ...],
        entry_ids: tuple[str | None, ...],
        reset_mask: tuple[bool, ...],
    ) -> tuple[_RawBoundary | None, ...]: ...

    def step(self, actions: tuple[int, ...]) -> tuple[_RawTransition, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _NativeRuntime:
    numpy: object
    native: object
    ironclad_vector_env: type
    run_env: type


def _load_native_runtime() -> _NativeRuntime:
    """Load the pinned optional runtime only after an explicit factory call."""
    import numpy
    from spire_lab import _native
    from spire_lab.gymnasium import IroncladCombatVectorEnv, RunEnv

    return _NativeRuntime(numpy, _native, IroncladCombatVectorEnv, RunEnv)


def _error_types(native: object, names: tuple[str, ...]) -> tuple[type[Exception], ...]:
    result = tuple(getattr(native, name, None) for name in names)
    if any(not isinstance(item, type) or not issubclass(item, Exception) for item in result):
        raise ContractError("installed Spire Gym lacks the accepted public error types")
    return result  # type: ignore[return-value]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed public mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError(f"{name} must be a sequence")
    return value


def _row_observation(observation: object, slot: int, *, batched: bool) -> Mapping[str, object]:
    values = _mapping(observation, "observation")
    if not batched:
        return values
    row: dict[str, object] = {}
    try:
        for key, value in values.items():
            row[key] = value[slot]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as error:
        raise ContractError("vector observation does not contain the requested public row") from error
    return row


def _public_scalar(value: object, name: str) -> object:
    """Convert NumPy scalar wrappers without accepting arrays or coercive strings."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if type(value) not in (bool, int, float):
        raise ContractError(f"{name} must be a public numeric scalar")
    return value


def _info_slots(info: object, batch_size: int) -> tuple[object, ...]:
    values = _mapping(info, "vector info")
    rows = _sequence(values.get("slots"), "vector info slots")
    if len(rows) != batch_size:
        raise ContractError("vector info slots do not match the configured batch")
    return tuple(rows)


def _legal_indices(mask: object, width: int) -> tuple[int, ...]:
    shape = getattr(mask, "shape", None)
    dtype = getattr(mask, "dtype", None)
    flags = getattr(mask, "flags", None)
    if shape != (width,) or getattr(dtype, "name", None) != "int8":
        raise ContractError("Run legal-action mask does not match its public descriptor")
    if getattr(dtype, "isnative", None) is not True:
        raise ContractError("Run legal-action mask must use native-endian int8")
    if flags is None or getattr(flags, "c_contiguous", None) is not True:
        raise ContractError("Run legal-action mask must be C-contiguous")
    result: list[int] = []
    try:
        values = tuple(mask)  # type: ignore[arg-type]
    except TypeError as error:
        raise ContractError("Run legal-action mask is not iterable") from error
    if len(values) != width:
        raise ContractError("Run legal-action mask has an invalid length")
    for index, raw in enumerate(values):
        value = _public_scalar(raw, "Run legal-action mask value")
        if type(value) is not int or value not in (0, 1):
            raise ContractError("Run legal-action mask values must be int8 zeroes or ones")
        if value:
            result.append(index)
    return tuple(result)


class _StandaloneProvider:
    mode = "standalone"

    def __init__(self, env: object, runtime: _NativeRuntime, batch_size: int):
        self._env = env
        self._numpy = runtime.numpy
        self.batch_size = batch_size
        self.input_errors = _error_types(
            runtime.native, ("CombatInputError", "RunInputError")
        )
        self.state_errors = _error_types(
            runtime.native, ("CombatStateError", "RunStateError")
        )

    def reset(
        self,
        seeds: tuple[int, ...],
        entry_ids: tuple[str | None, ...],
        reset_mask: tuple[bool, ...],
    ) -> tuple[_RawBoundary | None, ...]:
        np = self._numpy
        mask = np.asarray(reset_mask, dtype=np.dtype("bool"))  # type: ignore[attr-defined]
        with span("native_reset_public_export"):
            observations, info = self._env.reset(  # type: ignore[attr-defined]
                seed=list(seeds),
                options={"entry_ids": list(entry_ids), "reset_mask": mask},
            )
        try:
            info_rows = _info_slots(info, self.batch_size)
            rows: list[_RawBoundary | None] = []
            for slot in range(self.batch_size):
                rows.append(_RawBoundary(
                    _row_observation(observations, slot, batched=True),
                    _mapping(info_rows[slot], "vector slot info"),
                ))
            return tuple(rows)
        except BaseException as error:
            raise _PostReturnFailure(error) from error

    def step(self, actions: tuple[int, ...]) -> tuple[_RawTransition, ...]:
        np = self._numpy
        native_actions = np.asarray(actions, dtype=np.dtype("=i8"))  # type: ignore[attr-defined]
        with span("native_step_public_export"):
            observations, rewards, terminated, truncated, info = self._env.step(  # type: ignore[attr-defined]
                native_actions
            )
        try:
            info_rows = _info_slots(info, self.batch_size)
            result: list[_RawTransition] = []
            for slot in range(self.batch_size):
                reward = _public_scalar(rewards[slot], "native reward")
                ended = _public_scalar(terminated[slot], "terminated flag")
                cut = _public_scalar(truncated[slot], "truncated flag")
                if type(reward) not in (int, float) or not math.isfinite(reward):
                    raise ContractError("native reward must be finite")
                if type(ended) is not bool or type(cut) is not bool:
                    raise ContractError("native end flags must be boolean scalars")
                result.append(_RawTransition(
                    _RawBoundary(
                        _row_observation(observations, slot, batched=True),
                        _mapping(info_rows[slot], "vector slot info"),
                    ),
                    float(reward),
                    ended,
                    cut,
                ))
            return tuple(result)
        except BaseException as error:
            raise _PostReturnFailure(error) from error

    def close(self) -> None:
        self._env.close()  # type: ignore[attr-defined]


class _RunProvider:
    mode = "run"
    batch_size = 1

    def __init__(self, env: object, runtime: _NativeRuntime, run_action_count: int):
        self._env = env
        self._run_action_count = run_action_count
        self.dispatch_action_count = run_action_count
        self.input_errors = _error_types(runtime.native, ("RunInputError",))
        self.state_errors = _error_types(runtime.native, ("RunStateError",))

    def _capture(
        self,
        view: object,
        run_observation: object,
    ) -> _RawBoundary:
        sidecar = _mapping(view, "Run combat_view")
        observation = _mapping(sidecar.get("obs"), "Run combat_view obs")
        info = _mapping(sidecar.get("info"), "Run combat_view info")
        lifecycle = parse_lifecycle(info)
        if lifecycle.active:
            noncombat_indices: tuple[int, ...] = ()
        else:
            run_obs = _mapping(run_observation, "Run public observation")
            mask = run_obs.get("action_mask")
            if mask is None:
                raise ContractError("Run observation is missing its public action mask")
            noncombat_indices = _legal_indices(mask, self._run_action_count)
        return _RawBoundary(observation, info, noncombat_indices)

    def reset(
        self,
        seeds: tuple[int, ...],
        entry_ids: tuple[str | None, ...],
        reset_mask: tuple[bool, ...],
    ) -> tuple[_RawBoundary | None, ...]:
        if len(seeds) != 1 or len(entry_ids) != 1 or len(reset_mask) != 1 or reset_mask != (True,):
            raise DriverInputError("Run reset requires one selected row")
        with span("native_reset_public_export"):
            result = self._env.reset(seed=seeds[0])  # type: ignore[attr-defined]
        try:
            # The common sidecar and its token are valid only for this decision.
            with span("run_combat_public_export"):
                view = self._env.combat_view()  # type: ignore[attr-defined]
            run_observation, _info = result
            return (self._capture(view, run_observation),)
        except BaseException as error:
            raise _PostReturnFailure(error) from error

    def step(self, actions: tuple[int, ...]) -> tuple[_RawTransition, ...]:
        with span("native_step_public_export"):
            result = self._env.step(actions[0])  # type: ignore[attr-defined]
        try:
            # Read the sidecar before inspecting or dispatching another decision.
            with span("run_combat_public_export"):
                view = self._env.combat_view()  # type: ignore[attr-defined]
            run_observation, reward, terminated, truncated, _info = result
            ended = _public_scalar(terminated, "Run terminated flag")
            cut = _public_scalar(truncated, "Run truncated flag")
            numeric_reward = _public_scalar(reward, "Run native reward")
            if type(numeric_reward) not in (int, float) or not math.isfinite(numeric_reward):
                raise ContractError("Run native reward must be finite")
            if type(ended) is not bool or type(cut) is not bool:
                raise ContractError("Run end flags must be booleans")
            return (_RawTransition(
                self._capture(view, run_observation),
                float(numeric_reward),
                ended,
                cut,
            ),)
        except BaseException as error:
            raise _PostReturnFailure(error) from error

    def close(self) -> None:
        self._env.close()  # type: ignore[attr-defined]


def _clean_error(error: BaseException) -> str:
    value = str(error).replace("\x00", " ").replace("\n", " ").replace("\r", " ")
    if len(value) > _MAX_ERROR_TEXT:
        value = value[:_MAX_ERROR_TEXT] + "…"
    return value


@observed("snapshot_validation")
def _row_from_raw(
    raw: _RawBoundary,
    schema: CombatSchema,
    *,
    dispatch_action_count: int | None = None,
) -> DriverRow:
    frame = snapshot_common(raw.observation, schema)
    lifecycle = parse_lifecycle(raw.info)
    validate_frame(frame, schema)
    if lifecycle.active:
        if raw.noncombat_indices:
            raise ContractError("active Run row cannot expose fixed-controller indices")
        legal = candidates(frame, schema)
        noncombat: tuple[int, ...] = ()
    else:
        if any(frame.tensors["action_mask"].data):
            raise ContractError("inactive common view must expose an empty action mask")
        legal = ()
        width = schema.action_space.n if dispatch_action_count is None else dispatch_action_count
        noncombat = _validate_noncombat_indices(raw.noncombat_indices, width)
    return DriverRow(frame, lifecycle, legal, noncombat)


def _validate_noncombat_indices(values: Sequence[int], action_count: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ContractError("fixed-controller indices must be a sequence")
    result = tuple(values)
    if any(type(value) is not int or not 0 <= value < action_count for value in result):
        raise ContractError("fixed-controller indices must be legal in-range built-in integers")
    if tuple(sorted(set(result))) != result:
        raise ContractError("fixed-controller indices must be unique and increasing")
    return result


class CombatDriver:
    """Validated adapter over standalone combat or the single normal-Run facade."""

    def __init__(self, provider: _Provider, schema: CombatSchema):
        self._provider = provider
        self.schema = schema
        self.mode = provider.mode
        self.batch_size = provider.batch_size
        self._dispatch_action_count = getattr(provider, "dispatch_action_count", schema.action_space.n)
        if self.mode not in {"standalone", "run"}:
            raise ContractError("driver provider mode must be standalone or run")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ContractError("driver provider batch_size must be positive")
        if type(self._dispatch_action_count) is not int or self._dispatch_action_count < 1:
            raise ContractError("driver provider dispatch action count must be positive")
        if self.mode == "run" and self.batch_size != 1:
            raise ContractError("Run driver requires exactly one native row")
        self._rows: tuple[DriverRow | None, ...] = (None,) * self.batch_size
        self._terminal: tuple[bool, ...] = (False,) * self.batch_size
        self._accepted_steps = 0
        self._accepted_operations = 0
        self._failed = False
        self._closed = False

    @property
    def rows(self) -> tuple[DriverRow | None, ...]:
        """Current immutable rows; an unreset row remains ``None``."""
        return self._rows

    @property
    def accepted_steps(self) -> int:
        """Number of whole driver step calls whose public delivery succeeded."""
        return self._accepted_steps

    def _ensure_usable(self) -> None:
        if self._closed:
            raise DriverStateError("driver is closed")
        if self._failed:
            raise DriverStateError("driver is faulted; accepted actions cannot be retried")

    def _is_recoverable(self, error: BaseException) -> bool:
        if isinstance(error, self._provider.input_errors):
            return True
        if isinstance(error, self._provider.state_errors):
            reason = getattr(error, "reason", None)
            return reason == "stale_token" or "stale" in str(error).lower()
        return False

    def _close_after_failure(self) -> BaseException | None:
        if self._closed:
            return None
        self._closed = True
        try:
            self._provider.close()
        except BaseException as error:
            return error
        return None

    @staticmethod
    def _failure_context(
        error: BaseException,
        *,
        phase: str,
        delivery: str,
        operation_slots: int,
        known_current_delivery_slots: int,
        uncertain_current_slots: int,
    ) -> str:
        return (
            f"primary failure {type(error).__name__}: {_clean_error(error)}; "
            f"phase={phase}; delivery={delivery}; operation_slots={operation_slots}; "
            f"known_current_delivery_slots={known_current_delivery_slots}; "
            f"uncertain_current_slots={uncertain_current_slots}"
        )

    def _fault_slots(
        self,
        error: BaseException,
        operation_slot_indices: tuple[int, ...],
    ) -> tuple[tuple[int, ...], bool]:
        """Keep only a small set of in-range slot indices from public fault records."""
        faults = getattr(error, "faults", None)
        if not isinstance(faults, Sequence) or isinstance(faults, (str, bytes, bytearray)):
            return (), False
        slots: set[int] = set()
        truncated = len(faults) > 64
        for record in faults[:64]:
            if not isinstance(record, Mapping):
                continue
            slot = record.get("slot")
            if type(slot) is int and slot in operation_slot_indices:
                slots.add(slot)
        return tuple(sorted(slots)), truncated

    def _fault(
        self,
        error: BaseException,
        *,
        phase: str,
        delivery: str,
        operation_slots: int = 0,
        operation_slot_indices: Sequence[int] = (),
        known_current_delivery_slots: int = 0,
        uncertain_current_slots: int = 0,
    ) -> DriverFault:
        self._failed = True
        close_failure = self._close_after_failure()
        known_current_delivery_slots = min(
            max(known_current_delivery_slots, 0), operation_slots
        )
        uncertain_current_slots = min(max(uncertain_current_slots, 0), operation_slots)
        close_error: str | None = None
        if close_failure is not None:
            if not isinstance(close_failure, Exception):
                context = self._failure_context(
                    error,
                    phase=phase,
                    delivery=delivery,
                    operation_slots=operation_slots,
                    known_current_delivery_slots=known_current_delivery_slots,
                    uncertain_current_slots=uncertain_current_slots,
                )
                add_note = getattr(close_failure, "add_note", None)
                if callable(add_note):
                    add_note(
                        "provider close process-control failure; propagating close control "
                        f"with the primary failure context: {context}"
                    )
                raise close_failure from error
            close_error = f"{type(close_failure).__name__}: {_clean_error(close_failure)}"
        operation_slots = min(max(operation_slots, 0), _MAX_FAULT_PREFIX)
        public_operation_slots = tuple(
            slot for slot in operation_slot_indices
            if type(slot) is int and 0 <= slot < self.batch_size
        )
        public_operation_slots = tuple(sorted(set(public_operation_slots)))
        if not public_operation_slots and operation_slots == self.batch_size:
            public_operation_slots = tuple(range(min(self.batch_size, 64)))
        operation_slots_truncated = len(public_operation_slots) > 64
        public_operation_slots = public_operation_slots[:64]
        known_current_delivery_slots = min(known_current_delivery_slots, operation_slots)
        uncertain_current_slots = min(uncertain_current_slots, operation_slots)
        fault_slots, fault_slots_truncated = self._fault_slots(error, public_operation_slots)
        evidence = DriverFaultEvidence(
            phase=phase,
            accepted_prefix=min(self._accepted_steps, _MAX_FAULT_PREFIX),
            failed_operation=min(self._accepted_operations + 1, _MAX_FAULT_PREFIX),
            delivery=delivery,
            exception_type=type(error).__name__,
            message=_clean_error(error),
            accepted_prefix_operations=min(self._accepted_operations, _MAX_FAULT_PREFIX),
            accepted_prefix_slot_actions=min(
                self._accepted_steps * self.batch_size, _MAX_FAULT_PREFIX
            ),
            operation_slots=operation_slots,
            operation_slot_indices=public_operation_slots,
            operation_slot_indices_truncated=operation_slots_truncated,
            known_current_delivery_slots=known_current_delivery_slots,
            uncertain_current_slots=uncertain_current_slots,
            fault_slots=fault_slots,
            fault_slots_truncated=fault_slots_truncated,
            close_error=close_error,
        )
        return DriverFault(evidence)

    def _finish_failure(
        self,
        error: BaseException,
        *,
        phase: str,
        delivery: str,
        operation_slots: int,
        operation_slot_indices: Sequence[int] = (),
        known_current_delivery_slots: int | None = None,
    ) -> None:
        post_return = isinstance(error, _PostReturnFailure)
        if post_return:
            error = error.error
            phase = f"{phase}-export"
            delivery = "returned"
            known_current_delivery_slots = operation_slots
        if not post_return and self._is_recoverable(error):
            raise error
        if not isinstance(error, Exception):
            self._failed = True
            close_failure = self._close_after_failure()
            if close_failure is not None:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    context = self._failure_context(
                        error,
                        phase=phase,
                        delivery=delivery,
                        operation_slots=operation_slots,
                        known_current_delivery_slots=(
                            operation_slots
                            if delivery == "returned" and known_current_delivery_slots is None
                            else (known_current_delivery_slots or 0)
                        ),
                        uncertain_current_slots=(operation_slots if delivery == "unknown" else 0),
                    )
                    add_note(
                        f"provider close after primary process-control failure also raised "
                        f"{type(close_failure).__name__}: {_clean_error(close_failure)}; "
                        f"primary exception remains primary. {context}"
                    )
            raise error
        raise self._fault(
            error,
            phase=phase,
            delivery=delivery,
            operation_slots=operation_slots,
            operation_slot_indices=operation_slot_indices,
            known_current_delivery_slots=(
                operation_slots
                if delivery == "returned" and known_current_delivery_slots is None
                else (known_current_delivery_slots or 0)
            ),
            uncertain_current_slots=(operation_slots if delivery == "unknown" else 0),
        ) from error

    def _validate_reset(
        self,
        seeds: Sequence[int],
        entry_ids: Sequence[str | None],
        reset_mask: Sequence[bool] | None,
    ) -> tuple[tuple[int, ...], tuple[str | None, ...], tuple[bool, ...]]:
        seed_values = _sequence(seeds, "seeds")
        entry_values = _sequence(entry_ids, "entry_ids")
        if len(seed_values) != self.batch_size or len(entry_values) != self.batch_size:
            raise DriverInputError("seeds and entry_ids must match the native batch width")
        if any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seed_values):
            raise DriverInputError("each seed must be a built-in unsigned 64-bit integer")
        if any(entry is not None and (type(entry) is not str or not entry)
               for entry in entry_values):
            raise DriverInputError("each entry_id must be a non-empty string or None")
        if self.mode == "run" and tuple(entry_values) != (None,):
            raise DriverInputError("Run driver cannot receive catalog entry IDs")
        if reset_mask is None:
            mask_values = (True,) * self.batch_size
        else:
            raw_mask = _sequence(reset_mask, "reset_mask")
            if len(raw_mask) != self.batch_size or any(type(value) is not bool for value in raw_mask):
                raise DriverInputError("reset_mask must contain one built-in bool per row")
            mask_values = tuple(raw_mask)  # type: ignore[assignment]
        if not any(mask_values):
            raise DriverInputError("reset_mask must select at least one row")
        if self.mode == "run" and mask_values != (True,):
            raise DriverInputError("single-row Run resets must select their row")
        if any(not selected and self._rows[slot] is None
               for slot, selected in enumerate(mask_values)):
            raise DriverStateError("an unselected row must already have a delivered state")
        return tuple(seed_values), tuple(entry_values), tuple(mask_values)

    def reset(
        self,
        *,
        seeds: Sequence[int],
        entry_ids: Sequence[str | None],
        reset_mask: Sequence[bool] | None = None,
    ) -> tuple[DriverRow, ...]:
        """Explicitly reset selected rows; construction never resets implicitly."""
        self._ensure_usable()
        seed_values, entry_values, mask_values = self._validate_reset(
            seeds, entry_ids, reset_mask
        )
        try:
            raw_rows = self._provider.reset(seed_values, entry_values, mask_values)
        except BaseException as error:
            self._finish_failure(
                error,
                phase="reset",
                delivery="unknown",
                operation_slots=sum(mask_values),
                operation_slot_indices=tuple(
                    slot for slot, selected in enumerate(mask_values) if selected
                ),
            )
        try:
            reset_row_count = len(raw_rows)
        except BaseException as error:
            self._finish_failure(
                error,
                phase="reset-export",
                delivery="returned",
                operation_slots=sum(mask_values),
                operation_slot_indices=tuple(
                    slot for slot, selected in enumerate(mask_values) if selected
                ),
            )
        if reset_row_count != self.batch_size:
            error = ContractError("provider reset returned the wrong number of rows")
            raise self._fault(
                error,
                phase="reset-export",
                delivery="returned",
                operation_slots=sum(mask_values),
                operation_slot_indices=tuple(
                    slot for slot, selected in enumerate(mask_values) if selected
                ),
                known_current_delivery_slots=sum(mask_values),
            ) from error
        updated = list(self._rows)
        terminal = list(self._terminal)
        try:
            for slot, selected in enumerate(mask_values):
                if selected:
                    raw = raw_rows[slot]
                    if raw is None:
                        raise ContractError("provider omitted a selected reset row")
                    updated[slot] = _row_from_raw(
                        raw, self.schema, dispatch_action_count=self._dispatch_action_count
                    )
                    terminal[slot] = False
                else:
                    raw = raw_rows[slot]
                    if raw is None:
                        if self.mode == "standalone":
                            raise ContractError(
                                "provider omitted the public unselected reset boundary"
                            )
                        continue
                    observed = _row_from_raw(
                        raw, self.schema, dispatch_action_count=self._dispatch_action_count
                    )
                    current = self._rows[slot]
                    if current is None:
                        raise ContractError("unselected row has no cached public boundary")
                    if observed.frame != current.frame:
                        raise ContractError("unselected public frame changed during masked reset")
                    if observed.lifecycle != current.lifecycle:
                        raise ContractError("unselected public lifecycle changed during masked reset")
                    updated[slot] = current
        except BaseException as error:
            self._finish_failure(
                error,
                phase="reset-export",
                delivery="returned",
                operation_slots=sum(mask_values),
                operation_slot_indices=tuple(
                    slot for slot, selected in enumerate(mask_values) if selected
                ),
            )
        self._rows = tuple(updated)
        self._terminal = tuple(terminal)
        self._accepted_operations += 1
        return tuple(row for row in self._rows if row is not None)

    def _validate_actions(self, actions: Sequence[int]) -> tuple[int, ...]:
        values = _sequence(actions, "actions")
        if len(values) != self.batch_size:
            raise DriverInputError("actions must provide one built-in int per native row")
        if any(type(action) is not int or action < 0 for action in values):
            raise DriverInputError("actions must be non-negative built-in integers")
        if any(self._rows[slot] is None for slot in range(self.batch_size)):
            raise DriverStateError("reset every row before stepping")
        if any(self._terminal):
            raise DriverStateError("a terminal row rejects the entire native batch before dispatch")
        for slot, action in enumerate(values):
            row = self._rows[slot]
            assert row is not None
            legal_indices = (
                tuple(candidate.index for candidate in row.candidates)
                if row.lifecycle.active else row.noncombat_indices
            )
            action_limit = (
                self.schema.action_space.n
                if row.lifecycle.active else self._dispatch_action_count
            )
            if action >= action_limit:
                raise DriverInputError(f"action for row {slot} is outside its public dispatch space")
            if action not in legal_indices:
                raise DriverInputError(f"action for row {slot} is not in its current public legal set")
        return tuple(values)

    def step(self, actions: Sequence[int]) -> tuple[DriverTransition, ...]:
        """Dispatch one full legal batch and return owned source/successor rows."""
        self._ensure_usable()
        action_values = self._validate_actions(actions)
        sources = tuple(self._rows)
        try:
            raw_steps = self._provider.step(action_values)
        except BaseException as error:
            self._finish_failure(
                error,
                phase="step",
                delivery="unknown",
                operation_slots=self.batch_size,
                operation_slot_indices=tuple(range(self.batch_size)),
            )
        try:
            step_row_count = len(raw_steps)
        except BaseException as error:
            self._finish_failure(
                error,
                phase="step-export",
                delivery="returned",
                operation_slots=self.batch_size,
                operation_slot_indices=tuple(range(self.batch_size)),
            )
        if step_row_count != self.batch_size:
            error = ContractError("provider step returned the wrong number of rows")
            raise self._fault(
                error,
                phase="step-export",
                delivery="returned",
                operation_slots=self.batch_size,
                operation_slot_indices=tuple(range(self.batch_size)),
                known_current_delivery_slots=self.batch_size,
            ) from error
        transitions: list[DriverTransition] = []
        successors: list[DriverRow] = []
        terminal: list[bool] = []
        try:
            for slot, raw in enumerate(raw_steps):
                source = sources[slot]
                if source is None:
                    raise ContractError("step source disappeared before dispatch")
                if type(raw.terminated) is not bool or type(raw.truncated) is not bool:
                    raise ContractError("provider end flags must be built-in booleans")
                if raw.terminated and raw.truncated:
                    raise ContractError("provider cannot terminate and truncate one transition")
                reward = raw.reward
                if type(reward) not in (int, float) or not math.isfinite(reward):
                    raise ContractError("provider reward must be finite")
                successor = _row_from_raw(
                    raw.boundary,
                    self.schema,
                    dispatch_action_count=self._dispatch_action_count,
                )
                transitions.append(DriverTransition(
                    source=source,
                    successor=successor,
                    native_reward=float(reward),
                    terminated=raw.terminated,
                    truncated=raw.truncated,
                ))
                successors.append(successor)
                terminal.append(raw.terminated or raw.truncated)
        except BaseException as error:
            self._finish_failure(
                error,
                phase="step-export",
                delivery="returned",
                operation_slots=self.batch_size,
                operation_slot_indices=tuple(range(self.batch_size)),
            )
        self._accepted_steps += 1
        self._accepted_operations += 1
        self._rows = tuple(successors)
        self._terminal = tuple(terminal)
        return tuple(transitions)

    def close(self) -> None:
        """Close once; a close failure freezes the adapter and is never retried."""
        if self._closed:
            return
        self._closed = True
        try:
            self._provider.close()
        except BaseException as error:
            self._failed = True
            if not isinstance(error, Exception):
                raise
            evidence = DriverFaultEvidence(
                phase="close",
                accepted_prefix=min(self._accepted_steps, _MAX_FAULT_PREFIX),
                failed_operation=min(self._accepted_operations + 1, _MAX_FAULT_PREFIX),
                delivery="closed-with-error",
                exception_type=type(error).__name__,
                message=_clean_error(error),
                accepted_prefix_operations=min(self._accepted_operations, _MAX_FAULT_PREFIX),
                accepted_prefix_slot_actions=min(
                    self._accepted_steps * self.batch_size, _MAX_FAULT_PREFIX
                ),
            )
            self._failed = True
            raise DriverFault(evidence) from error


def _require_environment_identity(
    spec: object,
    *,
    expected: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    values = _mapping(spec, name)
    for key, value in expected.items():
        if values.get(key) != value:
            raise ContractError(f"{name} {key} does not match the accepted profile")
    return values


def _normalized_entry(entry: Mapping[str, object]) -> dict[str, object]:
    value = to_json_value(entry)
    if not isinstance(value, dict):
        raise ContractError("standalone case entry must thaw to a JSON object")
    result = dict(value)
    deck = result.get("deck")
    relics = result.get("relics")
    if not isinstance(deck, list) or not isinstance(relics, list):
        raise ContractError("standalone case entry has invalid public deck or relics")
    result["deck"] = sorted(deck, key=lambda row: (row["card"], row["upgrades"]))
    result["relics"] = sorted(relics)
    counters = dict(result.get("relic_counters", {}))
    if "neows_lament" in result["relics"]:
        counters.setdefault("neows_lament", 3)
    result["relic_counters"] = counters
    return result


def _verify_catalog(catalog_descriptor: object, entries: Sequence[Mapping[str, object]]) -> None:
    descriptor = _mapping(catalog_descriptor, "public catalog descriptor")
    if set(descriptor) != {"catalog_schema_id", "entries"}:
        raise ContractError("public catalog descriptor has unexpected fields")
    if descriptor.get("catalog_schema_id") != "spire-gym-combat-entry-catalog-v1":
        raise ContractError("public catalog descriptor has an unsupported schema")
    actual = descriptor.get("entries")
    if not isinstance(actual, list):
        raise ContractError("public catalog descriptor entries must be a list")
    expected = [_normalized_entry(entry) for entry in entries]
    if actual != expected:
        raise ContractError("Gym normalized catalog does not match the supplied public cases")


def _run_action_count(descriptor: object) -> int:
    values = _mapping(descriptor, "Run observation descriptor")
    action_space = _mapping(values.get("action_space"), "Run action space")
    width = action_space.get("n")
    if type(width) is not int or width < 1:
        raise ContractError("Run descriptor must publish a finite discrete action space")
    tensors = _sequence(values.get("tensors"), "Run descriptor tensors")
    masks = [
        _mapping(tensor, "Run descriptor tensor")
        for tensor in tensors
        if isinstance(tensor, Mapping) and tensor.get("key") == "action_mask"
    ]
    if len(masks) != 1 or masks[0].get("dtype") != "int8" or masks[0].get("shape") != [width]:
        raise ContractError("Run action mask does not match the descriptor action space")
    return width


def _validate_factory_inputs(config: NeuralConfig, cases: NeuralCases) -> None:
    if not isinstance(config, NeuralConfig) or not isinstance(cases, NeuralCases):
        raise TypeError("make_driver requires NeuralConfig and NeuralCases")
    if config.environment_revision != ACCEPTED_GYM_REVISION:
        raise ContractError("NeuralConfig does not pin the accepted Gym revision")
    if config.common_view_id != COMMON_VIEW_ID or config.common_view_fingerprint != COMMON_VIEW_FINGERPRINT:
        raise ContractError("NeuralConfig does not pin the accepted common combat view")
    if (cases.sha256 != config.cases.sha256 or cases.path != config.cases.path
            or cases.split != config.split or cases.partition != config.partition):
        raise ContractError("make_driver cases do not match the authenticated config manifest")
    if cases.partition not in {"train", "selection", "final"}:
        raise ContractError("unsupported neural case partition")
    if config.mode not in {"standalone", "run"}:
        raise ContractError("unsupported neural driver mode")
    if not cases.for_mode(config.mode):
        raise ContractError(f"case manifest has no {config.mode} cases")


def _verify_native_build(native: object, expected_revision: str) -> None:
    build_info = getattr(native, "build_info", None)
    if not callable(build_info):
        raise ContractError("installed Spire Gym lacks public build_info metadata")
    identity = _mapping(build_info(), "Spire Gym build_info")
    if (identity.get("extension") != "spire-python"
            or identity.get("revision") != expected_revision):
        raise ContractError("loaded Spire Gym extension does not match the accepted revision")


def _close_after_factory_error(env: object, error: BaseException) -> None:
    try:
        env.close()  # type: ignore[attr-defined]
    except BaseException as close_error:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(f"Gym close after factory failure also raised {type(close_error).__name__}")


def make_driver(config: NeuralConfig, cases: NeuralCases) -> CombatDriver:
    """Construct an authenticated public Gym adapter without resetting or stepping.

    Standalone mode creates one manually resettable public vector facade. Run
    mode creates exactly one normal ``m10-ironclad-policy`` Run facade. The
    Gym descriptor, environment identity and finite catalog are checked before
    the driver is returned.
    """
    _validate_factory_inputs(config, cases)
    runtime = _load_native_runtime()
    _verify_native_build(runtime.native, config.environment_revision)
    if config.mode == "standalone":
        mode_cases = cases.for_mode("standalone")
        entries = tuple(case.entry for case in mode_cases)
        if any(not isinstance(entry, Mapping) for entry in entries):
            raise ContractError("standalone cases must provide native public entries")
        catalog = {"entries": [to_json_value(entry) for entry in entries]}
        env = runtime.ironclad_vector_env(
            config.rollout_rows,
            catalog,
            partition=config.partition,
        )
        try:
            schema = parse_common_descriptor(env.observation_spec())
            if schema.schema_fingerprint != config.common_view_fingerprint:
                raise ContractError("Gym common descriptor fingerprint disagrees with config")
            _require_environment_identity(
                env.environment_spec(),
                expected={
                    "profile_id": "ironclad-combat-v1",
                    "schema_id": COMMON_VIEW_ID,
                    "observation_fingerprint": config.common_view_fingerprint,
                    "content_id": "spire-gym-ironclad-combat-content-v1",
                    "rules_id": "spire-gym-native-m10-rules-v1",
                    "reward_id": "spire-gym-combat-outcome-reward-v1",
                    "mode": "standalone_combat",
                    "partition": config.partition,
                    "autoreset": "disabled",
                },
                name="standalone environment identity",
            )
            _verify_catalog(env.catalog_descriptor(), entries)  # type: ignore[attr-defined]
            provider = _StandaloneProvider(env, runtime, config.rollout_rows)
            return CombatDriver(provider, schema)
        except BaseException as error:
            _close_after_factory_error(env, error)
            raise

    mode_cases = cases.for_mode("run")
    if len(mode_cases) != 1 or config.rollout_rows != 1:
        raise ContractError("Run driver requires exactly one case and one native row")
    run_case = mode_cases[0]
    if (run_case.profile_id != RUN_PROFILE_ID or run_case.character != "ironclad"
            or run_case.ascension != 0):
        raise ContractError("Run case must be Ironclad A0 m10-ironclad-policy")
    if config.profile_id != RUN_PROFILE_ID or config.run_observation_fingerprint != RUN_OBSERVATION_FINGERPRINT:
        raise ContractError("Run config does not pin the accepted v3 Run observation")
    env = runtime.run_env(
        "ironclad",
        0,
        profile=RUN_PROFILE_ID,
        partition=config.partition,
        record_replay=False,
    )
    try:
        run_descriptor = _mapping(env.observation_spec(), "Run observation descriptor")
        if (run_descriptor.get("schema_id") != "spire-gym-run-observation-v3"
                or run_descriptor.get("profile_id") != "M10IroncladPolicy"):
            raise ContractError("Gym Run descriptor is not the accepted policy profile")
        embedded_fingerprint = run_descriptor.get("fingerprint")
        if embedded_fingerprint is not None and embedded_fingerprint != config.run_observation_fingerprint:
            raise ContractError("Gym Run descriptor fingerprint disagrees with config")
        _require_environment_identity(
            env.environment_spec(),
            expected={
                "profile_id": "M10IroncladPolicy",
                "schema_id": "spire-gym-run-observation-v3",
                "observation_fingerprint": config.run_observation_fingerprint,
                "rules_id": "spire-gym-native-m10-rules-v1",
                "reward_id": "spire-gym-floor-key-terminal-reward-v1",
                "combat_view_id": "spire-gym-combat-policy-view-v1",
                "mode": "normal_run",
                "partition": config.partition,
                "autoreset": "disabled",
            },
            name="Run environment identity",
        )
        schema = parse_common_descriptor(env.combat_observation_spec())
        if schema.schema_fingerprint != config.common_view_fingerprint:
            raise ContractError("Run common descriptor fingerprint disagrees with config")
        provider = _RunProvider(env, runtime, _run_action_count(run_descriptor))
        return CombatDriver(provider, schema)
    except BaseException as error:
        _close_after_factory_error(env, error)
        raise


__all__ = [
    "CombatDriver",
    "DriverFault",
    "DriverFaultEvidence",
    "DriverInputError",
    "DriverRow",
    "DriverStateError",
    "DriverTransition",
    "make_driver",
]
