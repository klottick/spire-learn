"""One fixed, disposable CUDA throughput cohort for the accepted neural learner.

The module is dependency-free when imported.  Torch, Gym, the native extension,
and model factories are reached only by :func:`calibrate`, after the strict
calibration config and approved GPU UUID have passed the preflight boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from typing import Any

from spire_learn.neural.collector import Collector, CollectorRecovery
from spire_learn.neural.config import (
    COMMON_VIEW_FINGERPRINT,
    COMMON_VIEW_ID,
    RUN_OBSERVATION_FINGERPRINT,
    RUN_OBSERVATION_VERSION,
    RUN_PROFILE_ID,
    RUN_REPLAY_VERSION,
    NeuralConfig,
    SeedStreams,
)
from spire_learn.neural.runner import NeuralReport


GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
GPU_NAME = "NVIDIA GeForce RTX 5090"
GPU_DRIVER = "595.91.07"
TORCH_VERSION = "2.14.1+cu130"
TORCH_CUDA_VERSION_PREFIX = "13.0"
NUMPY_VERSION = "2.5.4"
GYMNASIUM_VERSION = "1.4.0"
SPIRE_LAB_VERSION = "0.1.0"
CUDA_LOCK_SHA256 = "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45"
NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
ALLOCATOR_CAP_BYTES = 8 * 1024**3
MIN_DEVICE_FREE_BYTES = 12 * 1024**3
MAX_COHORT_SECONDS = 300.0
MAX_ACTION_SLOTS = 1024
ABSOLUTE_ACTION_RESERVATION_CAP = 2048
MAX_SAMPLE_ROWS = 1024
MAX_OPTIMIZER_STEPS = 3
EXPECTED_PARAMETER_COUNT = 265_122
_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
_MEMORY_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class TrialSpec:
    """One immutable workload in the approved trial order."""

    trial_id: str
    mode: str
    batch_size: int
    horizon: int
    action_slots: int
    seed_ordinal: int
    optimizer_step: bool

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id:
            raise ValueError("trial_id must be a non-empty string")
        if self.mode not in {"standalone", "run"}:
            raise ValueError("trial mode must be standalone or run")
        for name in ("batch_size", "horizon", "action_slots", "seed_ordinal"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative primitive integer")
        if self.batch_size < 1 or self.horizon < 1 or self.action_slots != self.batch_size * self.horizon:
            raise ValueError("trial action slots must equal batch size times horizon")
        if type(self.optimizer_step) is not bool:
            raise TypeError("optimizer_step must be a boolean")
        if self.optimizer_step != (self.mode == "standalone"):
            raise ValueError("only standalone trials declare one disposable optimizer step")


TRIALS = (
    TrialSpec("standalone-b4-h32", "standalone", 4, 32, 128, 0, True),
    TrialSpec("standalone-b8-h32", "standalone", 8, 32, 256, 1024, True),
    TrialSpec("standalone-b16-h32", "standalone", 16, 32, 512, 2048, True),
    TrialSpec("run-b1-h128", "run", 1, 128, 128, 3072, False),
)


class CalibrationPreflightError(RuntimeError):
    """The exact approved CUDA runtime or device was not available."""


def _validate_manifest_release_binding(
    config: NeuralConfig,
    release_binding: Mapping[str, object],
) -> dict[str, str]:
    """Bind the reviewed raw manifest bytes while retaining loader identities."""
    manifest_release_sha256 = release_binding.get("train_manifest_sha256")
    config_file_sha256 = release_binding.get("calibration_config_file_sha256")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in (manifest_release_sha256, config_file_sha256)):
        raise CalibrationPreflightError("parent-reviewed manifest/config file identity is malformed")
    if (not isinstance(config.manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", config.manifest_sha256) is None
            or not isinstance(config.config_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", config.config_sha256) is None):
        raise CalibrationPreflightError("loaded manifest/config canonical identity is malformed")

    digest = hashlib.sha256()
    try:
        if not isinstance(config.manifest_path, Path) or not config.manifest_path.is_file():
            raise OSError("manifest path is not a readable regular file")
        with config.manifest_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CalibrationPreflightError("training manifest file cannot be read for release binding") from error

    manifest_file_sha256 = digest.hexdigest()
    if manifest_release_sha256 != manifest_file_sha256:
        raise CalibrationPreflightError("parent-reviewed manifest file binding differs")
    return {
        "train_manifest_file_sha256": manifest_file_sha256,
        "train_manifest_canonical_sha256": config.manifest_sha256,
        "calibration_config_file_sha256": config_file_sha256,
        "calibration_config_canonical_sha256": config.config_sha256,
    }


class CalibrationStop(RuntimeError):
    """A named time, device-memory, or cumulative calibration cap was reached."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Reservation:
    reservation_id: str
    amount: int


class _CalibrationBudget:
    """One monotonic cohort clock shared by setup, trials, probes, and output."""

    def __init__(self, clock: Callable[[], float], duration: float = MAX_COHORT_SECONDS):
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._start = self._read()
        self._duration = float(duration)
        self._memory_guard: Callable[[], object] | None = None

    def _read(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("calibration clock must return a finite primitive number")
        return float(value)

    def bind_memory_guard(self, guard: Callable[[], object]) -> None:
        if not callable(guard):
            raise TypeError("memory guard must be callable")
        self._memory_guard = guard

    def elapsed(self) -> float:
        return max(0.0, self._read() - self._start)

    def remaining(self) -> float:
        if self._memory_guard is not None:
            self._memory_guard()
        return max(0.0, self._duration - self.elapsed())

    def check(self, phase: str) -> None:
        if self.remaining() <= 0:
            raise CalibrationStop(f"wall_budget_before_{phase}")

    @property
    def allowed_seconds(self) -> float:
        return self._duration


class _CalibrationLedger:
    """Irreversible, cohort-local reservations and delivery accounting."""

    _COLLECTION_FIELDS = (
        "attempted_environment_slots", "forwarded_environment_slots",
        "known_accepted_environment_slots", "delivered_environment_slots",
        "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
        "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
        "reset_operations", "reset_slots", "learned_decisions", "fixed_decisions",
        "undispatched_actor_decisions", "undispatched_fixed_decisions",
        "tasks_started", "task_terminals", "tasks_completed", "victories", "deaths", "escaped",
        "faults", "incomplete_decisions", "incomplete_environment_slots",
    )

    def __init__(self) -> None:
        self._counts: dict[str, int] = {
            "reserved_action_slots": 0,
            "reserved_sample_rows": 0,
            "attempted_sample_rows": 0,
            "delivered_sample_rows": 0,
            "uncertain_sample_rows": 0,
            "optimizer_reserved_steps": 0,
            "optimizer_attempted_steps": 0,
            "optimizer_accepted_steps": 0,
            "optimizer_uncertain_steps": 0,
        }
        self._trial_reservations: set[str] = set()
        self._optimizer_states: dict[str, str] = {}
        self._sample_states: dict[str, str] = {}

    def reserve_actions(self, amount: int, *, name: str) -> _Reservation:
        if type(amount) is not int or amount <= 0:
            raise ValueError("action reservation must be a positive primitive integer")
        if not isinstance(name, str) or not name or name in self._trial_reservations:
            raise ValueError("action reservation name must be new and non-empty")
        total = self._counts["reserved_action_slots"] + amount
        if total > ABSOLUTE_ACTION_RESERVATION_CAP or total > MAX_ACTION_SLOTS:
            raise ValueError("cohort action-slot reservation exceeds 1024")
        self._trial_reservations.add(name)
        self._counts["reserved_action_slots"] = total
        return _Reservation(name, amount)

    def reserve_samples(self, amount: int, *, name: str) -> _Reservation:
        if type(amount) is not int or amount <= 0:
            raise ValueError("sample reservation must be a positive primitive integer")
        if not isinstance(name, str) or not name or name in self._sample_states:
            raise ValueError("sample reservation name must be new and non-empty")
        total = self._counts["reserved_sample_rows"] + amount
        if total > MAX_SAMPLE_ROWS:
            raise ValueError("cohort learned-sample reservation exceeds 1024")
        self._counts["reserved_sample_rows"] = total
        self._sample_states[name] = "reserved"
        return _Reservation(name, amount)

    def mark_sample_attempted(self, reservation: _Reservation) -> None:
        if self._sample_states.get(reservation.reservation_id) != "reserved":
            raise ValueError("sample reservation is not awaiting an attempt")
        self._sample_states[reservation.reservation_id] = "attempted"
        self._counts["attempted_sample_rows"] += reservation.amount

    def mark_sample_delivered(self, reservation: _Reservation) -> None:
        if self._sample_states.get(reservation.reservation_id) != "attempted":
            raise ValueError("sample reservation has no attempted call")
        self._sample_states[reservation.reservation_id] = "delivered"
        self._counts["delivered_sample_rows"] += reservation.amount

    def mark_sample_uncertain(self, reservation: _Reservation) -> None:
        if self._sample_states.get(reservation.reservation_id) != "attempted":
            raise ValueError("sample reservation has no attempted call")
        self._sample_states[reservation.reservation_id] = "uncertain"
        self._counts["uncertain_sample_rows"] += reservation.amount

    def reserve_optimizer_call(self, *, trial_id: str) -> _Reservation:
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("optimizer trial id must be non-empty")
        key = f"optimizer:{trial_id}"
        if key in self._optimizer_states:
            raise ValueError("optimizer call is already reserved for this trial")
        total = self._counts["optimizer_reserved_steps"] + 1
        if total > MAX_OPTIMIZER_STEPS:
            raise ValueError("cohort optimizer-step reservation exceeds three")
        self._counts["optimizer_reserved_steps"] = total
        self._optimizer_states[key] = "reserved"
        return _Reservation(key, 1)

    def mark_optimizer_attempted(self, reservation: _Reservation) -> None:
        if self._optimizer_states.get(reservation.reservation_id) != "reserved":
            raise ValueError("optimizer reservation is not awaiting an attempt")
        self._optimizer_states[reservation.reservation_id] = "attempted"
        self._counts["optimizer_attempted_steps"] += 1

    def mark_optimizer_accepted(self, reservation: _Reservation) -> None:
        if self._optimizer_states.get(reservation.reservation_id) != "attempted":
            raise ValueError("optimizer reservation has no attempted call")
        self._optimizer_states[reservation.reservation_id] = "accepted"
        self._counts["optimizer_accepted_steps"] += 1

    def mark_optimizer_uncertain(self, reservation: _Reservation) -> None:
        if self._optimizer_states.get(reservation.reservation_id) != "attempted":
            raise ValueError("optimizer reservation has no attempted call")
        self._optimizer_states[reservation.reservation_id] = "uncertain"
        self._counts["optimizer_uncertain_steps"] += 1

    def record_collection(self, diagnostics: object) -> None:
        if diagnostics is None:
            return
        for field in self._COLLECTION_FIELDS:
            value = getattr(diagnostics, field, None)
            if type(value) is not int or value < 0:
                raise ValueError(f"collector diagnostic {field} must be a non-negative integer")
            self._counts[field] = self._counts.get(field, 0) + value

    def snapshot(self) -> dict[str, int]:
        result = dict(self._counts)
        result["undispatched_sample_rows"] = (
            result["reserved_sample_rows"] - result["attempted_sample_rows"]
        )
        result["unclassified_attempted_sample_rows"] = (
            result["attempted_sample_rows"]
            - result["delivered_sample_rows"]
            - result["uncertain_sample_rows"]
        )
        return dict(sorted(result.items()))


@dataclass
class _TrialTiming:
    provider_reset_seconds: float = 0.0
    provider_step_seconds: float = 0.0
    driver_reset_seconds: float = 0.0
    driver_step_seconds: float = 0.0
    policy_act_seconds: float = 0.0
    policy_values_seconds: float = 0.0
    cpu_collation_seconds: float = 0.0
    cpu_model_copy_seconds: float = 0.0
    cpu_forward_seconds: float = 0.0
    cpu_backward_seconds: float = 0.0
    h2d_seconds: float = 0.0
    gpu_forward_seconds: float = 0.0
    gpu_backward_seconds: float = 0.0
    gpu_adam_seconds: float = 0.0
    close_seconds: float = 0.0
    _pending_instrumentation_error: BaseException | None = field(default=None, repr=False)

    @property
    def cpu_forward_backward_seconds(self) -> float:
        return self.cpu_forward_seconds + self.cpu_backward_seconds

    @property
    def gpu_forward_backward_seconds(self) -> float:
        return self.gpu_forward_seconds + self.gpu_backward_seconds

    def defer_instrumentation_error(self, error: BaseException, label: str) -> None:
        self._pending_instrumentation_error = _prefer_exception(
            self._pending_instrumentation_error, error, label,
        )

    def raise_pending_instrumentation_error(self) -> None:
        error = self._pending_instrumentation_error
        self._pending_instrumentation_error = None
        if error is not None:
            raise error

    def projection(self) -> dict[str, float]:
        return {
            "public_native_provider_dispatch_export_seconds":
                self.provider_reset_seconds + self.provider_step_seconds,
            "learner_snapshot_validation_residual_seconds": max(
                0.0,
                self.driver_reset_seconds + self.driver_step_seconds
                - self.provider_reset_seconds - self.provider_step_seconds,
            ),
            "outer_driver_reset_step_seconds": self.driver_reset_seconds + self.driver_step_seconds,
            "production_policy_act_inclusive_encoding_h2d_forward_sampling_seconds": self.policy_act_seconds,
            "production_policy_value_seconds": self.policy_values_seconds,
            "cpu_collation_seconds": self.cpu_collation_seconds,
            "cpu_model_copy_seconds": self.cpu_model_copy_seconds,
            "cpu_forward_seconds": self.cpu_forward_seconds,
            "cpu_backward_seconds": self.cpu_backward_seconds,
            "cpu_forward_backward_seconds": self.cpu_forward_backward_seconds,
            "h2d_seconds": self.h2d_seconds,
            "gpu_forward_seconds": self.gpu_forward_seconds,
            "gpu_backward_seconds": self.gpu_backward_seconds,
            "gpu_forward_backward_seconds": self.gpu_forward_backward_seconds,
            "gpu_adam_seconds": self.gpu_adam_seconds,
            "runtime_close_seconds": self.close_seconds,
            "native_timing_qualification": (
                "provider dispatch plus public export; subtraction residual is learner snapshot/validation "
                "overhead, not pure Rust or pure snapshot time"
            ),
            "stored_frame_timing_qualification": (
                "CPU encode_frames(cpu) followed by explicit full EncodedBatch transfer; "
                "this cached-frame path is not production act latency"
            ),
        }


class _TimedProvider:
    """Time only public provider calls, delegating lifecycle and return values."""

    def __init__(self, inner: object, timing: _TrialTiming, timer: Callable[[], float]):
        self._inner = inner
        self._timing = timing
        self._timer = timer

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def _call(self, name: str, *args: object) -> object:
        self._timing.raise_pending_instrumentation_error()
        operation = getattr(self._inner, name)
        started = self._timer()
        try:
            result = operation(*args)
        except BaseException as operation_error:
            try:
                elapsed = _elapsed(self._timer, started)
            except BaseException as timing_error:
                raise _prefer_exception(operation_error, timing_error, f"provider {name} timing")
            field = "provider_reset_seconds" if name == "reset" else "provider_step_seconds"
            setattr(self._timing, field, getattr(self._timing, field) + elapsed)
            raise
        try:
            elapsed = _elapsed(self._timer, started)
        except BaseException as timing_error:
            self._timing.defer_instrumentation_error(timing_error, f"provider {name} timing")
            return result
        field = "provider_reset_seconds" if name == "reset" else "provider_step_seconds"
        setattr(self._timing, field, getattr(self._timing, field) + elapsed)
        return result

    def reset(self, *args: object) -> object:
        return self._call("reset", *args)

    def step(self, *args: object) -> object:
        return self._call("step", *args)


class _TimedDriver:
    """Measure the accepted driver boundary while preserving its public protocol."""

    def __init__(
        self,
        inner: object,
        timing: _TrialTiming,
        timer: Callable[[], float],
        budget: _CalibrationBudget,
    ) -> None:
        self._inner = inner
        self._timing = timing
        self._timer = timer
        self._budget = budget

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    @property
    def rows(self) -> object:
        return getattr(self._inner, "rows")

    @property
    def accepted_steps(self) -> object:
        return getattr(self._inner, "accepted_steps")

    def reset(self, **kwargs: object) -> object:
        self._timing.raise_pending_instrumentation_error()
        self._budget.check("native_reset")
        return _timed_adapter_call(
            self._inner, "reset", self._timing, self._timer,
            "driver_reset_seconds", kwargs=kwargs,
        )

    def step(self, actions: Sequence[int]) -> object:
        self._timing.raise_pending_instrumentation_error()
        self._budget.check("native_step")
        return _timed_adapter_call(
            self._inner, "step", self._timing, self._timer,
            "driver_step_seconds", args=(actions,),
        )

    def close(self) -> object:
        try:
            return getattr(self._inner, "close")()
        except BaseException as error:
            self._timing.defer_instrumentation_error(error, "timed driver close")
            raise


class _TimedPolicy:
    """Count sampled learned rows at the call site before invoking model sampling."""

    def __init__(
        self,
        inner: object,
        *,
        ledger: _CalibrationLedger,
        trial_id: str,
        timing: _TrialTiming,
        timer: Callable[[], float],
        budget: _CalibrationBudget,
        synchronize: Callable[[], object] | None,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._trial_id = trial_id
        self._timing = timing
        self._timer = timer
        self._budget = budget
        self._synchronize = synchronize

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def act(self, frames: Sequence[object], *, generator: object, deterministic: bool = False) -> object:
        self._timing.raise_pending_instrumentation_error()
        self._budget.check("policy_sampling")
        reservation = self._ledger.reserve_samples(
            len(frames),
            name=f"{self._trial_id}:sample:{self._ledger.snapshot()['reserved_sample_rows']}",
        )
        if self._synchronize is not None:
            self._synchronize()
        started = self._timer()
        operation = getattr(self._inner, "act")
        self._ledger.mark_sample_attempted(reservation)
        try:
            result = operation(
                frames, generator=generator, deterministic=deterministic,
            )
        except BaseException as operation_error:
            self._ledger.mark_sample_uncertain(reservation)
            primary: BaseException | None = operation_error
        else:
            self._ledger.mark_sample_delivered(reservation)
            primary = None

        if self._synchronize is not None:
            try:
                self._synchronize()
            except BaseException as sync_error:
                primary = _prefer_exception(primary, sync_error, "policy post-call synchronization")
        try:
            elapsed = _elapsed(self._timer, started)
        except BaseException as timing_error:
            primary = _prefer_exception(primary, timing_error, "policy act timing")
        else:
            self._timing.policy_act_seconds += elapsed
        if primary is not None and self._ledger._sample_states.get(reservation.reservation_id) == "delivered":
            self._timing.defer_instrumentation_error(primary, "policy act instrumentation")
            return result
        if primary is not None:
            raise primary
        if self._timing._pending_instrumentation_error is not None:
            self._timing.raise_pending_instrumentation_error()
        return result

    def values(self, frames: Sequence[object]) -> object:
        self._timing.raise_pending_instrumentation_error()
        self._budget.check("policy_value")
        if self._synchronize is not None:
            self._synchronize()
        started = self._timer()
        try:
            result = getattr(self._inner, "values")(frames)
            succeeded = True
        except BaseException as operation_error:
            primary: BaseException | None = operation_error
            result = None
            succeeded = False
        else:
            primary = None
        if self._synchronize is not None:
            try:
                self._synchronize()
            except BaseException as sync_error:
                primary = _prefer_exception(primary, sync_error, "policy values synchronization")
        try:
            elapsed = _elapsed(self._timer, started)
        except BaseException as timing_error:
            primary = _prefer_exception(primary, timing_error, "policy values timing")
        else:
            self._timing.policy_values_seconds += elapsed
        if primary is not None and succeeded:
            self._timing.defer_instrumentation_error(primary, "policy values instrumentation")
            return result
        if primary is not None:
            raise primary
        return result


@dataclass
class _TrialAssembly:
    """One fresh model/collector/driver and its idempotent close boundary."""

    config: NeuralConfig
    spec: TrialSpec
    driver: object
    policy: object
    collector: Collector
    timing: _TrialTiming
    timer: Callable[[], float]
    _closed: bool = False
    close_count: int = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_count += 1
        started: float | None = None
        primary: BaseException | None = None
        try:
            started = self.timer()
        except BaseException as timer_error:
            primary = timer_error
        try:
            self.collector.close()
        except BaseException as close_error:
            primary = _prefer_exception(primary, close_error, "collector close")
        if started is not None:
            try:
                self.timing.close_seconds += _elapsed(self.timer, started)
            except BaseException as timer_error:
                primary = _prefer_exception(primary, timer_error, "collector close timing")
        if primary is not None:
            raise primary


def _assemble_trial(
    config: NeuralConfig,
    spec: TrialSpec,
    recovery: CollectorRecovery,
    budget: _CalibrationBudget,
    ledger: _CalibrationLedger,
    *,
    driver_factory: Callable[[NeuralConfig, object], object],
    policy_factory: Callable[..., object],
    generator_factory: Callable[[NeuralConfig], object],
    timer: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], object] | None = None,
) -> _TrialAssembly:
    """Shared production/test assembly path; all environment work uses Collector."""
    timing = _TrialTiming()
    driver: object | None = None
    collector: Collector | None = None
    try:
        budget.check("driver_factory")
        driver = driver_factory(config, config.cases)
        provider = getattr(driver, "_provider", None)
        if provider is not None:
            if not callable(getattr(provider, "reset", None)) or not callable(getattr(provider, "step", None)):
                raise TypeError("accepted driver provider lacks public reset/step boundaries")
            setattr(driver, "_provider", _TimedProvider(provider, timing, timer))
        budget.check("model_factory")
        policy = policy_factory(
            getattr(driver, "schema"), config.model,
            seed=config.seed_streams.policy, device=config.device,
        )
        if getattr(policy, "_parameter_count", None) != EXPECTED_PARAMETER_COUNT:
            raise ValueError("calibration model parameter count differs from the accepted 265122 default")
        generator = generator_factory(config)
        timed_policy = _TimedPolicy(
            policy,
            ledger=ledger,
            trial_id=spec.trial_id,
            timing=timing,
            timer=timer,
            budget=budget,
            synchronize=synchronize,
        )
        timed_driver = _TimedDriver(driver, timing, timer, budget)
        collector = Collector(
            timed_driver,
            timed_policy,
            config,
            controller_rng=random.Random(config.seed_streams.noncombat),
            curriculum_rng=random.Random(config.seed_streams.curriculum),
            budget=budget,
            policy_generator=generator,
            recovery=recovery,
        )
        return _TrialAssembly(config, spec, driver, policy, collector, timing, timer)
    except BaseException as error:
        primary: BaseException = error
        try:
            if collector is not None:
                collector.close()
            elif driver is not None:
                close = getattr(driver, "close", None)
                if callable(close):
                    close()
        except BaseException as close_error:
            primary = _prefer_exception(primary, close_error, "trial assembly close")
        if primary is error:
            raise
        raise primary from error


def _add_note(error: BaseException, message: str) -> None:
    try:
        BaseException.add_note(error, message)
    except Exception:
        pass


def _error_value(error: BaseException) -> dict[str, str]:
    try:
        args = BaseException.args.__get__(error)
        safe_types = (str, int, float, bool, type(None))
        if not isinstance(args, tuple) or any(type(value) not in safe_types for value in args):
            message = "exception details unavailable"
        elif len(args) == 0:
            message = ""
        elif len(args) == 1:
            message = str(args[0])
        else:
            message = str(args)
        message = message.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    except Exception:
        message = "exception message unavailable"
    return {"type": type(error).__name__, "message": message[:512]}


def _is_process_control(error: BaseException) -> bool:
    return not isinstance(error, Exception)


def _prefer_exception(
    primary: BaseException | None,
    secondary: BaseException,
    label: str,
) -> BaseException:
    """Keep the first process-control error and attach every later failure."""
    if primary is None:
        return secondary
    if primary is secondary:
        return primary
    if _is_process_control(primary):
        _add_note(primary, f"{label} also failed: {_error_value(secondary)}")
        return primary
    if _is_process_control(secondary):
        _add_note(secondary, f"{label} superseded earlier error: {_error_value(primary)}")
        return secondary
    _add_note(primary, f"{label} also failed: {_error_value(secondary)}")
    return primary


def _elapsed(timer: Callable[[], float], started: float) -> float:
    if type(started) not in (int, float) or not math.isfinite(float(started)):
        raise ValueError("calibration timing start must be a finite primitive number")
    ended = timer()
    if type(ended) not in (int, float) or not math.isfinite(float(ended)):
        raise ValueError("calibration timing callback must return a finite primitive number")
    return max(0.0, float(ended) - float(started))


def _timed_adapter_call(
    inner: object,
    name: str,
    timing: _TrialTiming,
    timer: Callable[[], float],
    field_name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object] | None = None,
) -> object:
    """Measure a returned adapter effect, deferring post-call clock failures."""
    timing.raise_pending_instrumentation_error()
    operation = getattr(inner, name)
    started = timer()
    try:
        result = operation(*args, **({} if kwargs is None else dict(kwargs)))
    except BaseException as operation_error:
        try:
            elapsed = _elapsed(timer, started)
        except BaseException as timing_error:
            raise _prefer_exception(operation_error, timing_error, f"{name} timing")
        setattr(timing, field_name, getattr(timing, field_name) + elapsed)
        raise
    try:
        elapsed = _elapsed(timer, started)
    except BaseException as timing_error:
        timing.defer_instrumentation_error(timing_error, f"{name} timing")
        return result
    setattr(timing, field_name, getattr(timing, field_name) + elapsed)
    return result


def _close_preserving(primary: BaseException | None, resource: object, label: str):
    try:
        close = getattr(resource, "close", None)
    except BaseException as lookup_error:
        return _prefer_exception(primary, lookup_error, f"{label} close lookup"), lookup_error
    if not callable(close):
        return primary, None
    try:
        close()
    except BaseException as close_error:
        return _prefer_exception(primary, close_error, f"{label} close"), close_error
    return primary, None


def _validate_calibration_config(config: NeuralConfig) -> None:
    if not isinstance(config, NeuralConfig):
        raise TypeError("calibrate requires a NeuralConfig")
    if (config.operation != "calibrate" or config.mode != "standalone"
            or config.split != "train" or config.partition != "train"):
        raise ValueError("calibration requires operation=calibrate, standalone, and train partition")
    if config.device != "cuda:0":
        raise ValueError("calibration requires the qualified cuda:0 device")
    if config.calibration_gpu_uuid != GPU_UUID or not isinstance(config.calibration_gpu_uuid, str):
        raise ValueError("calibration requires the one qualified Task 8 GPU UUID")
    if _UUID_RE.fullmatch(config.calibration_gpu_uuid) is None:
        raise ValueError("calibration GPU UUID is malformed")
    if (config.rollout_rows, config.rollout_steps) != (4, 32):
        raise ValueError("calibration config must retain the fixed standalone B4/horizon32 base")
    if (config.model.embedding_width, config.model.hidden_width, config.model.max_parameters) != (
        16, 64, 10_000_000,
    ):
        raise ValueError("calibration requires ModelConfig(16, 64, 10000000)")
    budgets = config.budgets
    if (budgets.max_optimizer_steps != MAX_OPTIMIZER_STEPS
            or budgets.max_training_decisions != MAX_SAMPLE_ROWS
            or budgets.max_environment_actions != MAX_ACTION_SLOTS
            or budgets.max_wall_seconds != MAX_COHORT_SECONDS):
        raise ValueError("calibration budgets must match the fixed 3-step/1024-slot/300-second cohort")
    if config.cases.split != "train" or config.cases.partition != "train":
        raise ValueError("calibration must use the authenticated train manifest")
    if not config.cases.for_mode("standalone") or len(config.cases.for_mode("run")) != 1:
        raise ValueError("calibration train manifest must contain standalone cases and one Run case")
    if len(config.accepted_identities) != 1 or not {
        "standalone", "run",
    } <= set(config.accepted_identities[0].accepted_raw_mode_mappings):
        raise ValueError("calibration requires the accepted shared standalone/Run identity")
    output_expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts/development/neural-combat/task-8/calibration/outputs"
    ).resolve()
    if config.output_root != output_expected:
        raise ValueError("calibration output_root must be the dedicated Task 8 outputs child")


def _trial_config(config: NeuralConfig, spec: TrialSpec, *, trial_index: int) -> NeuralConfig:
    if not isinstance(spec, TrialSpec):
        raise TypeError("trial config requires a fixed TrialSpec")
    if type(trial_index) is not int or not 0 <= trial_index < len(TRIALS):
        raise ValueError("trial index is outside the fixed calibration order")
    seeds: dict[str, int] = {}
    for name in ("policy", "optimizer", "curriculum", "noncombat"):
        authored = getattr(config.seed_streams, name)
        digest = hashlib.sha256(
            f"spire-learn-task8-calibration\0{config.config_sha256}\0{spec.trial_id}\0{name}\0{authored}".encode()
        ).digest()
        seeds[name] = int.from_bytes(digest[:8], "big")
    while len(set(seeds.values())) != len(seeds):
        seeds = {name: (value + 1) % 2**64 for name, value in seeds.items()}
    return replace(
        config,
        mode=spec.mode,
        operation="calibrate",
        split="train",
        partition="train",
        profile_id=RUN_PROFILE_ID if spec.mode == "run" else None,
        observation_version=RUN_OBSERVATION_VERSION if spec.mode == "run" else None,
        replay_version=RUN_REPLAY_VERSION if spec.mode == "run" else None,
        run_observation_fingerprint=RUN_OBSERVATION_FINGERPRINT if spec.mode == "run" else None,
        seed_streams=SeedStreams(**seeds),
        rollout_rows=spec.batch_size,
        rollout_steps=spec.horizon,
    )


def _recovery(spec: TrialSpec) -> CollectorRecovery:
    return CollectorRecovery(
        next_seed_ordinal=spec.seed_ordinal,
        next_decision_sequence=spec.seed_ordinal,
        next_episode_ordinals=(spec.seed_ordinal,) * spec.batch_size,
    )


def _frame_sha256(frame: object) -> str:
    digest = hashlib.sha256()
    digest.update(str(getattr(frame, "schema_fingerprint", "")).encode("ascii"))
    tensors = getattr(frame, "tensors", None)
    if not isinstance(tensors, Mapping):
        raise TypeError("calibration comparison frames must be owned public CombatFrame values")
    for name, buffer in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(buffer.dtype.encode("ascii"))
        digest.update(repr(buffer.shape).encode("ascii"))
        digest.update(buffer.data)
    return digest.hexdigest()


def _source_batch(rollout: object, batch_size: int) -> tuple[object, ...]:
    records = getattr(rollout, "records", None)
    if not isinstance(records, tuple):
        raise ValueError("collector did not return immutable rollout records")
    actor_records = tuple(record for record in records if getattr(record, "actor_training_mask", False))
    if not actor_records:
        return ()
    decision_id = actor_records[0].decision_id
    selected = tuple(record.source for record in actor_records if record.decision_id == decision_id)
    if len(selected) != batch_size:
        return ()
    return selected


def _collector_local_identities(trial_id: str, rollout: object | None) -> list[dict[str, str]]:
    records = getattr(rollout, "records", ()) if rollout is not None else ()
    if not isinstance(records, tuple):
        return []
    identities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        stream_id = getattr(record, "stream_id", None)
        task_id = getattr(record, "task_id", None)
        if not isinstance(stream_id, str) or not isinstance(task_id, str):
            continue
        identity = (stream_id, task_id)
        if identity in seen:
            continue
        seen.add(identity)
        identities.append({
            "stream_id": stream_id,
            "task_id": task_id,
            "qualified_id": f"{trial_id}/{stream_id}/{task_id}",
        })
    return identities


def _stored_frame_cpu_gpu_ratio(
    cpu_forward_seconds: float,
    cpu_backward_seconds: float,
    gpu_forward_seconds: float,
    gpu_backward_seconds: float,
) -> dict[str, object]:
    values = (cpu_forward_seconds, cpu_backward_seconds, gpu_forward_seconds, gpu_backward_seconds)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0
           for value in values):
        raise ValueError("stored-frame phase timings must be finite non-negative primitive numbers")
    numerator = float(cpu_forward_seconds) + float(cpu_backward_seconds)
    denominator = float(gpu_forward_seconds) + float(gpu_backward_seconds)
    boundary = (
        "same owned stored frames; CPU forward plus backward divided by GPU forward plus backward; "
        "CPU collation and GPU H2D excluded"
    )
    if denominator == 0.0:
        return {
            "cpu_over_gpu": None,
            "numerator_seconds": numerator,
            "denominator_seconds": denominator,
            "numerator": "CPU forward_seconds + backward_seconds",
            "denominator": "GPU forward_seconds + backward_seconds",
            "measurement_boundary": boundary,
            "unavailable_reason": "zero_gpu_forward_backward_seconds",
        }
    return {
        "cpu_over_gpu": numerator / denominator,
        "numerator_seconds": numerator,
        "denominator_seconds": denominator,
        "numerator": "CPU forward_seconds + backward_seconds",
        "denominator": "GPU forward_seconds + backward_seconds",
        "measurement_boundary": boundary,
        "unavailable_reason": None,
    }


def _numeric_values(value: object):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _numeric_values(item)
        return
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("logit comparison values must be finite primitive numbers")
    yield float(value)


def _all_candidate_logit_difference(first: object, second: object) -> dict[str, float]:
    first_values = tuple(_numeric_values(first))
    second_values = tuple(_numeric_values(second))
    if len(first_values) != len(second_values) or not first_values:
        raise ValueError("all-candidate logit comparison requires equal non-empty outputs")
    return {
        "all_candidate_logits_max_abs_difference": max(
            abs(left - right) for left, right in zip(first_values, second_values, strict=True)
        ),
    }


def _forward_backward_probe(
    model: object,
    batch: object,
    objective: Callable[[object, object], object],
    *,
    budget: object,
    phase: str,
    timer: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], object] | None = None,
) -> tuple[object, object, dict[str, float]]:
    """Admit and time one forward/loss and one backward, with optional CUDA sync."""
    def phase_call(label: str, operation: Callable[[], object]) -> tuple[object, float]:
        if synchronize is not None:
            synchronize()
        budget.check(f"{phase}_{label}")
        started = timer()
        primary: BaseException | None = None
        result: object | None = None
        try:
            result = operation()
        except BaseException as error:
            primary = error
        if synchronize is not None:
            try:
                synchronize()
            except BaseException as error:
                primary = _prefer_exception(primary, error, f"{phase} {label} synchronization")
        try:
            elapsed = _elapsed(timer, started)
        except BaseException as error:
            primary = _prefer_exception(primary, error, f"{phase} {label} timing")
            elapsed = 0.0
        if primary is not None:
            raise primary
        return result, elapsed

    output, forward_seconds = phase_call(
        "forward", lambda: (lambda output: (output, objective(output, batch)))(model.forward(batch)),
    )
    output, loss = output
    _backward_result, backward_seconds = phase_call("backward", getattr(loss, "backward"))
    return output, loss, {
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
    }


def _collection_projection(diagnostics: object | None, counters: object | None = None) -> dict[str, object]:
    names = (
        *_CalibrationLedger._COLLECTION_FIELDS,
        "tasks_started", "task_terminals", "tasks_completed", "victories", "deaths", "escaped",
        "faults", "incomplete_decisions", "incomplete_environment_slots",
        "elapsed_seconds", "wall_overshoot_seconds", "elapsed_seconds_is_fallback",
        "next_seed_ordinal", "stop_reason",
    )
    result: dict[str, object] = {}
    for name in names:
        value = getattr(diagnostics, name, None) if diagnostics is not None else None
        if type(value) in (str, int, float, bool) or value is None:
            result[name] = value
    incomplete = getattr(diagnostics, "incomplete_tasks", ()) if diagnostics is not None else ()
    result["incomplete_task_count"] = len(incomplete) if isinstance(incomplete, tuple) else 0
    fault = getattr(diagnostics, "fault", None) if diagnostics is not None else None
    result["fault"] = None if fault is None else {
        "phase": getattr(fault, "phase", None),
        "exception_type": getattr(fault, "exception_type", None),
    }
    for name in ("environment_terminations", "environment_truncations"):
        value = getattr(counters, name, 0) if counters is not None else 0
        result[name] = value if type(value) is int and value >= 0 else None
    return result


def _seed_projection(config: NeuralConfig) -> dict[str, int]:
    return {
        "policy": config.seed_streams.policy,
        "optimizer": config.seed_streams.optimizer,
        "curriculum": config.seed_streams.curriculum,
        "noncombat": config.seed_streams.noncombat,
    }


def _derived_config_projection(config: NeuralConfig) -> dict[str, object]:
    return {
        "config_sha256": config.config_sha256,
        "manifest_sha256": config.manifest_sha256,
        "mode": config.mode,
        "operation": config.operation,
        "split": config.split,
        "partition": config.partition,
        "profile_id": config.profile_id,
        "observation_version": config.observation_version,
        "replay_version": config.replay_version,
        "run_observation_fingerprint": config.run_observation_fingerprint,
        "rollout": {"rows": config.rollout_rows, "steps": config.rollout_steps},
        "device": config.device,
        "seed_streams": _seed_projection(config),
    }


def _config_projection(config: NeuralConfig) -> dict[str, object]:
    return {
        "schema": "spire-learn-neural-config-v1",
        "config_sha256": config.config_sha256,
        "mode": config.mode,
        "operation": config.operation,
        "split": config.split,
        "partition": config.partition,
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": config.manifest_sha256,
        "output_root": str(config.output_root),
        "environment_revision": config.environment_revision,
        "common_view_id": config.common_view_id,
        "common_view_fingerprint": config.common_view_fingerprint,
        "profile_id": config.profile_id,
        "run_observation_fingerprint": config.run_observation_fingerprint,
        "device": config.device,
        "calibration_gpu_uuid": config.calibration_gpu_uuid,
        "seed_streams": _seed_projection(config),
        "model": {
            "embedding_width": config.model.embedding_width,
            "hidden_width": config.model.hidden_width,
            "max_parameters": config.model.max_parameters,
        },
        "ppo": {
            "learning_rate": config.ppo.learning_rate,
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clip_epsilon": config.ppo.clip_epsilon,
            "value_coefficient": config.ppo.value_coefficient,
            "entropy_coefficient": config.ppo.entropy_coefficient,
            "max_grad_norm": config.ppo.max_grad_norm,
            "epochs": config.ppo.epochs,
            "minibatch_rows": config.ppo.minibatch_rows,
            "adam_eps": config.ppo.adam_eps,
        },
        "budgets": {
            "max_optimizer_steps": config.budgets.max_optimizer_steps,
            "max_training_decisions": config.budgets.max_training_decisions,
            "max_environment_actions": config.budgets.max_environment_actions,
            "max_wall_seconds": config.budgets.max_wall_seconds,
        },
    }


def _manifest_projection(config: NeuralConfig, device: Mapping[str, object]) -> dict[str, object]:
    return {
        "operation": "calibrate",
        "cohort_id": "task8-fixed-calibration-v1",
        "config_sha256": config.config_sha256,
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": config.manifest_sha256,
        "environment_revision": config.environment_revision,
        "common_view_id": config.common_view_id,
        "common_view_fingerprint": config.common_view_fingerprint,
        "gym_source_dirty_state": "not_mounted; accepted native build identity checked by make_driver",
        "device": dict(device),
        "trial_order": [spec.trial_id for spec in TRIALS],
        "trial_identity": {
            "qualified_identity_format": "{trial_id}/{stream_id}/{task_id}",
            "collector_stream_id_format": "stream-{slot}",
            "collector_task_id_formats": {
                "standalone": "task-{slot}-{episode}",
                "run": "task-{slot}-run-{episode}-visit-{visit}",
            },
            "identity_source": "actual public rollout.records; Collector IDs remain local and unmodified",
        },
        "rng_001": (
            "Project-owned reproducible randomness is not seed-for-seed or bit-for-bit "
            "original-game RNG parity; no original-runtime transfer is established."
        ),
        "time_001": (
            "The full-run Secret Portal elapsed-time assumption is retained; this "
            "calibration does not validate it."
        ),
        "python_version": platform.python_version(),
    }


def _write_lifecycle_receipt(run_dir: Path, receipt: Mapping[str, object]) -> None:
    path = run_dir / "lifecycle.json"
    if path.exists() or path.is_symlink():
        raise OSError("calibration lifecycle receipt already exists")
    encoded = json.dumps(dict(receipt), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _calibrate_with_dependencies(
    config: NeuralConfig,
    dependencies: object,
    *,
    clock: Callable[[], float] = time.monotonic,
    timer: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
) -> NeuralReport:
    """Injectable control plane used by real Collector-backed pure tests."""
    _validate_calibration_config(config)
    if not callable(clock) or not callable(timer):
        raise TypeError("calibration clock and timer must be callable")
    budget = _CalibrationBudget(clock)
    ledger = _CalibrationLedger()
    if run_id is None:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex

    try:
        budget.check("device_qualification")
        device = dependencies.verify_device(config, budget)
        if not isinstance(device, Mapping) or device.get("gpu_uuid") != config.calibration_gpu_uuid:
            raise CalibrationPreflightError("runtime GPU UUID does not match the config")
        memory_guard = getattr(dependencies, "check_memory", None)
        if callable(memory_guard):
            budget.bind_memory_guard(memory_guard)
        budget.check("output_creation")
    except BaseException as error:
        if _is_process_control(error):
            raise
        return NeuralReport({
            "schema_id": "spire-learn-neural-calibration-report-v1",
            "operation": "calibrate",
            "status": "setup_blocked",
            "stop_reason": "device_preflight_failed",
            "config_sha256": config.config_sha256,
            "manifest_sha256": config.manifest_sha256,
            "setup_error": _error_value(error),
            "artifacts_created": False,
        }, 2, None)

    writer: object | None = None
    run_dir: Path | None = None
    writer_close_attempted = False
    primary: BaseException | None = None
    fatal: BaseException | None = None
    close_errors: list[BaseException] = []
    serialization_seconds = 0.0
    writer_creation_seconds = 0.0
    receipt_serialization_seconds: float | None = None
    status = "complete"
    stop_reason = "fixed_cohort_complete"
    trials: list[dict[str, object]] = []
    tested_rates: list[tuple[float, dict[str, object]]] = []
    report: NeuralReport | None = None
    summary: dict[str, object] | None = None

    try:
        budget.check("artifact_writer_creation")
        started = timer()
        writer = dependencies.create_writer(config, run_id)
        writer_creation_seconds = max(0.0, timer() - started)
        run_dir_value = getattr(writer, "run_dir", None)
        run_dir = Path(run_dir_value) if run_dir_value is not None else None
        budget.check("artifact_run_start_serialization")
        started = timer()
        writer.start(_config_projection(config), _manifest_projection(config, device))
        serialization_seconds += max(0.0, timer() - started)

        for trial_index, spec in enumerate(TRIALS):
            if budget.remaining() <= 0:
                status, stop_reason = "incomplete", "wall_budget_before_trial"
                break
            trial_started = budget.elapsed()
            reservation = ledger.reserve_actions(spec.action_slots, name=spec.trial_id)
            derived = _trial_config(config, spec, trial_index=trial_index)
            session: _TrialAssembly | Any | None = None
            rollout: object | None = None
            diagnostics: object | None = None
            frame_hashes: tuple[str, ...] = ()
            comparison: Mapping[str, object] | None = None
            optimizer_result: Mapping[str, object] | None = None
            trial_error: BaseException | None = None
            trial_status = "incomplete"
            trial_stop_reason: str | None = None
            trial_rate: float | None = None
            try:
                budget.check("trial_factory")
                session = dependencies.build_trial(
                    derived, spec, _recovery(spec), budget, ledger, timer,
                )
                if not callable(getattr(session, "close", None)) or not callable(
                    getattr(getattr(session, "collector", None), "collect", None)
                ):
                    raise TypeError("calibration trial factory must return a closeable Collector session")
                policy_version = getattr(getattr(session, "policy", None), "_behavior_policy_version", 0)
                if type(policy_version) not in (int, str):
                    raise ValueError("calibration policy version must be a primitive integer or string")
                try:
                    rollout = session.collector.collect(
                        max_learned_decisions=spec.action_slots,
                        max_env_actions=spec.action_slots,
                        policy_version=policy_version,
                    )
                except BaseException as error:
                    trial_error = _prefer_exception(trial_error, error, "collector collection")
                    try:
                        rollout = getattr(session.collector, "partial_rollout", None)
                    except BaseException as partial_error:
                        trial_error = _prefer_exception(
                            trial_error, partial_error, "partial rollout lookup",
                        )
                diagnostics = getattr(rollout, "collection", None) if rollout is not None else None
                ledger.record_collection(diagnostics)
                trial_stop_reason = getattr(diagnostics, "stop_reason", None)

                try:
                    session.timing.raise_pending_instrumentation_error()
                except BaseException as instrumentation_error:
                    trial_error = _prefer_exception(
                        trial_error, instrumentation_error, "trial instrumentation",
                    )

                if trial_error is not None:
                    raise trial_error
                delivered = getattr(diagnostics, "delivered_environment_slots", 0)
                counters = getattr(rollout, "counters", None)
                run_terminations = getattr(counters, "environment_terminations", 0)
                run_truncations = getattr(counters, "environment_truncations", 0)
                run_faults = getattr(diagnostics, "faults", 0)
                accepted_natural_run_end = (
                    spec.mode == "run"
                    and trial_stop_reason == "run_episode_terminal"
                    and delivered > 0
                    and run_terminations > 0
                    and run_truncations == 0
                    and run_faults == 0
                )
                collection_complete = delivered == reservation.amount or accepted_natural_run_end
                if not collection_complete:
                    trial_status = "incomplete"
                    trial_stop_reason = trial_stop_reason or "action_or_wall_cap"
                    status, stop_reason = "incomplete", trial_stop_reason
                else:
                    trial_status = "complete"

                if trial_status == "complete" and spec.mode == "standalone":
                    frames = _source_batch(rollout, spec.batch_size)
                    if len(frames) != spec.batch_size:
                        trial_status = "incomplete"
                        trial_stop_reason = "no_exact_ongoing_source_batch"
                        status, stop_reason = "incomplete", trial_stop_reason
                    else:
                        frame_hashes = tuple(_frame_sha256(frame) for frame in frames)
                        budget.check("stored_frame_comparison")
                        comparison_value = dependencies.compare(session, frames, budget)
                        if not isinstance(comparison_value, Mapping):
                            raise TypeError("comparison probe must return a primitive mapping")
                        comparison = dict(comparison_value)
                        if (comparison.get("measurement_complete") is not True
                                or comparison.get("cpu_finite") is not True
                                or comparison.get("gpu_finite") is not True
                                or comparison.get("frame_sha256") != list(frame_hashes)
                                or comparison.get("same_initial_parameter_sha256") is not True):
                            raise RuntimeError("CPU/GPU comparison did not prove finite same-frame initial-model work")

                        budget.check("disposable_optimizer_reservation")
                        optimizer_reservation = ledger.reserve_optimizer_call(trial_id=spec.trial_id)
                        try:
                            update_value = dependencies.optimizer_probe(
                                session, frames, budget, ledger, optimizer_reservation,
                            )
                        except BaseException:
                            if ledger._optimizer_states.get(optimizer_reservation.reservation_id) == "attempted":
                                ledger.mark_optimizer_uncertain(optimizer_reservation)
                            raise
                        if not isinstance(update_value, Mapping):
                            raise TypeError("optimizer probe must return a primitive mapping")
                        optimizer_result = dict(update_value)
                        if (optimizer_result.get("attempted_steps") != 1
                                or optimizer_result.get("accepted_steps") != 1
                                or optimizer_result.get("uncertain_steps") != 0
                                or optimizer_result.get("finite") is not True
                                or optimizer_result.get("parameters_changed") is not True):
                            raise RuntimeError("standalone disposable Adam probe was not one finite accepted parameter update")

            except BaseException as error:
                trial_error = _prefer_exception(trial_error, error, "trial execution")
                primary = _prefer_exception(primary, trial_error, "trial failure")
                if _is_process_control(trial_error):
                    status, stop_reason = "interrupted", "process_control"
                elif isinstance(trial_error, CalibrationStop):
                    status = "incomplete"
                    stop_reason = trial_error.reason
                elif isinstance(trial_error, MemoryError):
                    status = "incomplete"
                    stop_reason = "memory_cap"
                else:
                    status = "operational_failure"
                    stop_reason = "trial_or_probe_failure"
                    fatal = _prefer_exception(fatal, trial_error, "trial failure")
                trial_status = "incomplete" if status == "incomplete" else "failed"
            finally:
                if session is not None:
                    primary, close_error = _close_preserving(
                        primary if primary is not None else trial_error,
                        session,
                        "calibration trial",
                    )
                    if close_error is not None:
                        close_errors.append(close_error)
                        if _is_process_control(primary):
                            status, stop_reason, trial_status = "interrupted", "process_control", "failed"
                        elif status == "complete":
                            status, stop_reason, trial_status = "operational_failure", "trial_close_failure", "failed"
                            fatal = _prefer_exception(fatal, primary or close_error, "trial close")

            elapsed = max(0.0, budget.elapsed() - trial_started)
            if trial_status == "complete" and spec.mode == "standalone" and elapsed > 0:
                trial_rate = float(getattr(diagnostics, "delivered_environment_slots", 0)) / elapsed
                tested_rates.append((trial_rate, {
                    "trial_id": spec.trial_id,
                    "useful_delivered_slots_per_second": trial_rate,
                    "measurement_boundary": "collection, same-frame probes, disposable Adam, and trial close",
                }))
            collection_projection = _collection_projection(
                diagnostics, getattr(rollout, "counters", None) if rollout is not None else None,
            )
            delivered_slots = collection_projection.get("delivered_environment_slots", 0)
            rate = None
            if type(delivered_slots) is int and elapsed > 0:
                rate = delivered_slots / elapsed
            trial_row: dict[str, object] = {
                "trial_id": spec.trial_id,
                "mode": spec.mode,
                "batch_size": spec.batch_size,
                "horizon": spec.horizon,
                "reserved_action_slots": reservation.amount,
                "seed_ordinal_start": spec.seed_ordinal,
                "derived_trial_config": _derived_config_projection(derived),
                "collector_local_identities": _collector_local_identities(spec.trial_id, rollout),
                "status": trial_status,
                "fully_tested": trial_status == "complete",
                "stop_reason": trial_stop_reason or stop_reason if trial_status != "complete" else trial_stop_reason,
                "collection": collection_projection,
                "sample_accounting": {
                    key: value for key, value in ledger.snapshot().items() if "sample" in key
                },
                "frame_sha256": list(frame_hashes),
                "comparison": dict(comparison) if comparison is not None else None,
                "optimizer_probe": dict(optimizer_result) if optimizer_result is not None else None,
                "timings": session.timing.projection() if session is not None else {},
                "trial_elapsed_seconds_including_runtime_close": elapsed,
                "useful_delivered_slots_per_second": rate,
                "trial_error": _error_value(trial_error) if trial_error is not None else None,
            }
            trials.append(trial_row)
            started: float | None = None
            event_error: BaseException | None = None
            try:
                started = timer()
                budget.check("trial_result_serialization")
                writer.event({"event": "calibration_trial", **trial_row})
            except BaseException as error:
                event_error = error
            if started is not None:
                try:
                    serialization_seconds += _elapsed(timer, started)
                except BaseException as timing_error:
                    event_error = _prefer_exception(
                        event_error, timing_error, "trial result serialization timing",
                    )
            if event_error is not None:
                primary = _prefer_exception(primary, event_error, "trial event")
                if _is_process_control(primary):
                    status, stop_reason = "interrupted", "process_control"
                elif isinstance(primary, CalibrationStop):
                    status, stop_reason = "incomplete", primary.reason
                elif isinstance(primary, MemoryError):
                    status, stop_reason = "incomplete", "memory_cap"
                else:
                    fatal = _prefer_exception(fatal, primary, "trial result serialization")
                    status, stop_reason = "artifact_failure", "trial_result_serialization_failed"
                break
            if trial_status != "complete":
                break
            if status != "complete":
                break

        selection: dict[str, object] | None = None
        if tested_rates:
            selection = max(tested_rates, key=lambda item: item[0])[1]
        if status == "complete" and selection is None:
            status, stop_reason = "incomplete", "no_fully_tested_standalone_size"

        budget.check("final_memory_report")
        memory = dependencies.memory_report() if callable(getattr(dependencies, "memory_report", None)) else {}
        summary = {
            "schema_id": "spire-learn-neural-calibration-report-v1",
            "operation": "calibrate",
            "cohort_id": "task8-fixed-calibration-v1",
            "status": status,
            "stop_reason": stop_reason,
            "run_directory": str(run_dir) if run_dir is not None else None,
            "config_sha256": config.config_sha256,
            "manifest_sha256": config.manifest_sha256,
            "environment_revision": config.environment_revision,
            "gym_source_dirty_state": "not_mounted; accepted native build identity checked by make_driver",
            "common_view_id": config.common_view_id,
            "common_view_fingerprint": config.common_view_fingerprint,
            "run_profile_id": RUN_PROFILE_ID,
            "run_observation_version": RUN_OBSERVATION_VERSION,
            "run_replay_version": RUN_REPLAY_VERSION,
            "run_observation_fingerprint": RUN_OBSERVATION_FINGERPRINT,
            "model": {
                "embedding_width": config.model.embedding_width,
                "hidden_width": config.model.hidden_width,
                "max_parameters": config.model.max_parameters,
                "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            },
            "budgets": {
                "soft_wall_seconds": MAX_COHORT_SECONDS,
                "whole_wall_seconds_including_cleanup": 600,
                "max_action_slots": MAX_ACTION_SLOTS,
                "absolute_action_reservation_guard": ABSOLUTE_ACTION_RESERVATION_CAP,
                "max_sample_rows": MAX_SAMPLE_ROWS,
                "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
                "max_allocator_bytes": ALLOCATOR_CAP_BYTES,
            },
            "cohort": ledger.snapshot(),
            "trials": trials,
            "selection": selection,
            "selection_rule": (
                "highest measured delivered native slot throughput among standalone trials that "
                "completed their reserved collection, same-frame CPU/GPU comparison, finite Adam "
                "probe, and runtime close; no unmeasured or incomplete size is selected"
            ),
            "device": dict(device),
            "memory": dict(memory) if isinstance(memory, Mapping) else {},
            "timing": {
                "serialization_seconds_before_final_summary": serialization_seconds,
                "total_elapsed_seconds_before_runtime_writer_close": budget.elapsed(),
                "measurement_uncertainty": (
                    "one fixed-order pass per size; provider/driver timing subtraction is an adapter residual; "
                    "no threaded engine scaling, parity threshold, or original-game transfer claim"
                ),
            },
            "rng_001": "Project-owned randomness is not original-game seed parity or bit-for-bit parity.",
            "time_001": "The full-run Secret Portal elapsed-time assumption remains unvalidated.",
            "fatal_error": _error_value(fatal) if fatal is not None else None,
        }
        summary["exit_code"] = 0 if status == "complete" else 1
        if _is_process_control(primary) if primary is not None else False:
            summary["status"] = "interrupted"
            summary["exit_code"] = 1
        report = NeuralReport(summary, int(summary["exit_code"]), run_dir)
    except BaseException as error:
        primary = _prefer_exception(primary, error, "calibration runner")
        if _is_process_control(primary):
            status, stop_reason = "interrupted", "process_control"
        elif isinstance(primary, CalibrationStop):
            status, stop_reason = "incomplete", primary.reason
        elif isinstance(primary, MemoryError):
            status, stop_reason = "incomplete", "memory_cap"
        else:
            fatal = _prefer_exception(fatal, primary, "calibration runner failure")
            status = "artifact_failure" if isinstance(primary, OSError) else "operational_failure"
            stop_reason = "calibration_runner_failure"

    # Finalization mirrors the accepted Task 7 authority: summary is provisional;
    # the post-close lifecycle receipt carries authoritative status and elapsed time.
    if writer is not None and summary is None:
        summary = {
            "schema_id": "spire-learn-neural-calibration-report-v1",
            "operation": "calibrate",
            "status": status,
            "stop_reason": stop_reason,
            "config_sha256": config.config_sha256,
            "manifest_sha256": config.manifest_sha256,
            "run_directory": str(run_dir) if run_dir is not None else None,
            "cohort": ledger.snapshot(),
            "trials": trials,
            "selection": None,
            "fatal_error": _error_value(fatal) if fatal is not None else None,
        }
    if writer is not None and summary is not None:
        started: float | None = None
        finalize_error: BaseException | None = None
        try:
            started = timer()
            budget.check("summary_serialization")
            writer.finalize({
                **summary,
                "lifecycle_receipt_required": True,
                "lifecycle": {
                    "status": "receipt_pending",
                    "timing_boundary": "after trial and writer close; before lifecycle receipt serialization and fsync",
                    "receipt_path": "lifecycle.json",
                },
            })
        except BaseException as error:
            finalize_error = error
        if started is not None:
            try:
                serialization_seconds += _elapsed(timer, started)
            except BaseException as timing_error:
                finalize_error = _prefer_exception(
                    finalize_error, timing_error, "summary finalization timing",
                )
        if finalize_error is not None:
            primary = _prefer_exception(primary, finalize_error, "summary finalization")
            if _is_process_control(primary):
                status, stop_reason = "interrupted", "process_control"
            elif isinstance(primary, CalibrationStop):
                status, stop_reason = "incomplete", primary.reason
            elif isinstance(primary, MemoryError):
                status, stop_reason = "incomplete", "memory_cap"
            else:
                fatal = _prefer_exception(fatal, primary, "summary finalization failure")
                status, stop_reason = "artifact_failure", "summary_finalization_failed"
            summary["status"] = status
            summary["stop_reason"] = stop_reason
            summary["exit_code"] = 1
            if fatal is not None:
                summary["fatal_error"] = _error_value(fatal)

    writer_close_seconds: float | None = None
    writer_close_timing_error: BaseException | None = None
    if writer is not None and not writer_close_attempted:
        writer_close_attempted = True
        close_started: float | None = None
        try:
            close_started = timer()
        except BaseException as timing_error:
            writer_close_timing_error = timing_error
            primary = _prefer_exception(primary, timing_error, "artifact writer close timing start")
        primary, writer_close_error = _close_preserving(primary, writer, "calibration artifact writer")
        if close_started is not None:
            try:
                writer_close_seconds = _elapsed(timer, close_started)
            except BaseException as timing_error:
                writer_close_timing_error = _prefer_exception(
                    writer_close_timing_error, timing_error, "artifact writer close timing",
                )
                primary = _prefer_exception(primary, timing_error, "artifact writer close timing")
        if writer_close_error is not None:
            close_errors.append(writer_close_error)
        if _is_process_control(primary) if primary is not None else False:
            status, stop_reason = "interrupted", "process_control"
        elif writer_close_error is not None and status == "complete":
            status, stop_reason = "artifact_failure", "writer_close_failure"
            fatal = _prefer_exception(fatal, primary or writer_close_error, "artifact writer close")
        elif writer_close_timing_error is not None and status == "complete":
            status, stop_reason = "operational_failure", "writer_close_timing_failure"
            fatal = _prefer_exception(fatal, writer_close_timing_error, "artifact writer close timing")

    final_clock_error: BaseException | None = None
    try:
        final_elapsed = budget.elapsed()
    except BaseException as error:
        final_elapsed = None
        final_clock_error = error
        primary = _prefer_exception(primary, error, "final calibration clock")
        if _is_process_control(primary):
            status, stop_reason = "interrupted", "process_control"
        else:
            status, stop_reason = "operational_failure", "final_clock_unavailable"
            fatal = _prefer_exception(fatal, error, "final calibration clock")
    if _is_process_control(primary) if primary is not None else False:
        status, stop_reason = "interrupted", "process_control"

    receipt = {
        "schema_id": "spire-learn-neural-calibration-lifecycle-v1",
        "status": status,
        "exit_code": 0 if status == "complete" else 1,
        "elapsed_seconds": final_elapsed,
        "wall_overshoot_seconds": (
            max(0.0, final_elapsed - MAX_COHORT_SECONDS) if final_elapsed is not None else None
        ),
        "timing_available": final_clock_error is None,
        "timing_boundary": "after runtime and artifact writer close; before lifecycle receipt serialization and fsync",
        "close_errors": [_error_value(item) for item in close_errors],
        "serialization_seconds": serialization_seconds,
        "serialization_seconds_boundary": "writer start/events/summary; excludes this lifecycle receipt write and fsync",
        "writer_creation_seconds": writer_creation_seconds,
        "writer_close_seconds": writer_close_seconds,
        "receipt_path": "lifecycle.json",
    }
    if final_clock_error is not None:
        receipt["timing_unavailable"] = {
            "phase": "post_close_final_measurement",
            "error": _error_value(final_clock_error),
        }
    if fatal is not None:
        receipt["operation_error"] = _error_value(fatal)
    receipt_error: BaseException | None = None
    if writer is not None and run_dir is not None:
        try:
            _write_lifecycle_receipt(run_dir, receipt)
        except BaseException as error:
            receipt_error = error
            primary = _prefer_exception(primary, error, "lifecycle receipt write")
            if _is_process_control(primary):
                status, stop_reason = "interrupted", "process_control"
                receipt["status"] = status
                receipt["exit_code"] = 1
            elif primary is not None and error is not primary:
                _add_note(primary, f"lifecycle receipt write failed: {_error_value(error)}")
            else:
                status, stop_reason = "artifact_failure", "lifecycle_receipt_write_failed"
                receipt["status"] = status
                receipt["exit_code"] = 1
                _add_note(error, "lifecycle receipt was not published")

    if summary is None:
        summary = {
            "schema_id": "spire-learn-neural-calibration-report-v1",
            "operation": "calibrate",
            "status": status,
            "stop_reason": stop_reason,
            "config_sha256": config.config_sha256,
            "manifest_sha256": config.manifest_sha256,
            "run_directory": str(run_dir) if run_dir is not None else None,
            "cohort": ledger.snapshot(),
            "trials": trials,
            "selection": None,
        }
    summary["status"] = status
    summary["stop_reason"] = stop_reason
    summary["elapsed_seconds"] = final_elapsed
    summary["wall_overshoot_seconds"] = (
        max(0.0, final_elapsed - MAX_COHORT_SECONDS) if final_elapsed is not None else None
    )
    summary["serialization_seconds"] = serialization_seconds
    summary["writer_creation_seconds"] = writer_creation_seconds
    summary["receipt_serialization_seconds"] = receipt_serialization_seconds
    summary["total_serialization_seconds_including_receipt"] = (
        None if receipt_serialization_seconds is None
        else serialization_seconds + receipt_serialization_seconds
    )
    timing_summary = summary.get("timing")
    if isinstance(timing_summary, dict):
        timing_summary["total_elapsed_seconds_after_runtime_writer_close_before_receipt"] = final_elapsed
        timing_summary["writer_creation_seconds"] = writer_creation_seconds
        timing_summary["writer_close_seconds"] = writer_close_seconds
        timing_summary["receipt_serialization_seconds_reported_after_write"] = receipt_serialization_seconds
    summary["exit_code"] = int(receipt["exit_code"])
    summary["lifecycle_receipt_required"] = writer is not None
    summary["lifecycle"] = receipt
    if receipt_error is not None:
        summary["lifecycle_receipt_error"] = _error_value(receipt_error)
    if fatal is not None:
        summary["fatal_error"] = _error_value(fatal)
    if report is None:
        report = NeuralReport(summary, int(receipt["exit_code"]), run_dir)
    else:
        report.summary = summary
        report.exit_code = int(receipt["exit_code"])
        report.run_dir = run_dir

    if _is_process_control(primary) if primary is not None else False:
        raise primary
    return report


class _CudaCalibrationDependencies:
    """Lazy CUDA runtime adapter for real calibration, with no PPO/checkpoint path."""

    def __init__(self) -> None:
        self.torch: Any | None = None
        self.device_report: dict[str, object] = {}
        self._last_memory: dict[str, object] = {}

    @staticmethod
    def _nvidia_query(query: str) -> list[list[str]]:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            )
        except Exception as error:
            raise CalibrationPreflightError("nvidia-smi device query failed") from error
        rows = []
        for line in result.stdout.splitlines():
            if line.strip():
                rows.append([part.strip() for part in line.split(",")])
        return rows

    @staticmethod
    def _native_payload_sha256(distribution: object) -> str:
        files = getattr(distribution, "files", None)
        locate = getattr(distribution, "locate_file", None)
        if not isinstance(files, (list, tuple)) or not callable(locate):
            raise CalibrationPreflightError("Spire Lab distribution lacks an inspectable file inventory")
        native_entries = [
            entry for entry in files
            if (Path(str(entry)).parts[:1] == ("spire_lab",)
                and Path(str(entry)).name.startswith("_native")
                and Path(str(entry)).suffix == ".so")
        ]
        if len(native_entries) != 1:
            raise CalibrationPreflightError("Spire Lab distribution must contain one native extension")
        native_path = Path(locate(native_entries[0]))
        if native_path.is_symlink() or not native_path.is_file():
            raise CalibrationPreflightError("installed Spire Lab native extension is missing or unsafe")
        native_path = native_path.resolve(strict=True)
        venv_root = Path(sys.prefix).resolve(strict=True)
        if not native_path.is_relative_to(venv_root):
            raise CalibrationPreflightError("installed Spire Lab native extension escapes the active venv")
        digest = hashlib.sha256()
        with native_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_device(self, config: NeuralConfig, budget: _CalibrationBudget) -> Mapping[str, object]:
        if config.calibration_gpu_uuid != GPU_UUID:
            raise CalibrationPreflightError("config GPU UUID is not the qualified device")
        if os.environ.get("SPIRE_LEARN_TASK8_GPU_UUID") != GPU_UUID:
            raise CalibrationPreflightError("runtime GPU UUID environment gate is not the qualified device")
        runtime_binding = {
            "source_snapshot_sha256": os.environ.get("SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256"),
            "runbook_sha256": os.environ.get("SPIRE_LEARN_TASK8_RUNBOOK_SHA256"),
            "calibration_config_file_sha256": os.environ.get("SPIRE_LEARN_TASK8_CONFIG_SHA256"),
            "train_manifest_sha256": os.environ.get("SPIRE_LEARN_TASK8_MANIFEST_SHA256"),
            "qualification_report_sha256": os.environ.get("SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256"),
            "qualification_summary_sha256": os.environ.get("SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256"),
            "cuda_lock_sha256": os.environ.get("SPIRE_LEARN_TASK8_LOCK_SHA256"),
            "native_sha256": os.environ.get("SPIRE_LEARN_TASK8_NATIVE_SHA256"),
        }
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in runtime_binding.values()):
            raise CalibrationPreflightError("parent-reviewed source/runtime release binding is missing")
        if (runtime_binding["cuda_lock_sha256"] != CUDA_LOCK_SHA256
                or runtime_binding["native_sha256"] != NATIVE_SHA256):
            raise CalibrationPreflightError("parent-reviewed lock/native release identity differs")
        input_identity_domains = _validate_manifest_release_binding(config, runtime_binding)
        for name in _MEMORY_ENV:
            if os.environ.get(name) != "4":
                raise CalibrationPreflightError(f"{name} must be fixed to four CPU threads")
        try:
            import torch
        except Exception as error:
            raise CalibrationPreflightError("qualified CUDA Torch could not be imported") from error
        self.torch = torch
        if str(torch.__version__) != TORCH_VERSION:
            raise CalibrationPreflightError("Torch version differs from the accepted CUDA lock")
        if not isinstance(torch.version.cuda, str) or not torch.version.cuda.startswith(TORCH_CUDA_VERSION_PREFIX):
            raise CalibrationPreflightError("Torch CUDA runtime differs from the accepted CUDA lock")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise CalibrationPreflightError("exactly one available CUDA device is required")
        try:
            torch.cuda.set_device(0)
            torch.cuda.init()
            properties = torch.cuda.get_device_properties(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            architecture = f"sm_{capability[0]}{capability[1]}"
            architectures = tuple(torch.cuda.get_arch_list())
            free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info(0))
        except Exception as error:
            raise CalibrationPreflightError("Torch CUDA device initialization or inventory failed") from error
        visible = self._nvidia_query("gpu=uuid,name,driver_version,memory.total,memory.free")
        if len(visible) != 1 or len(visible[0]) != 5:
            raise CalibrationPreflightError("container must expose exactly one well-formed GPU inventory row")
        visible_uuid, visible_name, driver, total_mib_raw, free_mib_raw = visible[0]
        try:
            total_mib, free_mib = int(total_mib_raw), int(free_mib_raw)
        except ValueError as error:
            raise CalibrationPreflightError("container GPU memory inventory is malformed") from error
        if (visible_uuid != GPU_UUID or visible_name != GPU_NAME or driver != GPU_DRIVER
                or not _UUID_RE.fullmatch(visible_uuid)):
            raise CalibrationPreflightError("container-visible GPU identity differs from the qualified device")
        if (capability != (12, 0) or architecture not in architectures
                or properties.name != GPU_NAME or total_bytes < 24 * 1024**3):
            raise CalibrationPreflightError("Torch-visible GPU capability or compiled architecture is unqualified")
        if free_bytes < MIN_DEVICE_FREE_BYTES or free_mib * 1024**2 < MIN_DEVICE_FREE_BYTES:
            raise CalibrationPreflightError("GPU has less than 12 GiB of free memory")
        distribution_objects: dict[str, object] = {}
        for distribution, expected in (
            ("numpy", NUMPY_VERSION), ("gymnasium", GYMNASIUM_VERSION), ("spire-lab", SPIRE_LAB_VERSION),
        ):
            try:
                package = importlib.metadata.distribution(distribution)
                distribution_objects[distribution] = package
                observed = package.version
            except importlib.metadata.PackageNotFoundError as error:
                raise CalibrationPreflightError(f"locked distribution {distribution} is missing") from error
            if observed != expected:
                raise CalibrationPreflightError(f"locked distribution {distribution} has an unexpected version")
        native_sha256 = self._native_payload_sha256(distribution_objects["spire-lab"])
        if native_sha256 != NATIVE_SHA256:
            raise CalibrationPreflightError("installed Spire Lab native extension differs from the accepted build")
        try:
            if torch.get_num_threads() != 4:
                torch.set_num_threads(4)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        except Exception as error:
            raise CalibrationPreflightError("Torch CPU thread controls could not be set") from error
        if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
            raise CalibrationPreflightError("Torch intra-op/interop thread limits are not four/one")
        fraction = min(1.0, ALLOCATOR_CAP_BYTES / float(total_bytes))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats(0)
        except Exception as error:
            raise CalibrationPreflightError("CUDA allocator could not enforce the 8 GiB ceiling") from error
        budget.bind_memory_guard(self.check_memory)
        process_rows = self._process_memory_rows()
        self.device_report = {
            "gpu_uuid": visible_uuid,
            "device_name": visible_name,
            "driver_version": driver,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "compute_capability": list(capability),
            "compiled_architecture": architecture,
            "compiled_architectures": list(architectures),
            "device_total_bytes": total_bytes,
            "device_free_bytes_torch": free_bytes,
            "device_total_mib_nvidia_smi": total_mib,
            "device_free_mib_nvidia_smi": free_mib,
            "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
            "allocator_fraction": fraction,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "numpy_version": NUMPY_VERSION,
            "gymnasium_version": GYMNASIUM_VERSION,
            "spire_lab_version": SPIRE_LAB_VERSION,
            "spire_lab_native_sha256": native_sha256,
            "parent_release_binding": dict(runtime_binding),
            "input_identity_domains": dict(input_identity_domains),
            "process_id": os.getpid(),
            "gpu_process_memory_rows": process_rows,
            "owned_process_identified": any(row["pid"] == os.getpid() for row in process_rows),
            "owned_process_memory_mib": next(
                (row["memory_mib"] for row in process_rows if row["pid"] == os.getpid()), None,
            ),
        }
        return dict(self.device_report)

    def _process_memory_rows(self) -> list[dict[str, object]]:
        rows = self._nvidia_query("compute-apps=pid,used_memory,gpu_uuid")
        result: list[dict[str, object]] = []
        for row in rows:
            if len(row) != 3 or row[2] != GPU_UUID:
                raise CalibrationPreflightError("GPU process inventory contains a malformed or unapproved row")
            try:
                result.append({"pid": int(row[0]), "memory_mib": int(row[1]), "gpu_uuid": row[2]})
            except ValueError as error:
                raise CalibrationPreflightError("GPU process memory inventory is malformed") from error
        return result

    def check_memory(self) -> None:
        torch = self.torch
        if torch is None:
            return
        try:
            torch.cuda.synchronize(0)
            current_allocated = int(torch.cuda.memory_allocated(0))
            current_reserved = int(torch.cuda.memory_reserved(0))
            peak_allocated = int(torch.cuda.max_memory_allocated(0))
            peak_reserved = int(torch.cuda.max_memory_reserved(0))
        except Exception as error:
            raise CalibrationStop("cuda_memory_observation_failed") from error
        self._last_memory = {
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        }
        if max(current_allocated, current_reserved, peak_allocated, peak_reserved) > ALLOCATOR_CAP_BYTES:
            raise MemoryError("CUDA allocator memory exceeded the approved 8 GiB ceiling")

    def memory_report(self) -> Mapping[str, object]:
        self.check_memory()
        rows = self._process_memory_rows()
        result = {
            **self._last_memory,
            "gpu_process_memory_rows": rows,
            "owned_process_identified": any(row["pid"] == os.getpid() for row in rows),
            "owned_process_memory_mib": next(
                (row["memory_mib"] for row in rows if row["pid"] == os.getpid()), None,
            ),
            "process_id": os.getpid(),
            "observation_qualification": (
                "Torch allocator peaks and nvidia-smi process rows are observed separately; "
                "device-wide external/context allocations are not a hard-process isolation claim"
            ),
        }
        return result

    def create_writer(self, config: NeuralConfig, run_id: str) -> object:
        from spire_learn.neural.artifacts import NeuralWriter

        return NeuralWriter.create(config.output_root, run_id)

    def build_trial(
        self,
        config: NeuralConfig,
        spec: TrialSpec,
        recovery: CollectorRecovery,
        budget: _CalibrationBudget,
        ledger: _CalibrationLedger,
        timer: Callable[[], float],
    ) -> _TrialAssembly:
        from spire_learn.neural.envs import make_driver
        from spire_learn.neural.model import build_policy

        torch = self.torch
        if torch is None:
            raise RuntimeError("CUDA device preflight did not initialize Torch")

        def generator_factory(selected: NeuralConfig) -> object:
            generator = torch.Generator(device="cuda:0")
            generator.manual_seed(selected.seed_streams.policy)
            return generator

        return _assemble_trial(
            config,
            spec,
            recovery,
            budget,
            ledger,
            driver_factory=lambda selected, cases: make_driver(selected, cases),
            policy_factory=build_policy,
            generator_factory=generator_factory,
            timer=timer,
            synchronize=lambda: torch.cuda.synchronize(0),
        )

    @staticmethod
    def _parameter_digest(policy: object, torch: Any) -> str:
        digest = hashlib.sha256()
        parameters = getattr(policy, "named_parameters", None)
        if not callable(parameters):
            raise TypeError("calibration model must expose named_parameters")
        for name, parameter in parameters():
            cpu = parameter.detach().to(device="cpu").contiguous()
            raw = cpu.view(torch.uint8)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes(raw.reshape(-1).tolist()))
        return digest.hexdigest()

    @staticmethod
    def _finite_output(output: object, torch: Any) -> bool:
        logits = getattr(output, "logits", None)
        values = getattr(output, "values", None)
        return bool(
            logits is not None and values is not None
            and torch.isfinite(logits).all().item()
            and torch.isfinite(values).all().item()
        )

    @staticmethod
    def _objective(output: object, batch: object, torch: Any) -> object:
        from spire_learn.neural.distribution import masked_categorical

        distribution = masked_categorical(output.logits, batch.action_mask)
        return -distribution.entropy.mean() + 0.5 * output.values.square().mean()

    @staticmethod
    def _validate_gradients(policy: object, torch: Any) -> None:
        gradients = [parameter.grad for parameter in policy.parameters()]
        present = [gradient for gradient in gradients if gradient is not None]
        if not present or any(not torch.isfinite(gradient).all().item() for gradient in present):
            raise RuntimeError("calibration produced absent or non-finite model gradients")

    def compare(
        self,
        session: _TrialAssembly,
        frames: tuple[object, ...],
        budget: _CalibrationBudget,
    ) -> Mapping[str, object]:
        torch = self.torch
        if torch is None:
            raise RuntimeError("Torch was not initialized")
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model import build_policy

        budget.check("cpu_collation")
        started = time.perf_counter()
        cpu_batch = encode_frames(frames, session.driver.schema, device="cpu")
        session.timing.cpu_collation_seconds += time.perf_counter() - started
        budget.check("cpu_model_copy")
        started = time.perf_counter()
        cpu_model = build_policy(
            session.driver.schema,
            session.config.model,
            seed=session.config.seed_streams.policy,
            device="cpu",
        )
        initial_sha256 = self._parameter_digest(session.policy, torch)
        cpu_model.load_state_dict(session.policy.state_dict(), strict=True)
        cpu_initial_sha256 = self._parameter_digest(cpu_model, torch)
        session.timing.cpu_model_copy_seconds += time.perf_counter() - started
        if initial_sha256 != cpu_initial_sha256:
            raise RuntimeError("CPU comparison model does not own identical initial parameter bytes")

        cpu_model.zero_grad(set_to_none=True)
        cpu_output, cpu_loss, cpu_timing = _forward_backward_probe(
            cpu_model,
            cpu_batch,
            lambda output, batch: self._objective(output, batch, torch),
            budget=budget,
            phase="cpu",
        )
        session.timing.cpu_forward_seconds += cpu_timing["forward_seconds"]
        session.timing.cpu_backward_seconds += cpu_timing["backward_seconds"]
        self._validate_gradients(cpu_model, torch)
        if not self._finite_output(cpu_output, torch) or not torch.isfinite(cpu_loss).item():
            raise RuntimeError("CPU stored-frame comparison produced a non-finite output")

        budget.check("stored_frame_h2d")
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        gpu_batch = _move_encoded_batch(cpu_batch, "cuda:0")
        torch.cuda.synchronize(0)
        session.timing.h2d_seconds += time.perf_counter() - started
        session.policy.zero_grad(set_to_none=True)
        gpu_output, gpu_loss, gpu_timing = _forward_backward_probe(
            session.policy,
            gpu_batch,
            lambda output, batch: self._objective(output, batch, torch),
            budget=budget,
            phase="gpu",
            synchronize=lambda: torch.cuda.synchronize(0),
        )
        session.timing.gpu_forward_seconds += gpu_timing["forward_seconds"]
        session.timing.gpu_backward_seconds += gpu_timing["backward_seconds"]
        self._validate_gradients(session.policy, torch)
        if not self._finite_output(gpu_output, torch) or not torch.isfinite(gpu_loss).item():
            raise RuntimeError("GPU stored-frame comparison produced a non-finite output")

        cpu_logits = cpu_output.logits.detach().cpu()
        gpu_logits = gpu_output.logits.detach().cpu()
        cpu_values = cpu_output.values.detach().cpu()
        gpu_values = gpu_output.values.detach().cpu()
        logits_metric = _all_candidate_logit_difference(cpu_logits, gpu_logits)
        values_delta = float((cpu_values - gpu_values).abs().max().item())
        if not math.isfinite(values_delta):
            raise RuntimeError("CPU/GPU stored-frame comparison difference was non-finite")
        stored_frame_ratio = _stored_frame_cpu_gpu_ratio(
            cpu_timing["forward_seconds"], cpu_timing["backward_seconds"],
            gpu_timing["forward_seconds"], gpu_timing["backward_seconds"],
        )
        frame_hashes = [_frame_sha256(frame) for frame in frames]
        return {
            "measurement_complete": True,
            "cpu_finite": True,
            "gpu_finite": True,
            "frame_sha256": frame_hashes,
            "frame_count": len(frames),
            "initial_policy_parameter_sha256": initial_sha256,
            "cpu_policy_parameter_sha256": cpu_initial_sha256,
            "same_initial_parameter_sha256": initial_sha256 == cpu_initial_sha256,
            "objective": "synthetic calibration: negative legal-action entropy + 0.5 * value.square().mean",
            "cpu_loss": float(cpu_loss.detach().item()),
            "gpu_loss": float(gpu_loss.detach().cpu().item()),
            **logits_metric,
            "values_max_abs_difference": values_delta,
            "stored_frame_cpu_gpu_ratio": stored_frame_ratio,
            "cpu_forward_seconds": cpu_timing["forward_seconds"],
            "cpu_backward_seconds": cpu_timing["backward_seconds"],
            "gpu_forward_seconds": gpu_timing["forward_seconds"],
            "gpu_backward_seconds": gpu_timing["backward_seconds"],
            "numeric_parity_threshold": None,
            "cpu_dtype": "float32",
            "gpu_dtype": "float32",
            "amp": False,
            "compile": False,
        }

    def optimizer_probe(
        self,
        session: _TrialAssembly,
        frames: tuple[object, ...],
        budget: _CalibrationBudget,
        ledger: _CalibrationLedger,
        reservation: _Reservation,
    ) -> Mapping[str, object]:
        del frames
        torch = self.torch
        if torch is None:
            raise RuntimeError("Torch was not initialized")
        budget.check("disposable_adam")
        before = self._parameter_digest(session.policy, torch)
        optimizer = torch.optim.Adam(
            session.policy.parameters(),
            lr=session.config.ppo.learning_rate,
            eps=session.config.ppo.adam_eps,
        )
        budget.check("optimizer_step")
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        ledger.mark_optimizer_attempted(reservation)
        try:
            optimizer.step()
        except BaseException:
            ledger.mark_optimizer_uncertain(reservation)
            raise
        ledger.mark_optimizer_accepted(reservation)
        torch.cuda.synchronize(0)
        session.timing.gpu_adam_seconds += time.perf_counter() - started
        after = self._parameter_digest(session.policy, torch)
        if before == after:
            raise RuntimeError("disposable Adam call did not change any model parameter bytes")
        parameters_finite = all(torch.isfinite(parameter).all().item() for parameter in session.policy.parameters())
        state_tensors = [
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        ]
        state_finite = all(torch.isfinite(value).all().item() for value in state_tensors)
        if not parameters_finite or not state_finite:
            raise RuntimeError("disposable Adam produced a non-finite parameter or optimizer state")
        return {
            "kind": "synthetic_calibration_adam_not_ppo",
            "attempted_steps": 1,
            "accepted_steps": 1,
            "uncertain_steps": 0,
            "finite": True,
            "parameters_changed": True,
            "initial_parameter_sha256": before,
            "updated_parameter_sha256": after,
            "parameter_sha256_changed": before != after,
            "optimizer_state_finite": True,
            "dtype": "float32",
            "amp": False,
            "compile": False,
        }


def _move_encoded_batch(batch: object, device: str) -> object:
    from spire_learn.neural.features import EncodedBatch

    def moved(values: Mapping[str, Any]) -> dict[str, Any]:
        return {name: value.to(device=device, non_blocking=False) for name, value in values.items()}

    return EncodedBatch(
        schema_fingerprint=batch.schema_fingerprint,
        values=moved(batch.values),
        known=moved(batch.known),
        occupied=moved(batch.occupied),
        action_mask=batch.action_mask.to(device=device, non_blocking=False),
        known_coverage=moved(batch.known_coverage),
    )


def calibrate(
    config: NeuralConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> NeuralReport:
    """Run the exact fixed cohort after static config and qualified-device gates."""
    _validate_calibration_config(config)
    dependencies = _CudaCalibrationDependencies()
    return _calibrate_with_dependencies(config, dependencies, clock=clock)


__all__ = ["CalibrationPreflightError", "CalibrationStop", "TRIALS", "TrialSpec", "calibrate"]
