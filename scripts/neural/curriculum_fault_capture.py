"""Bounded, dependency-light capture of public driver operation failures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "spire-learn-curriculum-fault-capture-v1"
MAX_BATCH_SIZE = 256
MAX_RECORD_BYTES = 64 * 1024
MAX_INTENT_BYTES = 48 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FAULT_BYTES = MAX_RECORD_BYTES
MAX_DIAGNOSTIC_BYTES = 8 * 1024
_MAX_TREE_DEPTH = 8
_MAX_TREE_NODES = 4096
_MAX_TREE_ITEMS = 512
_MAX_TEXT_CHARS = 4096
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CAPTURE_MARKER = "_spire_learn_curriculum_fault_capture"


class FaultCaptureError(ValueError):
    """Capture could not safely record an operation before forwarding it."""


@dataclass
class _Capture:
    collector: Any
    driver: Any
    directory: Path
    config_sha256: str
    manifest_sha256: str
    batch_size: int
    promote_finalization_control: Any
    bytes_written: int = 0
    ordinal: int = 0
    poisoned: bool = False
    first_fault_written: bool = False

    @property
    def operations_path(self) -> Path:
        return self.directory / "operations.jsonl"

    @property
    def first_fault_path(self) -> Path:
        return self.directory / "first-fault.json"


def _bounded_text(value: object, limit: int = 512) -> str:
    try:
        text = str(value)
    except Exception:
        text = "exception message unavailable"
    text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _exception_value(error: BaseException) -> dict[str, str]:
    return {"exception_type": _bounded_text(type(error).__name__, 128),
            "message": _bounded_text(error)}


def _control(capture: _Capture, primary: BaseException, control: BaseException,
             label: str) -> BaseException:
    return capture.promote_finalization_control(primary, control, label)


def _note(error: BaseException, detail: str) -> BaseException:
    add_note = getattr(error, "add_note", None)
    if not callable(add_note):
        return error
    try:
        add_note(_bounded_text(detail))
    except Exception:
        return error
    return error


def _copy_json(value: object, *, depth: int = 0, nodes: list[int] | None = None) -> object:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > _MAX_TREE_NODES or depth > _MAX_TREE_DEPTH:
        raise FaultCaptureError("public diagnostic exceeds the bounded JSON shape")
    if value is None or type(value) in (bool, int, str):
        if type(value) is int and not -(2**63) <= value < 2**64:
            raise FaultCaptureError("public diagnostic integer is outside the bounded JSON range")
        if type(value) is str and len(value) > _MAX_TEXT_CHARS:
            raise FaultCaptureError("public diagnostic text exceeds the bounded JSON shape")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FaultCaptureError("public diagnostic contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_TREE_ITEMS:
            raise FaultCaptureError("public diagnostic mapping exceeds the bounded JSON shape")
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or len(key) > _MAX_TEXT_CHARS:
                raise FaultCaptureError("public diagnostic mapping has an invalid key")
            copied[key] = _copy_json(item, depth=depth + 1, nodes=nodes)
        return copied
    if type(value) in (list, tuple):
        if len(value) > _MAX_TREE_ITEMS:
            raise FaultCaptureError("public diagnostic sequence exceeds the bounded JSON shape")
        return [_copy_json(item, depth=depth + 1, nodes=nodes) for item in value]
    raise FaultCaptureError("public diagnostic is not primitive JSON data")


def _encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _slot_state(capture: _Capture) -> list[dict[str, object]]:
    cases = getattr(capture.collector, "_slot_cases", None)
    seeds = getattr(capture.collector, "_slot_seeds", None)
    if (not isinstance(cases, (list, tuple)) or not isinstance(seeds, (list, tuple))
            or len(cases) != capture.batch_size or len(seeds) != capture.batch_size):
        raise FaultCaptureError("collector case and seed metadata must match the driver batch")
    result: list[dict[str, object]] = []
    for slot, (case, seed) in enumerate(zip(cases, seeds, strict=True)):
        if case is None:
            case_id = entry_id = None
        else:
            case_id = getattr(case, "case_id", None)
            entry_id = getattr(case, "entry_id", None)
            if type(case_id) is not str or len(case_id) > _MAX_TEXT_CHARS:
                raise FaultCaptureError("collector case_id must be bounded text")
            if entry_id is not None and (type(entry_id) is not str or len(entry_id) > _MAX_TEXT_CHARS):
                raise FaultCaptureError("collector entry_id must be bounded text or null")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise FaultCaptureError("collector actual_seed must be a uint64 or null")
        result.append({"slot": slot, "case_id": case_id, "entry_id": entry_id,
                       "actual_seed": seed})
    return result


def _sequence(value: object, name: str, *, types: tuple[type, ...], nullable: bool = False) -> list[object] | None:
    if value is None and nullable:
        return None
    if type(value) not in (tuple, list) or len(value) > MAX_BATCH_SIZE:
        raise FaultCaptureError(f"{name} must be a bounded primitive sequence")
    result: list[object] = []
    for item in value:
        if item is None and nullable:
            result.append(None)
        elif type(item) in types:
            if type(item) is int and not -(2**63) <= item < 2**64:
                raise FaultCaptureError(f"{name} integer is outside the bounded JSON range")
            result.append(item)
        else:
            raise FaultCaptureError(f"{name} contains a non-primitive value")
    return result


def _intent(capture: _Capture, operation: str, args: tuple[object, ...],
            kwargs: dict[str, object]) -> dict[str, object]:
    ordinal = capture.ordinal + 1
    record: dict[str, object] = {
        "schema": SCHEMA,
        "record": "intent",
        "ordinal": ordinal,
        "operation": operation,
        "config_sha256": capture.config_sha256,
        "manifest_sha256": capture.manifest_sha256,
        "slot_state": _slot_state(capture),
    }
    if operation == "reset":
        record["reset"] = {
            "seeds": _sequence(kwargs.get("seeds"), "reset seeds", types=(int,)),
            "entry_ids": _sequence(kwargs.get("entry_ids"), "reset entry_ids",
                                    types=(str,), nullable=True),
            "reset_mask": _sequence(kwargs.get("reset_mask"), "reset_mask", types=(bool,),
                                     nullable=True),
        }
        reset = record["reset"]
        assert isinstance(reset, dict)
        if (reset["seeds"] is not None and any(not 0 <= item < 2**64 for item in reset["seeds"])
                or reset["reset_mask"] is not None
                and any(type(item) is not bool for item in reset["reset_mask"])):
            raise FaultCaptureError("reset intent contains an invalid seed or mask")
    else:
        raw_actions = kwargs.get("actions") if "actions" in kwargs else (args[0] if args else None)
        actions = _sequence(raw_actions, "step actions", types=(int,))
        if actions is not None and any(item < 0 for item in actions):
            raise FaultCaptureError("step action indices must be non-negative integers")
        record["actions"] = actions
    return record


def _append_jsonl(capture: _Capture, record: dict[str, object], *, limit: int = MAX_RECORD_BYTES) -> None:
    payload = _encode(record)
    if len(payload) > limit or len(payload) > MAX_RECORD_BYTES:
        raise FaultCaptureError("operation record exceeds its byte bound")
    if capture.bytes_written + len(payload) > MAX_TOTAL_BYTES - MAX_FAULT_BYTES:
        raise FaultCaptureError("operation log reached its per-runtime byte bound")
    try:
        with capture.operations_path.open("ab") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("short write while persisting operation intent")
            stream.flush()
    except BaseException:
        capture.poisoned = True
        raise
    capture.bytes_written += len(payload)


def _write_first_fault(capture: _Capture, value: dict[str, object]) -> None:
    payload = _encode(value)
    if len(payload) > MAX_FAULT_BYTES:
        raise FaultCaptureError("first-fault record exceeds its byte bound")
    if capture.bytes_written + len(payload) > MAX_TOTAL_BYTES:
        raise FaultCaptureError("first-fault record exceeds the per-runtime byte bound")
    try:
        with capture.first_fault_path.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("short write while persisting first fault")
            stream.flush()
    except BaseException:
        capture.poisoned = True
        raise
    capture.bytes_written += len(payload)
    capture.first_fault_written = True


def _public_diagnostic(capture: _Capture) -> dict[str, object]:
    try:
        provider = getattr(capture.driver, "_provider", None)
        env = getattr(provider, "_env", None)
        diagnostic = getattr(env, "diagnostic", None)
        if not callable(diagnostic):
            return {"status": "unavailable", "reason": "public diagnostic unavailable"}
        copied = _copy_json(diagnostic())
        payload = _encode(copied)
        if len(payload) > MAX_DIAGNOSTIC_BYTES:
            return {"status": "unavailable", "reason": "public diagnostic exceeds 8192 bytes"}
        return {"status": "available", "data": copied}
    except FaultCaptureError:
        return {"status": "unavailable", "reason": "public diagnostic is not bounded primitive JSON"}
    except Exception as error:
        return {"status": "unavailable", "reason": _bounded_text(
            f"{type(error).__name__}: {_bounded_text(error)}")}


def _finalize_fault(capture: _Capture, pending: dict[str, object],
                    operation_error: BaseException) -> BaseException:
    if capture.first_fault_written:
        return operation_error
    propagated = operation_error
    control_error: BaseException | None = None
    if isinstance(operation_error, Exception):
        try:
            diagnostic = _public_diagnostic(capture)
        except BaseException as error:
            diagnostic = {"status": "unavailable", "reason": "diagnostic finalization interrupted"}
            if not isinstance(error, Exception):
                propagated = _control(capture, propagated, error, "public driver diagnostic")
                control_error = error
    else:
        diagnostic = {"status": "unavailable", "reason": "skipped for process-control failure"}

    value: dict[str, object] = {
        "schema": SCHEMA,
        "pending_intent": pending,
        "primary_error": _exception_value(operation_error),
        "public_diagnostic": diagnostic,
    }
    if control_error is not None:
        value["finalization_control"] = _exception_value(control_error)
    try:
        _write_first_fault(capture, value)
    except BaseException as error:
        if isinstance(error, Exception):
            _note(operation_error, f"first-fault capture failed: {type(error).__name__}: {_bounded_text(error)}")
        else:
            propagated = _control(capture, propagated, error, "first-fault record write")
    return propagated


def _forward(capture: _Capture, operation: str, original: Any,
             args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    if capture.poisoned:
        raise FaultCaptureError("capture storage failed; refusing a later native effect")
    intent = _intent(capture, operation, args, kwargs)
    _append_jsonl(capture, intent, limit=MAX_INTENT_BYTES)
    capture.ordinal = int(intent["ordinal"])
    try:
        result = original(*args, **kwargs)
    except BaseException as error:
        try:
            propagated = _finalize_fault(capture, intent, error)
        except BaseException as finalization_error:
            if isinstance(finalization_error, Exception):
                _note(error, f"fault finalization failed: {type(finalization_error).__name__}: "
                      f"{_bounded_text(finalization_error)}")
                propagated = error
            else:
                propagated = _control(capture, error, finalization_error, "fault capture finalization")
        if propagated is error:
            raise
        raise propagated from error
    delivered = {"schema": SCHEMA, "record": "delivered", "ordinal": capture.ordinal,
                 "operation": operation}
    try:
        _append_jsonl(capture, delivered)
    except BaseException as error:
        if isinstance(error, Exception):
            _note(error, f"{operation} returned normally but its delivered record was not persisted")
        raise
    return result


def install_fault_capture(runtime: Any, output_root: Path) -> Path:
    """Observe the existing collector driver and return its unique log directory."""
    collector = getattr(runtime, "collector", None)
    driver = getattr(collector, "driver", None)
    config = getattr(runtime, "config", None)
    if collector is None or driver is None or config is None:
        raise TypeError("runtime must expose collector.driver and config")
    if getattr(driver, _CAPTURE_MARKER, None) is not None:
        raise ValueError("driver already has curriculum fault capture installed")
    batch_size = getattr(collector, "_batch_size", getattr(driver, "batch_size", None))
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError("driver batch size must be between 1 and 256")
    if getattr(driver, "batch_size", batch_size) != batch_size:
        raise ValueError("collector and driver batch sizes must match")
    for name in ("reset", "step"):
        if not callable(getattr(driver, name, None)):
            raise TypeError(f"driver must expose callable {name}")
    config_sha256 = getattr(config, "config_sha256", None)
    manifest_sha256 = getattr(config, "manifest_sha256", None)
    if (not isinstance(config_sha256, str) or _HASH.fullmatch(config_sha256) is None
            or not isinstance(manifest_sha256, str) or _HASH.fullmatch(manifest_sha256) is None):
        raise ValueError("runtime config and manifest hashes must be lowercase SHA-256 values")

    from spire_learn.neural.runner import _promote_finalization_control

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="curriculum-runtime-", dir=root))
    capture = _Capture(collector, driver, directory, config_sha256, manifest_sha256,
                       batch_size, _promote_finalization_control)
    original_reset = driver.reset
    original_step = driver.step
    namespace = getattr(driver, "__dict__", {})
    had_reset, had_step = "reset" in namespace, "step" in namespace
    prior_reset, prior_step = namespace.get("reset"), namespace.get("step")

    def reset_wrapper(*args: object, **kwargs: object) -> object:
        return _forward(capture, "reset", original_reset, args, kwargs)

    def step_wrapper(*args: object, **kwargs: object) -> object:
        return _forward(capture, "step", original_step, args, kwargs)

    try:
        setattr(driver, "reset", reset_wrapper)
        setattr(driver, "step", step_wrapper)
        setattr(driver, _CAPTURE_MARKER, capture)
    except BaseException:
        if had_reset:
            setattr(driver, "reset", prior_reset)
        else:
            try:
                delattr(driver, "reset")
            except AttributeError:
                pass
        if had_step:
            setattr(driver, "step", prior_step)
        else:
            try:
                delattr(driver, "step")
            except AttributeError:
                pass
        raise
    return directory


__all__ = ["FaultCaptureError", "install_fault_capture"]
