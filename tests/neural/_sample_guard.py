"""Stdlib-only lifetime reservations and in-process sampled-row guard.

The shared ledger counts reservations permanently. A failed invocation never
refunds a reservation; its actual draw counts are appended as separate evidence.
The tensor test process installs the Torch hook only after its opt-in gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


LIFETIME_CAP = 4096
MAX_INVOCATION_ROWS = 64
LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts/development/neural-combat/diagnostic-sampled-row-ledger.jsonl"
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_TASKS = frozenset({"task4", "task6"})


class SampleBudgetError(RuntimeError):
    """A draw would exceed its durable per-invocation reservation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append(handle, event: dict[str, object]) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _events(handle) -> list[dict[str, object]]:
    handle.seek(0)
    lines = handle.readlines()
    if not lines:
        raise ValueError("sample ledger is empty; refusing to initialize or reset it")
    result: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"sample ledger line {number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"sample ledger line {number} is not an object")
        result.append(value)
    first = result[0]
    if first != {
        "event": "ledger_initialized",
        "ledger_version": 1,
        "lifetime_cap": LIFETIME_CAP,
    }:
        raise ValueError("sample ledger initialization record does not match version/cap")
    return result


def initialize_ledger(path: Path = LEDGER_PATH) -> None:
    """Create a new ledger once; an existing or empty ledger is never reset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            _events(handle)
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        _append(handle, {
            "event": "ledger_initialized",
            "ledger_version": 1,
            "lifetime_cap": LIFETIME_CAP,
        })


def _reservation_state(events: Sequence[dict[str, object]]) -> tuple[int, dict[str, dict[str, object]], set[str]]:
    reserved = 0
    reservations: dict[str, dict[str, object]] = {}
    finished: set[str] = set()
    for event in events[1:]:
        kind = event.get("event")
        reservation_id = event.get("reservation_id")
        if kind == "reservation":
            if not isinstance(reservation_id, str) or reservation_id in reservations:
                raise ValueError("sample ledger has a missing or duplicate reservation ID")
            task = event.get("task")
            rows = event.get("reserved_rows")
            if task not in _TASKS or type(rows) is not int or not 1 <= rows <= MAX_INVOCATION_ROWS:
                raise ValueError("sample ledger has an invalid task or per-invocation reservation")
            if event.get("reserved_before") != reserved or event.get("reserved_after") != reserved + rows:
                raise ValueError("sample ledger reservation totals are not monotonic")
            reserved += rows
            if reserved > LIFETIME_CAP:
                raise ValueError("sample ledger exceeds the lifetime sampled-row cap")
            reservations[reservation_id] = event
        elif kind == "completion":
            if reservation_id not in reservations or reservation_id in finished:
                raise ValueError("sample ledger completion has no unique reservation")
            reserved_rows = reservations[reservation_id]["reserved_rows"]
            status = event.get("status")
            attempted = event.get("attempted_rows")
            completed = event.get("completed_rows")
            rejected = event.get("rejected_rows")
            if status not in {"not_started", "succeeded", "failed", "uncertain"}:
                raise ValueError("sample ledger has an invalid completion status")
            for count, name in ((attempted, "attempted_rows"), (completed, "completed_rows"),
                                (rejected, "rejected_rows")):
                if count is not None and (type(count) is not int or count < 0):
                    raise ValueError(f"sample ledger {name} must be a non-negative integer or null")
            if (attempted is not None and attempted > reserved_rows
                    or completed is not None and completed > reserved_rows
                    or attempted is not None and completed is not None and completed > attempted):
                raise ValueError("sample ledger actual counts exceed or contradict the reservation")
            if status == "not_started" and (attempted, completed) != (0, 0):
                raise ValueError("not-started invocation must have zero attempted/completed draws")
            if status != "uncertain" and (attempted is None or completed is None):
                raise ValueError("known invocation status requires actual attempted/completed counts")
            finished.add(reservation_id)
        else:
            raise ValueError(f"sample ledger has unsupported event kind {kind!r}")
    return reserved, reservations, finished


def reserve_rows(
    *, task: str, reservation_id: str, rows: int = MAX_INVOCATION_ROWS,
    path: Path = LEDGER_PATH,
) -> dict[str, int]:
    """Durably reserve rows before any protected test process can start."""
    if task not in _TASKS:
        raise ValueError("task must be task4 or task6")
    if not isinstance(reservation_id, str) or _RUN_ID.fullmatch(reservation_id) is None:
        raise ValueError("reservation_id must be a unique 1..64 character run ID")
    if type(rows) is not int or not 1 <= rows <= MAX_INVOCATION_ROWS:
        raise ValueError("each invocation must reserve between 1 and 64 rows")
    path = Path(path)
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        events = _events(handle)
        reserved, reservations, _ = _reservation_state(events)
        if reservation_id in reservations:
            raise ValueError("reservation_id already exists; invocation IDs are never reused")
        if reserved + rows > LIFETIME_CAP:
            raise SampleBudgetError(
                f"reservation would exceed shared cap {LIFETIME_CAP}; remaining={LIFETIME_CAP - reserved}"
            )
        _append(handle, {
            "event": "reservation",
            "reservation_id": reservation_id,
            "task": task,
            "reserved_rows": rows,
            "reserved_before": reserved,
            "reserved_after": reserved + rows,
            "recorded_at_utc": _utc_now(),
        })
    return {"reserved_total": reserved + rows, "remaining": LIFETIME_CAP - reserved - rows}


def complete_reservation(
    *, reservation_id: str, status: str, attempted_rows: int | None,
    completed_rows: int | None, rejected_rows: int = 0, path: Path = LEDGER_PATH,
) -> dict[str, int]:
    """Append actual evidence without refunding or editing a reservation."""
    path = Path(path)
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        events = _events(handle)
        reserved, reservations, finished = _reservation_state(events)
        if reservation_id not in reservations:
            raise ValueError("completion references an unknown reservation")
        if reservation_id in finished:
            raise ValueError("reservation already has a completion record")
        if status not in {"not_started", "succeeded", "failed", "uncertain"}:
            raise ValueError("invalid completion status")
        reserved_rows = reservations[reservation_id]["reserved_rows"]
        for count, name in ((attempted_rows, "attempted_rows"),
                            (completed_rows, "completed_rows"),
                            (rejected_rows, "rejected_rows")):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if (attempted_rows is not None and attempted_rows > reserved_rows
                or completed_rows is not None and completed_rows > reserved_rows
                or attempted_rows is not None and completed_rows is not None
                and completed_rows > attempted_rows):
            raise ValueError("actual counts exceed or contradict the reservation")
        if status != "uncertain" and (attempted_rows is None or completed_rows is None):
            raise ValueError("known completion requires actual attempted/completed counts")
        if status == "not_started" and (attempted_rows, completed_rows) != (0, 0):
            raise ValueError("not-started invocation must have zero attempted/completed draws")
        _append(handle, {
            "event": "completion",
            "reservation_id": reservation_id,
            "status": status,
            "attempted_rows": attempted_rows,
            "completed_rows": completed_rows,
            "rejected_rows": rejected_rows,
            "recorded_at_utc": _utc_now(),
        })
    return {"reserved_total": reserved, "remaining": LIFETIME_CAP - reserved}


def _sample_rows(input_tensor: object, num_samples: object) -> int:
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError("multinomial num_samples must be a positive integer")
    shape = getattr(input_tensor, "shape", None)
    if shape is None:
        raise TypeError("sample guard requires a tensor with a shape")
    dimensions = tuple(shape)
    if not dimensions or any(type(size) is not int or size < 0 for size in dimensions):
        raise TypeError("sample guard received an invalid tensor shape")
    leading_rows = math.prod(dimensions[:-1]) if len(dimensions) > 1 else 1
    return leading_rows * num_samples


class SamplingGuard:
    """Count each multinomial row before calling Torch and refuse overshoot."""

    def __init__(self, *, reservation_id: str, max_rows: int, evidence_path: Path):
        if not isinstance(reservation_id, str) or _RUN_ID.fullmatch(reservation_id) is None:
            raise ValueError("reservation_id must be a unique 1..64 character run ID")
        if type(max_rows) is not int or not 1 <= max_rows <= MAX_INVOCATION_ROWS:
            raise ValueError("in-process sample ceiling must be between 1 and 64")
        self.reservation_id = reservation_id
        self.max_rows = max_rows
        self.evidence_path = Path(evidence_path)
        if not self.evidence_path.parent.is_dir():
            raise ValueError("sample evidence parent must be the owned writable run directory")
        self.attempted_rows = 0
        self.completed_rows = 0
        self.rejected_rows = 0
        self._write("ready")

    def _write(self, status: str, error: str | None = None) -> None:
        record: dict[str, object] = {
            "schema": "spire-learn-tensor-sampling-v1",
            "reservation_id": self.reservation_id,
            "reserved_rows": self.max_rows,
            "attempted_rows": self.attempted_rows,
            "completed_rows": self.completed_rows,
            "rejected_rows": self.rejected_rows,
            "status": status,
        }
        if error is not None:
            record["last_error"] = error[:240]
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.evidence_path.name}.", suffix=".tmp", dir=self.evidence_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.evidence_path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def wrap(self, sampler: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(sampler)
        def guarded(input_tensor: object, num_samples: object = None, *args, **kwargs):
            if num_samples is None and "num_samples" in kwargs:
                num_samples = kwargs.pop("num_samples")
            rows = _sample_rows(input_tensor, num_samples)
            if self.attempted_rows + rows > self.max_rows:
                self.rejected_rows += rows
                self._write("budget_rejected")
                raise SampleBudgetError(
                    f"sampled rows would exceed invocation reservation {self.max_rows}"
                )
            self.attempted_rows += rows
            self._write("sample_started")
            try:
                result = sampler(input_tensor, num_samples, *args, **kwargs)
            except BaseException as error:
                self._write("sample_failed", f"{type(error).__name__}: {error}")
                raise
            self.completed_rows += rows
            self._write("sample_completed")
            return result

        return guarded


_ACTIVE_GUARD: SamplingGuard | None = None
_ACTIVE_TORCH: object | None = None


def install_torch_guard(
    torch_module: object, *, reservation_id: str, max_rows: int, evidence_path: Path,
) -> SamplingGuard:
    """Patch Torch's multinomial entry point once per gated unittest process."""
    global _ACTIVE_GUARD, _ACTIVE_TORCH
    if _ACTIVE_GUARD is not None:
        if (_ACTIVE_TORCH is not torch_module
                or _ACTIVE_GUARD.reservation_id != reservation_id
                or _ACTIVE_GUARD.max_rows != max_rows
                or _ACTIVE_GUARD.evidence_path != Path(evidence_path)):
            raise RuntimeError("tensor test modules disagree about the active sample reservation")
        return _ACTIVE_GUARD
    original = getattr(torch_module, "multinomial", None)
    if not callable(original):
        raise TypeError("Torch module has no callable multinomial function")
    guard = SamplingGuard(
        reservation_id=reservation_id, max_rows=max_rows, evidence_path=evidence_path
    )
    setattr(torch_module, "multinomial", guard.wrap(original))
    _ACTIVE_GUARD = guard
    _ACTIVE_TORCH = torch_module
    return guard


def activate_from_environment() -> SamplingGuard:
    """Load Torch only after the explicit tensor-test opt-in has been set."""
    if os.environ.get("SPIRE_LEARN_RUN_NEURAL_TENSOR") != "1":
        raise RuntimeError("tensor test gate is not enabled")
    task = os.environ.get("SPIRE_LEARN_TENSOR_TASK")
    if task not in _TASKS:
        raise RuntimeError("SPIRE_LEARN_TENSOR_TASK must be task4 or task6")
    reservation_id = os.environ.get("SPIRE_LEARN_TENSOR_RESERVATION_ID", "")
    max_rows_text = os.environ.get("SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS", "")
    evidence_path = os.environ.get("SPIRE_LEARN_TENSOR_EVIDENCE_PATH", "")
    if max_rows_text != str(MAX_INVOCATION_ROWS):
        raise RuntimeError("each tensor-test invocation must enforce its full 64-row reservation")
    if not evidence_path:
        raise RuntimeError("SPIRE_LEARN_TENSOR_EVIDENCE_PATH is required")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(f"opt-in tensor runtime is unavailable: {error}") from error
    torch.set_num_threads(4)
    return install_torch_guard(
        torch,
        reservation_id=reservation_id,
        max_rows=int(max_rows_text),
        evidence_path=Path(evidence_path),
    )


def active_guard() -> SamplingGuard:
    if _ACTIVE_GUARD is None:
        raise RuntimeError("no tensor sampling guard is installed")
    return _ACTIVE_GUARD


def _finish_from_evidence(ledger_path: Path, reservation_id: str, requested_status: str,
                          evidence_path: Path) -> dict[str, int]:
    if requested_status not in {"not_started", "succeeded", "failed", "uncertain"}:
        raise ValueError("invalid requested completion status")
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("reservation_id") != reservation_id or evidence.get("reserved_rows") != MAX_INVOCATION_ROWS:
            raise ValueError("sample evidence does not match its durable reservation")
        attempted = evidence.get("attempted_rows")
        completed = evidence.get("completed_rows")
        rejected = evidence.get("rejected_rows")
        evidence_status = evidence.get("status")
        if requested_status == "uncertain":
            attempted = evidence.get("attempted_rows")
            completed = None
        elif evidence_status in {"sample_started", "sample_failed"}:
            requested_status = "uncertain"
            attempted = evidence.get("attempted_rows")
            completed = None
        elif requested_status == "succeeded" and evidence_status == "budget_rejected":
            requested_status = "failed"
    elif requested_status == "not_started":
        attempted = completed = rejected = 0
    else:
        requested_status = "uncertain"
        attempted = completed = rejected = None
    return complete_reservation(
        reservation_id=reservation_id,
        status=requested_status,
        attempted_rows=attempted,
        completed_rows=completed,
        rejected_rows=0 if rejected is None else rejected,
        path=ledger_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve_parser = commands.add_parser("reserve")
    reserve_parser.add_argument("--task", choices=sorted(_TASKS), required=True)
    reserve_parser.add_argument("--reservation-id", required=True)
    reserve_parser.add_argument("--rows", type=int, default=MAX_INVOCATION_ROWS)
    finish_parser = commands.add_parser("finish")
    finish_parser.add_argument("--reservation-id", required=True)
    finish_parser.add_argument("--status", choices=("not_started", "succeeded", "failed", "uncertain"), required=True)
    finish_parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "reserve":
            result = reserve_rows(
                task=args.task,
                reservation_id=args.reservation_id,
                rows=args.rows,
                path=args.ledger,
            )
        else:
            evidence = args.evidence or Path("/path/that/does/not/exist")
            result = _finish_from_evidence(
                args.ledger, args.reservation_id, args.status, evidence
            )
    except (OSError, ValueError, SampleBudgetError) as error:
        print(f"sample ledger refused operation: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
