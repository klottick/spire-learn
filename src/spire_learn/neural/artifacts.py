"""Small, strict JSON artifacts for neural training and evaluation runs.

This module is dependency-free. It deliberately accepts JSON primitives and
containers only; neural records are projected by their callers rather than
serialized from arbitrary dataclasses or live model/environment objects.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from spire_learn.artifacts.codec import canonical_fingerprint
from spire_learn.artifacts.provenance import collect_provenance


RUN_SCHEMA_ID = "spire-learn-neural-run-v1"
MAX_RUN_ID_LENGTH = 64
MAX_EVENT_BYTES = 64 * 1024
MAX_EVENT_LOG_BYTES = 64 * 1024 * 1024
MAX_EVENT_COUNT = 1_000_000
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_RAW_EVENT_FIELDS = frozenset({
    "obs", "observation", "observations", "raw_observation", "raw_frame",
    "frame", "tensor", "tensors", "buffer", "frame_bytes",
})


def _json_value(value: object, *, event: bool = False) -> object:
    """Return a bounded strict-JSON projection without widening core codecs."""
    remaining = [100_000]

    def visit(item: object, depth: int) -> object:
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 32:
            raise ValueError("JSON value exceeds the neural artifact structure limit")
        if item is None or type(item) in (str, bool):
            return item
        if type(item) is int:
            if not -(2**64 - 1) <= item <= 2**64 - 1:
                raise ValueError("JSON integer exceeds the supported 64-bit range")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return item
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON mapping keys must be strings")
                if event and key.casefold() in _RAW_EVENT_FIELDS:
                    raise ValueError(f"event field {key!r} may contain a raw observation")
                if key in result:
                    raise ValueError(f"duplicate JSON mapping key: {key}")
                result[key] = visit(child, depth + 1)
            return result
        if isinstance(item, (tuple, list)):
            return [visit(child, depth + 1) for child in item]
        raise TypeError(f"unsupported JSON value type: {type(item).__name__}")

    return visit(value, 0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    hashes: dict[str, str] = {}
    for path in sorted(package_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise OSError(f"neural source is not a regular file: {path}")
        relative = path.relative_to(package_root).as_posix()
        hashes[relative] = _sha256_bytes(path.read_bytes())
    if not hashes:
        raise OSError("neural source hash inventory is empty")
    return hashes


def _write_exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class NeuralWriter:
    """Append bounded neural run records in a directory created exclusively."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._events = (self.run_dir / "events.jsonl").open("x", encoding="utf-8", newline="\n")
        self._started = False
        self._finalized = False
        self._closed = False
        self._failed = False
        self._event_count = 0
        self._event_bytes = 0

    @classmethod
    def create(cls, root: Path, run_id: str) -> "NeuralWriter":
        if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None:
            raise ValueError("unsafe run_id")
        root = Path(root)
        if root.is_symlink():
            raise ValueError("artifact root may not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        run_dir.mkdir()
        try:
            return cls(run_dir)
        except BaseException:
            try:
                run_dir.rmdir()
            except OSError:
                pass
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("neural writer is closed")
        if self._failed:
            raise RuntimeError("neural writer failed; no further writes are allowed")

    def start(self, config: Mapping[str, object], manifest: Mapping[str, object]) -> None:
        self._ensure_open()
        if self._started:
            raise RuntimeError("neural writer was already started")
        if not isinstance(config, Mapping) or not isinstance(manifest, Mapping):
            raise TypeError("config and manifest must be mappings")
        safe_config = _json_value(config)
        safe_manifest = _json_value(manifest)
        if not isinstance(safe_config, dict) or not isinstance(safe_manifest, dict):
            raise TypeError("config and manifest must project to JSON objects")

        provenance = collect_provenance(learner_root=_repository_root(), gym_checkout=None)
        config_record = {
            "schema_id": RUN_SCHEMA_ID,
            "record_type": "config",
            "config_sha256": canonical_fingerprint(safe_config),
            "config": safe_config,
        }
        manifest_record = {
            "schema_id": RUN_SCHEMA_ID,
            "record_type": "manifest",
            "manifest_sha256": canonical_fingerprint(safe_manifest),
            "code_sha256": _code_hashes(),
            "provenance": _json_value(provenance),
            "manifest": safe_manifest,
        }
        try:
            _write_exclusive_json(self.run_dir / "config.json", config_record)
            _write_exclusive_json(self.run_dir / "manifest.json", manifest_record)
        except BaseException:
            self._failed = True
            raise
        self._started = True

    def event(self, record: Mapping[str, object]) -> None:
        self._ensure_open()
        if not self._started:
            raise RuntimeError("neural writer must be started before appending events")
        if self._finalized:
            raise RuntimeError("neural writer is finalized")
        if not isinstance(record, Mapping):
            raise TypeError("event record must be a mapping")
        safe_record = _json_value(record, event=True)
        if not isinstance(safe_record, dict):
            raise TypeError("event record must project to a JSON object")
        envelope = {
            "schema_id": RUN_SCHEMA_ID,
            "record_sha256": canonical_fingerprint(safe_record),
            "record": safe_record,
        }
        line = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if encoded_size > MAX_EVENT_BYTES:
            raise ValueError(f"neural event exceeds {MAX_EVENT_BYTES} bytes")
        if self._event_count >= MAX_EVENT_COUNT:
            raise ValueError("neural event count exceeds its configured limit")
        if self._event_bytes + encoded_size > MAX_EVENT_LOG_BYTES:
            raise ValueError("neural event log exceeds its configured byte limit")
        try:
            self._events.write(line)
            self._events.flush()
        except BaseException:
            self._failed = True
            raise
        self._event_count += 1
        self._event_bytes += encoded_size

    def finalize(self, summary: Mapping[str, object]) -> None:
        self._ensure_open()
        if not self._started:
            raise RuntimeError("neural writer must be started before finalization")
        if self._finalized:
            raise RuntimeError("neural writer is already finalized")
        if not isinstance(summary, Mapping):
            raise TypeError("summary must be a mapping")
        safe_summary = _json_value(summary)
        if not isinstance(safe_summary, dict):
            raise TypeError("summary must project to a JSON object")
        value = {
            "schema_id": RUN_SCHEMA_ID,
            "record_type": "summary",
            "summary_sha256": canonical_fingerprint(safe_summary),
            "summary": safe_summary,
        }
        temporary: Path | None = None
        try:
            self._events.flush()
            os.fsync(self._events.fileno())
            fd, name = tempfile.mkstemp(prefix=".summary-", suffix=".tmp", dir=self.run_dir)
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # A hard-link publishes atomically and fails if another writer placed
            # an unfamiliar summary at the destination in the meantime.
            os.link(temporary, self.run_dir / "summary.json", follow_symlinks=False)
            temporary.unlink()
            directory_fd = os.open(self.run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            self._failed = True
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            raise
        self._finalized = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        flush_error: BaseException | None = None
        try:
            self._events.flush()
            os.fsync(self._events.fileno())
        except BaseException as error:
            flush_error = error
        close_error: BaseException | None = None
        try:
            self._events.close()
        except BaseException as error:
            close_error = error

        def attach_secondary(primary: BaseException, stage: str, secondary: BaseException) -> None:
            primary.add_note(
                f"secondary {stage} failure: {type(secondary).__name__}: {secondary}"
            )

        primary_error: BaseException | None = (
            flush_error if flush_error is not None else close_error
        )
        secondary_error: BaseException | None = None
        secondary_stage = ""
        if flush_error is not None and close_error is not None:
            flush_is_control = not isinstance(flush_error, Exception)
            close_is_control = not isinstance(close_error, Exception)
            if close_is_control and not flush_is_control:
                primary_error = close_error
                secondary_error = flush_error
                secondary_stage = "flush/fsync"
            else:
                primary_error = flush_error
                secondary_error = close_error
                secondary_stage = "close"
        if primary_error is not None:
            self._failed = True
            if secondary_error is not None:
                attach_secondary(primary_error, secondary_stage, secondary_error)
            raise primary_error


__all__ = ["MAX_EVENT_BYTES", "MAX_EVENT_COUNT", "MAX_EVENT_LOG_BYTES", "NeuralWriter", "RUN_SCHEMA_ID"]
