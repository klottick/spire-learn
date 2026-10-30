"""Lazy outer runtime for the prepared Task 9 learning study.

This module is safe to import in the preparation environment. Runtime, model,
checkpoint and child-process boundaries are resolved only when an operation is
called. Live CLI entry always asks Task C's phase-context verifier first.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
from typing import Any


STUDY_RELATIVE = PurePosixPath("configs/neural/experiments/task9-v1/study.json")
SELECTION_OUTPUT_RELATIVE = PurePosixPath(
    "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
)
TRAINING_ORDER = ("standalone-a", "run-a", "standalone-b", "run-b")
TRAINING_SCRATCH_DIRS = (
    "tmp", "cache", "torch", "torch-extensions", "cuda", "triton", "home",
)
POLICY_ORDER = (
    "standalone-a", "run-a", "standalone-b", "run-b",
    "legal-random", "end-turn-last-v1",
)
SELECTION_ROWS = 192
SELECTION_RESERVATION = 256
SELECTION_RESERVED_MAX = SELECTION_ROWS * SELECTION_RESERVATION
SELECTION_OUTPUT_LIMIT = 2 * 1024**3
CHILD_STREAM_LIMIT = 64 * 1024
CHILD_CLOSE_GRACE_SECONDS = 60.0
MAX_JSON_BYTES = 8 * 1024 * 1024
_FORBIDDEN_REFERENCE_PREFIXES = ("torch", "tensordict", "torchrl")


@dataclass(frozen=True)
class RuntimeDependencies:
    """Narrow injection points for actual learner and child boundaries.

    ``factory`` has the ``create_runtime`` call shape. ``train`` and
    ``evaluate`` have the accepted runner call shapes. ``checkpoint_preflight``
    is the dependency-free ``preflight_checkpoint`` API. ``child_process`` has
    the bounded subprocess helper shape (argv plus timeout/output/progress
    keywords). The phase context and Python executable are only projected into
    fresh child argv; they do not bypass CLI verification.
    """

    factory: Callable[..., Any] | None = None
    train: Callable[..., Any] | None = None
    evaluate: Callable[..., Any] | None = None
    checkpoint_preflight: Callable[..., Any] | None = None
    child_process: Callable[..., Any] | None = None
    phase_context: Path | None = None
    python_executable: str | None = None

    def __post_init__(self) -> None:
        for name in ("factory", "train", "evaluate", "checkpoint_preflight", "child_process"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} dependency must be callable or None")
        if self.phase_context is not None and not isinstance(self.phase_context, Path):
            raise TypeError("phase_context must be pathlib.Path or None")
        if self.python_executable is not None and (
            not isinstance(self.python_executable, str) or not self.python_executable
        ):
            raise TypeError("python_executable must be non-empty text or None")


class RuntimeProtocolError(RuntimeError):
    """An actual runner result or study binding violates the Task 9 protocol."""


class ReferenceImportViolation(RuntimeProtocolError):
    """A stateless reference child attempted or already loaded a Torch package."""


class OutputAccountingError(RuntimeProtocolError):
    """Selection output is unsafe or exceeds its authored accounting ceiling."""


def _error_value(error: BaseException) -> dict[str, str]:
    try:
        message = str(error).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    except BaseException:
        message = "exception message unavailable"
    return {"exception_type": type(error).__name__, "message": message[:512]}


def _is_control(error: BaseException) -> bool:
    return not isinstance(error, Exception)


def _note(primary: BaseException, detail: str) -> None:
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(detail)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _read_json(path: Path, *, label: str, max_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, object], bytes]:
    _require_regular_file(path, label=label)
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"{label} exceeds its bounded JSON size")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value, raw


def _project_json(value: object) -> object:
    # Reuse the existing bounded immutable-record projection. Do not widen the
    # learner artifact codec to accept arbitrary dataclasses or tensor objects.
    from spire_learn.neural.artifacts import _json_value

    return _json_value(value)


def _mkdir_safe(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            cursor.mkdir()
            continue
        if cursor.is_symlink() or not cursor.is_dir():
            raise ValueError(f"owned output path has a symlink or non-directory component: {cursor}")


def _assert_safe_path(root: Path, path: Path, *, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository root") from error
    cursor = root
    for index, part in enumerate(relative.parts):
        if part in {"", ".", ".."}:
            raise ValueError(f"{label} contains an unsafe path component")
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if cursor.is_symlink():
            raise ValueError(f"{label} may not contain symlinks")
        if index < len(relative.parts) - 1 and not cursor.is_dir():
            raise ValueError(f"{label} has a non-directory parent")
    return path


def _root_path(root: Path) -> Path:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("repository root may not be a symlink")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("repository root must be a directory")
    return resolved


def _bound_path(root: Path, relative: str | PurePosixPath, *, label: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"{label} must be a safe repository-relative path")
    return _assert_safe_path(root, root.joinpath(*value.parts), label=label)


def _caller_path(root: Path, supplied: Path, *, expected: Path, label: str) -> Path:
    value = Path(supplied)
    if value.is_absolute():
        candidate = Path(os.path.abspath(value))
    else:
        if any(part == ".." for part in value.parts):
            raise ValueError(f"{label} may not traverse parent directories")
        candidate = Path(os.path.abspath(root / value))
    _assert_safe_path(root, candidate, label=label)
    if candidate != expected:
        raise ValueError(f"{label} must use its exact authored output slot")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    _require_regular_file(path, label="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_package(root: Path):
    source = _bound_path(root, "scripts/neural/task9_study_package.py", label="Task A validator")
    _require_regular_file(source, label="Task A validator")
    for import_root in (str(root), str(root / "src")):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
    module_name = "_spire_learn_task9_study_package_" + hashlib.sha256(
        os.fsencode(str(root))
    ).hexdigest()[:16]
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load the authored Task A study validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return module


def _validate_study(root: Path, study_path: Path) -> tuple[Path, dict[str, object], Path, bytes]:
    root_path = _root_path(root)
    study_file = _bound_path(root_path, STUDY_RELATIVE, label="Task 9 study")
    supplied = Path(study_path)
    if supplied.is_absolute():
        supplied_path = Path(os.path.abspath(supplied))
    else:
        if any(part == ".." for part in supplied.parts):
            raise ValueError("study path may not traverse parent directories")
        supplied_path = Path(os.path.abspath(root_path / supplied))
    if supplied_path != study_file:
        raise ValueError("study_path must name the authored Task 9 study.json")
    package = _load_task_package(root_path)
    validated = package.validate_study(root_path, study_file)
    if not isinstance(validated, dict):
        raise TypeError("Task A validate_study must return its validated study mapping")
    _, raw = _read_json(study_file, label="validated Task 9 study")
    return root_path, validated, study_file, raw


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{label} must be a regular non-symlink directory")


def _now(clock: Callable[[], float]) -> float:
    if not callable(clock):
        raise TypeError("clock must be callable")
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("monotonic clock must return a finite primitive number")
    return float(value)


def _deadline(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("deadline must be a finite primitive monotonic timestamp")
    return float(value)


def _repo_relative(root: Path, path: Path, *, label: str) -> str:
    safe = _assert_safe_path(root, Path(os.path.abspath(path)), label=label)
    return safe.relative_to(root).as_posix()


def _training_record(study: Mapping[str, object], policy_id: str) -> dict[str, object]:
    training = study.get("training")
    if not isinstance(training, Mapping) or not isinstance(training.get("runs"), list):
        raise ValueError("validated study training inventory is malformed")
    matches = [item for item in training["runs"]
               if isinstance(item, Mapping) and item.get("policy_id") == policy_id]
    if policy_id not in TRAINING_ORDER or len(matches) != 1:
        raise ValueError(f"unknown or ambiguous training policy_id {policy_id!r}")
    return dict(matches[0])


def _schedule_row(study: Mapping[str, object], operation_id: str) -> dict[str, object]:
    selection = study.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("schedule"), list):
        raise ValueError("validated study selection schedule is malformed")
    rows = [row for row in selection["schedule"]
            if isinstance(row, Mapping) and row.get("operation_id") == operation_id]
    if len(rows) != 1:
        raise ValueError(f"unknown or ambiguous selection operation_id {operation_id!r}")
    return dict(rows[0])


def _load_config(root: Path, relative: str):
    from spire_learn.neural.config import load_neural_config

    config_path = _bound_path(root, relative, label="bound neural config")
    _require_regular_file(config_path, label="bound neural config")
    return load_neural_config(config_path)


def _exact_output_root(root: Path, relative: str, config: Any) -> Path:
    expected = _bound_path(root, relative, label="authored output root")
    actual = Path(os.path.abspath(config.output_root))
    if actual != expected:
        raise ValueError("loaded config output_root differs from the authored output slot")
    _assert_safe_path(root, expected, label="authored output root")
    return expected


def _prepare_training_scratch(root: Path, output_root: Path) -> dict[str, object]:
    scratch_root = output_root / ".runtime"
    _assert_safe_path(root, scratch_root, label="training runtime scratch root")
    if scratch_root.exists() or scratch_root.is_symlink():
        raise FileExistsError("training runtime scratch already exists; replay is forbidden")
    scratch_root.mkdir()
    directories: dict[str, str] = {}
    for name in TRAINING_SCRATCH_DIRS:
        path = scratch_root / name
        _assert_safe_path(root, path, label=f"training runtime scratch {name}")
        path.mkdir()
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise RuntimeProtocolError(f"training runtime scratch {name} is not an empty ordinary directory")
        directories[name] = _repo_relative(root, path, label=f"training runtime scratch {name}")
    if ({path.name for path in scratch_root.iterdir()} != set(TRAINING_SCRATCH_DIRS)
            or scratch_root.is_symlink() or not scratch_root.is_dir()):
        raise RuntimeProtocolError("training runtime scratch root contains unexpected entries")
    return {
        "root": _repo_relative(root, scratch_root, label="training runtime scratch root"),
        "directories": directories,
    }


def _canonical_fingerprint(value: object) -> str:
    from spire_learn.artifacts.codec import canonical_fingerprint

    return canonical_fingerprint(value)


def _write_result_preserving_control(
    path: Path, result: Mapping[str, object], primary: BaseException | None, *,
    context: str = "result",
):
    try:
        return _exclusive_json(path, result)
    except BaseException as error:
        if primary is None:
            raise
        if _is_control(primary):
            _note(primary, f"{context} publication also failed: {type(error).__name__}: {_error_value(error)['message']}")
            raise primary
        if _is_control(error):
            _note(error, f"{context} publication followed prior {type(primary).__name__}: {_error_value(primary)['message']}")
            raise error
        _note(primary, f"{context} publication also failed: {type(error).__name__}: {_error_value(error)['message']}")
        raise primary


def _make_directories_for(path: Path) -> None:
    parent = path.parent
    anchor = parent
    while not anchor.exists() and not anchor.is_symlink():
        if anchor.parent == anchor:
            raise OSError("cannot find an existing output ancestor")
        anchor = anchor.parent
    if anchor.is_symlink() or not anchor.is_dir():
        raise ValueError(f"output ancestor is not a safe directory: {anchor}")
    anchor = anchor.resolve(strict=True)
    _mkdir_safe(anchor, parent)


def _exclusive_json(path: Path, value: Mapping[str, object]) -> bytes:
    _make_directories_for(path)
    safe = _project_json(value)
    if not isinstance(safe, dict):
        raise TypeError("exclusive JSON result must project to an object")
    encoded = (json.dumps(
        safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return encoded


def _resolve_dependency(dependencies: RuntimeDependencies | None, name: str, default):
    if dependencies is not None:
        selected = getattr(dependencies, name)
        if selected is not None:
            return selected
    return default


def _actual_train(config, *, runtime, clock):
    from spire_learn.neural.runner import train

    return train(config, runtime=runtime, clock=clock)


def _actual_evaluate(config, *, runtime, clock, allow_final=False):
    from spire_learn.neural.runner import evaluate_policy

    return evaluate_policy(config, runtime=runtime, clock=clock, allow_final=allow_final)


def _default_factory(
    config, *, purpose: str, checkpoint: Path | None, policy_kind: str,
    action_selection: str,
):
    """Assemble the accepted runtime after the relevant device/checkpoint gate."""
    from spire_learn.neural.device import plan_runtime_device, require_runtime_execution_gate
    from spire_learn.neural.evaluation import REFERENCE_POLICIES
    from spire_learn.neural.runner import create_runtime

    if purpose == "evaluate" and policy_kind in REFERENCE_POLICIES:
        # create_runtime's reference branch has no Torch import and is kept that way.
        return create_runtime(
            config, purpose=purpose, checkpoint=None, policy_kind=policy_kind,
            action_selection=action_selection,
        )

    device_plan = plan_runtime_device(config)
    # In particular, the approved GPU gate precedes importing Torch for training.
    require_runtime_execution_gate(device_plan)
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Task 9 requires Torch intra-op/inter-op threads 4/1")
    runtime = create_runtime(
        config, purpose=purpose, checkpoint=checkpoint, policy_kind=policy_kind,
        action_selection=action_selection,
    )
    try:
        intra = torch.get_num_threads()
        interop = torch.get_num_interop_threads()
        if intra != 4 or interop != 1:
            raise RuntimeError("Torch thread settings changed during Task 9 runtime assembly")
        runtime.task9_thread_settings = {
            "intraop_threads": intra,
            "interop_threads": interop,
            "process_id": os.getpid(),
            "verified_before_runtime_assembly": True,
            "verified_after_runtime_assembly": True,
        }
    except BaseException as error:
        primary = _close_preserving(error, runtime, "runtime thread verification")
        assert primary is not None
        raise primary
    return runtime


def _runtime_thread_evidence(runtime: Any) -> dict[str, object] | None:
    value = getattr(runtime, "task9_thread_settings", None)
    if value is None:
        return None
    if (not isinstance(value, Mapping)
            or type(value.get("intraop_threads")) is not int or value.get("intraop_threads") != 4
            or type(value.get("interop_threads")) is not int or value.get("interop_threads") != 1
            or type(value.get("process_id")) is not int
            or value.get("verified_before_runtime_assembly") is not True
            or value.get("verified_after_runtime_assembly") is not True):
        raise RuntimeProtocolError("default runtime thread evidence is malformed")
    return dict(value)


def _set_operation_deadline(runtime: Any, deadline: float) -> None:
    from spire_learn.neural.runner import RuntimeOperationLimits, _operation_limits

    current = _operation_limits(runtime)
    runtime.operation_limits = RuntimeOperationLimits(
        max_updates=current.max_updates, monotonic_deadline=deadline,
    )


def _close_preserving(primary: BaseException | None, resource: Any, label: str) -> BaseException | None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return primary
    try:
        close()
    except BaseException as error:
        if primary is None:
            return error
        if _is_control(error) and not _is_control(primary):
            _note(error, f"{label} close followed {type(primary).__name__}: {_error_value(primary)['message']}")
            return error
        _note(primary, f"{label} close also failed: {type(error).__name__}: {_error_value(error)['message']}")
    return primary


def _report_files(root: Path, config: Any, report: Any) -> dict[str, object]:
    from spire_learn.neural.runner import NeuralReport

    if not isinstance(report, NeuralReport):
        raise TypeError("accepted runner must return NeuralReport")
    summary = _project_json(report.summary)
    if not isinstance(summary, dict):
        raise TypeError("NeuralReport.summary must project to a JSON object")
    info: dict[str, object] = {
        "report": summary,
        "runner_exit_code": report.exit_code,
        "run_id": None,
        "run_directory": None,
        "summary_path": None,
        "summary_sha256": None,
        "lifecycle_path": None,
        "lifecycle_sha256": None,
        "lifecycle": None,
    }
    if report.run_dir is None:
        if summary.get("run_directory") is not None:
            raise RuntimeProtocolError("runner summary declares a run directory but report.run_dir is missing")
        return info
    run_dir = Path(report.run_dir)
    _assert_safe_path(root, Path(os.path.abspath(run_dir)), label="runner run_dir")
    if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.parent != config.output_root:
        raise RuntimeProtocolError("runner run_dir is not a fresh child of the authored output root")
    if summary.get("run_directory") != str(run_dir):
        raise RuntimeProtocolError("runner summary and NeuralReport run_dir identities disagree")
    summary_path = run_dir / "summary.json"
    lifecycle_path = run_dir / "lifecycle.json"
    summary_doc, summary_raw = _read_json(summary_path, label="runner summary.json")
    lifecycle, lifecycle_raw = _read_json(lifecycle_path, label="runner lifecycle.json")
    if (summary_doc.get("schema_id") != "spire-learn-neural-run-v1"
            or summary_doc.get("record_type") != "summary"
            or not isinstance(summary_doc.get("summary"), dict)):
        raise RuntimeProtocolError("runner summary.json has an unsupported envelope")
    provisional = summary_doc["summary"]
    if summary_doc.get("summary_sha256") != _canonical_fingerprint(provisional):
        raise RuntimeProtocolError("runner summary.json canonical digest mismatch")
    pending = provisional.get("lifecycle")
    if (not isinstance(pending, Mapping) or pending.get("status") != "receipt_pending"
            or pending.get("exit_code") != 1):
        raise RuntimeProtocolError("persisted runner summary is not the accepted provisional receipt")
    for key, expected in (
        ("operation", config.operation), ("mode", config.mode), ("split", config.split),
        ("config_sha256", config.config_sha256), ("manifest_sha256", config.manifest_sha256),
        ("environment_revision", config.environment_revision),
    ):
        if summary.get(key) != expected or provisional.get(key) != expected:
            raise RuntimeProtocolError(f"runner report {key} differs from the selected config")
    if (lifecycle.get("schema_id") != "spire-learn-neural-lifecycle-v1"
            or lifecycle.get("receipt_path") != "lifecycle.json"
            or type(lifecycle.get("exit_code")) is not int
            or lifecycle.get("status") != summary.get("status")
            or lifecycle.get("exit_code") != report.exit_code
            or summary.get("exit_code") != report.exit_code):
        raise RuntimeProtocolError("authoritative lifecycle.json disagrees with the returned NeuralReport")
    if summary.get("lifecycle") != lifecycle:
        raise RuntimeProtocolError("returned NeuralReport lifecycle differs from authoritative lifecycle.json")
    info.update({
        "run_id": run_dir.name,
        "run_directory": _repo_relative(root, run_dir, label="runner run_dir"),
        "summary_path": _repo_relative(root, summary_path, label="summary path"),
        "summary_sha256": _sha256_bytes(summary_raw),
        "lifecycle_path": _repo_relative(root, lifecycle_path, label="lifecycle path"),
        "lifecycle_sha256": _sha256_bytes(lifecycle_raw),
        "lifecycle": lifecycle,
    })
    return info


_SAVED_PROGRESS_FIELDS = (
    "attempted_environment_slots", "forwarded_environment_slots",
    "known_accepted_environment_slots", "delivered_environment_slots",
    "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
    "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
    "learned_decisions", "fixed_decisions", "optimizer_attempted_steps",
    "optimizer_accepted_steps", "optimizer_uncertain_steps", "update_count",
    "incomplete_decisions",
)


def _expected_saved_progress(report: Mapping[str, object]) -> dict[str, int]:
    collection = report.get("collection")
    optimizer = report.get("optimizer")
    if not isinstance(collection, Mapping) or not isinstance(optimizer, Mapping):
        raise RuntimeProtocolError("training report omits collection or optimizer counters")
    result: dict[str, int] = {}
    for name in _SAVED_PROGRESS_FIELDS:
        source = optimizer if name.startswith("optimizer_") or name == "update_count" else collection
        key = {
            "optimizer_attempted_steps": "attempted_steps",
            "optimizer_accepted_steps": "accepted_steps",
            "optimizer_uncertain_steps": "uncertain_steps",
            "update_count": "updates",
        }.get(name, name)
        value = source.get(key)
        if type(value) is not int or value < 0:
            raise RuntimeProtocolError(f"training report counter {name} is not a non-negative integer")
        result[name] = value
    return result


def _selected_identity(config: Any):
    matches = [identity for identity in config.accepted_identities
               if config.mode in identity.accepted_raw_mode_mappings]
    if len(matches) != 1:
        raise RuntimeProtocolError("selected config must resolve exactly one accepted learner identity")
    return matches[0]


def _checkpoint_inventory(
    root: Path, path: Path, config: Any, preflight_function: Callable[..., Any], *,
    expected_inventory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from spire_learn.neural.checkpoint import CheckpointPreflight, identity_payload

    checkpoint_path = Path(os.path.abspath(path))
    _assert_safe_path(root, checkpoint_path, label="checkpoint directory")
    preflight = preflight_function(checkpoint_path, _selected_identity(config), purpose="evaluate")
    if not isinstance(preflight, CheckpointPreflight):
        raise TypeError("checkpoint preflight must return the immutable CheckpointPreflight contract")
    if preflight.path != checkpoint_path or preflight.purpose != "evaluate":
        raise RuntimeProtocolError("checkpoint preflight returned a different path or purpose")
    metadata = _project_json(preflight.metadata)
    if not isinstance(metadata, dict):
        raise TypeError("CheckpointPreflight.metadata must project to an object")
    expected_identity = _project_json(identity_payload(_selected_identity(config)))
    if metadata.get("identity") != expected_identity:
        raise RuntimeProtocolError("checkpoint accepted learner identity differs from the selected config")
    run_metadata = metadata.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        raise RuntimeProtocolError("checkpoint metadata omits run_metadata")
    inventory_raw = metadata.get("payload_inventory")
    if not isinstance(inventory_raw, Mapping) or set(inventory_raw) != {"model.pt", "training.pt"}:
        raise RuntimeProtocolError("Task 9 frozen checkpoint must bind model.pt and training.pt")
    names = {"metadata.json", *inventory_raw.keys()}
    if path.is_symlink() or not path.is_dir():
        raise RuntimeProtocolError("checkpoint directory must be a regular non-symlink directory")
    children = list(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise RuntimeProtocolError("checkpoint directory contains an unsafe child")
    if {child.name for child in children} != names:
        raise RuntimeProtocolError("checkpoint directory inventory differs from metadata")
    inventory: dict[str, object] = {}
    for name in sorted(names):
        child = path / name
        _require_regular_file(child, label=f"checkpoint {name}")
        inventory[name] = {"size_bytes": child.stat().st_size, "sha256": _sha256_file(child)}
    if expected_inventory is not None and dict(expected_inventory) != inventory:
        raise RuntimeProtocolError("checkpoint raw file inventory changed from its frozen binding")
    if preflight.model_path != path / "model.pt" or preflight.training_path != path / "training.pt":
        raise RuntimeProtocolError("CheckpointPreflight payload paths disagree with the exact inventory")
    tensors = metadata.get("model_tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RuntimeProtocolError("checkpoint metadata omits the named model tensor descriptors")
    return {
        "directory": _repo_relative(root, path, label="checkpoint directory"),
        "inventory": inventory,
        "model_tensors": tensors,
        "metadata": metadata,
        "purpose": preflight.purpose,
        "stored_bytes": preflight.stored_bytes,
        "uncompressed_bytes": preflight.uncompressed_bytes,
        "logical_tensor_bytes": preflight.logical_tensor_bytes,
    }


def _validate_training_checkpoint_prefix(checkpoint: Mapping[str, object], report: Mapping[str, object], lifecycle: Mapping[str, object], config: Any) -> None:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeProtocolError("checkpoint inventory omitted verified metadata")
    training_state = metadata.get("training_state")
    run_metadata = metadata.get("run_metadata")
    if not isinstance(training_state, Mapping) or not isinstance(run_metadata, Mapping):
        raise RuntimeProtocolError("final checkpoint lacks training state or run metadata")
    saved = training_state.get("counters")
    if not isinstance(saved, Mapping) or set(saved) != {*_SAVED_PROGRESS_FIELDS, "elapsed_seconds"}:
        raise RuntimeProtocolError("checkpoint saved counters do not match the accepted progress schema")
    expected = _expected_saved_progress(report)
    for name, value in expected.items():
        if type(saved.get(name)) is not int or saved[name] != value:
            raise RuntimeProtocolError(f"checkpoint saved counter {name} differs from the actual report")
    elapsed = saved.get("elapsed_seconds")
    lifecycle_elapsed = lifecycle.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0
            or type(lifecycle_elapsed) not in (int, float)
            or not math.isfinite(float(lifecycle_elapsed)) or lifecycle_elapsed < 0
            or elapsed > lifecycle_elapsed):
        raise RuntimeProtocolError("checkpoint elapsed prefix is not an earlier lifecycle boundary")
    if _project_json(run_metadata.get("cumulative_prefix")) != _project_json(saved):
        raise RuntimeProtocolError("checkpoint run_metadata and training_state prefixes disagree")
    if training_state.get("optimizer_step") != expected["optimizer_accepted_steps"]:
        raise RuntimeProtocolError("checkpoint optimizer_step differs from saved accepted steps")
    expected_run = {
        "mode": config.mode, "operation": "train", "split": "train", "partition": "train",
        "config_sha256": config.config_sha256, "manifest_sha256": config.manifest_sha256,
        "environment_revision": config.environment_revision,
    }
    for name, value in expected_run.items():
        if run_metadata.get(name) != value:
            raise RuntimeProtocolError(f"checkpoint run_metadata {name} differs from the training config")


def _training_count_projection(report: Mapping[str, object] | None) -> dict[str, object]:
    collection_names = (
        "attempted_environment_slots", "forwarded_environment_slots",
        "known_accepted_environment_slots", "delivered_environment_slots",
        "uncertain_environment_slots", "attempted_reset_slots",
        "forwarded_reset_slots", "known_accepted_reset_slots",
        "delivered_reset_slots", "uncertain_reset_slots",
    )
    optimizer_names = (
        "optimizer_attempted_steps", "optimizer_accepted_steps",
        "optimizer_uncertain_steps", "update_count",
    )
    if report is None:
        return {name: None for name in (*collection_names, *optimizer_names)}
    collection = report.get("collection")
    optimizer = report.get("optimizer")
    values = {
        name: collection.get(name) if isinstance(collection, Mapping) else None
        for name in collection_names
    }
    optimizer_values = {
        "optimizer_attempted_steps": optimizer.get("attempted_steps") if isinstance(optimizer, Mapping) else None,
        "optimizer_accepted_steps": optimizer.get("accepted_steps") if isinstance(optimizer, Mapping) else None,
        "optimizer_uncertain_steps": optimizer.get("uncertain_steps") if isinstance(optimizer, Mapping) else None,
        "update_count": optimizer.get("updates") if isinstance(optimizer, Mapping) else None,
    }
    values.update({
        name: value if type(value) is int and value >= 0 else None
        for name, value in optimizer_values.items()
    })
    return {
        **{
            name: value if type(value) is int and value >= 0 else None
            for name, value in values.items() if name not in optimizer_names
        },
        **{name: values[name] for name in optimizer_names},
    }


def _attempted_effects(
    counts: Mapping[str, object] | None, *, optimizer: Mapping[str, object] | None = None,
    runner_handed_off: bool,
) -> bool | None:
    collection_names = (
        "attempted_environment_slots", "forwarded_environment_slots",
        "known_accepted_environment_slots", "delivered_environment_slots",
        "uncertain_environment_slots", "attempted_reset_slots",
        "forwarded_reset_slots", "known_accepted_reset_slots",
        "delivered_reset_slots", "uncertain_reset_slots",
    )
    values = [counts.get(name) if isinstance(counts, Mapping) else None
              for name in collection_names]
    optimizer_names = (
        ("optimizer_attempted_steps", "attempted_steps"),
        ("optimizer_accepted_steps", "accepted_steps"),
        ("optimizer_uncertain_steps", "uncertain_steps"),
    )
    for receipt_name, report_name in optimizer_names:
        if isinstance(counts, Mapping) and receipt_name in counts:
            values.append(counts.get(receipt_name))
        else:
            values.append(optimizer.get(report_name) if isinstance(optimizer, Mapping) else None)
    if isinstance(counts, Mapping) and "update_count" in counts:
        values.append(counts.get("update_count"))
    else:
        values.append(optimizer.get("updates") if isinstance(optimizer, Mapping) else None)
    if any(type(value) is int and value > 0 for value in values):
        return True
    if all(type(value) is int and value == 0 for value in values):
        return False
    return None if runner_handed_off else False


def _training_receipt_base(
    root: Path, study: Mapping[str, object], study_raw: bytes,
    record: Mapping[str, object], config: Any, deadline: float,
) -> dict[str, object]:
    manifest = study["training"]["manifest"]
    seeds = {
        "policy": config.seed_streams.policy,
        "optimizer": config.seed_streams.optimizer,
        "curriculum": config.seed_streams.curriculum,
        "noncombat": config.seed_streams.noncombat,
    }
    if seeds != record["seeds"]:
        raise RuntimeProtocolError("loaded training seed streams differ from the validated study")
    config_relative = record["config"]["path"]
    return {
        "schema_id": "spire-learn-task9-training-operation-v1",
        "policy_id": record["policy_id"],
        "mode": record["mode"],
        "study_raw_sha256": _sha256_bytes(study_raw),
        "config": {
            "path": config_relative,
            "canonical_sha256": config.config_sha256,
            "raw_sha256": record["config"]["raw_sha256"],
        },
        "manifest": {
            "path": manifest["path"],
            "canonical_sha256": config.manifest_sha256,
            "raw_sha256": manifest["raw_sha256"],
        },
        "seed_evidence": seeds,
        "rollout": dict(record["rollout"]),
        "environment_revision": config.environment_revision,
        "budgets": {
            "learned_decisions": config.budgets.max_training_decisions,
            "environment_step_slots": config.budgets.max_environment_actions,
            "optimizer_attempts": config.budgets.max_optimizer_steps,
            "soft_wall_seconds": config.budgets.max_wall_seconds,
            "shared_deadline_monotonic": deadline,
        },
        "outer_timing": {
            "start_monotonic": None,
            "finish_monotonic": None,
            "elapsed_seconds": None,
            "timing_boundary": "outer operation before factory through pre-publication finalization",
        },
        "run_id": None,
        "run_directory": None,
        "runner_exit_code": None,
        "report": None,
        "lifecycle": None,
        "summary_path": None,
        "summary_sha256": None,
        "lifecycle_path": None,
        "lifecycle_sha256": None,
        "checkpoint": None,
        "runtime_thread_settings": None,
        "runtime_scratch": None,
        "counts": _training_count_projection(None),
        "attempted_effects": False,
        "close_status": "not_acquired",
        "transport_exit_code": 1,
    }


def _training_status(
    info: Mapping[str, object], checkpoint: Mapping[str, object] | None,
) -> tuple[str, str | None]:
    report = info.get("report")
    lifecycle = info.get("lifecycle")
    if not isinstance(report, Mapping) or not isinstance(lifecycle, Mapping):
        return "operational_failure", "runner_report_or_lifecycle_missing"
    optimizer = report.get("optimizer")
    if not isinstance(optimizer, Mapping):
        return "operational_failure", "optimizer_evidence_missing"
    accepted = optimizer.get("accepted_steps")
    attempted = optimizer.get("attempted_steps")
    uncertain = optimizer.get("uncertain_steps")
    updates = optimizer.get("updates")
    if any(type(value) is not int or value < 0 for value in (accepted, attempted, uncertain, updates)):
        return "operational_failure", "optimizer_evidence_malformed"
    close_errors = lifecycle.get("close_errors")
    if (lifecycle.get("status") in {"operational_failure", "artifact_failure", "interrupted"}
            or not isinstance(close_errors, list) or close_errors
            or report.get("fault") is not None or uncertain != 0 or attempted != accepted):
        return "operational_failure", "training_lifecycle_or_optimizer_fault"
    if lifecycle.get("status") != "complete" or report.get("status") != "complete":
        return "unavailable", "training_lifecycle_not_complete"
    if accepted == 0 or updates == 0:
        return "unavailable", "zero_accepted_optimizer_updates"
    if checkpoint is None:
        return "operational_failure", "positive_update_checkpoint_missing_or_invalid"
    if info.get("runner_exit_code") != 0:
        return "operational_failure", "complete_training_report_has_nonzero_exit"
    return "eligible", None


def run_training_operation(
    root: Path, study_path: Path, policy_id: str, receipt_path: Path, *,
    deadline: float, clock: Callable[[], float] = time.monotonic,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, object]:
    """Run one fresh training operation with a shared absolute soft deadline."""
    deadline_value = _deadline(deadline)
    first_clock = _now(clock)
    root_path, study, study_file, study_raw = _validate_study(root, study_path)
    record = _training_record(study, policy_id)
    config = _load_config(root_path, record["config"]["path"])
    if (config.operation, config.split, config.partition, config.mode) != (
        "train", "train", "train", record["mode"],
    ):
        raise RuntimeProtocolError("training config is not a fresh authored train operation")
    output_root = _exact_output_root(root_path, record["output_root"], config)
    expected_receipt = output_root / "operation-receipt.json"
    receipt_file = _caller_path(
        root_path, Path(receipt_path), expected=expected_receipt, label="training receipt path",
    )
    reservation_file = output_root / "operation-reservation.json"
    _assert_safe_path(root_path, reservation_file, label="training reservation path")
    _require_directory(output_root, label="training output root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("training output slot is not empty; replacement or retry is forbidden")
    if reservation_file.exists() or reservation_file.is_symlink() or receipt_file.exists() or receipt_file.is_symlink():
        raise FileExistsError("training operation already has an exclusive reservation or receipt")
    base = _training_receipt_base(root_path, study, study_raw, record, config, deadline_value)
    reservation = {
        "schema_id": "spire-learn-task9-training-reservation-v1",
        "policy_id": policy_id,
        "config_raw_sha256": record["config"]["raw_sha256"],
        "study_raw_sha256": _sha256_bytes(study_raw),
        "deadline_monotonic": deadline_value,
        "state": "reserved_once",
    }
    _exclusive_json(reservation_file, reservation)

    setup_clock = _now(clock)
    if first_clock >= deadline_value or setup_clock >= deadline_value:
        base.update({
            "status": "not_started",
            "reason": "shared_soft_deadline_exhausted_before_factory",
            "transport_exit_code": 0,
            "outer_timing": {
                "start_monotonic": first_clock,
                "finish_monotonic": setup_clock,
                "elapsed_seconds": max(0.0, setup_clock - first_clock),
            "timing_boundary": "outer operation before factory through pre-publication finalization",
            },
        })
        _exclusive_json(receipt_file, base)
        return base

    factory = _resolve_dependency(dependencies, "factory", _default_factory)
    trainer = _resolve_dependency(dependencies, "train", _actual_train)
    runtime = None
    handed_to_runner = False
    primary: BaseException | None = None
    info: dict[str, object] | None = None
    checkpoint: dict[str, object] | None = None
    status = "operational_failure"
    reason: str | None = None
    try:
        base["runtime_scratch"] = _prepare_training_scratch(root_path, output_root)
        runtime = factory(
            config, purpose="train", checkpoint=None, policy_kind="checkpoint",
            action_selection="seeded-sampling",
        )
        base["runtime_thread_settings"] = _runtime_thread_evidence(runtime)
        # Ownership starts immediately when the factory returns.
        _set_operation_deadline(runtime, deadline_value)
        handed_to_runner = True
        report = trainer(config, runtime=runtime, clock=clock)
        info = _report_files(root_path, config, report)
        base.update(info)
        report_summary = info["report"]
        assert isinstance(report_summary, Mapping)
        base["counts"] = _training_count_projection(report_summary)
        base["attempted_effects"] = _attempted_effects(
            base["counts"], runner_handed_off=True,
        )
        lifecycle = info.get("lifecycle")
        publication = report_summary.get("checkpoint_publication")
        run_dir_value = info.get("run_directory")
        if (isinstance(publication, Mapping) and publication.get("publication_state") == "published"
                and publication.get("purpose") == "final" and isinstance(run_dir_value, str)):
            checkpoint_path = root_path / run_dir_value / "checkpoint-final"
            preflight = _resolve_dependency(
                dependencies, "checkpoint_preflight", _default_checkpoint_preflight,
            )
            checkpoint = _checkpoint_inventory(root_path, checkpoint_path, config, preflight)
            if isinstance(lifecycle, Mapping):
                _validate_training_checkpoint_prefix(checkpoint, report_summary, lifecycle, config)
            base["checkpoint"] = checkpoint
        status, reason = _training_status(info, checkpoint)
    except BaseException as error:
        primary = error
        if runtime is not None and not handed_to_runner:
            primary = _close_preserving(primary, runtime, "runtime setup")
        if _is_control(primary):
            status, reason = "interrupted", "process_control_exception"
        else:
            status, reason = "operational_failure", "training_operation_error"
        base["error"] = _error_value(error)
        base["counts"] = _training_count_projection(
            info["report"] if info is not None and isinstance(info.get("report"), Mapping) else None
        )
        base["attempted_effects"] = _attempted_effects(
            base["counts"], runner_handed_off=handed_to_runner,
        )
    final_clock: float | None = None
    try:
        final_clock = _now(clock)
    except BaseException as error:
        if primary is None or (_is_control(error) and not _is_control(primary)):
            if primary is not None:
                _note(error, f"final clock followed {type(primary).__name__}: {_error_value(primary)['message']}")
            primary = error
        else:
            _note(primary, f"final clock also failed: {type(error).__name__}: {_error_value(error)['message']}")
        if _is_control(primary):
            status, reason = "interrupted", "final_clock_process_control"
        else:
            status, reason = "operational_failure", "final_clock_error"
        base["final_clock_error"] = _error_value(error)
    base["status"] = status
    base["unavailable_reason"] = reason if status == "unavailable" else None
    base["reason"] = reason
    base["outer_timing"] = {
        "start_monotonic": first_clock,
        "finish_monotonic": final_clock,
        "elapsed_seconds": max(0.0, final_clock - first_clock) if final_clock is not None else None,
        "timing_boundary": "outer operation before factory through pre-publication finalization",
    }
    base["close_status"] = (
        "clean" if isinstance(base.get("lifecycle"), Mapping)
        and not base["lifecycle"].get("close_errors") else
        "reported_failure" if isinstance(base.get("lifecycle"), Mapping) else
        "runner_owned_or_unknown"
    )
    base["transport_exit_code"] = 0 if status in {"eligible", "unavailable", "not_started"} else 1
    result = _project_json(base)
    assert isinstance(result, dict)
    receipt_published = False
    try:
        _write_result_preserving_control(
            receipt_file, result, primary, context="training receipt",
        )
        receipt_published = True
    except BaseException as error:
        primary = error
    if primary is not None and (_is_control(primary) or not receipt_published):
        raise primary
    return result


def _default_checkpoint_preflight(path: Path, expected: Any, *, purpose: str):
    from spire_learn.neural.checkpoint import preflight_checkpoint

    return preflight_checkpoint(path, expected, purpose=purpose)


def _checkpoint_binding_path(root: Path) -> Path:
    return root / SELECTION_OUTPUT_RELATIVE / "control" / "checkpoint-bindings.json"


def _training_receipt_path(root: Path, policy_id: str, study: Mapping[str, object]) -> Path:
    record = _training_record(study, policy_id)
    return root / PurePosixPath(record["output_root"]) / "operation-receipt.json"


def freeze_checkpoint_bindings(
    root: Path, study_path: Path, training_receipts: object,
    destination: Path,
) -> dict[str, object]:
    """Freeze each actual training receipt and its immutable final checkpoint."""
    root_path, study, _, study_raw = _validate_study(root, study_path)
    if not isinstance(training_receipts, Mapping) or set(training_receipts) != set(TRAINING_ORDER):
        raise ValueError("training_receipts must map exactly the four authored policy IDs to paths or None")
    binding_file = _caller_path(
        root_path, Path(destination), expected=_checkpoint_binding_path(root_path),
        label="checkpoint binding destination",
    )
    if binding_file.exists() or binding_file.is_symlink():
        raise FileExistsError("checkpoint binding record already exists; rebinding is forbidden")
    _, study_bytes = _read_json(root_path / STUDY_RELATIVE, label="Task 9 study")
    if study_bytes != study_raw:
        raise RuntimeProtocolError("Task 9 study changed during binding preparation")
    policies: list[dict[str, object]] = []
    stop_required: list[str] = []
    for policy_id in TRAINING_ORDER:
        expected_receipt = _training_receipt_path(root_path, policy_id, study)
        supplied = training_receipts[policy_id]
        if supplied is None:
            policies.append({
                "policy_id": policy_id, "status": "unavailable",
                "reason": "training_receipt_missing", "training_receipt": None,
                "checkpoint_directory": None, "inventory": None, "model_tensors": None,
            })
            continue
        receipt_file = _caller_path(
            root_path, Path(supplied), expected=expected_receipt,
            label=f"training receipt path for {policy_id}",
        )
        if not receipt_file.exists():
            policies.append({
                "policy_id": policy_id, "status": "unavailable",
                "reason": "training_receipt_missing", "training_receipt": {
                    "path": _repo_relative(root_path, receipt_file, label="training receipt"),
                    "raw_sha256": None, "run_id": None,
                },
                "checkpoint_directory": None, "inventory": None, "model_tensors": None,
            })
            continue
        receipt, receipt_raw = _read_json(receipt_file, label=f"training receipt {policy_id}")
        if (receipt.get("schema_id") != "spire-learn-task9-training-operation-v1"
                or receipt.get("policy_id") != policy_id):
            raise RuntimeProtocolError(f"training receipt {policy_id} has a wrong schema or policy binding")
        receipt_binding = {
            "path": _repo_relative(root_path, receipt_file, label="training receipt"),
            "raw_sha256": _sha256_bytes(receipt_raw),
            "run_id": receipt.get("run_id"),
        }
        if receipt.get("status") == "unavailable":
            policies.append({
                "policy_id": policy_id, "status": "unavailable",
                "reason": receipt.get("unavailable_reason") or receipt.get("reason") or "training_unavailable",
                "training_receipt": receipt_binding, "checkpoint_directory": None,
                "inventory": None, "model_tensors": None,
            })
            continue
        if receipt.get("status") != "eligible":
            stop_required.append(policy_id)
            policies.append({
                "policy_id": policy_id, "status": "unavailable",
                "reason": f"training_operation_{receipt.get('status', 'malformed')}",
                "training_receipt": receipt_binding, "checkpoint_directory": None,
                "inventory": None, "model_tensors": None,
            })
            continue
        record = _training_record(study, policy_id)
        config = _load_config(root_path, record["config"]["path"])
        run_id = receipt.get("run_id")
        run_directory = receipt.get("run_directory")
        checkpoint_record = receipt.get("checkpoint")
        if (not isinstance(run_id, str) or not run_id or not isinstance(run_directory, str)
                or not isinstance(checkpoint_record, Mapping)):
            raise RuntimeProtocolError(f"eligible training receipt {policy_id} omits actual checkpoint identity")
        run_dir = _bound_path(root_path, run_directory, label=f"training run directory {policy_id}")
        checkpoint_path = _bound_path(
            root_path, checkpoint_record.get("directory"),
            label=f"final checkpoint directory {policy_id}",
        )
        if run_dir.name != run_id or checkpoint_path != run_dir / "checkpoint-final":
            raise RuntimeProtocolError(f"training receipt {policy_id} run/checkpoint path is not final")
        preflight = _default_checkpoint_preflight
        current = _checkpoint_inventory(root_path, checkpoint_path, config, preflight)
        if (current["inventory"] != checkpoint_record.get("inventory")
                or current["model_tensors"] != checkpoint_record.get("model_tensors")
                or current["metadata"] != checkpoint_record.get("metadata")):
            raise RuntimeProtocolError(f"training receipt {policy_id} checkpoint inventory changed before freeze")
        policies.append({
            "policy_id": policy_id,
            "status": "available",
            "reason": None,
            "training_receipt": receipt_binding,
            "checkpoint_directory": _repo_relative(root_path, checkpoint_path, label="frozen checkpoint"),
            "inventory": current["inventory"],
            "model_tensors": current["model_tensors"],
        })
    value = {
        "schema_id": "spire-learn-task9-checkpoint-bindings-v1",
        "state": "frozen_before_selection",
        "study_raw_sha256": _sha256_bytes(study_raw),
        "selection_policy_order": list(POLICY_ORDER),
        "policies": policies,
        "stop_required_policy_ids": stop_required,
        "selection_reserved_step_slots_max": SELECTION_RESERVED_MAX,
    }
    _exclusive_json(binding_file, value)
    safe = _project_json(value)
    assert isinstance(safe, dict)
    return safe


class _ReferenceImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".")
               for prefix in _FORBIDDEN_REFERENCE_PREFIXES):
            self.attempts.append(fullname)
            raise ImportError(f"reference child forbids import attempt {fullname}")
        return None

    def snapshot(self) -> dict[str, object]:
        loaded = sorted(
            name for name in sys.modules
            if any(name == prefix or name.startswith(prefix + ".")
                   for prefix in _FORBIDDEN_REFERENCE_PREFIXES)
        )
        return {
            "installed": True,
            "process_id": os.getpid(),
            "attempts": list(self.attempts),
            "loaded_modules": loaded,
        }

    def check(self) -> dict[str, object]:
        evidence = self.snapshot()
        loaded = evidence["loaded_modules"]
        if self.attempts or loaded:
            raise ReferenceImportViolation(
                f"reference child attempted/loaded forbidden imports; attempts={self.attempts}; loaded={loaded}"
            )
        return evidence


@contextmanager
def _reference_import_guard():
    guard = _ReferenceImportGuard()
    already_loaded = sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in _FORBIDDEN_REFERENCE_PREFIXES)
    )
    if already_loaded:
        raise ReferenceImportViolation(f"reference child already has forbidden imports loaded: {already_loaded}")
    sys.meta_path.insert(0, guard)
    primary: BaseException | None = None
    try:
        yield guard
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        if guard.attempts:
            violation = ReferenceImportViolation(
                f"reference child attempted forbidden imports: {guard.attempts}"
            )
            if primary is not None:
                _note(primary, str(violation))
            else:
                raise violation


def _binding_file(root: Path, study_raw: bytes, bindings_path: Path) -> tuple[dict[str, object], bytes]:
    expected_path = _checkpoint_binding_path(root)
    binding_file = _caller_path(
        root, bindings_path, expected=expected_path, label="checkpoint binding input",
    )
    bindings, raw = _read_json(binding_file, label="frozen checkpoint bindings")
    if (bindings.get("schema_id") != "spire-learn-task9-checkpoint-bindings-v1"
            or bindings.get("state") != "frozen_before_selection"
            or bindings.get("study_raw_sha256") != _sha256_bytes(study_raw)
            or bindings.get("selection_policy_order") != list(POLICY_ORDER)
            or bindings.get("selection_reserved_step_slots_max") != SELECTION_RESERVED_MAX):
        raise RuntimeProtocolError("checkpoint binding record does not match the validated study")
    policies = bindings.get("policies")
    if not isinstance(policies, list) or [row.get("policy_id") if isinstance(row, Mapping) else None
                                          for row in policies] != list(TRAINING_ORDER):
        raise RuntimeProtocolError("checkpoint binding policy inventory/order is malformed")
    for policy in policies:
        if policy.get("status") not in {"available", "unavailable"}:
            raise RuntimeProtocolError("checkpoint binding has an unsupported policy status")
        if policy["status"] == "available":
            receipt = policy.get("training_receipt")
            inventory = policy.get("inventory")
            tensors = policy.get("model_tensors")
            checkpoint_dir = policy.get("checkpoint_directory")
            if (not isinstance(receipt, Mapping) or not isinstance(receipt.get("run_id"), str)
                    or not isinstance(receipt.get("raw_sha256"), str)
                    or not isinstance(inventory, Mapping) or set(inventory) != {"metadata.json", "model.pt", "training.pt"}
                    or not isinstance(tensors, list) or not tensors
                    or not isinstance(checkpoint_dir, str)):
                raise RuntimeProtocolError(f"available binding {policy['policy_id']} is incomplete")
    stop_required = bindings.get("stop_required_policy_ids")
    if (not isinstance(stop_required, list)
            or any(not isinstance(policy_id, str) or policy_id not in TRAINING_ORDER
                   for policy_id in stop_required)
            or len(stop_required) != len(set(stop_required))):
        raise RuntimeProtocolError("checkpoint binding stop-required policy IDs are malformed")
    faulted_training = [
        policy["policy_id"] for policy in policies
        if policy.get("status") == "unavailable"
        and isinstance(policy.get("reason"), str)
        and policy["reason"].startswith("training_operation_")
    ]
    if stop_required != faulted_training:
        raise RuntimeProtocolError("checkpoint binding stop-required IDs disagree with faulted training receipts")
    return bindings, raw


def _selection_policy(study: Mapping[str, object], policy_id: str) -> dict[str, object]:
    selection = study.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("policies"), list):
        raise ValueError("validated study policy inventory is malformed")
    matches = [item for item in selection["policies"]
               if isinstance(item, Mapping) and item.get("policy_id") == policy_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous selection policy {policy_id!r}")
    return dict(matches[0])


def _case_binding(study: Mapping[str, object], row: Mapping[str, object]) -> dict[str, object]:
    selection = study.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("case_bindings"), list):
        raise ValueError("validated study case inventory is malformed")
    matches = [item for item in selection["case_bindings"]
               if isinstance(item, Mapping)
               and item.get("case_index") == row.get("case_index")
               and item.get("mode") == row.get("mode")]
    if len(matches) != 1:
        raise RuntimeProtocolError("selection schedule row has no unique exact case binding")
    return dict(matches[0])


def _selection_config(root: Path, row: Mapping[str, object], binding: Mapping[str, object]):
    config_path = _bound_path(root, row["config_path"], label="selection config")
    if (row.get("config_path") != binding.get("config_path")
            or row.get("config_sha256") != binding.get("config_sha256")
            or row.get("config_raw_sha256") != binding.get("config_raw_sha256")
            or row.get("one_case_manifest_path") != binding.get("manifest_path")
            or row.get("one_case_manifest_sha256") != binding.get("manifest_sha256")
            or row.get("one_case_manifest_raw_sha256") != binding.get("manifest_raw_sha256")
            or row.get("parent_selection_sha256") != binding.get("parent_selection_sha256")):
        raise RuntimeProtocolError("schedule row and exact authored case binding disagree")
    raw_hash = _sha256_file(config_path)
    if raw_hash != row["config_raw_sha256"]:
        raise RuntimeProtocolError("selection config raw file digest changed")
    config = _load_config(root, row["config_path"])
    if config.config_sha256 != row["config_sha256"] or config.mode != row["mode"]:
        raise RuntimeProtocolError("loaded selection config canonical identity or mode changed")
    expected_output = binding["output_root"]
    output_root = _exact_output_root(root, expected_output, config)
    case_manifest = _bound_path(root, row["one_case_manifest_path"], label="one-case manifest")
    if (_sha256_file(case_manifest) != row["one_case_manifest_raw_sha256"]
            or config.manifest_path != case_manifest
            or config.manifest_sha256 != row["one_case_manifest_sha256"]):
        raise RuntimeProtocolError("selection config is not bound to the exact one-case manifest")
    cases = config.cases.for_mode(config.mode)
    if len(cases) != 1 or cases[0].case_id != row["case_id"]:
        raise RuntimeProtocolError("selection config does not contain the schedule's exact single case")
    if (config.operation, config.split, config.partition, config.device,
            config.rollout_rows, config.rollout_steps) != (
            "evaluate", "selection", "selection", "cpu", 1, 1,
    ):
        raise RuntimeProtocolError("selection config violates the frozen CPU one-case evaluation contract")
    if (config.budgets.max_optimizer_steps, config.budgets.max_training_decisions,
            config.budgets.max_environment_actions) != (1, SELECTION_RESERVATION, SELECTION_RESERVATION):
        raise RuntimeProtocolError("selection config caps differ from the 256-slot reservation")
    return config, output_root, cases[0]


def _operation_receipt_file(root: Path, operation_id: str) -> Path:
    return root / SELECTION_OUTPUT_RELATIVE / "control" / "operations" / f"{operation_id}.json"


def _operation_reservation_file(root: Path, operation_id: str) -> Path:
    return root / SELECTION_OUTPUT_RELATIVE / "control" / "operations" / f"{operation_id}.reservation.json"


def _operation_entered_file(root: Path, operation_id: str) -> Path:
    return root / SELECTION_OUTPUT_RELATIVE / "control" / "operations" / f"{operation_id}.entered.json"


def _reserve_selection_operation(
    root: Path, row: Mapping[str, object], binding_hash: str, deadline: float,
    clock: Callable[[], float], *, reserved_by: str,
) -> tuple[dict[str, object], str]:
    operation_id = row["operation_id"]
    reservation_file = _operation_reservation_file(root, operation_id)
    entered_file = _operation_entered_file(root, operation_id)
    receipt_file = _operation_receipt_file(root, operation_id)
    for path, label in ((reservation_file, "operation reservation"),
                        (entered_file, "operation entry"), (receipt_file, "operation receipt")):
        _assert_safe_path(root, path, label=label)
    _make_directories_for(reservation_file)
    if entered_file.exists() or entered_file.is_symlink() or receipt_file.exists() or receipt_file.is_symlink():
        raise FileExistsError(f"selection operation {operation_id} was already entered or finalized")
    if reservation_file.exists():
        reservation, raw = _read_json(reservation_file, label=f"reservation {operation_id}")
        expected = {
            "schema_id": "spire-learn-task9-selection-reservation-v1",
            "operation_id": operation_id,
            "index": row["index"],
            "case_id": row["case_id"],
            "policy_id": row["policy_id"],
            "mode": row["mode"],
            "binding_raw_sha256": binding_hash,
            "reserved_step_slots": SELECTION_RESERVATION,
            "state": "reserved_once",
        }
        if any(reservation.get(key) != value for key, value in expected.items()):
            raise RuntimeProtocolError(f"operation reservation {operation_id} does not match its schedule row")
    else:
        reservation = {
            "schema_id": "spire-learn-task9-selection-reservation-v1",
            "operation_id": operation_id,
            "index": row["index"],
            "case_id": row["case_id"],
            "policy_id": row["policy_id"],
            "mode": row["mode"],
            "binding_raw_sha256": binding_hash,
            "reserved_step_slots": SELECTION_RESERVATION,
            "deadline_monotonic": deadline,
            "reserved_at_monotonic": _now(clock),
            "reserved_by": reserved_by,
            "state": "reserved_once",
        }
        raw = _exclusive_json(reservation_file, reservation)
    entered = {
        "schema_id": "spire-learn-task9-selection-entry-v1",
        "operation_id": operation_id,
        "reservation_raw_sha256": _sha256_bytes(raw),
        "entered_at_monotonic": _now(clock),
        "state": "attempted_once",
    }
    _exclusive_json(entered_file, entered)
    return reservation, _sha256_bytes(raw)


def _selection_count_projection(report: Mapping[str, object] | None) -> dict[str, object]:
    keys = (
        "attempted_environment_slots", "forwarded_environment_slots",
        "known_accepted_environment_slots", "delivered_environment_slots",
        "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
        "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
        "learned_decisions", "fixed_decisions", "incomplete_decisions",
    )
    if report is None:
        return {key: None for key in keys}
    collection = report.get("collection")
    if not isinstance(collection, Mapping):
        return {key: None for key in keys}
    return {
        key: (value if type(value := collection.get(key)) is int and value >= 0 else None)
        for key in keys
    }


def _selection_status(info: Mapping[str, object], expected_case_id: str) -> tuple[str, str | None]:
    report = info.get("report")
    lifecycle = info.get("lifecycle")
    if not isinstance(report, Mapping) or not isinstance(lifecycle, Mapping):
        return "operational_failure", "runner_report_or_lifecycle_missing"
    collection = report.get("collection")
    optimizer = report.get("optimizer")
    accounting = report.get("case_accounting")
    if not isinstance(collection, Mapping) or not isinstance(optimizer, Mapping) or not isinstance(accounting, Mapping):
        return "operational_failure", "runner_progress_evidence_missing"
    if (report.get("fault") is not None or lifecycle.get("status") not in {"complete", "incomplete"}
            or not isinstance(lifecycle.get("close_errors"), list) or lifecycle["close_errors"]):
        return "operational_failure", "evaluation_fault_or_close_failure"
    for name in ("uncertain_environment_slots", "uncertain_reset_slots"):
        if type(collection.get(name)) is not int or collection[name] != 0:
            return "operational_failure", "evaluation_environment_effect_uncertain"
    if any(type(optimizer.get(key)) is not int or optimizer[key] != 0
           for key in ("attempted_steps", "accepted_steps", "uncertain_steps", "updates")):
        return "operational_failure", "evaluation_made_or_uncertain_optimizer_updates"
    cases = accounting.get("cases")
    if (accounting.get("scheduled") != 1 or not isinstance(cases, list) or len(cases) != 1
            or not isinstance(cases[0], Mapping) or cases[0].get("case_id") != expected_case_id):
        return "operational_failure", "evaluation_case_accounting_malformed"
    status = report.get("status")
    exit_code = info.get("runner_exit_code")
    if status == "complete" and lifecycle.get("status") == "complete" and exit_code == 0:
        return "complete", None
    if status == "incomplete" and lifecycle.get("status") == "incomplete" and exit_code == 1:
        return "incomplete", "healthy_case_incomplete"
    return "operational_failure", "runner_and_lifecycle_status_disagree"


def _selection_receipt_base(
    root: Path, study: Mapping[str, object], study_raw: bytes, row: Mapping[str, object],
    config: Any, case: Any, binding_record: Mapping[str, object], binding_raw: bytes,
    deadline: float,
) -> dict[str, object]:
    seeds = {
        "policy": config.seed_streams.policy,
        "optimizer": config.seed_streams.optimizer,
        "curriculum": config.seed_streams.curriculum,
        "noncombat": config.seed_streams.noncombat,
    }
    return {
        "schema_id": "spire-learn-task9-selection-operation-v1",
        "operation_id": row["operation_id"],
        "index": row["index"],
        "case_id": row["case_id"],
        "case_index": row["case_index"],
        "policy_id": row["policy_id"],
        "mode": row["mode"],
        "cohort": row["cohort"],
        "source_policy_mode": row["source_policy_mode"],
        "classification": row["classification"],
        "study_raw_sha256": _sha256_bytes(study_raw),
        "binding_raw_sha256": _sha256_bytes(binding_raw),
        "config": {
            "path": row["config_path"], "canonical_sha256": config.config_sha256,
            "raw_sha256": row["config_raw_sha256"],
        },
        "manifest": {
            "path": row["one_case_manifest_path"],
            "canonical_sha256": config.manifest_sha256,
            "raw_sha256": row["one_case_manifest_raw_sha256"],
            "parent_selection_sha256": row["parent_selection_sha256"],
        },
        "seed_evidence": {
            "environment_seed": case.environment_seed,
            "lineage_id": case.lineage_id,
            "entry_id": case.entry_id,
            "streams": seeds,
        },
        "environment_revision": config.environment_revision,
        "budgets": {
            "reserved_step_slots": SELECTION_RESERVATION,
            "learned_decision_cap": config.budgets.max_training_decisions,
            "environment_action_cap": config.budgets.max_environment_actions,
            "optimizer_call_cap": config.budgets.max_optimizer_steps,
            "soft_deadline_monotonic": deadline,
        },
        "reservation": None,
        "checkpoint": {
            "training_receipt": binding_record.get("training_receipt"),
            "before": None,
            "after": None,
        },
        "run_id": None,
        "run_directory": None,
        "runner_exit_code": None,
        "transport_exit_code": 1,
        "report": None,
        "lifecycle": None,
        "summary_path": None,
        "summary_sha256": None,
        "lifecycle_path": None,
        "lifecycle_sha256": None,
        "case_accounting": None,
        "fight_summary": None,
        "optimizer": None,
        "counts": _selection_count_projection(None),
        "runtime_thread_settings": None,
        "reference_import_guard": None,
        "attempted_effects": None,
        "close_status": "not_acquired",
        "status": "operational_failure",
        "reason": None,
        "outer_timing": {
            "start_monotonic": None, "finish_monotonic": None,
            "elapsed_seconds": None,
            "timing_boundary": "outer selection operation before factory through pre-publication finalization",
        },
    }


def run_selection_operation(
    root: Path, study_path: Path, operation_id: str, bindings_path: Path,
    receipt_path: Path, *, deadline: float,
    clock: Callable[[], float] = time.monotonic,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, object]:
    """Run one fresh CPU selection operation for an exact frozen schedule row."""
    deadline_value = _deadline(deadline)
    first_clock = _now(clock)
    root_path, study, study_file, study_raw = _validate_study(root, study_path)
    row = _schedule_row(study, operation_id)
    case_binding = _case_binding(study, row)
    config, output_root, case = _selection_config(root_path, row, case_binding)
    binding_record, binding_raw = _binding_file(root_path, study_raw, bindings_path)
    receipt_file = _caller_path(
        root_path, Path(receipt_path), expected=_operation_receipt_file(root_path, operation_id),
        label="selection operation receipt path",
    )
    policy = _selection_policy(study, row["policy_id"])
    frozen_policies = {item["policy_id"]: item for item in binding_record["policies"]}
    learned = policy["kind"] == "learned"
    frozen = frozen_policies.get(row["policy_id"]) if learned else None
    if learned and not isinstance(frozen, Mapping):
        raise RuntimeProtocolError("learned selection policy has no frozen checkpoint binding record")
    if (policy["kind"] == "reference") != (row["policy_id"] in {"legal-random", "end-turn-last-v1"}):
        raise RuntimeProtocolError("selection policy kind disagrees with the authored policy identity")
    stop_required = binding_record["stop_required_policy_ids"]
    blocked_reason = (
        f"prior_training_fault:{','.join(stop_required)}" if stop_required else None
    )
    if blocked_reason is None and learned and frozen["status"] != "available":
        blocked_reason = frozen.get("reason") or "policy_unavailable"
    if blocked_reason is not None:
        reservation_file = _operation_reservation_file(root_path, operation_id)
        entered_file = _operation_entered_file(root_path, operation_id)
        if any(path.exists() or path.is_symlink()
               for path in (receipt_file, reservation_file, entered_file)):
            raise FileExistsError("selection operation already has a reservation, entry, or receipt")
        result = _selection_receipt_base(
            root_path, study, study_raw, row, config, case, frozen or {}, binding_raw, deadline_value,
        )
        result.update({
            "status": "not_started", "reason": blocked_reason,
            "attempted_effects": False, "counts": _selection_count_projection(None),
            "transport_exit_code": 0,
        })
        primary: BaseException | None = None
        try:
            finish = _now(clock)
        except BaseException as error:
            primary = error
            result["final_clock_error"] = _error_value(error)
            if _is_control(error):
                result.update({
                    "status": "interrupted", "reason": "final_clock_process_control",
                    "transport_exit_code": 1,
                })
            else:
                result.update({
                    "status": "operational_failure", "reason": "final_clock_error",
                    "transport_exit_code": 1,
                })
            finish = None
        result["outer_timing"] = {
            "start_monotonic": first_clock,
            "finish_monotonic": finish,
            "elapsed_seconds": max(0.0, finish - first_clock) if finish is not None else None,
            "timing_boundary": "outer selection operation through pre-publication finalization",
        }
        projected = _project_json(result)
        assert isinstance(projected, dict)
        published = False
        try:
            _write_result_preserving_control(
                receipt_file, projected, primary, context="selection receipt",
            )
            published = True
        except BaseException as error:
            primary = error
        if primary is not None and (_is_control(primary) or not published):
            raise primary
        return projected

    _require_directory(output_root, label="selection case output root")
    selection_output = root_path / SELECTION_OUTPUT_RELATIVE
    _make_directories_for(receipt_file)
    base = _selection_receipt_base(
        root_path, study, study_raw, row, config, case, frozen or {}, binding_raw, deadline_value,
    )
    reservation: dict[str, object] | None = None
    reservation_hash: str | None = None
    runtime = None
    handed_to_runner = False
    before_checkpoint: dict[str, object] | None = None
    after_checkpoint: dict[str, object] | None = None
    postflight_attempted = False
    info: dict[str, object] | None = None
    status = "operational_failure"
    reason: str | None = None
    primary: BaseException | None = None
    import_guard: _ReferenceImportGuard | None = None
    try:
        reservation, reservation_hash = _reserve_selection_operation(
            root_path, row, _sha256_bytes(binding_raw), deadline_value, clock,
            reserved_by="direct_or_schedule_child",
        )
        base["reservation"] = {
            "path": _repo_relative(root_path, _operation_reservation_file(root_path, operation_id),
                                   label="selection reservation"),
            "raw_sha256": reservation_hash,
            "reserved_step_slots": reservation["reserved_step_slots"],
        }
        if _now(clock) >= deadline_value:
            status, reason = "not_started", "shared_soft_deadline_exhausted_before_factory"
            base["attempted_effects"] = False
            base["counts"] = _selection_count_projection(None)
        else:
            checkpoint_path = None
            if learned:
                assert isinstance(frozen, Mapping)
                checkpoint_path = _bound_path(
                    root_path, frozen["checkpoint_directory"], label="frozen selection checkpoint",
                )
                preflight = _resolve_dependency(
                    dependencies, "checkpoint_preflight", _default_checkpoint_preflight,
                )
                before_checkpoint = _checkpoint_inventory(
                    root_path, checkpoint_path, config, preflight,
                    expected_inventory=frozen["inventory"],
                )
                base["checkpoint"]["before"] = before_checkpoint["inventory"]
            if _now(clock) >= deadline_value:
                status, reason = "not_started", "shared_soft_deadline_exhausted_before_factory"
                base["attempted_effects"] = False
                base["counts"] = _selection_count_projection(None)
            else:
                policy_kind = row["policy_id"] if not learned else "checkpoint"
                action_selection = "seeded-sampling" if row["policy_id"] == "legal-random" else "deterministic"
                factory = _resolve_dependency(dependencies, "factory", _default_factory)
                evaluator = _resolve_dependency(dependencies, "evaluate", _actual_evaluate)
                guard_context = _reference_import_guard() if not learned else nullcontext(None)
                with guard_context as import_guard:
                    if import_guard is not None:
                        base["reference_import_guard"] = import_guard.snapshot()
                    runtime = factory(
                        config, purpose="evaluate", checkpoint=checkpoint_path,
                        policy_kind=policy_kind, action_selection=action_selection,
                    )
                    base["runtime_thread_settings"] = _runtime_thread_evidence(runtime)
                    if learned:
                        observed_checkpoint = getattr(runtime, "parent_checkpoint", None)
                        if observed_checkpoint is None:
                            raise RuntimeProtocolError("neural runtime omitted its actual parent_checkpoint")
                        observed_checkpoint_path = Path(observed_checkpoint)
                        observed_checkpoint_relative = _repo_relative(
                            root_path, observed_checkpoint_path, label="runtime parent_checkpoint",
                        )
                        base["checkpoint"]["factory_checkpoint_path"] = _repo_relative(
                            root_path, checkpoint_path, label="factory checkpoint argument",
                        )
                        base["checkpoint"]["runtime_parent_checkpoint"] = observed_checkpoint_relative
                        if observed_checkpoint_path != checkpoint_path:
                            raise RuntimeProtocolError("neural runtime parent_checkpoint differs from its factory argument")
                    _set_operation_deadline(runtime, deadline_value)
                    if import_guard is not None:
                        base["reference_import_guard"] = import_guard.check()
                    handed_to_runner = True
                    report = evaluator(
                        config, runtime=runtime, clock=clock, allow_final=False,
                    )
                    info = _report_files(root_path, config, report)
                    base.update(info)
                    report_summary = info["report"]
                    assert isinstance(report_summary, Mapping)
                    base["case_accounting"] = report_summary.get("case_accounting")
                    base["fight_summary"] = report_summary.get("fights")
                    base["optimizer"] = report_summary.get("optimizer")
                    base["counts"] = _selection_count_projection(report_summary)
                    optimizer_summary = report_summary.get("optimizer")
                    base["attempted_effects"] = _attempted_effects(
                        base["counts"],
                        optimizer=optimizer_summary if isinstance(optimizer_summary, Mapping) else None,
                        runner_handed_off=True,
                    )
                    if (report_summary.get("allow_final") is not False
                            or report_summary.get("policy_kind") != policy_kind
                            or report_summary.get("action_selection") != action_selection):
                        raise RuntimeProtocolError("runner changed frozen evaluation policy controls")
                    status, reason = _selection_status(info, row["case_id"])
                    if import_guard is not None:
                        base["reference_import_guard"] = import_guard.check()
                    tree_bytes = _directory_bytes(selection_output)
                    if tree_bytes > SELECTION_OUTPUT_LIMIT:
                        raise OutputAccountingError("selection output exceeds its 2 GiB accounting ceiling")
        if before_checkpoint is not None and handed_to_runner:
            postflight_attempted = True
            after_checkpoint = _checkpoint_inventory(
                root_path, checkpoint_path, config, preflight,
                expected_inventory=frozen["inventory"],
            )
            base["checkpoint"]["after"] = after_checkpoint["inventory"]
            if before_checkpoint["inventory"] != after_checkpoint["inventory"]:
                raise RuntimeProtocolError("frozen checkpoint bytes changed during neural evaluation")
    except BaseException as error:
        primary = error
        if import_guard is not None:
            base["reference_import_guard"] = import_guard.snapshot()
        if runtime is not None and not handed_to_runner:
            primary = _close_preserving(primary, runtime, "runtime setup")
        if not handed_to_runner:
            base["attempted_effects"] = False
        elif info is None:
            base["counts"] = _selection_count_projection(None)
            base["attempted_effects"] = None
        if _is_control(primary):
            status, reason = "interrupted", "process_control_exception"
        else:
            status, reason = "operational_failure", "selection_operation_error"
        base["error"] = _error_value(error)
    if before_checkpoint is not None and handed_to_runner and not postflight_attempted:
        try:
            postflight_attempted = True
            after_checkpoint = _checkpoint_inventory(
                root_path, checkpoint_path, config, preflight,
                expected_inventory=frozen["inventory"],
            )
            base["checkpoint"]["after"] = after_checkpoint["inventory"]
        except BaseException as error:
            if primary is None or (_is_control(error) and not _is_control(primary)):
                if primary is not None:
                    _note(error, f"checkpoint postflight followed {type(primary).__name__}: {_error_value(primary)['message']}")
                primary = error
            else:
                _note(primary, f"checkpoint postflight also failed: {type(error).__name__}: {_error_value(error)['message']}")
            status, reason = ("interrupted", "checkpoint_postflight_process_control") if _is_control(primary) else (
                "operational_failure", "checkpoint_postflight_failure",
            )
            base["checkpoint"]["postflight_error"] = _error_value(error)
    final_clock: float | None = None
    try:
        final_clock = _now(clock)
    except BaseException as error:
        if primary is None or (_is_control(error) and not _is_control(primary)):
            if primary is not None:
                _note(error, f"final clock followed {type(primary).__name__}: {_error_value(primary)['message']}")
            primary = error
        else:
            _note(primary, f"final clock also failed: {type(error).__name__}: {_error_value(error)['message']}")
        if _is_control(primary):
            status, reason = "interrupted", "final_clock_process_control"
        else:
            status, reason = "operational_failure", "final_clock_error"
        base["final_clock_error"] = _error_value(error)
    base["status"] = status
    base["reason"] = reason
    base["close_status"] = (
        "clean" if isinstance(base.get("lifecycle"), Mapping)
        and not base["lifecycle"].get("close_errors") else
        "reported_failure" if isinstance(base.get("lifecycle"), Mapping) else
        "runner_owned_or_unknown"
    )
    base["transport_exit_code"] = 0 if status in {"complete", "incomplete", "not_started"} else 1
    base["outer_timing"] = {
        "start_monotonic": first_clock,
        "finish_monotonic": final_clock,
        "elapsed_seconds": max(0.0, final_clock - first_clock) if final_clock is not None else None,
            "timing_boundary": "outer selection operation before factory through pre-publication finalization",
    }
    result = _project_json(base)
    assert isinstance(result, dict)
    receipt_published = False
    try:
        _write_result_preserving_control(
            receipt_file, result, primary, context="selection receipt",
        )
        receipt_published = True
    except BaseException as error:
        primary = error
    if primary is not None and (_is_control(primary) or not receipt_published):
        raise primary
    return result


def _directory_bytes(path: Path) -> int:
    """Count regular output bytes without following links or special files."""
    if path.is_symlink() or not path.is_dir():
        raise OutputAccountingError("selection output root is missing or unsafe")
    total = 0
    pending = [path]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise OutputAccountingError("selection output contains a symlink")
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(mode):
                    total += entry.stat(follow_symlinks=False).st_size
                else:
                    raise OutputAccountingError("selection output contains a non-regular file")
    return total


def _schedule_identity(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "index", "operation_id", "case_index", "case_id", "mode", "cohort",
        "policy_id", "source_policy_mode", "classification", "config_path",
        "config_sha256", "config_raw_sha256", "one_case_manifest_path",
        "one_case_manifest_sha256", "one_case_manifest_raw_sha256",
        "parent_selection_sha256",
    )
    return {key: row[key] for key in keys}


def _reserve_scheduled_operation(
    root: Path, row: Mapping[str, object], binding_hash: str, deadline: float,
    clock: Callable[[], float],
) -> tuple[dict[str, object], str]:
    reservation_file = _operation_reservation_file(root, row["operation_id"])
    entered_file = _operation_entered_file(root, row["operation_id"])
    receipt_file = _operation_receipt_file(root, row["operation_id"])
    for path, label in (
        (reservation_file, "selection reservation"),
        (entered_file, "selection entry"),
        (receipt_file, "selection receipt"),
    ):
        _assert_safe_path(root, path, label=label)
    if any(path.exists() or path.is_symlink() for path in (reservation_file, entered_file, receipt_file)):
        raise FileExistsError(f"selection operation {row['operation_id']} already has a reservation or result")
    reservation = {
        "schema_id": "spire-learn-task9-selection-reservation-v1",
        "operation_id": row["operation_id"],
        "index": row["index"],
        "case_id": row["case_id"],
        "policy_id": row["policy_id"],
        "mode": row["mode"],
        "binding_raw_sha256": binding_hash,
        "reserved_step_slots": SELECTION_RESERVATION,
        "deadline_monotonic": deadline,
        "reserved_at_monotonic": _now(clock),
        "reserved_by": "parent_schedule",
        "state": "reserved_once",
    }
    raw = _exclusive_json(reservation_file, reservation)
    return reservation, _sha256_bytes(raw)


def _reservation_publication_evidence(
    root: Path, row: Mapping[str, object], binding_hash: str, deadline: float,
) -> dict[str, object]:
    """Observe failed publication without treating readable bytes as durable."""
    path = _operation_reservation_file(root, row["operation_id"])
    evidence: dict[str, object] = {
        "state": "uncertain", "footprint_present": None,
        "record_status": "unknown", "raw_sha256": None,
    }
    try:
        try:
            path.lstat()
        except FileNotFoundError:
            evidence.update(footprint_present=False, record_status="missing")
            return evidence
        evidence["footprint_present"] = True
        _assert_safe_path(root, path, label="ambiguous reservation footprint")
        record, raw = _read_json(path, label="ambiguous reservation footprint")
        expected = {
            "schema_id": "spire-learn-task9-selection-reservation-v1",
            "operation_id": row["operation_id"], "index": row["index"],
            "case_id": row["case_id"], "policy_id": row["policy_id"], "mode": row["mode"],
            "binding_raw_sha256": binding_hash, "reserved_step_slots": SELECTION_RESERVATION,
            "deadline_monotonic": deadline, "state": "reserved_once",
        }
        projected = {key: record.get(key) for key in expected}
        if _canonical_fingerprint(projected) != _canonical_fingerprint(expected):
            raise RuntimeProtocolError("ambiguous reservation footprint identity mismatch")
        evidence.update(record_status="matched", raw_sha256=_sha256_bytes(raw))
    except Exception as error:
        evidence.update(record_status="invalid_or_unreadable", error=_error_value(error))
    return evidence


def _default_child_process(
    argv: Sequence[str], *, capture_output: bool, text: bool, timeout: float,
    check: bool, output_limit_bytes: int, progress_check: Callable[[], None],
):
    from scripts.neural.gpu_runtime_proof_helpers import _bounded_subprocess_run

    return _bounded_subprocess_run(
        argv, capture_output=capture_output, text=text, timeout=timeout, check=check,
        output_limit_bytes=output_limit_bytes, progress_check=progress_check,
    )


def _check_operation_artifacts(root: Path, receipt: Mapping[str, object]) -> None:
    run_directory = _bound_path(root, receipt.get("run_directory"), label="selection run directory")
    summary_path = _bound_path(root, receipt.get("summary_path"), label="selection summary")
    lifecycle_path = _bound_path(root, receipt.get("lifecycle_path"), label="selection lifecycle")
    if (not run_directory.is_dir() or run_directory.is_symlink()
            or run_directory.name != receipt.get("run_id")
            or summary_path != run_directory / "summary.json"
            or lifecycle_path != run_directory / "lifecycle.json"):
        raise RuntimeProtocolError("selection report paths do not match the actual writer run directory")
    if (_sha256_file(summary_path) != receipt.get("summary_sha256")
            or _sha256_file(lifecycle_path) != receipt.get("lifecycle_sha256")):
        raise RuntimeProtocolError("selection summary or lifecycle raw hash differs from its receipt")
    summary_doc, _ = _read_json(summary_path, label="selection summary.json")
    lifecycle, _ = _read_json(lifecycle_path, label="selection lifecycle.json")
    if (summary_doc.get("schema_id") != "spire-learn-neural-run-v1"
            or summary_doc.get("record_type") != "summary"
            or not isinstance(summary_doc.get("summary"), Mapping)
            or lifecycle != receipt.get("lifecycle")
            or lifecycle.get("schema_id") != "spire-learn-neural-lifecycle-v1"):
        raise RuntimeProtocolError("selection summary/lifecycle files disagree with their operation receipt")
    provisional = summary_doc["summary"]
    report = receipt.get("report")
    pending = provisional.get("lifecycle")
    if (not isinstance(report, Mapping)
            or summary_doc.get("summary_sha256") != _canonical_fingerprint(provisional)
            or not isinstance(pending, Mapping)
            or pending.get("status") != "receipt_pending" or pending.get("exit_code") != 1
            or provisional.get("run_directory") != str(run_directory)
            or report.get("run_directory") != str(run_directory)
            or receipt.get("case_accounting") != report.get("case_accounting")):
        raise RuntimeProtocolError("selection provisional summary differs from its accepted runner report")
    for key in (
        "operation", "mode", "split", "config_sha256", "manifest_sha256",
        "environment_revision",
    ):
        if provisional.get(key) != report.get(key):
            raise RuntimeProtocolError(f"selection summary {key} differs from the accepted runner report")
    if (lifecycle.get("status") != receipt.get("status")
            or lifecycle.get("exit_code") != receipt.get("runner_exit_code")
            or lifecycle.get("close_errors") != []):
        raise RuntimeProtocolError("selection authoritative lifecycle is not healthy for its runner receipt")


def _reconcile_selection_receipt(
    root: Path, study: Mapping[str, object], study_raw: bytes,
    row: Mapping[str, object], bindings: Mapping[str, object], binding_raw: bytes,
    receipt_file: Path, reservation: Mapping[str, object] | None,
    reservation_hash: str | None, dependencies: RuntimeDependencies | None,
) -> tuple[dict[str, object], bytes]:
    receipt, raw = _read_json(receipt_file, label=f"selection receipt {row['operation_id']}")
    identity = {
        key: row[key] for key in (
            "index", "operation_id", "case_index", "case_id", "mode", "cohort",
            "policy_id", "source_policy_mode", "classification",
        )
    }
    expected_policy = _selection_policy(study, row["policy_id"])
    if (receipt.get("schema_id") != "spire-learn-task9-selection-operation-v1"
            or any(receipt.get(key) != value for key, value in identity.items())
            or receipt.get("study_raw_sha256") != _sha256_bytes(study_raw)
            or receipt.get("binding_raw_sha256") != _sha256_bytes(binding_raw)
            or receipt.get("policy_id") != expected_policy.get("policy_id")):
        raise RuntimeProtocolError("selection child receipt identity differs from the frozen schedule")
    config_binding = receipt.get("config")
    manifest_binding = receipt.get("manifest")
    if (not isinstance(config_binding, Mapping)
            or config_binding.get("path") != row["config_path"]
            or config_binding.get("canonical_sha256") != row["config_sha256"]
            or config_binding.get("raw_sha256") != row["config_raw_sha256"]
            or not isinstance(manifest_binding, Mapping)
            or manifest_binding.get("path") != row["one_case_manifest_path"]
            or manifest_binding.get("canonical_sha256") != row["one_case_manifest_sha256"]
            or manifest_binding.get("raw_sha256") != row["one_case_manifest_raw_sha256"]
            or manifest_binding.get("parent_selection_sha256") != row["parent_selection_sha256"]):
        raise RuntimeProtocolError("selection child receipt config/manifest bindings differ from the schedule")
    status = receipt.get("status")
    if status not in {"complete", "incomplete", "not_started", "operational_failure"}:
        raise RuntimeProtocolError("selection child receipt has an unsupported terminal status")
    transport = receipt.get("transport_exit_code")
    expected_transport = 0 if status in {"complete", "incomplete", "not_started"} else 1
    if transport != expected_transport:
        raise RuntimeProtocolError("selection receipt transport status is inconsistent with its status")
    reservation_info = receipt.get("reservation")
    if reservation is None:
        if reservation_info is not None:
            raise RuntimeProtocolError("unreserved selection operation reports a fabricated reservation")
    else:
        if (not isinstance(reservation_info, Mapping)
                or reservation_info.get("raw_sha256") != reservation_hash
                or reservation_info.get("reserved_step_slots") != SELECTION_RESERVATION
                or reservation_info.get("path") != _repo_relative(
                    root, _operation_reservation_file(root, row["operation_id"]),
                    label="selection reservation",
                )):
            raise RuntimeProtocolError("selection receipt does not bind its durable parent reservation")
        entered_file = _operation_entered_file(root, row["operation_id"])
        entered, _ = _read_json(entered_file, label="selection operation entry")
        if (entered.get("operation_id") != row["operation_id"]
                or entered.get("reservation_raw_sha256") != reservation_hash
                or entered.get("state") != "attempted_once"):
            raise RuntimeProtocolError("selection child entry does not bind its single reservation")
    if status in {"complete", "incomplete"}:
        report = receipt.get("report")
        lifecycle = receipt.get("lifecycle")
        if (not isinstance(report, Mapping) or not isinstance(lifecycle, Mapping)
                or lifecycle.get("status") != status
                or lifecycle.get("exit_code") != receipt.get("runner_exit_code")
                or receipt.get("close_status") != "clean"
                or receipt.get("case_id") != row["case_id"]):
            raise RuntimeProtocolError("healthy selection receipt has inconsistent runner lifecycle/case evidence")
        expected_runner_exit = 0 if status == "complete" else 1
        if receipt.get("runner_exit_code") != expected_runner_exit:
            raise RuntimeProtocolError("selection runner exit code is inconsistent with its lifecycle")
        accounting = receipt.get("case_accounting")
        if (not isinstance(accounting, Mapping) or accounting.get("scheduled") != 1
                or accounting.get("complete") != (1 if status == "complete" else 0)):
            raise RuntimeProtocolError("selection case accounting is not a one-case result")
        counts = receipt.get("counts")
        if not isinstance(counts, Mapping):
            raise RuntimeProtocolError("selection receipt omits actual counter projection")
        for name, value in counts.items():
            if value is not None and (type(value) is not int or value < 0):
                raise RuntimeProtocolError(f"selection counter {name} is not a non-negative integer or unknown")
        if (counts.get("uncertain_environment_slots") != 0
                or counts.get("uncertain_reset_slots") != 0):
            raise RuntimeProtocolError("selection receipt reports uncertain environment or reset effects")
        learned_decisions = counts.get("learned_decisions")
        if expected_policy.get("kind") == "learned":
            attempted_slots = counts.get("attempted_environment_slots")
            if (type(learned_decisions) is not int or type(attempted_slots) is not int
                    or learned_decisions > attempted_slots
                    or learned_decisions > SELECTION_RESERVATION):
                raise RuntimeProtocolError("learned selection choices exceed observed or reserved step slots")
        elif learned_decisions != 0:
            raise RuntimeProtocolError("reference selection receipt reports learned choices")
        optimizer = receipt.get("optimizer")
        if (not isinstance(optimizer, Mapping)
                or any(optimizer.get(name) != 0 for name in (
                    "attempted_steps", "accepted_steps", "uncertain_steps", "updates",
                ))):
            raise RuntimeProtocolError("selection receipt reports optimizer effects")
        _check_operation_artifacts(root, receipt)
    if expected_policy.get("kind") == "learned":
        frozen_policies = bindings.get("policies")
        if not isinstance(frozen_policies, list):
            raise RuntimeProtocolError("frozen policy inventory is malformed")
        frozen_matches = [item for item in frozen_policies
                          if isinstance(item, Mapping) and item.get("policy_id") == row["policy_id"]]
        if len(frozen_matches) != 1:
            raise RuntimeProtocolError("learned selection policy has no unique frozen binding")
        frozen = frozen_matches[0]
        if frozen.get("status") == "available" and status in {"complete", "incomplete"}:
            config, _, _ = _selection_config(root, row, _case_binding(study, row))
            checkpoint_path = _bound_path(
                root, frozen.get("checkpoint_directory"), label="frozen selection checkpoint",
            )
            preflight = _resolve_dependency(
                dependencies, "checkpoint_preflight", _default_checkpoint_preflight,
            )
            current = _checkpoint_inventory(
                root, checkpoint_path, config, preflight,
                expected_inventory=frozen.get("inventory"),
            )
            checkpoint = receipt.get("checkpoint")
            if (not isinstance(checkpoint, Mapping)
                    or checkpoint.get("training_receipt") != frozen.get("training_receipt")
                    or checkpoint.get("before") != frozen.get("inventory")
                    or checkpoint.get("after") != frozen.get("inventory")
                    or checkpoint.get("factory_checkpoint_path") != frozen.get("checkpoint_directory")
                    or checkpoint.get("runtime_parent_checkpoint") != frozen.get("checkpoint_directory")
                    or current.get("inventory") != frozen.get("inventory")):
                raise RuntimeProtocolError("selection child checkpoint evidence differs from the frozen bytes")
    return receipt, raw


def _schedule_not_started(row: Mapping[str, object], reason: str) -> dict[str, object]:
    return {
        **_schedule_identity(row),
        "status": "not_started",
        "reason": reason,
        "dispatch_state": "not_dispatched",
        "receipt_path": None,
        "receipt_raw_sha256": None,
        "reservation_raw_sha256": None,
        "reserved_step_slots": 0,
        "attempted_effects": False,
        "counts": _selection_count_projection(None),
        "child_exit_code": None,
    }


def _schedule_unresolved(
    root: Path, row: Mapping[str, object], reason: str,
    reservation_hash: str | None, error: BaseException | None = None, *,
    reservation_attempted: bool = False, child_dispatched: bool | None = None,
    reservation_publication: Mapping[str, object] | None = None,
) -> dict[str, object]:
    reserved = reservation_attempted or reservation_hash is not None
    dispatched = reservation_hash is not None if child_dispatched is None else child_dispatched
    result = {
        **_schedule_identity(row),
        "status": "unresolved",
        "reason": reason,
        "dispatch_state": "possibly_entered" if dispatched else "not_dispatched",
        "receipt_path": _repo_relative(
            root, _operation_receipt_file(root, row["operation_id"]), label="selection receipt",
        ) if reserved else None,
        "receipt_raw_sha256": None,
        "reservation_raw_sha256": reservation_hash,
        "reserved_step_slots": SELECTION_RESERVATION if reserved else 0,
        "attempted_effects": None if dispatched else False,
        "counts": _selection_count_projection(None),
        "child_exit_code": None,
        "error": _error_value(error) if error is not None else None,
    }
    if reservation_publication is not None:
        result["reservation_publication"] = dict(reservation_publication)
    return result


def _schedule_totals(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    names = (
        "attempted_environment_slots", "forwarded_environment_slots",
        "known_accepted_environment_slots", "delivered_environment_slots",
        "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
        "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
        "learned_decisions", "fixed_decisions", "incomplete_decisions",
    )
    totals: dict[str, object] = {}
    for name in names:
        known = [value for row in rows
                 if isinstance((counts := row.get("counts")), Mapping)
                 and type(value := counts.get(name)) is int]
        unknown = sum(
            1 for row in rows
            if not isinstance(row.get("counts"), Mapping) or row["counts"].get(name) is None
        )
        totals[name] = {"known_sum": sum(known), "unknown_rows": unknown}
    return totals


def run_selection_schedule(
    root: Path, study_path: Path, bindings_path: Path, output_dir: Path, *,
    deadline: float, clock: Callable[[], float] = time.monotonic,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, object]:
    """Dispatch each frozen selection identity once under one shared deadline."""
    deadline_value = _deadline(deadline)
    first_clock = _now(clock)
    root_path, study, study_file, study_raw = _validate_study(root, study_path)
    binding_record, binding_raw = _binding_file(root_path, study_raw, bindings_path)
    selection_root = _bound_path(root_path, SELECTION_OUTPUT_RELATIVE, label="selection output root")
    output = _caller_path(
        root_path, Path(output_dir), expected=selection_root, label="selection output directory",
    )
    _make_directories_for(output / "control" / "schedule-plan.json")
    plan_file = output / "control" / "schedule-plan.json"
    result_file = output / "control" / "schedule-result.json"
    for path, label in ((plan_file, "selection schedule plan"),
                        (result_file, "selection schedule result")):
        _assert_safe_path(root_path, path, label=label)
    if (plan_file.exists() or plan_file.is_symlink()
            or result_file.exists() or result_file.is_symlink()):
        raise FileExistsError("selection schedule already has a plan or result; replay is forbidden")
    selection = study.get("selection")
    schedule = selection.get("schedule") if isinstance(selection, Mapping) else None
    if (not isinstance(schedule, list) or len(schedule) != SELECTION_ROWS
            or len({row.get("operation_id") for row in schedule if isinstance(row, Mapping)}) != SELECTION_ROWS):
        raise RuntimeProtocolError("validated Task A schedule must contain 192 unique operation identities")
    frozen_policies = binding_record.get("policies")
    if not isinstance(frozen_policies, list):
        raise RuntimeProtocolError("checkpoint binding policy inventory is malformed")
    frozen_by_id = {
        item["policy_id"]: item for item in frozen_policies
        if isinstance(item, Mapping) and isinstance(item.get("policy_id"), str)
    }
    rows = [dict(row) for row in schedule]
    plan = {
        "schema_id": "spire-learn-task9-selection-schedule-plan-v1",
        "state": "planned_before_effects",
        "study_raw_sha256": _sha256_bytes(study_raw),
        "binding_raw_sha256": _sha256_bytes(binding_raw),
        "shared_soft_deadline_monotonic": deadline_value,
        "operation_count": len(rows),
        "reserved_step_slots_max": SELECTION_RESERVED_MAX,
        "rows": [_schedule_identity(row) for row in rows],
    }
    _exclusive_json(plan_file, plan)

    stop_required = binding_record["stop_required_policy_ids"]
    if stop_required:
        reason = f"prior_training_fault:{','.join(stop_required)}"
        result_rows = [_schedule_not_started(row, reason) for row in rows]
        primary: BaseException | None = None
        try:
            finished = _now(clock)
        except BaseException as error:
            primary = error
            finished = None
            reason = "final_clock_process_control" if _is_control(error) else "final_clock_error"
        result: dict[str, object] = {
            "schema_id": "spire-learn-task9-selection-schedule-result-v1",
            "status": (
                "interrupted" if primary is not None and _is_control(primary) else
                "operational_failure" if primary is not None else "not_started"
            ),
            "reason": reason,
            "study_raw_sha256": _sha256_bytes(study_raw),
            "binding_raw_sha256": _sha256_bytes(binding_raw),
            "shared_soft_deadline_monotonic": deadline_value,
            "rows": result_rows,
            "attempted_operation_count": 0,
            "complete_operation_count": 0,
            "incomplete_operation_count": 0,
            "not_started_operation_count": SELECTION_ROWS,
            "unresolved_operation_count": 0,
            "operational_failure_count": 0,
            "reserved_operation_count": 0,
            "reserved_step_slots": 0,
            "reserved_step_slots_max": SELECTION_RESERVED_MAX,
            "transport_exit_code": 1 if primary is not None else 0,
            "progress_totals": _schedule_totals(result_rows),
            "outer_timing": {
                "start_monotonic": first_clock, "finish_monotonic": finished,
                "elapsed_seconds": max(0.0, finished - first_clock) if finished is not None else None,
                "timing_boundary": "schedule admission through pre-publication finalization",
            },
        }
        if primary is not None:
            result["error"] = _error_value(primary)
        projected = _project_json(result)
        assert isinstance(projected, dict)
        published = False
        try:
            _write_result_preserving_control(
                result_file, projected, primary, context="selection schedule result",
            )
            published = True
        except BaseException as error:
            primary = error
        if primary is not None and (_is_control(primary) or not published):
            raise primary
        return projected

    phase_context = dependencies.phase_context if dependencies is not None else None
    if phase_context is None:
        # Keep the durable plan as a precise record of the refusal to enter a
        # live child phase; callers must supply Task C's verified phase context.
        result_rows = [_schedule_not_started(row, "phase_context_dependency_missing") for row in rows]
        primary: BaseException | None = None
        try:
            finished = _now(clock)
        except BaseException as error:
            primary = error
            finished = None
        reason = (
            "final_clock_process_control" if primary is not None and _is_control(primary)
            else "final_clock_error" if primary is not None
            else "phase_context_dependency_missing"
        )
        result = {
            "schema_id": "spire-learn-task9-selection-schedule-result-v1",
            "status": (
                "interrupted" if primary is not None and _is_control(primary) else
                "operational_failure" if primary is not None else "not_started"
            ),
            "reason": reason,
            "study_raw_sha256": _sha256_bytes(study_raw),
            "binding_raw_sha256": _sha256_bytes(binding_raw),
            "shared_soft_deadline_monotonic": deadline_value,
            "rows": result_rows,
            "attempted_operation_count": 0,
            "complete_operation_count": 0,
            "incomplete_operation_count": 0,
            "not_started_operation_count": SELECTION_ROWS,
            "unresolved_operation_count": 0,
            "operational_failure_count": 0,
            "reserved_operation_count": 0,
            "reserved_step_slots": 0,
            "reserved_step_slots_max": SELECTION_RESERVED_MAX,
            "transport_exit_code": 1 if primary is not None else 0,
            "progress_totals": _schedule_totals(result_rows),
            "outer_timing": {
                "start_monotonic": first_clock, "finish_monotonic": finished,
                "elapsed_seconds": max(0.0, finished - first_clock) if finished is not None else None,
                "timing_boundary": "schedule admission through pre-publication finalization",
            },
        }
        if primary is not None:
            result["error"] = _error_value(primary)
        projected = _project_json(result)
        assert isinstance(projected, dict)
        published = False
        try:
            _write_result_preserving_control(
                result_file, projected, primary, context="selection schedule result",
            )
            published = True
        except BaseException as error:
            primary = error
        if primary is not None and (_is_control(primary) or not published):
            raise primary
        return projected

    rows_out: list[dict[str, object]] = []
    primary: BaseException | None = None
    stop_reason: str | None = None
    current_index = 0
    binding_hash = _sha256_bytes(binding_raw)
    child_runner = _resolve_dependency(dependencies, "child_process", _default_child_process)
    executable = dependencies.python_executable if dependencies is not None else None
    executable = executable or sys.executable
    script = _bound_path(root_path, "scripts/neural/task9_study_runtime.py", label="selection child script")

    for current_index, row in enumerate(rows):
        reservation: dict[str, object] | None = None
        reservation_hash: str | None = None
        reservation_attempted = False
        child_dispatched = False
        if stop_reason is not None:
            rows_out.append(_schedule_not_started(row, stop_reason))
            continue
        try:
            now = _now(clock)
            if now >= deadline_value:
                stop_reason = "shared_soft_deadline_exhausted_before_child"
                rows_out.append(_schedule_not_started(row, stop_reason))
                continue
            if _directory_bytes(output) > SELECTION_OUTPUT_LIMIT:
                raise OutputAccountingError("selection output exceeds its 2 GiB accounting ceiling")
            policy = _selection_policy(study, row["policy_id"])
            frozen = frozen_by_id.get(row["policy_id"]) if policy.get("kind") == "learned" else None
            eligible = policy.get("kind") != "learned" or (
                isinstance(frozen, Mapping) and frozen.get("status") == "available"
            )
            if eligible:
                reservation_attempted = True
                reservation, reservation_hash = _reserve_scheduled_operation(
                    root_path, row, binding_hash, deadline_value, clock,
                )
            receipt_file = _operation_receipt_file(root_path, row["operation_id"])
            argv = [
                executable, str(script), "selection-child",
                "--root", str(root_path), "--study", study_file.relative_to(root_path).as_posix(),
                "--phase-context", str(phase_context), "--deadline", str(deadline_value),
                "--operation-id", str(row["operation_id"]), "--bindings", str(_checkpoint_binding_path(root_path)),
                "--receipt", str(receipt_file),
            ]

            def progress_check() -> None:
                current_bytes = _directory_bytes(output)
                if current_bytes > SELECTION_OUTPUT_LIMIT:
                    raise OutputAccountingError("selection output exceeds its 2 GiB accounting ceiling")

            # Charge traversal, reservation/fsync and argv construction to one
            # absolute close boundary; soft expiry never creates a new grace.
            dispatch_now = _now(clock)
            timeout = deadline_value + CHILD_CLOSE_GRACE_SECONDS - dispatch_now
            if timeout <= 0:
                rows_out.append(_schedule_unresolved(
                    root_path, row, "shared_close_grace_exhausted_before_child", reservation_hash,
                    reservation_attempted=reservation_attempted, child_dispatched=False,
                ))
                stop_reason = f"shared_close_grace_exhausted_before_child:{row['operation_id']}"
                continue
            child_dispatched = True
            child = child_runner(
                argv, capture_output=True, text=True, timeout=timeout, check=False,
                output_limit_bytes=CHILD_STREAM_LIMIT, progress_check=progress_check,
            )
            if not isinstance(child, subprocess.CompletedProcess):
                raise TypeError("bounded child boundary must return subprocess.CompletedProcess")
            stdout = child.stdout if isinstance(child.stdout, str) else ""
            stderr = child.stderr if isinstance(child.stderr, str) else ""
            stdout_raw, stderr_raw = stdout.encode("utf-8"), stderr.encode("utf-8")
            child_io = {
                "stdout_bytes": len(stdout_raw), "stdout_prefix_sha256": _sha256_bytes(stdout_raw),
                "stderr_bytes": len(stderr_raw), "stderr_prefix_sha256": _sha256_bytes(stderr_raw),
                "stream_limit_bytes": CHILD_STREAM_LIMIT,
            }
            if (getattr(child, "timed_out", False)
                    or getattr(child, "stdout_truncated", False)
                    or getattr(child, "stderr_truncated", False)
                    or len(stdout_raw) > CHILD_STREAM_LIMIT or len(stderr_raw) > CHILD_STREAM_LIMIT):
                rows_out.append(_schedule_unresolved(
                    root_path, row, "child_timeout_or_output_limit", reservation_hash,
                    reservation_attempted=reservation_attempted, child_dispatched=child_dispatched,
                ) | {"child_exit_code": child.returncode, "child_io": child_io})
                stop_reason = f"child_timeout_or_output_limit:{row['operation_id']}"
                continue
            if not receipt_file.is_file() or receipt_file.is_symlink():
                rows_out.append(_schedule_unresolved(
                    root_path, row, "child_result_missing", reservation_hash,
                    reservation_attempted=reservation_attempted, child_dispatched=child_dispatched,
                ) | {"child_exit_code": child.returncode, "child_io": child_io})
                stop_reason = f"child_result_missing:{row['operation_id']}"
                continue
            receipt, receipt_raw = _reconcile_selection_receipt(
                root_path, study, study_raw, row, binding_record, binding_raw,
                receipt_file, reservation, reservation_hash, dependencies,
            )
            child_status = receipt["status"]
            expected_child_exit = receipt["transport_exit_code"]
            if child.returncode != expected_child_exit:
                raise RuntimeProtocolError("child process exit differs from its actual operation receipt")
            if child_status == "operational_failure":
                row_status = "operational_failure"
                stop_reason = f"operation_fault:{row['operation_id']}"
            else:
                row_status = child_status
            counts = receipt.get("counts")
            rows_out.append({
                **_schedule_identity(row),
                "status": row_status,
                "reason": receipt.get("reason"),
                "dispatch_state": "receipt_reconciled",
                "receipt_path": _repo_relative(root_path, receipt_file, label="selection receipt"),
                "receipt_raw_sha256": _sha256_bytes(receipt_raw),
                "reservation_raw_sha256": reservation_hash,
                "reserved_step_slots": SELECTION_RESERVATION if reservation is not None else 0,
                "attempted_effects": receipt.get("attempted_effects"),
                "counts": counts if isinstance(counts, Mapping) else _selection_count_projection(None),
                "runner_exit_code": receipt.get("runner_exit_code"),
                "transport_exit_code": receipt.get("transport_exit_code"),
                "child_exit_code": child.returncode,
                "child_io": child_io,
                "run_id": receipt.get("run_id"),
                "summary_sha256": receipt.get("summary_sha256"),
                "lifecycle_sha256": receipt.get("lifecycle_sha256"),
            })
            if stop_reason is not None:
                continue
            if _directory_bytes(output) > SELECTION_OUTPUT_LIMIT:
                rows_out[-1]["status"] = "operational_failure"
                rows_out[-1]["reason"] = "selection_output_accounting_limit"
                stop_reason = f"selection_output_accounting_limit:{row['operation_id']}"
        except BaseException as error:
            if primary is None or (_is_control(error) and not _is_control(primary)):
                if primary is not None:
                    _note(error, f"selection schedule failure followed {type(primary).__name__}: {_error_value(primary)['message']}")
                primary = error
            else:
                _note(primary, f"selection schedule also failed: {type(error).__name__}: {_error_value(error)['message']}")
            publication = None
            if reservation_attempted and reservation_hash is None:
                try:
                    publication = _reservation_publication_evidence(root_path, row, binding_hash, deadline_value)
                    reservation_hash = publication.get("raw_sha256")
                except BaseException as inspection_error:
                    publication = {"state": "uncertain", "footprint_present": None,
                                   "record_status": "inspection_interrupted", "raw_sha256": None,
                                   "error": _error_value(inspection_error)}
                    if _is_control(inspection_error) and not _is_control(primary):
                        _note(inspection_error, f"reservation inspection followed {type(primary).__name__}: {_error_value(primary)['message']}")
                        primary = inspection_error
                    else:
                        _note(primary, f"reservation inspection also failed: {type(inspection_error).__name__}: {_error_value(inspection_error)['message']}")
            rows_out.append(_schedule_unresolved(
                root_path, row, "schedule_dispatch_or_reconciliation_error", reservation_hash, primary,
                reservation_attempted=reservation_attempted, child_dispatched=child_dispatched,
                reservation_publication=publication,
            ))
            stop_reason = f"schedule_error:{row['operation_id']}"

    if len(rows_out) < SELECTION_ROWS:
        for row in rows[len(rows_out):]:
            rows_out.append(_schedule_not_started(row, stop_reason or "campaign_stopped"))
    final_clock_error: BaseException | None = None
    try:
        final_clock = _now(clock)
    except BaseException as error:
        final_clock_error = error
        if primary is None or (_is_control(error) and not _is_control(primary)):
            if primary is not None:
                _note(error, f"schedule final clock followed {type(primary).__name__}: {_error_value(primary)['message']}")
            primary = error
        else:
            _note(primary, f"schedule final clock also failed: {type(error).__name__}: {_error_value(error)['message']}")
        final_clock = None
        stop_reason = (
            "schedule_final_clock_process_control" if _is_control(primary)
            else "schedule_final_clock_error"
        )

    counts_by_status = {
        status: sum(1 for row in rows_out if row.get("status") == status)
        for status in ("complete", "incomplete", "not_started", "operational_failure", "unresolved")
    }
    reserved_slots = sum(
        int(row.get("reserved_step_slots", 0))
        for row in rows_out if type(row.get("reserved_step_slots")) is int
    )
    reserved_operations = sum(1 for row in rows_out if row.get("reserved_step_slots") == SELECTION_RESERVATION)
    entered_count = sum(
        1 for row in rows_out
        if row.get("reserved_step_slots") == SELECTION_RESERVATION
        and (_operation_entered_file(root_path, row["operation_id"]).is_file()
             or row.get("dispatch_state") == "possibly_entered")
    )
    if stop_reason is None:
        status = "incomplete" if counts_by_status["incomplete"] else "complete"
    elif counts_by_status["operational_failure"] or counts_by_status["unresolved"]:
        status = "stopped"
    elif counts_by_status["not_started"]:
        status = "not_started"
    else:
        status = "incomplete" if counts_by_status["incomplete"] else "complete"
    if final_clock_error is not None:
        status = "interrupted" if _is_control(primary) else "operational_failure"
    result: dict[str, object] = {
        "schema_id": "spire-learn-task9-selection-schedule-result-v1",
        "status": status,
        "reason": stop_reason,
        "study_raw_sha256": _sha256_bytes(study_raw),
        "binding_raw_sha256": binding_hash,
        "shared_soft_deadline_monotonic": deadline_value,
        "rows": rows_out,
        "attempted_operation_count": entered_count,
        "complete_operation_count": counts_by_status["complete"],
        "incomplete_operation_count": counts_by_status["incomplete"],
        "not_started_operation_count": counts_by_status["not_started"],
        "operational_failure_count": counts_by_status["operational_failure"],
        "unresolved_operation_count": counts_by_status["unresolved"],
        "transport_exit_code": 1 if (
            final_clock_error is not None
            or counts_by_status["operational_failure"]
            or counts_by_status["unresolved"]
        ) else 0,
        "reserved_operation_count": reserved_operations,
        "reserved_step_slots": reserved_slots,
        "reserved_step_slots_max": SELECTION_RESERVED_MAX,
        "progress_totals": _schedule_totals(rows_out),
        "outer_timing": {
            "start_monotonic": first_clock,
            "finish_monotonic": final_clock,
            "elapsed_seconds": max(0.0, final_clock - first_clock) if final_clock is not None else None,
            "timing_boundary": "schedule admission through pre-publication finalization",
        },
    }
    if primary is not None:
        result["error"] = _error_value(primary)
    result = _project_json(result)
    assert isinstance(result, dict)
    result_published = False
    try:
        _exclusive_json(result_file, result)
        result_published = True
    except BaseException as error:
        if primary is None or (_is_control(error) and not _is_control(primary)):
            if primary is not None:
                _note(error, f"schedule result publication followed {type(primary).__name__}: {_error_value(primary)['message']}")
            primary = error
        else:
            _note(primary, f"schedule result publication also failed: {type(error).__name__}: {_error_value(error)['message']}")
    if primary is not None and (_is_control(primary) or not result_published):
        raise primary
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the prepared Task 9 learner study guest operations")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "selection", "selection-child"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--study", required=True, type=Path)
        command.add_argument("--phase-context", required=True, type=Path)
        command.add_argument("--deadline", required=True, type=float)
        if name == "train":
            command.add_argument("--policy-id", required=True)
            command.add_argument("--receipt", required=True, type=Path)
        elif name == "selection":
            command.add_argument("--bindings", required=True, type=Path)
            command.add_argument("--output-dir", required=True, type=Path)
        else:
            command.add_argument("--operation-id", required=True)
            command.add_argument("--bindings", required=True, type=Path)
            command.add_argument("--receipt", required=True, type=Path)
    return parser


def _verify_cli_context(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    phase = args.policy_id if args.command == "train" else "selection"
    kwargs = {
        "root": args.root,
        "study_path": args.study,
        "phase_context": args.phase_context,
        "phase": phase,
        "deadline": _deadline(args.deadline),
        "policy_id": args.policy_id if args.command == "train" else None,
        "operation_id": args.operation_id if args.command == "selection-child" else None,
        "bindings_path": args.bindings if args.command != "train" else None,
        "receipt_path": args.receipt if args.command != "selection" else None,
        "output_dir": args.output_dir if args.command == "selection" else None,
    }
    try:
        # This module is deliberately absent until Task C supplies it. Import it
        # only after argparse has handled --help and before runtime imports.
        import_root = str(args.root)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        controls = importlib.import_module("scripts.neural.task9_study_controls")
        verifier = getattr(controls, "verify_guest_context")
        verified = verifier(**kwargs)
        if not isinstance(verified, Mapping) or verified.get("verified") is not True:
            raise RuntimeProtocolError("phase-context verifier did not return verified=true")
    except Exception as error:
        parser.error(f"phase-context verification failed: {type(error).__name__}: {error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _verify_cli_context(parser, args)
    try:
        if args.command == "train":
            result = run_training_operation(
                args.root, args.study, args.policy_id, args.receipt,
                deadline=args.deadline,
            )
            return int(result["transport_exit_code"])
        if args.command == "selection":
            result = run_selection_schedule(
                args.root, args.study, args.bindings, args.output_dir,
                deadline=args.deadline,
                dependencies=RuntimeDependencies(
                    phase_context=args.phase_context,
                    python_executable=sys.executable,
                ),
            )
            return int(result["transport_exit_code"])
        result = run_selection_operation(
            args.root, args.study, args.operation_id, args.bindings, args.receipt,
            deadline=args.deadline,
        )
        return int(result["transport_exit_code"])
    except BaseException as error:
        if _is_control(error):
            raise
        print(f"Task 9 study operation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
