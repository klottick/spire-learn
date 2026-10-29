"""Small stdlib-only controls for the separately released Task 9 GPU proof."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import tomllib
from typing import Any


GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
ACCEPTED_LEARNER_HEAD = "7e37450c84f0893f5dc798ca4f23304c8b30fb14"
IMAGE_REFERENCE = (
    "docker.io/library/python@sha256:"
    "57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
)
IMAGE_ID = "sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
CUDA_LOCK_SHA256 = "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45"
NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
TASK8_RUNTIME = Path("artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a")
CUDA_VENV = TASK8_RUNTIME / "install-work/venv-cuda"
CUDA_LOCK = TASK8_RUNTIME / "resolve-work/requirements-cuda.lock"
CUDA_NATIVE = CUDA_VENV / "lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
IMAGE_IDENTITY = Path(
    "artifacts/development/neural-combat/task-8/calibration/runtime/"
    "calibration-20261028-c/image-identity.json"
)
GPU_RECORD = TASK8_RUNTIME / "device-check-work/device-check.json"
RELEASE_BINDING = Path(
    "artifacts/development/neural-combat/task-8/calibration/runtime/"
    "calibration-20261028-c/release-binding.json"
)

SNAPSHOT_SCHEMA = "spire-learn-neural-gpu-proof-source-snapshot-v1"
RELEASE_SCHEMA = "spire-learn-neural-gpu-proof-parent-release-v1"
CLAIM_SCHEMA = "spire-learn-neural-gpu-proof-release-claim-v1"
INVENTORY_SCHEMA = "spire-learn-neural-gpu-proof-evidence-inventory-v1"
MODE_RESULT_SCHEMA = "spire-learn-neural-gpu-proof-mode-result-v1"
PROOF_ROOT_GUEST = "/work/artifacts/development/neural-combat/task-9/gpu-proof"
GUEST_RUNTIME_SCRATCH = PROOF_ROOT_GUEST + "/.runtime"
CONTAINER_PYTHON = "/runtime/venv-cuda/bin/python3.12"
LOCAL_DOCKER_ENDPOINT = "unix:///var/run/docker.sock"
DOCKER_IMAGE_DEFAULT_ENV = {
    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "GPG_KEY": "7169605F62C751356D054A26A821E680E5FA6305",
    "PYTHON_VERSION": "3.12.15",
    "PYTHON_SHA256": "c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a",
}
GUEST_SCRATCH_ENV = {
    "TMPDIR": GUEST_RUNTIME_SCRATCH + "/tmp",
    "XDG_CACHE_HOME": GUEST_RUNTIME_SCRATCH + "/cache",
    "TORCH_HOME": GUEST_RUNTIME_SCRATCH + "/torch",
    "TORCH_EXTENSIONS_DIR": GUEST_RUNTIME_SCRATCH + "/torch-extensions",
    "CUDA_CACHE_PATH": GUEST_RUNTIME_SCRATCH + "/cuda",
    "TRITON_CACHE_DIR": GUEST_RUNTIME_SCRATCH + "/triton",
}
GUEST_SOURCE_PATHS = (
    "src", "tests", "configs", "scripts/neural/gpu_runtime_proof_helpers.py",
)
HOST_COMMAND_OUTPUT_LIMIT_BYTES = 1024 * 1024
CHECKPOINT_RESERVATION_BYTES = 384 * 1024 * 1024
NEURAL_EVENT_LOG_RESERVATION_BYTES = 64 * 1024 * 1024
HOST_LOG_RESERVATION_BYTES = 2 * 1024 * 1024
OUTPUT_METADATA_RESERVATION_BYTES = 16 * 1024 * 1024
RUNTIME_SCRATCH_RESERVATION_BYTES = 96 * 1024 * 1024
OUTPUT_ADMISSION_RESERVATION_BYTES = (
    2 * CHECKPOINT_RESERVATION_BYTES
    + 2 * NEURAL_EVENT_LOG_RESERVATION_BYTES
    + 2 * HOST_LOG_RESERVATION_BYTES
    + OUTPUT_METADATA_RESERVATION_BYTES
    + RUNTIME_SCRATCH_RESERVATION_BYTES
)
SNAPSHOT_PATHS = (
    "src", "tests", "configs",
    "scripts/neural/gpu_runtime_proof_helpers.py",
    "scripts/neural/run-gpu-runtime-proof.sh",
    "scripts/neural/gpu-runtime-proof-runbook.md",
)
LIMITS: dict[str, object] = {
    "soft_deadline_seconds": 120,
    "whole_run_seconds": 300,
    "cleanup_reserve_seconds": 30,
    "step_slots": 64,
    "reset_slots": 69,
    "optimizer_attempts": 8,
    "actor_sample_rows": 64,
    "diagnostic_sample_rows": 4,
    "sample_rows": 68,
    "output_bytes": 1024**3,
    "retained_neural_bytes": 40 * 1024**3,
    "minimum_gpu_free_bytes": 12 * 1024**3,
    "minimum_disk_free_bytes": 30 * 1024**3,
    "host_memory_bytes": 8 * 1024**3,
    "gpu_allocator_bytes": 8 * 1024**3,
    "cpus": 4,
    "pids": 512,
    "gpu_uuid": GPU_UUID,
}
MODE_LIMITS = {
    "standalone": {"steps": 32, "resets": 36, "actor_samples": 32, "diagnostic_samples": 2, "optimizer": 4},
    "run": {"steps": 32, "resets": 33, "actor_samples": 32, "diagnostic_samples": 2, "optimizer": 4},
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_ID = re.compile(r"[0-9a-f]{40,64}\Z")
_COUNTER_NAMES = ("steps", "resets", "actor_samples", "diagnostic_samples", "optimizer")
MODE_RESULT_COUNTER_FIELDS = frozenset({
    "steps", "resets", "actor_samples", "diagnostic_samples", "optimizer",
    "steps_attempted", "steps_delivered", "steps_uncertain",
    "resets_attempted", "resets_delivered", "resets_uncertain",
    "actor_samples_attempted", "actor_samples_delivered", "actor_samples_uncertain",
    "diagnostic_samples_attempted", "diagnostic_samples_delivered", "diagnostic_samples_uncertain",
    "optimizer_attempted", "optimizer_accepted", "optimizer_uncertain",
    "faults", "incomplete_environment_slots",
})


class EffectLimitError(RuntimeError):
    """An effect did not fit the remaining cumulative proof budget."""


class ProofDeadlineExceeded(TimeoutError):
    """The single shared absolute proof deadline has elapsed."""


class ProofCommandTimeout(TimeoutError):
    def __init__(self, argv, timeout, *, stdout_prefix="", stderr_prefix=""):
        self.argv = tuple(argv)
        self.timeout = timeout
        self.stdout_prefix = stdout_prefix
        self.stderr_prefix = stderr_prefix
        super().__init__(f"host command timed out after {timeout:.3f}s: {self.argv[0]}")


class ProofCommandOutputLimit(RuntimeError):
    def __init__(
        self, argv, limit, *, stdout_prefix="", stderr_prefix="",
        stdout_truncated=False, stderr_truncated=False,
    ):
        self.argv = tuple(argv)
        self.limit = limit
        self.stdout_prefix = stdout_prefix
        self.stderr_prefix = stderr_prefix
        self.stdout_truncated = bool(stdout_truncated)
        self.stderr_truncated = bool(stderr_truncated)
        super().__init__(f"host command output exceeded {limit} bytes per stream: {self.argv[0]}")


class _WatchdogTermination(BaseException):
    """TERM from the outer watchdog; caught only to run bounded owned cleanup."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_replace(path: Path, value: Mapping[str, object]) -> None:
    """Replace a mutable host status record while keeping file and directory durable."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing symlink report: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mode(path: Path) -> str:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"source is not a regular file: {path}")
    return "100755" if info.st_mode & 0o111 else "100644"


def _source_files(root: Path, roots: Sequence[str]) -> tuple[str, ...]:
    root = Path(root).resolve(strict=True)
    expanded: set[str] = set()
    for item in roots:
        pure = PurePosixPath(item)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe source root: {item}")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"source escapes or links outside root: {item}")
        candidates = sorted(path.rglob("*")) if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"source contains symlink: {candidate.relative_to(root)}")
            if candidate.is_dir() or "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
                continue
            if not candidate.is_file():
                raise ValueError(f"source is not a regular file: {candidate}")
            expanded.add(candidate.relative_to(root).as_posix())
    if not expanded:
        raise ValueError("source snapshot cannot be empty")
    return tuple(sorted(expanded))


def _source_inventory(root: Path, paths: Sequence[str]) -> dict[str, dict[str, object]]:
    inventory = {}
    for name in paths:
        path = Path(root) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source path is missing, linked, or non-regular: {name}")
        inventory[name] = {"sha256": sha256_file(path), "git_mode": _mode(path)}
    return inventory


def _git_state(root: Path, *, timeout: float = 10.0) -> tuple[str, dict[str, object]]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=timeout, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True, text=True, timeout=timeout, check=True,
    ).stdout
    return revision, {"dirty": bool(status), "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def task8_identity_inputs(root: Path) -> dict[str, object]:
    root = Path(root)
    lock = root / CUDA_LOCK
    native = root / CUDA_NATIVE
    image_record = load_json(root / IMAGE_IDENTITY)
    gpu_record = load_json(root / GPU_RECORD)
    release_record = load_json(root / RELEASE_BINDING)
    actual_lock, actual_native = sha256_file(lock), sha256_file(native)
    if actual_lock != CUDA_LOCK_SHA256 or actual_native != NATIVE_SHA256:
        raise ValueError("Task 8 lock or native binary hash differs from the brief")
    if image_record.get("image_reference") != IMAGE_REFERENCE or image_record.get("image_id") != IMAGE_ID:
        raise ValueError("Task 8 image identity record differs from the qualified immutable image")
    if (
        gpu_record.get("status") != "qualified"
        or gpu_record.get("gpu_uuid") != GPU_UUID
        or gpu_record.get("visible_gpu_uuid") != GPU_UUID
        or gpu_record.get("device_name") != "NVIDIA GeForce RTX 5090"
        or gpu_record.get("driver_version") != "595.91.07"
        or gpu_record.get("compute_capability") != [12, 0]
        or gpu_record.get("torch") != "2.14.1+cu130"
        or gpu_record.get("torch_cuda") != "13.0"
    ):
        raise ValueError("Task 8 GPU qualification identity differs from the brief")
    if (
        release_record.get("cuda_lock_sha256") != CUDA_LOCK_SHA256
        or release_record.get("native_sha256") != NATIVE_SHA256
        or release_record.get("gpu_uuid") != GPU_UUID
    ):
        raise ValueError("Task 8 release binding differs from the qualified inputs")
    return {
        "lock_path": CUDA_LOCK.as_posix(), "lock_sha256": actual_lock,
        "native_path": CUDA_NATIVE.as_posix(), "native_sha256": actual_native,
        "image_reference": IMAGE_REFERENCE, "image_id": IMAGE_ID,
        "image_identity_record_sha256": sha256_file(root / IMAGE_IDENTITY),
        "gpu_record_sha256": sha256_file(root / GPU_RECORD),
        "release_binding_sha256": sha256_file(root / RELEASE_BINDING),
        "gpu_uuid": GPU_UUID,
        "task8_image_identity": image_record,
        "task8_gpu_record": gpu_record,
        "task8_release_binding": release_record,
    }


def create_source_snapshot(
    root: Path,
    destination: Path,
    *,
    source_paths: Sequence[str] = SNAPSHOT_PATHS,
    learner_git_revision: str | None = None,
    dirty_state: Mapping[str, object] | None = None,
    identity_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    paths = _source_files(root, source_paths)
    if learner_git_revision is None or dirty_state is None:
        learner_git_revision, dirty_state = _git_state(root)
    if not _GIT_ID.fullmatch(learner_git_revision):
        raise ValueError("learner Git revision must be a full hexadecimal object ID")
    source_inventory = _source_inventory(root, paths)
    snapshot = {
        "schema_id": SNAPSHOT_SCHEMA,
        "status": "source_only",
        "learner_git_revision": learner_git_revision,
        "dirty_state": dict(dirty_state),
        "source_files": source_inventory,
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "identity_inputs": dict(identity_inputs or task8_identity_inputs(root)),
        "path_count": len(paths),
    }
    write_json_exclusive(Path(destination), snapshot)
    return snapshot


def verify_source_snapshot(
    root: Path, snapshot_path: Path, *, expected_paths: Sequence[str] | None = SNAPSHOT_PATHS
) -> dict[str, object]:
    snapshot = load_json(snapshot_path)
    if snapshot.get("schema_id") != SNAPSHOT_SCHEMA or snapshot.get("status") != "source_only":
        raise ValueError("source snapshot schema/status is invalid")
    source_files = snapshot.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source snapshot inventory is empty")
    current_paths = _source_files(root, expected_paths or tuple(source_files))
    if tuple(sorted(source_files)) != current_paths:
        raise ValueError("source snapshot path inventory differs from the captured tree")
    current = _source_inventory(root, current_paths)
    if current != source_files:
        raise ValueError("source snapshot bytes or Git modes changed")
    if snapshot.get("path_count") != len(current_paths):
        raise ValueError("source snapshot path count differs")
    if snapshot.get("source_inventory_sha256") != canonical_sha256(source_files):
        raise ValueError("source snapshot canonical inventory hash differs")
    return {
        "snapshot_raw_sha256": sha256_file(snapshot_path),
        "snapshot_canonical_sha256": canonical_sha256(snapshot),
        "learner_git_revision": snapshot["learner_git_revision"],
        "dirty_state": snapshot["dirty_state"],
        "path_count": len(current_paths),
        "identity_inputs": snapshot["identity_inputs"],
    }


def _guest_source_inventory(
    source_files: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if not isinstance(source_files, Mapping):
        raise ValueError("released source inventory is not a mapping")
    entries = {}
    for name, entry in source_files.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise ValueError("released source inventory has an invalid entry")
        if any(
            name == prefix or name.startswith(prefix.rstrip("/") + "/")
            for prefix in GUEST_SOURCE_PATHS
        ):
            if (
                set(entry) != {"sha256", "git_mode"}
                or not isinstance(entry.get("sha256"), str)
                or not _SHA256.fullmatch(entry["sha256"])
                or entry.get("git_mode") not in {"100644", "100755"}
            ):
                raise ValueError(f"released guest source entry is invalid: {name}")
            entries[name] = dict(entry)
    if not entries:
        raise ValueError("released inventory has no guest learner source files")
    return entries


def _released_guest_source_bytes(root: Path, source_files: Mapping[str, object]) -> int:
    """Count the apparent bytes that the released guest source stage will copy."""
    root = Path(root).resolve(strict=True)
    expected = _guest_source_inventory(source_files)
    canonical_paths = _source_files(root, GUEST_SOURCE_PATHS)
    if set(canonical_paths) != set(expected):
        raise ValueError("canonical guest source paths differ from the released inventory")
    total = 0
    for name in canonical_paths:
        path = root.joinpath(*PurePosixPath(name).parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"released guest source path escapes or links outside root: {name}")
        file_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"released guest source is not a regular file: {name}")
        total += file_stat.st_size
    return total


def verify_staged_source_tree(
    staged_root: Path, source_files: Mapping[str, object],
) -> dict[str, object]:
    staged_root = Path(staged_root)
    if staged_root.is_symlink() or not staged_root.is_dir():
        raise ValueError("staged learner source root is missing or linked")
    expected = _guest_source_inventory(source_files)
    expected_files = set(expected)
    expected_dirs = {"."}
    for name in expected_files:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_dirs: set[str] = {"."}
    for path in staged_root.rglob("*"):
        relative = path.relative_to(staged_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"staged learner source contains a symlink: {relative}")
        if path.is_dir():
            actual_dirs.add(relative)
            if path.stat(follow_symlinks=False).st_mode & 0o222:
                raise ValueError(f"staged learner source directory is writable: {relative}")
        elif path.is_file():
            actual_files.add(relative)
            if path.stat(follow_symlinks=False).st_mode & 0o222:
                raise ValueError(f"staged learner source file is writable: {relative}")
        else:
            raise ValueError(f"staged learner source contains a non-regular path: {relative}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("staged learner source paths differ from the released guest inventory")
    current = _source_inventory(staged_root, tuple(sorted(expected_files)))
    if current != expected:
        raise ValueError("staged learner source bytes or normalized Git modes differ from release")
    return {
        "path": str(staged_root.resolve(strict=True)),
        "path_count": len(current),
        "source_inventory_sha256": canonical_sha256(current),
        "read_only": True,
    }


def stage_released_source_tree(
    root: Path, source_files: Mapping[str, object], staged_root: Path,
) -> dict[str, object]:
    """Copy only released learner inputs, verify bytes/modes, and make them read-only."""
    root = Path(root).resolve(strict=True)
    staged_root = Path(staged_root)
    if staged_root.exists() or staged_root.is_symlink():
        raise FileExistsError("staged source destination must be new")
    if staged_root.parent.is_symlink() or not staged_root.parent.is_dir():
        raise ValueError("staged source parent must be an existing regular directory")
    expected = _guest_source_inventory(source_files)
    canonical_paths = _source_files(root, GUEST_SOURCE_PATHS)
    if set(canonical_paths) != set(expected):
        raise ValueError("canonical guest source paths differ from the released inventory")
    canonical = _source_inventory(root, canonical_paths)
    if canonical != expected:
        raise ValueError("canonical guest source bytes or modes changed before staging")

    staged_root.mkdir(mode=0o700)
    for name in sorted(expected):
        source = root / name
        destination = staged_root / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        digest = hashlib.sha256()
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, source_flags)
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"canonical guest source changed type during staging: {name}")
            with os.fdopen(source_fd, "rb", closefd=False) as source_stream:
                with destination.open("xb") as destination_stream:
                    for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        destination_stream.write(chunk)
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
        finally:
            os.close(source_fd)
        if digest.hexdigest() != expected[name]["sha256"]:
            raise ValueError(f"canonical guest source changed while staging: {name}")
        mode = 0o555 if expected[name]["git_mode"] == "100755" else 0o444
        destination.chmod(mode)
    for directory in sorted(
        (path for path in staged_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True,
    ):
        directory.chmod(0o555)
    staged_root.chmod(0o555)
    return verify_staged_source_tree(staged_root, source_files)


def prepare_guest_scratch(output_child: Path) -> dict[str, str]:
    """Create writable temp/cache targets inside the one guest RW output bind."""
    output_child = Path(output_child)
    if output_child.is_symlink() or not output_child.is_dir():
        raise ValueError("guest scratch requires the new owned output directory")
    scratch_root = output_child / ".runtime"
    scratch_root.mkdir(mode=0o700, exist_ok=True)
    if scratch_root.is_symlink() or not scratch_root.resolve(strict=True).is_relative_to(
        output_child.resolve(strict=True),
    ):
        raise ValueError("guest runtime scratch escaped the owned output directory")
    paths = {}
    for name, guest_path in GUEST_SCRATCH_ENV.items():
        relative = guest_path.removeprefix(PROOF_ROOT_GUEST + "/")
        if not relative.startswith(".runtime/"):
            raise ValueError("guest scratch environment escaped its reserved subtree")
        target = output_child / relative
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(
            output_child.resolve(strict=True),
        ):
            raise ValueError(f"guest scratch path is linked or escaped: {name}")
        target.chmod(0o700)
        paths[name] = guest_path
    expected_dirs = {".runtime"}
    expected_dirs.update(
        path.removeprefix(PROOF_ROOT_GUEST + "/") for path in GUEST_SCRATCH_ENV.values()
    )
    actual_dirs = {
        path.relative_to(output_child).as_posix()
        for path in output_child.rglob("*") if path.is_dir() and not path.is_symlink()
    }
    if actual_dirs != expected_dirs:
        raise ValueError("guest runtime scratch directory layout differs from its env allowlist")
    return paths


def output_tree_bytes(output_child: Path) -> int:
    """Count closed and in-progress output without following links."""
    output_child = Path(output_child)
    if output_child.is_symlink() or not output_child.is_dir():
        raise ValueError("proof output child is missing or linked")
    total = 0
    for path in output_child.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"proof output contains a symlink: {path.relative_to(output_child)}")
        if path.is_file():
            total += path.stat(follow_symlinks=False).st_size
        elif not path.is_dir():
            raise ValueError(f"proof output contains a non-regular path: {path.relative_to(output_child)}")
    return total


def validate_config_pair(root: Path) -> dict[str, dict[str, object]]:
    root = Path(root)
    result = {}
    for mode, rows, steps, resets in (("standalone", 4, 8, 36), ("run", 1, 32, 33)):
        source_path = root / "configs/neural" / f"{mode}.toml"
        proof_path = root / "configs/neural" / f"{mode}-gpu-smoke.toml"
        source = tomllib.loads(source_path.read_text(encoding="utf-8"))
        proof = tomllib.loads(proof_path.read_text(encoding="utf-8"))
        mutable = {
            "device", "output_root", "runtime", "rollout", "budgets",
        }
        for key in set(source) | set(proof):
            if key not in mutable and source.get(key) != proof.get(key):
                raise ValueError(f"{mode} config changed preserved identity section {key}")
        if proof.get("schema") != source["schema"] or proof.get("mode") != mode:
            raise ValueError(f"{mode} config schema/mode differs")
        if proof.get("operation") != "train" or proof.get("split") != "train" or proof.get("partition") != "train":
            raise ValueError(f"{mode} proof config must train on the train partition")
        if proof.get("manifest_path") != source["manifest_path"]:
            raise ValueError(f"{mode} manifest path differs")
        if proof.get("device") != "cuda:0" or proof.get("runtime") != {"gpu_uuid": GPU_UUID}:
            raise ValueError(f"{mode} config is not bound to the qualified CUDA UUID")
        if proof.get("output_root") != f"../../artifacts/development/neural-combat/task-9/gpu-proof/{mode}":
            raise ValueError(f"{mode} output root is not the stable Task 9 proof path")
        if proof.get("rollout") != {"rows": rows, "steps": steps}:
            raise ValueError(f"{mode} rollout geometry differs")
        if proof.get("budgets") != {
            "max_optimizer_steps": 4,
            "max_training_decisions": 32,
            "max_environment_actions": 32,
            "max_wall_seconds": 120.0,
        }:
            raise ValueError(f"{mode} budgets differ")
        result[mode] = {
            "path": proof_path.relative_to(root).as_posix(),
            "sha256": sha256_file(proof_path),
            "rollout": proof["rollout"],
            "reset_slots": resets,
        }
    return result


def verify_release(
    release_path: Path, *, snapshot_path: Path, run_id: str, limits: Mapping[str, object] = LIMITS
) -> dict[str, object]:
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run ID is unsafe")
    release = load_json(release_path)
    snapshot_hashes = {
        "source_snapshot_raw_sha256": sha256_file(snapshot_path),
        "source_snapshot_canonical_sha256": canonical_sha256(load_json(snapshot_path)),
    }
    if release.get("schema_id") != RELEASE_SCHEMA or release.get("status") != "released":
        raise ValueError("parent release schema/status is invalid")
    if release.get("run_id") != run_id:
        raise ValueError("parent release run_id differs")
    if release.get("limits") != dict(limits):
        raise ValueError("parent release limits differ")
    for key, value in snapshot_hashes.items():
        if release.get(key) != value:
            raise ValueError(f"parent release {key} differs")
    if not isinstance(release.get("release_id"), str) or not release["release_id"]:
        raise ValueError("parent release ID is missing")
    return {
        "release_raw_sha256": sha256_file(release_path),
        "release_canonical_sha256": canonical_sha256(release),
        **snapshot_hashes,
        "release_id": release["release_id"],
        "run_id": run_id,
    }


def create_release_claim(claim_path: Path, release_binding: Mapping[str, object]) -> dict[str, object]:
    claim = {
        "schema_id": CLAIM_SCHEMA,
        "status": "claimed_once",
        "release_raw_sha256": release_binding["release_raw_sha256"],
        "release_canonical_sha256": release_binding["release_canonical_sha256"],
        "snapshot_raw_sha256": release_binding["source_snapshot_raw_sha256"],
        "snapshot_canonical_sha256": release_binding["source_snapshot_canonical_sha256"],
        "release_id": release_binding["release_id"],
        "run_id": release_binding["run_id"],
    }
    write_json_exclusive(claim_path, claim)
    return claim


def verify_release_claim(claim_path: Path, release_binding: Mapping[str, object]) -> dict[str, object]:
    claim = load_json(claim_path)
    if claim.get("schema_id") != CLAIM_SCHEMA or claim.get("status") != "claimed_once":
        raise ValueError("exclusive release claim schema/status is invalid")
    expected = {
        "release_raw_sha256": release_binding["release_raw_sha256"],
        "release_canonical_sha256": release_binding["release_canonical_sha256"],
        "snapshot_raw_sha256": release_binding["source_snapshot_raw_sha256"],
        "snapshot_canonical_sha256": release_binding["source_snapshot_canonical_sha256"],
        "release_id": release_binding["release_id"],
        "run_id": release_binding["run_id"],
    }
    if any(claim.get(key) != value for key, value in expected.items()):
        raise ValueError("exclusive claim does not bind the captured release and snapshot")
    return claim


def create_evidence_inventory(root: Path, destination: Path) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    destination = Path(destination)
    files = {}
    for path in sorted(root.rglob("*")):
        if path == destination:
            continue
        if path.is_symlink():
            raise ValueError(f"evidence contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            files[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    inventory = {"schema_id": INVENTORY_SCHEMA, "files": files, "file_count": len(files)}
    write_json_exclusive(destination, inventory)
    return inventory


def verify_evidence_inventory(root: Path, inventory_path: Path) -> dict[str, object]:
    inventory = load_json(inventory_path)
    if inventory.get("schema_id") != INVENTORY_SCHEMA or not isinstance(inventory.get("files"), dict):
        raise ValueError("evidence inventory schema is invalid")
    current = {}
    for path in sorted(Path(root).rglob("*")):
        if path == Path(inventory_path):
            continue
        if path.is_symlink():
            raise ValueError(f"evidence contains a symlink: {path}")
        if path.is_file():
            current[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
    if current != inventory["files"] or inventory.get("file_count") != len(current):
        raise ValueError("evidence inventory differs from current files")
    return inventory


def verify_test_transcript(text: str) -> dict[str, object]:
    required = (
        r"^test_01_standalone_b4_h8 \(test_neural_gpu_runtime\.NeuralGpuRuntimeProofTests\.test_01_standalone_b4_h8\) \.\.\. ok$",
        r"^test_02_run_b1_h32 \(test_neural_gpu_runtime\.NeuralGpuRuntimeProofTests\.test_02_run_b1_h32\) \.\.\. ok$",
    )
    if any(len(re.findall(pattern, text, re.MULTILINE)) != 1 for pattern in required):
        raise ValueError("test transcript does not contain each exact mode once")
    if not all(re.search(pattern, text, re.MULTILINE) for pattern in required):
        raise ValueError("test transcript does not show both mode methods passing")
    if "Ran 2 tests" not in text or not re.search(r"(?m)^OK\s*$", text):
        raise ValueError("test transcript is not exactly two successful unskipped tests")
    if any(word in text for word in ("skipped", "FAILED", "ERROR", "failures=", "errors=")):
        raise ValueError("test transcript reports a skip, failure, or error")
    return {
        "test_count": 2,
        "test_names": [
            "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
            "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32",
        ],
        "skipped": 0, "status": "passed",
    }


def verify_mode_evidence(output_child: Path) -> dict[str, object]:
    """Read the two closed guest result files and reconcile cumulative budgets."""
    output_child = Path(output_child)
    definitions = {
        "standalone": "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
        "run": "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32",
    }
    results = {}
    totals = defaultdict(int)
    for mode, test_id in definitions.items():
        path = output_child / mode / "gpu-proof-checks.json"
        result = validate_mode_result(load_json(path), mode=mode, test_id=test_id)
        counters = result["counters"]
        for name in (
            "steps_attempted", "steps_delivered", "steps_uncertain",
            "resets_attempted", "resets_delivered", "resets_uncertain",
            "actor_samples_attempted", "actor_samples_delivered", "actor_samples_uncertain",
            "diagnostic_samples_attempted", "diagnostic_samples_delivered",
            "diagnostic_samples_uncertain", "optimizer_attempted", "optimizer_accepted",
            "optimizer_uncertain",
        ):
            totals[name] += counters.get(name, 0)
        results[mode] = {
            "relative_path": f"{mode}/gpu-proof-checks.json",
            "raw_sha256": sha256_file(path),
            "canonical_sha256": canonical_sha256(result),
            "test_id": test_id,
            "updates": result["updates"],
            "counters": counters,
            "weights": result.get("weights"),
        }
    limits = {
        "steps_attempted": 64,
        "resets_attempted": 69,
        "actor_samples_attempted": 64,
        "diagnostic_samples_attempted": 4,
        "optimizer_attempted": 8,
    }
    for name, maximum in limits.items():
        if type(totals[name]) is not int or totals[name] > maximum:
            raise ValueError(f"mode evidence cumulative {name} exceeds its released cap")
    for name in (
        "steps", "resets", "actor_samples", "diagnostic_samples", "optimizer",
    ):
        if totals[f"{name}_uncertain"] != 0:
            raise ValueError(f"mode evidence cumulative {name} has uncertain effects")
    for name in ("steps", "resets", "actor_samples", "diagnostic_samples"):
        if totals[f"{name}_attempted"] != totals[f"{name}_delivered"]:
            raise ValueError(f"mode evidence cumulative {name} has incomplete effects")
    if totals["diagnostic_samples_delivered"] != 4:
        raise ValueError("mode evidence is missing a reserved diagnostic pair")
    if totals["optimizer_attempted"] != totals["optimizer_accepted"] or totals["optimizer_accepted"] < 2:
        raise ValueError("mode evidence optimizer attempts are incomplete")
    return {"modes": results, "cumulative_counters": dict(totals)}


class EffectLedger:
    """Cumulative reservations made before driver, sample, or optimizer effects."""

    _MAXIMUMS = {
        "steps": LIMITS["step_slots"],
        "resets": LIMITS["reset_slots"],
        "actor_samples": LIMITS["actor_sample_rows"],
        "diagnostic_samples": LIMITS["diagnostic_sample_rows"],
        "optimizer": LIMITS["optimizer_attempts"],
    }

    def __init__(self, deadline: float, clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(float(deadline)) or not callable(clock):
            raise ValueError("ledger requires a finite absolute deadline and clock")
        self.deadline = float(deadline)
        self.clock = clock
        self.totals = defaultdict(int)
        self.modes = {mode: defaultdict(int) for mode in MODE_LIMITS}
        self.diagnostic_reserved = defaultdict(int)

    def check_deadline(self) -> None:
        if self.clock() >= self.deadline:
            raise ProofDeadlineExceeded("shared 120-second proof deadline expired")

    def reserve(self, mode: str, kind: str, count: int) -> None:
        self.check_deadline()
        if mode not in MODE_LIMITS or kind not in self._MAXIMUMS or type(count) is not int or count < 0:
            raise ValueError("invalid proof reservation")
        per_mode_key = {
            "steps": "steps", "resets": "resets", "actor_samples": "actor_samples",
            "diagnostic_samples": "diagnostic_samples", "optimizer": "optimizer",
        }[kind]
        if self.totals[kind] + count > self._MAXIMUMS[kind]:
            raise EffectLimitError(f"cumulative {kind} reservation exceeds the proof limit")
        if self.modes[mode][per_mode_key] + count > MODE_LIMITS[mode][per_mode_key]:
            raise EffectLimitError(f"{mode} {kind} reservation exceeds the mode limit")
        self.totals[kind] += count
        self.modes[mode][per_mode_key] += count
        self.totals[f"{kind}_attempted"] += count
        self.modes[mode][f"{kind}_attempted"] += count

    def reserve_diagnostic_pair(self, mode: str) -> None:
        if self.diagnostic_reserved[mode]:
            raise EffectLimitError("diagnostic pair was already reserved for this mode")
        self.reserve(mode, "diagnostic_samples", 2)
        self.diagnostic_reserved[mode] = 2

    def complete(self, mode: str, kind: str, *, delivered: int = 0, uncertain: int = 0, accepted: int = 0) -> None:
        for name, count in (("delivered", delivered), ("uncertain", uncertain), ("accepted", accepted)):
            if type(count) is not int or count < 0:
                raise ValueError("effect completion counts must be non-negative integers")
            if count:
                self.totals[f"{kind}_{name}"] += count
                self.modes[mode][f"{kind}_{name}"] += count

    def claim_diagnostic_rows(self, mode: str, count: int) -> None:
        self.check_deadline()
        if type(count) is not int or count < 1 or self.diagnostic_reserved[mode] < count:
            raise EffectLimitError("diagnostic sample rows were not reserved as a pair")
        self.diagnostic_reserved[mode] -= count

    def snapshot(self) -> dict[str, object]:
        return {
            **{key: self.totals[key] for key in (
                "steps", "resets", "actor_samples", "diagnostic_samples", "optimizer",
                "steps_attempted", "steps_delivered", "steps_uncertain",
                "resets_attempted", "resets_delivered", "resets_uncertain",
                "actor_samples_attempted", "actor_samples_delivered", "actor_samples_uncertain",
                "diagnostic_samples_attempted", "diagnostic_samples_delivered", "diagnostic_samples_uncertain",
                "optimizer_attempted", "optimizer_accepted", "optimizer_uncertain",
            )},
            "modes": {mode: dict(values) for mode, values in self.modes.items()},
            "diagnostic_rows_unclaimed": dict(self.diagnostic_reserved),
            "deadline": self.deadline,
            "clock": self.clock(),
        }

    def mode_snapshot(self, mode: str) -> dict[str, int]:
        if mode not in self.modes:
            raise ValueError("unknown proof mode")
        return dict(self.modes[mode])


def project_mode_counters(
    ledger: EffectLedger,
    mode: str,
    *,
    faults: int,
    incomplete_environment_slots: int,
) -> dict[str, int]:
    """Project one real sparse ledger snapshot into the strict public counter schema."""
    if not isinstance(ledger, EffectLedger):
        raise TypeError("mode counter projection requires an EffectLedger")
    snapshot = ledger.mode_snapshot(mode)
    internal_fields = MODE_RESULT_COUNTER_FIELDS - {"faults", "incomplete_environment_slots"}
    allowed_internal_fields = internal_fields | {"optimizer_delivered"}
    unknown = set(snapshot) - allowed_internal_fields
    if unknown:
        raise ValueError(f"mode ledger contains unknown counters: {sorted(map(repr, unknown))}")
    for name, value in snapshot.items():
        if not isinstance(name, str) or type(value) is not int or value < 0:
            raise ValueError("mode ledger counters must be named non-negative integers")

    projected = {name: 0 for name in MODE_RESULT_COUNTER_FIELDS}
    for name, value in snapshot.items():
        if name != "optimizer_delivered":
            projected[name] = value
    delivered = snapshot.get("optimizer_delivered", 0)
    accepted = snapshot.get("optimizer_accepted", 0)
    if delivered != accepted:
        raise ValueError("optimizer delivery does not reconcile with accepted attempts")
    for name, value in (
        ("faults", faults),
        ("incomplete_environment_slots", incomplete_environment_slots),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        projected[name] = value
    return projected


class GuardedDriver:
    def __init__(self, inner: Any, ledger: EffectLedger, mode: str):
        self.inner, self.ledger, self.mode = inner, ledger, mode
        self.close_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self, *, seeds, entry_ids, reset_mask):
        slots = sum(bool(value) for value in reset_mask)
        self.ledger.reserve(self.mode, "resets", slots)
        try:
            result = self.inner.reset(seeds=seeds, entry_ids=entry_ids, reset_mask=reset_mask)
        except BaseException as error:
            evidence = getattr(error, "evidence", None)
            known = int(getattr(evidence, "known_current_delivery_slots", 0)) if evidence is not None else 0
            uncertain = int(getattr(evidence, "uncertain_current_slots", slots - known)) if evidence is not None else slots
            self.ledger.complete(self.mode, "resets", delivered=max(0, min(slots, known)), uncertain=max(0, min(slots-known, uncertain)))
            raise
        self.ledger.complete(self.mode, "resets", delivered=slots)
        return result

    def step(self, actions):
        slots = len(actions)
        self.ledger.reserve(self.mode, "steps", slots)
        try:
            result = self.inner.step(actions)
        except BaseException as error:
            evidence = getattr(error, "evidence", None)
            known = int(getattr(evidence, "known_current_delivery_slots", 0)) if evidence is not None else 0
            uncertain = int(getattr(evidence, "uncertain_current_slots", slots - known)) if evidence is not None else slots
            self.ledger.complete(self.mode, "steps", delivered=max(0, min(slots, known)), uncertain=max(0, min(slots-known, uncertain)))
            raise
        self.ledger.complete(self.mode, "steps", delivered=slots)
        return result

    def close(self):
        self.close_calls += 1
        return self.inner.close()


class GuardedPolicy:
    def __init__(self, inner: Any, ledger: EffectLedger, mode: str, *, diagnostic: bool = False):
        self.inner, self.ledger, self.mode, self.diagnostic = inner, ledger, mode, diagnostic

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def act(self, frames, *, generator, deterministic=False):
        rows = len(frames)
        if deterministic:
            self.ledger.check_deadline()
        elif self.diagnostic:
            self.ledger.claim_diagnostic_rows(self.mode, rows)
        else:
            self.ledger.reserve(self.mode, "actor_samples", rows)
        try:
            result = self.inner.act(frames, generator=generator, deterministic=deterministic)
        except BaseException:
            if not deterministic:
                self.ledger.complete(
                    self.mode, "diagnostic_samples" if self.diagnostic else "actor_samples",
                    uncertain=rows,
                )
            raise
        if not deterministic:
            self.ledger.complete(
                self.mode, "diagnostic_samples" if self.diagnostic else "actor_samples",
                delivered=rows,
            )
        return result


class GuardedOptimizer:
    def __init__(self, inner: Any, ledger: EffectLedger, mode: str):
        self.inner, self.ledger, self.mode = inner, ledger, mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def step(self, *args, **kwargs):
        self.ledger.reserve(self.mode, "optimizer", 1)
        try:
            result = self.inner.step(*args, **kwargs)
        except BaseException:
            self.ledger.complete(self.mode, "optimizer", uncertain=1)
            raise
        self.ledger.complete(self.mode, "optimizer", delivered=1, accepted=1)
        return result


def validate_mode_result(
    result: Mapping[str, object], *, mode: str, test_id: str,
) -> dict[str, object]:
    if result.get("schema_id") != MODE_RESULT_SCHEMA or result.get("status") != "passed":
        raise ValueError("mode proof result schema/status is invalid")
    if result.get("mode") != mode or result.get("test_id") != test_id:
        raise ValueError("mode proof result identity differs")
    if result.get("skipped") is not False or result.get("exit_code") != 0:
        raise ValueError("mode proof cannot pass with a skip or nonzero exit")
    counters = result.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("mode result counters are missing")
    if set(counters) != MODE_RESULT_COUNTER_FIELDS:
        missing = MODE_RESULT_COUNTER_FIELDS - set(counters)
        extra = set(counters) - MODE_RESULT_COUNTER_FIELDS
        raise ValueError(f"mode result counter fields differ; missing={sorted(missing)}, extra={sorted(extra)}")
    for name in MODE_RESULT_COUNTER_FIELDS:
        if type(counters[name]) is not int or counters[name] < 0:
            raise ValueError(f"mode result counter {name} must be a non-negative integer")
    for name, value in counters.items():
        if name.endswith("_uncertain") and value != 0:
            raise ValueError(f"mode result counter {name} is uncertain")
    expected = MODE_LIMITS[mode]
    bounded = {
        "steps": expected["steps"], "resets": expected["resets"],
        "actor_samples": expected["actor_samples"],
        "diagnostic_samples": expected["diagnostic_samples"],
        "optimizer": expected["optimizer"],
        "steps_attempted": expected["steps"], "steps_delivered": expected["steps"],
        "resets_attempted": expected["resets"], "resets_delivered": expected["resets"],
        "actor_samples_attempted": expected["actor_samples"],
        "actor_samples_delivered": expected["actor_samples"],
        "optimizer_attempted": expected["optimizer"], "optimizer_accepted": expected["optimizer"],
    }
    for name, maximum in bounded.items():
        count = counters.get(name)
        if count > maximum:
            raise ValueError(f"mode result counter {name} exceeds its released limit")
    for aggregate, attempted, delivered, uncertain in (
        ("steps", "steps_attempted", "steps_delivered", "steps_uncertain"),
        ("resets", "resets_attempted", "resets_delivered", "resets_uncertain"),
        ("actor_samples", "actor_samples_attempted", "actor_samples_delivered", "actor_samples_uncertain"),
        ("diagnostic_samples", "diagnostic_samples_attempted", "diagnostic_samples_delivered", "diagnostic_samples_uncertain"),
        ("optimizer", "optimizer_attempted", "optimizer_accepted", "optimizer_uncertain"),
    ):
        if counters[aggregate] != counters[attempted]:
            raise ValueError(f"mode result counter {aggregate} does not match its attempted prefix")
        if counters[attempted] != counters[delivered] + counters[uncertain]:
            raise ValueError(f"mode result counter {aggregate} delivery is unresolved")
    if not counters["steps_delivered"] or not counters["resets_delivered"]:
        raise ValueError("mode result has no accepted environment or reset effects")
    if counters["diagnostic_samples_attempted"] != expected["diagnostic_samples"]:
        raise ValueError("mode result did not reserve exactly the diagnostic pair")
    if counters["diagnostic_samples_delivered"] != expected["diagnostic_samples"]:
        raise ValueError("mode result diagnostic pair is incomplete")
    if counters["optimizer_attempted"] != counters["optimizer_accepted"] or not counters["optimizer_accepted"]:
        raise ValueError("mode result optimizer attempt is unresolved or absent")
    if counters["faults"] != 0 or counters["incomplete_environment_slots"] != 0:
        raise ValueError("mode result contains a fault or incomplete environment effect")
    updates = result.get("updates")
    if (
        not isinstance(updates, Mapping)
        or type(updates.get("accepted")) is not int
        or updates.get("accepted", 0) < 1
        or updates.get("finite") is not True
        or updates.get("weights_changed") is not True
    ):
        raise ValueError("mode did not produce a finite accepted update with changed weights")
    checks = result.get("checks")
    required_checks = {
        "one_shared_deadline", "accepted_finite_update_and_changed_weights",
        "final_checkpoint_saved", "same_backend_weights_and_adam",
        "adam_scalar_steps_cpu_moments_cuda", "same_backend_torch_rng_and_python_rng",
        "same_backend_counters_version_and_recovery",
        "same_frame_deterministic_and_sampled_full_choice_parity",
        "portable_cpu_weights_and_tolerant_outputs",
        "warm_start_fresh_state_and_frozen_evaluation",
        "cpu_comparisons_used_inert_drivers_without_steps",
        "training_driver_closed_once", "actor_critic_and_shared_hashes_recorded",
        "zero_fault_uncertainty_and_incomplete_effects",
    }
    if (
        not isinstance(checks, Mapping)
        or not required_checks <= set(checks)
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("mode checkpoint parity/cleanup checks are incomplete")
    weights = result.get("weights")
    weight_names = {
        "model_sha256", "shared_body_sha256", "actor_head_sha256", "critic_head_sha256",
    }
    if (
        not isinstance(weights, Mapping)
        or set(weights) != weight_names
        or any(not isinstance(weights[name], str) or not _SHA256.fullmatch(weights[name]) for name in weight_names)
    ):
        raise ValueError("mode actor, critic, shared, and full-model hashes are incomplete")
    checkpoint = result.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint.get("path")
        or not isinstance(checkpoint.get("metadata_raw_sha256"), str)
        or not _SHA256.fullmatch(checkpoint["metadata_raw_sha256"])
        or not isinstance(checkpoint.get("metadata_canonical_sha256"), str)
        or not _SHA256.fullmatch(checkpoint["metadata_canonical_sha256"])
    ):
        raise ValueError("mode final checkpoint identity/hashes are incomplete")
    portable = result.get("portable_cpu")
    if (
        not isinstance(portable, Mapping)
        or portable.get("warm_start_optimizer_fresh") is not True
        or portable.get("warm_start_rng_fresh") is not True
        or portable.get("evaluation_optimizer_absent") is not True
        or portable.get("cpu_driver_resets") != 0
        or portable.get("cpu_driver_steps") != 0
    ):
        raise ValueError("mode portable CPU warm-start/evaluation evidence is incomplete")
    return dict(result)


def verify_test_exports(root: Path) -> tuple[str, ...]:
    """Lazy import check used inside the proof process after the gate."""
    import sys
    sys.path.insert(0, str(Path(root) / "src"))
    from spire_learn.neural import runner

    expected = {
        "NeuralReport", "RuntimeOperationLimits", "TrainingRuntime", "calibrate",
        "create_runtime", "evaluate_policy", "train",
    }
    if set(runner.__all__) != expected:
        raise ImportError("neural runner public export set changed")
    return tuple(sorted(expected))


def _container_environment(deadline: float) -> dict[str, str]:
    value = f"{deadline:.9f}"
    return {
        "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
        "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        "SPIRE_LEARN_GPU_RUNTIME_PROOF_HELPER": "/work/gpu_runtime_proof_helpers.py",
        "SPIRE_LEARN_NEURAL_GPU_SOFT_DEADLINE": value,
        "SPIRE_LEARN_GPU_RUNTIME_PROOF_OUTPUT": PROOF_ROOT_GUEST,
        **GUEST_SCRATCH_ENV,
        "PYTHONPATH": "/work/src:/work/tests/integration",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }


def assert_owned_container(
    inspected: Mapping[str, object], *, container_id: str, container_name: str,
    image_id: str, image_reference: str, user: str,
    expected_mounts: Mapping[str, tuple[str, bool]], lifecycle: str = "prestart",
) -> dict[str, object]:
    full_id = inspected.get("Id")
    if not isinstance(container_id, str) or not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError("container ownership requires one full 64-character ID")
    if full_id != container_id:
        raise ValueError("inspected container full ID differs")
    config = inspected.get("Config")
    host = inspected.get("HostConfig")
    if not isinstance(config, Mapping) or not isinstance(host, Mapping):
        raise ValueError("container inspect omits Config or HostConfig")
    if inspected.get("Name") != "/" + container_name:
        raise ValueError("container name differs")
    if inspected.get("Image") != image_id or config.get("Image") != image_reference:
        raise ValueError("container image digest/ID differs")
    expected_config = {
        "User": user, "WorkingDir": "/work",
        "Entrypoint": [CONTAINER_PYTHON],
        "Cmd": ["-B", "-m", "unittest", "-v", "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests"],
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("container Config does not match the released entrypoint, command, user, and network")

    env = config.get("Env")
    if not isinstance(env, list):
        raise ValueError("container environment inventory is missing")
    env_map: dict[str, str] = {}
    for item in env:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("container environment contains an invalid entry")
        key, value = item.split("=", 1)
        if not key or key in env_map:
            raise ValueError("container environment contains an empty or duplicate key")
        env_map[key] = value
    deadline = env_map.get("SPIRE_LEARN_NEURAL_GPU_SOFT_DEADLINE")
    try:
        deadline_number = float(deadline) if deadline is not None else math.nan
    except ValueError:
        deadline_number = math.nan
    if not math.isfinite(deadline_number):
        raise ValueError("container shared absolute deadline is missing or invalid")
    expected_env = {
        **DOCKER_IMAGE_DEFAULT_ENV,
        **_container_environment(deadline_number),
    }
    if env_map != expected_env:
        raise ValueError("container environment differs from pinned image defaults and proof allowlist")

    host_values = {
        "NetworkMode": "none", "ReadonlyRootfs": True, "NanoCpus": 4_000_000_000,
        "Memory": 8 * 1024**3, "MemorySwap": 8 * 1024**3, "PidsLimit": 512,
        "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges:true"],
        "Privileged": False,
    }
    if any(key not in host or host[key] != value for key, value in host_values.items()):
        raise ValueError("container HostConfig resource or privilege boundary differs")
    for key in ("CapAdd", "Devices", "DeviceCgroupRules", "GroupAdd"):
        if key not in host or host[key] not in (None, []):
            raise ValueError(f"container HostConfig {key} must be empty")
    requests = host.get("DeviceRequests")
    if (
        not isinstance(requests, list) or len(requests) != 1
        or not isinstance(requests[0], Mapping)
        or requests[0].get("Driver") != ""
        or requests[0].get("Count") != 0
        or requests[0].get("DeviceIDs") != [GPU_UUID]
        or requests[0].get("Capabilities") != [["gpu"]]
        or requests[0].get("Options") != {}
    ):
        raise ValueError("container does not request only the released GPU UUID")
    actual = {}
    mounts = inspected.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("container mount inventory is missing")
    for mount in mounts:
        if not isinstance(mount, Mapping) or mount.get("Type") != "bind":
            raise ValueError("container has a non-bind mount")
        destination = mount.get("Destination")
        if destination in actual:
            raise ValueError("container has a duplicate mount destination")
        actual[destination] = (mount.get("Source"), bool(mount.get("RW")))
    if actual != dict(expected_mounts):
        raise ValueError("container mount source/destination/permissions differ")
    state = inspected.get("State")
    if not isinstance(state, Mapping) or type(state.get("Running")) is not bool:
        raise ValueError("container lifecycle state is missing or invalid")
    status = state.get("Status")
    if lifecycle == "prestart":
        valid_state = state["Running"] is False and status == "created"
    elif lifecycle == "cleanup":
        valid_state = status in {"created", "running", "exited", "dead"}
        valid_state = valid_state and state["Running"] is (status == "running")
    else:
        raise ValueError("unknown container inspection lifecycle")
    if not valid_state:
        raise ValueError(f"container lifecycle state is invalid for {lifecycle} inspection")
    return {"container_id": full_id, "container_name": container_name, "owned": True}


def _docker_create_argv(
    *, docker: str, name: str, root: Path, source_tree: Path, output_child: Path,
    user: str, deadline: float, docker_client_config: Path,
) -> list[str]:
    root = Path(root).resolve(strict=True)
    source_tree = Path(source_tree).resolve(strict=True)
    client_config = Path(docker_client_config).resolve(strict=True)
    output_path = Path(output_child).resolve(strict=True)
    if not client_config.is_dir() or any(client_config.iterdir()):
        raise ValueError("Docker client config must be a new empty task-owned directory")
    prepare_guest_scratch(output_path)
    task8 = root / TASK8_RUNTIME
    native = root / CUDA_NATIVE
    mounts = [
        (source_tree / "src", "/work/src", "ro"),
        (source_tree / "tests", "/work/tests", "ro"),
        (source_tree / "configs", "/work/configs", "ro"),
        (source_tree / "scripts/neural/gpu_runtime_proof_helpers.py", "/work/gpu_runtime_proof_helpers.py", "ro"),
        (task8 / "resolve-work/requirements-cuda.lock", "/runtime/requirements-cuda.lock", "ro"),
        (task8 / "install-work/venv-cuda", "/runtime/venv-cuda", "ro"),
        (native, "/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so", "ro"),
        (output_path, PROOF_ROOT_GUEST, "rw"),
    ]
    argv = [
        docker, "--host", LOCAL_DOCKER_ENDPOINT, "--config", str(client_config),
        "create", "--name", name, "--user", user, "--workdir", "/work",
        "--entrypoint", CONTAINER_PYTHON, "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--cpus", "4", "--memory", str(8 * 1024**3), "--memory-swap", str(8 * 1024**3),
        "--pids-limit", "512", "--gpus", f"device={GPU_UUID}", "--pull=never",
    ]
    for source, destination, mode in mounts:
        specification = f"type=bind,src={source},dst={destination}"
        if mode == "ro":
            specification += ",readonly"
        argv.extend(["--mount", specification])
    for key, value in _container_environment(deadline).items():
        argv.extend(["--env", f"{key}={value}"])
    argv.extend([
        IMAGE_REFERENCE, "-B", "-m", "unittest", "-v",
        "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests",
    ])
    return argv


def _signal_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _bounded_subprocess_run(
    argv, *, capture_output, text, timeout, check, env=None,
    output_limit_bytes=HOST_COMMAND_OUTPUT_LIMIT_BYTES, progress_check=None,
):
    """Drain an owned child with per-stream prefix caps and deadline polling."""
    if not capture_output or not text or check:
        raise ValueError("proof host commands require captured text output and explicit status handling")
    if type(output_limit_bytes) is not int or output_limit_bytes < 1:
        raise ValueError("host command output limit must be a positive integer")
    process = subprocess.Popen(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=None if env is None else dict(env), start_new_session=(os.name == "posix"),
    )
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    timed_out = False
    callback_error = None
    deadline = time.monotonic() + timeout
    next_progress = time.monotonic()
    try:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            now = time.monotonic()
            if now >= deadline and not timed_out:
                timed_out = True
                _signal_owned_process(process)
            if progress_check is not None and callback_error is None and now >= next_progress and not timed_out:
                next_progress = now + 0.1
                try:
                    progress_check()
                except BaseException as error:
                    callback_error = error
                    _signal_owned_process(process)
            events = selector.select(0.05)
            for key, _mask in events:
                try:
                    chunk = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                room = output_limit_bytes - len(buffers[name])
                if room > 0:
                    buffers[name].extend(chunk[:room])
                if len(chunk) > max(0, room):
                    truncated[name] = True
                    _signal_owned_process(process)
        return_code = process.wait()
        if callback_error is not None:
            callback_error.add_note(
                "bounded host child was terminated after progress-control failure; "
                f"stdout bytes retained={len(buffers['stdout'])}, stderr bytes retained={len(buffers['stderr'])}"
            )
            raise callback_error
    except BaseException as error:
        _signal_owned_process(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        selector.close()
        if callback_error is not None and callback_error is not error:
            error.add_note(f"progress callback also raised {type(callback_error).__name__}")
        raise
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    result = subprocess.CompletedProcess(
        argv, return_code,
        buffers["stdout"].decode("utf-8", errors="replace"),
        buffers["stderr"].decode("utf-8", errors="replace"),
    )
    result.timed_out = timed_out
    result.stdout_truncated = truncated["stdout"]
    result.stderr_truncated = truncated["stderr"]
    return result


@dataclass
class HostDependencies:
    """Injectable subprocess, clock, and disk probes for pure control-flow tests."""

    clock: Callable[[], float] = time.monotonic
    run: Callable[..., subprocess.CompletedProcess] = _bounded_subprocess_run
    disk_usage: Callable[[Path], Any] = shutil.disk_usage
    uid: Callable[[], int] = os.getuid
    gid: Callable[[], int] = os.getgid


def _docker_command(docker: str, client_config: Path, *arguments: str) -> list[str]:
    client_config = Path(client_config).resolve(strict=True)
    if not client_config.is_dir() or any(client_config.iterdir()):
        raise ValueError("Docker client config must stay empty and task-owned")
    return [docker, "--host", LOCAL_DOCKER_ENDPOINT, "--config", str(client_config), *arguments]


def _docker_environment(client_config: Path) -> dict[str, str]:
    client_config = Path(client_config).resolve(strict=True)
    if not client_config.is_dir() or any(client_config.iterdir()):
        raise ValueError("Docker client config must stay empty and task-owned")
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "XDG_CONFIG_HOME": str(client_config),
        "LC_ALL": "C",
    }
    if "HOME" in os.environ:
        environment["HOME"] = os.environ["HOME"]
    return environment


def _run(
    deps: HostDependencies, argv: Sequence[str], *, timeout: float,
    env: Mapping[str, str] | None = None,
    progress_check: Callable[[], None] | None = None,
    output_limit_bytes: int = HOST_COMMAND_OUTPUT_LIMIT_BYTES,
) -> subprocess.CompletedProcess:
    if timeout <= 0:
        raise ProofDeadlineExceeded("whole-run deadline expired before a host command")
    result = deps.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=False,
        env=None if env is None else dict(env), output_limit_bytes=output_limit_bytes,
        progress_check=progress_check,
    )
    if not isinstance(result, subprocess.CompletedProcess):
        raise TypeError("host command dependency must return CompletedProcess")
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    stdout_prefix = stdout.encode("utf-8", errors="replace")[:output_limit_bytes].decode("utf-8", errors="replace")
    stderr_prefix = stderr.encode("utf-8", errors="replace")[:output_limit_bytes].decode("utf-8", errors="replace")
    if getattr(result, "timed_out", False):
        raise ProofCommandTimeout(argv, timeout, stdout_prefix=stdout_prefix, stderr_prefix=stderr_prefix)
    if (
        getattr(result, "stdout_truncated", False)
        or getattr(result, "stderr_truncated", False)
        or len(stdout.encode("utf-8", errors="replace")) > output_limit_bytes
        or len(stderr.encode("utf-8", errors="replace")) > output_limit_bytes
    ):
        raise ProofCommandOutputLimit(
            argv, output_limit_bytes, stdout_prefix=stdout_prefix, stderr_prefix=stderr_prefix,
            stdout_truncated=getattr(result, "stdout_truncated", False),
            stderr_truncated=getattr(result, "stderr_truncated", False),
        )
    return result


def _require_command(
    deps: HostDependencies, argv: Sequence[str], *, timeout: float = 10,
    env: Mapping[str, str] | None = None,
) -> str:
    result = _run(deps, argv, timeout=timeout, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {argv[0]} {result.stderr[-512:]}")
    return result.stdout.strip()


def _retained_bytes(root: Path) -> int:
    base = Path(root) / "artifacts/development/neural-combat"
    total = 0
    if not base.exists():
        return 0
    for path in base.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def _gpu_preflight(deps: HostDependencies, *, timeout: float = 10.0) -> dict[str, object]:
    text = _require_command(deps, [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,compute_cap,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=timeout)
    rows = [line.split(",") for line in text.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 5 and row[0].strip() == GPU_UUID]
    if len(matches) != 1:
        raise RuntimeError("GPU preflight did not return exactly the released UUID")
    uuid, name, driver, capability, free = (part.strip() for part in matches[0])
    if name != "NVIDIA GeForce RTX 5090" or driver != "595.91.07" or capability != "12.0":
        raise RuntimeError("GPU model, driver, or compute capability differs from Task 8")
    free_bytes = int(free) * 1024**2
    if free_bytes < LIMITS["minimum_gpu_free_bytes"]:
        raise RuntimeError("GPU has less than the released 12 GiB free-memory floor")
    return {"uuid": uuid, "name": name, "driver": driver, "compute_capability": capability, "free_bytes": free_bytes}


def _identity_and_preflight(
    root: Path, output_child: Path, deps: HostDependencies, *, whole_deadline: float,
    docker_binary: str, docker_client_config: Path,
) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    configs = validate_config_pair(root)
    identity = task8_identity_inputs(root)
    remaining = whole_deadline - deps.clock()
    if remaining <= 0:
        raise ProofDeadlineExceeded("whole-run deadline expired during source/config preflight")
    revision, dirty = _git_state(root, timeout=min(5.0, remaining / 2))
    if _retained_bytes(root) > LIMITS["retained_neural_bytes"]:
        raise RuntimeError("retained neural-combat storage exceeds 40 GiB")
    usage = deps.disk_usage(root)
    if int(usage.free) < LIMITS["minimum_disk_free_bytes"]:
        raise RuntimeError("repository filesystem has less than 30 GiB free")
    remaining = whole_deadline - deps.clock()
    if remaining <= 0:
        raise ProofDeadlineExceeded("whole-run deadline expired before GPU preflight")
    gpu = _gpu_preflight(deps, timeout=min(10.0, remaining))
    remaining = whole_deadline - deps.clock()
    if remaining <= 0:
        raise ProofDeadlineExceeded("whole-run deadline expired before image preflight")
    image_id = _require_command(deps, _docker_command(
        docker_binary, docker_client_config,
        "image", "inspect", IMAGE_REFERENCE, "--format", "{{.Id}}",
    ), timeout=min(10.0, remaining), env=_docker_environment(docker_client_config))
    if image_id != IMAGE_ID:
        raise RuntimeError("reviewed image digest is not present locally with the expected ID")
    output_child = Path(output_child)
    proof_root = (root / "artifacts/development/neural-combat/task-9/gpu-proof").resolve(strict=True)
    parent = output_child.parent.resolve(strict=True)
    if not parent.is_relative_to(proof_root) or output_child.is_symlink() or output_child.exists():
        raise ValueError("output child must be a new, non-symlink directory below the proof root")
    if OUTPUT_ADMISSION_RESERVATION_BYTES > LIMITS["output_bytes"]:
        raise RuntimeError("known proof outputs and runtime scratch exceed the 1 GiB admission cap")
    return {
        "configs": configs, "identity_inputs": identity, "learner_git_revision": revision,
        "dirty_state": dirty, "gpu": gpu, "image_id": image_id,
        "disk_free_bytes": int(usage.free), "retained_neural_bytes_before": _retained_bytes(root),
        "output_admission_reservation_bytes": OUTPUT_ADMISSION_RESERVATION_BYTES,
    }


def _exact_name_absent(
    deps: HostDependencies, docker: str, client_config: Path, name: str,
    *, timeout: float = 10.0,
) -> None:
    output = _require_command(deps, _docker_command(
        docker, client_config, "container", "ls", "--all", "--filter",
        f"name=^{name}$", "--format", "{{.ID}}",
    ), timeout=timeout, env=_docker_environment(client_config))
    if output:
        raise RuntimeError("exact proof container name already exists; refusing reuse")


def _inspect(
    deps: HostDependencies, docker: str, client_config: Path, container_id: str,
    *, timeout: float,
) -> dict[str, object]:
    raw = _require_command(
        deps, _docker_command(docker, client_config, "container", "inspect", container_id),
        timeout=timeout, env=_docker_environment(client_config),
    )
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("container inspect did not return one public object")
    return value[0]


def _owner_only(inspected: Mapping[str, object], container_id: str, name: str) -> bool:
    return (
        isinstance(container_id, str) and re.fullmatch(r"[0-9a-f]{64}", container_id) is not None
        and inspected.get("Id") == container_id and inspected.get("Name") == "/" + name
    )


def _is_process_control(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, SystemExit, _WatchdogTermination))


def _remaining_timeout(deps: HostDependencies, deadline: float, maximum: float, phase: str) -> float:
    remaining = deadline - deps.clock()
    if remaining <= 0:
        raise ProofDeadlineExceeded(f"deadline expired {phase}")
    return min(maximum, remaining)


def _expected_mounts(root: Path, source_tree: Path, output_child: Path) -> dict[str, tuple[str, bool]]:
    root = Path(root).resolve(strict=True)
    source_tree = Path(source_tree).resolve(strict=True)
    output_child = Path(output_child).resolve(strict=True)
    return {
        "/work/src": (str(source_tree / "src"), False),
        "/work/tests": (str(source_tree / "tests"), False),
        "/work/configs": (str(source_tree / "configs"), False),
        "/work/gpu_runtime_proof_helpers.py": (
            str(source_tree / "scripts/neural/gpu_runtime_proof_helpers.py"), False,
        ),
        "/runtime/requirements-cuda.lock": (str(root / CUDA_LOCK), False),
        "/runtime/venv-cuda": (str(root / CUDA_VENV), False),
        "/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so": (
            str(root / CUDA_NATIVE), False,
        ),
        PROOF_ROOT_GUEST: (str(output_child), True),
    }


def _write_output_log(path: Path, content: str) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing linked host output log: {path.name}")
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > HOST_COMMAND_OUTPUT_LIMIT_BYTES:
        raise ProofCommandOutputLimit(["container-log", path.name], HOST_COMMAND_OUTPUT_LIMIT_BYTES)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def run_protected(
    *, root: Path, snapshot_path: Path, release_path: Path, claim_path: Path,
    run_id: str, output_child: Path, docker_binary: str = "docker",
    dependencies: HostDependencies | None = None, staging_parent: Path | None = None,
) -> dict[str, object]:
    """Run one explicitly released container; no retry or name-based cleanup."""
    deps = dependencies or HostDependencies()
    started = deps.clock()
    whole_deadline = started + float(LIMITS["whole_run_seconds"])
    soft_deadline = started + float(LIMITS["soft_deadline_seconds"])
    work_deadline = min(soft_deadline, whole_deadline - float(LIMITS["cleanup_reserve_seconds"]))
    root = Path(root).resolve(strict=True)
    output_child = Path(output_child)
    proof_root = (root / "artifacts/development/neural-combat/task-9/gpu-proof").resolve(strict=True)
    name = f"spirelearn-task9-gpu-proof-{run_id}"
    report_path = proof_root / f"host-run-{run_id}.json"
    control_paths = tuple(Path(item) for item in (snapshot_path, release_path, claim_path))
    report: dict[str, object] = {
        "schema_id": "spire-learn-neural-gpu-proof-host-run-v1",
        "status": "preflight", "run_id": run_id, "started_monotonic": started,
        "whole_deadline_monotonic": whole_deadline, "soft_deadline_monotonic": soft_deadline,
        "container_name": name, "container_id": None, "container_removed": False,
        "exact_name_absent_before_create": True, "exact_name_absent_after_cleanup": False,
        "primary_error": None, "cleanup_errors": [],
    }
    primary: BaseException | None = None
    first_control: BaseException | None = None
    container_id: str | None = None
    client_config_temp: tempfile.TemporaryDirectory | None = None
    client_config: Path | None = None
    source_tree: Path | None = None
    source_files: Mapping[str, object] | None = None
    snapshot_binding: Mapping[str, object] | None = None
    release_binding: Mapping[str, object] | None = None
    expected_mounts: Mapping[str, tuple[str, bool]] | None = None
    claim_created = False
    name_absent_before_create = False
    body_completed = False
    transcript = ""

    def remember(error: BaseException, phase: str) -> None:
        nonlocal primary, first_control
        detail = f"{phase}: {type(error).__name__}: {str(error)[:384]}"
        if isinstance(error, (ProofCommandTimeout, ProofCommandOutputLimit)):
            report["host_command_failure"] = {
                "argv": list(error.argv), "timeout_seconds": getattr(error, "timeout", None),
                "output_limit_bytes": getattr(error, "limit", None),
                "stdout_prefix": error.stdout_prefix, "stderr_prefix": error.stderr_prefix,
                "stdout_truncated": getattr(error, "stdout_truncated", False),
                "stderr_truncated": getattr(error, "stderr_truncated", False),
            }
        if _is_process_control(error):
            if first_control is None:
                first_control = error
                report["first_process_control"] = {
                    "type": type(error).__name__, "message": str(error)[:384], "phase": phase,
                }
            elif first_control is not error:
                first_control.add_note(f"later process-control exception: {detail}")
        elif primary is None:
            primary = error
            report["primary_error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        else:
            try:
                primary.add_note(f"later {detail}")
            except BaseException:
                pass
        if phase.startswith("cleanup") or phase in {"source-postcheck", "output-postcheck", "finalization"}:
            report["cleanup_errors"].append(detail)
        report["status"] = "failed_or_incomplete"

    def check_before(deadline: float, phase: str) -> None:
        if deps.clock() >= deadline:
            raise ProofDeadlineExceeded(f"deadline expired {phase}")

    def check_output_progress() -> None:
        check_before(work_deadline, "while the protected command was running")
        actual = output_tree_bytes(output_child)
        report["output_bytes_last_polled"] = actual
        if actual > LIMITS["output_bytes"]:
            raise ProofCommandOutputLimit(
                ["protected-container-output"], LIMITS["output_bytes"],
            )

    try:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run ID is not a safe path/container-name component")
        for path in control_paths:
            if path.is_symlink():
                raise ValueError("snapshot, release, and claim controls cannot be symlinks")
            resolved = path.resolve(strict=path != Path(claim_path))
            if not resolved.is_relative_to(proof_root):
                raise ValueError("snapshot, release, and claim must stay under the Task 9 proof root")
        if Path(claim_path).exists():
            raise FileExistsError("exclusive invocation claim already exists; it cannot be reused")
        check_before(work_deadline, "before source preflight")
        snapshot_binding = verify_source_snapshot(root, snapshot_path)
        if snapshot_binding["learner_git_revision"] != ACCEPTED_LEARNER_HEAD:
            raise ValueError("source snapshot learner revision is not the accepted Task 1 core commit")
        if snapshot_binding["identity_inputs"] != task8_identity_inputs(root):
            raise ValueError("Task 8 runtime identity inputs changed after source snapshot capture")
        source_files = load_json(snapshot_path).get("source_files")
        if not isinstance(source_files, Mapping):
            raise ValueError("released snapshot has no exact source inventory")
        release_binding = verify_release(release_path, snapshot_path=snapshot_path, run_id=run_id)
        output_parent = output_child.parent.resolve(strict=True)
        if not output_parent.is_relative_to(proof_root) or output_child.exists() or output_child.is_symlink():
            raise ValueError("output child must be a new non-symlink directory below the proof root")
        client_config_temp = tempfile.TemporaryDirectory(
            prefix=".task9-docker-client-", dir=output_parent,
        )
        client_config = Path(client_config_temp.name)
        preflight = _identity_and_preflight(
            root, output_child, deps, whole_deadline=work_deadline,
            docker_binary=docker_binary, docker_client_config=client_config,
        )
        if snapshot_binding["learner_git_revision"] != preflight["learner_git_revision"]:
            raise ValueError("learner Git revision changed after snapshot capture")
        if snapshot_binding["dirty_state"] != preflight["dirty_state"]:
            raise ValueError("learner dirty state changed after snapshot capture")
        if deps.uid() == 0:
            raise RuntimeError("protected proof runner must be invoked by a non-root host user")
        user = f"{deps.uid()}:{deps.gid()}"
        stage_parent = Path(staging_parent) if staging_parent is not None else proof_root / "inputs"
        stage_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if stage_parent.is_symlink() or not stage_parent.resolve(strict=True).is_relative_to(proof_root):
            raise ValueError("released source staging must remain under ignored Task 9 proof inputs")
        source_tree = stage_parent / f"source-{run_id}"
        retained_cap = int(LIMITS["retained_neural_bytes"])
        future_output_reservation = OUTPUT_ADMISSION_RESERVATION_BYTES
        staged_source_reservation = _released_guest_source_bytes(root, source_files)
        retained_before_stage = _retained_bytes(root)
        projected_before_stage = (
            retained_before_stage + staged_source_reservation + future_output_reservation
        )
        report["retained_admission"] = {
            "retained_bytes_before_stage": retained_before_stage,
            "staged_source_bytes_reserved": staged_source_reservation,
            "future_output_reservation_bytes": future_output_reservation,
            "projected_retained_bytes_before_stage": projected_before_stage,
        }
        if projected_before_stage > retained_cap:
            raise RuntimeError(
                "retained storage admission before staging exceeds 40 GiB "
                f"({projected_before_stage} > {retained_cap} bytes)"
            )
        staged = stage_released_source_tree(root, source_files, source_tree)
        report.update({"snapshot": snapshot_binding, "release": release_binding,
                       "preflight": preflight, "staged_source": staged})
        retained_after_stage = _retained_bytes(root)
        projected_after_stage = retained_after_stage + future_output_reservation
        report["retained_admission"].update({
            "retained_bytes_after_stage": retained_after_stage,
            "remaining_future_output_reservation_bytes": future_output_reservation,
            "projected_retained_bytes_after_stage": projected_after_stage,
        })
        if projected_after_stage > retained_cap:
            raise RuntimeError(
                "retained storage admission after staging exceeds 40 GiB "
                f"({projected_after_stage} > {retained_cap} bytes)"
            )

        output_child.mkdir(mode=0o700)
        guest_scratch = prepare_guest_scratch(output_child)
        report["guest_runtime_scratch"] = guest_scratch
        projected_output = output_tree_bytes(output_child) + OUTPUT_ADMISSION_RESERVATION_BYTES
        if projected_output > LIMITS["output_bytes"]:
            raise RuntimeError("known outputs and guest scratch exceed the 1 GiB output admission cap")
        _exact_name_absent(
            deps, docker_binary, client_config, name,
            timeout=_remaining_timeout(deps, work_deadline, 10.0, "before exact-name admission"),
        )
        name_absent_before_create = True
        claim = create_release_claim(claim_path, release_binding)
        claim_created = True
        verify_release_claim(claim_path, release_binding)
        report["claim_raw_sha256"] = sha256_file(claim_path)
        report["status"] = "running"

        latest_source = verify_source_snapshot(root, snapshot_path)
        staged_now = verify_staged_source_tree(source_tree, source_files)
        if (
            latest_source["snapshot_raw_sha256"] != snapshot_binding["snapshot_raw_sha256"]
            or task8_identity_inputs(root) != snapshot_binding["identity_inputs"]
            or staged_now != staged
            or sha256_file(release_path) != release_binding["release_raw_sha256"]
        ):
            raise RuntimeError("released source, staged source, release, or Task 8 input changed before create")
        expected_mounts = _expected_mounts(root, source_tree, output_child)
        create_argv = _docker_create_argv(
            docker=docker_binary, name=name, root=root, source_tree=source_tree,
            output_child=output_child, user=user, deadline=soft_deadline,
            docker_client_config=client_config,
        )
        check_before(work_deadline, "before container create")
        created = _run(
            deps, create_argv,
            timeout=_remaining_timeout(deps, work_deadline, 20.0, "before container create"),
            env=_docker_environment(client_config),
        )
        if created.returncode != 0:
            raise RuntimeError(f"docker create failed ({created.returncode}); ownership was not established")
        candidate = created.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", candidate):
            raise RuntimeError("docker create did not return exactly one full container ID")
        container_id = candidate
        report["container_id"] = container_id
        inspected = _inspect(
            deps, docker_binary, client_config, container_id,
            timeout=_remaining_timeout(deps, work_deadline, 10.0, "before prestart inspection"),
        )
        assert_owned_container(
            inspected, container_id=container_id, container_name=name,
            image_id=IMAGE_ID, image_reference=IMAGE_REFERENCE, user=user,
            expected_mounts=expected_mounts,
        )
        if not all((output_child / guest.removeprefix(PROOF_ROOT_GUEST + "/")).is_dir()
                   for guest in GUEST_SCRATCH_ENV.values()):
            raise RuntimeError("guest TMPDIR/Torch/CUDA cache paths were not prepared before imports")
        started_container = _run(
            deps, _docker_command(docker_binary, client_config, "start", "--attach", container_id),
            timeout=_remaining_timeout(deps, work_deadline, work_deadline - deps.clock(), "before container start"),
            env=_docker_environment(client_config), progress_check=check_output_progress,
        )
        transcript = (started_container.stdout or "") + (started_container.stderr or "")
        _write_output_log(output_child / "container-stdout.log", started_container.stdout or "")
        _write_output_log(output_child / "container-stderr.log", started_container.stderr or "")
        check_output_progress()
        report["container_exit_code"] = started_container.returncode
        report["test_transcript_sha256"] = hashlib.sha256(transcript.encode()).hexdigest()
        report["test_transcript"] = verify_test_transcript(transcript)
        if started_container.returncode != 0:
            raise RuntimeError(f"protected unittest command exited {started_container.returncode}")
        report["mode_evidence"] = verify_mode_evidence(output_child)
        report["source_postcheck"] = verify_source_snapshot(root, snapshot_path)
        if report["source_postcheck"]["snapshot_raw_sha256"] != snapshot_binding["snapshot_raw_sha256"]:
            raise RuntimeError("source snapshot raw hash changed during the protected run")
        report["staged_source_postcheck"] = verify_staged_source_tree(source_tree, source_files)
        if task8_identity_inputs(root) != snapshot_binding["identity_inputs"]:
            raise RuntimeError("Task 8 runtime identity inputs changed during the protected run")
        check_before(work_deadline, "after protected source postcheck")
        body_completed = True
    except BaseException as error:
        remember(error, "protected-run")
    finally:
        if container_id is not None and client_config is not None and expected_mounts is not None:
            current = None
            try:
                current = _inspect(
                    deps, docker_binary, client_config, container_id,
                    timeout=_remaining_timeout(deps, whole_deadline, 10.0, "during owned-container cleanup"),
                )
                assert_owned_container(
                    current, container_id=container_id, container_name=name,
                    image_id=IMAGE_ID, image_reference=IMAGE_REFERENCE,
                    user=f"{deps.uid()}:{deps.gid()}", expected_mounts=expected_mounts,
                    lifecycle="cleanup",
                )
            except BaseException as cleanup_error:
                remember(cleanup_error, "cleanup-inspect")
                current = None
            if current is not None:
                state = current.get("State")
                running = isinstance(state, Mapping) and state.get("Running") is True
                if running:
                    try:
                        stop_timeout = _remaining_timeout(deps, whole_deadline, 10.0, "before stopping owned container")
                        stop_seconds = max(1, int(min(10, stop_timeout - 1)))
                        _require_command(
                            deps, _docker_command(
                                docker_binary, client_config, "stop", "--time", str(stop_seconds), container_id,
                            ), timeout=stop_timeout, env=_docker_environment(client_config),
                        )
                        current = _inspect(
                            deps, docker_binary, client_config, container_id,
                            timeout=_remaining_timeout(deps, whole_deadline, 10.0, "after stopping owned container"),
                        )
                        assert_owned_container(
                            current, container_id=container_id, container_name=name,
                            image_id=IMAGE_ID, image_reference=IMAGE_REFERENCE,
                            user=f"{deps.uid()}:{deps.gid()}", expected_mounts=expected_mounts,
                            lifecycle="cleanup",
                        )
                        running = current["State"]["Running"] is True
                        if running:
                            raise RuntimeError("owned container remained running after bounded stop")
                    except BaseException as cleanup_error:
                        remember(cleanup_error, "cleanup-stop")
                        current = None
                if current is not None and not running:
                    try:
                        _require_command(
                            deps, _docker_command(docker_binary, client_config, "container", "rm", container_id),
                            timeout=_remaining_timeout(deps, whole_deadline, 10.0, "before removing owned container"),
                            env=_docker_environment(client_config),
                        )
                        report["container_removed"] = True
                    except BaseException as cleanup_error:
                        remember(cleanup_error, "cleanup-remove")
            try:
                _exact_name_absent(
                    deps, docker_binary, client_config, name,
                    timeout=_remaining_timeout(deps, whole_deadline, 10.0, "after owned-container cleanup"),
                )
                report["exact_name_absent_after_cleanup"] = True
            except BaseException as cleanup_error:
                remember(cleanup_error, "cleanup-name-absence")
        elif client_config is not None and name_absent_before_create:
            try:
                _exact_name_absent(
                    deps, docker_binary, client_config, name,
                    timeout=_remaining_timeout(deps, whole_deadline, 10.0, "confirming post-create name absence"),
                )
                report["exact_name_absent_after_cleanup"] = True
            except BaseException as absence_error:
                remember(absence_error, "cleanup-name-absence")

        if snapshot_binding is not None:
            try:
                check_before(whole_deadline, "before canonical and staged source postchecks")
                report["source_postcheck_after_cleanup"] = verify_source_snapshot(root, snapshot_path)
                if report["source_postcheck_after_cleanup"]["snapshot_raw_sha256"] != snapshot_binding["snapshot_raw_sha256"]:
                    raise RuntimeError("source snapshot raw hash changed during cleanup")
                if task8_identity_inputs(root) != snapshot_binding["identity_inputs"]:
                    raise RuntimeError("Task 8 runtime identity inputs changed during cleanup")
                if source_tree is not None and source_files is not None:
                    report["staged_source_postcheck_after_cleanup"] = verify_staged_source_tree(
                        source_tree, source_files,
                    )
            except BaseException as source_error:
                remember(source_error, "source-postcheck")

        if output_child.exists() and claim_created:
            try:
                check_before(whole_deadline, "before output and retained-storage postchecks")
                output_bytes = output_tree_bytes(output_child)
                if output_bytes > LIMITS["output_bytes"]:
                    raise RuntimeError("proof output exceeds 1 GiB")
                report["output_bytes_before_inventory"] = output_bytes
                report["retained_neural_bytes_after"] = _retained_bytes(root)
                if report["retained_neural_bytes_after"] > LIMITS["retained_neural_bytes"]:
                    raise RuntimeError("retained neural-combat storage exceeds 40 GiB after run")
                inventory_path = output_child / "evidence-inventory.json"
                create_evidence_inventory(output_child, inventory_path)
                verify_evidence_inventory(output_child, inventory_path)
                report["evidence_inventory"] = {
                    "relative_path": "evidence-inventory.json",
                    "raw_sha256": sha256_file(inventory_path),
                    "canonical_sha256": canonical_sha256(load_json(inventory_path)),
                }
                report["output_bytes"] = output_tree_bytes(output_child)
                if report["output_bytes"] > LIMITS["output_bytes"]:
                    raise RuntimeError("proof output including inventory exceeds 1 GiB")
            except BaseException as evidence_error:
                remember(evidence_error, "output-postcheck")

        if client_config_temp is not None:
            try:
                client_config_temp.cleanup()
            except BaseException as client_cleanup_error:
                remember(client_cleanup_error, "cleanup-client-config")

        report["name_absent_before_create"] = name_absent_before_create
        report["cleanup_complete"] = bool(report["container_removed"] and report["exact_name_absent_after_cleanup"])
        if not report["exact_name_absent_after_cleanup"]:
            remember(RuntimeError("exact proof container name absence was not confirmed after cleanup"), "cleanup-final")
        if container_id is not None and not report["container_removed"]:
            remember(RuntimeError("protected container ownership was not removed by full ID"), "cleanup-final")
        if not body_completed:
            report["status"] = "failed_or_incomplete"
        elif primary is None and first_control is None and report["cleanup_complete"]:
            report["status"] = "finalizing"
        else:
            report["status"] = "failed_or_incomplete"

        if claim_created:
            report_written = False
            try:
                report["cleanup_errors"] = list(report["cleanup_errors"])
                check_before(whole_deadline, "before host report write")
                if report_path.exists():
                    raise FileExistsError(f"host report already exists: {report_path.name}")
                write_json_exclusive(report_path, report)
                report_written = True
                report["retained_neural_bytes_after_host_report"] = _retained_bytes(root)
                if report["retained_neural_bytes_after_host_report"] > LIMITS["retained_neural_bytes"]:
                    remember(RuntimeError("retained neural-combat storage exceeds 40 GiB after host report"), "finalization")
                check_before(whole_deadline, "after host report write")
                report["finished_monotonic"] = deps.clock()
                if report["finished_monotonic"] >= whole_deadline:
                    raise ProofDeadlineExceeded("300-second whole-run cap including cleanup expired")
                if report["status"] == "finalizing" and primary is None and first_control is None:
                    report["status"] = "passed"
                else:
                    report["status"] = "failed_or_incomplete"
                check_before(whole_deadline, "before final host status write")
                _write_json_replace(report_path, report)
                check_before(whole_deadline, "after final host status write")
            except BaseException as report_error:
                remember(report_error, "finalization")
                if report_written:
                    report["status"] = "failed_or_incomplete"
                    try:
                        _write_json_replace(report_path, report)
                    except BaseException as rewrite_error:
                        remember(rewrite_error, "finalization-status-rewrite")

    if first_control is not None:
        prefix = {
            "release_claim_created": claim_created,
            "container_id": container_id,
            "container_removed": report["container_removed"],
            "exact_name_absent_after_cleanup": report["exact_name_absent_after_cleanup"],
        }
        first_control.add_note(f"accepted proof prefix and cleanup result: {prefix}")
        if primary is not None and primary is not first_control:
            first_control.add_note(f"earlier ordinary failure: {type(primary).__name__}: {str(primary)[:384]}")
        for detail in report["cleanup_errors"]:
            first_control.add_note(f"cleanup/finalization evidence: {detail}")
        raise first_control
    if primary is not None:
        raise primary
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 9 bounded GPU proof controls")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--release", type=Path, required=True)
    run.add_argument("--claim", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-child", type=Path, required=True)
    run.add_argument("--docker-binary", default="docker")
    return parser


def main(argv: Sequence[str] | None = None, *, dependencies: HostDependencies | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        root = args.root.resolve(strict=True)
        validate_config_pair(root)
        revision, dirty = _git_state(root)
        if revision != ACCEPTED_LEARNER_HEAD:
            raise RuntimeError("source snapshot must be created from the accepted Task 1 core commit")
        snapshot = create_source_snapshot(
            root, args.output, learner_git_revision=revision, dirty_state=dirty,
            identity_inputs=task8_identity_inputs(root),
        )
        binding = verify_source_snapshot(root, args.output)
        print(json.dumps({
            "status": "source_only_snapshot_created", "path_count": snapshot["path_count"],
            "raw_sha256": binding["snapshot_raw_sha256"],
            "canonical_sha256": binding["snapshot_canonical_sha256"],
        }, sort_keys=True))
        return 0
    if args.command == "run":
        if (
            os.environ.get("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME") != "1"
            or os.environ.get("SPIRE_LEARN_NEURAL_GPU_UUID") != GPU_UUID
        ):
            raise RuntimeError("protected runner requires explicit opt-in and exact GPU UUID")
        previous_term = signal.getsignal(signal.SIGTERM)

        def terminate_for_cleanup(_signum, _frame):
            raise _WatchdogTermination("outer 300-second proof watchdog requested bounded cleanup")

        signal.signal(signal.SIGTERM, terminate_for_cleanup)
        try:
            result = run_protected(
                root=args.root, snapshot_path=args.snapshot, release_path=args.release,
                claim_path=args.claim, run_id=args.run_id, output_child=args.output_child,
                docker_binary=args.docker_binary, dependencies=dependencies,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result.get("status") == "passed" else 1
    raise AssertionError("unreachable command")


__all__ = [
    "ACCEPTED_LEARNER_HEAD", "CUDA_LOCK_SHA256", "EffectLedger", "EffectLimitError", "GPU_UUID", "GuardedDriver",
    "GuardedOptimizer", "GuardedPolicy", "HostDependencies", "IMAGE_ID", "IMAGE_REFERENCE",
    "INVENTORY_SCHEMA", "LIMITS", "MODE_LIMITS", "MODE_RESULT_SCHEMA", "ProofDeadlineExceeded",
    "PROOF_ROOT_GUEST", "RELEASE_SCHEMA", "SNAPSHOT_PATHS", "SNAPSHOT_SCHEMA",
    "assert_owned_container", "build_parser", "canonical_sha256", "create_evidence_inventory",
    "create_release_claim", "create_source_snapshot", "main", "run_protected", "sha256_file",
    "load_json", "project_mode_counters", "task8_identity_inputs", "validate_config_pair", "validate_mode_result",
    "verify_evidence_inventory", "verify_mode_evidence", "verify_release", "verify_release_claim",
    "verify_source_snapshot", "verify_test_exports", "verify_test_transcript", "write_json_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
