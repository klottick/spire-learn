"""Stdlib-only host controls for the gated Task 8 CUDA setup phases.

This module deliberately does not import Torch, NumPy, Gymnasium, Docker SDKs,
or the Spire Gym package.  Its commands validate evidence and build literal
argument vectors; the phase runner owns Docker/network/device lifecycle.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
from itertools import product
import json
import math
import os
import platform
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit


TASK_ROOT = Path("artifacts/development/neural-combat/task-8")
CUDA_TORCH_URL = (
    "https://download-r2.pytorch.org/whl/cu130/"
    "torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"
)
CUDA_TORCH_SHA256 = "5359b255ac68c0cabb8293952b38e00f0d1a570815cd87f19b61af702c071410"
CUDA_TORCH_FILENAME = "torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
CUDA_TORCH_METADATA_SHA256 = "f50fae9c2fd1e1d1e22f52426296a598741c51378935d0905c2c60288671729a"
CUDA_TORCH_METADATA_SIZE = 37891
CUDA_TORCH_SIZE = 554621169
PYTHON_INDEX_URL = "https://pypi.org/simple"
PYTHON_IMAGE_DIGEST = "57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
PYTHON_IMAGE_REFERENCE = f"docker.io/library/python@sha256:{PYTHON_IMAGE_DIGEST}"
PYTHON_CANONICAL_REPO_DIGEST = f"python@sha256:{PYTHON_IMAGE_DIGEST}"
PYTHON_QUALIFIED_REPO_DIGEST = f"docker.io/library/python@sha256:{PYTHON_IMAGE_DIGEST}"
GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
GPU_NAME = "NVIDIA GeForce RTX 5090"
GPU_DRIVER = "595.91.07"
GPU_MEMORY_MIB = 32607
CORRECTED_RNG_RUN_ID = "cuda-setup-20261028-a"
CORRECTED_RNG_ATTEMPT_ID = "corrected-1"
CORRECTED_RNG_PHASE_KEY = "rng-red-corrected-1"
CORRECTED_RNG_SOURCE_SNAPSHOT_NAME = "task-8-corrected-rng-reviewed-source-snapshot.json"
GPU_MIN_FREE_MIB = 12 * 1024
CPU_WHEEL_FILENAME = "spire_lab-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
_PROTECTED_CPU_VENV_LINKS = {
    "bin/python": "/usr/local/bin/python",
    "bin/python3": "python",
    "bin/python3.12": "python",
}
CPU_WHEEL_SHA256 = "088296349eff7253530db2a85ebfb3a4f88aff658fc9cf7e8ed41cd2acc25777"
CPU_NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
CUDA_DIRECT_PINS = {
    "torch": "2.14.1+cu130",
    "numpy": "2.5.4",
    "gymnasium": "1.4.0",
    "spire_lab": "0.1.0",
}
TORCH_REQUIRED_CUDA_DISTRIBUTIONS = frozenset({
    "cuda-toolkit",
    "cuda-bindings",
    "nvidia-cudnn-cu13",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvshmem-cu13",
    "triton",
})
TORCH_REQUIRED_ORDINARY_DISTRIBUTIONS = frozenset({
    "filelock", "typing-extensions", "setuptools", "sympy", "networkx", "jinja2", "fsspec",
})
CUDA_RESOLVER_ARGV = (
    "python3.12", "-B", "-m", "pip", "install", "--dry-run", "--ignore-installed",
    "--report", "/work/pip-resolution-report.json", "--only-binary=:all:",
    "--index-url", PYTHON_INDEX_URL, "-r", "/work/requirements-cuda.in",
)
EXPECTED_RNG_TESTS = (
    "test_cpu_and_initialized_cuda_rng_states_unchanged_by_build",
    "test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda",
    "test_same_seed_builds_identical_initial_parameter_bytes",
)
RNG_TEST_CLASS = "tests.neural.test_gpu_qualification.GpuRngRegressionTests"
EXPECTED_RNG_LABELS = tuple(
    f"{name} ({RNG_TEST_CLASS}.{name})" for name in EXPECTED_RNG_TESTS
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}\Z")
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TEST_LINE = re.compile(r"^(test_[A-Za-z0-9_]+ \([^\n]+\)) \.\.\. (ok|FAIL|ERROR|skipped .*)$")
_RAN_LINE = re.compile(r"^Ran ([0-9]+) tests? in [0-9.]+s$")


class Task8ControlError(ValueError):
    """A Task 8 setup invariant or retained evidence check failed."""


def canonical_distribution_name(value: str) -> str:
    if not isinstance(value, str) or not value or _PACKAGE_NAME.fullmatch(value) is None:
        raise Task8ControlError("distribution name is malformed")
    return re.sub(r"[-_.]+", "-", value).lower()


def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Task8ControlError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_safe_task8_inputs(task_root: Path, input_paths: Sequence[Path]) -> tuple[Path, ...]:
    """Require canonical absolute parent inputs contained under the Task 8 root."""
    task_root = Path(task_root)
    if not task_root.is_absolute() or task_root.is_symlink() or not task_root.is_dir():
        raise Task8ControlError("Task 8 evidence root must be an absolute real directory")
    root_real = task_root.resolve(strict=True)
    if task_root != root_real:
        raise Task8ControlError("Task 8 evidence root must use its canonical absolute path")
    checked: list[Path] = []
    for raw in input_paths:
        path = Path(raw)
        if not path.is_absolute():
            raise Task8ControlError("parent release and source snapshot paths must be absolute")
        if path.is_symlink():
            raise Task8ControlError(f"parent release/source input may not be a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise Task8ControlError(f"parent release/source input is missing: {path}") from error
        if path != resolved:
            raise Task8ControlError(f"parent release/source input must be canonical: {path}")
        if root_real not in resolved.parents or not resolved.is_file():
            raise Task8ControlError(f"parent release/source input must be a regular file inside Task 8: {path}")
        if resolved.is_symlink():
            raise Task8ControlError(f"parent release/source input may not be a symlink: {path}")
        checked.append(resolved)
    if len(set(checked)) != len(checked):
        raise Task8ControlError("parent release and source snapshot inputs must be distinct files")
    return tuple(checked)


def _source_snapshot_entries(repo_root: Path, scope_roots: Sequence[str]) -> list[dict[str, object]]:
    repo_root = Path(repo_root).resolve(strict=True)
    scopes: list[Path] = []
    for raw in scope_roots:
        rel = Path(raw)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise Task8ControlError("source snapshot scopes must be safe repository-relative paths")
        scope = repo_root / rel
        if scope.is_symlink() or not scope.exists():
            raise Task8ControlError(f"source snapshot input is missing or is a symlink: {scope}")
        if scope.resolve(strict=True) != scope:
            raise Task8ControlError(f"source snapshot input is not canonical: {scope}")
        scopes.append(scope)
    if len(set(scopes)) != len(scopes):
        raise Task8ControlError("source snapshot scopes must be unique")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_directory(path: Path) -> None:
        rel = path.relative_to(repo_root).as_posix()
        if rel in seen:
            return
        seen.add(rel)
        entries.append({"path": rel, "kind": "directory"})

    def add_file(path: Path) -> None:
        rel = path.relative_to(repo_root).as_posix()
        if rel in seen:
            raise Task8ControlError(f"source snapshot scopes overlap at {rel}")
        if path.is_symlink() or not path.is_file():
            raise Task8ControlError(f"source snapshot file is not a regular non-symlink file: {path}")
        info = path.stat(follow_symlinks=False)
        seen.add(rel)
        entries.append({
            "path": rel,
            "kind": "file",
            "size_bytes": info.st_size,
            "executable": bool(info.st_mode & 0o111),
            "sha256": sha256_file(path),
        })

    for scope in scopes:
        if scope.is_file():
            add_file(scope)
            continue
        if not scope.is_dir():
            raise Task8ControlError(f"source snapshot scope is not a file or directory: {scope}")
        add_directory(scope)
        for directory, dirnames, filenames in os.walk(scope, followlinks=False):
            directory_path = Path(directory)
            for name in sorted(tuple(dirnames)):
                child = directory_path / name
                if child.is_symlink():
                    raise Task8ControlError(f"source tree contains a symlink directory: {child}")
                if not child.is_dir():
                    raise Task8ControlError(f"source tree contains a non-directory entry: {child}")
                add_directory(child)
            for name in sorted(filenames):
                add_file(directory_path / name)
    entries.sort(key=lambda item: (str(item["path"]), str(item["kind"])))
    return entries


def capture_source_snapshot(
    repo_root: Path,
    output_path: Path,
    *,
    scope_roots: Sequence[str] = (
        "src", "tests", "scripts/neural/run-task8-gpu-setup.sh",
        "scripts/neural/task8-gpu-setup-runbook.md", "scripts/neural/task8_device_check.py",
        "scripts/neural/task8_gpu_setup_helpers.py",
    ),
) -> dict[str, object]:
    """Write a full byte/path/executable-mode snapshot of projected sources."""
    repo_root = Path(repo_root).resolve(strict=True)
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Task8ControlError("source snapshot output already exists")
    entries = _source_snapshot_entries(repo_root, scope_roots)
    value = {
        "format": "spire-learn-task8-source-snapshot-v1",
        "repository_root": str(repo_root),
        "scope_roots": list(scope_roots),
        "entries": entries,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(output_path), "sha256": hashlib.sha256(encoded).hexdigest(),
            "file_count": sum(item["kind"] == "file" for item in entries),
            "entry_count": len(entries)}


def verify_source_snapshot(snapshot_path: Path, repo_root: Path) -> dict[str, object]:
    snapshot = _json_object(snapshot_path)
    repo_root = Path(repo_root).resolve(strict=True)
    if snapshot.get("format") != "spire-learn-task8-source-snapshot-v1":
        raise Task8ControlError("source snapshot format is unsupported")
    if snapshot.get("repository_root") != str(repo_root):
        raise Task8ControlError("source snapshot repository identity differs")
    scopes = snapshot.get("scope_roots")
    entries = snapshot.get("entries")
    if not isinstance(scopes, list) or any(not isinstance(item, str) for item in scopes):
        raise Task8ControlError("source snapshot scope list is malformed")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise Task8ControlError("source snapshot entries are malformed")
    observed = _source_snapshot_entries(repo_root, scopes)
    if observed != entries:
        raise Task8ControlError("source snapshot mismatch: mounted source/test bytes, modes, or path set changed")
    return {"unchanged": True, "file_count": sum(item["kind"] == "file" for item in observed),
            "entry_count": len(observed), "snapshot_sha256": sha256_file(snapshot_path)}


def _json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Task8ControlError(f"expected a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Task8ControlError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Task8ControlError(f"JSON evidence must be an object: {path}")
    return value


def verify_candidate_metadata(record_path: Path, metadata_path: Path) -> dict[str, object]:
    """Reconcile the parent's read-only Torch index/HEAD/METADATA evidence."""
    record = _json_object(record_path)
    metadata_path = Path(metadata_path)
    metadata_bytes = metadata_path.read_bytes()
    observed_metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
    if record.get("wheel_url") != CUDA_TORCH_URL:
        raise Task8ControlError("CUDA Torch URL differs from the approved candidate")
    if record.get("expected_wheel_sha256") != CUDA_TORCH_SHA256:
        raise Task8ControlError("CUDA Torch advertised SHA256 differs from the approved candidate")
    if record.get("metadata_sha256") != CUDA_TORCH_METADATA_SHA256:
        raise Task8ControlError("CUDA Torch metadata digest differs from retained preflight")
    if observed_metadata_hash != CUDA_TORCH_METADATA_SHA256:
        raise Task8ControlError("retained CUDA Torch METADATA bytes fail SHA256")
    head = record.get("payload_head")
    metadata_response = record.get("metadata_response")
    if not isinstance(head, dict) or head.get("status") != 200:
        raise Task8ControlError("Torch candidate payload HEAD was not HTTP 200")
    if int(_header(head.get("headers", []), "content-length") or -1) != CUDA_TORCH_SIZE:
        raise Task8ControlError("Torch candidate payload size differs from retained HEAD")
    if not isinstance(metadata_response, dict) or metadata_response.get("status") != 200:
        raise Task8ControlError("Torch candidate metadata response was not HTTP 200")
    if int(_header(metadata_response.get("headers", []), "content-length") or -1) != CUDA_TORCH_METADATA_SIZE:
        raise Task8ControlError("Torch candidate metadata size differs from retained response")
    parsed = Parser().parsestr(metadata_bytes.decode("utf-8"))
    if parsed.get("Name", "").lower() != "torch" or parsed.get("Version") != CUDA_DIRECT_PINS["torch"]:
        raise Task8ControlError("Torch candidate METADATA name/version mismatch")
    requires = set(parsed.get_all("Requires-Dist", []))
    required = {
        "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"",
        "cuda-bindings<14,>=13.0.3; platform_system == \"Linux\" and python_version < \"3.15\"",
        "nvidia-cudnn-cu13==9.24.0.43; platform_system == \"Linux\"",
        "nvidia-cusparselt-cu13==0.8.1; platform_system == \"Linux\"",
        "nvidia-nccl-cu13==2.30.7; platform_system == \"Linux\"",
        "nvidia-nvshmem-cu13==3.4.5; platform_system == \"Linux\"",
        "triton~=3.8.0; platform_system == \"Linux\" and python_version < \"3.15\"",
    }
    if not required.issubset(requires):
        raise Task8ControlError("Torch CUDA Requires-Dist closure metadata is incomplete")
    link = record.get("link_attributes")
    if not isinstance(link, dict) or not str(link.get("href", "")).endswith(
        f"#sha256={CUDA_TORCH_SHA256}"
    ):
        raise Task8ControlError("PyTorch simple-index link does not advertise the expected hash")
    return {
        "wheel_url": CUDA_TORCH_URL,
        "expected_sha256": CUDA_TORCH_SHA256,
        "payload_head_bytes": CUDA_TORCH_SIZE,
        "metadata_sha256": observed_metadata_hash,
        "metadata_bytes": len(metadata_bytes),
        "requires_dist_count": len(requires),
        "payload_downloaded": False,
    }


def _header(headers: object, key: str) -> str | None:
    if not isinstance(headers, list):
        return None
    for item in headers:
        if (isinstance(item, (list, tuple)) and len(item) == 2
                and str(item[0]).lower() == key.lower()):
            return str(item[1])
    return None


def verify_cpu_wheel_payload(
    location_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    """Verify the independently built accepted Gym wheel as read-only input."""
    location = _json_object(location_path)
    audit = _json_object(audit_path)
    runtime_root = Path(str(location.get("runtime_root", ""))).resolve(strict=True)
    if location.get("state") != "built_installed_offline_identity_verified_conformance_pending":
        raise Task8ControlError("retained CPU runtime is not in the expected accepted state")
    if audit.get("new_native_built_from_accepted_readonly_source") is not True:
        raise Task8ControlError("retained Spire Lab wheel lacks accepted-source build evidence")
    wheel_path = Path(str(audit.get("wheel", "")))
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise Task8ControlError("retained Spire Lab wheel is missing or is a symlink")
    resolved_wheel = wheel_path.resolve(strict=True)
    if runtime_root not in resolved_wheel.parents:
        raise Task8ControlError("retained Spire Lab wheel escapes its runtime root")
    expected = audit.get("wheel_sha256")
    observed = sha256_file(resolved_wheel)
    if expected != observed or expected != CPU_WHEEL_SHA256:
        raise Task8ControlError("retained Spire Lab wheel SHA256 differs from accepted audit")
    if resolved_wheel.name != CPU_WHEEL_FILENAME:
        raise Task8ControlError("retained Spire Lab wheel filename/ABI differs from accepted audit")
    if audit.get("native_sha256") != CPU_NATIVE_SHA256:
        raise Task8ControlError("retained CPU report native identity differs from accepted audit")
    return {
        "wheel_path": str(resolved_wheel),
        "wheel_sha256": observed,
        "native_payload_mounted": False,
        "use": "hash-verified read-only dependency payload only",
    }


def verify_image_inspect(raw: str | bytes) -> dict[str, object]:
    """Require the exact canonical runtime digest and immutable local image ID."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Task8ControlError("docker image inspect output is not JSON") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise Task8ControlError("docker image inspect must describe exactly one image")
    image = values[0]
    image_id = image.get("Id")
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise Task8ControlError("runtime image .Id is missing or malformed")
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise Task8ControlError("runtime image must be linux/amd64")
    digests = image.get("RepoDigests")
    if not isinstance(digests, list) or not (
        PYTHON_CANONICAL_REPO_DIGEST in digests or PYTHON_QUALIFIED_REPO_DIGEST in digests
    ):
        raise Task8ControlError("runtime image lacks the exact canonical Python RepoDigest")
    observed_repo_digest = (
        PYTHON_CANONICAL_REPO_DIGEST
        if PYTHON_CANONICAL_REPO_DIGEST in digests else PYTHON_QUALIFIED_REPO_DIGEST
    )
    return {
        "image_id": image_id,
        "os": "linux",
        "architecture": "amd64",
        "repo_digest": observed_repo_digest,
        "image_reference": PYTHON_IMAGE_REFERENCE,
    }


def verify_local_image_alias(
    raw: str | bytes,
    *,
    alias: str,
    expected_image_id: str,
) -> dict[str, object]:
    """Verify a run-scoped local tag still identifies the pinned Python image."""
    base = verify_image_inspect(raw)
    values = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    image = values[0]
    if image.get("Id") != expected_image_id or base["image_id"] != expected_image_id:
        raise Task8ControlError("local image alias .Id changed from the canonical image")
    tags = image.get("RepoTags")
    if not isinstance(tags, list) or alias not in tags:
        raise Task8ControlError("run-scoped local alias is absent from RepoTags")
    return {**base, "local_alias": alias}


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise Task8ControlError("run ID must be unique and 1..40 safe characters")
    return run_id


def project_corrected_rng_attempt_paths(
    task_root: Path,
    *,
    run_id: str,
    attempt_id: str,
) -> dict[str, Path | str]:
    """Project the sole reviewed correction attempt without touching its paths."""
    root = Path(task_root)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or root.resolve(strict=True) != root):
        raise Task8ControlError("corrected RNG Task 8 root must be a canonical absolute directory")
    if run_id != CORRECTED_RNG_RUN_ID or attempt_id != CORRECTED_RNG_ATTEMPT_ID:
        raise Task8ControlError("only the approved corrected RNG run and attempt are allowed")
    runtime = root / "runtime" / run_id
    output = root / "runtime-output" / run_id
    control = root / "runtime-control" / run_id
    admin = root / "runtime-admin" / run_id
    original_output = output / "rng-red-work"
    original_control = control / "rng-red"
    original_admin = admin / "rng-red"
    corrected_output = output / f"{CORRECTED_RNG_PHASE_KEY}-work"
    corrected_control = control / CORRECTED_RNG_PHASE_KEY
    corrected_admin = admin / CORRECTED_RNG_PHASE_KEY
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "phase_key": CORRECTED_RNG_PHASE_KEY,
        "container_name": f"spirelearn-task8-{CORRECTED_RNG_PHASE_KEY}-{run_id}",
        "original_started_marker": control / "rng-red.started",
        "corrected_started_marker": control / f"{CORRECTED_RNG_PHASE_KEY}.started",
        "previous_failed_output_root": original_output,
        "previous_failed_control_root": original_control,
        "previous_failed_admin_root": original_admin,
        "previous_failed_inventory_path": original_admin / "evidence-inventory.json",
        "previous_failed_start_stdout_path": original_admin / "logs/container-start.stdout.log",
        "previous_failed_cleanup_status_path": original_control / "cleanup-status.json",
        "corrected_output_root": corrected_output,
        "corrected_control_root": corrected_control,
        "corrected_admin_root": corrected_admin,
        "corrected_log_root": corrected_admin / "logs",
        "corrected_phase_started_marker": corrected_control / "phase.started",
        "corrected_transcript_path": corrected_admin / "logs/rng-red-transcript.stdout.log",
        "corrected_inventory_path": corrected_admin / "evidence-inventory.json",
        "runtime_root": runtime,
        "output_root": output,
        "control_root": control,
        "admin_root": admin,
    }


def verify_previous_failed_rng_attempt(
    task_root: Path,
    *,
    run_id: str,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Verify the immutable original ModelConfig harness failure for correction."""
    paths = project_corrected_rng_attempt_paths(
        task_root, run_id=run_id, attempt_id=CORRECTED_RNG_ATTEMPT_ID,
    )
    marker = Path(paths["original_started_marker"])
    inventory_path = Path(paths["previous_failed_inventory_path"])
    cleanup_path = Path(paths["previous_failed_cleanup_status_path"])
    transcript_path = Path(paths["previous_failed_start_stdout_path"])
    admin_root = Path(paths["previous_failed_admin_root"])
    control_root = Path(paths["previous_failed_control_root"])
    output_root = Path(paths["previous_failed_output_root"])
    for path in (marker, inventory_path, cleanup_path, transcript_path):
        if path.is_symlink() or not path.is_file():
            raise Task8ControlError(f"original failed RNG evidence is missing or unsafe: {path}")
        if path.resolve(strict=True) != path:
            raise Task8ControlError(f"original failed RNG evidence is not canonical: {path}")
    inventory = verify_evidence_inventory(inventory_path, allowed_venv_roots=allowed_venv_roots)
    inventory_record = _json_object(inventory_path)
    expected_roots = [str(path) for path in (admin_root, control_root, output_root)]
    if inventory_record.get("roots") != expected_roots:
        raise Task8ControlError("original failed RNG inventory does not cover its exact phase roots")
    status = _json_object(cleanup_path)
    if (status.get("phase") != "rng-red" or status.get("run_id") != run_id
            or status.get("container_create_attempted") != 1
            or status.get("container_start_attempted") != 1
            or status.get("container_started") != 0
            or status.get("cleanup_failed") != 0 or status.get("status") != 2):
        raise Task8ControlError("original failed RNG cleanup record differs from the retained harness failure")
    transcript = transcript_path.read_text(encoding="utf-8")
    required_failure_text = (
        "ModelConfig.__init__() missing 1 required positional argument: 'max_parameters'",
        "Ran 3 tests",
        "FAILED (failures=1, errors=2)",
    )
    if any(text not in transcript for text in required_failure_text):
        raise Task8ControlError("original RNG transcript is not the reviewed ModelConfig harness failure")
    return {
        "run_id": run_id,
        "phase": "rng-red",
        "attempt_id": "original-failed",
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory["inventory_sha256"],
        "container_started_record_preserved": 0,
        "status": 2,
    }


def validate_corrected_rng_red_release(
    release_path: Path,
    *,
    task_root: Path,
    run_id: str,
    source_hash: str,
    closure_lock_hash: str,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Require the parent single-use release for the fixed correction attempt."""
    paths = project_corrected_rng_attempt_paths(
        task_root, run_id=run_id, attempt_id=CORRECTED_RNG_ATTEMPT_ID,
    )
    previous = verify_previous_failed_rng_attempt(
        task_root, run_id=run_id, allowed_venv_roots=allowed_venv_roots,
    )
    return validate_phase_release(
        release_path,
        phase="rng-red",
        source_hash=source_hash,
        required_fields={
            "run_id": run_id,
            "attempt_id": CORRECTED_RNG_ATTEMPT_ID,
            "corrected_source_snapshot_sha256": source_hash,
            "closure_lock_sha256": closure_lock_hash,
            "previous_failed_attempt_inventory_path": str(paths["previous_failed_inventory_path"]),
            "previous_failed_attempt_inventory_sha256": previous["inventory_sha256"],
        },
    )


def validate_corrected_rng_followup_release(
    release_path: Path,
    *,
    task_root: Path,
    phase: str,
    source_hash: str,
    closure_lock_hash: str,
    corrected_red_evidence: Mapping[str, object],
    rng_green_transcript_sha256: str | None = None,
) -> dict[str, object]:
    """Bind green/device release to the exact accepted corrected RED record."""
    if phase not in {"rng-green", "device-check"}:
        raise Task8ControlError("corrected RED evidence may release only RNG green or device-check")
    corrected_red_evidence = verify_corrected_rng_red_evidence(
        task_root,
        run_id=CORRECTED_RNG_RUN_ID,
        transcript_path=Path(str(corrected_red_evidence.get("transcript_path", ""))),
        transcript_sha256=str(corrected_red_evidence.get("transcript_sha256", "")),
        inventory_path=Path(str(corrected_red_evidence.get("inventory_path", ""))),
        inventory_sha256=str(corrected_red_evidence.get("inventory_sha256", "")),
        previous_failed_inventory_path=Path(str(
            corrected_red_evidence.get("previous_failed_attempt_inventory_path", "")
        )),
        previous_failed_inventory_sha256=str(
            corrected_red_evidence.get("previous_failed_attempt_inventory_sha256", "")
        ),
    )
    if (corrected_red_evidence.get("run_id") != CORRECTED_RNG_RUN_ID
            or corrected_red_evidence.get("phase") != "rng-red"
            or corrected_red_evidence.get("attempt_id") != CORRECTED_RNG_ATTEMPT_ID
            or corrected_red_evidence.get("status") != "expected_red"):
        raise Task8ControlError("follow-up release requires the accepted corrected RED evidence record")
    if corrected_red_evidence.get("closure_lock_sha256") != closure_lock_hash:
        raise Task8ControlError("corrected RED and follow-up phase do not use the same closure lock")
    required = {
        "run_id": CORRECTED_RNG_RUN_ID,
        "closure_lock_sha256": closure_lock_hash,
        "rng_red_attempt_id": CORRECTED_RNG_ATTEMPT_ID,
        "rng_red_transcript_path": corrected_red_evidence.get("transcript_path"),
        "rng_red_transcript_sha256": corrected_red_evidence.get("transcript_sha256"),
        "rng_red_inventory_path": corrected_red_evidence.get("inventory_path"),
        "rng_red_inventory_sha256": corrected_red_evidence.get("inventory_sha256"),
        "rng_red_source_snapshot_sha256": corrected_red_evidence.get("source_snapshot_sha256"),
        "rng_red_closure_lock_sha256": corrected_red_evidence.get("closure_lock_sha256"),
        "previous_failed_attempt_inventory_path": corrected_red_evidence.get(
            "previous_failed_attempt_inventory_path"
        ),
        "previous_failed_attempt_inventory_sha256": corrected_red_evidence.get(
            "previous_failed_attempt_inventory_sha256"
        ),
    }
    for key in ("rng_red_transcript_sha256", "rng_red_inventory_sha256",
                "rng_red_source_snapshot_sha256", "rng_red_closure_lock_sha256",
                "previous_failed_attempt_inventory_sha256"):
        value = required[key]
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise Task8ControlError(f"corrected RED release binding {key!r} is malformed")
    if any(not isinstance(required[key], str) for key in (
        "rng_red_transcript_path", "rng_red_inventory_path", "previous_failed_attempt_inventory_path",
    )):
        raise Task8ControlError("corrected RED release evidence paths are malformed")
    if phase == "device-check":
        if (not isinstance(rng_green_transcript_sha256, str)
                or _HEX64.fullmatch(rng_green_transcript_sha256) is None):
            raise Task8ControlError("device-check release must bind a verified green transcript")
        required["rng_regressions"] = "passed"
        required["rng_green_transcript_sha256"] = rng_green_transcript_sha256
    elif rng_green_transcript_sha256 is not None:
        raise Task8ControlError("RNG green release may not bind a future green transcript")
    return validate_phase_release(
        release_path, phase=phase, source_hash=source_hash, required_fields=required,
    )


def verify_corrected_rng_red_evidence(
    task_root: Path,
    *,
    run_id: str,
    transcript_path: Path,
    transcript_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
    previous_failed_inventory_path: Path,
    previous_failed_inventory_sha256: str,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Verify exact canonical accepted RED and both immutable inventory identities."""
    paths = project_corrected_rng_attempt_paths(
        task_root, run_id=run_id, attempt_id=CORRECTED_RNG_ATTEMPT_ID,
    )
    previous = verify_previous_failed_rng_attempt(
        task_root, run_id=run_id, allowed_venv_roots=allowed_venv_roots,
    )

    def exact_file(raw_path: Path, expected: Path, digest: str, label: str) -> Path:
        path = Path(raw_path)
        if (not path.is_absolute() or path != expected or path.is_symlink()
                or not path.is_file() or path.resolve(strict=True) != path):
            raise Task8ControlError(f"{label} path is not the exact canonical corrected RNG evidence file")
        if _HEX64.fullmatch(digest) is None or sha256_file(path) != digest:
            raise Task8ControlError(f"{label} SHA256 does not match the parent-bound evidence")
        return path

    if (Path(previous_failed_inventory_path) != Path(paths["previous_failed_inventory_path"])
            or previous_failed_inventory_sha256 != previous["inventory_sha256"]):
        raise Task8ControlError("previous failed RNG inventory path/hash differs from the retained original")
    transcript_file = exact_file(
        transcript_path, Path(paths["corrected_transcript_path"]), transcript_sha256,
        "corrected RED transcript",
    )
    inventory_file = exact_file(
        inventory_path, Path(paths["corrected_inventory_path"]), inventory_sha256,
        "corrected RED inventory",
    )
    inventory = verify_evidence_inventory(inventory_file, allowed_venv_roots=allowed_venv_roots)
    inventory_record = _json_object(inventory_file)
    expected_roots = [
        str(paths[key]) for key in (
            "corrected_admin_root", "corrected_control_root", "corrected_output_root",
        )
    ]
    if inventory_record.get("roots") != expected_roots:
        raise Task8ControlError("corrected RED inventory does not cover its exact phase roots")
    corrected_status_path = Path(paths["corrected_control_root"]) / "cleanup-status.json"
    corrected_status = _json_object(corrected_status_path)
    if (corrected_status.get("phase") != "rng-red" or corrected_status.get("run_id") != run_id
            or corrected_status.get("attempt_id") != CORRECTED_RNG_ATTEMPT_ID
            or corrected_status.get("container_create_attempted") != 1
            or corrected_status.get("container_start_attempted") != 1
            or corrected_status.get("container_start_invocation_returned") != 1
            or corrected_status.get("container_started") != 1
            or corrected_status.get("cleanup_failed") != 0 or corrected_status.get("status") != 0):
        raise Task8ControlError("corrected RED attempt did not finish with accepted cleanup/start evidence")
    transcript = _json_object(transcript_file)
    if (set(transcript) != {"expected_labels", "observed_labels", "ran", "failures", "skipped", "status"}
            or transcript.get("expected_labels") != list(EXPECTED_RNG_LABELS)
            or transcript.get("observed_labels") != list(EXPECTED_RNG_LABELS)
            or transcript.get("ran") != 3
            or transcript.get("failures") != list(EXPECTED_RNG_LABELS[:2])
            or transcript.get("skipped") != 0 or transcript.get("status") != "expected_red"):
        raise Task8ControlError("corrected RED transcript is not the exact accepted three-test expected RED")
    identity_path = Path(paths["corrected_control_root"]) / "phase-input-identities.json"
    identity = _json_object(identity_path)
    prior_rows = identity.get("prior_inventories")
    immutable_rows = identity.get("immutable_paths")
    if (identity.get("format") != "spire-learn-task8-phase-input-identities-v1"
            or not isinstance(prior_rows, list) or not isinstance(immutable_rows, list)
            or not any(
                isinstance(row, dict) and row.get("phase") == "rng-red"
                and row.get("inventory_path") == str(paths["previous_failed_inventory_path"])
                and row.get("inventory_sha256") == previous["inventory_sha256"]
                for row in prior_rows
            )):
        raise Task8ControlError("corrected RED phase inputs do not bind the original failed inventory")
    immutable_by_name = {
        row.get("name"): row for row in immutable_rows if isinstance(row, dict)
    }
    source_identity = immutable_by_name.get("source_snapshot")
    lock_identity = immutable_by_name.get("closure_lock")
    if (not isinstance(source_identity, dict) or not isinstance(source_identity.get("sha256"), str)
            or _HEX64.fullmatch(source_identity["sha256"]) is None
            or not isinstance(lock_identity, dict) or not isinstance(lock_identity.get("sha256"), str)
            or _HEX64.fullmatch(lock_identity["sha256"]) is None):
        raise Task8ControlError("corrected RED phase inputs lack source snapshot or closure lock identity")
    expected_source = Path(task_root) / CORRECTED_RNG_SOURCE_SNAPSHOT_NAME
    expected_lock = Path(paths["output_root"]) / "resolve-work/requirements-cuda.lock"
    for identity_row, expected_path, label in (
        (source_identity, expected_source, "source snapshot"),
        (lock_identity, expected_lock, "closure lock"),
    ):
        identity_path_value = identity_row.get("path")
        if (identity_path_value != str(expected_path) or expected_path.is_symlink()
                or not expected_path.is_file() or expected_path.resolve(strict=True) != expected_path
                or sha256_file(expected_path) != identity_row.get("sha256")):
            raise Task8ControlError(f"corrected RED {label} input identity is not the exact retained file")
    return {
        "run_id": run_id,
        "phase": "rng-red",
        "attempt_id": CORRECTED_RNG_ATTEMPT_ID,
        "transcript_path": str(transcript_file),
        "transcript_sha256": sha256_file(transcript_file),
        "inventory_path": str(inventory_file),
        "inventory_sha256": inventory["inventory_sha256"],
        "previous_failed_attempt_inventory_path": previous["inventory_path"],
        "previous_failed_attempt_inventory_sha256": previous["inventory_sha256"],
        "source_snapshot_sha256": source_identity["sha256"],
        "closure_lock_sha256": lock_identity["sha256"],
        "status": "expected_red",
    }


def validate_phase(phase: str) -> str:
    if phase not in {"resolve", "install", "rng-red", "rng-green", "device-check"}:
        raise Task8ControlError("phase must be resolve, install, rng-red, rng-green, or device-check")
    return phase


def _container_identity(run_id: str, phase: str, attempt_id: str | None = None) -> tuple[str, str]:
    if attempt_id is None:
        return phase, f"spirelearn-task8-{phase}-{run_id}"
    if (phase != "rng-red" or run_id != CORRECTED_RNG_RUN_ID
            or attempt_id != CORRECTED_RNG_ATTEMPT_ID):
        raise Task8ControlError("only the approved corrected RNG attempt may use an attempt-scoped container")
    return CORRECTED_RNG_PHASE_KEY, f"spirelearn-task8-{CORRECTED_RNG_PHASE_KEY}-{run_id}"


def build_docker_create_argv(
    *,
    image: str,
    run_id: str,
    phase: str,
    command: Sequence[str],
    mounts: Sequence[tuple[Path, str, bool]],
    network: str,
    gpu_uuid: str | None = None,
    attempt_id: str | None = None,
) -> tuple[str, ...]:
    """Build strict Docker argv for a later parent-released phase."""
    validate_run_id(run_id)
    validate_phase(phase)
    _phase_key, container_name = _container_identity(run_id, phase, attempt_id)
    if not image.startswith("spire-learn/task8-python:"):
        raise Task8ControlError("container image must be the verified run-scoped local alias")
    expected_network = "bridge" if phase == "resolve" else "none"
    if network != expected_network:
        raise Task8ControlError(f"{phase} phase requires network={expected_network}")
    gpu_phase = phase in {"rng-red", "rng-green", "device-check"}
    if gpu_phase and gpu_uuid != GPU_UUID:
        raise Task8ControlError("GPU phase must pass only the inventoried RTX 5090 UUID")
    if not gpu_phase and gpu_uuid is not None:
        raise Task8ControlError("setup resolution/install must not request a GPU")
    if not command or any(not isinstance(item, str) or "\x00" in item for item in command):
        raise Task8ControlError("container command must be a non-empty string argv")
    destinations: set[str] = set()
    mount_args: list[str] = []
    for source, destination, readonly in mounts:
        source = Path(source)
        if not source.is_absolute() or source.is_symlink() or not source.exists():
            raise Task8ControlError("every bind source must be an existing absolute non-symlink path")
        if not destination.startswith("/") or destination in destinations or "," in str(source):
            raise Task8ControlError("bind destinations must be unique absolute paths")
        destinations.add(destination)
        mode = ",readonly" if readonly else ""
        mount_args.extend(("--mount", f"type=bind,src={source},dst={destination}{mode}"))
    if "/work" not in destinations:
        raise Task8ControlError("one fresh output child must be mounted writable at /work")
    work_mount = next(item for item in mounts if item[1] == "/work")
    if work_mount[2] is not False:
        raise Task8ControlError("/work must be writable for phase-owned output only")
    if any(readonly is False and destination != "/work" for _source, destination, readonly in mounts):
        raise Task8ControlError("/work must be the only writable bind")
    args = [
        "docker", "create", "--pull=never", "--name", container_name,
        "--label", "spire-learn.task=task8-gpu-setup",
        "--label", f"spire-learn.phase={phase}",
        "--label", f"spire-learn.run-id={run_id}",
        "--platform", "linux/amd64", "--network", network, "--read-only",
        "--user", "1000:1000", "--cpus", "4", "--memory", "8g",
        "--pids-limit", "512", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--workdir", "/learner",
    ]
    if attempt_id is not None:
        args.extend(("--label", f"spire-learn.attempt-id={attempt_id}"))
    if gpu_phase:
        args.extend(("--gpus", f"device={GPU_UUID}"))
    args.extend(mount_args)
    args.extend((
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONNOUSERSITE=1",
        "--env", "PIP_CONFIG_FILE=/dev/null",
        "--env", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "--env", "PIP_NO_INPUT=1",
        "--env", "TMPDIR=/work/tmp",
        "--env", "PIP_CACHE_DIR=/work/pip-cache",
        "--env", "HOME=/tmp",
        "--entrypoint", "/bin/sh", image, *command,
    ))
    return tuple(args)


def assert_exact_container_name_absent(exit_status: int, stdout: bytes | str) -> None:
    """Require a successful exact-name Docker lookup with no matching rows."""
    if type(exit_status) is not int or not 0 <= exit_status <= 255:
        raise Task8ControlError("exact-name container lookup exit status is malformed")
    if exit_status != 0:
        raise Task8ControlError(
            f"exact-name container lookup failed with exit status {exit_status}; creation is not allowed"
        )
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if not isinstance(stdout, bytes):
        raise Task8ControlError("exact-name container lookup output is malformed")
    if stdout:
        raise Task8ControlError("exact-name container must be absent before creation")


def parse_created_container_id(exit_status: int, stdout: bytes | str) -> str:
    """Establish ownership only from a successful create's single full ID."""
    if type(exit_status) is not int or not 0 <= exit_status <= 255:
        raise Task8ControlError("docker create exit status is malformed; container ownership was not established")
    if exit_status != 0:
        raise Task8ControlError(
            f"docker create exit status {exit_status} did not establish container ownership"
        )
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if not isinstance(stdout, bytes):
        raise Task8ControlError("docker create output is malformed; container ownership was not established")
    match = re.fullmatch(rb"([0-9a-f]{64})\n?", stdout)
    if match is None:
        raise Task8ControlError("docker create must return exactly one full 64-digit container ID")
    return match.group(1).decode("ascii")


def _recorded_docker_exit_status(path: Path) -> int:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Task8ControlError(f"Docker command exit record is missing or unsafe: {path}")
    raw = path.read_bytes()
    match = re.fullmatch(rb"(0|[1-9][0-9]{0,2})\n", raw)
    if match is None:
        raise Task8ControlError(f"Docker command exit record is malformed: {path}")
    status = int(match.group(1))
    if status > 255:
        raise Task8ControlError(f"Docker command exit record is out of range: {path}")
    return status


def _recorded_docker_stdout(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Task8ControlError(f"Docker command stdout record is missing or unsafe: {path}")
    return path.read_bytes()


def authorize_container_removal(
    requested_container_id: str,
    *,
    create_exit_status: int,
    create_stdout: bytes | str,
    cleanup_assert_exit_status: int,
    claim_path: Path,
) -> dict[str, object]:
    """Write an exclusive one-use cleanup claim for the verified created ID."""
    created_id = parse_created_container_id(create_exit_status, create_stdout)
    if requested_container_id != created_id:
        raise Task8ControlError("container removal ID does not match the successful create return")
    if (type(cleanup_assert_exit_status) is not int or not 0 <= cleanup_assert_exit_status <= 255
            or cleanup_assert_exit_status != 0):
        raise Task8ControlError("container removal requires a successful cleanup spec assertion")
    claim_path = Path(claim_path)
    payload = json.dumps({
        "format": "spire-learn-task8-container-removal-authorization-v1",
        "container_id": created_id,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    try:
        fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise Task8ControlError("container removal was already authorized once") from error
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"claim_path": str(claim_path), "container_id": created_id}


def build_phase_spec(
    *,
    phase: str,
    run_id: str,
    image: str,
    expected_image_id: str,
    work: Path,
    helper: Path,
    local_spire_wheel: Path,
    resolve_output: Path,
    install_output: Path,
    source_root: Path,
    tests_root: Path,
    device_check_script: Path,
    gpu_inventory: Path,
    attempt_id: str | None = None,
) -> dict[str, object]:
    """Materialize the reviewed exact mount and command projection for a phase."""
    validate_run_id(run_id)
    validate_phase(phase)
    phase_key, container_name = _container_identity(run_id, phase, attempt_id)
    for path in (work, helper):
        if Path(path).is_symlink() or not Path(path).exists():
            raise Task8ControlError(f"phase input is missing or unsafe: {path}")
    mounts: list[dict[str, object]] = []

    def bind(source: Path, destination: str, readonly: bool) -> None:
        source = Path(source)
        if source.is_symlink() or not source.exists():
            raise Task8ControlError(f"phase bind input is missing or unsafe: {source}")
        mounts.append({"source": str(source.resolve(strict=True)), "destination": destination,
                       "readonly": readonly})

    bind(helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True)
    bind(work, "/work", False)
    if phase == "resolve":
        local_wheel_name = Path(local_spire_wheel).name
        if local_wheel_name != CPU_WHEEL_FILENAME:
            raise Task8ControlError("local Spire Lab bind must keep its exact accepted wheel filename")
        bind(local_spire_wheel, f"/inputs/{CPU_WHEEL_FILENAME}", True)
        script = (
            "set -eu\n"
            "export TMPDIR=/work/tmp PIP_CACHE_DIR=/work/pip-cache\n"
            "mkdir -p /work/tmp /work/pip-cache /work/wheelhouse\n"
            "python3.12 -B /learner/scripts/neural/task8_gpu_setup_helpers.py write-resolution-input /work/requirements-cuda.in\n"
            "python3.12 -B /learner/scripts/neural/task8_gpu_setup_helpers.py write-resolver-command /work/resolver-command.json\n"
            "python3.12 -B -m pip install --dry-run --ignore-installed --report /work/pip-resolution-report.json "
            "--only-binary=:all: --index-url https://pypi.org/simple -r /work/requirements-cuda.in\n"
            "python3.12 -B -m pip download --only-binary=:all: --index-url https://pypi.org/simple "
            "--dest /work/wheelhouse -r /work/requirements-cuda.in\n"
            "python3.12 -B /learner/scripts/neural/task8_gpu_setup_helpers.py capture-cuda-closure "
            "/work/pip-resolution-report.json /work/wheelhouse /work/cuda-closure.json "
            f"--local-spire-wheel /inputs/{CPU_WHEEL_FILENAME} --resolver-command-json /work/resolver-command.json\n"
            "python3.12 -B /learner/scripts/neural/task8_gpu_setup_helpers.py audit-cuda-closure "
            "/work/cuda-closure.json /work/wheelhouse /work/requirements-cuda.lock"
        )
        network, gpu_uuid = "bridge", None
    elif phase == "install":
        bind(Path(resolve_output) / "wheelhouse", "/inputs/wheelhouse", True)
        bind(Path(resolve_output) / "requirements-cuda.lock", "/inputs/requirements-cuda.lock", True)
        local_wheel_name = Path(local_spire_wheel).name
        if local_wheel_name != CPU_WHEEL_FILENAME:
            raise Task8ControlError("local Spire Lab bind must keep its exact accepted wheel filename")
        bind(local_spire_wheel, f"/inputs/{CPU_WHEEL_FILENAME}", True)
        script = (
            "set -eu\n"
            "export TMPDIR=/work/tmp PIP_CACHE_DIR=/work/pip-cache\n"
            "mkdir -p /work/tmp /work/pip-cache\n"
            "python3.12 -B -m venv --copies /work/venv-cuda\n"
            f"/work/venv-cuda/bin/python -B -m pip install --no-index --find-links /inputs/wheelhouse "
            "--require-hashes -r /inputs/requirements-cuda.lock\n"
            "/work/venv-cuda/bin/python -B -m pip check"
        )
        network, gpu_uuid = "none", None
    else:
        bind(source_root, "/learner/src", True)
        bind(tests_root, "/learner/tests", True)
        bind(Path(install_output) / "venv-cuda", "/runtime/venv-cuda", True)
        if phase == "device-check":
            bind(gpu_inventory, "/inputs/gpu-inventory.json", True)
            bind(device_check_script, "/learner/scripts/neural/task8_device_check.py", True)
            script = (
                "set -eu\n"
                "export SPIRE_LEARN_TASK8_DEVICE_CHECK=1\n"
                f"export SPIRE_LEARN_TASK8_GPU_UUID={GPU_UUID}\n"
                "exec /runtime/venv-cuda/bin/python -B /learner/scripts/neural/task8_device_check.py "
                "--inventory /inputs/gpu-inventory.json --output /work/device-check.json"
            )
        else:
            script = (
                "set -eu\n"
                "export SPIRE_LEARN_RUN_TASK8_RNG=1\n"
                "export PYTHONPATH=/learner/src\n"
                "exec /runtime/venv-cuda/bin/python -B -m unittest "
                "tests.neural.test_gpu_qualification -v 2>&1"
            )
        network, gpu_uuid = "none", GPU_UUID
    result = {
        "image": image, "expected_image_id": expected_image_id,
        "run_id": run_id, "phase": phase, "network": network,
        "gpu_uuid": gpu_uuid, "mounts": mounts, "command": ["-ec", script],
        "phase_key": phase_key, "container_name": container_name,
    }
    if attempt_id is not None:
        result["attempt_id"] = attempt_id
    return result


def assert_container_inspect(
    raw: str | bytes,
    *,
    expected_image_id: str,
    run_id: str,
    phase: str,
    expected_command: Sequence[str],
    expected_mounts: Sequence[tuple[Path, str, bool]],
    network: str,
    gpu_uuid: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, object]:
    """Fail closed unless Docker's created container matches the reviewed spec."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Task8ControlError("docker inspect output is not JSON") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise Task8ControlError("docker inspect must describe exactly one container")
    container = values[0]
    container_id = container.get("Id")
    if not isinstance(container_id, str) or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise Task8ControlError("created container ID must be one full 64-digit ID")
    if container.get("Image") != expected_image_id:
        raise Task8ControlError("container image ID changed after alias verification")
    config = container.get("Config")
    host = container.get("HostConfig")
    mounts = container.get("Mounts")
    if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(mounts, list):
        raise Task8ControlError("Docker inspect lacks Config, HostConfig, or Mounts")
    if config.get("User") != "1000:1000" or config.get("WorkingDir") != "/learner":
        raise Task8ControlError("container user or working directory differs from approved values")
    if config.get("Entrypoint") != ["/bin/sh"] or config.get("Cmd") != list(expected_command):
        raise Task8ControlError("container command differs from the exact reviewed argv")
    labels = config.get("Labels")
    expected_labels = {
        "spire-learn.task": "task8-gpu-setup",
        "spire-learn.phase": phase,
        "spire-learn.run-id": run_id,
    }
    phase_key, container_name = _container_identity(run_id, phase, attempt_id)
    if attempt_id is not None:
        expected_labels["spire-learn.attempt-id"] = attempt_id
        if container.get("Name") != f"/{container_name}":
            raise Task8ControlError("container name differs from the exact corrected RNG attempt")
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected_labels.items()):
        raise Task8ControlError("container ownership labels do not match this phase/run ID")
    env = config.get("Env")
    if not isinstance(env, list):
        raise Task8ControlError("container environment is missing")
    environment = dict(item.split("=", 1) for item in env if isinstance(item, str) and "=" in item)
    required_env = {
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": "/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1", "HOME": "/tmp",
        "TMPDIR": "/work/tmp", "PIP_CACHE_DIR": "/work/pip-cache",
    }
    if any(environment.get(key) != value for key, value in required_env.items()):
        raise Task8ControlError("container Python/pip isolation environment is incomplete")
    if any(any(secret in key.lower() for secret in ("token", "password", "credential", "secret"))
           for key in environment):
        raise Task8ControlError("secret-like environment variables are not allowed in the runtime")
    expected_network = "bridge" if phase == "resolve" else "none"
    if network != expected_network or host.get("NetworkMode") != expected_network:
        raise Task8ControlError("container network differs from the phase-specific boundary")
    if (host.get("ReadonlyRootfs") is not True or host.get("Privileged") is not False
            or host.get("AutoRemove") is not False or host.get("CapDrop") != ["ALL"]
            or host.get("SecurityOpt") != ["no-new-privileges"]
            or host.get("NanoCpus") != 4_000_000_000
            or host.get("Memory") != 8 * 1024**3 or host.get("PidsLimit") != 512
            or host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=64m"}):
        raise Task8ControlError("container rootfs, privilege, process, or resource limits differ")
    def absent(value: object) -> bool:
        return value is None or (isinstance(value, list) and not value)

    if (not absent(host.get("Devices")) or not absent(host.get("CapAdd"))
            or not absent(host.get("GroupAdd"))):
        raise Task8ControlError("container may not add devices, capabilities, or supplemental groups")
    if not absent(host.get("DeviceCgroupRules")):
        raise Task8ControlError("container may not add device cgroup rules")
    gpu_phase = phase in {"rng-red", "rng-green", "device-check"}
    if gpu_phase:
        if gpu_uuid != GPU_UUID:
            raise Task8ControlError("GPU container must request the approved UUID only")
        requests = host.get("DeviceRequests")
        if not isinstance(requests, list) or len(requests) != 1:
            raise Task8ControlError("GPU container must have exactly one device request")
        request = requests[0]
        if (not isinstance(request, dict) or request.get("DeviceIDs") != [GPU_UUID]
                or request.get("Capabilities") != [["gpu"]]):
            raise Task8ControlError("GPU container exposed a different device or capability set")
    elif not absent(host.get("DeviceRequests")) or gpu_uuid is not None:
        raise Task8ControlError("resolution/install container must not receive a GPU")
    expected_mount_rows = sorted(
        (str(Path(source).resolve(strict=True)), destination, not readonly)
        for source, destination, readonly in expected_mounts
    )
    observed_mount_rows: list[tuple[str, str, bool]] = []
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get("Type") != "bind":
            raise Task8ControlError("only exact bind mounts are allowed")
        source, destination, rw = mount.get("Source"), mount.get("Destination"), mount.get("RW")
        if not isinstance(source, str) or not isinstance(destination, str) or type(rw) is not bool:
            raise Task8ControlError("container bind mount entry is malformed")
        observed_mount_rows.append((str(Path(source).resolve(strict=True)), destination, rw))
    if sorted(observed_mount_rows) != expected_mount_rows:
        raise Task8ControlError("container mount table differs from the exact reviewed allowlist")
    if sum(1 for _source, destination, rw in observed_mount_rows if destination == "/work" and rw) != 1:
        raise Task8ControlError("container must have exactly one writable /work bind")
    return {
        "container_id": container_id,
        "image_id": expected_image_id,
        "phase": phase,
        "phase_key": phase_key,
        "run_id": run_id,
        "network": network,
        "gpu_uuid": GPU_UUID if gpu_phase else None,
        "mount_count": len(mounts),
        "read_only_rootfs": True,
        "user": "1000:1000",
    }


def reserve_cumulative(current: int, requested: int, *, cap: int, name: str) -> int:
    """Reserve an entire operation before dispatch; failed work never refunds it."""
    if any(type(value) is not int or value < 0 for value in (current, requested, cap)):
        raise Task8ControlError(f"{name} counters must be non-negative primitive integers")
    if requested == 0 or cap == 0 or current > cap or current + requested > cap:
        raise Task8ControlError(f"{name} reservation would exceed its cumulative cap")
    return current + requested


def verify_device_check_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate the non-gameplay CUDA qualification result after its one run."""
    if report.get("status") != "qualified":
        raise Task8ControlError("device report did not finish with qualified status")
    if report.get("gpu_uuid") != GPU_UUID or report.get("visible_gpu_uuid") != GPU_UUID:
        raise Task8ControlError("device report does not prove the visible GPU UUID")
    if report.get("torch") != "2.14.1+cu130":
        raise Task8ControlError("device report does not use the exact CUDA Torch candidate")
    cuda_version = report.get("torch_cuda")
    if not isinstance(cuda_version, str) or not cuda_version.startswith("13.0"):
        raise Task8ControlError("Torch CUDA runtime must report version 13.0.x")
    if report.get("device_name") != GPU_NAME:
        raise Task8ControlError("Torch-visible device name differs from the RTX 5090")
    if (report.get("torch_cuda_initialized") is not True
            or type(report.get("torch_cuda_device_count")) is not int
            or report.get("torch_cuda_device_count") != 1):
        raise Task8ControlError("device report lacks a completed single-device CUDA initialization")
    if report.get("driver_version") != GPU_DRIVER:
        raise Task8ControlError("device report driver differs from the inventoried driver")
    capability = report.get("compute_capability")
    architectures = report.get("compiled_architectures")
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
            or not isinstance(architectures, list) or not architectures
            or any(not isinstance(item, str) for item in architectures)):
        raise Task8ControlError("device capability and compiled architecture list are required")
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    if expected_arch not in architectures:
        raise Task8ControlError("Torch does not report a compiled architecture for this device")
    for field in ("finite_forward", "finite_backward", "finite_adam", "synchronized"):
        if report.get(field) is not True:
            raise Task8ControlError(f"device work did not prove {field}")
    if report.get("dtype") != "float32" or report.get("amp") is not False or report.get("compile") is not False:
        raise Task8ControlError("device qualification must use plain float32 without AMP or compile")
    if report.get("fallback") is not False:
        raise Task8ControlError("device qualification may not fall back to another device/package")
    peak = report.get("peak_allocated_bytes")
    free_before = report.get("memory_free_before_mib")
    if type(peak) is not int or not 0 <= peak <= 8 * 1024**3:
        raise Task8ControlError("Torch allocator peak exceeds the 8 GiB ceiling")
    counters = (
        report.get("optimizer_step_attempted"), report.get("optimizer_step_accepted"),
        report.get("optimizer_step_uncertain"), report.get("diagnostic_adam_calls"),
    )
    if (any(type(value) is not int for value in counters)
            or counters != (1, 1, 0, 1)):
        raise Task8ControlError("device report must show exactly one returned synthetic Adam step")
    if report.get("owned_process_identified") is not False or report.get("owned_process_memory_mib") is not None:
        raise Task8ControlError("GPU-wide process totals may not be labeled as this check's owned memory")
    process_memory = report.get("gpu_wide_process_memory_mib")
    processes = report.get("gpu_wide_processes")
    if type(process_memory) is not int or process_memory < 0 or not isinstance(processes, list):
        raise Task8ControlError("device report lacks GPU-wide process usage totals")
    if not processes:
        raise Task8ControlError("device report must contain at least one GPU-wide process row")
    process_total = 0
    for process in processes:
        if (not isinstance(process, dict) or process.get("gpu_uuid") != GPU_UUID
                or type(process.get("pid")) is not int or type(process.get("memory_mib")) is not int
                or process["memory_mib"] < 0):
            raise Task8ControlError("device process usage row is malformed or names another GPU")
        process_total += process["memory_mib"]
    if process_total != process_memory:
        raise Task8ControlError("GPU-wide process-memory total disagrees with nvidia-smi rows")
    if type(free_before) is not int or free_before < GPU_MIN_FREE_MIB:
        raise Task8ControlError("device check began below the required 12 GiB free memory")
    return {
        "status": "qualified",
        "gpu_uuid": GPU_UUID,
        "compiled_architecture": expected_arch,
        "peak_allocated_bytes": peak,
        "gpu_wide_process_memory_mib": process_memory,
        "owned_process_memory_identified": False,
        "hard_isolation_claim": False,
    }


def verify_gpu_inventory(raw: str, *, minimum_free_mib: int = GPU_MIN_FREE_MIB) -> dict[str, object]:
    """Check nvidia-smi CSV for the approved device and current free memory."""
    if type(minimum_free_mib) is not int or minimum_free_mib < GPU_MIN_FREE_MIB:
        raise Task8ControlError("GPU preflight may not lower the approved 12 GiB free-memory floor")
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    matching: list[tuple[str, str, str, int, int]] = []
    for line in rows:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            raise Task8ControlError("nvidia-smi inventory row must have five CSV fields")
        uuid, name, driver, total_text, free_text = fields
        try:
            total, free = int(total_text), int(free_text)
        except ValueError as error:
            raise Task8ControlError("nvidia-smi memory fields must be integer MiB") from error
        if uuid == GPU_UUID:
            matching.append((uuid, name, driver, total, free))
    if len(matching) != 1:
        raise Task8ControlError("exactly one approved RTX 5090 UUID must be inventoried")
    uuid, name, driver, total, free = matching[0]
    if name != GPU_NAME or driver != GPU_DRIVER or total != GPU_MEMORY_MIB:
        raise Task8ControlError("RTX 5090 identity differs from the reviewed inventory")
    if free < minimum_free_mib:
        raise Task8ControlError("approved GPU has less than 12 GiB free; stop before device work")
    return {
        "uuid": uuid, "name": name, "driver": driver,
        "memory_total_mib": total, "memory_free_mib": free,
        "minimum_free_mib": minimum_free_mib,
    }


def verify_storage_budget(
    *,
    new_gpu_bytes: int,
    neural_cumulative_bytes: int,
    free_bytes: int,
    setup: bool,
    reservation_bytes: int = 0,
) -> dict[str, int]:
    """Admit measured and reserved growth without pruning retained data."""
    values = (new_gpu_bytes, neural_cumulative_bytes, free_bytes, reservation_bytes)
    if any(type(value) is not int or value < 0 for value in values):
        raise Task8ControlError("storage accounting values must be non-negative integer bytes")
    gib = 1024**3
    new_projected = new_gpu_bytes + reservation_bytes
    neural_projected = neural_cumulative_bytes + reservation_bytes
    free_floor = 60 * gib if setup else 30 * gib
    if new_projected > 24 * gib:
        raise Task8ControlError("new Task 8 output plus reservation exceeds 24 GiB soft budget")
    if neural_projected > 40 * gib:
        raise Task8ControlError("neural cumulative allocation exceeds 40 GiB soft budget")
    if free_bytes < free_floor:
        raise Task8ControlError("free disk space is below the phase floor")
    return {
        "new_gpu_bytes": new_gpu_bytes,
        "reservation_bytes": reservation_bytes,
        "new_gpu_projected_bytes": new_projected,
        "neural_cumulative_bytes": neural_cumulative_bytes,
        "neural_projected_bytes": neural_projected,
        "free_bytes": free_bytes,
        "free_floor_bytes": free_floor,
    }


def _normalize_venv_roots(venv_roots: Sequence[Path]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for root in venv_roots:
        path = Path(root)
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_dir():
            raise Task8ControlError(f"owned venv root is missing or unsafe: {path}")
        resolved = path.resolve(strict=True)
        if path != resolved:
            raise Task8ControlError(f"owned venv root must be canonical: {path}")
        normalized.append(resolved)
    if len(set(normalized)) != len(normalized):
        raise Task8ControlError("owned venv roots must be unique")
    return tuple(normalized)


def _known_venv_link(path: Path, venv_roots: Sequence[Path]) -> str:
    path = Path(path)
    matching = [root for root in venv_roots if path == root / "lib64"]
    if len(matching) != 1:
        raise Task8ControlError(f"storage tree contains an unrecognized symlink: {path}")
    venv = matching[0]
    if os.readlink(path) != "lib":
        raise Task8ControlError(f"owned venv lib64 link must point to its sibling lib: {path}")
    library = venv / "lib"
    if library.is_symlink() or not library.is_dir():
        raise Task8ControlError(f"owned venv lib target is missing or unsafe: {library}")
    if venv.resolve(strict=True) not in library.resolve(strict=True).parents:
        raise Task8ControlError(f"owned venv lib target escapes the venv: {path}")
    return "lib"


def _normalize_protected_cpu_venv(root: Path | None) -> Path | None:
    if root is None:
        return None
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise Task8ControlError(f"protected CPU venv root is missing or unsafe: {path}")
    resolved = path.resolve(strict=True)
    if path != resolved:
        raise Task8ControlError(f"protected CPU venv root must be canonical: {path}")
    return resolved


def _known_protected_cpu_venv_link(path: Path, root: Path | None) -> str:
    if root is None or root not in path.parents:
        raise Task8ControlError(f"storage tree contains an unrecognized symlink: {path}")
    relative = path.relative_to(root).as_posix()
    expected_target = _PROTECTED_CPU_VENV_LINKS.get(relative)
    if expected_target is None or os.readlink(path) != expected_target:
        raise Task8ControlError(f"storage tree contains an unrecognized protected CPU venv symlink: {path}")
    # These are pre-existing protected-runtime aliases. Count only each link's
    # own inode bytes; never resolve or traverse their targets for accounting.
    return expected_target


def _storage_symlink_target(
    path: Path,
    venv_roots: Sequence[Path],
    protected_cpu_venv_root: Path | None,
) -> str:
    if any(path == root / "lib64" for root in venv_roots):
        return _known_venv_link(path, venv_roots)
    return _known_protected_cpu_venv_link(path, protected_cpu_venv_root)


def _tree_bytes(
    root: Path,
    *,
    reject_symlinks: bool = True,
    allowed_venv_roots: Sequence[Path] = (),
    protected_cpu_venv_root: Path | None = None,
) -> int:
    root = Path(root)
    if root.is_symlink():
        raise Task8ControlError(f"storage accounting root may not be a symlink: {root}")
    if not root.exists():
        return 0
    if not root.is_dir():
        raise Task8ControlError(f"storage accounting root is not a regular directory: {root}")
    venv_roots = _normalize_venv_roots(allowed_venv_roots)
    protected_cpu_root = _normalize_protected_cpu_venv(protected_cpu_venv_root)
    total = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(dirnames):
            path = directory_path / name
            if path.is_symlink():
                _storage_symlink_target(path, venv_roots, protected_cpu_root)
                total += path.lstat().st_size
                dirnames.remove(name)
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                _storage_symlink_target(path, venv_roots, protected_cpu_root)
                total += path.lstat().st_size
            elif path.is_file():
                total += path.stat(follow_symlinks=False).st_size
            else:
                raise Task8ControlError(f"storage tree contains a non-regular entry: {path}")
    return total


def check_storage_paths(
    filesystem_root: Path,
    neural_root: Path,
    new_gpu_roots: Sequence[Path],
    *,
    setup: bool,
    reservation_bytes: int = 0,
    reserve_remaining_setup: bool = False,
    allowed_venv_roots: Sequence[Path] = (),
    protected_cpu_venv_root: Path | None = None,
) -> dict[str, int]:
    """Conservatively count retained neural bytes and Task 8 output growth."""
    filesystem_root = Path(filesystem_root)
    if filesystem_root.is_symlink() or not filesystem_root.is_dir():
        raise Task8ControlError("filesystem accounting root is unsafe")
    neural_root = Path(neural_root)
    protected_cpu_root = _normalize_protected_cpu_venv(protected_cpu_venv_root)
    if protected_cpu_root is not None and neural_root.resolve(strict=True) not in protected_cpu_root.parents:
        raise Task8ControlError("protected CPU venv must be contained by the retained neural-combat tree")
    normalized_roots = [Path(path) for path in new_gpu_roots]
    if len({str(path.resolve(strict=True)) for path in normalized_roots}) != len(normalized_roots):
        raise Task8ControlError("new GPU storage roots must be unique")
    new_gpu_bytes = sum(
        _tree_bytes(
            path, reject_symlinks=True, allowed_venv_roots=allowed_venv_roots,
            protected_cpu_venv_root=protected_cpu_root,
        )
        for path in normalized_roots
    )
    neural_current_bytes = _tree_bytes(
        neural_root, reject_symlinks=True, allowed_venv_roots=allowed_venv_roots,
        protected_cpu_venv_root=protected_cpu_root,
    )
    if reserve_remaining_setup:
        if not setup:
            raise Task8ControlError("full setup reservation is valid only for setup admission")
        reservation_bytes = 24 * 1024**3 - new_gpu_bytes
        if reservation_bytes < 0:
            raise Task8ControlError("Task 8 output already exceeds the 24 GiB setup cap")
    free_bytes = shutil.disk_usage(filesystem_root).free
    result = verify_storage_budget(
        new_gpu_bytes=new_gpu_bytes,
        neural_cumulative_bytes=neural_current_bytes,
        free_bytes=free_bytes,
        setup=setup,
        reservation_bytes=reservation_bytes,
    )
    return {
        **result,
        "neural_current_bytes": neural_current_bytes,
    }


def project_offline_install_bytes(wheelhouse: Path, local_wheel: Path) -> dict[str, int]:
    """Bound offline install expansion, one-wheel staging, and venv/cache overhead."""
    wheelhouse = Path(wheelhouse)
    local_wheel = Path(local_wheel)
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise Task8ControlError("install projection wheelhouse is missing or unsafe")
    if local_wheel.is_symlink() or not local_wheel.is_file():
        raise Task8ControlError("install projection local Spire Lab wheel is missing or unsafe")
    wheel_paths = []
    for path in wheelhouse.iterdir():
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".whl"):
            raise Task8ControlError("install projection wheelhouse has a non-wheel or symlink entry")
        wheel_paths.append(path)
    wheel_paths.append(local_wheel)
    expanded_sizes: list[int] = []
    for path in wheel_paths:
        try:
            with zipfile.ZipFile(path) as archive:
                expanded_sizes.append(sum(item.file_size for item in archive.infolist()))
        except (OSError, zipfile.BadZipFile) as error:
            raise Task8ControlError(f"install projection cannot read wheel ZIP sizes: {path.name}") from error
    if not expanded_sizes:
        raise Task8ControlError("install projection found no wheels")
    uncompressed = sum(expanded_sizes)
    largest = max(expanded_sizes)
    bounded_overhead = 1024**3
    return {
        "wheel_count": len(wheel_paths),
        "uncompressed_wheel_bytes": uncompressed,
        "largest_uncompressed_wheel_bytes": largest,
        "bounded_temp_cache_venv_overhead_bytes": bounded_overhead,
        "projected_additional_bytes": uncompressed + largest + bounded_overhead,
    }


def verify_timeout_bounds(
    *,
    elapsed_seconds: float,
    phase: str,
    cleanup_completed: bool,
    active_seconds_before: int = 0,
    cleanup_reserve_seconds: int = 0,
    external_parent_pause_seconds: int = 0,
) -> dict[str, object]:
    if (isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds)) or elapsed_seconds < 0):
        raise Task8ControlError("elapsed time must be a finite non-negative number")
    if any(type(value) is not int or value < 0 for value in (
        active_seconds_before, cleanup_reserve_seconds, external_parent_pause_seconds,
    )):
        raise Task8ControlError("time budget counters must be non-negative primitive integers")
    validate_phase(phase)
    setup = phase in {"resolve", "install"}
    soft, whole = (600, 1200) if setup else (60, 120)
    if not setup and active_seconds_before != 0:
        raise Task8ControlError("RNG and device checks have independent per-invocation budgets")
    if active_seconds_before > whole:
        raise Task8ControlError("cumulative active time already exceeds the whole-run limit")
    elapsed_rounded_up = math.ceil(float(elapsed_seconds))
    charged = elapsed_rounded_up + cleanup_reserve_seconds
    cumulative = active_seconds_before + charged
    if cumulative > whole or not cleanup_completed:
        raise Task8ControlError("phase exceeded cumulative whole-run limit or lacks completed cleanup evidence")
    return {
        "phase": phase,
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_seconds_rounded_up": elapsed_rounded_up,
        "cleanup_reserve_seconds": cleanup_reserve_seconds,
        "active_seconds_before": active_seconds_before,
        "active_seconds_charged": cumulative,
        "phase_active_seconds_charged": charged,
        "soft_limit_seconds": soft,
        "whole_limit_seconds_including_cleanup": whole,
        "soft_overshoot_seconds": max(0, cumulative - soft),
        "remaining_soft_seconds": max(0, soft - cumulative),
        "remaining_hard_seconds": max(0, whole - cumulative),
        "external_parent_pause_seconds": external_parent_pause_seconds,
        "cleanup_completed": True,
    }


def write_phase_budget_state(
    output_path: Path,
    *,
    phase: str,
    started_epoch_seconds: float,
    active_seconds_before: int,
    external_parent_pause_seconds: int = 0,
    cleanup_completed: bool,
    cleanup_reserve_seconds: int = 15,
) -> dict[str, object]:
    if (isinstance(started_epoch_seconds, bool)
            or not isinstance(started_epoch_seconds, (int, float))
            or not math.isfinite(float(started_epoch_seconds))):
        raise Task8ControlError("phase start timestamp is malformed")
    elapsed = time.time() - float(started_epoch_seconds)
    bounds = verify_timeout_bounds(
        elapsed_seconds=elapsed, phase=phase, cleanup_completed=cleanup_completed,
        active_seconds_before=active_seconds_before,
        cleanup_reserve_seconds=cleanup_reserve_seconds,
        external_parent_pause_seconds=external_parent_pause_seconds,
    )
    state = {
        "format": "spire-learn-task8-phase-budget-v1",
        **bounds,
    }
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Task8ControlError("phase budget state already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {**state, "path": str(output_path), "sha256": hashlib.sha256(encoded).hexdigest()}


def read_setup_budget_state(path: Path) -> dict[str, object]:
    state = _json_object(path)
    if (state.get("format") != "spire-learn-task8-phase-budget-v1"
            or state.get("phase") != "resolve"
            or type(state.get("active_seconds_charged")) is not int
            or type(state.get("remaining_soft_seconds")) is not int
            or type(state.get("remaining_hard_seconds")) is not int):
        raise Task8ControlError("resolve setup budget state is malformed")
    if state["active_seconds_charged"] + state["remaining_hard_seconds"] != 1200:
        raise Task8ControlError("resolve setup budget hard-time accounting is inconsistent")
    expected_soft_remaining = max(0, 600 - state["active_seconds_charged"])
    if state["remaining_soft_seconds"] != expected_soft_remaining:
        raise Task8ControlError("resolve setup budget soft-time accounting is inconsistent")
    if state["remaining_soft_seconds"] <= 0 or state["remaining_hard_seconds"] <= 0:
        raise Task8ControlError("resolve consumed the setup budget; install phase is blocked")
    return {
        "active_seconds_before": state["active_seconds_charged"],
        "remaining_soft_seconds": state["remaining_soft_seconds"],
        "remaining_hard_seconds": state["remaining_hard_seconds"],
        "state_sha256": sha256_file(path),
    }


def parse_unittest_transcript(
    transcript: str,
    *,
    expected_labels: Sequence[str] = EXPECTED_RNG_LABELS,
    expected_failures: Sequence[str] = (),
) -> dict[str, object]:
    """Require exact labels, and prove the only accepted RED causes explicitly."""
    labels: list[str] = []
    outcomes: dict[str, str] = {}
    ran: int | None = None
    for line in transcript.splitlines():
        match = _TEST_LINE.match(line)
        if match:
            label, outcome = match.groups()
            labels.append(label)
            outcomes[label] = "skipped" if outcome.startswith("skipped ") else outcome
        ran_match = _RAN_LINE.match(line)
        if ran_match:
            ran = int(ran_match.group(1))
    expected = tuple(expected_labels)
    if tuple(labels) != expected or ran != len(expected):
        raise Task8ControlError("unittest verbose labels/count differ from exact reviewed inventory")
    if any(outcome in {"ERROR", "skipped"} for outcome in outcomes.values()):
        raise Task8ControlError("Task 8 gated unittest transcript contains an error or skip")
    failures = tuple(label for label, outcome in outcomes.items() if outcome == "FAIL")
    expected_failure_labels = tuple(expected_failures)
    if failures != expected_failure_labels:
        raise Task8ControlError("unittest failures differ from the exact expected phase outcome")
    if expected_failures:
        if expected_failure_labels != EXPECTED_RNG_LABELS[:2]:
            raise Task8ControlError("expected RED must identify the two reviewed RNG regression tests")
        if "FAILED (failures=" not in transcript:
            raise Task8ControlError("expected RED transcript lacks a failure summary")
        forbidden_markers = (
            "TASK8_RNG_CUDA_UNAVAILABLE=1",
            "TASK8_RNG_CUDA_INITIALIZATION_FAILED=",
            "TASK8_RNG_TORCH_LOAD_FAILED=",
            "TASK8_RNG_PRECONDITION_FAILED=",
            "TASK8_RNG_DEVICE_MISMATCH=",
            "TASK8_RNG_DEVICE_QUERY_FAILED=",
        )
        if any(marker in transcript for marker in forbidden_markers):
            raise Task8ControlError("expected RED transcript reports unavailable or failed CUDA preconditions")
        failure_blocks: dict[str, str] = {}
        current: str | None = None
        body: list[str] = []
        for line in transcript.splitlines():
            if line.startswith("FAIL: "):
                if current is not None:
                    failure_blocks[current] = "\n".join(body)
                current = line[6:]
                body = []
            elif line.startswith(("ERROR: ", "Ran ")):
                if current is not None:
                    failure_blocks[current] = "\n".join(body)
                    current = None
                body = []
            elif current is not None:
                body.append(line)
        if current is not None:
            failure_blocks[current] = "\n".join(body)
        first_cause = (
            "AssertionError: False is not true : CUDA RNG state changed after CPU policy construction"
        )
        first_block = failure_blocks.get(expected_failure_labels[0], "")
        second_block = failure_blocks.get(expected_failure_labels[1], "")
        first_markers = (
            "TASK8_RNG_CUDA_AVAILABLE=1",
            "TASK8_RNG_CUDA_INITIALIZED=1",
            f"TASK8_RNG_APPROVED_GPU_UUID={GPU_UUID}",
            "TASK8_EXPECTED_RED_CUDA_RNG_STATE_CHANGED=1",
            first_cause,
        )
        second_markers = (
            "TASK8_RNG_CUDA_AVAILABLE=1",
            f"TASK8_RNG_APPROVED_GPU_UUID={GPU_UUID}",
            "TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1",
            "TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1",
        )
        if (any(marker not in first_block for marker in first_markers)
                or any(marker not in second_block for marker in second_markers)):
            raise Task8ControlError("expected RED failure details differ from the exact reviewed regression causes")
    elif "OK" not in transcript or "FAILED" in transcript:
        raise Task8ControlError("unittest transcript did not finish with a clean OK summary")
    return {
        "expected_labels": list(expected),
        "observed_labels": labels,
        "ran": ran,
        "failures": list(failures),
        "skipped": 0,
        "status": "expected_red" if expected_failures else "passed",
    }


def _wheel_metadata(path: Path) -> tuple[bytes, dict[str, str], list[str], list[str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [
                name for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise Task8ControlError("wheel must contain one root-level dist-info/METADATA")
            data = archive.read(metadata_paths[0])
            wheel_data = archive.read(metadata_paths[0].removesuffix("METADATA") + "WHEEL")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise Task8ControlError(f"wheel archive is unreadable: {path}") from error
    parsed = Parser().parsestr(data.decode("utf-8"))
    name, version = parsed.get("Name"), parsed.get("Version")
    if not name or not version:
        raise Task8ControlError("wheel METADATA lacks Name or Version")
    wheel_fields = Parser().parsestr(wheel_data.decode("utf-8"))
    tags = wheel_fields.get_all("Tag", [])
    if not tags:
        raise Task8ControlError("wheel WHEEL metadata has no Tag field")
    return data, {"name": name, "version": version}, parsed.get_all("Requires-Dist", []), tags


def _target_compatible_wheel_tags() -> frozenset[str]:
    """Read CPython 3.12/Linux tags from pip only inside the resolver container."""
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise Task8ControlError("wheel tag audit must run in the pinned CPython 3.12 Linux x86_64 resolver")
    try:
        from pip._vendor.packaging.tags import sys_tags
    except ImportError as error:
        raise Task8ControlError("resolver pip lacks its vendored packaging tag provider") from error
    return frozenset(str(tag) for tag in sys_tags())


def _assert_target_wheel_tag(
    filename: str,
    *,
    compatible_tags: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> bool:
    if not filename.endswith(".whl"):
        raise Task8ControlError("source distribution payloads are forbidden")
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        raise Task8ControlError("wheel filename has no valid Python/ABI/platform tags")
    python_tags, abi_tags, platform_tags = parts[-3:]
    candidates = {
        f"{python}-{abi}-{platform}"
        for python, abi, platform in product(
            python_tags.split("."), abi_tags.split("."), platform_tags.split("."),
        )
    }
    supported = frozenset(compatible_tags) if compatible_tags is not None else _target_compatible_wheel_tags()
    if not candidates.intersection(supported):
        raise Task8ControlError(f"wheel {filename} has no pip-compatible CPython 3.12/Linux target tag")
    return True


def audit_cuda_closure(
    manifest_path: Path,
    wheelhouse: Path,
    lock_path: Path,
    *,
    compatible_tags: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    """Verify a resolver-produced complete binary closure, then write hash lock.

    The manifest is produced only after the approved resolver/download command
    exits successfully.  A fresh output lock is created exclusively; this
    function never edits an existing CPU or CUDA lock.
    """
    manifest = _json_object(manifest_path)
    wheelhouse = Path(wheelhouse)
    lock_path = Path(lock_path)
    if lock_path.exists() or lock_path.is_symlink():
        raise Task8ControlError("CUDA lock output already exists; evidence paths are never reused")
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise Task8ControlError("CUDA wheelhouse must be a regular directory")
    if manifest.get("format") != "spire-learn-task8-cuda-closure-v1":
        raise Task8ControlError("CUDA closure manifest format is unsupported")
    resolver = manifest.get("resolver")
    packages = manifest.get("packages")
    if not isinstance(resolver, dict) or resolver.get("exit_code") != 0:
        raise Task8ControlError("binary dependency resolver did not exit successfully")
    if resolver.get("only_binary") is not True or resolver.get("complete_resolution") is not True:
        raise Task8ControlError("resolver must prove a complete binary-only closure")
    if resolver.get("python") != "3.12" or resolver.get("platform") != "linux_x86_64":
        raise Task8ControlError("resolver target must be CPython 3.12 linux x86_64")
    if resolver.get("index_url") != PYTHON_INDEX_URL:
        raise Task8ControlError("resolver index differs from the single approved Python index")
    report_hash = resolver.get("report_sha256")
    command = resolver.get("command")
    if not isinstance(report_hash, str) or _HEX64.fullmatch(report_hash) is None:
        raise Task8ControlError("pip resolver report SHA256 is missing")
    if (not isinstance(command, list) or any(not isinstance(item, str) for item in command)
            or "--dry-run" not in command or "--only-binary=:all:" not in command
            or "--extra-index-url" in command or "--index-url" not in command
            or PYTHON_INDEX_URL not in command):
        raise Task8ControlError("closure manifest lacks the exact binary-only single-index resolver argv")
    if not isinstance(packages, list) or not packages:
        raise Task8ControlError("closure package inventory is empty")
    seen: dict[str, dict[str, Any]] = {}
    lock_rows: list[tuple[str, str, str]] = []
    expected_remote_wheels: set[str] = set()
    target_tags = frozenset(compatible_tags) if compatible_tags is not None else _target_compatible_wheel_tags()
    required_response_fields = {"status", "final_url", "headers"}
    for package in packages:
        if not isinstance(package, dict):
            raise Task8ControlError("closure package row is not an object")
        name = canonical_distribution_name(str(package.get("name", "")))
        version = package.get("version")
        filename = package.get("filename")
        if not isinstance(version, str) or not version or not isinstance(filename, str) or not filename.endswith(".whl"):
            raise Task8ControlError("closure inventory may contain only named wheel payloads")
        _assert_target_wheel_tag(filename, compatible_tags=target_tags)
        filename_tags = filename[:-4].split("-")[-3:]
        expected_tags = {
            f"{python}-{abi}-{platform}"
            for python, abi, platform in product(
                filename_tags[0].split("."), filename_tags[1].split("."),
                filename_tags[2].split("."),
            )
        }
        if name in seen:
            raise Task8ControlError(f"closure contains multiple selected wheels for {name}")
        if Path(filename).name != filename:
            raise Task8ControlError("wheel filename may not contain a path")
        if package.get("source_response", {}).get("kind") == "local-read-only-payload":
            local_payload = package.get("local_payload_path")
            if not isinstance(local_payload, str):
                raise Task8ControlError("local Spire Lab wheel lacks its read-only payload path")
            path = Path(local_payload)
        else:
            path = wheelhouse / filename
        observed = sha256_file(path)
        expected = package.get("expected_sha256")
        recorded_observed = package.get("observed_sha256")
        if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
            raise Task8ControlError(f"wheel {filename} lacks an expected SHA256")
        if observed != expected or recorded_observed != observed:
            raise Task8ControlError(f"wheel {filename} SHA256 does not match source evidence")
        source_url = package.get("source_url")
        response = package.get("source_response")
        if not isinstance(source_url, str) or not isinstance(response, dict):
            raise Task8ControlError(f"wheel {filename} lacks original source/response metadata")
        if response.get("kind") == "http":
            redacted = response.get("redacted_header_names")
            if (response.get("status") != 200 or response.get("method") != "HEAD"
                    or not isinstance(response.get("headers"), list)
                    or not isinstance(redacted, list)
                    or not {"set-cookie", "cookie", "authorization"}.issubset(redacted)):
                raise Task8ControlError(f"wheel {filename} source response is not retained HTTP 200 metadata")
            final_url = response.get("final_url")
            if not isinstance(final_url, str) or urlsplit(final_url).scheme != "https":
                raise Task8ControlError(f"wheel {filename} final source URL must use HTTPS")
            if not isinstance(source_url, str) or urlsplit(source_url).scheme != "https":
                raise Task8ControlError(f"wheel {filename} source URL must use HTTPS")
            expected_remote_wheels.add(filename)
        elif response.get("kind") == "local-read-only-payload":
            if (name != "spire-lab" or response.get("sha256") != expected
                    or source_url != f"file:///inputs/{CPU_WHEEL_FILENAME}"):
                raise Task8ControlError("only the audited local Spire Lab wheel may use local source evidence")
        else:
            raise Task8ControlError(f"wheel {filename} has an unsupported response metadata kind")
        metadata_bytes, wheel_metadata, wheel_requires_dist, wheel_tags = _wheel_metadata(path)
        if not expected_tags.intersection(wheel_tags) or not target_tags.intersection(wheel_tags):
            raise Task8ControlError(f"wheel {filename} WHEEL tags disagree with its filename")
        if canonical_distribution_name(wheel_metadata["name"]) != name or wheel_metadata["version"] != version:
            raise Task8ControlError(f"wheel {filename} METADATA disagrees with the resolver inventory")
        metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
        if package.get("metadata_sha256") != metadata_hash:
            raise Task8ControlError(f"wheel {filename} METADATA digest differs from the closure record")
        requires_dist = package.get("requires_dist")
        if not isinstance(requires_dist, list) or any(not isinstance(item, str) for item in requires_dist):
            raise Task8ControlError(f"wheel {filename} lacks complete Requires-Dist metadata")
        if requires_dist != wheel_requires_dist:
            raise Task8ControlError(f"wheel {filename} Requires-Dist differs from actual METADATA")
        if name == "torch":
            if (filename != CUDA_TORCH_FILENAME or expected != CUDA_TORCH_SHA256
                    or source_url.split("#", 1)[0] != CUDA_TORCH_URL
                    or metadata_hash != CUDA_TORCH_METADATA_SHA256):
                raise Task8ControlError("Torch wheel differs from the exact reviewed CUDA candidate")
            declared_names = {
                canonical_distribution_name(match.group(1))
                for requirement in requires_dist
                if (match := re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement))
            }
            required_torch_names = (
                TORCH_REQUIRED_CUDA_DISTRIBUTIONS | TORCH_REQUIRED_ORDINARY_DISTRIBUTIONS
            )
            if not required_torch_names.issubset(declared_names):
                raise Task8ControlError("Torch wheel METADATA omits declared runtime dependencies")
        if name == "spire-lab" and (
            filename != CPU_WHEEL_FILENAME or expected != CPU_WHEEL_SHA256
        ):
            raise Task8ControlError("Spire Lab wheel differs from the hash-verified accepted CPU build")
        resolver_names = resolver.get("resolved_packages")
        if not isinstance(resolver_names, list) or name not in {
            canonical_distribution_name(str(item)) for item in resolver_names
        }:
            raise Task8ControlError(f"wheel {filename} is absent from the resolver's complete package report")
        seen[name] = package
        lock_rows.append((name, version, observed))
    for name, version in CUDA_DIRECT_PINS.items():
        entry = seen.get(canonical_distribution_name(name))
        if entry is None or entry.get("version") != version:
            raise Task8ControlError(f"CUDA closure does not contain pinned {name}=={version}")
    missing_torch = (TORCH_REQUIRED_CUDA_DISTRIBUTIONS | TORCH_REQUIRED_ORDINARY_DISTRIBUTIONS) - seen.keys()
    if missing_torch:
        raise Task8ControlError("CUDA closure is missing Torch runtime dependencies: " + ", ".join(sorted(missing_torch)))
    resolved = resolver.get("resolved_packages")
    if not isinstance(resolved, list):
        raise Task8ControlError("resolver package report is missing")
    normalized_resolved = {canonical_distribution_name(str(name)) for name in resolved}
    if len(normalized_resolved) != len(resolved):
        raise Task8ControlError("resolver package report contains duplicate distributions")
    if normalized_resolved != seen.keys():
        raise Task8ControlError("wheelhouse package set differs from complete resolver report")
    actual_wheels: set[str] = set()
    for path in wheelhouse.iterdir():
        if path.is_symlink() or not path.is_file():
            raise Task8ControlError("wheelhouse contains a symlink or non-regular entry")
        if not path.name.endswith(".whl"):
            raise Task8ControlError("wheelhouse contains a non-wheel payload")
        actual_wheels.add(path.name)
    allowed_wheels = set(expected_remote_wheels)
    local_copy = wheelhouse / CPU_WHEEL_FILENAME
    if local_copy.exists():
        if local_copy.is_symlink() or sha256_file(local_copy) != CPU_WHEEL_SHA256:
            raise Task8ControlError("wheelhouse Spire Lab copy differs from the read-only accepted wheel")
        allowed_wheels.add(CPU_WHEEL_FILENAME)
    if actual_wheels != allowed_wheels:
        raise Task8ControlError("wheelhouse files do not exactly match the audited closure")
    lock_rows.sort()
    lock_lines = []
    for name, version, digest in lock_rows:
        requirement = (
            f"spire-lab @ file:///inputs/{CPU_WHEEL_FILENAME}"
            if name == "spire-lab" else f"{name}=={version}"
        )
        lock_lines.append(f"{requirement} --hash=sha256:{digest}")
    lock_content = "\n".join(lock_lines) + "\n"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(lock_content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise Task8ControlError("CUDA lock output was created concurrently") from error
    return {
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "package_count": len(lock_rows),
        "direct_pins": dict(CUDA_DIRECT_PINS),
        "torch_runtime_distributions": sorted(
            TORCH_REQUIRED_CUDA_DISTRIBUTIONS | TORCH_REQUIRED_ORDINARY_DISTRIBUTIONS
        ),
        "offline_install_authorized": False,
        "target_compatible_tag_count": len(target_tags),
        "target_compatible_tags_sha256": hashlib.sha256(
            "\n".join(sorted(target_tags)).encode("utf-8")
        ).hexdigest(),
        "target_runtime": "CPython-3.12/Linux-x86_64/pip-vendored-packaging",
    }


def validate_phase_release(
    release_path: Path,
    *,
    phase: str,
    source_hash: str,
    required_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require an explicit one-phase parent release tied to a source snapshot."""
    release = _json_object(release_path)
    if release.get("decision") != "internal_task8_phase_release" or release.get("phase") != phase:
        raise Task8ControlError("matching internal parent phase release is required")
    if release.get("preparation_review") != "passed" or release.get("independent_review") != "passed":
        raise Task8ControlError("Task 8 preparation and independent reviews must both pass")
    if release.get("source_snapshot_sha256") != source_hash:
        raise Task8ControlError("parent release does not match the reviewed source snapshot")
    if release.get("single_use") is not True or not release.get("release_id"):
        raise Task8ControlError("phase release must be unique and single-use")
    for key, expected in (required_fields or {}).items():
        if release.get(key) != expected:
            raise Task8ControlError(f"parent phase release field {key!r} does not match reviewed evidence")
    return release


def write_cuda_resolution_input(path: Path) -> None:
    """Create the exact, single-index resolution request as a new file."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise Task8ControlError("CUDA resolver input already exists; paths are never reused")
    content = (
        f"torch @ {CUDA_TORCH_URL}#sha256={CUDA_TORCH_SHA256}\n"
        "numpy==2.5.4\n"
        "gymnasium==1.4.0\n"
        f"spire_lab @ file:///inputs/{CPU_WHEEL_FILENAME}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise Task8ControlError("CUDA resolver input was created concurrently") from error


def write_resolver_command(path: Path) -> None:
    """Retain the exact argv audited by capture_cuda_closure before execution."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise Task8ControlError("resolver command record already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps({"argv": list(CUDA_RESOLVER_ARGV)}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise Task8ControlError("resolver command record was created concurrently") from error


def _safe_response_headers(headers: object) -> list[list[str]]:
    if not hasattr(headers, "items"):
        raise Task8ControlError("HTTP response headers are unavailable")
    allowed = {
        "cache-control", "content-length", "content-type", "etag", "last-modified",
        "server", "x-cache", "cf-cache-status", "x-amz-version-id",
    }
    return sorted(
        [[str(name), str(value)] for name, value in headers.items() if str(name).lower() in allowed],
        key=lambda item: item[0].lower(),
    )


def capture_cuda_closure(
    report_path: Path,
    wheelhouse: Path,
    output_path: Path,
    *,
    local_spire_wheel: Path,
    resolver_command: Sequence[str],
) -> dict[str, object]:
    """Attach source HEAD metadata and on-disk hashes to a pip dry-run report.

    This function performs bounded HEAD requests only for URLs already selected
    by the reviewed resolver. It downloads no payloads and performs no install.
    """
    report = _json_object(report_path)
    wheelhouse = Path(wheelhouse)
    local_spire_wheel = Path(local_spire_wheel)
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise Task8ControlError("CUDA wheelhouse must be a regular directory")
    if local_spire_wheel.is_symlink() or not local_spire_wheel.is_file():
        raise Task8ControlError("approved Spire Lab read-only wheel is unavailable")
    command = tuple(resolver_command)
    if ("--dry-run" not in command or "--only-binary=:all:" not in command
            or "--no-deps" in command or "--no-binary" in command
            or "--extra-index-url" in command or "--index-url" not in command
            or PYTHON_INDEX_URL not in command):
        raise Task8ControlError("resolver command must be a full binary-only dry run against one approved index")
    install = report.get("install")
    if not isinstance(install, list) or not install:
        raise Task8ControlError("pip resolver report has no selected package closure")
    package_rows: list[dict[str, object]] = []
    names: list[str] = []
    for item in install:
        if not isinstance(item, dict):
            raise Task8ControlError("pip resolver report contains a malformed package entry")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise Task8ControlError("pip resolver report entry lacks metadata or download source")
        name, version = metadata.get("name"), metadata.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise Task8ControlError("pip resolver report entry lacks a package name/version")
        normalized = canonical_distribution_name(name)
        if normalized in names:
            raise Task8ControlError(f"pip resolver selected multiple versions of {normalized}")
        names.append(normalized)
        source_url = download.get("url")
        archive = download.get("archive_info")
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        expected = hashes.get("sha256") if isinstance(hashes, dict) else None
        if not isinstance(source_url, str) or not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
            raise Task8ControlError(f"pip resolver entry for {name} lacks exact URL/SHA256")
        parsed_url = urlsplit(source_url)
        if parsed_url.username or parsed_url.password or parsed_url.query:
            raise Task8ControlError("package source URL contains credentials or query parameters")
        if parsed_url.scheme == "file":
            if normalized != "spire-lab" or Path(unquote(parsed_url.path)).name != CPU_WHEEL_FILENAME:
                raise Task8ControlError("only the approved local Spire Lab wheel may use a file URL")
            path = local_spire_wheel
            filename = CPU_WHEEL_FILENAME
            if sha256_file(path) != expected or expected != CPU_WHEEL_SHA256:
                raise Task8ControlError("local Spire Lab report hash differs from the approved read-only wheel")
            source_response: dict[str, object] = {
                "kind": "local-read-only-payload", "sha256": expected,
            }
        else:
            if parsed_url.scheme != "https":
                raise Task8ControlError("all remote wheel sources must use HTTPS")
            filename = unquote(Path(parsed_url.path).name)
            if not filename.endswith(".whl"):
                raise Task8ControlError("resolver selected a non-wheel source distribution")
            path = wheelhouse / filename
            observed = sha256_file(path)
            if observed != expected:
                raise Task8ControlError(f"downloaded wheel {filename} differs from pip's expected SHA256")
            source_url_no_fragment = source_url.split("#", 1)[0]
            request = urllib.request.Request(source_url_no_fragment, method="HEAD")
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    status = response.getcode()
                    final_url = response.geturl()
                    headers = _safe_response_headers(response.headers)
            except (OSError, urllib.error.URLError) as error:
                raise Task8ControlError(f"source HEAD failed for selected wheel {filename}") from error
            final = urlsplit(final_url)
            if status != 200 or final.scheme != "https" or final.username or final.password or final.query:
                raise Task8ControlError(f"selected wheel {filename} source response is not safe HTTPS 200")
            response_length = _header(headers, "content-length")
            if response_length is not None and int(response_length) != path.stat().st_size:
                raise Task8ControlError(f"selected wheel {filename} response size differs from payload")
            source_response = {
                "kind": "http", "method": "HEAD", "status": status,
                "final_url": final_url, "headers": headers,
                "redacted_header_names": ["set-cookie", "cookie", "authorization"],
            }
        observed = sha256_file(path)
        metadata_bytes, actual, requires_dist, _tags = _wheel_metadata(path)
        if canonical_distribution_name(actual["name"]) != normalized or actual["version"] != version:
            raise Task8ControlError(f"wheel {filename} METADATA differs from pip resolver report")
        row: dict[str, object] = {
            "name": name, "version": version, "filename": filename,
            "source_url": source_url,
            "source_response": source_response,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "requires_dist": requires_dist,
        }
        if parsed_url.scheme == "file":
            row["local_payload_path"] = str(local_spire_wheel)
        package_rows.append(row)
    output = {
        "format": "spire-learn-task8-cuda-closure-v1",
        "resolver": {
            "exit_code": 0,
            "command": list(command),
            "only_binary": True,
            "complete_resolution": True,
            "python": "3.12",
            "platform": "linux_x86_64",
            "index_url": PYTHON_INDEX_URL,
            "report_sha256": sha256_file(report_path),
            "resolved_packages": names,
            "pip_version": report.get("pip_version"),
        },
        "packages": package_rows,
    }
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Task8ControlError("CUDA closure manifest already exists; evidence paths are never reused")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    try:
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise Task8ControlError("CUDA closure manifest was created concurrently") from error
    return {
        "manifest_path": str(output_path),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "package_count": len(package_rows),
        "payload_downloaded_by_capture": False,
        "install_performed": False,
    }


def write_command_record(
    log_root: Path,
    name: str,
    argv: Sequence[str],
    stdout: bytes,
    stderr: bytes,
    exit_code: int,
) -> None:
    """Write closed per-command evidence files under a fresh directory."""
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) is None:
        raise Task8ControlError("command record name is malformed")
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise Task8ControlError("command exit status must be 0..255")
    log_root = Path(log_root)
    if log_root.is_symlink() or not log_root.is_dir():
        raise Task8ControlError("command log root must be a fresh real directory")
    files = {
        f"{name}.command.log": (shlex.join(tuple(argv)) + "\n").encode("utf-8"),
        f"{name}.stdout.log": bytes(stdout),
        f"{name}.stderr.log": bytes(stderr),
        f"{name}.exit.log": f"{exit_code}\n".encode("ascii"),
    }
    for filename, content in files.items():
        path = log_root / filename
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise Task8ControlError(f"command evidence path already exists: {path}") from error
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def _inventory_entries(
    roots: Sequence[Path],
    output_path: Path,
    *,
    allowed_venv_roots: Sequence[Path] = (),
) -> list[dict[str, object]]:
    normalized_roots = []
    for root in roots:
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise Task8ControlError(f"inventory root must be an existing regular directory: {root}")
        resolved = root.resolve(strict=True)
        if root != resolved:
            raise Task8ControlError(f"inventory root must be canonical: {root}")
        normalized_roots.append(resolved)
    if len(set(normalized_roots)) != len(normalized_roots):
        raise Task8ControlError("evidence inventory roots must be unique")
    for left in normalized_roots:
        for right in normalized_roots:
            if left != right and left in right.parents:
                raise Task8ControlError("evidence inventory roots may not overlap")
    venv_roots = _normalize_venv_roots(allowed_venv_roots)
    output_abs = Path(output_path).absolute()
    excluded = {
        output_abs,
        output_abs.parent / "evidence-inventory.command.log",
        output_abs.parent / "evidence-inventory.stdout.log",
        output_abs.parent / "evidence-inventory.stderr.log",
        output_abs.parent / "evidence-inventory.exit.log",
    }
    entries: list[dict[str, object]] = []
    for root in normalized_roots:
        for directory, names, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in tuple(names):
                path = directory_path / name
                if path.is_symlink():
                    target = _known_venv_link(path, venv_roots)
                    info = path.lstat()
                    entries.append({
                        "kind": "symlink", "path": path.relative_to(root).as_posix(),
                        "root": str(root), "mode": oct(info.st_mode & 0o777),
                        "size_bytes": info.st_size, "link_target": target,
                        "link_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    })
                    names.remove(name)
                else:
                    if not path.is_dir():
                        raise Task8ControlError(f"evidence tree contains a non-directory entry: {path}")
                    info = path.stat(follow_symlinks=False)
                    entries.append({
                        "kind": "directory", "path": path.relative_to(root).as_posix(),
                        "root": str(root), "mode": oct(info.st_mode & 0o777),
                    })
            for name in filenames:
                path = directory_path / name
                if path.absolute() in excluded:
                    continue
                if path.is_symlink():
                    target = _known_venv_link(path, venv_roots)
                    info = path.lstat()
                    entries.append({
                        "kind": "symlink", "path": path.relative_to(root).as_posix(),
                        "root": str(root), "mode": oct(info.st_mode & 0o777),
                        "size_bytes": info.st_size, "link_target": target,
                        "link_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    })
                    continue
                if not path.is_file():
                    raise Task8ControlError(f"evidence tree contains a non-regular file: {path}")
                info = path.stat(follow_symlinks=False)
                entries.append({
                    "kind": "file", "path": path.relative_to(root).as_posix(),
                    "root": str(root), "size_bytes": info.st_size,
                    "mode": oct(info.st_mode & 0o777), "sha256": sha256_file(path),
                })
    entries.sort(key=lambda item: (str(item["root"]), str(item["path"]), str(item["kind"])))
    return entries


def inventory_roots(
    roots: Sequence[Path],
    output_path: Path,
    *,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Freeze one phase's own closed output/control/admin roots exclusively."""
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Task8ControlError("final evidence inventory path already exists")
    entries = _inventory_entries(roots, output_path, allowed_venv_roots=allowed_venv_roots)
    root_paths = [str(Path(root).resolve(strict=True)) for root in roots]
    inventory = {
        "format": "spire-learn-task8-evidence-inventory-v2",
        "roots": sorted(root_paths),
        "entries": entries,
    }
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"output_path": str(output_path), "entry_count": len(entries),
            "inventory_sha256": hashlib.sha256(encoded).hexdigest()}


def verify_evidence_inventory(
    inventory_path: Path,
    *,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Prove a prior phase's exact inventoried paths, bytes, modes, and links remain unchanged."""
    inventory_path = Path(inventory_path)
    inventory = _json_object(inventory_path)
    entries = inventory.get("entries")
    roots_value = inventory.get("roots")
    if (inventory.get("format") != "spire-learn-task8-evidence-inventory-v2"
            or not isinstance(entries, list) or not isinstance(roots_value, list)
            or any(not isinstance(item, str) for item in roots_value)):
        raise Task8ControlError("prior evidence inventory format is unsupported")
    roots = [Path(item) for item in roots_value]
    if not roots:
        raise Task8ControlError("prior evidence inventory has no phase-owned roots")
    if len(set(roots)) != len(roots):
        raise Task8ControlError("prior evidence inventory repeats an owned root")
    observed = _inventory_entries(roots, inventory_path, allowed_venv_roots=allowed_venv_roots)
    if observed != entries:
        raise Task8ControlError("prior phase evidence inventory mismatch")
    return {"unchanged": True, "entry_count": len(entries), "inventory_sha256": sha256_file(inventory_path)}


def phase_tree_identity(inventory_path: Path, root: Path) -> dict[str, object]:
    inventory = _json_object(inventory_path)
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise Task8ControlError("phase inventory entries are malformed")
    target = Path(root).resolve(strict=True)
    root_text = str(target)
    selected = []
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("root"), str):
            continue
        inventory_root = Path(item["root"])
        candidate = inventory_root / str(item.get("path", ""))
        if candidate == target or target in candidate.parents:
            selected.append(item)
    if not selected:
        raise Task8ControlError(f"phase inventory does not include the requested root: {root_text}")
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"root": root_text, "sha256": hashlib.sha256(encoded).hexdigest(), "entry_count": len(selected)}


def _phase_input_payload(
    prior_inventories: Mapping[str, Path],
    immutable_paths: Mapping[str, Path],
    venv_path: Path | None,
    venv_inventory_path: Path | None,
    allowed_venv_roots: Sequence[Path],
) -> dict[str, object]:
    prior_rows = []
    for phase, inventory_path in sorted(prior_inventories.items()):
        if phase not in {
            "resolve", "install", "rng-red", "rng-red-corrected-1", "rng-green", "device-check",
        }:
            raise Task8ControlError(f"prior phase name is invalid: {phase}")
        verified = verify_evidence_inventory(inventory_path, allowed_venv_roots=allowed_venv_roots)
        prior_rows.append({
            "phase": phase,
            "inventory_path": str(Path(inventory_path).resolve(strict=True)),
            "inventory_sha256": verified["inventory_sha256"],
            "entry_count": verified["entry_count"],
        })
    immutable_rows = []
    for name, path in sorted(immutable_paths.items()):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise Task8ControlError(f"phase immutable input is missing or unsafe: {name}")
        immutable_rows.append({
            "name": name,
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
        })
    venv_identity = None
    if venv_path is not None or venv_inventory_path is not None:
        if venv_path is None or venv_inventory_path is None:
            raise Task8ControlError("venv identity requires both its path and owning phase inventory")
        venv_identity = {
            "path": str(Path(venv_path).resolve(strict=True)),
            "inventory_path": str(Path(venv_inventory_path).resolve(strict=True)),
            **phase_tree_identity(venv_inventory_path, venv_path),
        }
    return {
        "format": "spire-learn-task8-phase-input-identities-v1",
        "prior_inventories": prior_rows,
        "immutable_paths": immutable_rows,
        "venv_identity": venv_identity,
    }


def write_phase_input_identities(
    output_path: Path,
    *,
    prior_inventories: Mapping[str, Path],
    immutable_paths: Mapping[str, Path],
    venv_path: Path | None = None,
    venv_inventory_path: Path | None = None,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    payload = _phase_input_payload(
        prior_inventories, immutable_paths, venv_path, venv_inventory_path, allowed_venv_roots,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Task8ControlError("phase input identity record already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(output_path), "sha256": hashlib.sha256(encoded).hexdigest(),
            "prior_phase_count": len(payload["prior_inventories"])}


def verify_phase_input_identities(
    identity_path: Path,
    *,
    allowed_venv_roots: Sequence[Path] = (),
) -> dict[str, object]:
    recorded = _json_object(identity_path)
    if recorded.get("format") != "spire-learn-task8-phase-input-identities-v1":
        raise Task8ControlError("phase input identity format is unsupported")
    priors = recorded.get("prior_inventories")
    immutable = recorded.get("immutable_paths")
    venv = recorded.get("venv_identity")
    if not isinstance(priors, list) or not isinstance(immutable, list):
        raise Task8ControlError("phase input identity lists are malformed")
    prior_paths: dict[str, Path] = {}
    for item in priors:
        if not isinstance(item, dict) or not isinstance(item.get("phase"), str) or not isinstance(item.get("inventory_path"), str):
            raise Task8ControlError("prior phase inventory identity is malformed")
        prior_paths[item["phase"]] = Path(item["inventory_path"])
    immutable_paths: dict[str, Path] = {}
    for item in immutable:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("path"), str):
            raise Task8ControlError("immutable phase path identity is malformed")
        immutable_paths[item["name"]] = Path(item["path"])
    if venv is not None:
        if not isinstance(venv, dict) or not isinstance(venv.get("path"), str) or not isinstance(venv.get("inventory_path"), str):
            raise Task8ControlError("installed venv identity is malformed")
        venv_path = Path(venv["path"])
        venv_inventory_path = Path(venv["inventory_path"])
    else:
        venv_path = venv_inventory_path = None
    observed = _phase_input_payload(
        prior_paths, immutable_paths, venv_path, venv_inventory_path, allowed_venv_roots,
    )
    if observed != recorded:
        raise Task8ControlError("prior phase input identities changed after release")
    return {"unchanged": True, "identity_sha256": sha256_file(identity_path),
            "prior_phase_count": len(prior_paths)}


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    metadata = subparsers.add_parser("verify-candidate-metadata")
    metadata.add_argument("record", type=Path)
    metadata.add_argument("metadata", type=Path)
    wheel = subparsers.add_parser("verify-cpu-wheel")
    wheel.add_argument("location", type=Path)
    wheel.add_argument("audit", type=Path)
    image = subparsers.add_parser("verify-image-inspect")
    image.add_argument("inspect", type=Path)
    image_alias = subparsers.add_parser("verify-local-image-alias")
    image_alias.add_argument("inspect", type=Path)
    image_alias.add_argument("--alias", required=True)
    image_alias.add_argument("--expected-image-id", required=True)
    transcript = subparsers.add_parser("verify-rng-transcript")
    transcript.add_argument("transcript", type=Path)
    transcript.add_argument("--expected-failure", action="append", default=[])
    resolution_input = subparsers.add_parser("write-resolution-input")
    resolution_input.add_argument("output", type=Path)
    source_snapshot = subparsers.add_parser("capture-source-snapshot")
    source_snapshot.add_argument("repo_root", type=Path)
    source_snapshot.add_argument("output", type=Path)
    source_snapshot.add_argument("--scope-root", action="append")
    verify_source = subparsers.add_parser("verify-source-snapshot")
    verify_source.add_argument("snapshot", type=Path)
    verify_source.add_argument("repo_root", type=Path)
    safe_inputs = subparsers.add_parser("verify-safe-inputs")
    safe_inputs.add_argument("task_root", type=Path)
    safe_inputs.add_argument("inputs", nargs="+", type=Path)
    corrected_projection = subparsers.add_parser("project-corrected-rng-attempt")
    corrected_projection.add_argument("task_root", type=Path)
    corrected_projection.add_argument("run_id")
    corrected_projection.add_argument("attempt_id")
    previous_rng = subparsers.add_parser("verify-previous-failed-rng-attempt")
    previous_rng.add_argument("task_root", type=Path)
    previous_rng.add_argument("run_id")
    previous_rng.add_argument("--allow-venv", action="append", type=Path, default=[])
    corrected_release = subparsers.add_parser("verify-corrected-rng-red-release")
    corrected_release.add_argument("release", type=Path)
    corrected_release.add_argument("task_root", type=Path)
    corrected_release.add_argument("run_id")
    corrected_release.add_argument("source_hash")
    corrected_release.add_argument("closure_lock_hash")
    corrected_release.add_argument("--allow-venv", action="append", type=Path, default=[])
    corrected_followup = subparsers.add_parser("verify-corrected-rng-followup-release")
    corrected_followup.add_argument("release", type=Path)
    corrected_followup.add_argument("task_root", type=Path)
    corrected_followup.add_argument("phase", choices=("rng-green", "device-check"))
    corrected_followup.add_argument("source_hash")
    corrected_followup.add_argument("closure_lock_hash")
    corrected_followup.add_argument("red_evidence", type=Path)
    corrected_followup.add_argument("--rng-green-transcript-sha256")
    corrected_red = subparsers.add_parser("verify-corrected-rng-red-evidence")
    corrected_red.add_argument("task_root", type=Path)
    corrected_red.add_argument("run_id")
    corrected_red.add_argument("transcript", type=Path)
    corrected_red.add_argument("transcript_sha256")
    corrected_red.add_argument("inventory", type=Path)
    corrected_red.add_argument("inventory_sha256")
    corrected_red.add_argument("previous_inventory", type=Path)
    corrected_red.add_argument("previous_inventory_sha256")
    corrected_red.add_argument("--allow-venv", action="append", type=Path, default=[])
    phase_budget = subparsers.add_parser("write-phase-budget-state")
    phase_budget.add_argument("phase", choices=("resolve", "install", "rng-red", "rng-green", "device-check"))
    phase_budget.add_argument("output", type=Path)
    phase_budget.add_argument("started_epoch_seconds", type=float)
    phase_budget.add_argument("--active-before", type=int, default=0)
    phase_budget.add_argument("--external-parent-pause", type=int, default=0)
    phase_budget.add_argument("--cleanup-reserve", type=int, default=15)
    phase_budget.add_argument("--cleanup-completed", action="store_true")
    setup_budget = subparsers.add_parser("read-setup-budget-state")
    setup_budget.add_argument("state", type=Path)
    phase_inputs = subparsers.add_parser("record-phase-input-identities")
    phase_inputs.add_argument("output", type=Path)
    phase_inputs.add_argument("--prior", action="append", default=[])
    phase_inputs.add_argument("--immutable", action="append", default=[])
    phase_inputs.add_argument("--venv", type=Path)
    phase_inputs.add_argument("--venv-inventory", type=Path)
    phase_inputs.add_argument("--allow-venv", action="append", type=Path, default=[])
    verify_phase_inputs = subparsers.add_parser("verify-phase-input-identities")
    verify_phase_inputs.add_argument("identity", type=Path)
    verify_phase_inputs.add_argument("--allow-venv", action="append", type=Path, default=[])
    resolver_command = subparsers.add_parser("write-resolver-command")
    resolver_command.add_argument("output", type=Path)
    capture = subparsers.add_parser("capture-cuda-closure")
    capture.add_argument("report", type=Path)
    capture.add_argument("wheelhouse", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--local-spire-wheel", type=Path, required=True)
    capture.add_argument("--resolver-command-json", type=Path, required=True)
    audit = subparsers.add_parser("audit-cuda-closure")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("wheelhouse", type=Path)
    audit.add_argument("lock", type=Path)
    release = subparsers.add_parser("verify-phase-release")
    release.add_argument("record", type=Path)
    release.add_argument("--phase", required=True)
    release.add_argument("--source-hash", required=True)
    release.add_argument("--require", action="append", default=[])
    inventory = subparsers.add_parser("verify-gpu-inventory")
    inventory.add_argument("csv", type=Path)
    device_report = subparsers.add_parser("verify-device-report")
    device_report.add_argument("report", type=Path)
    docker_argv = subparsers.add_parser("docker-create-argv")
    docker_argv.add_argument("spec", type=Path)
    name_preflight = subparsers.add_parser("assert-container-name-absent")
    name_preflight.add_argument("exit_status", type=Path)
    name_preflight.add_argument("stdout", type=Path)
    create_assertion = subparsers.add_parser("assert-container-create")
    create_assertion.add_argument("exit_status", type=Path)
    create_assertion.add_argument("stdout", type=Path)
    removal_authorization = subparsers.add_parser("authorize-container-removal")
    removal_authorization.add_argument("container_id")
    removal_authorization.add_argument("create_exit_status", type=Path)
    removal_authorization.add_argument("create_stdout", type=Path)
    removal_authorization.add_argument("cleanup_assert_exit_status", type=Path)
    removal_authorization.add_argument("claim_path", type=Path)
    container = subparsers.add_parser("assert-container-inspect")
    container.add_argument("inspect", type=Path)
    container.add_argument("spec", type=Path)
    inventory_check = subparsers.add_parser("verify-inventory")
    inventory_check.add_argument("inventory", type=Path)
    inventory_check.add_argument("--allow-venv", action="append", type=Path, default=[])
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("output", type=Path)
    inventory.add_argument("roots", nargs="+", type=Path)
    inventory.add_argument("--allow-venv", action="append", type=Path, default=[])
    storage = subparsers.add_parser("verify-storage")
    storage.add_argument("filesystem_root", type=Path)
    storage.add_argument("neural_root", type=Path)
    storage.add_argument("scope", choices=("setup", "workload"))
    storage.add_argument("new_gpu_roots", nargs="+", type=Path)
    storage.add_argument("--reservation-bytes", type=int, default=0)
    storage.add_argument("--reserve-remaining-setup", action="store_true")
    storage.add_argument("--project-install-wheelhouse", type=Path)
    storage.add_argument("--local-wheel", type=Path)
    storage.add_argument("--allow-venv", action="append", type=Path, default=[])
    storage.add_argument("--protected-cpu-venv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        if args.command == "verify-candidate-metadata":
            value = verify_candidate_metadata(args.record, args.metadata)
        elif args.command == "verify-cpu-wheel":
            value = verify_cpu_wheel_payload(args.location, args.audit)
        elif args.command == "verify-image-inspect":
            value = verify_image_inspect(args.inspect.read_bytes())
        elif args.command == "verify-local-image-alias":
            value = verify_local_image_alias(
                args.inspect.read_bytes(), alias=args.alias,
                expected_image_id=args.expected_image_id,
            )
        elif args.command == "verify-rng-transcript":
            value = parse_unittest_transcript(
                args.transcript.read_text(encoding="utf-8"), expected_failures=args.expected_failure,
            )
        elif args.command == "write-resolution-input":
            write_cuda_resolution_input(args.output)
            value = {"path": str(args.output), "sha256": sha256_file(args.output)}
        elif args.command == "capture-source-snapshot":
            value = capture_source_snapshot(
                args.repo_root, args.output,
                scope_roots=tuple(args.scope_root) if args.scope_root else (
                    "src", "tests", "scripts/neural/run-task8-gpu-setup.sh",
                    "scripts/neural/task8-gpu-setup-runbook.md", "scripts/neural/task8_device_check.py",
                    "scripts/neural/task8_gpu_setup_helpers.py",
                ),
            )
        elif args.command == "verify-source-snapshot":
            value = verify_source_snapshot(args.snapshot, args.repo_root)
        elif args.command == "verify-safe-inputs":
            value = {"paths": [str(path) for path in validate_safe_task8_inputs(args.task_root, args.inputs)]}
        elif args.command == "project-corrected-rng-attempt":
            projected = project_corrected_rng_attempt_paths(
                args.task_root, run_id=args.run_id, attempt_id=args.attempt_id,
            )
            value = {key: str(item) if isinstance(item, Path) else item
                     for key, item in projected.items()}
        elif args.command == "verify-previous-failed-rng-attempt":
            value = verify_previous_failed_rng_attempt(
                args.task_root, run_id=args.run_id, allowed_venv_roots=args.allow_venv,
            )
        elif args.command == "verify-corrected-rng-red-release":
            value = validate_corrected_rng_red_release(
                args.release, task_root=args.task_root, run_id=args.run_id,
                source_hash=args.source_hash, closure_lock_hash=args.closure_lock_hash,
                allowed_venv_roots=args.allow_venv,
            )
        elif args.command == "verify-corrected-rng-followup-release":
            value = validate_corrected_rng_followup_release(
                args.release, task_root=args.task_root, phase=args.phase, source_hash=args.source_hash,
                closure_lock_hash=args.closure_lock_hash,
                corrected_red_evidence=_json_object(args.red_evidence),
                rng_green_transcript_sha256=args.rng_green_transcript_sha256,
            )
        elif args.command == "verify-corrected-rng-red-evidence":
            value = verify_corrected_rng_red_evidence(
                args.task_root, run_id=args.run_id,
                transcript_path=args.transcript, transcript_sha256=args.transcript_sha256,
                inventory_path=args.inventory, inventory_sha256=args.inventory_sha256,
                previous_failed_inventory_path=args.previous_inventory,
                previous_failed_inventory_sha256=args.previous_inventory_sha256,
                allowed_venv_roots=args.allow_venv,
            )
        elif args.command == "write-phase-budget-state":
            value = write_phase_budget_state(
                args.output, phase=args.phase, started_epoch_seconds=args.started_epoch_seconds,
                active_seconds_before=args.active_before,
                external_parent_pause_seconds=args.external_parent_pause,
                cleanup_completed=args.cleanup_completed,
                cleanup_reserve_seconds=args.cleanup_reserve,
            )
        elif args.command == "read-setup-budget-state":
            value = read_setup_budget_state(args.state)
        elif args.command == "record-phase-input-identities":
            def named_paths(items: Sequence[str], option: str) -> dict[str, Path]:
                result: dict[str, Path] = {}
                for item in items:
                    if "=" not in item:
                        raise Task8ControlError(f"{option} entries must use NAME=PATH")
                    name, path = item.split("=", 1)
                    if not name or not path or name in result:
                        raise Task8ControlError(f"{option} names must be non-empty and unique")
                    result[name] = Path(path)
                return result
            value = write_phase_input_identities(
                args.output,
                prior_inventories=named_paths(args.prior, "--prior"),
                immutable_paths=named_paths(args.immutable, "--immutable"),
                venv_path=args.venv, venv_inventory_path=args.venv_inventory,
                allowed_venv_roots=args.allow_venv,
            )
        elif args.command == "verify-phase-input-identities":
            value = verify_phase_input_identities(args.identity, allowed_venv_roots=args.allow_venv)
        elif args.command == "write-resolver-command":
            write_resolver_command(args.output)
            value = {"path": str(args.output), "sha256": sha256_file(args.output),
                     "argv": list(CUDA_RESOLVER_ARGV)}
        elif args.command == "capture-cuda-closure":
            command_value = _json_object(args.resolver_command_json)
            command = command_value.get("argv")
            if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
                raise Task8ControlError("resolver command JSON must contain an argv string array")
            value = capture_cuda_closure(
                args.report, args.wheelhouse, args.output,
                local_spire_wheel=args.local_spire_wheel, resolver_command=command,
            )
        elif args.command == "audit-cuda-closure":
            value = audit_cuda_closure(args.manifest, args.wheelhouse, args.lock)
        elif args.command == "verify-phase-release":
            required = {}
            for item in args.require:
                if "=" not in item:
                    raise Task8ControlError("--require values must use KEY=VALUE")
                key, expected = item.split("=", 1)
                if key in {
                    "setup_active_seconds_before", "setup_soft_remaining_seconds",
                    "setup_hard_remaining_seconds", "external_parent_review_pause_seconds",
                }:
                    try:
                        required[key] = int(expected)
                    except ValueError as error:
                        raise Task8ControlError(f"parent release field {key!r} must be an integer") from error
                else:
                    required[key] = expected
            value = validate_phase_release(
                args.record, phase=args.phase, source_hash=args.source_hash,
                required_fields=required,
            )
        elif args.command == "verify-gpu-inventory":
            value = verify_gpu_inventory(args.csv.read_text(encoding="utf-8"))
        elif args.command == "verify-device-report":
            value = verify_device_check_report(_json_object(args.report))
        elif args.command == "assert-container-name-absent":
            assert_exact_container_name_absent(
                _recorded_docker_exit_status(args.exit_status),
                _recorded_docker_stdout(args.stdout),
            )
            value = {"name_absent": True}
        elif args.command == "assert-container-create":
            value = {"container_id": parse_created_container_id(
                _recorded_docker_exit_status(args.exit_status),
                _recorded_docker_stdout(args.stdout),
            )}
        elif args.command == "authorize-container-removal":
            value = authorize_container_removal(
                args.container_id,
                create_exit_status=_recorded_docker_exit_status(args.create_exit_status),
                create_stdout=_recorded_docker_stdout(args.create_stdout),
                cleanup_assert_exit_status=_recorded_docker_exit_status(args.cleanup_assert_exit_status),
                claim_path=args.claim_path,
            )
        elif args.command == "docker-create-argv":
            spec = _json_object(args.spec)
            mounts = spec.get("mounts")
            command = spec.get("command")
            if not isinstance(mounts, list) or not isinstance(command, list):
                raise Task8ControlError("Docker phase spec requires mount and command arrays")
            parsed_mounts = []
            for item in mounts:
                if not isinstance(item, dict) or type(item.get("readonly")) is not bool:
                    raise Task8ControlError("Docker phase mount spec is malformed")
                parsed_mounts.append((
                    Path(str(item.get("source", ""))),
                    str(item.get("destination", "")),
                    item["readonly"],
                ))
            value = {"argv": list(build_docker_create_argv(
                image=str(spec.get("image", "")), run_id=str(spec.get("run_id", "")),
                phase=str(spec.get("phase", "")), command=command, mounts=parsed_mounts,
                network=str(spec.get("network", "")), gpu_uuid=spec.get("gpu_uuid"),
                attempt_id=spec.get("attempt_id"),
            ))}
        elif args.command == "assert-container-inspect":
            spec = _json_object(args.spec)
            mounts = spec.get("mounts")
            command = spec.get("command")
            if not isinstance(mounts, list) or not isinstance(command, list):
                raise Task8ControlError("Docker phase spec requires mount and command arrays")
            parsed_mounts = []
            for item in mounts:
                if not isinstance(item, dict) or type(item.get("readonly")) is not bool:
                    raise Task8ControlError("Docker phase mount spec is malformed")
                parsed_mounts.append((
                    Path(str(item.get("source", ""))),
                    str(item.get("destination", "")),
                    item["readonly"],
                ))
            value = assert_container_inspect(
                args.inspect.read_bytes(), expected_image_id=str(spec.get("expected_image_id", "")),
                run_id=str(spec.get("run_id", "")), phase=str(spec.get("phase", "")),
                expected_command=command, expected_mounts=parsed_mounts,
                network=str(spec.get("network", "")), gpu_uuid=spec.get("gpu_uuid"),
                attempt_id=spec.get("attempt_id"),
            )
        elif args.command == "inventory":
            value = inventory_roots(args.roots, args.output, allowed_venv_roots=args.allow_venv)
        elif args.command == "verify-inventory":
            value = verify_evidence_inventory(args.inventory, allowed_venv_roots=args.allow_venv)
        elif args.command == "verify-storage":
            reservation_bytes = args.reservation_bytes
            projection = None
            if args.project_install_wheelhouse is not None:
                if args.local_wheel is None:
                    raise Task8ControlError("install projection requires --local-wheel")
                projection = project_offline_install_bytes(args.project_install_wheelhouse, args.local_wheel)
                reservation_bytes = projection["projected_additional_bytes"]
            value = check_storage_paths(
                args.filesystem_root, args.neural_root, args.new_gpu_roots,
                setup=args.scope == "setup", reservation_bytes=reservation_bytes,
                reserve_remaining_setup=args.reserve_remaining_setup,
                allowed_venv_roots=args.allow_venv,
                protected_cpu_venv_root=args.protected_cpu_venv,
            )
            if projection is not None:
                value["offline_install_projection"] = projection
        else:
            raise AssertionError(args.command)
    except (Task8ControlError, OSError) as error:
        print(f"TASK8_CONTROL_ERROR={error}", file=sys.stderr)
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

