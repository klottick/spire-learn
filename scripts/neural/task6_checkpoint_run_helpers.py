"""Stdlib-only controls for the Task 6 checkpoint CPU test wrapper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Sequence


EXPECTED_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
EXPECTED_GYM_SOURCE_MAP_SHA256 = "dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3"
EXPECTED_CPU_LOCK_SHA256 = "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0"
EXPECTED_NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
CPU_RUNTIME_RELATIVE = Path("artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc")
MIN_FREE_BYTES = 30 * 1024**3
IMAGE = "docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
_PINNED_IMAGE_REPO_DIGESTS = frozenset({
    IMAGE,
    "python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0",
})
CONTAINER_TIMEOUT_SECONDS = 120
CONTROL_FILES = (
    "scripts/neural/run-task6-checkpoint-tests.sh",
    "scripts/neural/task6_checkpoint_run_helpers.py",
    "scripts/neural/task6-checkpoint-test-runbook.md",
)
EXPECTED_CHECKPOINT_TEST_METHODS = (
    "test_evaluation_load_is_frozen_and_omits_optimizer_and_rng",
    "test_resume_round_trips_named_weights_optimizer_rng_and_counters",
    "test_transfer_allows_mode_and_seed_change_but_rejects_encoder_change",
    "test_warm_start_transfers_weights_but_drops_optimizer_rng_and_counters",
)
_EXPECTED_RUNTIME_PACKAGES = {
    "spire-lab": "0.1.0",
    "gymnasium": "1.4.0",
    "numpy": "2.5.4",
    "torch": "2.14.1+cpu",
}
_FINAL_WRITER_FILES = (
    "evidence-inventory.json",
    "evidence-inventory.command.log",
    "evidence-inventory.stdout.log",
    "evidence-inventory.stderr.log",
    "evidence-inventory.exit.log",
)
_TEST_RESULT_RE = re.compile(
    r"^(?P<label>.+ \([^)]+\)) \.\.\. (?P<result>ok|FAIL|ERROR|skipped(?: \(.*\))?)$"
)
_COMMAND_RE = re.compile(
    r"^CONTAINER_COMMAND=\$\(cat <<'COMMAND'\n(?P<body>.*?)\nCOMMAND\n\)$",
    re.MULTILINE | re.DOTALL,
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,57}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def control_hashes(root: Path) -> dict[str, dict[str, object]]:
    root = Path(root).resolve()
    result: dict[str, dict[str, object]] = {}
    for relative in CONTROL_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Task 6 control file is missing or unsafe: {relative}")
        result[relative] = {"sha256": _sha256(path), "mode": stat.S_IMODE(path.stat().st_mode)}
    return result


def mounted_inputs(root: Path) -> dict[str, dict[str, object]]:
    root = Path(root).resolve()
    result: dict[str, dict[str, object]] = {}
    for top in ("src", "tests"):
        base = root / top
        if not base.is_dir():
            raise ValueError(f"Task 6 read-only input directory is missing: {top}")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are not allowed in mounted inputs: {path.relative_to(root)}")
            if path.is_file():
                result[path.relative_to(root).as_posix()] = {
                    "sha256": _sha256(path),
                    "mode": stat.S_IMODE(path.stat().st_mode),
                }
    return result


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def runtime_identity(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    location_path = root / "artifacts/development/neural-combat/cpu-runtime-location.json"
    location = json.loads(location_path.read_text(encoding="utf-8"))
    runtime_root = (root / CPU_RUNTIME_RELATIVE).resolve()
    if Path(location.get("runtime_root", "")).resolve() != runtime_root:
        raise ValueError("Task 6 must use the retained qualified CPU runtime location")
    durable_source = Path(location.get("durable_source_root", ""))
    if not str(durable_source).endswith(f"gym66-{EXPECTED_GYM_REVISION}/source"):
        raise ValueError("retained runtime no longer names the accepted Gym source revision")
    venv = runtime_root / "data/venv-cpu"
    lock = runtime_root / "data/requirements-cpu.lock"
    native = venv / "lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
    reconciliation_path = runtime_root / "cpu-source-reconciliation.json"
    for path in (venv, lock, native, reconciliation_path):
        if not path.exists():
            raise ValueError(f"retained CPU runtime input is missing: {path.name}")
    if venv.is_symlink() or lock.is_symlink() or native.is_symlink():
        raise ValueError("retained CPU runtime inputs must not be symlinks")
    lock_hash, native_hash = _sha256(lock), _sha256(native)
    if lock_hash != EXPECTED_CPU_LOCK_SHA256 or native_hash != EXPECTED_NATIVE_SHA256:
        raise ValueError("retained CPU lock or native extension hash changed")
    python_link = venv / "bin/python"
    if not python_link.is_symlink() or os.readlink(python_link) != "/usr/local/bin/python":
        raise ValueError("qualified CPU virtualenv Python link changed")
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if (reconciliation.get("source_map_sha256") != EXPECTED_GYM_SOURCE_MAP_SHA256
            or reconciliation.get("verified_files") != 636
            or reconciliation.get("normalized_git_modes_match") is not True
            or reconciliation.get("original_source_posthash_available") is not False):
        raise ValueError("retained accepted Gym source reconciliation changed")
    report_path = root / "artifacts/development/neural-combat/cpu-runtime-report.md"
    return {
        "runtime_root": str(runtime_root),
        "venv_path": str(venv),
        "venv_python_link": os.readlink(python_link),
        "cpu_lock_sha256": lock_hash,
        "native_extension_sha256": native_hash,
        "gym_revision": EXPECTED_GYM_REVISION,
        "gym_source_map_sha256": EXPECTED_GYM_SOURCE_MAP_SHA256,
        "gym_original_source_posthash_available": False,
        "gym_durable_verified_files": 636,
        "cpu_setup_report_sha256": _sha256(report_path),
        "docker_executable": shutil.which("docker"),
    }


def require_free_space(path: Path, minimum_bytes: int = MIN_FREE_BYTES) -> int:
    if type(minimum_bytes) is not int or minimum_bytes <= 0:
        raise ValueError("minimum free-space threshold must be a positive integer")
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < minimum_bytes:
        raise ValueError(f"Task 6 requires at least 30 GiB free; found {free_bytes} bytes")
    return free_bytes


def source_before(root: Path, run_root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    run_root = Path(run_root).resolve()
    expected_parent = (root / "artifacts/development/neural-combat/task-6/runtime").resolve()
    if run_root.parent != expected_parent or not run_root.is_dir():
        raise ValueError("Task 6 output must be a fresh child of its runtime directory")
    free_bytes = require_free_space(run_root)
    inputs = mounted_inputs(root)
    controls = control_hashes(root)
    record: dict[str, object] = {
        "git_head": _git(root, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": _git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        "mounted_input_sha256": inputs,
        "mounted_input_tree_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "control_sha256": controls,
        "runtime_identity": runtime_identity(root),
        "free_bytes_at_preflight": free_bytes,
        "required_free_bytes": MIN_FREE_BYTES,
        "container_writable_bind_paths": ["/run"],
        "container_ephemeral_tmpfs": {"/tmp": "64 MiB, noexec, nosuid"},
        "container_read_only_paths": ["/learner/src", "/learner/tests", "/work/venv-cpu", "/work/requirements-cpu.lock"],
        "sampled_rows_reserved": 64,
        "native_resets": 0,
        "native_steps": 0,
    }
    _write_json(run_root / "source-preflight.json", record)
    return record


def source_after(root: Path, run_root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    run_root = Path(run_root).resolve()
    before = json.loads((run_root / "source-preflight.json").read_text(encoding="utf-8"))
    inputs = mounted_inputs(root)
    controls = control_hashes(root)
    runtime = runtime_identity(root)
    input_tree = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if inputs != before["mounted_input_sha256"] or input_tree != before["mounted_input_tree_sha256"]:
        raise ValueError("Task 6 learner source/tests changed during the protected test")
    if controls != before["control_sha256"]:
        raise ValueError("Task 6 runner/helper/runbook changed during the protected test")
    if runtime != before["runtime_identity"]:
        raise ValueError("retained CPU runtime or accepted Gym provenance changed")
    result = {
        "mounted_input_tree_sha256_before": before["mounted_input_tree_sha256"],
        "mounted_input_tree_sha256_after": input_tree,
        "control_sha256_after": controls,
        "runtime_identity_after": runtime,
        "unchanged": True,
    }
    _write_json(run_root / "source-postcheck.json", result)
    return result


def extract_container_command(script: Path) -> str:
    text = Path(script).read_text(encoding="utf-8")
    match = _COMMAND_RE.search(text)
    if match is None:
        raise ValueError("Task 6 runner has no literal container command")
    return match.group("body")


def parse_container_command(script: Path) -> dict[str, object]:
    body = extract_container_command(script)
    result = subprocess.run(["/bin/sh", "-n", "-c", body], capture_output=True, text=True)
    return {"parsed": result.returncode == 0, "exit_code": result.returncode, "stderr": result.stderr, "command": body}


def expected_checkpoint_test_ids() -> tuple[str, ...]:
    """Derive the exact stdlib verbose labels from the authored gated class."""
    test_path = Path(__file__).resolve().parents[2] / "tests/neural/test_checkpoint.py"
    module = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    classes = [
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NeuralCheckpointTensorTests"
    ]
    if len(classes) != 1:
        raise ValueError("gated checkpoint test class is missing or ambiguous")
    methods = tuple(sorted(
        node.name for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ))
    if methods != tuple(sorted(EXPECTED_CHECKPOINT_TEST_METHODS)):
        raise ValueError("gated checkpoint test method inventory differs from its four-test contract")
    module_name = "tests.neural.test_checkpoint"
    class_name = "NeuralCheckpointTensorTests"
    return tuple(f"{name} ({module_name}.{class_name}.{name})" for name in methods)


def _runtime_marker(transcript: str) -> dict[str, object] | None:
    marker_lines = [
        line for line in transcript.splitlines()
        if line.startswith("TASK6_RUNTIME_IDENTITY=")
    ]
    if len(marker_lines) != 1:
        return None
    try:
        value = json.loads(marker_lines[0].split("=", 1)[1])
    except (json.JSONDecodeError, TypeError):
        return None
    expected = {
        "packages": _EXPECTED_RUNTIME_PACKAGES,
        "lock_sha256": EXPECTED_CPU_LOCK_SHA256,
        "native_sha256": EXPECTED_NATIVE_SHA256,
    }
    return value if value == expected else None


def assess_checkpoint_test_run(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    expected_outcome: str,
) -> dict[str, object]:
    """Classify only exact four-test GREEN or explicit product/behavior RED."""
    if expected_outcome not in {"red", "green"}:
        raise ValueError("expected outcome must be red or green")
    if type(exit_code) is not int:
        raise TypeError("checkpoint child exit code must be an integer")
    transcript = f"{stdout}\n{stderr}"
    runtime = _runtime_marker(transcript)
    expected_ids = set(expected_checkpoint_test_ids())
    result_lines: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        match = _TEST_RESULT_RE.fullmatch(line)
        if match:
            result_lines.append((match.group("label"), match.group("result")))
    labels = [label for label, _status in result_lines]
    statuses = [status for _label, status in result_lines]
    method_lines = [
        (label, status) for label, status in result_lines
        if not label.startswith("setUpClass (")
    ]
    setup_lines = [
        (label, status) for label, status in result_lines
        if label.startswith("setUpClass (")
    ]
    ran_values = re.findall(r"^Ran\s+(\d+)\s+tests?\s+in\s+[0-9.]+s$", transcript, re.MULTILINE)
    summary_lines = re.findall(r"^(?:OK(?: \(skipped=\d+\))?|FAILED \(.*\))$", transcript, re.MULTILINE)
    tests_ran = int(ran_values[0]) if len(ran_values) == 1 else None
    failed_count = statuses.count("FAIL")
    error_count = statuses.count("ERROR")
    skipped_count = sum(status.startswith("skipped") for status in statuses)
    setup_error_count = sum(status == "ERROR" for _label, status in setup_lines)
    classification = "unclassified"
    reasons: list[str] = []
    if runtime is None:
        reasons.append("runtime identity marker is missing, malformed, duplicated, or does not match the pin")
    if tests_ran is None or len(summary_lines) != 1:
        reasons.append("unittest transcript has no unique Ran count and terminal summary")

    api_marker_lines = [
        line for line in transcript.splitlines()
        if "TASK6_CHECKPOINT_API_MISSING" in line
    ]
    exact_api_marker = (
        len(api_marker_lines) == 1
        and api_marker_lines[0] == (
            "RuntimeError: TASK6_CHECKPOINT_API_MISSING "
            "module=spire_learn.neural.checkpoint names=load_checkpoint,save_checkpoint"
        )
    )
    expected_setup_label = (
        "setUpClass (tests.neural.test_checkpoint.NeuralCheckpointTensorTests)"
    )
    if (
        runtime is not None and exact_api_marker and exit_code != 0
        and tests_ran == 0 and labels == [expected_setup_label]
        and setup_lines == [(expected_setup_label, "ERROR")]
        and not method_lines and error_count == 1 and failed_count == 0 and skipped_count == 0
        and summary_lines == ["FAILED (errors=1)"]
    ):
        classification = "product_api_missing"
    elif (
        runtime is not None and not api_marker_lines and exit_code != 0
        and tests_ran == len(expected_ids) and not setup_lines
        and len(method_lines) == len(expected_ids)
        and set(labels) == expected_ids and len(set(labels)) == len(expected_ids)
        and all(status in {"ok", "FAIL"} for _label, status in method_lines)
        and failed_count > 0 and error_count == 0 and skipped_count == 0
        and summary_lines == [f"FAILED (failures={failed_count})"]
    ):
        classification = "behavioral_red"
    elif (
        runtime is not None and exit_code == 0 and tests_ran == len(expected_ids)
        and not setup_lines and len(method_lines) == len(expected_ids)
        and set(labels) == expected_ids and len(set(labels)) == len(expected_ids)
        and statuses == ["ok"] * len(expected_ids)
        and failed_count == 0 and error_count == 0 and skipped_count == 0
        and summary_lines == ["OK"]
    ):
        classification = "green"
    else:
        if labels and (set(labels) != expected_ids and labels != [expected_setup_label]):
            reasons.append("unittest labels do not equal the four expected checkpoint tests")
        if skipped_count:
            reasons.append("checkpoint suite skipped one or more tests")
        if error_count or setup_error_count:
            reasons.append("checkpoint suite has an unexpected test or setup error")
        if tests_ran == 0 and not exact_api_marker:
            reasons.append("zero-test setup failure is not the exact missing checkpoint API condition")
        if expected_outcome == "green" and exit_code != 0:
            reasons.append("GREEN expected child exit code 0")
        if expected_outcome == "red" and exit_code == 0:
            reasons.append("RED expected a nonzero child exit code")

    accepted = (expected_outcome == "green" and classification == "green") or (
        expected_outcome == "red" and classification in {"product_api_missing", "behavioral_red"}
    )
    return {
        "classification": classification,
        "accepted": accepted,
        "expected_outcome": expected_outcome,
        "child_exit_code": exit_code,
        "tests_ran": tests_ran,
        "failed_tests": failed_count,
        "errored_tests": error_count,
        "skipped_tests": skipped_count,
        "expected_test_ids": sorted(expected_ids),
        "observed_test_ids": labels,
        "runtime_identity": runtime,
        "rejection_reasons": reasons,
    }


def assert_container_inspect(
    inspect_value: object, *, run_id: str, source_root: Path, tests_root: Path,
    venv: Path, lock: Path, writable: Path, command: str,
    container_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("inspect check received an unsafe run ID")
    if isinstance(inspect_value, list):
        if len(inspect_value) != 1:
            raise ValueError("inspect output must contain exactly one created container")
        inspect_value = inspect_value[0]
    if not isinstance(inspect_value, dict):
        raise ValueError("inspect output must be a container object")
    if container_id is not None:
        if (re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None
                or inspect_value.get("Id") != container_id):
            raise ValueError("inspected container ID does not match the selected Task 6 object")
    config = inspect_value.get("Config")
    host = inspect_value.get("HostConfig")
    mounts = inspect_value.get("Mounts")
    if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(mounts, list):
        raise ValueError("inspect output is missing Config, HostConfig, or Mounts")
    if config.get("Image") != IMAGE or config.get("User") != "1000:1000":
        raise ValueError("created container image or non-root user differs from the reviewed contract")
    if inspect_value.get("Name") != f"/spirelearn-task6-{run_id}":
        raise ValueError("created container name does not match its unique Task 6 run ID")
    if config.get("Entrypoint") != ["/bin/sh"] or config.get("Cmd") != ["-ec", command]:
        raise ValueError("created container command differs from the literal reviewed script")
    if config.get("WorkingDir") != "/learner" or host.get("Privileged") is not False:
        raise ValueError("container workdir or privilege mode differs from the reviewed controls")
    if host.get("ReadonlyRootfs") is not True or host.get("NetworkMode") != "none":
        raise ValueError("container root filesystem or network mode is not protected")
    if host.get("Memory") != 8 * 1024**3 or host.get("NanoCpus") != 4_000_000_000 or host.get("PidsLimit") != 512:
        raise ValueError("container CPU, memory, or PID budget differs from the reviewed cap")
    security = host.get("SecurityOpt") or []
    if host.get("CapDrop") != ["ALL"] or not any(
        option == "no-new-privileges" or option.startswith("no-new-privileges:")
        for option in security
    ):
        raise ValueError("container capability or privilege controls are incomplete")
    if host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=64m"}:
        raise ValueError("container temporary filesystem settings differ from the reviewed controls")
    if host.get("DeviceRequests") or host.get("Devices"):
        raise ValueError("Task 6 CPU checkpoint tests may not receive a device")
    if config.get("User") == "0" or host.get("Privileged") is not False:
        raise ValueError("Task 6 container must run non-root without privileged mode")
    environment = set(config.get("Env") or [])
    required_environment = {
        "PYTHONPATH=/learner/src", "PYTHONDONTWRITEBYTECODE=1", "PIP_NO_INDEX=1",
        "PIP_NO_CACHE_DIR=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", "HOME=/run/home",
        "TMPDIR=/run/tmp", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
        "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
        "SPIRE_LEARN_RUN_NEURAL_TENSOR=1", "SPIRE_LEARN_TENSOR_TASK=task6",
        f"SPIRE_LEARN_TENSOR_RESERVATION_ID=task6-{run_id}",
        "SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64",
        "SPIRE_LEARN_TENSOR_EVIDENCE_PATH=/run/tensor-sampling.json",
    }
    if not required_environment.issubset(environment):
        raise ValueError("container environment is missing a Task 6 offline/budget control")
    labels = config.get("Labels") or {}
    if labels.get("spire-learn.phase") != "task6-checkpoint-cpu-tests" or labels.get("spire-learn.owner") != run_id:
        raise ValueError("container ownership labels do not match its Task 6 run ID")
    expected_mounts = {
        "/learner/src": (Path(source_root).resolve(), False),
        "/learner/tests": (Path(tests_root).resolve(), False),
        "/work/venv-cpu": (Path(venv).resolve(), False),
        "/work/requirements-cpu.lock": (Path(lock).resolve(), False),
        "/run": (Path(writable).resolve(), True),
    }
    observed: dict[str, tuple[Path, bool]] = {}
    for item in mounts:
        if not isinstance(item, dict) or not isinstance(item.get("Destination"), str):
            raise ValueError("container has an invalid mount entry")
        if item.get("Type") != "bind":
            raise ValueError("Task 6 mounts must be explicit bind mounts")
        destination = item["Destination"]
        observed[destination] = (Path(item.get("Source", "")).resolve(), item.get("RW") is True)
    if observed != expected_mounts:
        raise ValueError("container mount table differs from the exact Task 6 allowlist")
    return {
        "name": inspect_value.get("Name"),
        "image": config["Image"],
        "user": config["User"],
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "mounts": {name: {"source": str(source), "read_write": writable_flag}
                   for name, (source, writable_flag) in observed.items()},
        "cpu_nanos": host["NanoCpus"],
        "memory_bytes": host["Memory"],
        "pids_limit": host["PidsLimit"],
        "network": host["NetworkMode"],
        "root_read_only": host["ReadonlyRootfs"],
        "container_id": inspect_value.get("Id"),
    }


def assert_local_image_identity(inspect_value: object) -> dict[str, object]:
    """Require the retained exact Linux/amd64 digest to already be local."""
    if isinstance(inspect_value, list):
        if len(inspect_value) != 1:
            raise ValueError("local pinned image inspection must return exactly one image")
        inspect_value = inspect_value[0]
    if not isinstance(inspect_value, dict):
        raise ValueError("local pinned image inspection is not an object")
    image_id = inspect_value.get("Id")
    repo_digests = inspect_value.get("RepoDigests")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("local pinned image has an invalid immutable image ID")
    if (not isinstance(repo_digests, list)
            or any(not isinstance(value, str) for value in repo_digests)
            or not _PINNED_IMAGE_REPO_DIGESTS.intersection(repo_digests)):
        raise ValueError("exact pinned image digest is not present in the local image identity")
    if inspect_value.get("Os") != "linux" or inspect_value.get("Architecture") != "amd64":
        raise ValueError("local pinned image platform is not linux/amd64")
    return {
        "image": IMAGE,
        "image_id": image_id,
        "os": "linux",
        "architecture": "amd64",
        "repo_digests": sorted(repo_digests),
    }


def resolve_container_id(candidate_output: str) -> str | None:
    """Select only a single syntactically valid ID from an exact-name query."""
    if not isinstance(candidate_output, str):
        raise TypeError("container ID query output must be text")
    candidates = [line for line in candidate_output.splitlines() if line]
    if not candidates:
        return None
    if len(candidates) != 1 or re.fullmatch(r"[0-9a-f]{12,64}", candidates[0]) is None:
        raise ValueError("exact-name container lookup must return zero or one valid container ID")
    return candidates[0]


def cleanup_status(run_root: Path, *, container_create_attempted: bool,
                   container_created: bool, container_started: bool,
                   cleanup_failed: bool) -> dict[str, object]:
    result = {
        "schema_id": "spire-learn-task6-cleanup-v1",
        "container_create_attempted": container_create_attempted,
        "container_created": container_created,
        "container_started": container_started,
        "cleanup_failed": cleanup_failed,
    }
    _write_json(Path(run_root) / "cleanup-status.json", result)
    return result


def run_summary(run_root: Path, *, expected_outcome: str,
                container_created: bool, container_started: bool,
                start_status: int, cleanup_failed: bool) -> dict[str, object]:
    if expected_outcome not in {"red", "green"}:
        raise ValueError("expected outcome must be red or green")
    if type(start_status) is not int:
        raise TypeError("container start status must be an integer")
    stderr_path = Path(run_root) / "logs/container-start.stderr.log"
    stdout_path = Path(run_root) / "logs/container-start.stdout.log"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    exit_path = Path(run_root) / "logs/container-start.exit.log"
    recorded_status: int | None = None
    if exit_path.is_file():
        try:
            recorded_status = int(exit_path.read_text(encoding="utf-8").strip())
        except ValueError:
            recorded_status = None
    assessment = assess_checkpoint_test_run(
        stdout, stderr, start_status, expected_outcome=expected_outcome
    )
    accepted = (
        assessment["accepted"] is True
        and container_created is True
        and container_started is True
        and cleanup_failed is False
        and recorded_status == start_status
    )
    rejection_reasons = list(assessment["rejection_reasons"])
    if not container_created:
        rejection_reasons.append("container was not created")
    if not container_started:
        rejection_reasons.append("container was not started")
    if cleanup_failed:
        rejection_reasons.append("container cleanup or evidence finalization failed")
    if recorded_status != start_status:
        rejection_reasons.append("container start exit record differs from the exact child status")
    result = {
        "schema_id": "spire-learn-task6-checkpoint-run-v1",
        "expected_outcome": expected_outcome,
        "container_created": container_created,
        "container_started": container_started,
        "container_start_exit": start_status,
        "container_start_exit_recorded": recorded_status,
        "tests_ran": assessment["tests_ran"],
        "failed_tests": assessment["failed_tests"],
        "errored_tests": assessment["errored_tests"],
        "skipped_tests": assessment["skipped_tests"],
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest() if stdout_path.is_file() else None,
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest() if stderr_path.is_file() else None,
        "cleanup_failed": cleanup_failed,
        "classification": assessment["classification"],
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "expected_test_ids": assessment["expected_test_ids"],
        "observed_test_ids": assessment["observed_test_ids"],
        "runtime_identity": assessment["runtime_identity"],
    }
    _write_json(Path(run_root) / "run-summary.json", result)
    return result


def _resolve_evidence_root(path: Path, label: str) -> Path:
    original = Path(path)
    if original.is_symlink() or not original.is_dir():
        raise ValueError(f"evidence {label} root must be a regular directory")
    return original.resolve()


def _evidence_files(run_root: Path, control_root: Path) -> dict[str, dict[str, object]]:
    roots = (("run", run_root), ("control", control_root))
    files: dict[str, dict[str, object]] = {}
    resolved_roots: list[Path] = []
    for prefix, original_root in roots:
        root = _resolve_evidence_root(original_root, prefix)
        if any(root == other or root in other.parents or other in root.parents for other in resolved_roots):
            raise ValueError("evidence run and control roots must be disjoint")
        resolved_roots.append(root)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"evidence inventory may not include symlinks: {prefix}/{relative}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"evidence inventory contains a non-regular file: {prefix}/{relative}")
            files[f"{prefix}/{relative}"] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return files


def close_inventory(
    run_root: Path,
    control_root: Path,
    output_path: Path,
    *,
    admin_root: Path,
) -> dict[str, object]:
    run_root = _resolve_evidence_root(run_root, "run")
    control_root = _resolve_evidence_root(control_root, "control")
    admin_path = Path(admin_root)
    if admin_path.is_symlink() or not admin_path.is_dir():
        raise ValueError("final inventory admin output must be a regular directory")
    admin_root = admin_path.resolve()
    output_path = Path(output_path)
    if output_path.parent.resolve() != admin_root or output_path.name != "evidence-inventory.json":
        raise ValueError("final inventory output must be evidence-inventory.json in the admin root")
    if any(admin_root == root or admin_root in root.parents or root in admin_root.parents
           for root in (run_root, control_root)):
        raise ValueError("final inventory admin output must be disjoint from evidence roots")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("final evidence inventory is never overwritten")
    allowed_admin = set(_FINAL_WRITER_FILES) - {"evidence-inventory.json"}
    for child in admin_root.iterdir():
        if child.name not in allowed_admin or child.is_symlink() or not child.is_file():
            raise ValueError("admin output may contain only final writer bookkeeping files")
    record = {
        "schema_id": "spire-learn-task6-evidence-inventory-v1",
        "roots": {"run": str(run_root), "control": str(control_root)},
        "files": _evidence_files(run_root, control_root),
        "excluded_final_writer": {
            "root": str(admin_root),
            "reason": "The final inventory command writes its manifest and invocation logs here; hashing those outputs would make the manifest self-changing.",
            "bookkeeping_files": list(_FINAL_WRITER_FILES),
        },
    }
    _write_json(output_path, record)
    return record


def verify_inventory(
    record: object,
    run_root: Path,
    control_root: Path,
    admin_root: Path,
) -> bool:
    if not isinstance(record, dict) or record.get("schema_id") != "spire-learn-task6-evidence-inventory-v1":
        raise ValueError("evidence inventory record has an invalid schema")
    run_root = _resolve_evidence_root(run_root, "run")
    control_root = _resolve_evidence_root(control_root, "control")
    admin_path = Path(admin_root)
    if admin_path.is_symlink() or not admin_path.is_dir():
        raise ValueError("final inventory admin output must be a regular directory")
    admin_root = admin_path.resolve()
    if record.get("roots") != {"run": str(run_root), "control": str(control_root)}:
        raise ValueError("evidence inventory roots do not match the current run")
    exclusion = record.get("excluded_final_writer")
    if exclusion != {
        "root": str(admin_root),
        "reason": "The final inventory command writes its manifest and invocation logs here; hashing those outputs would make the manifest self-changing.",
        "bookkeeping_files": list(_FINAL_WRITER_FILES),
    }:
        raise ValueError("evidence inventory has an undocumented final writer exclusion")
    if set(child.name for child in admin_root.iterdir()) != set(_FINAL_WRITER_FILES):
        raise ValueError("admin output contains files outside the final writer bookkeeping exclusion")
    if any(child.is_symlink() or not child.is_file() for child in admin_root.iterdir()):
        raise ValueError("admin output contains unsafe final writer bookkeeping")
    actual_files = _evidence_files(run_root, control_root)
    if record.get("files") != actual_files:
        raise ValueError("evidence inventory mismatch: run or control files changed")
    manifest_path = admin_root / "evidence-inventory.json"
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("final evidence inventory output is missing or invalid") from error
    if stored != record:
        raise ValueError("final evidence inventory output differs from the returned manifest")
    return True


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("source-before")
    before.add_argument("root", type=Path)
    before.add_argument("run_root", type=Path)
    after = commands.add_parser("source-after")
    after.add_argument("root", type=Path)
    after.add_argument("run_root", type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("inspect_json", type=Path)
    inspect.add_argument("output", type=Path)
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--source", required=True, type=Path)
    inspect.add_argument("--tests", required=True, type=Path)
    inspect.add_argument("--venv", required=True, type=Path)
    inspect.add_argument("--lock", required=True, type=Path)
    inspect.add_argument("--writable", required=True, type=Path)
    inspect.add_argument("--command-file", required=True, type=Path)
    inspect.add_argument("--container-id")
    inventory = commands.add_parser("inventory")
    inventory.add_argument("run_root", type=Path)
    inventory.add_argument("control_root", type=Path)
    inventory.add_argument("output", type=Path)
    inventory.add_argument("--admin-root", required=True, type=Path)
    image = commands.add_parser("assert-local-image")
    image.add_argument("inspect_json", type=Path)
    image.add_argument("output", type=Path)
    container_id = commands.add_parser("resolve-container-id")
    container_id.add_argument("candidate_output", type=Path)
    cleanup = commands.add_parser("cleanup-status")
    cleanup.add_argument("run_root", type=Path)
    cleanup.add_argument("--container-create-attempted", action="store_true")
    cleanup.add_argument("--container-created", action="store_true")
    cleanup.add_argument("--container-started", action="store_true")
    cleanup.add_argument("--cleanup-failed", action="store_true")
    summary = commands.add_parser("run-summary")
    summary.add_argument("run_root", type=Path)
    summary.add_argument("--expected-outcome", choices=("red", "green"), required=True)
    summary.add_argument("--container-created", action="store_true")
    summary.add_argument("--container-started", action="store_true")
    summary.add_argument("--start-status", type=int, default=143)
    summary.add_argument("--cleanup-failed", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "source-before":
            result = source_before(arguments.root, arguments.run_root)
        elif arguments.command == "source-after":
            result = source_after(arguments.root, arguments.run_root)
        elif arguments.command == "inspect":
            command = arguments.command_file.read_text(encoding="utf-8")
            result = assert_container_inspect(
                json.loads(arguments.inspect_json.read_text(encoding="utf-8")),
                run_id=arguments.run_id,
                source_root=arguments.source,
                tests_root=arguments.tests,
                venv=arguments.venv,
                lock=arguments.lock,
                writable=arguments.writable,
                command=command,
                container_id=arguments.container_id,
            )
            _write_json(arguments.output, result)
        elif arguments.command == "assert-local-image":
            result = assert_local_image_identity(
                json.loads(arguments.inspect_json.read_text(encoding="utf-8"))
            )
            _write_json(arguments.output, result)
        elif arguments.command == "resolve-container-id":
            candidate = resolve_container_id(
                arguments.candidate_output.read_text(encoding="utf-8")
            )
            result = {"container_id": candidate}
        elif arguments.command == "inventory":
            result = close_inventory(
                arguments.run_root,
                arguments.control_root,
                arguments.output,
                admin_root=arguments.admin_root,
            )
        elif arguments.command == "cleanup-status":
            result = cleanup_status(
                arguments.run_root,
                container_create_attempted=arguments.container_create_attempted,
                container_created=arguments.container_created,
                container_started=arguments.container_started,
                cleanup_failed=arguments.cleanup_failed,
            )
        elif arguments.command == "run-summary":
            result = run_summary(
                arguments.run_root,
                expected_outcome=arguments.expected_outcome,
                container_created=arguments.container_created,
                container_started=arguments.container_started,
                start_status=arguments.start_status,
                cleanup_failed=arguments.cleanup_failed,
            )
        else:
            raise AssertionError("unreachable command")
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"Task 6 control check failed: {error}", file=sys.stderr)
        return 2
    if arguments.command == "resolve-container-id":
        print(result["container_id"] or "")
    elif arguments.command != "inspect":
        print(json.dumps(result, sort_keys=True))
    if arguments.command == "run-summary" and result["accepted"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
