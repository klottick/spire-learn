"""Stdlib provenance, bounded-result assessment and closed evidence helpers."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib


EXPECTED_LOCK_SHA256 = "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0"
EXPECTED_NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
EXPECTED_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
EXPECTED_GYM_SOURCE_MAP_SHA256 = "dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3"
MIN_FREE_BYTES = 30 * 1024**3
MAX_SAMPLED_ROWS = 64
MAX_PARAMETERS = 10_000_000
MAX_PROCESS_BYTES = 8 * 1024**3
CONTROL_FILES = (
    "scripts/neural/run-task4-tensor-tests.sh",
    "scripts/neural/task4_tensor_run_helpers.py",
    "scripts/neural/task4_tensor_inspect.py",
    "scripts/neural/task4-tensor-test-runbook.md",
)
_TEST_MODULES = ("test_model.py", "test_distribution.py", "test_ppo.py")
_TEST_LINE = re.compile(r"^(?P<label>.+?) \.\.\. (?P<status>ok|FAIL|ERROR|skipped)(?:\s+.*)?$")
_RUN_LINE = re.compile(r"Ran\s+(?P<count>\d+)\s+tests?\s+in\s+[0-9.]+s")
_PRODUCT_API_MISSING = re.compile(
    r"(?:"
    r"No module named ['\"]spire_learn\.neural\.(?:features|model|distribution|ppo)['\"]"
    r"|cannot import name ['\"](?:features|model|distribution|ppo)['\"] "
    r"from ['\"]spire_learn\.neural['\"](?:\s|\()"
    r")"
)
_API_IMPORT_FAILURE = re.compile(
    r"opt-in neural (?:model|distribution|PPO) API is unavailable:"
)
_OPTIONAL_RUNTIME_MISSING = re.compile(
    r"No module named ['\"](?:numpy|gymnasium|torch|torchrl|tensordict|spire_lab)(?:\.|['\"])",
)
_CONTAINER_HEREDOC = re.compile(
    r"^CONTAINER_COMMAND=\$\(cat <<'COMMAND'\n(?P<body>.*?)\nCOMMAND\n\)$",
    re.MULTILINE | re.DOTALL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mounted_inputs(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for top in ("src", "tests", "configs"):
        base = root / top
        if not base.is_dir():
            raise ValueError(f"mounted input directory is missing: {top}")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink is not permitted in read-only inputs: {path.relative_to(root)}")
            if path.is_file():
                files[path.relative_to(root).as_posix()] = _sha256(path)
    return files


def control_hashes(root: Path) -> dict[str, str]:
    """Hash every host-side runner/helper/runbook file that controls a run."""
    root = Path(root).resolve()
    result: dict[str, str] = {}
    for relative in CONTROL_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Task 4 control file is missing or not regular: {relative}")
        result[relative] = _sha256(path)
    return result


def verify_control_hashes(root: Path, expected: dict[str, str]) -> None:
    current = control_hashes(root)
    if current != expected:
        changed = sorted(set(current) | set(expected))
        changed = [name for name in changed if current.get(name) != expected.get(name)]
        raise ValueError(f"Task 4 control files changed during the run: {changed}")


def require_free_space(free_bytes: int, minimum_bytes: int = MIN_FREE_BYTES) -> None:
    if type(free_bytes) is not int or type(minimum_bytes) is not int or minimum_bytes <= 0:
        raise ValueError("free-space values must be positive primitive integers")
    if free_bytes < minimum_bytes:
        required_gib = math.ceil(minimum_bytes / 1024**3)
        raise ValueError(f"Task 4 requires at least {required_gib} GiB free; found {free_bytes} bytes")


def _input_tree_hash(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _runtime_identity(runtime_root: Path) -> dict[str, object]:
    lock = runtime_root / "data/requirements-cpu.lock"
    native = runtime_root / "data/venv-cpu/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
    reconciliation_path = runtime_root / "cpu-source-reconciliation.json"
    for path in (lock, native, reconciliation_path):
        if not path.is_file():
            raise ValueError(f"qualified CPU runtime input is missing: {path.name}")
    lock_hash = _sha256(lock)
    native_hash = _sha256(native)
    if lock_hash != EXPECTED_LOCK_SHA256 or native_hash != EXPECTED_NATIVE_SHA256:
        raise ValueError("qualified CPU lock or native extension hash changed")
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if reconciliation.get("source_map_sha256") != EXPECTED_GYM_SOURCE_MAP_SHA256:
        raise ValueError("durable accepted Gym source-map hash changed")
    if reconciliation.get("original_source_posthash_available") is not False:
        raise ValueError("Gym original-source posthash qualification changed")
    if (reconciliation.get("verified_files") != 636
            or reconciliation.get("normalized_git_modes_match") is not True):
        raise ValueError("durable accepted Gym source reconciliation changed")
    return {
        "cpu_lock_sha256": lock_hash,
        "native_extension_sha256": native_hash,
        "gym_revision": EXPECTED_GYM_REVISION,
        "gym_source_map_sha256": reconciliation["source_map_sha256"],
        "gym_source_postcheck_kind": reconciliation["source_postcheck_kind"],
        "gym_original_source_posthash_available": reconciliation["original_source_posthash_available"],
        "gym_durable_verified_files": reconciliation["verified_files"],
        "gym_durable_normalized_modes_match": reconciliation["normalized_git_modes_match"],
    }


def source_before(root: Path, run_root: Path, runtime_root: Path) -> dict[str, object]:
    root = root.resolve()
    run_root = run_root.resolve()
    free_bytes = shutil.disk_usage(run_root).free
    require_free_space(free_bytes)
    files = _mounted_inputs(root)
    controls = control_hashes(root)
    record: dict[str, object] = {
        "git_head": _git(root, "rev-parse", "HEAD").strip(),
        "git_status_porcelain_v1": _git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        "mounted_input_file_sha256": files,
        "mounted_input_tree_sha256": _input_tree_hash(files),
        "host_control_file_sha256": controls,
        "runtime_identity": _runtime_identity(runtime_root),
        "free_bytes_at_preflight": free_bytes,
        "required_free_bytes": MIN_FREE_BYTES,
        "container_writable_path_aliases": [
            {"host": str((run_root / "writable").resolve()), "container": "/run", "read_write": True},
            {
                "host": str((run_root / "writable").resolve()),
                "container": "/learner/artifacts/development/neural-combat",
                "read_write": True,
            },
        ],
        "native_resets": 0,
        "native_steps": 0,
        "policy_samples": "bounded by the shared diagnostic ledger and in-process guard",
    }
    path = run_root / "source-preflight.json"
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def source_after(root: Path, run_root: Path, runtime_root: Path) -> dict[str, object]:
    root = root.resolve()
    run_root = run_root.resolve()
    before = json.loads((run_root / "source-preflight.json").read_text(encoding="utf-8"))
    files = _mounted_inputs(root)
    controls = control_hashes(root)
    runtime = _runtime_identity(runtime_root)
    same_inputs = files == before["mounted_input_file_sha256"]
    same_tree = _input_tree_hash(files) == before["mounted_input_tree_sha256"]
    if not same_inputs or not same_tree:
        raise ValueError("read-only learner src/tests/configs inputs changed during tensor checks")
    if controls != before["host_control_file_sha256"]:
        raise ValueError("Task 4 host runner/helper/runbook files changed during tensor checks")
    if runtime != before["runtime_identity"]:
        raise ValueError("qualified CPU runtime or accepted Gym provenance changed")
    record = {
        "mounted_input_file_sha256_after": files,
        "mounted_input_tree_sha256_before": before["mounted_input_tree_sha256"],
        "mounted_input_tree_sha256_after": _input_tree_hash(files),
        "host_control_file_sha256_after": controls,
        "runtime_identity_after": runtime,
        "unchanged": True,
    }
    (run_root / "source-postcheck.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return record


def expected_test_names(tests_root: Path) -> tuple[str, ...]:
    """Derive the required suite from test methods in the three gated modules."""
    tests_root = Path(tests_root)
    names: list[str] = []
    for filename in _TEST_MODULES:
        path = tests_root / filename
        if not path.is_file():
            raise ValueError(f"expected tensor test module is missing: {filename}")
        # unittest discover -s tests/neural sets that directory as top-level.
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test_"):
                    names.append(f"{member.name} ({module}.{node.name}.{member.name})")
    if not names:
        raise ValueError("expected authored tensor test inventory is empty")
    if len(names) != len(set(names)):
        raise ValueError("expected authored tensor test inventory contains duplicate names")
    return tuple(sorted(names))


def extract_container_command(script_path: Path) -> str:
    """Return the literal shell heredoc passed as Docker's -ec command."""
    source = Path(script_path).read_text(encoding="utf-8")
    match = _CONTAINER_HEREDOC.search(source)
    if match is None:
        raise ValueError("Task 4 wrapper no longer contains the literal container command heredoc")
    return match.group("body")


def parse_container_command(script_path: Path) -> dict[str, object]:
    """Ask /bin/sh to parse the exact body without executing any command."""
    command = extract_container_command(script_path)
    result = subprocess.run(
        ["/bin/sh", "-n", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "parsed": result.returncode == 0,
        "exit_code": result.returncode,
        "stderr": result.stderr,
        "body_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "body": command,
    }


def parse_unittest_output(
    stdout: str,
    stderr: str,
    expected_names: tuple[str, ...] | list[str],
    *,
    exit_code: int | None,
) -> dict[str, object]:
    combined = f"{stdout}\n{stderr}"
    observed: list[dict[str, str]] = []
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        match = _TEST_LINE.fullmatch(line)
        if match is not None:
            status = match.group("status")
            observed.append({"test_id": match.group("label"), "status": status})
    summary = _RUN_LINE.search(combined)
    ran = int(summary.group("count")) if summary is not None else None
    counts = Counter(item["status"] for item in observed)
    normalized_counts = {
        "ok": counts.get("ok", 0),
        "failure": counts.get("FAIL", 0),
        "error": counts.get("ERROR", 0),
        "skipped": counts.get("skipped", 0),
    }
    expected = tuple(expected_names)
    expected_set = set(expected)
    observed_names = tuple(item["test_id"] for item in observed)
    observed_set = set(observed_names)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    complete = (
        summary is not None
        and ran == len(expected)
        and len(observed_names) == len(expected)
        and not missing
        and not unexpected
    )
    exit_is_zero = type(exit_code) is int and exit_code == 0
    product_api_missing = bool(_PRODUCT_API_MISSING.search(combined))
    api_import_failure = bool(_API_IMPORT_FAILURE.search(combined))
    optional_runtime_missing = bool(_OPTIONAL_RUNTIME_MISSING.search(combined))
    green_candidate = (
        complete
        and exit_is_zero
        and normalized_counts["failure"] == 0
        and normalized_counts["error"] == 0
        and normalized_counts["skipped"] == 0
    )
    return {
        "exit_code": exit_code,
        "expected_count": len(expected),
        "expected_tests": list(expected),
        "ran": ran,
        "observed_count": len(observed_names),
        "observed_tests": observed,
        "counts": normalized_counts,
        "missing_expected_tests": missing,
        "unexpected_tests": unexpected,
        "suite_complete": complete,
        "product_api_missing": product_api_missing,
        "api_import_failure": api_import_failure,
        "optional_runtime_missing": optional_runtime_missing,
        "green_candidate": green_candidate,
    }


def validate_sampling_evidence(
    evidence: object,
    *,
    reservation_id: str,
    max_rows: int = MAX_SAMPLED_ROWS,
) -> dict[str, object]:
    reasons: list[str] = []
    if not isinstance(evidence, dict):
        return {
            "present": False,
            "coherent": False,
            "completed_draws": False,
            "status": "missing",
            "attempted_rows": None,
            "completed_rows": None,
            "rejected_rows": None,
            "reasons": ["sampling evidence is missing or is not an object"],
        }
    required = {"schema", "reservation_id", "reserved_rows", "attempted_rows",
                "completed_rows", "rejected_rows", "status"}
    if not required <= evidence.keys():
        reasons.append("sampling evidence lacks required fields")
    if evidence.get("schema") != "spire-learn-tensor-sampling-v1":
        reasons.append("sampling evidence schema differs")
    if evidence.get("reservation_id") != reservation_id:
        reasons.append("sampling evidence reservation ID differs")
    if type(evidence.get("reserved_rows")) is not int or evidence.get("reserved_rows") != max_rows:
        reasons.append("sampling evidence reservation differs from the invocation ceiling")
    attempted = evidence.get("attempted_rows")
    completed = evidence.get("completed_rows")
    rejected = evidence.get("rejected_rows")
    for count, name in ((attempted, "attempted_rows"), (completed, "completed_rows"),
                        (rejected, "rejected_rows")):
        if type(count) is not int or count < 0:
            reasons.append(f"sampling {name} is not a non-negative integer")
    if all(type(count) is int and count >= 0 for count in (attempted, completed, rejected)):
        if attempted > max_rows or completed > attempted:
            reasons.append("sampling counts exceed or contradict the reserved rows")
    status = evidence.get("status")
    if status == "ready":
        if (attempted, completed, rejected) != (0, 0, 0):
            reasons.append("ready sampling evidence has nonzero counters")
    elif status == "sample_completed":
        if not (type(attempted) is int and attempted > 0 and attempted == completed and rejected == 0):
            reasons.append("completed sampling evidence has inconsistent counters")
    elif status == "budget_rejected":
        if type(rejected) is not int or rejected <= 0:
            reasons.append("budget-rejected evidence has no refused rows")
    elif status in {"sample_started", "sample_failed"}:
        reasons.append("sampling call is uncertain or failed")
    else:
        reasons.append("sampling status is unknown")
    coherent = not reasons
    completed_draws = coherent and status == "sample_completed" and attempted > 0
    return {
        "present": True,
        "coherent": coherent,
        "completed_draws": completed_draws,
        "status": status,
        "reservation_id": evidence.get("reservation_id"),
        "reserved_rows": evidence.get("reserved_rows"),
        "attempted_rows": attempted,
        "completed_rows": completed,
        "rejected_rows": rejected,
        "reasons": reasons,
    }


def validate_model_metrics(
    metrics: object,
    *,
    expected_schema_fingerprint: str | None = None,
) -> dict[str, object]:
    reasons: list[str] = []
    if not isinstance(metrics, dict):
        return {"present": False, "coherent": False, "reasons": ["model metrics are missing or not an object"]}
    fingerprint = metrics.get("schema_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        reasons.append("schema_fingerprint is not a lowercase SHA-256")
    elif expected_schema_fingerprint is not None and fingerprint != expected_schema_fingerprint:
        reasons.append("schema_fingerprint differs from the accepted run configuration")
    parameter_count = metrics.get("parameter_count")
    if type(parameter_count) is not int or not 1 <= parameter_count <= MAX_PARAMETERS:
        reasons.append("parameter_count is outside the approved model cap")
    frame_bytes = metrics.get("snapshot_bytes_per_frame")
    if type(frame_bytes) is not int or frame_bytes <= 0:
        reasons.append("snapshot_bytes_per_frame must be positive")
    timings = metrics.get("forward_backward_ms")
    if not isinstance(timings, dict):
        reasons.append("forward_backward_ms is missing")
    else:
        for batch_name in ("B1", "B4"):
            batch = timings.get(batch_name)
            if not isinstance(batch, dict):
                reasons.append(f"{batch_name} timing is missing")
                continue
            for phase in ("forward_ms", "backward_ms"):
                value = batch.get(phase)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    reasons.append(f"{batch_name}.{phase} must be finite and non-negative")
    peak = metrics.get("peak_process_bytes")
    if type(peak) is not int or not 0 < peak < MAX_PROCESS_BYTES:
        reasons.append("peak_process_bytes is outside the strict 8 GiB process cap")
    for field in ("native_resets", "native_steps", "policy_samples"):
        if type(metrics.get(field)) is not int or metrics.get(field) != 0:
            reasons.append(f"{field} must be the literal integer zero")
    return {
        "present": True,
        "coherent": not reasons,
        "parameter_count": parameter_count,
        "snapshot_bytes_per_frame": frame_bytes,
        "forward_backward_ms": timings,
        "peak_process_bytes": peak,
        "native_resets": metrics.get("native_resets"),
        "native_steps": metrics.get("native_steps"),
        "policy_samples": metrics.get("policy_samples"),
        "reasons": reasons,
    }


def assess_run(
    parsed: dict[str, object],
    *,
    expected_outcome: str,
    sampling: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    if expected_outcome not in {"red", "green"}:
        raise ValueError("expected_outcome must be red or green")
    counts = parsed["counts"]
    assert isinstance(counts, dict)
    failure_count = counts["failure"]
    error_count = counts["error"]
    skip_count = counts["skipped"]
    result: dict[str, object] = {
        "expected_outcome": expected_outcome,
        "test_run": parsed,
        "sampling": sampling,
        "model_metrics": metrics,
        "behavioral_failure": False,
        "accepted": False,
    }
    if not parsed["suite_complete"]:
        result["classification"] = "runtime_or_loader_error"
        result["reason"] = "unittest did not execute the complete authored test inventory"
        return result
    if (
        parsed["optional_runtime_missing"]
        or error_count
        or (parsed["api_import_failure"] and not parsed["product_api_missing"])
    ):
        result["classification"] = "runtime_or_loader_error"
        result["reason"] = "optional runtime or unittest error occurred"
        return result
    if skip_count:
        result["classification"] = "skipped_tests"
        result["reason"] = "a GREEN result requires zero skipped tests"
        return result
    if expected_outcome == "red":
        if failure_count == 0 and parsed["exit_code"] == 0:
            result["classification"] = "unexpected_green"
            result["reason"] = "the RED invocation unexpectedly passed"
            return result
        if failure_count > 0 and parsed["exit_code"] not in (None, 0):
            if parsed["product_api_missing"]:
                result["classification"] = "product_api_missing"
                result["reason"] = "expected product API absence was recorded separately from test behavior"
                result["accepted"] = True
                return result
            result["classification"] = "behavioral_red"
            result["reason"] = "complete authored suite recorded assertion failures"
            result["behavioral_failure"] = True
            result["accepted"] = True
            return result
        result["classification"] = "unclassified_red"
        result["reason"] = "RED outcome did not contain a complete classified failure result"
        return result
    if (
        parsed["green_candidate"]
        and failure_count == 0
        and sampling.get("coherent") is True
        and sampling.get("completed_draws") is True
        and sampling.get("rejected_rows") == 0
        and metrics.get("coherent") is True
    ):
        result["classification"] = "green"
        result["reason"] = "all authored tests and required sampling/benchmark evidence passed"
        result["accepted"] = True
        return result
    result["classification"] = "incomplete_green_evidence"
    result["reason"] = "GREEN requires zero failures/skips and coherent sampling plus model metrics"
    return result


def close_inventory(run_root: Path) -> dict[str, object]:
    run_root = run_root.resolve()
    inventory_path = run_root / "evidence-inventory.json"
    files: dict[str, str] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence inventory refuses symlinks: {path.relative_to(run_root)}")
        if path.is_file() and path != inventory_path:
            files[path.relative_to(run_root).as_posix()] = _sha256(path)
    record = {
        "inventory_version": 1,
        "run_root": str(run_root),
        "file_count": len(files),
        "sha256": files,
        "inventory_self_excluded": True,
    }
    inventory_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def _read_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_exit(log_root: Path, command_name: str) -> int | None:
    path = log_root / f"{command_name}.exit.log"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    try:
        return int(value)
    except ValueError:
        return None


def evaluate_run_directory(
    *,
    root: Path,
    run_root: Path,
    expected_outcome: str,
    reservation_id: str,
    max_rows: int = MAX_SAMPLED_ROWS,
) -> dict[str, object]:
    run_root = run_root.resolve()
    log_root = run_root / "logs"
    stdout_path = log_root / "container-start.stdout.log"
    stderr_path = log_root / "container-start.stderr.log"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    parsed = parse_unittest_output(
        stdout,
        stderr,
        expected_test_names(root / "tests/neural"),
        exit_code=_read_exit(log_root, "container-start"),
    )
    sampling = validate_sampling_evidence(
        _read_json(run_root / "writable/tensor-sampling.json"),
        reservation_id=reservation_id,
        max_rows=max_rows,
    )
    config_path = root / "configs/neural/run.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    environment = config.get("environment")
    expected_fingerprint = environment.get("common_view_fingerprint") if isinstance(environment, dict) else None
    metrics = validate_model_metrics(
        _read_json(run_root / "writable/model-metrics.json"),
        expected_schema_fingerprint=expected_fingerprint,
    )
    result = assess_run(
        parsed,
        expected_outcome=expected_outcome,
        sampling=sampling,
        metrics=metrics,
    )
    result["reservation_id"] = reservation_id
    result["max_sampled_rows"] = max_rows
    result_path = run_root / "test-result.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def write_cleanup_status(
    run_root: Path,
    *,
    create_attempted: bool,
    container_created: bool,
    cleanup_failed: bool,
) -> dict[str, object]:
    run_root = run_root.resolve()
    logs = run_root / "logs"
    inventory_exit = _read_exit(logs, "container-final-inventory")
    inventory_stdout = logs / "container-final-inventory.stdout.log"
    inventory_output = inventory_stdout.read_text(encoding="utf-8", errors="replace").strip() \
        if inventory_stdout.is_file() else ""
    inventory_result = (
        "absent" if create_attempted and inventory_exit == 0 and not inventory_output
        else "present" if create_attempted and inventory_exit == 0 and inventory_output
        else "unknown" if create_attempted
        else "not_attempted"
    )
    record = {
        "create_attempted": create_attempted,
        "container_created": container_created,
        "remove_exit_status": _read_exit(logs, "container-remove"),
        "final_named_container_inventory_exit_status": inventory_exit,
        "final_named_container_inventory_output": inventory_output,
        "resource_state_after_cleanup": inventory_result,
        "source_postcheck_exit_status": _read_exit(logs, "source-postcheck"),
        "sample_ledger_finish_exit_status": _read_exit(logs, "sample-row-finish"),
        "cleanup_failed": cleanup_failed or (create_attempted and inventory_result != "absent"),
    }
    path = run_root / "cleanup-status.json"
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def write_run_summary(
    run_root: Path,
    *,
    pre_inventory_exit_status: int,
    expected_outcome: str,
    container_created: bool,
    container_started: bool,
) -> dict[str, object]:
    run_root = run_root.resolve()
    logs = run_root / "logs"
    command_statuses: dict[str, int | None] = {}
    for path in sorted(logs.glob("*.exit.log")):
        command_statuses[path.name.removesuffix(".exit.log")] = _read_exit(logs, path.name.removesuffix(".exit.log"))
    record = {
        "summary_version": 1,
        "pre_inventory_exit_status": pre_inventory_exit_status,
        "expected_outcome": expected_outcome,
        "container_created": container_created,
        "container_started": container_started,
        "command_exit_statuses": command_statuses,
        "test_result": _read_json(run_root / "test-result.json"),
        "cleanup_status": _read_json(run_root / "cleanup-status.json"),
        "source_postcheck": _read_json(run_root / "source-postcheck.json"),
    }
    path = run_root / "run-summary.json"
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def _runtime_identity_cli(path: Path) -> dict[str, object]:
    return _runtime_identity(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("source-before")
    before.add_argument("root", type=Path)
    before.add_argument("run_root", type=Path)
    before.add_argument("runtime_root", type=Path)
    after = commands.add_parser("source-after")
    after.add_argument("root", type=Path)
    after.add_argument("run_root", type=Path)
    after.add_argument("runtime_root", type=Path)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("run_root", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("root", type=Path)
    evaluate.add_argument("run_root", type=Path)
    evaluate.add_argument("--expected-outcome", choices=("red", "green"), required=True)
    evaluate.add_argument("--reservation-id", required=True)
    evaluate.add_argument("--max-rows", type=int, default=MAX_SAMPLED_ROWS)
    cleanup = commands.add_parser("cleanup-status")
    cleanup.add_argument("run_root", type=Path)
    cleanup.add_argument("--create-attempted", action="store_true")
    cleanup.add_argument("--container-created", action="store_true")
    cleanup.add_argument("--cleanup-failed", action="store_true")
    summary = commands.add_parser("run-summary")
    summary.add_argument("run_root", type=Path)
    summary.add_argument("--pre-inventory-exit-status", type=int, required=True)
    summary.add_argument("--expected-outcome", choices=("red", "green"), required=True)
    summary.add_argument("--container-created", action="store_true")
    summary.add_argument("--container-started", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "source-before":
            result = source_before(args.root, args.run_root, args.runtime_root)
        elif args.command == "source-after":
            result = source_after(args.root, args.run_root, args.runtime_root)
        elif args.command == "inventory":
            result = close_inventory(args.run_root)
        elif args.command == "evaluate":
            result = evaluate_run_directory(
                root=args.root,
                run_root=args.run_root,
                expected_outcome=args.expected_outcome,
                reservation_id=args.reservation_id,
                max_rows=args.max_rows,
            )
        elif args.command == "cleanup-status":
            result = write_cleanup_status(
                args.run_root,
                create_attempted=args.create_attempted,
                container_created=args.container_created,
                cleanup_failed=args.cleanup_failed,
            )
        else:
            result = write_run_summary(
                args.run_root,
                pre_inventory_exit_status=args.pre_inventory_exit_status,
                expected_outcome=args.expected_outcome,
                container_created=args.container_created,
                container_started=args.container_started,
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Task 4 provenance/evidence helper failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    if args.command == "evaluate" and not result.get("accepted"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
