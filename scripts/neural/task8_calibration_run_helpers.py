"""Small stdlib-only gates for the one reviewed Task 8 calibration cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tomllib
import time
from typing import Callable, Mapping, Sequence

import task8_gpu_setup_helpers as setup


TASK_RELATIVE = Path("artifacts/development/neural-combat/task-8")
TASK_ROOT = TASK_RELATIVE
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}\Z")
FULL_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
UUID = setup.GPU_UUID
CUDA_RUN_ID = setup.CORRECTED_RNG_RUN_ID
IMAGE_ALIAS = "spire-learn/task8-python:cuda-setup-20261028-a"
IMAGE_REFERENCE = setup.PYTHON_IMAGE_REFERENCE
LOCK_SHA256 = "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45"
NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
TORCH = "2.14.1+cu130"
NUMPY = "2.5.4"
GYMNASIUM = "1.4.0"
SPIRE_LAB = "0.1.0"
TEST_LABEL = (
    "test_task8_cuda_calibration_runs_one_fixed_cohort "
    "(tests.integration.test_neural_calibration.NeuralCalibrationIntegrationTests."
    "test_task8_cuda_calibration_runs_one_fixed_cohort)"
)

SOURCE_SCOPES = (
    "src",
    "tests",
    "configs/neural/calibration.toml",
    "configs/cases/neural/train.json",
    "scripts/neural/run-task8-calibration.sh",
    "scripts/neural/task8-calibration-runbook.md",
    "scripts/neural/task8_calibration_run_helpers.py",
    "scripts/neural/task8_gpu_setup_helpers.py",
)

RELEASE_FIELDS = frozenset({
    "decision", "parent_source_review", "independent_review", "single_use", "release_id",
    "source_snapshot_sha256", "runbook_sha256", "calibration_config_sha256",
    "train_manifest_sha256", "qualification_report_sha256", "qualification_summary_sha256",
    "cuda_lock_sha256", "native_sha256", "gpu_uuid", "learner_head", "learner_dirty",
    "previous_claim_sha256", "previous_failure_proof_sha256", "previous_inventory_sha256",
})

PREDECESSOR_RUN_ID = "calibration-20261028-b"
PREDECESSOR_RELEASE_SHA256 = "64f00b405b1c1520541e68aef77015cc2d446a73d5e4701fc7683f2d35ceb20e"
PREDECESSOR_CLAIM_SHA256 = "cd023e95b890cbd5f9b1c8604e8c2ed35001fe0644c12cd1af0a1fe47cde2164"
PREDECESSOR_FAILURE_PROOF_SHA256 = "4ab61300d71d06ad2c4dc434a0c2741e8f61d162756871a52ea39e3350c1f5ee"
PREDECESSOR_INVENTORY_SHA256 = "a3eb9438f730092b5ad29b24f4299efb04bafeeca682dbcf67e603b3fef4189e"
CONTINUATION_GENERATION = 1
CONTINUATION_RELEASE_PATH = Path("calibration/phase-release-v3.json")
CONTINUATION_SNAPSHOT_PATH = Path("calibration/calibration-source-snapshot-v3.json")
CONTINUATION_CLAIM_PATH = Path("calibration/phase-release-continuation-1.claim.json")
PREDECESSOR_EVIDENCE_PATHS = {
    "previous_claim_sha256": Path("calibration/phase-release.claim.json"),
    "previous_failure_proof_sha256": Path(
        "calibration/parent-invocation-v2/runtime-parent-proof.json"
    ),
    "previous_inventory_sha256": Path(
        "calibration/runtime-admin/calibration-20261028-b/evidence-inventory.json"
    ),
}
PREDECESSOR_EVIDENCE_HASHES = {
    "previous_claim_sha256": PREDECESSOR_CLAIM_SHA256,
    "previous_failure_proof_sha256": PREDECESSOR_FAILURE_PROOF_SHA256,
    "previous_inventory_sha256": PREDECESSOR_INVENTORY_SHA256,
}

CALIBRATION_SCRIPT = """set -eu
/runtime/venv-cuda/bin/python -B -m pip check
/runtime/venv-cuda/bin/python -B - <<'PY'
import importlib.metadata
import json
import sys
expected = {
    "torch": "2.14.1+cu130",
    "numpy": "2.5.4",
    "gymnasium": "1.4.0",
    "spire-lab": "0.1.0",
}
observed = {name: importlib.metadata.version(name) for name in expected}
if sys.version_info[:2] != (3, 12) or observed != expected:
    raise SystemExit("locked Python/runtime package identity differs")
print("TASK8_CALIBRATION_PACKAGE_PREFLIGHT=" + json.dumps(observed, sort_keys=True))
PY
export PYTHONPATH=/learner/src
exec /runtime/venv-cuda/bin/python -B -m unittest -v \\
  tests.integration.test_neural_calibration.NeuralCalibrationIntegrationTests.test_task8_cuda_calibration_runs_one_fixed_cohort 2>&1
"""


class CalibrationControlError(ValueError):
    """A missing or mismatched release, source, host, or Docker invariant."""


def _canonical(path: Path, *, kind: str | None = None) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise CalibrationControlError(f"symlink input is forbidden: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CalibrationControlError(f"required path is missing: {path}") from error
    if path != resolved:
        raise CalibrationControlError(f"input path must be canonical: {path}")
    if kind == "file" and (not path.is_file() or not stat.S_ISREG(path.lstat().st_mode)):
        raise CalibrationControlError(f"input must be a regular file: {path}")
    if kind == "directory" and not path.is_dir():
        raise CalibrationControlError(f"input must be a directory: {path}")
    return path


def _repo_root(root: Path) -> Path:
    root = _canonical(root, kind="directory")
    if not (root / ".git").exists() and not (root / ".git").is_file():
        raise CalibrationControlError("repository root does not contain its Git metadata")
    return root


def _sha256(path: Path) -> str:
    try:
        return setup.sha256_file(_canonical(path, kind="file"))
    except (OSError, setup.Task8ControlError) as error:
        raise CalibrationControlError(str(error)) from error


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(_canonical(path, kind="file").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationControlError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise CalibrationControlError(f"JSON input must be an object: {path}")
    return value


def verify_predecessor_evidence(
    root: Path,
    *,
    hash_file: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    """Verify the fixed consumed claim and its reviewed no-effect failure proof."""
    root = _repo_root(root)
    hash_file = _sha256 if hash_file is None else hash_file
    paths = {
        key: root / TASK_RELATIVE / relative
        for key, relative in PREDECESSOR_EVIDENCE_PATHS.items()
    }
    for key, path in paths.items():
        path = _canonical(path, kind="file")
        if hash_file(path) != PREDECESSOR_EVIDENCE_HASHES[key]:
            raise CalibrationControlError(f"fixed predecessor evidence hash differs: {key}")

    claim = _json_object(paths["previous_claim_sha256"])
    if (claim != {
            "container_create_attempts_allowed": 1,
            "release_sha256": PREDECESSOR_RELEASE_SHA256,
            "run_id": PREDECESSOR_RUN_ID,
            "schema_id": "spire-learn-task8-calibration-release-claim-v1",
        }
            or type(claim.get("container_create_attempts_allowed")) is not int):
        raise CalibrationControlError("original consumed claim does not bind the exact failed run/release")

    proof = _json_object(paths["previous_failure_proof_sha256"])
    expected_effects = {
        "calibration_trials_started": 0,
        "native_factories": 0,
        "native_resets": 0,
        "native_steps": 0,
        "optimizer_calls": 0,
        "policy_samples": 0,
    }
    if any(type(proof.get(key)) is not int or proof.get(key) != 0
           for key in expected_effects):
        raise CalibrationControlError("predecessor proof records a calibration or native effect")
    enabled_tests = proof.get("enabled_tests")
    if (enabled_tests != {"ran": 1, "failures": 1, "errors": 0, "skipped": 0}
            or not isinstance(enabled_tests, dict)
            or any(type(enabled_tests.get(key)) is not int for key in enabled_tests)):
        raise CalibrationControlError("predecessor proof is not exactly one failed enabled test")
    cleanup = proof.get("cleanup")
    expected_cleanup = {
        "container_create_attempted": 1,
        "create_returned_full_id": 1,
        "container_owned": 1,
        "container_started_once": 1,
        "container_removed": 1,
        "cleanup_failed": 0,
    }
    if (cleanup != expected_cleanup or not isinstance(cleanup, dict)
            or any(type(cleanup.get(key)) is not int for key in expected_cleanup)):
        raise CalibrationControlError("predecessor container cleanup was not fully successful")
    if (proof.get("run_id") != PREDECESSOR_RUN_ID
            or proof.get("claim_consumed") is not True
            or type(proof.get("wrapper_exit")) is not int
            or proof.get("wrapper_exit") != 2
            or proof.get("failure_phase") != (
                "verify_device release binding before CUDA initialization/model or driver factory"
            )):
        raise CalibrationControlError("predecessor proof does not document the fixed pre-effect failure")
    prior_inventory = proof.get("inventory")
    if (not isinstance(prior_inventory, dict)
            or prior_inventory.get("sha256") != PREDECESSOR_INVENTORY_SHA256
            or prior_inventory.get("entries") != 120
            or prior_inventory.get("file_hashes_verified") != 116
            or prior_inventory.get("mismatches") != []):
        raise CalibrationControlError("predecessor proof does not bind the fully checked runtime inventory")
    lifecycle = proof.get("host_lifecycle")
    if (not isinstance(lifecycle, dict)
            or lifecycle.get("status") != "failed"
            or lifecycle.get("cleanup_failed") is not False
            or lifecycle.get("wrapper_exit_code") != 2
            or lifecycle.get("run_id") not in (None, PREDECESSOR_RUN_ID)):
        raise CalibrationControlError("predecessor lifecycle does not record a failed, cleaned-up attempt")

    inventory = _json_object(paths["previous_inventory_sha256"])
    entries = inventory.get("entries")
    roots = inventory.get("roots")
    expected_root_suffixes = {
        f"{TASK_RELATIVE.as_posix()}/calibration/runtime/{PREDECESSOR_RUN_ID}",
        f"{TASK_RELATIVE.as_posix()}/calibration/runtime-control/{PREDECESSOR_RUN_ID}",
    }
    inventory_kinds = [entry.get("kind") for entry in entries
                       if isinstance(entry, dict)] if isinstance(entries, list) else []
    observed_root_suffixes = {
        suffix for suffix in expected_root_suffixes
        if isinstance(roots, list)
        and any(isinstance(value, str) and value.endswith("/" + suffix) for value in roots)
    }
    if (inventory.get("schema_id") != "spire-learn-task8-calibration-closed-inventory-v1"
            or inventory.get("logs_closed_before_inventory") is not True
            or not isinstance(entries, list) or len(entries) != 120
            or not isinstance(roots, list) or len(roots) != 2
            or len(inventory_kinds) != len(entries)
            or inventory_kinds.count("file") != 116
            or inventory_kinds.count("directory") != 4
            or observed_root_suffixes != expected_root_suffixes):
        raise CalibrationControlError("predecessor runtime inventory is incomplete or names other roots")

    return {
        "continuation_generation": CONTINUATION_GENERATION,
        "previous_run_id": PREDECESSOR_RUN_ID,
        "previous_release_sha256": PREDECESSOR_RELEASE_SHA256,
        **PREDECESSOR_EVIDENCE_HASHES,
    }


def capture_source_snapshot(root: Path, output: Path) -> dict[str, object]:
    root = _repo_root(root)
    output = Path(output)
    if output.is_symlink() or output.exists():
        raise CalibrationControlError("source snapshot is immutable and its path already exists")
    output_parent = _canonical(output.parent, kind="directory")
    if output.parent != output_parent:
        raise CalibrationControlError("source snapshot parent must be canonical and contain no symlinks")
    output = output_parent / output.name
    task_root = _canonical(root / TASK_RELATIVE, kind="directory")
    try:
        if task_root not in output.parents:
            raise CalibrationControlError("source snapshot output must be inside Task 8 evidence")
        value = setup.capture_source_snapshot(root, output, scope_roots=SOURCE_SCOPES)
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    return value


def _expected_input_paths(root: Path) -> dict[str, Path]:
    return {
        "calibration_config": root / "configs/neural/calibration.toml",
        "train_manifest": root / "configs/cases/neural/train.json",
        "qualification_report": root / TASK_RELATIVE / "task-8-gpu-qualification-report.md",
        "qualification_summary": root / TASK_RELATIVE / "task-8-gpu-qualification-summary.json",
        "cuda_lock": root / TASK_RELATIVE / "runtime-output" / CUDA_RUN_ID / "resolve-work" / "requirements-cuda.lock",
    }


def verify_calibration_config(config_path: Path, manifest_path: Path) -> dict[str, str]:
    config_path = _canonical(config_path, kind="file")
    manifest_path = _canonical(manifest_path, kind="file")
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CalibrationControlError("calibration TOML is malformed") from error
    expected = {
        "schema": "spire-learn-neural-config-v1",
        "mode": "standalone",
        "operation": "calibrate",
        "split": "train",
        "partition": "train",
        "manifest_path": "../cases/neural/train.json",
        "output_root": "../../artifacts/development/neural-combat/task-8/calibration/outputs",
        "device": "cuda:0",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise CalibrationControlError("calibration config does not target the exact approved cohort inputs")
    calibration = config.get("calibration")
    if calibration != {"gpu_uuid": UUID}:
        raise CalibrationControlError("calibration GPU table differs from the exact qualified UUID")
    if config.get("budgets") != {
        "max_optimizer_steps": 3,
        "max_training_decisions": 1024,
        "max_environment_actions": 1024,
        "max_wall_seconds": 300.0,
    }:
        raise CalibrationControlError("calibration budgets differ from the approved fixed cohort")
    rollout = config.get("rollout")
    if rollout != {"rows": 4, "steps": 32}:
        raise CalibrationControlError("calibration base rollout differs from the approved standalone trial")
    if Path(config_path.parent / str(config["manifest_path"])).resolve(strict=True) != manifest_path:
        raise CalibrationControlError("calibration config manifest path does not resolve to the mounted manifest")
    return {
        "calibration_config_sha256": _sha256(config_path),
        "train_manifest_sha256": _sha256(manifest_path),
    }


def verify_qualification(summary_path: Path, report_path: Path, lock_path: Path) -> dict[str, str]:
    summary_path = _canonical(summary_path, kind="file")
    report_path = _canonical(report_path, kind="file")
    lock_path = _canonical(lock_path, kind="file")
    summary = _json_object(summary_path)
    if (summary.get("status") != "RNG_fix_and_finite_device_qualified;calibration_pending"
            or summary.get("run_id") != CUDA_RUN_ID or summary.get("lock_sha256") != LOCK_SHA256):
        raise CalibrationControlError("retained CUDA qualification is missing or no longer accepted")
    green = summary.get("rng_green")
    if not isinstance(green, dict) or any(green.get(key) != expected for key, expected in {
        "tests": 3, "failures": 0, "errors": 0, "skipped": 0, "wrapper_exit": 0,
    }.items()):
        raise CalibrationControlError("retained RNG qualification no longer passes its exact checks")
    device = summary.get("device_check")
    if not isinstance(device, dict) or any(device.get(key) != expected for key, expected in {
        "status": "qualified", "gpu_uuid": UUID, "torch": TORCH, "torch_cuda": "13.0",
    }.items()):
        raise CalibrationControlError("retained device qualification differs from the accepted CUDA device")
    architectures = device.get("compiled_architectures")
    if (not isinstance(architectures, list) or "sm_120" not in architectures
            or device.get("compute_capability") != [12, 0]):
        raise CalibrationControlError("retained device qualification lacks compiled sm_120 support")
    inventory_check = summary.get("device_inventory_verification")
    if not isinstance(inventory_check, dict) or inventory_check.get("unchanged") is not True:
        raise CalibrationControlError("retained approved GPU inventory is not unchanged")
    if _sha256(lock_path) != LOCK_SHA256:
        raise CalibrationControlError("retained CUDA lock SHA256 differs from the accepted closure")
    return {
        "qualification_report_sha256": _sha256(report_path),
        "qualification_summary_sha256": _sha256(summary_path),
        "cuda_lock_sha256": LOCK_SHA256,
        "native_sha256": NATIVE_SHA256,
    }


def read_learner_git_identity(root: Path) -> dict[str, object]:
    """Read only the host checkout HEAD and dirty bit with bounded Git calls."""
    root = _repo_root(root)
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5, cwd=root, env=environment,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=5, cwd=root, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CalibrationControlError("bounded host Git identity query failed") from error
    head = head_result.stdout.strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None:
        raise CalibrationControlError("host Git HEAD is not a full supported object ID")
    if len(status_result.stdout.encode("utf-8")) > 1024 * 1024:
        raise CalibrationControlError("host Git dirty-state output exceeds its 1 MiB bound")
    return {"learner_head": head, "learner_dirty": bool(status_result.stdout)}


def verify_learner_git_release(root: Path, release: Mapping[str, object]) -> dict[str, object]:
    head = release.get("learner_head")
    dirty = release.get("learner_dirty")
    if (not isinstance(head, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None
            or type(dirty) is not bool):
        raise CalibrationControlError("parent release learner Git identity is malformed")
    observed = read_learner_git_identity(root)
    if observed != {"learner_head": head, "learner_dirty": dirty}:
        raise CalibrationControlError("parent release learner HEAD or dirty state is stale")
    return observed


def verify_release(root: Path, release_path: Path, snapshot_path: Path) -> dict[str, object]:
    """Fail closed unless the parent release binds every reviewed input byte."""
    root = _repo_root(root)
    task_root = _canonical(root / TASK_RELATIVE, kind="directory")
    if (Path(release_path) != task_root / CONTINUATION_RELEASE_PATH
            or Path(snapshot_path) != task_root / CONTINUATION_SNAPSHOT_PATH):
        raise CalibrationControlError("only the fixed v3 continuation release and snapshot are accepted")
    predecessor = verify_predecessor_evidence(root)
    try:
        release_path, snapshot_path = setup.validate_safe_task8_inputs(
            task_root, (Path(release_path), Path(snapshot_path)),
        )
        snapshot = setup.verify_source_snapshot(snapshot_path, root)
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    raw_snapshot = _json_object(snapshot_path)
    if raw_snapshot.get("scope_roots") != list(SOURCE_SCOPES):
        raise CalibrationControlError("source snapshot scope is not the exact calibration projection")
    paths = _expected_input_paths(root)
    hashes = verify_calibration_config(paths["calibration_config"], paths["train_manifest"])
    qualification = verify_qualification(
        paths["qualification_summary"], paths["qualification_report"], paths["cuda_lock"],
    )
    runbook = root / "scripts/neural/task8-calibration-runbook.md"
    release = _json_object(release_path)
    if set(release) != RELEASE_FIELDS:
        raise CalibrationControlError("parent calibration release fields are incomplete or unexpected")
    if (release.get("decision") != "one_task8_calibration_cohort"
            or release.get("parent_source_review") != "passed"
            or release.get("independent_review") != "passed"
            or release.get("single_use") is not True
            or not isinstance(release.get("release_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", release["release_id"])):
        raise CalibrationControlError("parent release is not an explicit one-cohort source-review release")
    learner_identity = verify_learner_git_release(root, release)
    expected = {
        "source_snapshot_sha256": _sha256(snapshot_path),
        "runbook_sha256": _sha256(runbook),
        **hashes,
        **qualification,
        **PREDECESSOR_EVIDENCE_HASHES,
        "gpu_uuid": UUID,
        **learner_identity,
    }
    if any(release.get(key) != value for key, value in expected.items()):
        raise CalibrationControlError("parent release does not bind the current source/config/runtime evidence")
    return {
        "release_id": release["release_id"],
        "continuation_generation": CONTINUATION_GENERATION,
        "previous_run_id": predecessor["previous_run_id"],
        "previous_release_sha256": predecessor["previous_release_sha256"],
        **expected,
        "source_file_count": snapshot.get("file_count"),
        "release_sha256": _sha256(release_path),
    }


def claim_continuation_release(
    root: Path,
    release_path: Path,
    snapshot_path: Path,
    marker_path: Path,
    run_id: str,
    *,
    expected_release_sha256: str,
) -> dict[str, object]:
    """Verify the fixed continuation release and claim its only marker once."""
    if (not isinstance(expected_release_sha256, str)
            or FULL_ID_PATTERN.fullmatch(expected_release_sha256) is None):
        raise CalibrationControlError("captured release SHA256 is missing or malformed")
    root = _repo_root(root)
    verified = verify_release(root, release_path, snapshot_path)
    if verified.get("release_sha256") != expected_release_sha256:
        raise CalibrationControlError("verified release SHA256 differs from captured binding")
    validate_run_id(run_id)
    if run_id == PREDECESSOR_RUN_ID:
        raise CalibrationControlError("continuation requires a fresh run ID")
    expected_marker = root / TASK_RELATIVE / CONTINUATION_CLAIM_PATH
    marker_path = Path(marker_path)
    if marker_path != expected_marker:
        raise CalibrationControlError("only the fixed generation-one continuation marker is accepted")
    payload_value = {
        "schema_id": "spire-learn-task8-calibration-continuation-claim-v1",
        "continuation_generation": CONTINUATION_GENERATION,
        "run_id": run_id,
        "release_sha256": verified["release_sha256"],
        "container_create_attempts_allowed": 1,
        "previous_run_id": verified["previous_run_id"],
        "previous_release_sha256": verified["previous_release_sha256"],
        **PREDECESSOR_EVIDENCE_HASHES,
    }
    payload = json.dumps(
        payload_value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode() + b"\n"
    _write_exclusive(marker_path, payload)
    return payload_value


def assert_claim_release_binding(claim_path: Path, expected_release_sha256: str) -> None:
    """Stop before create unless the exclusive claim matches the captured binding."""
    if (not isinstance(expected_release_sha256, str)
            or FULL_ID_PATTERN.fullmatch(expected_release_sha256) is None):
        raise CalibrationControlError("captured release SHA256 is missing or malformed")
    claim = _json_object(_canonical(Path(claim_path), kind="file"))
    if claim.get("release_sha256") != expected_release_sha256:
        raise CalibrationControlError("exclusive claim release SHA256 differs from captured binding")


def verify_image_alias(raw: str | bytes) -> dict[str, object]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        values = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationControlError("local image inspect output is malformed JSON") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise CalibrationControlError("local alias inspect must return exactly one image")
    try:
        identity = setup.verify_image_inspect(raw)
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    tags = values[0].get("RepoTags")
    if not isinstance(tags, list) or IMAGE_ALIAS not in tags:
        raise CalibrationControlError("the exact already-local CUDA setup alias is absent")
    return {**identity, "local_alias": IMAGE_ALIAS}


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise CalibrationControlError("run ID must be a unique 1..40 character safe identifier")
    return run_id


def verify_gpu_inventory_csv(raw: str) -> dict[str, object]:
    try:
        return setup.verify_gpu_inventory(raw, minimum_free_mib=12 * 1024)
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error


def verify_host_storage(
    root: Path,
    run_root: Path,
    control_root: Path,
    admin_root: Path,
    output_child: Path,
    *,
    reservation_bytes: int | None = None,
) -> dict[str, int]:
    root = _repo_root(root)
    task_root = _canonical(root / TASK_RELATIVE, kind="directory")
    run_root = _canonical(run_root, kind="directory")
    control_root = _canonical(control_root, kind="directory")
    admin_root = _canonical(admin_root, kind="directory")
    output_child = _canonical(output_child, kind="directory")
    neural_root = _canonical(root / "artifacts/development/neural-combat", kind="directory")
    runtime_venv = _canonical(
        task_root / "runtime-output" / CUDA_RUN_ID / "install-work" / "venv-cuda",
        kind="directory",
    )
    cpu_venv = _canonical(
        root / "artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc/data/venv-cpu",
        kind="directory",
    )
    if reservation_bytes is not None and (
        type(reservation_bytes) is not int or not 0 <= reservation_bytes <= 1024**3
    ):
        raise CalibrationControlError("calibration storage reservation must be between zero and 1 GiB")
    try:
        measured = setup.check_storage_paths(
            root,
            neural_root,
            (run_root, control_root, admin_root),
            setup=False,
            reservation_bytes=0,
            allowed_venv_roots=(runtime_venv, cpu_venv),
            protected_cpu_venv_root=cpu_venv,
        )
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    if reservation_bytes is None:
        reservation_bytes = 1024**3 - measured["new_gpu_bytes"]
        if reservation_bytes < 0:
            raise CalibrationControlError("calibration-owned growth exceeds the fixed 1 GiB cap")
    if measured["new_gpu_bytes"] + reservation_bytes > 1024**3:
        raise CalibrationControlError("calibration-owned growth plus reservation exceeds 1 GiB")
    try:
        budget = setup.verify_storage_budget(
            new_gpu_bytes=measured["new_gpu_bytes"],
            neural_cumulative_bytes=measured["neural_current_bytes"],
            free_bytes=measured["free_bytes"],
            setup=False,
            reservation_bytes=reservation_bytes,
        )
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    budget["neural_current_bytes"] = measured["neural_current_bytes"]
    if run_root.resolve(strict=True) not in output_child.resolve(strict=True).parents:
        raise CalibrationControlError("writable output child must be inside this fresh calibration run root")
    return budget


def build_mounts(root: Path, output_child: Path) -> tuple[tuple[Path, str, bool], ...]:
    root = _repo_root(root)
    output_child = _canonical(output_child, kind="directory")
    runtime_root = root / TASK_RELATIVE / "calibration" / "runtime"
    try:
        output_relative = output_child.relative_to(runtime_root)
    except ValueError as error:
        raise CalibrationControlError("writable mount must be this repository's fresh calibration output child") from error
    if (len(output_relative.parts) != 2 or output_relative.parts[1] != "output-child"
            or not RUN_ID_PATTERN.fullmatch(output_relative.parts[0])):
        raise CalibrationControlError("writable mount path must be runtime/<fresh-run-id>/output-child")
    if any(output_child.iterdir()):
        raise CalibrationControlError("writable output child must be empty before container creation")
    paths = _expected_input_paths(root)
    runtime_venv = root / TASK_RELATIVE / "runtime-output" / CUDA_RUN_ID / "install-work" / "venv-cuda"
    helper = root / "scripts/neural/task8_calibration_run_helpers.py"
    setup_helper = root / "scripts/neural/task8_gpu_setup_helpers.py"
    mounts: tuple[tuple[Path, str, bool], ...] = (
        (root / "src", "/learner/src", True),
        (root / "tests", "/learner/tests", True),
        (paths["calibration_config"], "/learner/configs/neural/calibration.toml", True),
        (paths["train_manifest"], "/learner/configs/cases/neural/train.json", True),
        (helper, "/learner/scripts/neural/task8_calibration_run_helpers.py", True),
        (setup_helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
        (paths["cuda_lock"], "/learner/evidence/requirements-cuda.lock", True),
        (paths["qualification_report"], "/learner/evidence/task-8-gpu-qualification-report.md", True),
        (paths["qualification_summary"], "/learner/evidence/task-8-gpu-qualification-summary.json", True),
        (runtime_venv, "/runtime/venv-cuda", True),
        (output_child, "/learner/artifacts/development/neural-combat/task-8/calibration/outputs", False),
    )
    destinations: set[str] = set()
    for source, destination, _readonly in mounts:
        _canonical(source, kind="directory" if destination in {"/learner/src", "/learner/tests", "/runtime/venv-cuda"} or not source.is_file() else "file")
        if destination in destinations:
            raise CalibrationControlError("container mount destinations must be unique")
        destinations.add(destination)
    if sum(not readonly for _source, _destination, readonly in mounts) != 1:
        raise CalibrationControlError("the fresh output child must be the only writable bind")
    if any(not readonly and destination != "/learner/artifacts/development/neural-combat/task-8/calibration/outputs"
           for _source, destination, readonly in mounts):
        raise CalibrationControlError("only the exact calibration output child may be writable")
    return mounts


def container_name(run_id: str) -> str:
    return f"spirelearn-task8-calibration-{validate_run_id(run_id)}"


def build_container_argv(
    root: Path,
    output_child: Path,
    run_id: str,
    *,
    image_id: str,
    release_binding: Mapping[str, object],
) -> tuple[str, ...]:
    run_id = validate_run_id(run_id)
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise CalibrationControlError("container image identity must be sha256 plus a full 64-digit ID")
    try:
        output_relative = Path(output_child).relative_to(
            Path(root) / TASK_RELATIVE / "calibration" / "runtime"
        )
    except ValueError as error:
        raise CalibrationControlError("container run ID must own the exact writable output child") from error
    if output_relative.parts[0] != run_id:
        raise CalibrationControlError("container run ID must own the exact writable output child")
    required_hashes = {
        "source_snapshot_sha256", "runbook_sha256", "calibration_config_sha256",
        "train_manifest_sha256", "qualification_report_sha256", "qualification_summary_sha256",
        "cuda_lock_sha256", "native_sha256", *PREDECESSOR_EVIDENCE_HASHES,
    }
    required_lineage = {
        "continuation_generation", "previous_run_id", "previous_release_sha256",
    }
    required_binding = required_hashes | required_lineage
    if not required_binding.issubset(release_binding):
        raise CalibrationControlError("verified parent release binding is incomplete")
    for field in required_hashes:
        digest = release_binding[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise CalibrationControlError(f"parent release {field} must be a SHA256 digest")
    if (release_binding["cuda_lock_sha256"] != LOCK_SHA256
            or release_binding["native_sha256"] != NATIVE_SHA256):
        raise CalibrationControlError("verified parent release CUDA/native identities differ")
    if (type(release_binding.get("continuation_generation")) is not int
            or release_binding.get("continuation_generation") != CONTINUATION_GENERATION
            or release_binding.get("previous_run_id") != PREDECESSOR_RUN_ID
            or release_binding.get("previous_release_sha256") != PREDECESSOR_RELEASE_SHA256
            or any(release_binding.get(key) != value
                   for key, value in PREDECESSOR_EVIDENCE_HASHES.items())):
        raise CalibrationControlError("verified parent release has the wrong continuation lineage")
    mounts = build_mounts(root, output_child)
    args = [
        "docker", "create", "--pull=never", "--name", container_name(run_id),
        "--label", "spire-learn.task=task8-calibration",
        "--label", f"spire-learn.run-id={run_id}",
        "--platform", "linux/amd64", "--network", "none", "--read-only",
        "--user", "1000:1000", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
        "--pids-limit", "512", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--workdir", "/learner",
        "--gpus", f"device={UUID}",
    ]
    environment = (
        "HOME=/tmp", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
        "PIP_CONFIG_FILE=/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK=1", "PIP_NO_INPUT=1",
        "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4", "NUMEXPR_NUM_THREADS=4",
        "SPIRE_LEARN_RUN_TASK8_CALIBRATION=1", f"SPIRE_LEARN_TASK8_GPU_UUID={UUID}",
        "SPIRE_LEARN_TASK8_EVIDENCE_DIR=/learner/evidence",
        f"SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256={release_binding['source_snapshot_sha256']}",
        f"SPIRE_LEARN_TASK8_RUNBOOK_SHA256={release_binding['runbook_sha256']}",
        f"SPIRE_LEARN_TASK8_CONFIG_SHA256={release_binding['calibration_config_sha256']}",
        f"SPIRE_LEARN_TASK8_MANIFEST_SHA256={release_binding['train_manifest_sha256']}",
        f"SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256={release_binding['qualification_report_sha256']}",
        f"SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256={release_binding['qualification_summary_sha256']}",
        f"SPIRE_LEARN_TASK8_LOCK_SHA256={release_binding['cuda_lock_sha256']}",
        f"SPIRE_LEARN_TASK8_NATIVE_SHA256={release_binding['native_sha256']}",
    )
    for value in environment:
        args.extend(("--env", value))
    for source, destination, readonly in mounts:
        mount = f"type=bind,src={source},dst={destination}"
        if readonly:
            mount += ",readonly"
        args.extend(("--mount", mount))
    args.extend(("--entrypoint", "/bin/sh", IMAGE_ALIAS, "-ec", CALIBRATION_SCRIPT))
    return tuple(args)


def assert_name_absent(raw_output: str) -> None:
    if raw_output.strip():
        raise CalibrationControlError("the exact container name is already present; run IDs are never reused")


def parse_created_container_id(raw_output: str) -> str:
    value = raw_output.strip()
    if FULL_ID_PATTERN.fullmatch(value) is None:
        raise CalibrationControlError("successful docker create did not return exactly one full container ID")
    return value


def _inspect_values(raw: str | bytes) -> dict[str, object]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        values = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationControlError("docker inspect output is malformed JSON") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise CalibrationControlError("docker inspect must return exactly one container")
    return values[0]


def assert_container_owner(raw: str | bytes, *, container_id: str, run_id: str) -> dict[str, object]:
    if FULL_ID_PATTERN.fullmatch(container_id) is None:
        raise CalibrationControlError("container ownership requires the recorded full ID")
    value = _inspect_values(raw)
    labels = value.get("Config", {}).get("Labels") if isinstance(value.get("Config"), dict) else None
    expected_name = "/" + container_name(run_id)
    if (value.get("Id") != container_id or value.get("Name") != expected_name
            or not isinstance(labels, dict)
            or labels.get("spire-learn.task") != "task8-calibration"
            or labels.get("spire-learn.run-id") != run_id):
        raise CalibrationControlError("recorded full container ID does not prove this run owns the container")
    return {"container_id": container_id, "name": expected_name, "owned": True}


def assert_container_finished(raw: str | bytes, *, container_id: str, run_id: str) -> dict[str, object]:
    owner = assert_container_owner(raw, container_id=container_id, run_id=run_id)
    value = _inspect_values(raw)
    state = value.get("State")
    if not isinstance(state, dict):
        raise CalibrationControlError("Docker inspect lacks container state")
    result = {
        **owner,
        "state": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
    }
    if state.get("Status") != "exited" or state.get("Running") is not False:
        raise CalibrationControlError("one-shot calibration container did not finish")
    if state.get("OOMKilled") is not False:
        raise CalibrationControlError("calibration container was killed by its memory cap")
    if type(state.get("ExitCode")) is not int:
        raise CalibrationControlError("container exit code is not a primitive integer")
    if state.get("ExitCode") != 0:
        raise CalibrationControlError("one-shot calibration container returned a nonzero exit code")
    return result


def container_is_running(raw: str | bytes, *, container_id: str, run_id: str) -> bool:
    assert_container_owner(raw, container_id=container_id, run_id=run_id)
    state = _inspect_values(raw).get("State")
    if not isinstance(state, dict) or type(state.get("Running")) is not bool:
        raise CalibrationControlError("Docker inspect lacks a primitive running-state value")
    return state["Running"]


def assert_container_inspect(
    raw: str | bytes,
    *,
    container_id: str,
    run_id: str,
    image_id: str,
    root: Path,
    output_child: Path,
) -> dict[str, object]:
    owner = assert_container_owner(raw, container_id=container_id, run_id=run_id)
    value = _inspect_values(raw)
    config = value.get("Config")
    host = value.get("HostConfig")
    mounts = value.get("Mounts")
    if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(mounts, list):
        raise CalibrationControlError("Docker inspect lacks Config, HostConfig, or Mounts")
    if value.get("Image") != image_id:
        raise CalibrationControlError("container image ID differs from the preflighted local alias")
    if (config.get("User") != "1000:1000" or config.get("WorkingDir") != "/learner"
            or config.get("Entrypoint") != ["/bin/sh"]
            or config.get("Cmd") != ["-ec", CALIBRATION_SCRIPT]):
        raise CalibrationControlError("container user, working directory, or command differs from the reviewed plan")
    env_raw = config.get("Env")
    if not isinstance(env_raw, list):
        raise CalibrationControlError("container environment is missing")
    environment = {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in env_raw if isinstance(entry, str) and "=" in entry
    }
    required_environment = {
        "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": "/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4", "SPIRE_LEARN_RUN_TASK8_CALIBRATION": "1",
        "SPIRE_LEARN_TASK8_GPU_UUID": UUID, "SPIRE_LEARN_TASK8_EVIDENCE_DIR": "/learner/evidence",
    }
    release_env = (
        "SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256", "SPIRE_LEARN_TASK8_RUNBOOK_SHA256",
        "SPIRE_LEARN_TASK8_CONFIG_SHA256", "SPIRE_LEARN_TASK8_MANIFEST_SHA256",
        "SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256",
        "SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256", "SPIRE_LEARN_TASK8_LOCK_SHA256",
        "SPIRE_LEARN_TASK8_NATIVE_SHA256",
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", environment.get(key, "")) is None for key in release_env):
        raise CalibrationControlError("container release binding is missing or malformed")
    required_environment.update({
        key: environment[key] for key in release_env
    })
    if (environment["SPIRE_LEARN_TASK8_LOCK_SHA256"] != LOCK_SHA256
            or environment["SPIRE_LEARN_TASK8_NATIVE_SHA256"] != NATIVE_SHA256):
        raise CalibrationControlError("container release binding has an unqualified lock/native hash")
    if any(environment.get(key) != expected for key, expected in required_environment.items()):
        raise CalibrationControlError("container environment lacks an exact isolation/thread/device value")
    if any(any(secret in name.lower() for secret in ("token", "password", "credential", "secret"))
           for name in environment):
        raise CalibrationControlError("secret-like environment variable names are forbidden")
    if (host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True
            or host.get("Privileged") is not False or host.get("AutoRemove") is not False
            or host.get("CapDrop") != ["ALL"] or host.get("SecurityOpt") != ["no-new-privileges"]
            or host.get("NanoCpus") != 4_000_000_000 or host.get("Memory") != 8 * 1024**3
            or host.get("MemorySwap") != 8 * 1024**3
            or host.get("PidsLimit") != 512
            or host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=64m"}):
        raise CalibrationControlError("container resource or privilege bounds differ")
    def absent(value: object) -> bool:
        return value is None or value == [] or value == {}
    if (not absent(host.get("Devices")) or not absent(host.get("CapAdd"))
            or not absent(host.get("GroupAdd")) or not absent(host.get("DeviceCgroupRules"))):
        raise CalibrationControlError("container may not add devices, capabilities, or groups")
    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise CalibrationControlError("container must request only one approved GPU UUID")
    request = requests[0]
    if (not isinstance(request, dict) or request.get("DeviceIDs") != [UUID]
            or request.get("Capabilities") != [["gpu"]]
            or request.get("Count") not in (None, 0)
            or not absent(request.get("Options"))):
        raise CalibrationControlError("container GPU request differs from the exact UUID-only request")
    expected_mounts = sorted(
        (str(source.resolve(strict=True)), destination, not readonly)
        for source, destination, readonly in build_mounts(root, output_child)
    )
    observed_mounts: list[tuple[str, str, bool]] = []
    for mount in mounts:
        if (not isinstance(mount, dict) or mount.get("Type") != "bind"
                or not isinstance(mount.get("Source"), str)
                or not isinstance(mount.get("Destination"), str)
                or type(mount.get("RW")) is not bool):
            raise CalibrationControlError("container bind mount is malformed or not a bind mount")
        observed_mounts.append((str(Path(mount["Source"]).resolve(strict=True)),
                                mount["Destination"], mount["RW"]))
    if sorted(observed_mounts) != expected_mounts:
        raise CalibrationControlError("container mount table differs from the exact read-only allowlist")
    if sum(1 for _source, _dest, writable in observed_mounts if writable) != 1:
        raise CalibrationControlError("only the one fresh output child may be writable")
    return {**owner, "image_id": image_id, "mount_count": len(mounts),
            "gpu_uuid": UUID, "network": "none", "read_only_rootfs": True, "user": "1000:1000"}


def verify_unittest_transcript(transcript: str) -> dict[str, object]:
    try:
        return setup.parse_unittest_transcript(transcript, expected_labels=(TEST_LABEL,))
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error


def write_closed_inventory(
    roots: Sequence[Path],
    output_path: Path,
    *,
    all_logs_closed: bool,
) -> dict[str, object]:
    """Freeze only complete roots after the wrapper has closed every log descriptor."""
    if all_logs_closed is not True:
        raise CalibrationControlError("evidence inventory is allowed only after every log is closed")
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise CalibrationControlError("closed evidence inventory path already exists")
    normalized = tuple(_canonical(root, kind="directory") for root in roots)
    if len(set(normalized)) != len(normalized):
        raise CalibrationControlError("closed evidence inventory roots must be unique")
    for index, left in enumerate(normalized):
        for right in normalized[index + 1:]:
            if left in right.parents or right in left.parents:
                raise CalibrationControlError("closed evidence inventory roots must not overlap")
    output_parent = _canonical(output_path.parent, kind="directory")
    if output_path.parent != output_parent:
        raise CalibrationControlError("closed evidence inventory parent must be canonical")
    output_path = output_parent / output_path.name
    if any(root == output_parent or root in output_parent.parents for root in normalized):
        raise CalibrationControlError("closed evidence inventory must be outside all inventoried roots")
    entries: list[dict[str, object]] = []
    for root in normalized:
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(current)
            relative_dir = directory.relative_to(root).as_posix()
            entries.append({"root": str(root), "path": "." if relative_dir == "." else relative_dir,
                            "kind": "directory", "mode": f"0{directory.stat(follow_symlinks=False).st_mode & 0o777:o}"})
            for name in tuple(dirnames):
                item = directory / name
                if item.is_symlink() or not item.is_dir():
                    raise CalibrationControlError(f"inventory rejects directory symlink/special file: {item}")
            for name in sorted(filenames):
                item = directory / name
                if item.is_symlink() or not item.is_file():
                    raise CalibrationControlError(f"inventory rejects symlink/special file: {item}")
                info = item.stat(follow_symlinks=False)
                entries.append({"root": str(root), "path": item.relative_to(root).as_posix(),
                                "kind": "file", "mode": f"0{info.st_mode & 0o777:o}",
                                "size_bytes": info.st_size, "sha256": _sha256(item)})
    entries.sort(key=lambda row: (str(row["root"]), str(row["path"]), str(row["kind"])))
    value = {"schema_id": "spire-learn-task8-calibration-closed-inventory-v1",
             "roots": [str(root) for root in normalized], "entries": entries,
             "logs_closed_before_inventory": True}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(payload) > 16 * 1024**2:
        raise CalibrationControlError("closed output inventory exceeds its 16 MiB evidence reservation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise CalibrationControlError("closed evidence inventory path already exists") from error
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(output_path), "entry_count": len(entries),
            "sha256": hashlib.sha256(payload).hexdigest()}


def host_lifecycle_record(
    started_epoch: float,
    finished_epoch: float,
    *,
    wrapper_exit_code: int,
    cleanup_failed: bool,
    release_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    numbers = (started_epoch, finished_epoch)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in numbers):
        raise CalibrationControlError("host lifecycle timestamps must be finite numbers")
    if finished_epoch < started_epoch or type(wrapper_exit_code) is not int or type(cleanup_failed) is not bool:
        raise CalibrationControlError("host lifecycle status or timestamp ordering is malformed")
    elapsed = float(finished_epoch - started_epoch)
    record: dict[str, object] = {
        "schema_id": "spire-learn-task8-calibration-host-lifecycle-v1",
        "status": "complete" if wrapper_exit_code == 0 and not cleanup_failed and elapsed <= 600 else "failed",
        "wrapper_exit_code": wrapper_exit_code,
        "cleanup_failed": cleanup_failed,
        "started_epoch_seconds": float(started_epoch),
        "finished_epoch_seconds": float(finished_epoch),
        "elapsed_seconds_including_container_cleanup_and_inventory": elapsed,
        "cohort_soft_deadline_seconds": 300,
        "wrapper_term_deadline_seconds": 510,
        "wrapper_kill_deadline_seconds": 600,
        "wrapper_elapsed_after_300_seconds": max(0.0, elapsed - 300.0),
        "hard_overshoot_seconds": max(0.0, elapsed - 600.0),
        "timing_boundary": "after container cleanup and closed-output inventory; before this admin receipt write",
        "receipt_path": "host-lifecycle.json",
    }
    if release_binding is None:
        record["provenance_status"] = "release_not_verified"
    else:
        head = release_binding.get("learner_head")
        dirty = release_binding.get("learner_dirty")
        required_hashes = (
            "source_snapshot_sha256", "release_sha256", "release_binding_sha256",
            "previous_release_sha256", *PREDECESSOR_EVIDENCE_HASHES,
        )
        if (not isinstance(head, str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None
                or type(dirty) is not bool
                or type(release_binding.get("continuation_generation")) is not int
                or release_binding.get("continuation_generation") != CONTINUATION_GENERATION
                or release_binding.get("previous_run_id") != PREDECESSOR_RUN_ID
                or release_binding.get("previous_release_sha256") != PREDECESSOR_RELEASE_SHA256
                or any(not isinstance(release_binding.get(key), str)
                       or re.fullmatch(r"[0-9a-f]{64}", release_binding[key]) is None
                       for key in required_hashes)):
            raise CalibrationControlError("host lifecycle Git/release provenance is malformed")
        record.update({
            "provenance_status": "verified_parent_release",
            "learner_head": head,
            "learner_dirty": dirty,
            "source_snapshot_sha256": release_binding["source_snapshot_sha256"],
            "release_sha256": release_binding["release_sha256"],
            "release_binding_sha256": release_binding["release_binding_sha256"],
            "continuation": {
                "generation": CONTINUATION_GENERATION,
                "previous_run_id": PREDECESSOR_RUN_ID,
                "previous_release_sha256": PREDECESSOR_RELEASE_SHA256,
                "previous_outcome": "one_enabled_test_failed_before_calibration_effects",
                **PREDECESSOR_EVIDENCE_HASHES,
            },
        })
    return record


def write_host_lifecycle(
    output_path: Path,
    started_epoch: float,
    *,
    wrapper_exit_code: int,
    cleanup_failed: bool,
    finished_epoch: float | None = None,
    release_binding_path: Path | None = None,
) -> dict[str, object]:
    release_binding: dict[str, object] | None = None
    if release_binding_path is not None:
        candidate = Path(release_binding_path)
        if candidate.is_symlink():
            raise CalibrationControlError("host lifecycle release binding may not be a symlink")
        if candidate.exists():
            path = _canonical(candidate, kind="file")
            release_binding = _json_object(path)
            release_binding["release_binding_sha256"] = _sha256(path)
    record = host_lifecycle_record(
        started_epoch,
        time.time() if finished_epoch is None else finished_epoch,
        wrapper_exit_code=wrapper_exit_code,
        cleanup_failed=cleanup_failed,
        release_binding=release_binding,
    )
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    _write_exclusive(Path(output_path), payload)
    return record


def verify_source_snapshot(root: Path, snapshot_path: Path) -> dict[str, object]:
    root = _repo_root(root)
    snapshot_path = _canonical(snapshot_path, kind="file")
    try:
        result = setup.verify_source_snapshot(snapshot_path, root)
    except setup.Task8ControlError as error:
        raise CalibrationControlError(str(error)) from error
    value = _json_object(snapshot_path)
    if value.get("scope_roots") != list(SOURCE_SCOPES):
        raise CalibrationControlError("source snapshot does not bind the exact calibration projection")
    return {**result, "snapshot_sha256": _sha256(snapshot_path)}


def _write_exclusive(path: Path, data: bytes) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise CalibrationControlError(f"output is immutable and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise CalibrationControlError(f"output is immutable and already exists: {path}") from error
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise CalibrationControlError(
            f"failed to fsync output parent directory after exclusive create: {path.parent}"
        ) from error


def _read_json(path: Path) -> dict[str, object]:
    return _json_object(path)


def _cmd_capture(args: argparse.Namespace) -> int:
    print(json.dumps(capture_source_snapshot(Path(args.root), Path(args.output)), sort_keys=True))
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    value = verify_release(Path(args.root), Path(args.release), Path(args.snapshot))
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if args.output:
        _write_exclusive(Path(args.output), encoded)
    else:
        sys.stdout.buffer.write(encoded)
    return 0


def _cmd_predecessor(args: argparse.Namespace) -> int:
    print(json.dumps(verify_predecessor_evidence(Path(args.root)), sort_keys=True))
    return 0


def _cmd_gpu(args: argparse.Namespace) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=8,
    )
    print(json.dumps(verify_gpu_inventory_csv(result.stdout), sort_keys=True))
    return 0


def _cmd_storage(args: argparse.Namespace) -> int:
    print(json.dumps(verify_host_storage(
        Path(args.root), Path(args.run_root), Path(args.control_root), Path(args.admin_root),
        Path(args.output_child),
        reservation_bytes=args.reservation_bytes,
    ), sort_keys=True))
    return 0


def _cmd_snapshot_check(args: argparse.Namespace) -> int:
    print(json.dumps(verify_source_snapshot(Path(args.root), Path(args.snapshot)), sort_keys=True))
    return 0


def _cmd_image(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.inspect), kind="file").read_bytes()
    value = verify_image_alias(raw)
    if args.output:
        _write_exclusive(Path(args.output), json.dumps(value, sort_keys=True).encode() + b"\n")
    else:
        print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_name(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.output), kind="file").read_text(encoding="utf-8")
    assert_name_absent(raw)
    return 0


def _cmd_create_id(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.output), kind="file").read_text(encoding="utf-8")
    print(parse_created_container_id(raw))
    return 0


def _cmd_claim_continuation(args: argparse.Namespace) -> int:
    value = claim_continuation_release(
        Path(args.root), Path(args.release), Path(args.snapshot), Path(args.marker), args.run_id,
        expected_release_sha256=args.expected_release_sha256,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_assert_claim_release_binding(args: argparse.Namespace) -> int:
    assert_claim_release_binding(Path(args.claim), args.expected_release_sha256)
    return 0


def _cmd_argv(args: argparse.Namespace) -> int:
    binding = _read_json(Path(args.release_binding))
    argv = build_container_argv(
        Path(args.root), Path(args.output_child), args.run_id,
        image_id=args.image_id, release_binding=binding,
    )
    encoded = b"\0".join(item.encode() for item in argv) + b"\0"
    _write_exclusive(Path(args.output), encoded)
    return 0


def _cmd_owner(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.inspect), kind="file").read_bytes()
    value = assert_container_owner(raw, container_id=args.container_id, run_id=args.run_id)
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.inspect), kind="file").read_bytes()
    value = assert_container_inspect(
        raw, container_id=args.container_id, run_id=args.run_id,
        image_id=args.image_id, root=Path(args.root), output_child=Path(args.output_child),
    )
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_finished(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.inspect), kind="file").read_bytes()
    value = assert_container_finished(raw, container_id=args.container_id, run_id=args.run_id)
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_running(args: argparse.Namespace) -> int:
    raw = _canonical(Path(args.inspect), kind="file").read_bytes()
    value = container_is_running(raw, container_id=args.container_id, run_id=args.run_id)
    print("true" if value else "false")
    return 0


def _cmd_transcript(args: argparse.Namespace) -> int:
    value = verify_unittest_transcript(_canonical(Path(args.transcript), kind="file").read_text(encoding="utf-8"))
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    roots = tuple(Path(value) for value in args.roots)
    value = write_closed_inventory(
        roots, Path(args.output), all_logs_closed=args.logs_closed,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_host_lifecycle(args: argparse.Namespace) -> int:
    value = write_host_lifecycle(
        Path(args.output), float(args.started_epoch),
        wrapper_exit_code=args.wrapper_exit_code, cleanup_failed=args.cleanup_failed == "1",
        release_binding_path=Path(args.release_binding) if args.release_binding else None,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


def _cmd_main() -> int:
    parser = argparse.ArgumentParser(prog="task8_calibration_run_helpers.py")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture-source-snapshot")
    capture.add_argument("root")
    capture.add_argument("output")
    capture.set_defaults(handler=_cmd_capture)
    release = commands.add_parser("verify-release")
    release.add_argument("root")
    release.add_argument("release")
    release.add_argument("snapshot")
    release.add_argument("--output")
    release.set_defaults(handler=_cmd_release)
    predecessor = commands.add_parser("verify-predecessor")
    predecessor.add_argument("root")
    predecessor.set_defaults(handler=_cmd_predecessor)
    gpu = commands.add_parser("verify-gpu")
    gpu.set_defaults(handler=_cmd_gpu)
    storage = commands.add_parser("verify-storage")
    storage.add_argument("root")
    storage.add_argument("run_root")
    storage.add_argument("control_root")
    storage.add_argument("admin_root")
    storage.add_argument("output_child")
    storage.add_argument("--reservation-bytes", type=int)
    storage.set_defaults(handler=_cmd_storage)
    snapshot_check = commands.add_parser("verify-source-snapshot")
    snapshot_check.add_argument("root")
    snapshot_check.add_argument("snapshot")
    snapshot_check.set_defaults(handler=_cmd_snapshot_check)
    image = commands.add_parser("verify-image-alias")
    image.add_argument("inspect")
    image.add_argument("--output")
    image.set_defaults(handler=_cmd_image)
    name = commands.add_parser("assert-name-absent")
    name.add_argument("output")
    name.set_defaults(handler=_cmd_name)
    create_id = commands.add_parser("parse-created-id")
    create_id.add_argument("output")
    create_id.set_defaults(handler=_cmd_create_id)
    claim = commands.add_parser("claim-continuation")
    claim.add_argument("root")
    claim.add_argument("release")
    claim.add_argument("snapshot")
    claim.add_argument("marker")
    claim.add_argument("run_id")
    claim.add_argument("--expected-release-sha256", required=True)
    claim.set_defaults(handler=_cmd_claim_continuation)
    claim_binding = commands.add_parser("assert-claim-release-binding")
    claim_binding.add_argument("claim")
    claim_binding.add_argument("expected_release_sha256")
    claim_binding.set_defaults(handler=_cmd_assert_claim_release_binding)
    argv = commands.add_parser("write-container-argv")
    argv.add_argument("root")
    argv.add_argument("output_child")
    argv.add_argument("run_id")
    argv.add_argument("image_id")
    argv.add_argument("release_binding")
    argv.add_argument("output")
    argv.set_defaults(handler=_cmd_argv)
    owner = commands.add_parser("assert-owner")
    owner.add_argument("inspect")
    owner.add_argument("container_id")
    owner.add_argument("run_id")
    owner.set_defaults(handler=_cmd_owner)
    inspect = commands.add_parser("assert-inspect")
    inspect.add_argument("inspect")
    inspect.add_argument("container_id")
    inspect.add_argument("run_id")
    inspect.add_argument("image_id")
    inspect.add_argument("root")
    inspect.add_argument("output_child")
    inspect.set_defaults(handler=_cmd_inspect)
    finished = commands.add_parser("assert-finished")
    finished.add_argument("inspect")
    finished.add_argument("container_id")
    finished.add_argument("run_id")
    finished.set_defaults(handler=_cmd_finished)
    running = commands.add_parser("container-running")
    running.add_argument("inspect")
    running.add_argument("container_id")
    running.add_argument("run_id")
    running.set_defaults(handler=_cmd_running)
    transcript = commands.add_parser("verify-transcript")
    transcript.add_argument("transcript")
    transcript.set_defaults(handler=_cmd_transcript)
    inventory = commands.add_parser("inventory-closed")
    inventory.add_argument("output")
    inventory.add_argument("--logs-closed", action="store_true", required=True)
    inventory.add_argument("roots", nargs="+")
    inventory.set_defaults(handler=_cmd_inventory)
    lifecycle = commands.add_parser("write-host-lifecycle")
    lifecycle.add_argument("output")
    lifecycle.add_argument("started_epoch")
    lifecycle.add_argument("wrapper_exit_code", type=int)
    lifecycle.add_argument("cleanup_failed", choices=("0", "1"))
    lifecycle.add_argument("--release-binding")
    lifecycle.set_defaults(handler=_cmd_host_lifecycle)
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Task8 calibration control: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cmd_main())
