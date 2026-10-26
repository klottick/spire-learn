"""Pure host-side controls for the held Task 7 CPU smoke runbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import shutil
import stat
import textwrap
from typing import Any


IMAGE_DIGEST = "57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
IMAGE_REFERENCE = f"docker.io/library/python@sha256:{IMAGE_DIGEST}"
LOCK_SHA256 = "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0"
NATIVE_SHA256 = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
EXPECTED_TESTS = (
    "test_run_bounded_training_and_reload_parity",
    "test_standalone_bounded_training_and_reload_parity",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "BLIS_NUM_THREADS": "4",
}
RUNTIME_PREFLIGHT_CODE = textwrap.dedent("""\
    import importlib.metadata as metadata
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import torch
    versions = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": metadata.version("torch"),
        "numpy": metadata.version("numpy"),
        "gymnasium": metadata.version("gymnasium"),
    }
    expected = {"python": "3.12.15", "torch": "2.14.1+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0"}
    if versions != expected or torch.version.cuda is not None:
        raise RuntimeError("installed CPU runtime identity differs: " + repr(versions))
    if any(os.environ.get(name) != value for name, value in {
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4", "BLIS_NUM_THREADS": "4",
    }.items()):
        raise RuntimeError("bounded BLAS/OpenMP thread environment is missing")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch thread limits differ from the approved CPU profile")
    pip = subprocess.run([sys.executable, "-B", "-m", "pip", "check"], capture_output=True, text=True)
    sys.stderr.write(pip.stdout)
    sys.stderr.write(pip.stderr)
    if pip.returncode != 0:
        raise RuntimeError("same-container pip check failed")
    thread_environment = {name: os.environ[name] for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    )}
    preflight_process = {
        "role": "preflight_process", "process_id": os.getpid(),
        **versions, "torch_cuda": None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": thread_environment,
        "configuration_boundary": "preflight_process_before_pip_check",
    }
    identity = {
        **versions, "torch_cuda": None,
        "preflight_process": preflight_process, "training_process": None,
        "pip_check": "passed",
        "check_order": [
            "installed_versions", "preflight_process_thread_gate", "pip_check",
            "training_process_thread_gate", "unittest",
        ],
    }
    target = Path("/learner/artifacts/development/neural-combat/runtime-identity.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\\n")
        stream.flush()
        os.fsync(stream.fileno())
    sys.stderr.write("TASK7_RUNTIME_IDENTITY=" + json.dumps(identity, sort_keys=True) + "\\n")
    sys.stderr.write("TASK7_PREFLIGHT_PROCESS_IDENTITY=" + json.dumps(preflight_process, sort_keys=True) + "\\n")
    sys.stderr.write("TASK7_PIP_CHECK=passed\\n")
""")
CONTAINER_SCRIPT = (
    "set -eu\n"
    + "export " + " ".join(f"{name}={value}" for name, value in THREAD_ENVIRONMENT.items()) + "\n"
    + "/work/venv-cpu/bin/python -B -c \"$1\" >&2\n"
    + "exec /work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_training -f -v"
)
CONTAINER_COMMAND_ARGS = ("-ec", CONTAINER_SCRIPT, "task7", RUNTIME_PREFLIGHT_CODE)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")

TASK7_FILES = (
    "AGENTS.md",
    "README.md",
    "docs/development.md",
    "docs/reference/README.md",
    "artifacts/development/neural-combat/future-task-rulings.md",
    "artifacts/development/neural-combat/cpu-runtime-location.json",
    "artifacts/development/neural-combat/cpu-runtime-report.md",
    "artifacts/development/neural-combat/task-7/task-7-brief.md",
    "artifacts/development/neural-combat/task-7/collector-consumer-rulings.md",
    "artifacts/development/neural-combat/task-7/task-7-review.md",
    "artifacts/development/neural-combat/task-7/task-7-review-package.diff",
    "artifacts/development/neural-combat/task-7/task-7-first-freeze.zip",
    "artifacts/development/neural-combat/task-7/parent-first-review-hashes.json",
    "artifacts/development/neural-combat/task-7/parent-runner-counterexamples.json",
    "artifacts/development/neural-combat/task-7/parent-short-rollout-update-counterexample-v2.json",
    "artifacts/development/neural-combat/task-7/parent-output-scaffold-reproducer.json",
    "artifacts/development/neural-combat/task-7/task-7-fix-round-1.md",
    "artifacts/development/neural-combat/task-7/task-7-fix-round-1.diff",
    "artifacts/development/neural-combat/task-7/task-7-report.md",
    "src/spire_learn/neural/collector.py",
    "src/spire_learn/neural/evaluation.py",
    "src/spire_learn/neural/runner.py",
    "src/spire_learn/neural/cli.py",
    "src/spire_learn/neural/__main__.py",
    "tests/unit/test_neural_collector.py",
    "tests/unit/test_neural_runner.py",
    "tests/unit/test_neural_cli.py",
    "tests/unit/test_neural_task7_run_helpers.py",
    "tests/integration/test_neural_training.py",
    "configs/neural/standalone.toml",
    "configs/neural/run.toml",
    "configs/neural/standalone-selection.toml",
    "configs/neural/run-selection.toml",
    "configs/neural/standalone-final.toml",
    "configs/neural/run-final.toml",
    "docs/neural-combat.md",
    "scripts/neural/task7_training_run_helpers.py",
    "scripts/neural/run-task7-training-smoke.sh",
    "scripts/neural/task7-training-smoke-runbook.md",
)
PRESERVED_FILES = {
    "artifacts/development/foundation/task-6-report.md":
        "e7ea42e58871d492d4b5ca2e408f782155526fe80258df1a44bad2cd02285d2a",
    "task-6-core-review.diff":
        "a52df58775d2c5fc0d8f1cbbe1946af787839e5d1379cd752e2663a4557edc29",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_tree(output_child: Path) -> tuple[Path, ...]:
    """Create the exact fresh mode output roots used by the smoke container."""
    output_child = Path(output_child)
    if output_child.is_symlink() or not output_child.is_dir():
        raise ValueError("fresh output child must be an existing regular directory")
    root = output_child.resolve(strict=True)
    relative_paths = (
        Path("development"),
        Path("development/neural-combat"),
        Path("development/neural-combat/runs"),
        Path("development/neural-combat/runs/standalone"),
        Path("development/neural-combat/runs/run"),
    )
    created: list[Path] = []
    for relative in relative_paths:
        path = root / relative
        path.mkdir(mode=0o700)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"output scaffold path is not a regular directory: {relative}")
        path.chmod(0o700)
        created.append(path)
    return tuple(created)


def record_training_process_identity(torch_runtime: Any, identity_path: Path) -> dict[str, object]:
    """Set and record CPU thread limits in the active gated unittest process."""
    identity_path = Path(identity_path)
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError("preflight runtime identity must be an existing regular file")
    thread_environment = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if thread_environment != THREAD_ENVIRONMENT:
        raise RuntimeError("training process thread environment differs from the approved CPU profile")

    torch_runtime.set_num_threads(4)
    torch_runtime.set_num_interop_threads(1)
    torch_threads = torch_runtime.get_num_threads()
    torch_interop_threads = torch_runtime.get_num_interop_threads()
    torch_version = str(getattr(torch_runtime, "__version__", ""))
    torch_cuda = getattr(getattr(torch_runtime, "version", None), "cuda", "missing")
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    if (python_version != "3.12.15" or torch_version != "2.14.1+cpu"
            or torch_cuda is not None or torch_threads != 4 or torch_interop_threads != 1):
        raise RuntimeError("training process runtime/thread identity differs from the approved CPU profile")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict) or not _preflight_identity_is_qualified(identity):
        raise ValueError("preflight runtime identity is not qualified")
    if identity.get("training_process") is not None:
        raise ValueError("training process identity is already recorded")
    record = {
        "role": "unittest_training_process", "process_id": os.getpid(),
        "python": python_version, "torch": torch_version, "torch_cuda": torch_cuda,
        "torch_num_threads": torch_threads,
        "torch_num_interop_threads": torch_interop_threads,
        "thread_environment": thread_environment,
        "configuration_boundary": "setUpClass_before_model_or_environment_factory",
    }
    identity["training_process"] = record
    with identity_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(identity, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def source_snapshot(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    head = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain=v1",
         "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    files: dict[str, str] = {}
    evidence_root = root / "artifacts/development/neural-combat/task-7/logs"
    evidence_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(evidence_root.glob("task7-*"))
        if path.is_file() and not path.is_symlink()
    ) if evidence_root.is_dir() else ()
    for relative in (*TASK7_FILES, *evidence_files):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Task 7 source path is missing or not a regular file: {relative}")
        files[relative] = sha256(path)
    preserved: dict[str, str] = {}
    for relative, expected in PRESERVED_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"protected pre-existing path is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"protected pre-existing file changed: {relative}")
        preserved[relative] = actual
    mounted_inputs = mounted_input_inventory(root)
    return {"head": head, "status_porcelain": status, "task7_files_sha256": files,
            "mounted_inputs": mounted_inputs,
            "preserved_files_sha256": preserved}


def mounted_input_inventory(root: Path) -> dict[str, dict[str, object]]:
    """Bind every regular file and Git-compatible mode in mounted inputs."""
    root = Path(root).resolve(strict=True)
    inventory: dict[str, dict[str, object]] = {}
    for mount_name in ("src", "tests", "configs"):
        mount_root = root / mount_name
        if mount_root.is_symlink() or not mount_root.is_dir():
            raise ValueError(f"mounted input root is missing or not a directory: {mount_name}")
        for current, directory_names, file_names in os.walk(mount_root, followlinks=False):
            current_path = Path(current)
            for name in tuple(directory_names):
                path = current_path / name
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"mounted input contains an unsupported directory entry: {path}")
            for name in file_names:
                path = current_path / name
                if path.is_symlink():
                    raise ValueError(f"mounted input contains a symlink: {path}")
                file_stat = path.lstat()
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError(f"mounted input is not a regular file: {path}")
                mode = "100755" if file_stat.st_mode & 0o111 else "100644"
                relative = path.relative_to(root).as_posix()
                inventory[relative] = {"mode": mode, "sha256": sha256(path)}
    if not inventory:
        raise ValueError("mounted input inventory is empty")
    return dict(sorted(inventory.items()))


def compare_snapshots(before: Path, after: Path) -> None:
    left = json.loads(Path(before).read_text(encoding="utf-8"))
    right = json.loads(Path(after).read_text(encoding="utf-8"))
    if left != right:
        raise ValueError("source HEAD, working-tree status, Task 7 hashes, or protected hashes changed")


def snapshot_sha256(path: Path) -> str:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_release(record_path: Path, snapshot_path: Path) -> dict[str, object]:
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    required = {
        "decision": "internal_cpu_smoke_release",
        "first_delivery_review": "passed",
        "fresh_independent_review": "passed",
        "task3_native_conformance": "accepted",
        "task5_native_collection": "accepted",
        "task7_source_snapshot_sha256": snapshot_sha256(snapshot_path),
    }
    if not isinstance(record, dict) or any(record.get(name) != value for name, value in required.items()):
        raise ValueError("parent release record is absent, incomplete, or does not match this source freeze")
    return record


def verify_runtime_identity(location_path: Path, lock_path: Path, native_path: Path) -> dict[str, object]:
    location = json.loads(Path(location_path).read_text(encoding="utf-8"))
    expected_venv = Path(location["venv_host_path"])
    venv = expected_venv.resolve(strict=True)
    if not venv.is_dir():
        raise ValueError("retained CPU runtime location is not the reviewed one")
    python_link = venv / "bin/python"
    if not python_link.is_symlink() or os.readlink(python_link) != "/usr/local/bin/python":
        raise ValueError("retained CPU venv interpreter link changed")
    expected_lock = Path(location["runtime_root"]) / "data/requirements-cpu.lock"
    if Path(lock_path).resolve() != expected_lock.resolve() or Path(lock_path).is_symlink() \
            or sha256(lock_path) != LOCK_SHA256:
        raise ValueError("retained CPU lock SHA-256 changed")
    native = Path(native_path)
    if (native.is_symlink() or not native.is_file() or not native.resolve().is_relative_to(venv)
            or sha256(native) != NATIVE_SHA256):
        raise ValueError("retained CPU native extension SHA-256 changed")
    expected_native = "/work/venv-cpu/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
    return {
        "runtime_root": str(Path(location["runtime_root"]).resolve()),
        "runtime_state": location.get("state"),
        "venv": str(venv),
        "lock_sha256": sha256(lock_path),
        "native_extension": str(native.resolve()),
        "native_sha256": sha256(native),
        "container_native_path": expected_native,
        "image_reference": IMAGE_REFERENCE,
        "image_digest": IMAGE_DIGEST,
    }


def verify_image_repo_digests(raw: str) -> None:
    values = json.loads(raw)
    accepted = {f"python@sha256:{IMAGE_DIGEST}", IMAGE_REFERENCE}
    if not isinstance(values, list) or not accepted.intersection(values):
        raise ValueError("pinned Python image digest is not present in local RepoDigests")


def verify_image_identity(raw: str) -> dict[str, object]:
    parts = raw.strip().split("|", 3)
    if len(parts) != 4:
        raise ValueError("image inspection must bind immutable ID, OS, architecture, and RepoDigests")
    image_id, os_name, architecture, digests_raw = parts
    digests = json.loads(digests_raw)
    if _IMAGE_ID.fullmatch(image_id) is None:
        raise ValueError("container image ID must be an immutable sha256 identifier")
    if os_name != "linux" or architecture != "amd64":
        raise ValueError("pinned CPU image must be linux/amd64")
    verify_image_repo_digests(json.dumps(digests))
    return {"image_id": image_id, "os": os_name, "architecture": architecture,
            "repo_digests": digests, "image_reference": IMAGE_REFERENCE}


def verify_free_space(path: Path, minimum_gib: int = 30) -> int:
    free_bytes = shutil.disk_usage(path).free
    minimum = minimum_gib * 1024**3
    if free_bytes < minimum:
        raise ValueError(f"only {free_bytes} bytes free; require at least {minimum} bytes")
    return free_bytes


def verify_container_status(raw: str) -> None:
    if raw.strip() != "exited 0":
        raise ValueError(f"container did not exit cleanly: {raw.strip()!r}")


def parse_transcript(stdout: str, stderr: str = "") -> dict[str, object]:
    text = f"{stdout}\n{stderr}"
    found = []
    for line in text.splitlines():
        if line.startswith("test_"):
            method = line.split(" ", 1)[0]
            found.append(method)
    if tuple(found) != EXPECTED_TESTS:
        raise ValueError(f"smoke transcript must contain exactly {EXPECTED_TESTS}, got {tuple(found)}")
    if "Ran 2 tests" not in text or "\nOK" not in text:
        raise ValueError("smoke transcript did not report exactly two successful tests")
    if re.search(r"\b(skipped|FAILED|ERROR)\b", text, re.IGNORECASE):
        raise ValueError("smoke transcript contains a skip, failure, or error")
    for marker in (
        "TASK7_RUNTIME_IDENTITY=", "TASK7_PREFLIGHT_PROCESS_IDENTITY=",
        "TASK7_PIP_CHECK=passed", "TASK7_TRAINING_PROCESS_IDENTITY=",
    ):
        if marker not in stderr:
            label = "training process thread identity" if "TRAINING_PROCESS" in marker else "same-container runtime preflight"
            raise ValueError(f"{label} marker is missing: {marker}")
    first_test = text.find(EXPECTED_TESTS[0])
    if first_test < 0 or any(text.find(marker) > first_test for marker in (
        "TASK7_RUNTIME_IDENTITY=", "TASK7_PREFLIGHT_PROCESS_IDENTITY=",
        "TASK7_PIP_CHECK=passed", "TASK7_TRAINING_PROCESS_IDENTITY=",
    )):
        raise ValueError("preflight and training process checks must finish before the first smoke method")
    identity_line = next((line for line in stderr.splitlines() if line.startswith("TASK7_RUNTIME_IDENTITY=")), None)
    preflight_line = next((line for line in stderr.splitlines()
                           if line.startswith("TASK7_PREFLIGHT_PROCESS_IDENTITY=")), None)
    training_line = next((line for line in stderr.splitlines()
                          if line.startswith("TASK7_TRAINING_PROCESS_IDENTITY=")), None)
    try:
        identity = json.loads(identity_line.split("=", 1)[1]) if identity_line else None
        preflight_process = json.loads(preflight_line.split("=", 1)[1]) if preflight_line else None
        training_process = json.loads(training_line.split("=", 1)[1]) if training_line else None
    except (ValueError, AttributeError) as error:
        raise ValueError("runtime or process identity marker is malformed") from error
    if (not isinstance(identity, dict)
            or identity.get("training_process") is not None
            or identity.get("preflight_process") != preflight_process):
        raise ValueError("preflight process marker does not match runtime identity")
    complete_identity = dict(identity)
    complete_identity["training_process"] = training_process
    if not runtime_identity_is_qualified(complete_identity):
        raise ValueError("same-container preflight and training process identity is not qualified")
    return {"test_ids": list(found), "test_count": 2, "skipped": 0, "failed": 0,
            "training_process_id": training_process["process_id"]}


def _thread_process_identity_is_qualified(
    process: object, *, role: str, configuration_boundary: str,
) -> bool:
    if not isinstance(process, dict):
        return False
    if (process.get("role") != role
            or type(process.get("process_id")) is not int
            or process["process_id"] <= 0
            or process.get("python") != "3.12.15"
            or process.get("torch") != "2.14.1+cpu"
            or process.get("torch_cuda") is not None
            or type(process.get("torch_num_threads")) is not int
            or process["torch_num_threads"] != 4
            or type(process.get("torch_num_interop_threads")) is not int
            or process["torch_num_interop_threads"] != 1
            or process.get("thread_environment") != THREAD_ENVIRONMENT
            or process.get("configuration_boundary") != configuration_boundary):
        return False
    return True


def _preflight_identity_is_qualified(identity: object) -> bool:
    if not isinstance(identity, dict):
        return False
    expected = {
        "python": "3.12.15", "torch": "2.14.1+cpu", "numpy": "2.5.4",
        "gymnasium": "1.4.0", "torch_cuda": None, "pip_check": "passed",
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        return False
    if identity.get("check_order") != [
        "installed_versions", "preflight_process_thread_gate", "pip_check",
        "training_process_thread_gate", "unittest",
    ]:
        return False
    return _thread_process_identity_is_qualified(
        identity.get("preflight_process"), role="preflight_process",
        configuration_boundary="preflight_process_before_pip_check",
    )


def runtime_identity_is_qualified(identity: object) -> bool:
    return (
        _preflight_identity_is_qualified(identity)
        and _thread_process_identity_is_qualified(
            identity.get("training_process"), role="unittest_training_process",
            configuration_boundary="setUpClass_before_model_or_environment_factory",
        )
    )


def _container_mounts(container: MappingLike) -> dict[str, tuple[str, bool]]:
    mounts = container.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("container inspect has no mount inventory")
    result: dict[str, tuple[str, bool]] = {}
    for item in mounts:
        if not isinstance(item, dict):
            raise ValueError("container mount inventory contains a malformed item")
        destination = item.get("Destination")
        source = item.get("Source")
        rw = item.get("RW")
        if not isinstance(destination, str) or not isinstance(source, str) or type(rw) is not bool:
            raise ValueError("container mount has invalid source/destination/RW fields")
        if item.get("Type") != "bind":
            raise ValueError("container contains a non-bind data mount")
        if destination in result:
            raise ValueError("container inspect repeats a mount destination")
        result[destination] = (str(Path(source).resolve()), rw)
    return result


MappingLike = dict[str, Any]


def assert_container_inspect(
    payload: object,
    *,
    run_id: str,
    container_id: str,
    image_identity: object,
    repo_root: Path,
    venv: Path,
    lock: Path,
    output_child: Path,
) -> dict[str, object]:
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("docker inspect must return exactly one container object")
    container = payload[0]
    if not isinstance(image_identity, dict):
        raise ValueError("verified immutable image identity is missing")
    image_id = image_identity.get("image_id")
    if (not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None
            or image_identity.get("os") != "linux"
            or image_identity.get("architecture") != "amd64"
            or image_identity.get("image_reference") != IMAGE_REFERENCE):
        raise ValueError("verified image identity is not the pinned linux/amd64 image")
    verify_image_repo_digests(json.dumps(image_identity.get("repo_digests")))
    host = container.get("HostConfig")
    config = container.get("Config")
    if not isinstance(host, dict) or not isinstance(config, dict):
        raise ValueError("docker inspect lacks container or host configuration")
    if container.get("Id") != container_id:
        raise ValueError("container full ID differs from the verified create result")
    if container.get("Image") != image_id or _IMAGE_ID.fullmatch(image_id) is None:
        raise ValueError("container immutable image ID differs from the verified linux/amd64 image")
    if config.get("Image") != IMAGE_REFERENCE or config.get("User") != "1000:1000":
        raise ValueError("container image or user differs from the approved CPU profile")
    if config.get("WorkingDir") != "/learner":
        raise ValueError("container working directory differs from the approved mount layout")
    if config.get("Entrypoint") != ["/bin/sh"]:
        raise ValueError("container entrypoint is not the preflight shell")
    expected_command = ["-ec", CONTAINER_SCRIPT, "task7", RUNTIME_PREFLIGHT_CODE]
    if config.get("Cmd") != expected_command:
        raise ValueError("container command is not the exact two-mode training-smoke invocation")
    environment = config.get("Env")
    if not isinstance(environment, list):
        raise ValueError("container environment is missing")
    required_env = {
        "SPIRE_LEARN_RUN_NEURAL_NATIVE=1",
        "SPIRE_LEARN_RUN_NEURAL_TENSOR=1",
        "PYTHONPATH=/learner/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1",
        "HOME=/tmp",
        *(f"{name}={value}" for name, value in THREAD_ENVIRONMENT.items()),
    }
    if not required_env.issubset(set(environment)):
        raise ValueError("container opt-in or Python isolation environment differs")
    if host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True:
        raise ValueError("container must have no network and a read-only root filesystem")
    if host.get("CapDrop") != ["ALL"] or "no-new-privileges" not in (host.get("SecurityOpt") or []):
        raise ValueError("container privilege controls differ")
    if host.get("Privileged") is not False:
        raise ValueError("container must not be privileged")
    if host.get("Devices") not in (None, []) or host.get("DeviceRequests") not in (None, []):
        raise ValueError("container must not request host devices or accelerators")
    if host.get("NanoCpus") != 4_000_000_000 or host.get("Memory") != 8 * 1024**3:
        raise ValueError("container CPU or memory limit differs")
    if host.get("PidsLimit") != 512 or host.get("AutoRemove") is not False:
        raise ValueError("container PID or lifecycle setting differs")
    if host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=64m"}:
        raise ValueError("container temporary filesystem differs")
    root = Path(repo_root).resolve()
    expected_mounts = {
        "/learner/src": (str((root / "src").resolve()), False),
        "/learner/tests": (str((root / "tests").resolve()), False),
        "/learner/configs": (str((root / "configs").resolve()), False),
        "/learner/scripts/neural/task7_training_run_helpers.py": (
            str((root / "scripts/neural/task7_training_run_helpers.py").resolve()), False,
        ),
        "/learner/artifacts": (str(Path(output_child).resolve()), True),
        "/work/venv-cpu": (str(Path(venv).resolve()), False),
        "/work/requirements-cpu.lock": (str(Path(lock).resolve()), False),
    }
    actual_mounts = _container_mounts(container)
    if actual_mounts != expected_mounts:
        raise ValueError(f"container mount table differs: {actual_mounts!r}")
    labels = config.get("Labels")
    if not isinstance(labels, dict) or labels.get("spire-learn.task") != "task7-training-smoke" \
            or labels.get("spire-learn.run-id") != run_id:
        raise ValueError("container ownership labels do not match this unique Task 7 run")
    return {"container_id": container_id, "image": config["Image"], "image_id": image_id,
            "platform": "linux/amd64", "user": config["User"],
            "network": host["NetworkMode"], "readonly_rootfs": host["ReadonlyRootfs"],
            "mounts": {path: {"source": item[0], "read_write": item[1]}
                       for path, item in sorted(actual_mounts.items())}}


def resolve_container_id(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) != 1 or _HEX64.fullmatch(lines[0]) is None:
        raise ValueError("exact-name container lookup did not return zero or one full container ID")
    return lines[0]


def assert_smoke_artifacts(output_root: Path) -> dict[str, object]:
    summaries = sorted(Path(output_root).rglob("summary.json"))
    if len(summaries) != 2:
        raise ValueError(f"expected two completed Task 7 run summaries, got {len(summaries)}")
    identity_path = Path(output_root) / "development/neural-combat/runtime-identity.json"
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError("same-container installed runtime identity artifact is missing")
    runtime_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not runtime_identity_is_qualified(runtime_identity):
        raise ValueError("same-container installed runtime identity artifact is not qualified")
    totals = {"learned_decisions": 0, "environment_slots": 0,
              "optimizer_attempted_steps": 0, "optimizer_accepted_steps": 0}
    modes: set[str] = set()
    checkpoints = []
    for path in summaries:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        summary = envelope.get("summary") if isinstance(envelope, dict) else None
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            raise ValueError(f"Task 7 training artifact is incomplete: {path}")
        mode = summary.get("mode")
        if summary.get("lifecycle_receipt_required") is not True or summary.get("exit_code") != 1:
            raise ValueError(f"{path} does not fail closed while its lifecycle receipt is pending")
        receipt_path = path.parent / "lifecycle.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(f"{path.parent} has no final lifecycle receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("schema_id") != "spire-learn-neural-lifecycle-v1"
                or receipt.get("status") != "complete" or receipt.get("exit_code") != 0
                or receipt.get("receipt_path") != "lifecycle.json"):
            raise ValueError(f"{mode} final lifecycle receipt is not successful")
        for field in ("elapsed_seconds", "wall_overshoot_seconds"):
            value = receipt.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"lifecycle receipt {field} is invalid")
        if "after runtime and artifact writer close" not in receipt.get("timing_boundary", ""):
            raise ValueError("lifecycle receipt timing boundary is not explicit")
        if mode not in {"standalone", "run"} or mode in modes:
            raise ValueError("smoke artifacts must contain one standalone and one Run mode")
        modes.add(mode)
        if summary.get("optimizer", {}).get("updates") != 2:
            raise ValueError(f"{mode} smoke did not perform exactly two rollout updates")
        learned = summary.get("collection", {}).get("learned_decisions")
        slots = summary.get("collection", {}).get("attempted_environment_slots")
        optimizer = summary.get("optimizer", {})
        attempted = optimizer.get("attempted_steps")
        accepted = optimizer.get("accepted_steps")
        uncertain = optimizer.get("uncertain_steps")
        if any(type(value) is not int or value < 0 for value in
               (learned, slots, attempted, accepted, uncertain)):
            raise ValueError(f"{mode} smoke summary counters are invalid")
        if accepted <= 0 or uncertain != 0 or attempted != accepted:
            raise ValueError(f"{mode} smoke requires accepted optimizer work and zero uncertainty")
        if attempted > 16:
            raise ValueError(f"{mode} smoke exceeded its 16-attempt optimizer cap")
        totals["learned_decisions"] += learned
        totals["environment_slots"] += slots
        totals["optimizer_attempted_steps"] += attempted
        totals["optimizer_accepted_steps"] += accepted
        checks_path = path.parent / "smoke-checks.json"
        if checks_path.is_symlink() or not checks_path.is_file():
            raise ValueError(f"{mode} model parity and weight-change checks are missing")
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        if (checks.get("schema_id") != "spire-learn-task7-smoke-checks-v1"
                or checks.get("all_final_weights_finite") is not True
                or checks.get("at_least_one_weight_changed") is not True
                or checks.get("deterministic_reload_parity") is not True
                or checks.get("resume_round_trip") is not True):
            raise ValueError(f"{mode} smoke checks do not prove finite changed weights and round trips")
        checkpoint = path.parent / "checkpoint-final"
        if any(not (checkpoint / name).is_file() for name in ("metadata.json", "model.pt", "training.pt")):
            raise ValueError(f"{mode} final checkpoint is incomplete")
        checkpoints.append(str(checkpoint))
    if modes != {"standalone", "run"}:
        raise ValueError("smoke artifacts do not contain both accepted modes")
    caps = {"learned_decisions": 512, "environment_slots": 2048,
            "optimizer_attempted_steps": 32, "optimizer_accepted_steps": 32}
    for name, cap in caps.items():
        if totals[name] > cap:
            raise ValueError(f"combined Task 7 {name} exceeded its cap")
    if totals["optimizer_accepted_steps"] <= 0 or totals["optimizer_attempted_steps"] != totals["optimizer_accepted_steps"]:
        raise ValueError("combined smoke must report accepted optimizer effects with zero uncertainty")
    return {"modes": sorted(modes), "totals": totals, "caps": caps,
            "checkpoints": checkpoints, "runtime_identity": runtime_identity}


def inventory(run_root: Path, control_root: Path, admin_root: Path) -> dict[str, object]:
    roots = (("run", Path(run_root)), ("control", Path(control_root)), ("admin", Path(admin_root)))
    excluded_admin = {
        "evidence-inventory.json", "evidence-inventory.command.log",
        "evidence-inventory.stdout.log", "evidence-inventory.stderr.log",
        "evidence-inventory.exit.log",
    }
    files = []
    for label, root in roots:
        root = root.resolve(strict=True)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"evidence tree contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if label == "admin" and relative in excluded_admin:
                continue
            files.append({"tree": label, "path": relative, "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    return {"schema": "spire-learn-task7-evidence-inventory-v1", "files": files,
            "excluded_final_writer_files": sorted(excluded_admin)}


def _write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                     encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("source-snapshot")
    snap.add_argument("root", type=Path)
    snap.add_argument("output", type=Path)
    compare = commands.add_parser("compare-snapshots")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    release = commands.add_parser("verify-release")
    release.add_argument("record", type=Path)
    release.add_argument("snapshot", type=Path)
    runtime = commands.add_parser("runtime-identity")
    runtime.add_argument("location", type=Path)
    runtime.add_argument("lock", type=Path)
    runtime.add_argument("native", type=Path)
    runtime.add_argument("output", type=Path)
    image = commands.add_parser("verify-image")
    image.add_argument("repo_digests", type=Path)
    image.add_argument("output", type=Path)
    space = commands.add_parser("verify-free-space")
    space.add_argument("path", type=Path)
    space.add_argument("--minimum-gib", type=int, default=30)
    state = commands.add_parser("verify-container-status")
    state.add_argument("status", type=Path)
    transcript = commands.add_parser("transcript")
    transcript.add_argument("stdout", type=Path)
    transcript.add_argument("stderr", type=Path)
    ids = commands.add_parser("resolve-container-id")
    ids.add_argument("lookup", type=Path)
    inspect = commands.add_parser("assert-inspect")
    inspect.add_argument("inspect_json", type=Path)
    inspect.add_argument("output", type=Path)
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--container-id", required=True)
    inspect.add_argument("--image-identity", type=Path, required=True)
    inspect.add_argument("--root", type=Path, required=True)
    inspect.add_argument("--venv", type=Path, required=True)
    inspect.add_argument("--lock", type=Path, required=True)
    inspect.add_argument("--output-child", type=Path, required=True)
    smoke = commands.add_parser("assert-smoke-artifacts")
    smoke.add_argument("output_root", type=Path)
    smoke.add_argument("output", type=Path)
    output_tree = commands.add_parser("prepare-output-tree")
    output_tree.add_argument("output_child", type=Path)
    commands.add_parser("container-args")
    inv = commands.add_parser("inventory")
    inv.add_argument("run_root", type=Path)
    inv.add_argument("control_root", type=Path)
    inv.add_argument("admin_root", type=Path)
    inv.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "source-snapshot":
            _write_json(args.output, source_snapshot(args.root))
        elif args.command == "compare-snapshots":
            compare_snapshots(args.before, args.after)
            print("source snapshot unchanged")
        elif args.command == "verify-release":
            value = verify_release(args.record, args.snapshot)
            print(json.dumps({"decision": value["decision"],
                              "task7_source_snapshot_sha256": value["task7_source_snapshot_sha256"]},
                             sort_keys=True))
        elif args.command == "runtime-identity":
            value = verify_runtime_identity(args.location, args.lock, args.native)
            _write_json(args.output, value)
        elif args.command == "verify-image":
            raw = args.repo_digests.read_text(encoding="utf-8")
            value = verify_image_identity(raw)
            _write_json(args.output, value)
            print(json.dumps(value, sort_keys=True))
        elif args.command == "verify-free-space":
            print(verify_free_space(args.path, args.minimum_gib))
        elif args.command == "verify-container-status":
            verify_container_status(args.status.read_text(encoding="utf-8"))
            print("container exited 0")
        elif args.command == "transcript":
            result = parse_transcript(args.stdout.read_text(encoding="utf-8"),
                                      args.stderr.read_text(encoding="utf-8"))
            print(json.dumps(result, sort_keys=True))
        elif args.command == "resolve-container-id":
            print(resolve_container_id(args.lookup.read_text(encoding="utf-8")))
        elif args.command == "assert-inspect":
            result = assert_container_inspect(
                json.loads(args.inspect_json.read_text(encoding="utf-8")),
                run_id=args.run_id, container_id=args.container_id,
                image_identity=json.loads(args.image_identity.read_text(encoding="utf-8")),
                repo_root=args.root, venv=args.venv, lock=args.lock,
                output_child=args.output_child,
            )
            _write_json(args.output, result)
        elif args.command == "assert-smoke-artifacts":
            result = assert_smoke_artifacts(args.output_root)
            _write_json(args.output, result)
        elif args.command == "prepare-output-tree":
            result = prepare_output_tree(args.output_child)
            print(json.dumps({"created": [str(path) for path in result]}, sort_keys=True))
        elif args.command == "container-args":
            sys.stdout.buffer.write(b"\0".join(value.encode("utf-8") for value in CONTAINER_COMMAND_ARGS) + b"\0")
        elif args.command == "inventory":
            result = inventory(args.run_root, args.control_root, args.admin_root)
            _write_json(args.output, result)
        else:
            raise AssertionError(args.command)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"task7 helper: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
