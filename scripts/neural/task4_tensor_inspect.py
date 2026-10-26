"""Strict stdlib assertion for a Task 4 tensor-test container inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def expected_environment(reservation_id: str) -> set[str]:
    return {
        "PYTHONPATH=/learner/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PIP_NO_INDEX=1",
        "PIP_NO_CACHE_DIR=1",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "HOME=/run/home",
        "TMPDIR=/run/tmp",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "SPIRE_LEARN_RUN_NEURAL_TENSOR=1",
        "SPIRE_LEARN_TENSOR_TASK=task4",
        f"SPIRE_LEARN_TENSOR_RESERVATION_ID={reservation_id}",
        "SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64",
        "SPIRE_LEARN_TENSOR_EVIDENCE_PATH=/run/tensor-sampling.json",
        "SPIRE_LEARN_TENSOR_METRICS_PATH=/run/model-metrics.json",
    }


def assert_container_inspect(
    inspect: Mapping[str, object], *, image_id: str, root: Path, venv: Path,
    writable: Path, command: str, reservation_id: str,
) -> None:
    """Reject any control, mount, entrypoint, or command drift before start."""
    assert inspect["image_id"] == image_id, "container image ID differs from the pinned local image"
    assert inspect["user"] == "1000:1000", "container must run as the qualified non-root user"
    assert inspect["workdir"] == "/learner", "container working directory changed"
    assert inspect["entrypoint"] == ["/bin/sh"], "container entrypoint changed"
    assert inspect["cmd"] == ["-ec", command], "reviewed container command changed"
    assert inspect["readonly_rootfs"] is True, "container root filesystem must be read-only"
    assert inspect["privileged"] is False, "privileged containers are not permitted"
    assert set(inspect["cap_drop"]) == {"ALL"}, "all Linux capabilities must be dropped"
    assert any(item in {"no-new-privileges", "no-new-privileges:true"}
               for item in inspect["security_opt"])
    assert inspect["nano_cpus"] == 4_000_000_000, "container CPU limit must be four CPUs"
    assert inspect["memory"] == 8 * 1024**3, "container memory limit must be 8 GiB"
    assert inspect["pids_limit"] == 512, "container PID limit changed"
    assert inspect["network"] == "none", "tensor checks must run without network access"
    assert set(inspect["tmpfs"]) == {"/tmp"}
    tmpfs_options = set(inspect["tmpfs"]["/tmp"].split(","))
    assert {"rw", "noexec", "nosuid"} <= tmpfs_options
    tmpfs_size = next((item.split("=", 1)[1] for item in tmpfs_options
                       if item.startswith("size=")), None)
    assert tmpfs_size in {"64m", str(64 * 1024 * 1024)}

    mounts = {item["Destination"]: item for item in inspect["mounts"]}
    expected_mounts = {
        "/learner/src": (str(root / "src"), False),
        "/learner/tests": (str(root / "tests"), False),
        "/learner/configs": (str(root / "configs"), False),
        "/work/venv-cpu": (str(venv), False),
        "/run": (str(writable), True),
        "/learner/artifacts/development/neural-combat": (str(writable), True),
    }
    assert set(mounts) == set(expected_mounts), "unexpected bind mount set"
    for destination, (source, read_write) in expected_mounts.items():
        mount = mounts[destination]
        assert mount["Type"] == "bind", f"{destination} must be a bind mount"
        assert mount["Source"] == source, f"{destination} source changed"
        assert mount["RW"] is read_write, f"{destination} access mode changed"

    environment = set(inspect["env"])
    required = expected_environment(reservation_id)
    assert required <= environment, "required offline/task-4 guard environment changed"
    assert not any(name in item for item in environment for name in (
        "AWS_SECRET", "TOKEN=", "PASSWORD=", "DOCKER_HOST=", "SSH_AUTH_SOCK=",
    )), "credential or host socket environment leaked into container"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect", type=Path)
    parser.add_argument("image_id")
    parser.add_argument("root", type=Path)
    parser.add_argument("venv", type=Path)
    parser.add_argument("writable", type=Path)
    parser.add_argument("reservation_id")
    parser.add_argument("command_file", type=Path)
    args = parser.parse_args()
    inspect = json.loads(args.inspect.read_text(encoding="utf-8"))
    command = args.command_file.read_text(encoding="utf-8").rstrip("\n")
    assert_container_inspect(
        inspect,
        image_id=args.image_id,
        root=args.root,
        venv=args.venv,
        writable=args.writable,
        command=command,
        reservation_id=args.reservation_id,
    )
    print(json.dumps({"inspection_valid": True, "entrypoint": ["/bin/sh"], "mount_count": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
