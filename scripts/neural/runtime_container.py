#!/usr/bin/env python3.12
"""Run a learner entry point in the pinned, offline Spire Gym runtime."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Mapping, Sequence


IMAGE = "spire-learn/task8-python:cuda-setup-20261028-a"
SCHEMA = "spire-learn-gym-runtime-profile-v2"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ENTRY = re.compile(r"[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class RuntimeProfile:
    path: Path
    sha256: str
    gym_revision: str
    build_identity: Mapping[str, object]
    wheel_path: Path
    wheel_sha256: str
    overlay: Path
    native_sha256: str
    dependency_venv: Path
    dependency_lock_sha256: str


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> RuntimeProfile:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"unsafe or oversized runtime profile: {path}")
    raw = path.read_bytes()
    try:
        data = json.loads(raw, object_pairs_hook=_json_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid runtime profile JSON: {path}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"runtime profile schema must be {SCHEMA}")
    required = ("gym_revision", "build_identity", "wheel_path", "wheel_sha256",
                "package_overlay", "native_sha256", "dependency_venv", "dependency_lock_sha256")
    if any(key not in data for key in required):
        raise ValueError("runtime profile is missing required fields")
    for key in ("wheel_sha256", "native_sha256", "dependency_lock_sha256"):
        if not isinstance(data[key], str) or HEX64.fullmatch(data[key]) is None:
            raise ValueError(f"runtime profile {key} must be a lowercase SHA-256")
    if not isinstance(data["gym_revision"], str) or not data["gym_revision"]:
        raise ValueError("runtime profile gym_revision is invalid")
    if not isinstance(data["build_identity"], dict):
        raise ValueError("runtime profile build_identity must be an object")
    paths = [data[key] for key in ("wheel_path", "package_overlay", "dependency_venv")]
    if any(not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts for value in paths):
        raise ValueError("runtime profile paths must be repository-relative and traversal-free")
    return RuntimeProfile(path=path, sha256=hashlib.sha256(raw).hexdigest(),
        gym_revision=data["gym_revision"], build_identity=data["build_identity"],
        wheel_path=Path(data["wheel_path"]), wheel_sha256=data["wheel_sha256"],
        overlay=Path(data["package_overlay"]), native_sha256=data["native_sha256"],
        dependency_venv=Path(data["dependency_venv"]),
        dependency_lock_sha256=data["dependency_lock_sha256"])


def _repo_path(root: Path, relative: Path, *, exists=False) -> Path:
    root = Path(root).resolve(strict=True)
    path = root / relative
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("runtime path escapes repository") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"symlink in runtime path: {current}")
    if exists and not path.exists():
        raise ValueError(f"missing runtime input: {path}")
    return path


def _output_path(root: Path, value: Path) -> Path:
    root = Path(root).resolve(strict=True)
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("output path traversal is forbidden")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the repository") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"symlink in output path: {current}")
    result = root / relative
    if not result.is_relative_to(root / "artifacts/development/neural-combat"):
        raise ValueError("writable output must be under artifacts/development/neural-combat")
    return result


def verify_profile(profile: RuntimeProfile, root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    wheel = _repo_path(root, profile.wheel_path, exists=True)
    overlay = _repo_path(root, profile.overlay, exists=True)
    venv = _repo_path(root, profile.dependency_venv, exists=True)
    if not wheel.is_file() or not overlay.is_dir() or not venv.is_dir():
        raise ValueError("wheel, overlay, and dependency venv must be the expected filesystem types")
    wheel_hash = _sha256(wheel)
    if wheel_hash != profile.wheel_sha256:
        raise ValueError("runtime wheel SHA-256 mismatch")
    for path in (overlay, *overlay.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in runtime overlay: {path}")
        if path.stat().st_mode & 0o222:
            raise ValueError(f"runtime overlay is writable: {path}")
    native_files = list((overlay / "spire_lab").glob("_native*.so")) if (overlay / "spire_lab").is_dir() else []
    if len(native_files) != 1:
        raise ValueError("runtime overlay must contain exactly one spire_lab/_native shared library")
    native_hash = _sha256(native_files[0])
    if native_hash != profile.native_sha256:
        raise ValueError("runtime overlay native SHA-256 mismatch")
    # Task 8's qualified venv and its lock are sibling outputs under one runtime directory.
    lock = venv.parent.parent / "resolve-work/requirements-cuda.lock"
    lock = _repo_path(root, lock.relative_to(root), exists=True)
    lock_hash = _sha256(lock)
    if lock_hash != profile.dependency_lock_sha256:
        raise ValueError("dependency lock SHA-256 mismatch")
    return {"verified": True, "profile_sha256": profile.sha256,
            "wheel_sha256": wheel_hash, "native_sha256": native_hash,
            "dependency_lock_sha256": lock_hash,
            "wheel_path": profile.wheel_path.as_posix(), "overlay": profile.overlay.as_posix(),
            "dependency_venv": profile.dependency_venv.as_posix()}


def docker_command(profile: RuntimeProfile, *, root: Path, entry: str, args: Sequence[str],
                   output: Path, name: str, gpu_uuid: str | None, cpus: float,
                   memory: str, env: Mapping[str, str]) -> list[str]:
    root = Path(root).resolve(strict=True)
    if not isinstance(entry, str) or ENTRY.fullmatch(entry) is None:
        raise ValueError("entry must be a simple script filename")
    if not name or any(char.isspace() for char in name):
        raise ValueError("invalid owned container name")
    output = _output_path(root, output)
    if not math.isfinite(cpus) or cpus <= 0:
        raise ValueError("cpus must be positive")
    if not isinstance(memory, str) or not memory:
        raise ValueError("memory limit is required")
    for host_path in (root, output):
        if any(char in str(host_path) for char in (",", "\n", "\r", "\x00")):
            raise ValueError("Docker bind path contains an unsafe character")
    guest_entry = root / "scripts/neural" / entry
    python = root / profile.dependency_venv / "bin/python"
    bootstrap = (
        "import runpy,sys; "
        f"sys.path[:0]=[{str(root / profile.overlay)!r},{str(root / 'src')!r},{str(root)!r}]; "
        f"sys.argv=['entry',{','.join(repr(str(arg)) for arg in args)}]; "
        f"runpy.run_path({str(guest_entry)!r},run_name='__main__')"
    )
    command = ["docker", "create", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp", "--cpus", str(cpus), "--memory", memory,
               "-v", f"{root}:{root}:ro", "-v", f"{output}:{output}", "-w", "/tmp"]
    if gpu_uuid:
        command.extend(("--gpus", f"device={gpu_uuid}"))
    else:
        command.extend(("-e", "CUDA_VISIBLE_DEVICES="))
    for key, value in env.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(value, str):
            raise ValueError("environment entries must have simple names and string values")
        if key == "CUDA_VISIBLE_DEVICES":
            raise ValueError("CUDA_VISIBLE_DEVICES is controlled by runner device selection")
        command.extend(("-e", f"{key}={value}"))
    command.extend((IMAGE, str(python), "-I", "-B", "-c", bootstrap))
    return command


def write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def git_state(root: Path) -> dict:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    return {"head": head, "dirty": dirty}


def _invoke(docker, argv, *, timeout=None):
    return docker.run(argv, capture_output=True, text=True, check=False, timeout=timeout)


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _termination_scope():
    state = {"cleanup": False, "pending": None}
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def handle(number, frame):
            if state["cleanup"]:
                state["pending"] = number
                return
            raise KeyboardInterrupt(f"received signal {number}")
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, handle)
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run(profile_path: Path, *, entry: str, args: Sequence[str], output: Path,
        authority: str, gpu_uuid: str | None = None, timeout_seconds: float = 14400,
        cpus: float = 4, memory: str = "16g", env: Mapping[str, str] | None = None,
        docker=subprocess, root: Path | None = None) -> int:
    root = Path(root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    profile_path = Path(profile_path)
    if not profile_path.is_absolute():
        profile_path = root / profile_path
    profile = load_profile(profile_path)
    verification = verify_profile(profile, root)
    learner = git_state(root)
    output = _output_path(root, Path(output))
    name = "spire-learn-runtime-" + uuid.uuid4().hex[:20]
    create_command = docker_command(profile, root=root, entry=entry, args=args, output=output,
        name=name, gpu_uuid=gpu_uuid, cpus=cpus, memory=memory, env=env or {})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = _now()
    manifest = {"schema": "spire-learn-runtime-container-manifest-v1", "authority": authority,
        "date": started[:10], "profile": {"path": str(profile.path), "sha256": profile.sha256,
        "gym_revision": profile.gym_revision, "build_identity": profile.build_identity},
        "learner": learner, "verification": verification, "container_name": name,
        "command": create_command, "limits": {"timeout_seconds": timeout_seconds, "cpus": cpus,
        "memory": memory, "gpu_uuid": gpu_uuid}, "started_at": started}
    write_exclusive(output / "manifest.json", manifest)
    created = False
    cid = name
    timed_out = False
    status = 1
    failure = None
    removed = False
    with _termination_scope() as termination:
        try:
            created_result = _invoke(docker, create_command, timeout=30)
            if created_result.returncode != 0:
                failure = f"docker create failed: {created_result.stderr.strip()}"
            else:
                cid = created_result.stdout.strip()
                if not cid:
                    failure = "docker create returned no container id"
                else:
                    created = True
                    start_result = _invoke(docker, ["docker", "start", cid], timeout=30)
                    if start_result.returncode != 0:
                        failure = f"docker start failed: {start_result.stderr.strip()}"
                    else:
                        try:
                            wait_result = _invoke(docker, ["docker", "wait", cid], timeout=timeout_seconds)
                            if wait_result.returncode != 0:
                                failure = f"docker wait failed: {wait_result.stderr.strip()}"
                            else:
                                try:
                                    status = int(wait_result.stdout.strip())
                                except ValueError:
                                    failure = "docker wait returned an invalid exit status"
                                    status = 1
                        except (TimeoutError, subprocess.TimeoutExpired):
                            timed_out = True
                            status = 124
                            failure = "container exceeded outer timeout"
                            _invoke(docker, ["docker", "kill", cid], timeout=30)
                    logs = _invoke(docker, ["docker", "logs", cid], timeout=30)
                    (output / "stdout.txt").write_text(logs.stdout or "", encoding="utf-8")
                    (output / "stderr.txt").write_text(logs.stderr or "", encoding="utf-8")
                    if logs.returncode != 0 and failure is None:
                        failure = f"docker logs failed: {logs.stderr.strip()}"
                        status = 1
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            status = 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            termination["cleanup"] = True
            try:
                _invoke(docker, ["docker", "rm", "--force", cid], timeout=30)
                inspect = _invoke(docker, ["docker", "inspect", cid], timeout=30)
                removed = (inspect.returncode != 0 and
                           re.search(r"no such (?:object|container)", inspect.stderr or "", re.I) is not None)
                if not removed and failure is None:
                    failure = "container remained after docker rm"
                    status = 1
            except BaseException as exc:
                failure = (failure + "; " if failure else "") + f"cleanup failed: {type(exc).__name__}: {exc}"
                status = 1
            if termination["pending"] is not None:
                failure = (failure + "; " if failure else "") + f"received signal {termination['pending']} during cleanup"
                status = 130
            ended = _now()
            try:
                final_learner = git_state(root)
            except BaseException as exc:
                final_learner = {"head": None, "dirty": None,
                                 "error": f"{type(exc).__name__}: {exc}"}
                failure = (failure + "; " if failure else "") + "could not record final learner state"
                status = 1
            final = {"schema": "spire-learn-runtime-container-final-v1", "started_at": started,
                "ended_at": ended, "exit_status": int(status), "timed_out": timed_out,
                "container_created": created, "container_removed": removed,
                "learner": final_learner, "profile_verification": verification, "failure": failure}
            write_exclusive(output / "manifest-final.json", final)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gpu")
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--cpus", type=float, default=4)
    parser.add_argument("--memory", default="16g")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    values = parser.parse_args(argv)
    entry_args = values.entry_args[1:] if values.entry_args[:1] == ["--"] else values.entry_args
    env = {}
    for item in values.env:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error("--env must be KEY=VALUE")
        env[key] = value
    return run(values.profile, entry=values.entry, args=entry_args, output=values.output,
        authority=values.authority, gpu_uuid=values.gpu, timeout_seconds=values.timeout,
        cpus=values.cpus, memory=values.memory, env=env, root=values.root)


if __name__ == "__main__":
    raise SystemExit(main())
