#!/usr/bin/env python3.12
"""Run a learner entry point in the pinned, offline Spire Gym runtime."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import errno
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from collections.abc import Mapping, Sequence


IMAGE = "spire-learn/task8-python:cuda-setup-20261028-a"
SCHEMA = "spire-learn-gym-runtime-profile-v2"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
ENTRY = re.compile(r"[-A-Za-z0-9_][A-Za-z0-9_.-]*\.py\Z")
MAX_LOG_BYTES = 8 * 1024 * 1024
TMPFS_SPEC = "/tmp:rw,noexec,nosuid,size=64m"
CREATE_TIMEOUT_SECONDS = 30.0
CREATE_RECONCILE_SECONDS = CREATE_TIMEOUT_SECONDS
RECONCILE_INTERVAL_SECONDS = 0.25
ABSENCE_CONFIRMATIONS = 3
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


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
    # Layout assumption: Task 8 stores install-work/venv-cuda and resolve-work/
    # requirements-cuda.lock as siblings below the same runtime directory.
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
    if not isinstance(entry, str) or ENTRY.fullmatch(entry) is None or "\\" in entry:
        raise ValueError("entry must be a simple <name>.py filename")
    entry_path = root / "scripts/neural" / entry
    if entry_path.is_symlink() or not entry_path.is_file():
        raise ValueError("entry must exist as a regular non-symlink script")
    if gpu_uuid is not None and (not isinstance(gpu_uuid, str) or GPU_UUID.fullmatch(gpu_uuid) is None):
        raise ValueError("gpu_uuid must be one canonical NVIDIA GPU UUID")
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
    guest_entry = entry_path
    python = root / profile.dependency_venv / "bin/python"
    bootstrap = (
        "import runpy,sys; "
        f"sys.path[:0]=[{str(root / profile.overlay)!r},{str(root / 'src')!r},{str(root)!r}]; "
        f"sys.argv=['entry',{','.join(repr(str(arg)) for arg in args)}]; "
        f"runpy.run_path({str(guest_entry)!r},run_name='__main__')"
    )
    command = ["docker", "create", "--name", name, "--network", "none", "--read-only",
               "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "512",
               "--tmpfs", TMPFS_SPEC, "--cpus", str(cpus), "--memory", memory,
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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_no_replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rename_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        if error_number not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(error_number, os.strerror(error_number), target)
    # Portable fallback retains atomic no-overwrite publication where renameat2 is absent.
    os.link(source, target)
    source.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git_state(root: Path) -> dict:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    return {"head": head, "dirty": dirty}


def _invoke(docker, argv, *, timeout=None):
    return docker.run(argv, capture_output=True, text=True, check=False, timeout=timeout)


def _docker_metadata(docker) -> dict:
    image = _invoke(docker, ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], timeout=30)
    if image.returncode != 0 or not image.stdout.strip():
        raise RuntimeError(f"could not identify Docker image: {(image.stderr or '').strip()}")
    version = _invoke(docker, ["docker", "version", "--format", "{{json .}}"], timeout=30)
    if version.returncode != 0 or not version.stdout.strip():
        raise RuntimeError(f"could not identify Docker client/server version: {(version.stderr or '').strip()}")
    return {"image": IMAGE, "image_id": image.stdout.strip(), "docker_version": version.stdout.strip()}


def _no_such_container(result) -> bool:
    return result.returncode != 0 and re.search(
        r"no such (?:object|container)", result.stderr or "", re.I) is not None


def _error_text(operation: str, result=None, exception: BaseException | None = None) -> str:
    if exception is not None:
        return f"{operation}: {type(exception).__name__}: {exception}"
    return f"{operation}: exit {result.returncode}: {(result.stderr or '').strip()}"


def _collect_logs(docker, target: str, output: Path) -> tuple[str | None, bool]:
    argv = ["docker", "logs", "--tail", "10000", target]
    popen = getattr(docker, "Popen", subprocess.Popen)
    total_bytes = {"stdout": 0, "stderr": 0}
    truncated = False
    stderr_tail = bytearray()
    error = None
    process = None
    with (output / "stdout.txt").open("wb") as stdout_file, \
         (output / "stderr.txt").open("wb") as stderr_file:
        process = popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        selector = selectors.DefaultSelector()
        files = {"stdout": stdout_file, "stderr": stderr_file}
        try:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + 30.0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = "docker logs timed out"
                    process.kill()
                    break
                events = selector.select(remaining)
                if not events:
                    error = "docker logs timed out"
                    process.kill()
                    break
                for key, _ in events:
                    stream = key.data
                    chunk = os.read(key.fd, 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = max(0, MAX_LOG_BYTES - total_bytes[stream])
                    if available:
                        files[stream].write(chunk[:available])
                        total_bytes[stream] += min(available, len(chunk))
                    if len(chunk) > available:
                        truncated = True
                    if stream == "stderr":
                        stderr_tail.extend(chunk)
                        if len(stderr_tail) > 4096:
                            del stderr_tail[:-4096]
            try:
                returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                error = error or "docker logs timed out waiting for exit"
                process.kill()
                returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
    if returncode != 0:
        detail = stderr_tail.decode("utf-8", errors="replace").strip()
        error = error or f"docker logs exited {returncode}: {detail}"
    return error, truncated


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _termination_scope(finalize_signals=None):
    state = {"cleanup": False, "pending": [], "restore_errors": [],
             "handled": set(), "blocked": False, "old_mask": None}
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def handle(number, frame):
            if state["cleanup"]:
                state["pending"].append(number)
                callback = state.get("handoff_callback")
                if callback is not None and not state.get("post_callback_active"):
                    callback(state)
                return
            raise KeyboardInterrupt(f"received signal {number}")
        state["recording_handler"] = handle
        for number in HANDLED_SIGNALS:
            previous[number] = signal.signal(number, handle)
        state["handled"] = set(previous)
    try:
        yield state
    finally:
        _restore_signal_handlers(previous, state, finalize_signals=finalize_signals)


def _restore_signal_handlers(previous, state, *, finalize_signals=None) -> None:
    state["handoff_callback"] = finalize_signals
    for number in HANDLED_SIGNALS:
        if number not in previous:
            continue
        try:
            signal.signal(number, state["recording_handler"])
        except BaseException as exc:
            state["restore_errors"].append(
                f"reinstall runner signal handler {number}: {type(exc).__name__}: {exc}")
    if state["blocked"]:
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, state["handled"])
            state["blocked"] = False
        except BaseException as exc:
            state["restore_errors"].append(
                f"unblock signals for handoff: {type(exc).__name__}: {exc}")
    if finalize_signals is not None:
        finalize_signals(state)
        state["status_decided"] = True

    # SIGTERM is restored last so a signal observed during earlier disposition
    # handoffs is accounted for before its caller handler becomes active.
    restore_order = (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)
    for number in restore_order:
        if number not in previous:
            continue
        try:
            signal.signal(number, previous[number])
        except BaseException as exc:
            state["restore_errors"].append(
                f"restore signal {number}: {type(exc).__name__}: {exc}")
        if finalize_signals is not None:
            finalize_signals(state)
            state["status_decided"] = True

    # From this point, signals belong to the caller. In the CLI the caller's
    # default disposition terminates the process with a non-zero signal status.
    if state["old_mask"] is not None:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, state["old_mask"])
        except BaseException as exc:
            state["restore_errors"].append(
                f"restore signal mask: {type(exc).__name__}: {exc}")


def _block_finalization_signals(state) -> None:
    if not state["handled"] or not hasattr(signal, "pthread_sigmask"):
        return
    try:
        state["old_mask"] = signal.pthread_sigmask(signal.SIG_BLOCK, state["handled"])
        state["blocked"] = True
    except BaseException as exc:
        state["restore_errors"].append(
            f"block signals for finalization: {type(exc).__name__}: {exc}")


def _pending_handled_signals(state) -> set[int]:
    if not state["handled"] or not hasattr(signal, "sigpending"):
        return set()
    try:
        return set(signal.sigpending()) & state["handled"]
    except BaseException as exc:
        state["restore_errors"].append(
            f"inspect pending signals: {type(exc).__name__}: {exc}")
        return set()


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
    if output.exists():
        raise FileExistsError(output)
    runtime = _docker_metadata(docker)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = _now()
    manifest = {"schema": "spire-learn-runtime-container-manifest-v1", "authority": authority,
        "date": started[:10], "profile": {"path": str(profile.path), "sha256": profile.sha256,
        "gym_revision": profile.gym_revision, "build_identity": profile.build_identity},
        "learner": learner, "verification": verification, "runtime": runtime, "container_name": name,
        "command": create_command, "limits": {"timeout_seconds": timeout_seconds, "cpus": cpus,
        "memory": memory, "gpu_uuid": gpu_uuid}, "started_at": started}
    write_exclusive(output / "manifest.json", manifest)
    created = False
    container_target = name
    container_id = None
    create_attempted = False
    create_uncertain = False
    timed_out = False
    status = 1
    failure = None
    removed = False
    logs_truncated = False
    cleanup_errors = {"logs": [], "kill": [], "rm": [], "inspect": []}
    published_signal_count = 0

    def write_post_manifest(termination):
        nonlocal status
        if termination.get("post_callback_active"):
            return
        termination["post_callback_active"] = True
        try:
            post_signals = set(termination["pending"][published_signal_count:])
            restore_errors = termination["restore_errors"]
            if not post_signals and not restore_errors:
                return
            if termination.get("post_manifest_failed"):
                status = 75
                return
            status = 130 if post_signals else 1
            if termination.get("post_manifest_written"):
                return
            post = {"schema": "spire-learn-runtime-container-post-v1",
                "final_manifest_sha256": _sha256(output / "manifest-final.json"),
                "signals": sorted(post_signals), "handler_restore_errors": restore_errors,
                "exit_status": status,
                "failure": "termination signal pending after final manifest publication" if post_signals else
                           "signal handler restoration failed"}
            try:
                write_exclusive(output / "manifest-post.json", post)
                termination["post_manifest_written"] = True
            except BaseException as exc:
                status = 75
                termination["post_manifest_failed"] = True
                print(f"run is not successful: could not write manifest-post.json: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if termination["pending"][published_signal_count:]:
                status = 130 if not termination.get("post_manifest_failed") else 75
        finally:
            termination["post_callback_active"] = False
            termination["status_decided"] = True

    with _termination_scope(finalize_signals=write_post_manifest) as termination:
        try:
            create_attempted = True
            created_result = _invoke(docker, create_command, timeout=CREATE_TIMEOUT_SECONDS)
            if created_result.returncode != 0:
                failure = f"docker create failed: {created_result.stderr.strip()}"
                status = 1
                create_uncertain = True
            elif not CONTAINER_ID.fullmatch(created_result.stdout.strip()):
                failure = "docker create returned an invalid or empty container ID"
                status = 1
                create_uncertain = True
            else:
                container_id = created_result.stdout.strip()
                container_target = container_id
                created = True
                start_result = _invoke(docker, ["docker", "start", container_id], timeout=30)
                if start_result.returncode != 0:
                    failure = _error_text("docker start", result=start_result)
                    status = 1
                else:
                    try:
                        wait_result = _invoke(docker, ["docker", "wait", container_id], timeout=timeout_seconds)
                        if wait_result.returncode != 0:
                            failure = _error_text("docker wait", result=wait_result)
                            status = 1
                        else:
                            try:
                                status = int(wait_result.stdout.strip())
                            except ValueError:
                                failure = "docker wait returned an invalid exit status"
                                status = 1
                    except (TimeoutError, subprocess.TimeoutExpired) as exc:
                        timed_out = True
                        status = 124
                        failure = _error_text("container wait timed out", exception=exc)
                        try:
                            killed = _invoke(docker, ["docker", "kill", container_id], timeout=30)
                            if killed.returncode != 0:
                                cleanup_errors["kill"].append(_error_text("docker kill", result=killed))
                        except BaseException as kill_error:
                            cleanup_errors["kill"].append(_error_text("docker kill", exception=kill_error))
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if create_attempted and not created:
                create_uncertain = True
            timed_out = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
            status = 124 if timed_out else (130 if isinstance(exc, KeyboardInterrupt) else 1)
        finally:
            termination["cleanup"] = True
            if created or create_uncertain or (create_attempted and timed_out):
                try:
                    log_error, logs_truncated = _collect_logs(docker, container_target, output)
                    if log_error:
                        cleanup_errors["logs"].append(log_error)
                except BaseException as log_exception:
                    cleanup_errors["logs"].append(_error_text("docker logs", exception=log_exception))
            else:
                for filename in ("stdout.txt", "stderr.txt"):
                    try:
                        (output / filename).write_bytes(b"")
                    except BaseException as log_exception:
                        cleanup_errors["logs"].append(_error_text(f"write {filename}", exception=log_exception))

            def remove_target(target: str, label: str) -> bool:
                try:
                    result = _invoke(docker, ["docker", "rm", "--force", target], timeout=30)
                    if result.returncode != 0 and not _no_such_container(result):
                        message = _error_text(f"docker rm {label}", result=result)
                        if message not in cleanup_errors["rm"]:
                            cleanup_errors["rm"].append(message)
                        return False
                    return True
                except BaseException as exc:
                    message = _error_text(f"docker rm {label}", exception=exc)
                    if message not in cleanup_errors["rm"]:
                        cleanup_errors["rm"].append(message)
                    return False

            remove_by_name_succeeded = remove_target(name, "by name")
            if container_id is not None and not remove_by_name_succeeded:
                remove_target(container_id, "by ID")

            def inspect_once() -> bool | None:
                try:
                    result = _invoke(docker, ["docker", "inspect", name], timeout=10)
                    if _no_such_container(result):
                        return False
                    if result.returncode == 0:
                        return True
                    message = _error_text("docker inspect", result=result)
                except BaseException as exc:
                    message = _error_text("docker inspect", exception=exc)
                if message not in cleanup_errors["inspect"]:
                    cleanup_errors["inspect"].append(message)
                return None

            if create_uncertain:
                reconcile_deadline = time.monotonic() + CREATE_RECONCILE_SECONDS
                absent_count = 0
                while True:
                    observed = inspect_once()
                    if observed is False:
                        absent_count += 1
                    elif observed is True:
                        absent_count = 0
                        remove_target(name, "by name during create reconciliation")
                    else:
                        absent_count = 0
                    remaining = reconcile_deadline - time.monotonic()
                    if remaining <= 0:
                        removed = absent_count >= ABSENCE_CONFIRMATIONS
                        break
                    time.sleep(min(RECONCILE_INTERVAL_SECONDS, remaining))
            else:
                for attempt in range(3):
                    observed = inspect_once()
                    if observed is False:
                        removed = True
                        break
                    if attempt < 2:
                        if observed is True:
                            remove_target(name, "by name during cleanup verification")
                        time.sleep(0.05)
            if not removed and "owned container still exists after cleanup" not in cleanup_errors["inspect"]:
                cleanup_errors["inspect"].append("owned container still exists after cleanup")
            for kind in ("logs", "kill", "rm", "inspect"):
                for error in cleanup_errors[kind]:
                    failure = (failure + "; " if failure else "") + error
            if status == 0 and any(cleanup_errors.values()):
                status = 1
            if not removed:
                failure = (failure + "; " if failure else "") + "container absence was not verified"
                status = 1
            if create_uncertain:
                failure = (failure + "; " if failure else "") + "container creation outcome uncertain"
                if status == 0:
                    status = 1
            _block_finalization_signals(termination)
            try:
                final_learner = git_state(root)
            except BaseException as exc:
                final_learner = {"head": None, "dirty": None,
                                 "error": f"{type(exc).__name__}: {exc}"}
                failure = (failure + "; " if failure else "") + "could not record final learner state"
                status = 1
            ended = _now()
            finalization_signals = _pending_handled_signals(termination)
            for number in sorted(finalization_signals):
                if number not in termination["pending"]:
                    termination["pending"].append(number)
            if termination["pending"]:
                signal_names = ",".join(str(item) for item in termination["pending"])
                failure = (failure + "; " if failure else "") + \
                    f"received signal(s) during finalization: {signal_names}"
                status = 130
            if termination["restore_errors"]:
                failure = (failure + "; " if failure else "") + \
                    "could not safely finalize signal handling"
                status = 1
            published_signal_count = len(termination["pending"])
            final = {"schema": "spire-learn-runtime-container-final-v1", "started_at": started,
                "ended_at": ended, "exit_status": int(status), "timed_out": timed_out,
                "container_created": created if not create_uncertain else None,
                "container_removed": "uncertain" if create_uncertain else removed,
                "learner": final_learner, "profile_verification": verification,
                "cleanup_errors": cleanup_errors, "logs_truncated": logs_truncated, "failure": failure,
                "success_contract": "Success requires this manifest exit_status to be 0, the process to exit 0, and manifest-post.json to be absent."}
            write_exclusive(output / "manifest-final.json", final)
    if termination["restore_errors"] and status == 0:
        status = 1
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
