"""Runtime container lifecycle tests use a temporary profile tree and fake Docker CLI."""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neural import runtime_container as runner


CID = "a" * 64
GPU = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"


class FakeDocker:
    """Stateful fake for Docker CLI outcomes; no external daemon is contacted."""
    def __init__(self, *, failure=None, logs_size=0):
        self.failure = failure
        self.logs_size = logs_size
        self.calls = []
        self.name = None
        self.container_id = CID
        self.exists = False
        self.created = False
        self.rm_targets = []
        self.inspect_targets = []
        self.inspect_count = 0
        self.pending_create = False
        self.popen_calls = []

    def run(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        operation = argv[1]
        if operation == "image":
            return SimpleNamespace(returncode=0, stdout="sha256:image-id\n", stderr="")
        if operation == "version":
            return SimpleNamespace(returncode=0, stdout="Docker version 99.0.0\n", stderr="")
        if operation == "create":
            self.name = argv[argv.index("--name") + 1]
            if self.failure == "create":
                return SimpleNamespace(returncode=125, stdout="", stderr="create failed")
            self.created = True
            self.exists = True
            if self.failure == "create_timeout":
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            if self.failure == "delayed_create":
                self.exists = False
                self.pending_create = True
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            if self.failure == "empty_create":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout=self.container_id + "\n", stderr="")
        if operation == "start":
            if self.failure == "start":
                return SimpleNamespace(returncode=1, stdout="", stderr="start failed")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "wait":
            if self.failure == "timeout":
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            if self.failure == "sigterm":
                signal.raise_signal(signal.SIGTERM)
            if self.failure == "sighup":
                signal.raise_signal(signal.SIGHUP)
            return SimpleNamespace(returncode=0,
                stdout="7\n" if self.failure == "nonzero" else "0\n", stderr="")
        if operation == "kill":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "logs":
            if self.failure == "logs":
                return SimpleNamespace(returncode=1, stdout="partial", stderr="logs failed")
            stdout = "x" * self.logs_size if self.logs_size else "captured out"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="captured err")
        if operation == "rm":
            targets = [item for item in argv[2:] if not item.startswith("-")]
            self.rm_targets.extend(targets)
            if self.failure == "rm_exception":
                raise RuntimeError("rm transport failed")
            if self.failure == "rm_timeout":
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            self.exists = False
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "inspect":
            target = argv[-1]
            self.inspect_targets.append(target)
            self.inspect_count += 1
            if self.pending_create and self.inspect_count == 1:
                self.pending_create = False
                self.exists = True
                return SimpleNamespace(returncode=1, stdout="", stderr=f"Error: No such object: {target}")
            if self.exists:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"Id": self.container_id}), stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=f"Error: No such object: {target}")
        raise AssertionError(f"unexpected docker operation: {argv}")

    def Popen(self, argv, **kwargs):
        argv = list(argv)
        self.popen_calls.append(argv)
        self.calls.append((argv, kwargs))
        if self.failure == "logs":
            code = "import sys; sys.stdout.write('partial'); sys.stderr.write('logs failed'); sys.exit(1)"
        elif self.logs_size:
            code = f"import sys; sys.stdout.buffer.write(b'x' * {self.logs_size}); sys.stderr.write('captured err')"
        else:
            code = "import sys; sys.stdout.write('captured out'); sys.stderr.write('captured err')"
        return subprocess.Popen([sys.executable, "-c", code], stdout=kwargs["stdout"],
                                stderr=kwargs["stderr"], bufsize=0)

    def operations(self):
        return [call[0][1] for call in self.calls]


class RuntimeContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile_path = self.root / "configs/neural/test/runtime.json"
        self.profile_path.parent.mkdir(parents=True)
        self.wheel = self.root / "artifacts/input/test.whl"
        self.wheel.parent.mkdir(parents=True)
        self.wheel.write_bytes(b"wheel payload")
        self.overlay = self.root / "artifacts/site"
        native_dir = self.overlay / "spire_lab"
        native_dir.mkdir(parents=True)
        self.native = native_dir / "_native.cpython-312-x86_64-linux-gnu.so"
        self.native.write_bytes(b"native payload")
        self.venv = self.root / "artifacts/runtime/install-work/venv-cuda"
        self.venv.mkdir(parents=True)
        (self.venv / "bin").mkdir()
        (self.venv / "bin/python").touch()
        self.lock = self.root / "artifacts/runtime/resolve-work/requirements-cuda.lock"
        self.lock.parent.mkdir(parents=True)
        self.lock.write_bytes(b"lock payload")
        self.profile_data = {
            "schema": "spire-learn-gym-runtime-profile-v2",
            "gym_revision": "a" * 40,
            "native_sha256": self.sha(self.native),
            "wheel_sha256": self.sha(self.wheel),
            "wheel_path": self.wheel.relative_to(self.root).as_posix(),
            "package_overlay": self.overlay.relative_to(self.root).as_posix(),
            "dependency_venv": self.venv.relative_to(self.root).as_posix(),
            "dependency_lock_sha256": self.sha(self.lock),
            "build_identity": {"revision": "a" * 40, "dirty": False},
        }
        self.write_profile()
        entry_dir = self.root / "scripts/neural"
        entry_dir.mkdir(parents=True)
        (entry_dir / "gated_tests_entry.py").write_text("pass\n", encoding="utf-8")
        (entry_dir / "entry.py").write_text("pass\n", encoding="utf-8")
        self.output = self.root / "artifacts/development/neural-combat/test-run"
        self.output.parent.mkdir(parents=True)
        for path in sorted(self.overlay.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        self.overlay.chmod(0o555)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def tearDown(self):
        for path in self.overlay.rglob("*"):
            try:
                path.chmod(0o755 if path.is_dir() else 0o644)
            except FileNotFoundError:
                pass
        self.overlay.chmod(0o755)
        self.temp.cleanup()

    def write_profile(self):
        self.profile_path.write_text(json.dumps(self.profile_data), encoding="utf-8")

    def fake_git(self, *, late_signal=False):
        calls = []
        def state(root):
            calls.append("state")
            if late_signal and len(calls) == 2:
                signal.raise_signal(signal.SIGTERM)
            return {"head": "1" * 40, "dirty": []}
        return state

    def run_runner(self, docker, **overrides):
        values = dict(entry="gated_tests_entry.py", args=["--smoke"], output=self.output,
                      authority="approved unit fixture", gpu_uuid=None, timeout_seconds=10,
                      cpus=2.0, memory="4g", env={})
        values.update(overrides)
        with patch.object(runner, "git_state", side_effect=self.fake_git()), \
             patch.object(runner, "CREATE_RECONCILE_SECONDS", 0.05, create=True), \
             patch.object(runner, "RECONCILE_INTERVAL_SECONDS", 0.005, create=True):
            return runner.run(self.profile_path, root=self.root, docker=docker, **values)

    def test_command_contains_network_mount_and_runtime_bootstrap_contract(self):
        profile = runner.load_profile(self.profile_path)
        command = runner.docker_command(profile, root=self.root, entry="gated_tests_entry.py",
            args=["--smoke"], output=self.output, name="owned", gpu_uuid=None, cpus=2,
            memory="4g", env={"MODE": "test"})
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        mounts = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "-v"]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(mounts[0].endswith(":ro"))
        self.assertFalse(mounts[1].endswith(":ro"))
        self.assertEqual(mounts[1].split(":")[0], str(self.output))
        self.assertIn("-I", command)
        self.assertIn("-B", command)
        self.assertIn("runpy.run_path", command[-1])

    def test_command_applies_container_hardening_and_host_identity(self):
        profile = runner.load_profile(self.profile_path)
        command = runner.docker_command(profile, root=self.root, entry="entry.py", args=[],
            output=self.output, name="owned", gpu_uuid=None, cpus=1, memory="2g", env={})
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertEqual(command[command.index("--security-opt") + 1], "no-new-privileges")
        self.assertEqual(command[command.index("--pids-limit") + 1], "512")
        self.assertEqual(command[command.index("--tmpfs") + 1], "/tmp:rw,noexec,nosuid,size=64m")
        self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")

    def test_gpu_uuid_selects_only_one_canonical_device(self):
        profile = runner.load_profile(self.profile_path)
        command = runner.docker_command(profile, root=self.root, entry="entry.py", args=[],
            output=self.output, name="owned", gpu_uuid=GPU, cpus=1, memory="2g", env={})
        self.assertEqual(command[command.index("--gpus") + 1], f"device={GPU}")
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", command)

    def test_invalid_gpu_selectors_are_rejected_before_any_docker_call(self):
        for selector in ('GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,GPU-ffffffff-1111-2222-3333-444444444444',
                         '"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"',
                         'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,capabilities=compute'):
            with self.subTest(selector=selector):
                docker = FakeDocker()
                with self.assertRaises(ValueError):
                    self.run_runner(docker, gpu_uuid=selector)
                self.assertEqual(docker.calls, [])
                if self.output.exists():
                    shutil.rmtree(self.output)
                self.assertFalse(self.output.exists())

    def test_invalid_and_missing_entry_paths_are_rejected_before_docker(self):
        for entry in (".hidden.py", "../entry.py", "sub/entry.py", "entry", "missing.py"):
            with self.subTest(entry=entry):
                docker = FakeDocker()
                with self.assertRaises(ValueError):
                    self.run_runner(docker, entry=entry)
                self.assertEqual(docker.calls, [])
                if self.output.exists():
                    shutil.rmtree(self.output)
                self.assertFalse(self.output.exists())

    def test_symlink_entry_is_rejected_before_docker(self):
        link = self.root / "scripts/neural/link.py"
        link.symlink_to(self.root / "scripts/neural/entry.py")
        docker = FakeDocker()
        with self.assertRaises(ValueError):
            self.run_runner(docker, entry="link.py")
        self.assertEqual(docker.calls, [])

    def test_wheel_native_lock_and_overlay_verification_rejects_mismatch(self):
        mutations = (
            ("native_sha256", "0" * 64),
            ("dependency_lock_sha256", "0" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                self.profile_data[field] = value
                self.write_profile()
                docker = FakeDocker()
                with self.assertRaises(ValueError):
                    self.run_runner(docker)
                self.assertEqual(docker.calls, [])
                self.profile_data[field] = self.sha(self.native if field == "native_sha256" else self.lock)
        self.write_profile()
        self.overlay.chmod(0o755)
        (self.overlay / "writable.py").write_text("pass\n", encoding="utf-8")
        (self.overlay / "writable.py").chmod(0o644)
        self.overlay.chmod(0o555)
        docker = FakeDocker()
        with self.assertRaises(ValueError):
            self.run_runner(docker)
        self.assertEqual(docker.calls, [])
        self.overlay.chmod(0o755)
        (self.overlay / "writable.py").unlink()
        self.overlay.chmod(0o555)

    def test_existing_output_directory_is_rejected(self):
        self.output.mkdir()
        docker = FakeDocker()
        with self.assertRaises(FileExistsError):
            self.run_runner(docker)
        self.assertEqual(docker.calls, [])

    def test_timeout_kills_removes_verifies_absence_and_collects_logs(self):
        docker = FakeDocker(failure="timeout")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertIn("kill", docker.operations())
        self.assertIn("logs", docker.operations())
        self.assertTrue(docker.exists is False)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["timed_out"])
        self.assertTrue(final["container_removed"])
        self.assertNotEqual(final["exit_status"], 0)

    def test_create_timeout_removes_by_owned_name_and_rechecks_absence(self):
        docker = FakeDocker(failure="create_timeout")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertTrue(docker.name in docker.rm_targets)
        self.assertGreaterEqual(len(docker.inspect_targets), 1)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["container_removed"])
        self.assertTrue(final["timed_out"])
        self.assertIsNone(final["container_created"])

    def test_empty_create_output_keeps_owned_name_as_cleanup_target(self):
        docker = FakeDocker(failure="empty_create")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertIn(docker.name, docker.rm_targets)
        self.assertNotIn("", docker.rm_targets)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["container_removed"])
        self.assertIsNone(final["container_created"])

    def test_start_failure_still_collects_logs_and_removes(self):
        docker = FakeDocker(failure="start")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertIn("logs", docker.operations())
        self.assertTrue(json.loads((self.output / "manifest-final.json").read_text())["container_removed"])

    def test_logs_failure_is_recorded_as_nonzero(self):
        docker = FakeDocker(failure="logs")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertIn("logs failed", final["failure"])
        self.assertTrue(final["container_removed"])

    def test_saved_logs_are_bounded(self):
        docker = FakeDocker(logs_size=9 * 1024 * 1024)
        self.run_runner(docker)
        self.assertEqual(len(docker.popen_calls), 1)
        self.assertLessEqual((self.output / "stdout.txt").stat().st_size, 8 * 1024 * 1024)

    def test_delayed_create_after_first_absence_is_reconciled(self):
        docker = FakeDocker(failure="delayed_create")
        status = self.run_runner(docker)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertFalse(final["container_removed"] and docker.exists)
        self.assertFalse(docker.exists)
        self.assertGreaterEqual(docker.inspect_count, 4)

    def test_stdout_file_error_after_create_failure_still_cleans_and_publishes(self):
        docker = FakeDocker(failure="create")
        original_open = Path.open
        def fail_stdout(path, *args, **kwargs):
            if Path(path).name == "stdout.txt" and args[:1] == ("wb",):
                raise OSError("disk full for stdout")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", fail_stdout):
            status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertIn("rm", docker.operations())
        self.assertIn("inspect", docker.operations())
        self.assertTrue((self.output / "manifest-final.json").is_file())
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["cleanup_errors"]["logs"])

    def test_rm_exception_still_inspects_and_persisting_container_is_failure(self):
        docker = FakeDocker(failure="rm_exception")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertGreaterEqual(len(docker.inspect_targets), 1)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertFalse(final["container_removed"])
        self.assertNotEqual(final["exit_status"], 0)
        self.assertTrue(final["cleanup_errors"]["rm"])
        self.assertTrue(final["cleanup_errors"]["inspect"])
        self.assertIn(docker.name, docker.rm_targets)
        self.assertIn(docker.container_id, docker.rm_targets)

    def test_rm_timeout_still_inspects_container(self):
        docker = FakeDocker(failure="rm_timeout")
        self.run_runner(docker)
        self.assertGreaterEqual(len(docker.inspect_targets), 1)

    def test_sigterm_during_final_git_state_cannot_be_recorded_as_success(self):
        docker = FakeDocker()
        with patch.object(runner, "git_state", side_effect=self.fake_git(late_signal=True)):
            status = runner.run(self.profile_path, root=self.root, docker=docker,
                entry="gated_tests_entry.py", args=[], output=self.output,
                authority="approved unit fixture")
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertNotEqual(final["exit_status"], 0)
        self.assertIsNotNone(final["failure"])

    def test_signal_after_manifest_publication_leaves_it_unchanged_and_returns_nonzero(self):
        docker = FakeDocker()
        original_write = runner.write_exclusive
        signaled = [False]
        published_bytes = []
        def write(path, value):
            if Path(path).name == "manifest-final.json" and not signaled[0]:
                signaled[0] = True
                original_write(path, value)
                published_bytes.append(Path(path).read_bytes())
                signal.raise_signal(signal.SIGTERM)
                return None
            return original_write(path, value)
        with patch.object(runner, "git_state", side_effect=self.fake_git()), \
             patch.object(runner, "write_exclusive", side_effect=write):
            status = runner.run(self.profile_path, root=self.root, docker=docker,
                entry="gated_tests_entry.py", args=[], output=self.output,
                authority="approved unit fixture")
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertEqual((self.output / "manifest-final.json").read_bytes(), published_bytes[0])
        post = json.loads((self.output / "manifest-post.json").read_text())
        self.assertIn(signal.SIGTERM, post["signals"])

    def test_sigterm_during_handler_restoration_writes_post_manifest(self):
        docker = FakeDocker()
        original_signal = signal.signal
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        original_signal(signal.SIGTERM, lambda number, frame: None)
        calls = []
        def restore_hook(number, handler):
            calls.append((number, handler))
            if len(calls) == 4:
                signal.raise_signal(signal.SIGTERM)
            return original_signal(number, handler)
        try:
            with patch.object(runner.signal, "signal", side_effect=restore_hook):
                status = self.run_runner(docker)
        finally:
            original_signal(signal.SIGTERM, previous_sigterm)
        final = json.loads((self.output / "manifest-final.json").read_text())
        post = json.loads((self.output / "manifest-post.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertEqual(final["exit_status"], 0)
        self.assertIn(signal.SIGTERM, post["signals"])

    def test_signal_during_end_time_capture_forces_failure_manifest(self):
        docker = FakeDocker()
        original_now = runner._now
        calls = []
        def clock():
            calls.append(True)
            if len(calls) == 2:
                signal.raise_signal(signal.SIGTERM)
            return original_now()
        with patch.object(runner, "git_state", side_effect=self.fake_git()), \
             patch.object(runner, "_now", side_effect=clock):
            status = runner.run(self.profile_path, root=self.root, docker=docker,
                entry="gated_tests_entry.py", args=[], output=self.output,
                authority="approved unit fixture")
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertNotEqual(final["exit_status"], 0)
        self.assertIn("signal", final["failure"])

    def test_sighup_during_wait_cleans_container_and_records_failure(self):
        docker = FakeDocker(failure="sighup")
        status = self.run_runner(docker)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertNotEqual(status, 0)
        self.assertIn("logs", docker.operations())
        self.assertTrue(final["container_removed"])
        self.assertIn("KeyboardInterrupt", final["failure"])

    def test_nonzero_exit_is_not_retried_and_dirty_state_is_recorded(self):
        docker = FakeDocker(failure="nonzero")
        status = self.run_runner(docker)
        self.assertEqual(status, 7)
        self.assertEqual(docker.operations().count("create"), 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertEqual(manifest["learner"]["dirty"], [])
        self.assertEqual(final["learner"]["dirty"], [])
        self.assertEqual((self.output / "stdout.txt").read_text(), "captured out")
        self.assertEqual((self.output / "stderr.txt").read_text(), "captured err")
        self.assertEqual(manifest["runtime"]["image_id"], "sha256:image-id")
        self.assertEqual(manifest["runtime"]["docker_version"], "Docker version 99.0.0")

    def test_manifests_are_exclusive_and_existing_file_is_not_overwritten(self):
        docker = FakeDocker()
        self.run_runner(docker)
        manifest = self.output / "manifest.json"
        original = manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.write_exclusive(manifest, {"replacement": True})
        self.assertEqual(manifest.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            runner.write_exclusive(self.output / "manifest-final.json", {})
        self.assertEqual(list(self.output.glob(".manifest*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
