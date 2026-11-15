"""Profile-driven runtime container tests use temporary trees and a fake Docker CLI."""
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neural import runtime_container as runner


class FakeDocker:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.calls = []
        self.created = False
        self.removed = False
        self.killed = False

    def run(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        operation = argv[1]
        if operation == "create":
            if self.failure == "create":
                return SimpleNamespace(returncode=125, stdout="", stderr="create failed")
            self.created = True
            return SimpleNamespace(returncode=0, stdout="fake-id\n", stderr="")
        if operation == "start":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "wait":
            if self.failure == "timeout":
                raise TimeoutError("wait exceeded")
            if self.failure == "signal":
                signal.raise_signal(signal.SIGTERM)
            return SimpleNamespace(returncode=0, stdout="7\n" if self.failure == "nonzero" else "0\n", stderr="")
        if operation == "kill":
            self.killed = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if operation == "logs":
            return SimpleNamespace(returncode=0, stdout="captured out", stderr="captured err")
        if operation == "rm":
            self.removed = True
            return SimpleNamespace(returncode=0, stdout="fake-id\n", stderr="")
        if operation == "inspect":
            return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object: fake-id")
        raise AssertionError(f"unexpected docker operation: {operation}")


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
        native = native_dir / "_native.cpython-312-x86_64-linux-gnu.so"
        native.write_bytes(b"native payload")
        self.venv = self.root / "artifacts/runtime/install-work/venv-cuda"
        self.venv.mkdir(parents=True)
        (self.venv / "bin").mkdir()
        (self.venv / "bin/python").touch()
        lock = self.root / "artifacts/runtime/resolve-work/requirements-cuda.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"lock payload")
        self.profile_data = {
            "schema": "spire-learn-gym-runtime-profile-v2",
            "gym_revision": "a" * 40,
            "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "wheel_sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
            "wheel_path": self.wheel.relative_to(self.root).as_posix(),
            "package_overlay": self.overlay.relative_to(self.root).as_posix(),
            "dependency_venv": self.venv.relative_to(self.root).as_posix(),
            "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "build_identity": {"revision": "a" * 40, "dirty": False},
        }
        self.write_profile()
        self.output = self.root / "artifacts/development/neural-combat/test-run"
        self.output.parent.mkdir(parents=True)
        # The profile verifier requires an immutable overlay, including directories.
        for path in sorted(self.overlay.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        self.overlay.chmod(0o555)

    def tearDown(self):
        # Restore permissions so TemporaryDirectory can clean its contents.
        for path in self.overlay.rglob("*"):
            try:
                path.chmod(0o755 if path.is_dir() else 0o644)
            except FileNotFoundError:
                pass
        self.overlay.chmod(0o755)
        self.temp.cleanup()

    def write_profile(self):
        self.profile_path.write_text(json.dumps(self.profile_data), encoding="utf-8")

    def run_runner(self, docker, **overrides):
        git = overrides.pop("_git_state", {"head": "1" * 40, "dirty": []})
        values = dict(entry="gated_tests_entry.py", args=["--smoke"], output=self.output,
                      authority="approved unit fixture", gpu_uuid=None, timeout_seconds=10,
                      cpus=2.0, memory="4g", env={})
        values.update(overrides)
        with patch.object(runner, "git_state", return_value=git):
            return runner.run(self.profile_path, root=self.root, docker=docker, **values)

    def test_command_is_offline_readonly_and_has_exactly_one_writable_mount(self):
        profile = runner.load_profile(self.profile_path)
        command = runner.docker_command(profile, root=self.root, entry="gated_tests_entry.py",
            args=["--smoke"], output=self.output, name="owned", gpu_uuid=None, cpus=2,
            memory="4g", env={"MODE": "test"})
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        mounts = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "-v"]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(mounts[0].endswith(":ro"))
        self.assertFalse(mounts[1].endswith(":ro"))
        self.assertEqual(mounts[1].split(":")[0], str(self.output))
        self.assertFalse(any("--gpus" in value for value in command))
        self.assertIn("spire-learn/task8-python:cuda-setup-20261028-a", command)
        self.assertIn("-I", command)
        self.assertIn("-B", command)
        self.assertIn("runpy.run_path", command[-1])

    def test_gpu_uuid_adds_device_flag_and_preserves_cuda_visibility(self):
        profile = runner.load_profile(self.profile_path)
        command = runner.docker_command(profile, root=self.root, entry="entry.py", args=[],
            output=self.output, name="owned", gpu_uuid="GPU-abc", cpus=1, memory="2g", env={})
        self.assertEqual(command[command.index("--gpus") + 1], "device=GPU-abc")
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", command)

    def test_profile_hash_mismatch_is_rejected_before_docker(self):
        self.profile_data["wheel_sha256"] = "0" * 64
        self.write_profile()
        docker = FakeDocker()
        with self.assertRaises(ValueError):
            self.run_runner(docker)
        self.assertEqual(docker.calls, [])

    def test_existing_output_directory_is_rejected(self):
        self.output.mkdir()
        docker = FakeDocker()
        with self.assertRaises(FileExistsError):
            self.run_runner(docker)
        self.assertEqual(docker.calls, [])

    def test_timeout_kills_removes_and_verifies_absence(self):
        docker = FakeDocker(failure="timeout")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        operations = [call[1] for call in docker.calls]
        self.assertIn("kill", operations)
        self.assertIn("rm", operations)
        self.assertEqual(operations[-1], "inspect")
        self.assertTrue(docker.killed and docker.removed)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["timed_out"])
        self.assertNotEqual(final["exit_status"], 0)
        self.assertTrue(final["container_removed"])

    def test_termination_signal_removes_container_and_records_failure(self):
        docker = FakeDocker(failure="signal")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertTrue(docker.removed)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertTrue(final["container_removed"])
        self.assertIn("KeyboardInterrupt", final["failure"])

    def test_create_failure_publishes_final_manifest_without_container(self):
        docker = FakeDocker(failure="create")
        status = self.run_runner(docker)
        self.assertNotEqual(status, 0)
        self.assertFalse(docker.created)
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertFalse(final["container_created"])
        self.assertTrue(final["container_removed"])

    def test_nonzero_exit_is_not_retried_and_dirty_state_is_recorded(self):
        docker = FakeDocker(failure="nonzero")
        status = self.run_runner(docker, _git_state={"head": "1" * 40, "dirty": [" M file.py"]})
        self.assertEqual(status, 7)
        self.assertEqual(sum(call[1] == "create" for call in docker.calls), 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        final = json.loads((self.output / "manifest-final.json").read_text())
        self.assertEqual(manifest["learner"]["dirty"], [" M file.py"])
        self.assertEqual(final["learner"]["dirty"], [" M file.py"])
        self.assertEqual((self.output / "stdout.txt").read_text(), "captured out")
        self.assertEqual((self.output / "stderr.txt").read_text(), "captured err")

    def test_manifest_publication_is_exclusive(self):
        docker = FakeDocker()
        self.run_runner(docker)
        with self.assertRaises(FileExistsError):
            runner.write_exclusive(self.output / "manifest.json", {})
        with self.assertRaises(FileExistsError):
            runner.write_exclusive(self.output / "manifest-final.json", {})


if __name__ == "__main__":
    unittest.main()
