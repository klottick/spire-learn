"""Task 9 pure proof-control, effect-guard, and projected-import checks."""

from __future__ import annotations

import builtins
from contextlib import contextmanager, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/neural/gpu_runtime_proof_helpers.py"
INTEGRATION_PATH = ROOT / "tests/integration/test_neural_gpu_runtime.py"
PROOF_RELATIVE = Path("artifacts/development/neural-combat/task-9/gpu-proof")
GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
OPTIONAL_PREFIXES = (
    "numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab", "cuda", "nvidia",
)
sys.path.insert(0, str(ROOT / "scripts/neural"))
sys.path.insert(0, str(ROOT / "src"))
import gpu_runtime_proof_helpers as proof  # noqa: E402


class _CounterDriver:
    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.closes = 0

    def reset(self, *, seeds, entry_ids, reset_mask):
        self.resets += sum(reset_mask)
        return tuple("row" for _ in reset_mask)

    def step(self, actions):
        self.steps += len(actions)
        return tuple("transition" for _ in actions)

    def close(self):
        self.closes += 1


class _CounterPolicy:
    def __init__(self):
        self.rows = 0

    def act(self, frames, *, generator, deterministic=False):
        if not deterministic:
            self.rows += len(frames)
        return tuple("choice" for _ in frames)


class _CounterOptimizer:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return self.steps


class _VersionPolicy:
    def __init__(self):
        self._behavior_policy_version = 0

    def act(self, frames, *, generator, deterministic=False):
        return tuple(
            type("Choice", (), {"behavior_policy_version": self._behavior_policy_version})()
            for _ in frames
        )


class _FakeHost:
    """Strict fake command surface for a complete, non-Docker host-control pass."""

    def __init__(self):
        self.calls = []
        self.command_envs = []
        self.client_config_paths = []
        self.create_argv = None
        self.container_id = "a" * 64
        self.inspections = 0
        self.clock_calls = 0
        self.clock_exception_at = None
        self.command_exceptions = {}
        self.start_returncode = 0
        self.started = False
        self.removed = False
        self.create_scratch_ready = False
        self.name_queries = 0
        self.cleanup_spec_change = None
        self.clock_control_after_cleanup = False
        self.clock_after_cleanup = None

    def clock(self):
        self.clock_calls += 1
        if self.clock_control_after_cleanup and self.removed and self.name_queries >= 2:
            self.clock_control_after_cleanup = False
            raise KeyboardInterrupt("fake final clock control")
        if self.clock_after_cleanup is not None and self.removed and self.name_queries >= 2:
            return self.clock_after_cleanup
        if self.clock_exception_at == self.clock_calls:
            raise KeyboardInterrupt("fake final clock control")
        return 100.0 + len(self.calls) * 0.01

    def disk_usage(self, _path):
        return type("Usage", (), {"free": 50 * 1024**3})()

    def _inspect_payload(self):
        argv = self.create_argv
        assert argv is not None
        def option(name):
            return argv[argv.index(name) + 1]

        mounts = []
        for index, item in enumerate(argv):
            if item == "--mount":
                fields = dict(part.split("=", 1) for part in argv[index + 1].split(",") if "=" in part)
                mounts.append({
                    "Type": "bind", "Source": fields["src"], "Destination": fields["dst"],
                    "RW": "readonly" not in argv[index + 1].split(","),
                })
        env = [argv[index + 1] for index, item in enumerate(argv) if item == "--env"]
        image_defaults = [
            "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8",
            "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
            "PYTHON_VERSION=3.12.15",
            "PYTHON_SHA256=c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a",
        ]
        inspected = {
            "Id": self.container_id,
            "Name": "/" + option("--name"),
            "Image": proof.IMAGE_ID,
            "Config": {
                "Image": proof.IMAGE_REFERENCE,
                "User": option("--user"),
                "WorkingDir": option("--workdir"),
                "Entrypoint": [option("--entrypoint")],
                "Cmd": ["-B", "-m", "unittest", "-v", "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests"],
                "Env": image_defaults + env,
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "NanoCpus": 4_000_000_000,
                "Memory": 8 * 1024**3,
                "MemorySwap": 8 * 1024**3,
                "PidsLimit": 512,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Privileged": False,
                "CapAdd": [],
                "Devices": [],
                "DeviceCgroupRules": [],
                "GroupAdd": [],
                "DeviceRequests": [{
                    "Driver": "", "Count": 0, "DeviceIDs": [GPU_UUID],
                    "Capabilities": [["gpu"]], "Options": {},
                }],
            },
            "Mounts": mounts,
            "State": {
                "Running": False,
                "Status": "exited" if self.started else "created",
            },
        }

        if self.started:
            field, value = self.cleanup_spec_change or (None, None)
            if field is not None:
                inspected["HostConfig"][field] = value
        return inspected

    def run(
        self, argv, *, capture_output, text, timeout, check, env=None,
        output_limit_bytes=None, progress_check=None,
    ):
        self.calls.append(tuple(argv))
        self.command_envs.append(env)
        self.assert_host_command_args(capture_output, text, timeout, check)
        if argv[0] == "nvidia-smi":
            out = f"{GPU_UUID}, NVIDIA GeForce RTX 5090, 595.91.07, 12.0, 30000\n"
            command = None
        elif argv[0] == "docker":
            if len(argv) < 6 or argv[1] != "--host" or argv[2] != "unix:///var/run/docker.sock":
                raise AssertionError("Docker command did not pin the local Unix endpoint")
            if argv[3] != "--config" or env is None:
                raise AssertionError("Docker command did not use an isolated client config and environment")
            expected_environment = {"PATH", "XDG_CONFIG_HOME", "LC_ALL"}
            if "HOME" in os.environ:
                expected_environment.add("HOME")
            if set(env) != expected_environment:
                raise AssertionError("Docker command environment is not on the isolated allowlist")
            if env.get("HOME") != os.environ.get("HOME"):
                raise AssertionError("Docker command changed or manufactured caller HOME")
            if env["XDG_CONFIG_HOME"] != argv[4]:
                raise AssertionError("Docker XDG config does not select the isolated directory")
            config_dir = Path(argv[4])
            if not config_dir.is_dir() or any(config_dir.iterdir()):
                raise AssertionError("Docker client config directory is not empty")
            self.client_config_paths.append(config_dir)
            command = argv[5:]
        else:
            command = None
        if command is None:
            return subprocess.CompletedProcess(argv, 0, out, "")
        if command:
            key = tuple(command[:2]) if command[:1] == ["container"] and command[1:2] == ["inspect"] else tuple(command[:2])
            if command[:2] == ["container", "inspect"]:
                key = ("container", "inspect", self.inspections + 1)
            if key in self.command_exceptions:
                raise self.command_exceptions[key]
        if command and command[:2] == ["image", "inspect"]:
            out = proof.IMAGE_ID + "\n"
        elif command and command[:3] == ["container", "ls", "--all"]:
            self.name_queries += 1
            out = ""
        elif command and command[0] == "create":
            self.create_argv = list(argv)
            output_mount = next(
                item for item in argv if item.startswith("type=bind,")
                and proof.PROOF_ROOT_GUEST in item
            )
            output_root = Path(
                dict(part.split("=", 1) for part in output_mount.split(",") if "=" in part)["src"]
            )
            if config_dir.resolve().is_relative_to(output_root.resolve()):
                raise AssertionError("Docker client config is inside the guest writable output bind")
            create_env = {
                argv[index + 1].split("=", 1)[0]: argv[index + 1].split("=", 1)[1]
                for index, item in enumerate(argv) if item == "--env"
            }
            self.create_scratch_ready = set(proof.GUEST_SCRATCH_ENV) <= set(create_env) and all(
                create_env[key] == value
                and (output_root / value.removeprefix(proof.PROOF_ROOT_GUEST + "/")).is_dir()
                for key, value in proof.GUEST_SCRATCH_ENV.items()
            )
            out = self.container_id + "\n"
        elif command and command[:2] == ["container", "inspect"]:
            self.inspections += 1
            out = json.dumps([self._inspect_payload()])
        elif command and command[:2] == ["start", "--attach"]:
            proof_mount = next(
                item for item in self.create_argv
                if item.startswith("type=bind,") and proof.PROOF_ROOT_GUEST in item
            )
            output_root = Path(
                dict(part.split("=", 1) for part in proof_mount.split(",") if "=" in part)["src"]
            )
            checks = {
                "one_shared_deadline": True,
                "accepted_finite_update_and_changed_weights": True,
                "final_checkpoint_saved": True,
                "same_backend_weights_and_adam": True,
                "adam_scalar_steps_cpu_moments_cuda": True,
                "same_backend_torch_rng_and_python_rng": True,
                "same_backend_counters_version_and_recovery": True,
                "same_frame_deterministic_and_sampled_full_choice_parity": True,
                "portable_cpu_weights_and_tolerant_outputs": True,
                "warm_start_fresh_state_and_frozen_evaluation": True,
                "cpu_comparisons_used_inert_drivers_without_steps": True,
                "training_driver_closed_once": True,
                "actor_critic_and_shared_hashes_recorded": True,
                "zero_fault_uncertainty_and_incomplete_effects": True,
            }
            for mode, test_id, reset_slots in (
                ("standalone", "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8", 36),
                ("run", "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32", 33),
            ):
                mode_root = output_root / mode
                mode_root.mkdir()
                counters = {
                    "steps": 32, "resets": reset_slots, "actor_samples": 32,
                    "diagnostic_samples": 2, "optimizer": 4,
                    "steps_attempted": 32, "steps_delivered": 32, "steps_uncertain": 0,
                    "resets_attempted": reset_slots, "resets_delivered": reset_slots, "resets_uncertain": 0,
                    "actor_samples_attempted": 32, "actor_samples_delivered": 32, "actor_samples_uncertain": 0,
                    "diagnostic_samples_attempted": 2, "diagnostic_samples_delivered": 2,
                    "diagnostic_samples_uncertain": 0, "optimizer_attempted": 4,
                    "optimizer_accepted": 4, "optimizer_uncertain": 0,
                    "faults": 0, "incomplete_environment_slots": 0,
                }
                result = {
                    "schema_id": proof.MODE_RESULT_SCHEMA, "status": "passed",
                    "mode": mode, "test_id": test_id, "exit_code": 0, "skipped": False,
                    "updates": {"accepted": 1, "finite": True, "weights_changed": True},
                    "counters": counters, "checks": checks,
                    "weights": {
                        "model_sha256": "1" * 64, "shared_body_sha256": "2" * 64,
                        "actor_head_sha256": "3" * 64, "critic_head_sha256": "4" * 64,
                    },
                    "checkpoint": {
                        "path": "standalone/run/checkpoint-final",
                        "metadata_raw_sha256": "5" * 64,
                        "metadata_canonical_sha256": "6" * 64,
                    },
                    "portable_cpu": {
                        "warm_start_optimizer_fresh": True, "warm_start_rng_fresh": True,
                        "evaluation_optimizer_absent": True,
                        "cpu_driver_resets": 0, "cpu_driver_steps": 0,
                    },
                }
                (mode_root / "gpu-proof-checks.json").write_text(
                    json.dumps(result, sort_keys=True) + "\n", encoding="utf-8",
                )
            out = (
                "test_01_standalone_b4_h8 (test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8) ... ok\n"
                "test_02_run_b1_h32 (test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32) ... ok\n\n"
                "Ran 2 tests in 0.010s\n\nOK\n"
            )
            self.started = True
            if progress_check is not None:
                progress_check()
            returncode = self.start_returncode
        elif command and command[:2] == ["container", "rm"]:
            self.removed = True
            out = self.container_id
        else:
            raise AssertionError(f"unexpected host command: {argv}")
        return subprocess.CompletedProcess(argv, locals().get("returncode", 0), out, "")

    @staticmethod
    def assert_host_command_args(capture_output, text, timeout, check):
        if not capture_output or not text or timeout <= 0 or check:
            raise AssertionError("host command call did not use bounded capture mode")


class GpuRuntimeProofHelperTests(unittest.TestCase):
    def test_docker_environment_preserves_or_omits_caller_home(self):
        with tempfile.TemporaryDirectory() as temporary:
            client_config = Path(temporary) / "docker-config"
            client_config.mkdir()
            caller_home = "/caller/home-distinct-from-docker-config"
            with patch.dict(os.environ, {"HOME": caller_home}, clear=False):
                environment = proof._docker_environment(client_config)
            self.assertEqual(environment["HOME"], caller_home)
            self.assertNotEqual(environment["HOME"], str(client_config.resolve()))
            self.assertEqual(environment["XDG_CONFIG_HOME"], str(client_config.resolve()))

            with patch.dict(os.environ, {}, clear=True):
                environment_without_home = proof._docker_environment(client_config)
            self.assertNotIn("HOME", environment_without_home)
            self.assertEqual(environment_without_home["XDG_CONFIG_HOME"], str(client_config.resolve()))

    def test_parser_binds_the_explicit_snapshot_release_claim_and_run(self):
        args = proof.build_parser().parse_args([
            "run", "--root", "/repo/spire-learn", "--snapshot", "/proof/source.json",
            "--release", "/proof/release.json", "--claim", "/proof/claim.json",
            "--run-id", "gpu-proof-20261028-a", "--output-child", "/proof/output",
            "--docker-binary", "docker",
        ])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.root, Path("/repo/spire-learn"))
        self.assertEqual(args.snapshot, Path("/proof/source.json"))
        self.assertEqual(args.release, Path("/proof/release.json"))
        self.assertEqual(args.claim, Path("/proof/claim.json"))
        self.assertEqual(args.run_id, "gpu-proof-20261028-a")
        self.assertEqual(args.output_child, Path("/proof/output"))
        self.assertEqual(args.docker_binary, "docker")

    def test_generated_create_argv_uses_default_writable_mount_and_never_pulls(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_tree = base / "source"
            for relative in ("src", "tests", "configs"):
                (source_tree / relative).mkdir(parents=True)
            helper = source_tree / "scripts/neural/gpu_runtime_proof_helpers.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("helper\n", encoding="utf-8")
            output_child = base / "output"
            output_child.mkdir()
            proof.prepare_guest_scratch(output_child)
            client_config = base / "docker-client"
            client_config.mkdir()
            argv = proof._docker_create_argv(
                docker="/usr/bin/docker", name="spirelearn-task9-gpu-proof-argv-test",
                root=ROOT, source_tree=source_tree, output_child=output_child,
                user="1000:1000", deadline=123.5, docker_client_config=client_config,
            )
        mounts = [argv[index + 1] for index, item in enumerate(argv) if item == "--mount"]
        self.assertIn("--pull=never", argv)
        self.assertTrue(any(item.endswith(",dst=" + proof.PROOF_ROOT_GUEST) for item in mounts))
        self.assertFalse(any(item.endswith(",rw") for item in mounts))

    def test_create_argv_exactly_matches_the_released_no_pull_mount_form(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_tree = base / "staged-source"
            for relative in (
                "src", "tests", "configs", "scripts/neural/gpu_runtime_proof_helpers.py",
            ):
                target = source_tree / relative
                if relative.endswith(".py"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("helper\n", encoding="utf-8")
                else:
                    target.mkdir(parents=True, exist_ok=True)
            output_child = base / "output"
            output_child.mkdir()
            proof.prepare_guest_scratch(output_child)
            docker_client_config = base / "docker-client"
            docker_client_config.mkdir()
            argv = proof._docker_create_argv(
                docker="/usr/bin/docker", name="spirelearn-task9-gpu-proof-argv-test",
                root=ROOT, source_tree=source_tree, output_child=output_child,
                user="1000:1000", deadline=123.5,
                docker_client_config=docker_client_config,
            )
        mounts = [
            (str(source_tree / "src"), "/work/src", "readonly"),
            (str(source_tree / "tests"), "/work/tests", "readonly"),
            (str(source_tree / "configs"), "/work/configs", "readonly"),
            (str(source_tree / "scripts/neural/gpu_runtime_proof_helpers.py"), "/work/gpu_runtime_proof_helpers.py", "readonly"),
            (str(ROOT / proof.CUDA_LOCK), "/runtime/requirements-cuda.lock", "readonly"),
            (str(ROOT / proof.CUDA_VENV), "/runtime/venv-cuda", "readonly"),
            (str(ROOT / proof.CUDA_NATIVE), "/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so", "readonly"),
            (str(output_child.resolve()), proof.PROOF_ROOT_GUEST, None),
        ]
        expected = [
            "/usr/bin/docker", "--host", "unix:///var/run/docker.sock",
            "--config", str(docker_client_config.resolve()),
            "create", "--name", "spirelearn-task9-gpu-proof-argv-test",
            "--user", "1000:1000", "--workdir", "/work",
            "--entrypoint", proof.CONTAINER_PYTHON, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--cpus", "4", "--memory", str(8 * 1024**3), "--memory-swap", str(8 * 1024**3),
            "--pids-limit", "512", "--gpus", f"device={GPU_UUID}", "--pull=never",
        ]
        for source, destination, mode in mounts:
            specification = f"type=bind,src={source},dst={destination}"
            if mode is not None:
                specification += f",{mode}"
            expected.extend(["--mount", specification])
        env = {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
            "SPIRE_LEARN_GPU_RUNTIME_PROOF_HELPER": "/work/gpu_runtime_proof_helpers.py",
            "SPIRE_LEARN_NEURAL_GPU_SOFT_DEADLINE": "123.500000000",
            "SPIRE_LEARN_GPU_RUNTIME_PROOF_OUTPUT": proof.PROOF_ROOT_GUEST,
            "TMPDIR": proof.PROOF_ROOT_GUEST + "/.runtime/tmp",
            "XDG_CACHE_HOME": proof.PROOF_ROOT_GUEST + "/.runtime/cache",
            "TORCH_HOME": proof.PROOF_ROOT_GUEST + "/.runtime/torch",
            "TORCH_EXTENSIONS_DIR": proof.PROOF_ROOT_GUEST + "/.runtime/torch-extensions",
            "CUDA_CACHE_PATH": proof.PROOF_ROOT_GUEST + "/.runtime/cuda",
            "TRITON_CACHE_DIR": proof.PROOF_ROOT_GUEST + "/.runtime/triton",
            "PYTHONPATH": "/work/src:/work/tests/integration",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        for key, value in env.items():
            expected.extend(["--env", f"{key}={value}"])
        expected.extend([
            proof.IMAGE_REFERENCE, "-B", "-m", "unittest", "-v",
            "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests",
        ])
        self.assertEqual(argv, expected)

    def test_guest_runtime_scratch_is_precreated_inside_the_owned_output_and_reserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_child = Path(temporary) / "output"
            output_child.mkdir(mode=0o700)
            paths = proof.prepare_guest_scratch(output_child)
            self.assertEqual(paths, proof.GUEST_SCRATCH_ENV)
            self.assertEqual(proof.output_tree_bytes(output_child), 0)
            for guest_path in paths.values():
                host_path = output_child / guest_path.removeprefix(proof.PROOF_ROOT_GUEST + "/")
                self.assertTrue(host_path.is_dir())
                self.assertEqual(host_path.stat().st_mode & 0o777, 0o700)
                self.assertTrue(host_path.resolve().is_relative_to(output_child.resolve()))
            self.assertNotIn("HOME", paths)
            self.assertEqual(
                proof.OUTPUT_ADMISSION_RESERVATION_BYTES,
                2 * proof.CHECKPOINT_RESERVATION_BYTES
                + 2 * proof.NEURAL_EVENT_LOG_RESERVATION_BYTES
                + 2 * proof.HOST_LOG_RESERVATION_BYTES
                + proof.OUTPUT_METADATA_RESERVATION_BYTES
                + proof.RUNTIME_SCRATCH_RESERVATION_BYTES,
            )
            self.assertLessEqual(
                proof.OUTPUT_ADMISSION_RESERVATION_BYTES,
                proof.LIMITS["output_bytes"],
            )

    def test_bounded_host_command_capture_keeps_prefix_and_reports_timeout(self):
        dependencies = proof.HostDependencies()
        with self.assertRaises(proof.ProofCommandOutputLimit) as output_error:
            proof._run(
                dependencies,
                [sys.executable, "-B", "-c", "import sys; sys.stdout.write('x' * 1500000)"],
                timeout=5.0,
            )
        self.assertLessEqual(len(output_error.exception.stdout_prefix.encode("utf-8")), 1024 * 1024)
        self.assertEqual(len(output_error.exception.stdout_prefix.encode("utf-8")), 1024 * 1024)
        with self.assertRaises(proof.ProofCommandTimeout) as caught:
            proof._run(
                dependencies,
                [sys.executable, "-B", "-c", "import sys,time; print('prefix', flush=True); time.sleep(2)"],
                timeout=0.1,
            )
        self.assertIn("prefix", caught.exception.stdout_prefix)

    def test_staged_source_tree_is_immutable_to_later_canonical_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            root = scratch / "checkout"
            paths = {
                "src/model.py": "model-v1\n",
                "tests/integration/test_proof.py": "test-v1\n",
                "configs/neural/proof.toml": "[proof]\n",
                "scripts/neural/gpu_runtime_proof_helpers.py": "helper-v1\n",
                "scripts/neural/run-gpu-runtime-proof.sh": "wrapper-v1\n",
                "scripts/neural/gpu-runtime-proof-runbook.md": "runbook-v1\n",
            }
            for relative, contents in paths.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents, encoding="utf-8")
            snapshot = proof.create_source_snapshot(
                root, scratch / "snapshot.json", source_paths=(
                    "src", "tests", "configs", "scripts/neural/gpu_runtime_proof_helpers.py",
                    "scripts/neural/run-gpu-runtime-proof.sh", "scripts/neural/gpu-runtime-proof-runbook.md",
                ),
                learner_git_revision="a" * 40,
                dirty_state={"dirty": True, "status_sha256": "b" * 64},
                identity_inputs={"test": "fixture"},
            )
            staged_root = scratch / "staged-source"
            staged = proof.stage_released_source_tree(root, snapshot["source_files"], staged_root)
            self.assertEqual(staged["path_count"], 4)
            self.assertFalse((staged_root / "scripts/neural/run-gpu-runtime-proof.sh").exists())
            self.assertFalse((staged_root / "scripts/neural/gpu-runtime-proof-runbook.md").exists())
            staged_model = staged_root / "src/model.py"
            before = staged_model.read_bytes()
            (root / "src/model.py").write_text("canonical-edited-later\n", encoding="utf-8")
            self.assertEqual(staged_model.read_bytes(), before)
            self.assertEqual(
                proof.verify_staged_source_tree(staged_root, snapshot["source_files"]),
                staged,
            )
            self.assertEqual(staged_model.stat().st_mode & 0o777, 0o444)
            outside = scratch / "outside.py"
            outside.write_text("outside\n", encoding="utf-8")
            linked = root / "tests/integration/linked.py"
            linked.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                proof.stage_released_source_tree(root, snapshot["source_files"], scratch / "linked-stage")
            linked.unlink()
            escaped_inventory = dict(snapshot["source_files"])
            escaped_inventory["src/../../outside.py"] = {"sha256": "f" * 64, "git_mode": "100644"}
            with self.assertRaisesRegex(ValueError, "paths differ"):
                proof.stage_released_source_tree(
                    root, escaped_inventory, scratch / "escaped-stage",
                )

    def test_wrapper_installs_a_300_second_outer_watchdog_with_cleanup_grace(self):
        wrapper = (ROOT / "scripts/neural/run-gpu-runtime-proof.sh").read_text(encoding="utf-8")
        self.assertIn("/usr/bin/timeout --signal=TERM --kill-after=30s 270s", wrapper)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            proof.build_parser().parse_args(["run", "--root", "/repo"])
        self.assertEqual(proof.build_parser().parse_args([
            "snapshot", "--root", "/repo", "--output", "/proof/source.json",
        ]).command, "snapshot")

    def test_configs_validate_against_preserved_mode_inputs(self):
        validated = proof.validate_config_pair(ROOT)
        self.assertEqual(set(validated), {"standalone", "run"})
        expected = {
            "standalone": ("standalone-gpu-smoke.toml", {"rows": 4, "steps": 8}, 36),
            "run": ("run-gpu-smoke.toml", {"rows": 1, "steps": 32}, 33),
        }
        from spire_learn.neural.config import load_neural_config

        for mode, (filename, rollout, reset_slots) in expected.items():
            config = load_neural_config(ROOT / "configs/neural" / filename)
            source = load_neural_config(ROOT / "configs/neural" / f"{mode}.toml")
            self.assertEqual(config.mode, mode)
            self.assertEqual(config.operation, "train")
            self.assertEqual(config.device, "cuda:0")
            self.assertEqual(config.runtime_gpu_uuid, GPU_UUID)
            self.assertEqual(config.rollout_rows, rollout["rows"])
            self.assertEqual(config.rollout_steps, rollout["steps"])
            self.assertEqual(config.budgets.max_training_decisions, 32)
            self.assertEqual(config.budgets.max_environment_actions, 32)
            self.assertEqual(config.budgets.max_optimizer_steps, 4)
            self.assertEqual(config.budgets.max_wall_seconds, 120.0)
            self.assertEqual(config.model, source.model)
            self.assertEqual(config.ppo, source.ppo)
            self.assertEqual(config.seed_streams, source.seed_streams)
            self.assertEqual(config.manifest_sha256, source.manifest_sha256)
            self.assertEqual(config.accepted_identities, source.accepted_identities)
            self.assertNotEqual(config.output_root, source.output_root)
            self.assertIn("task-9/gpu-proof", config.output_root.as_posix())
            self.assertEqual(validated[mode]["reset_slots"], reset_slots)

    def test_snapshot_binds_working_bytes_modes_and_raw_canonical_domains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/module.py").write_bytes(b"module-v1\n")
            source = root / "src/module.py"
            source.chmod(0o644)
            snapshot_path = root / "snapshot.json"
            snapshot = proof.create_source_snapshot(
                root, snapshot_path, source_paths=("src/module.py",),
                learner_git_revision="a" * 40,
                dirty_state={"dirty": True, "status_sha256": "b" * 64},
                identity_inputs={"lock_sha256": "c" * 64},
            )
            binding = proof.verify_source_snapshot(root, snapshot_path, expected_paths=("src/module.py",))
            self.assertEqual(snapshot["schema_id"], proof.SNAPSHOT_SCHEMA)
            self.assertEqual(binding["snapshot_raw_sha256"], proof.sha256_file(snapshot_path))
            self.assertEqual(binding["snapshot_canonical_sha256"], proof.canonical_sha256(snapshot))
            source.write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "bytes or Git modes"):
                proof.verify_source_snapshot(root, snapshot_path, expected_paths=("src/module.py",))

    def _release_fixture(self, root):
        (root / "src").mkdir()
        (root / "src/module.py").write_text("immutable\n", encoding="utf-8")
        snapshot_path = root / "snapshot.json"
        proof.create_source_snapshot(
            root, snapshot_path, source_paths=("src/module.py",),
            learner_git_revision="d" * 40,
            dirty_state={"dirty": False, "status_sha256": "e" * 64},
            identity_inputs={"source": "test"},
        )
        release_path = root / "release.json"
        release = {
            "schema_id": proof.RELEASE_SCHEMA,
            "status": "released",
            "release_id": "task9-review-release-1",
            "run_id": "gpu-proof-20261028-a",
            "source_snapshot_raw_sha256": proof.sha256_file(snapshot_path),
            "source_snapshot_canonical_sha256": proof.canonical_sha256(proof.load_json(snapshot_path)),
            "limits": proof.LIMITS,
        }
        release_path.write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
        return snapshot_path, release_path

    def _fixture_identity_inputs(self):
        return {"fixture_identity": "authored-task8-boundary"}

    def _fixture_dirty_state(self):
        return {"dirty": True, "status_sha256": "d" * 64}

    @contextmanager
    def _patched_fixture_metadata(self, root):
        root = Path(root).resolve(strict=True)
        if root.is_relative_to(ROOT.resolve()):
            raise AssertionError("fake host fixture unexpectedly uses the live checkout")
        identity_inputs = self._fixture_identity_inputs()
        dirty_state = self._fixture_dirty_state()

        def identity_for(candidate):
            if Path(candidate).resolve(strict=True) != root:
                raise AssertionError("fake Task 8 identity boundary received a foreign checkout")
            return identity_inputs

        def git_for(candidate, *, timeout=10.0):
            if Path(candidate).resolve(strict=True) != root:
                raise AssertionError("fake Git boundary received a foreign checkout")
            return proof.ACCEPTED_LEARNER_HEAD, dirty_state

        def configs_for(candidate):
            if Path(candidate).resolve(strict=True) != root:
                raise AssertionError("fake config boundary received a foreign checkout")
            return {"fixture": {"validated": True}}

        with patch.object(proof, "task8_identity_inputs", side_effect=identity_for), \
                patch.object(proof, "_git_state", side_effect=git_for), \
                patch.object(proof, "validate_config_pair", side_effect=configs_for):
            yield

    def _fake_run_arguments(self, root, run_id, fake):
        root = self._create_fake_checkout(root).resolve(strict=True)
        proof_root = root / PROOF_RELATIVE
        snapshot_path = proof_root / f"{run_id}-snapshot.json"
        proof.create_source_snapshot(
            root, snapshot_path,
            learner_git_revision=proof.ACCEPTED_LEARNER_HEAD,
            dirty_state=self._fixture_dirty_state(),
            identity_inputs=self._fixture_identity_inputs(),
        )
        binding = proof.verify_source_snapshot(root, snapshot_path)
        release_path = proof_root / f"{run_id}-release.json"
        release_path.write_text(json.dumps({
            "schema_id": proof.RELEASE_SCHEMA,
            "status": "released",
            "release_id": f"release-{run_id}",
            "run_id": run_id,
            "source_snapshot_raw_sha256": binding["snapshot_raw_sha256"],
            "source_snapshot_canonical_sha256": binding["snapshot_canonical_sha256"],
            "limits": proof.LIMITS,
        }, sort_keys=True) + "\n", encoding="utf-8")
        output_parent = proof_root / "outputs"
        output_parent.mkdir(parents=True, exist_ok=True)
        arguments = {
            "root": root,
            "snapshot_path": snapshot_path,
            "release_path": release_path,
            "claim_path": proof_root / f"{run_id}-claim.json",
            "run_id": run_id,
            "output_child": output_parent / run_id,
            "staging_parent": proof_root / "inputs",
            "dependencies": proof.HostDependencies(
                clock=fake.clock, run=fake.run, disk_usage=fake.disk_usage,
                uid=lambda: 1000, gid=lambda: 1000,
            ),
        }
        self._assert_fake_lifecycle_paths(arguments)
        return arguments

    def _create_fake_checkout(self, root):
        root = Path(root)
        files = {
            "src/fixture_module.py": "VALUE = 'fixture'\n",
            "tests/integration/test_fixture.py": "# authored fixture\n",
            "configs/neural/fixture.toml": "[fixture]\nvalue = 1\n",
            "scripts/neural/gpu_runtime_proof_helpers.py": "# projected helper fixture\n",
            "scripts/neural/run-gpu-runtime-proof.sh": "#!/usr/bin/env bash\nexit 0\n",
            "scripts/neural/gpu-runtime-proof-runbook.md": "fixture runbook\n",
        }
        for relative, contents in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        (root / "artifacts/development/neural-combat/task-9/gpu-proof").mkdir(
            parents=True, exist_ok=True,
        )
        return root

    def _fake_report_path(self, root, run_id):
        return Path(root) / PROOF_RELATIVE / f"host-run-{run_id}.json"

    def _assert_fake_lifecycle_paths(self, arguments):
        root = Path(arguments["root"]).resolve(strict=True)
        proof_root = root / PROOF_RELATIVE
        run_id = arguments["run_id"]
        paths = {
            "root": root,
            "snapshot": Path(arguments["snapshot_path"]),
            "release": Path(arguments["release_path"]),
            "claim": Path(arguments["claim_path"]),
            "output": Path(arguments["output_child"]),
            "stage parent": Path(arguments["staging_parent"]),
            "stage": Path(arguments["staging_parent"]) / f"source-{run_id}",
            "report": self._fake_report_path(root, run_id),
        }
        for name, path in paths.items():
            self.assertTrue(path.resolve().is_relative_to(root), f"{name} escapes temporary checkout: {path}")
            if name != "root":
                self.assertTrue(path.resolve().is_relative_to(proof_root), f"{name} escapes temporary proof root")
        self.assertFalse(root.is_relative_to(ROOT.resolve()), "fake lifecycle fixture uses the live checkout")
        for relative in (proof.TASK8_RUNTIME, proof.CUDA_LOCK, proof.CUDA_NATIVE):
            self.assertFalse((root / relative).exists(), f"fixture unexpectedly contains Task 8 runtime input: {relative}")

    def _run_fake_host(self, arguments):
        with self._patched_fixture_metadata(arguments["root"]):
            return proof.run_protected(**arguments)

    def _retained_budget_run_fixture(self, root, label, fake):
        run_id = f"gpu-proof-20261029-budget-{label}"
        arguments = self._fake_run_arguments(root, run_id, fake)
        source_files = proof.load_json(arguments["snapshot_path"])["source_files"]
        guest_inventory = proof._guest_source_inventory(source_files)
        staged_bytes = sum((Path(arguments["root"]) / name).stat().st_size for name in guest_inventory)
        report_path = self._fake_report_path(arguments["root"], run_id)
        stage_root = Path(arguments["staging_parent"]) / f"source-{run_id}"
        return arguments, stage_root, staged_bytes, report_path

    def test_failed_fake_fixture_cleanup_preserves_foreign_report_sentinel(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-fixture-sentinel-") as temporary:
            outer = Path(temporary)
            foreign_root = outer / "foreign-reports"
            foreign_root.mkdir()
            run_id = "gpu-proof-20261029-sentinel-failure"
            sentinel = foreign_root / f"host-run-{run_id}.json"
            sentinel_bytes = b"foreign report sentinel\n"
            sentinel.write_bytes(sentinel_bytes)
            fixture_temporary = tempfile.TemporaryDirectory(dir=outer)
            self.addCleanup(fixture_temporary.cleanup)
            checkout_root = self._create_fake_checkout(Path(fixture_temporary.name) / "checkout")
            fake = _FakeHost()
            with patch.object(sys.modules[__name__], "PROOF_ROOT", foreign_root, create=True):
                arguments = self._fake_run_arguments(checkout_root, run_id, fake)
                report_path = self._fake_report_path(checkout_root, run_id)
                self._assert_fake_lifecycle_paths(arguments)
                self.assertTrue(report_path.resolve().is_relative_to(checkout_root.resolve()))
                self.assertNotEqual(report_path, sentinel)
                invalid_release = json.loads(Path(arguments["release_path"]).read_text(encoding="utf-8"))
                invalid_release["run_id"] = "different-run"
                Path(arguments["release_path"]).write_text(
                    json.dumps(invalid_release, sort_keys=True) + "\n", encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "run_id"):
                    self._run_fake_host(arguments)
                self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
                self.assertFalse(Path(arguments["claim_path"]).exists())
                self.assertFalse(Path(arguments["output_child"]).exists())
                self.assertEqual(fake.calls, [])
            fixture_temporary.cleanup()
            self.doCleanups()
            self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)

    def test_retained_admission_rejects_before_staging_when_projected_total_exceeds_cap(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-retained-pre-") as temporary:
            root = Path(temporary) / "checkout"
            fake = _FakeHost()
            arguments, stage_root, staged_bytes, _report_path = self._retained_budget_run_fixture(
                root, "prestage", fake,
            )
            retained_cap = proof.LIMITS["retained_neural_bytes"]
            output_reserve = proof.OUTPUT_ADMISSION_RESERVATION_BYTES
            current_retained = retained_cap - staged_bytes - output_reserve + 1
            retained_calls = 0

            def retained_bytes(_root):
                nonlocal retained_calls
                retained_calls += 1
                return 0 if retained_calls < 3 else current_retained

            with patch.object(proof, "_retained_bytes", side_effect=retained_bytes), \
                    patch.object(proof, "stage_released_source_tree", wraps=proof.stage_released_source_tree) as stage:
                with self.assertRaisesRegex(RuntimeError, "retained storage admission before staging"):
                    self._run_fake_host(arguments)
            stage.assert_not_called()
            self.assertFalse(stage_root.exists())
            self.assertFalse(arguments["claim_path"].exists())
            self.assertFalse(arguments["output_child"].exists())
            self.assertFalse(any(call[0] == "docker" and call[5] == "create" for call in fake.calls))

    def test_retained_admission_rechecks_actual_stage_before_claim_or_create(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-retained-post-") as temporary:
            root = Path(temporary) / "checkout"
            fake = _FakeHost()
            arguments, stage_root, staged_bytes, _report_path = self._retained_budget_run_fixture(
                root, "poststage", fake,
            )
            retained_cap = proof.LIMITS["retained_neural_bytes"]
            output_reserve = proof.OUTPUT_ADMISSION_RESERVATION_BYTES
            pre_stage_retained = retained_cap - staged_bytes - output_reserve

            def retained_bytes(_root):
                if stage_root.exists():
                    return retained_cap - output_reserve + 1
                return pre_stage_retained

            with patch.object(proof, "_retained_bytes", side_effect=retained_bytes), \
                    patch.object(proof, "stage_released_source_tree", wraps=proof.stage_released_source_tree) as stage, \
                    patch.object(proof, "create_release_claim", wraps=proof.create_release_claim) as create_claim:
                with self.assertRaisesRegex(RuntimeError, "retained storage admission after staging"):
                    self._run_fake_host(arguments)
            stage.assert_called_once()
            create_claim.assert_not_called()
            self.assertTrue(stage_root.is_dir())
            self.assertFalse(arguments["claim_path"].exists())
            self.assertFalse(arguments["output_child"].exists())
            self.assertFalse(any(call[0] == "docker" and call[5] == "create" for call in fake.calls))

    def test_retained_admission_accepts_boundary_fit_with_stage_and_output_reserve(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-retained-fit-") as temporary:
            root = Path(temporary) / "checkout"
            fake = _FakeHost()
            arguments, stage_root, staged_bytes, report_path = self._retained_budget_run_fixture(
                root, "boundary", fake,
            )
            retained_cap = proof.LIMITS["retained_neural_bytes"]
            output_reserve = proof.OUTPUT_ADMISSION_RESERVATION_BYTES
            pre_stage_retained = retained_cap - staged_bytes - output_reserve
            post_stage_retained = retained_cap - output_reserve

            def retained_bytes(_root):
                if not stage_root.exists():
                    return pre_stage_retained
                actual_output = proof.output_tree_bytes(arguments["output_child"]) \
                    if arguments["output_child"].exists() else 0
                claim_bytes = arguments["claim_path"].stat().st_size if arguments["claim_path"].exists() else 0
                host_report_bytes = report_path.stat().st_size if report_path.exists() else 0
                return post_stage_retained + actual_output + claim_bytes + host_report_bytes

            with patch.object(proof, "_retained_bytes", side_effect=retained_bytes):
                result = self._run_fake_host(arguments)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["retained_admission"]["staged_source_bytes_reserved"], staged_bytes)
            self.assertEqual(result["retained_admission"]["future_output_reservation_bytes"], output_reserve)
            self.assertEqual(result["retained_admission"]["projected_retained_bytes_before_stage"], retained_cap)
            self.assertEqual(result["retained_admission"]["projected_retained_bytes_after_stage"], retained_cap)
            self.assertTrue(fake.removed)

    def test_release_and_one_use_claim_have_positive_and_negative_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_path, release_path = self._release_fixture(root)
            binding = proof.verify_release(
                release_path, snapshot_path=snapshot_path, run_id="gpu-proof-20261028-a",
            )
            self.assertEqual(binding["release_raw_sha256"], proof.sha256_file(release_path))
            claim_path = root / "claim.json"
            claim = proof.create_release_claim(claim_path, binding)
            self.assertEqual(proof.verify_release_claim(claim_path, binding), claim)
            with self.assertRaises(FileExistsError):
                proof.create_release_claim(claim_path, binding)
            invalid_claim = dict(claim)
            invalid_claim["release_raw_sha256"] = "0" * 64
            claim_path.write_text(json.dumps(invalid_claim), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "captured release"):
                proof.verify_release_claim(claim_path, binding)
            with self.assertRaisesRegex(ValueError, "run_id"):
                proof.verify_release(
                    release_path, snapshot_path=snapshot_path, run_id="other-run",
                )
            release = json.loads(release_path.read_text(encoding="utf-8"))
            release["limits"] = {**proof.LIMITS, "step_slots": 65}
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "limits"):
                proof.verify_release(
                    release_path, snapshot_path=snapshot_path, run_id="gpu-proof-20261028-a",
                )

    def _container_fixture(self):
        container_id = "b" * 64
        name = "spirelearn-task9-gpu-proof-proof-run"
        env = {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
            "SPIRE_LEARN_GPU_RUNTIME_PROOF_HELPER": "/work/gpu_runtime_proof_helpers.py",
            "SPIRE_LEARN_GPU_RUNTIME_PROOF_OUTPUT": proof.PROOF_ROOT_GUEST,
            "SPIRE_LEARN_NEURAL_GPU_SOFT_DEADLINE": "123.500000000",
            **proof.GUEST_SCRATCH_ENV,
            "PYTHONPATH": "/work/src:/work/tests/integration",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        image_defaults = [
            "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8",
            "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
            "PYTHON_VERSION=3.12.15",
            "PYTHON_SHA256=c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a",
        ]
        mounts = {
            "/work/src": ("/repo/src", False),
            "/work/tests": ("/repo/tests", False),
            "/work/configs": ("/repo/configs", False),
            "/work/gpu_runtime_proof_helpers.py": ("/repo/helper.py", False),
            "/runtime/requirements-cuda.lock": ("/repo/lock", False),
            "/runtime/venv-cuda": ("/repo/venv", False),
            "/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so": ("/repo/native.so", False),
            proof.PROOF_ROOT_GUEST: ("/repo/output", True),
        }
        inspected = {
            "Id": container_id, "Name": "/" + name, "Image": proof.IMAGE_ID,
            "Config": {
                "Image": proof.IMAGE_REFERENCE, "User": "1000:1000", "WorkingDir": "/work",
                "Entrypoint": [proof.CONTAINER_PYTHON],
                "Cmd": ["-B", "-m", "unittest", "-v", "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests"],
                "Env": image_defaults + [f"{key}={value}" for key, value in env.items()],
            },
            "HostConfig": {
                "NetworkMode": "none", "ReadonlyRootfs": True, "NanoCpus": 4_000_000_000,
                "Memory": 8 * 1024**3, "MemorySwap": 8 * 1024**3, "PidsLimit": 512,
                "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges:true"],
                "Privileged": False, "CapAdd": [], "Devices": [],
                "DeviceCgroupRules": [], "GroupAdd": [],
                "DeviceRequests": [{
                    "Driver": "", "Count": 0, "DeviceIDs": [GPU_UUID],
                    "Capabilities": [["gpu"]], "Options": {},
                }],
            },
            "Mounts": [
                {"Type": "bind", "Source": source, "Destination": destination, "RW": rw}
                for destination, (source, rw) in mounts.items()
            ],
            "State": {"Running": False, "Status": "created"},
        }
        return inspected, container_id, name, mounts

    def test_prestart_public_inspection_accepts_exact_controls_and_rejects_changes(self):
        inspected, container_id, name, mounts = self._container_fixture()
        self.assertTrue(proof.assert_owned_container(
            inspected, container_id=container_id, container_name=name, image_id=proof.IMAGE_ID,
            image_reference=proof.IMAGE_REFERENCE, user="1000:1000", expected_mounts=mounts,
        ))
        for change in (
            "network", "mount", "gpu", "state", "id", "privileged", "cap_add",
            "devices", "device_cgroup_rules", "group_add", "gpu_driver", "gpu_count",
            "gpu_options", "extra_env", "duplicate_env",
        ):
            altered = json.loads(json.dumps(inspected))
            if change == "network":
                altered["HostConfig"]["NetworkMode"] = "bridge"
            elif change == "mount":
                altered["Mounts"][0]["RW"] = True
            elif change == "gpu":
                altered["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-wrong"]
            elif change == "state":
                altered["State"]["Running"] = True
            elif change == "id":
                altered["Id"] = "c" * 64
            elif change == "privileged":
                altered["HostConfig"]["Privileged"] = True
            elif change == "cap_add":
                altered["HostConfig"]["CapAdd"] = ["SYS_ADMIN"]
            elif change == "devices":
                altered["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/nvidia0"}]
            elif change == "device_cgroup_rules":
                altered["HostConfig"]["DeviceCgroupRules"] = ["c 195:* rwm"]
            elif change == "group_add":
                altered["HostConfig"]["GroupAdd"] = ["docker"]
            elif change == "gpu_driver":
                altered["HostConfig"]["DeviceRequests"][0]["Driver"] = "nvidia"
            elif change == "gpu_count":
                altered["HostConfig"]["DeviceRequests"][0]["Count"] = 1
            elif change == "gpu_options":
                altered["HostConfig"]["DeviceRequests"][0]["Options"] = {"foo": "bar"}
            elif change == "extra_env":
                altered["Config"]["Env"].append("DOCKER_HOST=tcp://attacker.invalid:2375")
            elif change == "duplicate_env":
                altered["Config"]["Env"].append("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=0")
            with self.subTest(change=change), self.assertRaises(ValueError):
                proof.assert_owned_container(
                    altered, container_id=container_id, container_name=name,
                    image_id=proof.IMAGE_ID, image_reference=proof.IMAGE_REFERENCE,
                    user="1000:1000", expected_mounts=mounts,
                )

        altered_cleanup = json.loads(json.dumps(inspected))
        altered_cleanup["HostConfig"]["Privileged"] = True
        with self.assertRaisesRegex(ValueError, "HostConfig"):
            proof.assert_owned_container(
                altered_cleanup, container_id=container_id, container_name=name,
                image_id=proof.IMAGE_ID, image_reference=proof.IMAGE_REFERENCE,
                user="1000:1000", expected_mounts=mounts, lifecycle="cleanup",
            )

    def test_effect_limits_reserve_before_delegation_and_share_deadline(self):
        now = [1.0]
        ledger = proof.EffectLedger(deadline=10.0, clock=lambda: now[0])
        standalone_driver = _CounterDriver()
        run_driver = _CounterDriver()
        standalone = proof.GuardedDriver(standalone_driver, ledger, "standalone")
        run = proof.GuardedDriver(run_driver, ledger, "run")
        standalone.reset(seeds=(), entry_ids=(), reset_mask=(True,) * 36)
        run.reset(seeds=(), entry_ids=(), reset_mask=(True,) * 33)
        standalone.step((0,) * 32)
        run.step((0,) * 32)
        before = (standalone_driver.resets + run_driver.resets, standalone_driver.steps + run_driver.steps)
        with self.assertRaises(proof.EffectLimitError):
            run.reset(seeds=(), entry_ids=(), reset_mask=(True,))
        with self.assertRaises(proof.EffectLimitError):
            standalone.step((0,))
        self.assertEqual(before, (standalone_driver.resets + run_driver.resets, standalone_driver.steps + run_driver.steps))

        standalone_policy = _CounterPolicy()
        run_policy = _CounterPolicy()
        policy_a = proof.GuardedPolicy(standalone_policy, ledger, "standalone")
        policy_b = proof.GuardedPolicy(run_policy, ledger, "run")
        policy_a.act((object(),) * 32, generator=None)
        policy_b.act((object(),) * 32, generator=None)
        before_samples = standalone_policy.rows + run_policy.rows
        with self.assertRaises(proof.EffectLimitError):
            policy_b.act((object(),), generator=None)
        self.assertEqual(standalone_policy.rows + run_policy.rows, before_samples)

        ledger.reserve_diagnostic_pair("standalone")
        diagnostic = proof.GuardedPolicy(
            _CounterPolicy(), ledger, "standalone", diagnostic=True,
        )
        diagnostic.act((object(),), generator=None)
        diagnostic.act((object(),), generator=None)
        optimizer_a, optimizer_b = _CounterOptimizer(), _CounterOptimizer()
        guarded_a = proof.GuardedOptimizer(optimizer_a, ledger, "standalone")
        guarded_b = proof.GuardedOptimizer(optimizer_b, ledger, "run")
        for _ in range(4):
            guarded_a.step()
            guarded_b.step()
        attempts_before = optimizer_a.steps + optimizer_b.steps
        with self.assertRaises(proof.EffectLimitError):
            guarded_b.step()
        self.assertEqual(optimizer_a.steps + optimizer_b.steps, attempts_before)
        self.assertEqual(ledger.snapshot()["optimizer_attempted"], 8)
        self.assertEqual(ledger.snapshot()["diagnostic_samples_delivered"], 2)

        now[0] = 10.1
        later_driver = _CounterDriver()
        with self.assertRaises(proof.ProofDeadlineExceeded):
            proof.GuardedDriver(later_driver, ledger, "run").step((0,))
        self.assertEqual(later_driver.steps, 0)

    def test_collector_sampling_guard_observes_ppo_version_mutation_on_inner_policy(self):
        ledger = proof.EffectLedger(deadline=100.0, clock=lambda: 1.0)
        runtime_policy = _VersionPolicy()
        collector_policy = proof.GuardedPolicy(runtime_policy, ledger, "standalone")
        # PPO mutates runtime.policy directly; the collector wrapper must keep
        # reading the current model version instead of shadowing it on a proxy.
        runtime_policy._behavior_policy_version += 1
        choices = collector_policy.act((object(),), generator=None, deterministic=False)
        self.assertEqual(choices[0].behavior_policy_version, 1)
        self.assertEqual(collector_policy._behavior_policy_version, 1)
        self.assertEqual(ledger.mode_snapshot("standalone")["actor_samples_delivered"], 1)

    def test_projected_warm_start_check_uses_the_real_counting_wrapper(self):
        sys.path.insert(0, str(ROOT / "src"))
        from spire_learn.neural.runner import _CountingOptimizer

        class StrictAdam:
            def __init__(self):
                self.state = {}
                self.param_groups = []

            def state_dict(self):
                return {"state": self.state, "param_groups": self.param_groups}

        adam = StrictAdam()
        optimizer = _CountingOptimizer(adam)
        self.assertFalse(hasattr(adam, "attempted"))
        self.assertFalse(hasattr(adam, "accepted"))
        with tempfile.TemporaryDirectory() as temporary:
            _, module = self._load_projected(Path(temporary), enabled=True)
            test_instance = module.NeuralGpuRuntimeProofTests("test_01_standalone_b4_h8")
            module._assert_fresh_warm_optimizer(test_instance, optimizer)

    def test_projected_resume_check_uses_wrapper_counters_with_restored_adam_state(self):
        sys.path.insert(0, str(ROOT / "src"))
        from spire_learn.neural.runner import _CountingOptimizer

        class StrictAdam:
            def __init__(self):
                self.state = {"parameter": {"step": 3, "exp_avg": (0.25,), "exp_avg_sq": (0.5,)}}
                self.param_groups = []

            def state_dict(self):
                return {"state": self.state, "param_groups": self.param_groups}

        adam = StrictAdam()
        optimizer = _CountingOptimizer(adam)
        self.assertTrue(adam.state, "resume fixture must retain nonempty Adam moments")
        self.assertFalse(hasattr(adam, "attempted"))
        self.assertFalse(hasattr(adam, "accepted"))
        with tempfile.TemporaryDirectory() as temporary:
            _, module = self._load_projected(Path(temporary), enabled=True)
            test_instance = module.NeuralGpuRuntimeProofTests("test_01_standalone_b4_h8")
            module._assert_fresh_resume_optimizer(test_instance, optimizer)

            for counter in ("attempted", "accepted"):
                setattr(optimizer, counter, 1)
                with self.subTest(counter=counter), self.assertRaises(AssertionError):
                    module._assert_fresh_resume_optimizer(test_instance, optimizer)
                setattr(optimizer, counter, 0)

    def test_mode_result_and_test_transcript_reject_skips_incomplete_effects(self):
        mode = "standalone"
        counters = {
            "steps": 32, "resets": 36, "actor_samples": 32,
            "diagnostic_samples": 2, "optimizer": 4,
            "steps_attempted": 32, "steps_delivered": 32, "steps_uncertain": 0,
            "resets_attempted": 36, "resets_delivered": 36, "resets_uncertain": 0,
            "actor_samples_attempted": 32, "actor_samples_delivered": 32, "actor_samples_uncertain": 0,
            "diagnostic_samples_attempted": 2, "diagnostic_samples_delivered": 2,
            "diagnostic_samples_uncertain": 0, "optimizer_attempted": 4,
            "optimizer_accepted": 4, "optimizer_uncertain": 0,
            "faults": 0, "incomplete_environment_slots": 0,
        }
        checks = {
            "one_shared_deadline": True,
            "accepted_finite_update_and_changed_weights": True,
            "final_checkpoint_saved": True,
            "same_backend_weights_and_adam": True,
            "adam_scalar_steps_cpu_moments_cuda": True,
            "same_backend_torch_rng_and_python_rng": True,
            "same_backend_counters_version_and_recovery": True,
            "same_frame_deterministic_and_sampled_full_choice_parity": True,
            "portable_cpu_weights_and_tolerant_outputs": True,
            "warm_start_fresh_state_and_frozen_evaluation": True,
            "cpu_comparisons_used_inert_drivers_without_steps": True,
            "training_driver_closed_once": True,
            "actor_critic_and_shared_hashes_recorded": True,
            "zero_fault_uncertainty_and_incomplete_effects": True,
        }
        result = {
            "schema_id": proof.MODE_RESULT_SCHEMA, "status": "passed", "mode": mode,
            "test_id": "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
            "exit_code": 0, "skipped": False,
            "updates": {"accepted": 1, "finite": True, "weights_changed": True},
            "counters": counters, "checks": checks,
            "weights": {
                "model_sha256": "1" * 64, "shared_body_sha256": "2" * 64,
                "actor_head_sha256": "3" * 64, "critic_head_sha256": "4" * 64,
            },
            "checkpoint": {
                "path": "standalone/run/checkpoint-final",
                "metadata_raw_sha256": "5" * 64,
                "metadata_canonical_sha256": "6" * 64,
            },
            "portable_cpu": {
                "warm_start_optimizer_fresh": True, "warm_start_rng_fresh": True,
                "evaluation_optimizer_absent": True,
                "cpu_driver_resets": 0, "cpu_driver_steps": 0,
            },
        }
        self.assertEqual(proof.validate_mode_result(
            result, mode=mode, test_id=result["test_id"],
        ), result)
        invalid = dict(result, skipped=True)
        with self.assertRaisesRegex(ValueError, "skip"):
            proof.validate_mode_result(invalid, mode=mode, test_id=result["test_id"])
        invalid = json.loads(json.dumps(result))
        invalid["counters"]["optimizer_uncertain"] = 1
        with self.assertRaisesRegex(ValueError, "uncertain"):
            proof.validate_mode_result(invalid, mode=mode, test_id=result["test_id"])
        for missing in ("steps_uncertain", "faults", "incomplete_environment_slots"):
            invalid = json.loads(json.dumps(result))
            del invalid["counters"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "counter"):
                proof.validate_mode_result(invalid, mode=mode, test_id=result["test_id"])
        for field, invalid_value in (("faults", True), ("steps_uncertain", 0.0)):
            invalid = json.loads(json.dumps(result))
            invalid["counters"][field] = invalid_value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "integer"):
                proof.validate_mode_result(invalid, mode=mode, test_id=result["test_id"])
        invalid = json.loads(json.dumps(result))
        invalid["counters"]["unlisted_counter"] = 0
        with self.assertRaisesRegex(ValueError, "counter"):
            proof.validate_mode_result(invalid, mode=mode, test_id=result["test_id"])
        good = (
            "test_01_standalone_b4_h8 (test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8) ... ok\n"
            "test_02_run_b1_h32 (test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32) ... ok\n\n"
            "Ran 2 tests in 0.01s\n\nOK\n"
        )
        self.assertEqual(proof.verify_test_transcript(good)["skipped"], 0)
        with self.assertRaisesRegex(ValueError, "exact mode once"):
            proof.verify_test_transcript(good.replace("... ok", "... skipped 'held'", 1))

    def test_evidence_inventory_is_independent_and_detects_late_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.json").write_text('{"status":"passed"}\n', encoding="utf-8")
            (root / "container.stdout.log").write_text("2 tests passed\n", encoding="utf-8")
            inventory_path = root / "evidence-inventory.json"
            inventory = proof.create_evidence_inventory(root, inventory_path)
            self.assertEqual(proof.verify_evidence_inventory(root, inventory_path), inventory)
            (root / "container.stdout.log").write_text("changed after inventory\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                proof.verify_evidence_inventory(root, inventory_path)

    def _load_projected(self, root, *, enabled, helper_present=True):
        projected = root / "projected"
        test_path = projected / "tests/integration/test_neural_gpu_runtime.py"
        test_path.parent.mkdir(parents=True)
        shutil.copy2(INTEGRATION_PATH, test_path)
        helper_path = projected / "gpu_runtime_proof_helpers.py"
        if helper_present:
            shutil.copy2(HELPER_PATH, helper_path)
        module_name = f"projected_gpu_integration_{int(enabled)}_{int(helper_present)}"
        spec = importlib.util.spec_from_file_location(module_name, test_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        env = {"SPIRE_LEARN_GPU_RUNTIME_PROOF_HELPER": str(helper_path)}
        if enabled:
            env["SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME"] = "1"
            env["SPIRE_LEARN_NEURAL_GPU_UUID"] = GPU_UUID
        else:
            env.pop("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME", None)
        with patch.dict(os.environ, env, clear=False):
            if not enabled:
                os.environ.pop("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME", None)
            spec.loader.exec_module(module)
        return projected, module

    def test_gate_off_import_and_discovery_block_all_optional_prefixes(self):
        original_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".", 1)[0] in OPTIONAL_PREFIXES:
                raise AssertionError(f"optional import while gate is off: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch("builtins.__import__", blocked):
            _, module = self._load_projected(Path(temporary), enabled=False)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.NeuralGpuRuntimeProofTests)
            result = unittest.TestResult()
            suite.run(result)
            self.assertEqual(result.testsRun, 2)
            self.assertEqual(len(result.skipped), 2)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])

    def test_projected_enabled_methods_bind_runner_aliases_without_extra_self(self):
        class _ReachedCreateRuntime(Exception):
            pass

        class StrictLedger:
            @staticmethod
            def check_deadline():
                return None

        class Budgets:
            max_training_decisions = 32
            max_environment_actions = 32
            max_optimizer_steps = 4
            max_wall_seconds = 120.0

        class Config:
            rollout_rows = 4
            rollout_steps = 8
            budgets = Budgets()

        calls = []

        def load_neural_config(path, /):
            calls.append(("load_neural_config", path))
            return "loaded-config"

        def create_runtime(config, *, purpose):
            calls.append(("create_runtime", config, purpose))
            raise _ReachedCreateRuntime()

        def evaluate_policy(config, *, runtime, clock):
            calls.append(("evaluate_policy", config, runtime, clock))
            return "evaluated"

        def train(config, *, runtime, clock):
            calls.append(("train", config, runtime, clock))
            return "trained"

        with tempfile.TemporaryDirectory() as temporary:
            projected, module = self._load_projected(Path(temporary), enabled=True)
            test_case = module.NeuralGpuRuntimeProofTests
            names = tuple(name for name in test_case.__dict__ if name.startswith("test_"))
            self.assertEqual(names, ("test_01_standalone_b4_h8", "test_02_run_b1_h32"))
            module._install_runner_api_aliases(
                test_case,
                load_neural_config=load_neural_config,
                create_runtime=create_runtime,
                evaluate_policy=evaluate_policy,
                train=train,
            )
            instance = test_case("test_01_standalone_b4_h8")
            instance.ledger = StrictLedger()
            instance.configs = {"standalone": Config()}
            bound_test = instance.test_01_standalone_b4_h8
            self.assertIs(bound_test.__self__, instance)
            with self.assertRaises(_ReachedCreateRuntime):
                bound_test()

            config = instance.configs["standalone"]
            runtime = object()
            clock = lambda: 0.0
            self.assertEqual(instance.load_neural_config("cpu.toml"), "loaded-config")
            self.assertEqual(instance.evaluate_policy(config, runtime=runtime, clock=clock), "evaluated")
            self.assertEqual(instance.train(config, runtime=runtime, clock=clock), "trained")
            self.assertEqual(calls, [
                ("create_runtime", config, "train"),
                ("load_neural_config", "cpu.toml"),
                ("evaluate_policy", config, runtime, clock),
                ("train", config, runtime, clock),
            ])

            self.assertTrue((projected / "gpu_runtime_proof_helpers.py").is_file())
            helper_exports = set(proof.__all__)
            self.assertEqual(len(helper_exports), len(proof.__all__))
            self.assertTrue(all(hasattr(proof, name) for name in helper_exports))
            self.assertTrue({
                "EffectLedger", "GuardedDriver", "GuardedPolicy", "GuardedOptimizer",
                "create_source_snapshot", "verify_release", "run_protected",
            } <= helper_exports)

        sys.path.insert(0, str(ROOT / "src"))
        from spire_learn.neural import runner
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.runner import RuntimeOperationLimits, create_runtime, train

        expected_runner_exports = {
            "NeuralReport", "RuntimeOperationLimits", "TrainingRuntime", "calibrate",
            "create_runtime", "evaluate_policy", "train",
        }
        self.assertEqual(set(runner.__all__), expected_runner_exports)
        self.assertEqual(proof.verify_test_exports(ROOT), tuple(sorted(expected_runner_exports)))
        self.assertIs(runner.train, train)
        self.assertIs(runner.create_runtime, create_runtime)
        self.assertIs(runner.RuntimeOperationLimits, RuntimeOperationLimits)
        self.assertTrue(callable(load_neural_config))

    def test_missing_projected_helper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ImportError, "proof helper is missing"):
                self._load_projected(Path(temporary), enabled=True, helper_present=False)

    def test_full_protected_control_flow_with_fake_host_and_positive_negative_transcript(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-pure-") as temporary:
            checkout = self._create_fake_checkout(Path(temporary) / "checkout")
            run_id = "fake-control-20261028-a"
            fake = _FakeHost()
            arguments = self._fake_run_arguments(checkout, run_id, fake)
            self._assert_fake_lifecycle_paths(arguments)
            snapshot_path = Path(arguments["snapshot_path"])
            release_path = Path(arguments["release_path"])
            claim_path = Path(arguments["claim_path"])
            output_child = Path(arguments["output_child"])
            report_path = self._fake_report_path(checkout, run_id)
            with patch.dict(os.environ, {
                "DOCKER_HOST": "tcp://hostile.invalid:2375",
                "DOCKER_CONTEXT": "hostile-context",
                "DOCKER_CONFIG": str(Path(temporary) / "hostile-docker-config"),
                "DOCKER_TLS_VERIFY": "1",
                "DOCKER_CERT_PATH": str(Path(temporary) / "hostile-certs"),
            }, clear=False):
                result = self._run_fake_host(arguments)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["container_removed"])
            self.assertTrue(fake.create_scratch_ready)
            self.assertTrue(fake.client_config_paths)
            self.assertTrue(all(not path.exists() for path in fake.client_config_paths))
            self.assertEqual(result["test_transcript"]["skipped"], 0)
            self.assertEqual(result["evidence_inventory"]["relative_path"], "evidence-inventory.json")
            self.assertTrue(claim_path.is_file())
            self.assertEqual(proof.verify_evidence_inventory(
                output_child, output_child / "evidence-inventory.json",
            )["file_count"], 4)
            self.assertEqual(set(result["mode_evidence"]["modes"]), {"standalone", "run"})
            self.assertTrue(report_path.is_file())
            commands = [call[5:] for call in fake.calls if call[0] == "docker"]
            self.assertLess(commands.index(("container", "ls", "--all", "--filter",
                                            f"name=^spirelearn-task9-gpu-proof-{run_id}$",
                                            "--format", "{{.ID}}")),
                            next(index for index, command in enumerate(commands) if command[0] == "create"))
            self.assertTrue(any(GPU_UUID in item for item in fake.create_argv))
            with self.assertRaisesRegex(ValueError, "each exact mode once"):
                proof.verify_test_transcript("Ran 2 tests\n\nOK\n")
            bad_release = checkout / PROOF_RELATIVE / "bad-parent-release.json"
            invalid = json.loads(release_path.read_text(encoding="utf-8"))
            invalid["run_id"] = "different-released-run"
            bad_release.write_text(json.dumps(invalid), encoding="utf-8")
            bad_claim = checkout / PROOF_RELATIVE / "bad-claim.json"
            bad_output = checkout / PROOF_RELATIVE / "outputs/must-not-be-created"
            rejected_host = _FakeHost()
            bad_arguments = {
                **arguments,
                "release_path": bad_release,
                "claim_path": bad_claim,
                "output_child": bad_output,
                "dependencies": proof.HostDependencies(
                    clock=rejected_host.clock, run=rejected_host.run,
                    disk_usage=rejected_host.disk_usage, uid=lambda: 1000,
                    gid=lambda: 1000,
                ),
            }
            self._assert_fake_lifecycle_paths(bad_arguments)
            with self.assertRaisesRegex(ValueError, "run_id"):
                self._run_fake_host(bad_arguments)
            self.assertEqual(rejected_host.calls, [])
            self.assertFalse(bad_claim.exists())
            self.assertFalse(bad_output.exists())

    def test_cleanup_rechecks_full_spec_and_preserves_first_process_control(self):
        with tempfile.TemporaryDirectory(prefix="gpu-proof-control-") as temporary:
            checkout = self._create_fake_checkout(Path(temporary) / "checkout")
            report_paths = []

            def arguments(label, fake):
                run_id = f"control-{label}-20261029-a"
                values = self._fake_run_arguments(checkout, run_id, fake)
                report_path = self._fake_report_path(checkout, run_id)
                report_paths.append(report_path)
                self._assert_fake_lifecycle_paths(values)
                return values

            def invoke(label, fake):
                return self._run_fake_host(arguments(label, fake))

            changed_spec = _FakeHost()
            changed_spec.cleanup_spec_change = ("Privileged", True)
            with self.assertRaisesRegex(ValueError, "HostConfig"):
                invoke("changed-spec", changed_spec)
            changed_commands = [call[5:] for call in changed_spec.calls if call[0] == "docker"]
            self.assertFalse(changed_spec.removed)
            self.assertFalse(any(command[:2] == ("container", "rm") for command in changed_commands))

            ordinary_then_control = _FakeHost()
            ordinary_then_control.start_returncode = 1
            ordinary_then_control.command_exceptions[("container", "inspect", 2)] = KeyboardInterrupt(
                "cleanup interrupt after ordinary start failure",
            )
            with self.assertRaises(KeyboardInterrupt) as caught:
                invoke("ordinary-first", ordinary_then_control)
            self.assertTrue(any("earlier ordinary failure" in note for note in caught.exception.__notes__))
            self.assertFalse(ordinary_then_control.removed)

            two_controls = _FakeHost()
            two_controls.command_exceptions[("start", "--attach")] = KeyboardInterrupt("first control")
            two_controls.command_exceptions[("container", "inspect", 2)] = SystemExit("second control")
            with self.assertRaises(KeyboardInterrupt) as caught:
                invoke("two-controls", two_controls)
            self.assertEqual(str(caught.exception), "first control")
            self.assertTrue(any("later process-control" in note for note in caught.exception.__notes__))

            control_only = _FakeHost()
            control_only.command_exceptions[("start", "--attach")] = KeyboardInterrupt("only control")
            with self.assertRaisesRegex(KeyboardInterrupt, "only control"):
                invoke("control-only", control_only)
            self.assertTrue(control_only.removed)

            final_clock = _FakeHost()
            final_clock.clock_control_after_cleanup = True
            with self.assertRaisesRegex(KeyboardInterrupt, "final clock control"):
                invoke("final-clock", final_clock)
            self.assertTrue(final_clock.removed)
            final_clock_report = report_paths[-1]
            self.assertNotEqual(
                json.loads(final_clock_report.read_text(encoding="utf-8"))["status"], "passed",
            )

            expired_deadline = _FakeHost()
            expired_deadline.clock_after_cleanup = 1000.0
            with self.assertRaises(proof.ProofDeadlineExceeded):
                invoke("expired-deadline", expired_deadline)
            self.assertTrue(expired_deadline.removed)

            final_write = _FakeHost()
            final_write_args = arguments("final-write", final_write)
            failed_report_path = report_paths[-1]
            real_exclusive_write = proof.write_json_exclusive

            def fail_final_host_report(path, value):
                if Path(path) == failed_report_path:
                    raise SystemExit("final report write control")
                return real_exclusive_write(path, value)

            with patch.object(proof, "write_json_exclusive", side_effect=fail_final_host_report):
                with self.assertRaisesRegex(SystemExit, "final report write control"):
                    self._run_fake_host(final_write_args)
            self.assertTrue(final_write.removed)
            self.assertFalse(failed_report_path.exists())


if __name__ == "__main__":
    unittest.main()
