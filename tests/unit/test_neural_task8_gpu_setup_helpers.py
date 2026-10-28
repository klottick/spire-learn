"""Pure, stdlib-only checks for the held Task 8 GPU setup controls."""

from __future__ import annotations

import hashlib
import importlib
import io
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neural import task8_gpu_setup_helpers as helpers


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TARGET_TAGS = frozenset({
    "py3-none-any",
    "py3-none-manylinux_2_17_x86_64",
    "cp312-cp312-manylinux_2_17_x86_64",
    "cp312-cp312-manylinux_2_28_x86_64",
})


def _image_inspect(*, image_id="sha256:" + "a" * 64, architecture="amd64", repo_digests=None,
                   repo_tags=None):
    return json.dumps([{
        "Id": image_id,
        "Os": "linux",
        "Architecture": architecture,
        "RepoDigests": repo_digests if repo_digests is not None else [helpers.PYTHON_CANONICAL_REPO_DIGEST],
        "RepoTags": repo_tags if repo_tags is not None else ["spire-learn/task8-python:run-1"],
    }])


def _matching_corrected_container_inspect(helper: Path, work: Path, *, container_id="c" * 64):
    command = ["-ec", "true"]
    inspect = [{
        "Id": container_id,
        "Name": "/spirelearn-task8-rng-red-corrected-1-cuda-setup-20261028-a",
        "Image": "sha256:" + "a" * 64,
        "Config": {
            "User": "1000:1000", "WorkingDir": "/learner",
            "Entrypoint": ["/bin/sh"], "Cmd": command,
            "Labels": {
                "spire-learn.task": "task8-gpu-setup", "spire-learn.phase": "rng-red",
                "spire-learn.run-id": "cuda-setup-20261028-a",
                "spire-learn.attempt-id": "corrected-1",
            },
            "Env": [
                "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
                "PIP_CONFIG_FILE=/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                "PIP_NO_INPUT=1", "HOME=/tmp", "TMPDIR=/work/tmp",
                "PIP_CACHE_DIR=/work/pip-cache",
            ],
        },
        "HostConfig": {
            "NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
            "AutoRemove": False, "CapDrop": ["ALL"], "CapAdd": None,
            "GroupAdd": None, "DeviceCgroupRules": None,
            "SecurityOpt": ["no-new-privileges"], "NanoCpus": 4_000_000_000,
            "Memory": 8 * 1024**3, "PidsLimit": 512, "Devices": [],
            "DeviceRequests": [{"DeviceIDs": [helpers.GPU_UUID], "Capabilities": [["gpu"]]}],
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
        },
        "NetworkSettings": {"Networks": {"none": {}}},
        "Mounts": [
            {"Type": "bind", "Source": str(helper.resolve()),
             "Destination": "/learner/scripts/neural/task8_gpu_setup_helpers.py", "RW": False},
            {"Type": "bind", "Source": str(work.resolve()), "Destination": "/work", "RW": True},
        ],
    }]
    inspect_args = {
        "expected_image_id": "sha256:" + "a" * 64,
        "run_id": "cuda-setup-20261028-a", "phase": "rng-red",
        "attempt_id": "corrected-1", "expected_command": command,
        "expected_mounts": (
            (helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
            (work, "/work", False),
        ),
        "network": "none", "gpu_uuid": helpers.GPU_UUID,
    }
    return inspect, inspect_args


def _make_wheel(
    directory: Path, package: str, version: str, requires=(), *,
    filename_override=None, wheel_tag_override=None,
) -> tuple[str, str, list[str]]:
    normalized = helpers.canonical_distribution_name(package).replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    filename = filename_override or f"{normalized}-{version}-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {package}\n"
        f"Version: {version}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requires)
        + "\n"
    ).encode("utf-8")
    path = directory / filename
    parts = filename[:-4].split("-")[-3:]
    wheel_tags = wheel_tag_override or (
        f"{python}-{abi}-{platform}"
        for python, abi, platform in product(
            parts[0].split("."), parts[1].split("."), parts[2].split("."),
        )
    )
    tags = "".join(f"Tag: {tag}\n" for tag in wheel_tags)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\n{tags}")
    return filename, hashlib.sha256(path.read_bytes()).hexdigest(), list(requires)


def _closure_fixture(
    directory: Path, *, compressed_cuda_bindings=False,
    incompatible_cuda_bindings_metadata=False,
) -> dict[str, object]:
    packages = []
    names_and_versions = {
        "torch": "2.14.1+cu130",
        "numpy": "2.5.4",
        "gymnasium": "1.4.0",
        "spire_lab": "0.1.0",
        "filelock": "3.0",
        "typing-extensions": "4.10.0",
        "setuptools": "77.0.3",
        "sympy": "1.13.3",
        "networkx": "2.5.1",
        "jinja2": "3.0",
        "fsspec": "0.8.5",
        "cuda-toolkit": "13.0.3",
        "cuda-bindings": "13.0.3",
        "nvidia-cudnn-cu13": "9.24.0.43",
        "nvidia-cusparselt-cu13": "0.8.1",
        "nvidia-nccl-cu13": "2.30.7",
        "nvidia-nvshmem-cu13": "3.4.5",
        "triton": "3.8.0",
    }
    runtime = ROOT / "artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
    wheel_audit_path = runtime / "new-wheel-audit.json"
    wheel_audit = json.loads(wheel_audit_path.read_text(encoding="utf-8"))
    local_wheel = Path(wheel_audit["wheel"])
    local_metadata, local_identity, local_requires, _local_tags = helpers._wheel_metadata(local_wheel)
    packages.append({
        "name": local_identity["name"],
        "version": local_identity["version"],
        "filename": local_wheel.name,
        "source_url": f"file:///inputs/{helpers.CPU_WHEEL_FILENAME}",
        "source_response": {"kind": "local-read-only-payload", "sha256": wheel_audit["wheel_sha256"]},
        "local_payload_path": str(local_wheel),
        "expected_sha256": wheel_audit["wheel_sha256"],
        "observed_sha256": wheel_audit["wheel_sha256"],
        "metadata_sha256": hashlib.sha256(local_metadata).hexdigest(),
        "requires_dist": local_requires,
    })
    torch_requires = [
        "filelock",
        "typing-extensions>=4.10.0",
        "setuptools>=77.0.3",
        "sympy>=1.13.3",
        "networkx>=2.5.1",
        "jinja2",
        "fsspec>=0.8.5",
        "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"",
        "cuda-bindings<14,>=13.0.3; platform_system == \"Linux\" and python_version < \"3.15\"",
        "nvidia-cudnn-cu13==9.24.0.43; platform_system == \"Linux\"",
        "nvidia-cusparselt-cu13==0.8.1; platform_system == \"Linux\"",
        "nvidia-nccl-cu13==2.30.7; platform_system == \"Linux\"",
        "nvidia-nvshmem-cu13==3.4.5; platform_system == \"Linux\"",
        "triton~=3.8.0; platform_system == \"Linux\" and python_version < \"3.15\"",
    ]
    for package, version in names_and_versions.items():
        if package == "spire_lab":
            continue
        requires = torch_requires if package == "torch" else []
        filename_override = helpers.CUDA_TORCH_FILENAME if package == "torch" else None
        wheel_tag_override = None
        if package == "cuda-bindings" and compressed_cuda_bindings:
            filename_override = (
                "cuda_bindings-13.0.3-py2.py3-none.abi3-"
                "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            )
            if incompatible_cuda_bindings_metadata:
                wheel_tag_override = ("cp311-cp311-win_amd64",)
        filename, digest, requires_dist = _make_wheel(
            directory, package, version, requires, filename_override=filename_override,
            wheel_tag_override=wheel_tag_override,
        )
        if package == "torch":
            source_url = f"{helpers.CUDA_TORCH_URL}#sha256={digest}"
            source_response = {
                "kind": "http", "status": 200,
                "method": "HEAD",
                "final_url": helpers.CUDA_TORCH_URL,
                "headers": [["Content-Length", str((directory / filename).stat().st_size)]],
                "redacted_header_names": ["set-cookie", "cookie", "authorization"],
            }
        else:
            source_url = f"https://files.pythonhosted.org/packages/test/{filename}#sha256={digest}"
            source_response = {
                "kind": "http", "method": "HEAD", "status": 200,
                "final_url": source_url.split("#", 1)[0],
                "headers": [["Content-Length", str((directory / filename).stat().st_size)]],
                "redacted_header_names": ["set-cookie", "cookie", "authorization"],
            }
        packages.append({
            "name": package,
            "version": version,
            "filename": filename,
            "source_url": source_url,
            "source_response": source_response,
            "expected_sha256": digest,
            "observed_sha256": digest,
            "metadata_sha256": _wheel_metadata_hash(directory / filename),
            "requires_dist": requires_dist,
        })
    torch_record = next(item for item in packages if item["name"] == "torch")
    return {
        "format": "spire-learn-task8-cuda-closure-v1",
        "resolver": {
            "exit_code": 0,
            "command": [
                "python3.12", "-B", "-m", "pip", "install", "--dry-run", "--ignore-installed",
                "--only-binary=:all:", "--index-url", helpers.PYTHON_INDEX_URL,
            ],
            "only_binary": True,
            "complete_resolution": True,
            "python": "3.12",
            "platform": "linux_x86_64",
            "index_url": helpers.PYTHON_INDEX_URL,
            "report_sha256": "f" * 64,
            "resolved_packages": sorted(names_and_versions),
        },
        "fixture_torch_sha256": torch_record["observed_sha256"],
        "fixture_torch_metadata_sha256": torch_record["metadata_sha256"],
        "packages": packages,
    }


def _wheel_metadata_hash(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return hashlib.sha256(archive.read(metadata_path)).hexdigest()


class _FakeTensor:
    def __init__(self, finite=True):
        self.finite = finite
        self.raise_on_finite = False
        self.grad = self

    def to(self, **_kwargs):
        return self

    def reshape(self, *_shape):
        return self

    def square(self):
        return self

    def mean(self):
        return self

    def backward(self):
        return None

    def all(self):
        return self

    def item(self):
        return self.finite


class _FakeOptimizer:
    def __init__(self, parameters, *, fail_after_step=False, raise_during_step=False):
        self.parameters = list(parameters)
        self.step_calls = 0
        self.fail_after_step = fail_after_step
        self.raise_during_step = raise_during_step
        self.state = {
            parameter: {
                "step": _FakeTensor(), "exp_avg": _FakeTensor(), "exp_avg_sq": _FakeTensor(),
            }
            for parameter in self.parameters
        }

    def step(self):
        self.step_calls += 1
        if self.raise_during_step:
            raise RuntimeError("synthetic Adam step exception")
        if self.fail_after_step:
            self.state[self.parameters[0]]["exp_avg"].raise_on_finite = True


class _FakeTorch:
    __version__ = "2.14.1+cu130"
    float32 = "float32"

    def __init__(self, *, fail_after_step=False, raise_during_step=False):
        self.version = SimpleNamespace(cuda="13.0.3")
        self.optimizer = None
        self.fail_after_step = fail_after_step
        self.raise_during_step = raise_during_step
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            set_device=lambda _index: None,
            init=lambda: None,
            is_initialized=lambda: True,
            get_device_properties=lambda _index: SimpleNamespace(
                name=helpers.GPU_NAME, total_memory=helpers.GPU_MEMORY_MIB * 1024**2,
            ),
            get_device_capability=lambda _index: (12, 0),
            get_arch_list=lambda: ["sm_120"],
            set_per_process_memory_fraction=lambda *_args: None,
            reset_peak_memory_stats=lambda *_args: None,
            synchronize=lambda *_args: None,
            max_memory_allocated=lambda _index: 1024,
            max_memory_reserved=lambda _index: 2048,
            memory_allocated=lambda _index: 512,
            memory_reserved=lambda _index: 1024,
        )
        self.nn = SimpleNamespace(
            Linear=lambda *_args: object(),
            Tanh=lambda: object(),
            Sequential=lambda *_args: self._model(),
        )
        self.optim = SimpleNamespace(Adam=self._adam)

    def _model(self):
        class FakeModel:
            def __init__(self):
                self.parameter = _FakeTensor()

            def to(self, **_kwargs):
                return self

            def parameters(self):
                return [self.parameter]

            def __call__(self, _inputs):
                return _FakeTensor()

        return FakeModel()

    def _adam(self, parameters, **_kwargs):
        self.optimizer = _FakeOptimizer(
            parameters, fail_after_step=self.fail_after_step,
            raise_during_step=self.raise_during_step,
        )
        return self.optimizer

    @staticmethod
    def linspace(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        if not isinstance(value, _FakeTensor):
            raise TypeError("isfinite accepts tensor values only")
        if value.raise_on_finite:
            raise RuntimeError("synthetic post-Adam finite inspection fault")
        return value


class Task8GpuSetupHelperTests(unittest.TestCase):
    def test_parent_cuda_candidate_metadata_is_reconciled_without_payload_download(self):
        result = helpers.verify_candidate_metadata(
            ROOT / "artifacts/development/neural-combat/task-8/parent-package-preflight/metadata-preflight.json",
            ROOT / "artifacts/development/neural-combat/task-8/parent-package-preflight/torch-wheel.metadata",
        )
        self.assertEqual(result["expected_sha256"], helpers.CUDA_TORCH_SHA256)
        self.assertEqual(result["payload_head_bytes"], 554621169)
        self.assertIs(result["payload_downloaded"], False)

    def test_independently_built_gym_wheel_is_verified_as_only_read_only_dependency(self):
        runtime = ROOT / "artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
        result = helpers.verify_cpu_wheel_payload(
            ROOT / "artifacts/development/neural-combat/cpu-runtime-location.json",
            runtime / "new-wheel-audit.json",
        )
        self.assertEqual(result["wheel_sha256"], "088296349eff7253530db2a85ebfb3a4f88aff658fc9cf7e8ed41cd2acc25777")
        self.assertIs(result["native_payload_mounted"], False)

    def test_canonical_image_digest_alias_and_architecture_are_pinned(self):
        raw = _image_inspect()
        identity = helpers.verify_image_inspect(raw)
        alias = helpers.verify_local_image_alias(
            raw, alias="spire-learn/task8-python:run-1", expected_image_id=identity["image_id"],
        )
        self.assertEqual(alias["architecture"], "amd64")
        self.assertEqual(alias["repo_digest"], helpers.PYTHON_CANONICAL_REPO_DIGEST)
        qualified = helpers.verify_image_inspect(_image_inspect(
            repo_digests=[helpers.PYTHON_QUALIFIED_REPO_DIGEST],
        ))
        self.assertEqual(qualified["repo_digest"], helpers.PYTHON_QUALIFIED_REPO_DIGEST)
        for bad in (
            _image_inspect(architecture="arm64"),
            _image_inspect(repo_digests=["python:3.12-slim"]),
            _image_inspect(image_id="bad"),
        ):
            with self.subTest(bad=bad), self.assertRaises(helpers.Task8ControlError):
                helpers.verify_image_inspect(bad)
        with self.assertRaisesRegex(helpers.Task8ControlError, r"\.Id changed"):
            helpers.verify_local_image_alias(
                _image_inspect(image_id="sha256:" + "b" * 64),
                alias="spire-learn/task8-python:run-1",
                expected_image_id="sha256:" + "a" * 64,
            )

    def test_docker_create_argv_is_network_and_uuid_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "fresh-work"
            helper = root / "task8_gpu_setup_helpers.py"
            work.mkdir()
            helper.write_text("# fixture\n", encoding="utf-8")
            mounts = ((helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
                      (work, "/work", False))
            setup = helpers.build_docker_create_argv(
                image="spire-learn/task8-python:run-1", run_id="run-1", phase="resolve",
                command=("-ec", "python -B -m pip download --only-binary=:all:"),
                mounts=mounts, network="bridge",
            )
            self.assertIn("--network", setup)
            self.assertIn("bridge", setup)
            self.assertNotIn("--gpus", setup)
            self.assertIn("--pull=never", setup)
            self.assertIn("--read-only", setup)
            self.assertIn("--user", setup)
            self.assertIn("1000:1000", setup)
            self.assertIn("--cap-drop", setup)
            self.assertIn("ALL", setup)
            self.assertIn("no-new-privileges", setup)
            self.assertIn("--pids-limit", setup)
            self.assertIn("512", setup)
            self.assertIn("--cpus", setup)
            self.assertIn("4", setup)
            self.assertIn("--memory", setup)
            self.assertIn("8g", setup)
            self.assertTrue(any(item.endswith(",readonly") for item in setup if item.startswith("type=bind")))
            self.assertTrue(any(item.startswith("type=bind") and item.endswith("dst=/work") for item in setup))
            device = helpers.build_docker_create_argv(
                image="spire-learn/task8-python:run-1", run_id="run-1", phase="device-check",
                command=("-ec", "python -B -m scripts.neural.task8_device_check"),
                mounts=mounts, network="none", gpu_uuid=helpers.GPU_UUID,
            )
            self.assertIn("none", device)
            gpu_index = device.index("--gpus")
            self.assertEqual(device[gpu_index + 1], f"device={helpers.GPU_UUID}")
            with self.assertRaisesRegex(helpers.Task8ControlError, "only the inventoried"):
                helpers.build_docker_create_argv(
                    image="spire-learn/task8-python:run-1", run_id="run-1", phase="device-check",
                    command=("-ec", "true"), mounts=mounts, network="none", gpu_uuid="all",
                )
            with self.assertRaisesRegex(helpers.Task8ControlError, "requires network=none"):
                helpers.build_docker_create_argv(
                    image="spire-learn/task8-python:run-1", run_id="run-1", phase="install",
                    command=("-ec", "true"), mounts=mounts, network="bridge",
                )

    def test_phase_specs_project_read_only_inputs_and_separate_network_install_and_gpu_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper_file = root / "task8_gpu_setup_helpers.py"
            device_script = root / "task8_device_check.py"
            local_wheel = root / helpers.CPU_WHEEL_FILENAME
            inventory = root / "gpu-inventory.json"
            for path in (helper_file, device_script, local_wheel, inventory):
                path.write_text("{}\n", encoding="utf-8")
            work = root / "work"
            resolve_output = root / "resolve"
            install_output = root / "install"
            source_root = root / "src"
            tests_root = root / "tests"
            (resolve_output / "wheelhouse").mkdir(parents=True)
            (resolve_output / "requirements-cuda.lock").write_text("torch==2.14.1+cu130\n", encoding="utf-8")
            (install_output / "venv-cuda").mkdir(parents=True)
            for path in (work, source_root, tests_root):
                path.mkdir()
            common = {
                "run_id": "run-1", "image": "spire-learn/task8-python:run-1",
                "expected_image_id": "sha256:" + "a" * 64,
                "work": work, "helper": helper_file, "local_spire_wheel": local_wheel,
                "resolve_output": resolve_output, "install_output": install_output,
                "source_root": source_root, "tests_root": tests_root,
                "device_check_script": device_script, "gpu_inventory": inventory,
            }
            resolution = helpers.build_phase_spec(phase="resolve", **common)
            resolution_script = resolution["command"][1]
            self.assertEqual(resolution["network"], "bridge")
            self.assertIsNone(resolution["gpu_uuid"])
            self.assertEqual(resolution_script.count("pip install --dry-run"), 1)
            self.assertEqual(resolution_script.count("pip download --only-binary=:all:"), 1)
            self.assertIn("capture-cuda-closure", resolution_script)
            self.assertIn("audit-cuda-closure", resolution_script)
            self.assertNotIn("pip install --no-index", resolution_script)

            install = helpers.build_phase_spec(phase="install", **common)
            install_script = install["command"][1]
            self.assertEqual(install["network"], "none")
            self.assertIsNone(install["gpu_uuid"])
            self.assertIn("--no-index", install_script)
            self.assertIn("--require-hashes", install_script)
            self.assertIn("pip check", install_script)
            self.assertFalse(any(not row["readonly"] and row["destination"] != "/work"
                                 for row in install["mounts"]))

            device = helpers.build_phase_spec(phase="device-check", **common)
            self.assertEqual(device["network"], "none")
            self.assertEqual(device["gpu_uuid"], helpers.GPU_UUID)
            device_argv = helpers.build_docker_create_argv(
                image=device["image"], run_id=device["run_id"], phase=device["phase"],
                command=device["command"],
                mounts=tuple((Path(row["source"]), row["destination"], row["readonly"])
                             for row in device["mounts"]),
                network=device["network"], gpu_uuid=device["gpu_uuid"],
            )
            self.assertIn("--gpus", device_argv)
            self.assertIn("--network", device_argv)
            self.assertIn("none", device_argv)
            self.assertIn("--read-only", device_argv)

            corrected = helpers.build_phase_spec(
                phase="rng-red", attempt_id="corrected-1",
                **{**common, "run_id": helpers.CORRECTED_RNG_RUN_ID},
            )
            self.assertEqual(corrected["phase_key"], "rng-red-corrected-1")
            self.assertEqual(
                corrected["container_name"],
                "spirelearn-task8-rng-red-corrected-1-cuda-setup-20261028-a",
            )
            self.assertEqual(corrected["attempt_id"], "corrected-1")

    def test_exact_resolver_inputs_and_phase_release_are_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirement_path = root / "requirements-cuda.in"
            helpers.write_cuda_resolution_input(requirement_path)
            requirements = requirement_path.read_text(encoding="utf-8")
            self.assertIn(f"torch @ {helpers.CUDA_TORCH_URL}#sha256={helpers.CUDA_TORCH_SHA256}", requirements)
            self.assertIn("numpy==2.5.4", requirements)
            self.assertIn("gymnasium==1.4.0", requirements)
            self.assertIn(f"spire_lab @ file:///inputs/{helpers.CPU_WHEEL_FILENAME}", requirements)
            with self.assertRaisesRegex(helpers.Task8ControlError, "already exists"):
                helpers.write_cuda_resolution_input(requirement_path)
            command_path = root / "resolver-command.json"
            helpers.write_resolver_command(command_path)
            command = json.loads(command_path.read_text(encoding="utf-8"))["argv"]
            self.assertEqual(tuple(command), helpers.CUDA_RESOLVER_ARGV)
            self.assertIn("--dry-run", command)
            self.assertIn("--only-binary=:all:", command)
            self.assertNotIn("--extra-index-url", command)
            release_path = root / "release.json"
            release_path.write_text(json.dumps({
                "decision": "internal_task8_phase_release", "phase": "install",
                "preparation_review": "passed", "independent_review": "passed",
                "source_snapshot_sha256": "a" * 64, "single_use": True,
                "release_id": "release-1", "closure_lock_sha256": "b" * 64,
            }), encoding="utf-8")
            result = helpers.validate_phase_release(
                release_path, phase="install", source_hash="a" * 64,
                required_fields={"closure_lock_sha256": "b" * 64},
            )
            self.assertEqual(result["release_id"], "release-1")
            with self.assertRaisesRegex(helpers.Task8ControlError, "field"):
                helpers.validate_phase_release(
                    release_path, phase="install", source_hash="a" * 64,
                    required_fields={"closure_lock_sha256": "c" * 64},
                )

    def test_corrected_rng_attempt_projection_is_single_run_and_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task-8"
            root.mkdir()
            paths = helpers.project_corrected_rng_attempt_paths(
                root, run_id="cuda-setup-20261028-a", attempt_id="corrected-1",
            )
            self.assertEqual(paths["phase_key"], "rng-red-corrected-1")
            self.assertEqual(
                paths["container_name"],
                "spirelearn-task8-rng-red-corrected-1-cuda-setup-20261028-a",
            )
            self.assertEqual(
                paths["previous_failed_inventory_path"],
                root / "runtime-admin/cuda-setup-20261028-a/rng-red/evidence-inventory.json",
            )
            self.assertEqual(
                paths["corrected_transcript_path"],
                root / "runtime-admin/cuda-setup-20261028-a/rng-red-corrected-1/logs/rng-red-transcript.stdout.log",
            )
            self.assertEqual(
                paths["corrected_inventory_path"],
                root / "runtime-admin/cuda-setup-20261028-a/rng-red-corrected-1/evidence-inventory.json",
            )
            for run_id, attempt_id in (("other-run", "corrected-1"),
                                       ("cuda-setup-20261028-a", "corrected-2")):
                with self.subTest(run_id=run_id, attempt_id=attempt_id), \
                        self.assertRaises(helpers.Task8ControlError):
                    helpers.project_corrected_rng_attempt_paths(
                        root, run_id=run_id, attempt_id=attempt_id,
                    )
            alias = Path(directory) / "task-8-alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(helpers.Task8ControlError, "canonical"):
                helpers.project_corrected_rng_attempt_paths(
                    alias, run_id="cuda-setup-20261028-a", attempt_id="corrected-1",
                )

    def test_corrected_rng_release_and_evidence_bind_fresh_red_and_immutable_failure(self):
        run_id = "cuda-setup-20261028-a"
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory) / "task-8"
            task_root.mkdir()
            source_snapshot = task_root / helpers.CORRECTED_RNG_SOURCE_SNAPSHOT_NAME
            source_snapshot.write_text("{}\n", encoding="utf-8")
            lock_path = task_root / "runtime-output" / run_id / "resolve-work/requirements-cuda.lock"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text("torch==2.14.1+cu130\n", encoding="utf-8")
            source_hash, lock_hash = helpers.sha256_file(source_snapshot), helpers.sha256_file(lock_path)
            paths = helpers.project_corrected_rng_attempt_paths(
                task_root, run_id=run_id, attempt_id="corrected-1",
            )
            Path(paths["original_started_marker"]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths["original_started_marker"]).write_text("\n", encoding="utf-8")
            for key in ("previous_failed_output_root", "previous_failed_control_root",
                        "previous_failed_admin_root"):
                paths[key].mkdir(parents=True)
            old_admin = paths["previous_failed_admin_root"]
            (old_admin / "logs").mkdir()
            (paths["previous_failed_control_root"] / "cleanup-status.json").write_text(
                json.dumps({"phase": "rng-red", "run_id": run_id,
                            "container_create_attempted": 1, "container_start_attempted": 1,
                            "container_started": 0, "cleanup_failed": 0, "status": 2}) + "\n",
                encoding="utf-8",
            )
            (old_admin / "logs/container-start.stdout.log").write_text(
                "TypeError: ModelConfig.__init__() missing 1 required positional argument: 'max_parameters'\n"
                "Ran 3 tests in 2.075s\nFAILED (failures=1, errors=2)\n",
                encoding="utf-8",
            )
            old_inventory_path = paths["previous_failed_inventory_path"]
            old_inventory = helpers.inventory_roots(
                (paths["previous_failed_output_root"], paths["previous_failed_control_root"], old_admin),
                old_inventory_path,
            )
            old_hash = old_inventory["inventory_sha256"]

            for key in ("corrected_output_root", "corrected_control_root", "corrected_admin_root"):
                paths[key].mkdir(parents=True)
            corrected_admin = paths["corrected_admin_root"]
            (corrected_admin / "logs").mkdir()
            (paths["corrected_control_root"] / "cleanup-status.json").write_text(
                json.dumps({"phase": "rng-red", "run_id": run_id, "attempt_id": "corrected-1",
                            "container_create_attempted": 1, "container_start_attempted": 1,
                            "container_start_invocation_returned": 1, "container_started": 1,
                            "cleanup_failed": 0, "status": 0}) + "\n",
                encoding="utf-8",
            )
            transcript = {
                "expected_labels": list(helpers.EXPECTED_RNG_LABELS),
                "observed_labels": list(helpers.EXPECTED_RNG_LABELS),
                "ran": 3, "failures": list(helpers.EXPECTED_RNG_LABELS[:2]),
                "skipped": 0, "status": "expected_red",
            }
            corrected_transcript = paths["corrected_transcript_path"]
            corrected_transcript.write_text(json.dumps(transcript) + "\n", encoding="utf-8")
            (paths["corrected_control_root"] / "phase-input-identities.json").write_text(
                json.dumps({
                    "format": "spire-learn-task8-phase-input-identities-v1",
                    "prior_inventories": [{
                        "phase": "rng-red",
                        "inventory_path": str(old_inventory_path),
                        "inventory_sha256": old_hash,
                    }],
                    "immutable_paths": [
                        {"name": "source_snapshot", "path": str(source_snapshot), "sha256": source_hash},
                        {"name": "closure_lock", "path": str(lock_path), "sha256": lock_hash},
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            corrected_inventory = helpers.inventory_roots(
                (paths["corrected_output_root"], paths["corrected_control_root"], corrected_admin),
                paths["corrected_inventory_path"],
            )
            release_path = Path(directory) / "release.json"
            release_path.write_text(json.dumps({
                "decision": "internal_task8_phase_release", "phase": "rng-red",
                "preparation_review": "passed", "independent_review": "passed",
                "source_snapshot_sha256": source_hash,
                "corrected_source_snapshot_sha256": source_hash,
                "single_use": True, "release_id": "rng-red-correction-1",
                "run_id": run_id, "attempt_id": "corrected-1",
                "closure_lock_sha256": lock_hash,
                "previous_failed_attempt_inventory_path": str(old_inventory_path),
                "previous_failed_attempt_inventory_sha256": old_hash,
            }), encoding="utf-8")

            release = helpers.validate_corrected_rng_red_release(
                release_path, task_root=task_root, run_id=run_id,
                source_hash=source_hash, closure_lock_hash=lock_hash,
            )
            self.assertEqual(release["attempt_id"], "corrected-1")
            red = helpers.verify_corrected_rng_red_evidence(
                task_root, run_id=run_id,
                transcript_path=corrected_transcript,
                transcript_sha256=helpers.sha256_file(corrected_transcript),
                inventory_path=paths["corrected_inventory_path"],
                inventory_sha256=corrected_inventory["inventory_sha256"],
                previous_failed_inventory_path=old_inventory_path,
                previous_failed_inventory_sha256=old_hash,
            )
            self.assertEqual(red["attempt_id"], "corrected-1")
            self.assertEqual(red["status"], "expected_red")
            self.assertEqual(red["previous_failed_attempt_inventory_sha256"], old_hash)
            transcript_copy = Path(directory) / "red-transcript-copy.json"
            transcript_copy.write_bytes(corrected_transcript.read_bytes())
            corrected_transcript.unlink()
            corrected_transcript.symlink_to(transcript_copy)
            with self.assertRaisesRegex(helpers.Task8ControlError, "exact canonical"):
                helpers.verify_corrected_rng_red_evidence(
                    task_root, run_id=run_id,
                    transcript_path=corrected_transcript,
                    transcript_sha256=helpers.sha256_file(transcript_copy),
                    inventory_path=paths["corrected_inventory_path"],
                    inventory_sha256=corrected_inventory["inventory_sha256"],
                    previous_failed_inventory_path=old_inventory_path,
                    previous_failed_inventory_sha256=old_hash,
                )
            corrected_transcript.unlink()
            corrected_transcript.write_bytes(transcript_copy.read_bytes())

            followup_path = Path(directory) / "green-release.json"
            followup_source_hash = "c" * 64
            for phase in ("rng-green", "device-check"):
                fields = {
                    "decision": "internal_task8_phase_release", "phase": phase,
                    "preparation_review": "passed", "independent_review": "passed",
                    "source_snapshot_sha256": followup_source_hash,
                    "single_use": True, "release_id": f"{phase}-release-1",
                    "run_id": run_id,
                    "closure_lock_sha256": lock_hash,
                    "rng_red_attempt_id": red["attempt_id"],
                    "rng_red_transcript_path": red["transcript_path"],
                    "rng_red_transcript_sha256": red["transcript_sha256"],
                    "rng_red_inventory_path": red["inventory_path"],
                    "rng_red_inventory_sha256": red["inventory_sha256"],
                    "rng_red_source_snapshot_sha256": red["source_snapshot_sha256"],
                    "rng_red_closure_lock_sha256": red["closure_lock_sha256"],
                    "previous_failed_attempt_inventory_path": red["previous_failed_attempt_inventory_path"],
                    "previous_failed_attempt_inventory_sha256": red["previous_failed_attempt_inventory_sha256"],
                }
                green_hash = "d" * 64 if phase == "device-check" else None
                if phase == "device-check":
                    fields.update({"rng_regressions": "passed", "rng_green_transcript_sha256": green_hash})
                followup_path.write_text(json.dumps(fields), encoding="utf-8")
                accepted = helpers.validate_corrected_rng_followup_release(
                    followup_path, task_root=task_root, phase=phase, source_hash=followup_source_hash,
                    closure_lock_hash=lock_hash, corrected_red_evidence=red,
                    rng_green_transcript_sha256=green_hash,
                )
                self.assertEqual(accepted["phase"], phase)
                if phase == "rng-green":
                    fields["rng_red_transcript_path"] = str(paths["previous_failed_admin_root"] / "logs/old.log")
                    followup_path.write_text(json.dumps(fields), encoding="utf-8")
                    with self.assertRaisesRegex(helpers.Task8ControlError, "field"):
                        helpers.validate_corrected_rng_followup_release(
                            followup_path, task_root=task_root, phase=phase, source_hash=followup_source_hash,
                            closure_lock_hash=lock_hash, corrected_red_evidence=red,
                        )

            with self.assertRaisesRegex(helpers.Task8ControlError, "exact canonical"):
                helpers.verify_corrected_rng_red_evidence(
                    task_root, run_id=run_id,
                    transcript_path=paths["previous_failed_admin_root"] / "logs/rng-red-transcript.stdout.log",
                    transcript_sha256="e" * 64,
                    inventory_path=paths["corrected_inventory_path"],
                    inventory_sha256=corrected_inventory["inventory_sha256"],
                    previous_failed_inventory_path=old_inventory_path,
                    previous_failed_inventory_sha256=old_hash,
                )
            release_value = json.loads(release_path.read_text(encoding="utf-8"))
            release_value["previous_failed_attempt_inventory_sha256"] = "f" * 64
            release_path.write_text(json.dumps(release_value), encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "field"):
                helpers.validate_corrected_rng_red_release(
                    release_path, task_root=task_root, run_id=run_id,
                    source_hash=source_hash, closure_lock_hash=lock_hash,
                )
            old_start_output = paths["previous_failed_start_stdout_path"]
            old_start_output.write_text(old_start_output.read_text(encoding="utf-8") + "changed\n",
                                        encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "inventory mismatch"):
                helpers.verify_corrected_rng_red_evidence(
                    task_root, run_id=run_id,
                    transcript_path=corrected_transcript,
                    transcript_sha256=helpers.sha256_file(corrected_transcript),
                    inventory_path=paths["corrected_inventory_path"],
                    inventory_sha256=corrected_inventory["inventory_sha256"],
                    previous_failed_inventory_path=old_inventory_path,
                    previous_failed_inventory_sha256=old_hash,
                )

    def test_exact_container_name_lookup_rejects_a_matching_preexisting_container(self):
        container_id = "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            work = root / "work"
            helper.write_text("# fixture\n", encoding="utf-8")
            work.mkdir()
            inspect, inspect_args = _matching_corrected_container_inspect(helper, work)
            self.assertEqual(
                helpers.assert_container_inspect(json.dumps(inspect), **inspect_args)["container_id"],
                container_id,
            )
            collision_lookup = f"{container_id}\n".encode("ascii")
            with self.assertRaisesRegex(helpers.Task8ControlError, "must be absent"):
                helpers.assert_exact_container_name_absent(0, collision_lookup)

            removal_claim = root / "container-removal-authorization.json"
            with self.assertRaisesRegex(helpers.Task8ControlError, "create exit status"):
                helpers.authorize_container_removal(
                    container_id, create_exit_status=125, create_stdout=b"",
                    cleanup_assert_exit_status=0, claim_path=removal_claim,
                )
            self.assertFalse(removal_claim.exists())

    def test_container_name_lookup_rejects_malformed_output_and_daemon_failures(self):
        helpers.assert_exact_container_name_absent(0, b"")
        for raw in (b"\n", b"not-an-id\n", ("a" * 64 + "\n" + "b" * 64 + "\n").encode()):
            with self.subTest(raw=raw), self.assertRaisesRegex(helpers.Task8ControlError, "must be absent"):
                helpers.assert_exact_container_name_absent(0, raw)
        for status in (1, 124, 137):
            with self.subTest(status=status), self.assertRaisesRegex(helpers.Task8ControlError, "lookup failed"):
                helpers.assert_exact_container_name_absent(status, b"")

    def test_container_creation_establishes_ownership_only_for_one_exact_id(self):
        container_id = "d" * 64
        self.assertEqual(helpers.parse_created_container_id(0, f"{container_id}\n".encode()), container_id)
        for status in (1, 124, 125, 137):
            with self.subTest(status=status), self.assertRaisesRegex(
                helpers.Task8ControlError, "create exit status",
            ):
                helpers.parse_created_container_id(status, f"{container_id}\n".encode())
        for raw in (
            b"", b"malformed\n", ("d" * 64 + "\n" + "e" * 64 + "\n").encode(),
            f"{container_id}\n\n".encode(), f"{container_id} extra\n".encode(),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(helpers.Task8ControlError, "exactly one"):
                helpers.parse_created_container_id(0, raw)

    def test_container_removal_claim_is_bound_to_created_id_and_spec_check_once(self):
        container_id = "e" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_stdout = f"{container_id}\n".encode()
            claim_path = root / "container-removal-authorization.json"
            with self.assertRaisesRegex(helpers.Task8ControlError, "does not match"):
                helpers.authorize_container_removal(
                    "f" * 64, create_exit_status=0, create_stdout=create_stdout,
                    cleanup_assert_exit_status=0, claim_path=claim_path,
                )
            with self.assertRaisesRegex(helpers.Task8ControlError, "spec assertion"):
                helpers.authorize_container_removal(
                    container_id, create_exit_status=0, create_stdout=create_stdout,
                    cleanup_assert_exit_status=1, claim_path=claim_path,
                )
            authorization = helpers.authorize_container_removal(
                container_id, create_exit_status=0, create_stdout=create_stdout,
                cleanup_assert_exit_status=0, claim_path=claim_path,
            )
            self.assertEqual(authorization["container_id"], container_id)
            self.assertEqual(json.loads(claim_path.read_text(encoding="utf-8"))["container_id"], container_id)
            with self.assertRaisesRegex(helpers.Task8ControlError, "already authorized once"):
                helpers.authorize_container_removal(
                    container_id, create_exit_status=0, create_stdout=create_stdout,
                    cleanup_assert_exit_status=0, claim_path=claim_path,
                )

    def test_container_ownership_cli_consumes_recorded_fake_docker_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lookup_status = root / "name-lookup.exit.log"
            lookup_stdout = root / "name-lookup.stdout.log"
            create_status = root / "container-create.exit.log"
            create_stdout = root / "container-create.stdout.log"
            inspect_status = root / "cleanup-assert.exit.log"
            claim = root / "container-removal-authorization.json"

            def invoke(argv):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    status = helpers.main(argv)
                return status, stdout.getvalue(), stderr.getvalue()

            lookup_status.write_text("0\n", encoding="ascii")
            lookup_stdout.write_bytes(b"")
            status, output, _error = invoke([
                "assert-container-name-absent", str(lookup_status), str(lookup_stdout),
            ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output)["name_absent"], True)

            lookup_stdout.write_bytes(("a" * 64 + "\n").encode("ascii"))
            status, _output, error = invoke([
                "assert-container-name-absent", str(lookup_status), str(lookup_stdout),
            ])
            self.assertEqual(status, 2)
            self.assertIn("must be absent", error)

            container_id = "b" * 64
            create_status.write_text("0\n", encoding="ascii")
            create_stdout.write_bytes(f"{container_id}\n".encode("ascii"))
            status, output, _error = invoke([
                "assert-container-create", str(create_status), str(create_stdout),
            ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output)["container_id"], container_id)

            create_status.write_text("124\n", encoding="ascii")
            status, _output, error = invoke([
                "assert-container-create", str(create_status), str(create_stdout),
            ])
            self.assertEqual(status, 2)
            self.assertIn("did not establish container ownership", error)
            create_status.write_text("0\n", encoding="ascii")
            inspect_status.write_text("0\n", encoding="ascii")
            args = [
                "authorize-container-removal", container_id, str(create_status), str(create_stdout),
                str(inspect_status), str(claim),
            ]
            status, output, _error = invoke(args)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output)["container_id"], container_id)
            authorized_bytes = claim.read_bytes()
            status, _output, error = invoke(args)
            self.assertEqual(status, 2)
            self.assertIn("already authorized once", error)
            self.assertEqual(claim.read_bytes(), authorized_bytes)

    def test_corrected_rng_docker_identity_is_unique_and_attempt_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            helper = root / "helper.py"
            work.mkdir()
            helper.write_text("# helper\n", encoding="utf-8")
            argv = helpers.build_docker_create_argv(
                image="spire-learn/task8-python:cuda-setup-20261028-a",
                run_id="cuda-setup-20261028-a", phase="rng-red", attempt_id="corrected-1",
                command=("-ec", "true"),
                mounts=((helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
                        (work, "/work", False)),
                network="none", gpu_uuid=helpers.GPU_UUID,
            )
            self.assertEqual(
                argv[argv.index("--name") + 1],
                "spirelearn-task8-rng-red-corrected-1-cuda-setup-20261028-a",
            )
            self.assertIn("spire-learn.attempt-id=corrected-1", argv)
            inspect, inspect_args = _matching_corrected_container_inspect(helper, work)
            self.assertEqual(
                helpers.assert_container_inspect(json.dumps(inspect), **inspect_args)["phase_key"],
                "rng-red-corrected-1",
            )
            wrong_name = json.loads(json.dumps(inspect))
            wrong_name[0]["Name"] = "/spirelearn-task8-rng-red-cuda-setup-20261028-a"
            with self.assertRaisesRegex(helpers.Task8ControlError, "container name"):
                helpers.assert_container_inspect(json.dumps(wrong_name), **inspect_args)
            with self.assertRaisesRegex(helpers.Task8ControlError, "only the approved corrected"):
                helpers.build_docker_create_argv(
                    image="spire-learn/task8-python:cuda-setup-20261028-a",
                    run_id="cuda-setup-20261028-a", phase="rng-green", attempt_id="corrected-1",
                    command=("-ec", "true"),
                    mounts=((helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
                            (work, "/work", False)),
                    network="none", gpu_uuid=helpers.GPU_UUID,
                )

    def test_container_inspect_rejects_extra_writable_or_unexpected_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            helper = root / "helper.py"
            work.mkdir()
            helper.write_text("# helper\n", encoding="utf-8")
            expected_mounts = ((helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
                               (work, "/work", False))
            command = ["-ec", "python -B -m pip download --only-binary=:all:"]
            inspect = [{
                "Id": "c" * 64,
                "Image": "sha256:" + "a" * 64,
                "Config": {
                    "User": "1000:1000", "WorkingDir": "/learner", "Entrypoint": ["/bin/sh"],
                    "Cmd": command,
                    "Labels": {
                        "spire-learn.task": "task8-gpu-setup",
                        "spire-learn.phase": "resolve",
                        "spire-learn.run-id": "run-1",
                    },
                    "Env": [
                        "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
                        "PIP_CONFIG_FILE=/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                        "PIP_NO_INPUT=1", "HOME=/tmp", "TMPDIR=/work/tmp",
                        "PIP_CACHE_DIR=/work/pip-cache",
                    ],
                },
                "HostConfig": {
                    "NetworkMode": "bridge", "ReadonlyRootfs": True, "Privileged": False,
                    "AutoRemove": False, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                    "NanoCpus": 4_000_000_000, "Memory": 8 * 1024**3, "PidsLimit": 512,
                    "Devices": [], "DeviceRequests": [],
                    "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
                },
                "Mounts": [
                    {"Type": "bind", "Source": str(helper.resolve()),
                     "Destination": "/learner/scripts/neural/task8_gpu_setup_helpers.py", "RW": False},
                    {"Type": "bind", "Source": str(work.resolve()), "Destination": "/work", "RW": True},
                ],
            }]
            args = {
                "expected_image_id": "sha256:" + "a" * 64,
                "run_id": "run-1", "phase": "resolve", "expected_command": command,
                "expected_mounts": expected_mounts, "network": "bridge",
            }
            result = helpers.assert_container_inspect(json.dumps(inspect), **args)
            self.assertEqual(result["mount_count"], 2)
            altered = json.loads(json.dumps(inspect))
            altered[0]["Mounts"].append({
                "Type": "bind", "Source": str(root), "Destination": "/root", "RW": True,
            })
            with self.assertRaisesRegex(helpers.Task8ControlError, "mount table"):
                helpers.assert_container_inspect(json.dumps(altered), **args)

    def test_container_inspect_normalizes_null_absence_and_rejects_added_privilege_or_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            helper = root / "helper.py"
            work.mkdir()
            helper.write_text("# helper\n", encoding="utf-8")
            command = ["-ec", "true"]
            config = {
                "User": "1000:1000", "WorkingDir": "/learner", "Entrypoint": ["/bin/sh"],
                "Cmd": command,
                "Labels": {
                    "spire-learn.task": "task8-gpu-setup", "spire-learn.phase": "resolve",
                    "spire-learn.run-id": "run-null",
                },
                "Env": [
                    "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
                    "PIP_CONFIG_FILE=/dev/null", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                    "PIP_NO_INPUT=1", "HOME=/tmp", "TMPDIR=/work/tmp",
                    "PIP_CACHE_DIR=/work/pip-cache",
                ],
            }
            host = {
                "NetworkMode": "bridge", "ReadonlyRootfs": True, "Privileged": False,
                "AutoRemove": False, "CapDrop": ["ALL"], "CapAdd": None,
                "GroupAdd": None, "SecurityOpt": ["no-new-privileges"],
                "NanoCpus": 4_000_000_000, "Memory": 8 * 1024**3, "PidsLimit": 512,
                # Retained inspect evidence uses empty Devices plus null requests.
                "Devices": [], "DeviceRequests": None, "DeviceCgroupRules": [],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
            }
            inspect = [{
                "Id": "c" * 64, "Image": "sha256:" + "a" * 64,
                "Config": config, "HostConfig": host,
                "Mounts": [
                    {"Type": "bind", "Source": str(helper.resolve()),
                     "Destination": "/learner/scripts/neural/task8_gpu_setup_helpers.py", "RW": False},
                    {"Type": "bind", "Source": str(work.resolve()), "Destination": "/work", "RW": True},
                ],
            }]
            expected_mounts = ((helper, "/learner/scripts/neural/task8_gpu_setup_helpers.py", True),
                               (work, "/work", False))
            args = {
                "expected_image_id": "sha256:" + "a" * 64, "run_id": "run-null",
                "phase": "resolve", "expected_command": command,
                "expected_mounts": expected_mounts, "network": "bridge",
            }
            self.assertEqual(helpers.assert_container_inspect(json.dumps(inspect), **args)["phase"], "resolve")
            null_absence = json.loads(json.dumps(inspect))
            null_absence[0]["HostConfig"]["Devices"] = None
            null_absence[0]["HostConfig"]["DeviceCgroupRules"] = None
            self.assertEqual(
                helpers.assert_container_inspect(json.dumps(null_absence), **args)["phase"], "resolve",
            )
            for field, value in (
                ("GroupAdd", ["video"]), ("CapAdd", ["SYS_ADMIN"]),
                ("Devices", [{"PathOnHost": "/dev/nvidia0"}]),
                ("DeviceCgroupRules", ["c 195:* rwm"]),
                ("DeviceCgroupRules", ""), ("DeviceCgroupRules", 0),
                ("DeviceRequests", [{"DeviceIDs": [helpers.GPU_UUID], "Capabilities": [["gpu"]]}]),
                ("Privileged", True),
            ):
                with self.subTest(field=field), self.assertRaises(helpers.Task8ControlError):
                    altered = json.loads(json.dumps(inspect))
                    altered[0]["HostConfig"][field] = value
                    helpers.assert_container_inspect(json.dumps(altered), **args)
            gpu_inspect = json.loads(json.dumps(inspect))
            gpu_inspect[0]["Config"]["Labels"]["spire-learn.phase"] = "rng-red"
            gpu_inspect[0]["HostConfig"]["NetworkMode"] = "none"
            gpu_inspect[0]["HostConfig"]["DeviceRequests"] = [{
                "DeviceIDs": [helpers.GPU_UUID], "Capabilities": [["gpu"]],
            }]
            gpu_args = {**args, "phase": "rng-red", "network": "none", "gpu_uuid": helpers.GPU_UUID}
            self.assertEqual(
                helpers.assert_container_inspect(json.dumps(gpu_inspect), **gpu_args)["gpu_uuid"],
                helpers.GPU_UUID,
            )
            gpu_inspect[0]["HostConfig"]["DeviceCgroupRules"] = None
            self.assertEqual(
                helpers.assert_container_inspect(json.dumps(gpu_inspect), **gpu_args)["gpu_uuid"],
                helpers.GPU_UUID,
            )
            gpu_inspect[0]["HostConfig"]["DeviceCgroupRules"] = ["c 195:* rwm"]
            with self.assertRaisesRegex(helpers.Task8ControlError, "device cgroup"):
                helpers.assert_container_inspect(json.dumps(gpu_inspect), **gpu_args)
            gpu_inspect[0]["HostConfig"]["DeviceCgroupRules"] = []
            gpu_inspect[0]["HostConfig"]["GroupAdd"] = ["video"]
            with self.assertRaisesRegex(helpers.Task8ControlError, "supplemental groups"):
                helpers.assert_container_inspect(json.dumps(gpu_inspect), **gpu_args)

    def test_phase_spec_uses_exact_tagged_local_wheel_and_owned_pip_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            device_script = root / "device.py"
            local_wheel = root / helpers.CPU_WHEEL_FILENAME
            inventory = root / "gpu.json"
            for path in (helper, device_script, local_wheel, inventory):
                path.write_text("fixture\n", encoding="utf-8")
            work, resolve_output, install_output = root / "work", root / "resolve", root / "install"
            source_root, tests_root = root / "src", root / "tests"
            (resolve_output / "wheelhouse").mkdir(parents=True)
            (resolve_output / "requirements-cuda.lock").write_text("fixture\n", encoding="utf-8")
            (install_output / "venv-cuda").mkdir(parents=True)
            for path in (work, source_root, tests_root):
                path.mkdir()
            spec = helpers.build_phase_spec(
                phase="resolve", run_id="run-tags", image="spire-learn/task8-python:run-tags",
                expected_image_id="sha256:" + "a" * 64, work=work, helper=helper,
                local_spire_wheel=local_wheel, resolve_output=resolve_output,
                install_output=install_output, source_root=source_root, tests_root=tests_root,
                device_check_script=device_script, gpu_inventory=inventory,
            )
            wheel_mount = next(row for row in spec["mounts"] if row["destination"].startswith("/inputs/"))
            self.assertEqual(wheel_mount["destination"], f"/inputs/{helpers.CPU_WHEEL_FILENAME}")
            self.assertIn("TMPDIR=/work/tmp", spec["command"][1])
            self.assertIn("PIP_CACHE_DIR=/work/pip-cache", spec["command"][1])
            install = helpers.build_phase_spec(
                phase="install", run_id="run-tags", image="spire-learn/task8-python:run-tags",
                expected_image_id="sha256:" + "a" * 64, work=work, helper=helper,
                local_spire_wheel=local_wheel, resolve_output=resolve_output,
                install_output=install_output, source_root=source_root, tests_root=tests_root,
                device_check_script=device_script, gpu_inventory=inventory,
            )
            self.assertIn("TMPDIR=/work/tmp", install["command"][1])
            self.assertIn("PIP_CACHE_DIR=/work/pip-cache", install["command"][1])
            self.assertIn(
                f"/inputs/{helpers.CPU_WHEEL_FILENAME}",
                [row["destination"] for row in install["mounts"]],
            )

    def test_target_wheel_tags_use_explicit_supported_cpython312_tags(self):
        compatible = {
            "cp312-cp312-manylinux_2_28_x86_64",
            "py3-none-manylinux_2_17_x86_64",
            "cp39-abi3-manylinux_2_17_x86_64",
        }
        for filename in (
            "sample-1.0-py3-none-manylinux_2_17_x86_64.whl",
            "sample-1.0-cp39-abi3-manylinux_2_17_x86_64.whl",
            "sample-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        ):
            with self.subTest(filename=filename):
                self.assertTrue(helpers._assert_target_wheel_tag(filename, compatible_tags=compatible))
        for filename in (
            "sample-1.0-cp311-cp311-manylinux_2_17_x86_64.whl",
            "sample-1.0-py3-none-win_amd64.whl",
        ):
            with self.subTest(filename=filename), self.assertRaises(helpers.Task8ControlError):
                helpers._assert_target_wheel_tag(filename, compatible_tags=compatible)

    def test_gpu_inventory_requires_exact_uuid_identity_and_memory_headroom(self):
        line = f"{helpers.GPU_UUID}, {helpers.GPU_NAME}, {helpers.GPU_DRIVER}, {helpers.GPU_MEMORY_MIB}, 14000\n"
        result = helpers.verify_gpu_inventory(line)
        self.assertEqual(result["memory_free_mib"], 14000)
        with self.assertRaisesRegex(helpers.Task8ControlError, "less than 12 GiB"):
            helpers.verify_gpu_inventory(line.replace(", 14000", ", 11999"))
        with self.assertRaisesRegex(helpers.Task8ControlError, "exactly one approved"):
            helpers.verify_gpu_inventory(line.replace(helpers.GPU_UUID, "GPU-other"))
        with self.assertRaises(helpers.Task8ControlError):
            helpers.verify_gpu_inventory(line, minimum_free_mib=1)

    def test_storage_timeout_and_cumulative_reservations_fail_closed(self):
        gib = 1024**3
        self.assertEqual(
            helpers.verify_storage_budget(
                new_gpu_bytes=24 * gib, neural_cumulative_bytes=40 * gib,
                free_bytes=60 * gib, setup=True,
            )["free_floor_bytes"], 60 * gib,
        )
        for kwargs in (
            {"new_gpu_bytes": 24 * gib + 1, "neural_cumulative_bytes": 1,
             "free_bytes": 100 * gib, "setup": True},
            {"new_gpu_bytes": 1, "neural_cumulative_bytes": 40 * gib + 1,
             "free_bytes": 100 * gib, "setup": True},
            {"new_gpu_bytes": 1, "neural_cumulative_bytes": 1,
             "free_bytes": 59 * gib, "setup": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(helpers.Task8ControlError):
                helpers.verify_storage_budget(**kwargs)
        self.assertEqual(helpers.reserve_cumulative(1024, 512, cap=2048, name="slot"), 1536)
        self.assertEqual(helpers.reserve_cumulative(2, 1, cap=3, name="optimizer"), 3)
        with self.assertRaisesRegex(helpers.Task8ControlError, "exceed"):
            helpers.reserve_cumulative(1792, 257, cap=2048, name="slot")
        with self.assertRaisesRegex(helpers.Task8ControlError, "exceed"):
            helpers.reserve_cumulative(3, 1, cap=3, name="optimizer")
        self.assertEqual(
            helpers.verify_timeout_bounds(elapsed_seconds=59, phase="device-check", cleanup_completed=True)["soft_overshoot_seconds"],
            0.0,
        )
        with self.assertRaisesRegex(helpers.Task8ControlError, "cleanup"):
            helpers.verify_timeout_bounds(elapsed_seconds=31, phase="device-check", cleanup_completed=False)

    def test_setup_budget_carries_consumed_prefix_and_excludes_parent_pause(self):
        result = helpers.verify_timeout_bounds(
            elapsed_seconds=285, phase="install", cleanup_completed=True,
            active_seconds_before=300, cleanup_reserve_seconds=15,
            external_parent_pause_seconds=1800,
        )
        self.assertEqual(result["active_seconds_before"], 300)
        self.assertEqual(result["active_seconds_charged"], 600)
        self.assertEqual(result["external_parent_pause_seconds"], 1800)
        self.assertEqual(result["soft_overshoot_seconds"], 0)
        overshoot = helpers.verify_timeout_bounds(
            elapsed_seconds=286, phase="install", cleanup_completed=True,
            active_seconds_before=300, cleanup_reserve_seconds=15,
        )
        self.assertEqual(overshoot["soft_overshoot_seconds"], 1)
        with self.assertRaisesRegex(helpers.Task8ControlError, "whole-run"):
            helpers.verify_timeout_bounds(
                elapsed_seconds=1, phase="install", cleanup_completed=True,
                active_seconds_before=1190, cleanup_reserve_seconds=15,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolve-budget.json"
            with patch.object(helpers.time, "time", return_value=1270.0):
                helpers.write_phase_budget_state(
                    path, phase="resolve", started_epoch_seconds=1000.0,
                    active_seconds_before=300, external_parent_pause_seconds=1800,
                    cleanup_completed=True, cleanup_reserve_seconds=15,
                )
            remaining = helpers.read_setup_budget_state(path)
            self.assertEqual(remaining["active_seconds_before"], 585)
            self.assertEqual(remaining["remaining_soft_seconds"], 15)
            self.assertEqual(remaining["remaining_hard_seconds"], 615)
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(state["external_parent_pause_seconds"], 1800)
            self.assertEqual(state["active_seconds_charged"], 585)

    def test_storage_growth_is_conservative_and_reserves_before_allocation(self):
        gib = 1024**3
        admitted = helpers.verify_storage_budget(
            new_gpu_bytes=8 * gib, neural_cumulative_bytes=20 * gib,
            free_bytes=60 * gib, setup=True, reservation_bytes=16 * gib,
        )
        self.assertEqual(admitted["new_gpu_projected_bytes"], 24 * gib)
        self.assertEqual(admitted["neural_projected_bytes"], 36 * gib)
        for kwargs in (
            {"new_gpu_bytes": 8 * gib, "neural_cumulative_bytes": 20 * gib,
             "free_bytes": 60 * gib, "setup": True, "reservation_bytes": 17 * gib},
            {"new_gpu_bytes": 8 * gib, "neural_cumulative_bytes": 25 * gib,
             "free_bytes": 60 * gib, "setup": True, "reservation_bytes": 16 * gib},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(helpers.Task8ControlError):
                helpers.verify_storage_budget(**kwargs)

    def test_storage_counts_full_neural_tree_and_only_known_venv_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neural = root / "neural-combat"
            neural.mkdir()
            previous = neural / "accepted-existing-report.json"
            previous.write_bytes(b"prior evidence")
            runtime_roots = [neural / name for name in ("runtime", "runtime-output")]
            for path in runtime_roots:
                path.mkdir()
            venv = runtime_roots[1] / "venv-cuda"
            (venv / "lib").mkdir(parents=True)
            (venv / "lib" / "module.py").write_bytes(b"venv bytes")
            os.symlink("lib", venv / "lib64")
            (runtime_roots[0] / "image-identity.json").write_text("{}\n", encoding="utf-8")
            with patch.object(helpers.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                measured = helpers.check_storage_paths(
                    root, neural, runtime_roots, setup=True,
                    reservation_bytes=24 * 1024**3 - 50,
                    allowed_venv_roots=(venv,),
                )
            current = helpers._tree_bytes(neural, reject_symlinks=True, allowed_venv_roots=(venv,))
            self.assertEqual(measured["neural_cumulative_bytes"], current)
            self.assertGreater(measured["neural_cumulative_bytes"], len(b"prior evidence"))
            self.assertEqual(measured["reservation_bytes"], 24 * 1024**3 - 50)
            args = helpers._cli().parse_args([
                "verify-storage", str(root), str(neural), "setup",
                "--reservation-bytes", str(24 * 1024**3),
                *(str(path) for path in runtime_roots),
            ])
            self.assertEqual(args.new_gpu_roots, runtime_roots)
            invalid = neural / "runtime" / "unfamiliar-link"
            os.symlink("../../outside", invalid)
            with self.assertRaisesRegex(helpers.Task8ControlError, "unrecognized symlink"):
                helpers._tree_bytes(neural, reject_symlinks=True, allowed_venv_roots=(venv,))

    def test_storage_counts_only_observed_protected_cpu_venv_aliases_without_following_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neural = root / "neural-combat"
            cpu_runtime = neural / "runtime" / "retained-cpu"
            cpu_venv = cpu_runtime / "data" / "venv-cpu"
            output = neural / "runtime-output"
            (cpu_venv / "bin").mkdir(parents=True)
            (cpu_venv / "lib").mkdir()
            output.mkdir(parents=True)
            (cpu_venv / "lib" / "module.py").write_bytes(b"retained CPU runtime bytes")
            os.symlink("/usr/local/bin/python", cpu_venv / "bin" / "python")
            os.symlink("python", cpu_venv / "bin" / "python3")
            os.symlink("python", cpu_venv / "bin" / "python3.12")
            os.symlink("lib", cpu_venv / "lib64")
            with patch.object(helpers.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                measured = helpers.check_storage_paths(
                    root, neural, (cpu_runtime, output), setup=True,
                    allowed_venv_roots=(cpu_venv,), protected_cpu_venv_root=cpu_venv,
                )
            self.assertGreaterEqual(
                measured["neural_current_bytes"],
                sum((cpu_venv / "bin" / name).lstat().st_size for name in ("python", "python3", "python3.12")),
            )
            with patch.object(helpers.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch("sys.stdout", new=io.StringIO()):
                cli_status = helpers.main([
                    "verify-storage", str(root), str(neural), "setup", str(cpu_runtime), str(output),
                    "--protected-cpu-venv", str(cpu_venv), "--allow-venv", str(cpu_venv),
                ])
            self.assertEqual(cli_status, 0)
            os.unlink(cpu_venv / "bin" / "python3.12")
            os.symlink("../../outside", cpu_venv / "bin" / "python3.12")
            with patch.object(helpers.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                with self.assertRaisesRegex(helpers.Task8ControlError, "unrecognized.*symlink"):
                    helpers.check_storage_paths(
                        root, neural, (cpu_runtime, output), setup=True,
                        allowed_venv_roots=(cpu_venv,), protected_cpu_venv_root=cpu_venv,
                    )

    def test_install_projection_uses_wheel_uncompressed_sizes_and_bounded_overhead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            _make_wheel(wheelhouse, "sample", "1.0")
            local = root / "local.whl"
            _make_wheel(root, "local", "1.0", filename_override="local-1.0-py3-none-any.whl")
            (root / "local-1.0-py3-none-any.whl").replace(local)
            projection = helpers.project_offline_install_bytes(wheelhouse, local)
            self.assertEqual(projection["wheel_count"], 2)
            self.assertGreater(projection["uncompressed_wheel_bytes"], 0)
            self.assertGreater(projection["projected_additional_bytes"], projection["uncompressed_wheel_bytes"])

    def test_safe_parent_inputs_require_absolute_canonical_non_symlink_task8_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory) / "task8"
            task_root.mkdir()
            (task_root / "sub").mkdir()
            release = task_root / "release.json"
            snapshot = task_root / "snapshot.json"
            release.write_text("{}\n", encoding="utf-8")
            snapshot.write_text("{}\n", encoding="utf-8")
            result = helpers.validate_safe_task8_inputs(task_root, (release, snapshot))
            self.assertEqual(result, (release, snapshot))
            with self.assertRaisesRegex(helpers.Task8ControlError, "absolute"):
                helpers.validate_safe_task8_inputs(task_root, (Path("release.json"),))
            with self.assertRaisesRegex(helpers.Task8ControlError, "canonical"):
                helpers.validate_safe_task8_inputs(task_root, (task_root / "sub" / ".." / "release.json",))
            escaped = Path(directory) / "escaped.json"
            escaped.write_text("{}\n", encoding="utf-8")
            link = task_root / "linked.json"
            os.symlink(escaped, link)
            with self.assertRaisesRegex(helpers.Task8ControlError, "symlink|escape"):
                helpers.validate_safe_task8_inputs(task_root, (link,))

    def test_corrected_rng_attempt_rejects_noncanonical_source_before_claiming_phase_roots(self):
        task_root = ROOT / "artifacts/development/neural-combat/task-8"
        run_id = helpers.CORRECTED_RNG_RUN_ID
        wrong_snapshot = task_root / "task-8-preparation-fix4-final-source-snapshot.json"
        release = task_root / "phase-releases/rng-red.json"
        with tempfile.TemporaryDirectory() as directory:
            boundary_root = Path(directory)
            fake_bin = boundary_root / "bin"
            fake_bin.mkdir()
            call_log = boundary_root / "command-calls.log"
            for command in ("python3.12", "timeout", "docker"):
                fake = fake_bin / command
                fake.write_text(
                    "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> \"$SPIRE_LEARN_TASK8_PREFLIGHT_CALL_LOG\"\n"
                    "exit 97\n",
                    encoding="utf-8",
                )
                fake.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = os.pathsep.join((str(fake_bin), os.defpath))
            env["SPIRE_LEARN_TASK8_PREFLIGHT_CALL_LOG"] = str(call_log)

            run_roots = tuple(
                task_root / parent / run_id
                for parent in ("runtime", "runtime-output", "runtime-control", "runtime-admin", "docker-config")
            )

            def filesystem_state(path):
                if not path.exists() and not path.is_symlink():
                    return ("missing",)
                info = path.lstat()
                metadata = (
                    info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns,
                )
                if path.is_symlink():
                    return ("symlink", metadata, os.readlink(path))
                if path.is_dir():
                    children = tuple(
                        (child.name, filesystem_state(child))
                        for child in sorted(path.iterdir(), key=lambda item: item.name)
                    )
                    return ("directory", metadata, children)
                return ("file", metadata)

            corrected_phase_root = task_root / "runtime-control" / run_id / "rng-red-corrected-1"
            corrected_started_marker = task_root / "runtime-control" / run_id / "rng-red-corrected-1.started"
            corrected_roots = (
                task_root / "runtime-output" / run_id / "rng-red-corrected-1-work",
                corrected_phase_root,
                task_root / "runtime-admin" / run_id / "rng-red-corrected-1",
            )
            self.assertTrue(wrong_snapshot.is_file())
            self.assertTrue(release.is_file())
            self.assertTrue(all(path.is_dir() and not path.is_symlink() for path in corrected_roots))
            self.assertTrue(corrected_started_marker.is_file() and not corrected_started_marker.is_symlink())
            before = tuple(filesystem_state(path) for path in run_roots)

            result = subprocess.run(
                [
                    "/bin/bash", str(ROOT / "scripts/neural/run-task8-gpu-setup.sh"),
                    "rng-red", run_id, str(release), str(wrong_snapshot),
                    "--attempt-id", helpers.CORRECTED_RNG_ATTEMPT_ID,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("exact reviewed corrected RED source snapshot", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(call_log.exists(), "invalid arguments reached Python, timeout, or Docker")
            self.assertEqual(
                before,
                tuple(filesystem_state(path) for path in run_roots),
                "invalid arguments changed retained Task 8 phase data",
            )

    def test_source_snapshot_binds_exact_tree_file_set_bytes_and_executable_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            task = repo / "artifacts" / "task8"
            (repo / "src" / "spire_learn").mkdir(parents=True)
            (repo / "tests").mkdir()
            (repo / "scripts" / "neural").mkdir(parents=True)
            (repo / "src" / "spire_learn" / "module.py").write_text("value = 1\n", encoding="utf-8")
            (repo / "tests" / "test_module.py").write_text("assert True\n", encoding="utf-8")
            runner = repo / "scripts" / "neural" / "runner.sh"
            runner.write_text("#!/bin/sh\n", encoding="utf-8")
            runner.chmod(0o755)
            task.mkdir(parents=True)
            snapshot_path = task / "source-snapshot.json"
            helpers.capture_source_snapshot(
                repo, snapshot_path, scope_roots=("src", "tests", "scripts/neural/runner.sh"),
            )
            verified = helpers.verify_source_snapshot(snapshot_path, repo)
            self.assertEqual(verified["file_count"], 3)
            runner.chmod(0o644)
            with self.assertRaisesRegex(helpers.Task8ControlError, "snapshot mismatch"):
                helpers.verify_source_snapshot(snapshot_path, repo)
            runner.chmod(0o755)
            (repo / "src" / "spire_learn" / "module.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "snapshot mismatch"):
                helpers.verify_source_snapshot(snapshot_path, repo)

    def test_phase_inventory_is_exclusive_verifiable_and_records_only_known_venv_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, control, admin = (root / name for name in ("phase-output", "phase-control", "phase-admin"))
            for path in (output, control, admin):
                path.mkdir()
            sibling_phase = root / "sibling-phase"
            sibling_phase.mkdir()
            (control / "phase-status.json").write_text("{}\n", encoding="utf-8")
            venv = output / "venv-cuda"
            (venv / "lib").mkdir(parents=True)
            os.symlink("lib", venv / "lib64")
            inventory_path = admin / "evidence-inventory.json"
            result = helpers.inventory_roots(
                (output, control, admin), inventory_path, allowed_venv_roots=(venv,),
            )
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            links = [item for item in inventory["entries"] if item.get("kind") == "symlink"]
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]["link_target"], "lib")
            self.assertTrue(helpers.verify_evidence_inventory(
                inventory_path, allowed_venv_roots=(venv,),
            )["unchanged"])
            unsafe_output, unsafe_control, unsafe_admin = (
                root / "unsafe-output", root / "unsafe-control", root / "unsafe-admin",
            )
            for path in (unsafe_output, unsafe_control, unsafe_admin):
                path.mkdir()
            unapproved_venv = unsafe_output / "another-venv"
            (unapproved_venv / "bin").mkdir(parents=True)
            os.symlink("/usr/local/bin/python", unapproved_venv / "bin" / "python")
            with self.assertRaisesRegex(helpers.Task8ControlError, "unrecognized symlink"):
                helpers.inventory_roots(
                    (unsafe_output, unsafe_control, unsafe_admin), unsafe_admin / "rejected-inventory.json",
                    allowed_venv_roots=(venv, unapproved_venv),
                )
            (sibling_phase / "next-phase.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(helpers.verify_evidence_inventory(
                inventory_path, allowed_venv_roots=(venv,),
            )["unchanged"])
            (admin / "later.log").write_text("late mutation\n", encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "inventory mismatch"):
                helpers.verify_evidence_inventory(inventory_path, allowed_venv_roots=(venv,))
            self.assertGreater(result["entry_count"], 0)

    def test_later_phase_identities_bind_prior_inventory_and_source_lock_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, control, admin = (root / name for name in ("resolve-output", "resolve-control", "resolve-admin"))
            for path in (output, control, admin):
                path.mkdir()
            retained = output / "closure.json"
            retained.write_text("{\"closure\":1}\n", encoding="utf-8")
            inventory_path = admin / "evidence-inventory.json"
            helpers.inventory_roots((output, control, admin), inventory_path)
            source_snapshot = root / "source-snapshot.json"
            lock = root / "requirements-cuda.lock"
            source_snapshot.write_bytes(b"source snapshot v1\n")
            lock.write_bytes(b"torch==2.14.1+cu130\n")
            next_control = root / "rng-control"
            next_control.mkdir()
            identities_path = next_control / "phase-input-identities.json"
            recorded = helpers.write_phase_input_identities(
                identities_path,
                prior_inventories={"resolve": inventory_path},
                immutable_paths={"source_snapshot": source_snapshot, "closure_lock": lock},
            )
            self.assertEqual(recorded["prior_phase_count"], 1)
            self.assertTrue(helpers.verify_phase_input_identities(identities_path)["unchanged"])
            source_snapshot.write_bytes(b"source snapshot changed\n")
            with self.assertRaisesRegex(helpers.Task8ControlError, "identities changed"):
                helpers.verify_phase_input_identities(identities_path)
            source_snapshot.write_bytes(b"source snapshot v1\n")
            retained.write_text("{\"closure\":2}\n", encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "inventory mismatch"):
                helpers.verify_phase_input_identities(identities_path)

    def test_cuda_closure_audit_emits_hash_locked_all_wheels_only_after_complete_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            manifest_path = root / "closure.json"
            lock_path = root / "requirements-cuda.lock"
            manifest = _closure_fixture(wheelhouse)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(helpers, "CUDA_TORCH_SHA256", manifest["fixture_torch_sha256"]), \
                    patch.object(helpers, "CUDA_TORCH_METADATA_SHA256", manifest["fixture_torch_metadata_sha256"]):
                result = helpers.audit_cuda_closure(
                    manifest_path, wheelhouse, lock_path, compatible_tags=FIXTURE_TARGET_TAGS,
                )
            self.assertEqual(result["package_count"], len(manifest["packages"]))
            self.assertFalse(result["offline_install_authorized"])
            lock = lock_path.read_text(encoding="utf-8")
            self.assertIn("torch==2.14.1+cu130", lock)
            self.assertIn("numpy==2.5.4", lock)
            self.assertIn("gymnasium==1.4.0", lock)
            self.assertIn("triton==3.8.0", lock)
            self.assertIn(f"spire-lab @ file:///inputs/{helpers.CPU_WHEEL_FILENAME}", lock)
            with self.assertRaisesRegex(helpers.Task8ControlError, "already exists"):
                helpers.audit_cuda_closure(manifest_path, wheelhouse, lock_path)

    def test_cuda_closure_audit_expands_all_compressed_filename_tags_against_wheel_metadata(self):
        def audit(manifest, wheelhouse, lock_path):
            with patch.object(helpers, "CUDA_TORCH_SHA256", manifest["fixture_torch_sha256"]), \
                    patch.object(helpers, "CUDA_TORCH_METADATA_SHA256", manifest["fixture_torch_metadata_sha256"]):
                return helpers.audit_cuda_closure(
                    manifest_path, wheelhouse, lock_path, compatible_tags=FIXTURE_TARGET_TAGS,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            manifest = _closure_fixture(wheelhouse, compressed_cuda_bindings=True)
            package = next(item for item in manifest["packages"] if item["name"] == "cuda-bindings")
            wheel_path = wheelhouse / package["filename"]
            self.assertIn("py2.py3-none.abi3-", package["filename"])
            self.assertIn("py3-none-manylinux_2_17_x86_64", helpers._wheel_metadata(wheel_path)[3])
            manifest_path = root / "closure.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = audit(manifest, wheelhouse, root / "requirements-cuda.lock")
            self.assertEqual(result["package_count"], len(manifest["packages"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            manifest = _closure_fixture(
                wheelhouse, compressed_cuda_bindings=True,
                incompatible_cuda_bindings_metadata=True,
            )
            package = next(item for item in manifest["packages"] if item["name"] == "cuda-bindings")
            self.assertEqual(helpers._wheel_metadata(wheelhouse / package["filename"])[3], [
                "cp311-cp311-win_amd64",
            ])
            manifest_path = root / "closure.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(helpers.Task8ControlError, "WHEEL tags disagree"):
                audit(manifest, wheelhouse, root / "must-not-write.lock")

    def test_cuda_closure_rejects_missing_binary_dependency_and_tampered_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            manifest = _closure_fixture(wheelhouse)
            manifest["packages"] = [item for item in manifest["packages"] if item["name"] != "triton"]
            manifest["resolver"]["resolved_packages"].remove("triton")
            missing_path = root / "missing.json"
            missing_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(helpers, "CUDA_TORCH_SHA256", manifest["fixture_torch_sha256"]), \
                    patch.object(helpers, "CUDA_TORCH_METADATA_SHA256", manifest["fixture_torch_metadata_sha256"]):
                with self.assertRaisesRegex(helpers.Task8ControlError, "missing Torch runtime dependencies"):
                    helpers.audit_cuda_closure(
                        missing_path, wheelhouse, root / "never.lock", compatible_tags=FIXTURE_TARGET_TAGS,
                    )
            manifest = _closure_fixture(wheelhouse)
            manifest["packages"][0]["expected_sha256"] = "0" * 64
            tampered_path = root / "tampered.json"
            tampered_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(helpers, "CUDA_TORCH_SHA256", manifest["fixture_torch_sha256"]), \
                    patch.object(helpers, "CUDA_TORCH_METADATA_SHA256", manifest["fixture_torch_metadata_sha256"]):
                with self.assertRaisesRegex(helpers.Task8ControlError, "SHA256"):
                    helpers.audit_cuda_closure(
                        tampered_path, wheelhouse, root / "never.lock", compatible_tags=FIXTURE_TARGET_TAGS,
                    )

    def test_device_report_requires_real_architecture_finite_update_and_unowned_gpu_totals(self):
        report = {
            "status": "qualified", "gpu_uuid": helpers.GPU_UUID,
            "visible_gpu_uuid": helpers.GPU_UUID,
            "torch": "2.14.1+cu130", "torch_cuda": "13.0.3",
            "torch_cuda_initialized": True, "torch_cuda_device_count": 1,
            "device_name": helpers.GPU_NAME, "compute_capability": [12, 0],
            "driver_version": helpers.GPU_DRIVER,
            "compiled_architectures": ["sm_120"], "finite_forward": True,
            "finite_backward": True, "finite_adam": True, "synchronized": True,
            "dtype": "float32", "amp": False, "compile": False, "fallback": False,
            "peak_allocated_bytes": 1024,
            "optimizer_step_attempted": 1, "optimizer_step_accepted": 1,
            "optimizer_step_uncertain": 0, "diagnostic_adam_calls": 1,
            "owned_process_identified": False, "owned_process_memory_mib": None,
            "gpu_wide_process_memory_mib": 2048,
            "gpu_wide_processes": [{"pid": 123, "memory_mib": 2048, "gpu_uuid": helpers.GPU_UUID}],
            "memory_free_before_mib": 14000,
        }
        result = helpers.verify_device_check_report(report)
        self.assertEqual(result["status"], "qualified")
        self.assertIs(result["hard_isolation_claim"], False)
        empty_process_evidence = {
            **report, "gpu_wide_process_memory_mib": 0, "gpu_wide_processes": [],
        }
        with self.assertRaisesRegex(helpers.Task8ControlError, "at least one GPU-wide process row"):
            helpers.verify_device_check_report(empty_process_evidence)
        for field, value in (
            ("fallback", True), ("finite_adam", False), ("amp", True),
            ("owned_process_identified", True), ("optimizer_step_accepted", 2),
        ):
            with self.subTest(field=field):
                altered = dict(report)
                altered[field] = value
                with self.assertRaises(helpers.Task8ControlError):
                    helpers.verify_device_check_report(altered)

    def test_device_check_fake_torch_checks_uuid_counts_step_states_and_persists_faults(self):
        from scripts.neural import task8_device_check

        def run_fake(
            *, report_path, fail_after_step=False, raise_during_step=False,
            visible_uuid=helpers.GPU_UUID, process_output=None, expected_error=None,
        ):
            root = report_path.parent
            inventory_path = root / "gpu-inventory.json"
            inventory_path.write_text(json.dumps({
                "uuid": helpers.GPU_UUID, "memory_free_mib": 14000,
                "driver": helpers.GPU_DRIVER,
            }), encoding="utf-8")
            fake_torch = _FakeTorch(
                fail_after_step=fail_after_step, raise_during_step=raise_during_step,
            )

            def fake_run(argv, **_kwargs):
                if any("--query-gpu=" in item for item in argv):
                    return SimpleNamespace(stdout=(
                        f"{visible_uuid}, {helpers.GPU_NAME}, {helpers.GPU_DRIVER}\n"
                    ))
                if any("--query-compute-apps=" in item for item in argv):
                    return SimpleNamespace(
                        stdout=(f"4321, 498, {helpers.GPU_UUID}\n"
                                if process_output is None else process_output),
                    )
                raise AssertionError(f"unexpected subprocess argv: {argv!r}")

            with patch.dict(os.environ, {
                "SPIRE_LEARN_TASK8_DEVICE_CHECK": "1",
                "SPIRE_LEARN_TASK8_GPU_UUID": helpers.GPU_UUID,
            }), patch.dict("sys.modules", {"torch": fake_torch}), \
                    patch.object(task8_device_check.subprocess, "run", side_effect=fake_run):
                if expected_error:
                    with self.assertRaisesRegex(Exception, expected_error):
                        task8_device_check.run_device_check(inventory_path, report_path)
                    return fake_torch, json.loads(report_path.read_text(encoding="utf-8"))
                result = task8_device_check.run_device_check(inventory_path, report_path)
                return fake_torch, result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success_torch, success = run_fake(report_path=root / "success.json")
            self.assertEqual(success_torch.optimizer.step_calls, 1)
            self.assertEqual(success["diagnostic_adam_calls"], 1)
            self.assertEqual(success["optimizer_step_attempted"], 1)
            self.assertEqual(success["optimizer_step_accepted"], 1)
            self.assertEqual(success["gpu_wide_process_memory_mib"], 498)
            self.assertIs(success["owned_process_identified"], False)
            self.assertIsNone(success["owned_process_memory_mib"])
            empty_torch, empty_processes = run_fake(
                process_output="", expected_error="no GPU-wide process rows",
                report_path=root / "empty-process-query.json",
            )
            self.assertEqual(empty_torch.optimizer.step_calls, 1)
            self.assertEqual(empty_processes["status"], "failed")
            self.assertEqual(empty_processes["diagnostic_adam_calls"], 1)
            self.assertEqual(empty_processes["optimizer_step_attempted"], 1)
            self.assertEqual(empty_processes["optimizer_step_accepted"], 1)
            self.assertEqual(empty_processes["optimizer_step_uncertain"], 0)
            self.assertEqual(empty_processes["gpu_wide_processes"], [])
            self.assertEqual(empty_processes["gpu_wide_process_memory_mib"], 0)
            self.assertIn("no GPU-wide process rows", empty_processes["fault"]["message"])
            fault_torch, failed = run_fake(
                fail_after_step=True, expected_error="post-Adam", report_path=root / "failed.json",
            )
            self.assertEqual(fault_torch.optimizer.step_calls, 1)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["optimizer_step_accepted"], 1)
            self.assertEqual(failed["optimizer_step_uncertain"], 0)
            self.assertIn("synthetic post-Adam", failed["fault"]["message"])
            raised_torch, raised = run_fake(
                raise_during_step=True, expected_error="synthetic Adam step exception",
                report_path=root / "raised.json",
            )
            self.assertEqual(raised_torch.optimizer.step_calls, 1)
            self.assertEqual(raised["diagnostic_adam_calls"], 1)
            self.assertEqual(raised["optimizer_step_attempted"], 1)
            self.assertEqual(raised["optimizer_step_accepted"], 0)
            self.assertEqual(raised["optimizer_step_uncertain"], 1)
            rejected_torch, rejected = run_fake(
                visible_uuid="GPU-not-approved", expected_error="visible GPU UUID/name",
                report_path=root / "wrong-visible-uuid.json",
            )
            self.assertIsNone(rejected_torch.optimizer)
            self.assertEqual(rejected["status"], "failed")
            self.assertEqual(rejected["optimizer_step_attempted"], 0)
            self.assertIn("visible GPU UUID/name", rejected["fault"]["message"])

    def test_rng_test_inventory_matches_module_and_parser_uses_actual_text_test_runner(self):
        from tests.neural.test_gpu_qualification import GpuRngRegressionTests

        observed_names = tuple(unittest.defaultTestLoader.getTestCaseNames(GpuRngRegressionTests))
        self.assertEqual(observed_names, helpers.EXPECTED_RNG_TESTS)

        fixture_methods = {name: (lambda self: None) for name in helpers.EXPECTED_RNG_TESTS}
        fixture_case = type("GpuRngRegressionTests", (unittest.TestCase,), fixture_methods)
        fixture_case.__module__ = "tests.neural.test_gpu_qualification"
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(fixture_case)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        self.assertTrue(result.wasSuccessful())
        parsed = helpers.parse_unittest_transcript(stream.getvalue())
        self.assertEqual(parsed["observed_labels"], list(helpers.EXPECTED_RNG_LABELS))
        self.assertEqual(parsed["skipped"], 0)

    def test_rng_cuda_precondition_uses_visible_approved_uuid_with_strict_fake(self):
        from tests.neural import test_gpu_qualification as gated

        class FakeCuda:
            def __init__(self):
                self.initialized = False

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            def init(self):
                self.initialized = True

            @staticmethod
            def set_device(_index):
                return None

            def is_initialized(self):
                return self.initialized

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        approved = SimpleNamespace(stdout=f"{helpers.GPU_UUID}\n")
        with patch.object(gated.subprocess, "run", return_value=approved):
            markers = gated._require_cuda_available(fake_torch, initialize=True)
        self.assertIn("TASK8_RNG_CUDA_AVAILABLE=1", markers)
        self.assertIn("TASK8_RNG_CUDA_INITIALIZED=1", markers)
        self.assertIn(f"TASK8_RNG_APPROVED_GPU_UUID={helpers.GPU_UUID}", markers)
        with patch.object(gated.subprocess, "run", return_value=SimpleNamespace(stdout="GPU-other\n")):
            with self.assertRaisesRegex(AssertionError, "TASK8_RNG_DEVICE_MISMATCH=1"):
                gated._require_cuda_available(fake_torch, initialize=True)
        with patch.object(gated.subprocess, "run", side_effect=OSError("nvidia-smi unavailable")):
            with self.assertRaisesRegex(AssertionError, "TASK8_RNG_DEVICE_QUERY_FAILED="):
                gated._require_cuda_available(fake_torch, initialize=True)

    def test_expected_red_transcript_requires_exact_failure_set_and_no_skips(self):
        first, second, passing = helpers.EXPECTED_RNG_LABELS
        transcript = f"""{first} ... FAIL
{second} ... FAIL
{passing} ... ok

======================================================================
FAIL: {first}
Traceback (most recent call last):
TASK8_RNG_CUDA_AVAILABLE=1
TASK8_RNG_CUDA_INITIALIZED=1
TASK8_RNG_APPROVED_GPU_UUID={helpers.GPU_UUID}
TASK8_EXPECTED_RED_CUDA_RNG_STATE_CHANGED=1
AssertionError: False is not true : CUDA RNG state changed after CPU policy construction

======================================================================
FAIL: {second}
Traceback (most recent call last):
TASK8_RNG_CUDA_AVAILABLE=1
TASK8_RNG_APPROVED_GPU_UUID={helpers.GPU_UUID}
TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1
AssertionError: isolated CPU-build seeding trap failed:
TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1

Ran 3 tests in 0.001s

FAILED (failures=2)
"""
        expected_failures = helpers.EXPECTED_RNG_LABELS[:2]
        parsed = helpers.parse_unittest_transcript(
            transcript, expected_failures=expected_failures,
        )
        self.assertEqual(parsed["status"], "expected_red")
        for altered in (
            transcript.replace("TASK8_RNG_CUDA_AVAILABLE=1", ""),
            transcript.replace("TASK8_RNG_CUDA_AVAILABLE=1", "TASK8_RNG_CUDA_UNAVAILABLE=1"),
            transcript.replace("AssertionError: False is not true : CUDA RNG state changed after CPU policy construction",
                               "AssertionError: some unrelated issue"),
            transcript.replace("TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1", "TASK8_OTHER_FAILURE=1"),
            transcript.replace(f"{first} ... FAIL", f"{first} ... skipped 'gate closed'")
                .replace("FAILED (failures=2)", "OK (skipped=1)"),
            transcript.replace(f"{second} ... FAIL", f"{second} ... ERROR")
                .replace("FAILED (failures=2)", "FAILED (errors=1)"),
        ):
            with self.subTest(altered=altered), self.assertRaises(helpers.Task8ControlError):
                helpers.parse_unittest_transcript(altered, expected_failures=expected_failures)
        changed_failure_set = transcript.replace(f"{first} ... FAIL", f"{first} ... ok")
        with self.assertRaisesRegex(helpers.Task8ControlError, "failures differ"):
            helpers.parse_unittest_transcript(changed_failure_set, expected_failures=expected_failures)

    def test_expected_red_parser_accepts_real_asserttrue_text_runner_output_with_strict_fakes(self):
        from tests.neural import test_gpu_qualification as gated
        with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "1"}):
            gated = importlib.reload(gated)

        class FakeRngState:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return FakeRngState(self.value)

        class FakeParameter:
            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def view(self, _dtype):
                return self

            def reshape(self, *_shape):
                return self

            @staticmethod
            def tolist():
                return [1, 2, 3]

        class FakeCuda:
            def __init__(self):
                self.initialized = False
                self.rng_state = 0

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            def init(self):
                self.initialized = True

            @staticmethod
            def set_device(index):
                if index != 0:
                    raise AssertionError("only visible device zero is allowed")

            def is_initialized(self):
                return self.initialized

            def manual_seed_all(self, seed):
                self.rng_state = seed

            def get_rng_state(self, index):
                if index != 0 or not self.initialized:
                    raise AssertionError("CUDA RNG read requires initialized device zero")
                return FakeRngState(self.rng_state)

        class FakeTorch:
            uint8 = object()

            def __init__(self):
                self.cuda = FakeCuda()
                self.cpu_rng_state = 0

            def manual_seed(self, seed):
                self.cpu_rng_state = seed

            def get_rng_state(self):
                return FakeRngState(self.cpu_rng_state)

            @staticmethod
            def equal(left, right):
                return left.value == right.value

        class FakeSchema:
            def __enter__(self):
                return object()

            def __exit__(self, *_args):
                return False

        fake_torch = FakeTorch()

        def fake_build_policy(schema, config, *, seed, device):
            if device != "cpu" or not isinstance(config, dict) or not schema:
                raise AssertionError("strict fake accepts only the synthetic CPU policy request")
            fake_torch.cuda.rng_state += 1  # Simulate only the expected CUDA RNG regression.
            return SimpleNamespace(named_parameters=lambda: [("weight", FakeParameter())])

        def fake_api():
            return fake_torch, lambda **kwargs: kwargs, fake_build_policy

        def fake_subprocess_run(argv, **kwargs):
            if argv[0] == "nvidia-smi":
                self.assertEqual(argv[1], "--query-gpu=uuid")
                self.assertEqual(kwargs, {"check": True, "capture_output": True, "text": True, "timeout": 5})
                return SimpleNamespace(stdout=f"{helpers.GPU_UUID}\n")
            if argv[0] == gated.sys.executable and len(argv) == 4 and argv[1:3] == ["-B", "-c"]:
                child_code = argv[3]
                for required in (
                    "torch.cuda.is_available()", "torch.cuda.device_count() != 1",
                    "TASK8_RNG_DEVICE_MISMATCH=1", "TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1",
                    "TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1",
                ):
                    self.assertIn(required, child_code)
                child_stdout = (
                    "TASK8_RNG_CUDA_AVAILABLE=1\n"
                    f"TASK8_RNG_APPROVED_GPU_UUID={helpers.GPU_UUID}\n"
                    "TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1\n"
                )
                child_stderr = (
                    "Traceback (most recent call last):\n"
                    "AssertionError: TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1\n"
                )
                return SimpleNamespace(returncode=1, stdout=child_stdout, stderr=child_stderr)
            raise AssertionError(f"unexpected subprocess command in strict fake: {argv!r}")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(gated.GpuRngRegressionTests)
        stream = io.StringIO()
        try:
            with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "1"}), \
                    patch.object(gated, "_api", side_effect=fake_api), \
                    patch.object(gated, "synthetic_schema", side_effect=FakeSchema), \
                    patch.object(gated.subprocess, "run", side_effect=fake_subprocess_run):
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        finally:
            with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "0"}):
                importlib.reload(gated)
        transcript = stream.getvalue()
        self.assertFalse(
            result.wasSuccessful(),
            f"fake TestTextRunner outcomes={result.testsRun} skips={len(result.skipped)} errors={len(result.errors)}\n{transcript}",
        )
        self.assertEqual(len(result.skipped), 0, transcript)
        self.assertEqual(len(result.errors), 0, transcript)
        self.assertEqual(len(result.failures), 2, transcript)
        self.assertIn(
            "AssertionError: False is not true : CUDA RNG state changed after CPU policy construction",
            transcript,
        )
        try:
            parsed = helpers.parse_unittest_transcript(
                transcript, expected_failures=helpers.EXPECTED_RNG_LABELS[:2],
            )
        except helpers.Task8ControlError as error:
            self.fail(f"real TextTestRunner RED was rejected: {error}\n{transcript}")
        self.assertEqual(parsed["status"], "expected_red")

    def test_rng_constructor_sites_supply_bounded_actual_model_config_to_inert_builder(self):
        import contextlib
        import subprocess as subprocess_module
        import sys
        import traceback
        from types import ModuleType

        from spire_learn.neural.contracts import ModelConfig
        from tests.neural._support import synthetic_schema
        from tests.neural import test_gpu_qualification as gated

        direct_calls = []

        def strict_direct_build(schema, config, *, seed, device):
            self.assertIsNotNone(schema)
            self.assertIs(type(config), ModelConfig)
            self.assertEqual(
                (config.embedding_width, config.hidden_width, config.max_parameters),
                (16, 64, 10_000_000),
            )
            self.assertEqual((seed, device), (20261027, "cpu"))
            direct_calls.append((schema, config, seed, device))
            return SimpleNamespace(named_parameters=lambda: ())

        with synthetic_schema() as schema:
            policy = gated._policy(schema, ModelConfig, strict_direct_build, seed=20261027)
        self.assertEqual(len(direct_calls), 1)
        self.assertEqual(direct_calls[0][1].max_parameters, 10_000_000)
        self.assertEqual(policy.named_parameters(), ())

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def is_initialized():
                return False

        class FakeTorch(ModuleType):
            def __init__(self):
                super().__init__("torch")
                self.cuda = FakeCuda()

            @staticmethod
            def manual_seed(_seed):
                raise AssertionError("fake torch seed trap was not patched")

        fake_torch = FakeTorch()
        fake_model = ModuleType("spire_learn.neural.model")
        child_calls = []

        def strict_child_build(schema, config, *, seed, device):
            self.assertIsNotNone(schema)
            self.assertIs(type(config), ModelConfig)
            self.assertEqual(
                (config.embedding_width, config.hidden_width, config.max_parameters),
                (16, 64, 10_000_000),
            )
            self.assertEqual((seed, device), (19, "cpu"))
            child_calls.append((schema, config, seed, device))
            return object()

        fake_model.build_policy = strict_child_build

        def strict_nvidia_smi(argv, **kwargs):
            self.assertEqual(argv, ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"])
            self.assertEqual(
                kwargs,
                {"check": True, "capture_output": True, "text": True, "timeout": 5},
            )
            return SimpleNamespace(stdout=f"{helpers.GPU_UUID}\n")

        def run_child(argv, **kwargs):
            self.assertEqual(argv[:3], [sys.executable, "-B", "-c"])
            self.assertEqual(set(kwargs), {"check", "capture_output", "text", "env"})
            self.assertIs(kwargs["check"], False)
            self.assertIs(kwargs["capture_output"], True)
            self.assertIs(kwargs["text"], True)
            self.assertEqual(kwargs["env"]["SPIRE_LEARN_RUN_TASK8_RNG"], "1")
            child_stdout, child_stderr = io.StringIO(), io.StringIO()
            child_error = ""
            with patch.dict(sys.modules, {
                "torch": fake_torch,
                "spire_learn.neural.model": fake_model,
            }), patch.object(
                subprocess_module, "run", side_effect=strict_nvidia_smi,
            ), contextlib.redirect_stdout(child_stdout), contextlib.redirect_stderr(child_stderr):
                try:
                    exec(argv[3], {"__name__": "__main__"})
                except BaseException:
                    child_error = traceback.format_exc()
            return SimpleNamespace(
                returncode=1 if child_error else 0,
                stdout=child_stdout.getvalue(),
                stderr=child_stderr.getvalue() + child_error,
            )

        try:
            with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "1"}):
                gated = importlib.reload(gated)
            case = gated.GpuRngRegressionTests(
                "test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda",
            )
            with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "1"}), \
                    patch.object(gated.subprocess, "run", side_effect=run_child):
                case.test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda()
        finally:
            with patch.dict(os.environ, {"SPIRE_LEARN_RUN_TASK8_RNG": "0"}):
                importlib.reload(gated)
        self.assertEqual(len(child_calls), 1)
        self.assertEqual(child_calls[0][1].max_parameters, 10_000_000)

    def test_expected_red_markers_must_be_bound_to_their_exact_failure_blocks(self):
        first, second, passing = helpers.EXPECTED_RNG_LABELS
        transcript = f"""TASK8_RNG_CUDA_AVAILABLE=1
TASK8_RNG_CUDA_INITIALIZED=1
TASK8_RNG_APPROVED_GPU_UUID={helpers.GPU_UUID}
TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1
{first} ... FAIL
{second} ... FAIL
{passing} ... ok

======================================================================
FAIL: {first}
Traceback (most recent call last):
TASK8_EXPECTED_RED_CUDA_RNG_STATE_CHANGED=1
AssertionError: CUDA RNG state changed after CPU policy construction

======================================================================
FAIL: {second}
Traceback (most recent call last):
TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1
AssertionError: isolated CPU-build seeding trap failed

Ran 3 tests in 0.001s

FAILED (failures=2)
"""
        with self.assertRaisesRegex(helpers.Task8ControlError, "failure details"):
            helpers.parse_unittest_transcript(
                transcript, expected_failures=helpers.EXPECTED_RNG_LABELS[:2],
            )

    def test_command_records_are_unique_exact_and_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helpers.write_command_record(root, "pure-check", ("python3.12", "-B", "-m", "unittest"),
                                         b"ok\n", b"", 0)
            self.assertEqual((root / "pure-check.command.log").read_text(), "python3.12 -B -m unittest\n")
            self.assertEqual((root / "pure-check.stdout.log").read_bytes(), b"ok\n")
            self.assertEqual((root / "pure-check.stderr.log").read_bytes(), b"")
            self.assertEqual((root / "pure-check.exit.log").read_text(), "0\n")
            with self.assertRaisesRegex(helpers.Task8ControlError, "already exists"):
                helpers.write_command_record(root, "pure-check", ("python3.12",), b"", b"", 0)

    def test_final_inventory_excludes_only_its_own_four_log_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            admin = root / "admin"
            other = root / "other"
            for path in (runtime, admin, other):
                path.mkdir()
            (runtime / "evidence.json").write_text("{}\n", encoding="utf-8")
            for suffix in ("command", "stdout", "stderr", "exit"):
                (admin / f"evidence-inventory.{suffix}.log").write_text("self\n", encoding="utf-8")
            foreign_log = other / "evidence-inventory.command.log"
            foreign_log.write_text("included\n", encoding="utf-8")
            output = admin / "evidence-inventory.json"
            result = helpers.inventory_roots((runtime, admin, other), output)
            inventory = json.loads(output.read_text(encoding="utf-8"))
            paths = {(Path(item["root"]).name, item["path"]) for item in inventory["entries"]}
            self.assertEqual(result["entry_count"], 2)
            self.assertIn(("runtime", "evidence.json"), paths)
            self.assertIn(("other", "evidence-inventory.command.log"), paths)
            self.assertNotIn(("admin", "evidence-inventory.command.log"), paths)


if __name__ == "__main__":
    unittest.main()

