"""Pure device planning and qualification tests with strict fake boundaries."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
GPU_NAME = "NVIDIA GeForce RTX 5090"
GPU_DRIVER = "595.91.07"


def _device_module():
    return importlib.import_module("spire_learn.neural.device")


def _cpu_config():
    return SimpleNamespace(device="cpu", runtime_gpu_uuid=None)


def _cuda_config():
    return SimpleNamespace(device="cuda:0", runtime_gpu_uuid=GPU_UUID)


class FakeCuda:
    def __init__(self, *, device_count=1, uuid=GPU_UUID, free_bytes=16 * 1024**3):
        self.device_count_value = device_count
        self.uuid = uuid
        self.free_bytes = free_bytes
        self.calls = []

    def is_available(self):
        self.calls.append(("is_available",))
        return True

    def device_count(self):
        self.calls.append(("device_count",))
        return self.device_count_value

    def set_device(self, index):
        self.calls.append(("set_device", index))

    def init(self):
        self.calls.append(("init",))

    def get_device_properties(self, index):
        self.calls.append(("get_device_properties", index))
        return SimpleNamespace(name=GPU_NAME, total_memory=32 * 1024**3)

    def get_device_capability(self, index):
        self.calls.append(("get_device_capability", index))
        return (12, 0)

    def get_arch_list(self):
        self.calls.append(("get_arch_list",))
        return ["sm_120"]

    def mem_get_info(self, index):
        self.calls.append(("mem_get_info", index))
        return (self.free_bytes, 32 * 1024**3)

    def set_per_process_memory_fraction(self, fraction, index):
        self.calls.append(("set_per_process_memory_fraction", fraction, index))


class FakeTorch:
    __version__ = "2.14.1+cu130"

    def __init__(self, cuda=None):
        self.version = SimpleNamespace(cuda="13.0")
        self.cuda = cuda or FakeCuda()
        self.intraop = 2
        self.interop = 2
        self.calls = []

    def get_num_threads(self):
        return self.intraop

    def set_num_threads(self, value):
        self.calls.append(("set_num_threads", value))
        self.intraop = value

    def get_num_interop_threads(self):
        return self.interop

    def set_num_interop_threads(self, value):
        self.calls.append(("set_num_interop_threads", value))
        self.interop = value


class NeuralDeviceTests(unittest.TestCase):
    def test_planner_selects_cuda_actor_rng_and_cpu_shuffle_rng(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        self.assertEqual(
            (plan.device, plan.policy_generator_device, plan.shuffle_generator_device,
             plan.policy_rng_backend, plan.gpu_uuid),
            ("cuda:0", "cuda:0", "cpu", "cuda", GPU_UUID),
        )
        with self.assertRaises(AttributeError):
            plan.device = "cpu"

    def test_planner_selects_cpu_and_rejects_unqualified_or_inconsistent_devices(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cpu_config())
        self.assertEqual(
            (plan.device, plan.policy_generator_device, plan.shuffle_generator_device,
             plan.policy_rng_backend, plan.gpu_uuid),
            ("cpu", "cpu", "cpu", "cpu", None),
        )
        invalid = (
            SimpleNamespace(device="cuda", runtime_gpu_uuid=GPU_UUID),
            SimpleNamespace(device="cuda:1", runtime_gpu_uuid=GPU_UUID),
            SimpleNamespace(device="cuda:0", runtime_gpu_uuid=None),
            SimpleNamespace(device="cuda:0", runtime_gpu_uuid=True),
            SimpleNamespace(device="cuda:0", runtime_gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"),
            SimpleNamespace(device="cpu", runtime_gpu_uuid=GPU_UUID),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                device.plan_runtime_device(config)

    def test_execution_gate_is_cpu_noop_and_requires_exact_cuda_opt_in_and_uuid(self):
        device = _device_module()
        cpu_plan = device.plan_runtime_device(_cpu_config())
        cuda_plan = device.plan_runtime_device(_cuda_config())
        with patch.dict(os.environ, {}, clear=True):
            device.require_runtime_execution_gate(cpu_plan)
            with self.assertRaises(RuntimeError):
                device.require_runtime_execution_gate(cuda_plan)
        with patch.dict(os.environ, {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        }, clear=True):
            device.require_runtime_execution_gate(cuda_plan)
        for environment in (
            {"SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "true", "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID},
            {"SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1", "SPIRE_LEARN_NEURAL_GPU_UUID": "wrong"},
        ):
            with patch.dict(os.environ, environment, clear=True):
                with self.subTest(environment=environment), self.assertRaises(RuntimeError):
                    device.require_runtime_execution_gate(cuda_plan)

    def test_cpu_qualification_never_accesses_torch_or_cuda(self):
        device = _device_module()

        class Bomb:
            def __getattr__(self, name):
                raise AssertionError(f"CPU qualification accessed {name}")

        plan = device.plan_runtime_device(_cpu_config())
        report = device.qualify_runtime_device(Bomb(), plan)
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["gpu_uuid"], None)
        json.dumps(report, allow_nan=False)

    def test_cuda_qualification_checks_pins_and_applies_thread_and_allocator_caps(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        torch = FakeTorch()
        inventory = SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")
        with patch.object(device.subprocess, "run", return_value=inventory) as run:
            report = device.qualify_runtime_device(torch, plan)
        self.assertEqual(run.call_args.args[0], [
            "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(torch.intraop, 4)
        self.assertEqual(torch.interop, 1)
        self.assertIn(("set_per_process_memory_fraction", 0.25, 0), torch.cuda.calls)
        self.assertEqual(report["gpu_uuid"], GPU_UUID)
        self.assertEqual(report["device_name"], GPU_NAME)
        self.assertEqual(report["driver_version"], GPU_DRIVER)
        self.assertEqual(report["compiled_architecture"], "sm_120")
        self.assertEqual(report["allocator_cap_bytes"], 8 * 1024**3)
        json.dumps(report, allow_nan=False)

    def test_cuda_qualification_rejects_cuda_versions_that_only_share_the_13_0_prefix(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        inventory = SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")
        for version in ("13.01", "13.0bad"):
            with self.subTest(cuda_version=version):
                torch = FakeTorch()
                torch.version.cuda = version
                with patch.object(device.subprocess, "run", return_value=inventory) as query:
                    with self.assertRaisesRegex(RuntimeError, "Torch CUDA version"):
                        device.qualify_runtime_device(torch, plan)
                    query.assert_not_called()

    def test_cuda_qualification_rejects_wrong_uuid_and_low_free_memory(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        wrong_inventory = SimpleNamespace(stdout=f"GPU-00000000-0000-0000-0000-000000000000, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")
        with patch.object(device.subprocess, "run", return_value=wrong_inventory):
            with self.assertRaisesRegex(RuntimeError, "UUID|identity"):
                device.qualify_runtime_device(FakeTorch(), plan)

        low_memory = FakeTorch(FakeCuda(free_bytes=8 * 1024**3))
        inventory = SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 8192\n")
        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "free memory"):
                device.qualify_runtime_device(low_memory, plan)

    def test_cuda_qualification_rejects_boolean_device_count_and_noninteger_memory(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        inventory = SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")

        boolean_count = FakeTorch(FakeCuda(device_count=True))
        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "exactly one available CUDA device"):
                device.qualify_runtime_device(boolean_count, plan)

        class NonintegerMemoryCuda(FakeCuda):
            def mem_get_info(self, index):
                self.calls.append(("mem_get_info", index))
                return 16.5 * 1024**3, 32 * 1024**3

        malformed_memory = FakeTorch(NonintegerMemoryCuda())
        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "device inventory"):
                device.qualify_runtime_device(malformed_memory, plan)

        class InconsistentMemoryCuda(FakeCuda):
            def mem_get_info(self, index):
                self.calls.append(("mem_get_info", index))
                return 33 * 1024**3, 32 * 1024**3

        inconsistent_memory = FakeTorch(InconsistentMemoryCuda())
        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "device inventory"):
                device.qualify_runtime_device(inconsistent_memory, plan)

    def test_cuda_qualification_contextualizes_ordinary_errors_and_preserves_process_control(self):
        device = _device_module()
        plan = device.plan_runtime_device(_cuda_config())
        inventory = SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")

        class BrokenCudaAccess:
            __version__ = "2.14.1+cu130"
            version = SimpleNamespace(cuda="13.0")

            @property
            def cuda(self):
                raise ValueError("cuda attribute failed")

        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "Torch CUDA runtime") as caught:
                device.qualify_runtime_device(BrokenCudaAccess(), plan)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

        class BrokenPropertiesCuda(FakeCuda):
            def get_device_properties(self, index):
                self.calls.append(("get_device_properties", index))
                raise OSError("device properties failed")

        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "device inventory") as caught:
                device.qualify_runtime_device(FakeTorch(BrokenPropertiesCuda()), plan)
        self.assertIsInstance(caught.exception.__cause__, OSError)

        interruption = KeyboardInterrupt("stop requested")

        class InterruptedPropertiesCuda(FakeCuda):
            def get_device_properties(self, index):
                raise interruption

        with patch.object(device.subprocess, "run", return_value=inventory):
            with self.assertRaises(KeyboardInterrupt) as caught:
                device.qualify_runtime_device(FakeTorch(InterruptedPropertiesCuda()), plan)
        self.assertIs(caught.exception, interruption)

    def test_resume_device_contract_is_atomic_and_backend_specific(self):
        device = _device_module()
        cpu_plan = device.plan_runtime_device(_cpu_config())
        cuda_plan = device.plan_runtime_device(_cuda_config())
        device.validate_resume_device({}, cpu_plan)
        device.validate_resume_device({
            "runtime_device": "cpu", "policy_rng_backend": "cpu", "runtime_gpu_uuid": None,
        }, cpu_plan)
        device.validate_resume_device({
            "runtime_device": "cuda:0", "policy_rng_backend": "cuda", "runtime_gpu_uuid": GPU_UUID,
        }, cuda_plan)
        invalid = (
            ({"runtime_device": "cpu"}, cpu_plan),
            ({"runtime_device": "cuda:0", "policy_rng_backend": "cuda", "runtime_gpu_uuid": GPU_UUID}, cpu_plan),
            ({"runtime_device": "cuda:0", "policy_rng_backend": "cpu", "runtime_gpu_uuid": GPU_UUID}, cuda_plan),
            ({"runtime_device": "cuda:0", "policy_rng_backend": "cuda", "runtime_gpu_uuid": True}, cuda_plan),
        )
        for metadata, plan in invalid:
            with self.subTest(metadata=metadata, plan=plan), self.assertRaises(ValueError):
                device.validate_resume_device(metadata, plan)

    def test_import_orders_are_acyclic_and_optional_free_in_fresh_processes(self):
        scripts = (
            "import spire_learn.neural.config; import spire_learn.neural.device",
            "import spire_learn.neural.device; import spire_learn.neural.config",
        )
        for imports in scripts:
            script = "\n".join((
                "import builtins, os, sys",
                "blocked = {'torch', 'numpy', 'gymnasium', 'spire_lab'}",
                "attempts = []",
                "original = builtins.__import__",
                "def guarded(name, *args, **kwargs):",
                "    if name.split('.', 1)[0] in blocked:",
                "        attempts.append(name)",
                "        raise AssertionError(name)",
                "    return original(name, *args, **kwargs)",
                "builtins.__import__ = guarded",
                "os.environ.pop('SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME', None)",
                "os.environ.pop('SPIRE_LEARN_NEURAL_GPU_UUID', None)",
                imports,
                "from types import SimpleNamespace",
                "from spire_learn.neural.device import plan_runtime_device",
                f"plan_runtime_device(SimpleNamespace(device='cuda:0', runtime_gpu_uuid='{GPU_UUID}'))",
                "assert attempts == [], attempts",
                "assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)",
                "print('fresh import order ok')",
            ))
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                env=dict(os.environ, PYTHONPATH="src"), capture_output=True, text=True,
            )
            with self.subTest(imports=imports):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "fresh import order ok")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
