"""Opt-in CUDA RNG isolation regressions for Task 8.

The gate is checked by unittest decorators before Torch is imported.  These
tests use only a learner-authored synthetic public descriptor; they never make
a Gym driver, reset/step an environment, sample an action, or update a policy.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.neural._support import synthetic_schema


_RNG_GATE = "SPIRE_LEARN_RUN_TASK8_RNG"
_APPROVED_GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"


def _api():
    try:
        import torch

        from spire_learn.neural.contracts import ModelConfig
        from spire_learn.neural.model import build_policy
    except Exception as error:
        raise RuntimeError(f"TASK8_RNG_TORCH_LOAD_FAILED={type(error).__name__}:{error}") from error

    return torch, ModelConfig, build_policy


def _require_cuda_available(torch, *, initialize: bool) -> tuple[str, ...]:
    if not torch.cuda.is_available():
        raise AssertionError("TASK8_RNG_CUDA_UNAVAILABLE=1: approved CUDA device is unavailable")
    if torch.cuda.device_count() != 1:
        raise AssertionError("TASK8_RNG_PRECONDITION_FAILED=device_count: exactly one device must be visible")
    markers = ["TASK8_RNG_CUDA_AVAILABLE=1"]
    if initialize:
        try:
            torch.cuda.init()
            torch.cuda.set_device(0)
        except Exception as error:
            raise AssertionError(
                f"TASK8_RNG_CUDA_INITIALIZATION_FAILED={type(error).__name__}:{error}"
            ) from error
        if not torch.cuda.is_initialized():
            raise AssertionError("TASK8_RNG_CUDA_INITIALIZATION_FAILED=not_initialized")
        markers.append("TASK8_RNG_CUDA_INITIALIZED=1")
    try:
        visible = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except Exception as error:
        raise AssertionError(
            f"TASK8_RNG_DEVICE_QUERY_FAILED={type(error).__name__}:{error}"
        ) from error
    rows = [line.strip() for line in visible.stdout.splitlines() if line.strip()]
    if rows != [_APPROVED_GPU_UUID]:
        raise AssertionError(f"TASK8_RNG_DEVICE_MISMATCH=1: observed visible GPU UUID rows {rows!r}")
    markers.append(f"TASK8_RNG_APPROVED_GPU_UUID={rows[0]}")
    return tuple(markers)


def _policy(schema, model_config, build_policy, *, seed: int):
    return build_policy(
        schema,
        model_config(embedding_width=16, hidden_width=64, max_parameters=10_000_000),
        seed=seed,
        device="cpu",
    )


def _parameter_digest(policy, torch) -> str:
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        raw = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(bytes(raw.reshape(-1).tolist()))
    return digest.hexdigest()


class GpuRngRegressionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(_RNG_GATE) == "1", "set SPIRE_LEARN_RUN_TASK8_RNG=1")
    def test_cpu_and_initialized_cuda_rng_states_unchanged_by_build(self):
        torch, ModelConfig, build_policy = _api()
        gate_markers = _require_cuda_available(torch, initialize=True)
        torch.manual_seed(711)
        torch.cuda.manual_seed_all(991)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(0).clone()
        with synthetic_schema() as schema:
            _policy(schema, ModelConfig, build_policy, seed=20261027)
        cpu_unchanged = torch.equal(cpu_before, torch.get_rng_state())
        cuda_unchanged = torch.equal(cuda_before, torch.cuda.get_rng_state(0))
        self.assertTrue(cpu_unchanged, "CPU RNG state changed after CPU policy construction")
        self.assertTrue(
            cuda_unchanged,
            "CUDA RNG state changed after CPU policy construction\n"
            + " ".join((*gate_markers, "TASK8_EXPECTED_RED_CUDA_RNG_STATE_CHANGED=1")),
        )

    @unittest.skipUnless(os.environ.get(_RNG_GATE) == "1", "set SPIRE_LEARN_RUN_TASK8_RNG=1")
    def test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda(self):
        # A child process starts with CUDA uninitialized even if another test
        # intentionally initialized the parent process's CUDA generator.
        code = r'''
try:
    import torch
except Exception as error:
    raise RuntimeError(f"TASK8_RNG_TORCH_LOAD_FAILED={type(error).__name__}:{error}") from error
import subprocess
if not torch.cuda.is_available():
    raise RuntimeError("TASK8_RNG_CUDA_UNAVAILABLE=1: approved CUDA device is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError("TASK8_RNG_PRECONDITION_FAILED=device_count: exactly one device must be visible")
try:
    visible = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
except Exception as error:
    raise RuntimeError(f"TASK8_RNG_DEVICE_QUERY_FAILED={type(error).__name__}:{error}") from error
rows = [line.strip() for line in visible.stdout.splitlines() if line.strip()]
if rows != ["GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"]:
    raise RuntimeError(f"TASK8_RNG_DEVICE_MISMATCH=1: observed visible GPU UUID rows {rows!r}")
print("TASK8_RNG_CUDA_AVAILABLE=1")
print("TASK8_RNG_APPROVED_GPU_UUID=" + rows[0])
if torch.cuda.is_initialized():
    raise RuntimeError("TASK8_RNG_PRECONDITION_FAILED=child_cuda_already_initialized")
print("TASK8_RNG_CHILD_CUDA_UNINITIALIZED=1")
from unittest.mock import patch
from tests.neural._support import synthetic_schema
from spire_learn.neural.contracts import ModelConfig
from spire_learn.neural.model import build_policy
def forbidden_global_seed(*args, **kwargs):
    raise AssertionError("TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1")
with patch.object(torch, "manual_seed", side_effect=forbidden_global_seed) as trap:
    with synthetic_schema() as schema:
        build_policy(
            schema,
            ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000),
            seed=19,
            device="cpu",
        )
    trap.assert_not_called()
if torch.cuda.is_initialized():
    raise AssertionError("TASK8_RNG_PRECONDITION_FAILED=cpu_build_initialized_cuda")
print("TASK8_CPU_BUILD_CUDA_UNINITIALIZED=1")
'''
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        child_output = completed.stdout + completed.stderr
        cause_marker = (
            "TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1"
            if "TASK8_EXPECTED_RED_GLOBAL_MANUAL_SEED_CALLED=1" in child_output
            else "TASK8_RNG_CHILD_FAILURE_NOT_EXPECTED=1"
        )
        self.assertEqual(
            completed.returncode,
            0,
            cause_marker + "\nisolated CPU-build seeding trap failed:\n" + child_output,
        )
        self.assertIn("TASK8_CPU_BUILD_CUDA_UNINITIALIZED=1", completed.stdout)

    @unittest.skipUnless(os.environ.get(_RNG_GATE) == "1", "set SPIRE_LEARN_RUN_TASK8_RNG=1")
    def test_same_seed_builds_identical_initial_parameter_bytes(self):
        torch, ModelConfig, build_policy = _api()
        with synthetic_schema() as schema:
            first = _policy(schema, ModelConfig, build_policy, seed=54017)
            second = _policy(schema, ModelConfig, build_policy, seed=54017)
        self.assertEqual(_parameter_digest(first, torch), _parameter_digest(second, torch))


if __name__ == "__main__":
    unittest.main()

