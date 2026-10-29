"""Pure device planning and explicit runtime qualification for neural runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import re
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spire_learn.neural.config import NeuralConfig


GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
GPU_NAME = "NVIDIA GeForce RTX 5090"
GPU_DRIVER = "595.91.07"
TORCH_VERSION = "2.14.1+cu130"
TORCH_CUDA_VERSION = "13.0"
ALLOCATOR_CAP_BYTES = 8 * 1024**3
MIN_DEVICE_FREE_BYTES = 12 * 1024**3
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
_RESUME_DEVICE_FIELDS = frozenset({
    "runtime_device", "policy_rng_backend", "runtime_gpu_uuid",
})


@dataclass(frozen=True)
class RuntimeDevicePlan:
    """Immutable model, policy RNG, and shuffle RNG device assignment."""

    device: str
    policy_generator_device: str
    shuffle_generator_device: str
    policy_rng_backend: str
    gpu_uuid: str | None

    def __post_init__(self) -> None:
        expected = {
            "cpu": ("cpu", "cpu", "cpu", None),
            "cuda:0": ("cuda:0", "cpu", "cuda", GPU_UUID),
        }.get(self.device)
        actual = (
            self.policy_generator_device,
            self.shuffle_generator_device,
            self.policy_rng_backend,
            self.gpu_uuid,
        )
        if expected is None or actual != expected:
            raise ValueError("runtime device plan is not a supported CPU or qualified CUDA assignment")


def plan_runtime_device(config: NeuralConfig) -> RuntimeDevicePlan:
    """Validate the config-only device contract without importing optional packages."""
    device = getattr(config, "device", None)
    gpu_uuid = getattr(config, "runtime_gpu_uuid", None)
    if device == "cpu":
        if gpu_uuid is not None:
            raise ValueError("CPU runtime config must not declare runtime_gpu_uuid")
        return RuntimeDevicePlan("cpu", "cpu", "cpu", "cpu", None)
    if device == "cuda:0":
        if (not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None
                or gpu_uuid != GPU_UUID):
            raise ValueError("CUDA runtime config must declare the exact qualified GPU UUID")
        return RuntimeDevicePlan("cuda:0", "cuda:0", "cpu", "cuda", GPU_UUID)
    raise ValueError("neural runtime supports only device=cpu or the qualified device=cuda:0")


def require_runtime_execution_gate(plan: RuntimeDevicePlan) -> None:
    """Require the explicit protected opt-in before CUDA imports or initialization."""
    if not isinstance(plan, RuntimeDevicePlan):
        raise TypeError("plan must be RuntimeDevicePlan")
    if plan.device == "cpu":
        return
    if (os.environ.get("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME") != "1"
            or os.environ.get("SPIRE_LEARN_NEURAL_GPU_UUID") != plan.gpu_uuid):
        raise RuntimeError(
            "CUDA neural runtime requires SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=1 "
            "and the exact SPIRE_LEARN_NEURAL_GPU_UUID"
        )


def _nvidia_gpu_inventory() -> tuple[str, str, str, int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as error:
        raise RuntimeError("bounded nvidia-smi GPU inventory query failed") from error
    output = getattr(result, "stdout", None)
    if not isinstance(output, str):
        raise RuntimeError("nvidia-smi GPU inventory output must be text")
    rows = [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("nvidia-smi must report exactly one well-formed GPU inventory row")
    gpu_uuid, name, driver, total_raw, free_raw = rows[0]
    try:
        total_mib, free_mib = int(total_raw), int(free_raw)
    except ValueError as error:
        raise RuntimeError("nvidia-smi GPU memory inventory is malformed") from error
    if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
        raise RuntimeError("nvidia-smi GPU memory inventory is out of range")
    return gpu_uuid, name, driver, total_mib, free_mib


def qualify_runtime_device(
    torch_module: Any,
    plan: RuntimeDevicePlan,
) -> dict[str, object]:
    """Qualify the selected execution device and return JSON-safe observed metadata."""
    if not isinstance(plan, RuntimeDevicePlan):
        raise TypeError("plan must be RuntimeDevicePlan")
    report: dict[str, object] = {
        "device": plan.device,
        "policy_generator_device": plan.policy_generator_device,
        "shuffle_generator_device": plan.shuffle_generator_device,
        "policy_rng_backend": plan.policy_rng_backend,
        "gpu_uuid": plan.gpu_uuid,
    }
    if plan.device == "cpu":
        return report
    if torch_module is None:
        raise RuntimeError("CUDA device qualification requires the lazily imported Torch module")

    try:
        torch_version = getattr(torch_module, "__version__", None)
        torch_version_info = getattr(torch_module, "version", None)
        cuda_version = getattr(torch_version_info, "cuda", None)
    except Exception as error:
        raise RuntimeError("Torch CUDA version metadata could not be read") from error
    if torch_version != TORCH_VERSION:
        raise RuntimeError("Torch version differs from the qualified CUDA runtime")
    if cuda_version != TORCH_CUDA_VERSION:
        raise RuntimeError("Torch CUDA version differs from the qualified CUDA runtime")

    visible_uuid, visible_name, driver, total_mib, free_mib = _nvidia_gpu_inventory()
    if (not _GPU_UUID_RE.fullmatch(visible_uuid)
            or visible_uuid != plan.gpu_uuid
            or visible_name != GPU_NAME
            or driver != GPU_DRIVER):
        raise RuntimeError("nvidia-smi GPU UUID, name, or driver differs from the qualified device")

    try:
        cuda = getattr(torch_module, "cuda", None)
    except Exception as error:
        raise RuntimeError("Torch CUDA runtime could not be accessed") from error
    if cuda is None:
        raise RuntimeError("Torch module has no CUDA runtime")
    try:
        available = cuda.is_available()
        device_count = cuda.device_count()
    except Exception as error:
        raise RuntimeError("Torch CUDA availability check failed") from error
    if available is not True or type(device_count) is not int or device_count != 1:
        raise RuntimeError("exactly one available CUDA device is required")
    try:
        cuda.set_device(0)
        cuda.init()
        properties = cuda.get_device_properties(0)
        raw_capability = cuda.get_device_capability(0)
        if (not isinstance(raw_capability, (tuple, list))
                or len(raw_capability) != 2
                or any(type(value) is not int for value in raw_capability)):
            raise ValueError("Torch CUDA compute capability is malformed")
        capability = tuple(raw_capability)
        architecture = f"sm_{capability[0]}{capability[1]}" if len(capability) == 2 else ""
        architectures = tuple(cuda.get_arch_list())
        raw_memory = cuda.mem_get_info(0)
        if (not isinstance(raw_memory, (tuple, list))
                or len(raw_memory) != 2
                or any(type(value) is not int for value in raw_memory)):
            raise ValueError("Torch CUDA memory inventory is malformed")
        free_bytes, total_bytes = raw_memory
        if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
            raise ValueError("Torch CUDA memory inventory is out of range")
    except Exception as error:
        raise RuntimeError("Torch CUDA initialization or device inventory failed") from error

    try:
        device_name = getattr(properties, "name", None)
    except Exception as error:
        raise RuntimeError("Torch-visible GPU properties could not be read") from error
    if (device_name != GPU_NAME
            or capability != (12, 0)
            or architecture != "sm_120"
            or architecture not in architectures):
        raise RuntimeError("Torch-visible GPU name, capability, or compiled architecture is unqualified")
    if free_bytes < MIN_DEVICE_FREE_BYTES or free_mib * 1024**2 < MIN_DEVICE_FREE_BYTES:
        raise RuntimeError("GPU has less than 12 GiB of free memory")
    try:
        if torch_module.get_num_threads() != 4:
            torch_module.set_num_threads(4)
        if torch_module.get_num_interop_threads() != 1:
            torch_module.set_num_interop_threads(1)
        intraop_threads = torch_module.get_num_threads()
        interop_threads = torch_module.get_num_interop_threads()
    except Exception as error:
        raise RuntimeError("Torch CPU thread limits could not be configured") from error
    if intraop_threads != 4 or interop_threads != 1:
        raise RuntimeError("Torch intra-op/inter-op thread limits must be four/one")

    allocator_fraction = min(1.0, ALLOCATOR_CAP_BYTES / float(total_bytes))
    try:
        cuda.set_per_process_memory_fraction(allocator_fraction, 0)
    except Exception as error:
        raise RuntimeError("Torch CUDA allocator could not enforce the 8 GiB ceiling") from error

    report.update({
        "gpu_uuid": visible_uuid,
        "device_name": visible_name,
        "driver_version": driver,
        "torch_version": str(torch_version),
        "torch_cuda_version": str(cuda_version),
        "cuda_device_count": 1,
        "compute_capability": list(capability),
        "compiled_architecture": architecture,
        "compiled_architectures": list(architectures),
        "device_total_bytes_torch": total_bytes,
        "device_free_bytes_torch": free_bytes,
        "device_total_mib_nvidia_smi": total_mib,
        "device_free_mib_nvidia_smi": free_mib,
        "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        "allocator_fraction": allocator_fraction,
        "torch_intraop_threads": intraop_threads,
        "torch_interop_threads": interop_threads,
    })
    return report


def validate_resume_device(metadata: Mapping[str, object], plan: RuntimeDevicePlan) -> None:
    """Validate atomic resume device metadata; an all-absent legacy payload is CPU."""
    if not isinstance(metadata, Mapping):
        raise TypeError("resume run metadata must be a mapping")
    if not isinstance(plan, RuntimeDevicePlan):
        raise TypeError("plan must be RuntimeDevicePlan")
    present = _RESUME_DEVICE_FIELDS.intersection(metadata)
    if not present:
        if plan.device == "cpu":
            return
        raise ValueError("legacy checkpoint device metadata is CPU-only")
    if present != _RESUME_DEVICE_FIELDS:
        raise ValueError("resume runtime device metadata fields must be present atomically")

    expected_uuid = plan.gpu_uuid
    if (metadata.get("runtime_device") != plan.device
            or metadata.get("policy_rng_backend") != plan.policy_rng_backend
            or metadata.get("runtime_gpu_uuid") != expected_uuid):
        raise ValueError("resume checkpoint runtime device/backend/UUID does not match the selected config")


__all__ = [
    "RuntimeDevicePlan",
    "plan_runtime_device",
    "qualify_runtime_device",
    "require_runtime_execution_gate",
    "validate_resume_device",
]
