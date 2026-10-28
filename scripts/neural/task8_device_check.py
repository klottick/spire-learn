"""One-shot, non-gameplay CUDA device qualification for Task 8.

Torch is imported only after the explicit phase gate.  The check performs one
small float32 forward/backward/Adam update on synthetic tensors, never imports
the learner policy or Spire Gym, and creates a new report file exclusively.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


EXPECTED_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
EXPECTED_TORCH = "2.14.1+cu130"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5090"
MEMORY_CAP_BYTES = 8 * 1024**3


class DeviceCheckError(RuntimeError):
    """A required CUDA runtime identity or operation failed."""


def _exclusive_json(path: Path, value: dict[str, object]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise DeviceCheckError("device report path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _finite_adam_state(torch, optimizer) -> bool:
    """Check each tensor value held in Adam state, never the containing dict."""
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            raise DeviceCheckError("Adam optimizer state entry is not a mapping")
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _query_visible_gpu(subprocess_module) -> dict[str, str]:
    result = subprocess_module.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise DeviceCheckError("container-visible GPU query must return exactly one device")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise DeviceCheckError("container-visible GPU query returned malformed CSV")
    uuid, name, driver = fields
    if uuid != EXPECTED_UUID or name != EXPECTED_DEVICE:
        raise DeviceCheckError("container-visible GPU UUID/name differs from the approved device")
    return {"uuid": uuid, "name": name, "driver": driver}


def run_device_check(inventory_path: Path, report_path: Path) -> dict[str, object]:
    """Run one finite CUDA optimizer step after all host-side gates pass."""
    inventory_path, report_path = Path(inventory_path), Path(report_path)
    if report_path.exists() or report_path.is_symlink():
        raise DeviceCheckError("device report path already exists")
    report: dict[str, object] = {
        "status": "failed",
        "gpu_uuid": EXPECTED_UUID,
        "visible_gpu_uuid": None,
        "torch": "not_loaded",
        "torch_cuda": None,
        "memory_free_before_mib": None,
        "dtype": "float32",
        "amp": False,
        "compile": False,
        "fallback": False,
        "finite_forward": False,
        "finite_backward": False,
        "finite_adam": False,
        "synchronized": False,
        "optimizer_step_attempted": 0,
        "optimizer_step_accepted": 0,
        "optimizer_step_uncertain": 0,
        "diagnostic_adam_calls": 0,
        "peak_allocated_bytes": 0,
        "owned_process_identified": False,
        "owned_process_memory_mib": None,
        "gpu_wide_process_memory_mib": 0,
        "gpu_wide_processes": [],
        "gym_calls": 0,
    }
    try:
        if os.environ.get("SPIRE_LEARN_TASK8_DEVICE_CHECK") != "1":
            raise DeviceCheckError("SPIRE_LEARN_TASK8_DEVICE_CHECK=1 is required")
        if os.environ.get("SPIRE_LEARN_TASK8_GPU_UUID") != EXPECTED_UUID:
            raise DeviceCheckError("device phase must pass the inventoried GPU UUID")
        if inventory_path.is_symlink() or not inventory_path.is_file():
            raise DeviceCheckError("host GPU inventory is missing or is a symlink")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if not isinstance(inventory, dict) or inventory.get("uuid") != EXPECTED_UUID:
            raise DeviceCheckError("host GPU inventory is missing or names a different UUID")
        free_before = inventory.get("memory_free_mib")
        if type(free_before) is not int or free_before < 12 * 1024:
            raise DeviceCheckError("host GPU inventory is below 12 GiB free memory")
        report["memory_free_before_mib"] = free_before

        import torch

        report["torch"] = str(torch.__version__)
        report["torch_cuda"] = torch.version.cuda
        if str(torch.__version__) != EXPECTED_TORCH:
            raise DeviceCheckError("installed Torch version differs from the audited CUDA lock")
        if not isinstance(torch.version.cuda, str) or not torch.version.cuda.startswith("13.0"):
            raise DeviceCheckError("installed Torch does not report its CUDA 13.0 runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise DeviceCheckError("container must expose exactly the approved CUDA device")
        visible_gpu = _query_visible_gpu(subprocess)
        report["visible_gpu_uuid"] = visible_gpu["uuid"]
        if visible_gpu["driver"] != inventory.get("driver"):
            raise DeviceCheckError("container-visible GPU driver differs from the host inventory")
        torch.cuda.set_device(0)
        torch.cuda.init()
        if not torch.cuda.is_initialized():
            raise DeviceCheckError("Torch CUDA initialization did not complete")
        properties = torch.cuda.get_device_properties(0)
        device_name = str(properties.name)
        capability = list(torch.cuda.get_device_capability(0))
        architectures = list(torch.cuda.get_arch_list())
        expected_arch = f"sm_{capability[0]}{capability[1]}"
        if device_name != EXPECTED_DEVICE or expected_arch not in architectures:
            raise DeviceCheckError("Torch-visible device or compiled architecture is not qualified")
        if int(properties.total_memory) < 24 * 1024**3:
            raise DeviceCheckError("Torch-visible GPU memory is below the RTX 5090 profile")

        # This limits the PyTorch caching allocator; it is not a driver-wide cap.
        fraction = min(1.0, MEMORY_CAP_BYTES / float(properties.total_memory))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.cuda.reset_peak_memory_stats(0)
        device = torch.device("cuda:0")
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 16),
        ).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, eps=0.00001)
        inputs = torch.linspace(-1.0, 1.0, 64 * 32, dtype=torch.float32, device=device).reshape(32, 64)
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        output = model(inputs)
        loss = output.square().mean()
        torch.cuda.synchronize(0)
        forward_seconds = time.perf_counter() - start
        report["finite_forward"] = bool(torch.isfinite(output).all().item() and torch.isfinite(loss).item())

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(0)
        backward_seconds = time.perf_counter() - start
        parameters = list(model.parameters())
        gradients = [parameter.grad for parameter in parameters]
        report["finite_backward"] = bool(
            gradients and all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients)
        )

        start = time.perf_counter()
        report["optimizer_step_attempted"] = 1
        report["diagnostic_adam_calls"] = 1
        try:
            optimizer.step()
        except Exception:
            report["optimizer_step_uncertain"] = 1
            raise
        report["optimizer_step_accepted"] = 1
        torch.cuda.synchronize(0)
        adam_seconds = time.perf_counter() - start
        report["finite_adam"] = bool(
            all(torch.isfinite(parameter).all().item() for parameter in parameters)
            and _finite_adam_state(torch, optimizer)
        )

        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        current_allocated = int(torch.cuda.memory_allocated(0))
        current_reserved = int(torch.cuda.memory_reserved(0))
        torch.cuda.synchronize(0)
        process_query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        process_rows = []
        for line in process_query.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                raise DeviceCheckError("nvidia-smi process query returned malformed CSV")
            try:
                process_pid, process_mib = int(fields[0]), int(fields[1])
            except ValueError as error:
                raise DeviceCheckError("nvidia-smi process PID/memory fields are not integers") from error
            if fields[2] != EXPECTED_UUID:
                raise DeviceCheckError("container process query exposed an unapproved GPU")
            process_rows.append({"pid": process_pid, "memory_mib": process_mib, "gpu_uuid": fields[2]})
        report.update({
            "device_name": device_name,
            "compute_capability": capability,
            "compiled_architectures": architectures,
            "device_total_memory_bytes": int(properties.total_memory),
            "driver_version": visible_gpu["driver"],
            "torch_cuda_device_count": int(torch.cuda.device_count()),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "torch_allocator_peak_reserved_mib": (peak_reserved + 1024**2 - 1) // 1024**2,
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
            "gpu_wide_process_memory_mib": sum(row["memory_mib"] for row in process_rows),
            "gpu_wide_processes": process_rows,
            "owned_process_identified": False,
            "owned_process_memory_mib": None,
            "allocator_fraction": fraction,
            "forward_seconds_synchronized": forward_seconds,
            "backward_seconds_synchronized": backward_seconds,
            "adam_seconds_synchronized": adam_seconds,
            "synchronized": True,
            "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
            "python": platform.python_version(),
        })
        if not process_rows:
            raise DeviceCheckError("nvidia-smi process query returned no GPU-wide process rows")
        if not all(report[key] is True for key in ("finite_forward", "finite_backward", "finite_adam")):
            raise DeviceCheckError("finite float32 forward/backward/Adam gate failed")
        if peak_allocated > MEMORY_CAP_BYTES:
            raise DeviceCheckError("PyTorch allocator peak exceeded 8 GiB")
        report["status"] = "qualified"
        _exclusive_json(report_path, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["fault"] = {"type": type(error).__name__, "message": str(error)}
        if not report_path.exists() and not report_path.is_symlink():
            _exclusive_json(report_path, report)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_device_check(args.inventory, args.output)
    except Exception as error:
        print(f"TASK8_DEVICE_CHECK_FAILED={type(error).__name__}:{error}", file=sys.stderr)
        return 2
    print("TASK8_DEVICE_CHECK=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

