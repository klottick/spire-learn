"""Pure checkpoint identity and byte-inventory preflight.

Tensor export and restricted deserialization are kept behind the separately
reviewed CPU checkpoint-test phase. This module intentionally imports no ML
framework and rejects incompatible metadata before such a loader is entered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

from spire_learn.neural.contracts import NeuralIdentity
from spire_learn.neural.modes import NEURAL_MODES


CHECKPOINT_SCHEMA_ID = "spire-learn-neural-checkpoint-v1"
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 65_536
MAX_TENSOR_RANK = 64
MAX_TENSOR_DIMENSION = MAX_CHECKPOINT_BYTES
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TENSOR_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}
_PURPOSES = frozenset({"evaluate", "warm_start", "resume"})
_PAYLOAD_NAMES = frozenset({"model.pt", "training.pt"})
_IDENTITY_KEYS = frozenset({
    "model_id", "features_id", "action_id", "objective_id",
    "accepted_raw_mode_mappings", "config_hashes",
})
_TOP_LEVEL_KEYS = frozenset({
    "schema_id", "identity", "payload_inventory", "model_tensors",
    "training_state", "run_metadata", "parent_checkpoint",
})


class CheckpointCompatibilityError(ValueError):
    """Checkpoint identities or purpose-specific state contracts disagree."""


@dataclass(frozen=True)
class CheckpointPreflight:
    """Verified JSON/file inventory returned before optional tensor loading."""

    path: Path
    metadata: Mapping[str, object]
    purpose: str
    model_path: Path
    training_path: Path | None
    stored_bytes: int
    uncompressed_bytes: int
    logical_tensor_bytes: int


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded named state with recovery policy made explicit."""

    metadata: Mapping[str, object]
    model_tensors: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None
    rng_state: Mapping[str, Any] | None
    optimizer_step: int
    inflight_record_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping) or not isinstance(self.model_tensors, Mapping):
            raise TypeError("loaded checkpoint metadata and model tensors must be mappings")
        if any(not isinstance(name, str) or not name for name in self.model_tensors):
            raise ValueError("loaded model tensor names must be non-empty strings")
        if self.optimizer_state is not None and not isinstance(self.optimizer_state, Mapping):
            raise TypeError("loaded optimizer state must be a mapping or None")
        if self.rng_state is not None and not isinstance(self.rng_state, Mapping):
            raise TypeError("loaded RNG state must be a mapping or None")
        if (self.optimizer_state is None) != (self.rng_state is None):
            raise ValueError("optimizer and RNG state must be present or absent together")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("loaded optimizer_step must be a non-negative integer")
        if type(self.inflight_record_count) is not int or self.inflight_record_count != 0:
            raise ValueError("loaded checkpoints must drop in-flight records")


def _strict_json(value: object, *, depth: int = 0, count: list[int] | None = None) -> object:
    if count is None:
        count = [100_000]
    count[0] -= 1
    if count[0] < 0 or depth > 32:
        raise ValueError("checkpoint JSON exceeds its structure limit")
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        if not -(2**64 - 1) <= value <= 2**64 - 1:
            raise ValueError("checkpoint JSON integer exceeds the supported 64-bit range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("checkpoint JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint JSON mapping keys must be strings")
            if key in result:
                raise ValueError(f"duplicate checkpoint JSON key: {key}")
            result[key] = _strict_json(child, depth=depth + 1, count=count)
        return result
    if isinstance(value, (list, tuple)):
        return [_strict_json(child, depth=depth + 1, count=count) for child in value]
    raise TypeError(f"unsupported checkpoint JSON value type: {type(value).__name__}")


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def identity_payload(identity: NeuralIdentity) -> dict[str, object]:
    if not isinstance(identity, NeuralIdentity):
        raise TypeError("expected must be a NeuralIdentity")
    return {
        "model_id": identity.model_id,
        "features_id": identity.features_id,
        "action_id": identity.action_id,
        "objective_id": identity.objective_id,
        "accepted_raw_mode_mappings": dict(identity.accepted_raw_mode_mappings),
        "config_hashes": dict(identity.config_hashes),
    }


def _require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_identity(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != _IDENTITY_KEYS:
        raise ValueError("checkpoint identity has an invalid field set")
    result = _strict_json(raw)
    assert isinstance(result, dict)
    for name in ("model_id", "features_id", "action_id", "objective_id"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"checkpoint identity {name} must be non-empty text")
    mappings = result["accepted_raw_mode_mappings"]
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("checkpoint identity raw-mode mappings must be a non-empty object")
    if any(mode not in NEURAL_MODES for mode in mappings):
        raise ValueError("checkpoint identity declares an unsupported raw mode")
    if any(not isinstance(mapping, str) or not mapping for mapping in mappings.values()):
        raise ValueError("checkpoint identity raw-mode mapping IDs must be non-empty text")
    config_hashes = result["config_hashes"]
    if not isinstance(config_hashes, dict) or not config_hashes:
        raise ValueError("checkpoint config hashes must be a non-empty object")
    for name, digest in config_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint config hash names must be non-empty text")
        _require_hash(digest, f"checkpoint config hash {name}")
    if "model" not in config_hashes:
        raise ValueError("checkpoint identity must include the model config hash")
    return result


def _validate_tensor_inventory(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("model_tensors must be a non-empty named tensor inventory")
    names: set[str] = set()
    result: list[dict[str, object]] = []
    aggregate_logical_bytes = 0
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"name", "dtype", "shape", "sha256"}:
            raise ValueError("model tensor inventory entries have an invalid field set")
        name, dtype, shape = item["name"], item["dtype"], item["shape"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("model tensor names must be unique non-empty strings")
        if not isinstance(dtype, str) or dtype not in _TENSOR_DTYPE_BYTES:
            raise ValueError(f"model tensor {name} has unsupported model tensor dtype")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"model tensor {name} shape must be a list of non-negative integers")
        if len(shape) > MAX_TENSOR_RANK:
            raise ValueError(f"model tensor {name} rank exceeds {MAX_TENSOR_RANK}")
        if any(size > MAX_TENSOR_DIMENSION for size in shape):
            raise ValueError(f"model tensor {name} dimension exceeds {MAX_TENSOR_DIMENSION}")
        logical_bytes = _tensor_logical_bytes(name, dtype, shape)
        if logical_bytes > MAX_CHECKPOINT_BYTES:
            raise ValueError(f"model tensor {name} logical tensor bytes exceed 256 MiB")
        aggregate_logical_bytes += logical_bytes
        if aggregate_logical_bytes > MAX_CHECKPOINT_BYTES:
            raise ValueError("aggregate logical tensor bytes exceed 256 MiB")
        names.add(name)
        result.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "sha256": _require_hash(item["sha256"], f"model tensor {name} hash"),
        })
    return result


def _tensor_logical_bytes(name: str, dtype: str, shape: list[int]) -> int:
    """Compute checked dense logical bytes for one named tensor."""
    if any(dimension == 0 for dimension in shape):
        return 0
    width = _TENSOR_DTYPE_BYTES[dtype]
    elements = 1
    for dimension in shape:
        # Divide before multiplying so the check remains safe in fixed-width
        # implementations too, and never forms an unbounded intermediate.
        if elements > MAX_CHECKPOINT_BYTES // width // dimension:
            return MAX_CHECKPOINT_BYTES + 1
        elements *= dimension
    return elements * width


def _validate_payload_inventory(raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, Mapping) or "model.pt" not in raw or set(raw) - _PAYLOAD_NAMES:
        raise ValueError("payload_inventory must declare model.pt and only known payload names")
    result: dict[str, dict[str, object]] = {}
    for name, item in raw.items():
        if not isinstance(name, str) or name not in _PAYLOAD_NAMES:
            raise ValueError("checkpoint payload filename is not allowed")
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts or "\\" in name:
            raise ValueError("checkpoint payload filename must be a safe relative name")
        if not isinstance(item, Mapping) or set(item) != {"size_bytes", "sha256"}:
            raise ValueError(f"payload inventory entry {name} has an invalid field set")
        size = item["size_bytes"]
        if type(size) is not int or size <= 0 or size > MAX_CHECKPOINT_BYTES:
            raise ValueError(f"payload inventory size for {name} is outside the checkpoint limit")
        result[name] = {
            "size_bytes": size,
            "sha256": _require_hash(item["sha256"], f"payload hash for {name}"),
        }
    return result


def _validate_training_state(raw: object, *, has_training_payload: bool) -> dict[str, object] | None:
    if raw is None:
        if has_training_payload:
            raise ValueError("training.pt requires a training_state metadata record")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("training_state must be an object or null")
    required = {
        "optimizer_step", "hyperparameter_id", "rng_domains", "counters",
        "update_id", "behavior_policy_version", "next_environment_seed_ordinal",
        "next_decision_sequence", "saved_inflight_record_count",
    }
    if set(raw) != required:
        raise ValueError(f"training_state must contain exactly {sorted(required)}")
    state = _strict_json(raw)
    assert isinstance(state, dict)
    for name in ("optimizer_step", "next_environment_seed_ordinal", "next_decision_sequence", "saved_inflight_record_count"):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError(f"training_state {name} must be a non-negative integer")
    _require_hash(state["hyperparameter_id"], "training_state hyperparameter_id")
    for name in ("update_id", "behavior_policy_version"):
        if type(state[name]) not in (str, int) or state[name] == "":
            raise ValueError(f"training_state {name} must be a non-empty string or integer")
        if type(state[name]) is int and state[name] < 0:
            raise ValueError(f"training_state {name} cannot be negative")
    if (not isinstance(state["rng_domains"], list) or len(state["rng_domains"]) != 4
            or set(state["rng_domains"]) != {"policy", "minibatch", "curriculum", "controller"}):
        raise ValueError("training_state rng_domains must declare policy/minibatch/curriculum/controller")
    if not isinstance(state["counters"], dict):
        raise ValueError("training_state counters must be an object")
    return state


def validate_checkpoint_metadata(
    metadata: Mapping[str, object], expected: NeuralIdentity, *, purpose: str,
) -> dict[str, object]:
    """Validate checkpoint identity and state requirements without tensor imports."""
    if purpose not in _PURPOSES:
        raise ValueError("purpose must be evaluate, warm_start, or resume")
    if not isinstance(expected, NeuralIdentity):
        raise TypeError("expected must be a NeuralIdentity")
    safe = _strict_json(metadata)
    if not isinstance(safe, dict) or set(safe) != _TOP_LEVEL_KEYS:
        raise ValueError("checkpoint metadata has an invalid top-level field set")
    if safe.get("schema_id") != CHECKPOINT_SCHEMA_ID:
        raise ValueError("checkpoint schema_id is unsupported")

    saved_identity = _validate_identity(safe["identity"])
    payload_inventory = _validate_payload_inventory(safe["payload_inventory"])
    model_tensors = _validate_tensor_inventory(safe["model_tensors"])
    training_state = _validate_training_state(
        safe["training_state"], has_training_payload="training.pt" in payload_inventory
    )
    run_metadata = safe["run_metadata"]
    if not isinstance(run_metadata, dict):
        raise ValueError("run_metadata must be a strict JSON object")
    parent = safe["parent_checkpoint"]
    if parent is not None and not isinstance(parent, dict):
        raise ValueError("parent_checkpoint must be an object or null")

    expected_payload = identity_payload(expected)
    for name in ("model_id", "features_id", "action_id", "objective_id"):
        if saved_identity[name] != expected_payload[name]:
            raise CheckpointCompatibilityError(f"checkpoint {name} is incompatible")
    saved_mappings = saved_identity["accepted_raw_mode_mappings"]
    expected_mappings = expected_payload["accepted_raw_mode_mappings"]
    assert isinstance(saved_mappings, dict) and isinstance(expected_mappings, dict)
    for mode, mapping in expected_mappings.items():
        if saved_mappings.get(mode) != mapping:
            raise CheckpointCompatibilityError(
                f"checkpoint does not declare the expected {mode} raw-mode mapping"
            )
    saved_hashes = saved_identity["config_hashes"]
    expected_hashes = expected_payload["config_hashes"]
    assert isinstance(saved_hashes, dict) and isinstance(expected_hashes, dict)
    if saved_hashes.get("model") != expected_hashes.get("model"):
        raise CheckpointCompatibilityError("checkpoint model config hash is incompatible")
    if purpose == "resume":
        if training_state is None or "training.pt" not in payload_inventory:
            raise CheckpointCompatibilityError("resume requires optimizer, RNG, and training state")
        if saved_hashes != expected_hashes:
            raise CheckpointCompatibilityError("resume requires matching learner config hashes")
        if training_state["hyperparameter_id"] != expected_hashes.get("ppo"):
            raise CheckpointCompatibilityError("resume hyperparameter ID is incompatible")
    if "training.pt" in payload_inventory and training_state is None:
        raise ValueError("training payload is missing its explicit state contract")
    safe["identity"] = saved_identity
    safe["payload_inventory"] = payload_inventory
    safe["model_tensors"] = model_tensors
    safe["training_state"] = training_state
    safe["run_metadata"] = run_metadata
    return safe


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint metadata key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_uncompressed_bytes(path: Path) -> int:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"checkpoint payload is not a restricted tensor archive: {path.name}")
    total = 0
    names: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("checkpoint tensor archive has too many entries")
            for item in members:
                name = item.filename
                relative = PurePosixPath(name)
                if (not name or "\x00" in name or "\\" in name or not relative.parts or relative.is_absolute()
                        or any(part in {"", ".", ".."} for part in relative.parts)):
                    raise ValueError("checkpoint archive contains an unsafe member path")
                if name in names:
                    raise ValueError("checkpoint archive contains duplicate member paths")
                names.add(name)
                mode = (item.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ValueError("checkpoint archive may not contain symlinks")
                if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError("checkpoint archive contains a non-regular member")
                if item.flag_bits & 0x1:
                    raise ValueError("encrypted checkpoint archive members are unsupported")
                if item.file_size < 0:
                    raise ValueError("checkpoint archive has a negative declared member size")
                total += item.file_size
                if total > MAX_CHECKPOINT_BYTES:
                    raise ValueError("checkpoint uncompressed tensor bytes exceed 256 MiB")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("checkpoint tensor archive inventory is invalid") from error
    return total


def preflight_checkpoint(
    path: Path, expected: NeuralIdentity, *, purpose: str,
) -> CheckpointPreflight:
    """Verify strict metadata, identity, payload hashes and archive bounds."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise ValueError("checkpoint path must be an owned regular directory")
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("checkpoint metadata.json is missing or not a regular file")
    metadata_size = metadata_path.stat().st_size
    if metadata_size <= 0 or metadata_size > MAX_METADATA_BYTES:
        raise ValueError("checkpoint metadata.json exceeds its strict JSON size limit")
    try:
        raw_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint metadata.json is not strict UTF-8 JSON") from error
    metadata = validate_checkpoint_metadata(raw_metadata, expected, purpose=purpose)
    tensor_inventory = metadata["model_tensors"]
    assert isinstance(tensor_inventory, list)
    logical_tensor_bytes = sum(
        _tensor_logical_bytes(item["name"], item["dtype"], item["shape"])
        for item in tensor_inventory
    )

    inventory = metadata["payload_inventory"]
    assert isinstance(inventory, dict)
    expected_names = {"metadata.json", *inventory}
    actual_names: set[str] = set()
    for child in checkpoint_path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("checkpoint directory may contain only regular non-symlink files")
        actual_names.add(child.name)
    if actual_names != expected_names:
        raise ValueError("checkpoint directory files disagree with the JSON payload inventory")

    stored_bytes = metadata_size
    uncompressed_bytes = metadata_size
    payload_paths: dict[str, Path] = {}
    for name, item in inventory.items():
        assert isinstance(item, dict)
        payload_path = checkpoint_path / name
        if payload_path.is_symlink() or not payload_path.is_file():
            raise ValueError(f"checkpoint payload is missing or unsafe: {name}")
        size = payload_path.stat().st_size
        if size != item["size_bytes"]:
            raise ValueError(f"checkpoint payload size mismatch: {name}")
        stored_bytes += size
        payload_paths[name] = payload_path
    if stored_bytes > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint aggregate stored bytes exceed the 256 MiB limit")

    for name, item in inventory.items():
        payload_path = payload_paths[name]
        if _sha256_file(payload_path) != item["sha256"]:
            raise ValueError(f"checkpoint payload SHA-256 mismatch: {name}")
        uncompressed = _archive_uncompressed_bytes(payload_path)
        uncompressed_bytes += uncompressed
        if max(stored_bytes, uncompressed_bytes) > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint aggregate exceeds the 256 MiB limit")

    return CheckpointPreflight(
        path=checkpoint_path,
        metadata=_freeze_json(metadata),
        purpose=purpose,
        model_path=payload_paths["model.pt"],
        training_path=payload_paths.get("training.pt"),
        stored_bytes=stored_bytes,
        uncompressed_bytes=uncompressed_bytes,
        logical_tensor_bytes=logical_tensor_bytes,
    )


def _checkpoint_identity(raw: object) -> NeuralIdentity:
    identity = _validate_identity(raw)
    return NeuralIdentity(
        model_id=identity["model_id"],
        features_id=identity["features_id"],
        action_id=identity["action_id"],
        objective_id=identity["objective_id"],
        accepted_raw_mode_mappings=identity["accepted_raw_mode_mappings"],
        config_hashes=identity["config_hashes"],
    )


def _torch_tensor_info(name: str, tensor: object, torch: Any) -> tuple[dict[str, object], int]:
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 4096:
        raise ValueError("checkpoint tensor names must be non-empty bounded text")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"checkpoint tensor {name} is not a Torch tensor")
    if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise ValueError(f"checkpoint tensor {name} must be a dense supported tensor")
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _TENSOR_DTYPE_BYTES:
        raise ValueError(f"checkpoint tensor {name} has unsupported dtype {dtype}")
    shape = [int(size) for size in tensor.shape]
    if len(shape) > MAX_TENSOR_RANK:
        raise ValueError(f"checkpoint tensor {name} rank exceeds {MAX_TENSOR_RANK}")
    if any(size < 0 or size > MAX_TENSOR_DIMENSION for size in shape):
        raise ValueError(f"checkpoint tensor {name} has an unsupported dimension")
    logical_bytes = _tensor_logical_bytes(name, dtype, shape)
    if logical_bytes > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint tensor {name} exceeds the 256 MiB logical tensor limit")
    if tensor.numel() != math.prod(shape):
        raise ValueError(f"checkpoint tensor {name} has inconsistent logical geometry")
    if tensor.element_size() != _TENSOR_DTYPE_BYTES[dtype]:
        raise ValueError(f"checkpoint tensor {name} has inconsistent dtype width")
    return {"name": name, "dtype": dtype, "shape": shape}, logical_bytes


def _require_finite_tensor(name: str, tensor: Any, torch: Any) -> None:
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"checkpoint tensor {name} cannot be checked for finite values") from error
    if not finite:
        raise ValueError(f"checkpoint tensor {name} must contain only finite values")


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    raw = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _snapshot_tensor(name: str, tensor: Any, torch: Any) -> Any:
    cpu_tensor = tensor.detach().to(device="cpu")
    _require_finite_tensor(name, cpu_tensor, torch)
    return cpu_tensor.contiguous().clone()


def _scan_training_value(
    value: object,
    torch: Any,
    *,
    name: str,
    aggregate_bytes: list[int],
    node_count: list[int],
    scalar_bytes: list[int],
    tensor_leaves: list[tuple[str, Any]],
    depth: int = 0,
) -> object:
    node_count[0] += 1
    if node_count[0] > 100_000 or depth > 32:
        raise ValueError("training checkpoint state exceeds its structure limit")
    if isinstance(value, torch.Tensor):
        _, logical_bytes = _torch_tensor_info(name, value, torch)
        aggregate_bytes[0] += logical_bytes
        if aggregate_bytes[0] > MAX_CHECKPOINT_BYTES:
            raise ValueError("aggregate logical checkpoint tensors exceed 256 MiB")
        tensor_leaves.append((name, value))
        return value
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -(2**64 - 1) <= value <= 2**64 - 1:
            raise ValueError("training checkpoint integer exceeds the supported 64-bit range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("training checkpoint numbers must be finite")
        return value
    if isinstance(value, str):
        size = len(value.encode("utf-8"))
        if size > 1024 * 1024:
            raise ValueError("training checkpoint text exceeds its size limit")
        scalar_bytes[0] += size
        if scalar_bytes[0] > 16 * 1024 * 1024:
            raise ValueError("training checkpoint scalar data exceeds its size limit")
        return value
    if isinstance(value, Mapping):
        result: dict[str | int, object] = {}
        for key, child in value.items():
            if type(key) not in (str, int):
                raise TypeError("training checkpoint mapping keys must be strings or integers")
            if type(key) is int and not -(2**64 - 1) <= key <= 2**64 - 1:
                raise ValueError("training checkpoint mapping key exceeds the supported integer range")
            if type(key) is str:
                key_size = len(key.encode("utf-8"))
                if key_size > 4096:
                    raise ValueError("training checkpoint mapping key exceeds its size limit")
                scalar_bytes[0] += key_size
                if scalar_bytes[0] > 16 * 1024 * 1024:
                    raise ValueError("training checkpoint scalar data exceeds its size limit")
            if key in result:
                raise ValueError("training checkpoint contains a duplicate mapping key")
            result[key] = _scan_training_value(
                child,
                torch,
                name=f"{name}.{key}",
                aggregate_bytes=aggregate_bytes,
                node_count=node_count,
                scalar_bytes=scalar_bytes,
                tensor_leaves=tensor_leaves,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (tuple, list)):
        items = [
            _scan_training_value(
                child,
                torch,
                name=f"{name}[{index}]",
                aggregate_bytes=aggregate_bytes,
                node_count=node_count,
                scalar_bytes=scalar_bytes,
                tensor_leaves=tensor_leaves,
                depth=depth + 1,
            )
            for index, child in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    raise TypeError(f"unsupported training checkpoint value type: {type(value).__name__}")


def _snapshot_training_value(value: object, torch: Any, *, name: str) -> object:
    if isinstance(value, torch.Tensor):
        return _snapshot_tensor(name, value, torch)
    if isinstance(value, dict):
        return {
            key: _snapshot_training_value(child, torch, name=f"{name}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _snapshot_training_value(child, torch, name=f"{name}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _snapshot_training_value(child, torch, name=f"{name}[{index}]")
            for index, child in enumerate(value)
        ]
    return value


def _write_torch_payload(path: Path, value: object, torch: Any) -> None:
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if os.name != "posix":
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unsupported")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "renameat2 is required for safe checkpoint publication") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _remove_owned_temporary_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity):
        return
    shutil.rmtree(path)


def save_checkpoint(
    path: Path,
    *,
    policy: object,
    optimizer: object,
    rng_state: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Atomically publish a bounded, named tensor checkpoint without mutating inputs."""
    destination = Path(path)
    if destination.name in {"", ".", ".."}:
        raise ValueError("checkpoint destination must name a directory")
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, "checkpoint destination already exists", str(destination))
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("checkpoint parent must be an existing regular directory")
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    base = _strict_json(metadata)
    input_keys = {"identity", "run_metadata", "training_state", "parent_checkpoint"}
    if not isinstance(base, dict) or set(base) != input_keys:
        raise ValueError("checkpoint save metadata has an invalid field set")
    identity = _checkpoint_identity(base["identity"])
    training_state = _validate_training_state(base["training_state"], has_training_payload=True)
    if training_state is None:
        raise ValueError("training checkpoint save requires explicit training metadata")
    if not isinstance(rng_state, Mapping):
        raise TypeError("rng_state must be an explicit mapping")
    expected_rng_domains = set(training_state["rng_domains"])
    if set(rng_state) != expected_rng_domains:
        raise ValueError("rng_state domains disagree with training metadata")

    state_dict_method = getattr(policy, "state_dict", None)
    if not callable(state_dict_method):
        raise TypeError("policy must expose a named state_dict")
    model_state = state_dict_method()
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("policy state_dict must be a non-empty named tensor mapping")
    optimizer_state_method = getattr(optimizer, "state_dict", None)
    if not callable(optimizer_state_method):
        raise TypeError("optimizer must expose a state_dict")
    optimizer_state = optimizer_state_method()
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("optimizer state_dict must be a mapping")

    import torch

    model_sources: dict[str, Any] = {}
    model_inventory: list[dict[str, object]] = []
    aggregate_logical_bytes = 0
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not name or name in model_sources:
            raise ValueError("policy state_dict tensor names must be unique non-empty strings")
        descriptor, logical_bytes = _torch_tensor_info(name, tensor, torch)
        aggregate_logical_bytes += logical_bytes
        if aggregate_logical_bytes > MAX_CHECKPOINT_BYTES:
            raise ValueError("aggregate logical model tensor bytes exceed 256 MiB")
        model_sources[name] = tensor
        model_inventory.append(descriptor)

    tree_budget = [aggregate_logical_bytes]
    tree_nodes = [0]
    scalar_bytes = [0]
    training_tensor_leaves: list[tuple[str, Any]] = []
    checked_optimizer_state = _scan_training_value(
        optimizer_state,
        torch,
        name="optimizer_state",
        aggregate_bytes=tree_budget,
        node_count=tree_nodes,
        scalar_bytes=scalar_bytes,
        tensor_leaves=training_tensor_leaves,
    )
    checked_rng_state = _scan_training_value(
        rng_state,
        torch,
        name="rng_state",
        aggregate_bytes=tree_budget,
        node_count=tree_nodes,
        scalar_bytes=scalar_bytes,
        tensor_leaves=training_tensor_leaves,
    )
    if not isinstance(checked_optimizer_state, dict) or not isinstance(checked_rng_state, dict):
        raise TypeError("optimizer and RNG checkpoint state must be mappings")

    model_tensors: dict[str, Any] = {
        name: _snapshot_tensor(name, tensor, torch)
        for name, tensor in model_sources.items()
    }
    safe_training_state = {
        "optimizer_state": _snapshot_training_value(checked_optimizer_state, torch, name="optimizer_state"),
        "rng_state": _snapshot_training_value(checked_rng_state, torch, name="rng_state"),
    }
    for item in model_inventory:
        tensor = model_tensors[item["name"]]
        item["sha256"] = _tensor_sha256(tensor, torch)

    temporary = Path(tempfile.mkdtemp(prefix=".task6-checkpoint-", dir=parent))
    temporary_stat = temporary.lstat()
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    try:
        temporary.chmod(0o700)
        _write_torch_payload(temporary / "model.pt", model_tensors, torch)
        _write_torch_payload(temporary / "training.pt", safe_training_state, torch)
        payload_inventory = {
            name: {
                "size_bytes": (temporary / name).stat().st_size,
                "sha256": _sha256_file(temporary / name),
            }
            for name in ("model.pt", "training.pt")
        }
        full_metadata = {
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "identity": base["identity"],
            "payload_inventory": payload_inventory,
            "model_tensors": model_inventory,
            "training_state": training_state,
            "run_metadata": base["run_metadata"],
            "parent_checkpoint": base["parent_checkpoint"],
        }
        validated_metadata = validate_checkpoint_metadata(
            full_metadata, identity, purpose="resume"
        )
        metadata_bytes = json.dumps(
            validated_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(metadata_bytes) > MAX_METADATA_BYTES:
            raise ValueError("checkpoint metadata.json exceeds its 1 MiB limit")
        with (temporary / "metadata.json").open("xb") as stream:
            stream.write(metadata_bytes)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / "metadata.json").chmod(0o600)
        preflight_checkpoint(temporary, identity, purpose="resume")
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(parent)
    except BaseException as error:
        try:
            _remove_owned_temporary_directory(temporary, temporary_identity)
        except Exception as cleanup_error:
            error.add_note(f"checkpoint temporary cleanup also failed: {cleanup_error!r}")
        raise


def load_checkpoint(
    path: Path,
    expected: NeuralIdentity,
    *,
    purpose: str,
    device: str,
) -> LoadedCheckpoint:
    """Preflight first, then load restricted tensor state on CPU without applying it."""
    if device != "cpu":
        raise ValueError("Task 6 checkpoint loading currently supports only device='cpu'")
    preflight = preflight_checkpoint(path, expected, purpose=purpose)

    import torch

    model_payload = torch.load(
        preflight.model_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(model_payload, Mapping) or not model_payload:
        raise ValueError("model.pt must contain a non-empty named tensor mapping")
    declared = {
        item["name"]: item
        for item in preflight.metadata["model_tensors"]
    }
    if set(model_payload) != set(declared):
        raise ValueError("loaded model tensor names disagree with the preflight inventory")
    model_tensors: dict[str, Any] = {}
    model_leaves: list[tuple[str, Any]] = []
    aggregate_logical_bytes = 0
    for name, tensor in model_payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError("loaded model tensor names must be non-empty strings")
        descriptor, logical_bytes = _torch_tensor_info(name, tensor, torch)
        saved_descriptor = declared[name]
        if descriptor["dtype"] != saved_descriptor["dtype"] or descriptor["shape"] != list(saved_descriptor["shape"]):
            raise ValueError(f"loaded model tensor {name} dtype or shape disagrees with metadata")
        if tensor.device.type != "cpu":
            raise ValueError(f"loaded model tensor {name} was not mapped to CPU")
        aggregate_logical_bytes += logical_bytes
        if aggregate_logical_bytes > MAX_CHECKPOINT_BYTES:
            raise ValueError("aggregate logical model tensor bytes exceed 256 MiB")
        model_tensors[name] = tensor
        model_leaves.append((name, tensor))
    if aggregate_logical_bytes != preflight.logical_tensor_bytes:
        raise ValueError("loaded model tensor geometry disagrees with the metadata inventory")

    training_payload: dict[str, object] | None = None
    training_leaves: list[tuple[str, Any]] = []
    if purpose == "resume":
        if preflight.training_path is None:
            raise CheckpointCompatibilityError("resume requires optimizer and RNG payloads")
        raw_training_payload = torch.load(
            preflight.training_path,
            map_location="cpu",
            weights_only=True,
        )
        if (not isinstance(raw_training_payload, Mapping)
                or set(raw_training_payload) != {"optimizer_state", "rng_state"}):
            raise ValueError("training.pt has an unsupported field set")
        raw_optimizer_state = raw_training_payload["optimizer_state"]
        raw_rng_state = raw_training_payload["rng_state"]
        if not isinstance(raw_optimizer_state, Mapping) or not isinstance(raw_rng_state, Mapping):
            raise ValueError("training.pt optimizer and RNG state must be mappings")
        training_state = preflight.metadata["training_state"]
        if not isinstance(training_state, Mapping):
            raise CheckpointCompatibilityError("resume requires explicit training metadata")
        if set(raw_rng_state) != set(training_state["rng_domains"]):
            raise ValueError("loaded RNG domains disagree with training metadata")
        aggregate_budget = [aggregate_logical_bytes]
        node_count = [0]
        scalar_size = [0]
        checked_optimizer_state = _scan_training_value(
            raw_optimizer_state,
            torch,
            name="optimizer_state",
            aggregate_bytes=aggregate_budget,
            node_count=node_count,
            scalar_bytes=scalar_size,
            tensor_leaves=training_leaves,
        )
        checked_rng_state = _scan_training_value(
            raw_rng_state,
            torch,
            name="rng_state",
            aggregate_bytes=aggregate_budget,
            node_count=node_count,
            scalar_bytes=scalar_size,
            tensor_leaves=training_leaves,
        )
        if not isinstance(checked_optimizer_state, dict) or not isinstance(checked_rng_state, dict):
            raise ValueError("training.pt state mappings are unsupported")
        training_payload = {
            "optimizer_state": checked_optimizer_state,
            "rng_state": checked_rng_state,
        }

    for name, tensor in model_leaves + training_leaves:
        if tensor.device.type != "cpu":
            raise ValueError(f"loaded checkpoint tensor {name} was not mapped to CPU")
        _require_finite_tensor(name, tensor, torch)
    for name, tensor in model_leaves:
        if _tensor_sha256(tensor, torch) != declared[name]["sha256"]:
            raise ValueError(f"loaded model tensor {name} SHA-256 disagrees with metadata")

    training_state = preflight.metadata["training_state"]
    saved_inflight = 0
    saved_step = 0
    if isinstance(training_state, Mapping):
        saved_inflight = training_state["saved_inflight_record_count"]
        saved_step = training_state["optimizer_step"]
    recovery = {
        "purpose": purpose,
        "parent_checkpoint": str(Path(path)),
        "dropped_inflight_record_count": saved_inflight if purpose == "resume" else 0,
    }
    loaded_metadata = dict(preflight.metadata)
    loaded_metadata["recovery"] = recovery
    frozen_metadata = _freeze_json(loaded_metadata)
    assert isinstance(frozen_metadata, Mapping)
    if purpose == "resume":
        assert training_payload is not None
        optimizer_state = training_payload["optimizer_state"]
        rng_state = training_payload["rng_state"]
        assert isinstance(optimizer_state, Mapping) and isinstance(rng_state, Mapping)
        optimizer_step = saved_step
    else:
        optimizer_state = None
        rng_state = None
        optimizer_step = 0
    return LoadedCheckpoint(
        metadata=frozen_metadata,
        model_tensors=model_tensors,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        optimizer_step=optimizer_step,
        inflight_record_count=0,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "MAX_CHECKPOINT_BYTES",
    "CheckpointCompatibilityError",
    "CheckpointPreflight",
    "LoadedCheckpoint",
    "identity_payload",
    "load_checkpoint",
    "preflight_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_metadata",
]
