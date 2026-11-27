"""Sparse decision shards for search distillation (numpy; runtime container only).

A shard is a directory of ``.npy`` files written with ``np.save(allow_pickle=False)`` plus a
deterministic ``manifest.json`` (file sha256s, counts and provenance; no timestamps).

Observation files hold one compact combat frame per decision, stored sparsely: a row group keeps
only its occupied rows (CSR style: ``<group>.offsets``, ``.rows``, ``.values``, ``.known``), the
groups in ``optional_known_empty_groups`` also keep their full ``_known`` array
(``<group>.known_full``), single-row groups and ``action_mask`` are dense. Every label lives in a
separate ``labels.*`` file, so a trainer can build batches from observation files alone.

Only the host-importable modules must stay free of numpy: nothing imported at startup may import
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from spire_learn.contracts import ContractError
from spire_learn.neural.contracts import BufferArray, CombatFrame
from spire_learn.neural.schema import CombatSchema, TensorSpec, validate_frame

SHARD_SCHEMA = "spire-learn-distill-shard-v1"
MANIFEST = "manifest.json"
LABEL_PREFIX = "labels."
HOLDOUT_FRACTION = 0.05
_HOLDOUT_DOMAIN = b"spire-learn-distill-validation-v1:"

# Enums stored as int8 in the label files (the index in the tuple is the stored value).
ROOT_KINDS = ("searched", "single_legal", "refused")
OUTCOMES = ("win", "death", "escaped", "none")

_LABEL_FILES = (
    "legal_offsets", "legal", "values", "samples", "choice", "played", "root", "outcome",
    "hp_kept", "realised_v2", "value_ok", "policy_ok", "research_seed", "case", "run_hash", "turn",
)


class ShardError(ContractError):
    """A shard is malformed, inconsistent with its schema or fails verification."""


def run_hash(run_id: str) -> int:
    """The big-endian first 8 bytes of ``sha256(run_id)``; stored in label files only."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big")


def validation_runs(train_run_ids: Iterable[str]) -> frozenset[str]:
    """The validation holdout (delta 8): the first ``round(0.05 * N)`` runs by hash rank."""
    ids = sorted(set(train_run_ids))
    if any(not isinstance(run_id, str) or not run_id for run_id in ids):
        raise ValueError("run ids must be non-empty strings")
    ranked = sorted(
        ids,
        key=lambda run_id: (
            int.from_bytes(hashlib.sha256(_HOLDOUT_DOMAIN + run_id.encode()).digest()[:8], "big"),
            run_id,
        ),
    )
    return frozenset(ranked[:round(HOLDOUT_FRACTION * len(ids))])


# --- observation encoding -------------------------------------------------------------------

def _layout(schema: CombatSchema) -> tuple[list[TensorSpec], list[TensorSpec]]:
    """Value tensors split into (row groups, single-row groups); anything else must be a mask."""
    row_groups, single_groups = [], []
    companions = {"action_mask"}
    for spec in schema.tensors.values():
        if spec.columns:
            (row_groups if spec.occupied_key is not None else single_groups).append(spec)
            companions.add(spec.known_key)
            if spec.occupied_key is not None:
                companions.add(spec.occupied_key)
    if "action_mask" not in schema.tensors or "actions" not in schema.tensors:
        raise ShardError("schema has no actions group or action_mask")
    stray = set(schema.tensors) - {spec.key for spec in row_groups + single_groups} - companions
    if stray:
        raise ShardError(f"schema has tensors a shard cannot store: {sorted(stray)}")
    return row_groups, single_groups


def _view_array(buffer: BufferArray) -> np.ndarray:
    return np.frombuffer(buffer.data, dtype=np.dtype(buffer.dtype)).reshape(buffer.shape)


def _row_dtype(spec: TensorSpec) -> np.dtype:
    return np.dtype(np.int16 if spec.shape[0] <= np.iinfo(np.int16).max else np.int32)


def encode_decision(frame: CombatFrame, schema: CombatSchema) -> dict[str, np.ndarray]:
    """One validated frame as sparse per-decision arrays, keyed by observation file stem."""
    validate_frame(frame, schema)
    row_groups, single_groups = _layout(schema)
    view = schema.view
    arrays: dict[str, np.ndarray] = {}
    for spec in row_groups:
        values = _view_array(frame.tensors[spec.key])
        known = _view_array(frame.tensors[spec.known_key])
        occupied = _view_array(frame.tensors[spec.occupied_key])
        rows = np.flatnonzero(occupied).astype(_row_dtype(spec))
        arrays[f"{spec.key}.rows"] = rows
        arrays[f"{spec.key}.values"] = np.ascontiguousarray(values[rows])
        arrays[f"{spec.key}.known"] = np.ascontiguousarray(known[rows])
        if spec.key in view.optional_known_empty_groups:
            arrays[f"{spec.key}.known_full"] = np.ascontiguousarray(known)
    for spec in single_groups:
        arrays[f"{spec.key}.values"] = np.ascontiguousarray(_view_array(frame.tensors[spec.key]))
        arrays[f"{spec.key}.known"] = np.ascontiguousarray(_view_array(frame.tensors[spec.known_key]))
    arrays["action_mask"] = np.ascontiguousarray(_view_array(frame.tensors["action_mask"]))
    return arrays


def decode_decision(arrays: Mapping[str, np.ndarray], schema: CombatSchema) -> CombatFrame:
    """The exact compact frame for one decision's sparse arrays (inverse of ``encode_decision``)."""
    row_groups, single_groups = _layout(schema)
    view = schema.view
    tensors: dict[str, BufferArray] = {}

    def put(key: str, spec: TensorSpec, array: np.ndarray) -> None:
        if array.shape != spec.shape or array.dtype != np.dtype(spec.dtype):
            raise ShardError(f"decoded {key} does not match the schema dtype/shape")
        tensors[key] = BufferArray(spec.dtype, spec.shape, np.ascontiguousarray(array).tobytes())

    for spec in row_groups:
        capacity, columns = spec.shape
        rows = np.asarray(arrays[f"{spec.key}.rows"])
        if rows.size and (rows.min() < 0 or rows.max() >= capacity or np.any(np.diff(rows) <= 0)):
            raise ShardError(f"{spec.key} rows are not strictly increasing within capacity")
        dtype = np.dtype(spec.dtype)
        if spec.key in view.required_known_empty_groups:
            known = np.ones(spec.shape, dtype=np.int8)
        elif spec.key in view.optional_known_empty_groups:
            known = np.array(arrays[f"{spec.key}.known_full"], dtype=np.int8, copy=True)
        else:
            known = np.zeros(spec.shape, dtype=np.int8)
        fill = 0 if spec.key in view.required_known_empty_groups | view.optional_known_empty_groups \
            else spec.padding
        values = np.full(spec.shape, fill, dtype=dtype)
        values[rows] = arrays[f"{spec.key}.values"]
        known[rows] = arrays[f"{spec.key}.known"]
        occupied = np.zeros(capacity, dtype=np.int8)
        occupied[rows] = 1
        put(spec.key, spec, values)
        put(spec.known_key, schema.tensors[spec.known_key], known)
        put(spec.occupied_key, schema.tensors[spec.occupied_key], occupied)
    for spec in single_groups:
        put(spec.key, spec, np.asarray(arrays[f"{spec.key}.values"]))
        put(spec.known_key, schema.tensors[spec.known_key], np.asarray(arrays[f"{spec.key}.known"]))
    put("action_mask", schema.tensors["action_mask"], np.asarray(arrays["action_mask"]))
    return CombatFrame(tensors, schema.schema_fingerprint)


# --- labels ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionLabels:
    """Search-teacher labels of one decision (never a model input)."""

    legal: Sequence[int]
    values: Sequence[float] | None  # one per legal action; None when the root was not searched
    samples: int
    choice: int
    played: int
    root: str
    outcome: str
    hp_kept: float
    realised_v2: float
    value_ok: bool
    policy_ok: bool
    research_seed: int
    case: tuple[int, int]  # (shard index, local case index)
    run_id: str
    turn: int


@dataclass(frozen=True)
class StoredLabels:
    legal: np.ndarray
    values: np.ndarray
    samples: int
    choice: int
    played: int
    root: str
    outcome: str
    hp_kept: float
    realised_v2: float
    value_ok: bool
    policy_ok: bool
    research_seed: int
    case: tuple[int, int]
    run_hash: int
    turn: int


def _label_record(labels: DecisionLabels, frame_mask: np.ndarray) -> dict[str, object]:
    legal = [int(action) for action in labels.legal]
    if legal != [int(a) for a in np.flatnonzero(frame_mask)]:
        raise ShardError("labels.legal must equal the frame's action_mask legal set")
    if labels.values is None:
        values = [math.nan] * len(legal)
    else:
        values = [float(v) for v in labels.values]
        if len(values) != len(legal):
            raise ShardError("labels.values needs one value per legal action")
    if labels.choice not in legal or labels.played not in legal:
        raise ShardError("choice and played must be legal actions")
    if labels.root not in ROOT_KINDS or labels.outcome not in OUTCOMES:
        raise ShardError("unknown root kind or outcome")
    for name in ("hp_kept", "realised_v2"):
        value = float(getattr(labels, name))
        if labels.value_ok != math.isfinite(value):
            raise ShardError(f"{name} must be finite exactly when value_ok is true")
    case = tuple(int(part) for part in labels.case)
    if len(case) != 2 or min(case) < 0:
        raise ShardError("case must be (shard index, local index)")
    if not 0 <= int(labels.research_seed) < 2 ** 64:
        raise ShardError("research_seed must fit uint64")
    return {
        "legal": legal, "values": values, "samples": int(labels.samples),
        "choice": int(labels.choice), "played": int(labels.played),
        "root": ROOT_KINDS.index(labels.root), "outcome": OUTCOMES.index(labels.outcome),
        "hp_kept": float(labels.hp_kept), "realised_v2": float(labels.realised_v2),
        "value_ok": bool(labels.value_ok), "policy_ok": bool(labels.policy_ok),
        "research_seed": int(labels.research_seed), "case": case,
        "run_hash": run_hash(labels.run_id), "turn": int(labels.turn),
    }


# --- writer ----------------------------------------------------------------------------------

def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_identity(schema: CombatSchema) -> dict[str, str]:
    return {
        "schema_id": schema.schema_id, "schema_fingerprint": schema.schema_fingerprint,
        "descriptor_digest": schema.descriptor_digest, "gym_revision": schema.gym_revision,
    }


class ShardWriter:
    """Accumulate decisions and write one shard directory on ``close``."""

    def __init__(self, path: Path | str, schema: CombatSchema, provenance: Mapping[str, object]):
        self._path = Path(path)
        self._schema = schema
        self._provenance = json.loads(json.dumps(provenance, allow_nan=False))
        _layout(schema)
        self._pieces: dict[str, list[np.ndarray]] = {}
        self._offsets: dict[str, list[int]] = {}
        self._labels: dict[str, list] = {name: [] for name in _LABEL_FILES}
        self._legal_offsets = [0]
        self._rows = 0
        self._closed = False

    def add(self, frame: CombatFrame, labels: DecisionLabels) -> None:
        if self._closed:
            raise ShardError("shard writer is closed")
        arrays = encode_decision(frame, self._schema)
        record = _label_record(labels, arrays["action_mask"])
        for name, array in arrays.items():
            self._pieces.setdefault(name, []).append(array)
            if name.endswith(".rows"):
                offsets = self._offsets.setdefault(name[:-len(".rows")], [0])
                offsets.append(offsets[-1] + len(array))
        out = self._labels
        out["legal"].extend(record["legal"])
        out["values"].extend(record["values"])
        self._legal_offsets.append(len(out["legal"]))
        for name in ("samples", "choice", "played", "root", "outcome", "hp_kept", "realised_v2",
                     "value_ok", "policy_ok", "research_seed", "case", "run_hash", "turn"):
            out[name].append(record[name])
        self._rows += 1

    def _arrays(self) -> dict[str, np.ndarray]:
        files: dict[str, np.ndarray] = {}
        for name, pieces in self._pieces.items():
            if name.endswith(".rows"):
                group = name[:-len(".rows")]
                files[f"{group}.offsets"] = np.asarray(self._offsets[group], dtype=np.int64)
                files[name] = np.concatenate(pieces)
            elif name.endswith(".known_full") or name == "action_mask" or self._is_single(name):
                files[name] = np.stack(pieces)
            else:
                files[name] = np.concatenate(pieces)
        out = self._labels
        files.update({
            "labels.legal_offsets": np.asarray(self._legal_offsets, dtype=np.int64),
            "labels.legal": np.asarray(out["legal"], dtype=np.int32),
            "labels.values": np.asarray(out["values"], dtype=np.float64),
            "labels.samples": np.asarray(out["samples"], dtype=np.int32),
            "labels.choice": np.asarray(out["choice"], dtype=np.int32),
            "labels.played": np.asarray(out["played"], dtype=np.int32),
            "labels.root": np.asarray(out["root"], dtype=np.int8),
            "labels.outcome": np.asarray(out["outcome"], dtype=np.int8),
            "labels.hp_kept": np.asarray(out["hp_kept"], dtype=np.float64),
            "labels.realised_v2": np.asarray(out["realised_v2"], dtype=np.float64),
            "labels.value_ok": np.asarray(out["value_ok"], dtype=np.bool_),
            "labels.policy_ok": np.asarray(out["policy_ok"], dtype=np.bool_),
            "labels.research_seed": np.asarray(out["research_seed"], dtype=np.uint64),
            "labels.case": np.asarray(out["case"], dtype=np.int64).reshape(-1, 2),
            "labels.run_hash": np.asarray(out["run_hash"], dtype=np.uint64),
            "labels.turn": np.asarray(out["turn"], dtype=np.int32),
        })
        return files

    def _is_single(self, name: str) -> bool:
        stem = name.rsplit(".", 1)[0]
        spec = self._schema.tensors.get(stem)
        return spec is not None and spec.occupied_key is None

    def close(self) -> dict[str, object]:
        if self._closed:
            raise ShardError("shard writer is closed")
        self._closed = True
        self._path.mkdir(parents=True, exist_ok=False)
        files: dict[str, dict[str, object]] = {}
        if self._rows:
            for name, array in sorted(self._arrays().items()):
                target = self._path / f"{name}.npy"
                np.save(target, array, allow_pickle=False)
                files[name] = {
                    "sha256": _file_sha256(target), "dtype": array.dtype.str, "shape": list(array.shape),
                }
        manifest = {
            "schema": SHARD_SCHEMA,
            "observation": _schema_identity(self._schema),
            "rows": self._rows,
            "files": files,
            "provenance": self._provenance,
        }
        (self._path / MANIFEST).write_bytes(_json_bytes(manifest))
        return manifest


# --- reader ----------------------------------------------------------------------------------

class ShardReader:
    """Read-only, memory-mapped access to one shard."""

    def __init__(self, path: Path | str, schema: CombatSchema | None = None):
        self.path = Path(path)
        try:
            self.manifest = json.loads((self.path / MANIFEST).read_text())
        except (OSError, ValueError) as error:
            raise ShardError(f"unreadable shard manifest: {error}") from None
        if self.manifest.get("schema") != SHARD_SCHEMA:
            raise ShardError("shard schema is not accepted")
        self.rows: int = self.manifest["rows"]
        self.schema = schema
        if schema is not None and self.manifest["observation"] != _schema_identity(schema):
            raise ShardError("shard observation schema does not match the given CombatSchema")
        self._arrays = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.manifest["files"]
        }

    def __len__(self) -> int:
        return self.rows

    @property
    def observation_files(self) -> tuple[str, ...]:
        return tuple(name for name in self.manifest["files"] if not name.startswith(LABEL_PREFIX))

    @property
    def label_files(self) -> tuple[str, ...]:
        return tuple(name for name in self.manifest["files"] if name.startswith(LABEL_PREFIX))

    def array(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def verify(self) -> None:
        """Hashes, shapes and CSR structure; raises ``ShardError`` on any mismatch."""
        listed = self.manifest["files"]
        on_disk = {p.name for p in self.path.iterdir()} - {MANIFEST}
        if on_disk != {f"{name}.npy" for name in listed}:
            raise ShardError("shard directory files differ from its manifest")
        for name, entry in listed.items():
            if _file_sha256(self.path / f"{name}.npy") != entry["sha256"]:
                raise ShardError(f"shard file {name} fails its sha256")
            array = self._arrays[name]
            if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
                raise ShardError(f"shard file {name} dtype/shape differs from its manifest")
        if self.rows == 0:
            if listed:
                raise ShardError("a zero-row shard lists files")
            return
        for name in listed:
            if name.endswith(".offsets"):
                offsets = self._arrays[name]
                stem = name[:-len(".offsets")]
                total = len(self._arrays[f"{stem}.rows"])
                if (len(offsets) != self.rows + 1 or offsets[0] != 0 or offsets[-1] != total
                        or np.any(np.diff(offsets) < 0)):
                    raise ShardError(f"{name} is not a valid offsets array")
        legal_offsets = self._arrays["labels.legal_offsets"]
        if (len(legal_offsets) != self.rows + 1 or legal_offsets[-1] != len(self._arrays["labels.legal"])
                or len(self._arrays["labels.values"]) != len(self._arrays["labels.legal"])):
            raise ShardError("labels.legal_offsets is inconsistent")
        for name in self.label_files:
            if name in ("labels.legal", "labels.values", "labels.legal_offsets"):
                continue
            if len(self._arrays[name]) != self.rows:
                raise ShardError(f"{name} does not have one entry per decision")

    def _check_index(self, index: int) -> int:
        if not 0 <= index < self.rows:
            raise IndexError(f"decision {index} outside shard of {self.rows} rows")
        return index

    def decision_arrays(self, index: int) -> dict[str, np.ndarray]:
        """The sparse per-decision arrays (same keys as ``encode_decision``)."""
        index = self._check_index(index)
        arrays: dict[str, np.ndarray] = {}
        for name in self.observation_files:
            if name.endswith(".offsets"):
                continue
            stem, _, kind = name.rpartition(".")
            offsets = self._arrays.get(f"{stem}.offsets") if kind in ("rows", "values", "known") else None
            if name == "action_mask" or kind == "known_full" or offsets is None:
                arrays[name] = self._arrays[name][index]
            else:
                low, high = int(offsets[index]), int(offsets[index + 1])
                arrays[name] = self._arrays[name][low:high]
        return arrays

    def frame(self, index: int, schema: CombatSchema | None = None) -> CombatFrame:
        schema = schema or self.schema
        if schema is None:
            raise ShardError("reading a frame needs the CombatSchema")
        return expand_decision(self, index, schema)

    def labels(self, index: int) -> StoredLabels:
        index = self._check_index(index)
        a = self._arrays
        low, high = int(a["labels.legal_offsets"][index]), int(a["labels.legal_offsets"][index + 1])
        return StoredLabels(
            legal=a["labels.legal"][low:high], values=a["labels.values"][low:high],
            samples=int(a["labels.samples"][index]), choice=int(a["labels.choice"][index]),
            played=int(a["labels.played"][index]), root=ROOT_KINDS[int(a["labels.root"][index])],
            outcome=OUTCOMES[int(a["labels.outcome"][index])],
            hp_kept=float(a["labels.hp_kept"][index]), realised_v2=float(a["labels.realised_v2"][index]),
            value_ok=bool(a["labels.value_ok"][index]), policy_ok=bool(a["labels.policy_ok"][index]),
            research_seed=int(a["labels.research_seed"][index]),
            case=(int(a["labels.case"][index][0]), int(a["labels.case"][index][1])),
            run_hash=int(a["labels.run_hash"][index]), turn=int(a["labels.turn"][index]),
        )


def expand_decision(shard: ShardReader, index: int, schema: CombatSchema) -> CombatFrame:
    """The compact observation the Gym returned for decision ``index``, byte for byte."""
    if shard.manifest["observation"] != _schema_identity(schema):
        raise ShardError("shard observation schema does not match the given CombatSchema")
    return decode_decision(shard.decision_arrays(shard._check_index(index)), schema)


__all__ = [
    "SHARD_SCHEMA", "ROOT_KINDS", "OUTCOMES", "ShardError", "ShardWriter", "ShardReader",
    "DecisionLabels", "StoredLabels", "encode_decision", "decode_decision", "expand_decision",
    "run_hash", "validation_runs",
]
