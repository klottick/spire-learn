"""Sparse decision shards for search distillation (numpy; runtime container only).

A shard is a directory of ``.npy`` files written with ``np.save(allow_pickle=False)`` plus a
deterministic ``manifest.json`` (file sha256s, counts and provenance; no timestamps).

Observation files hold one compact combat frame per decision, stored sparsely: a row group keeps
only its occupied rows (CSR style: ``<group>.offsets``, ``.rows``, ``.values``, ``.known``), the
groups in ``optional_known_empty_groups`` also keep their full ``_known`` array
(``<group>.known_full``), single-row groups and ``action_mask`` are dense. Every label lives in a
separate ``labels.*`` file, so a trainer can build batches from observation files alone.

Label conventions: ``played`` is always a legal action. ``choice`` is the teacher's pick and is
legal too, except that a refused root (the search refused it; no teacher choice exists) stores
``CHOICE_NONE`` (-1) with ``policy_ok`` false. Unsearched roots (single legal action, refused)
store NaN ``values``; ``value_ok`` false stores NaN ``hp_kept`` and ``realised_v2``; every NaN is the
canonical quiet NaN so shard bytes are deterministic.

Only the host-importable modules must stay free of numpy: nothing imported at startup may import
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
CHOICE_NONE = -1
_INT32_LIMIT = 2 ** 31
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
        key=lambda run_id: int.from_bytes(hashlib.sha256(_HOLDOUT_DOMAIN + run_id.encode()).digest(), "big"),
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


def _canonical(value: float) -> float:
    return math.nan if math.isnan(value) else value


def _label_record(labels: DecisionLabels, frame_mask: np.ndarray) -> dict[str, object]:
    legal = [int(action) for action in labels.legal]
    if legal != [int(a) for a in np.flatnonzero(frame_mask)]:
        raise ShardError("labels.legal must equal the frame's action_mask legal set")
    if labels.root not in ROOT_KINDS or labels.outcome not in OUTCOMES:
        raise ShardError("unknown root kind or outcome")
    searched = labels.root == "searched"
    if labels.values is None:
        values = [math.nan] * len(legal)
    else:
        values = [_canonical(float(v)) for v in labels.values]
        if len(values) != len(legal):
            raise ShardError("labels.values needs one value per legal action")
    if searched and not all(math.isfinite(v) for v in values):
        raise ShardError("a searched root needs a finite value for every legal action")
    if not searched and not all(math.isnan(v) for v in values):
        raise ShardError("an unsearched root must store NaN values")
    if labels.policy_ok and not searched:
        raise ShardError("policy_ok requires a searched root")
    played, choice = int(labels.played), int(labels.choice)
    if played not in legal:
        raise ShardError("played must be a legal action")
    if choice == CHOICE_NONE:
        if labels.root != "refused" or labels.policy_ok:
            raise ShardError("choice -1 is only for a refused root with policy_ok false")
    elif choice not in legal:
        raise ShardError("choice must be a legal action")
    scalars = {}
    for name in ("hp_kept", "realised_v2"):
        value = float(getattr(labels, name))
        if math.isinf(value) or (math.isnan(value) == bool(labels.value_ok)):
            raise ShardError(f"{name} must be finite exactly when value_ok is true (and never infinite)")
        scalars[name] = _canonical(value)
    for name in ("samples", "turn"):
        if not 0 <= int(getattr(labels, name)) < _INT32_LIMIT:
            raise ShardError(f"{name} must be in [0, 2**31)")
    case = tuple(int(part) for part in labels.case)
    if len(case) != 2 or min(case) < 0:
        raise ShardError("case must be (shard index, local index)")
    if not 0 <= int(labels.research_seed) < 2 ** 64:
        raise ShardError("research_seed must fit uint64")
    return {
        "legal": legal, "values": values, "samples": int(labels.samples),
        "choice": choice, "played": played,
        "root": ROOT_KINDS.index(labels.root), "outcome": OUTCOMES.index(labels.outcome),
        "hp_kept": scalars["hp_kept"], "realised_v2": scalars["realised_v2"],
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
        """Write the shard atomically (temp sibling, manifest last, rename); retry-safe on failure."""
        if self._closed:
            raise ShardError("shard writer is closed")
        if self._path.exists():
            raise FileExistsError(f"shard path exists: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_name(f"{self._path.name}.tmp-{os.getpid()}")
        shutil.rmtree(temp, ignore_errors=True)
        try:
            temp.mkdir()
            files: dict[str, dict[str, object]] = {}
            if self._rows:
                for name, array in sorted(self._arrays().items()):
                    target = temp / f"{name}.npy"
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
            with (temp / MANIFEST).open("wb") as handle:
                handle.write(_json_bytes(manifest))
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temp, self._path)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        self._closed = True
        return manifest


# --- reader ----------------------------------------------------------------------------------

def _expected_observation_files(schema: CombatSchema) -> set[str]:
    row_groups, single_groups = _layout(schema)
    view = schema.view
    names = {"action_mask"}
    for spec in row_groups:
        names |= {f"{spec.key}.{kind}" for kind in ("offsets", "rows", "values", "known")}
        if spec.key in view.optional_known_empty_groups:
            names.add(f"{spec.key}.known_full")
    for spec in single_groups:
        names |= {f"{spec.key}.values", f"{spec.key}.known"}
    return names


_LABEL_NAMES = {f"{LABEL_PREFIX}{name}" for name in _LABEL_FILES}
_PER_ROW_LABELS = _LABEL_NAMES - {"labels.legal_offsets", "labels.legal", "labels.values"}


class ShardReader:
    """Read-only access to one shard; arrays are memory-mapped lazily on first use."""

    def __init__(self, path: Path | str, schema: CombatSchema | None = None):
        self.path = Path(path)
        try:
            manifest = json.loads((self.path / MANIFEST).read_text())
            if manifest.get("schema") != SHARD_SCHEMA:
                raise ShardError("shard schema is not accepted")
            rows, files, observation = manifest["rows"], manifest["files"], manifest["observation"]
            if type(rows) is not int or rows < 0 or not isinstance(files, dict):
                raise ShardError("shard manifest is malformed")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise ShardError(f"unreadable shard manifest: {error}") from None
        self.manifest = manifest
        self.rows: int = rows
        self.schema = schema
        if schema is not None and observation != _schema_identity(schema):
            raise ShardError("shard observation schema does not match the given CombatSchema")
        self._cache: dict[str, np.ndarray] = {}

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Release every mapped file; later access re-opens lazily."""
        for array in self._cache.values():
            mapped = getattr(array, "_mmap", None)
            if mapped is not None:
                try:
                    mapped.close()
                except (BufferError, ValueError):
                    pass  # a live view still holds it; the cache drop lets it be collected
        self._cache.clear()

    @property
    def open_files(self) -> int:
        return len(self._cache)

    def __len__(self) -> int:
        return self.rows

    @property
    def observation_files(self) -> tuple[str, ...]:
        return tuple(name for name in self.manifest["files"] if not name.startswith(LABEL_PREFIX))

    @property
    def label_files(self) -> tuple[str, ...]:
        return tuple(name for name in self.manifest["files"] if name.startswith(LABEL_PREFIX))

    def array(self, name: str) -> np.ndarray:
        if name not in self.manifest["files"]:
            raise ShardError(f"shard has no file {name}")
        if name not in self._cache:
            try:
                self._cache[name] = np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError, EOFError) as error:
                raise ShardError(f"shard file {name} is unreadable: {error}") from None
        return self._cache[name]

    def verify(self, schema: CombatSchema | None = None) -> None:
        """Hashes, required files, shapes, CSR structure and label codes; only ``ShardError`` escapes."""
        schema = schema or self.schema
        try:
            self._verify(schema)
        except ShardError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise ShardError(f"shard fails verification: {type(error).__name__}: {error}") from None

    def _verify(self, schema: CombatSchema | None) -> None:
        listed = self.manifest["files"]
        on_disk = {p.name for p in self.path.iterdir()} - {MANIFEST}
        if on_disk != {f"{name}.npy" for name in listed}:
            raise ShardError("shard directory files differ from its manifest")
        if self.rows == 0:
            if listed:
                raise ShardError("a zero-row shard lists files")
            return
        required = set(_LABEL_NAMES)
        if schema is not None:
            required |= _expected_observation_files(schema)
        if missing := required - set(listed):
            raise ShardError(f"shard manifest lacks required files: {sorted(missing)}")
        for name, entry in listed.items():
            if _file_sha256(self.path / f"{name}.npy") != entry["sha256"]:
                raise ShardError(f"shard file {name} fails its sha256")
            array = self.array(name)
            if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
                raise ShardError(f"shard file {name} dtype/shape differs from its manifest")
        rows = self.rows
        stems = {name[:-len(".offsets")] for name in listed if name.endswith(".offsets")}
        for name in self.observation_files:
            stem, _, kind = name.rpartition(".")
            if kind == "offsets" or (stem in stems and kind in ("rows", "values", "known")):
                continue
            if kind == "rows":
                raise ShardError(f"{name} has no matching CSR files")
            if len(self.array(name)) != rows:
                raise ShardError(f"dense file {name} does not have one entry per decision")
        for stem in stems:
            offsets = self.array(f"{stem}.offsets")
            if len(offsets) != rows + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                raise ShardError(f"{stem}.offsets is not a valid offsets array")
            for kind in ("rows", "values", "known"):
                if f"{stem}.{kind}" not in listed or len(self.array(f"{stem}.{kind}")) != offsets[-1]:
                    raise ShardError(f"{stem}.{kind} length differs from its offsets")
        legal_offsets = self.array("labels.legal_offsets")
        legal = self.array("labels.legal")
        if (len(legal_offsets) != rows + 1 or legal_offsets[0] != 0 or np.any(np.diff(legal_offsets) < 0)
                or legal_offsets[-1] != len(legal) or len(self.array("labels.values")) != len(legal)):
            raise ShardError("labels.legal_offsets is inconsistent")
        for name in _PER_ROW_LABELS:
            if len(self.array(name)) != rows:
                raise ShardError(f"{name} does not have one entry per decision")
        root, outcome = self.array("labels.root"), self.array("labels.outcome")
        if root.min() < 0 or root.max() >= len(ROOT_KINDS) or outcome.min() < 0 or outcome.max() >= len(OUTCOMES):
            raise ShardError("label enum code out of range")
        row_of = np.repeat(np.arange(rows), np.diff(legal_offsets))
        for name, allow_none in (("labels.played", False), ("labels.choice", True)):
            column = np.asarray(self.array(name))
            hits = np.bincount(row_of, weights=(legal == column[row_of]), minlength=rows)
            absent = hits < 1
            if allow_none:
                none = column == CHOICE_NONE
                if np.any(none & ((np.asarray(root) != ROOT_KINDS.index("refused"))
                                  | np.asarray(self.array("labels.policy_ok")))):
                    raise ShardError("choice -1 is only valid for a refused root with policy_ok false")
                absent &= ~none
            if np.any(absent):
                raise ShardError(f"{name} is not a legal action of its row")

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
            offsets = f"{stem}.offsets" in self.manifest["files"] and kind in ("rows", "values", "known")
            if offsets:
                bounds = self.array(f"{stem}.offsets")
                arrays[name] = self.array(name)[int(bounds[index]):int(bounds[index + 1])]
            else:
                arrays[name] = self.array(name)[index]
        return arrays

    def frame(self, index: int, schema: CombatSchema | None = None) -> CombatFrame:
        schema = schema or self.schema
        if schema is None:
            raise ShardError("reading a frame needs the CombatSchema")
        return expand_decision(self, index, schema)

    def labels(self, index: int) -> StoredLabels:
        index = self._check_index(index)
        a = self.array
        offsets = a("labels.legal_offsets")
        low, high = int(offsets[index]), int(offsets[index + 1])
        case = a("labels.case")[index]
        return StoredLabels(
            legal=a("labels.legal")[low:high], values=a("labels.values")[low:high],
            samples=int(a("labels.samples")[index]), choice=int(a("labels.choice")[index]),
            played=int(a("labels.played")[index]), root=ROOT_KINDS[int(a("labels.root")[index])],
            outcome=OUTCOMES[int(a("labels.outcome")[index])],
            hp_kept=float(a("labels.hp_kept")[index]), realised_v2=float(a("labels.realised_v2")[index]),
            value_ok=bool(a("labels.value_ok")[index]), policy_ok=bool(a("labels.policy_ok")[index]),
            research_seed=int(a("labels.research_seed")[index]),
            case=(int(case[0]), int(case[1])),
            run_hash=int(a("labels.run_hash")[index]), turn=int(a("labels.turn")[index]),
        )


def expand_decision(shard: ShardReader, index: int, schema: CombatSchema) -> CombatFrame:
    """The compact observation the Gym returned for decision ``index``, byte for byte."""
    if shard.manifest["observation"] != _schema_identity(schema):
        raise ShardError("shard observation schema does not match the given CombatSchema")
    return decode_decision(shard.decision_arrays(shard._check_index(index)), schema)


__all__ = [
    "SHARD_SCHEMA", "CHOICE_NONE", "ROOT_KINDS", "OUTCOMES", "ShardError", "ShardWriter", "ShardReader",
    "DecisionLabels", "StoredLabels", "encode_decision", "decode_decision", "expand_decision",
    "run_hash", "validation_runs",
]
