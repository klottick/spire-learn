"""Supervised distillation of search-teacher labels into a combat student.

The trainer reads the labelled decision shards of ``distill_data`` (chunk directories written by
``scripts/neural/distill_label_entry.py``) and fits a ``value_heads = 3`` FF student:

* **Policy:** cross-entropy to ``softmax(v / tau)`` over the legal actions, on ``policy_ok`` rows.
* **Value:** three MSE terms with weight 1/3 each, on ``value_ok`` rows: the outcome head
  (``PolicyOutput.values``; +1 win, -1 death, 0 escape) and the two ``aux_values`` heads (HP kept,
  realised v2).

Each term is a mean over its selected rows and 0 when none is selected. Rows are selected by
index before any arithmetic, so a NaN label of a masked-out row never enters an expression.
Adam runs at a fixed learning rate with ``max_grad_norm`` 1.0 for every student.

**Loader.** Labels are copied into RAM once per shard (a few hundred bytes per decision).
Observations stay in the memory-mapped ``.npy`` files and batches are collated straight from
their CSR arrays into the trimmed ``EncodedBatch`` that ``encode_frames(..., trim_padding=True)``
builds for the expanded frames; label files are never read while collating. Batch order comes
from a dedicated seeded RNG domain: every pass permutes the training shards, takes them in
windows of ``SHUFFLE_WINDOW_SHARDS`` and permutes the rows within each window. At most
``OPEN_SHARD_LIMIT`` shards are mapped at a time (least recently used are unmapped), which bounds
the resident observation pages to about two windows (~1.2 GB at the real ~38 MB per shard), far
under the 8 GiB budget.

**Validation** runs every ``eval_every`` steps and at the last step on the ``validation`` shard
series: regret ``max v - v(student argmax)`` over searched decisions that are not all-way ties,
strict (argmax equals the teacher's choice) and tie-aware (argmax has the maximal value)
top-1 agreement, and every loss term. The result is the checkpoint with the lowest regret;
candidates within one run-level SE (delta 11) of it go to the lowest value loss (delta 12).
Only that checkpoint and the final one are kept.

Importing this module imports neither numpy nor torch; the dataset discovery and the checkpoint
choice are pure Python.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.config import (
    DISTILL_DATASET_SCHEMA, DISTILL_HASH_KEY, DISTILL_OBJECTIVE_ID, DistillConfig, distill_config_hash,
    distill_hash_payload, model_config_hash_payload,
)

BATCH_ROWS = 512
EVAL_EVERY = 1000
MAX_GRAD_NORM = 1.0
ADAM_EPS = 1e-8  # torch's Adam default, fixed for every student
VALUE_HEAD_WEIGHT = 1.0 / 3.0
VALUE_HEADS = ("outcome", "hp_kept", "realised_v2")
SHUFFLE_WINDOW_SHARDS = 16
OPEN_SHARD_LIMIT = 2 * SHUFFLE_WINDOW_SHARDS
RSS_BUDGET_BYTES = 8 * 1024 ** 3
RNG_DOMAINS = ("policy", "minibatch", "curriculum", "controller")
_SEED_DOMAIN = "spire-learn-distill-v1"
CHUNK_SCHEMA = "spire-learn-distill-label-chunk-v1"
CHUNK_MANIFEST = "chunk-manifest.json"
SERIES = ("train", "validation")
# OUTCOMES codes of distill_data: win, death, escaped (none never has value labels).
_OUTCOME_TARGETS = (1.0, -1.0, 0.0)
_ROOT_SEARCHED = 0  # distill_data.ROOT_KINDS.index("searched")


class DistillError(ContractError):
    """The dataset, its labels or a training invariant is malformed."""


def domain_seed(seed: int, domain: str) -> int:
    """A 63-bit seed for one RNG domain of a student seed (domains never share a stream)."""
    if domain not in RNG_DOMAINS:
        raise ValueError(f"unknown RNG domain {domain!r}")
    digest = hashlib.sha256(f"{_SEED_DOMAIN}:{domain}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


# --- dataset discovery (stdlib) -------------------------------------------------------------

@dataclass(frozen=True)
class ShardRef:
    series: str
    index: int
    path: Path
    rows: int
    manifest_sha256: str


@dataclass(frozen=True)
class Dataset:
    chunks: tuple[Path, ...]
    chunk_manifest_sha256: Mapping[str, str]
    train: tuple[ShardRef, ...]  # every training shard, in index order
    validation: tuple[ShardRef, ...]

    def train_prefix(self, fraction: float) -> tuple[ShardRef, ...]:
        """``data_fraction`` of the training shards: a prefix in index order."""
        if type(fraction) not in (int, float) or not 0 < fraction <= 1:
            raise ValueError("data_fraction must be in (0, 1]")
        count = round(fraction * len(self.train))
        if count < 1:
            raise DistillError("data_fraction selects no training shard")
        return self.train[:count]

    def identity(self) -> dict[str, object]:
        """Run metadata: chunk manifests and the shard lists (never part of an identity hash)."""
        return {
            "chunks": [str(path) for path in self.chunks],
            "chunk_manifest_sha256": dict(self.chunk_manifest_sha256),
            "dataset_sha256": _canonical_sha256(sorted(self.chunk_manifest_sha256.values())),
            "train_shards": [ref.index for ref in self.train],
            "validation_shards": [ref.index for ref in self.validation],
            "train_rows": sum(ref.rows for ref in self.train),
            "validation_rows": sum(ref.rows for ref in self.validation),
        }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_chunks(config: DistillConfig) -> tuple[Path, ...]:
    """The config's chunk dirs, or those listed by its dataset manifest (relative to it)."""
    if config.chunks:
        return config.chunks
    if config.dataset_manifest is None:
        raise DistillError("the config names neither chunks nor a dataset manifest")
    try:
        document = json.loads(config.dataset_manifest.read_text())
    except (OSError, ValueError) as error:
        raise DistillError(f"unreadable dataset manifest: {error}") from None
    if (not isinstance(document, dict) or set(document) != {"schema", "chunks"}
            or document["schema"] != DISTILL_DATASET_SCHEMA or not isinstance(document["chunks"], list)
            or not document["chunks"] or not all(isinstance(item, str) and item for item in document["chunks"])):
        raise DistillError(f"a dataset manifest is {{schema: {DISTILL_DATASET_SCHEMA}, chunks: [dirs]}}")
    base = config.dataset_manifest.parent
    chunks = tuple((Path(item) if Path(item).is_absolute() else base / item).resolve()
                   for item in document["chunks"])
    if len(set(chunks)) != len(chunks):
        raise DistillError("the dataset manifest lists a chunk twice")
    return chunks


def discover_dataset(chunks: Sequence[Path]) -> Dataset:
    """Finished chunks' train and validation shards, sorted by shard index (stdlib only).

    Each chunk must hold its ``chunk-manifest.json`` (written last by the labeller); every shard
    directory must match it, and each shard manifest must have the recorded sha256 and rows.
    File hashes are the labeller's ``verify_chunk``; this does not re-hash the arrays.
    """
    shards: dict[str, dict[int, ShardRef]] = {series: {} for series in SERIES}
    manifests: dict[str, str] = {}
    observation = None
    for chunk in chunks:
        chunk = Path(chunk)
        path = chunk / CHUNK_MANIFEST
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise DistillError(f"{chunk} is not a finished label chunk: {error}") from None
        if not isinstance(manifest, dict) or manifest.get("schema") != CHUNK_SCHEMA:
            raise DistillError(f"{chunk} is not a distillation label chunk")
        if observation is None:
            observation = manifest.get("observation")
        elif manifest.get("observation") != observation:
            raise DistillError("chunks record different observation schemas")
        manifests[str(chunk)] = _file_sha256(path)
        for series in SERIES:
            listed = manifest["shards"][series]
            on_disk = {item.name for item in (chunk / series).iterdir()} if (chunk / series).is_dir() else set()
            if set(listed) != on_disk:
                raise DistillError(f"{chunk}/{series} does not hold exactly the chunk's shards")
            for name, entry in listed.items():
                prefix, _, digits = name.partition("-")
                if prefix != "shard" or not digits.isdigit():
                    raise DistillError(f"unexpected shard directory {name}")
                index = int(digits)
                shard_path = chunk / series / name
                if _file_sha256(shard_path / "manifest.json") != entry["manifest_sha256"]:
                    raise DistillError(f"{shard_path} manifest differs from its chunk manifest")
                rows = json.loads((shard_path / "manifest.json").read_text())["rows"]
                if rows != entry["rows"]:
                    raise DistillError(f"{shard_path} rows differ from its chunk manifest")
                if index in shards[series]:
                    raise DistillError(f"shard {index} appears in two chunks")
                shards[series][index] = ShardRef(series, index, shard_path, rows, entry["manifest_sha256"])
    if not shards["train"]:
        raise DistillError("the dataset has no training shard")
    return Dataset(
        chunks=tuple(Path(chunk) for chunk in chunks),
        chunk_manifest_sha256=manifests,
        train=tuple(shards["train"][index] for index in sorted(shards["train"])),
        validation=tuple(shards["validation"][index] for index in sorted(shards["validation"])),
    )


# --- checkpoint choice (delta 12; pure) -----------------------------------------------------

def run_level_se(differences: Sequence[float], run_hashes: Sequence[int]) -> tuple[float, float]:
    """``(d_bar, SE)`` of per-row differences grouped by run (delta 11); SE is 0 below 2 runs."""
    if len(differences) != len(run_hashes):
        raise ValueError("one run hash per difference")
    total = len(differences)
    if total == 0:
        return 0.0, 0.0
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for difference, run in zip(differences, run_hashes):
        run = int(run)
        sums[run] = sums.get(run, 0.0) + float(difference)
        counts[run] = counts.get(run, 0) + 1
    d_bar = sum(sums.values()) / total
    groups = len(sums)
    if groups < 2:
        return d_bar, 0.0
    spread = sum((sums[run] - counts[run] * d_bar) ** 2 for run in sums)
    return d_bar, math.sqrt(groups / (groups - 1) * spread) / total


@dataclass(frozen=True)
class EvalRecord:
    """One validation pass, as the checkpoint choice needs it."""

    step: int
    regret: float
    value_loss: float
    regret_rows: Sequence[float]  # per regret row, the same validation rows for every record
    run_hashes: Sequence[int]


def choose_checkpoint(records: Sequence[EvalRecord]) -> tuple[EvalRecord, list[int]]:
    """Lowest regret; candidates within one run-level SE of it go to the lowest value loss.

    Returns the chosen record and the steps tied with the best (ties on equal keys go to the
    earlier step).
    """
    if not records:
        raise ValueError("no validation record to choose from")
    best = min(records, key=lambda item: (item.regret, item.step))
    tied = []
    for record in records:
        if len(record.regret_rows) != len(best.regret_rows) or list(record.run_hashes) != list(best.run_hashes):
            raise ValueError("validation records cover different rows")
        differences = [float(a) - float(b) for a, b in zip(record.regret_rows, best.regret_rows)]
        d_bar, se = run_level_se(differences, best.run_hashes)
        if d_bar <= se:
            tied.append(record)
    chosen = min(tied, key=lambda item: (item.value_loss, item.step))
    return chosen, [record.step for record in tied]


# --- labels and shards (numpy) --------------------------------------------------------------

def _np():
    import numpy
    return numpy


@dataclass
class ShardLabels:
    """The label columns a trainer needs, copied into RAM (never a model input)."""

    rows: int
    legal_offsets: Any
    values: Any
    choice: Any
    root: Any
    outcome: Any
    hp_kept: Any
    realised_v2: Any
    value_ok: Any
    policy_ok: Any
    run_hash: Any

    @classmethod
    def read(cls, reader) -> "ShardLabels":
        np = _np()
        a = lambda name: np.array(reader.array(f"labels.{name}"))  # noqa: E731 - copies out of the map
        labels = cls(rows=reader.rows, legal_offsets=a("legal_offsets").astype(np.int64), values=a("values"),
                     choice=a("choice"), root=a("root"), outcome=a("outcome"), hp_kept=a("hp_kept"),
                     realised_v2=a("realised_v2"), value_ok=a("value_ok").astype(bool),
                     policy_ok=a("policy_ok").astype(bool), run_hash=a("run_hash"))
        labels._check()
        return labels

    def _check(self) -> None:
        np = _np()
        value_rows = self.value_ok
        if np.any(self.outcome[value_rows] > 2):
            raise DistillError("a value_ok row has no fight outcome")
        if not (np.all(np.isfinite(self.hp_kept[value_rows])) and np.all(np.isfinite(self.realised_v2[value_rows]))):
            raise DistillError("a value_ok row has a non-finite value target")
        counts = np.diff(self.legal_offsets)
        for row in np.flatnonzero(self.policy_ok):
            values = self.values[self.legal_offsets[row]:self.legal_offsets[row + 1]]
            if self.root[row] != _ROOT_SEARCHED or counts[row] < 1 or not np.all(np.isfinite(values)):
                raise DistillError("a policy_ok row needs a searched root with finite values")

    def row_values(self, row: int):
        return self.values[self.legal_offsets[row]:self.legal_offsets[row + 1]]

    def regret_row(self, row: int) -> bool:
        """Searched and not an all-way tie (the rows regret and agreement are measured on)."""
        if self.root[row] != _ROOT_SEARCHED:
            return False
        values = self.row_values(row)
        return len(values) > 0 and bool(values.max() != values.min())


class ShardSet:
    """Labels of every shard in RAM; observation files mapped lazily, at most ``open_limit`` shards."""

    def __init__(self, refs: Sequence[ShardRef], schema, *, open_limit: int = OPEN_SHARD_LIMIT):
        from spire_learn.neural.distill_data import ShardReader

        if open_limit < 1:
            raise ValueError("open_limit must be positive")
        self.refs = tuple(refs)
        self.schema = schema
        self._open_limit = open_limit
        self._readers: OrderedDict[int, Any] = OrderedDict()
        self.labels: list[ShardLabels] = []
        for ref in self.refs:
            with ShardReader(ref.path, schema) as reader:
                if reader.rows != ref.rows:
                    raise DistillError(f"{ref.path} has {reader.rows} rows, its chunk records {ref.rows}")
                self.labels.append(ShardLabels.read(reader) if reader.rows else None)
        self.row_counts = tuple(ref.rows for ref in self.refs)
        self.peak_open = 0

    def reader(self, position: int):
        from spire_learn.neural.distill_data import ShardReader

        reader = self._readers.pop(position, None)
        if reader is None:
            reader = ShardReader(self.refs[position].path, self.schema)
            while len(self._readers) >= self._open_limit:
                _, oldest = self._readers.popitem(last=False)
                oldest.close()
        self._readers[position] = reader
        self.peak_open = max(self.peak_open, len(self._readers))
        return reader

    @property
    def open_shards(self) -> int:
        return len(self._readers)

    def close(self) -> None:
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()

    def __enter__(self) -> "ShardSet":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# --- batch order (dedicated RNG domain) -----------------------------------------------------

class BatchOrder:
    """Seeded batches of ``(shard position, row)`` over passes of shard windows.

    The same seed and the same row counts always give the same batches. The CPU torch generator
    of the ``minibatch`` domain draws every permutation.
    """

    def __init__(self, row_counts: Sequence[int], *, seed: int, batch_rows: int,
                 window: int = SHUFFLE_WINDOW_SHARDS):
        import torch

        if sum(row_counts) < 1:
            raise DistillError("no training rows")
        if batch_rows < 1 or window < 1:
            raise ValueError("batch_rows and window must be positive")
        self._counts = tuple(int(count) for count in row_counts)
        self._batch_rows = batch_rows
        self._window = window
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(domain_seed(seed, "minibatch"))
        self._buffer = _np().zeros((0, 2), dtype=_np().int64)
        self.passes = 0

    def _next_pass(self):
        import torch

        np = _np()
        order = torch.randperm(len(self._counts), generator=self.generator).tolist()
        pieces = []
        for start in range(0, len(order), self._window):
            window = order[start:start + self._window]
            items = np.concatenate([
                np.stack((np.full(self._counts[shard], shard, dtype=np.int64),
                          np.arange(self._counts[shard], dtype=np.int64)), axis=1)
                for shard in window])
            permutation = torch.randperm(len(items), generator=self.generator).numpy()
            pieces.append(items[permutation])
        self.passes += 1
        return np.concatenate(pieces)

    def next_batch(self):
        np = _np()
        while len(self._buffer) < self._batch_rows:
            self._buffer = np.concatenate((self._buffer, self._next_pass()))
        batch, self._buffer = self._buffer[:self._batch_rows], self._buffer[self._batch_rows:]
        return batch


# --- collation (observation files only) -----------------------------------------------------

def _layout(schema):
    from spire_learn.neural.distill_data import _layout as layout
    return layout(schema)


def _csr_index(offsets, rows):
    """Flat indices of the CSR entries of ``rows`` (in order) and their per-row lengths."""
    np = _np()
    low = np.asarray(offsets[rows], dtype=np.int64)
    high = np.asarray(offsets[rows + 1], dtype=np.int64)
    lengths = high - low
    total = int(lengths.sum())
    starts = np.repeat(low - (np.cumsum(lengths) - lengths), lengths)
    return starts + np.arange(total, dtype=np.int64), lengths


def collate_observations(shards: ShardSet, items, *, device: str = "cpu"):
    """The trimmed ``EncodedBatch`` of ``items`` (``(shard position, row)`` pairs), from observation files.

    It equals ``encode_frames([expand_decision(...) for each item], schema, device=...,
    trim_padding=True)`` tensor for tensor; no label file is read.
    """
    import torch

    from spire_learn.neural.features import EncodedBatch

    np = _np()
    schema = shards.schema
    view = schema.view
    row_groups, single_groups = _layout(schema)
    items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
    batch = len(items)
    if batch < 1:
        raise ValueError("a batch needs at least one item")
    sparse: dict[str, list[tuple[Any, Any, Any, Any]]] = {spec.key: [] for spec in row_groups}
    dense_names = ["action_mask"]
    dense_names += [f"{spec.key}.{kind}" for spec in single_groups for kind in ("values", "known")]
    dense_names += [f"{spec.key}.known_full" for spec in row_groups if spec.key in view.optional_known_empty_groups]
    dense: dict[str, Any] = {}
    for position in np.unique(items[:, 0]):
        selected = np.flatnonzero(items[:, 0] == position)
        rows = items[selected, 1]
        order = np.argsort(rows, kind="stable")
        selected, rows = selected[order], rows[order]
        reader = shards.reader(int(position))
        if rows.min() < 0 or rows.max() >= reader.rows:
            raise IndexError("batch row outside its shard")
        for spec in row_groups:
            key = spec.key
            flat, lengths = _csr_index(reader.array(f"{key}.offsets"), rows)
            sparse[key].append((np.repeat(selected, lengths), np.asarray(reader.array(f"{key}.rows")[flat]),
                                reader.array(f"{key}.values")[flat], reader.array(f"{key}.known")[flat]))
        for name in dense_names:
            source = reader.array(name)
            if name not in dense:
                dense[name] = np.empty((batch, *source.shape[1:]), dtype=source.dtype)
            dense[name][selected] = source[rows]

    values_out: dict[str, Any] = {}
    known_out: dict[str, Any] = {}
    occupied_out: dict[str, Any] = {}
    coverage_out: dict[str, Any] = {}
    row_padding: list[tuple[str, int]] = []
    known_padding: dict[str, Any] = {}
    dense_rows = view.dense_rows
    singles = {spec.key for spec in single_groups}
    by_key = {spec.key: spec for spec in row_groups}
    for group, spec in schema.tensors.items():
        if not spec.columns:
            continue
        columns = len(spec.columns)
        if group in singles:
            values = torch.from_numpy(dense[f"{group}.values"].reshape(batch, 1, columns)).to(torch.int64)
            known = torch.from_numpy(dense[f"{group}.known"].reshape(batch, 1, columns)).to(torch.bool)
            values_out[group], known_out[group] = values, known
            coverage_out[group] = known[:, 0, :].to(torch.float32)
            continue
        assert group in by_key
        capacity = spec.shape[0]
        pieces = sparse[group]
        batch_index = np.concatenate([piece[0] for piece in pieces])
        row_index = np.concatenate([piece[1] for piece in pieces]).astype(np.int64)
        extent = int(row_index.max()) + 1 if row_index.size else 0
        rows = max(1, extent)
        required = group in view.required_known_empty_groups
        optional = group in view.optional_known_empty_groups
        fill = 0 if required or optional else spec.padding
        values = np.full((batch, rows, columns), fill, dtype=np.dtype(spec.dtype))
        if required:
            known = np.ones((batch, rows, columns), dtype=np.int8)
        elif optional:
            known = np.array(dense[f"{group}.known_full"][:, :rows], dtype=np.int8, copy=True)
        else:
            known = np.zeros((batch, rows, columns), dtype=np.int8)
        occupied = np.zeros((batch, rows), dtype=np.int8)
        if row_index.size:
            values[batch_index, row_index] = np.concatenate([piece[2] for piece in pieces])
            known[batch_index, row_index] = np.concatenate([piece[3] for piece in pieces])
            occupied[batch_index, row_index] = 1
        stop = rows if rows < capacity else None
        if stop is not None:
            row_padding.append((group, capacity - rows))
        values_t = torch.from_numpy(values).to(torch.int64)
        known_t = torch.from_numpy(known).to(torch.bool)
        values_out[group], known_out[group] = values_t, known_t
        occupied_out[group] = torch.from_numpy(occupied).to(torch.bool)
        # Known coverage exactly as encode_frames computes it for the trimmed rows.
        dense_count = dense_rows.get(group, capacity)
        coverage_known = known_t
        if stop is not None and optional:
            tail = torch.from_numpy(np.ascontiguousarray(dense[f"{group}.known_full"][:, rows:capacity])).to(torch.bool)
            known_padding[group] = tail
            coverage_known = torch.cat((known_t, tail), dim=1)
        elif stop is not None and (dense_count == capacity or required):
            coverage_known = torch.nn.functional.pad(known_t, (0, 0, 0, capacity - rows), value=required)
        if dense_count == capacity:
            coverage_out[group] = coverage_known.to(torch.float32).mean(dim=1)
        else:
            coverage_out[group] = coverage_known.to(torch.float32).sum(dim=1).div_(dense_count)
    action_mask = torch.from_numpy(dense["action_mask"]).to(torch.bool)

    def move(mapping: Mapping[str, Any]) -> dict[str, Any]:
        return {key: tensor.to(device=device) for key, tensor in mapping.items()}

    return EncodedBatch(
        schema_fingerprint=schema.schema_fingerprint,
        values=move(values_out), known=move(known_out), occupied=move(occupied_out),
        action_mask=action_mask.to(device=device), known_coverage=move(coverage_out),
        _row_padding=tuple(row_padding), _known_padding=move(known_padding),
    )


# --- targets (labels in RAM) ----------------------------------------------------------------

@dataclass
class BatchTargets:
    """Index-selected targets; every tensor holds only selected rows, so none holds a NaN."""

    policy_index: Any  # int64 (P,) batch positions with policy_ok
    policy_target: Any  # float32 (P, W) softmax(v / tau) over the legal actions, 0 elsewhere
    value_index: Any  # int64 (V,) batch positions with value_ok
    value_target: Any  # float32 (V, 3): outcome, hp_kept, realised_v2
    regret_index: Any  # int64 (R,) searched rows that are not all-way ties
    regret_values: Any  # float64 (R, W) teacher values, -inf on illegal actions
    regret_choice: Any  # int64 (R,) teacher choice
    regret_run_hash: Any  # uint64 (R,)

    @property
    def policy_rows(self) -> int:
        return int(self.policy_index.numel())

    @property
    def value_rows(self) -> int:
        return int(self.value_index.numel())


def batch_targets(shards: ShardSet, items, action_mask, *, tau: float, device: str = "cpu") -> BatchTargets:
    """Targets of ``items`` from the in-RAM labels; ``action_mask`` is the batch's public mask."""
    import torch

    np = _np()
    items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
    mask = action_mask.detach().cpu().numpy().astype(bool)
    width = mask.shape[1]
    policy, value, regret = [], [], []
    value_rows, regret_rows, regret_choice, regret_runs = [], [], [], []
    policy_values = []
    for position, (shard, row) in enumerate(items.tolist()):
        labels = shards.labels[shard]
        if labels is None:
            raise IndexError("batch row in an empty shard")
        legal = np.flatnonzero(mask[position])
        row_values = labels.row_values(row)
        if len(row_values) != len(legal):
            raise DistillError("labels.values do not match the batch action mask")
        if labels.policy_ok[row]:
            policy.append(position)
            policy_values.append((legal, row_values))
        if labels.value_ok[row]:
            value.append(position)
            value_rows.append((_OUTCOME_TARGETS[int(labels.outcome[row])], float(labels.hp_kept[row]),
                               float(labels.realised_v2[row])))
        if labels.regret_row(row):
            regret.append(position)
            dense = np.full(width, -np.inf)
            dense[legal] = row_values
            regret_rows.append(dense)
            regret_choice.append(int(labels.choice[row]))
            regret_runs.append(int(labels.run_hash[row]))
    target = np.zeros((len(policy), width), dtype=np.float64)
    for index, (legal, row_values) in enumerate(policy_values):
        scaled = np.asarray(row_values, dtype=np.float64) / tau
        weights = np.exp(scaled - scaled.max())
        target[index, legal] = weights / weights.sum()
    as_index = lambda rows: torch.tensor(rows, dtype=torch.int64, device=device)  # noqa: E731
    return BatchTargets(
        policy_index=as_index(policy),
        policy_target=torch.from_numpy(target).to(device=device, dtype=torch.float32),
        value_index=as_index(value),
        value_target=torch.tensor(value_rows, dtype=torch.float32, device=device).reshape(-1, 3),
        regret_index=as_index(regret),
        regret_values=(np.stack(regret_rows) if regret_rows else np.zeros((0, width))),
        regret_choice=np.asarray(regret_choice, dtype=np.int64),
        regret_run_hash=np.asarray(regret_runs, dtype=np.uint64),
    )


# --- loss -----------------------------------------------------------------------------------

@dataclass
class LossTerms:
    """Sums of each term over the selected rows, divided by the given row counts."""

    total: Any
    policy: Any
    values: tuple[Any, Any, Any]
    policy_rows: int
    value_rows: int


def loss_terms(output, action_mask, targets: BatchTargets, *, policy_count: int | None = None,
               value_count: int | None = None) -> LossTerms:
    """Masked distillation loss. Rows are selected by index first, so masked labels never enter.

    ``policy_count``/``value_count`` default to this batch's selected rows (a mean per term). A
    microbatch passes the full batch's counts, so the accumulated gradient equals one full batch.
    """
    import torch

    if output.aux_values is None:
        raise ContractError("distillation needs a value_heads = 3 student")
    policy_count = targets.policy_rows if policy_count is None else policy_count
    value_count = targets.value_rows if value_count is None else value_count
    zero = output.logits.new_zeros(())
    policy = zero
    if targets.policy_rows:
        logits = output.logits.index_select(0, targets.policy_index)
        mask = action_mask.index_select(0, targets.policy_index)
        log_probs = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        # Illegal entries are replaced, never multiplied: 0 * -inf cannot appear.
        safe = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        policy = -(targets.policy_target * safe).sum() / policy_count
    heads = [zero, zero, zero]
    if targets.value_rows:
        predicted = torch.cat((output.values.unsqueeze(-1), output.aux_values), dim=-1)
        predicted = predicted.index_select(0, targets.value_index)
        squared = (predicted - targets.value_target).square()
        heads = [squared[:, head].sum() / value_count for head in range(3)]
    total = policy + VALUE_HEAD_WEIGHT * (heads[0] + heads[1] + heads[2])
    if not total.requires_grad and output.logits.requires_grad:
        # Nothing selected: anchor the graph so backward gives exact zero gradients.
        total = total + 0.0 * (output.logits.sum() + output.values.sum() + output.aux_values.sum())
    return LossTerms(total, policy, tuple(heads), policy_count, value_count)


def train_step(policy, optimizer, shards: ShardSet, items, *, tau: float, microbatch_rows: int,
               device: str = "cpu") -> dict[str, float]:
    """One optimizer step over ``items``, accumulating microbatches to the full-batch gradient."""
    import torch

    np = _np()
    items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
    # The full batch's selected-row counts come from the in-RAM labels, before any collation, so
    # each microbatch is collated (and its activations freed) one at a time.
    policy_count = sum(int(shards.labels[shard].policy_ok[row]) for shard, row in items.tolist())
    value_count = sum(int(shards.labels[shard].value_ok[row]) for shard, row in items.tolist())
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {"policy_loss": 0.0, **{f"{name}_loss": 0.0 for name in VALUE_HEADS}}
    for start in range(0, len(items), microbatch_rows):
        chunk = items[start:start + microbatch_rows]
        batch = collate_observations(shards, chunk, device=device)
        targets = batch_targets(shards, chunk, batch.action_mask, tau=tau, device=device)
        output = policy(batch)
        terms = loss_terms(output, batch.action_mask, targets, policy_count=max(policy_count, 1),
                           value_count=max(value_count, 1))
        terms.total.backward()
        sums["policy_loss"] += float(terms.policy.detach())
        for name, head in zip(VALUE_HEADS, terms.values):
            sums[f"{name}_loss"] += float(head.detach())
    parameters = [parameter for parameter in policy.parameters() if parameter.grad is not None]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)) if parameters else 0.0
    if not math.isfinite(grad_norm):
        raise DistillError("non-finite gradient norm")
    optimizer.step()
    value_loss = VALUE_HEAD_WEIGHT * sum(sums[f"{name}_loss"] for name in VALUE_HEADS)
    return {**sums, "value_loss": value_loss, "loss": sums["policy_loss"] + value_loss,
            "grad_norm": grad_norm, "policy_rows": policy_count, "value_rows": value_count,
            "rows": len(items)}


# --- validation -----------------------------------------------------------------------------

@dataclass
class ValidationResult:
    metrics: dict[str, float | int]
    regret_rows: list[float]
    run_hashes: list[int]


def evaluate(policy, shards: ShardSet, *, tau: float, batch_rows: int, device: str = "cpu") -> ValidationResult:
    """Every validation row in shard order: regret, strict/tie-aware agreement and each loss term."""
    import torch

    np = _np()
    sums = {"policy": 0.0, "outcome": 0.0, "hp_kept": 0.0, "realised_v2": 0.0}
    policy_rows = value_rows = 0
    regrets: list[float] = []
    runs: list[int] = []
    strict = tie_aware = 0
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for position, count in enumerate(shards.row_counts):
                for start in range(0, count, batch_rows):
                    rows = np.arange(start, min(count, start + batch_rows), dtype=np.int64)
                    items = np.stack((np.full(len(rows), position, dtype=np.int64), rows), axis=1)
                    batch = collate_observations(shards, items, device=device)
                    targets = batch_targets(shards, items, batch.action_mask, tau=tau, device=device)
                    output = policy(batch)
                    terms = loss_terms(output, batch.action_mask, targets, policy_count=1, value_count=1)
                    sums["policy"] += float(terms.policy)
                    for name, head in zip(VALUE_HEADS, terms.values):
                        sums[name] += float(head)
                    policy_rows += targets.policy_rows
                    value_rows += targets.value_rows
                    if targets.regret_index.numel():
                        logits = output.logits.index_select(0, targets.regret_index)
                        mask = batch.action_mask.index_select(0, targets.regret_index)
                        choice = logits.masked_fill(~mask, -torch.inf).argmax(dim=-1).cpu().numpy()
                        values = targets.regret_values
                        best = values.max(axis=1)
                        chosen = values[np.arange(len(choice)), choice]
                        regrets.extend((best - chosen).tolist())
                        runs.extend(int(run) for run in targets.regret_run_hash)
                        strict += int(np.sum(choice == targets.regret_choice))
                        tie_aware += int(np.sum(chosen == best))
            shards.close()
    finally:
        policy.train(was_training)
    policy_loss = sums["policy"] / policy_rows if policy_rows else 0.0
    heads = {name: (sums[name] / value_rows if value_rows else 0.0) for name in VALUE_HEADS}
    value_loss = VALUE_HEAD_WEIGHT * sum(heads.values())
    regret_count = len(regrets)
    metrics = {
        "regret": sum(regrets) / regret_count if regret_count else 0.0,
        "strict_agreement": strict / regret_count if regret_count else 0.0,
        "tie_aware_agreement": tie_aware / regret_count if regret_count else 0.0,
        "policy_loss": policy_loss,
        **{f"{name}_mse": heads[name] for name in VALUE_HEADS},
        "value_loss": value_loss,
        "loss": policy_loss + value_loss,
        "regret_rows": regret_count, "policy_rows": policy_rows, "value_rows": value_rows,
        "rows": sum(shards.row_counts),
    }
    return ValidationResult(metrics, regrets, runs)


# --- the training run -----------------------------------------------------------------------

class _NoGate:
    waited_seconds = 0.0

    def wait(self, stage: str) -> None:
        return None

    def release(self) -> None:
        return None


def _peak_rss_bytes() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _rng_state(policy_generator, batch_order: BatchOrder, seed: int) -> dict[str, object]:
    # curriculum and controller are unused by distillation; their seeded states are recorded so the
    # four RNG domains of a training checkpoint stay explicit.
    return {
        "policy": {"torch": policy_generator.get_state()},
        "minibatch": {"torch": batch_order.generator.get_state()},
        "curriculum": {"python": random.Random(domain_seed(seed, "curriculum")).getstate()},
        "controller": {"python": random.Random(domain_seed(seed, "controller")).getstate()},
    }


def build_student(config: DistillConfig, schema, *, device: str):
    from spire_learn.neural.model import build_policy

    identity = config.identity
    return build_policy(schema, config.model, seed=domain_seed(config.seed, "policy"), device=device,
                        objective_id=DISTILL_OBJECTIVE_ID, model_id=identity.model_id,
                        features_id=identity.features_id)


def _save(path: Path, *, policy, optimizer, policy_generator, batch_order, config: DistillConfig,
          step: int, counters: Mapping[str, int], run_metadata: Mapping[str, object]) -> None:
    from spire_learn.neural.checkpoint import identity_payload, save_checkpoint

    if config.identity.objective_id != DISTILL_OBJECTIVE_ID:
        raise ContractError("a distillation checkpoint carries the distillation objective")
    save_checkpoint(
        path, policy=policy, optimizer=optimizer,
        rng_state=_rng_state(policy_generator, batch_order, config.seed),
        metadata={
            "identity": identity_payload(config.identity),
            "run_metadata": {**run_metadata, "step": step},
            "training_state": {
                "optimizer_step": step,
                "hyperparameter_id": config.identity.config_hashes[DISTILL_HASH_KEY],
                "rng_domains": list(RNG_DOMAINS),
                "counters": dict(counters),
                "update_id": f"step-{step}",
                "behavior_policy_version": step,
                "next_environment_seed_ordinal": 0,
                "next_decision_sequence": 0,
                "saved_inflight_record_count": 0,
            },
            "parent_checkpoint": None,
        })


def _json_line(handle, event: Mapping[str, object]) -> None:
    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def train_student(config: DistillConfig, schema, dataset: Dataset, output: Path, *, device: str = "cpu",
                  gate=None, log_every: int = 50, clock: Callable[[], float] = time.perf_counter,
                  provenance: Mapping[str, object] | None = None) -> dict[str, object]:
    """Train one student; writes ``train_log.jsonl``, the kept checkpoints and returns the summary.

    ``output`` must be a fresh or empty directory. ``gate`` (``wait(stage)``/``release()``) is consulted before
    training and at every validation interval. Faults raise; nothing is retried.
    """
    import torch

    if config.identity.config_hashes.get(DISTILL_HASH_KEY) != distill_config_hash(config.distill):
        raise ContractError("the identity's distill hash disagrees with the config")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DistillError(f"{output} must be a fresh or empty directory")
    candidates = output / "candidates"
    candidates.mkdir()
    gate = gate or _NoGate()
    hyper = config.distill
    started = clock()
    train_refs = dataset.train_prefix(hyper.data_fraction)
    if not dataset.validation or not any(ref.rows for ref in dataset.validation):
        raise DistillError("the dataset has no validation rows")
    train_set = ShardSet(train_refs, schema)
    validation_set = ShardSet(dataset.validation, schema)
    run_metadata = {
        "operation": "distill-train",
        "config_sha256": config.config_sha256,
        "seed": config.seed,
        "eval_every": config.eval_every,
        "microbatch_rows": hyper.effective_microbatch_rows,
        "shuffle_window_shards": SHUFFLE_WINDOW_SHARDS,
        "environment_revision": config.environment_revision,
        "standard_view_fingerprint": config.standard_view_fingerprint,
        "device": device,
        "data_fraction": hyper.data_fraction,
        "train_shards": [ref.index for ref in train_refs],
        "dataset_sha256": dataset.identity()["dataset_sha256"],
    }
    summary: dict[str, object] = {
        "schema": "spire-learn-distill-train-summary-v1",
        "identity": {"model_id": config.identity.model_id, "features_id": config.identity.features_id,
                     "action_id": config.identity.action_id, "objective_id": config.identity.objective_id,
                     "config_hashes": dict(config.identity.config_hashes)},
        "model": model_config_hash_payload(config.model),
        "distill": distill_hash_payload(hyper),
        "optimizer": {"name": "adam", "learning_rate": hyper.learning_rate, "eps": ADAM_EPS,
                      "max_grad_norm": MAX_GRAD_NORM},
        "run_metadata": run_metadata,
        "dataset": {**dataset.identity(), "train_prefix_rows": sum(ref.rows for ref in train_refs)},
        "seeds": {domain: domain_seed(config.seed, domain) for domain in RNG_DOMAINS},
        "provenance": dict(provenance or {}),
    }
    log = (output / "train_log.jsonl").open("x", encoding="utf-8")
    train_seconds = eval_seconds = 0.0
    records: list[EvalRecord] = []
    evaluations: list[dict[str, object]] = []
    counters = {"optimizer_steps": 0, "rows_seen": 0, "policy_rows": 0, "value_rows": 0}
    try:
        gate.wait("train-start")
        policy = build_student(config, schema, device=device)
        summary["parameter_count"] = int(sum(parameter.numel() for parameter in policy.parameters()))
        optimizer = torch.optim.Adam(policy.parameters(), lr=hyper.learning_rate, eps=ADAM_EPS)
        policy_generator = torch.Generator(device="cpu")
        policy_generator.manual_seed(domain_seed(config.seed, "policy"))
        order = BatchOrder(train_set.row_counts, seed=config.seed, batch_rows=hyper.batch_rows)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        _json_line(log, {"event": "start", "parameter_count": summary["parameter_count"],
                         "train_shards": len(train_refs), "train_rows": sum(train_set.row_counts),
                         "validation_rows": sum(validation_set.row_counts)})
        window_rows = window_seconds = 0.0
        for step in range(1, hyper.gradient_steps + 1):
            began = clock()
            metrics = train_step(policy, optimizer, train_set, order.next_batch(), tau=hyper.tau,
                                 microbatch_rows=hyper.effective_microbatch_rows, device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = clock() - began
            train_seconds += elapsed
            window_seconds += elapsed
            window_rows += metrics["rows"]
            counters["optimizer_steps"] = step
            counters["rows_seen"] += metrics["rows"]
            counters["policy_rows"] += metrics["policy_rows"]
            counters["value_rows"] += metrics["value_rows"]
            if step % log_every == 0 or step == 1:
                _json_line(log, {"event": "train", "step": step, "passes": order.passes,
                                 "rows_per_second": window_rows / window_seconds if window_seconds else 0.0,
                                 "open_shards": train_set.open_shards, "peak_rss_bytes": _peak_rss_bytes(),
                                 **{key: value for key, value in metrics.items() if key != "rows"}})
                window_rows = window_seconds = 0.0
            if step % config.eval_every == 0 or step == hyper.gradient_steps:
                gate.wait(f"eval-{step}")
                began = clock()
                result = evaluate(policy, validation_set, tau=hyper.tau, batch_rows=hyper.effective_microbatch_rows,
                                  device=device)
                eval_seconds += clock() - began
                path = candidates / f"step-{step:08d}"
                _save(path, policy=policy, optimizer=optimizer, policy_generator=policy_generator,
                      batch_order=order, config=config, step=step, counters=counters, run_metadata=run_metadata)
                records.append(EvalRecord(step, result.metrics["regret"], result.metrics["value_loss"],
                                          result.regret_rows, result.run_hashes))
                evaluation = {"step": step, **result.metrics}
                evaluations.append(evaluation)
                _json_line(log, {"event": "validation", **evaluation, "peak_rss_bytes": _peak_rss_bytes()})
                gate.release()
        chosen, tied = choose_checkpoint(records)
        final_step = hyper.gradient_steps
        final = output / "checkpoint-final"
        (candidates / f"step-{final_step:08d}").rename(final)
        best_path = final
        if chosen.step != final_step:
            best_path = output / "checkpoint-best"
            (candidates / f"step-{chosen.step:08d}").rename(best_path)
        shutil.rmtree(candidates)
        best_eval = next(item for item in evaluations if item["step"] == chosen.step)
        peak_vram = (int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None)
        peak_reserved = (int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else None)
        total_rows = counters["rows_seen"]
        summary.update({
            "status": "complete",
            "counters": counters,
            "passes": order.passes,
            "evaluations": evaluations,
            "chosen": {"step": chosen.step, "tied_steps": tied, "regret": best_eval["regret"],
                       "value_loss": best_eval["value_loss"],
                       "value_mse": {name: best_eval[f"{name}_mse"] for name in VALUE_HEADS},
                       "checkpoint": str(best_path)},
            "final": {"step": final_step, "checkpoint": str(final)},
            "timing": {"train_seconds": train_seconds, "eval_seconds": eval_seconds,
                       "gpu_seconds": train_seconds + eval_seconds if device.startswith("cuda") else 0.0,
                       "gpu_wait_seconds": float(getattr(gate, "waited_seconds", 0.0)),
                       "wall_seconds": clock() - started,
                       "rows_per_second": total_rows / train_seconds if train_seconds else 0.0},
            "memory": {"peak_rss_bytes": _peak_rss_bytes(), "rss_budget_bytes": RSS_BUDGET_BYTES,
                       "within_rss_budget": _peak_rss_bytes() <= RSS_BUDGET_BYTES,
                       "peak_open_shards": max(train_set.peak_open, validation_set.peak_open),
                       "peak_vram_allocated_bytes": peak_vram, "peak_vram_reserved_bytes": peak_reserved},
        })
        _json_line(log, {"event": "complete", "chosen_step": chosen.step, "tied_steps": tied})
        return summary
    finally:
        log.close()
        train_set.close()
        validation_set.close()


__all__ = [
    "ADAM_EPS", "BATCH_ROWS", "EVAL_EVERY", "MAX_GRAD_NORM", "OPEN_SHARD_LIMIT", "RNG_DOMAINS",
    "SHUFFLE_WINDOW_SHARDS", "VALUE_HEADS", "VALUE_HEAD_WEIGHT", "BatchOrder", "BatchTargets", "Dataset",
    "DistillError", "EvalRecord", "LossTerms", "ShardLabels", "ShardRef", "ShardSet", "ValidationResult",
    "batch_targets", "build_student", "choose_checkpoint", "collate_observations", "dataset_chunks",
    "discover_dataset", "domain_seed", "evaluate", "loss_terms", "run_level_se", "train_step",
    "train_student",
]
