#!/usr/bin/env python3.12
"""Delta 11's tau candidates and delta 12's gradient steps from labelled shards (runtime container; numpy).

    distill_tau_entry.py --pilot CHUNK_DIR [--dataset DATASET.json]

Delta 11: the candidates are the 25th, 50th and 75th percentiles of the strictly positive gaps between
the top two distinct teacher values. They are computed over the pilot chunk's ``train`` series only,
on searched decisions, with a floor of 0.01. "Top two distinct" means the largest value minus the
largest value strictly below it, so a decision where every legal action has the same value has no
gap and is excluded. Percentiles use linear interpolation between order statistics (numpy's default,
pinned here as ``method="linear"``).

Delta 12: with ``--dataset``, ``gradient_steps = round(6 * train_rows / BATCH_ROWS)``, where
``train_rows`` counts every stored row of the dataset's ``train`` series (from the verified chunk
manifests).

Writes ``tau.json`` to ``$SPIRE_LEARN_RUNNER_OUTPUT`` (or prints it when unset).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

BATCH_ROWS = 512
TAU_FLOOR = 0.01
PERCENTILES = (25, 50, 75)
PASSES = 6


def top_two_gap(values) -> float | None:
    """The top value minus the largest strictly smaller value; None when all values are equal."""
    ordered = sorted(set(float(v) for v in values), reverse=True)
    if len(ordered) < 2:
        return None
    return ordered[0] - ordered[1]


def tau_candidates(gaps, *, floor: float = TAU_FLOOR) -> dict:
    """Delta 11: linear-interpolated percentiles of the strictly positive gaps, floored."""
    positive = sorted(g for g in gaps if g > 0)
    if not positive:
        raise ValueError("no strictly positive teacher-value gaps")
    out = {}
    for q in PERCENTILES:
        rank = (len(positive) - 1) * q / 100
        low, frac = int(rank), rank - int(rank)
        high = min(low + 1, len(positive) - 1)
        value = positive[low] + (positive[high] - positive[low]) * frac
        out[f"p{q}"] = max(value, floor)
    return out


def gradient_steps(train_rows: int) -> int:
    return round(PASSES * train_rows / BATCH_ROWS)


def _pilot_gaps(chunk: Path) -> tuple[list[float], dict]:
    import numpy as np

    from spire_learn.neural.distill_data import ROOT_KINDS

    searched = ROOT_KINDS.index("searched")
    gaps, counts = [], {"shards": 0, "rows": 0, "searched": 0, "all_tie": 0}
    for shard in sorted((chunk / "train").iterdir()):
        load = lambda name: np.load(shard / f"{name}.npy", mmap_mode="r")
        root, values, offsets = load("labels.root"), load("labels.values"), load("labels.legal_offsets")
        counts["shards"] += 1
        counts["rows"] += len(root)
        for row in np.flatnonzero(np.asarray(root) == searched):
            gap = top_two_gap(values[offsets[row]:offsets[row + 1]])
            counts["searched"] += 1
            if gap is None:
                counts["all_tie"] += 1
            else:
                gaps.append(gap)
    return gaps, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", type=Path, required=True, help="the pilot chunk directory (shards 1000-1007)")
    parser.add_argument("--dataset", type=Path, help="the dataset manifest, for delta 12's gradient steps")
    args = parser.parse_args(argv)
    gaps, counts = _pilot_gaps(args.pilot)
    result = {"schema": "spire-learn-distill-tau-v1", "pilot": str(args.pilot), "counts": counts,
              "positive_gaps": sum(g > 0 for g in gaps), "percentile_method": "linear", "floor": TAU_FLOOR,
              "tau": tau_candidates(gaps)}
    if args.dataset is not None:
        from spire_learn.neural.distill_train import open_dataset

        dataset = open_dataset(args.dataset)
        train_rows = sum(ref.rows for ref in dataset.train)
        result.update(dataset=str(args.dataset), train_rows=train_rows, batch_rows=BATCH_ROWS, passes=PASSES,
                      gradient_steps=gradient_steps(train_rows))
    text = json.dumps(result, indent=1, sort_keys=True)
    output = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if output:
        (Path(output) / "tau.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
