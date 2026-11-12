"""One-reset Gym 1.0.0 adapter qualification; no training or gameplay steps."""

from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import random
import sys
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
_MODES = ("standalone", "run", "standard-standalone", "standard-run")


class _NoTrainingImports(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.attempted: list[str] = []

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname.split(".", 1)[0] in {"torch", "torchrl", "tensordict"}:
            self.attempted.append(fullname)
            raise AssertionError("training import during Gym 1.0.0 qualification: " + fullname)
        return None


def _entry_ids(config: Any) -> list[str | None]:
    """Select cases in the same slot order as Collector's first training reset."""
    mode_cases = config.cases.for_mode(config.mode)
    if not mode_cases:
        raise AssertionError(f"config has no cases for {config.mode}")
    if config.mode in {"run", "standard-run"}:
        return [None] * config.rollout_rows
    if config.split == "train":
        rng = random.Random(config.seed_streams.curriculum)
        return [mode_cases[rng.randrange(len(mode_cases))].entry_id
                for _ in range(config.rollout_rows)]
    return [mode_cases[index % len(mode_cases)].entry_id
            for index in range(config.rollout_rows)]


def _check_build(info: dict[str, object], contract: Any) -> dict[str, object]:
    expected = contract.build
    if expected is None:
        raise AssertionError("Gym 1.0.0 contract is missing its build identity")
    wanted = {
        "extension": "spire-python",
        "revision": expected.revision,
        "dirty": False,
        "profile": expected.profile,
        "features": list(expected.features),
        "source_digest": expected.source_digest,
        "file_digest": expected.file_digest,
    }
    for key, value in wanted.items():
        actual = list(info[key]) if key == "features" and isinstance(info.get(key), tuple) else info.get(key)
        if actual != value:
            raise AssertionError(f"Gym build_info {key} does not match GYM_V1_CONTRACT.build")
    if info.get("dirty") is not False:
        raise AssertionError("Gym native build must report dirty=False")
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in info.items()}


def run(output: Path) -> dict[str, object]:
    """Run one public reset on each authored Gym 1.0.0 mode and save its record."""
    guard = _NoTrainingImports()
    sys.meta_path.insert(0, guard)
    try:
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.envs import _load_native_runtime, make_driver
        from spire_learn.neural.gym_contracts import GYM_V1_CONTRACT

        runtime = _load_native_runtime()
        build_info = _check_build(dict(runtime.native.build_info()), GYM_V1_CONTRACT)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        mode_records: list[dict[str, object]] = []
        close_counts: dict[str, int] = {}
        for mode in _MODES:
            config = load_neural_config(_ROOT / "configs" / "neural" / "gym-v1.0.0" / f"{mode}.toml")
            if config.mode != mode or config.environment_revision != GYM_V1_CONTRACT.source_revision:
                raise AssertionError(f"{mode} config does not select the Gym 1.0.0 contract")
            driver = make_driver(config, config.cases)
            try:
                expected_driver_mode = "run" if mode.endswith("run") else "standalone"
                if driver.mode != expected_driver_mode or driver.batch_size != config.rollout_rows:
                    raise AssertionError(f"{mode} driver dimensions disagree with its config")
                rows = driver.reset(
                    seeds=[0] * driver.batch_size,
                    entry_ids=_entry_ids(config),
                )
                if not isinstance(rows, tuple) or len(rows) != driver.batch_size:
                    raise AssertionError(f"{mode} reset did not return a complete parsed row tuple")
                candidate_counts: list[int] = []
                noncombat_counts: list[int] = []
                active_rows = 0
                for slot, row in enumerate(rows):
                    if row is None or row.frame is None or row.lifecycle is None:
                        raise AssertionError(f"{mode} reset row {slot} lacks a parsed frame or lifecycle")
                    if row.frame.schema_fingerprint != driver.schema.schema_fingerprint:
                        raise AssertionError(f"{mode} reset row {slot} fingerprint disagrees with driver schema")
                    candidates = len(row.candidates)
                    noncombat = len(row.noncombat_indices)
                    if candidates == 0 and noncombat == 0:
                        raise AssertionError(f"{mode} reset row {slot} has no public legal actions")
                    candidate_counts.append(candidates)
                    noncombat_counts.append(noncombat)
                    active_rows += int(row.lifecycle.active)
                mode_records.append({
                    "mode": mode,
                    "config_sha256": config.config_sha256,
                    "schema_fingerprint": driver.schema.schema_fingerprint,
                    "view_id": driver.schema.view.view_id,
                    "batch_size": driver.batch_size,
                    "active_rows": active_rows,
                    "candidate_counts": candidate_counts,
                    "noncombat_counts": noncombat_counts,
                })
            finally:
                driver.close()
                close_counts[mode] = close_counts.get(mode, 0) + 1
        if close_counts != {mode: 1 for mode in _MODES}:
            raise AssertionError("each Gym driver must be closed exactly once")
        if guard.attempted:
            raise AssertionError("training imports were attempted during qualification")
        record: dict[str, object] = {
            "status": "qualified_one_reset_per_mode",
            "build_info": build_info,
            "modes": mode_records,
            "reset_calls": len(mode_records),
            "step_calls": 0,
            "training_import_attempts": guard.attempted,
            "close_counts": close_counts,
        }
        with (output / "qualification.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps(record, sort_keys=True, indent=2))
        return record
    finally:
        sys.meta_path.remove(guard)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory for qualification.json")
    args = parser.parse_args(argv)
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
