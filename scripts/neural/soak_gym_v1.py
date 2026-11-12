#!/usr/bin/env python3
"""Bounded random-legal-action soak of Learn's Gym 1.0.0 public drivers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[2]
_MODES = ("standalone", "run", "standard-standalone", "standard-run")
_CHARACTERS = ("ironclad", "silent", "defect", "watcher")


class _NoTrainingImports:
    def __init__(self):
        self.attempted: list[str] = []

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname.split(".", 1)[0] in {"torch", "torchrl", "tensordict"}:
            self.attempted.append(fullname)
            raise AssertionError("training import during Gym 1.0.0 soak: " + fullname)
        return None


def _entry_ids(cases: Any, mode: str, rng: random.Random, batch_size: int) -> list[str | None]:
    mode_cases = cases.for_mode(mode)
    if mode in {"run", "standard-run"}:
        return [None] * batch_size
    return [rng.choice(mode_cases).entry_id for _ in range(batch_size)]


def _drive(driver: Any, *, mode: str, steps: int, rng: random.Random,
           entry_ids: Callable[[], list[str | None]]) -> dict[str, int | str]:
    stats: Counter[str] = Counter()
    standalone = mode in {"standalone", "standard-standalone"}
    batch = driver.batch_size
    rows = driver.reset(seeds=[rng.randrange(2**31) for _ in range(batch)], entry_ids=entry_ids())
    for _ in range(steps):
        actions = []
        for row in rows:
            legal = ([candidate.index for candidate in row.candidates]
                     if row.lifecycle.active else list(row.noncombat_indices))
            if not legal:
                raise RuntimeError("driver row exposed no legal action")
            actions.append(rng.choice(legal))
        transitions = driver.step(actions)
        stats["steps"] += 1
        for transition in transitions:
            if transition.terminated:
                stats["terminated"] += 1
            lifecycle = transition.lifecycle
            if (lifecycle.last_exit is not None
                    and transition.source.lifecycle.combats_settled != lifecycle.combats_settled):
                stats["settled_" + lifecycle.last_exit["outcome"]] += 1
            stats["active" if lifecycle.active else "noncombat"] += 1
            if lifecycle.routing == "combat_selection":
                stats["selection"] += 1
        rows = [transition.successor for transition in transitions]
        done = [
            transition.terminated or transition.truncated
            or (standalone and not transition.lifecycle.active)
            for transition in transitions
        ]
        if any(done):
            reset_mask = done if standalone else [True] * batch
            rows = driver.reset(
                seeds=[rng.randrange(2**31) for _ in range(batch)],
                entry_ids=entry_ids(), reset_mask=reset_mask,
            )
            stats["resets"] += 1
    return {"status": "ok", **dict(stats)}


def run(output: Path, *, steps: int, characters: bool = False,
        root: Path = _ROOT) -> dict[str, object]:
    """Run the four configured modes and optionally eight Standard Run variants."""
    guard = _NoTrainingImports()
    sys.meta_path.insert(0, guard)
    try:
        from spire_learn.neural import envs
        from spire_learn.neural.config import load_neural_config

        result: dict[str, object] = {"schema": "spire-learn-gym-v1-soak-v1", "steps_per_mode": steps,
                                    "modes": {}, "standard_runs": {}}
        runtime = envs._load_native_runtime() if characters else None
        for mode in _MODES:
            config = load_neural_config(root / "configs/neural/gym-v1.0.0" / f"{mode}.toml")
            if config.mode != mode:
                raise ValueError(f"{mode} template selects {config.mode}")
            driver = envs.make_driver(config, config.cases)
            rng = random.Random(11)
            try:
                get_entries = lambda c=config, d=driver, r=rng, m=mode: _entry_ids(
                    c.cases, m, r, d.batch_size)
                result["modes"][mode] = _drive(
                    driver, mode=mode, steps=steps, rng=rng, entry_ids=get_entries)
            finally:
                driver.close()
        if characters:
            assert runtime is not None
            config = load_neural_config(root / "configs/neural/gym-v1.0.0/standard-run.toml")
            for character in _CHARACTERS:
                for ascension in (0, 20):
                    label = f"{character}-a{ascension}"
                    case = SimpleNamespace(character=character, ascension=ascension, creation=None)
                    driver = envs._make_standard_run_driver(config, runtime, case)
                    rng = random.Random(f"{character}-a{ascension}")
                    try:
                        result["standard_runs"][label] = _drive(
                            driver, mode="standard-run", steps=steps, rng=rng,
                            entry_ids=lambda: [None],
                        )
                    finally:
                        driver.close()
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "soak.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        sys.meta_path.remove(guard)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--characters", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    run(args.output, steps=args.steps, characters=args.characters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
