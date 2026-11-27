#!/usr/bin/env python3.12
"""Train one distillation student from labelled decision shards (runtime container; numpy + torch).

Run it through ``runtime_container.py`` (with ``--gpu <uuid>`` for a CUDA config):

    distill_train_entry.py --config CONFIG.toml --output OUTPUT [--descriptor COMPACT.json]
                           [--gpu-wait MIN_FREE_BYTES:POLL_SECONDS:MAX_TOTAL_SECONDS]

``CONFIG`` is a ``load_distill_config`` TOML; its ``[data]`` table names the label chunk dirs or a
dataset manifest. ``OUTPUT`` is a fresh (or empty) directory under the config's ``output_root``; when
``SPIRE_LEARN_RUNNER_OUTPUT`` is set it must be that directory. The compact combat descriptor comes
from ``--descriptor`` (a captured descriptor JSON) or, by default, from the installed Gym without a
reset; either way it must match the config's fingerprint and every chunk's observation schema.

``gpu_wait`` yields the shared GPU: before training and at every validation interval the entry
waits until ``MIN_FREE_BYTES`` are free, polling every ``POLL_SECONDS``, for at most
``MAX_TOTAL_SECONDS`` in total; waits are excluded from GPU time and reported. Faults stop the run
(exit 1); nothing is retried.

Writes ``train_log.jsonl`` (per-step and per-validation events), ``summary.json`` (rows per second,
GPU time and waits, peak VRAM and RSS, every validation, the chosen and final checkpoints) and the
kept checkpoints ``checkpoint-best`` (when it differs from the final one) and ``checkpoint-final``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / "src"), str(_repo)]

_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = "summary.json"
# The descriptor capture's Ironclad entry (no reset, no step).
_CAPTURE_ENTRY = {
    "id": "distill-train-descriptor", "character": "ironclad", "ascension": 0,
    "encounter": "cultist", "hp": 80, "max_hp": 80, "gold": 99,
    "deck": [{"card": card, "upgrades": 0} for card in ["strike"] * 5 + ["defend"] * 4 + ["bash"]],
    "relics": ["burning_blood"], "potions": [None] * 3,
}


class GpuWaitExhausted(RuntimeError):
    """The shared GPU stayed busy for the whole allowed wait."""


class GpuGate:
    """Wait (never fault) while other services hold the GPU; inactive on CPU or without settings."""

    def __init__(self, settings: tuple[int, int, int] | None, *, device: str,
                 free_bytes: Callable[[], int], release: Callable[[], None],
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 journal: Callable[[Mapping[str, object]], None] = lambda event: None):
        self._settings = settings if device.startswith("cuda") else None
        self._free, self._release, self._sleep, self._clock, self._journal = free_bytes, release, sleep, clock, journal
        self.waited_seconds = 0.0
        self.events: list[dict[str, object]] = []

    def wait(self, stage: str) -> None:
        if self._settings is None:
            return
        minimum, poll, cap = self._settings
        started = None
        while True:
            free = self._free()
            if free >= minimum:
                break
            if started is None:
                started = self._clock()
                self._note({"event": "gpu_wait", "stage": stage, "free_bytes": free, "min_free_bytes": minimum})
            if self.waited_seconds + (self._clock() - started) >= cap:
                self.waited_seconds += self._clock() - started
                raise GpuWaitExhausted(f"gpu_wait exhausted at {stage}")
            self._sleep(poll)
        if started is not None:
            waited = self._clock() - started
            self.waited_seconds += waited
            self._note({"event": "gpu_resumed", "stage": stage, "waited_seconds": round(waited, 3),
                        "total_waited_seconds": round(self.waited_seconds, 3)})

    def release(self) -> None:
        if self._settings is not None:
            self._release()

    def _note(self, event: dict[str, object]) -> None:
        self.events.append(event)
        self._journal(event)


def parse_gpu_wait(text: str) -> tuple[int, int, int]:
    parts = text.split(":")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError("--gpu-wait is MIN_FREE_BYTES:POLL_SECONDS:MAX_TOTAL_SECONDS (positive)")
    return values  # type: ignore[return-value]


def _git_state() -> dict[str, object]:
    try:
        head = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(_ROOT), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=30, check=True).stdout.strip())
        return {"revision": head, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "note": "git unavailable in this process"}


def _gym_state() -> dict[str, object]:
    try:
        from spire_lab import _native
    except ImportError:
        return {"build": None, "note": "Spire Gym not importable"}
    info = dict(_native.build_info())
    return {"build": {key: info.get(key) for key in ("revision", "dirty", "source_digest", "file_digest")}}


def live_descriptor(preset: str) -> dict:
    """The installed Gym's compact Standard combat descriptor (constructs an env; never resets)."""
    from spire_lab.gymnasium import StandardCombatVectorEnv

    env = StandardCombatVectorEnv({
        "catalog_schema_id": "spire-gym-standard-combat-catalog-v1",
        "profile_id": "standard-combat-v1", "entries": [_CAPTURE_ENTRY],
    }, 1, observation="compact", capacity_preset=preset)
    try:
        return env.observation_spec()
    finally:
        env.close()


def load_schema(config, descriptor_path: Path | None):
    from spire_learn.neural.schema import parse_combat_descriptor

    descriptor = (json.loads(descriptor_path.read_text()) if descriptor_path is not None
                  else live_descriptor(config.standard_view_preset))
    schema = parse_combat_descriptor(descriptor, revision=config.environment_revision)
    if schema.schema_fingerprint != config.standard_view_fingerprint:
        raise ValueError("the compact descriptor's fingerprint differs from the config")
    return schema


def check_observation(dataset, schema) -> None:
    """Every chunk records the observation schema the student is built for."""
    expected = {"schema_id": schema.schema_id, "schema_fingerprint": schema.schema_fingerprint,
                "descriptor_digest": schema.descriptor_digest, "gym_revision": schema.gym_revision}
    for chunk in dataset.chunks:
        manifest = json.loads((Path(chunk) / "chunk-manifest.json").read_text())
        if manifest.get("observation") != expected:
            raise ValueError(f"{chunk} was labelled under a different observation schema")


def _check_output(output: Path, config) -> None:
    if not output.is_absolute():
        raise ValueError("--output must be an absolute path")
    if not output.resolve().is_relative_to(config.output_root):
        raise ValueError(f"--output must be under the config's output_root {config.output_root}")
    runner = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if runner is not None and Path(runner).resolve() != output.resolve():
        raise ValueError("--output must be SPIRE_LEARN_RUNNER_OUTPUT")
    if output.exists() and any(output.iterdir()):
        raise ValueError("--output must be a fresh or empty directory")


def _write_summary(output: Path, summary: Mapping[str, object]) -> None:
    path = output / SUMMARY
    with path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=1, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, help="captured compact descriptor JSON (default: the Gym's)")
    parser.add_argument("--gpu-wait", type=parse_gpu_wait, metavar="MIN_FREE_BYTES:POLL_SECONDS:MAX_TOTAL_SECONDS")
    args = parser.parse_args(argv)

    from spire_learn.neural.config import load_distill_config
    from spire_learn.neural.device import (
        plan_runtime_device, qualify_runtime_device, require_runtime_execution_gate,
    )
    from spire_learn.neural.distill_train import dataset_chunks, discover_dataset, train_student

    config = load_distill_config(args.config)
    _check_output(args.output, config)
    plan = plan_runtime_device(config)
    require_runtime_execution_gate(plan)
    dataset = discover_dataset(dataset_chunks(config))
    schema = load_schema(config, args.descriptor)
    check_observation(dataset, schema)

    import torch

    gate = GpuGate(args.gpu_wait, device=plan.device,
                   free_bytes=lambda: int(torch.cuda.mem_get_info(0)[0]),
                   release=torch.cuda.empty_cache)
    started = time.monotonic()
    cpu_started = time.process_time()
    gate.wait("qualify")
    device_report = qualify_runtime_device(torch, plan)
    provenance = {"learner": _git_state(), "gym": _gym_state(), "device": device_report,
                  "config": str(config.source), "descriptor": None if args.descriptor is None else str(args.descriptor),
                  "torch_version": torch.__version__,
                  "note": "Supervised training is seeded; GPU nondeterminism is not claimed away."}
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        summary = train_student(config, schema, dataset, args.output, device=plan.device, gate=gate,
                                provenance=provenance)
    except BaseException as error:
        failure = {"schema": "spire-learn-distill-train-summary-v1", "status": "failed",
                   "error": f"{type(error).__name__}: {error}", "provenance": provenance,
                   "gpu_waits": gate.events, "gpu_wait_seconds": gate.waited_seconds}
        if not (args.output / SUMMARY).exists():
            _write_summary(args.output, failure)
        if isinstance(error, Exception):
            print(failure["error"], file=sys.stderr)
            return 1
        raise
    summary["gpu_waits"] = gate.events
    summary["timing"]["cpu_seconds"] = time.process_time() - cpu_started
    summary["timing"]["entry_wall_seconds"] = time.monotonic() - started
    _write_summary(args.output, summary)
    print(json.dumps({"status": summary["status"], "chosen": summary["chosen"],
                      "rows_per_second": summary["timing"]["rows_per_second"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
