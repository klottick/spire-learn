#!/usr/bin/env python3.12
"""Export a distilled S1 student as a PPO warm-start checkpoint (runtime container; numpy + torch).

Run it through ``runtime_container.py`` on the CPU, in two steps:

    distill_export.py critic-scale --dataset DATASET.json --template TEMPLATE.toml --output OUTPUT
    distill_export.py export --checkpoint S1_CHECKPOINT --template TEMPLATE.toml
                             --critic-scale OUTPUT/critic-scale.json --output OUTPUT [--descriptor COMPACT.json]

``critic-scale`` computes the critic scale once (delta 18) from the dataset's training shards:
``c = mean(gamma ** remaining)`` over the value rows (``labels.value_ok``: teacher fights that were
not censored; mixed and censored fights carry no value target), where ``remaining = (fight_rows - 1)
- row_in_fight`` counts the decisions to the fight's end (0 on its last decision). A fight is a
contiguous run of rows with one ``labels.case``; both counts come from ``labels.case`` and row order.
``gamma`` is the template's PPO discount. The record (``OUTPUT/critic-scale.json``) holds ``c`` and
what it was computed from; the same record is used for every template.

``export`` runs once per PPO template. It refuses anything but an S1 distillation checkpoint: the
distillation objective, the Standard FF learner ids, the template's raw-mode mapping and the
template's FF model with ``value_heads = 3``, trained on exactly the shards the record covers. It
copies the trunk, the policy head and the outcome head into the PPO policy's ``value_head``, then
multiplies that head's weight and bias by ``c`` (the outcome head is undiscounted; PPO's critic is
discounted). The two auxiliary heads (``aux_value_head``: HP kept, realised v2) are dropped. The
checkpoint carries the template's identity (``spire-learn-combat-ff-standard-v1``,
``spire-learn-combat-features-standard-v1``, ``spire-learn-combat-dispatch-v1``,
``spire-learn-combat-winloss-v1``) with its model and PPO hashes, a fresh Adam optimizer, the
template's freshly seeded RNG streams, zero training counters, and ``parent_checkpoint`` with the
source checkpoint's path and ``checkpoint_sha256``. ``OUTPUT`` (absolute; ``SPIRE_LEARN_RUNNER_OUTPUT``
when set) may hold the runner's logs but no earlier export; it receives
``checkpoint/`` (the PPO checkpoint, preflighted for ``warm_start`` and ``evaluate`` before the
entry returns) and ``export.json`` (the export record). Faults stop the export; nothing is retried.

The outcome head was fit on teacher play, not student play; the scale corrects only the discount.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / "src"), str(_repo)]

from scripts.neural.baseline_entry import checkpoint_sha256  # noqa: E402  (after the sys.path set-up)

CRITIC_SCALE_SCHEMA = "spire-learn-distill-critic-scale-v1"
EXPORT_SCHEMA = "spire-learn-distill-export-v1"
CRITIC_SCALE_FILE = "critic-scale.json"
CRITIC_RULE = ("mean(gamma ** remaining) over labels.value_ok rows of the training shards; "
               "remaining = (fight_rows - 1) - row_in_fight, a fight being a contiguous labels.case run")
DROPPED_PREFIX = "aux_value_head."
SCALED_TENSORS = ("value_head.weight", "value_head.bias")
_RNG_DOMAINS = ("policy", "minibatch", "curriculum", "controller")


class ExportError(ValueError):
    """The source is not an exportable S1 student, or the export's inputs disagree."""


# --- the critic scale -----------------------------------------------------------------------

def critic_scale_details(shards: Sequence, *, gamma: float = 0.99) -> dict:
    """``c`` and its counts over the shards (paths or ``ShardReader``s), in the given order."""
    import numpy as np

    from spire_learn.neural.distill_data import ShardReader, _file_sha256

    if type(gamma) is not float or not 0 < gamma <= 1:
        raise ExportError("gamma must be a float in (0, 1]")
    terms: list[float] = []
    seen: set[tuple[int, int]] = set()
    fights = value_fights = 0
    for item in shards:
        reader = item if isinstance(item, ShardReader) else ShardReader(Path(item))
        try:
            if reader.rows == 0:
                continue  # a zero-row shard lists no files
            for name in ("labels.case", "labels.value_ok"):  # the two files read, against the shard manifest
                if _file_sha256(reader.path / f"{name}.npy") != reader.manifest["files"][name]["sha256"]:
                    raise ExportError(f"shard {reader.path.name} file {name} fails its sha256")
            # Copies: closing the reader unmaps its files.
            case = np.array(reader.array("labels.case"), dtype=np.int64)
            value_ok = np.array(reader.array("labels.value_ok"), dtype=np.bool_)
        finally:
            if reader is not item:
                reader.close()
        rows = len(value_ok)
        if rows != reader.rows or case.shape != (rows, 2):
            raise ExportError("labels.case and labels.value_ok must hold one entry per row")
        change = np.ones(rows, dtype=np.bool_)
        change[1:] = np.any(case[1:] != case[:-1], axis=1)
        starts = np.flatnonzero(change)
        lengths = np.diff(np.append(starts, rows))
        for start in starts.tolist():
            key = (int(case[start, 0]), int(case[start, 1]))
            if key in seen:
                raise ExportError(f"fight {key} is not one contiguous run of rows")
            seen.add(key)
        fight = np.cumsum(change) - 1
        remaining = lengths[fight] - 1 - (np.arange(rows) - starts[fight])
        flags = value_ok.astype(np.int8)
        if np.any(np.minimum.reduceat(flags, starts) != np.maximum.reduceat(flags, starts)):
            raise ExportError("value_ok must be the same for every row of a fight")
        fights += len(starts)
        value_fights += int(value_ok[starts].sum())
        terms.extend(np.power(np.float64(gamma), remaining[value_ok].astype(np.float64)).tolist())
    if not terms:
        raise ExportError("the shards hold no value rows")
    return {"critic_scale": math.fsum(terms) / len(terms), "gamma": gamma, "value_rows": len(terms),
            "fights": fights, "value_fights": value_fights, "rule": CRITIC_RULE}


def compute_critic_scale(shards: Sequence, *, gamma: float = 0.99) -> float:
    """``c = mean(gamma ** remaining)`` over the value rows of ``shards`` (delta 18)."""
    return critic_scale_details(shards, gamma=gamma)["critic_scale"]


# --- the S1 check (stdlib) ------------------------------------------------------------------

def _template(template):
    from spire_learn.neural.config import NeuralConfig, load_neural_config

    config = template if isinstance(template, NeuralConfig) else load_neural_config(Path(template))
    return config


def _require_ppo_template(config) -> None:
    from spire_learn.neural.config import STANDARD_LEARNER_IDENTITY_IDS, WINLOSS_OBJECTIVE_ID

    identity = config.accepted_identities[0] if len(config.accepted_identities) == 1 else None
    if (config.ppo is None or config.mode != "standard-standalone" or identity is None
            or (identity.model_id, identity.features_id, identity.action_id) != STANDARD_LEARNER_IDENTITY_IDS
            or identity.objective_id != WINLOSS_OBJECTIVE_ID or config.model.value_heads != 1
            or set(identity.config_hashes) != {"model", "ppo"}):
        raise ExportError("the template must be a standard-standalone PPO config with one Standard FF "
                          "win/loss identity and one critic")


def s1_identity(config, distill_hash: str):
    """The distillation identity an S1 student of this template carries."""
    from spire_learn.neural.config import DISTILL_OBJECTIVE_ID, model_config_hash
    from spire_learn.neural.contracts import NeuralIdentity

    template = config.accepted_identities[0]
    return NeuralIdentity(
        model_id=template.model_id, features_id=template.features_id, action_id=template.action_id,
        objective_id=DISTILL_OBJECTIVE_ID, accepted_raw_mode_mappings=dict(template.accepted_raw_mode_mappings),
        config_hashes={"model": model_config_hash(replace(config.model, value_heads=3)), "distill": distill_hash})


def check_s1(distill_ckpt: Path, config):
    """Preflight ``distill_ckpt`` as an S1 student of ``config``; returns (preflight, identity). Stdlib only."""
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.config import DISTILL_HASH_KEY, DISTILL_OBJECTIVE_ID

    _require_ppo_template(config)
    path = Path(distill_ckpt)
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ExportError(f"{path} is not a checkpoint directory")
    try:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))["identity"]
        hashes = saved["config_hashes"]
    except (ValueError, KeyError, TypeError) as error:
        raise ExportError(f"{path} has no readable checkpoint identity") from error
    if saved.get("objective_id") != DISTILL_OBJECTIVE_ID:
        raise ExportError(f"only a distillation checkpoint exports; {path} has objective "
                          f"{saved.get('objective_id')!r}")
    if not isinstance(hashes, dict) or set(hashes) != {"model", DISTILL_HASH_KEY}:
        raise ExportError("an S1 checkpoint identity carries exactly the model and distill hashes")
    expected = s1_identity(config, hashes[DISTILL_HASH_KEY])
    try:
        # Model, features, action, objective, raw-mode mapping and model hash (the template's FF with 3 heads).
        preflight = preflight_checkpoint(path, expected, purpose="evaluate")
    except ValueError as error:
        raise ExportError(f"{path} is not an S1 student of this template: {error}") from error
    if preflight.metadata["training_state"] is None:
        raise ExportError("an S1 checkpoint records its distillation training state")
    return preflight, expected


# --- the export -----------------------------------------------------------------------------

def _progress_counters() -> dict:
    from spire_learn.neural.runner import _PROGRESS_FIELDS

    return {**{name: 0 for name in sorted(_PROGRESS_FIELDS)}, "elapsed_seconds": 0.0}


def fresh_rng_state(config, torch) -> dict:
    """The four RNG streams exactly as a fresh PPO runtime of ``config`` seeds them (CPU generators)."""
    seeds = config.seed_streams
    policy = torch.Generator(device="cpu")
    policy.manual_seed(seeds.policy)
    minibatch = torch.Generator(device="cpu")
    minibatch.manual_seed(seeds.optimizer)
    return {"policy": policy.get_state(), "minibatch": minibatch.get_state(),
            "curriculum": random.Random(seeds.curriculum).getstate(),
            "controller": random.Random(seeds.noncombat).getstate()}


def _check_critic(c: float, record: Mapping | None, config, source_metadata: Mapping) -> None:
    if type(c) is not float or not math.isfinite(c) or not 0 < c <= 1:
        raise ExportError("the critic scale c must be a finite float in (0, 1]")
    if record is None:
        return
    if record.get("critic_scale") != c:
        raise ExportError("c disagrees with its critic-scale record")
    if record.get("gamma") != config.ppo.gamma:
        raise ExportError("the critic scale was computed with a different gamma than the template's")
    run_metadata = source_metadata.get("run_metadata") or {}
    if (list(run_metadata.get("train_shards", ())) != list(record.get("train_shards", ()))
            or run_metadata.get("dataset_sha256") != record.get("dataset_sha256")):
        raise ExportError("the S1 student was trained on other shards than the critic scale covers")


def export_ppo_checkpoint(distill_ckpt, template, c: float, *, output: Path, schema,
                          critic_record: Mapping | None = None) -> Path:
    """Write the PPO warm-start checkpoint of an S1 student to ``output`` (a new directory); returns it."""
    from spire_learn.neural.checkpoint import identity_payload, load_checkpoint, preflight_checkpoint, save_checkpoint
    from spire_learn.neural.config import DISTILL_OBJECTIVE_ID

    config = _template(template)
    source = Path(distill_ckpt)
    preflight, expected = check_s1(source, config)
    _check_critic(c, critic_record, config, preflight.metadata)
    if schema.schema_fingerprint != config.standard_view_fingerprint:
        raise ExportError("the compact schema's fingerprint differs from the template's")
    output = Path(output)
    if os.path.lexists(output):
        raise ExportError(f"{output} already exists")
    source_sha256 = checkpoint_sha256(source)

    import torch

    from spire_learn.neural.model import build_policy

    identity = config.accepted_identities[0]
    names = dict(model_id=identity.model_id, features_id=identity.features_id)
    loaded = load_checkpoint(source, expected, purpose="evaluate", device="cpu")
    student = build_policy(schema, replace(config.model, value_heads=3), seed=config.seed_streams.policy,
                           device="cpu", objective_id=DISTILL_OBJECTIVE_ID, **names)
    student.load_state_dict(dict(loaded.model_tensors), strict=True)  # exactly the S1 architecture
    tensors = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()
               if not name.startswith(DROPPED_PREFIX)}
    dropped = sorted(name for name in student.state_dict() if name.startswith(DROPPED_PREFIX))
    with torch.no_grad():
        for name in SCALED_TENSORS:
            tensors[name] = tensors[name] * c
    policy = build_policy(schema, config.model, seed=config.seed_streams.policy, device="cpu", **names)
    policy.load_state_dict(tensors, strict=True)  # trunk, policy head and the scaled outcome head
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate, eps=config.ppo.adam_eps)
    counters = _progress_counters()
    run_metadata = {
        "operation": "distill-export",
        "mode": config.mode,
        "config_sha256": config.config_sha256,
        "environment_revision": config.environment_revision,
        "standard_view_fingerprint": config.standard_view_fingerprint,
        "policy_kind": "checkpoint",
        "controller_id": "distill-export-v1",
        "critic_scale": c,
        "critic_scale_rule": CRITIC_RULE,
        "gamma": config.ppo.gamma,
        "scaled_tensors": list(SCALED_TENSORS),
        "dropped_tensors": dropped,
        "source_identity": dict(preflight.metadata["identity"]),
        "source_step": (preflight.metadata["run_metadata"] or {}).get("step"),
    }
    save_checkpoint(
        output, policy=policy, optimizer=optimizer, rng_state=fresh_rng_state(config, torch),
        metadata={
            "identity": identity_payload(identity),
            "run_metadata": json.loads(json.dumps(run_metadata, default=_json_default)),
            "training_state": {
                "optimizer_step": 0,
                "hyperparameter_id": identity.config_hashes["ppo"],
                "rng_domains": list(_RNG_DOMAINS),
                "counters": counters,
                "update_id": "update-0",
                "behavior_policy_version": 0,
                "next_environment_seed_ordinal": 0,
                "next_decision_sequence": 0,
                "saved_inflight_record_count": 0,
            },
            "parent_checkpoint": {"path": str(source), "sha256": source_sha256},
        })
    for purpose in ("warm_start", "evaluate"):
        preflight_checkpoint(output, identity, purpose=purpose)
    return output


def _json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


# --- the entry ------------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, document: Mapping) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


def _check_runner_output(output: Path) -> None:
    if not output.is_absolute():
        raise ExportError("--output must be an absolute path")
    runner = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if runner is not None and Path(runner).resolve() != output.resolve():
        raise ExportError("--output must be SPIRE_LEARN_RUNNER_OUTPUT")


def critic_scale_record(dataset_path: Path, template) -> dict:
    """The critic-scale record over a dataset manifest's training shards."""
    from spire_learn.neural.distill_train import open_dataset

    config = _template(template)
    _require_ppo_template(config)
    dataset = open_dataset(dataset_path)
    details = critic_scale_details([ref.path for ref in dataset.train], gamma=config.ppo.gamma)
    identity = dataset.identity()
    return {"schema": CRITIC_SCALE_SCHEMA, **details,
            "dataset_manifest": str(dataset_path), "dataset_manifest_sha256": dataset.manifest_sha256,
            "dataset_sha256": identity["dataset_sha256"], "train_shards": identity["train_shards"],
            "train_rows": identity["train_rows"],
            "template": None if template is config else str(template), "template_config_sha256": config.config_sha256}


def _load_critic_record(path: Path) -> dict:
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict) or record.get("schema") != CRITIC_SCALE_SCHEMA:
        raise ExportError("unsupported critic-scale record")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    scale = commands.add_parser("critic-scale", help="compute c once from the dataset's training shards")
    scale.add_argument("--dataset", type=Path, required=True, help="dataset manifest (build_dataset_manifest)")
    scale.add_argument("--template", type=Path, required=True, help="a PPO template (its gamma)")
    scale.add_argument("--output", type=Path, required=True, help="directory for critic-scale.json")
    export = commands.add_parser("export", help="export one S1 checkpoint for one PPO template")
    export.add_argument("--checkpoint", type=Path, required=True, help="the S1 distillation checkpoint")
    export.add_argument("--template", type=Path, required=True)
    export.add_argument("--critic-scale", type=Path, required=True, help="the critic-scale record")
    export.add_argument("--output", type=Path, required=True, help="directory for checkpoint/ and export.json")
    export.add_argument("--descriptor", type=Path, help="captured compact descriptor JSON (default: the Gym's)")
    args = parser.parse_args(argv)
    from scripts.neural.distill_train_entry import _git_state, _gym_state, load_schema

    try:
        _check_runner_output(args.output)
        if args.command == "critic-scale":
            record = critic_scale_record(args.dataset, args.template)
            record["provenance"] = {"learner": _git_state(), "gym": _gym_state()}
            args.output.mkdir(parents=True, exist_ok=True)
            _write_new(args.output / CRITIC_SCALE_FILE, record)
            print(json.dumps({"critic_scale": record["critic_scale"], "value_rows": record["value_rows"]}))
            return 0
        if any(os.path.lexists(args.output / name) for name in ("checkpoint", "export.json")):
            raise ExportError("--output already holds an export")
        record = _load_critic_record(args.critic_scale)
        config = _template(args.template)
        schema = load_schema(config, args.descriptor)
        args.output.mkdir(parents=True, exist_ok=True)
        checkpoint = export_ppo_checkpoint(args.checkpoint, config, record["critic_scale"],
                                           output=args.output / "checkpoint", schema=schema,
                                           critic_record=record)
        import torch

        _write_new(args.output / "export.json", {
            "schema": EXPORT_SCHEMA, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "source": str(args.checkpoint), "source_sha256": checkpoint_sha256(args.checkpoint),
            "template": str(args.template), "template_sha256": _file_sha256(args.template),
            "config_sha256": config.config_sha256,
            "identity": {"model_id": config.accepted_identities[0].model_id,
                         "features_id": config.accepted_identities[0].features_id,
                         "action_id": config.accepted_identities[0].action_id,
                         "objective_id": config.accepted_identities[0].objective_id,
                         "config_hashes": dict(config.accepted_identities[0].config_hashes)},
            "seeds": {"policy": config.seed_streams.policy, "optimizer": config.seed_streams.optimizer,
                      "curriculum": config.seed_streams.curriculum, "noncombat": config.seed_streams.noncombat},
            "critic_scale": record["critic_scale"], "critic_scale_record": str(args.critic_scale),
            "critic_scale_record_sha256": _file_sha256(args.critic_scale),
            "descriptor": None if args.descriptor is None else str(args.descriptor),
            "provenance": {"learner": _git_state(), "gym": _gym_state(), "torch_version": torch.__version__},
            "note": "the outcome head was fit on teacher play, not student play; c corrects only the discount",
        })
    except (OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"checkpoint": str(checkpoint), "critic_scale": record["critic_scale"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
