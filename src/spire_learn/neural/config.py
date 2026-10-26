"""Strict pure-Python dual-mode neural experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

from spire_learn.contracts import ContractError
from spire_learn.neural.cases import (
    NeuralCases,
    load_neural_cases,
    validate_neural_case_lineage,
)
from spire_learn.neural.contracts import ModelConfig, NeuralIdentity, PPOConfig


ACCEPTED_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
COMMON_VIEW_ID = "spire-gym-combat-policy-view-v1"
COMMON_VIEW_FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"
RUN_PROFILE_ID = "m10-ironclad-policy"
RUN_OBSERVATION_VERSION = 3
RUN_REPLAY_VERSION = 3
RUN_OBSERVATION_FINGERPRINT = "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8"
STANDALONE_RAW_MAPPING_ID = (
    "spire-learn-standalone-common-v1@"
    f"{COMMON_VIEW_FINGERPRINT}"
)
RUN_RAW_MAPPING_ID = (
    "spire-learn-run-policy-v3-common-v1@"
    f"{RUN_OBSERVATION_FINGERPRINT}:{COMMON_VIEW_FINGERPRINT}"
)
LEARNER_IDENTITY_IDS = (
    "spire-learn-combat-ff-v1",
    "spire-learn-combat-features-v1",
    "spire-learn-combat-dispatch-v1",
    "spire-learn-combat-winloss-v1",
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _exact_keys(value: object, required: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    return value


@dataclass(frozen=True)
class SeedStreams:
    """Independent policy, optimizer/shuffle, curriculum, and fixed-controller seeds."""

    policy: int
    optimizer: int
    curriculum: int
    noncombat: int

    def __post_init__(self) -> None:
        values = (self.policy, self.optimizer, self.curriculum, self.noncombat)
        if any(type(value) is not int or value < 0 or value >= 2**64 for value in values):
            raise ContractError("seed streams must be non-negative uint64 integers")
        if len(set(values)) != len(values):
            raise ContractError("policy, optimizer, curriculum, and noncombat seeds must be distinct")


@dataclass(frozen=True)
class NeuralBudget:
    max_optimizer_steps: int
    max_training_decisions: int
    max_environment_actions: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        for name in ("max_optimizer_steps", "max_training_decisions", "max_environment_actions"):
            _positive_int(getattr(self, name), f"budget {name}")
        object.__setattr__(self, "max_wall_seconds",
                           _finite_positive(self.max_wall_seconds, "budget max_wall_seconds"))


@dataclass(frozen=True)
class NeuralConfig:
    mode: str
    operation: str
    split: str
    partition: str
    manifest_path: Path
    output_root: Path
    cases: NeuralCases
    manifest_sha256: str
    config_sha256: str
    environment_revision: str
    common_view_id: str
    common_view_fingerprint: str
    run_observation_fingerprint: str | None
    profile_id: str | None
    observation_version: int | None
    replay_version: int | None
    accepted_identities: tuple[NeuralIdentity, ...]
    seed_streams: SeedStreams
    model: ModelConfig
    ppo: PPOConfig
    rollout_rows: int
    rollout_steps: int
    device: str
    budgets: NeuralBudget
    final_requires_opt_in: bool


def _identity(raw: object, *, model_hash: str, ppo_hash: str) -> NeuralIdentity:
    required = {
        "model_id", "features_id", "action_id", "objective_id",
        "accepted_raw_mode_mappings", "config_hashes",
    }
    item = _exact_keys(raw, required, "accepted identity")
    mappings = item["accepted_raw_mode_mappings"]
    hashes = item["config_hashes"]
    if not isinstance(mappings, dict) or not isinstance(hashes, dict):
        raise ValueError("identity mappings and hashes must be tables")
    if set(mappings) - {"standalone", "run"}:
        raise ValueError("identity declares an unsupported raw mode")
    for raw_mode, mapping_id in mappings.items():
        expected_mapping = {
            "standalone": STANDALONE_RAW_MAPPING_ID,
            "run": RUN_RAW_MAPPING_ID,
        }[raw_mode]
        if not isinstance(mapping_id, str) or mapping_id != expected_mapping:
            raise ValueError(f"identity {raw_mode} mapping does not match the accepted raw-mode contract")
    if set(hashes) != {"model", "ppo"}:
        raise ValueError("identity config_hashes must declare model and ppo hashes")
    if hashes["model"] != model_hash or hashes["ppo"] != ppo_hash:
        raise ValueError("identity component hash disagrees with authored model/PPO settings")
    identity = NeuralIdentity(
        model_id=item["model_id"],
        features_id=item["features_id"],
        action_id=item["action_id"],
        objective_id=item["objective_id"],
        accepted_raw_mode_mappings=mappings,
        config_hashes=hashes,
    )
    if (identity.model_id, identity.features_id, identity.action_id, identity.objective_id
            ) != LEARNER_IDENTITY_IDS:
        raise ValueError("identity must use the supported combat learner model/features/action/objective IDs")
    return identity


def load_neural_config(path: Path) -> NeuralConfig:
    """Parse a strict TOML config and its case manifest without runtime side effects."""
    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read neural config {source}: {exc}") from exc

    required = {
        "schema", "mode", "operation", "split", "partition", "manifest_path",
        "output_root", "device", "environment", "seeds", "model", "ppo",
        "rollout", "budgets", "accepted_identities",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid neural config root fields")
    if raw["schema"] != "spire-learn-neural-config-v1":
        raise ValueError("unsupported neural config schema")
    mode, operation, split, partition = raw["mode"], raw["operation"], raw["split"], raw["partition"]
    if not isinstance(mode, str) or mode not in {"standalone", "run"}:
        raise ValueError("mode must be standalone or run")
    if not isinstance(operation, str) or operation not in {"train", "evaluate"}:
        raise ValueError("operation must be train or evaluate")
    if not isinstance(split, str) or split not in {"train", "selection", "final"} or partition != split:
        raise ValueError("split and partition must match train/selection/final")
    if operation == "train" and split != "train":
        raise ValueError("training is permitted only with the train split")
    if operation == "evaluate" and split not in {"selection", "final"}:
        raise ValueError("evaluation requires selection or final split")

    environment = raw["environment"]
    expected_environment_keys = {
        "gym_revision", "common_view_id", "common_view_fingerprint",
    }
    if mode == "run":
        expected_environment_keys |= {
            "profile_id", "observation_version", "replay_version", "run_observation_fingerprint",
        }
    environment = _exact_keys(environment, expected_environment_keys, "environment")
    if environment["gym_revision"] != ACCEPTED_GYM_REVISION:
        raise ValueError("unsupported Gym revision")
    if environment["common_view_id"] != COMMON_VIEW_ID:
        raise ValueError("unsupported common combat view")
    if environment["common_view_fingerprint"] != COMMON_VIEW_FINGERPRINT:
        raise ValueError("unsupported common combat view fingerprint")
    profile_id: str | None = None
    observation_version: int | None = None
    replay_version: int | None = None
    run_observation_fingerprint: str | None = None
    if mode == "run":
        if environment["profile_id"] != RUN_PROFILE_ID:
            raise ValueError("Run mode requires profile m10-ironclad-policy")
        if type(environment["observation_version"]) is not int or environment["observation_version"] != RUN_OBSERVATION_VERSION:
            raise ValueError("Run mode requires observation schema v3")
        if type(environment["replay_version"]) is not int or environment["replay_version"] != RUN_REPLAY_VERSION:
            raise ValueError("Run mode requires replay schema v3")
        if environment["run_observation_fingerprint"] != RUN_OBSERVATION_FINGERPRINT:
            raise ValueError("Run mode requires the accepted Run observation fingerprint")
        profile_id = RUN_PROFILE_ID
        observation_version = RUN_OBSERVATION_VERSION
        replay_version = RUN_REPLAY_VERSION
        run_observation_fingerprint = RUN_OBSERVATION_FINGERPRINT

    seed_values = _exact_keys(raw["seeds"], {"policy", "optimizer", "curriculum", "noncombat"}, "seeds")
    seeds = SeedStreams(**seed_values)

    model_raw = _exact_keys(
        raw["model"], {"embedding_width", "hidden_width", "max_parameters"}, "model"
    )
    model = ModelConfig(**model_raw)

    ppo_required = {
        "learning_rate", "gamma", "gae_lambda", "clip_epsilon", "value_coefficient",
        "entropy_coefficient", "max_grad_norm", "epochs", "minibatch_rows", "adam_eps",
    }
    ppo_raw = _exact_keys(raw["ppo"], ppo_required, "ppo")
    if not isinstance(ppo_raw, dict):
        raise ValueError("invalid PPO config fields")
    ppo = PPOConfig(**ppo_raw)

    rollout_raw = _exact_keys(raw["rollout"], {"rows", "steps"}, "rollout")
    rollout_rows = _positive_int(rollout_raw["rows"], "rollout.rows")
    rollout_steps = _positive_int(rollout_raw["steps"], "rollout.steps")
    budget_raw = _exact_keys(
        raw["budgets"],
        {"max_optimizer_steps", "max_training_decisions", "max_environment_actions", "max_wall_seconds"},
        "budgets",
    )
    budgets = NeuralBudget(**budget_raw)
    if mode == "run" and rollout_rows != 1:
        raise ValueError("Run mode requires exactly one native rollout slot")
    if mode == "standalone" and rollout_rows > 256:
        raise ValueError("standalone rollout rows must be between 1 and 256 native slots")
    rollout_batch = rollout_rows * rollout_steps
    if rollout_batch > budgets.max_training_decisions:
        raise ValueError("rollout rows times steps exceeds the training decision budget")
    if rollout_batch > budgets.max_environment_actions:
        raise ValueError("rollout rows times steps exceeds the environment-action budget")
    device = raw["device"]
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ValueError("device must be cpu, cuda, or cuda:<index>")

    model_hash = _canonical_hash(model_raw)
    ppo_hash = _canonical_hash(ppo_raw)
    identity_values = raw["accepted_identities"]
    if not isinstance(identity_values, list) or not identity_values:
        raise ValueError("accepted_identities must be a non-empty array of tables")
    identities = tuple(_identity(item, model_hash=model_hash, ppo_hash=ppo_hash)
                       for item in identity_values)
    if any(mode not in identity.accepted_raw_mode_mappings for identity in identities):
        raise ValueError("every accepted identity must declare the selected raw mode")
    identity_keys = [
        (item.model_id, item.features_id, item.action_id, item.objective_id)
        for item in identities
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("duplicate accepted identity")

    manifest_raw = raw["manifest_path"]
    output_raw = raw["output_root"]
    if not isinstance(manifest_raw, str) or not manifest_raw:
        raise ValueError("manifest_path must be a non-empty path")
    if not isinstance(output_raw, str) or not output_raw:
        raise ValueError("output_root must be a non-empty path")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_absolute():
        manifest_path = source.parent / manifest_path
    manifest_path = manifest_path.resolve()
    output_root = Path(output_raw)
    if not output_root.is_absolute():
        output_root = source.parent / output_root
    output_root = output_root.resolve()
    artifact_root = Path(__file__).resolve().parents[3] / "artifacts" / "development" / "neural-combat"
    if not output_root.is_relative_to(artifact_root.resolve()):
        raise ValueError("output_root must stay under artifacts/development/neural-combat")

    cases = load_neural_cases(manifest_path)
    if cases.split != split or cases.partition != partition:
        raise ValueError("manifest split/partition disagrees with neural config")
    if not cases.for_mode(mode):
        raise ValueError(f"manifest contains no cases for mode {mode}")
    related_manifests = [cases]
    for related_split in ("train", "selection", "final"):
        sibling_path = manifest_path.with_name(f"{related_split}.json")
        if related_split != cases.split and sibling_path.is_file():
            related_manifests.append(load_neural_cases(sibling_path))
    validate_neural_case_lineage(tuple(related_manifests))
    if operation == "train":
        validate_neural_case_lineage((cases,), training=True)
    elif split == "final":
        validate_neural_case_lineage((cases,), training=False)

    return NeuralConfig(
        mode=mode,
        operation=operation,
        split=split,
        partition=partition,
        manifest_path=manifest_path,
        output_root=output_root,
        cases=cases,
        manifest_sha256=cases.sha256,
        config_sha256=_canonical_hash(raw),
        environment_revision=environment["gym_revision"],
        common_view_id=environment["common_view_id"],
        common_view_fingerprint=environment["common_view_fingerprint"],
        run_observation_fingerprint=run_observation_fingerprint,
        profile_id=profile_id,
        observation_version=observation_version,
        replay_version=replay_version,
        accepted_identities=identities,
        seed_streams=seeds,
        model=model,
        ppo=ppo,
        rollout_rows=rollout_rows,
        rollout_steps=rollout_steps,
        device=device,
        budgets=budgets,
        final_requires_opt_in=split == "final",
    )


__all__ = [
    "ACCEPTED_GYM_REVISION",
    "COMMON_VIEW_FINGERPRINT",
    "COMMON_VIEW_ID",
    "NeuralBudget",
    "NeuralConfig",
    "RUN_PROFILE_ID",
    "SeedStreams",
    "load_neural_config",
]
