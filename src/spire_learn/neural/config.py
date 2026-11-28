"""Strict pure-Python dual-mode neural experiment configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

from spire_learn.contracts import ContractError
from spire_learn.neural.cases import (
    NeuralCases,
    _freeze_json,
    load_neural_cases,
    validate_neural_case_lineage,
)
from spire_learn.neural.contracts import ModelConfig, NeuralIdentity, PPOConfig
from spire_learn.neural.gym_contracts import GymContract, RUNTIME_REVISIONS, contract_for_revision
from spire_learn.neural.modes import NEURAL_MODES


ACCEPTED_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
# The original descriptor/support provenance stays frozen. Runtime revisions
# are explicit: schema compatibility does not imply arbitrary build acceptance.
FIXED_GYM_REVISION = "a86df922a9dc598645820cc6544df36aefb3d2cc"
ACCEPTED_RUNTIME_GYM_REVISIONS = RUNTIME_REVISIONS
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
WINLOSS_OBJECTIVE_ID = LEARNER_IDENTITY_IDS[3]
# Model and features for the Standard combat view (16 groups, 2048 actions). M10/common-view
# checkpoints (LEARNER_IDENTITY_IDS) never load into them. Standard identities may declare only
# Standard raw modes; the action dispatch identity is shared.
STANDARD_LEARNER_IDENTITY_IDS = (
    "spire-learn-combat-ff-standard-v1",
    "spire-learn-combat-features-standard-v1",
    LEARNER_IDENTITY_IDS[2],
)
_STANDARD_MODES = frozenset({"standard-standalone", "standard-run"})
# Win/loss plus a potential-based enemy-HP term; requires the [objective] table.
HP_SHAPED_OBJECTIVE_ID = "spire-learn-combat-winloss-hpshaped-v1"
OBJECTIVE_IDS = (WINLOSS_OBJECTIVE_ID, HP_SHAPED_OBJECTIVE_ID)
# Supervised distillation from search-teacher labels (load_distill_config; evaluation through
# load_neural_config with config_hashes = {model}). Its checkpoints never load as PPO ones.
DISTILL_OBJECTIVE_ID = "spire-learn-combat-distill-v1"
DISTILL_CONFIG_SCHEMA = "spire-learn-distill-config-v1"
DISTILL_DATASET_SCHEMA = "spire-learn-distill-dataset-v1"
# The hash key of each objective's training hyperparameters (identity config_hashes and the
# checkpoint training_state hyperparameter_id).
DISTILL_HASH_KEY = "distill"
PPO_HASH_KEY = "ppo"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_DEVICE = re.compile(r"(?:cpu|cuda:0)\Z")
_CALIBRATION_GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
_RUNTIME_GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_config_hash_payload(model_config: object) -> dict[str, object]:
    """The canonical payload every model-config hash site hashes.

    It is the dataclass fields, except that ``value_heads`` is left out when it is 1, so an
    explicit ``value_heads = 1`` hashes exactly like the historical configs without it.
    """
    if not is_dataclass(model_config) or isinstance(model_config, type):
        raise TypeError("model_config must be a model-config dataclass instance")
    payload = asdict(model_config)
    if payload.get("value_heads") == 1:
        del payload["value_heads"]
    return payload


def model_config_hash(model_config: object) -> str:
    """The identity's ``config_hashes["model"]``."""
    return _canonical_hash(model_config_hash_payload(model_config))


def objective_hash_key(objective_id: str) -> str:
    """``distill`` for the distillation objective, ``ppo`` for every PPO objective."""
    return DISTILL_HASH_KEY if objective_id == DISTILL_OBJECTIVE_ID else PPO_HASH_KEY


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
class HpShapedObjective:
    """Potential-based enemy-HP shaping coefficient; gamma is the PPO discount."""

    coefficient: float

    def __post_init__(self) -> None:
        value = self.coefficient
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("objective.coefficient must be a finite number in (0, 1]")
        object.__setattr__(self, "coefficient", float(value))


def _objective(raw: object) -> HpShapedObjective:
    table = _exact_keys(raw, {"kind", "coefficient"}, "objective")
    if table["kind"] != "hp_shaped":
        raise ValueError('objective.kind must be "hp_shaped"')
    return HpShapedObjective(coefficient=table["coefficient"])


@dataclass(frozen=True)
class NeuralBudget:
    max_optimizer_steps: int
    max_training_decisions: int
    max_environment_actions: int
    max_wall_seconds: float
    max_turn_actions: int | None = None
    validation_stride: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_optimizer_steps", "max_training_decisions", "max_environment_actions"):
            _positive_int(getattr(self, name), f"budget {name}")
        if self.max_turn_actions is not None:
            _positive_int(self.max_turn_actions, "budget max_turn_actions")
        if self.validation_stride is not None:
            _positive_int(self.validation_stride, "budget validation_stride")
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
    common_view_id: str | None  # None only in Standard-only configs
    common_view_fingerprint: str | None
    run_observation_fingerprint: str | None
    standard_view_fingerprint: str | None
    standard_run_fingerprint: str | None
    profile_id: str | None
    observation_version: int | None
    replay_version: int | None
    accepted_identities: tuple[NeuralIdentity, ...]
    seed_streams: SeedStreams
    model: ModelConfig  # or a registered model family's config (see MODEL_FAMILY_TABLES)
    ppo: PPOConfig | None  # None only for a distillation evaluation config
    rollout_rows: int
    rollout_steps: int
    device: str
    budgets: NeuralBudget
    final_requires_opt_in: bool
    calibration_gpu_uuid: str | None = None
    runtime_gpu_uuid: str | None = None
    scenario_stream: Mapping[str, object] | None = None
    workers: int = 1
    # None is the win/loss objective; configs without [objective] keep it.
    objective: HpShapedObjective | None = None
    # Standard combat observation layout: "dense" (the default and the only layout before
    # Gym v1.2.0) or "compact" with a capacity preset (standard-standalone only).
    standard_view_layout: str = "dense"
    standard_view_preset: str | None = None
    # None records omission; the effective setting is false.
    trim_padding: bool | None = None
    experimental_trim_compute: bool | None = None
    # The TOML table the model settings came from ("model" for ModelConfig).
    model_table: str = "model"
    # Distillation hyperparameters, when a config carries them (evaluation configs do not).
    distill: "DistillHyperparameters | None" = None

    @property
    def effective_validation_stride(self) -> int:
        if self.operation == "train" and self.budgets.validation_stride is not None:
            return self.budgets.validation_stride
        return 1


def _stream_runtime_projection(config: NeuralConfig) -> dict[str, object]:
    """Project accepted normalized settings for loader binding and stream admission."""
    runtime = {name: getattr(config, name) for name in (
        'mode', 'operation', 'split', 'partition', 'environment_revision',
        'common_view_id', 'common_view_fingerprint', 'run_observation_fingerprint',
        'profile_id', 'observation_version', 'replay_version', 'rollout_rows',
        'rollout_steps', 'device', 'final_requires_opt_in', 'calibration_gpu_uuid',
        'runtime_gpu_uuid')}
    # Same key order as before; the model goes through the shared hash payload (value_heads = 1
    # is left out, as before it existed) and ppo is written only when present.
    runtime['seed_streams'] = asdict(config.seed_streams)
    runtime['model'] = model_config_hash_payload(config.model)
    if config.ppo is not None:
        runtime['ppo'] = asdict(config.ppo)
    runtime['budgets'] = asdict(config.budgets)
    if config.trim_padding is not None:
        runtime['model']['trim_padding'] = config.trim_padding
    if config.experimental_trim_compute is not None:
        runtime['model']['experimental_trim_compute'] = config.experimental_trim_compute
    if config.budgets.max_turn_actions is None:
        del runtime['budgets']['max_turn_actions']
    if config.budgets.validation_stride is None:
        del runtime['budgets']['validation_stride']
    runtime['accepted_identities'] = [{
        'model_id': identity.model_id, 'features_id': identity.features_id,
        'action_id': identity.action_id, 'objective_id': identity.objective_id,
        'accepted_raw_mode_mappings': dict(identity.accepted_raw_mode_mappings),
        'config_hashes': dict(identity.config_hashes),
    } for identity in config.accepted_identities]
    if config.objective is not None:
        # Only shaped configs gain this key; win/loss projections stay byte-identical.
        runtime['objective'] = {'kind': 'hp_shaped', 'coefficient': config.objective.coefficient}
    if config.standard_view_layout != "dense":
        # Likewise only compact configs gain the layout; dense projections stay byte-identical.
        runtime['standard_view_layout'] = config.standard_view_layout
        runtime['standard_view_preset'] = config.standard_view_preset
        runtime['standard_view_fingerprint'] = config.standard_view_fingerprint
    return runtime


def _identity(raw: object, *, model_hash: str, ppo_hash: str | None = None, contract: GymContract,
              standard_view_fingerprint: str | None = None,
              component_hashes: Mapping[str, str] | None = None,
              objective_ids: tuple[str, ...] = OBJECTIVE_IDS) -> NeuralIdentity:
    """One accepted identity whose component hashes match the authored settings.

    PPO configs declare ``{model, ppo}`` (the default); distillation configs pass
    ``component_hashes`` (``{model}`` to evaluate, ``{model, distill}`` to train) and the
    distillation objective.
    """
    required = {
        "model_id", "features_id", "action_id", "objective_id",
        "accepted_raw_mode_mappings", "config_hashes",
    }
    item = _exact_keys(raw, required, "accepted identity")
    mappings = item["accepted_raw_mode_mappings"]
    hashes = item["config_hashes"]
    if not isinstance(mappings, dict) or not isinstance(hashes, dict):
        raise ValueError("identity mappings and hashes must be tables")
    if set(mappings) - NEURAL_MODES:
        raise ValueError("identity declares an unsupported raw mode")
    accepted_mappings = (contract.raw_mode_mappings if standard_view_fingerprint is None
                         else contract.raw_mode_mappings_for_view(standard_view_fingerprint))
    for raw_mode, mapping_id in mappings.items():
        expected_mapping = accepted_mappings.get(raw_mode)
        if expected_mapping is None:
            raise ValueError(f"identity declares {raw_mode}, which the release does not map for this "
                             "Standard combat layout (compact maps standard-standalone only)")
        if not isinstance(mapping_id, str) or mapping_id != expected_mapping:
            raise ValueError(f"identity {raw_mode} mapping does not match the accepted raw-mode contract")
    if component_hashes is None:
        if set(hashes) != {"model", "ppo"}:
            raise ValueError("identity config_hashes must declare model and ppo hashes")
        if hashes["model"] != model_hash or hashes["ppo"] != ppo_hash:
            raise ValueError("identity component hash disagrees with authored model/PPO settings")
    else:
        if set(hashes) != set(component_hashes):
            raise ValueError(f"identity config_hashes must declare exactly {sorted(component_hashes)}")
        if any(hashes[name] != digest for name, digest in component_hashes.items()):
            raise ValueError("identity component hash disagrees with the authored settings")
    identity = NeuralIdentity(
        model_id=item["model_id"],
        features_id=item["features_id"],
        action_id=item["action_id"],
        objective_id=item["objective_id"],
        accepted_raw_mode_mappings=mappings,
        config_hashes=hashes,
    )
    learner_ids = (identity.model_id, identity.features_id, identity.action_id)
    if (learner_ids not in (LEARNER_IDENTITY_IDS[:3], STANDARD_LEARNER_IDENTITY_IDS)
            or identity.objective_id not in objective_ids):
        raise ValueError("identity must use the supported combat learner model/features/action/objective IDs")
    if learner_ids == STANDARD_LEARNER_IDENTITY_IDS and not set(mappings) <= _STANDARD_MODES:
        raise ValueError("the Standard learner identity declares only standard-standalone/standard-run mappings")
    return identity


_MODE_FAMILIES = {
    "standalone": "standalone", "standard-standalone": "standalone",
    "run": "run", "standard-run": "run",
}


def require_standard_learner_identity(identities) -> None:
    """Curriculum streams of Standard recipes train only the Standard learner identity.

    Each identity must use the Standard model/features IDs and declare only Standard raw
    modes. (Historical non-stream Standard configs keep their v1 identities.) ``identities``
    may be NeuralIdentity values or raw template tables.
    """
    for identity in identities:
        get = identity.get if isinstance(identity, Mapping) else lambda name: getattr(identity, name, None)
        mappings = get("accepted_raw_mode_mappings")
        if ((get("model_id"), get("features_id"), get("action_id")) != STANDARD_LEARNER_IDENTITY_IDS
                or not isinstance(mappings, Mapping) or not mappings or not set(mappings) <= _STANDARD_MODES):
            raise ValueError("a Standard curriculum stream requires the Standard learner identity "
                             "(model/features *-standard-v1) with only Standard raw-mode mappings")


def mode_family(mode: str) -> str:
    """Driver family of a config mode: standalone combat or combat inside a Run."""
    try:
        return _MODE_FAMILIES[mode]
    except (KeyError, TypeError):
        raise ValueError("mode must be standalone, run, standard-standalone, or standard-run") from None


def _validate_experimental_trim_compute(model: Mapping[str, object], *, operation: str,
                                        allow_experimental_trim_compute: bool) -> None:
    if "experimental_trim_compute" not in model:
        return
    if type(model["experimental_trim_compute"]) is not bool:
        raise ValueError("model.experimental_trim_compute must be a boolean")
    if model.get("trim_padding") is not True:
        raise ValueError("model.experimental_trim_compute requires model.trim_padding = true")
    if not allow_experimental_trim_compute or operation != "train":
        raise ValueError("model.experimental_trim_compute is restricted to profile_entry training")


def _parse_ff_model(table: dict[str, object]) -> ModelConfig:
    return ModelConfig(**table)


# Model-family tables an evaluation config may carry, keyed by TOML table name. ``model`` is the
# feed-forward ModelConfig (its trim flags are handled by the loader); a later family registers
# its own table here (``transformer``). PPO and calibration configs use [model] only.
KNOWN_MODEL_TABLES = ("model", "transformer")
MODEL_FAMILY_TABLES: dict[str, Callable[[dict[str, object]], object]] = {"model": _parse_ff_model}


def _declares_distill(raw: object) -> bool:
    """Whether any accepted identity table names the distillation objective."""
    identities = raw.get("accepted_identities") if isinstance(raw, dict) else None
    return isinstance(identities, list) and any(
        isinstance(item, dict) and item.get("objective_id") == DISTILL_OBJECTIVE_ID for item in identities)


def _model_table_name(raw: dict[str, object]) -> str:
    tables = [name for name in KNOWN_MODEL_TABLES if name in raw]
    if len(tables) != 1:
        raise ValueError("a distillation evaluation config declares exactly one model table: "
                         "[model] or [transformer]")
    if tables[0] not in MODEL_FAMILY_TABLES:
        raise ValueError(f"the [{tables[0]}] model table has no registered model family")
    return tables[0]


def load_neural_config(path: Path, *, allow_experimental_trim_compute: bool = False) -> NeuralConfig:
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
    # A distillation evaluation config has no [ppo] table and one model-family table.
    distill = _declares_distill(raw)
    model_table = "model"
    if distill:
        model_table = _model_table_name(raw)
        required = (required - {"model", "ppo"}) | {model_table}
    expected_root = set(required)
    if isinstance(raw, dict) and "scenario_stream" in raw:
        expected_root.add("scenario_stream")
    if isinstance(raw, dict) and "objective" in raw:
        expected_root.add("objective")
    if isinstance(raw, dict) and raw.get("operation") == "calibrate":
        expected_root.add("calibration")
    elif isinstance(raw, dict) and raw.get("device") == "cuda:0":
        expected_root.add("runtime")
    if not isinstance(raw, dict) or set(raw) != expected_root:
        raise ValueError("invalid neural config root fields")
    if raw["schema"] != "spire-learn-neural-config-v1":
        raise ValueError("unsupported neural config schema")
    mode, operation, split, partition = raw["mode"], raw["operation"], raw["split"], raw["partition"]
    if not isinstance(mode, str) or mode not in NEURAL_MODES:
        raise ValueError("mode must be standalone, run, standard-standalone, or standard-run")
    standard_mode = mode in {"standard-standalone", "standard-run"}
    if not isinstance(operation, str) or operation not in {"train", "evaluate", "calibrate"}:
        raise ValueError("operation must be train, evaluate, or calibrate")
    if not isinstance(split, str) or split not in {"train", "selection", "final"} or partition != split:
        raise ValueError("split and partition must match train/selection/final")
    if operation == "train" and split != "train":
        raise ValueError("training is permitted only with the train split")
    if operation == "evaluate" and split not in {"selection", "final"}:
        raise ValueError("evaluation requires selection or final split")
    if distill and operation != "evaluate":
        raise ValueError(f"load_neural_config admits {DISTILL_OBJECTIVE_ID} only to evaluate; "
                         "distillation training uses load_distill_config")
    calibration_gpu_uuid: str | None = None
    if operation == "calibrate":
        if mode != "standalone" or split != "train" or partition != "train":
            raise ValueError("calibration requires standalone mode and split=partition=train")
        calibration_raw = _exact_keys(raw.get("calibration"), {"gpu_uuid"}, "calibration")
        calibration_gpu_uuid = calibration_raw["gpu_uuid"]
        if (not isinstance(calibration_gpu_uuid, str)
                or _GPU_UUID.fullmatch(calibration_gpu_uuid) is None):
            raise ValueError("calibration.gpu_uuid must be a well-formed GPU UUID string")
        if calibration_gpu_uuid != _CALIBRATION_GPU_UUID:
            raise ValueError("calibration.gpu_uuid does not match the qualified Task 8 device")
    elif "calibration" in raw:
        raise ValueError("calibration table is valid only for operation=calibrate")

    runtime_gpu_uuid: str | None = None
    if operation != "calibrate" and raw.get("device") == "cuda:0":
        runtime_raw = _exact_keys(raw.get("runtime"), {"gpu_uuid"}, "runtime")
        runtime_gpu_uuid = runtime_raw["gpu_uuid"]
        if (not isinstance(runtime_gpu_uuid, str)
                or _GPU_UUID.fullmatch(runtime_gpu_uuid) is None):
            raise ValueError("runtime.gpu_uuid must be a well-formed GPU UUID string")
        if runtime_gpu_uuid != _RUNTIME_GPU_UUID:
            raise ValueError("runtime.gpu_uuid does not match the qualified Task 8 device")
    elif "runtime" in raw:
        raise ValueError("runtime table is valid only for general device=cuda:0")

    environment = raw["environment"]
    expected_environment_keys = {
        "gym_revision", "common_view_id", "common_view_fingerprint",
    }
    # A Standard-only config (no identity declaring an M10 mode, checked below) may omit
    # the common view; it is then recorded as None. Both common fields or neither.
    common_omitted = (standard_mode and isinstance(environment, dict)
                      and "common_view_id" not in environment and "common_view_fingerprint" not in environment)
    if common_omitted:
        expected_environment_keys -= {"common_view_id", "common_view_fingerprint"}
    if standard_mode:
        expected_environment_keys.add("standard_view_fingerprint")
        if isinstance(environment, dict) and "standard_view_layout" in environment:
            expected_environment_keys.add("standard_view_layout")
            if environment["standard_view_layout"] == "compact":
                expected_environment_keys.add("standard_view_preset")
        if mode == "standard-run":
            expected_environment_keys.add("standard_run_fingerprint")
        if isinstance(environment, dict) and "workers" in environment:
            expected_environment_keys.add("workers")
    if mode == "run":
        expected_environment_keys |= {
            "profile_id", "observation_version", "replay_version", "run_observation_fingerprint",
        }
    environment = _exact_keys(environment, expected_environment_keys, "environment")
    contract = contract_for_revision(environment["gym_revision"])
    common_view_id: str | None = None
    common_view_fingerprint: str | None = None
    if not common_omitted:
        if environment["common_view_id"] != COMMON_VIEW_ID:
            raise ValueError("unsupported common combat view")
        if environment["common_view_fingerprint"] != contract.common_fingerprint:
            raise ValueError("unsupported common combat view fingerprint")
        common_view_id = environment["common_view_id"]
        common_view_fingerprint = environment["common_view_fingerprint"]
    standard_view_fingerprint: str | None = None
    standard_run_fingerprint: str | None = None
    standard_view_layout = "dense"
    standard_view_preset: str | None = None
    workers = 1
    if standard_mode:
        standard_view_fingerprint = environment["standard_view_fingerprint"]
        standard_view_layout = environment.get("standard_view_layout", "dense")
        if standard_view_layout not in ("dense", "compact"):
            raise ValueError('environment.standard_view_layout must be "dense" or "compact"')
        if standard_view_layout == "compact":
            presets = contract.standard_compact_fingerprints
            if not presets:
                raise ValueError("the compact Standard combat layout requires Gym v1.2.0; "
                                 "earlier releases are dense only")
            if mode != "standard-standalone":
                raise ValueError("the compact Standard combat layout requires standard-standalone mode "
                                 "(the Run's combat projection is dense only)")
            standard_view_preset = environment["standard_view_preset"]
            if not isinstance(standard_view_preset, str) or standard_view_preset not in presets:
                raise ValueError("environment.standard_view_preset is not an accepted compact preset")
            if standard_view_fingerprint != presets[standard_view_preset]:
                raise ValueError("the compact Standard layout requires its preset's compact view fingerprint")
        elif (contract.standard_combat_fingerprint is None
                or standard_view_fingerprint != contract.standard_combat_fingerprint):
            raise ValueError("Standard mode requires the accepted standard combat view fingerprint")
        if "workers" in environment:
            workers = environment["workers"]
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("environment.workers must be an integer in [1, 256]")
        if mode == "standard-run":
            if workers != 1:
                raise ValueError("standard-run requires workers = 1")
            standard_run_fingerprint = environment["standard_run_fingerprint"]
            if (contract.standard_run_fingerprint is None
                    or standard_run_fingerprint != contract.standard_run_fingerprint):
                raise ValueError("standard-run requires the accepted standard Run fingerprint")
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
        if environment["run_observation_fingerprint"] != contract.run_fingerprint:
            raise ValueError("Run mode requires the accepted Run observation fingerprint")
        profile_id = RUN_PROFILE_ID
        observation_version = RUN_OBSERVATION_VERSION
        replay_version = RUN_REPLAY_VERSION
        run_observation_fingerprint = contract.run_fingerprint
    seed_values = _exact_keys(raw["seeds"], {"policy", "optimizer", "curriculum", "noncombat"}, "seeds")
    seeds = SeedStreams(**seed_values)

    trim_padding = None
    experimental_trim_compute = None
    if model_table == "model":
        model_fields = {"embedding_width", "hidden_width", "max_parameters"}
        if isinstance(raw["model"], dict):
            model_fields.update(raw["model"].keys() & {"trim_padding", "experimental_trim_compute", "value_heads"})
        model_raw = dict(_exact_keys(raw["model"], model_fields, "model"))
        _validate_experimental_trim_compute(model_raw, operation=raw["operation"],
                                            allow_experimental_trim_compute=allow_experimental_trim_compute)
        trim_padding = model_raw.pop("trim_padding", None)
        if trim_padding is not None and type(trim_padding) is not bool:
            raise ValueError("model.trim_padding must be a boolean")
        experimental_trim_compute = model_raw.pop("experimental_trim_compute", None)
        model = ModelConfig(**model_raw)
        if not distill and model.value_heads != 1:
            raise ValueError("model.value_heads = 3 is for distillation students; PPO configs keep one critic")
    else:
        if not isinstance(raw[model_table], dict):
            raise ValueError(f"[{model_table}] must be a table")
        model = MODEL_FAMILY_TABLES[model_table](dict(raw[model_table]))

    ppo: PPOConfig | None = None
    ppo_hash: str | None = None
    if not distill:
        ppo_required = {
            "learning_rate", "gamma", "gae_lambda", "clip_epsilon", "value_coefficient",
            "entropy_coefficient", "max_grad_norm", "epochs", "minibatch_rows", "adam_eps",
        }
        ppo_raw = _exact_keys(raw["ppo"], ppo_required, "ppo")
        if not isinstance(ppo_raw, dict):
            raise ValueError("invalid PPO config fields")
        ppo = PPOConfig(**ppo_raw)
        ppo_hash = _canonical_hash(ppo_raw)

    rollout_raw = _exact_keys(raw["rollout"], {"rows", "steps"}, "rollout")
    rollout_rows = _positive_int(rollout_raw["rows"], "rollout.rows")
    rollout_steps = _positive_int(rollout_raw["steps"], "rollout.steps")
    budget_fields = {"max_optimizer_steps", "max_training_decisions", "max_environment_actions", "max_wall_seconds"}
    if isinstance(raw["budgets"], dict):
        budget_fields.update(raw["budgets"].keys() & {"max_turn_actions", "validation_stride"})
    budget_raw = _exact_keys(raw["budgets"], budget_fields, "budgets")
    budgets = NeuralBudget(**budget_raw)
    if mode in {"run", "standard-run"} and rollout_rows != 1:
        raise ValueError("Run mode requires exactly one native rollout slot")
    if mode in {"standalone", "standard-standalone"} and rollout_rows > 256:
        raise ValueError("standalone rollout rows must be between 1 and 256 native slots")
    rollout_batch = rollout_rows * rollout_steps
    if rollout_batch > budgets.max_training_decisions:
        raise ValueError("rollout rows times steps exceeds the training decision budget")
    if rollout_batch > budgets.max_environment_actions:
        raise ValueError("rollout rows times steps exceeds the environment-action budget")
    device = raw["device"]
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ValueError("device must be cpu or cuda:0")
    if operation == "calibrate":
        if device != "cuda:0":
            raise ValueError("calibration requires device=cuda:0")
        if (model.embedding_width, model.hidden_width, model.max_parameters) != (16, 64, 10_000_000):
            raise ValueError("calibration requires model settings 16/64/10000000")
        if (rollout_rows, rollout_steps) != (4, 32):
            raise ValueError("calibration base rollout must be rows=4 and steps=32")
        if (budgets.max_optimizer_steps, budgets.max_training_decisions,
                budgets.max_environment_actions, budgets.max_wall_seconds) != (3, 1024, 1024, 300.0):
            raise ValueError("calibration budgets must be 3 optimizer calls, 1024 decisions/actions, 300 seconds")

    # Every model hash site hashes model_config_hash_payload (identical to the authored [model]
    # table for ModelConfig: value_heads = 1 is left out).
    model_hash = model_config_hash(model)
    identity_values = raw["accepted_identities"]
    if not isinstance(identity_values, list) or not identity_values:
        raise ValueError("accepted_identities must be a non-empty array of tables")
    identity_options = ({"component_hashes": {"model": model_hash}, "objective_ids": (DISTILL_OBJECTIVE_ID,)}
                        if distill else {"ppo_hash": ppo_hash})
    identities = tuple(_identity(item, model_hash=model_hash, contract=contract,
                                 standard_view_fingerprint=(standard_view_fingerprint
                                                            if standard_view_layout == "compact" else None),
                                 **identity_options)
                       for item in identity_values)
    if any(mode not in identity.accepted_raw_mode_mappings for identity in identities):
        raise ValueError("every accepted identity must declare the selected raw mode")
    if common_omitted and any(not set(identity.accepted_raw_mode_mappings) <= _STANDARD_MODES
                              for identity in identities):
        raise ValueError("a config declaring an M10 raw mode must pin the common combat view")
    identity_keys = [
        (item.model_id, item.features_id, item.action_id, item.objective_id)
        for item in identities
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("duplicate accepted identity")
    if distill and "objective" in raw:
        raise ValueError("the [objective] table is for the hp-shaped PPO objective, not distillation")
    objective = _objective(raw["objective"]) if "objective" in raw else None
    expected_objective_id = (DISTILL_OBJECTIVE_ID if distill
                             else WINLOSS_OBJECTIVE_ID if objective is None else HP_SHAPED_OBJECTIVE_ID)
    if any(identity.objective_id != expected_objective_id for identity in identities):
        raise ValueError(
            "the hp-shaped objective ID requires the [objective] table, and the table requires that ID"
        )

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
    if operation in {"train", "calibrate"}:
        validate_neural_case_lineage((cases,), training=True)
    elif split == "final":
        validate_neural_case_lineage((cases,), training=False)

    scenario_stream = None
    if distill and "scenario_stream" in raw:
        raise ValueError("a distillation evaluation config has no scenario stream")
    if "scenario_stream" in raw:
        from .stream_resume import normalize_stream_config
        scenario_stream = normalize_stream_config(raw, source, manifest_path, cases.sha256)

    config = NeuralConfig(
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
        common_view_id=common_view_id,
        common_view_fingerprint=common_view_fingerprint,
        run_observation_fingerprint=run_observation_fingerprint,
        standard_view_fingerprint=standard_view_fingerprint,
        standard_run_fingerprint=standard_run_fingerprint,
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
        calibration_gpu_uuid=calibration_gpu_uuid,
        runtime_gpu_uuid=runtime_gpu_uuid,
        scenario_stream=scenario_stream,
        workers=workers,
        objective=objective,
        trim_padding=trim_padding,
        experimental_trim_compute=experimental_trim_compute,
        standard_view_layout=standard_view_layout,
        standard_view_preset=standard_view_preset,
        model_table=model_table,
    )
    if scenario_stream is not None:
        # Bind only after this parser has accepted and normalized every setting.
        # dataclasses.replace retains this immutable provenance at later API calls.
        config = replace(config, scenario_stream=_freeze_json({
            **scenario_stream, 'runtime_projection': _stream_runtime_projection(config),
        }))
    return config


# --- distillation training configs -----------------------------------------------------------

_DISTILL_HASHED = ("tau", "learning_rate", "batch_rows", "gradient_steps", "data_fraction")


@dataclass(frozen=True)
class DistillHyperparameters:
    """The hashed distillation settings (identity ``config_hashes["distill"]``).

    ``tau`` is the soft-target temperature, ``data_fraction`` the index-order prefix of the
    training shards. ``microbatch_rows`` is a memory setting outside the hash: gradients are
    accumulated to ``batch_rows`` and equal one full batch.
    """

    tau: float
    learning_rate: float
    batch_rows: int
    gradient_steps: int
    data_fraction: float
    microbatch_rows: int | None = None

    def __post_init__(self) -> None:
        for name in ("tau", "learning_rate"):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), f"distill.{name}"))
        for name in ("batch_rows", "gradient_steps"):
            _positive_int(getattr(self, name), f"distill.{name}")
        fraction = self.data_fraction
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("distill.data_fraction must be in (0, 1]")
        object.__setattr__(self, "data_fraction", float(fraction))
        if self.microbatch_rows is not None:
            _positive_int(self.microbatch_rows, "distill.microbatch_rows")
            if self.microbatch_rows > self.batch_rows:
                raise ValueError("distill.microbatch_rows cannot exceed distill.batch_rows")

    @property
    def effective_microbatch_rows(self) -> int:
        return self.batch_rows if self.microbatch_rows is None else self.microbatch_rows


def distill_hash_payload(hyperparameters: DistillHyperparameters) -> dict[str, object]:
    """Only tau, learning_rate, batch_rows, gradient_steps and data_fraction (normalized)."""
    if not isinstance(hyperparameters, DistillHyperparameters):
        raise TypeError("hyperparameters must be DistillHyperparameters")
    return {name: getattr(hyperparameters, name) for name in _DISTILL_HASHED}


def distill_config_hash(hyperparameters: DistillHyperparameters) -> str:
    return _canonical_hash(distill_hash_payload(hyperparameters))


@dataclass(frozen=True)
class DistillConfig:
    """A strict distillation training config: no manifest_path, [rollout], [ppo] or shard paths.

    The shards come from a dataset manifest given to the trainer (``distill_train_entry.py
    --dataset``). The shards, ``seed`` and ``eval_every`` are run metadata, outside every
    identity hash.
    """

    source: Path
    config_sha256: str
    mode: str
    environment_revision: str
    standard_view_layout: str
    standard_view_preset: str
    standard_view_fingerprint: str
    output_root: Path
    device: str
    runtime_gpu_uuid: str | None
    seed: int
    eval_every: int
    model: ModelConfig
    distill: DistillHyperparameters
    identity: NeuralIdentity


def _resolve(source: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def load_distill_config(path: Path) -> DistillConfig:
    """Parse a strict distillation training TOML without touching shards, devices or torch."""
    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read distillation config {source}: {exc}") from exc
    required = {"schema", "mode", "output_root", "device", "seed", "eval_every", "environment",
                "model", "distill", "identity"}
    if isinstance(raw, dict) and raw.get("device") == "cuda:0":
        required.add("runtime")
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"a distillation config must contain exactly {sorted(required)}")
    if raw["schema"] != DISTILL_CONFIG_SCHEMA:
        raise ValueError("unsupported distillation config schema")
    if raw["mode"] != "standard-standalone":
        raise ValueError("distillation trains on standard-standalone compact decisions only")
    device = raw["device"]
    if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
        raise ValueError("device must be cpu or cuda:0")
    runtime_gpu_uuid = None
    if device == "cuda:0":
        runtime_gpu_uuid = _exact_keys(raw["runtime"], {"gpu_uuid"}, "runtime")["gpu_uuid"]
        if not isinstance(runtime_gpu_uuid, str) or _GPU_UUID.fullmatch(runtime_gpu_uuid) is None:
            raise ValueError("runtime.gpu_uuid must be a well-formed GPU UUID string")
        if runtime_gpu_uuid != _RUNTIME_GPU_UUID:
            raise ValueError("runtime.gpu_uuid does not match the qualified Task 8 device")
    seed = raw["seed"]
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be a non-negative uint64 integer")
    eval_every = _positive_int(raw["eval_every"], "eval_every")

    environment = _exact_keys(raw["environment"], {
        "gym_revision", "standard_view_layout", "standard_view_preset", "standard_view_fingerprint",
    }, "environment")
    contract = contract_for_revision(environment["gym_revision"])
    if environment["standard_view_layout"] != "compact":
        raise ValueError("distillation shards hold the compact Standard combat layout")
    preset = environment["standard_view_preset"]
    presets = contract.standard_compact_fingerprints
    if not isinstance(preset, str) or preset not in presets:
        raise ValueError("environment.standard_view_preset is not an accepted compact preset")
    fingerprint = environment["standard_view_fingerprint"]
    if fingerprint != presets[preset]:
        raise ValueError("the compact Standard layout requires its preset's compact view fingerprint")

    model_fields = {"embedding_width", "hidden_width", "max_parameters", "value_heads"}
    model = ModelConfig(**_exact_keys(raw["model"], model_fields, "model"))
    if model.value_heads != 3:
        raise ValueError("a distillation student has three value heads (model.value_heads = 3)")
    distill_raw = raw["distill"]
    distill_fields = set(_DISTILL_HASHED)
    if isinstance(distill_raw, dict) and "microbatch_rows" in distill_raw:
        distill_fields.add("microbatch_rows")
    hyperparameters = DistillHyperparameters(**_exact_keys(distill_raw, distill_fields, "distill"))

    identity = _identity(
        raw["identity"], model_hash=model_config_hash(model), contract=contract,
        standard_view_fingerprint=fingerprint,
        component_hashes={"model": model_config_hash(model),
                          DISTILL_HASH_KEY: distill_config_hash(hyperparameters)},
        objective_ids=(DISTILL_OBJECTIVE_ID,))
    if (identity.model_id, identity.features_id, identity.action_id) != STANDARD_LEARNER_IDENTITY_IDS:
        raise ValueError("the distillation student uses the Standard FF learner identity")
    if set(identity.accepted_raw_mode_mappings) != {"standard-standalone"}:
        raise ValueError("the distillation identity maps standard-standalone only")

    output_root = _resolve(source, raw["output_root"], "output_root")
    artifact_root = Path(__file__).resolve().parents[3] / "artifacts" / "development" / "neural-combat"
    if not output_root.is_relative_to(artifact_root.resolve()):
        raise ValueError("output_root must stay under artifacts/development/neural-combat")
    return DistillConfig(
        source=source, config_sha256=_canonical_hash(raw), mode=raw["mode"],
        environment_revision=environment["gym_revision"], standard_view_layout="compact",
        standard_view_preset=preset, standard_view_fingerprint=fingerprint, output_root=output_root,
        device=device, runtime_gpu_uuid=runtime_gpu_uuid, seed=seed, eval_every=eval_every,
        model=model, distill=hyperparameters,
        identity=identity,
    )


__all__ = [
    "ACCEPTED_GYM_REVISION",
    "ACCEPTED_RUNTIME_GYM_REVISIONS",
    "DISTILL_CONFIG_SCHEMA",
    "DISTILL_DATASET_SCHEMA",
    "DISTILL_HASH_KEY",
    "DISTILL_OBJECTIVE_ID",
    "DistillConfig",
    "DistillHyperparameters",
    "FIXED_GYM_REVISION",
    "HP_SHAPED_OBJECTIVE_ID",
    "HpShapedObjective",
    "COMMON_VIEW_FINGERPRINT",
    "COMMON_VIEW_ID",
    "KNOWN_MODEL_TABLES",
    "MODEL_FAMILY_TABLES",
    "NeuralBudget",
    "NeuralConfig",
    "NEURAL_MODES",
    "PPO_HASH_KEY",
    "RUN_OBSERVATION_FINGERPRINT",
    "RUN_OBSERVATION_VERSION",
    "RUN_PROFILE_ID",
    "RUN_REPLAY_VERSION",
    "STANDARD_LEARNER_IDENTITY_IDS",
    "distill_config_hash",
    "distill_hash_payload",
    "load_distill_config",
    "model_config_hash",
    "model_config_hash_payload",
    "objective_hash_key",
    "require_standard_learner_identity",
    "SeedStreams",
    "WINLOSS_OBJECTIVE_ID",
    "load_neural_config",
    "mode_family",
]
