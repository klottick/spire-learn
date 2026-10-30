"""Pure authoring-time validation for the frozen Task 9 study package.

This module reads TOML/JSON and the learner's pure config/case loaders only.
It never imports a runtime backend, constructs an environment, or dispatches an
operation.  Future checkpoint digests are intentionally absent until a
successful training operation has produced its post-run inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
from typing import Any

from spire_learn.neural.cases import load_neural_cases, validate_neural_case_lineage
from spire_learn.neural.config import load_neural_config


STUDY_SCHEMA = "spire-learn-task9-study-v1"
STUDY_STATE = "prepared_not_released"
TRAINING_ORDER = ("standalone-a", "run-a", "standalone-b", "run-b")
POLICY_ORDER = (
    "standalone-a", "run-a", "standalone-b", "run-b",
    "legal-random", "end-turn-last-v1",
)
POLICY_MODES: dict[str, str | None] = {
    "standalone-a": "standalone",
    "run-a": "run",
    "standalone-b": "standalone",
    "run-b": "run",
    "legal-random": None,
    "end-turn-last-v1": None,
}
TRAINING_SEEDS: dict[str, dict[str, int]] = {
    "standalone-a": {"policy": 101103, "optimizer": 102207,
                     "curriculum": 103301, "noncombat": 104409},
    "run-a": {"policy": 301103, "optimizer": 302207,
              "curriculum": 303301, "noncombat": 304409},
    "standalone-b": {"policy": 201103, "optimizer": 202207,
                     "curriculum": 203301, "noncombat": 204409},
    "run-b": {"policy": 401103, "optimizer": 402207,
              "curriculum": 403301, "noncombat": 404409},
}
EVAL_SEEDS = {"policy": 91003, "optimizer": 92009,
              "curriculum": 93011, "noncombat": 94007}
TRAINING_CASES = "configs/cases/neural/train.json"
PARENT_SELECTION = "configs/cases/neural/task9-selection.json"
ENTRY_TEMPLATE = "configs/cases/neural/selection.json"
FINAL_CASES = "configs/cases/neural/final.json"
BASE_PACKAGE = "configs/neural/experiments/task9-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

_MODEL_DEFAULT = {
    "embedding_width": 16,
    "hidden_width": 64,
    "max_parameters": 10_000_000,
}
_PPO_DEFAULT = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "max_grad_norm": 0.5,
    "epochs": 4,
    "minibatch_rows": 64,
    "adam_eps": 0.00001,
}
_MODEL_HASH = "4fb628e06bf4952847795ac5d8548d12dfeede4c97e874ab843857ae0f91cd2f"
_PPO_HASH = "f08075e0b01a15095b542fc20cba11c3258381bcb5721bb5066a4448e43c9e2a"
_IDENTITY_IDS = {
    "model_id": "spire-learn-combat-ff-v1",
    "features_id": "spire-learn-combat-features-v1",
    "action_id": "spire-learn-combat-dispatch-v1",
    "objective_id": "spire-learn-combat-winloss-v1",
}
_COMMON_FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"
_RUN_FINGERPRINT = "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8"
_GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"

_RUNTIME_PINS: dict[str, object] = {
    "python": "3.12",
    "torch": "2.14.1+cu130",
    "gym_revision": _GYM_REVISION,
    "common_fingerprint": _COMMON_FINGERPRINT,
    "run_observation_fingerprint": _RUN_FINGERPRINT,
    "cuda_lock_sha256": "92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45",
    "native_extension_sha256": "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9",
    "container_image": "python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0",
    "gpu_uuid": _GPU_UUID,
    "train_device": "cuda:0",
    "selection_device": "cpu",
    "cpu_threads": 4,
    "interop_threads": 1,
    "host_memory_bytes": 8 * 1024**3,
    "gpu_memory_bytes": 8 * 1024**3,
    "minimum_free_gpu_bytes": 12 * 1024**3,
    "pids_max": 512,
}
_BUDGETS: dict[str, object] = {
    "training": {
        "learned_decisions": 16_384,
        "environment_step_slots": 65_536,
        "optimizer_attempts": 1_024,
        "soft_wall_seconds": 600,
        "phase_wall_seconds": 900,
        "output_ceiling_bytes": 1024**3,
        "final_checkpoint_slots": 1,
        "midpoint_checkpoint_slots": 0,
        "whole_phase_includes_setup_cleanup": True,
    },
    "selection": {
        "case_count": 32,
        "policy_count": 6,
        "operation_count": 192,
        "step_slots_per_operation": 256,
        "reserved_step_slots": 49_152,
        "max_reserved_step_slots": 65_536,
        "soft_wall_seconds": 600,
        "shared_soft_deadline_start": "before_setup",
        "phase_wall_seconds": 900,
        "phase_wall_limit_includes_cleanup": True,
        "output_ceiling_bytes": 2 * 1024**3,
        "optimizer_calls": 0,
        "training_resets": 0,
        "eval_seeds": EVAL_SEEDS,
    },
    "campaign": {
        "phase_count": 5,
        "max_phase_wall_seconds": 900,
        "whole_wall_seconds": 4_500,
    },
    "storage": {
        "total_output_ceiling_bytes": 6 * 1024**3,
        "retained_aggregate_ceiling_bytes": 40 * 1024**3,
        "minimum_free_bytes": 30 * 1024**3,
        "accounting_only": True,
    },
}
_INTERPRETATION = {
    "final_split": "not_used",
    "standalone_seen_encounters": ["cultist", "jaw_worm", "two_louse", "small_slimes"],
    "standalone_held_out_encounters": ["large_slime", "three_louse"],
    "held_out_scope": "excluded from the standalone training catalog only; no Run-policy non-exposure claim",
    "selection_backend_difference": "training uses cuda:0; frozen and reference selection uses cpu",
    "training_reset_seed_policy": "environment-seed-for-replica0-v1 ordinal; matched native reset streams, not disjoint",
    "selection_environment_seed_domain": "selection split seeds at (1<<40)+4096+i for i=0..31",
    "frozen_policy_action_rule": "highest legal logit; smallest original action index on ties",
    "reference_torch_policy": "legal-random and end-turn-last-v1 run in fresh processes without attempting or loading Torch/CUDA",
    "no_resume_retry_replacement": True,
    "selection_may_be_partial": True,
    "operational_fault_policy": "halt remaining campaign and persist not_started rows; genuine game death remains a valid outcome",
    "storage_limits_are_quotas": False,
    "qualifications": ["RNG-001", "TIME-001", "finite supported content", "simulator-only evidence",
                       "partial selection coverage", "checkpoint elapsed prefix ends at snapshot"],
}


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} must contain exactly {sorted(expected)}; found {actual}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _same_json_value(actual: object, expected: object) -> bool:
    """JSON equality that keeps booleans distinct from integers."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_same_json_value(actual[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _check_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> tuple[bytes, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    return raw, value


def _ensure_inside(root: Path, candidate: Path, label: str) -> Path:
    root = root.resolve()
    absolute = Path(candidate).absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root: {candidate}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {cursor}")
    resolved = absolute.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root through a symlink")
    return resolved


def _bound_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty repository-relative POSIX path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"{label} must stay relative to repository root")
    if posix.as_posix() != relative:
        raise ValueError(f"{label} must use its normalized repository-relative spelling")
    return _ensure_inside(root, root / Path(*posix.parts), label)


def _config_relative_path(root: Path, config_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty config-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to its config")
    cursor = config_path.parent
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            cursor = cursor.parent
        else:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"{label} path contains a symlink: {cursor}")
        cursor_real = cursor.resolve(strict=False)
        if not cursor_real.is_relative_to(root.resolve()):
            raise ValueError(f"{label} escapes repository root")
    return cursor.resolve(strict=False)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")


def _manifest_binding(root: Path, value: object, expected_path: str,
                      label: str) -> tuple[object, dict[str, object], Path]:
    binding = _exact_keys(value, {"path", "canonical_sha256", "raw_sha256"}, label)
    if binding["path"] != expected_path:
        raise ValueError(f"{label} path must be {expected_path}")
    path = _bound_path(root, binding["path"], f"{label} path")
    _require_file(path, label)
    raw, parsed = _read_json(path, label)
    _check_sha(binding["canonical_sha256"], f"{label} canonical_sha256")
    _check_sha(binding["raw_sha256"], f"{label} raw_sha256")
    cases = load_neural_cases(path)
    if binding["raw_sha256"] != _sha256(raw):
        raise ValueError(f"{label} raw SHA-256 mismatch")
    if binding["canonical_sha256"] != cases.sha256:
        raise ValueError(f"{label} canonical SHA-256 mismatch")
    return cases, parsed, path


def _config_binding(root: Path, value: object, expected_path: str,
                    label: str) -> tuple[object, dict[str, Any], Path]:
    binding = _exact_keys(value, {"path", "config_sha256", "raw_sha256"}, label)
    if binding["path"] != expected_path:
        raise ValueError(f"{label} path must be {expected_path}")
    path = _bound_path(root, binding["path"], f"{label} path")
    _require_file(path, label)
    try:
        raw = path.read_bytes()
        parsed = tomllib.loads(raw.decode("utf-8"))
        config = load_neural_config(path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise ValueError(f"cannot load {label} {path}: {exc}") from exc
    _check_sha(binding["config_sha256"], f"{label} config_sha256")
    _check_sha(binding["raw_sha256"], f"{label} raw_sha256")
    if binding["raw_sha256"] != _sha256(raw):
        raise ValueError(f"{label} raw SHA-256 mismatch")
    if binding["config_sha256"] != config.config_sha256:
        raise ValueError(f"{label} canonical SHA-256 mismatch")

    manifest_ref = _config_relative_path(root, path, parsed.get("manifest_path"),
                                         f"{label} manifest_path")
    if manifest_ref != config.manifest_path:
        raise ValueError(f"{label} manifest path differs from the actual config loader result")
    output_ref = _config_relative_path(root, path, parsed.get("output_root"),
                                       f"{label} output_root")
    if output_ref != config.output_root:
        raise ValueError(f"{label} output path differs from the actual config loader result")
    artifact_root = (root / "artifacts/development/neural-combat").resolve()
    if not output_ref.is_relative_to(artifact_root):
        raise ValueError(f"{label} output_root is outside artifacts/development/neural-combat")
    return config, parsed, path


def _check_model_identity(config: object, label: str) -> None:
    from dataclasses import asdict

    if asdict(config.model) != _MODEL_DEFAULT:
        raise ValueError(f"{label} model settings differ from the accepted defaults")
    if asdict(config.ppo) != _PPO_DEFAULT:
        raise ValueError(f"{label} PPO settings differ from the accepted defaults")
    if len(config.accepted_identities) != 1:
        raise ValueError(f"{label} must contain the single accepted model/PPO identity")
    identity = config.accepted_identities[0]
    for key, expected in _IDENTITY_IDS.items():
        if getattr(identity, key) != expected:
            raise ValueError(f"{label} identity {key} differs from the accepted default")
    expected_mappings = {
        "standalone": f"spire-learn-standalone-common-v1@{_COMMON_FINGERPRINT}",
        "run": f"spire-learn-run-policy-v3-common-v1@{_RUN_FINGERPRINT}:{_COMMON_FINGERPRINT}",
    }
    if dict(identity.accepted_raw_mode_mappings) != expected_mappings:
        raise ValueError(f"{label} raw mode identity mappings differ from the accepted defaults")
    expected_hashes = {"model": _MODEL_HASH, "ppo": _PPO_HASH}
    if dict(identity.config_hashes) != expected_hashes:
        raise ValueError(f"{label} model/PPO identity hashes differ from accepted defaults")


def _validate_common_config(config: object, label: str) -> None:
    _check_model_identity(config, label)
    if (config.environment_revision != _GYM_REVISION
            or config.common_view_fingerprint != _COMMON_FINGERPRINT):
        raise ValueError(f"{label} environment fingerprints differ from the accepted contract")


def _expected_case_records(train_cases: object, template_raw: dict[str, object]) -> list[dict[str, object]]:
    seen_catalog = {
        case.entry["encounter"] for case in train_cases.cases
        if case.mode == "standalone"
    }
    expected_seen = {"cultist", "jaw_worm", "two_louse", "small_slimes"}
    held_out = {"large_slime", "three_louse"}
    if seen_catalog != expected_seen:
        raise ValueError("actual standalone training catalog differs from the specified seen catalog")
    templates = template_raw.get("cases") if isinstance(template_raw, dict) else None
    if not isinstance(templates, list):
        raise ValueError("entry template manifest has no cases array")
    standalone_templates = {
        row["entry"]["encounter"]: row for row in templates
        if isinstance(row, dict) and row.get("mode") == "standalone"
    }
    expected_encounters = [
        "cultist", "large_slime", "jaw_worm", "three_louse", "two_louse",
        "large_slime", "small_slimes", "three_louse", "cultist", "large_slime",
        "jaw_worm", "three_louse", "two_louse", "large_slime", "small_slimes",
        "three_louse",
    ]
    records: list[dict[str, object]] = []
    run_templates = [row for row in templates if isinstance(row, dict) and row.get("mode") == "run"]
    if len(standalone_templates) != 6 or len(run_templates) != 1:
        raise ValueError("selection templates must contain six standalone encounters and one Run case")
    for index, encounter in enumerate(expected_encounters):
        cohort = "seen" if encounter in seen_catalog else "held_out_encounter"
        if (encounter in seen_catalog) == (encounter in held_out):
            raise ValueError(f"encounter {encounter} is not classified exactly once")
        entry_id = f"task9-entry-standalone-{index:02d}"
        template = standalone_templates.get(encounter)
        if template is None:
            raise ValueError(f"missing selection entry template for {encounter}")
        entry = json.loads(json.dumps(template["entry"]))
        entry["id"] = entry_id
        standalone = {
            "case_id": f"task9-selection-standalone-{index:02d}",
            "lineage_id": f"lineage:task9-selection:standalone:{index:02d}",
            "mode": "standalone",
            "environment_seed": 2**40 + 4096 + index,
            "cohort": cohort,
            "entry_id": entry_id,
            "entry": entry,
        }
        run = json.loads(json.dumps(run_templates[0]))
        run.update({
            "case_id": f"task9-selection-run-{index:02d}",
            "lineage_id": f"lineage:task9-selection:run:{index:02d}",
            "environment_seed": 2**40 + 4096 + 16 + index,
            "cohort": "normal_run",
        })
        records.extend((standalone, run))
    return records


def build_round_robin(cases: list[dict], policies: list[str]) -> tuple[dict, ...]:
    """Build the fixed 192-row policy-by-case schedule from bound case records.

    Each item in ``cases`` is one ``selection.case_bindings`` record from a
    validated study.  The sequence is 16 standalone/Run pairs; each record
    carries config and manifest hashes that are copied into its operation row.
    """
    if type(cases) is not list or len(cases) != 32:
        raise ValueError("round-robin cases must be the 32 ordered case bindings")
    if type(policies) is not list or tuple(policies) != POLICY_ORDER:
        raise ValueError("round-robin policy order must match the six approved policy IDs")
    expected_case_keys = {
        "case_index", "case_id", "lineage_id", "mode", "cohort", "encounter", "entry_id",
        "environment_seed", "config_path", "config_sha256", "config_raw_sha256",
        "manifest_path", "manifest_sha256", "manifest_raw_sha256",
        "parent_selection_sha256", "output_root",
    }
    by_pair: dict[tuple[int, str], dict[str, object]] = {}
    all_case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case = _exact_keys(case, expected_case_keys, f"round-robin case binding {index}")
        pair_index = index // 2
        expected_mode = "standalone" if index % 2 == 0 else "run"
        if type(case["case_index"]) is not int:
            raise ValueError("round-robin case_index values must be integers")
        if case["case_index"] != pair_index or case["mode"] != expected_mode:
            raise ValueError("round-robin cases must be ordered standalone/Run for indices 0..15")
        if not isinstance(case["case_id"], str) or case["case_id"] in all_case_ids:
            raise ValueError("round-robin case IDs must be unique strings")
        all_case_ids.add(case["case_id"])
        for digest_key in ("config_sha256", "config_raw_sha256", "manifest_sha256",
                           "manifest_raw_sha256", "parent_selection_sha256"):
            _check_sha(case[digest_key], f"round-robin {digest_key}")
        for path_key in ("config_path", "manifest_path", "output_root"):
            if not isinstance(case[path_key], str) or not case[path_key]:
                raise ValueError(f"round-robin {path_key} must be a non-empty path")
        by_pair[(pair_index, expected_mode)] = case

    rows: list[dict[str, object]] = []
    for pair_index in range(16):
        shift = pair_index % len(POLICY_ORDER)
        rotated = POLICY_ORDER[shift:] + POLICY_ORDER[:shift]
        modes = ("standalone", "run") if pair_index % 2 == 0 else ("run", "standalone")
        for policy_id in rotated:
            source_mode = POLICY_MODES[policy_id]
            for mode in modes:
                case = by_pair[(pair_index, mode)]
                classification = (
                    "reference" if source_mode is None
                    else "same-mode" if source_mode == mode
                    else "transfer"
                )
                index = len(rows)
                rows.append({
                    "index": index,
                    "operation_id": f"selection-{index:03d}",
                    "case_index": pair_index,
                    "case_id": case["case_id"],
                    "mode": mode,
                    "cohort": case["cohort"],
                    "policy_id": policy_id,
                    "source_policy_mode": source_mode,
                    "classification": classification,
                    "config_path": case["config_path"],
                    "config_sha256": case["config_sha256"],
                    "config_raw_sha256": case["config_raw_sha256"],
                    "one_case_manifest_path": case["manifest_path"],
                    "one_case_manifest_sha256": case["manifest_sha256"],
                    "one_case_manifest_raw_sha256": case["manifest_raw_sha256"],
                    "parent_selection_sha256": case["parent_selection_sha256"],
                })
    if len(rows) != 192 or len({(row["policy_id"], row["case_id"]) for row in rows}) != 192:
        raise ValueError("round-robin schedule does not contain 192 unique policy/case pairs")
    return tuple(rows)


def _validate_layout(root: Path) -> None:
    base = _bound_path(root, BASE_PACKAGE, "Task 9 package directory")
    selection_dir = _bound_path(root, f"{BASE_PACKAGE}/selection", "Task 9 selection directory")
    _require_directory(base, "Task 9 package directory")
    _require_directory(selection_dir, "Task 9 selection directory")
    expected_root = {"study.json", "selection"} | {f"{name}.toml" for name in TRAINING_ORDER}
    actual_root = {path.name for path in base.iterdir()}
    if actual_root != expected_root:
        raise ValueError(f"Task 9 package inventory/order has unexpected root entries: {sorted(actual_root)}")
    if any(path.is_symlink() for path in base.iterdir()):
        raise ValueError("Task 9 package inventory contains a symlink")
    expected_selection = {f"case-{index:02d}-{mode}.toml"
                          for index in range(16) for mode in ("standalone", "run")}
    actual_selection = {path.name for path in selection_dir.iterdir()}
    if actual_selection != expected_selection:
        raise ValueError("Task 9 selection config inventory/order is not exactly 32 case TOMLs")
    if any(path.is_symlink() for path in selection_dir.iterdir()):
        raise ValueError("Task 9 selection config inventory contains a symlink")
    case_dir = _bound_path(root, "configs/cases/neural", "neural case directory")
    expected_manifests = {"task9-selection.json"} | {
        f"task9-selection-case-{index:02d}-{mode}.json"
        for index in range(16) for mode in ("standalone", "run")
    }
    actual_manifests = {path.name for path in case_dir.iterdir()
                        if path.name.startswith("task9-selection")}
    if actual_manifests != expected_manifests:
        raise ValueError("Task 9 selection manifest inventory/order is not exactly parent plus 32 one-case files")


def validate_study(root: Path, study_path: Path) -> dict[str, object]:
    """Load and verify the immutable Task 9 study and every authored binding."""
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("root must be an existing repository directory")
    supplied_study = Path(study_path)
    if supplied_study.is_absolute():
        study_file = _ensure_inside(root_path, supplied_study, "study path")
    else:
        study_file = _bound_path(root_path, supplied_study.as_posix(), "study path")
    _require_file(study_file, "study file")
    _validate_layout(root_path)
    raw_study, parsed_study = _read_json(study_file, "study file")
    study = _exact_keys(parsed_study, {
        "schema", "state", "base_commit", "runtime_pins", "budgets", "interpretation",
        "training", "selection", "checkpoint_binding",
    }, "study")
    if study["schema"] != STUDY_SCHEMA or study["state"] != STUDY_STATE:
        raise ValueError("study must be an unreleased spire-learn-task9-study-v1 protocol")
    if not isinstance(study["base_commit"], str) or _COMMIT.fullmatch(study["base_commit"]) is None:
        raise ValueError("study base_commit must be a full lowercase Git commit ID")
    if study_file != _bound_path(root_path, f"{BASE_PACKAGE}/study.json", "study file"):
        raise ValueError("study file must be the authored Task 9 study.json")
    if not raw_study:
        raise ValueError("study file cannot be empty")

    if not _same_json_value(study["runtime_pins"], _RUNTIME_PINS):
        raise ValueError("runtime pins differ from the specified qualified environment")
    if not _same_json_value(study["budgets"], _BUDGETS):
        raise ValueError("study caps, seeds, and storage budgets differ from the specification")
    interpretation = _exact_keys(study["interpretation"],
                                 set(_INTERPRETATION) | {"final_manifest"}, "interpretation")
    for key, expected in _INTERPRETATION.items():
        if not _same_json_value(interpretation[key], expected):
            raise ValueError(f"interpretation.{key} differs from the approved qualification")

    training = _exact_keys(study["training"], {"order", "manifest", "identity_hashes", "runs"},
                           "training")
    if not _same_json_value(training["order"], list(TRAINING_ORDER)):
        raise ValueError("training run inventory/order must be standalone-a, run-a, standalone-b, run-b")
    if not _same_json_value(training["identity_hashes"],
                            {"model": _MODEL_HASH, "ppo": _PPO_HASH}):
        raise ValueError("training model/PPO hashes differ from the accepted defaults")
    train_cases, train_raw, train_path = _manifest_binding(
        root_path, training["manifest"], TRAINING_CASES, "training manifest")
    if not isinstance(train_raw, dict) or train_cases.split != "train":
        raise ValueError("training manifest must be the existing train split")

    final_cases, final_raw, _ = _manifest_binding(
        root_path, interpretation["final_manifest"], FINAL_CASES, "final manifest")
    if not isinstance(final_raw, dict) or final_cases.split != "final":
        raise ValueError("final manifest binding must identify the unused final split")

    run_values = training["runs"]
    if not isinstance(run_values, list) or len(run_values) != 4:
        raise ValueError("training must contain exactly four runs")
    output_roots: set[str] = set()
    run_keys = {"policy_id", "mode", "config", "seeds", "rollout", "output_root"}
    for index, (policy_id, run_value) in enumerate(zip(TRAINING_ORDER, run_values)):
        run = _exact_keys(run_value, run_keys, f"training run {index}")
        if run["policy_id"] != policy_id or run["mode"] != policy_id.split("-")[0]:
            raise ValueError("training run inventory/order or mode substitution detected")
        expected_config_path = f"{BASE_PACKAGE}/{policy_id}.toml"
        config, _, _ = _config_binding(
            root_path, run["config"], expected_config_path, f"training config {policy_id}")
        _validate_common_config(config, f"training config {policy_id}")
        expected_mode = policy_id.split("-")[0]
        expected_rows = 16 if expected_mode == "standalone" else 1
        expected_steps = 32 if expected_mode == "standalone" else 128
        if (config.mode, config.operation, config.split, config.partition) != (
                expected_mode, "train", "train", "train"):
            raise ValueError(f"training config {policy_id} must be fresh {expected_mode} train mode")
        if config.device != "cuda:0" or config.runtime_gpu_uuid != _GPU_UUID:
            raise ValueError(f"training config {policy_id} must pin the approved cuda:0 GPU")
        if config.calibration_gpu_uuid is not None:
            raise ValueError(f"training config {policy_id} cannot use calibration-only GPU controls")
        if (config.environment_revision != _GYM_REVISION
                or config.common_view_fingerprint != _COMMON_FINGERPRINT):
            raise ValueError(f"training config {policy_id} has an unsupported environment fingerprint")
        if expected_mode == "run" and config.run_observation_fingerprint != _RUN_FINGERPRINT:
            raise ValueError(f"training config {policy_id} has an unsupported Run observation fingerprint")
        if (config.rollout_rows, config.rollout_steps) != (expected_rows, expected_steps):
            raise ValueError(f"training config {policy_id} has the wrong native rollout shape")
        if (config.budgets.max_training_decisions, config.budgets.max_environment_actions,
                config.budgets.max_optimizer_steps, config.budgets.max_wall_seconds) != (
                16_384, 65_536, 1_024, 600.0):
            raise ValueError(f"training config {policy_id} differs from the per-run ceilings")
        if run["seeds"] != TRAINING_SEEDS[policy_id] or {
                "policy": config.seed_streams.policy,
                "optimizer": config.seed_streams.optimizer,
                "curriculum": config.seed_streams.curriculum,
                "noncombat": config.seed_streams.noncombat,
        } != TRAINING_SEEDS[policy_id]:
            raise ValueError(f"training config {policy_id} seed quartet differs from the specification")
        if run["rollout"] != {"rows": expected_rows, "steps": expected_steps}:
            raise ValueError(f"training run {policy_id} rollout binding differs from its config")
        if config.manifest_path != train_path:
            raise ValueError(f"training config {policy_id} does not consume the bound train manifest")
        output_root = _bound_path(root_path, run["output_root"],
                                  f"training output root {policy_id}")
        if (output_root != config.output_root
                or run["output_root"] != f"artifacts/development/neural-combat/experiments/task9-v1/training/{policy_id}"):
            raise ValueError(f"training output root for {policy_id} must be its unique relative slot")
        if run["output_root"] in output_roots:
            raise ValueError("training/evaluation output roots must be unique")
        output_roots.add(run["output_root"])

    selection = _exact_keys(study["selection"], {
        "parent_manifest", "entry_template_manifest", "policy_order", "policies", "eval_seeds",
        "case_bindings", "schedule",
    }, "selection")
    if selection["policy_order"] != list(POLICY_ORDER):
        raise ValueError("selection policy order differs from the six approved policies")
    parent_cases, parent_raw, parent_path = _manifest_binding(
        root_path, selection["parent_manifest"], PARENT_SELECTION, "parent selection manifest")
    if not isinstance(parent_raw, dict) or parent_cases.split != "selection":
        raise ValueError("parent selection manifest must be the new selection split")
    if selection["eval_seeds"] != EVAL_SEEDS:
        raise ValueError("selection evaluation seed quartet differs from the matched seed set")

    # The selection template binding also lives under selection.  This read is
    # the source for existing entry shapes; no simulator or guessed shape is used.
    template_binding = selection["entry_template_manifest"]
    template_cases, template_raw, _ = _manifest_binding(
        root_path, template_binding, ENTRY_TEMPLATE, "selection entry template manifest")
    if not isinstance(template_raw, dict) or template_cases.split != "selection":
        raise ValueError("selection entry template manifest must remain the existing selection split")

    validate_neural_case_lineage((train_cases, parent_cases, final_cases))
    expected_records = _expected_case_records(train_cases, template_raw)
    parent_records = parent_raw.get("cases")
    if not _same_json_value(parent_records, expected_records):
        raise ValueError("parent selection cases differ from exact template-derived identity/order")
    if len(parent_cases.cases) != 32:
        raise ValueError("parent selection must contain exactly 32 cases")

    case_values = selection["case_bindings"]
    if not isinstance(case_values, list) or len(case_values) != 32:
        raise ValueError("selection case binding inventory must contain exactly 32 ordered cases")
    case_keys = {
        "case_index", "case_id", "lineage_id", "mode", "cohort", "encounter", "entry_id",
        "environment_seed", "config_path", "config_sha256", "config_raw_sha256",
        "manifest_path", "manifest_sha256", "manifest_raw_sha256",
        "parent_selection_sha256", "output_root",
    }
    validated_case_bindings: list[dict[str, object]] = []
    case_ids: set[str] = set()
    case_manifest_paths: set[str] = set()
    config_paths: set[str] = set()
    for index, (binding_value, record) in enumerate(zip(case_values, expected_records)):
        binding = _exact_keys(binding_value, case_keys, f"selection case binding {index}")
        pair_index = index // 2
        mode = "standalone" if index % 2 == 0 else "run"
        expected_config_name = f"{BASE_PACKAGE}/selection/case-{pair_index:02d}-{mode}.toml"
        expected_manifest_name = (
            f"configs/cases/neural/task9-selection-case-{pair_index:02d}-{mode}.json"
        )
        expected_case_id = record["case_id"]
        if type(binding["case_index"]) is not int:
            raise ValueError(f"selection case binding {index} case_index must be an integer")
        if (binding["case_index"], binding["mode"], binding["case_id"], binding["lineage_id"],
                binding["cohort"], binding["environment_seed"]) != (
                pair_index, mode, expected_case_id, record["lineage_id"], record["cohort"],
                record["environment_seed"]):
            raise ValueError(f"selection case binding {index} identity/order differs from its parent record")
        encounter = record["entry"]["encounter"] if mode == "standalone" else None
        entry_id = record["entry_id"] if mode == "standalone" else None
        if binding["encounter"] != encounter or binding["entry_id"] != entry_id:
            raise ValueError(f"selection case binding {index} encounter/entry identity differs")
        if binding["config_path"] != expected_config_name:
            raise ValueError(f"selection config inventory/order differs at case {index}")
        if binding["manifest_path"] != expected_manifest_name:
            raise ValueError(f"selection one-case manifest inventory/order differs at case {index}")
        if binding["parent_selection_sha256"] != parent_cases.sha256:
            raise ValueError(f"selection case binding {index} has the wrong parent selection hash")
        if binding["case_id"] in case_ids or binding["manifest_path"] in case_manifest_paths:
            raise ValueError("selection case IDs and one-case manifest paths must be unique")
        case_ids.add(binding["case_id"])
        case_manifest_paths.add(binding["manifest_path"])

        one_case_path = _bound_path(root_path, binding["manifest_path"],
                                    f"one-case manifest {binding['case_id']}")
        _require_file(one_case_path, f"one-case manifest {binding['case_id']}")
        child_cases, child_raw = load_neural_cases(one_case_path), _read_json(
            one_case_path, f"one-case manifest {binding['case_id']}"
        )[1]
        if (not isinstance(child_raw, dict) or not _same_json_value(child_raw.get("cases"), [record])
                or child_cases.split != "selection"):
            raise ValueError(f"one-case manifest {binding['case_id']} does not equal its exact parent record")
        _check_sha(binding["manifest_sha256"], f"one-case {binding['case_id']} manifest_sha256")
        _check_sha(binding["manifest_raw_sha256"], f"one-case {binding['case_id']} manifest_raw_sha256")
        if binding["manifest_sha256"] != child_cases.sha256:
            raise ValueError(f"one-case manifest {binding['case_id']} canonical SHA-256 mismatch")
        if binding["manifest_raw_sha256"] != _sha256(one_case_path.read_bytes()):
            raise ValueError(f"one-case manifest {binding['case_id']} raw SHA-256 mismatch")

        config, _, _ = _config_binding(
            root_path, {
                "path": binding["config_path"],
                "config_sha256": binding["config_sha256"],
                "raw_sha256": binding["config_raw_sha256"],
            }, expected_config_name, f"selection config {binding['case_id']}")
        _validate_common_config(config, f"selection config {binding['case_id']}")
        if (config.mode, config.operation, config.split, config.partition, config.device) != (
                mode, "evaluate", "selection", "selection", "cpu"):
            raise ValueError(f"selection config {binding['case_id']} has the wrong mode/backend")
        if config.manifest_path != one_case_path:
            raise ValueError(f"selection config {binding['case_id']} does not load its one-case manifest")
        if (config.rollout_rows, config.rollout_steps) != (1, 1):
            raise ValueError(f"selection config {binding['case_id']} must use rows=1 and steps=1")
        if (config.budgets.max_optimizer_steps, config.budgets.max_training_decisions,
                config.budgets.max_environment_actions, config.budgets.max_wall_seconds) != (
                1, 256, 256, 600.0):
            raise ValueError(f"selection config {binding['case_id']} differs from the per-case caps")
        if config.runtime_gpu_uuid is not None or config.calibration_gpu_uuid is not None:
            raise ValueError(f"selection config {binding['case_id']} must remain CPU-only")
        if config.seed_streams != type(config.seed_streams)(**EVAL_SEEDS):
            raise ValueError(f"selection config {binding['case_id']} eval seed quartet differs")
        expected_output = (
            f"artifacts/development/neural-combat/experiments/task9-v1/selection/{record['case_id']}"
        )
        if binding["output_root"] != expected_output or config.output_root != _bound_path(
                root_path, expected_output, f"selection output root {record['case_id']}"):
            raise ValueError(f"selection output root for {record['case_id']} is not its unique relative slot")
        if expected_output in output_roots:
            raise ValueError("training/evaluation output roots must be globally unique")
        output_roots.add(expected_output)
        if binding["config_path"] in config_paths:
            raise ValueError("selection config paths must be unique")
        config_paths.add(binding["config_path"])
        validated_case_bindings.append(binding)

    policy_values = selection["policies"]
    if not isinstance(policy_values, list) or len(policy_values) != 6:
        raise ValueError("selection policy inventory must contain exactly six policies")
    policy_keys = {
        "policy_id", "kind", "source_policy_mode", "source_config_path", "source_config_sha256",
        "source_config_raw_sha256", "reference_id", "device", "imports_torch",
    }
    for index, (policy_value, policy_id) in enumerate(zip(policy_values, POLICY_ORDER)):
        policy = _exact_keys(policy_value, policy_keys, f"selection policy {index}")
        if policy["policy_id"] != policy_id:
            raise ValueError("selection policy order or policy substitution detected")
        if type(policy["imports_torch"]) is not bool:
            raise ValueError(f"selection policy {policy_id} imports_torch must be a JSON boolean")
        source_mode = POLICY_MODES[policy_id]
        if policy["device"] != "cpu" or policy["source_policy_mode"] != source_mode:
            raise ValueError(f"selection policy {policy_id} mode/backend binding is invalid")
        if source_mode is None:
            if (policy["kind"], policy["source_config_path"], policy["source_config_sha256"],
                    policy["source_config_raw_sha256"], policy["reference_id"],
                    policy["imports_torch"]) != (
                    "reference", None, None, None, policy_id, False):
                raise ValueError(f"reference policy {policy_id} was substituted or enables Torch")
        else:
            run = next(item for item in run_values if item["policy_id"] == policy_id)
            source_config = run["config"]
            if (policy["kind"], policy["source_config_path"], policy["source_config_sha256"],
                    policy["source_config_raw_sha256"], policy["reference_id"],
                    policy["imports_torch"]) != (
                    "learned", source_config["path"], source_config["config_sha256"],
                    source_config["raw_sha256"], None, True):
                raise ValueError(f"learned selection policy {policy_id} is not bound to its train config")
    if selection["eval_seeds"] != EVAL_SEEDS:
        raise ValueError("selection policy seed binding differs from the specified quartet")

    checkpoint = _exact_keys(study["checkpoint_binding"], {
        "state", "training_run_id_source", "final_slot", "inventory_policy",
        "require_finite_accepted_update", "freeze_before_selection",
        "verify_before_and_after_each_use", "unavailable_policy", "policies",
    }, "checkpoint_binding")
    if (type(checkpoint["require_finite_accepted_update"]) is not bool
            or type(checkpoint["freeze_before_selection"]) is not bool
            or type(checkpoint["verify_before_and_after_each_use"]) is not bool):
        raise ValueError("checkpoint binding flags must be JSON booleans")
    if (checkpoint["state"], checkpoint["training_run_id_source"], checkpoint["final_slot"],
            checkpoint["inventory_policy"], checkpoint["require_finite_accepted_update"],
            checkpoint["freeze_before_selection"], checkpoint["verify_before_and_after_each_use"],
            checkpoint["unavailable_policy"]) != (
            "future_inventory_required", "successful-training-receipt.run_id", "final",
            "post-run-inventory", True, True, True, "not_started_no_substitute"):
        raise ValueError("future checkpoint binding protocol differs from the specified inventory policy")
    checkpoint_policies = checkpoint["policies"]
    if not isinstance(checkpoint_policies, list) or len(checkpoint_policies) != 4:
        raise ValueError("checkpoint binding must have one future slot for each trained policy")
    expected_checkpoint = [
        {"policy_id": policy_id,
         "training_run_id_source": "successful-training-receipt.run_id",
         "final_slot": "final",
         "checkpoint_directory_source": "final-checkpoint-from-post-run-inventory",
         "metadata_and_payload_hashes_source": "post-run-inventory"}
        for policy_id in TRAINING_ORDER
    ]
    if checkpoint_policies != expected_checkpoint:
        raise ValueError("checkpoint inventory policies contain a substitution or invented binding")

    expected_schedule = build_round_robin(validated_case_bindings, list(POLICY_ORDER))
    if not _same_json_value(selection["schedule"], list(expected_schedule)):
        raise ValueError("selection schedule does not match the 192 ordered round-robin bindings")
    if len({(row["policy_id"], row["case_id"]) for row in expected_schedule}) != 192:
        raise ValueError("selection schedule has a duplicate or missing policy/case pair")
    return study


def _describe() -> dict[str, object]:
    return {
        "schema": STUDY_SCHEMA,
        "state": STUDY_STATE,
        "apis": [
            "validate_study(root: Path, study_path: Path) -> dict[str, object]",
            "build_round_robin(cases: list[dict], policies: list[str]) -> tuple[dict, ...]",
        ],
        "study_keys": ["schema", "state", "base_commit", "runtime_pins", "budgets",
                       "interpretation", "training", "selection", "checkpoint_binding"],
        "validation_only": True,
        "runtime_dispatch": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="repository root")
    parser.add_argument("--study", type=Path, help="repository-relative study JSON path")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--validate", action="store_true", help="validate all authored bindings")
    operation.add_argument("--describe", action="store_true", help="describe the pure package API")
    args = parser.parse_args(argv)
    if args.describe:
        if args.root is not None or args.study is not None:
            parser.error("--describe does not accept --root or --study")
        print(json.dumps(_describe(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.root is None or args.study is None:
        parser.error("--validate requires both --root and --study")
    try:
        validated = validate_study(args.root, args.study)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"task9-study-package: {exc}", file=sys.stderr)
        return 2
    summary = {
        "schema": validated["schema"],
        "state": validated["state"],
        "base_commit": validated["base_commit"],
        "training_run_count": len(validated["training"]["runs"]),
        "selection_case_count": len(validated["selection"]["case_bindings"]),
        "policy_count": len(validated["selection"]["policies"]),
        "schedule_count": len(validated["selection"]["schedule"]),
        "future_checkpoint_count": len(validated["checkpoint_binding"]["policies"]),
        "runtime_dispatch": False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
