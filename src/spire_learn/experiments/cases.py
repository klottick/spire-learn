"""Strict, explicit case manifest loading and split-overlap checks."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from spire_learn.contracts import CaseExecution

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_FINGERPRINT = re.compile(r"[0-9a-fA-F]{64}\Z")
_SPLITS = {"train", "selection", "final"}


@dataclass(frozen=True)
class CaseManifest:
    path: Path
    split: str
    executions: tuple[CaseExecution, ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def load_cases(path: Path) -> CaseManifest:
    """Load and strictly validate one versioned JSON case manifest."""
    resolved_path = Path(path).resolve()
    try:
        with resolved_path.open(encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cases manifest {resolved_path}: {exc}") from exc

    if not isinstance(raw, dict) or set(raw) != {"schema", "split", "cases"}:
        raise ValueError("manifest must contain only schema, split, and cases")
    if raw["schema"] != "spire-learn-cases-v1":
        raise ValueError("unsupported manifest schema")
    split = raw["split"]
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("invalid manifest split")
    if not isinstance(raw["cases"], list) or not raw["cases"]:
        raise ValueError("cases must be a non-empty array")

    case_ids: set[str] = set()
    execution_ids: set[str] = set()
    executions: list[CaseExecution] = []
    required_case = {"case_id", "scenario_id", "partition", "executions"}
    allowed_case = required_case | {"expected_schema_id", "expected_fingerprint"}
    for case in raw["cases"]:
        if not isinstance(case, dict) or not required_case <= case.keys() or case.keys() - allowed_case:
            raise ValueError("invalid case fields")
        case_id = _identifier(case["case_id"], "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        scenario_id = _identifier(case["scenario_id"], "scenario_id")
        partition = _identifier(case["partition"], "partition")
        schema_id = case.get("expected_schema_id")
        if schema_id is not None:
            schema_id = _identifier(schema_id, "expected_schema_id")
        fingerprint = case.get("expected_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None):
            raise ValueError("expected_fingerprint must be 64 hexadecimal characters")
        case_executions = case["executions"]
        if not isinstance(case_executions, list) or not case_executions:
            raise ValueError("executions must be a non-empty array")
        for execution in case_executions:
            if not isinstance(execution, dict) or set(execution) != {"execution_id", "environment_seed", "policy_seed"}:
                raise ValueError("invalid execution fields")
            execution_id = _identifier(execution["execution_id"], "execution_id")
            if execution_id in execution_ids:
                raise ValueError(f"duplicate execution_id: {execution_id}")
            execution_ids.add(execution_id)
            environment_seed = execution["environment_seed"]
            policy_seed = execution["policy_seed"]
            if any(type(seed) is not int or seed < 0 for seed in (environment_seed, policy_seed)):
                raise ValueError("seeds must be non-negative integers")
            executions.append(CaseExecution(
                execution_id, case_id, split, partition, scenario_id,
                environment_seed, policy_seed, schema_id, fingerprint,
            ))
    return CaseManifest(resolved_path, split, tuple(executions))


def validate_splits(manifests: Sequence[CaseManifest]) -> None:
    """Reject duplicate identities and scenario/seed overlap across splits."""
    case_ids: set[str] = set()
    execution_ids: set[str] = set()
    identities: dict[tuple[str, int], str] = {}
    for manifest in manifests:
        if manifest.split not in _SPLITS:
            raise ValueError(f"invalid manifest split: {manifest.split}")
        for execution in manifest.executions:
            if execution.split != manifest.split:
                raise ValueError("execution split disagrees with manifest split")
            if execution.case_id in case_ids:
                raise ValueError(f"duplicate case_id across manifests: {execution.case_id}")
            if execution.execution_id in execution_ids:
                raise ValueError(f"duplicate execution_id across manifests: {execution.execution_id}")
            case_ids.add(execution.case_id)
            execution_ids.add(execution.execution_id)
            key = (execution.scenario_id, execution.environment_seed)
            previous_split = identities.get(key)
            if previous_split is not None and previous_split != manifest.split:
                raise ValueError("cross-split case overlap")
            identities[key] = manifest.split
