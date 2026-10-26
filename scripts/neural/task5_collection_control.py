"""Pure host-side control checks for the protected Task 5 collection wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


EXPECTED_IMAGE = "docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
EXPECTED_GYM_REVISION = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
EXPECTED_COMMON_FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"
EXPECTED_RUN_FINGERPRINT = "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8"
EXPECTED_TEST_IDS = (
    "tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_run_natural_task_end_without_run_reset",
    "tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_standalone_manual_resets",
)
PUBLIC_DIAGNOSTIC_KIND = "public-diagnostic-lowest-legal-v1"
PUBLIC_DIAGNOSTIC_VERSION = "public-diagnostic-lowest-legal-v1"
PUBLIC_DIAGNOSTIC_MODEL_ID = "spire-learn-public-diagnostic-lowest-legal-v1"
PUBLIC_DIAGNOSTIC_ACTION_ID = "spire-learn-public-lowest-legal-action-v1"
PUBLIC_DIAGNOSTIC_SPEC = (
    "frames-only; choose the minimum index whose public action_mask value is 1; "
    "point-mass distribution; log_probability=0.0; value=0.0; "
    "behavior_policy_version=public-diagnostic-lowest-legal-v1"
)
PUBLIC_DIAGNOSTIC_SPEC_SHA256 = hashlib.sha256(PUBLIC_DIAGNOSTIC_SPEC.encode("utf-8")).hexdigest()
SHARED_COHORT_BUDGET_ID = "task5-native-cohort-v1"
SHARED_COHORT_BUDGET_SECONDS = 120.0
CLEANUP_SHELL_FILESYSTEM_RESERVE_SECONDS = 45
MAX_MODE_SLOT_ACTIONS = 2048
MAX_TOTAL_SLOT_ACTIONS = 4096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_COUNTER_FIELDS = (
    "attempted_environment_slots", "forwarded_environment_slots",
    "known_accepted_environment_slots", "delivered_environment_slots",
    "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
    "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
    "reset_operations", "reset_slots", "next_seed_ordinal", "tasks_completed",
    "task_terminals",
)


def extract_container_command(script_text: str) -> str:
    """Return the exact reviewed command body passed after ``/bin/sh -ec``."""
    if not isinstance(script_text, str):
        raise TypeError("container command source must be text")
    match = re.search(
        r"CONTAINER_COMMAND=\$\(cat <<'COMMAND'\n(.*?)\nCOMMAND\n\)",
        script_text,
        re.DOTALL,
    )
    if match is None or not match.group(1):
        raise ValueError("the protected wrapper must contain one non-empty reviewed command body")
    if re.search(r"CONTAINER_COMMAND=\$\(cat <<'COMMAND'", script_text[match.end():]):
        raise ValueError("the protected wrapper contains multiple container command bodies")
    return match.group(1)


def cleanup_timeout_worst_case_seconds(script_text: str) -> int:
    """Include every hard cleanup deadline plus the reserved shell/filesystem margin."""
    start = script_text.find("finish() {")
    end = script_text.find("\n}\n\ntrap finish EXIT", start)
    if start < 0 or end < 0:
        raise ValueError("protected wrapper cleanup function is missing or malformed")
    cleanup = script_text[start:end]
    calls = re.findall(
        r"record\s+([A-Za-z0-9_-]+)\s+timeout\s+--signal=TERM\s+"
        r"--kill-after=(\d+)s\s+(\d+)s",
        cleanup,
    )
    expected = (
        "exact-name-recovery-inventory", "container-remove", "final-name-inventory",
        "source-after", "source-compare", "final-evidence-inventory",
    )
    record_names = re.findall(r"^\s*if record ([A-Za-z0-9_-]+)\b", cleanup, re.MULTILINE)
    timeout_names = [name for name, _kill_after, _soft in calls]
    if record_names != list(expected) or timeout_names != list(expected):
        raise ValueError("cleanup subprocess deadlines do not cover each required bounded phase exactly once")
    bounded_children = sum(int(kill_after) + int(soft) for _name, kill_after, soft in calls)
    return bounded_children + CLEANUP_SHELL_FILESYSTEM_RESERVE_SECONDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _regular_tree(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        top = root / name
        if not top.is_dir():
            raise ValueError(f"required source input directory is missing: {name}")
        for current, directories, files in os.walk(top, followlinks=False):
            directories[:] = sorted(
                directory for directory in directories
                if not (Path(current) / directory).is_symlink()
            )
            for filename in sorted(files):
                path = Path(current) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                result[path.relative_to(root).as_posix()] = _sha256(path)
    for relative in (
        "scripts/neural/run-task5-collection.sh",
        "scripts/neural/task5_collection_control.py",
        "scripts/neural/task5-collection-runbook.md",
    ):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            result[relative] = _sha256(path)
    if not result:
        raise ValueError("source input inventory is empty")
    return result


def source_snapshot(root: Path) -> dict[str, object]:
    root = root.resolve()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status_text = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    files = _regular_tree(root, ("src", "tests", "configs"))
    return {
        "learner_head": head,
        "dirty_status": status_text.splitlines(),
        "input_files": files,
        "input_file_count": len(files),
        "input_tree_sha256": _canonical_hash(files),
        "config_sha256": {
            name: files[name]
            for name in (
                "configs/neural/standalone.toml",
                "configs/neural/run.toml",
                "configs/cases/neural/train.json",
            )
        },
        "gym_revision": EXPECTED_GYM_REVISION,
        "common_view_fingerprint": EXPECTED_COMMON_FINGERPRINT,
        "run_observation_fingerprint": EXPECTED_RUN_FINGERPRINT,
        "cpu_image": EXPECTED_IMAGE,
        "cpu_runtime_root": "artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc",
        "cpu_runtime_venv": "data/venv-cpu (read-only container mount)",
        "cpu_runtime_packages": {
            "spire-lab": "0.1.0",
            "gymnasium": "1.4.0",
            "numpy": "2.5.4",
            "torch": "2.14.1+cpu",
        },
        "cpu_lock_sha256": "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0",
        "gym_native_sha256": "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9",
        "gym_source_map_sha256": "dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3",
        "gym_durable_source_files": 636,
        "gym_original_post_build_source_hash_available": False,
    }


def compare_snapshots(before: dict[str, object], after: dict[str, object]) -> None:
    if before != after:
        raise ValueError("mounted learner inputs, HEAD, or dirty state changed during the run")


def pre_dispatch_manifest(snapshot: object) -> dict[str, object]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("config_sha256"), dict):
        raise ValueError("pre-dispatch source snapshot is missing exact config hashes")
    return {
        "schema": "spire-learn-neural-collection-pre-dispatch-v1",
        "expected_test_ids": list(EXPECTED_TEST_IDS),
        "limits": {
            "standalone": {"attempted_environment_slots": MAX_MODE_SLOT_ACTIONS,
                           "forwarded_environment_slots": MAX_MODE_SLOT_ACTIONS,
                           "known_accepted_environment_slots": MAX_MODE_SLOT_ACTIONS,
                           "delivered_environment_slots": MAX_MODE_SLOT_ACTIONS,
                           "uncertain_environment_slots": 0},
            "run": {"attempted_environment_slots": MAX_MODE_SLOT_ACTIONS,
                    "forwarded_environment_slots": MAX_MODE_SLOT_ACTIONS,
                    "known_accepted_environment_slots": MAX_MODE_SLOT_ACTIONS,
                    "delivered_environment_slots": MAX_MODE_SLOT_ACTIONS,
                    "uncertain_environment_slots": 0},
            "combined_attempted_environment_slots": MAX_TOTAL_SLOT_ACTIONS,
            "soft_seconds": 120,
            "whole_wrapper_active_seconds": 210,
            "whole_wrapper_cleanup_seconds": 90,
            "shared_budget_id": SHARED_COHORT_BUDGET_ID,
        },
        "initial_counters": {
            mode: {name: 0 for name in (
                "attempted_environment_slots", "forwarded_environment_slots",
                "known_accepted_environment_slots", "delivered_environment_slots",
                "uncertain_environment_slots",
            )}
            for mode in ("standalone", "run")
        },
        "initial_reset_counters": {
            mode: {name: 0 for name in (
                "attempted_reset_slots", "forwarded_reset_slots",
                "known_accepted_reset_slots", "delivered_reset_slots",
                "uncertain_reset_slots", "reset_operations", "reset_slots",
            )}
            for mode in ("standalone", "run")
        },
        "optimizer_updates": 0,
        "gym_revision": EXPECTED_GYM_REVISION,
        "common_view_fingerprint": EXPECTED_COMMON_FINGERPRINT,
        "run_observation_fingerprint": EXPECTED_RUN_FINGERPRINT,
        "cpu_image": EXPECTED_IMAGE,
        "cpu_lock_sha256": snapshot["cpu_lock_sha256"],
        "gym_native_sha256": snapshot["gym_native_sha256"],
        "gym_source_map_sha256": snapshot["gym_source_map_sha256"],
        "config_sha256": snapshot["config_sha256"],
        "learner_head": snapshot["learner_head"],
        "dirty_status": snapshot["dirty_status"],
        "input_tree_sha256": snapshot["input_tree_sha256"],
        "provenance_qualification": {
            "gym_durable_source_files": snapshot["gym_durable_source_files"],
            "original_post_build_source_hash_available": False,
        },
    }


def evidence_inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.is_dir():
        raise ValueError("run evidence directory is missing")
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            directory for directory in directories
            if not (Path(current) / directory).is_symlink()
        )
        for filename in sorted(files):
            path = Path(current) / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"run evidence contains a non-regular file: {path}")
            result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _verbose_test_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    label = re.compile(r"^(test_[A-Za-z0-9_]+) \(([^)]+)\) \.\.\. (.+)$")
    for line in text.splitlines():
        match = label.fullmatch(line)
        if match is not None:
            rows.append((match.group(2), match.group(3).strip()))
    return rows


def parse_text_test_result(
    stdout: str,
    stderr: str,
    *,
    exit_code: int,
) -> dict[str, object]:
    text = f"{stdout}\n{stderr}"
    rows = _verbose_test_rows(text)
    summary = re.search(r"^Ran (\d+) tests? in [0-9.]+s?$", text, re.MULTILINE)
    observed = [test_id for test_id, _ in rows]
    statuses = {test_id: status for test_id, status in rows}
    passed = (
        exit_code == 0
        and summary is not None
        and int(summary.group(1)) == len(EXPECTED_TEST_IDS)
        and tuple(sorted(observed)) == tuple(sorted(EXPECTED_TEST_IDS))
        and all(statuses.get(test_id) == "ok" for test_id in EXPECTED_TEST_IDS)
        and re.search(r"^OK$", text, re.MULTILINE) is not None
        and "skipped=" not in text
        and "FAILED" not in text
    )
    return {
        "passed": passed,
        "exit_code": exit_code,
        "expected_test_ids": list(EXPECTED_TEST_IDS),
        "observed_test_ids": observed,
        "test_statuses": statuses,
        "summary_count": None if summary is None else int(summary.group(1)),
        "skipped": len([status for status in statuses.values() if status.startswith("skipped")]),
    }


def _nonnegative_int(record: dict[str, object], name: str) -> int:
    value = record.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"collection evidence {name} must be a non-negative integer")
    return value


def _run_counters(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_RUN_COUNTER_FIELDS):
        raise ValueError(f"{name} must report every Run boundary counter exactly once")
    return {field: _nonnegative_int(value, field) for field in _RUN_COUNTER_FIELDS}


def _counter_delta(
    before: dict[str, int], after: dict[str, int], *, context: str,
) -> dict[str, int]:
    if any(after[name] < before[name] for name in _RUN_COUNTER_FIELDS):
        raise ValueError(f"{context} Run counters moved backwards")
    return {name: after[name] - before[name] for name in _RUN_COUNTER_FIELDS}


def _require_action_delta(delta: dict[str, int], *, context: str) -> None:
    expected = {
        "attempted_environment_slots": 1,
        "forwarded_environment_slots": 1,
        "known_accepted_environment_slots": 1,
        "delivered_environment_slots": 1,
        "uncertain_environment_slots": 0,
        "attempted_reset_slots": 0,
        "forwarded_reset_slots": 0,
        "known_accepted_reset_slots": 0,
        "uncertain_reset_slots": 0,
        "reset_operations": 0,
        "reset_slots": 0,
        "delivered_reset_slots": 0,
        "next_seed_ordinal": 0,
    }
    if any(delta[name] != value for name, value in expected.items()):
        raise ValueError(f"{context} does not prove one delivered action without a reset")


def _finite_nonnegative(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"collection evidence {name} must be finite and non-negative")
    return float(value)


def _diagnostic_policy_metadata(value: object) -> None:
    expected = {
        "kind": PUBLIC_DIAGNOSTIC_KIND,
        "behavior_policy_version": PUBLIC_DIAGNOSTIC_VERSION,
        "distribution": "point-mass-on-minimum-legal-action-mask-index",
        "value_estimator": "constant-zero-v1",
        "policy_spec": PUBLIC_DIAGNOSTIC_SPEC,
        "identity": {
            "model_id": PUBLIC_DIAGNOSTIC_MODEL_ID,
            "features_id": "spire-learn-combat-features-v1",
            "action_id": PUBLIC_DIAGNOSTIC_ACTION_ID,
            "objective_id": "spire-learn-combat-winloss-v1",
            "policy_spec_sha256": PUBLIC_DIAGNOSTIC_SPEC_SHA256,
        },
    }
    if value != expected:
        raise ValueError("Run evidence diagnostic policy identity or truthful behavior metadata changed")


def _run_task_identity(value: object, *, stream_id: str, visit: int) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError("Run boundary task ID must be episode and visit qualified")
    match = re.fullmatch(r"task-(\d+)-run-(\d+)-visit-(\d+)", value)
    stream_match = re.fullmatch(r"stream-(\d+)", stream_id)
    if match is None or stream_match is None or int(match.group(1)) != int(stream_match.group(1)):
        raise ValueError("Run boundary task ID is not qualified to its stream and episode")
    episode, task_visit = int(match.group(2)), int(match.group(3))
    if episode < 0 or task_visit != visit:
        raise ValueError("Run boundary task ID visit disagrees with public settlement metadata")
    return episode, task_visit


def _validate_run_continuation(
    item: dict[str, object], checkpoint_rows: list[object],
) -> None:
    evidence = item.get("run_continuation")
    if not isinstance(evidence, dict):
        raise ValueError("Run evidence lacks a victory and same-Run continuation trace")
    _diagnostic_policy_metadata(evidence.get("policy_metadata"))
    settlement = evidence.get("settlement")
    continuation = evidence.get("continuation")
    if not isinstance(settlement, dict) or not isinstance(continuation, dict):
        raise ValueError("Run evidence lacks settlement or continuation boundary details")

    sequence = _nonnegative_int(settlement, "sequence")
    if _nonnegative_int(continuation, "sequence") != sequence + 1:
        raise ValueError("Run continuation is not the immediately following delivered decision")
    stream_id = settlement.get("stream_id")
    if not isinstance(stream_id, str) or continuation.get("stream_id") != stream_id:
        raise ValueError("Run continuation changed native stream")
    if not isinstance(settlement.get("decision_id"), str) or not settlement["decision_id"]:
        raise ValueError("Run settlement lacks a unique decision ID")
    if (not isinstance(continuation.get("decision_id"), str)
            or not continuation["decision_id"]
            or continuation["decision_id"] == settlement["decision_id"]):
        raise ValueError("Run continuation decision ID was reused or missing")

    visit = _nonnegative_int(settlement, "visit")
    episode, _ = _run_task_identity(settlement.get("task_id"), stream_id=stream_id, visit=visit)
    continuation_visit = continuation.get("visit")
    if type(continuation_visit) is not int or continuation_visit <= visit:
        raise ValueError("Run continuation task must use a later episode-qualified visit")
    continuation_episode, _ = _run_task_identity(
        continuation.get("task_id"), stream_id=stream_id, visit=continuation_visit,
    )
    if continuation_episode != episode or continuation.get("task_id") == settlement.get("task_id"):
        raise ValueError("Run continuation crossed a Run episode or reused the settled task ID")

    if (settlement.get("outcome") != "victory"
            or settlement.get("task_terminal") is not True
            or settlement.get("native_terminated") is not False
            or settlement.get("native_truncated") is not False
            or settlement.get("actor_training_mask") is not True
            or type(settlement.get("log_probability")) not in (int, float)
            or settlement.get("log_probability") != 0.0
            or type(settlement.get("value")) not in (int, float)
            or settlement.get("value") != 0.0
            or settlement.get("behavior_policy_version") != PUBLIC_DIAGNOSTIC_VERSION):
        raise ValueError("Run gate requires a nonterminal victory task settlement")
    encounter = settlement.get("encounter")
    before_hp = settlement.get("hp_before_healing")
    after_hp = settlement.get("hp_after_cleanup")
    if (not isinstance(encounter, str) or not encounter
            or type(before_hp) is not int or before_hp < 0
            or type(after_hp) is not int or after_hp < 0):
        raise ValueError("Run victory checkpoint lacks immutable public exit metadata")
    source_hash = settlement.get("source_frame_sha256")
    successor_hash = settlement.get("successor_frame_sha256")
    checkpoint_hash = settlement.get("immediate_checkpoint_sha256")
    if (not all(isinstance(value, str) and _SHA256.fullmatch(value)
                for value in (source_hash, successor_hash, checkpoint_hash))
            or successor_hash != checkpoint_hash):
        raise ValueError("Run victory settlement does not bind its immediate checkpoint frame")
    matches = [row for row in checkpoint_rows if isinstance(row, dict)
               and row.get("stream_id") == stream_id
               and row.get("task_id") == settlement.get("task_id")
               and row.get("visit") == visit
               and row.get("outcome") == "victory"
               and row.get("encounter") == encounter
               and row.get("hp_before_healing") == before_hp
               and row.get("hp_after_cleanup") == after_hp
               and row.get("public_frame_sha256") == checkpoint_hash]
    if len(matches) != 1:
        raise ValueError("Run victory boundary does not match one immediate resource checkpoint")

    before = _run_counters(settlement.get("counters_before"), "Run settlement counters_before")
    after = _run_counters(settlement.get("counters_after"), "Run settlement counters_after")
    if (_nonnegative_int(settlement, "reset_operations_before") != before["reset_operations"]
            or _nonnegative_int(settlement, "reset_operations_after") != after["reset_operations"]):
        raise ValueError("Run settlement explicit reset counts disagree with its counters")
    delta = _counter_delta(before, after, context="Run settlement")
    _require_action_delta(delta, context="Run settlement")
    if delta["tasks_completed"] != 1 or delta["task_terminals"] != 1:
        raise ValueError("Run victory settlement counters do not record exactly one task boundary")
    for field in ("reset_operations", "reset_slots", "delivered_reset_slots", "next_seed_ordinal"):
        if before[field] != 1 or after[field] != 1:
            raise ValueError("Run victory evidence must retain its one initial reset and seed allocation")

    if (continuation.get("source_frame_sha256") != checkpoint_hash
            or not isinstance(continuation.get("successor_frame_sha256"), str)
            or _SHA256.fullmatch(continuation["successor_frame_sha256"]) is None
            or type(continuation.get("native_terminated")) is not bool
            or type(continuation.get("native_truncated")) is not bool):
        raise ValueError("Run continuation is not a delivered action from the settled successor frame")
    if continuation.get("continuation_kind") == "fixed_noncombat":
        if (continuation.get("actor_training_mask") is not False
                or continuation.get("log_probability") is not None
                or continuation.get("value") is not None
                or continuation.get("behavior_policy_version") is not None):
            raise ValueError("fixed Run continuation must not claim actor statistics")
    elif continuation.get("continuation_kind") == "next_combat":
        if (continuation.get("actor_training_mask") is not True
                or type(continuation.get("log_probability")) not in (int, float)
                or continuation.get("log_probability") != 0.0
                or type(continuation.get("value")) not in (int, float)
                or continuation.get("value") != 0.0
                or continuation.get("behavior_policy_version") != PUBLIC_DIAGNOSTIC_VERSION):
            raise ValueError("next-combat Run continuation lacks truthful diagnostic policy statistics")
    else:
        raise ValueError("Run continuation must be a fixed noncombat or next-combat action")
    continuation_before = _run_counters(
        continuation.get("counters_before"), "Run continuation counters_before",
    )
    continuation_after = _run_counters(
        continuation.get("counters_after"), "Run continuation counters_after",
    )
    if (_nonnegative_int(continuation, "reset_operations_before")
            != continuation_before["reset_operations"]
            or _nonnegative_int(continuation, "reset_operations_after")
            != continuation_after["reset_operations"]):
        raise ValueError("Run continuation explicit reset counts disagree with its counters")
    if continuation_before != after:
        raise ValueError("Run continuation counters show an intervening action or reset")
    _require_action_delta(
        _counter_delta(continuation_before, continuation_after, context="Run continuation"),
        context="Run continuation",
    )
    if any(item.get(name) != continuation_after[name] for name in _RUN_COUNTER_FIELDS):
        raise ValueError("Run record totals do not end at the reported continuation boundary")


def _validate_manual_reset_boundary(item: dict[str, object]) -> None:
    boundary = item.get("manual_reset_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("standalone evidence lacks an actual manual reset boundary")
    reset_before = _nonnegative_int(boundary, "reset_operations_before")
    reset_after = _nonnegative_int(boundary, "reset_operations_after")
    tasks_before = _nonnegative_int(boundary, "tasks_completed_before")
    slots_before = _nonnegative_int(boundary, "reset_slots_before")
    slots_after = _nonnegative_int(boundary, "reset_slots_after")
    delivered_before = _nonnegative_int(boundary, "delivered_reset_slots_before")
    delivered_after = _nonnegative_int(boundary, "delivered_reset_slots_after")
    ordinal_before = _nonnegative_int(boundary, "next_seed_ordinal_before")
    ordinal_after = _nonnegative_int(boundary, "next_seed_ordinal_after")
    ordinals = boundary.get("seed_ordinals")
    expected_ordinals = list(range(ordinal_before, ordinal_after))
    if (reset_before < 1 or reset_after <= reset_before or tasks_before < 1
            or slots_after <= slots_before or delivered_after - delivered_before != slots_after - slots_before
            or ordinal_after <= ordinal_before or ordinals != expected_ordinals
            or len(ordinals) != slots_after - slots_before
            or delivered_after != item.get("delivered_reset_slots")
            or slots_after != item.get("reset_slots")
            or ordinal_after != item.get("next_seed_ordinal")
            or reset_after != item.get("reset_operations")
            or item.get("tasks_completed", 0) < tasks_before):
        raise ValueError("standalone manual reset boundary does not bind completed work to a later delivered reset")
    seeds = item.get("seed_allocations")
    observed = [seed.get("ordinal") for seed in seeds if isinstance(seed, dict)
                and ordinal_before <= seed.get("ordinal", -1) < ordinal_after]
    if observed != expected_ordinals:
        raise ValueError("standalone manual reset boundary seed ordinals do not match delivered allocations")


def validate_collection_evidence(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or document.get("schema") != "spire-learn-neural-collection-evidence-v1":
        raise ValueError("collection evidence has an unsupported schema")
    records = document.get("tests")
    if not isinstance(records, list) or len(records) != 2 or any(not isinstance(item, dict) for item in records):
        raise ValueError("collection evidence must contain exactly two test records")
    by_id = {item.get("test_id"): item for item in records}
    if len(by_id) != 2 or set(by_id) != set(EXPECTED_TEST_IDS):
        raise ValueError("collection evidence test IDs do not match the required full unittest IDs")

    total_attempted = 0
    total_forwarded = 0
    result_rows = []
    for test_id in EXPECTED_TEST_IDS:
        item = by_id[test_id]
        mode = "run" if ".test_collect_run_" in test_id else "standalone"
        if item.get("mode") != mode:
            raise ValueError(f"{mode} test evidence has the wrong mode")
        if item.get("environment_revision") != EXPECTED_GYM_REVISION:
            raise ValueError("collection evidence uses an unaccepted Gym revision")
        if item.get("common_view_fingerprint") != EXPECTED_COMMON_FINGERPRINT:
            raise ValueError("collection evidence uses an unaccepted common view")
        if mode == "run" and (
            item.get("run_profile_id") != "m10-ironclad-policy"
            or item.get("run_observation_fingerprint") != EXPECTED_RUN_FINGERPRINT
        ):
            raise ValueError("Run evidence does not identify the accepted v3 policy profile")
        if mode == "standalone" and item.get("run_profile_id") is not None:
            raise ValueError("standalone evidence must not claim a Run profile")

        if item.get("shared_budget_id") != SHARED_COHORT_BUDGET_ID:
            raise ValueError("both native modes must identify the one shared cohort Budget")
        if item.get("wall_budget_limit_seconds") != SHARED_COHORT_BUDGET_SECONDS:
            raise ValueError("native collection must use the accepted shared 120-second soft Budget")
        elapsed = _finite_nonnegative(item.get("elapsed_seconds"), "elapsed_seconds")
        overshoot = _finite_nonnegative(item.get("wall_overshoot_seconds"), "wall_overshoot_seconds")
        if overshoot > elapsed or type(item.get("elapsed_seconds_is_fallback")) is not bool:
            raise ValueError("native collection timing or fallback qualification is malformed")

        attempted = _nonnegative_int(item, "attempted_environment_slots")
        forwarded = _nonnegative_int(item, "forwarded_environment_slots")
        known = _nonnegative_int(item, "known_accepted_environment_slots")
        delivered = _nonnegative_int(item, "delivered_environment_slots")
        uncertain = _nonnegative_int(item, "uncertain_environment_slots")
        if not (delivered <= known <= forwarded <= attempted):
            raise ValueError("environment slot acceptance counters are inconsistent")
        if uncertain != 0 or attempted > MAX_MODE_SLOT_ACTIONS:
            raise ValueError("collection contains uncertain slots or exceeded its per-mode native slot-action bound")
        total_attempted += attempted
        total_forwarded += forwarded

        reset_operations = _nonnegative_int(item, "reset_operations")
        tasks_completed = _nonnegative_int(item, "tasks_completed")
        faults = _nonnegative_int(item, "faults")
        updates = _nonnegative_int(item, "optimizer_updates")
        if faults != 0 or updates != 0 or tasks_completed < 1:
            raise ValueError("collection must complete a natural task with no faults or optimizer updates")
        if mode == "standalone" and reset_operations < 2:
            raise ValueError("standalone evidence lacks an actual manual reset")
        if mode == "run" and reset_operations != 1:
            raise ValueError("Run evidence must retain one initial reset and no Run restart")
        attempted_reset = _nonnegative_int(item, "attempted_reset_slots")
        forwarded_reset = _nonnegative_int(item, "forwarded_reset_slots")
        known_reset = _nonnegative_int(item, "known_accepted_reset_slots")
        delivered_reset = _nonnegative_int(item, "delivered_reset_slots")
        uncertain_reset = _nonnegative_int(item, "uncertain_reset_slots")
        reset_slots = _nonnegative_int(item, "reset_slots")
        if not (delivered_reset <= known_reset <= forwarded_reset <= attempted_reset):
            raise ValueError("reset slot acceptance counters are inconsistent")
        if uncertain_reset != 0 or delivered_reset != reset_slots or reset_operations > reset_slots:
            raise ValueError("collection reset evidence is uncertain or undelivered")
        checkpoint_rows = item.get("resource_checkpoints")
        if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) < tasks_completed:
            raise ValueError("collection evidence is missing settlement resource checkpoints")
        for checkpoint in checkpoint_rows:
            if (not isinstance(checkpoint, dict)
                    or checkpoint.get("outcome") not in {"victory", "death"}
                    or type(checkpoint.get("visit")) is not int
                    or checkpoint["visit"] < 0
                    or not isinstance(checkpoint.get("stream_id"), str)
                    or not isinstance(checkpoint.get("task_id"), str)
                    or not isinstance(checkpoint.get("encounter"), str)
                    or not checkpoint["encounter"]
                    or type(checkpoint.get("hp_before_healing")) is not int
                    or checkpoint["hp_before_healing"] < 0
                    or type(checkpoint.get("hp_after_cleanup")) is not int
                    or checkpoint["hp_after_cleanup"] < 0
                    or not _SHA256.fullmatch(str(checkpoint.get("public_frame_sha256", "")))):
                raise ValueError("resource checkpoint evidence is malformed")
        seeds = item.get("seed_allocations")
        if not isinstance(seeds, list) or len(seeds) != reset_slots:
            raise ValueError("seed allocation evidence does not match the reset slots")
        next_seed_ordinal = _nonnegative_int(item, "next_seed_ordinal")
        if next_seed_ordinal != len(seeds):
            raise ValueError("next seed ordinal does not match delivered reset allocations")
        ordinals = []
        for seed in seeds:
            if (not isinstance(seed, dict) or seed.get("delivery") != "delivered"
                    or type(seed.get("actual_seed")) is not int
                    or type(seed.get("authored_reference_seed")) is not int
                    or type(seed.get("ordinal")) is not int
                    or not isinstance(seed.get("case_id"), str)
                    or not isinstance(seed.get("allocation_policy_id"), str)):
                raise ValueError("seed allocation evidence is malformed")
            ordinals.append(seed["ordinal"])
        if ordinals != list(range(next_seed_ordinal)):
            raise ValueError("seed allocation ordinals are not monotonic and contiguous")
        if mode == "run":
            _validate_run_continuation(item, checkpoint_rows)
        else:
            _validate_manual_reset_boundary(item)
        result_rows.append({"test_id": test_id, "mode": mode, "attempted_slots": attempted,
                            "forwarded_slots": forwarded, "tasks_completed": tasks_completed,
                            "reset_operations": reset_operations})

    if total_attempted > MAX_TOTAL_SLOT_ACTIONS:
        raise ValueError("two-mode collection exceeded 4,096 combined slot actions")
    run_elapsed = _finite_nonnegative(by_id[EXPECTED_TEST_IDS[0]].get("elapsed_seconds"),
                                      "Run elapsed_seconds")
    standalone_elapsed = _finite_nonnegative(by_id[EXPECTED_TEST_IDS[1]].get("elapsed_seconds"),
                                             "standalone elapsed_seconds")
    if standalone_elapsed < run_elapsed:
        raise ValueError("shared cohort Budget elapsed time moved backwards between native tests")
    return {
        "passed": True,
        "tests": result_rows,
        "total_attempted_environment_slots": total_attempted,
        "total_forwarded_environment_slots": total_forwarded,
        "optimizer_updates": 0,
    }


def classify_collection_result(
    stdout: str,
    stderr: str,
    evidence: object,
    *,
    exit_code: int,
) -> dict[str, object]:
    test_result = parse_text_test_result(stdout, stderr, exit_code=exit_code)
    result: dict[str, object] = {
        "schema": "spire-learn-neural-collection-result-v1",
        "tests": test_result,
        "collection_evidence": None,
        "classification": "incomplete_or_failed",
    }
    if not test_result["passed"]:
        result["reason"] = "required unittest IDs/count/status or process exit did not pass"
        return result
    try:
        evidence_result = validate_collection_evidence(evidence)
    except (TypeError, ValueError) as error:
        result["reason"] = str(error)
        return result
    result["collection_evidence"] = evidence_result
    result["classification"] = "passed"
    result["reason"] = "both native mode tests and their bounded outcome evidence passed"
    return result


def verify_container_inspect(
    raw: str,
    *,
    root: Path,
    venv: Path,
    run_root: Path,
    image_id: str,
    expected_command: str,
) -> dict[str, object]:
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Docker inspect must return one JSON object")
    expected_base = {
        "image_id": image_id,
        "user": "1000:1000",
        "workdir": "/learner",
        "entrypoint": ["/bin/sh"],
        "readonly_rootfs": True,
        "privileged": False,
        "cap_drop": ["ALL"],
        "nano_cpus": 4_000_000_000,
        "memory": 8 * 1024**3,
        "pids_limit": 512,
        "network": "none",
    }
    for key, expected in expected_base.items():
        if document.get(key) != expected:
            raise ValueError(f"container inspect {key} does not match the reviewed CPU boundary")
    if not isinstance(expected_command, str) or not expected_command:
        raise ValueError("reviewed container command must be non-empty text")
    if document.get("cmd") != ["-ec", expected_command]:
        raise ValueError("container inspect Cmd does not match the exact reviewed command and argv")
    if not any(item in {"no-new-privileges", "no-new-privileges:true"}
               for item in document.get("security_opt", [])):
        raise ValueError("container lacks no-new-privileges")
    tmpfs = document.get("tmpfs")
    if not isinstance(tmpfs, dict) or set(tmpfs) != {"/tmp"}:
        raise ValueError("container /tmp mount does not match the reviewed noexec tmpfs")
    tmpfs_options = set(str(tmpfs["/tmp"]).split(","))
    tmpfs_size = next((item.split("=", 1)[1] for item in tmpfs_options
                       if item.startswith("size=")), None)
    if (not {"rw", "noexec", "nosuid"} <= tmpfs_options
            or tmpfs_size not in {"64m", str(64 * 1024 * 1024)}):
        raise ValueError("container /tmp mount does not match the reviewed noexec tmpfs")

    expected_mounts = {
        (str(root / "src"), "/learner/src", False),
        (str(root / "tests"), "/learner/tests", False),
        (str(root / "configs"), "/learner/configs", False),
        (str(root / "scripts/neural/task5_collection_control.py"),
         "/learner/scripts/neural/task5_collection_control.py", False),
        (str(venv), "/work/venv-cpu", False),
        (str(venv.parent / "requirements-cpu.lock"), "/work/requirements-cpu.lock", False),
        (str(run_root), "/run", True),
        (str(run_root), "/learner/artifacts/development/neural-combat", True),
    }
    mounts = document.get("mounts")
    if not isinstance(mounts, list):
        raise ValueError("container inspect mounts must be a list")
    observed = set()
    for mount in mounts:
        if not isinstance(mount, dict):
            raise ValueError("container mount record must be an object")
        observed.add((mount.get("Source", mount.get("source")),
                      mount.get("Destination", mount.get("destination")),
                      mount.get("RW", mount.get("rw"))))
    if observed != expected_mounts or len(observed) != len(mounts):
        raise ValueError("container mounts differ from the reviewed read-only inputs/fresh-child outputs")
    helper_mounts = [
        mount for mount in mounts
        if mount.get("Destination", mount.get("destination"))
        == "/learner/scripts/neural/task5_collection_control.py"
    ]
    if (len(helper_mounts) != 1
            or helper_mounts[0].get("Type", helper_mounts[0].get("type")) != "bind"):
        raise ValueError("container mounts must include exactly one bind for the control helper file")
    environment = document.get("env")
    if not isinstance(environment, list) or any(not isinstance(item, str) for item in environment):
        raise ValueError("container inspect environment must be a string list")
    environment_map = dict(item.split("=", 1) for item in environment if "=" in item)
    expected_environment = {
        "PYTHONPATH": "/learner/src",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HOME": "/run/home",
        "TMPDIR": "/run/tmp",
    }
    if any(environment_map.get(name) != value for name, value in expected_environment.items()):
        raise ValueError("container environment is missing an exact reviewed offline/path setting")
    if any(name.startswith(("AWS_", "SSH_", "GITHUB_", "GH_"))
           or "TOKEN" in name.upper() or "PASSWORD" in name.upper()
           for name in environment_map):
        raise ValueError("container environment contains a credential-like variable")
    return {
        "passed": True,
        "mount_count": len(observed),
        "image_id": image_id,
        "entrypoint": ["/bin/sh"],
        "cmd_sha256": hashlib.sha256(expected_command.encode("utf-8")).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--root", type=Path, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--before", type=Path, required=True)
    compare_parser.add_argument("--after", type=Path, required=True)
    pre_dispatch_parser = subparsers.add_parser("pre-dispatch")
    pre_dispatch_parser.add_argument("--source-before", type=Path, required=True)
    pre_dispatch_parser.add_argument("--output", type=Path, required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--root", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("--stdout", type=Path, required=True)
    classify_parser.add_argument("--stderr", type=Path, required=True)
    classify_parser.add_argument("--evidence", type=Path, required=True)
    classify_parser.add_argument("--exit-code", type=int, required=True)
    classify_parser.add_argument("--output", type=Path, required=True)
    inspect_parser = subparsers.add_parser("verify-inspect")
    inspect_parser.add_argument("--input", type=Path, required=True)
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--venv", type=Path, required=True)
    inspect_parser.add_argument("--run-root", type=Path, required=True)
    inspect_parser.add_argument("--image-id", required=True)
    inspect_parser.add_argument("--expected-command-file", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            _write_json(args.output, source_snapshot(args.root))
        elif args.command == "compare":
            compare_snapshots(json.loads(args.before.read_text(encoding="utf-8")),
                              json.loads(args.after.read_text(encoding="utf-8")))
            print("input snapshot unchanged")
        elif args.command == "pre-dispatch":
            snapshot = json.loads(args.source_before.read_text(encoding="utf-8"))
            _write_json(args.output, pre_dispatch_manifest(snapshot))
        elif args.command == "inventory":
            _write_json(args.output, evidence_inventory(args.root))
        elif args.command == "classify":
            result = classify_collection_result(
                args.stdout.read_text(encoding="utf-8"),
                args.stderr.read_text(encoding="utf-8"),
                json.loads(args.evidence.read_text(encoding="utf-8")),
                exit_code=args.exit_code,
            )
            _write_json(args.output, result)
            print(result["classification"])
            if not result["tests"]["passed"] or result["classification"] != "passed":
                return 2
        elif args.command == "verify-inspect":
            result = verify_container_inspect(
                args.input.read_text(encoding="utf-8"),
                root=args.root.resolve(),
                venv=args.venv.resolve(),
                run_root=args.run_root.resolve(),
                image_id=args.image_id,
                expected_command=args.expected_command_file.read_text(encoding="utf-8"),
            )
            _write_json(args.output, result)
            print("container boundary accepted")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Task 5 collection control failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
