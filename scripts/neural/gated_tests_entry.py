#!/usr/bin/env python3
"""Run the gated tensor suite or the opt-in Standard native turn-cap checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import unittest
from typing import Sequence


TASK = "gym-8726d467"
MAX_ROWS = 64
MODULES = (
    "tests.neural.test_model",
    "tests.neural.test_distribution",
    "tests.neural.test_ppo",
    "tests.neural.test_checkpoint",
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,50}\Z")
_EVIDENCE_FILE = "tensor-sampling.json"
_METRICS_FILE = "model-metrics.json"


def _detail(item: object) -> dict[str, str]:
    if isinstance(item, tuple) and len(item) == 2:
        test, message = item
    else:
        test, message = item, ""
    identifier = test.id() if callable(getattr(test, "id", None)) else str(test)
    return {"test": identifier, "message": str(message)}


def _test_cases(suite: object) -> list[object]:
    if isinstance(suite, unittest.TestSuite):
        result = []
        for child in suite:
            result.extend(_test_cases(child))
        return result
    if callable(getattr(suite, "id", None)) and callable(getattr(suite, "run", None)):
        return [suite]
    raise TypeError(f"test loader returned unsupported object {type(suite).__name__}")


def _sampling_summary(path: Path, reservation_id: str) -> dict[str, object]:
    summary: dict[str, object] = {"clean": False, "path": str(path)}
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        return summary
    if not isinstance(evidence, dict):
        summary["error"] = "sampling evidence must be a JSON object"
        return summary
    summary.update({
        "status": evidence.get("status"),
        "attempted_rows": evidence.get("attempted_rows"),
        "completed_rows": evidence.get("completed_rows"),
        "rejected_rows": evidence.get("rejected_rows"),
    })
    counts_valid = all(
        type(evidence.get(name)) is int and evidence[name] >= 0
        for name in ("attempted_rows", "completed_rows", "rejected_rows")
    )
    clean = (
        isinstance(evidence, dict)
        and evidence.get("schema") == "spire-learn-tensor-sampling-v1"
        and evidence.get("reservation_id") == reservation_id
        and evidence.get("reserved_rows") == MAX_ROWS
        and counts_valid
        and evidence["attempted_rows"] <= MAX_ROWS
        and evidence["completed_rows"] <= evidence["attempted_rows"]
        and evidence["rejected_rows"] == 0
        and isinstance(evidence.get("status"), str)
        and evidence.get("status") in {"ready", "sample_completed"}
        and (evidence.get("status") != "ready" or evidence["attempted_rows"] == 0)
        and (evidence.get("status") != "sample_completed"
             or evidence["completed_rows"] == evidence["attempted_rows"])
    )
    summary["clean"] = clean
    if not clean:
        summary["error"] = "sampling evidence is missing, inconsistent, rejected, or incomplete"
    return summary


def _write_exclusive(path: Path, document: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _completion_for(path: Path, reservation_id: str) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        for line in handle:
            event = json.loads(line)
            if (event.get("event") == "completion"
                    and event.get("reservation_id") == reservation_id):
                return event
    return None


def _result_document(
    *, run_id: str, reservation_id: str, modules: list[dict[str, object]],
    result: unittest.TestResult, sampling: dict[str, object],
    reservation_completion: dict[str, object], runner_error: str | None,
    was_successful: bool,
) -> dict[str, object]:
    skipped = [_detail(item) for item in result.skipped]
    failures = [_detail(item) for item in result.failures]
    errors = [_detail(item) for item in result.errors]
    expected_failures = [_detail(item) for item in getattr(result, "expectedFailures", ())]
    unexpected_successes = [
        _detail(item) for item in getattr(result, "unexpectedSuccesses", ())
    ]
    expected = sum(int(module["collected"]) for module in modules)
    ran = int(result.testsRun)
    collected_nonempty = len(modules) == len(MODULES) and all(
        int(module["collected"]) > 0 and module.get("load_error") is None
        for module in modules
    )
    green = (
        expected > 0 and collected_nonempty and ran == expected
        and was_successful and not skipped and not failures and not errors
        and not unexpected_successes and runner_error is None
        and sampling.get("clean") is True
        and reservation_completion.get("completed") is True
        and reservation_completion.get("ledger_status") == "succeeded"
    )
    passed = max(0, ran - len(skipped) - len(failures) - len(errors)
                 - len(expected_failures) - len(unexpected_successes))
    return {
        "schema": "spire-learn-gated-tensor-tests-v1",
        "task": TASK,
        "run_id": run_id,
        "reservation_id": reservation_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "modules": modules,
        "status": "green" if green else "red",
        "tests": {"expected": expected, "run": ran, "passed": passed},
        "unittest": {
            "was_successful": was_successful,
            "skipped": skipped,
            "failures": failures,
            "errors": errors,
            "expected_failures": expected_failures,
            "unexpected_successes": unexpected_successes,
        },
        "sampling": sampling,
        "reservation_completion": reservation_completion,
        "runner_error": runner_error,
    }


def _claim_evidence_directory(evidence_dir: Path, run_id: str) -> None:
    if not evidence_dir.parent.is_dir():
        raise FileNotFoundError("evidence parent directory must already exist")
    evidence_dir.mkdir(mode=0o700)
    claim = {
        "schema": "spire-learn-gated-tensor-claim-v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_exclusive(evidence_dir / "claim.json", claim)
    for filename in (_EVIDENCE_FILE, _METRICS_FILE, "result.json"):
        if (evidence_dir / filename).exists():
            raise FileExistsError(f"refusing pre-existing evidence file {filename}")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--standard-turncap"]:
        if os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") != "1":
            print("--standard-turncap requires SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1", file=sys.stderr)
            return 2
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.integration.test_standard_turncap")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.testsRun and result.wasSuccessful() and not result.skipped else 1

    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Use --standard-turncap alone for the Standard native checks.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path,
                        help="new evidence directory under the owned writable output")
    args = parser.parse_args(argv)
    if _RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id must be 1..51 safe filename characters and start with a letter or digit")
    if not args.ledger.is_absolute():
        print("--ledger must be an absolute path", file=sys.stderr)
        return 2
    if not args.evidence.is_absolute():
        print("--evidence must be an absolute path", file=sys.stderr)
        return 2
    evidence_dir = args.evidence
    evidence_path = evidence_dir / _EVIDENCE_FILE
    metrics_path = evidence_dir / _METRICS_FILE
    reservation_id = f"{TASK}-{args.run_id}"
    if not args.ledger.is_file():
        print("sample ledger must already exist; refusing to initialize or reset it", file=sys.stderr)
        return 2

    try:
        _claim_evidence_directory(evidence_dir, args.run_id)
    except OSError as error:
        print(f"could not exclusively claim evidence directory: {error}", file=sys.stderr)
        return 2

    from tests.neural._sample_guard import (
        SampleBudgetError, _finish_from_evidence, reserve_rows,
    )

    try:
        reserve_rows(task=TASK, reservation_id=reservation_id,
                     rows=MAX_ROWS, path=args.ledger)
    except (OSError, ValueError, SampleBudgetError) as error:
        print(f"sample ledger refused invocation: {error}", file=sys.stderr)
        return 2

    # These are all environment variables read by the four gated modules and guard.
    # Override inherited values so neither evidence path can escape the owned directory.
    os.environ.update({
        "SPIRE_LEARN_RUN_NEURAL_TENSOR": "1",
        "SPIRE_LEARN_TENSOR_TASK": TASK,
        "SPIRE_LEARN_TENSOR_RESERVATION_ID": reservation_id,
        "SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS": str(MAX_ROWS),
        "SPIRE_LEARN_TENSOR_EVIDENCE_PATH": str(evidence_path),
        "SPIRE_LEARN_TENSOR_METRICS_PATH": str(metrics_path),
    })

    modules: list[dict[str, object]] = []
    tests: list[object] = []
    for module_name in MODULES:
        module_record: dict[str, object] = {
            "module": module_name,
            "collected": 0,
            "test_ids": [],
            "load_error": None,
        }
        try:
            module_suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
            module_tests = _test_cases(module_suite)
            test_ids = [test.id() for test in module_tests]
            module_record["collected"] = len(test_ids)
            module_record["test_ids"] = test_ids
            tests.extend(module_tests)
            if not module_tests:
                module_record["load_error"] = "module collected no tests"
        except Exception as error:
            module_record["load_error"] = f"{type(error).__name__}: {error}"
        modules.append(module_record)

    result = unittest.TestResult()
    runner_error = None
    runner_started = False
    interrupted: BaseException | None = None
    if all(int(module["collected"]) > 0 and module["load_error"] is None for module in modules):
        suite = unittest.TestSuite(tests)
        try:
            runner_started = True
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        except BaseException as error:
            # Process control still settles the reservation and publishes a red result, then propagates.
            if not isinstance(error, Exception):
                interrupted = error
            runner_error = f"{type(error).__name__}: {error}"
            result.errors.append(("gated tensor test runner", runner_error))

    try:
        was_successful = bool(result.wasSuccessful())
    except Exception as error:
        was_successful = False
        runner_error = runner_error or f"{type(error).__name__}: {error}"

    sampling = _sampling_summary(evidence_path, reservation_id)
    tests_passed = (
        sum(int(module["collected"]) for module in modules) > 0
        and all(int(module["collected"]) > 0 and module["load_error"] is None for module in modules)
        and result.testsRun == sum(int(module["collected"]) for module in modules)
        and was_successful and not result.skipped and not result.failures and not result.errors
        and not getattr(result, "unexpectedSuccesses", ()) and runner_error is None
    )
    requested_status = "succeeded" if tests_passed and sampling["clean"] else (
        "failed" if runner_started else "not_started"
    )
    completion: dict[str, object] = {"completed": False, "requested_status": requested_status}
    try:
        _finish_from_evidence(
            args.ledger, reservation_id, requested_status, evidence_path
        )
        event = _completion_for(args.ledger, reservation_id)
        if event is None:
            raise ValueError("ledger finish returned without a completion event")
        completion.update({"completed": True, "ledger_status": event["status"],
                           "attempted_rows": event["attempted_rows"],
                           "completed_rows": event["completed_rows"],
                           "rejected_rows": event["rejected_rows"]})
    except Exception as error:  # includes malformed ledger events (KeyError, TypeError)
        completion["error"] = f"{type(error).__name__}: {error}"

    document = _result_document(
        run_id=args.run_id, reservation_id=reservation_id, modules=modules,
        result=result, sampling=sampling, reservation_completion=completion,
        runner_error=runner_error, was_successful=was_successful,
    )
    try:
        _write_exclusive(evidence_dir / "result.json", document)
    except (OSError, ValueError) as error:
        print(f"could not write exclusive result.json: {error}", file=sys.stderr)
        if interrupted is not None:
            raise interrupted
        return 2
    if interrupted is not None:
        raise interrupted
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
