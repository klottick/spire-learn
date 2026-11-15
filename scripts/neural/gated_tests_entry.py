#!/usr/bin/env python3
"""Run the explicit Gym 8726d467 gated tensor-test suite in one process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def _detail(item: object) -> dict[str, str]:
    test, message = item
    identifier = test.id() if callable(getattr(test, "id", None)) else str(test)
    return {"test": identifier, "message": str(message)}


def _result_document(run_id: str, reservation_id: str, expected: int,
                     result: unittest.TestResult, *, runner_error: str | None = None) -> dict[str, object]:
    skipped = [_detail(item) for item in result.skipped]
    failures = [_detail(item) for item in result.failures]
    errors = [_detail(item) for item in result.errors]
    ran = result.testsRun
    passed = max(0, ran - len(skipped) - len(failures) - len(errors))
    green = (
        expected > 0 and ran == expected and not skipped and not failures and not errors
        and runner_error is None
    )
    return {
        "schema": "spire-learn-gated-tensor-tests-v1",
        "task": TASK,
        "run_id": run_id,
        "reservation_id": reservation_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "modules": list(MODULES),
        "status": "green" if green else "red",
        "tests": {
            "expected": expected,
            "run": ran,
            "passed": passed,
            "skipped": skipped,
            "failures": failures,
            "errors": errors,
        },
        "runner_error": runner_error,
    }


def _write_result_exclusive(path: Path, document: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path,
                        help="existing owned writable directory for evidence files")
    args = parser.parse_args(argv)
    if _RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id must be 1..51 safe filename characters and start with a letter or digit")
    evidence_dir = args.evidence
    result_path = evidence_dir / "result.json"
    evidence_path = evidence_dir / "tensor-sampling.json"
    reservation_id = f"{TASK}-{args.run_id}"
    if not evidence_dir.is_dir():
        print("evidence directory must already exist", file=sys.stderr)
        return 2
    if result_path.exists():
        print("result.json already exists; refusing to overwrite evidence", file=sys.stderr)
        return 2
    if not args.ledger.is_file():
        print("sample ledger must already exist; refusing to initialize or reset it", file=sys.stderr)
        return 2

    from tests.neural._sample_guard import SampleBudgetError, reserve_rows

    try:
        reserve_rows(task=TASK, reservation_id=reservation_id, rows=MAX_ROWS, path=args.ledger)
    except (OSError, ValueError, SampleBudgetError) as error:
        print(f"sample ledger refused invocation: {error}", file=sys.stderr)
        return 2

    os.environ["SPIRE_LEARN_RUN_NEURAL_TENSOR"] = "1"
    os.environ["SPIRE_LEARN_TENSOR_TASK"] = TASK
    os.environ["SPIRE_LEARN_TENSOR_RESERVATION_ID"] = reservation_id
    os.environ["SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS"] = str(MAX_ROWS)
    os.environ["SPIRE_LEARN_TENSOR_EVIDENCE_PATH"] = str(evidence_path)

    suite = unittest.TestSuite()
    runner_error = None
    result = unittest.TestResult()
    try:
        for module in MODULES:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
        expected = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    except Exception as error:
        expected = suite.countTestCases()
        runner_error = f"{type(error).__name__}: {error}"
        result.errors.append(("gated tensor test runner", runner_error))
    document = _result_document(run_id=args.run_id, reservation_id=reservation_id,
                                expected=expected, result=result, runner_error=runner_error)
    try:
        _write_result_exclusive(result_path, document)
    except (OSError, ValueError) as error:
        print(f"could not write exclusive result.json: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
