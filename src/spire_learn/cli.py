"""Dependency-light command-line entry points for evaluation and probes."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
import time
import uuid

from spire_learn.artifacts.writer import RunWriter
from spire_learn.evaluation.runner import EvaluationReport, evaluate
from spire_learn.experiments.cases import CaseManifest, load_cases, validate_splits
from spire_learn.experiments.config import RunConfig, load_config, to_probe_config, validate_expectations


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit command, returning an exit status instead of exiting."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if type(error.code) is int else 2

    try:
        config = load_config(Path(arguments.config), expected_mode=arguments.command)
        manifest = load_cases(config.cases)
        guards = tuple(load_cases(path) for path in config.split_guard_manifests)
        validate_splits((manifest, *guards))
        validate_expectations(config, manifest)
        if arguments.command == "evaluate" and manifest.split == "final" and not arguments.allow_final:
            raise ValueError("final split requires --allow-final")
        probe_config = None if arguments.command == "evaluate" else to_probe_config(config, manifest)
    except (OSError, ValueError, TypeError) as error:
        print(f"spire-learn: {error}", file=sys.stderr)
        return 2

    if arguments.validate_only:
        message = (
            "configuration valid; no evaluation or artifacts created"
            if arguments.command == "evaluate"
            else "configuration valid; no probe or artifacts created"
        )
        try:
            print(message)
        except Exception as error:
            print(f"spire-learn: output failed: {error}", file=sys.stderr)
            return 1
        return 0

    if arguments.command == "evaluate":
        try:
            report = evaluate(config, allow_final=arguments.allow_final)
            _print_evaluation_report(report)
            return report.exit_code
        except Exception as error:
            print(f"spire-learn: evaluation failed: {error}", file=sys.stderr)
            return 1

    return _run_probe(arguments.command, config, manifest, probe_config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spire-learn")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "probe-vector", "probe-torchrl"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, metavar="PATH")
        command.add_argument("--validate-only", action="store_true")
        if name == "evaluate":
            command.add_argument("--allow-final", action="store_true")
    return parser


def _print_evaluation_report(report: EvaluationReport) -> None:
    print(json.dumps(dict(report.summary), sort_keys=True, allow_nan=False))
    if report.run_dir is not None:
        print(f"run directory: {report.run_dir}")
    if report.fatal_error is not None:
        print(
            f"spire-learn: {report.fatal_error.kind} during {report.fatal_error.phase}: "
            f"{report.fatal_error.reason}",
            file=sys.stderr,
        )


def _run_probe(command: str, config: RunConfig, manifest: CaseManifest, probe_config) -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex
    writer = None
    status = 2
    primary_exception: BaseException | None = None
    try:
        try:
            writer = RunWriter.create(config.output_root, run_id)
            writer.write_start(_config_record(config), _manifest_record(manifest))
            if command == "probe-vector":
                from spire_learn.probes.vector import probe_report_to_json, run_vector_probe

                report = run_vector_probe(probe_config, writer, clock=time.monotonic)
                print(json.dumps(probe_report_to_json(report), sort_keys=True, allow_nan=False))
            else:
                from spire_learn.probes.torchrl import run_torchrl_probe
                from spire_learn.probes.vector import probe_report_to_json

                report = run_torchrl_probe(probe_config, writer, clock=time.monotonic)
                print(json.dumps(probe_report_to_json(report), sort_keys=True, allow_nan=False))
            status = _probe_exit_code(report)
        except ImportError as error:
            print(f"spire-learn: probe dependency/setup unavailable: {error}", file=sys.stderr)
            status = 2
        except Exception as error:
            print(f"spire-learn: probe failed: {error}", file=sys.stderr)
            status = 1
    except BaseException as error:
        primary_exception = error
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except BaseException as error:
                if primary_exception is not None:
                    _note_probe_close_failure(primary_exception, error)
                elif isinstance(error, Exception):
                    print(f"spire-learn: artifact close failed: {error}", file=sys.stderr)
                    status = 1
                else:
                    raise
    return status


def _note_probe_close_failure(primary: BaseException, close_failure: BaseException) -> None:
    try:
        detail = str(close_failure)
    except BaseException:
        detail = "exception message unavailable"
    try:
        primary.add_note(
            "probe artifact close also failed: "
            f"{type(close_failure).__name__}: {detail}"
        )
    except BaseException:
        pass


def _probe_exit_code(report) -> int:
    """Map a probe report to CLI status while distinguishing setup from failure."""
    verdict = report.verdict
    failures = tuple(report.failures)
    delivered = (
        report.batch_steps > 0
        or report.slot_decisions > 0
        or report.replacement_decisions > 0
    )

    if verdict in {"compatible", "adaptable"} and not failures:
        return 0
    if verdict == "blocked":
        if delivered or any(failure.kind != "setup_error" for failure in failures):
            return 1
        return 2
    return 1


def _config_record(config: RunConfig) -> dict[str, object]:
    probe = None
    if config.probe is not None:
        probe = {"num_envs": config.probe.num_envs, "workers": config.probe.workers}
    return {
        "mode": config.mode,
        "profile": config.profile,
        "agent": config.agent,
        "cases": str(config.cases),
        "output_root": str(config.output_root),
        "per_case_decisions": config.per_case_decisions,
        "max_batch_steps": config.max_batch_steps,
        "max_wall_seconds": config.max_wall_seconds,
        "traces": config.traces,
        "expected_partition": config.expected_partition,
        "expected_schema_id": config.expected_schema_id,
        "expected_fingerprint": config.expected_fingerprint,
        "gym_checkout": str(config.gym_checkout) if config.gym_checkout is not None else None,
        "split_guard_manifests": [str(path) for path in config.split_guard_manifests],
        "probe": probe,
    }


def _manifest_record(manifest: CaseManifest) -> dict[str, object]:
    return {
        "path": str(manifest.path),
        "split": manifest.split,
        "executions": manifest.executions,
    }
