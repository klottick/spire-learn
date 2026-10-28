"""Lazy command-line entry point for bounded neural work."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from spire_learn.neural.config import load_neural_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m spire_learn.neural")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run bounded CPU training")
    train.add_argument("--config", required=True, metavar="PATH")
    train.add_argument("--validate-only", action="store_true")
    training_source = train.add_mutually_exclusive_group()
    training_source.add_argument("--resume", metavar="CHECKPOINT")
    training_source.add_argument("--warm-start", metavar="CHECKPOINT")

    evaluate = commands.add_parser("evaluate", help="evaluate one frozen policy or reference")
    evaluate.add_argument("--config", required=True, metavar="PATH")
    evaluate.add_argument("--validate-only", action="store_true")
    evaluate.add_argument(
        "--policy", choices=("checkpoint", "legal-random", "end-turn-last-v1"),
        default="checkpoint",
    )
    evaluate.add_argument("--checkpoint", metavar="PATH")
    evaluate.add_argument(
        "--action-selection", choices=("deterministic", "seeded-sampling"),
    )
    evaluate.add_argument("--allow-final", action="store_true")

    calibrate = commands.add_parser(
        "calibrate", help="run the fixed Task 8 CUDA calibration cohort"
    )
    calibrate.add_argument("--config", required=True, metavar="PATH")
    calibrate.add_argument("--validate-only", action="store_true")
    return parser


def _validated_arguments(arguments: argparse.Namespace) -> tuple[NeuralConfig, str, Path | None, str, bool]:
    config = load_neural_config(Path(arguments.config))
    if arguments.command == "train":
        if config.operation != "train" or config.split != "train":
            raise ValueError("train requires a train-operation config with split=train")
        checkpoint = arguments.resume or arguments.warm_start
        purpose = "resume" if arguments.resume else "warm_start" if arguments.warm_start else "train"
        return config, purpose, Path(checkpoint) if checkpoint else None, "seeded-sampling", False

    if arguments.command == "evaluate":
        if config.operation != "evaluate" or config.split not in {"selection", "final"}:
            raise ValueError("evaluate requires an evaluate-operation selection or final config")
        if config.split == "final" and not arguments.allow_final:
            raise ValueError("final split requires --allow-final")
        policy_kind = arguments.policy
        checkpoint = Path(arguments.checkpoint) if arguments.checkpoint is not None else None
        if policy_kind == "checkpoint" and checkpoint is None:
            raise ValueError("--policy checkpoint requires --checkpoint")
        if policy_kind != "checkpoint" and checkpoint is not None:
            raise ValueError("--checkpoint is only valid with --policy checkpoint")
        default_selection = "seeded-sampling" if policy_kind == "legal-random" else "deterministic"
        action_selection = arguments.action_selection or default_selection
        if policy_kind == "legal-random" and action_selection != "seeded-sampling":
            raise ValueError("legal-random requires seeded-sampling action selection")
        if policy_kind == "end-turn-last-v1" and action_selection != "deterministic":
            raise ValueError("end-turn-last-v1 requires deterministic action selection")
        return config, "evaluate", checkpoint, policy_kind, arguments.allow_final

    if arguments.command == "calibrate":
        if (config.operation != "calibrate" or config.mode != "standalone"
                or config.split != "train" or config.partition != "train"):
            raise ValueError("calibrate requires a calibration-operation standalone train config")
        if config.device != "cuda:0" or config.calibration_gpu_uuid is None:
            raise ValueError("calibrate requires the qualified cuda:0 device and GPU UUID")
        return config, "calibrate", None, "calibration", False
    raise ValueError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit neural command, returning a CLI status code."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if type(error.code) is int else 2

    try:
        config, purpose, checkpoint, policy_kind, allow_final = _validated_arguments(arguments)
    except (OSError, TypeError, ValueError) as error:
        print(f"spire-learn neural: {error}", file=sys.stderr)
        return 2

    if arguments.validate_only:
        message = (
            "configuration valid; no neural runtime or artifacts created"
            if arguments.command != "calibrate"
            else "calibration configuration valid; CUDA runtime and calibration were not executed"
        )
        print(message)
        return 0

    if arguments.command == "calibrate":
        from spire_learn.neural.runner import calibrate

        try:
            report = calibrate(config)
        except Exception as error:
            print(f"spire-learn neural: calibration failed: {error}", file=sys.stderr)
            return 1
        try:
            print(json.dumps(dict(report.summary), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, OSError) as error:
            print(f"spire-learn neural: report output failed: {error}", file=sys.stderr)
            return 1
        return report.exit_code

    from spire_learn.neural.runner import create_runtime, evaluate_policy, train

    action_selection = "seeded-sampling"
    if arguments.command == "evaluate":
        action_selection = arguments.action_selection or (
            "seeded-sampling" if policy_kind == "legal-random" else "deterministic"
        )
    try:
        runtime = create_runtime(
            config,
            purpose=purpose,
            checkpoint=checkpoint,
            policy_kind=policy_kind if arguments.command == "evaluate" else "checkpoint",
            action_selection=action_selection,
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"spire-learn neural: runtime setup blocked: {error}", file=sys.stderr)
        return 2

    try:
        report = (
            evaluate_policy(config, runtime=runtime, allow_final=allow_final)
            if arguments.command == "evaluate"
            else train(config, runtime=runtime)
        )
    except Exception as error:
        print(f"spire-learn neural: operation failed: {error}", file=sys.stderr)
        return 1
    try:
        print(json.dumps(dict(report.summary), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, OSError) as error:
        print(f"spire-learn neural: report output failed: {error}", file=sys.stderr)
        return 1
    return report.exit_code


__all__ = ["main"]
