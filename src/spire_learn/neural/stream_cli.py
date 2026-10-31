"""Dependency-free preparation CLI. This entry never runs training."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


def add_arguments(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest='curriculum_command', required=True)
    prepare = commands.add_parser('prepare', help='prepare reproducible files without runtime authority')
    prepare.add_argument('--recipe', required=True, type=Path)
    prepare.add_argument('--destination', required=True, type=Path,
                         help='absent destination under artifacts/development/neural-combat')
    prepare.add_argument('--arm', required=True, choices=('fixed', 'randomized', 'mixed'))
    prepare.add_argument('--split', default='train', choices=('train', 'selection'))
    prepare.add_argument('--shard-index', required=True, type=int)
    prepare.add_argument('--validate-only', action='store_true')


def execute(arguments: argparse.Namespace) -> int:
    from .stream_runner import _preparation, prepare_stream
    try:
        kwargs = dict(arm=arguments.arm, split=arguments.split, shard_index=arguments.shard_index)
        if arguments.validate_only:
            _, result = _preparation(arguments.recipe, arguments.destination, **kwargs)
            result['validation_only'] = True
        else:
            result = prepare_stream(arguments.recipe, arguments.destination, **kwargs)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f'spire-learn curriculum: {error}', file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m spire_learn.neural.stream_cli')
    add_arguments(parser)
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if type(error.code) is int else 2
    return execute(arguments)


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['main']
