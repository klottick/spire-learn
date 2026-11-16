"""cProfile breakdown of driver-side frame validation on the real Gym build; no training.

Drives one configured mode with random legal actions, as ``soak_gym_v1.py`` does,
under ``cProfile`` and writes the functions with the most own and cumulative time
to ``validation-profile.json`` and ``validation-profile.txt``. Run it inside the
offline runtime container so the pinned native overlay is importable.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import random
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

_ROOT = Path(__file__).resolve().parents[2]


def _rows(stats: pstats.Stats, key: str, limit: int) -> list[dict[str, object]]:
    entries = []
    for (filename, line, name), (_calls, calls, own, cumulative, _callers) in stats.stats.items():
        entries.append({'function': f'{Path(filename).name}:{line}:{name}', 'calls': calls,
                        'own_seconds': round(own, 4), 'cumulative_seconds': round(cumulative, 4)})
    entries.sort(key=lambda row: -row[key])
    return entries[:limit]


def run(output: Path, *, config_path: Path, steps: int, limit: int = 40) -> dict[str, object]:
    from scripts.neural.soak_gym_v1 import _drive, _entry_ids
    from spire_learn.neural import envs
    from spire_learn.neural.config import load_neural_config

    config = load_neural_config(config_path)
    driver = envs.make_driver(config, config.cases)
    rng = random.Random(11)
    profiler = cProfile.Profile()
    try:
        started = time.perf_counter()
        profiler.enable()
        stats = _drive(driver, mode=config.mode, steps=steps, rng=rng,
                       entry_ids=lambda: _entry_ids(config.cases, config.mode, rng, driver.batch_size))
        profiler.disable()
        wall = time.perf_counter() - started
    finally:
        driver.close()
    profile = pstats.Stats(profiler)
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats('tottime').print_stats(limit)
    result = {'schema': 'spire-learn-validation-profile-v1', 'config': str(config_path),
              'mode': config.mode, 'batch_size': driver.batch_size, 'steps': steps,
              'wall_seconds_under_profiler': round(wall, 3), 'drive': stats,
              'by_own_time': _rows(profile, 'own_seconds', limit),
              'by_cumulative_time': _rows(profile, 'cumulative_seconds', limit)}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation-profile.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'validation-profile.txt').write_text(text.getvalue())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/neural/gym-1.1.0/standalone.toml'))
    parser.add_argument('--steps', type=int, default=300)
    arguments = parser.parse_args(argv)
    if not arguments.output.is_absolute():
        print('--output must be an absolute path', file=sys.stderr)
        return 2
    expected = os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT')
    if expected is not None and Path(expected).resolve() != arguments.output.resolve():
        print('output must match SPIRE_LEARN_RUNNER_OUTPUT', file=sys.stderr)
        return 2
    config = arguments.config if arguments.config.is_absolute() else _ROOT / arguments.config
    result = run(arguments.output, config_path=config, steps=arguments.steps)
    print(json.dumps({key: result[key] for key in ('mode', 'batch_size', 'steps',
                                                   'wall_seconds_under_profiler')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
