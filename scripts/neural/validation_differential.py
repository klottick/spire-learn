"""Compare the fast validation path with the reference validator on real Gym frames.

Drives each configured mode with random legal actions, as ``soak_gym_v1.py`` does.
Every frame Learn validates is also checked by both paths directly. Every
``--mutate-every``-th frame additionally gets ``--mutations`` random single-cell
changes, and each changed frame is checked by both paths. Any disagreement, in
either direction, is recorded and makes the run fail. No training runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import struct
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

_ROOT = Path(__file__).resolve().parents[2]
_MODES = ('standalone', 'run', 'standard-standalone', 'standard-run')


def _reference_accepts(schema_module, frame, parsed) -> tuple[bool, str | None]:
    try:
        schema_module._validate_frame_contents_reference(frame, parsed)
    except Exception as error:
        return False, f'{type(error).__name__}: {error}'
    return True, None


def _fast_accepts(schema_module, frame, parsed) -> bool:
    try:
        return bool(schema_module._frame_is_valid_fast(frame, parsed))
    except Exception:
        return False


def _mutated(frame, rng: random.Random):
    from spire_learn.neural.contracts import BufferArray, CombatFrame
    key = rng.choice(sorted(frame.tensors))
    array = frame.tensors[key]
    code, size = ('b', 1) if array.dtype == 'int8' else ('q', 8)
    cells = len(array.data) // size
    if cells == 0:
        return None, None
    cell = rng.randrange(cells)
    old = struct.unpack_from('=' + code, array.data, cell * size)[0]
    choices = [0, 1, -1, 2, old + 1, old - 1] if code == 'q' else [0, 1, 2, -1]
    value = rng.choice([choice for choice in choices if choice != old] or [old ^ 1])
    try:
        packed = struct.pack('=' + code, value)
    except struct.error:
        return None, None
    data = array.data[:cell * size] + packed + array.data[(cell + 1) * size:]
    tensors = dict(frame.tensors)
    tensors[key] = BufferArray(array.dtype, array.shape, data)
    return CombatFrame(tensors, frame.schema_fingerprint), f'{key}[{cell}] {old}->{value}'


def run(output: Path, *, profile_dir: Path, steps: int, mutate_every: int, mutations: int,
        seed: int = 17) -> dict[str, object]:
    from scripts.neural.soak_gym_v1 import _drive, _entry_ids
    from spire_learn.neural import envs, schema as schema_module
    from spire_learn.neural.config import load_neural_config

    rng = random.Random(seed)
    totals = {'frames': 0, 'frames_fast_accepted': 0, 'mutations': 0,
              'mutations_rejected': 0, 'seconds_reference': 0.0, 'seconds_fast': 0.0}
    disagreements: list[dict[str, object]] = []
    original = schema_module._validate_frame_contents

    def check(frame, parsed, label):
        started = time.perf_counter()
        accepted, error = _reference_accepts(schema_module, frame, parsed)
        middle = time.perf_counter()
        fast = _fast_accepts(schema_module, frame, parsed)
        totals['seconds_reference'] += middle - started
        totals['seconds_fast'] += time.perf_counter() - middle
        if fast != accepted and len(disagreements) < 50:
            disagreements.append({'case': label, 'reference_accepts': accepted,
                                  'fast_accepts': fast, 'reference_error': error})
        return accepted, fast

    def observed(frame, parsed):
        totals['frames'] += 1
        _, fast = check(frame, parsed, f'frame {totals["frames"]}')
        totals['frames_fast_accepted'] += fast
        if totals['frames'] % mutate_every == 0:
            for _ in range(mutations):
                changed, label = _mutated(frame, rng)
                if changed is None:
                    continue
                totals['mutations'] += 1
                accepted, _ = check(changed, parsed, f'frame {totals["frames"]} {label}')
                totals['mutations_rejected'] += not accepted
        return original(frame, parsed)

    modes: dict[str, object] = {}
    schema_module._validate_frame_contents = observed
    try:
        for mode in _MODES:
            config = load_neural_config(profile_dir / f'{mode}.toml')
            driver = envs.make_driver(config, config.cases)
            mode_rng = random.Random(11)
            try:
                modes[mode] = _drive(driver, mode=mode, steps=steps, rng=mode_rng,
                                     entry_ids=lambda c=config, d=driver, r=mode_rng, m=mode:
                                     _entry_ids(c.cases, m, r, d.batch_size))
            finally:
                driver.close()
    finally:
        schema_module._validate_frame_contents = original
    result = {'schema': 'spire-learn-validation-differential-v1', 'profile_dir': str(profile_dir),
              'steps_per_mode': steps, 'mutate_every': mutate_every, 'mutations_per_frame': mutations,
              'modes': modes, **{key: round(value, 3) if isinstance(value, float) else value
                                 for key, value in totals.items()},
              'disagreements': disagreements,
              'status': 'disagree' if disagreements else ('agree' if totals['frames'] else 'no_frames')}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation-differential.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile-dir', type=Path, default=Path('configs/neural/gym-1.1.1'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--mutate-every', type=int, default=25)
    parser.add_argument('--mutations', type=int, default=20)
    arguments = parser.parse_args(argv)
    if not arguments.output.is_absolute():
        print('--output must be an absolute path', file=sys.stderr)
        return 2
    expected = os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT')
    if expected is not None and Path(expected).resolve() != arguments.output.resolve():
        print('output must match SPIRE_LEARN_RUNNER_OUTPUT', file=sys.stderr)
        return 2
    profile_dir = arguments.profile_dir
    if not profile_dir.is_absolute():
        profile_dir = _ROOT / profile_dir
    result = run(arguments.output, profile_dir=profile_dir, steps=arguments.steps,
                 mutate_every=arguments.mutate_every, mutations=arguments.mutations)
    print(json.dumps({key: result[key] for key in ('status', 'frames', 'frames_fast_accepted',
                                                   'mutations', 'mutations_rejected',
                                                   'seconds_reference', 'seconds_fast')}))
    return 0 if result['status'] == 'agree' else 1


if __name__ == '__main__':
    raise SystemExit(main())
