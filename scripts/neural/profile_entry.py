"""Throughput profile of baseline-shaped training segments for a baseline-style plan.

Runs inside the runtime container started by ``runtime_container.py``. It trains
a few ordinary stream segments from fresh weights, exactly as the baseline does,
with timing spans around the native driver, frame encoding, policy inference,
bootstrap values, PPO updates and checkpoint saves. Learn's existing opt-in
spans (frame validation, public export, PPO phases) report through the same
recorder. Checkpoints stay in the output; nothing here is a training result.
"""
from __future__ import annotations

import argparse
import cProfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
import json
import os
import pstats
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

PROFILE_SCHEMA = 'spire-learn-throughput-profile-v1'
# The first segment carries CUDA/allocator warm-up; the last runs without device
# synchronization so its wall time is the honest, unperturbed total.
DEFAULT_SYNCHRONIZED = (True, True, False)


def _default_dependencies() -> dict[str, Callable]:
    from functools import partial

    from scripts.neural.baseline_entry import load_plan
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.runner import create_runtime, train
    from spire_learn.neural.stream_runner import prepare_stream, run_stream_segment
    return dict(load_plan=load_plan,
                prepare_stream=partial(prepare_stream, allow_experimental_trim_compute=True),
                load_config=partial(load_neural_config, allow_experimental_trim_compute=True),
                create_runtime=partial(create_runtime, allow_experimental_trim_compute=True),
                train=train,
                run_stream_segment=partial(run_stream_segment, allow_experimental_trim_compute=True))


def _gpu_wait(plan: Mapping, config, deps: Mapping, clock: Callable[[], float], profile: dict,
              stage: str) -> None:
    """Yield the shared GPU: wait, rather than fault, until ``min_free_bytes`` is free.

    Active only with a ``gpu_wait`` plan entry on a CUDA device; waited time is excluded from
    segment timings because the wait happens before the segment clock starts. Exceeding
    ``max_total_seconds`` raises, so a permanently busy GPU is an error, not a slow run.
    """
    settings = plan.get('gpu_wait')
    if settings is None or not str(getattr(config, 'device', '')).startswith('cuda'):
        return
    from scripts.neural import baseline_entry
    free_bytes = deps.get('gpu_free_bytes', baseline_entry._gpu_free_bytes)
    sleep = deps.get('sleep', time.sleep)
    waits = profile.setdefault('gpu_waits', {'total_waited_seconds': 0.0, 'events': []})
    started = None
    while True:
        free = free_bytes()
        if free >= settings['min_free_bytes']:
            break
        if started is None:
            started = clock()
        if waits['total_waited_seconds'] + (clock() - started) >= settings['max_total_seconds']:
            waits['total_waited_seconds'] += clock() - started
            raise RuntimeError('gpu_wait_exhausted')
        sleep(settings['poll_seconds'])
    if started is not None:
        waited = clock() - started
        waits['total_waited_seconds'] = round(waits['total_waited_seconds'] + waited, 3)
        waits['events'].append({'stage': stage, 'waited_seconds': round(waited, 3), 'free_bytes': free})


def _patch(stack: ExitStack, obj: object, name: str, replacement: Callable) -> Callable:
    own = name in vars(obj)
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    stack.callback(setattr, obj, name, original) if own else stack.callback(delattr, obj, name)
    return original


def _timed(timing, label: str, function: Callable) -> Callable:
    def call(*args, **kwargs):
        with timing.span(label):
            return function(*args, **kwargs)
    return call


def instrument(runtime, timing, stack: ExitStack) -> list[str]:
    """Wrap the runtime's own collaborators; every wrapper is restored by ``stack``.

    Returns the labels that could not be wrapped (for example a native object
    without an instance ``__dict__``), so a missing span is reported, not silent.
    """
    collector = runtime.collector
    optimizer = getattr(runtime, 'optimizer', None)
    targets = [(collector.driver, 'reset', 'driver_reset'),
               (collector.driver, 'step', 'driver_step'),
               (collector.policy, 'act', 'policy_act'),
               (collector.policy, 'values', 'bootstrap_values'),
               (collector, 'collect', 'collection'),
               (runtime, 'update', 'ppo_update'),
               (runtime, 'save', 'checkpoint_save')]
    if optimizer is not None:
        targets.append((optimizer, 'step', 'adam_step'))
    missing = []
    for obj, name, label in targets:
        if not callable(getattr(obj, name, None)):
            continue
        try:
            _patch(stack, obj, name, _timed(timing, label, getattr(obj, name)))
        except (TypeError, AttributeError):
            missing.append(label)
    return missing


def _encoding(timing, stack: ExitStack) -> None:
    # model.py binds encode_frames at import; ppo.py imports it from features per call.
    from spire_learn.neural import features, model
    for module in (features, model):
        _patch(stack, module, 'encode_frames', _timed(timing, 'encode_frames', module.encode_frames))


def _recorder(runtime, synchronized: bool):
    from scripts.neural.sustained_profile import TimingRecorder
    sync = None
    device = str(runtime.config.device)
    if synchronized and device.startswith('cuda') and runtime.torch is not None:
        sync = lambda: runtime.torch.cuda.synchronize(device)
    return TimingRecorder(synchronize=sync)


def _top(rows: Mapping[str, Mapping], wall: float) -> list[dict]:
    # Exclusive times partition the spanned time; the remainder (runtime creation,
    # checkpoint preflight, untimed loop code) is reported as ``unattributed``.
    seconds = {name: row['exclusive_wall_seconds'] for name, row in rows.items()}
    counts = {name: row['count'] for name, row in rows.items()}
    seconds['unattributed'] = max(0., wall - sum(seconds.values()))
    counts['unattributed'] = None
    return [{'span': name, 'count': counts[name], 'exclusive_wall_seconds': round(value, 3),
             'share_of_segment': round(value / wall, 4) if wall > 0 else None}
            for name, value in sorted(seconds.items(), key=lambda item: -item[1])]


def _cprofile_rows(profiler, limit: int = 60) -> list[dict]:
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append({'function': f'{Path(filename).name}:{line}:{name}', 'calls': calls,
                     'own_seconds': round(own, 4), 'cumulative_seconds': round(cumulative, 4)})
    return sorted(rows, key=lambda row: -row['cumulative_seconds'])[:limit]


def run_profile(root: Path, plan_path: Path, output: Path, *, checkpoint: Path | None = None,
                synchronized=DEFAULT_SYNCHRONIZED, cprofile_segment: int | None = None,
                dependencies: Mapping[str, Callable] | None = None,
                clock: Callable[[], float] = time.monotonic) -> dict:
    from scripts.neural.baseline_entry import _limits
    from spire_learn.neural.profiling import observe
    deps = dict(_default_dependencies() if dependencies is None else dependencies)
    root, output = Path(root).resolve(strict=True), Path(output)
    if not output.is_absolute():
        raise ValueError('output must be absolute')
    plan = deps['load_plan'](plan_path if Path(plan_path).is_absolute() else root / plan_path)
    profile: dict = {'schema': PROFILE_SCHEMA, 'baseline_plan': plan['baseline_id'],
                     'gym_revision': plan['gym_revision'], 'segment_allowance': plan['segment'],
                     'segments': [], 'status': 'incomplete'}
    prepared = output / 'prepared'
    prepared.mkdir()
    # Warm-starting from a fresh-initial checkpoint follows the baseline's own path.
    parent, purpose, prefix, shard_index = checkpoint, 'train' if checkpoint is None else 'warm_start', None, 0
    profile['parent_checkpoint'] = None if checkpoint is None else str(checkpoint)
    try:
        for index, sync in enumerate(synchronized):
            destination = prepared / f'segment-{index}-shard-{shard_index}'
            deps['prepare_stream'](root / plan['recipe'], destination, arm=plan['arm'], split='train',
                                   shard_index=shard_index, template=root / plan['template'])
            config = deps['load_config'](destination / 'config.toml')
            if config.environment_revision != plan['gym_revision']:
                raise ValueError('template revision differs from plan')
            _gpu_wait(plan, config, deps, clock, profile, f'segment-{index}')
            row: dict = {'segment': index, 'shard_index': shard_index, 'synchronized': sync}
            profile['segments'].append(row)

            phases = row['phases'] = {}
            timer = clock  # train() receives its own clock argument; phases use this one

            def create_runtime(*args, **kwargs):
                begun = timer()
                try:
                    runtime = deps['create_runtime'](*args, **kwargs)
                finally:
                    phases['create_runtime_seconds'] = round(timer() - begun, 3)
                row['device'] = str(runtime.config.device)
                row['_timing'] = _recorder(runtime, sync)
                return runtime

            def train(config, *, runtime, clock):
                timing = row['_timing']
                begun = timer()
                try:
                    with ExitStack() as stack:
                        row['not_instrumented'] = instrument(runtime, timing, stack)
                        _encoding(timing, stack)
                        with observe(timing):
                            return deps['train'](config, runtime=runtime, clock=clock)
                finally:
                    phases['train_seconds'] = round(timer() - begun, 3)

            profiler = cProfile.Profile() if index == cprofile_segment else None
            started = clock()
            try:
                if profiler is not None:
                    profiler.enable()
                result = deps['run_stream_segment'](
                    config, purpose=purpose, checkpoint=parent,
                    operation_limits=_limits(prefix, plan['segment'], clock() + 900),
                    dependencies={'create_runtime': create_runtime, 'train': train, 'clock': clock})
            finally:
                # A faulting segment still reports the time it spent and where.
                wall = clock() - started
                if profiler is not None:
                    profiler.disable()
                    try:
                        row['cprofile_top_cumulative'] = _cprofile_rows(profiler)
                    except Exception as error:  # never mask the segment's own outcome
                        row['cprofile_error'] = f'{type(error).__name__}: {error}'
                phases['after_train_seconds'] = round(
                    wall - phases.get('create_runtime_seconds', 0) - phases.get('train_seconds', 0), 3)
                timing = row.pop('_timing', None)
                rows = timing.snapshot() if timing is not None else {}
                row.update(wall_seconds=round(wall, 3), spans=rows, ranked=_top(rows, wall))
            previous = (prefix or {}).get('learned_decisions', 0)
            if result['checkpoint'] is None:
                row['report'] = getattr(result['report'], 'summary', None)
                profile['stop_reason'] = 'segment_without_checkpoint'
                break
            prefix = result['saved_prefix']
            decisions = prefix['learned_decisions'] - previous
            row.update(learned_decisions=decisions,
                       decisions_per_second=round(decisions / wall, 2) if wall > 0 else None)
            parent, purpose = result['checkpoint'], 'resume'
            if result['next_shard'] is None:
                profile['stop_reason'] = 'stream_exhausted'
                break
            shard_index = result['next_shard']['shard_index']
        else:
            profile['status'] = 'complete'
    except Exception as error:
        profile['fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
    finally:
        for row in profile['segments']:
            row.pop('_timing', None)
        with (output / 'profile.json').open('x') as stream:
            stream.write(json.dumps(profile, indent=2, sort_keys=True, default=str) + '\n')
    return profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, help='fresh-initial checkpoint to warm-start from')
    parser.add_argument('--cprofile-segment', type=int,
                        help='also run cProfile over this segment index (slows that segment)')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args(argv)
    if not arguments.root.is_absolute() or not arguments.output.is_absolute():
        print('--root and --output must be absolute paths', file=sys.stderr)
        return 2
    expected = os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT')
    if expected is not None and Path(expected).resolve() != arguments.output.resolve():
        print('output must match SPIRE_LEARN_RUNNER_OUTPUT', file=sys.stderr)
        return 2
    profile = run_profile(arguments.root, arguments.plan, arguments.output, checkpoint=arguments.checkpoint,
                          cprofile_segment=arguments.cprofile_segment)
    print(json.dumps({'status': profile['status'], 'segments': [
        {key: row.get(key) for key in ('segment', 'synchronized', 'wall_seconds', 'decisions_per_second')}
        for row in profile['segments']]}, sort_keys=True))
    return 0 if profile['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
