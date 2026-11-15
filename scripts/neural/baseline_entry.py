"""Gym 8726d467 fresh-weights baseline: qualification, mixed-stream training, panel evaluation.

Runs inside the runtime container started by ``runtime_container.py``. It composes
Learn's own building blocks: ``prepare_stream``, ``run_stream_segment``,
``build_development_panel`` and ``summarize_development_reports``. Faults stop
the baseline with a recorded reason; nothing is retried and no completed segment
is replayed.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

PLAN_SCHEMA = 'spire-learn-baseline-plan-v1'
MIN_FREE_BYTES = 30 * 2**30
_EVALUATION_ORDER = ('fresh-initial', 'final', 'near-135168', 'near-270336')


def _default_dependencies() -> dict[str, Callable]:
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.runner import create_runtime, evaluate_policy
    from spire_learn.neural.scenario_stream import load_scenario_recipe
    from spire_learn.neural.stream_runner import (
        build_development_panel, prepare_stream, run_stream_segment, summarize_development_reports,
    )
    return dict(
        prepare_stream=prepare_stream, load_config=load_neural_config, create_runtime=create_runtime,
        run_stream_segment=run_stream_segment, evaluate=evaluate_policy,
        preflight=preflight_checkpoint, load_recipe=load_scenario_recipe,
        build_panel=build_development_panel, summarize=summarize_development_reports,
        disk_free=lambda path: shutil.disk_usage(path).free, tree_bytes=_tree_bytes,
    )


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('unsupported baseline plan schema')
    target = plan.get('target_learned_decisions')
    if type(target) is not int or target < 1:
        raise ValueError('baseline plan needs a positive learned-decision target')
    return plan


def _json(value: object) -> object:
    return json.loads(json.dumps(value, default=str, sort_keys=True))


class _Journal:
    """Append-only progress record; one JSON object per line, flushed per event."""

    def __init__(self, path: Path, clock: Callable[[], float], started: float):
        self._path, self._clock, self._started = path, clock, started

    def write(self, event: str, **fields: object) -> None:
        row = {'event': event, 'elapsed_seconds': round(self._clock() - self._started, 3), **fields}
        with self._path.open('a') as stream:
            stream.write(json.dumps(_json(row), sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def _nearest(checkpoints: list[dict], decisions: int) -> dict | None:
    return min(checkpoints, key=lambda row: (abs(row['learned_decisions'] - decisions),
                                             row['learned_decisions']), default=None)


def _panel_configs(panel: Mapping, template: Path, directory: Path, plan: Mapping) -> list[dict]:
    """One evaluate config per development case, mirroring the campaign panel package."""
    import tomllib
    from spire_learn.neural.stream_runner import _toml
    evaluation = plan['evaluation']
    base = tomllib.loads(Path(template).read_text())
    directory.mkdir()
    rows = []
    for ordinal, case in enumerate(panel['manifest']['cases']):
        document = json.loads(json.dumps(base))
        document.update(operation='evaluate', split='selection', partition='selection',
                        manifest_path=f'case-{ordinal}.json', output_root='runs')
        document['rollout'] = {'rows': 1, 'steps': 1}
        document['budgets'] = dict(max_training_decisions=evaluation['case_actions'],
                                   max_environment_actions=evaluation['case_actions'],
                                   max_optimizer_steps=1, max_wall_seconds=float(evaluation['case_seconds']))
        manifest = {**panel['manifest'], 'cases': [case]}
        with (directory / f'case-{ordinal}.json').open('x') as stream:
            stream.write(json.dumps(manifest, sort_keys=True) + '\n')
        with (directory / f'config-{ordinal}.toml').open('x') as stream:
            stream.write(_toml(document))
        rows.append({'case_id': case['case_id'], 'config': directory / f'config-{ordinal}.toml'})
    return rows


def run_baseline(root: Path, plan_path: Path, output: Path, *,
                 dependencies: Mapping[str, Callable] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> dict:
    from spire_learn.neural.runner import RuntimeOperationLimits
    deps = dict(_default_dependencies() if dependencies is None else dependencies)
    root, output = Path(root), Path(output)
    plan = load_plan(root / plan_path)
    started = clock()
    training_deadline = started + plan['wall_seconds']['training']
    total_deadline = started + plan['wall_seconds']['total']
    template = root / plan['template']
    recipe_path = root / plan['recipe']
    prepared = output / 'prepared'
    prepared.mkdir()
    journal = _Journal(output / 'progress.jsonl', clock, started)
    summary: dict = {'schema': 'spire-learn-baseline-summary-v1', 'plan': plan, 'stop_reason': None,
                     'checkpoints': [], 'segments': 0, 'learned_decisions': 0, 'evaluations': {}}

    def prepare(name: str, *, split: str, shard_index: int):
        destination = prepared / name
        deps['prepare_stream'](recipe_path, destination, arm=plan['arm'], split=split,
                               shard_index=shard_index, template=template)
        return deps['load_config'](destination / 'config.toml')

    def storage_stop() -> str | None:
        if deps['disk_free'](output) < MIN_FREE_BYTES:
            return 'disk_free_below_minimum'
        if deps['tree_bytes'](output) > plan['max_new_output_bytes']:
            return 'output_byte_cap'
        return None

    fresh_path = None
    final = None
    try:
        reason = storage_stop()
        if reason is not None:
            raise _Stop(reason)
        # Fresh-initial: untrained weights saved once; every later segment descends from them.
        config = prepare('shard-0-initial', split='train', shard_index=0)
        if config.environment_revision != plan['gym_revision']:
            raise _Stop('template_revision_differs_from_plan')
        summary['config_sha256'] = config.config_sha256
        runtime = deps['create_runtime'](config, purpose='train')
        try:
            fresh_path = output / 'fresh-initial'
            runtime.save(fresh_path, 'final')
        finally:
            runtime.close()
        summary['checkpoints'].append({'label': 'fresh-initial', 'path': str(fresh_path), 'learned_decisions': 0})
        journal.write('fresh_initial_saved', checkpoint=fresh_path)

        # Qualification: one short warm start validates scenarios, sampling and updates natively.
        qualification = plan['qualification']
        config = prepare('qualification', split='train', shard_index=0)
        result = deps['run_stream_segment'](config, purpose='warm_start', checkpoint=fresh_path,
            operation_limits=RuntimeOperationLimits(
                max_updates=qualification['max_updates'],
                max_environment_actions=qualification['max_environment_actions'],
                monotonic_deadline=min(training_deadline, clock() + 600)))
        if result['checkpoint'] is None:
            journal.write('qualification_failed', report=getattr(result['report'], 'summary', None))
            raise _Stop('qualification_without_checkpoint')
        journal.write('qualification_passed', checkpoint=result['checkpoint'],
                      learned_decisions=result['saved_prefix']['learned_decisions'])

        segment = plan['segment']
        parent, purpose, shard_index = fresh_path, 'warm_start', 0
        recorded: list[dict] = []
        while True:
            if clock() >= training_deadline:
                raise _Stop('training_wall')
            reason = storage_stop()
            if reason is not None:
                raise _Stop(reason)
            config = prepare(f'train-{summary["segments"]}-shard-{shard_index}', split='train',
                             shard_index=shard_index)
            result = deps['run_stream_segment'](config, purpose=purpose, checkpoint=parent,
                operation_limits=RuntimeOperationLimits(
                    max_updates=segment['max_updates'],
                    max_environment_actions=segment['max_environment_actions'],
                    monotonic_deadline=min(training_deadline, clock() + 900)))
            summary['segments'] += 1
            if result['checkpoint'] is None:
                journal.write('segment_without_checkpoint', shard_index=shard_index,
                              report=getattr(result['report'], 'summary', None))
                raise _Stop('segment_without_checkpoint')
            decisions = result['saved_prefix']['learned_decisions']
            row = {'path': str(result['checkpoint']), 'learned_decisions': decisions,
                   'shard_index': shard_index, 'segment': summary['segments']}
            recorded.append(row)
            final = row
            summary['learned_decisions'] = decisions
            journal.write('segment_complete', **row, prefix=result['saved_prefix'])
            parent, purpose = result['checkpoint'], 'resume'
            if decisions >= plan['target_learned_decisions']:
                raise _Stop('target_reached')
            if result['next_shard'] is None:
                raise _Stop('stream_exhausted')
            shard_index = result['next_shard']['shard_index']
    except _Stop as stop:
        summary['stop_reason'] = stop.reason
    except Exception as error:
        summary['stop_reason'] = 'fault'
        summary['fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
        journal.write('fault', **summary['fault'])

    if final is not None:
        summary['checkpoints'].append({'label': 'final', **final})
        for near in plan['retain_near_decisions']:
            chosen = _nearest(recorded, near)
            if chosen is not None and chosen is not final:
                summary['checkpoints'].append({'label': f'near-{near}', **chosen})
    journal.write('training_stopped', stop_reason=summary['stop_reason'],
                  learned_decisions=summary['learned_decisions'])

    if fresh_path is not None:
        try:
            _evaluate(summary, deps, plan, template, recipe_path, output, total_deadline, clock, journal)
        except Exception as error:
            # An evaluation fault never rewrites training results; it is reported, not retried.
            summary['evaluation_fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            journal.write('evaluation_fault', **summary['evaluation_fault'])
    with (output / 'summary.json').open('x') as stream:
        stream.write(json.dumps(_json(summary), indent=2, sort_keys=True) + '\n')
    return summary


def _evaluate(summary, deps, plan, template, recipe_path, output, deadline, clock, journal) -> None:
    from spire_learn.neural.runner import RuntimeOperationLimits
    panel = deps['build_panel'](deps['load_recipe'](recipe_path))
    with (output / 'development-panel.json').open('x') as stream:
        stream.write(json.dumps(_json(panel), sort_keys=True) + '\n')
    rows = _panel_configs(panel, template, output / 'panel', plan)
    labelled = {row['label']: row for row in summary['checkpoints']}
    for label in _EVALUATION_ORDER:
        if label not in labelled:
            continue
        if clock() >= deadline:
            summary['evaluations'][label] = {'status': 'skipped_wall'}
            continue
        checkpoint = Path(labelled[label]['path'])
        reports = {}
        for row in rows:
            if clock() >= deadline:
                break
            config = deps['load_config'](row['config'])
            deps['preflight'](checkpoint, config.accepted_identities[0], purpose='evaluate')
            runtime = deps['create_runtime'](config, purpose='evaluate', checkpoint=checkpoint,
                                             policy_kind='checkpoint',
                                             action_selection=plan['evaluation']['action_selection'])
            runtime.operation_limits = RuntimeOperationLimits(
                max_environment_actions=plan['evaluation']['case_actions'],
                monotonic_deadline=min(deadline, clock() + plan['evaluation']['case_seconds']))
            # evaluate_policy owns and closes the runtime.
            reports[row['case_id']] = deps['evaluate'](config, runtime=runtime, clock=clock,
                                                       allow_final=False)
        summary['evaluations'][label] = {
            'checkpoint': str(checkpoint), 'cases_evaluated': len(reports),
            'summary': deps['summarize'](panel, reports)}
        journal.write('evaluation_complete', label=label, cases_evaluated=len(reports))


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args(argv)
    summary = run_baseline(arguments.root, arguments.plan, arguments.output)
    print(json.dumps({'stop_reason': summary['stop_reason'], 'learned_decisions': summary['learned_decisions'],
                      'segments': summary['segments']}, sort_keys=True))
    return 0 if summary['stop_reason'] == 'target_reached' else 1


if __name__ == '__main__':
    raise SystemExit(main())
