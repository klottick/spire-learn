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
import math
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
# Final first: it is the result the baseline exists to measure.
_EVALUATION_ORDER = ('final', 'fresh-initial', 'near-135168', 'near-270336')


def _default_dependencies() -> dict[str, Callable]:
    from spire_learn.neural.checkpoint import load_checkpoint, preflight_checkpoint
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
        load_checkpoint=load_checkpoint, weights_snapshot=_weights_snapshot, weights_equal=_weights_equal,
    )


def _weights_snapshot(runtime) -> dict:
    return {name: value.detach().cpu().clone() for name, value in runtime.policy.state_dict().items()}


def _weights_equal(before: Mapping, after: Mapping) -> bool:
    import torch
    return set(before) == set(after) and all(
        torch.equal(before[name], after[name].detach().cpu()) for name in before)


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


def _limits(prefix: Mapping | None, allowance: Mapping, deadline: float):
    """Per-segment allowance on top of the parent's saved cumulative counters.

    The runner compares ``max_updates`` and ``max_environment_actions`` with
    cumulative progress including the restored prefix, so a fixed value would
    stop every resumed segment immediately.
    """
    from spire_learn.neural.runner import RuntimeOperationLimits
    prefix = prefix or {}
    return RuntimeOperationLimits(
        max_updates=prefix.get('update_count', 0) + allowance['max_updates'],
        max_environment_actions=prefix.get('attempted_environment_slots', 0)
        + allowance['max_environment_actions'],
        monotonic_deadline=deadline)


def _validate_runner_contract(root: Path, output: Path, plan: Mapping, *, require_environment: bool) -> None:
    if not root.is_absolute():
        raise ValueError('root must be absolute')
    if not output.is_absolute():
        raise ValueError('output must be absolute')
    expected_output = os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT')
    if expected_output is None:
        if require_environment:
            raise ValueError('SPIRE_LEARN_RUNNER_OUTPUT is required')
    else:
        runner_output = Path(expected_output)
        if not runner_output.is_absolute():
            raise ValueError('SPIRE_LEARN_RUNNER_OUTPUT must be an absolute path')
        if runner_output.resolve() != output.resolve():
            raise ValueError('output must match SPIRE_LEARN_RUNNER_OUTPUT')
    timeout_text = os.environ.get('SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS')
    if timeout_text is None:
        if require_environment:
            raise ValueError('SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS is required')
    else:
        try:
            timeout = float(timeout_text)
        except ValueError as error:
            raise ValueError('SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS must be numeric') from error
        minimum = plan.get('runner_timeout_seconds')
        if (not math.isfinite(timeout) or type(minimum) not in (int, float)
                or not math.isfinite(float(minimum)) or timeout < minimum):
            raise ValueError(f'SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS must be at least {minimum}')


# Training stops after which evaluating the reached checkpoints is meaningful.
_EVALUABLE_STOPS = frozenset({'target_reached', 'training_wall', 'stream_exhausted',
                              'segment_without_progress', 'segment_without_checkpoint'})


def run_baseline(root: Path, plan_path: Path, output: Path, *,
                 dependencies: Mapping[str, Callable] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> dict:
    deps = dict(_default_dependencies() if dependencies is None else dependencies)
    root, output = Path(root), Path(output)
    if not root.is_absolute():
        raise ValueError('root must be absolute')
    if not output.is_absolute():
        raise ValueError('output must be absolute')
    root = root.resolve(strict=True)
    plan_path = Path(plan_path)
    plan = load_plan(plan_path if plan_path.is_absolute() else root / plan_path)
    _validate_runner_contract(root, output, plan, require_environment=False)
    started = clock()
    training_deadline = started + plan['wall_seconds']['training']
    total_deadline = started + plan['wall_seconds']['total']
    template = root / plan['template']
    recipe_path = root / plan['recipe']
    prepared = output / 'prepared'
    journal = _Journal(output / 'progress.jsonl', clock, started)
    summary: dict = {'schema': 'spire-learn-baseline-summary-v1', 'plan': plan, 'stop_reason': None,
                     'checkpoints': [], 'segments': 0, 'learned_decisions': 0, 'evaluations': {},
                     'result': 'incomplete'}

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
    recorded: list[dict] = []
    try:
        prepared.mkdir()
        try:
            _train(summary, deps, plan, prepare, storage_stop, recorded, training_deadline, clock,
                   journal, output)
        except _Stop as stop:
            summary['stop_reason'] = stop.reason
        except Exception as error:
            summary['stop_reason'] = 'fault'
            summary['fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            journal.write('fault', **summary['fault'])
        fresh_path = next((row['path'] for row in summary['checkpoints']
                           if row['label'] == 'fresh-initial'), None)
        if recorded:
            final = recorded[-1]
            summary['checkpoints'].append({'label': 'final', **final})
            for near in plan['retain_near_decisions']:
                chosen = _nearest(recorded, near)
                if chosen is not None and chosen is not final:
                    summary['checkpoints'].append({'label': f'near-{near}', **chosen})
        journal.write('training_stopped', stop_reason=summary['stop_reason'],
                      learned_decisions=summary['learned_decisions'])
        if fresh_path is not None and summary['stop_reason'] in _EVALUABLE_STOPS:
            try:
                _evaluate(summary, deps, plan, template, recipe_path, output, total_deadline, clock,
                          journal, storage_stop)
            except Exception as error:
                # An evaluation fault never rewrites training results; it is reported, not retried.
                summary['evaluation_fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
                journal.write('evaluation_fault', **summary['evaluation_fault'])
                for evaluation in summary['evaluations'].values():
                    if evaluation.get('status') == 'running':
                        evaluation['status'] = 'fault'  # partial rows kept; panel not complete
        else:
            summary['evaluation_skipped'] = summary['stop_reason']
        evaluated = summary['evaluations']
        summary['result'] = ('complete' if summary['stop_reason'] == 'target_reached'
                             and 'evaluation_fault' not in summary
                             and all(evaluated.get(row['label'], {}).get('status') == 'complete'
                                     for row in summary['checkpoints'])
                             else 'incomplete')
    except BaseException as interrupt:
        # Process control (interrupt, exit) still publishes what is known, then propagates.
        summary['interrupted'] = type(interrupt).__name__
        summary['result'] = 'interrupted'
        raise
    finally:
        with (output / 'summary.json').open('x') as stream:
            stream.write(json.dumps(_json(summary), indent=2, sort_keys=True) + '\n')
    return summary


def _train(summary, deps, plan, prepare, storage_stop, recorded, deadline, clock, journal, output) -> None:
    reason = storage_stop()
    if reason is not None:
        raise _Stop(reason)
    # Fresh-initial: untrained weights saved once; every later segment descends from them.
    config = prepare('shard-0-initial', split='train', shard_index=0)
    if config.environment_revision != plan['gym_revision']:
        raise _Stop('template_revision_differs_from_plan')
    summary['config_sha256'] = config.config_sha256
    summary['device'] = getattr(config, 'device', None)
    fresh_path = output / 'fresh-initial'
    runtime = deps['create_runtime'](config, purpose='train')
    try:
        before = deps['weights_snapshot'](runtime)
        runtime.save(fresh_path, 'final')
    finally:
        runtime.close()
    # The saved fresh weights must load back exactly, untrained, before anything trains on them.
    deps['preflight'](fresh_path, config.accepted_identities[0], purpose='evaluate')
    loaded = deps['load_checkpoint'](fresh_path, config.accepted_identities[0], purpose='resume', device='cpu')
    state = loaded.metadata['training_state']
    if (not deps['weights_equal'](before, loaded.model_tensors)
            or state['optimizer_step'] != 0 or state['behavior_policy_version'] != 0):
        raise _Stop('fresh_initial_roundtrip_mismatch')
    summary['checkpoints'].append({'label': 'fresh-initial', 'path': str(fresh_path), 'learned_decisions': 0})
    journal.write('fresh_initial_saved', checkpoint=fresh_path)

    # Qualification: one short warm start validates scenarios, sampling and updates natively.
    config = prepare('qualification', split='train', shard_index=0)
    result = deps['run_stream_segment'](config, purpose='warm_start', checkpoint=fresh_path,
        operation_limits=_limits(None, plan['qualification'], min(deadline, clock() + 600)))
    if result['checkpoint'] is None:
        journal.write('qualification_failed', report=getattr(result['report'], 'summary', None))
        raise _Stop('qualification_without_checkpoint')
    if result['saved_prefix']['learned_decisions'] < 1:
        raise _Stop('qualification_without_progress')
    journal.write('qualification_passed', checkpoint=result['checkpoint'],
                  learned_decisions=result['saved_prefix']['learned_decisions'])

    parent, purpose, shard_index, prefix = fresh_path, 'warm_start', 0, None
    while True:
        if clock() >= deadline:
            raise _Stop('training_wall')
        reason = storage_stop()
        if reason is not None:
            raise _Stop(reason)
        config = prepare(f'train-{summary["segments"]}-shard-{shard_index}', split='train',
                         shard_index=shard_index)
        result = deps['run_stream_segment'](config, purpose=purpose, checkpoint=parent,
            operation_limits=_limits(prefix, plan['segment'], min(deadline, clock() + 900)))
        summary['segments'] += 1
        if result['checkpoint'] is None:
            report = result['report']
            report_summary = getattr(report, 'summary', {}) or {}
            journal.write('segment_without_checkpoint', shard_index=shard_index, report=report_summary)
            if getattr(report, 'exit_code', 1) != 0 or report_summary.get('status') != 'incomplete':
                # A failed segment is a training fault: earlier checkpoints are not evaluated as results.
                summary['fault'] = {'type': 'SegmentFailure', 'message': str(report_summary.get('status'))}
                raise _Stop('segment_fault')
            raise _Stop('segment_without_checkpoint')
        previous = (prefix or {}).get('learned_decisions', 0)
        prefix = result['saved_prefix']
        decisions = prefix['learned_decisions']
        row = {'path': str(result['checkpoint']), 'learned_decisions': decisions,
               'shard_index': shard_index, 'segment': summary['segments']}
        recorded.append(row)
        summary['learned_decisions'] = decisions
        journal.write('segment_complete', **row, prefix=prefix)
        if decisions <= previous:
            raise _Stop('training_wall' if clock() >= deadline else 'segment_without_progress')
        parent, purpose = result['checkpoint'], 'resume'
        if decisions >= plan['target_learned_decisions']:
            raise _Stop('target_reached')
        if result['next_shard'] is None:
            raise _Stop('stream_exhausted')
        shard_index = result['next_shard']['shard_index']


def _case_outcome(report) -> str:
    summary = getattr(report, 'summary', {}) or {}
    if getattr(report, 'exit_code', 1) != 0 or summary.get('status') != 'complete':
        return 'failed'
    fight = (summary.get('fights') or {}).get('standalone_fights')
    if not isinstance(fight, Mapping):
        return 'invalid'  # a healthy report must carry its fight metrics
    if fight.get('victories'):
        return 'win'
    if fight.get('deaths'):
        return 'loss'
    if fight.get('escaped'):
        return 'escaped'
    return 'incomplete'


def _paired(fresh: Mapping[str, str], final: Mapping[str, str], *, total: int) -> dict:
    """Fresh vs final on cases both evaluated healthily; exact two-sided McNemar test."""
    from math import comb
    shared = sorted(set(fresh) & set(final))
    settled = ('win', 'loss', 'escaped')
    healthy = [c for c in shared if fresh[c] in settled and final[c] in settled]
    censored = [c for c in shared if c not in healthy
                and {fresh[c], final[c]} <= {'win', 'loss', 'escaped', 'incomplete'}]
    both = sum(fresh[c] == 'win' and final[c] == 'win' for c in healthy)
    fresh_only = sum(fresh[c] == 'win' and final[c] != 'win' for c in healthy)
    final_only = sum(fresh[c] != 'win' and final[c] == 'win' for c in healthy)
    neither = len(healthy) - both - fresh_only - final_only
    discordant = fresh_only + final_only
    tail = sum(comb(discordant, k) for k in range(min(fresh_only, final_only) + 1)) / 2**discordant \
        if discordant else 1.0
    return {'paired_cases': len(healthy), 'excluded_censored': len(censored),
            'excluded_failed_invalid_or_missing': total - len(healthy) - len(censored),
            'both_win': both, 'fresh_only_win': fresh_only, 'final_only_win': final_only,
            'neither_win': neither, 'fresh_wins': both + fresh_only, 'final_wins': both + final_only,
            'exact_mcnemar_two_sided_p': min(1.0, 2 * tail) if discordant else 1.0}


def _evaluate(summary, deps, plan, template, recipe_path, output, deadline, clock, journal,
              storage_stop) -> None:
    panel = deps['build_panel'](deps['load_recipe'](recipe_path))
    with (output / 'development-panel.json').open('x') as stream:
        stream.write(json.dumps(_json(panel), sort_keys=True) + '\n')
    rows = _panel_configs(panel, template, output / 'panel', plan)
    labelled = {row['label']: row for row in summary['checkpoints']}
    outcomes: dict[str, dict[str, str]] = {}
    for label in _EVALUATION_ORDER:
        if label not in labelled:
            continue
        if clock() >= deadline:
            summary['evaluations'][label] = {'status': 'skipped_wall'}
            continue
        checkpoint = Path(labelled[label]['path'])
        reports, cases, status = {}, {}, 'running'
        # Registered before the first case so exceptional exits still publish partial rows.
        evaluation = {'status': status, 'checkpoint': str(checkpoint), 'cases': cases, 'device': None}
        summary['evaluations'][label] = evaluation
        for row in rows:
            if clock() >= deadline:
                status = 'incomplete_wall'
                break
            reason = storage_stop()
            if reason is not None:
                status = 'stopped_' + reason
                break
            config = deps['load_config'](row['config'])
            deps['preflight'](checkpoint, config.accepted_identities[0], purpose='evaluate')
            runtime = deps['create_runtime'](config, purpose='evaluate', checkpoint=checkpoint,
                                             policy_kind='checkpoint',
                                             action_selection=plan['evaluation']['action_selection'])
            from spire_learn.neural.runner import RuntimeOperationLimits
            runtime.operation_limits = RuntimeOperationLimits(
                max_environment_actions=plan['evaluation']['case_actions'],
                monotonic_deadline=min(deadline, clock() + plan['evaluation']['case_seconds']))
            # evaluate_policy owns and closes the runtime.
            report = deps['evaluate'](config, runtime=runtime, clock=clock, allow_final=False)
            reports[row['case_id']] = report
            evaluation['device'] = getattr(config, 'device', None)
            evaluation['cases_evaluated'] = len(reports)
            outcome = _case_outcome(report)
            cases[row['case_id']] = {
                'outcome': outcome, 'config_sha256': getattr(config, 'config_sha256', None),
                'run_dir': str(getattr(report, 'run_dir', None))}
            if outcome in ('failed', 'invalid'):
                status = 'operational_failure' if outcome == 'failed' else 'invalid_report'
                break
        else:
            status = 'complete'
        outcomes[label] = {case: row['outcome'] for case, row in cases.items()}
        evaluation['cases_evaluated'] = len(reports)
        try:
            evaluation['summary'] = deps['summarize'](panel, reports)
        except Exception as error:
            evaluation['summary_error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            if status == 'complete':
                status = 'summary_error'
        evaluation['status'] = status
        journal.write('evaluation_complete', label=label, status=status, cases_evaluated=len(reports))
        if status != 'complete':
            break
    if 'fresh-initial' in outcomes and 'final' in outcomes:
        summary['paired_fresh_vs_final'] = _paired(outcomes['fresh-initial'], outcomes['final'], total=len(rows))


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
    if not arguments.root.is_absolute():
        print('--root must be an absolute path', file=sys.stderr)
        return 2
    if not arguments.output.is_absolute():
        print('--output must be an absolute path', file=sys.stderr)
        return 2
    plan_path = arguments.plan if arguments.plan.is_absolute() else arguments.root / arguments.plan
    try:
        plan = load_plan(plan_path)
        _validate_runner_contract(arguments.root, arguments.output, plan, require_environment=True)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    summary = run_baseline(arguments.root, arguments.plan, arguments.output)
    print(json.dumps({'result': summary['result'], 'stop_reason': summary['stop_reason'],
                      'learned_decisions': summary['learned_decisions'],
                      'segments': summary['segments']}, sort_keys=True))
    return 0 if summary['result'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
