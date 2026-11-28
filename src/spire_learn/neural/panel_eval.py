"""Panel evaluation shared by the PPO baselines and the distilled students (stdlib at import).

Extracted unchanged from ``scripts/neural/baseline_entry.py``: one evaluate config per panel case,
the shared-GPU gate re-checked before every case, the per-case outcome (win / loss / escaped /
incomplete, with the censor reason) and the panel summary. ``baseline_entry`` calls it and its
outputs are byte-identical; ``student_eval_entry`` evaluates a distilled student through it, so
both write the same per-case ``evaluations`` format that ``standard_results`` and ``teacher_audit``
read. A failed or invalid case report is an operational failure that stops the panel; nothing is
retried. Importing this module imports neither torch nor numpy nor the Gym.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path


def json_safe(value: object) -> object:
    return json.loads(json.dumps(value, default=str, sort_keys=True))


class Journal:
    """Append-only progress record; one JSON object per line, flushed per event."""

    def __init__(self, path: Path, clock: Callable[[], float], started: float, *, panel: str | None = None):
        self._path, self._clock, self._started = path, clock, started
        self._panel = panel

    def write(self, event: str, **fields: object) -> None:
        row = {'event': event, 'elapsed_seconds': round(self._clock() - self._started, 3), **fields}
        if self._panel is not None:
            row.setdefault('panel', self._panel)
        with self._path.open('a') as stream:
            stream.write(json.dumps(json_safe(row), sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def panel_configs(panel: Mapping, template: Path, directory: Path, plan: Mapping,
                  evaluation: Mapping | None = None) -> list[dict]:
    """One evaluate config per panel case, mirroring the campaign panel package."""
    import tomllib
    from spire_learn.neural.stream_runner import _toml
    evaluation = plan['evaluation'] if evaluation is None else evaluation
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
        if plan.get('max_turn_actions') is not None:
            document['budgets']['max_turn_actions'] = plan['max_turn_actions']
        manifest = {**panel['manifest'], 'cases': [case]}
        with (directory / f'case-{ordinal}.json').open('x') as stream:
            stream.write(json.dumps(manifest, sort_keys=True) + '\n')
        with (directory / f'config-{ordinal}.toml').open('x') as stream:
            stream.write(_toml(document))
        rows.append({'case_id': case['case_id'], 'config': directory / f'config-{ordinal}.toml'})
    return rows


class Budget:
    """Training and total wall deadlines; GPU waits move both later by the time waited."""

    def __init__(self, *, training: float, total: float):
        self.training, self.total = training, total


class GpuGate:
    """Yield the shared GPU: wait, rather than fault, while other services hold it.

    Training is second class on this host. Before each segment and evaluation case
    the gate waits until the device has ``min_free_bytes`` free, polling every
    ``poll_seconds``. Waiting does not consume the wall budgets, but the total wait
    is capped by ``max_total_seconds``. After each unit of work the CUDA cache is
    released so other services can load into it. Without a ``gpu_wait`` plan entry,
    or on a non-CUDA device, the gate does nothing. Memory taken by another service
    in the middle of a unit can still fault that unit; faults are never retried.
    """

    def __init__(self, plan: Mapping, deps: Mapping, clock: Callable[[], float], journal, budget: Budget):
        self._settings = plan.get('gpu_wait')
        self._deps, self._clock, self._journal, self._budget = deps, clock, journal, budget
        self.waited = 0.0

    def _active(self, config) -> bool:
        return self._settings is not None and str(getattr(config, 'device', '')).startswith('cuda')

    def wait(self, config, stage: str) -> None:
        if not self._active(config):
            return
        minimum, poll = self._settings['min_free_bytes'], self._settings['poll_seconds']
        started = None
        while True:
            free = self._deps['gpu_free_bytes']()
            if free >= minimum:
                break
            if started is None:
                started = self._clock()
                self._journal.write('gpu_wait', stage=stage, free_bytes=free, min_free_bytes=minimum)
            if self.waited + (self._clock() - started) >= self._settings['max_total_seconds']:
                self._account(started)
                raise Stop('gpu_wait_exhausted')
            self._deps['sleep'](poll)
        if started is not None:
            waited = self._account(started)
            self._journal.write('gpu_resumed', stage=stage, waited_seconds=round(waited, 3),
                                total_waited_seconds=round(self.waited, 3))

    def _account(self, started: float) -> float:
        waited = self._clock() - started
        self.waited += waited
        self._budget.training += waited
        self._budget.total += waited
        return waited

    def release(self, config) -> None:
        if self._active(config):
            self._deps['release_gpu_cache']()


def case_outcome(report) -> str:
    """Win/loss/escaped settle a case; 'incomplete' is censored; anything unhealthy fails.

    A clean fixed-case budget exhaustion (no fault, clean lifecycle) is censoring,
    not an operational failure: the evaluation continues and pairing excludes it.
    """
    from spire_learn.neural.stream_runner import is_censored_case_report
    summary = getattr(report, 'summary', {}) or {}
    if getattr(report, 'exit_code', 1) != 0 or summary.get('status') != 'complete':
        return 'incomplete' if is_censored_case_report(report) else 'failed'
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


def paired(fresh: Mapping[str, str], final: Mapping[str, str], *, total: int,
            standard: bool = False) -> dict:
    """Fresh vs final on cases both evaluated healthily; exact two-sided McNemar test.

    Standard panels only: a case escaped (Smoke Bomb) under either checkpoint is neither a
    win nor a loss, so it is excluded from pairing and counted in ``excluded_escaped``. M10
    panels keep the original dictionary exactly (escape is unreachable in their view).
    """
    from math import comb
    shared = sorted(set(fresh) & set(final))
    settled = ('win', 'loss') if standard else ('win', 'loss', 'escaped')
    healthy = [c for c in shared if fresh[c] in settled and final[c] in settled]
    escaped = [c for c in shared if c not in healthy and 'escaped' in (fresh[c], final[c])
               and {fresh[c], final[c]} <= {'win', 'loss', 'escaped'}] if standard else []
    censored = [c for c in shared if c not in healthy and c not in escaped
                and {fresh[c], final[c]} <= {'win', 'loss', 'escaped', 'incomplete'}]
    both = sum(fresh[c] == 'win' and final[c] == 'win' for c in healthy)
    fresh_only = sum(fresh[c] == 'win' and final[c] != 'win' for c in healthy)
    final_only = sum(fresh[c] != 'win' and final[c] == 'win' for c in healthy)
    neither = len(healthy) - both - fresh_only - final_only
    discordant = fresh_only + final_only
    tail = sum(comb(discordant, k) for k in range(min(fresh_only, final_only) + 1)) / 2**discordant \
        if discordant else 1.0
    result = {'paired_cases': len(healthy), 'excluded_censored': len(censored),
              'excluded_failed_invalid_or_missing': total - len(healthy) - len(censored) - len(escaped),
              'both_win': both, 'fresh_only_win': fresh_only, 'final_only_win': final_only,
              'neither_win': neither, 'fresh_wins': both + fresh_only, 'final_wins': both + final_only,
              'exact_mcnemar_two_sided_p': min(1.0, 2 * tail) if discordant else 1.0}
    if standard:
        result['excluded_escaped'] = len(escaped)
    return result


def evaluate_panel(summary, deps, plan, template, recipe_path, output, budget, gate, clock, journal,
                   storage_stop, *, labels, result_container=None,
                   panel_id='primary', evaluation=None, panel_kind='selection') -> None:
    """Evaluate each labelled checkpoint of ``summary['checkpoints']`` on one panel, case by case.

    ``deps`` supplies ``load_recipe``, ``build_panel``, ``load_config``, ``preflight``,
    ``create_runtime``, ``evaluate`` and ``summarize``. ``plan`` supplies ``gym_revision``,
    ``evaluation`` (``action_selection``, ``case_actions``, ``case_seconds``), the optional
    ``max_turn_actions`` and, for the boss panel, ``wall_seconds.total``. Labels missing from
    the checkpoints are skipped. Results go to ``result_container['evaluations'][label]``
    (``summary`` by default); ``gate`` waits before, and releases after, every case. The first
    label that does not complete stops the panel.
    """
    result_container = summary if result_container is None else result_container
    settings = plan['evaluation'] if evaluation is None else evaluation
    evaluations = result_container.setdefault('evaluations', {})
    output.mkdir(parents=True, exist_ok=True)
    from spire_learn.neural.gym_contracts import accepts_recorded_revision
    recipe = deps['load_recipe'](recipe_path)
    recipe_revision = getattr(recipe, 'gym_revision', None)  # Standard recipes carry their table's Gym revision
    # The table may be from another revision of the plan's rules epoch (a drop-in Gym), never another epoch.
    if recipe_revision is not None and not accepts_recorded_revision(plan['gym_revision'], recipe_revision):
        raise ValueError(f"panel recipe {recipe_path} is for Gym {recipe_revision}, "
                         f"but the plan is for Gym {plan['gym_revision']}")
    if panel_kind == 'boss':
        panel = deps['build_panel'](recipe, 'selection', kind='boss')
        result_container.update(panel_sha256=panel['panel_sha256'], manifest_sha256=panel['manifest_sha256'])
        # The runner config requires a finite wall budget. The run's total budget
        # bounds every boss evaluation without adding a shorter per-case deadline.
        boss_case_seconds = plan['wall_seconds']['total']
        settings = {**settings, 'case_seconds': boss_case_seconds}
        result_container['boss_case_seconds'] = boss_case_seconds
        panel_id = 'boss'
        panel_filename = 'boss-panel.json'
    else:
        panel = deps['build_panel'](recipe)
        panel_filename = 'development-panel.json'
    with (output / panel_filename).open('x') as stream:
        stream.write(json.dumps(json_safe(panel), sort_keys=True) + '\n')
    rows = panel_configs(panel, template, output / 'panel', plan, settings)
    labelled = {row['label']: row for row in summary['checkpoints']}
    outcomes: dict[str, dict[str, str]] = {}
    for label in labels:
        if label not in labelled:
            continue
        if clock() >= budget.total:
            evaluations[label] = {'status': 'skipped_wall'}
            continue
        checkpoint = Path(labelled[label]['path'])
        reports, cases, status = {}, {}, 'running'
        # Registered before the first case so exceptional exits still publish partial rows.
        evaluation = {'status': status, 'checkpoint': str(checkpoint), 'cases': cases, 'device': None}
        evaluations[label] = evaluation
        for row in rows:
            if clock() >= budget.total:
                status = 'incomplete_wall'
                break
            reason = storage_stop()
            if reason is not None:
                status = 'stopped_' + reason
                break
            config = deps['load_config'](row['config'])
            try:
                stage = f'evaluation {label}'
                if panel_id != 'primary':
                    stage += f' panel {panel_id}'
                gate.wait(config, stage)
            except Stop as stop:
                status = stop.reason
                break
            deps['preflight'](checkpoint, config.accepted_identities[0], purpose='evaluate')
            try:
                runtime = deps['create_runtime'](config, purpose='evaluate', checkpoint=checkpoint,
                                                 policy_kind='checkpoint',
                                                 action_selection=settings['action_selection'])
                from spire_learn.neural.runner import RuntimeOperationLimits
                runtime.operation_limits = RuntimeOperationLimits(
                    max_environment_actions=settings['case_actions'],
                    monotonic_deadline=min(budget.total, clock() + settings['case_seconds']))
                # evaluate_policy owns and closes the runtime.
                report = deps['evaluate'](config, runtime=runtime, clock=clock, allow_final=False)
            finally:
                gate.release(config)
            reports[row['case_id']] = report
            evaluation['device'] = getattr(config, 'device', None)
            evaluation['cases_evaluated'] = len(reports)
            outcome = case_outcome(report)
            cases[row['case_id']] = {
                'outcome': outcome, 'config_sha256': getattr(config, 'config_sha256', None),
                'run_dir': str(getattr(report, 'run_dir', None))}
            if outcome == 'incomplete':
                report_summary = report.summary
                collection_stop = (report_summary.get('collection') or {}).get('stop_reason')
                cases[row['case_id']]['censor_reason'] = collection_stop if collection_stop in {
                    'environment_action_budget', 'turn_action_budget',
                } else report_summary.get('stop_reason')
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
        journal.write('evaluation_complete', panel=panel_id, label=label, status=status,
                      cases_evaluated=len(reports))
        if status != 'complete':
            break
    if 'fresh-initial' in outcomes and 'final' in outcomes:
        result_container['paired_fresh_vs_final'] = paired(
            outcomes['fresh-initial'], outcomes['final'], total=len(rows),
            standard=panel.get('schema') == 'spire-learn-standard-panel-v2')


class Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


__all__ = [
    "Budget", "GpuGate", "Journal", "Stop", "case_outcome", "evaluate_panel", "json_safe", "paired",
    "panel_configs",
]
