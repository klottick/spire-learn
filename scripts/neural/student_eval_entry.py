#!/usr/bin/env python3.12
"""Evaluate one distilled student on the selection or boss panel (runtime container; torch).

    student_eval_entry.py --plan PLAN.json --student DISTILL_OUTPUT --output OUTPUT [--panel selection|boss]

``DISTILL_OUTPUT`` is a finished ``distill_train_entry.py`` output. Its ``summary.json`` names the
chosen checkpoint (delta 12: lowest validation regret, ``checkpoint-best`` or ``checkpoint-final``),
which is the student's result and is evaluated under the label ``final``, the label the gates,
``standard_results.load_outcomes(..., "final")`` and ``teacher_audit.load_ppo`` read.

Evaluation is the baselines' own panel evaluation (``panel_eval``), so ``summary.json`` has the
same per-case ``evaluations`` format: one evaluate config per case from the plan's template, at
``FIGHT_CAP`` actions per fight and ``TURN_CAP`` actions per public turn, with the shared-GPU gate
re-checked before every case. Actions are the reflex argmax over legal actions (``deterministic``),
never sampled; the policy is built through the model-family registry, and a legal set with no
finite logit raises (an operational failure, never a loss). A censored or escaped case is kept as
such. Faults stop the evaluation with the reason recorded; nothing is retried.

After the panel, the entry times warmed batch-1 ``forward`` passes of the same checkpoint on CPU and,
for a CUDA config, on the GPU: the median of ``INFERENCE_PASSES`` after ``INFERENCE_WARMUP``.
The frame is the first panel case's public start frame.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

from spire_learn.neural.panel_eval import (  # noqa: E402  (after the sys.path set-up)
    Budget, GpuGate, Journal, Stop, evaluate_panel, json_safe,
)

PLAN_SCHEMA = 'spire-learn-student-eval-plan-v1'
SUMMARY_SCHEMA = 'spire-learn-student-eval-summary-v1'
DISTILL_SUMMARY_SCHEMA = 'spire-learn-distill-train-summary-v1'
DISTILL_OBJECTIVE_ID = 'spire-learn-combat-distill-v1'
COMPACT_PRESET = 'standard-combat-compact-v1'
FIGHT_CAP = 1024  # actions in one fight
TURN_CAP = 500  # actions in one public turn
INFERENCE_PASSES = 1000
INFERENCE_WARMUP = 50
MIN_FREE_BYTES = 30 * 2**30
LABEL = 'final'
CHECKPOINT_NAMES = ('checkpoint-best', 'checkpoint-final')
_PLAN_KEYS = {'schema', 'student_id', 'gym_revision', 'recipe', 'template', 'evaluation', 'max_turn_actions',
              'wall_seconds', 'runner_timeout_seconds', 'max_new_output_bytes'}
_OPTIONAL_PLAN_KEYS = {'gpu_wait', 'notes', 'qualifications'}


def _positive_int(value: object) -> bool:
    return type(value) is int and value >= 1


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('unsupported student evaluation plan schema')
    keys = set(plan)
    if not _PLAN_KEYS <= keys or keys - _PLAN_KEYS - _OPTIONAL_PLAN_KEYS:
        raise ValueError(f'a student evaluation plan has keys {sorted(_PLAN_KEYS)} '
                         f'and optionally {sorted(_OPTIONAL_PLAN_KEYS)}')
    for key in ('student_id', 'gym_revision', 'recipe', 'template'):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError(f'{key} must be a non-empty string')
    evaluation = plan['evaluation']
    if (not isinstance(evaluation, dict) or set(evaluation) != {'action_selection', 'case_actions', 'case_seconds'}
            or type(evaluation['case_seconds']) not in (int, float)
            or not math.isfinite(evaluation['case_seconds']) or evaluation['case_seconds'] <= 0):
        raise ValueError('evaluation needs action_selection, case_actions and a positive case_seconds')
    if evaluation['action_selection'] != 'deterministic':
        raise ValueError('students are evaluated by reflex argmax: action_selection must be deterministic')
    if evaluation['case_actions'] != FIGHT_CAP or type(evaluation['case_actions']) is not int:
        raise ValueError(f'evaluation.case_actions must be FIGHT_CAP ({FIGHT_CAP})')
    if plan['max_turn_actions'] != TURN_CAP or type(plan['max_turn_actions']) is not int:
        raise ValueError(f'max_turn_actions must be TURN_CAP ({TURN_CAP})')
    wall = plan['wall_seconds']
    if not isinstance(wall, dict) or set(wall) != {'total'} or not _positive_int(wall['total']):
        raise ValueError('wall_seconds needs exactly a positive integer total')
    if not _positive_int(plan['max_new_output_bytes']):
        raise ValueError('max_new_output_bytes must be a positive integer')
    needed = wall['total'] + 1200
    wait = plan.get('gpu_wait')
    if wait is not None:
        if (not isinstance(wait, dict) or set(wait) != {'min_free_bytes', 'poll_seconds', 'max_total_seconds'}
                or any(not _positive_int(wait[key]) for key in wait)):
            raise ValueError('gpu_wait needs positive integer min_free_bytes, poll_seconds, max_total_seconds')
        needed += wait['max_total_seconds'] + wait['poll_seconds']
    # The outer runner kill must not pre-empt the summary after the longest allowed wait.
    if not _positive_int(plan['runner_timeout_seconds']) or plan['runner_timeout_seconds'] < needed:
        raise ValueError('runner_timeout_seconds must cover the total wall, the maximum GPU wait, '
                         'one poll and 1200 s of slack')
    return plan


def check_template(template: Path, plan: Mapping) -> dict:
    """The per-case template: a compact Standard distillation evaluation config on the plan's Gym."""
    import tomllib
    document = tomllib.loads(Path(template).read_text())
    if document.get('mode') != 'standard-standalone':
        raise ValueError('a student template is a standard-standalone config')
    if str(document.get('device', '')).startswith('cuda') and plan.get('gpu_wait') is None:
        raise ValueError('a CUDA template needs the plan gpu_wait: the GPU is shared, so gating is mandatory')
    if 'ppo' in document or 'scenario_stream' in document:
        raise ValueError('a student template has no [ppo] table or scenario stream')
    environment = document.get('environment', {})
    if environment.get('gym_revision') != plan['gym_revision']:
        raise ValueError("the template's gym_revision differs from the plan's")
    if environment.get('standard_view_layout') != 'compact' or environment.get('standard_view_preset') != COMPACT_PRESET:
        raise ValueError(f'a student template uses the compact {COMPACT_PRESET} view')
    for table in document.values():
        if isinstance(table, dict) and 'experimental_trim_compute' in table:
            raise ValueError('student evaluation refuses experimental_trim_compute')
    identities = document.get('accepted_identities')
    if not isinstance(identities, list) or len(identities) != 1:
        raise ValueError('a student template declares exactly one accepted identity')
    identity = identities[0]
    if identity.get('objective_id') != DISTILL_OBJECTIVE_ID:
        raise ValueError(f'a student template accepts the {DISTILL_OBJECTIVE_ID} objective only')
    return identity


def load_student(student: Path, identity: Mapping) -> dict:
    """The finished distillation run's chosen checkpoint, checked against the template identity."""
    student = Path(student)
    if not student.is_absolute():
        raise ValueError('--student must be an absolute path')
    data = (student / 'summary.json').read_bytes()
    summary = json.loads(data)
    if not isinstance(summary, dict) or summary.get('schema') != DISTILL_SUMMARY_SCHEMA:
        raise ValueError('the student summary is not a distillation training summary')
    if summary.get('status') != 'complete':
        raise ValueError(f"the student run is not complete (status {summary.get('status')!r})")
    trained = summary.get('identity') or {}
    for key in ('model_id', 'features_id', 'action_id', 'objective_id'):
        if trained.get(key) != identity.get(key):
            raise ValueError(f'the student {key} {trained.get(key)!r} differs from the template '
                             f'{identity.get(key)!r}')
    if (trained.get('config_hashes') or {}).get('model') != (identity.get('config_hashes') or {}).get('model'):
        raise ValueError("the student's model config hash differs from the template's")
    chosen = summary.get('chosen') or {}
    name = Path(str(chosen.get('checkpoint', ''))).name
    if name not in CHECKPOINT_NAMES or type(chosen.get('step')) is not int:
        raise ValueError('the student summary names no chosen checkpoint-best or checkpoint-final')
    # Resolved in the run's own directory, so a run mounted elsewhere still evaluates its checkpoint.
    checkpoint = student / name
    if not checkpoint.is_dir():
        raise ValueError(f'the chosen checkpoint is missing: {checkpoint}')
    return {'training_output': str(student), 'training_summary_sha256': hashlib.sha256(data).hexdigest(),
            'chosen_step': chosen['step'], 'chosen_name': name, 'checkpoint': str(checkpoint),
            'tied_steps': chosen.get('tied_steps'), 'final_step': (summary.get('final') or {}).get('step'),
            'identity': trained}


def _default_dependencies() -> dict[str, Callable]:
    from scripts.neural.baseline_entry import _gpu_free_bytes, _release_gpu_cache, _tree_bytes
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.recipes import build_panel, load_recipe, summarize_panel_reports
    from spire_learn.neural.runner import create_runtime, evaluate_policy
    return dict(
        load_config=load_neural_config, create_runtime=create_runtime, evaluate=evaluate_policy,
        preflight=preflight_checkpoint, load_recipe=load_recipe, build_panel=build_panel,
        summarize=summarize_panel_reports, disk_free=lambda path: shutil.disk_usage(path).free,
        tree_bytes=_tree_bytes, gpu_free_bytes=_gpu_free_bytes, release_gpu_cache=_release_gpu_cache,
        sleep=time.sleep, measure_inference=measure_inference, provenance=_provenance,
    )


def _provenance() -> dict:
    from scripts.neural.distill_train_entry import _git_state, _gym_state
    return {'learner': _git_state(), 'gym': _gym_state()}


def _start_frame(config):
    """The first panel case's public start frame and the driver's schema (one reset, then closed)."""
    from spire_learn.neural.envs import make_driver
    case = config.cases.for_mode(config.mode)[0]
    driver = make_driver(config, config.cases)
    try:
        row, = driver.reset(seeds=[case.environment_seed], entry_ids=[case.entry_id])
        return row.frame, driver.schema
    finally:
        driver.close()


def measure_inference(config, checkpoint: Path, gate) -> dict:
    """Warmed batch-1 forward timing of ``checkpoint`` on CPU and, for a CUDA config, on the GPU."""
    import torch

    from spire_learn.neural.checkpoint import load_checkpoint
    from spire_learn.neural.model_families import build_family_policy, model_family, time_batch1_forward

    identity = config.accepted_identities[0]
    family = model_family(identity.model_id)
    frame, schema = _start_frame(config)
    loaded = load_checkpoint(Path(checkpoint), identity, purpose='evaluate', device='cpu')
    trim = bool(config.trim_padding)
    result = {'family': family.family_id, 'model_id': identity.model_id, 'frame': 'first panel case start',
              'passes': INFERENCE_PASSES, 'warmup': INFERENCE_WARMUP, 'trim_padding': trim}

    def timed(device: str) -> dict:
        policy = build_family_policy(identity.model_id, schema, config.model, seed=0, device=device,
                                     objective_id=identity.objective_id, model_id=identity.model_id,
                                     features_id=identity.features_id, trim_padding=trim)
        policy.load_state_dict(dict(loaded.model_tensors), strict=True)
        policy.eval()
        return time_batch1_forward(policy, family, frame, schema, device=device, trim_padding=trim,
                                   passes=INFERENCE_PASSES, warmup=INFERENCE_WARMUP, torch=torch)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # one intra-op thread, as labelling and full-run workers use
    try:
        result['cpu'] = {**timed('cpu'), 'intra_op_threads': 1}
    finally:
        torch.set_num_threads(threads)
    if str(config.device).startswith('cuda'):
        gate.wait(config, 'inference')
        try:
            result['gpu'] = {**timed(str(config.device)), 'name': torch.cuda.get_device_name(0)}
        finally:
            gate.release(config)
    else:
        result['gpu'] = {'skipped': 'the evaluation config device is cpu'}
    return result


def run_student_eval(root: Path, plan_path: Path, student: Path, output: Path, *, panel: str = 'selection',
                     dependencies: Mapping[str, Callable] | None = None,
                     clock: Callable[[], float] = time.monotonic) -> dict:
    """Evaluate the student's chosen checkpoint on one panel and time its inference; writes summary.json."""
    from scripts.neural.baseline_entry import _validate_runner_contract
    if panel not in ('selection', 'boss'):
        raise ValueError(f'unknown panel {panel!r}')
    deps = dict(_default_dependencies() if dependencies is None else dependencies)
    root, output = Path(root), Path(output)
    if not root.is_absolute() or not output.is_absolute():
        raise ValueError('root and output must be absolute')
    root = root.resolve(strict=True)
    plan_path = Path(plan_path)
    plan = load_plan(plan_path if plan_path.is_absolute() else root / plan_path)
    _validate_runner_contract(root, output, plan, require_environment=False)
    template = root / plan['template']
    identity = check_template(template, plan)
    record = load_student(student, identity)
    # Like the baselines: the runner may already hold its logs here; summary.json and panel/ must be new.
    output.mkdir(parents=True, exist_ok=True)
    started, cpu_started = clock(), time.process_time()
    budget = Budget(training=started, total=started + plan['wall_seconds']['total'])
    journal = Journal(output / 'progress.jsonl', clock, started, panel=panel if panel == 'boss' else None)
    gate = GpuGate(plan, deps, clock, journal, budget)
    inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ('plan_sha256', plan_path if plan_path.is_absolute() else root / plan_path),
        ('template_sha256', template), ('recipe_sha256', root / plan['recipe']))}
    summary: dict = {'schema': SUMMARY_SCHEMA, 'plan': plan, 'inputs': inputs, 'panel': panel, 'student': record,
                     'checkpoints': [{'label': LABEL, 'path': record['checkpoint'],
                                      'chosen_step': record['chosen_step'], 'chosen_name': record['chosen_name']}],
                     'evaluations': {}, 'result': 'incomplete'}

    def storage_stop() -> str | None:
        if deps['disk_free'](output) < MIN_FREE_BYTES:
            return 'disk_free_below_minimum'
        if deps['tree_bytes'](output) > plan['max_new_output_bytes']:
            return 'output_byte_cap'
        return None

    try:
        summary['provenance'] = deps['provenance']()
        journal.write('student_evaluation_started', student=record['training_output'],
                      checkpoint=record['checkpoint'], chosen_step=record['chosen_step'])
        try:
            evaluate_panel(summary, deps, plan, template, root / plan['recipe'], output, budget, gate, clock,
                           journal, storage_stop, labels=(LABEL,), panel_kind=panel)
        except Exception as error:
            # An evaluation fault is reported, never retried; partial case rows are kept.
            summary['evaluation_fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            journal.write('evaluation_fault', **summary['evaluation_fault'])
            evaluation = summary['evaluations'].get(LABEL, {})
            if evaluation.get('status') == 'running':
                evaluation['status'] = 'fault'
        evaluated = summary['evaluations'].get(LABEL, {})
        if evaluated.get('status') == 'complete':
            try:
                config = deps['load_config'](output / 'panel' / 'config-0.toml')
                summary['inference'] = deps['measure_inference'](config, Path(record['checkpoint']), gate)
            except Stop as stop:
                summary['inference_fault'] = {'type': 'Stop', 'message': stop.reason}
            except Exception as error:
                summary['inference_fault'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            if 'inference_fault' in summary:
                journal.write('inference_fault', **summary['inference_fault'])
        else:
            summary['inference_skipped'] = f"evaluation status {evaluated.get('status')!r}"
        summary['result'] = ('complete' if evaluated.get('status') == 'complete'
                             and 'evaluation_fault' not in summary and 'inference' in summary else 'incomplete')
    except BaseException as interrupt:
        summary['interrupted'] = type(interrupt).__name__
        summary['result'] = 'interrupted'
        raise
    finally:
        summary['gpu_wait_seconds'] = gate.waited
        summary['timing'] = {'wall_seconds': clock() - started, 'cpu_seconds': time.process_time() - cpu_started}
        with (output / 'summary.json').open('x') as stream:
            stream.write(json.dumps(json_safe(summary), indent=2, sort_keys=True) + '\n')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--student', required=True, type=Path, help='a finished distill_train_entry output')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--panel', choices=('selection', 'boss'), default='selection')
    arguments = parser.parse_args(argv)
    from scripts.neural.baseline_entry import _validate_runner_contract
    for name in ('root', 'output', 'student'):
        if not getattr(arguments, name).is_absolute():
            print(f'--{name} must be an absolute path', file=sys.stderr)
            return 2
    plan_path = arguments.plan if arguments.plan.is_absolute() else arguments.root / arguments.plan
    try:
        plan = load_plan(plan_path)
        _validate_runner_contract(arguments.root, arguments.output, plan, require_environment=True)
        load_student(arguments.student, check_template(arguments.root / plan['template'], plan))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    summary = run_student_eval(arguments.root, arguments.plan, arguments.student, arguments.output,
                               panel=arguments.panel)
    final = summary['evaluations'].get(LABEL, {})
    print(json.dumps({'result': summary['result'], 'status': final.get('status'),
                      'cases_evaluated': final.get('cases_evaluated'), 'panel': summary['panel']}, sort_keys=True))
    return 0 if summary['result'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
