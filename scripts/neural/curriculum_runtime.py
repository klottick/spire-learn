"""Lazy single-job producer for curriculum-campaign-v1.

The host owns admission, shared leases and scheduling. Every operation checks a
closed reviewed context before resolving a backend. This module owns no release,
campaign loop, score-based decision, process management, or retry.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
import json
import math
import os
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ''):
    _root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_root / 'src'), str(_root)]

from scripts.neural.curriculum_package import (CAMPAIGN_ID, SCHEMA as PACKAGE_SCHEMA,
    read_json, relative_file, verify_inventory, sha256)

CONTEXT_SCHEMA = 'spire-learn-curriculum-job-context-v1'
RECEIPT_SCHEMA = 'spire-learn-curriculum-job-receipt-v1'
EFFECT_KINDS = ('environment_attempts', 'sample_rows', 'optimizer_attempts',
                'reset_slots', 'fixed_random_choices')
COMMON_KEYS = frozenset(('schema', 'campaign_id', 'job_id', 'kind', 'runtime_class',
    'package_path', 'package_sha256', 'input_root', 'input_inventory',
    'source_inventory', 'source_sha256', 'release_sha256', 'claim_sha256',
    'parent', 'deadline', 'whole_deadline', 'reservation', 'output_root'))
KIND_KEYS = {'segment': frozenset(('arm', 'shard_index', 'prefix', 'max_updates',
    'max_environment_actions', 'predecessor')), 'qualification': frozenset(),
    'panel': frozenset(('case_ids', 'slot'))}
DEPENDENCY_KEYS = frozenset(('factory', 'train', 'evaluate', 'load_checkpoint',
    'equal', 'fresh_rng_states', 'qualification_api', 'threads'))
QUALIFICATION_HASHES = ('02643c5784a2c7d9bee3b7323b3583a53d49fdd93e419a6bcc5b8fcb06296f31',
                        '612172ae211ba5d3d7efd6d0799a6da6053565fdb0f72b1fc6a01e0d68dc7586')
_ADMITTED_CONTEXT = None


def _integer(value, label, maximum=None):
    if type(value) is not int or value < 0 or maximum is not None and value > maximum:
        raise ValueError(label + ' must be a bounded nonnegative integer')
    return value


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(label + ' must be finite and nonnegative')
    return float(value)


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(label + ' must be a SHA256')
    return value


def _json(value):
    from spire_learn.neural.scenario_support import json_value
    return json_value(value)


def _write(path, value):
    raw = json.dumps(_json(value), sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode() + b'\n'
    if len(raw) > 32 * 1024**2:
        raise ValueError('campaign receipt exceeds 32 MiB')
    with Path(path).open('xb') as stream:
        stream.write(raw)


def source_paths(root: Path) -> list[str]:
    """Minimum staged Python scope; host may bind additional reviewed files."""
    root = Path(root)
    paths = [str(p.relative_to(root)) for p in (root / 'src/spire_learn').rglob('*.py')]
    paths += ['scripts/neural/' + name + '.py' for name in
              ('curriculum_package', 'curriculum_runtime', 'sustained_runtime',
               'sustained_profile', 'gpu_runtime_proof_helpers', 'task9_study_controls',
               'task9_nvidia_cdi', 'task8_gpu_setup_helpers', 'sustained_controls',
               'sustained_package')]
    paths += ['tests/__init__.py', 'tests/neural/__init__.py', 'tests/neural/test_curriculum.py']
    return sorted(paths)


def _external_path(root, value, *, parent=False):
    """Only /learner paths or the dedicated three-file /inputs/parent mount."""
    if parent and value == '/inputs/parent':
        from scripts.neural.task9_study_controls import _guest_path
        return _guest_path(value)
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('output/parent path must be absolute')
    root = Path(root).resolve()
    path = Path(value)
    if not path.is_relative_to(root) or '..' in path.parts:
        raise ValueError('output/parent path escaped source root')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink runtime path')
    return path


def validate_context(root: Path, context: Mapping, *, kind=None, clock=time.monotonic):
    """Pure admission of the exact producer contract; backend resolution follows."""
    root = Path(root).resolve()
    if not isinstance(context, Mapping) or context.get('kind') not in KIND_KEYS:
        raise ValueError('context kind is missing or unsupported')
    selected_kind = context['kind']
    if (set(context) != COMMON_KEYS | KIND_KEYS[selected_kind]
            or context.get('schema') != CONTEXT_SCHEMA
            or context.get('campaign_id') != CAMPAIGN_ID
            or kind is not None and selected_kind != kind):
        raise ValueError('context fields/schema/identity differ')
    if not isinstance(context['job_id'], str) or re.fullmatch(r'curriculum-[a-z0-9-]{1,100}', context['job_id']) is None:
        raise ValueError('context job ID must use its own namespace')
    if context['runtime_class'] != ('cpu' if selected_kind == 'panel' else 'gpu'):
        raise ValueError('context runtime class differs')
    for key in ('package_sha256', 'source_sha256', 'release_sha256', 'claim_sha256'):
        _digest(context[key], key)
    deadline, whole = (_finite(context[k], k) for k in ('deadline', 'whole_deadline'))
    now = _finite(clock(), 'clock')
    cap = 120 if selected_kind == 'qualification' else 480
    if not now < deadline <= whole or deadline - now > cap or whole - now > 600:
        raise ValueError('context deadline bounds differ or exhausted')
    limits = context['reservation']
    if not isinstance(limits, Mapping) or set(limits) != set(EFFECT_KINDS):
        raise ValueError('context effect reservation fields differ')
    ceilings = (dict(environment_attempts=512, sample_rows=512, optimizer_attempts=8,
                     reset_slots=256, fixed_random_choices=0) if selected_kind == 'qualification'
                else dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
                          reset_slots=4112, fixed_random_choices=0) if selected_kind == 'segment'
                else dict(environment_attempts=64*256, sample_rows=0, optimizer_attempts=0,
                          reset_slots=64, fixed_random_choices=0))
    for key in EFFECT_KINDS:
        _integer(limits[key], key, ceilings[key])
    if selected_kind == 'qualification' and dict(limits) != ceilings:
        raise ValueError('context qualification requires the exact shared ceilings')
    verify_inventory(root, context['source_inventory'])
    if not set(source_paths(root)) <= set(context['source_inventory']):
        raise ValueError('context source inventory omits required producer sources')
    package_path = relative_file(root, context['package_path'])
    if sha256(package_path) != context['package_sha256']:
        raise ValueError('context package raw identity differs')
    package = read_json(package_path)
    if package.get('schema') != PACKAGE_SCHEMA or package.get('campaign_id') != CAMPAIGN_ID:
        raise ValueError('context package schema/identity differs')
    input_root = relative_file(root, context['input_root'])
    verify_inventory(input_root, context['input_inventory'])
    parent = context['parent']
    if not isinstance(parent, Mapping) or set(parent) != {'path', 'inventory'}:
        raise ValueError('context parent fields differ')
    from scripts.neural.sustained_runtime import checkpoint_inventory
    parent_path = _external_path(root, parent['path'], parent=True)
    if checkpoint_inventory(parent_path) != parent['inventory']:
        raise ValueError('context parent inventory differs')
    output = _external_path(root, context['output_root'])
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('context output root must be fresh, empty and owned')
    config = None
    from spire_learn.neural.config import load_neural_config
    if selected_kind == 'segment':
        arm = context['arm']
        if arm not in package['training']:
            raise ValueError('context training arm differs')
        index = _integer(context['shard_index'], 'shard index', len(package['training'][arm]) - 1)
        binding = package['training'][arm][index]
        expected_inputs = {name: package['inventory'][name] for name in binding['inputs']}
        if context['input_inventory'] != expected_inputs:
            raise ValueError('context selected training input inventory differs')
        config = load_neural_config(relative_file(input_root, binding['config_path']))
        from spire_learn.neural.stream_resume import stream_identity
        if (config.config_sha256 != binding['config_sha256'] or config.manifest_sha256 != binding['manifest_sha256']
                or stream_identity(config)['stable_training_sha256'] != binding['stable_training_sha256']
                or config.scenario_stream['shard_index'] != index
                or config.scenario_stream['arm'] != arm):
            raise ValueError('context loaded training identity differs')
        if config.output_root != output:
            raise ValueError('context actual config output differs')
        prefix = context['prefix']
        if not isinstance(prefix, Mapping):
            raise ValueError('context prefix must be a mapping')
        if index == 0:
            expected_parent = package['original_parent']
            if (prefix != {} or context['predecessor'] is not None
                    or parent['inventory'] != expected_parent['inventory']):
                raise ValueError('context warm start requires original B and fresh progress')
        else:
            if not isinstance(context['predecessor'], Mapping) or set(context['predecessor']) != {'path', 'sha256'}:
                raise ValueError('context resume needs exact predecessor receipt')
            predecessor_path = relative_file(root, context['predecessor']['path'])
            if sha256(predecessor_path) != context['predecessor']['sha256']:
                raise ValueError('context predecessor receipt changed')
            previous = read_json(predecessor_path, maximum=32*1024**2)
            previous_binding = package['training'][arm][index - 1]
            if (previous.get('schema') != RECEIPT_SCHEMA or previous.get('status') != 'complete'
                    or previous.get('campaign_id') != context['campaign_id']
                    or previous.get('source_sha256') != context['source_sha256']
                    or previous.get('package_sha256') != context['package_sha256']
                    or previous.get('kind') != 'segment' or previous.get('arm') != arm
                    or previous.get('shard_index') != index - 1
                    or previous.get('config_sha256') != previous_binding['config_sha256']
                    or previous.get('manifest_sha256') != previous_binding['manifest_sha256']
                    or previous.get('next_shard', {}).get('shard_index') != index
                    or previous.get('report', {}).get('status') != 'complete'
                    or previous.get('report', {}).get('fault')
                    or set(previous.get('effects', {})) != set(EFFECT_KINDS)
                    or any(row.get('uncertain') != 0 for row in previous.get('effects', {}).values())
                    or previous.get('checkpoint', {}).get('inventory') != parent['inventory']
                    or previous.get('checkpoint', {}).get('counters') != prefix):
                raise ValueError('context predecessor is not healthy immediate parent')
        updates = _integer(context['max_updates'], 'max updates')
        actions = _integer(context['max_environment_actions'], 'max actions', 1000000)
        if (not prefix.get('update_count', 0) < updates <= prefix.get('update_count', 0) + 8
                or not prefix.get('attempted_environment_slots', 0) < actions
                <= prefix.get('attempted_environment_slots', 0) + limits['environment_attempts']):
            raise ValueError('context cumulative operation limits differ')
    elif selected_kind == 'qualification':
        configs = [load_neural_config(input_root / f'qualification-{i}.toml') for i in (0, 1)]
        if [c.config_sha256 for c in configs] != list(QUALIFICATION_HASHES):
            raise ValueError('context qualification config identity differs')
        if any(c.output_root != output for c in configs):
            raise ValueError('context qualification output differs')
        original = package['original_parent']
        if parent_path != root / original['checkpoint_directory'] or parent['inventory'] != original['inventory']:
            raise ValueError('context qualification requires exact original-B location')
        parents = read_json(input_root / 'pilot-parents.json')
        if parents['standalone_b'] != original:
            raise ValueError('context qualification held parent changed')
        required = {'qualification-0.toml', 'qualification-1.toml', 'pilot-parents.json',
                    'recipe.json', 'support.json', 'shard-0.json', 'shard-1.json'}
        if not required <= set(context['input_inventory']):
            raise ValueError('context qualification inventory incomplete')
    else:
        if context['slot'] not in package['evaluation_slots']:
            raise ValueError('context panel slot differs')
        rows = package['panel']['cases']
        ids = [row['case_id'] for row in rows]
        chosen = context['case_ids']
        if (not isinstance(chosen, list) or not chosen or len(chosen) != len(set(chosen))
                or any(value not in ids for value in chosen)
                or chosen != [value for value in ids if value in chosen]):
            raise ValueError('context panel cases must be unique and in fixed order')
        names = {package['panel']['path']} | {name for row in rows for name in row['inputs']}
        if context['input_inventory'] != {name: package['inventory'][name] for name in names}:
            raise ValueError('context complete panel inventory differs')
        panel = read_json(input_root / package['panel']['path'])
        from spire_learn.neural.scenario_support import content_sha256
        if (len(panel['manifest']['cases']) != 64 or panel['manifest_sha256'] != content_sha256(panel['manifest'])
                or set(panel['annotations']) != set(ids)
                or [c['case_id'] for c in panel['manifest']['cases']] != ids):
            raise ValueError('context complete panel structure differs')
        for row, case in zip(rows, panel['manifest']['cases']):
            actual = load_neural_config(input_root / row['config_path'])
            if (actual.config_sha256 != row['config_sha256'] or actual.manifest_sha256 != row['manifest_sha256']
                    or actual.output_root != output or actual.device != 'cpu' or actual.scenario_stream is not None
                    or (actual.rollout_rows, actual.rollout_steps, actual.operation, actual.split) != (1, 1, 'evaluate', 'selection')
                    or len(actual.cases.cases) != 1 or actual.cases.cases[0].case_id != case['case_id']
                    or actual.cases.cases[0].environment_seed != case['environment_seed']):
                raise ValueError('context CPU single-case config differs')
    return package, input_root, parent_path, output, config


def _equal(left, right):
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _fresh_rng(config):
    import random
    import torch
    from spire_learn.neural.device import plan_runtime_device
    plan = plan_runtime_device(config)
    policy, minibatch = torch.Generator(device=plan.policy_generator_device), torch.Generator(device=plan.shuffle_generator_device)
    policy.manual_seed(config.seed_streams.policy)
    minibatch.manual_seed(config.seed_streams.optimizer)
    return dict(policy=policy.get_state(), minibatch=minibatch.get_state(),
        curriculum=random.Random(config.seed_streams.curriculum).getstate(),
        controller=random.Random(config.seed_streams.noncombat).getstate())


def _threads():
    import torch
    if str(torch.__version__) != '2.14.1+cu130' or torch.version.cuda != '13.0':
        raise RuntimeError('actual executing Torch/CUDA versions differ')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (4, 1):
        raise RuntimeError('actual qualification process threads differ')
    return {'intraop': 4, 'interop': 1}


def _default_gate(context, dependencies):
    if dependencies is None and _ADMITTED_CONTEXT is not context:
        raise RuntimeError('real runtime requires verified guest entry admission')
    if dependencies is not None and _ADMITTED_CONTEXT is not context:
        required = {'segment': {'factory', 'load_checkpoint', 'equal', 'fresh_rng_states'},
                    'qualification': {'qualification_api', 'threads'}, 'panel': {'factory'}}[context['kind']]
        if not isinstance(dependencies, Mapping) or not required <= set(dependencies):
            raise TypeError('unadmitted execution requires explicit fake runtime boundaries')


def _factory(config, **kwargs):
    from spire_learn.neural.runner import create_runtime
    from spire_learn.neural.device import plan_runtime_device, require_runtime_execution_gate
    require_runtime_execution_gate(plan_runtime_device(config))
    _threads()
    runtime = create_runtime(config, **kwargs)
    try:
        _threads()
    except BaseException as error:
        from spire_learn.neural.runner import _close_preserving
        final = _close_preserving(error, runtime, 'tensor process verification')
        if final is not error:
            raise final
        raise
    return runtime


def _dependencies(supplied):
    supplied = dict(supplied or {})
    if set(supplied) - DEPENDENCY_KEYS or any(not callable(v) for v in supplied.values()):
        raise TypeError('dependencies must be known plain runtime callables')
    from spire_learn.neural.runner import train, evaluate_policy
    from spire_learn.neural.checkpoint import load_checkpoint
    return dict(factory=supplied.get('factory', _factory), train=supplied.get('train', train),
        evaluate=supplied.get('evaluate', evaluate_policy), load_checkpoint=supplied.get('load_checkpoint', load_checkpoint),
        equal=supplied.get('equal', _equal), fresh_rng_states=supplied.get('fresh_rng_states', _fresh_rng),
        qualification_api=supplied.get('qualification_api'), threads=supplied.get('threads', _threads))


@contextmanager
def _owned(runtime):
    """Runner may close; setup/assertion/serialization paths close exactly once."""
    from spire_learn.neural.runner import _close_preserving
    original, closed, primary = runtime.close, False, None
    def close():
        nonlocal closed
        if not closed:
            closed = True
            return original()
    runtime.close = close
    try:
        yield runtime
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            final = _close_preserving(primary, runtime, 'curriculum runtime')
            if final is not None and final is not primary:
                raise final
        finally:
            runtime.close = original


def _runtime_state(runtime, loaded, config, deps, *, resume, prefix):
    equal = deps['equal']
    def require(value, label):
        if not value:
            raise ValueError('factory state differs: ' + label)
    require(getattr(runtime, 'config', None) is config, 'actual config')
    require(equal(dict(runtime.policy.state_dict()), dict(loaded.model_tensors)), 'model')
    state = runtime.optimizer.state_dict()
    require(equal(state, loaded.optimizer_state) if resume else state.get('state') == {}, 'Adam')
    require(runtime.policy._behavior_policy_version == (loaded.metadata['training_state']['behavior_policy_version'] if resume else 0), 'behavior version')
    require(_json(runtime.recovered_progress) == _json(prefix), 'progress')
    actual_rng = dict(policy=runtime.policy_generator.get_state(), minibatch=runtime.shuffle_generator.get_state(),
        curriculum=runtime.curriculum_rng.getstate(), controller=runtime.controller_rng.getstate())
    expected_rng = loaded.rng_state if resume else deps['fresh_rng_states'](config)
    for domain in ('policy', 'minibatch', 'curriculum', 'controller'):
        require(equal(actual_rng[domain], expected_rng[domain]), domain + ' RNG')
    if not resume:
        recovery = runtime.collector.recovery_state
        require((recovery.next_seed_ordinal, recovery.next_decision_sequence,
                 list(recovery.next_episode_ordinals)) == (0, 0, [0] * config.rollout_rows), 'fresh collector recovery')
        expected_sampler = None
        if config.scenario_stream['arm'] == 'mixed':
            from spire_learn.neural.curriculum import CurriculumSampler
            from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard
            recipe = load_scenario_recipe(Path(config.scenario_stream['resolved_recipe_path']))
            shard = generate_scenario_shard(recipe, split='train', arm='mixed', shard_index=0)
            expected_sampler = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                                  categories=recipe.category_keys).snapshot()
        require(_json(runtime.collector.sampler_state) == _json(expected_sampler), 'fresh sampler')
    if resume:
        recovery = runtime.collector.recovery_state
        expected = loaded.metadata['run_metadata']['collector_recovery']
        require((recovery.next_seed_ordinal, recovery.next_decision_sequence, list(recovery.next_episode_ordinals)) ==
                (expected['next_seed_ordinal'], expected['next_decision_sequence'], list(expected['next_episode_ordinals'])), 'collector recovery')
        from spire_learn.neural.stream_resume import validate_stream_resume
        transition = validate_stream_resume(loaded.metadata['run_metadata'], config)
        expected_sampler = transition['sampler_state'] if config.scenario_stream['arm'] == 'mixed' else None
        require(_json(runtime.collector.sampler_state) == _json(expected_sampler), 'sampler')


@contextmanager
def _panel_effects(runtime, ledger):
    """Frozen deterministic evaluation: no optimizer, sample or random effects."""
    from scripts.neural.sustained_profile import _patch
    if getattr(runtime, 'optimizer', None) is not None:
        raise ValueError('panel runtime must have no optimizer')
    if (runtime.policy_kind, runtime.action_selection) != ('checkpoint', 'deterministic'):
        raise ValueError('panel runtime policy is not deterministic checkpoint')
    with ExitStack() as stack:
        driver = runtime.collector.driver
        def reset(*args, **kwargs):
            count = sum(kwargs['reset_mask']) if kwargs.get('reset_mask') is not None else len(kwargs['seeds'])
            return ledger.call('reset_slots', count, 'panel-reset', lambda: original_reset(*args, **kwargs))
        original_reset = _patch(stack, driver, 'reset', reset)
        def step(actions):
            return ledger.call('environment_attempts', len(actions), 'panel-step', lambda: original_step(actions))
        original_step = _patch(stack, driver, 'step', step)
        def act(frames, **kwargs):
            ledger.check()
            return original_act(frames, **kwargs)
        original_act = _patch(stack, runtime.collector.policy, 'act', act)
        def deny(*args, **kwargs):
            raise RuntimeError('panel forbids sampling/optimizer/random effects')
        _patch(stack, runtime.collector.controller_rng, 'choice', deny)
        torch = sys.modules.get('torch')
        if torch is not None:
            _patch(stack, torch, 'multinomial', deny)
            _patch(stack, torch.optim.Adam, 'step', deny)
        yield


@contextmanager
def _ledger(context, output, clock):
    from scripts.neural.sustained_runtime import EffectLedger, Journal
    control = output / 'campaign-control'
    control.mkdir()
    journal = Journal(control)
    class CampaignLedger(EffectLedger):
        """Reuse accounting while preserving late process controls at settle."""
        def __init__(self):
            from types import SimpleNamespace
            self.settlement_control = None
            def settle(row):
                try:
                    journal.settle(row)
                except BaseException as error:
                    if not isinstance(error, Exception) and self.settlement_control is None:
                        self.settlement_control = error
                    raise
            super().__init__(context['job_id'], SimpleNamespace(reserve=journal.reserve, settle=settle),
                             deadline=context['deadline'], clock=clock, limits=context['reservation'])
        @contextmanager
        def reservation(self, *args, **kwargs):
            from spire_learn.neural.runner import _promote_finalization_control
            primary = None
            try:
                with super().reservation(*args, **kwargs) as lease:
                    yield lease
            except BaseException as error:
                primary = error
                raise
            finally:
                if self.settlement_control is not None and self.settlement_control is not primary:
                    final = _promote_finalization_control(primary, self.settlement_control, 'effect settlement')
                    if final is not primary:
                        raise final
    ledger = CampaignLedger()
    primary = None
    try:
        yield ledger, journal
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            journal.flush()
            _write(control / 'attempts.json', {'effects': ledger.snapshot(), 'categories': ledger.categories,
                'journal': journal.parts, 'unresolved_effect_ids': sorted(journal.open),
                'qualification_attempts': getattr(ledger, 'qualification_attempts', None),
                'error': None if primary is None else {'type': type(primary).__name__, 'message': str(primary)[:1024]}})
        except BaseException as error:
            if primary is None:
                raise
            if not isinstance(error, Exception):
                from spire_learn.neural.runner import _promote_finalization_control
                final = _promote_finalization_control(primary, error, 'ledger retention')
                if final is not primary:
                    raise final
            primary.add_note(f'ledger retention also failed: {error}')


def _receipt(context, started, clock, **values):
    from scripts.neural.sustained_runtime import process_evidence
    elapsed = _finite(clock() - started, 'job elapsed')
    if clock() >= context['whole_deadline']:
        raise TimeoutError('whole deadline expired after lifecycle closure')
    return dict(schema=RECEIPT_SCHEMA, campaign_id=context['campaign_id'], job_id=context['job_id'],
        kind=context['kind'], runtime_class=context['runtime_class'], status='complete',
        package_sha256=context['package_sha256'], source_sha256=context['source_sha256'],
        release_sha256=context['release_sha256'], claim_sha256=context['claim_sha256'],
        parent=_json(context['parent']), input_inventory=_json(context['input_inventory']),
        elapsed_seconds=elapsed, deadline=context['deadline'], whole_deadline=context['whole_deadline'],
        process=process_evidence(), **values)


def _finished(receipt, context, started, clock):
    # Ledger serialization and closure belong to the enclosing invocation, while
    # checkpoint saved elapsed remains the exact earlier producer boundary.
    receipt['elapsed_seconds'] = _finite(clock() - started, 'enclosing job elapsed')
    if clock() >= context['whole_deadline']:
        raise TimeoutError('whole deadline expired during final ledger retention')
    return receipt


def run_segment(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping:
    """One real accepted stream segment, with cumulative high-water controls."""
    started = clock()
    package, input_root, parent, output, config = validate_context(root, context, kind='segment', clock=clock)
    _default_gate(context, dependencies)
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.runner import _checkpoint_recovery, RuntimeOperationLimits
    from spire_learn.neural.stream_runner import run_stream_segment
    from scripts.neural.sustained_runtime import effects, checkpoint_inventory, report_files, reconcile_segment
    resume = context['shard_index'] > 0
    purpose = 'resume' if resume else 'warm_start'
    preflight = preflight_checkpoint(parent, config.accepted_identities[0], purpose=purpose)
    prefix = _checkpoint_recovery(preflight, config)[1] if resume else {}
    if _json(prefix) != context['prefix']:
        raise ValueError('incoming preflight checkpoint prefix differs')
    if resume and (preflight.metadata['run_metadata']['scenario_stream']['current']['shard_index']
                   != context['shard_index'] - 1 or any(prefix[key] for key in
                   ('uncertain_environment_slots', 'uncertain_reset_slots', 'optimizer_uncertain_steps'))):
        raise ValueError('incoming checkpoint is not the healthy preceding shard')
    deps = _dependencies(dependencies)
    loaded = deps['load_checkpoint'](parent, config.accepted_identities[0], purpose=purpose, device='cpu')
    with _ledger(context, output, clock) as (ledger, journal):
        ledger.check()
        def factory(actual, **kwargs):
            if actual is not config or kwargs != dict(purpose=purpose, checkpoint=parent,
                    policy_kind='checkpoint', action_selection='seeded-sampling'):
                raise ValueError('stream factory invocation differs')
            runtime = deps['factory'](actual, **kwargs)
            try:
                _runtime_state(runtime, loaded, config, deps, resume=resume, prefix=prefix)
            except BaseException as error:
                from spire_learn.neural.runner import _close_preserving
                final = _close_preserving(error, runtime, 'curriculum factory')
                if final is not error:
                    raise final
                raise
            return runtime
        def train(actual, *, runtime, clock):
            with _owned(runtime), effects(runtime, config, ledger, training=True):
                return deps['train'](actual, runtime=runtime, clock=clock)
        result = run_stream_segment(config, purpose=purpose, checkpoint=parent,
            operation_limits=RuntimeOperationLimits(max_updates=context['max_updates'],
                max_environment_actions=context['max_environment_actions'], monotonic_deadline=context['deadline']),
            dependencies=dict(create_runtime=factory, train=train, clock=clock))
        if ledger.first_error is not None:
            raise ledger.first_error
        if result['checkpoint'] is None:
            raise RuntimeError('stream did not publish a healthy checkpoint')
        checkpoint = result['checkpoint']
        saved = preflight_checkpoint(checkpoint, config.accepted_identities[0], purpose='resume')
        # The accepted reconciler checks the physical current parent. Wire parent
        # aliases are never written into learner metadata or dataclass settings.
        physical_context = {**context, 'parent': {**context['parent'], 'path': str(parent)}}
        saved_prefix, delta = reconcile_segment(config, physical_context, result['report'].summary,
            saved.metadata, ledger.snapshot(), elapsed=clock() - started)
        if saved_prefix != result['saved_prefix']:
            raise ValueError('stream/reconciled saved prefix differs')
        receipt = _receipt(context, started, clock, arm=context['arm'], shard_index=context['shard_index'],
            prefix=_json(prefix), predecessor=_json(context['predecessor']), config_sha256=config.config_sha256,
            manifest_sha256=config.manifest_sha256, report=_json(result['report'].summary),
            report_files=report_files(result['report'], output), next_shard=_json(result['next_shard']),
            checkpoint={'path': str(checkpoint), 'inventory': checkpoint_inventory(checkpoint), 'counters': saved_prefix},
            delta=delta, effects=ledger.snapshot(), effect_categories=_json(ledger.categories))
    return _finished(receipt, context, started, clock)


def run_qualification(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping:
    """Invoke the exact held TestCase method once, narrowing its shared clock."""
    started = clock()
    _, input_root, _, output, _ = validate_context(root, context, kind='qualification', clock=clock)
    _default_gate(context, dependencies)
    deps = _dependencies(dependencies)
    from tests.neural import test_curriculum as held
    from unittest.mock import patch
    from scripts.neural.sustained_runtime import checkpoint_inventory
    original_caps, original_api = held._Caps, held._runtime_api
    observed, thread_evidence, segments = [], [], []
    original_exercise = held._exercise
    def exercise(*args, **kwargs):
        result = original_exercise(*args, **kwargs)
        segments.extend(result)
        return result
    with _ledger(context, output, clock) as (ledger, journal):
        leases = []
        class Caps(original_caps):
            def __init__(self):
                super().__init__(clock=clock)
                self.deadline = min(self.deadline, context['deadline'])
                observed.append(self)
                ledger.qualification_attempts = self.counts
            def charge(self, kind, count):
                ledger.check()
                # The held cap retains its exact named, pre-forward attempts.
                super().charge(kind, count)
                target = dict(actions='environment_attempts', samples='sample_rows',
                              adam='optimizer_attempts', resets='reset_slots')[kind]
                reservation = ledger.reservation(target, count, 'qualification-' + kind)
                lease = reservation.__enter__()
                lease['attempted'] = count
                # Remain unresolved until the actual held harness has completed
                # all effect/state/lifecycle assertions. A failed harness retains
                # every unresolved attempt conservatively as uncertain.
                lease['uncertain'] = count
                leases.append((reservation, lease, count))
        @contextmanager
        def api(caps):
            evidence = deps['threads']()
            if evidence != {'intraop': 4, 'interop': 1}:
                raise RuntimeError('actual qualification thread settings differ')
            thread_evidence.append(evidence)
            boundary = deps['qualification_api'] or original_api
            with boundary(caps) as value:
                yield value
            if deps['threads']() != evidence:
                raise RuntimeError('qualification process threads changed')
        case = held.CurriculumIntegrationTests('test_public_catalog_tensor_resume_and_tiny_healthy_segments')
        # The exact method remains unchanged and is invoked once. The class gate
        # remains untouched for discovery and every other process.
        success, primary = False, None
        try:
            with patch.object(held, '_Caps', Caps), patch.object(held, '_runtime_api', api), \
                 patch.object(held, '_exercise', exercise), \
                 patch.dict(os.environ, {'SPIRE_LEARN_CURRICULUM_INPUT_ROOT': str(input_root)}):
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    case.test_public_catalog_tensor_resume_and_tiny_healthy_segments()
                finally:
                    os.chdir(previous)
            success = True
        except BaseException as error:
            primary = error
            raise
        finally:
            final = primary
            for reservation, lease, count in leases:
                if success:
                    lease.update(delivered=count, uncertain=0)
                try:
                    reservation.__exit__(None, None, None)
                except BaseException as error:
                    if final is None:
                        final = error
                    elif not isinstance(error, Exception):
                        from spire_learn.neural.runner import _promote_finalization_control
                        final = _promote_finalization_control(final, error, 'qualification settlement')
                    else:
                        final.add_note(f'qualification settlement also failed: {error}')
            if final is not None and final is not primary:
                raise final
        if len(observed) != 1 or len(thread_evidence) != 1:
            raise RuntimeError('qualification did not execute exactly one held method')
        attempts = dict(observed[0].counts)
        receipt = _receipt(context, started, clock, qualification_attempts=attempts,
            qualification_method='tests.neural.test_curriculum.CurriculumIntegrationTests.test_public_catalog_tensor_resume_and_tiny_healthy_segments',
            threads=thread_evidence[0], effects=ledger.snapshot(),
            effect_delivery_basis='successful exact held harness', healthy_harness=True,
            qualification_segments=[{'report': _json(segment['report'].summary),
                'checkpoint': {'path': str(segment['checkpoint']),
                    'inventory': checkpoint_inventory(segment['checkpoint'])},
                'saved_prefix': _json(segment['saved_prefix']), 'next_shard': _json(segment['next_shard'])}
                for segment in segments])
    return _finished(receipt, context, started, clock)


def run_panel(root: Path, context: Mapping, *, dependencies=None, clock=time.monotonic) -> Mapping:
    """One bounded CPU panel batch; missing rows remain missing in real summary."""
    started = clock()
    package, input_root, parent, output, _ = validate_context(root, context, kind='panel', clock=clock)
    _default_gate(context, dependencies)
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.runner import RuntimeOperationLimits, NeuralReport
    from spire_learn.neural.stream_runner import summarize_development_reports
    from scripts.neural.sustained_runtime import report_files
    deps = _dependencies(dependencies)
    panel = read_json(input_root / package['panel']['path'])
    reports, retained = {}, []
    with _ledger(context, output, clock) as (ledger, journal):
        for row in package['panel']['cases']:
            if row['case_id'] not in context['case_ids']:
                continue
            if clock() >= context['deadline']:
                break
            config = load_neural_config(input_root / row['config_path'])
            preflight_checkpoint(parent, config.accepted_identities[0], purpose='evaluate')
            ledger.check()
            runtime = deps['factory'](config, purpose='evaluate', checkpoint=parent,
                policy_kind='checkpoint', action_selection='deterministic')
            with _owned(runtime), _panel_effects(runtime, ledger):
                runtime.operation_limits = RuntimeOperationLimits(max_environment_actions=256,
                    monotonic_deadline=min(context['deadline'], clock() + 30))
                report = deps['evaluate'](config, runtime=runtime, clock=clock, allow_final=False)
                if not isinstance(report, NeuralReport):
                    raise TypeError('CPU evaluator must return actual NeuralReport')
                # Serialize/assert inside ownership, including setup-only reports.
                summary = _json(report.summary)
                files = report_files(report, output) if report.run_dir is not None else None
            reports[row['case_id']] = report
            retained.append({'case_id': row['case_id'], 'config_sha256': config.config_sha256,
                'manifest_sha256': config.manifest_sha256, 'environment_seed': row['environment_seed'],
                'exit_code': report.exit_code, 'report': summary, 'report_files': files})
            if ledger.first_error is not None:
                raise ledger.first_error
            if summary.get('fault') or summary.get('status') not in ('complete', 'incomplete'):
                break
        aggregate = summarize_development_reports(panel, reports)
        receipt = _receipt(context, started, clock, slot=context['slot'], reports=retained,
            summary=_json(aggregate), missing_case_ids=[c['case_id'] for c in package['panel']['cases'] if c['case_id'] not in reports],
            effects=ledger.snapshot(), effect_categories=_json(ledger.categories))
        if aggregate['status'] == 'operational_failure':
            receipt['status'] = 'operational_failure'
    return _finished(receipt, context, started, clock)


def _guest_entry(root, context_path, *, dependencies=None, clock=time.monotonic, admission=None):
    """Actual entry: host verifier checks source/runtime/CDI before lazy imports."""
    if admission is None:
        from scripts.neural.curriculum_controls import verify_guest_context
        admission = verify_guest_context
    context = admission(Path(root), Path(context_path), clock=clock)
    if not isinstance(context, Mapping):
        raise TypeError('guest admission must return the exact context mapping')
    output = _external_path(root, context['output_root'])
    global _ADMITTED_CONTEXT
    if _ADMITTED_CONTEXT is not None:
        raise RuntimeError('guest entry already active')
    _ADMITTED_CONTEXT = context
    try:
        result = {'qualification': run_qualification, 'segment': run_segment, 'panel': run_panel}[context['kind']](
            Path(root), context, dependencies=dependencies, clock=clock)
        # Materialize bounded JSON before the last whole-deadline check.
        result = _json(result)
        if clock() >= context['whole_deadline']:
            raise TimeoutError('whole deadline expired before receipt publication')
        path = output / 'job-receipt.json'
        _write(path, result)
        if clock() >= context['whole_deadline']:
            raise TimeoutError('whole deadline expired during receipt publication')
        if result['status'] != 'complete':
            raise RuntimeError('guest producer retained operational_failure receipt')
        return {'status': 'complete', 'job_id': context['job_id'],
                'receipt_path': str(path), 'receipt_sha256': sha256(path)}
    except BaseException as error:
        try:
            _write(output / 'guest-failure.json', {'job_id': context['job_id'],
                'error': {'type': type(error).__name__, 'message': str(error)[:1024]}})
        except BaseException as retention:
            if not isinstance(retention, Exception):
                from spire_learn.neural.runner import _promote_finalization_control
                final = _promote_finalization_control(error, retention, 'guest failure retention')
                if final is not error:
                    raise final
            error.add_note(f'guest failure retention also failed: {retention}')
        raise
    finally:
        _ADMITTED_CONTEXT = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('_guest',))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    args = parser.parse_args(argv)
    from scripts.neural.sustained_controls import signal_scope
    with signal_scope():
        result = _guest_entry(args.root, args.context)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
