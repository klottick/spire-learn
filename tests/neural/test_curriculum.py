"""Held native/tensor curriculum qualification; no optional import at discovery.

This single class is UNEXECUTED. Its flag is an opt-in, not runtime authority.
The protected host protocol is a proposal in curriculum-qualification-runbook.md.
Pure tests call the same _exercise with explicit stdlib runtime substitutes.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

GATE = 'SPIRE_LEARN_RUN_NEURAL_CURRICULUM_INTEGRATION'


class _Caps:
    """One attempt ledger shared by construction, parity probes and both segments."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + 120.0
        self.counts = dict(actions=0, samples=0, adam=0, resets=0)

    def charge(self, kind, count):
        if self.clock() >= self.deadline:
            raise RuntimeError('qualification shared soft deadline exhausted')
        cap = dict(actions=512, samples=512, adam=8, resets=256)[kind]
        if self.counts[kind] + count > cap:
            raise RuntimeError(f'qualification {kind} cap exhausted')
        # Charge before forwarding: exceptions never refund attempts.
        self.counts[kind] += count


class _BoundedDriver:
    def __init__(self, driver, caps):
        self.driver, self.caps = driver, caps

    def __getattr__(self, name):
        return getattr(self.driver, name)

    def reset(self, *, seeds, entry_ids, reset_mask=None):
        selected = self.batch_size if reset_mask is None else sum(reset_mask)
        self.caps.charge('resets', selected)
        return self.driver.reset(seeds=seeds, entry_ids=entry_ids, reset_mask=reset_mask)

    def step(self, actions):
        self.caps.charge('actions', len(actions))
        return self.driver.step(actions)


def _fresh_rng_states(config, generator_factory):
    """Build isolated reference streams using the producer's device assignment."""
    from spire_learn.neural.device import plan_runtime_device
    plan = plan_runtime_device(config)
    policy = generator_factory(device=plan.policy_generator_device)
    policy.manual_seed(config.seed_streams.policy)
    minibatch = generator_factory(device=plan.shuffle_generator_device)
    minibatch.manual_seed(config.seed_streams.optimizer)
    return dict(policy=policy.get_state(), minibatch=minibatch.get_state(),
        curriculum=random.Random(config.seed_streams.curriculum).getstate(),
        controller=random.Random(config.seed_streams.noncombat).getstate())


@contextmanager
def _runtime_api(caps):
    # This boundary is entered only inside the gated method, never setUpModule.
    import torch
    from spire_learn.neural import envs, runner
    from spire_learn.neural.checkpoint import load_checkpoint

    original_driver, multinomial, adam_step = envs.make_driver, torch.multinomial, torch.optim.Adam.step

    def make_driver(config, cases):
        return _BoundedDriver(original_driver(config, cases), caps)

    def sample(input, num_samples, *args, **kwargs):
        caps.charge('samples', 1 if input.ndim == 1 else input.shape[0])
        return multinomial(input, num_samples, *args, **kwargs)

    def step(optimizer, *args, **kwargs):
        caps.charge('adam', 1)
        return adam_step(optimizer, *args, **kwargs)

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    def finite_changed(before, after):
        return all(torch.isfinite(t).all().item() for t in after.values()) and not equal(before, after)

    with patch.object(envs, 'make_driver', side_effect=make_driver), \
         patch.object(torch, 'multinomial', side_effect=sample), \
         patch.object(torch.optim.Adam, 'step', new=step):
        yield SimpleNamespace(make_driver=make_driver, create_runtime=runner.create_runtime,
            train=runner.train, load_checkpoint=load_checkpoint, equal=equal,
            finite_changed=finite_changed, clock=caps.clock,
            fresh_rng_states=lambda config: _fresh_rng_states(config, torch.Generator))


def _check_restored(test, runtime, loaded, api):
    """Compare against saved state, without elapsed-time recomputation or RNG draws."""
    from spire_learn.neural.scenario_support import json_value
    state = loaded.metadata['training_state']
    metadata = loaded.metadata['run_metadata']
    test.assertTrue(api.equal(dict(runtime.policy.state_dict()), dict(loaded.model_tensors)))
    test.assertTrue(api.equal(runtime.optimizer.state_dict(), loaded.optimizer_state))
    actual_rng = dict(policy=runtime.policy_generator.get_state(),
        minibatch=runtime.shuffle_generator.get_state(),
        curriculum=runtime.curriculum_rng.getstate(), controller=runtime.controller_rng.getstate())
    test.assertTrue(api.equal(actual_rng, dict(loaded.rng_state)))
    test.assertEqual(runtime.policy._behavior_policy_version, state['behavior_policy_version'])
    test.assertEqual(runtime.recovered_progress, state['counters'])
    recovery = runtime.collector.recovery_state
    test.assertEqual(recovery.next_seed_ordinal, state['next_environment_seed_ordinal'])
    test.assertEqual(recovery.next_decision_sequence, state['next_decision_sequence'])
    test.assertEqual(list(recovery.next_episode_ordinals), list(metadata['collector_recovery']['next_episode_ordinals']))
    test.assertEqual(runtime.collector.sampler_state, json_value(metadata['scenario_stream']['sampler_state']))


def _exercise(test, configs, parent, api, caps):
    from spire_learn.neural.runner import RuntimeOperationLimits, _close_preserving
    from spire_learn.neural.stream_runner import run_stream_segment
    first, second = configs
    test.assertEqual([c.scenario_stream['shard_index'] for c in configs], [0, 1])
    test.assertNotEqual(first.config_sha256, second.config_sha256)
    test.assertNotEqual(first.manifest_sha256, second.manifest_sha256)
    # Public normalization happens in actual make_driver before it returns.
    # Construction closes without resets or steps; this is not executed coverage.
    for config in configs:
        driver = api.make_driver(config, config.cases)
        driver.close()
    old = api.load_checkpoint(parent, first.accepted_identities[0], purpose='warm_start', device='cpu')

    def warm_factory(config, *, purpose, checkpoint, policy_kind, action_selection):
        test.assertEqual((purpose, checkpoint, policy_kind, action_selection),
                         ('warm_start', parent, 'checkpoint', 'seeded-sampling'))
        runtime = api.create_runtime(config, purpose=purpose, checkpoint=checkpoint,
            policy_kind=policy_kind, action_selection=action_selection)
        try:
            test.assertTrue(api.equal(dict(runtime.policy.state_dict()), dict(old.model_tensors)))
            test.assertEqual(runtime.optimizer.state_dict()['state'], {})
            test.assertEqual(runtime.policy._behavior_policy_version, 0)
            test.assertEqual(runtime.recovered_progress, {})
            expected_rng = api.fresh_rng_states(config)
            actual_rng = dict(policy=runtime.policy_generator.get_state(),
                minibatch=runtime.shuffle_generator.get_state(),
                curriculum=runtime.curriculum_rng.getstate(), controller=runtime.controller_rng.getstate())
            for domain in ('policy', 'minibatch', 'curriculum', 'controller'):
                test.assertTrue(api.equal(actual_rng[domain], expected_rng[domain]),
                                f'fresh warm-start {domain} RNG does not match config seed')
        except BaseException as error:
            primary = _close_preserving(error, runtime, 'qualification warm-start runtime')
            if primary is not error:
                raise primary
            raise
        return runtime

    result = run_stream_segment(first, purpose='warm_start', checkpoint=parent,
        operation_limits=RuntimeOperationLimits(max_updates=1, monotonic_deadline=caps.deadline,
                                               max_environment_actions=512),
        dependencies=dict(create_runtime=warm_factory, train=api.train, clock=api.clock))
    test.assertEqual(result['report'].exit_code, 0, result['report'].summary)
    test.assertIsNotNone(result['checkpoint'])
    test.assertEqual(result['next_shard']['shard_index'], 1)
    loaded = api.load_checkpoint(result['checkpoint'], first.accepted_identities[0], purpose='resume', device='cpu')
    test.assertTrue(api.finite_changed(dict(old.model_tensors), dict(loaded.model_tensors)))
    test.assertGreater(result['saved_prefix']['optimizer_accepted_steps'], 0)
    same = api.create_runtime(first, purpose='resume', checkpoint=result['checkpoint'],
        policy_kind='checkpoint', action_selection='seeded-sampling')
    try:
        _check_restored(test, same, loaded, api)
    finally:
        same.close()

    def resume_factory(config, *, purpose, checkpoint, policy_kind, action_selection):
        test.assertEqual((purpose, checkpoint, policy_kind, action_selection),
                         ('resume', result['checkpoint'], 'checkpoint', 'seeded-sampling'))
        runtime = api.create_runtime(config, purpose=purpose, checkpoint=checkpoint,
            policy_kind=policy_kind, action_selection=action_selection)
        try:
            _check_restored(test, runtime, loaded, api)
        except BaseException:
            runtime.close()
            raise
        return runtime

    # max_updates is the producer's cumulative high-water mark, not a local count.
    resumed = run_stream_segment(second, purpose='resume', checkpoint=result['checkpoint'],
        operation_limits=RuntimeOperationLimits(max_updates=2, monotonic_deadline=caps.deadline,
                                               max_environment_actions=512),
        dependencies=dict(create_runtime=resume_factory, train=api.train, clock=api.clock))
    test.assertEqual(resumed['report'].exit_code, 0, resumed['report'].summary)
    test.assertIsNotNone(resumed['checkpoint'])
    test.assertEqual(resumed['saved_prefix']['update_count'], 2)
    test.assertGreater(resumed['saved_prefix']['optimizer_accepted_steps'], result['saved_prefix']['optimizer_accepted_steps'])
    test.assertLessEqual(resumed['saved_prefix']['optimizer_attempted_steps'], 8)
    final = api.load_checkpoint(resumed['checkpoint'], second.accepted_identities[0], purpose='resume', device='cpu')
    test.assertTrue(api.finite_changed(dict(loaded.model_tensors), dict(final.model_tensors)))
    return result, resumed


@unittest.skipUnless(os.environ.get(GATE) == '1', f'set {GATE}=1 after separate protected release')
class CurriculumIntegrationTests(unittest.TestCase):
    def test_public_catalog_tensor_resume_and_tiny_healthy_segments(self):
        from spire_learn.neural.config import load_neural_config
        root = Path(os.environ['SPIRE_LEARN_CURRICULUM_INPUT_ROOT']).resolve()
        # The protected launcher must bind these exact already prepared bytes.
        configs = [load_neural_config(root / f'qualification-{i}.toml') for i in (0, 1)]
        parents = json.loads((root / 'pilot-parents.json').read_text())
        parent = Path(parents['standalone_b']['checkpoint_directory'])
        import hashlib
        for name, binding in parents['standalone_b']['inventory'].items():
            data = (parent / name).read_bytes()
            self.assertEqual(len(data), binding['size_bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), binding['sha256'])
        caps = _Caps()
        try:
            with _runtime_api(caps) as api:
                _exercise(self, configs, parent, api, caps)
        finally:
            # The future host owns durable stdout and whole-lifecycle accounting.
            print(json.dumps({'qualification_attempts': caps.counts}, sort_keys=True))
