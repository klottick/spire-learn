"""Pure generated-stream config and accepted checkpoint-boundary regressions."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from spire_learn.neural.config import load_neural_config
from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard
from spire_learn.neural.scenario_support import json_value


class StreamFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe_path = Path('configs/neural/curriculum-v1/recipe.json').resolve()
        cls.recipe = load_scenario_recipe(cls.recipe_path)
        cls.shards = [generate_scenario_shard(cls.recipe, split='train', shard_index=i)
                      for i in range(4)]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, index=0, arm='mixed'):
        shard = self.shards[index] if arm == 'mixed' else generate_scenario_shard(
            self.recipe, split='train', shard_index=index, arm=arm)
        (self.root / f'shard-{index}.json').write_text(json.dumps(json_value(shard.manifest)))
        source = Path('configs/neural/standalone.toml').read_text().replace(
            'manifest_path = "../cases/neural/train.json"',
            f'manifest_path = "shard-{index}.json"').replace(
            'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
            f'output_root = "{Path("artifacts/development/neural-combat/runs/stream-tests").resolve()}"')
        return source + ('\n[scenario_stream]\nschema = "spire-learn-scenario-stream-v1"\n'
                         f'recipe_path = "{self.recipe_path}"\narm = "{arm}"\n'
                         f'shard_index = {index}\ndistribution_sha256 = "{shard.distribution_sha256}"\n')

    def config(self, index=0, arm='mixed', source=None):
        path = self.root / f'config-{index}.toml'
        path.write_text(self.source(index, arm) if source is None else source)
        return load_neural_config(path)


class StreamResumeTests(StreamFixture):
    def test_loader_binds_actual_manifest_recipe_and_preserves_real_hashes(self):
        config = self.config()
        from spire_learn.neural.stream_resume import stream_identity
        self.assertEqual(config.manifest_sha256, self.shards[0].manifest_sha256)
        self.assertEqual(stream_identity(config)['distribution_sha256'], self.shards[0].distribution_sha256)
        self.assertIsNone(load_neural_config(Path('configs/neural/standalone.toml')).scenario_stream)
        text = self.source()
        for invalid in (text.replace('arm = "mixed"', 'arm = "randomized"'),
                        text.replace('shard_index = 0', 'shard_index = 1'),
                        text.replace(self.shards[0].distribution_sha256, '0' * 64),
                        text.replace('operation = "train"', 'operation = "evaluate"'),
                        text.replace('mode = "standalone"', 'mode = "run"'),
                        text + 'unknown = 1\n',
                        text.replace(str(self.recipe_path), '/missing/recipe.json')):
            with self.subTest(invalid=invalid[-100:]), self.assertRaises(ValueError):
                self.config(source=invalid)
        self.config()
        payload = json.loads((self.root / 'shard-0.json').read_text())
        payload['cases'][0]['entry']['hp'] = 24
        (self.root / 'shard-0.json').write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_neural_config(self.root / 'config-0.toml')

    def test_loader_bounds_retained_source_and_rejects_missing_support_binding(self):
        source = self.source()
        with self.assertRaises(ValueError):
            self.config(source=source + "#" + "x" * 32768 + "\n")
        last = generate_scenario_shard(self.recipe, split='train', shard_index=2**24 - 1)
        (self.root / 'shard-0.json').write_text(json.dumps(json_value(last.manifest)))
        with self.assertRaises(ValueError):
            self.config(source=source.replace('shard_index = 0', f'shard_index = {2**24 - 1}'))
        recipe_document = json.loads(self.recipe_path.read_text())
        recipe_document['support_file'] = str((self.recipe_path.parent / 'support.json').resolve())
        del recipe_document['support_sha256']
        recipe = self.root / 'unbound-recipe.json'
        recipe.write_text(json.dumps(recipe_document))
        with self.assertRaises(ValueError):
            self.config(source=source.replace(str(self.recipe_path), str(recipe)))

    def test_replaced_runtime_settings_reject_at_all_stream_entry_boundaries(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from spire_learn.neural import runner
        from spire_learn.neural.stream_resume import stream_identity
        from spire_learn.neural.stream_runner import run_stream_segment
        config = self.config()
        replacements = {
            'model': replace(config.model, hidden_width=32),
            'ppo': replace(config.ppo, learning_rate=0.01),
            'budgets': replace(config.budgets, max_optimizer_steps=17),
            'seed_streams': replace(config.seed_streams, policy=9999),
            'rollout_rows': 2, 'rollout_steps': config.rollout_steps + 1,
            'device': 'cuda:0', 'environment_revision': 'changed',
            'common_view_id': 'changed', 'common_view_fingerprint': '0' * 64,
            'run_observation_fingerprint': '0' * 64, 'profile_id': 'changed',
            'observation_version': 99, 'replay_version': 99,
            'final_requires_opt_in': True, 'calibration_gpu_uuid': 'changed',
            'runtime_gpu_uuid': 'changed',
            'accepted_identities': (replace(config.accepted_identities[0],
                config_hashes={'model': '0' * 64, 'ppo': '0' * 64}),),
        }
        original_import = __import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'gymnasium', 'spire_lab', 'torchrl', 'tensordict'}:
                raise AssertionError('optional import reached with stale config')
            return original_import(name, *args, **kwargs)
        variants = [(field, replace(config, **{field: value}))
                    for field, value in replacements.items()]
        variants.extend((f'budgets.{field}', replace(config, budgets=replace(
            config.budgets, **{field: getattr(config.budgets, field) + 1})))
            for field in ('max_training_decisions', 'max_environment_actions', 'max_wall_seconds'))
        variants.extend((f'seeds.{field}', replace(config, seed_streams=replace(
            config.seed_streams, **{field: getattr(config.seed_streams, field) + 10000})))
            for field in ('optimizer', 'curriculum', 'noncombat'))
        for field, changed in variants:
            self.assertEqual(changed.config_sha256, config.config_sha256)
            self.assertEqual(changed.manifest_sha256, config.manifest_sha256)
            with patch('builtins.__import__', side_effect=guard):
                with self.subTest(field=field, boundary='identity'), self.assertRaisesRegex(ValueError, 'runtime projection'):
                    stream_identity(changed)
                with self.subTest(field=field, boundary='create_runtime'), patch.object(
                        runner, 'plan_runtime_device', side_effect=ValueError('device planning reached')) as device:
                    with self.assertRaisesRegex(ValueError, 'runtime projection'):
                        runner.create_runtime(changed, purpose='train')
                    device.assert_not_called()
                factory, train, clock = Mock(side_effect=ValueError('factory reached')), Mock(), Mock()
                with self.subTest(field=field, boundary='run_stream_segment'), self.assertRaisesRegex(ValueError, 'runtime projection'):
                    run_stream_segment(changed, purpose='train', checkpoint=None,
                        operation_limits=runner.RuntimeOperationLimits(),
                        dependencies={'create_runtime': factory, 'train': train, 'clock': clock})
                factory.assert_not_called()
                train.assert_not_called()
                clock.assert_not_called()
                runtime = SimpleNamespace(close=Mock())
                with self.subTest(field=field, boundary='train'), patch.object(
                        runner, '_writer_start') as writer, \
                     patch.object(runner, '_set_budget', side_effect=AssertionError('budget setup reached')) as setup:
                    report = runner.train(changed, runtime=runtime, clock=clock)
                    self.assertEqual(report.exit_code, 2)
                    self.assertIn('runtime projection', report.summary['stop_reason'])
                    runtime.close.assert_called_once_with()
                    setup.assert_not_called()
                    writer.assert_not_called()
                    clock.assert_not_called()

    def test_resume_only_same_or_next_derived_shard_and_stable_settings(self):
        from spire_learn.neural.stream_resume import stream_checkpoint_state, validate_stream_resume
        from spire_learn.neural.curriculum import CurriculumSampler
        first = self.config(0)
        sampler = CurriculumSampler(self.shards[0].annotations,
            distribution_sha256=self.shards[0].distribution_sha256, categories=self.recipe.category_keys)
        state = stream_checkpoint_state(first, sampler.snapshot())
        metadata = {'scenario_stream': state, 'config_sha256': first.config_sha256,
                    'manifest_sha256': first.manifest_sha256}
        frozen = copy.deepcopy(metadata)
        for destination in (first, self.config(1)):
            restored = validate_stream_resume(metadata, destination)
            self.assertEqual(restored['sampler_state'], sampler.snapshot())
        self.assertEqual(metadata, frozen)
        restored = validate_stream_resume(metadata, self.config(1))
        self.assertEqual(restored["destination"]["config_sha256"], self.config(1).config_sha256)
        self.assertEqual(restored["destination"]["manifest_raw_sha256"],
                         hashlib.sha256((self.root / "shard-1.json").read_bytes()).hexdigest())
        for invalid in (self.config(2), self.config(3),
                        replace(first, rollout_steps=first.rollout_steps + 1),
                        replace(first, budgets=replace(first.budgets, max_optimizer_steps=17)),
                        replace(first, device='cuda:0'),
                        replace(first, model=replace(first.model, hidden_width=32)),
                        replace(first, ppo=replace(first.ppo, learning_rate=0.01)),
                        replace(first, seed_streams=replace(first.seed_streams, policy=9999))):
            with self.subTest(invalid=invalid.config_sha256), self.assertRaises(ValueError):
                validate_stream_resume(metadata, invalid)
        for key in ('config_sha256', 'manifest_sha256'):
            tampered = copy.deepcopy(metadata)
            tampered[key] = '0' * 64
            with self.assertRaises(ValueError):
                validate_stream_resume(tampered, first)
        for invalid_state in (None, {}, []):
            wrong = copy.deepcopy(metadata)
            wrong['scenario_stream']['sampler_state'] = invalid_state
            with self.assertRaises(ValueError):
                validate_stream_resume(wrong, first)
        wrong = copy.deepcopy(metadata)
        wrong['scenario_stream']['sampler_state']['completed'] = True
        with self.assertRaises(ValueError):
            validate_stream_resume(wrong, first)
        second = self.config(1)
        metadata = {'scenario_stream': stream_checkpoint_state(second, sampler.snapshot()),
                    'config_sha256': second.config_sha256, 'manifest_sha256': second.manifest_sha256}
        with self.assertRaises(ValueError):
            validate_stream_resume(metadata, first)


class StatefulGenerator:
    """A fake device generator with actual independent deterministic advancement."""
    def __init__(self, *, device):
        if device != 'cpu':
            raise AssertionError('this pure test only uses CPU generator contracts')
        self.rng = random.Random()

    def manual_seed(self, seed):
        self.rng.seed(seed)

    def get_state(self):
        return self.rng.getstate()

    def set_state(self, state):
        self.rng.setstate(state)

    def draw(self):
        return self.rng.random()


class StatefulAdam:
    def __init__(self, parameters, *, lr, eps):
        self.param_groups = [{'lr': lr, 'eps': eps}]
        self.state = {'step': 0, 'exp_avg': 0.0, 'exp_avg_sq': 0.0}

    def step(self):
        self.state['step'] += 1
        self.state['exp_avg'] += 0.125
        self.state['exp_avg_sq'] += 0.03125

    def state_dict(self):
        return copy.deepcopy({'state': self.state, 'param_groups': self.param_groups})

    def load_state_dict(self, state):
        if set(state) != {'state', 'param_groups'}:
            raise AssertionError('malformed Adam state')
        self.state = copy.deepcopy(state['state'])
        self.param_groups = copy.deepcopy(state['param_groups'])


class StreamRuntimeTests(StreamFixture):
    def assembly(self, config, *, purpose, loaded=None):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        from spire_learn.neural import runner, model, envs
        from tests.unit.test_neural_collector import FakePolicy, ScriptedDriver, row, transition, exit_data

        class Policy(FakePolicy):
            def __init__(self):
                super().__init__(config)
                self._behavior_policy_version = 0
                self.weights = {'weight': 17.0}

            def parameters(self):
                return (self.weights,)

            def state_dict(self):
                return dict(self.weights)

            def load_state_dict(self, state, *, strict):
                if not strict or set(state) != {'weight'}:
                    raise AssertionError('weight-only load is strict')
                self.weights = dict(state)

            def act(self, frames, *, generator, deterministic=False):
                generator.draw()
                self.behavior_version = self._behavior_policy_version
                return super().act(frames, generator=generator, deterministic=deterministic)

        def delivered(before, actions):
            return tuple(transition(source, row(44, active=False, entered=1, settled=1, exit_value=exit_data()),
                                    terminated=True, reward=1.0) for source in before)

        driver = ScriptedDriver('standalone', config.rollout_rows, steps=[delivered] * config.rollout_steps)
        driver.schema = SimpleNamespace(descriptor_digest='pure-stream-schema', tensors={})
        policy = Policy()
        stack = ExitStack()
        self.addCleanup(stack.close)
        fake_torch = SimpleNamespace(Generator=StatefulGenerator, optim=SimpleNamespace(Adam=StatefulAdam))
        stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
        stack.enter_context(patch.object(envs, 'make_driver', return_value=driver))
        stack.enter_context(patch.object(model, 'build_policy', return_value=policy))
        if loaded is not None:
            stack.enter_context(patch.object(runner, 'preflight_checkpoint',
                                            return_value=SimpleNamespace(metadata=loaded.metadata)))
            stack.enter_context(patch.object(runner, 'load_checkpoint', return_value=loaded))
        runtime = runner.create_runtime(config, purpose=purpose,
            checkpoint=None if purpose == 'train' else Path('/pure/frozen-parent'))
        return runtime, driver

    def capture(self, runtime):
        from types import SimpleNamespace
        from unittest.mock import patch
        from spire_learn.neural import runner
        captured = []

        def save(path, *, policy, optimizer, rng_state, metadata):
            self.assertEqual(path, Path('/pure/checkpoint'))
            self.assertFalse(runtime._closed)
            self.assertEqual(set(rng_state), {'policy', 'minibatch', 'curriculum', 'controller'})
            captured.append(SimpleNamespace(metadata=copy.deepcopy(metadata),
                rng_state=copy.deepcopy(rng_state), model_tensors=policy.state_dict(),
                optimizer_state=optimizer.state_dict()))

        with patch.object(runner, 'save_checkpoint', side_effect=save):
            runtime.save(Path('/pure/checkpoint'), 'resume')
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_real_save_recovery_warm_start_and_two_transitions_restore_exact_boundary(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from spire_learn.neural import runner
        from spire_learn.neural.ppo import UpdateMetrics
        from spire_learn.neural.stream_resume import validate_stream_resume

        old_parent = SimpleNamespace(model_tensors={'weight': 23.0}, metadata={},
            optimizer_state={'forbidden': 'warm start must discard'}, rng_state=None)
        configs = [self.config(i) for i in range(3)]
        # The source settings and every concrete identity must remain untouched.
        identities = [(c.config_sha256, c.manifest_sha256) for c in configs]
        loaded = old_parent
        previous_ids = set()
        previous_allocations = set()
        previous_sampler = None
        bindings = []
        for index, config in enumerate(configs):
            runtime, driver = self.assembly(config,
                purpose='warm_start' if index == 0 else 'resume', loaded=loaded)
            if index == 0:
                self.assertEqual(runtime.policy.weights, {'weight': 23.0})
                self.assertEqual(runtime.optimizer.state_dict()['state']['step'], 0)
                self.assertEqual(runtime.policy._behavior_policy_version, 0)
                self.assertEqual(runtime.recovered_progress, {})
                self.assertEqual(runtime.curriculum_rng.getstate(), random.Random(config.seed_streams.curriculum).getstate())
            else:
                self.assertEqual(runtime.policy.weights, loaded.model_tensors)
                self.assertEqual(runtime.optimizer.state_dict(), loaded.optimizer_state)
                self.assertEqual(runtime.policy._behavior_policy_version, index)
                self.assertEqual(runtime.recovered_progress, loaded.metadata['training_state']['counters'])
                for name, generator in (('policy', runtime.policy_generator), ('minibatch', runtime.shuffle_generator)):
                    self.assertEqual(generator.get_state(), loaded.rng_state[name])
                self.assertEqual(runtime.curriculum_rng.getstate(), loaded.rng_state['curriculum'])
                self.assertEqual(runtime.controller_rng.getstate(), loaded.rng_state['controller'])
                self.assertEqual(runtime.collector.sampler_state, previous_sampler)
                self.assertEqual(driver.reset_calls, [])

            before = {name: generator.get_state() for name, generator in (
                ('policy', runtime.policy_generator), ('minibatch', runtime.shuffle_generator))}
            controller_before = runtime.controller_rng.getstate()
            curriculum_before = runtime.curriculum_rng.getstate()
            from spire_learn.neural.curriculum import CurriculumSampler
            expected_sampler = CurriculumSampler(self.shards[index].annotations,
                distribution_sha256=self.shards[index].distribution_sha256,
                categories=self.recipe.category_keys, state=runtime.collector.sampler_state)
            expected_rng = random.Random()
            expected_rng.setstate(curriculum_before)
            expected_cases = [expected_sampler.choose_case(config.cases.for_mode('standalone'),
                                                           rng=expected_rng) for _ in range(4)]
            for case in expected_cases:
                expected_sampler.observe(case.case_id, 'victory')
            rollout = runtime.collector.collect(max_learned_decisions=4, max_env_actions=4,
                policy_version=runtime.policy._behavior_policy_version)
            ids = {(r.task_id, r.decision_id) for r in rollout.records}
            seeds = {a.actual_seed for a in rollout.collection.seed_allocations}
            self.assertTrue(ids.isdisjoint(previous_ids))
            self.assertTrue(seeds.isdisjoint(previous_allocations))
            self.assertTrue({r.task_id for r in rollout.records}.isdisjoint(
                {task for task, _decision in previous_ids}))
            self.assertTrue({r.decision_id for r in rollout.records}.isdisjoint(
                {decision for _task, decision in previous_ids}))
            previous_ids.update(ids)
            previous_allocations.update(seeds)
            self.assertEqual(len(rollout.records), 4)
            self.assertNotEqual(runtime.policy_generator.get_state(), before['policy'])
            self.assertEqual(runtime.shuffle_generator.get_state(), before['minibatch'])
            self.assertNotEqual(runtime.curriculum_rng.getstate(), curriculum_before)
            self.assertEqual(runtime.controller_rng.getstate(), controller_before)
            self.assertEqual(runtime.curriculum_rng.getstate(), expected_rng.getstate())
            self.assertEqual([a.case_id for a in rollout.collection.seed_allocations],
                             [case.case_id for case in expected_cases])
            self.assertEqual(runtime.collector.sampler_state, expected_sampler.snapshot())
            exact_policy = random.Random()
            exact_policy.setstate(before['policy'])
            exact_policy.random()
            self.assertEqual(runtime.policy_generator.get_state(), exact_policy.getstate())

            def update(policy, optimizer, rollout, ppo_config, **kwargs):
                # Fake only the tensor/PPO arithmetic, retaining the real runtime counter boundary.
                runtime.shuffle_generator.draw()
                optimizer.step()
                version = policy._behavior_policy_version
                policy._behavior_policy_version += 1
                policy.weights['weight'] += 0.5
                return UpdateMetrics(0.0, 0.0, 0.0, 1, 1, None, version, version + 1, 4)

            with patch.object(runner, 'ppo_update', side_effect=update):
                metrics = runtime.update(rollout)
            runtime.record_update(metrics)
            exact_minibatch = random.Random()
            exact_minibatch.setstate(before['minibatch'])
            exact_minibatch.random()
            self.assertEqual(runtime.shuffle_generator.get_state(), exact_minibatch.getstate())
            previous_sampler = runtime.collector.sampler_state
            loaded = self.capture(runtime)
            current = loaded.metadata['run_metadata']['scenario_stream']['current']
            self.assertEqual(current['config_sha256'], config.config_sha256)
            self.assertEqual(current['manifest_sha256'], config.manifest_sha256)
            self.assertEqual(current['config_raw_sha256'], hashlib.sha256(
                (self.root / f'config-{index}.toml').read_bytes()).hexdigest())
            self.assertEqual(current['manifest_raw_sha256'], hashlib.sha256(
                (self.root / f'shard-{index}.json').read_bytes()).hexdigest())
            if index:
                transition = loaded.metadata['run_metadata']['scenario_stream_transition']
                self.assertEqual(transition['from'], bindings[-1])
                self.assertEqual(transition['to'], current)
                self.assertEqual(transition['from']['shard_index'], index - 1)
                self.assertEqual(transition['to']['shard_index'], index)
                self.assertEqual(transition['identity'], loaded.metadata['run_metadata']['scenario_stream']['identity'])
            bindings.append(copy.deepcopy(current))
            self.assertLess(len(json.dumps(loaded.metadata).encode()), 1048576)
            self.assertEqual(loaded.metadata['run_metadata']['scenario_stream']['sampler_state'], previous_sampler)
            recovery, progress = runner._checkpoint_recovery(loaded, config)
            self.assertEqual(recovery.next_seed_ordinal, 4 * (index + 1))
            self.assertEqual(recovery.next_decision_sequence, index + 1)
            self.assertEqual(progress['optimizer_accepted_steps'], index + 1)
            self.assertEqual(progress, loaded.metadata['training_state']['counters'])
            runtime.close()
            with self.assertRaises(RuntimeError):
                _ = runtime.collector.sampler_state
            self.assertEqual(driver.close_calls, 1)
        self.assertEqual(identities, [(c.config_sha256, c.manifest_sha256) for c in configs])
        bad = copy.deepcopy(loaded)
        del bad.metadata['run_metadata']['scenario_stream_transition']
        with self.assertRaises(ValueError):
            runner._checkpoint_recovery(bad, configs[-1])
        for key in ('identity', 'from', 'to'):
            bad = copy.deepcopy(loaded)
            if key == 'identity':
                bad.metadata['run_metadata']['scenario_stream_transition'][key]['stable_training_sha256'] = '0' * 64
            else:
                bad.metadata['run_metadata']['scenario_stream_transition'][key]['manifest_sha256'] = '0' * 64
            with self.subTest(transition_key=key), self.assertRaises(ValueError):
                runner._checkpoint_recovery(bad, configs[-1])
        bad = copy.deepcopy(loaded)
        bad.metadata['run_metadata']['scenario_stream_transition']['from'] = bindings[0]
        with self.assertRaises(ValueError):
            runner._checkpoint_recovery(bad, configs[-1])
        # A later config cannot pretend the original warm-start parent carries stream continuity.
        with self.assertRaises(ValueError):
            runner._checkpoint_recovery(old_parent, configs[-1])
        self.assertEqual(validate_stream_resume(loaded.metadata['run_metadata'], configs[-1])['sampler_state'], previous_sampler)
        same, same_driver = self.assembly(configs[-1], purpose='resume', loaded=loaded)
        same_saved = self.capture(same)
        transition = same_saved.metadata['run_metadata']['scenario_stream_transition']
        self.assertEqual(transition['from'], bindings[-1])
        self.assertEqual(transition['to'], bindings[-1])
        self.assertEqual(same_saved.rng_state, loaded.rng_state)
        self.assertEqual(same_saved.optimizer_state, loaded.optimizer_state)
        runner._checkpoint_recovery(same_saved, configs[-1])
        same.close()
        self.assertEqual(same_driver.reset_calls, [])
        self.assertEqual(same_driver.step_calls, [])

    def test_core_train_observer_fault_retains_real_delivered_prefix_without_update(self):
        from unittest.mock import patch
        from spire_learn.neural import runner
        from spire_learn.neural.curriculum import CurriculumSampler
        from tests.unit.test_neural_runner import FakeWriter
        config = self.config()
        runtime, driver = self.assembly(config, purpose='train')
        writer = FakeWriter()
        first = ValueError('first sampler fault after delivery')
        with patch.object(runner, '_writer_start', return_value=(writer, Path('/pure/fault'))), \
             patch.object(CurriculumSampler, 'observe', side_effect=first), \
             patch.object(runtime, 'update', side_effect=AssertionError('postfault optimizer work')) as update, \
             patch.object(runner, 'save_checkpoint') as save:
            report = runner.train(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(len(driver.reset_calls), 1)
        self.assertEqual(len(driver.step_calls), 1)
        self.assertEqual(sum(e['event'] == 'accepted_action' for e in writer.events), 4)
        self.assertEqual(update.call_count, 0)
        self.assertEqual(save.call_count, 0)
        self.assertIn('first sampler fault after delivery', str(report.summary))
        self.assertEqual(driver.close_calls, 1)
        with self.assertRaises(RuntimeError):
            _ = runtime.collector.sampler_state

    def test_partial_event_control_propagates_after_fault_and_cleanup(self):
        from unittest.mock import patch
        from spire_learn.neural import runner
        from spire_learn.neural.curriculum import CurriculumSampler
        from tests.unit.test_neural_runner import FakeWriter
        config = self.config()
        scenarios = [(first, later, cleanup)
            for first, later in ((ValueError, KeyboardInterrupt), (ValueError, SystemExit),
                                 (KeyboardInterrupt, SystemExit), (SystemExit, KeyboardInterrupt))
            for cleanup in (RuntimeError, SystemExit)]
        for first_kind, later_kind, cleanup_kind in scenarios:
            with self.subTest(first=first_kind.__name__, later=later_kind.__name__,
                              cleanup=cleanup_kind.__name__):
                runtime, driver = self.assembly(config, purpose='train')
                first, later = first_kind('collection fault'), later_kind('event interruption')
                writer = FakeWriter(close_error=cleanup_kind('writer cleanup failure'))
                close_runtime = runtime.close
                def fail_after_close():
                    close_runtime()
                    raise cleanup_kind('runtime cleanup failure')
                write = runner._write_collection_events
                def interrupt_after_prefix(*args, **kwargs):
                    write(*args, **kwargs)
                    raise later
                with patch.object(runner, '_writer_start', return_value=(writer, Path('/pure/fault'))), \
                     patch.object(CurriculumSampler, 'observe', side_effect=first), \
                     patch.object(runner, '_write_collection_events', side_effect=interrupt_after_prefix), \
                     patch.object(runtime, 'update') as update, \
                     patch.object(runtime, 'save') as save, \
                     patch.object(runtime, 'close', side_effect=fail_after_close) as close:
                    expected = later if isinstance(first, Exception) else first
                    with self.assertRaises(type(expected)) as caught:
                        runner.train(config, runtime=runtime, clock=lambda: 0.0)
                self.assertIs(caught.exception, expected)
                self.assertIn('collection fault' if expected is later else 'event interruption',
                              str(caught.exception.__notes__))
                self.assertIn('runtime cleanup failure', str(caught.exception.__notes__))
                self.assertIn('writer cleanup failure', str(caught.exception.__notes__))
                self.assertEqual(sum(e['event'] == 'accepted_action' for e in writer.events), 4)
                update.assert_not_called()
                save.assert_not_called()
                close.assert_called_once_with()
                self.assertEqual(driver.close_calls, 1)
                self.assertEqual(writer.close_calls, 1)

    def test_direct_train_admission_control_closes_owned_runtime_once(self):
        from unittest.mock import Mock, patch
        from spire_learn.neural import runner, stream_resume
        config = self.config()
        original_import = __import__
        for boundary in ('import', 'identity'):
            for control_kind in (KeyboardInterrupt, SystemExit):
                for cleanup_kind in (None, RuntimeError, SystemExit):
                    with self.subTest(boundary=boundary, control=control_kind.__name__,
                                      cleanup=cleanup_kind):
                        runtime, driver = self.assembly(config, purpose='train')
                        primary = control_kind('admission interrupted')
                        close_runtime = runtime.close
                        def close_owned():
                            close_runtime()
                            if cleanup_kind is not None:
                                raise cleanup_kind('admission cleanup failed')
                        def import_admission(name, *args, **kwargs):
                            if boundary == 'import' and name == 'stream_resume':
                                raise primary
                            return original_import(name, *args, **kwargs)
                        clock = Mock()
                        with patch('builtins.__import__', side_effect=import_admission), \
                             patch.object(stream_resume, 'stream_identity', side_effect=primary) as admission, \
                             patch.object(runtime, 'close', side_effect=close_owned) as close, \
                             patch.object(runtime, 'update') as update, \
                             patch.object(runtime, 'save') as save, \
                             patch.object(runner, '_set_budget') as setup, \
                             patch.object(runner, '_writer_start') as writer:
                            with self.assertRaises(control_kind) as caught:
                                runner.train(config, runtime=runtime, clock=clock)
                        self.assertIs(caught.exception, primary)
                        close.assert_called_once_with()
                        self.assertEqual(driver.close_calls, 1)
                        self.assertEqual(driver.reset_calls, [])
                        self.assertEqual(driver.step_calls, [])
                        update.assert_not_called()
                        save.assert_not_called()
                        setup.assert_not_called()
                        writer.assert_not_called()
                        clock.assert_not_called()
                        self.assertEqual(admission.call_count, int(boundary == 'identity'))
                        if cleanup_kind is not None:
                            self.assertIn('admission cleanup failed', str(primary.__notes__))

    def test_segment_real_train_admission_control_closes_factory_runtime(self):
        from unittest.mock import Mock, patch
        from spire_learn.neural import runner, stream_resume
        from spire_learn.neural.stream_runner import run_stream_segment
        config = self.config()
        identity = stream_resume.stream_identity
        for control_kind in (KeyboardInterrupt, SystemExit):
            for cleanup_kind in (None, RuntimeError, SystemExit):
                with self.subTest(control=control_kind.__name__, cleanup=cleanup_kind):
                    primary = control_kind('repeat admission interrupted')
                    acquired = []
                    def admission(value):
                        if acquired:
                            raise primary
                        return identity(value)
                    def factory(value, *, purpose, checkpoint, policy_kind, action_selection):
                        runtime, driver = self.assembly(value, purpose=purpose)
                        close_runtime = runtime.close
                        def close_owned():
                            close_runtime()
                            if cleanup_kind is not None:
                                raise cleanup_kind('segment cleanup failed')
                        close = Mock(side_effect=close_owned)
                        runtime.close = close
                        runtime.update = Mock()
                        runtime.save = Mock()
                        acquired.append((runtime, driver, close))
                        return runtime
                    create = Mock(side_effect=factory)
                    clock = Mock()
                    with patch.object(stream_resume, 'stream_identity', side_effect=admission), \
                         patch.object(runner, '_writer_start') as writer:
                        with self.assertRaises(control_kind) as caught:
                            run_stream_segment(config, purpose='train', checkpoint=None,
                                operation_limits=runner.RuntimeOperationLimits(),
                                dependencies={'create_runtime': create, 'train': runner.train, 'clock': clock})
                    self.assertIs(caught.exception, primary)
                    create.assert_called_once()
                    self.assertEqual(len(acquired), 1)
                    runtime, driver, close = acquired[0]
                    close.assert_called_once_with()
                    self.assertEqual(driver.close_calls, 1)
                    self.assertEqual(driver.reset_calls, [])
                    self.assertEqual(driver.step_calls, [])
                    runtime.update.assert_not_called()
                    runtime.save.assert_not_called()
                    writer.assert_not_called()
                    clock.assert_not_called()
                    if cleanup_kind is not None:
                        self.assertIn('segment cleanup failed', str(primary.__notes__))

    def test_wrong_parent_rejected_before_optional_import_or_driver(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from spire_learn.neural import runner, envs
        import builtins
        config = self.config()
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'gymnasium', 'spire_lab', 'torchrl', 'tensordict'}:
                raise AssertionError('optional runtime reached during invalid resume')
            return original_import(name, *args, **kwargs)
        with patch.object(runner, 'preflight_checkpoint', return_value=SimpleNamespace(metadata={})), \
             patch.object(envs, 'make_driver', side_effect=AssertionError('driver reached')), \
             patch('builtins.__import__', side_effect=guard):
            with self.assertRaises(ValueError):
                runner.create_runtime(config, purpose='resume', checkpoint=Path('/wrong-parent'))

    def test_exhausted_restored_budget_forwards_no_reset_step_or_update(self):
        from unittest.mock import patch
        from spire_learn.neural import runner
        from tests.unit.test_neural_runner import FakeWriter
        config = self.config()
        runtime, driver = self.assembly(config, purpose='train')
        # Exercise the actual optimizer/counter save and restore boundary with a consumed budget.
        for _ in range(config.budgets.max_optimizer_steps):
            runtime.optimizer.step()
        runtime.optimizer_attempted_steps = config.budgets.max_optimizer_steps
        runtime.optimizer_accepted_steps = config.budgets.max_optimizer_steps
        saved = self.capture(runtime)
        runtime.close()
        resumed, fresh_driver = self.assembly(config, purpose='resume', loaded=saved)
        with patch.object(runner, '_writer_start', return_value=(FakeWriter(), Path('/pure/exhausted'))), \
             patch.object(runner, 'save_checkpoint'), \
             patch.object(resumed, 'update', side_effect=AssertionError('budget forwarded optimizer work')):
            report = runner.train(config, runtime=resumed, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(fresh_driver.reset_calls, [])
        self.assertEqual(fresh_driver.step_calls, [])
        self.assertEqual(fresh_driver.close_calls, 1)

    def test_fixed_and_randomized_never_inject_adaptive_sampler(self):
        for arm in ('fixed', 'randomized'):
            with self.subTest(arm=arm):
                config = self.config(arm=arm)
                runtime, driver = self.assembly(config, purpose='train')
                self.assertIsNone(runtime.collector.sampler_state)
                saved = self.capture(runtime)
                self.assertEqual(saved.metadata['run_metadata']['scenario_stream']['sampler_state'], {})
                runtime.close()
