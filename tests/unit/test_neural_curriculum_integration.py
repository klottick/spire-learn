"""Actual held harness under strict stdlib runtime boundaries, without packages."""
from contextlib import contextmanager
import builtins
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/development/neural-combat'
PREFIXES = ('torch', 'numpy', 'gymnasium', 'spire_lab', 'torchrl', 'tensordict')


@contextmanager
def blocked_optional():
    attempts = []
    original = builtins.__import__
    def forbidden(name):
        return any(name == p or name.startswith(p + '.') for p in PREFIXES)
    def guarded(name, *args, **kwargs):
        if forbidden(name):
            attempts.append(name)
            raise AssertionError(f'optional import attempted: {name}')
        return original(name, *args, **kwargs)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if forbidden(fullname):
                attempts.append(fullname)
                raise AssertionError(f'optional resolution attempted: {fullname}')
    blocker = Block()
    sys.meta_path.insert(0, blocker)
    try:
        with patch.object(builtins, '__import__', side_effect=guarded):
            yield attempts
    finally:
        sys.meta_path.remove(blocker)
    if attempts or any(forbidden(n) for n in sys.modules):
        raise AssertionError('optional package escaped pure boundary')


class CurriculumIntegrationHarnessTests(unittest.TestCase):
    def test_gate_off_skips_before_all_six_optional_imports(self):
        code = '''
import unittest
from tests.unit.test_neural_curriculum_integration import blocked_optional
with blocked_optional() as attempts:
    from tests.neural.test_curriculum import CurriculumIntegrationTests
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(CurriculumIntegrationTests).run(result)
    assert result.testsRun == 1 and len(result.skipped) == 1
    assert not result.errors and not result.failures
    assert not attempts
print('gate_off: one skip; optional attempts/loaded: []')
'''
        env = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
        env['PYTHONPATH'] = str(ROOT / 'src')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('one skip', result.stdout)

    def test_enabled_harness_runs_real_producers_with_actual_config_and_save_flow(self):
        # The break caught here is a wrong integration factory/segment/save/resume
        # contract. Tensor math, native dispatch and tensor archives are the only
        # external boundaries; reports/progress/lifecycle are real producers.
        from tests.neural import test_curriculum as held
        from spire_learn.neural import runner, envs
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.stream_runner import prepare_stream, _toml
        from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard
        from spire_learn.neural.stream_resume import validate_stream_resume
        from spire_learn.neural.curriculum import CurriculumSampler
        from spire_learn.neural.collector import Collector
        from spire_learn.neural.device import plan_runtime_device
        from spire_learn.neural.ppo import UpdateMetrics
        from tests.unit.test_neural_stream_resume import StatefulGenerator
        from tests.unit.test_neural_collector import FakePolicy, ScriptedDriver, row, transition, exit_data
        from tests.unit.test_neural_checkpoint import _write_archive
        from spire_learn.neural.checkpoint import CHECKPOINT_SCHEMA_ID
        import tomllib
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary, blocked_optional() as attempts:
            root = Path(temporary) / 'prepared'
            prepare_stream(ROOT / 'configs/neural/curriculum-v1/recipe.json', root,
                           arm='mixed', split='train', shard_index=0)
            document = tomllib.loads((root / 'config.toml').read_text())
            document['rollout']['steps'] = 16
            document['budgets'].update(max_optimizer_steps=8, max_training_decisions=512,
                                      max_environment_actions=512)
            configs = []
            for i in (0, 1):
                document['manifest_path'] = f'shard-{i}.json'
                document['scenario_stream']['shard_index'] = i
                path = root / f'qualification-{i}.toml'
                path.write_text(_toml(document))
                configs.append(load_neural_config(path))
            caps = held._Caps(clock=lambda: 0.0)
            parent = root / 'original-parent'
            parent.mkdir()
            inventory = {}
            for name in ('metadata.json', 'model.pt', 'training.pt'):
                data = ('synthetic parent ' + name).encode()
                (parent / name).write_bytes(data)
                inventory[name] = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            (root / 'pilot-parents.json').write_text(json.dumps(dict(standalone_b=dict(
                checkpoint_directory=str(parent), inventory=inventory))))
            loaded = {parent: SimpleNamespace(model_tensors={'weight': 23.0})}
            runtimes, drivers, factory_calls, catalogs = [], [], [], []
            def catalog_driver(config, cases):
                class CatalogFacade:
                    def __init__(inner, rows, catalog, *, partition):
                        self.assertEqual((rows, partition), (4, 'train'))
                        self.assertEqual(len(catalog['entries']), 256)
                        inner.entries = catalog['entries']
                        inner.close_calls = 0
                        catalogs.append(inner)
                    def observation_spec(inner):
                        return {'synthetic': 'descriptor boundary'}
                    def environment_spec(inner):
                        return dict(profile_id='ironclad-combat-v1', schema_id=envs.COMMON_VIEW_ID,
                            observation_fingerprint=config.common_view_fingerprint,
                            content_id='spire-gym-ironclad-combat-content-v1',
                            rules_id='spire-gym-native-m10-rules-v1',
                            reward_id='spire-gym-combat-outcome-reward-v1', mode='standalone_combat',
                            partition='train', autoreset='disabled')
                    def catalog_descriptor(inner):
                        return dict(catalog_schema_id='spire-gym-combat-entry-catalog-v1',
                                    entries=[envs._normalized_entry(e) for e in inner.entries])
                    def reset(inner, **kwargs):
                        self.fail('normalization probe reset')
                    def step(inner, actions):
                        self.fail('normalization probe step')
                    def close(inner):
                        inner.close_calls += 1
                native = SimpleNamespace(CombatInputError=ValueError, RunInputError=ValueError,
                    CombatStateError=RuntimeError, RunStateError=RuntimeError,
                    build_info=lambda: dict(extension='spire-python', revision=config.environment_revision))
                facade = envs._NativeRuntime(SimpleNamespace(), native, CatalogFacade, object)
                schema = SimpleNamespace(schema_fingerprint=config.common_view_fingerprint,
                                         action_space=SimpleNamespace(n=10))
                with patch.object(envs, '_load_native_runtime', return_value=facade), \
                     patch.object(envs, 'parse_common_descriptor', return_value=schema):
                    return held._BoundedDriver(envs.make_driver(config, cases), caps)
            class Policy(FakePolicy):
                def __init__(self, config):
                    super().__init__(config)
                    self._behavior_policy_version = 0
                    self.weights = {'weight': 0.0}
                def state_dict(self):
                    return dict(self.weights)
                def act(self, frames, *, generator, deterministic=False):
                    caps.charge('samples', len(frames))
                    generator.draw()
                    self.behavior_version = self._behavior_policy_version
                    return super().act(frames, generator=generator, deterministic=deterministic)
            class Adam:
                def __init__(self):
                    self.state = {}
                    self.param_groups = [{'lr': .0003, 'eps': .00001}]
                def step(self):
                    caps.charge('adam', 1)
                    self.state['step'] = self.state.get('step', 0) + 1
                    self.state['exp_avg'] = self.state.get('exp_avg', 0.0) + .125
                def state_dict(self):
                    return copy.deepcopy(dict(state=self.state, param_groups=self.param_groups))
                def load_state_dict(self, value):
                    self.state, self.param_groups = copy.deepcopy((value['state'], value['param_groups']))
            def make_driver(config, cases):
                self.assertIs(cases, config.cases)
                def delivered(before, actions):
                    return tuple(transition(source, row(44, active=False, entered=1,
                        settled=1, exit_value=exit_data()), terminated=True, reward=1.) for source in before)
                driver = ScriptedDriver('standalone', config.rollout_rows,
                                        steps=[delivered] * config.rollout_steps)
                driver.schema = SimpleNamespace(descriptor_digest='pure-integration-schema', tensors={})
                drivers.append(driver)
                return held._BoundedDriver(driver, caps)
            def load_checkpoint(path, identity, *, purpose, device):
                self.assertEqual(device, 'cpu')
                self.assertIn(purpose, ('warm_start', 'resume'))
                self.assertEqual(identity, configs[0].accepted_identities[0])
                return loaded[path]
            def create_runtime(config, *, purpose, checkpoint, policy_kind, action_selection):
                factory_calls.append((purpose, checkpoint))
                self.assertEqual((policy_kind, action_selection), ('checkpoint', 'seeded-sampling'))
                saved = loaded[checkpoint]
                recovery, progress, stream_transition = None, {}, None
                recipe = load_scenario_recipe(Path(config.scenario_stream['resolved_recipe_path']))
                shard = generate_scenario_shard(recipe, split='train', shard_index=config.scenario_stream['shard_index'])
                sampler_state = None
                if purpose == 'resume':
                    recovery, progress = runner._checkpoint_recovery(saved, config)
                    state = validate_stream_resume(saved.metadata['run_metadata'], config)
                    stream_transition = dict(identity=state['identity'], **{'from': state['current'], 'to': state['destination']})
                    sampler_state = state['sampler_state']
                sampler = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                             categories=recipe.category_keys, state=sampler_state)
                policy = Policy(config)
                policy.weights = dict(saved.model_tensors)
                optimizer = runner._CountingOptimizer(Adam())
                policy_rng, shuffle = StatefulGenerator(device='cpu'), StatefulGenerator(device='cpu')
                policy_rng.manual_seed(config.seed_streams.policy)
                shuffle.manual_seed(config.seed_streams.optimizer)
                curriculum, controller = random.Random(config.seed_streams.curriculum), random.Random(config.seed_streams.noncombat)
                if purpose == 'resume':
                    policy._behavior_policy_version = saved.metadata['training_state']['behavior_policy_version']
                    optimizer.load_state_dict(saved.optimizer_state)
                    policy_rng.set_state(saved.rng_state['policy']); shuffle.set_state(saved.rng_state['minibatch'])
                    curriculum.setstate(saved.rng_state['curriculum']); controller.setstate(saved.rng_state['controller'])
                budget = runner._RunnerBudget(120., lambda: 0., None)
                collector = Collector(make_driver(config, config.cases), policy, config,
                    controller_rng=controller, curriculum_rng=curriculum, budget=budget,
                    policy_generator=policy_rng, recovery=recovery, case_sampler=sampler)
                runtime = runner._NeuralTrainingRuntime(config=config, collector=collector, policy=policy,
                    optimizer=optimizer, torch=SimpleNamespace(), policy_generator=policy_rng,
                    shuffle_generator=shuffle, curriculum_rng=curriculum, controller_rng=controller,
                    device_plan=plan_runtime_device(config), device_report={}, budget=budget, recovery=recovery,
                    recovered_progress=progress, parent_checkpoint=checkpoint, stream_transition=stream_transition,
                    controller_id='neural-policy-resume-v1' if purpose == 'resume' else 'neural-policy-v1')
                runtimes.append(runtime)
                return runtime
            def update(policy, optimizer, rollout, ppo_config, *, shuffle_generator, budget):
                self.assertEqual(len(rollout.records), 64)
                before = policy._behavior_policy_version
                for _ in range(4):
                    shuffle_generator.draw(); optimizer.step(); policy.weights['weight'] += .125
                policy._behavior_policy_version += 1
                return UpdateMetrics(0., 0., 0., 4, 4, None, before, before + 1, 64)
            def save(path, *, policy, optimizer, rng_state, metadata):
                self.assertFalse(runtimes[-1]._closed)
                path.mkdir()
                metadata = copy.deepcopy(metadata)
                metadata.update(schema_id=CHECKPOINT_SCHEMA_ID, model_tensors=[dict(
                    name='weight', dtype='float32', shape=[1], sha256='e' * 64)], payload_inventory={})
                for name in ('model.pt', 'training.pt'):
                    _write_archive(path / name)
                    data = (path / name).read_bytes()
                    metadata['payload_inventory'][name] = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                (path / 'metadata.json').write_text(json.dumps(metadata))
                frozen = runner.preflight_checkpoint(path, configs[0].accepted_identities[0], purpose='resume').metadata
                loaded[path] = SimpleNamespace(metadata=frozen, model_tensors=policy.state_dict(),
                    optimizer_state=optimizer.state_dict(), rng_state=copy.deepcopy(rng_state))
            api = SimpleNamespace(make_driver=catalog_driver, create_runtime=create_runtime,
                train=runner.train, load_checkpoint=load_checkpoint, equal=lambda a, b: a == b,
                finite_changed=lambda a, b: a != b, clock=lambda: 0.)
            captured = []
            exercise = held._exercise
            def observe_exercise(test, actual_configs, actual_parent, actual_api, actual_caps):
                self.assertIs(actual_api, api)
                self.assertEqual([c.config_sha256 for c in actual_configs], [c.config_sha256 for c in configs])
                self.assertIs(actual_caps, caps)
                result = exercise(test, actual_configs, actual_parent, actual_api, actual_caps)
                captured.extend(result)
                return result
            @contextmanager
            def fake_api(actual_caps):
                self.assertIs(actual_caps, caps)
                yield api
            with patch.object(runner, 'ppo_update', side_effect=update), \
                 patch.object(runner, 'save_checkpoint', side_effect=save), \
                 patch.object(held, '_runtime_api', side_effect=fake_api), \
                 patch.object(held, '_Caps', return_value=caps), \
                 patch.object(held, '_exercise', side_effect=observe_exercise), \
                 patch.dict(os.environ, {'SPIRE_LEARN_CURRICULUM_INPUT_ROOT': str(root)}):
                # Direct method invocation exercises the enabled body, while its
                # class gate remains disabled for every actual unittest loader.
                held.CurriculumIntegrationTests('test_public_catalog_tensor_resume_and_tiny_healthy_segments').test_public_catalog_tensor_resume_and_tiny_healthy_segments()
            first, second = captured
            self.assertEqual(caps.counts, dict(actions=128, samples=128, adam=8, resets=128))
            self.assertEqual(second['saved_prefix']['delivered_environment_slots'], 128)
            self.assertEqual(second['saved_prefix']['delivered_reset_slots'], 128)
            self.assertEqual(second['saved_prefix']['optimizer_attempted_steps'], 8)
            self.assertEqual([p for p, _ in factory_calls], ['warm_start', 'resume', 'resume'])
            self.assertEqual(len(catalogs), 2)
            self.assertTrue(all(c.close_calls == 1 for c in catalogs))
            self.assertEqual(len(drivers), 3)
            self.assertTrue(all(d.close_calls == 1 for d in drivers))
            self.assertEqual(drivers[1].reset_calls, [])
            self.assertEqual(attempts, [])
            for result in (first, second):
                self.assertEqual(json.loads((result['run_dir'] / 'lifecycle.json').read_text())['status'], 'complete')

    def test_shared_caps_refuse_before_dispatch_and_do_not_refund_attempts(self):
        from tests.neural.test_curriculum import _Caps, _BoundedDriver
        driver = SimpleNamespace(batch_size=4, step=lambda actions: self.fail('over-cap dispatch'),
                                 reset=lambda **kw: self.fail('over-cap reset'))
        caps = _Caps(clock=lambda: 0.)
        caps.charge('actions', 512); caps.charge('resets', 256)
        bounded = _BoundedDriver(driver, caps)
        with self.assertRaisesRegex(RuntimeError, 'actions'):
            bounded.step([0] * 4)
        with self.assertRaisesRegex(RuntimeError, 'resets'):
            bounded.reset(seeds=[0]*4, entry_ids=['x']*4)
        caps.charge('adam', 8); caps.charge('samples', 512)
        for kind in ('adam', 'samples'):
            with self.assertRaises(RuntimeError):
                caps.charge(kind, 1)
        caps.clock = lambda: 121.
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            caps.charge('actions', 0)
