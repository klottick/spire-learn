"""Campaign guest tests use actual producers with strict runtime substitutes."""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.unit.test_neural_curriculum_package import ROOT, ARTIFACTS, package_fixture
from tests.unit.test_neural_curriculum_integration import blocked_optional


class ProducerHarness:
    """Tensor math/native/archive boundaries only; learner producers stay real."""
    def __init__(self, test, root):
        from spire_learn.neural import runner
        self.test, self.root, self.runner = test, root, runner
        self.caps = None
        self.loaded, self.runtimes, self.drivers, self.catalogs = {}, [], [], []
        self.wrong = None
        self.close_error = None
        self.save_error = None
        self.step_error = None
        self.after_close = None
        self.factory_calls = []

    def original(self, parent):
        self.loaded[Path(parent).resolve()] = SimpleNamespace(model_tensors={'weight': 23.0},
            metadata=json.loads((Path(parent) / 'metadata.json').read_text()))

    def load(self, path, identity, *, purpose, device):
        self.test.assertEqual(device, 'cpu')
        self.test.assertIn(purpose, ('warm_start', 'resume'))
        return self.loaded[Path(path).resolve()]

    def fresh(self, config):
        from tests.neural import test_curriculum as held
        from tests.unit.test_neural_stream_resume import StatefulGenerator
        return held._fresh_rng_states(config, lambda *, device: StatefulGenerator(device='cpu'))

    def factory(self, config, *, purpose, checkpoint, policy_kind, action_selection):
        from spire_learn.neural.collector import Collector
        from spire_learn.neural.curriculum import CurriculumSampler
        from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard
        from spire_learn.neural.stream_resume import validate_stream_resume
        from spire_learn.neural.device import plan_runtime_device
        from tests.unit.test_neural_collector import FakePolicy, ScriptedDriver, row, transition, exit_data
        from tests.unit.test_neural_stream_resume import StatefulGenerator
        from tests.neural import test_curriculum as held
        owner, runner = self, self.runner
        self.factory_calls.append((purpose, Path(checkpoint).resolve(), action_selection))
        saved = self.loaded[Path(checkpoint).resolve()]
        recovery, progress, stream_transition, sampler = None, {}, None, None
        if config.scenario_stream is not None:
            recipe = load_scenario_recipe(Path(config.scenario_stream['resolved_recipe_path']))
            shard = generate_scenario_shard(recipe, split='train', arm=config.scenario_stream['arm'],
                                             shard_index=config.scenario_stream['shard_index'])
            sampler_state = None
            if purpose == 'resume':
                recovery, progress = runner._checkpoint_recovery(saved, config)
                state = validate_stream_resume(saved.metadata['run_metadata'], config)
                stream_transition = dict(identity=state['identity'], **{'from': state['current'], 'to': state['destination']})
                sampler_state = state['sampler_state']
            if config.scenario_stream['arm'] == 'mixed':
                sampler = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                             categories=recipe.category_keys, state=sampler_state)
        class Policy(FakePolicy):
            def __init__(self):
                super().__init__(config)
                self._behavior_policy_version = 0
                self.weights = dict(saved.model_tensors)
            def state_dict(self):
                return dict(self.weights)
            def act(self, frames, *, generator, deterministic=False):
                if owner.caps is not None:
                    owner.caps.charge('samples', len(frames))
                if purpose == 'evaluate':
                    owner.test.assertTrue(deterministic)
                else:
                    generator.draw()
                self.behavior_version = self._behavior_policy_version
                return super().act(frames, generator=generator, deterministic=deterministic)
        class Adam:
            def __init__(self):
                self.state, self.param_groups = {}, [{'lr': .0003, 'eps': .00001}]
            def step(self):
                if owner.caps is not None:
                    owner.caps.charge('adam', 1)
                self.state['step'] = self.state.get('step', 0) + 1
                self.state['exp_avg'] = self.state.get('exp_avg', 0.) + .125
            def state_dict(self):
                return copy.deepcopy(dict(state=self.state, param_groups=self.param_groups))
            def load_state_dict(self, value):
                self.state, self.param_groups = copy.deepcopy((value['state'], value['param_groups']))
        def delivered(before, actions):
            if self.step_error is not None:
                raise self.step_error
            return tuple(transition(source, row(44, active=False, entered=1,
                settled=1, exit_value=exit_data()), terminated=True, reward=1.) for source in before)
        driver = ScriptedDriver('standalone', config.rollout_rows,
                                reset_rows=lambda slot, call, seed, entry: row(
                                    10 + (call + slot) % 200, active=True, entered=1, visit=0),
                                steps=[delivered] * (config.rollout_steps * (8 if purpose != 'evaluate' else 1)))
        driver.schema = SimpleNamespace(descriptor_digest='pure-campaign-schema', tensors={})
        self.drivers.append(driver)
        bounded_driver = held._BoundedDriver(driver, self.caps) if self.caps is not None else driver
        policy = Policy()
        if purpose == 'evaluate':
            policy._behavior_policy_version = saved.metadata['training_state']['behavior_policy_version']
        optimizer = None if purpose == 'evaluate' else runner._CountingOptimizer(Adam())
        policy_rng, shuffle = StatefulGenerator(device='cpu'), StatefulGenerator(device='cpu')
        policy_rng.manual_seed(config.seed_streams.policy)
        shuffle.manual_seed(config.seed_streams.optimizer)
        curriculum, controller = random.Random(config.seed_streams.curriculum), random.Random(config.seed_streams.noncombat)
        if purpose == 'resume':
            policy._behavior_policy_version = saved.metadata['training_state']['behavior_policy_version']
            optimizer.load_state_dict(saved.optimizer_state)
            policy_rng.set_state(saved.rng_state['policy']); shuffle.set_state(saved.rng_state['minibatch'])
            curriculum.setstate(saved.rng_state['curriculum']); controller.setstate(saved.rng_state['controller'])
        budget = runner._RunnerBudget(config.budgets.max_wall_seconds, lambda: 0., None)
        collector_policy = runner._DeterministicPolicy(policy) if purpose == 'evaluate' else policy
        collector = Collector(bounded_driver, collector_policy, config,
            controller_rng=controller, curriculum_rng=curriculum, budget=budget,
            policy_generator=policy_rng, recovery=recovery, case_sampler=sampler)
        runtime = runner._NeuralTrainingRuntime(config=config, collector=collector, policy=policy,
            optimizer=optimizer, torch=SimpleNamespace(), policy_generator=policy_rng,
            shuffle_generator=shuffle, curriculum_rng=curriculum, controller_rng=controller,
            device_plan=plan_runtime_device(config), device_report={}, budget=budget, recovery=recovery,
            recovered_progress=progress, parent_checkpoint=checkpoint, stream_transition=stream_transition,
            policy_kind=policy_kind, action_selection=action_selection,
            controller_id='neural-policy-resume-v1' if purpose == 'resume' else 'neural-policy-v1')
        if self.wrong in ('policy', 'minibatch', 'curriculum', 'controller'):
            generator = dict(policy=policy_rng, minibatch=shuffle, curriculum=curriculum, controller=controller)[self.wrong]
            if self.wrong in ('policy', 'minibatch'):
                generator.draw()
            else:
                generator.random()
        elif self.wrong == 'model':
            policy.weights['weight'] += 1
        elif self.wrong == 'Adam':
            optimizer.step()
        elif self.wrong == 'progress':
            runtime.recovered_progress = {'bogus': 1}
        elif self.wrong == 'recovery':
            collector._seed_ordinal = 99999
        elif self.wrong == 'staleconfig':
            from dataclasses import replace
            runtime.config = replace(config, rollout_steps=1)
        elif self.wrong == 'eval-policy-version':
            policy._behavior_policy_version = 0
        original_close = runtime.close
        def close():
            original_close()
            if self.after_close is not None:
                self.after_close()
            if self.close_error is not None:
                raise self.close_error
        runtime.close = close
        self.runtimes.append(runtime)
        return runtime

    def update(self, policy, optimizer, rollout, ppo_config, *, shuffle_generator, budget):
        from spire_learn.neural.ppo import UpdateMetrics
        before = policy._behavior_policy_version
        count = ppo_config.epochs * math.ceil(len(rollout.records) / ppo_config.minibatch_rows)
        for _ in range(count):
            shuffle_generator.draw(); optimizer.step(); policy.weights['weight'] += .125
        policy._behavior_policy_version += 1
        return UpdateMetrics(0., 0., 0., count, count, None, before, before + 1, len(rollout.records))

    def save(self, path, *, policy, optimizer, rng_state, metadata):
        from tests.unit.test_neural_checkpoint import _write_archive
        from spire_learn.neural.checkpoint import CHECKPOINT_SCHEMA_ID
        if self.save_error is not None:
            raise self.save_error
        self.test.assertFalse(self.runtimes[-1]._closed)
        path.mkdir()
        metadata = copy.deepcopy(metadata)
        metadata.update(schema_id=CHECKPOINT_SCHEMA_ID, model_tensors=[dict(
            name='weight', dtype='float32', shape=[1], sha256='e' * 64)], payload_inventory={})
        for name in ('model.pt', 'training.pt'):
            _write_archive(path / name)
            data = (path / name).read_bytes()
            metadata['payload_inventory'][name] = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (path / 'metadata.json').write_text(json.dumps(metadata))
        config = self.runtimes[-1].config
        frozen = self.runner.preflight_checkpoint(path, config.accepted_identities[0], purpose='resume').metadata
        self.loaded[path.resolve()] = SimpleNamespace(metadata=frozen, model_tensors=policy.state_dict(),
            optimizer_state=optimizer.state_dict(), rng_state=copy.deepcopy(rng_state))

    def catalog(self, config, cases):
        from spire_learn.neural import envs
        from tests.neural import test_curriculum as held
        owner = self
        class Catalog:
            def __init__(self, rows, catalog, *, partition):
                owner.test.assertEqual((rows, partition, len(catalog['entries'])), (4, 'train', 256))
                self.entries, self.close_calls = catalog['entries'], 0
                owner.catalogs.append(self)
            def observation_spec(self):
                return {'synthetic': 'native descriptor boundary'}
            def environment_spec(self):
                return dict(profile_id='ironclad-combat-v1', schema_id=envs.COMMON_VIEW_ID,
                    observation_fingerprint=config.common_view_fingerprint,
                    content_id='spire-gym-ironclad-combat-content-v1', rules_id='spire-gym-native-m10-rules-v1',
                    reward_id='spire-gym-combat-outcome-reward-v1', mode='standalone_combat', partition='train', autoreset='disabled')
            def catalog_descriptor(self):
                return dict(catalog_schema_id='spire-gym-combat-entry-catalog-v1',
                    entries=[envs._normalized_entry(e) for e in self.entries])
            def reset(self, **kwargs):
                owner.test.fail('catalog probe reset')
            def step(self, actions):
                owner.test.fail('catalog probe step')
            def close(self):
                self.close_calls += 1
        native = SimpleNamespace(CombatInputError=ValueError, RunInputError=ValueError,
            CombatStateError=RuntimeError, RunStateError=RuntimeError,
            build_info=lambda: dict(extension='spire-python', revision=config.environment_revision))
        facade = envs._NativeRuntime(SimpleNamespace(), native, Catalog, object)
        schema = SimpleNamespace(schema_fingerprint=config.common_view_fingerprint, action_space=SimpleNamespace(n=10))
        with patch.object(envs, '_load_native_runtime', return_value=facade), \
             patch.object(envs, 'parse_common_descriptor', return_value=schema):
            return held._BoundedDriver(envs.make_driver(config, cases), self.caps)

    @contextmanager
    def qualification_api(self, caps):
        self.caps = caps
        try:
            yield SimpleNamespace(make_driver=self.catalog, create_runtime=self.factory,
                train=self.runner.train, load_checkpoint=self.load, equal=lambda a, b: a == b,
                finite_changed=lambda a, b: a != b, clock=caps.clock, fresh_rng_states=self.fresh)
        finally:
            self.caps = None

    def dependencies(self):
        return dict(factory=self.factory, load_checkpoint=self.load, equal=lambda a, b: a == b,
            fresh_rng_states=self.fresh, qualification_api=self.qualification_api,
            threads=lambda: {'intraop': 4, 'interop': 1})

    @contextmanager
    def producers(self):
        with blocked_optional(), patch.object(self.runner, 'ppo_update', side_effect=self.update), \
             patch.object(self.runner, 'save_checkpoint', side_effect=self.save):
            yield


class CampaignProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path, cls.package = package_fixture()

    def setUp(self):
        from scripts.neural import curriculum_runtime as api
        from scripts.neural.curriculum_package import inventory
        self.api = api
        self.temporary = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'learner'
        self.root.mkdir()
        self.input_root = self.root / 'inputs/package'
        shutil.copytree(self.fixture_path, self.input_root, copy_function=os.link)
        for relative in api.source_paths(ROOT):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / relative, target)
        self.source_inventory = inventory(self.root, api.source_paths(self.root))
        original = self.package['original_parent']
        self.parent = self.root / original['checkpoint_directory']
        shutil.copytree(ROOT / original['checkpoint_directory'], self.parent, copy_function=os.link)
        self.harness = ProducerHarness(self, self.root)
        self.harness.original(self.parent)
        self.now = 0.

    def context(self, kind='segment', *, index=0, arm='mixed', prefix=None, parent=None, predecessor=None, case_ids=None):
        from scripts.neural.curriculum_package import inventory, sha256
        package = self.package
        input_root = self.input_root
        if kind == 'segment':
            binding = package['training'][arm][index]
            names = binding['inputs']
            output = input_root / f'training/{arm}/runs'
            extra = dict(arm=arm, shard_index=index, prefix=prefix or {},
                max_updates=(prefix or {}).get('update_count', 0) + 1,
                max_environment_actions=(prefix or {}).get('attempted_environment_slots', 0) + 4096,
                predecessor=predecessor)
            reservation = dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
                               reset_slots=4112, fixed_random_choices=0)
        elif kind == 'panel':
            names = sorted({package['panel']['path']} | {name for row in package['panel']['cases'] for name in row['inputs']})
            output = input_root / 'panel/runs'
            extra = dict(slot='original-b', case_ids=case_ids or [package['panel']['cases'][0]['case_id']])
            reservation = dict(environment_attempts=64*256, sample_rows=0, optimizer_attempts=0,
                               reset_slots=64, fixed_random_choices=0)
        else:
            source = ROOT / 'artifacts/development/neural-combat/curriculum-v1/task-5/prepared-mixed'
            input_root = self.root / source.relative_to(ROOT)
            if not input_root.exists():
                shutil.copytree(source, input_root, copy_function=os.link)
            names = sorted(p.name for p in input_root.iterdir() if p.is_file())
            output = input_root / 'runs'
            extra = {}
            reservation = dict(environment_attempts=512, sample_rows=512, optimizer_attempts=8,
                               reset_slots=256, fixed_random_choices=0)
        output.mkdir(exist_ok=True)
        return dict(schema=self.api.CONTEXT_SCHEMA, campaign_id='curriculum-20261101-a',
            kind=kind, job_id=f'curriculum-test-{kind}-{index}', runtime_class='cpu' if kind == 'panel' else 'gpu',
            package_path=str((self.input_root / 'package.json').relative_to(self.root)),
            package_sha256=sha256(self.input_root / 'package.json'), input_root=str(input_root.relative_to(self.root)),
            input_inventory=inventory(input_root, names), source_inventory=self.source_inventory,
            source_sha256='1'*64, release_sha256='2'*64, claim_sha256='3'*64,
            parent=parent or dict(path=str(self.parent), inventory=package['original_parent']['inventory']),
            output_root=str(output), deadline=self.now + (100 if kind == 'qualification' else 400),
            whole_deadline=self.now + 500, reservation=reservation, **extra)

    def entry(self, context, *, dependencies=None):
        path = self.root / 'job-context.json'
        path.write_text(json.dumps(context))
        verifier = Mock(side_effect=lambda root, path, **kwargs: json.loads(path.read_text()))
        controls = ModuleType('scripts.neural.curriculum_controls')
        controls.verify_guest_context = verifier
        with patch.dict(sys.modules, {'scripts.neural.curriculum_controls': controls}):
            try:
                return self.api._guest_entry(self.root, path,
                    dependencies=dependencies or self.harness.dependencies(), clock=lambda: self.now)
            finally:
                verifier.assert_called_once()

    def test_projected_qualification_warm_start_save_next_shard_panel_actual_producers(self):
        with self.harness.producers():
            qualification = self.entry(self.context('qualification'))
            q = json.loads(Path(qualification['receipt_path']).read_text())
            self.assertEqual(q['qualification_attempts'], dict(actions=128, samples=128, adam=8, resets=128))
            self.assertTrue(all(row['uncertain'] == 0 for row in q['effects'].values()))
            self.assertEqual(len(self.harness.catalogs), 2)
            self.assertTrue(all(d.close_calls == 1 for d in self.harness.catalogs))
            self.assertEqual(self.harness.factory_calls[0][1], self.parent)
            first = self.entry(self.context())
            self.assertEqual(self.harness.factory_calls[-1][:2], ('warm_start', self.parent))
            receipt_path = Path(first['receipt_path'])
            initial = json.loads(receipt_path.read_text())
            self.assertEqual(initial['prefix'], {})
            self.assertEqual(initial['checkpoint']['counters']['update_count'], 1)
            self.assertEqual(initial['effects']['environment_attempts']['attempted'], 512)
            # Preserve first output and expose its checkpoint through the next
            # fresh output overlay; physical parent bytes stay immutable.
            old_output = receipt_path.parent
            retained = self.root / 'retained-first'
            old_output.rename(retained)
            old_output.mkdir()
            saved_relative = Path(initial['checkpoint']['path']).relative_to(old_output)
            parent = retained / saved_relative
            self.harness.loaded[parent.resolve()] = self.harness.loaded[Path(initial['checkpoint']['path']).resolve()]
            original_receipt = retained / 'job-receipt.json'
            successor_context = self.context(index=1, prefix=initial['checkpoint']['counters'],
                parent=dict(path=str(parent), inventory=initial['checkpoint']['inventory']),
                predecessor=dict(path=str(original_receipt.relative_to(self.root)),
                                 sha256=hashlib.sha256(original_receipt.read_bytes()).hexdigest()))
            second = self.entry(successor_context)
            successor = json.loads(Path(second['receipt_path']).read_text())
            self.assertEqual(successor['checkpoint']['counters']['update_count'], 2)
            self.assertEqual(successor['delta']['attempted_environment_slots'], 512)
            panel_context = self.context('panel', parent=successor['checkpoint'] | {})
            panel_context['parent'] = {k: successor['checkpoint'][k] for k in ('path', 'inventory')}
            panel = self.entry(panel_context)
            evaluated = json.loads(Path(panel['receipt_path']).read_text())
            self.assertEqual(len(evaluated['reports']), 1)
            self.assertEqual(evaluated['reports'][0]['exit_code'], 0, evaluated['reports'])
            self.assertEqual(len(evaluated['missing_case_ids']), 63)
            self.assertEqual(evaluated['effects']['environment_attempts']['attempted'], 1)
            self.assertEqual(evaluated['effects']['sample_rows']['attempted'], 0)
            self.assertEqual(evaluated['effects']['optimizer_attempts']['attempted'], 0)
        self.assertTrue(all(driver.close_calls == 1 for driver in self.harness.drivers))

    def test_each_warm_state_mismatch_rejected_before_environment_effects(self):
        for wrong in ('model', 'Adam', 'policy', 'minibatch', 'curriculum', 'controller', 'progress', 'recovery', 'staleconfig'):
            with self.subTest(wrong=wrong):
                context = self.context()
                self.harness.wrong = wrong
                with self.harness.producers(), self.assertRaisesRegex(ValueError, 'factory state differs'):
                    self.entry(context)
                driver = self.harness.drivers[-1]
                self.assertEqual(driver.reset_calls + driver.step_calls, [])
                self.assertEqual(driver.close_calls, 1)
                shutil.rmtree(context['output_root'])

    def test_fixed_and_randomized_use_real_nonsampler_runtime_and_successor(self):
        from scripts.neural.curriculum_package import sha256
        for arm in ('fixed', 'randomized'):
            with self.subTest(arm=arm), self.harness.producers():
                context = self.context(arm=arm)
                first = self.entry(context)
                receipt_path = Path(first['receipt_path'])
                receipt = json.loads(receipt_path.read_text())
                output = receipt_path.parent
                retained = self.root / ('retained-' + arm)
                output.rename(retained)
                output.mkdir()
                parent = retained / Path(receipt['checkpoint']['path']).relative_to(output)
                self.harness.loaded[parent.resolve()] = self.harness.loaded[Path(receipt['checkpoint']['path']).resolve()]
                predecessor = retained / 'job-receipt.json'
                successor = self.entry(self.context(arm=arm, index=1,
                    prefix=receipt['checkpoint']['counters'],
                    parent=dict(path=str(parent), inventory=receipt['checkpoint']['inventory']),
                    predecessor=dict(path=str(predecessor.relative_to(self.root)), sha256=sha256(predecessor))))
                resumed = json.loads(Path(successor['receipt_path']).read_text())
                self.assertEqual(resumed['checkpoint']['counters']['update_count'], 2)
                self.assertEqual(resumed['delta']['attempted_environment_slots'], 512)

    def test_resume_rejects_rebound_foreign_source_receipt_before_factory(self):
        from scripts.neural.curriculum_package import sha256
        with self.harness.producers():
            first = self.entry(self.context())
            receipt_path = Path(first['receipt_path'])
            receipt = json.loads(receipt_path.read_text())
            output = receipt_path.parent
            retained = self.root / 'retained-lineage'
            output.rename(retained)
            output.mkdir()
            parent = retained / Path(receipt['checkpoint']['path']).relative_to(output)
            self.harness.loaded[parent.resolve()] = self.harness.loaded[Path(receipt['checkpoint']['path']).resolve()]
            # A negative adversarial receipt is derived from a real producer;
            # no hand-authored receipt authorizes the positive flow.
            receipt['source_sha256'] = '9' * 64
            foreign = retained / 'foreign-source-receipt.json'
            foreign.write_text(json.dumps(receipt))
            context = self.context(index=1, prefix=receipt['checkpoint']['counters'],
                parent=dict(path=str(parent), inventory=receipt['checkpoint']['inventory']),
                predecessor=dict(path=str(foreign.relative_to(self.root)), sha256=sha256(foreign)))
            with self.assertRaisesRegex(ValueError, 'predecessor'):
                self.entry(context)
            self.assertEqual(len(self.harness.factory_calls), 1)

    def test_malformed_deadlines_limits_identity_and_source_reject_before_factory(self):
        original = self.context()
        mutations = [dict(deadline=0), dict(deadline=501), dict(whole_deadline=601),
            dict(max_updates=9), dict(max_environment_actions=4097), dict(source_sha256='x'),
            dict(runtime_class='cpu'), dict(extra=True), dict(prefix={'update_count': 1}),
            dict(package_sha256='0'*64), dict(input_inventory={}), dict(shard_index=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.harness.producers(), self.assertRaises((ValueError, TypeError)):
                self.entry({**original, **mutation})
            failure = Path(original['output_root']) / 'guest-failure.json'
            if failure.exists():
                failure.unlink()
        self.assertEqual(self.harness.factory_calls, [])

    def test_caps_faults_save_failures_and_primary_controls_close_once(self):
        scenarios = ('sample-cap', 'adam-cap', 'reset-cap', 'native-fault', 'save-fault', 'control')
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                context = self.context()
                if scenario == 'sample-cap':
                    context['reservation']['sample_rows'] = 0
                elif scenario == 'adam-cap':
                    context['reservation']['optimizer_attempts'] = 0
                elif scenario == 'reset-cap':
                    context['reservation']['reset_slots'] = 0
                elif scenario == 'native-fault':
                    self.harness.step_error = RuntimeError('native-fault')
                elif scenario == 'save-fault':
                    self.harness.save_error = RuntimeError('save-fault')
                elif scenario == 'control':
                    self.harness.step_error = KeyboardInterrupt('primary control')
                    self.harness.close_error = RuntimeError('secondary close')
                expected = KeyboardInterrupt if scenario == 'control' else Exception
                with self.harness.producers(), self.assertRaises(expected):
                    self.entry(context)
                self.assertEqual(self.harness.drivers[-1].close_calls, 1)
                self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())
                counters = json.loads((Path(context['output_root']) / 'campaign-control/attempts.json').read_text())
                self.assertIsNotNone(counters['error'])
                if scenario in ('native-fault', 'control'):
                    self.assertGreater(counters['effects']['environment_attempts']['uncertain'], 0)
                self.harness.step_error = self.harness.save_error = self.harness.close_error = None
                shutil.rmtree(context['output_root'])

    def test_late_close_never_publishes_passing_receipt(self):
        context = self.context()
        self.harness.after_close = lambda: setattr(self, 'now', 501.)
        with self.harness.producers(), self.assertRaises(TimeoutError):
            self.entry(context)
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)
        self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())

    def test_qualification_narrows_shared_deadline_and_archives_attempts_on_failure(self):
        context = self.context('qualification')
        context['deadline'] = 1.
        real_api = self.harness.qualification_api
        @contextmanager
        def late(caps):
            self.assertEqual(caps.deadline, 1.)
            self.now = 2.
            with real_api(caps) as api:
                yield api
        dependencies = self.harness.dependencies()
        dependencies['qualification_api'] = late
        with self.harness.producers(), self.assertRaises((RuntimeError, TimeoutError, AssertionError)):
            self.entry(context, dependencies=dependencies)
        self.assertEqual(self.harness.drivers[0].reset_calls + self.harness.drivers[0].step_calls, [])
        self.assertEqual(self.harness.drivers[0].close_calls, 1)
        counters = json.loads((Path(context['output_root']) / 'campaign-control/attempts.json').read_text())
        self.assertEqual(counters['effects']['environment_attempts']['attempted'], 0)

    def test_action_high_water_stops_normally_at_actual_short_checkpoint(self):
        context = self.context()
        context['reservation']['environment_attempts'] = 16
        context['max_environment_actions'] = 16
        with self.harness.producers():
            published = self.entry(context)
        receipt = json.loads(Path(published['receipt_path']).read_text())
        self.assertEqual(receipt['checkpoint']['counters']['attempted_environment_slots'], 16)
        self.assertEqual(receipt['effects']['environment_attempts']['attempted'], 16)
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)

    def test_full_eight_update_job_obeys_4096_actions_and_256_adam_caps(self):
        context = self.context()
        context['max_updates'] = 8
        with self.harness.producers():
            published = self.entry(context)
        receipt = json.loads(Path(published['receipt_path']).read_text())
        saved = receipt['checkpoint']['counters']
        self.assertEqual((saved['update_count'], saved['attempted_environment_slots'],
                          saved['learned_decisions'], saved['optimizer_attempted_steps']),
                         (8, 4096, 4096, 256))
        for kind, count in (('environment_attempts', 4096), ('sample_rows', 4096), ('optimizer_attempts', 256)):
            self.assertEqual(receipt['effects'][kind]['attempted'], count)
            self.assertEqual(receipt['effects'][kind]['uncertain'], 0)
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)

    def test_settlement_process_control_supersedes_fault_and_preserves_first_control(self):
        from scripts.neural.sustained_runtime import Journal
        original = Journal.settle
        for primary in (RuntimeError('ordinary native fault'), KeyboardInterrupt('first control')):
            context = self.context()
            self.harness.step_error = primary
            later = SystemExit('settlement control')
            def settle(journal, row):
                if row['label'] == 'native-step':
                    raise later
                return original(journal, row)
            with self.harness.producers(), patch.object(Journal, 'settle', settle):
                with self.assertRaises(SystemExit if isinstance(primary, Exception) else KeyboardInterrupt) as caught:
                    self.entry(context)
            self.assertIs(caught.exception, later if isinstance(primary, Exception) else primary)
            self.assertEqual(self.harness.drivers[-1].close_calls, 1)
            self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())
            shutil.rmtree(context['output_root'])

    def test_earlier_settlement_control_reaches_runner_before_later_close_control(self):
        from scripts.neural.sustained_runtime import Journal
        context = self.context()
        original = Journal.settle
        self.harness.step_error = RuntimeError('ordinary native fault')
        primary = KeyboardInterrupt('earlier settlement control')
        self.harness.close_error = SystemExit('later runtime close control')
        def settle(journal, row):
            if row['label'] == 'native-step':
                raise primary
            return original(journal, row)
        with self.harness.producers(), patch.object(Journal, 'settle', settle):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.entry(context)
        self.assertIs(caught.exception, primary)
        self.assertTrue(any('later runtime close control' in note for note in primary.__notes__))
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)
        self.assertEqual(len(self.harness.drivers[-1].step_calls), 1)
        self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())

    def test_setup_assertion_promotes_close_control_and_retains_original_failure(self):
        context = self.context()
        self.harness.wrong = 'model'
        self.harness.close_error = KeyboardInterrupt('setup close control')
        with self.harness.producers(), self.assertRaises(KeyboardInterrupt) as caught:
            self.entry(context)
        self.assertIs(caught.exception, self.harness.close_error)
        self.assertTrue(any('factory state differs' in note for note in caught.exception.__notes__))
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)

    def test_qualification_settlement_preserves_first_process_control(self):
        from scripts.neural.sustained_runtime import Journal
        original = Journal.settle
        context = self.context('qualification')
        primary = KeyboardInterrupt('first qualification control')
        self.harness.step_error = primary
        def settle(journal, row):
            if row['label'] == 'qualification-actions':
                raise SystemExit('later qualification settlement control')
            return original(journal, row)
        with self.harness.producers(), patch.object(Journal, 'settle', settle):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.entry(context)
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)
        self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())

    def test_panel_setup_and_bad_summary_close_once_with_primary_control(self):
        from spire_learn.neural.runner import NeuralReport
        scenarios = ('optimizer', 'serialization', 'control')
        for scenario in scenarios:
            context = self.context('panel')
            dependencies = self.harness.dependencies()
            if scenario == 'optimizer':
                factory = self.harness.factory
                def wrong_factory(*args, **kwargs):
                    runtime = factory(*args, **kwargs)
                    runtime.optimizer = object()
                    return runtime
                dependencies['factory'] = wrong_factory
            elif scenario == 'serialization':
                def bad_json(*args, **kwargs):
                    report = self.harness.runner.evaluate_policy(*args, **kwargs)
                    self.assertEqual(report.exit_code, 0)
                    report.summary = {**report.summary, 'unserializable_metric': float('nan')}
                    return report
                dependencies['evaluate'] = bad_json
            else:
                def control(*args, **kwargs):
                    raise KeyboardInterrupt('panel primary control')
                dependencies['evaluate'] = control
                self.harness.close_error = RuntimeError('panel secondary close')
            with self.harness.producers(), self.assertRaises(KeyboardInterrupt if scenario == 'control' else (TypeError, ValueError)):
                self.entry(context, dependencies=dependencies)
            self.assertEqual(self.harness.drivers[-1].close_calls, 1)
            self.assertFalse((Path(context['output_root']) / 'job-receipt.json').exists())
            self.harness.close_error = None
            shutil.rmtree(context['output_root'])

    def test_panel_missing_metrics_remain_unavailable_and_failures_are_explicit(self):
        from spire_learn.neural.runner import NeuralReport
        context = self.context('panel')
        dependencies = self.harness.dependencies()
        dependencies['evaluate'] = lambda *args, **kwargs: NeuralReport(
            {'status': 'setup_blocked', 'mode': 'standalone', 'reason': 'strict fake setup failure'}, 2, None)
        with self.harness.producers(), patch.object(self.api, '_ADMITTED_CONTEXT', context):
            receipt = self.api.run_panel(self.root, context, dependencies=dependencies, clock=lambda: self.now)
        self.assertEqual(receipt['summary']['status'], 'operational_failure')
        self.assertEqual(receipt['status'], 'operational_failure')
        self.assertEqual(len(receipt['summary']['failures']), 1)
        self.assertEqual(len(receipt['missing_case_ids']), 63)
        cohorts = receipt['summary']['strata']['cohort']
        self.assertEqual(sum(row['completed'] for row in cohorts.values()), 0)
        self.assertTrue(all(row['mean_hp_retention'] is None for row in cohorts.values()))
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)

    def test_actual_panel_contract_fault_retains_failed_receipt_and_stops_next_case(self):
        context = self.context('panel', case_ids=[row['case_id'] for row in self.package['panel']['cases'][:2]])
        self.harness.wrong = 'eval-policy-version'
        with self.harness.producers(), self.assertRaisesRegex(RuntimeError, 'operational_failure'):
            self.entry(context)
        receipt = json.loads((Path(context['output_root']) / 'job-receipt.json').read_text())
        self.assertEqual(receipt['status'], 'operational_failure')
        self.assertEqual(receipt['summary']['status'], 'operational_failure')
        self.assertEqual(len(receipt['reports']), 1)
        self.assertEqual(receipt['reports'][0]['report']['status'], 'operational_failure')
        self.assertEqual(len(self.harness.factory_calls), 1)
        self.assertEqual(self.harness.drivers[-1].close_calls, 1)

    def test_complete_panel_inventory_and_parent_preflight_before_factory(self):
        context = self.context('panel')
        omitted = self.package['panel']['cases'][-1]['config_path']
        del context['input_inventory'][omitted]
        with self.harness.producers(), patch.object(self.api, '_ADMITTED_CONTEXT', context), \
             self.assertRaisesRegex(ValueError, 'complete panel inventory'):
            self.api.run_panel(self.root, context, dependencies=self.harness.dependencies(), clock=lambda: self.now)
        self.assertEqual(self.harness.factory_calls, [])

    def test_unadmitted_default_boundary_never_resolves_optional_packages(self):
        context = self.context()
        with blocked_optional(), self.assertRaisesRegex(RuntimeError, 'verified guest entry'):
            self.api.run_segment(self.root, context, clock=lambda: self.now)

    def test_named_dependency_mapping_cannot_admit_any_unverified_operation(self):
        for kind, operation in (('segment', self.api.run_segment),
                                ('qualification', self.api.run_qualification),
                                ('panel', self.api.run_panel)):
            context = self.context(kind)
            for dependencies in (None, {}, self.harness.dependencies()):
                with self.subTest(kind=kind, dependencies=None if dependencies is None else sorted(dependencies)), blocked_optional(), \
                     patch.object(self.api, '_dependencies', side_effect=AssertionError('unadmitted dependency resolution')), \
                     patch.object(self.api, '_ledger', side_effect=AssertionError('unadmitted cap creation')):
                    with self.assertRaisesRegex(RuntimeError, 'verified guest entry'):
                        operation(self.root, context, dependencies=dependencies, clock=lambda: self.now)
            self.assertEqual(self.harness.factory_calls, [])
            self.assertEqual(list(Path(context['output_root']).iterdir()), [])

    def test_removed_admission_hook_cannot_unlock_default_producer(self):
        context = self.context()
        path = self.root / 'unverified-context.json'
        path.write_text(json.dumps(context))
        hook = lambda *args, **kwargs: context
        with blocked_optional(), patch.object(self.api, 'run_segment',
                side_effect=AssertionError('unverified producer reached')) as producer:
            with self.assertRaises(TypeError):
                self.api._guest_entry(self.root, path, clock=lambda: self.now, admission=hook)
            producer.assert_not_called()
        self.assertIsNone(self.api._ADMITTED_CONTEXT)

    def test_failed_real_verifier_never_opens_default_runtime_boundary(self):
        context = self.context()
        path = self.root / 'rejected-context.json'
        path.write_text(json.dumps(context))
        controls = ModuleType('scripts.neural.curriculum_controls')
        controls.verify_guest_context = Mock(side_effect=PermissionError('host admission rejected'))
        with blocked_optional(), patch.dict(sys.modules, {'scripts.neural.curriculum_controls': controls}), \
             patch.object(self.api, 'run_segment') as producer, \
             patch.object(self.api, '_dependencies') as dependencies:
            with self.assertRaisesRegex(PermissionError, 'host admission rejected'):
                self.api._guest_entry(self.root, path, clock=lambda: self.now)
            controls.verify_guest_context.assert_called_once()
            producer.assert_not_called()
            dependencies.assert_not_called()
        self.assertIsNone(self.api._ADMITTED_CONTEXT)
        self.assertEqual(list(Path(context['output_root']).iterdir()), [])

    def test_parent_mount_projection_uses_exact_current_parent_and_wire_alias(self):
        from scripts.neural import task9_study_controls as low
        mounted = self.root / 'inputs/parent'
        shutil.copytree(self.parent, mounted, copy_function=os.link)
        self.harness.loaded[mounted.resolve()] = self.harness.loaded[self.parent.resolve()]
        context = self.context(parent=dict(path='/inputs/parent', inventory=self.package['original_parent']['inventory']))
        context['reservation']['environment_attempts'] = 16
        context['max_environment_actions'] = 16
        with self.harness.producers(), patch.object(low, '_guest_path', side_effect=lambda value: mounted if value == '/inputs/parent' else self.fail('unexpected namespace')):
            result = self.entry(context)
        receipt = json.loads(Path(result['receipt_path']).read_text())
        self.assertEqual(receipt['parent']['path'], '/inputs/parent')
        metadata = json.loads((Path(receipt['checkpoint']['path']) / 'metadata.json').read_text())
        self.assertEqual(metadata['parent_checkpoint'], {'path': str(mounted)})
        self.assertEqual(self.harness.factory_calls[0][1], mounted)

    def test_actual_projected_entry_imports_and_help_discovery_block_all_six_backends(self):
        # Import from the staged tree in a fresh interpreter, so the repository's
        # already imported modules cannot disguise a missing mount/import path.
        code = r'''
import builtins, importlib.abc, json, runpy, sys, unittest
prefixes=('torch','numpy','tensordict','torchrl','gymnasium','spire_lab')
attempts=[]
def forbidden(name):
    return any(name==p or name.startswith(p+'.') for p in prefixes)
original=builtins.__import__
def guard(name,*args,**kwargs):
    if forbidden(name):
        attempts.append(name); raise AssertionError('optional import: '+name)
    return original(name,*args,**kwargs)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if forbidden(fullname):
            attempts.append(fullname); raise AssertionError('optional resolution: '+fullname)
sys.meta_path.insert(0,Block()); builtins.__import__=guard
from scripts.neural import curriculum_package, curriculum_runtime
assert curriculum_runtime.__file__.startswith(sys.argv[1])
from scripts.neural.sustained_controls import signal_scope
from scripts.neural.task9_study_controls import _guest_path
assert sys.modules['scripts.neural.sustained_controls'].__file__.startswith(sys.argv[1])
assert sys.modules['scripts.neural.task8_gpu_setup_helpers'].__file__.startswith(sys.argv[1])
for name in ('curriculum_package','curriculum_runtime'):
    sys.argv=[name,'--help']
    try: runpy.run_path('scripts/neural/'+name+'.py',run_name='__main__')
    except SystemExit as error: assert error.code==0
suite=unittest.defaultTestLoader.discover('tests/neural',pattern='test_curriculum.py')
result=unittest.TestResult(); suite.run(result)
assert result.testsRun==1 and len(result.skipped)==1 and not result.errors and not result.failures
assert not attempts and not any(forbidden(name) for name in sys.modules)
print('projected entry/help/discovery; optional attempts/loaded: []')
'''
        environment = {key: value for key, value in os.environ.items() if not key.startswith('SPIRE_LEARN_RUN_')}
        environment['PYTHONPATH'] = str(self.root / 'src') + ':' + str(self.root)
        child = subprocess.run([sys.executable, '-B', '-c', code, str(self.root)],
            cwd=self.root, env=environment, text=True, capture_output=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('optional attempts/loaded: []', child.stdout)


class RuntimeContractTests(unittest.TestCase):
    def test_unverified_context_rejects_before_runtime_boundary(self):
        api = importlib.import_module('scripts.neural.curriculum_runtime')
        for run in (api.run_qualification, api.run_segment, api.run_panel):
            with self.assertRaisesRegex(RuntimeError, 'verified guest entry'):
                run('.', {}, dependencies={'factory': lambda: self.fail('runtime effect')})


if __name__ == '__main__':
    unittest.main()
