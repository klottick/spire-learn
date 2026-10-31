"""Preparation, panel denominators and strict one-segment dispatch."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural.config import load_neural_config
from spire_learn.neural.runner import NeuralReport, RuntimeOperationLimits
from spire_learn.neural.scenario_stream import load_scenario_recipe, _composition_split
from spire_learn.neural.scenario_support import content_sha256

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/development/neural-combat'
RECIPE = ROOT / 'configs/neural/curriculum-v1/recipe.json'


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = load_scenario_recipe(RECIPE)

    def test_prepare_roundtrips_real_config_inventory_and_entire_cycle(self):
        from spire_learn.neural.stream_runner import prepare_stream
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            destination = Path(temporary) / 'prepared'
            result = prepare_stream(RECIPE, destination, arm='mixed', split='train', shard_index=2)
            config = load_neural_config(destination / 'config.toml')
            self.assertEqual(config.scenario_stream['shard_index'], 2)
            self.assertEqual(config.manifest_sha256, result['manifest_sha256'])
            self.assertEqual(config.config_sha256, result['config_sha256'])
            self.assertEqual(config.output_root, destination / 'runs')
            self.assertEqual(result['coverage']['base_upgraded_count'], 150)
            self.assertEqual(result['coverage']['encounter_count'], 43)
            self.assertEqual(len(result['cycle']), 4)
            for name, binding in result['inventory'].items():
                self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), binding['raw_sha256'])
            self.assertFalse(any(p.suffix == '.pt' for p in destination.rglob('*')))
            self.assertLess((destination / 'preparation.json').stat().st_size, 32768)

    def test_infeasible_recipe_and_foreign_output_fail_without_writes(self):
        from spire_learn.neural.stream_runner import prepare_stream
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            root = Path(temporary)
            payload = json.loads(RECIPE.read_text())
            payload['support_file'] = str(RECIPE.with_name('support.json'))
            payload['shard_size'] = 8
            path = root / 'tiny.json'
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                prepare_stream(path, root / 'absent', arm='mixed', split='train', shard_index=0)
            self.assertFalse((root / 'absent').exists())
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                prepare_stream(RECIPE, Path(temporary) / 'absent', arm='fixed', split='train', shard_index=0)
            self.assertFalse((Path(temporary) / 'absent').exists())

    def test_fixed_control_and_selection_configs_use_the_real_loader(self):
        from spire_learn.neural.stream_runner import prepare_stream
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            for arm, split, expected in [('fixed', 'train', 78), ('randomized', 'selection', 256)]:
                destination = Path(temporary) / arm
                result = prepare_stream(RECIPE, destination, arm=arm, split=split, shard_index=0)
                config = load_neural_config(destination / 'config.toml')
                self.assertEqual(len(config.cases.cases), expected)
                self.assertEqual(config.operation, 'train' if split == 'train' else 'evaluate')
                self.assertEqual(config.scenario_stream is None, split == 'selection')
                self.assertEqual(result['coverage']['broad_coverage'], arm != 'fixed')

    def test_panel_disjointness_and_public_support_are_explicit(self):
        from spire_learn.neural.stream_runner import build_development_panel
        panel = build_development_panel(self.recipe)
        self.assertEqual(panel, build_development_panel(self.recipe))
        cases = panel['manifest']['cases']
        self.assertEqual(len({c['environment_seed'] for c in cases}), 64)
        self.assertEqual(len({c['lineage_id'] for c in cases}), 64)
        for cohort in ('familiar_new_seed', 'novel_deck', 'reserved_pair', 'resource_later_tier'):
            self.assertEqual(sum(c['cohort'] == cohort for c in cases), 16)
        for case in cases:
            self.recipe.support.validate_entry(case['entry'])
            self.assertGreaterEqual(case['environment_seed'], 2**40)
            annotation = panel['annotations'][case['case_id']]
            if case['cohort'] in ('novel_deck', 'reserved_pair'):
                self.assertEqual(_composition_split(case['entry']['deck']), 'selection')
                self.assertEqual(annotation['composition_scope'], 'novel_composition')
            else:
                self.assertEqual(annotation['composition_scope'], 'seen_composition_diagnostic')
            if case['cohort'] == 'reserved_pair':
                ids = {r['card'] for r in case['entry']['deck']}
                self.assertTrue(any(set(pair) <= ids for pair in self.recipe.reserved_pairs))

    def test_pilot_arms_share_full_parent_inventory_and_initialization(self):
        from spire_learn.neural.stream_runner import pilot_plan
        plan = pilot_plan(self.recipe)
        self.assertFalse(plan['execution_authorized'])
        self.assertEqual(set(plan['arms']), {'fixed', 'randomized', 'mixed'})
        self.assertEqual(len({content_sha256(x['initialization']) for x in plan['arms'].values()}), 1)
        for arm in plan['arms'].values():
            self.assertEqual(set(arm['parent']['inventory']), {'model.pt', 'training.pt', 'metadata.json'})
            self.assertIn('/standalone-b/', arm['parent']['checkpoint_directory'])

    def test_stratified_reports_preserve_faults_missing_and_resource_denominators(self):
        from spire_learn.neural.stream_runner import build_development_panel, summarize_development_reports
        panel = build_development_panel(self.recipe)
        ids = [c['case_id'] for c in panel['manifest']['cases']]
        healthy = NeuralReport({'status': 'complete', 'mode': 'standalone',
            'case_accounting': {'cases': [{'case_id': ids[0], 'status': 'complete'}]},
            'fights': {'standalone_fights': {'reached_combat_denominator': 1,
                'completed': 1, 'victories': 1, 'deaths': 0, 'fights': [
                    {'hp_before_healing': 40, 'hp_after_cleanup': 46, 'hp_retention': .575}]}},
            'collection': {'uncertain_environment_slots': 0}}, 0, None)
        fault = NeuralReport({'status': 'operational_failure', 'mode': 'standalone',
            'case_accounting': {'cases': [{'case_id': ids[1], 'status': 'incomplete'}]},
            'fights': {'standalone_fights': {'reached_combat_denominator': 1,
                'completed': 0, 'victories': 0, 'deaths': 0, 'fights': []}},
            'collection': {'uncertain_environment_slots': 1}}, 1, None)
        result = summarize_development_reports(panel, {ids[0]: healthy, ids[1]: fault})
        row = result['strata']['cohort']['familiar_new_seed']
        self.assertEqual((row['scheduled'], row['reported'], row['started'], row['completed'], row['wins']), (16, 2, 2, 1, 1))
        self.assertEqual(row['operational_failures'], 1)
        self.assertEqual(row['censored'], 1)
        self.assertEqual(row['hp_samples'], 1)
        self.assertIsNone(row['potion_use'])
        self.assertEqual(row['potion_use_samples'], 0)
        self.assertEqual(result['status'], 'operational_failure')
        self.assertEqual(result['worst_strata']['cohort']['completed'], 1)
        malformed = copy.deepcopy(healthy)
        malformed.summary['fights']['standalone_fights']['fights'] *= 2
        with self.assertRaises(ValueError):
            summarize_development_reports(panel, {ids[0]: malformed})
        unstarted = {}
        for case_id in ids:
            report = copy.deepcopy(fault)
            report.exit_code = 0
            report.summary.update(status='complete', collection={'uncertain_environment_slots': 0})
            report.summary['case_accounting']['cases'] = [{'case_id': case_id, 'status': 'not_started'}]
            report.summary['fights']['standalone_fights']['reached_combat_denominator'] = 0
            unstarted[case_id] = report
        self.assertEqual(summarize_development_reports(panel, unstarted)['status'], 'incomplete')
        blocked = NeuralReport({'status': 'setup_blocked', 'mode': 'standalone',
            'reason': 'first setup cause', 'case_accounting': {'cases': [
                {'case_id': ids[0], 'status': 'not_started'}]}}, 2, None)
        missing = summarize_development_reports(panel, {ids[0]: blocked})
        self.assertEqual(missing['status'], 'operational_failure')
        self.assertEqual(missing['strata']['cohort']['familiar_new_seed']['fight_reports_missing'], 1)
        self.assertEqual(missing['failures'][0]['reason'], 'first setup cause')


class SegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.unit.test_neural_stream_resume import StreamFixture
        StreamFixture.setUpClass()
        cls.fixture_type = StreamFixture

    def setUp(self):
        self.fixture = self.fixture_type()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.config(0, arm='fixed')

    def test_plain_callable_contract_limits_and_failure_never_offer_next(self):
        from spire_learn.neural.stream_runner import run_stream_segment
        limits = RuntimeOperationLimits(max_updates=3, monotonic_deadline=55.0, max_environment_actions=9)
        runtime = SimpleNamespace(operation_limits=None)
        calls = []
        def factory(config, *, purpose, checkpoint, policy_kind, action_selection):
            self.assertIs(config, self.config)
            calls.append((purpose, checkpoint, policy_kind, action_selection))
            return runtime
        def train(config, *, runtime, clock):
            self.assertIs(runtime.operation_limits, limits)
            self.assertEqual(clock(), 7.0)
            return NeuralReport({'status': 'operational_failure', 'fault': 'first cause'}, 1, None)
        result = run_stream_segment(self.config, purpose='resume', checkpoint=Path('/current-parent'),
            operation_limits=limits, dependencies={'create_runtime': factory, 'train': train, 'clock': lambda: 7.0})
        self.assertEqual(calls, [('resume', Path('/current-parent'), 'checkpoint', 'seeded-sampling')])
        self.assertIsNone(result['next_shard'])
        self.assertEqual(result['report'].summary['fault'], 'first cause')

    def test_invalid_purpose_or_limits_do_not_construct_runtime(self):
        from spire_learn.neural.stream_runner import run_stream_segment
        def forbidden(*args, **kwargs):
            self.fail('invalid input reached factory')
        for purpose, checkpoint, limits in [('train', Path('/bad'), RuntimeOperationLimits()),
                ('resume', None, RuntimeOperationLimits()), ('evaluate', None, RuntimeOperationLimits()),
                ('train', None, object())]:
            with self.subTest(purpose=purpose), self.assertRaises((ValueError, TypeError)):
                run_stream_segment(self.config, purpose=purpose, checkpoint=checkpoint,
                    operation_limits=limits, dependencies={'create_runtime': forbidden, 'train': forbidden, 'clock': forbidden})

    def test_setup_and_control_failures_propagate_without_retry(self):
        from spire_learn.neural.stream_runner import run_stream_segment
        for error in (RuntimeError('setup'), KeyboardInterrupt('control')):
            calls = []
            def factory(config, **kwargs):
                calls.append(config)
                raise error
            with self.assertRaises(type(error)):
                run_stream_segment(self.config, purpose='train', checkpoint=None,
                    operation_limits=RuntimeOperationLimits(), dependencies={
                        'create_runtime': factory, 'train': lambda *a, **k: self.fail('train after setup fault'), 'clock': lambda: 0.0})
            self.assertEqual(len(calls), 1)

    def write_checkpoint(self, path, metadata):
        from tests.unit.test_neural_checkpoint import _write_archive
        from spire_learn.neural.checkpoint import CHECKPOINT_SCHEMA_ID
        path.mkdir()
        value = copy.deepcopy(metadata)
        value.update(schema_id=CHECKPOINT_SCHEMA_ID, model_tensors=[{
            'name': 'weight', 'dtype': 'float32', 'shape': [1], 'sha256': 'e' * 64}])
        value['payload_inventory'] = {}
        for name in ('model.pt', 'training.pt'):
            _write_archive(path / name)
            data = (path / name).read_bytes()
            value['payload_inventory'][name] = {'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        (path / 'metadata.json').write_text(json.dumps(value))
        return value

    def published_report(self, *, parent=None):
        from tests.unit.test_neural_checkpoint import _metadata
        from spire_learn.neural.runner import _PROGRESS_FIELDS
        from spire_learn.neural.stream_resume import stream_checkpoint_state
        run = self.fixture.root / 'published'
        run.mkdir()
        path = run / 'checkpoint-final'
        metadata = _metadata(identity=self.config.accepted_identities[0], training=True)
        prefix = {k: 0 for k in _PROGRESS_FIELDS}
        prefix['elapsed_seconds'] = 2.0
        metadata['training_state'].update(optimizer_step=0, counters=prefix,
            hyperparameter_id=self.config.accepted_identities[0].config_hashes['ppo'],
            update_id='update-0', behavior_policy_version=0, next_environment_seed_ordinal=0,
            next_decision_sequence=0, saved_inflight_record_count=0)
        metadata['run_metadata'] = {'mode': 'standalone', 'operation': 'train', 'split': 'train',
            'partition': 'train', 'config_sha256': self.config.config_sha256,
            'manifest_sha256': self.config.manifest_sha256,
            'environment_revision': self.config.environment_revision,
            'scenario_stream': stream_checkpoint_state(self.config, {}),
            'cumulative_prefix': prefix,
            'collector_recovery': {'next_seed_ordinal': 0, 'next_decision_sequence': 0,
                'next_episode_ordinals': [0] * self.config.rollout_rows}}
        metadata['parent_checkpoint'] = None if parent is None else {'path': str(parent)}
        self.write_checkpoint(path, metadata)
        publication = {'publication_state': 'published', 'purpose': 'final', 'checkpoint_path': str(path)}
        lifecycle = {'status': 'complete', 'exit_code': 0, 'close_errors': [],
            'checkpoint_publication': publication}
        return NeuralReport({'status': 'complete', 'lifecycle_receipt_required': True,
            'checkpoint_publication': publication, 'lifecycle': lifecycle}, 0, run)

    def dispatch_report(self, report, *, purpose='train', checkpoint=None):
        from spire_learn.neural.stream_runner import run_stream_segment
        return run_stream_segment(self.config, purpose=purpose, checkpoint=checkpoint,
            operation_limits=RuntimeOperationLimits(), dependencies={
                'create_runtime': lambda config, **kwargs: SimpleNamespace(),
                'train': lambda config, **kwargs: report, 'clock': lambda: 100.0})

    def test_healthy_publication_returns_actual_checkpoint_prefix_and_next_only(self):
        parent = Path('/immediate-parent')
        report = self.published_report(parent=parent)
        result = self.dispatch_report(report, purpose='resume', checkpoint=parent)
        self.assertIs(result['report'], report)
        self.assertEqual(result['checkpoint'], report.run_dir / 'checkpoint-final')
        self.assertEqual(result['saved_prefix']['elapsed_seconds'], 2.0)
        self.assertEqual(result['next_shard']['shard_index'], 1)
        with self.assertRaisesRegex(ValueError, 'parent'):
            self.dispatch_report(report, purpose='resume', checkpoint=Path('/different-parent'))
        (result['checkpoint'] / 'model.pt').write_bytes(b'corrupt archive')
        with self.assertRaises(ValueError):
            self.dispatch_report(report, purpose='resume', checkpoint=parent)

    def test_uncertain_or_inconsistent_saved_progress_cannot_authorize_successor(self):
        report = self.published_report()
        path = report.run_dir / 'checkpoint-final/metadata.json'
        original = json.loads(path.read_text())
        for field in ('uncertain_environment_slots', 'optimizer_uncertain_steps', 'delivered_reset_slots'):
            metadata = copy.deepcopy(original)
            metadata['training_state']['counters'][field] = 1
            metadata['run_metadata']['cumulative_prefix'][field] = 1
            path.write_text(json.dumps(metadata))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.dispatch_report(report)

    def test_late_lifecycle_or_publication_failure_never_yields_successor(self):
        report = self.published_report()
        for transform in (
            lambda s: s['lifecycle'].update(status='operational_failure', close_errors=['close']),
            lambda s: s.update(lifecycle_receipt_error={'message': 'write failed'}),
            lambda s: s['checkpoint_publication'].update(publication_state='possibly_published'),
            lambda s: s['lifecycle'].update(clock_error={'message': 'clock failed'}),
        ):
            changed = NeuralReport(copy.deepcopy(report.summary), 0, report.run_dir)
            transform(changed.summary)
            with self.assertRaises(ValueError):
                self.dispatch_report(changed)

    def test_actual_train_factory_save_and_lifecycle_projection(self):
        from spire_learn.experiments.budget import Budget
        from spire_learn.neural import runner
        from spire_learn.neural.ppo import UpdateMetrics
        from spire_learn.neural.stream_runner import run_stream_segment
        from tests.unit.test_neural_stream_resume import StreamRuntimeTests
        from spire_learn.neural.artifacts import NeuralWriter
        helper = StreamRuntimeTests()
        self.addCleanup(helper.doCleanups)
        run = self.fixture.root / 'real-train'
        run.mkdir()
        writer = NeuralWriter(run)
        writer.start({'test': 'projection'}, {'test': 'projection'})
        runtimes = []
        def factory(config, *, purpose, checkpoint, policy_kind, action_selection):
            self.assertEqual((purpose, checkpoint, policy_kind, action_selection),
                             ('train', None, 'checkpoint', 'seeded-sampling'))
            runtime, driver = helper.assembly(config, purpose=purpose)
            runtimes.append((runtime, driver))
            return runtime
        def update(policy, optimizer, rollout, ppo_config, *, shuffle_generator, budget):
            self.assertIsInstance(budget, Budget)
            self.assertEqual(budget.allowed_seconds, 50.0)
            optimizer.step()
            version = policy._behavior_policy_version
            policy._behavior_policy_version += 1
            return UpdateMetrics(0.0, 0.0, 0.0, 1, 1, None, version, version + 1, len(rollout.records))
        saved = []
        def save(path, *, policy, optimizer, rng_state, metadata):
            self.assertFalse(runtimes[0][0]._closed)
            saved.append(self.write_checkpoint(path, metadata))
        with patch.object(runner, '_writer_start', return_value=(writer, run)), \
             patch.object(runner, 'ppo_update', side_effect=update), \
             patch.object(runner, 'save_checkpoint', side_effect=save):
            result = run_stream_segment(self.config, purpose='train', checkpoint=None,
                operation_limits=RuntimeOperationLimits(max_updates=1, monotonic_deadline=50.0,
                    max_environment_actions=4), dependencies={
                    'create_runtime': factory, 'train': runner.train, 'clock': lambda: 0.0})
        self.assertEqual(result['report'].exit_code, 0)
        self.assertEqual(result['saved_prefix']['delivered_environment_slots'], 4)
        self.assertEqual(result['saved_prefix']['optimizer_accepted_steps'], 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(runtimes[0][1].close_calls, 1)
        self.assertEqual(json.loads((run / 'lifecycle.json').read_text())['status'], 'complete')
        self.assertEqual(result['next_shard']['shard_index'], 1)

    def test_actual_runner_late_failures_and_control_halt_without_retry(self):
        from spire_learn.neural import runner
        from spire_learn.neural.ppo import UpdateMetrics
        from spire_learn.neural.stream_runner import run_stream_segment
        from tests.unit.test_neural_stream_resume import StreamRuntimeTests
        from tests.unit.test_neural_runner import FakeWriter
        for phase in ('update', 'close', 'receipt', 'observer', 'clock', 'control'):
            with self.subTest(phase=phase):
                helper = StreamRuntimeTests()
                self.addCleanup(helper.doCleanups)
                runtime, driver = helper.assembly(self.config, purpose='train')
                first = RuntimeError(f'first {phase} failure')
                closed = []
                saves = []
                updates = []
                writer = FakeWriter(event_error=first if phase == 'observer' else None)
                original_close = runtime.close
                def close():
                    original_close()
                    closed.append(True)
                    if phase == 'close':
                        raise first
                def update(policy, optimizer, rollout, ppo_config, **kwargs):
                    updates.append(True)
                    if phase == 'update':
                        raise first
                    if phase == 'control':
                        raise KeyboardInterrupt('control at update')
                    optimizer.step()
                    version = policy._behavior_policy_version
                    policy._behavior_policy_version += 1
                    return UpdateMetrics(0.0, 0.0, 0.0, 1, 1, None, version, version + 1, len(rollout.records))
                def clock():
                    if phase == 'clock' and closed:
                        raise first
                    return 0.0
                def receipt(*args):
                    if phase == 'receipt':
                        raise first
                with patch.object(runner, '_writer_start', return_value=(writer, self.fixture.root / phase)), \
                     patch.object(runner, 'ppo_update', side_effect=update), \
                     patch.object(runner, '_write_lifecycle_receipt', side_effect=receipt), \
                     patch.object(runner, 'save_checkpoint', side_effect=lambda *a, **k: saves.append(a)), \
                     patch.object(runtime, 'close', side_effect=close):
                    def invoke():
                        return run_stream_segment(self.config, purpose='train', checkpoint=None,
                            operation_limits=RuntimeOperationLimits(max_updates=1, max_environment_actions=4),
                            dependencies={'create_runtime': lambda config, **kwargs: runtime,
                                          'train': runner.train, 'clock': clock})
                    if phase == 'control':
                        with self.assertRaises(KeyboardInterrupt):
                            invoke()
                    else:
                        result = invoke()
                        self.assertEqual(result['report'].exit_code, 1)
                        self.assertIsNone(result['next_shard'])
                        self.assertIn(f'first {phase} failure', str(result['report'].summary))
                self.assertEqual(driver.close_calls, 1)
                self.assertEqual(len(updates), 0 if phase == 'observer' else 1)
                self.assertEqual(len(saves), 0 if phase in ('update', 'observer', 'control') else 1)
