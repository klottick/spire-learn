"""Focused contracts for the separately budgeted twenty-hour continuation."""
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from scripts.neural import curriculum_runtime as producer
from scripts.neural import task9_study_controls as low
from spire_learn.neural import stream_resume


ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'ironclad-twenty-hour-continuation-20261103'


class ContinuationStorageAdmissionTests(unittest.TestCase):
    def test_global_aggregate_scan_tolerates_only_live_unlink_at_closed_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root/'owned-runtime'
            aggregate = root/'artifacts/development/neural-combat'
            host.mkdir()
            aggregate.mkdir(parents=True)
            real_walk = low.os.walk

            def unlinking_walk(directory, *, followlinks, onerror):
                if Path(directory) == aggregate:
                    onerror(FileNotFoundError(2, 'entry disappeared', str(aggregate/'writer.tmp')))
                    return
                yield from real_walk(directory, followlinks=followlinks, onerror=onerror)

            deps = SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=64*controls.GIB))
            with patch.object(low.os, 'walk', side_effect=unlinking_walk):
                result = controls.storage_admission(root, host, deps)

            self.assertEqual(result['new_bytes'], 0)
            self.assertEqual(result['aggregate_bytes'], 0)

    def test_owned_tree_stays_strict_and_aggregate_permission_errors_propagate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root/'owned-runtime'
            aggregate = root/'artifacts/development/neural-combat'
            host.mkdir()
            aggregate.mkdir(parents=True)
            real_walk = low.os.walk
            deps = SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=64*controls.GIB))

            def disappearing_owned(directory, *, followlinks, onerror):
                if Path(directory) == host:
                    onerror(FileNotFoundError(2, 'owned entry disappeared', str(host/'tmp')))
                    return
                yield from real_walk(directory, followlinks=followlinks, onerror=onerror)

            with patch.object(low.os, 'walk', side_effect=disappearing_owned):
                with self.assertRaises(FileNotFoundError):
                    controls.storage_admission(root, host, deps)

            def denied_aggregate(directory, *, followlinks, onerror):
                if Path(directory) == aggregate:
                    onerror(PermissionError(13, 'aggregate scan denied', str(aggregate/'private')))
                    return
                yield from real_walk(directory, followlinks=followlinks, onerror=onerror)

            with patch.object(low.os, 'walk', side_effect=denied_aggregate):
                with self.assertRaises(PermissionError):
                    controls.storage_admission(root, host, deps)


class ContinuationProfileAndBudgetTests(unittest.TestCase):
    def test_profile_routes_a_new_namespace_without_changing_the_original(self):
        with profiles.scope(PROFILE):
            selected = profiles.current()
            self.assertEqual(selected.campaign_id, 'curriculum-20261103-twenty-hour-continuation-1')
            self.assertEqual(selected.domain,
                Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1'))
            self.assertEqual((selected.whole_seconds, selected.training_stop_second), (65998, 64198))
            self.assertIn('configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json',
                profiles.source_paths(ROOT))
            self.assertIn('tests/unit/test_neural_curriculum_continuation_1.py', profiles.source_paths(ROOT))
        with profiles.scope('ironclad-twenty-hour-20261103'):
            self.assertEqual((profiles.current().whole_seconds, profiles.current().training_stop_second),
                (72000, 70200))

    def test_caps_debit_original_charges_and_full_unknown_shard_338_lease(self):
        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        with profiles.scope(PROFILE):
            caps = controls._effective_global_effect_caps(ROOT, campaign)
            self.assertEqual(caps, {
                'environment_attempts': 4538880,
                'sample_rows': 4375040,
                'optimizer_attempts': 273416,
                'reset_slots': 4392512,
                'fixed_random_choices': 0,
            })
            audit = controls.validate_predecessor(ROOT)
            self.assertEqual(audit['charged_or_reserved'], {
                'environment_attempts': 341567,
                'sample_rows': 340096,
                'optimizer_attempts': 21256,
                'reset_slots': 19543,
                'fixed_random_choices': 0,
            })
            self.assertEqual(audit['failed_shard']['actual'], None)


class ContinuationRuntimeAdmissionTests(unittest.TestCase):
    def _context_fixture(self, root, *, prefix_actions, selected=profiles.TWENTY_HOUR_CONTINUATION):
        root = Path(root)
        input_root = root/'inputs'
        output = root/'output'
        parent_path = root/'parent'
        input_root.mkdir()
        output.mkdir()
        parent_path.mkdir()
        prefix = dict(attempted_environment_slots=prefix_actions, update_count=294000)
        start_index = 339 if selected is profiles.TWENTY_HOUR_CONTINUATION else 0
        current_index = start_index+1
        parent_inventory = {'metadata.json': {'sha256': 'a'*64, 'size_bytes': 1}}
        input_binding = {'sha256': 'b'*64, 'size_bytes': 1}
        bindings = [
            dict(inputs=['config.toml'], config_path='config.toml', config_sha256=f'config-{start_index}',
                manifest_sha256=f'manifest-{start_index}', stable_training_sha256='stable'),
            dict(inputs=['config.toml'], config_path='config.toml', config_sha256=f'config-{current_index}',
                manifest_sha256=f'manifest-{current_index}', stable_training_sha256='stable'),
        ]
        package_doc = dict(schema=producer.PACKAGE_SCHEMA,
            campaign_id=selected.campaign_id,
            campaign=dict(training_start_indices={'mixed': start_index},
                training_caps={'environment_attempts': 4710400} if selected is profiles.TWENTY_HOUR_CONTINUATION else {}),
            training={'mixed': bindings}, inventory={'config.toml': input_binding})
        context = dict(schema=producer.CONTEXT_SCHEMA,
            campaign_id=selected.campaign_id,
            job_id='curriculum-continuation-context-340', kind='segment', runtime_class='gpu',
            package_path='package.json', package_sha256='c'*64, input_root='inputs',
            input_inventory={'config.toml': input_binding}, source_inventory={}, source_sha256='d'*64,
            release_sha256='e'*64, claim_sha256='f'*64,
            parent={'path': str(parent_path), 'inventory': parent_inventory},
            deadline=400., whole_deadline=500.,
            reservation=dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
                reset_slots=4112, fixed_random_choices=0), output_root=str(output), arm='mixed',
            shard_index=current_index, prefix=prefix, max_updates=294008,
            max_environment_actions=prefix_actions+4096,
            predecessor={'path': 'predecessor.json', 'sha256': '1'*64})
        previous = dict(schema=producer.RECEIPT_SCHEMA, status='complete',
            campaign_id=context['campaign_id'], source_sha256=context['source_sha256'],
            package_sha256=context['package_sha256'], kind='segment', arm='mixed', shard_index=start_index,
            config_sha256=f'config-{start_index}', manifest_sha256=f'manifest-{start_index}',
            next_shard={'shard_index': current_index}, report={'status': 'complete', 'fault': None},
            effects={key: {'uncertain': 0} for key in producer.EFFECT_KINDS},
            checkpoint={'inventory': parent_inventory, 'counters': prefix})
        config = SimpleNamespace(environment_revision=selected.revision,
            config_sha256=f'config-{current_index}', manifest_sha256=f'manifest-{current_index}', output_root=output,
            scenario_stream={'shard_index': current_index, 'arm': 'mixed'})
        return context, package_doc, previous, config, parent_inventory

    def _admit(self, root, context, package_doc, previous, config, parent_inventory, *, selected=PROFILE):
        def read(path, **kwargs):
            return package_doc if Path(path).name == 'package.json' else previous

        def digest(path):
            return context['package_sha256'] if Path(path).name == 'package.json' else context['predecessor']['sha256']

        with profiles.scope(selected), patch.object(producer, 'verify_inventory'), \
                patch.object(producer, 'source_paths', return_value=[]), \
                patch.object(producer, 'relative_file', side_effect=lambda base, name: Path(base)/name), \
                patch.object(producer, 'sha256', side_effect=digest), \
                patch.object(producer, 'read_json', side_effect=read), \
                patch('scripts.neural.sustained_runtime.checkpoint_inventory', return_value=parent_inventory), \
                patch('spire_learn.neural.config.load_neural_config', return_value=config), \
                patch('spire_learn.neural.stream_resume.stream_identity',
                    return_value={'stable_training_sha256': 'stable'}):
            return producer.validate_context(root, context, clock=lambda: 0.)

    def test_guest_context_admits_past_legacy_cap_and_rejects_above_released_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, package_doc, previous, config, inventory = self._context_fixture(
                root, prefix_actions=1000001)
            admitted = self._admit(root, context, package_doc, previous, config, inventory)
            self.assertIs(admitted[0], package_doc)
            self.assertEqual(admitted[-1], config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, package_doc, previous, config, inventory = self._context_fixture(
                root, prefix_actions=4710400-4096)
            admitted = self._admit(root, context, package_doc, previous, config, inventory)
            self.assertEqual(context['max_environment_actions'], 4710400)
            self.assertIs(admitted[0], package_doc)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, package_doc, previous, config, inventory = self._context_fixture(
                root, prefix_actions=4710400-4095)
            with self.assertRaisesRegex(ValueError, 'max actions'):
                self._admit(root, context, package_doc, previous, config, inventory)

    def test_legacy_context_keeps_the_one_million_action_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = profiles.FRESH_GYM_C
            context, package_doc, previous, config, inventory = self._context_fixture(
                root, prefix_actions=1000001, selected=selected)
            with self.assertRaisesRegex(ValueError, 'max actions'):
                self._admit(root, context, package_doc, previous, config, inventory,
                    selected=selected.name)


class ContinuationSourceAndResumeTests(unittest.TestCase):
    def test_source_transition_is_bound_only_to_shard_339_and_the_retained_parent(self):
        parent = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/parents.json')['standalone_b']['inventory']
        bridge = {
            'schema': producer.SOURCE_TRANSITION_SCHEMA,
            'purpose': 'authorized-checkpoint-continuation',
            'from_source_sha256': 'd363f588bdafb4f2013828c149edeaff819eb422f6f48a6ff34493a257f7eb3a',
            'to_source_sha256': 'a'*64,
            'predecessor_sha256': 'b2afb649a3887bbcc101c397831cbe6b5c0e639476c377cbace20f4f1e333fd0',
            'checkpoint_inventory_sha256': '071a05b8bdec51f38b9f7591dec553a0d7f5547cc2dd2131094d678fe617054e',
            'user_approval_sha256': '1c1c1d85b858caf1900db84bfdec5ba9696cdd8d71823b9fbada0bebae2e35c4',
        }
        context = {'source_transition': bridge, 'campaign_id': 'curriculum-20261103-twenty-hour-continuation-1',
            'kind': 'segment', 'arm': 'mixed', 'shard_index': 339, 'source_sha256': 'a'*64,
            'predecessor': {'path': 'inputs/predecessor.json', 'sha256': bridge['predecessor_sha256']},
            'parent': {'path': '/inputs/parent', 'inventory': parent}}
        with profiles.scope(PROFILE):
            self.assertEqual(producer.validate_source_transition(context), bridge['from_source_sha256'])
            wrong = dict(context, shard_index=340)
            with self.assertRaisesRegex(ValueError, 'source transition'):
                producer.validate_source_transition(wrong)
            wrong_bridge = dict(bridge, predecessor_sha256='c'*64)
            with self.assertRaisesRegex(ValueError, 'source transition'):
                producer.validate_source_transition(dict(context, source_transition=wrong_bridge))

    def test_skip_authorization_preserves_full_338_lease_and_null_unknown_effects(self):
        import json
        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        expected = campaign['continuation']['resume_skip_authorization']
        self.assertEqual(stream_resume.validate_continuation_skip_authorization(expected), expected)
        encoded = json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.assertEqual(stream_resume.validate_continuation_skip_authorization(encoded), expected)
        for key, value in (('actual', 0), ('receipt_binding', {'sha256': '0'*64}),
                           ('skipped_shard_index', 339)):
            altered = dict(expected, **{key: value})
            with self.assertRaisesRegex(ValueError, 'skip authorization'):
                stream_resume.validate_continuation_skip_authorization(altered)

    def test_package_marks_only_shard_339_as_the_authorized_stream_gap(self):
        import tomllib
        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        template = tomllib.loads((ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/template.toml').read_text())
        with profiles.scope(PROFILE):
            first = package._training_document(template, campaign, 'mixed', 339, '0'*64)
            next_ = package._training_document(template, campaign, 'mixed', 340, '0'*64)
        import json
        self.assertEqual(json.loads(first['scenario_stream']['resume_skip']),
            campaign['continuation']['resume_skip_authorization'])
        self.assertNotIn('resume_skip', next_['scenario_stream'])

    def test_checkpoint_stream_admits_only_the_authorized_337_to_339_gap(self):
        import json
        import tomllib
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.curriculum import CurriculumSampler
        from spire_learn.neural.scenario_support import json_value
        from spire_learn.neural.scenario_stream import generate_scenario_shard, load_scenario_recipe
        from spire_learn.neural.stream_resume import stream_checkpoint_state
        from spire_learn.neural.stream_runner import _toml

        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        source_recipe = ROOT/'configs/neural/curriculum-v1/recipe.json'
        source_support = source_recipe.parent/'support.json'
        recipe = load_scenario_recipe(source_recipe)
        template = tomllib.loads((ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/template.toml').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/development/neural-combat') as temporary:
            base = Path(temporary)
            shutil.copyfile(source_recipe, base/'recipe.json')
            shutil.copyfile(source_support, base/'support.json')
            configs, shards = {}, {}
            with profiles.scope(PROFILE):
                for index in (337, 339, 340):
                    shard = generate_scenario_shard(recipe, split='train', arm='mixed', shard_index=index)
                    shards[index] = shard
                    (base/f'shard-{index}.json').write_text(json.dumps(json_value(shard.manifest), sort_keys=True)+'\n')
                    document = package._training_document(template, campaign, 'mixed', index,
                        shard.distribution_sha256)
                    path = base/f'config-{index}.toml'
                    path.write_text(_toml(document))
                    configs[index] = load_neural_config(path)
                sampler = CurriculumSampler(shards[337].annotations,
                    distribution_sha256=shards[337].distribution_sha256,
                    categories=recipe.category_keys).snapshot()
                state = stream_checkpoint_state(configs[337], sampler)
                metadata = dict(scenario_stream=state,
                    config_sha256=configs[337].config_sha256,
                    manifest_sha256=configs[337].manifest_sha256)
                resumed = stream_resume.validate_stream_resume(metadata, configs[339])
                self.assertEqual(resumed['current']['shard_index'], 337)
                self.assertEqual(resumed['destination']['shard_index'], 339)
                self.assertEqual(resumed['next']['shard_index'], 338)
                self.assertEqual(resumed['resume_skip_authorization'], campaign['continuation']['resume_skip_authorization'])
                with self.assertRaisesRegex(ValueError, 'same or exactly next shard'):
                    stream_resume.validate_stream_resume(metadata, configs[340])

    def test_first_panel_evaluates_checkpoint_337_then_resumes_at_339(self):
        from scripts.neural.curriculum_controls import TwentyHourSchedule
        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        parent = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/parents.json')['standalone_b']
        continuation = campaign['continuation']
        package_stub = {'campaign': campaign, 'evaluation_slots': campaign['evaluation_slots'],
            'continuation': continuation, 'original_parent': parent}
        receipt = {'path': '/runtime/lineage/shard337-receipt.json', 'sha256': continuation['parent_receipt_sha256']}
        with profiles.scope(PROFILE):
            plan = TwentyHourSchedule(package_stub, start=1000.,
                continuation_parent_receipt=receipt)
            self.assertEqual(plan.prefixes['mixed'], continuation['parent_prefix'])
            self.assertEqual(plan.indices['mixed'], 339)
            self.assertEqual(plan.next(1000.), {'kind': 'qualification'})
            plan.accept_qualification()
            panel = plan.next(1000.)
            self.assertEqual(panel, {'kind': 'panel', 'slot': 'continuation-parent-shard337', 'arm': 'mixed'})
            plan.finish_panel(panel['slot'], status='complete', reason='test')
            self.assertEqual(plan.next(1000.), {'kind': 'segment', 'arm': 'mixed', 'index': 339,
                'arm_deadline': 1000. + 64198})

    def test_panel_deadlines_preserve_training_cutoff_and_final_reserve(self):
        campaign = package.read_json(ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json')
        now = [100.]
        with profiles.scope(PROFILE):
            budget = controls.CampaignBudget(campaign['global_effect_caps'], start=now[0],
                deadline=now[0]+65998., clock=lambda: now[0], evaluation_cap=6600,
                final_panel_reserve=600)
            now[0] = 100.+64198.-198.
            _, nonfinal_whole = budget.job_deadlines('panel', final_panel=False)
            self.assertLessEqual(nonfinal_whole, 100.+64198.)
            now[0] = 100.+64198.
            with self.assertRaisesRegex(TimeoutError, 'time exhausted'):
                budget.job_deadlines('panel', final_panel=False)
            _, final_whole = budget.job_deadlines('panel', final_panel=True)
            self.assertEqual(final_whole-now[0], 600.)
            now[0] = 100.+64198.+100.
            _, late_whole = budget.job_deadlines('panel', final_panel=True)
            self.assertEqual(late_whole, 100.+64198.+600.)
            now[0] = 100.+64198.+1000.
            with self.assertRaisesRegex(TimeoutError, 'time exhausted'):
                budget.job_deadlines('panel', final_panel=True)


if __name__ == '__main__':
    unittest.main()
