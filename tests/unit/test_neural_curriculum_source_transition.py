"""Pure admission tests: source bridges never relax checkpoint-chain checks."""
import copy
import unittest
from scripts.neural import curriculum_runtime as api
from scripts.neural.gpu_runtime_proof_helpers import canonical_sha256

OLD = '53cd32b7be83bde5c78f773a914dba7f89a0937840214e272dbbfaab71ff7d2e'

class SourceTransitionTests(unittest.TestCase):
    def context(self):
        inventory = {'metadata.json': {'sha256': 'a'*64, 'size_bytes': 12}}
        context = dict(kind='segment', campaign_id='curriculum-20261102-c', arm='mixed',
            shard_index=31, source_sha256='b'*64, predecessor={'path': 'prior.json', 'sha256': 'c'*64},
            parent={'path': '/inputs/parent', 'inventory': inventory})
        context['source_transition'] = dict(schema='spire-learn-curriculum-source-transition-v1',
            purpose='authorized-checkpoint-continuation', from_source_sha256=OLD,
            to_source_sha256=context['source_sha256'], predecessor_sha256='c'*64,
            checkpoint_inventory_sha256=canonical_sha256(inventory), user_approval_sha256='d'*64)
        return context

    def test_bridge_exact_and_absent_legacy(self):
        context = self.context()
        self.assertEqual(api.validate_source_transition(context, previous_source=OLD, authority='d'*64), OLD)
        del context['source_transition']
        self.assertEqual(api.validate_source_transition(context), 'b'*64)

    def test_bridge_closed_bindings(self):
        changes = {'schema': 'other', 'purpose': 'retry', 'from_source_sha256': 'e'*64,
            'to_source_sha256': 'e'*64, 'predecessor_sha256': 'e'*64,
            'checkpoint_inventory_sha256': 'e'*64, 'user_approval_sha256': 'bad', 'extra': True}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                context = self.context(); context['source_transition'][key] = value
                api.validate_source_transition(context, previous_source=OLD, authority='d'*64)
        for key in self.context()['source_transition']:
            with self.subTest(missing=key), self.assertRaises(ValueError):
                context = self.context(); del context['source_transition'][key]
                api.validate_source_transition(context)

    def test_wrong_authority_or_actual_old_source(self):
        for kwargs in ({'authority': 'f'*64}, {'previous_source': 'f'*64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.validate_source_transition(self.context(), **kwargs)

    def test_only_c_mixed_resume_and_never_equal_source(self):
        for key, value in (('kind', 'panel'), ('arm', 'fixed'), ('campaign_id', 'other'),
                           ('shard_index', 0), ('shard_index', True), ('source_sha256', OLD)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                context = self.context(); context[key] = value
                api.validate_source_transition(context)

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from scripts.neural import curriculum_profiles as profiles

class ContextChainTests(unittest.TestCase):
    def fixture(self, root):
        context = SourceTransitionTests().context()
        context['shard_index'] = 1
        context.update(schema=api.CONTEXT_SCHEMA, job_id='curriculum-opt-mixed-001', runtime_class='gpu',
            package_path='package.json', package_sha256='e'*64, input_root='inputs', input_inventory={},
            source_inventory={}, release_sha256='f'*64, claim_sha256='a'*64,
            deadline=120., whole_deadline=220., reservation=dict(environment_attempts=4096,
                sample_rows=4096, optimizer_attempts=256, reset_slots=4112, fixed_random_choices=0),
            output_root=str(root/'output'), prefix={'update_count': 8, 'attempted_environment_slots': 4096},
            max_updates=16, max_environment_actions=8192)
        context['parent']['path'] = str(root/'parent')
        (root/'output').mkdir()
        binding = dict(inputs=[], config_path='config.toml', config_sha256='config',
            manifest_sha256='manifest', stable_training_sha256='stable')
        package = dict(schema=api.PACKAGE_SCHEMA, campaign_id=context['campaign_id'],
            training={'mixed': [binding, binding]}, inventory={})
        prior = dict(schema=api.RECEIPT_SCHEMA, status='complete', campaign_id=context['campaign_id'],
            source_sha256=OLD, package_sha256=context['package_sha256'], kind='segment', arm='mixed',
            shard_index=0, config_sha256='config', manifest_sha256='manifest', next_shard={'shard_index': 1},
            report={'status': 'complete', 'fault': None},
            effects={key: dict(uncertain=0) for key in api.EFFECT_KINDS},
            checkpoint={'inventory': context['parent']['inventory'], 'counters': context['prefix']})
        config = SimpleNamespace(environment_revision=profiles.FRESH_GYM_C.revision,
            config_sha256='config', manifest_sha256='manifest', output_root=root/'output',
            scenario_stream={'shard_index': 1, 'arm': 'mixed'})
        return context, package, prior, config

    def admit(self, root, context, package, prior, config):
        with ExitStack() as stack:
            stack.enter_context(profiles.scope('fresh-gym-afb51f5f-c'))
            stack.enter_context(patch.object(api, 'verify_inventory'))
            stack.enter_context(patch.object(api, 'source_paths', return_value=[]))
            stack.enter_context(patch.object(api, 'relative_file', side_effect=lambda root, name: root/name))
            stack.enter_context(patch.object(api, 'sha256', side_effect=lambda p: context['package_sha256'] if p.name == 'package.json' else context['predecessor']['sha256']))
            stack.enter_context(patch.object(api, 'read_json', side_effect=lambda p, **kw: package if p.name == 'package.json' else prior))
            stack.enter_context(patch('scripts.neural.sustained_runtime.checkpoint_inventory', return_value=context['parent']['inventory']))
            stack.enter_context(patch('spire_learn.neural.config.load_neural_config', return_value=config))
            stack.enter_context(patch('spire_learn.neural.stream_resume.stream_identity', return_value={'stable_training_sha256': 'stable'}))
            return api.validate_context(root, context, clock=lambda: 0.)

    def test_exact_bridge_and_normal_legacy_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.fixture(root)
            self.admit(root, *args)
            context, package, prior, config = args
            del context['source_transition']; prior['source_sha256'] = context['source_sha256']
            self.admit(root, *args)
            prior['source_sha256'] = OLD
            with self.assertRaisesRegex(ValueError, 'healthy immediate parent'):
                self.admit(root, *args)

    def test_bridge_does_not_relax_healthy_immediate_checkpoint(self):
        for change in ('arm', 'shard', 'fault', 'uncertain', 'prefix', 'weights', 'source'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); context, package, prior, config = self.fixture(root)
                if change == 'arm': prior['arm'] = 'fixed'
                elif change == 'shard': prior['shard_index'] = 2
                elif change == 'fault': prior['report']['fault'] = 'native fault'
                elif change == 'uncertain': prior['effects']['environment_attempts']['uncertain'] = 1
                elif change == 'prefix': prior['checkpoint']['counters'] = {}
                elif change == 'weights': prior['checkpoint']['inventory'] = {}
                else: prior['source_sha256'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'healthy immediate parent'):
                    self.admit(root, context, package, prior, config)
