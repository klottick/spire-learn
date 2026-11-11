"""Fresh ownership after failed guest admission; no native or tensor execution."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from scripts.neural import curriculum_runtime as producer
from spire_learn.neural import stream_resume
from spire_learn.neural.config import load_neural_config
from tests.unit.test_neural_curriculum_integration import blocked_optional

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'ironclad-twenty-hour-continuation-20261103-repaired'
CONFIG = ROOT/'configs/neural/experiments/twenty-hour-20261103-continuation-3'
CACHED = ROOT/'artifacts/development/neural-combat/twenty-hour-20261103-continuation-2/inputs/package-a'


class RepairedContinuationTests(unittest.TestCase):
    def test_fresh_identity_debits_both_unknown_leases_without_replaying_them(self):
        with profiles.scope(PROFILE), blocked_optional() as imports:
            self.assertEqual(profiles.current().campaign_id, 'curriculum-20261103-twenty-hour-continuation-3')
            self.assertEqual(str(profiles.current().domain),
                'artifacts/development/neural-combat/twenty-hour-20261103-continuation-3')
            caps = controls.release_caps(ROOT)['global_effect_caps']
            audit = controls.validate_predecessor(ROOT)
            self.assertEqual(caps, dict(environment_attempts=4538880, sample_rows=4375040,
                optimizer_attempts=273416, reset_slots=4392512, fixed_random_choices=0))
            self.assertEqual(audit['failed_shard']['lease']['environment_attempts'], 4096)
            additional = package.read_json(ROOT/profiles.current().additional_predecessor_path)
            self.assertEqual(additional['charged_or_reserved'], dict(environment_attempts=512,
                sample_rows=512, optimizer_attempts=8, reset_slots=256, fixed_random_choices=0))
            authority = controls._validate_repaired_continuation_authority(ROOT)
            self.assertEqual(authority['standing_user_authority']['instruction'], 'Fix that, then start more training')
            self.assertIsNone(additional['failed_job']['actual'])
            self.assertIsNone(additional['failed_job']['receipt_binding'])
            for key, value in caps.items():
                self.assertLessEqual(value, controls.TWENTY_HOUR_LIFETIME_CAPS[key]
                    - audit['charged_or_reserved'][key] - additional['charged_or_reserved'][key])
        self.assertEqual(imports, [])

    def test_structural_caps_need_no_historical_files_and_cannot_refund_admission(self):
        with profiles.scope(PROFILE), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (profiles.current().campaign_path, profiles.current().predecessor_path,
                         profiles.current().additional_predecessor_path):
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            self.assertEqual(controls.release_caps(root)['global_effect_caps']['sample_rows'], 4375040)
            path = root/profiles.current().additional_predecessor_path
            document = package.read_json(path)
            document['charged_or_reserved']['sample_rows'] = 0
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                controls.release_caps(root)

    def test_host_release_admission_authenticates_failed_admission_history(self):
        with profiles.scope(PROFILE), tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            release, snapshot = base/'release.json', base/'snapshot.json'
            release.write_text('{}\n')
            snapshot.write_text(json.dumps(dict(schema=controls.SOURCE_SCHEMA, state='source_only',
                learner_head='a'*40, dirty={}, files={'entry': {}}, modes={'entry': 0o644})))
            additional = package.read_json(ROOT/profiles.current().additional_predecessor_path)
            changed = ROOT/additional['artifacts']['attempt']['path']
            real_sha = package.sha256
            def corrupt(path):
                return '0'*64 if Path(path) == changed else real_sha(path)
            # Inject a changed byte hash at the filesystem boundary, preserving
            # every host admission call and leaving retained artifacts untouched.
            with patch.object(package, 'sha256', side_effect=corrupt), \
                    self.assertRaisesRegex(ValueError, 'failed admission raw artifact changed: attempt'):
                controls.validate_release(ROOT, package_path=base/'package.json',
                    source_snapshot=snapshot, release_path=release)

    def test_only339_bridges_source_then_saved339_accepts_ordinary340(self):
        from spire_learn.neural.scenario_support import json_value
        from spire_learn.neural.stream_runner import _toml
        from spire_learn.neural.stream_resume import stream_checkpoint_state
        import tomllib
        parent_doc = package.read_json(CONFIG/'parents.json')['standalone_b']
        metadata = package.read_json(ROOT/parent_doc['checkpoint_directory']/'metadata.json')['run_metadata']
        campaign = package.read_json(CONFIG/'campaign.json')
        with profiles.scope(PROFILE), blocked_optional() as imports, \
                tempfile.TemporaryDirectory(dir=ROOT/'artifacts/development/neural-combat') as temporary:
            base = Path(temporary)
            for name in ('recipe.json', 'support.json', 'shard-339.json', 'shard-340.json', 'config-340.toml'):
                shutil.copyfile(CACHED/'training/mixed'/name, base/name)
            template = tomllib.loads((CONFIG/'template.toml').read_text())
            previous = tomllib.loads((CACHED/'training/mixed/config-339.toml').read_text())
            first = package._training_document(template, campaign, 'mixed', 339,
                previous['scenario_stream']['distribution_sha256'])
            (base/'config-339.toml').write_text(_toml(first))
            config339, config340 = [load_neural_config(base/f'config-{index}.toml') for index in (339, 340)]
            transition = stream_resume.validate_stream_resume(metadata, config339)
            self.assertEqual((transition['current']['shard_index'], transition['destination']['shard_index']), (337, 339))
            self.assertEqual(transition['resume_skip_authorization']['campaign_id'], campaign['campaign_id'])
            saved339 = dict(metadata, controller_id='neural-policy-resume-v1',
                config_sha256=config339.config_sha256, manifest_sha256=config339.manifest_sha256,
                scenario_stream=json_value(stream_checkpoint_state(config339, transition['sampler_state'])),
                scenario_stream_transition=dict(identity=transition['identity'],
                    **{'from': transition['current'], 'to': transition['destination']},
                    resume_skip_authorization=transition['resume_skip_authorization']))
            next_ = stream_resume.validate_stream_resume(saved339, config340)
            self.assertEqual((next_['current']['shard_index'], next_['destination']['shard_index']), (339, 340))
            self.assertEqual(next_['sampler_state'], transition['sampler_state'])
            wrong = copy.deepcopy(saved339)
            wrong['scenario_stream_transition']['resume_skip_authorization']['skipped_lease']['sample_rows'] = 0
            with self.assertRaises(ValueError):
                stream_resume.validate_stream_resume(wrong, config340)
            bridge = dict(schema=producer.SOURCE_TRANSITION_SCHEMA,
                purpose='authorized-checkpoint-continuation',
                from_source_sha256=campaign['continuation']['parent_source_sha256'],
                to_source_sha256='a'*64,
                predecessor_sha256=campaign['continuation']['parent_receipt_sha256'],
                checkpoint_inventory_sha256=campaign['continuation']['parent_checkpoint_inventory_sha256'],
                user_approval_sha256=package.sha256(CONFIG/'launch-authority.json'))
            context = dict(source_transition=bridge, campaign_id=campaign['campaign_id'], kind='segment',
                arm='mixed', shard_index=339, source_sha256='a'*64,
                predecessor=dict(path='inputs/predecessor.json', sha256=bridge['predecessor_sha256']),
                parent=dict(path='/inputs/parent', inventory=parent_doc['inventory']))
            self.assertEqual(producer.validate_source_transition(context), bridge['from_source_sha256'])
            with self.assertRaises(ValueError):
                producer.validate_source_transition(dict(context, shard_index=340))
        self.assertEqual(imports, [])
