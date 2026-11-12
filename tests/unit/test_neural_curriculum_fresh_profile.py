"""Current Gym campaign profile and immutable package projection checks."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'fresh-gym-afb51f5f'
REVISION = 'afb51f5f222b4c4331ed34067c5eebeda606db91'
CURRENT_COMMON = '4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463'
CURRENT_RUN = '28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64'
CURRENT_NATIVE = '1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac'
CURRENT_OVERLAY = 'artifacts/development/neural-combat/gym-migration-20261102/site'
OLD_STANDALONE = 'spire-learn-standalone-common-v1@02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78'


def staged_profile_root(base: Path) -> Path:
    """Build isolated authored inputs and an explicit untrained parent fixture."""
    root = base / 'root'
    current = root / 'configs/neural/experiments/curriculum-v4'
    current.mkdir(parents=True)
    for name in ('bootstrap.toml', 'campaign.json', 'template.toml', 'overlay.json'):
        shutil.copyfile(ROOT / current.relative_to(root) / name, current / name)
    (root / 'configs/neural/gym-20261102').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'configs/neural/gym-20261102/runtime.json',
                    root / 'configs/neural/gym-20261102/runtime.json')

    # This synthetic parent belongs only to the temporary test root. The live
    # fresh parent is created later by the separately bounded bootstrap.
    parent = {
        'schema': 'spire-learn-curriculum-fresh-parents-v1',
        'standalone_b': {
            'checkpoint_directory': 'artifacts/development/neural-combat/curriculum-campaign-v4/bootstrap/checkpoint',
            'inventory': {'metadata.json': {'sha256': 'a' * 64, 'size_bytes': 128}},
            'initialization': 'fresh-current-schema-untrained',
            'trained_decisions': 0,
            'update_count': 0,
        },
    }
    (current / 'parents.json').write_text(json.dumps(parent))
    qsource = current / 'qualification'
    qsource.mkdir()
    (qsource / 'qualification-0.toml').write_text('profile = "temporary-current-0"\n')
    (qsource / 'qualification-1.toml').write_text('profile = "temporary-current-1"\n')
    (current / 'qualification-identities.json').write_text(json.dumps(['b' * 64, 'c' * 64]))

    old_experiment = root / 'configs/neural/experiments/curriculum-v2'
    shutil.copytree(ROOT / 'configs/neural/experiments/curriculum-v2', old_experiment)

    source = ROOT / package.REUSABLE_PACKAGE
    destination = root / package.REUSABLE_PACKAGE
    destination.parent.parent.mkdir(parents=True)
    shutil.copytree(source.parent, destination.parent)
    return root


class FreshProfileSelectionTests(unittest.TestCase):
    def test_current_profile_routes_authored_sources_without_changing_legacy_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_profile_root(Path(temporary))
            with profiles.scope(PROFILE):
                selected = profiles.current()
                self.assertEqual(selected.name, PROFILE)
                self.assertEqual(selected.campaign_id, 'curriculum-20261102-a')
                self.assertEqual(selected.campaign_path,
                                 'configs/neural/experiments/curriculum-v4/campaign.json')
                self.assertEqual(selected.domain,
                                 Path('artifacts/development/neural-combat/curriculum-campaign-v4'))
                self.assertEqual(selected.qualification_input,
                                 Path('artifacts/development/neural-combat/curriculum-campaign-v4/qualification'))
                self.assertEqual(selected.revision, REVISION)
                self.assertEqual(selected.baseline_slot, 'fresh-initial')
                self.assertEqual(selected.template,
                                 'configs/neural/experiments/curriculum-v4/template.toml')
                self.assertEqual(selected.parents,
                                 'configs/neural/experiments/curriculum-v4/parents.json')
                self.assertEqual(selected.runtime_profile,
                                 'configs/neural/gym-20261102/runtime.json')
                self.assertEqual(selected.overlay_manifest,
                                 'configs/neural/experiments/curriculum-v4/overlay.json')
                self.assertEqual(selected.native_sha256, CURRENT_NATIVE)
                self.assertEqual(selected.package_overlay, CURRENT_OVERLAY)
                self.assertEqual(profiles.qualification_source(root),
                                 root / 'configs/neural/experiments/curriculum-v4/qualification')
                self.assertEqual(profiles.qualification_hashes(root), ['b' * 64, 'c' * 64])
                paths = set(profiles.source_paths(root))
                self.assertTrue({
                    'configs/neural/experiments/curriculum-v4/campaign.json',
                    'configs/neural/experiments/curriculum-v4/bootstrap.toml',
                    'configs/neural/experiments/curriculum-v4/template.toml',
                    'configs/neural/experiments/curriculum-v4/parents.json',
                    'configs/neural/experiments/curriculum-v4/qualification-identities.json',
                    'configs/neural/experiments/curriculum-v4/qualification/qualification-0.toml',
                    'configs/neural/experiments/curriculum-v4/overlay.json',
                    'configs/neural/gym-20261102/runtime.json',
                    'tests/unit/test_neural_curriculum_fresh_profile.py',
                } <= paths)
            with profiles.scope('fixed-gym-a86df922'):
                self.assertEqual(profiles.qualification_source(root),
                                 root / 'configs/neural/experiments/curriculum-v2/qualification')
                self.assertEqual(profiles.qualification_hashes(root), json.loads(
                    (root / 'configs/neural/experiments/curriculum-v2/qualification-identities.json').read_text()))
                legacy_paths = set(profiles.source_paths(root))
                self.assertIn('configs/neural/experiments/curriculum-v2/campaign.json', legacy_paths)
                self.assertIn('tests/unit/test_neural_curriculum_fixed_profile.py', legacy_paths)
                self.assertNotIn('configs/neural/experiments/curriculum-v4/campaign.json', legacy_paths)


class CurrentContractProjectionTests(unittest.TestCase):
    def test_projection_refreshes_every_current_fingerprint_and_mapping(self):
        document = {
            'environment': {
                'gym_revision': '533dad1c8aec86df624b3742b6b3ad1dbf1d0033',
                'common_view_fingerprint': 'old-common',
                'run_observation_fingerprint': 'old-run',
            },
            'accepted_identities': [{
                'accepted_raw_mode_mappings': {'standalone': OLD_STANDALONE, 'run': 'old-run-map'},
            }],
        }
        with profiles.scope(PROFILE):
            package._project_gym_contract(document, profiles.current().revision)
        self.assertEqual(document['environment'], {
            'gym_revision': REVISION,
            'common_view_fingerprint': CURRENT_COMMON,
            'run_observation_fingerprint': CURRENT_RUN,
        })
        self.assertEqual(document['accepted_identities'][0]['accepted_raw_mode_mappings'], {
            'standalone': 'spire-learn-standalone-common-v1@' + CURRENT_COMMON,
            'run': 'spire-learn-run-policy-v3-common-v1@' + CURRENT_RUN + ':' + CURRENT_COMMON,
        })

    def test_current_package_refresh_uses_real_loader_and_preserves_scenario_bytes(self):
        from spire_learn.neural.config import load_neural_config

        source_path = ROOT / package.REUSABLE_PACKAGE
        source = package.read_json(source_path)
        test_work = ROOT / 'artifacts/development/neural-combat/curriculum-campaign-v4/task-1'
        test_work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_work) as temporary:
            root = staged_profile_root(Path(temporary))
            destination = (root / 'artifacts/development/neural-combat/curriculum-campaign-v4'
                           / 'preparation/package-current')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with profiles.scope(PROFILE):
                result = package.refresh_package(root, root / package.REUSABLE_PACKAGE, destination)
                self.assertEqual(result['campaign_id'], 'curriculum-20261102-a')
                current_campaign = json.loads((root / profiles.current().campaign_path).read_text())
                self.assertEqual(result['campaign'], current_campaign)
                self.assertEqual(result['evaluation_slots'], current_campaign['evaluation_slots'])
                self.assertEqual(result['evaluation_slots'][0], 'fresh-initial')
                self.assertEqual(result['original_parent'], json.loads(
                    (root / profiles.current().parents).read_text())['standalone_b'])
                for arm in ('fixed', 'randomized', 'mixed'):
                    config_path = destination / result['training'][arm][0]['config_path']
                    config = load_neural_config(config_path)
                    self.assertEqual(config.environment_revision, REVISION)
                    self.assertEqual(config.common_view_fingerprint, CURRENT_COMMON)
                    self.assertEqual(dict(config.accepted_identities[0].accepted_raw_mode_mappings), {
                        'standalone': 'spire-learn-standalone-common-v1@' + CURRENT_COMMON,
                        'run': 'spire-learn-run-policy-v3-common-v1@' + CURRENT_RUN + ':' + CURRENT_COMMON,
                    })
                for name in source['inventory']:
                    if name in {'campaign.json', 'original-parent.json'} or name.endswith('.toml'):
                        continue
                    self.assertEqual((destination / name).read_bytes(),
                                     (source_path.parent / name).read_bytes(), name)
                for arm in ('fixed', 'randomized', 'mixed'):
                    for name in ('shard-0.json', 'recipe.json', 'support.json'):
                        relative = f'training/{arm}/{name}'
                        self.assertEqual((destination / relative).read_bytes(),
                                         (source_path.parent / relative).read_bytes())

                config_path = destination / result['training']['fixed'][0]['config_path']
                original = config_path.read_text()
                config_path.write_text(original.replace(
                    'spire-learn-run-policy-v3-common-v1@' + CURRENT_RUN + ':' + CURRENT_COMMON,
                    'spire-learn-run-policy-v3-common-v1@30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8:02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78'))
                with self.assertRaisesRegex(ValueError, 'mapping does not match'):
                    load_neural_config(config_path)

    def test_current_overlay_and_receipt_gates_bind_the_new_native_and_origin(self):
        from scripts.neural import sustained_controls

        with profiles.scope(PROFILE):
            pins = profiles.overlay_pins(ROOT)
            self.assertEqual(pins['gym_revision'], REVISION)
            self.assertEqual(pins['native_sha256'], CURRENT_NATIVE)
            self.assertEqual(pins['package_overlay']['path'], CURRENT_OVERLAY)
            self.assertEqual(pins['package_overlay']['guest_destination'], '/runtime/gym-overlay')
            runtime = {**pins, 'versions': {'spire-lab': '0.1.0'}}
            receipt = {'process': {'modules': {
                'spire_lab': {'file': '/runtime/gym-overlay/spire_lab/__init__.py', 'version': '0.1.0'},
                'numpy': {'file': '/runtime/venv-cuda/lib/python3.12/site-packages/numpy/__init__.py',
                          'version': '2.3.2'},
            }}}
            with patch.object(sustained_controls, 'validate_process_evidence') as transport:
                profiles.validate_process_evidence(ROOT, receipt, runtime_pins=runtime)
                forwarded = transport.call_args.args[0]
                self.assertEqual(set(forwarded['process']['modules']), {'numpy'})
                self.assertIs(transport.call_args.kwargs['reference'], False)
                alien = json.loads(json.dumps(receipt))
                alien['process']['modules']['spire_lab']['file'] = '/alien/spire_lab/__init__.py'
                with self.assertRaisesRegex(ValueError, 'overlay origin/version'):
                    profiles.validate_process_evidence(ROOT, alien, runtime_pins=runtime)
                downgraded = {**runtime, 'native_sha256': '96d54621aa533f27ca720621dfa17664609e51e5b9311fed1636cffa8cd9d787'}
                with self.assertRaisesRegex(ValueError, 'package overlay differs'):
                    profiles.validate_process_evidence(ROOT, receipt, runtime_pins=downgraded)


if __name__ == '__main__':
    unittest.main()
