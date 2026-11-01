"""Fixed Gym deployment: real source/guest/producer/reconciler, pure OS fakes."""
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit.test_neural_curriculum_package import ROOT, package_fixture
from tests.unit import test_neural_curriculum_controls as shared
from tests.unit.test_neural_curriculum_integration import blocked_optional

PROFILE = 'fixed-gym-a86df922'
DOMAIN = Path('artifacts/development/neural-combat/curriculum-campaign-v2')
REVISION = 'a86df922a9dc598645820cc6544df36aefb3d2cc'


def fixed_fixture():
    from scripts.neural import curriculum_package as package
    destination = ROOT/DOMAIN/'preparation/package-b'
    if not destination.exists():
        package.refresh_package(ROOT, package_fixture()[0]/'package.json', destination)
    return destination, package.read_json(destination/'package.json')


class FixedProfileContractTests(unittest.TestCase):
    def test_explicit_profile_capture_and_unknown_selection_are_closed(self):
        from scripts.neural import curriculum_controls as controls
        with tempfile.TemporaryDirectory(dir=ROOT/DOMAIN/'preparation') as temporary, blocked_optional():
            destination = Path(temporary)/'snapshot.json'
            with patch.object(controls, 'capture_source', return_value={'captured': True}), \
                 patch('scripts.neural.gpu_runtime_proof_helpers._git_state', return_value=('a'*40, {})):
                self.assertEqual(controls.main(['capture', '--profile', PROFILE,
                    '--root', str(ROOT), '--destination', str(destination)]), 0)
            with self.assertRaises(SystemExit) as stopped:
                controls.main(['capture', '--profile', 'unreviewed', '--root', str(ROOT)])
            self.assertEqual(stopped.exception.code, 2)

    def test_refresh_uses_fixed_revision_same_data_and_fresh_parent(self):
        from scripts.neural import curriculum_profiles as profiles, curriculum_package as package
        from spire_learn.neural.config import load_neural_config
        with profiles.scope(PROFILE), blocked_optional():
            destination, result = fixed_fixture()
            old_path, old = package_fixture()
            self.assertEqual(result['campaign_id'], 'curriculum-20261101-b')
            self.assertEqual(result['evaluation_slots'][0], 'warm-start-fixed-003')
            self.assertIn('curriculum-fixed-003/output/', result['original_parent']['checkpoint_directory'])
            for arm in ('fixed', 'randomized', 'mixed'):
                row = result['training'][arm][0]
                actual = load_neural_config(destination/row['config_path'])
                self.assertEqual(actual.environment_revision, REVISION)
                self.assertEqual((actual.rollout_rows, actual.rollout_steps), (16, 32))
                self.assertNotEqual(row['config_sha256'], old['training'][arm][0]['config_sha256'])
                for name in ('shard-0.json', 'recipe.json', 'support.json'):
                    self.assertEqual((destination/f'training/{arm}'/name).read_bytes(),
                                     (old_path/f'training/{arm}'/name).read_bytes())
            self.assertEqual(result['coverage'], old['coverage'])
            q = ROOT/profiles.current().qualification_input
            for index in (0, 1):
                self.assertEqual(load_neural_config(q/f'qualification-{index}.toml').environment_revision, REVISION)

    def test_overlay_authenticates_closed_inventory_and_actual_import_origin(self):
        from scripts.neural import curriculum_profiles as profiles
        from scripts.neural import curriculum_controls as controls
        with profiles.scope(PROFILE), blocked_optional():
            pins = controls.runtime_pins(ROOT)
            self.assertEqual(pins['native_sha256'], '96d54621aa533f27ca720621dfa17664609e51e5b9311fed1636cffa8cd9d787')
            self.assertGreater(len(pins['package_overlay']['inventory']), 20)
            self.assertEqual(pins['gym_revision'], REVISION)
            self.assertEqual(pins['package_overlay'].get('guest_destination'), '/runtime/gym-overlay')
            self.assertIs(pins['package_overlay'].get('readonly'), True)
            self.assertEqual(pins['package_overlay'].get('pythonpath'), '/runtime/gym-overlay:/learner/src:/learner')
            with tempfile.TemporaryDirectory() as directory:
                overlay = Path(directory)/'overlay'
                shutil.copytree(ROOT/pins['package_overlay']['path'], overlay)
                expected = pins['package_overlay']['inventory']
                profiles.verify_overlay(overlay, expected)
                (overlay/'spire_lab/readiness.py').chmod(0o644)
                (overlay/'spire_lab/readiness.py').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    profiles.verify_overlay(overlay, expected)
                shutil.copyfile(ROOT/pins['package_overlay']['path']/'spire_lab/readiness.py', overlay/'spire_lab/readiness.py')
                overlay.chmod(0o755)
                (overlay/'extra.py').write_text('unbound')
                with self.assertRaisesRegex(ValueError, 'closed'):
                    profiles.verify_overlay(overlay, expected)
            site = ROOT/pins['package_overlay']['path']
            fake = SimpleNamespace(__file__=str(site/'spire_lab/__init__.py'))
            native = SimpleNamespace(__file__=str(site/'spire_lab/_native.cpython-312-x86_64-linux-gnu.so'),
                build_info=lambda: {'extension': 'spire-python', 'revision': REVISION})
            with patch.dict('sys.modules', {'spire_lab': fake, 'spire_lab._native': native}):
                actual = profiles.verify_imported_overlay(site, pins)
                self.assertEqual(actual['build_info']['revision'], REVISION)
                native.build_info = lambda: {'extension': 'spire-python', 'revision': '533dad1'}
                with self.assertRaisesRegex(ValueError, 'revision'):
                    profiles.verify_imported_overlay(site, pins)
                native.build_info = lambda: {'extension': 'spire-python', 'revision': REVISION}
                fake.__file__ = '/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/__init__.py'
                with self.assertRaisesRegex(ValueError, 'origin'):
                    profiles.verify_imported_overlay(site, pins)


class FixedProjectedTests(unittest.TestCase):
    """Reuse strict transport harness; select only meaningful projected cases."""
    fixture_provider = staticmethod(fixed_fixture)
    test_domain = DOMAIN
    for _name in ('prepare', 'docker', 'project', 'copy_output', 'execute'):
        locals()[_name] = getattr(shared.ProjectedHostTests, _name)

    def setUp(self):
        from scripts.neural import curriculum_profiles as profiles
        self.profile_stack = ExitStack()
        self.addCleanup(self.profile_stack.close)
        self.profile_stack.enter_context(profiles.scope(PROFILE))
        (ROOT/DOMAIN/'task-2').mkdir(exist_ok=True)
        shared.ProjectedHostTests.setUp(self)
        overlay = ROOT/'artifacts/development/neural-combat/gym-refresh-20261101/site'
        self.mapping['/runtime/gym-overlay'] = self.base/'gym-overlay'
        self.mapping['/runtime/requirements-cuda.lock'] = ROOT/self.api.low.LOCK
        self.imports = []

    def guest_boundaries(self):
        from scripts.neural import curriculum_profiles as profiles
        stack = shared.ProjectedHostTests.guest_boundaries(self)
        site = self.mapping['/runtime/gym-overlay']
        # External package import only; actual full inventory/native/lock gates run.
        fake = SimpleNamespace(__file__=str(site/'spire_lab/__init__.py'))
        native = SimpleNamespace(__file__=str(site/'spire_lab/_native.cpython-312-x86_64-linux-gnu.so'),
            build_info=lambda: {'extension': 'spire-python', 'revision': REVISION})
        stack.enter_context(patch.dict('sys.modules', {'spire_lab': fake, 'spire_lab._native': native}))
        return stack

    def test_fixed_projection_qualification_warm_resume_panel_and_cleanup(self):
        result = self.execute(self.prepare('qualification', job='fixed-qualification'))
        self.assertEqual(result['receipt']['qualification_attempts']['adam'], 8)
        self.assertEqual(dict(self.spec.environment)['PYTHONPATH'], '/runtime/gym-overlay:/learner/src:/learner')
        self.assertIn('--profile', self.spec.command)
        first = self.execute(self.prepare(arm='mixed', index=0, job='fixed-mixed-0', max_updates=1))
        self.assertEqual(first['checkpoint']['counters']['update_count'], 1)
        second = self.execute(self.prepare(arm='mixed', index=1, job='fixed-mixed-1', max_updates=2,
            prefix=first['checkpoint']['counters'], parent=first['checkpoint'], predecessor=first['receipt_binding']))
        self.assertEqual(second['checkpoint']['counters']['update_count'], 2)
        panel = self.execute(self.prepare('panel', job='fixed-panel', slot='warm-start-fixed-003'))
        self.assertEqual(len(panel['receipt']['reports']), 64)
        self.assertEqual(len(self.removed), 4)
        self.assertTrue(all(driver.close_calls == 1 for driver in self.harness.drivers))

    def test_profile_in_controller_supervision_and_fixed_baseline_schedule(self):
        argv = self.api._controller_argv(ROOT, self.fixture/'package.json', self.source_path, self.release_path)
        self.assertEqual(argv[-2:], ['--profile', PROFILE])
        schedule = self.api.CampaignSchedule(self.package, start=100.)
        schedule.accept_qualification()
        self.assertEqual(schedule.next(101.), {'kind': 'panel', 'slot': 'warm-start-fixed-003', 'arm': None})

    def test_actual_host_release_validation_and_downgrade_rejection(self):
        with blocked_optional():
            accepted = self.api.validate_release(ROOT, package_path=self.fixture/'package.json',
                source_snapshot=self.source_path, release_path=self.release_path)
        self.assertEqual(accepted['package']['campaign_id'], 'curriculum-20261101-b')
        release = json.loads(self.release_path.read_text())
        release['runtime_pins']['native_sha256'] = self.api.low.NATIVE_SHA256
        self.release_path.write_text(json.dumps(release))
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            self.api.validate_release(ROOT, package_path=self.fixture/'package.json',
                source_snapshot=self.source_path, release_path=self.release_path)

    def test_rebound_old_revision_config_is_rejected_before_backend(self):
        prepared = self.prepare(arm='fixed', index=0, job='bad-revision')
        self.project = self._wrong_revision_projection
        with self.assertRaises(ValueError):
            self.execute(prepared)
        self.assertEqual(self.harness.factory_calls, [])
        self.assertEqual(self.removed, ['a'*64])

    def test_overlay_readwrite_mount_is_rejected_before_backend(self):
        destination = str(self.mapping['/runtime/gym-overlay'])
        def mutate(facts):
            facts['mounts'] = [(path, False if path == destination else readonly)
                               for path, readonly in facts['mounts']]
        self.guest_mutation = mutate
        with self.assertRaises(ValueError):
            self.execute(self.prepare(arm='fixed', index=0, job='bad-overlay-mount'))
        self.assertEqual(self.harness.factory_calls, [])
        self.assertEqual(self.removed, ['a'*64])

    def _wrong_revision_projection(self):
        shared.ProjectedHostTests.project(self)
        # Rebind selected bytes and package/context consistently: revision gate
        # must still refuse this old configuration under the fixed deployment.
        from scripts.neural import curriculum_package as package
        config = self.guest/DOMAIN/'inputs/training/fixed/config-0.toml'
        config.write_text(config.read_text().replace(REVISION, '533dad1c8aec86df624b3742b6b3ad1dbf1d0033'))
        binding = package.inventory(self.guest/DOMAIN/'inputs', ['training/fixed/config-0.toml'])
        manifest = self.guest/DOMAIN/'inputs/package.json'
        document = json.loads(manifest.read_text())
        document['inventory'].update(binding)
        document['training']['fixed'][0] = package._config_row(manifest.parent, 'training/fixed/config-0.toml')
        manifest.write_text(json.dumps(document))
        self.context['package_sha256'] = package.sha256(manifest)
        self.context['input_inventory'].update(binding)
        # Exercise producer's actual closed selected-config validation directly;
        # acyclic release chain would independently reject these rebound bytes.
        self.api.producer.validate_context(self.guest, self.context, clock=lambda: self.now)


if __name__ == '__main__':
    unittest.main()
