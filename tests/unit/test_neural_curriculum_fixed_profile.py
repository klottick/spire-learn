"""Fixed Gym deployment: real source/guest/producer/reconciler, pure OS fakes."""
from contextlib import ExitStack
import copy
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
FRESH_PROFILE = 'fixed-gym-a86df922-c'
FRESH_DOMAIN = Path('artifacts/development/neural-combat/curriculum-campaign-v3')


def retained_qualification():
    """Read-only retrospective inputs; fresh in-memory budget, historical clock."""
    from scripts.neural import curriculum_controls as controls
    runtime = ROOT/DOMAIN/'runtime'
    job = runtime/'jobs/curriculum-qualification'
    context = controls._read(job/'control/job.json')
    release = controls._read(job/'control/release.json')
    package_path = ROOT/DOMAIN/'preparation/package-b/package.json'
    attempt = controls._read(runtime/'attempts/curriculum-qualification.json')
    commands = attempt['commands']
    create = next(row for row in commands if len(row['argv']) > 5 and row['argv'][5] == 'create')
    removed = next(row for row in commands if len(row['argv']) > 5 and row['argv'][5] == 'rm')
    start = next(row for row in commands if len(row['argv']) > 5 and row['argv'][5] == 'start')
    admission = json.loads(next(line[len(controls.transport.MARKER):]
        for line in start['stderr'].splitlines() if line.startswith(controls.transport.MARKER)))
    cid = create['stdout'].strip()
    assert create['exit_code'] == removed['exit_code'] == start['exit_code'] == 0
    assert removed['argv'][-1] == cid
    assert any(row['argv'][5:] == ['ps', '-a', '--filter', 'id='+cid, '--format', '{{.ID}}']
        and row['exit_code'] == 0 and not row['stdout'].strip() for row in commands if len(row['argv']) > 5)
    seal = controls._read(runtime/'campaign-result.json')
    historical_end = seal['parent_start']+seal['elapsed_seconds']
    assert historical_end < context['whole_deadline']
    budget = controls.CampaignBudget(controls.CAPS, start=seal['parent_start'],
        deadline=seal['whole_deadline'], clock=lambda: historical_end)
    budget.reserve(context['job_id'], context['reservation'])
    verified = {'release': release, 'package': controls._read(package_path), 'package_path': package_path}
    return controls, verified, context, job/'output', dict(container_id=cid,
        removal_entered=True, admission=admission), budget, historical_end


class RetainedReceiptTests(unittest.TestCase):
    def test_real_completed_qualification_crosses_actual_process_validator(self):
        from scripts.neural import curriculum_profiles as profiles
        controls, verified, context, output, lifecycle, budget, end = retained_qualification()
        original = (output/'job-receipt.json').read_bytes()
        with profiles.scope(PROFILE), blocked_optional():
            result = controls.reconcile_job(ROOT, verified, context, output,
                lifecycle=lifecycle, budget=budget, clock=lambda: end)
        self.assertEqual(result['receipt']['qualification_attempts'], {'actions': 128, 'adam': 8, 'resets': 13, 'samples': 128})
        self.assertEqual((output/'job-receipt.json').read_bytes(), original)

    def test_real_module_facts_reject_alien_origins_and_old_profile(self):
        from scripts.neural import curriculum_profiles as profiles
        controls, verified, context, output, lifecycle, budget, end = retained_qualification()
        receipt = controls._read(output/'job-receipt.json')
        with profiles.scope(PROFILE), blocked_optional():
            for name, path in [('torch', '/runtime/gym-overlay/torch/__init__.py'),
                               ('numpy', '/alien/numpy/__init__.py'),
                               ('gymnasium', '/runtime/gym-overlay/gymnasium/__init__.py'),
                               ('spire_lab', '/alien/spire_lab/__init__.py'),
                               ('spire_lab', '/runtime/gym-overlay/spire_lab/../spire_lab/__init__.py'),
                               ('spire_lab', '/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/__init__.py')]:
                changed = copy.deepcopy(receipt)
                changed['process']['modules'][name]['file'] = path
                with self.subTest(module=name, path=path), self.assertRaises(ValueError):
                    profiles.validate_process_evidence(ROOT, changed, runtime_pins=verified['release']['runtime_pins'])
            profiles.validate_process_evidence(ROOT, receipt, runtime_pins=verified['release']['runtime_pins'])
        with profiles.scope('historical'), self.assertRaises(ValueError):
            profiles.validate_process_evidence(ROOT, receipt, runtime_pins=verified['release']['runtime_pins'])

    def test_fresh_profile_uses_new_claim_domain_and_same_weights_budget(self):
        from scripts.neural import curriculum_profiles as profiles
        from scripts.neural import curriculum_package as package
        with profiles.scope(FRESH_PROFILE):
            selected = profiles.current()
            self.assertEqual(selected.campaign_id, 'curriculum-20261101-c')
            self.assertEqual(selected.domain, FRESH_DOMAIN)
            self.assertEqual(selected.qualification_input, FRESH_DOMAIN/'qualification')
            self.assertEqual(package.read_json(ROOT/selected.parents), package.read_json(ROOT/profiles.FIXED.parents))
            fresh = package.read_json(ROOT/selected.campaign_path)
            old = package.read_json(ROOT/profiles.FIXED.campaign_path)
            self.assertEqual({k:v for k,v in fresh.items() if k != 'campaign_id'},
                             {k:v for k,v in old.items() if k != 'campaign_id'})


def fixed_fixture():
    from scripts.neural import curriculum_package as package
    destination = ROOT/DOMAIN/'preparation/package-b'
    if not destination.exists():
        package.refresh_package(ROOT, package_fixture()[0]/'package.json', destination)
    return destination, package.read_json(destination/'package.json')


def fresh_fixture():
    from scripts.neural import curriculum_package as package
    destination = ROOT/FRESH_DOMAIN/'preparation/package-c'
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
    profile_name = PROFILE
    campaign_id = 'curriculum-20261101-b'
    for _name in ('prepare', 'docker', 'project', 'copy_output', 'execute'):
        locals()[_name] = getattr(shared.ProjectedHostTests, _name)

    def setUp(self):
        from scripts.neural import curriculum_profiles as profiles
        self.profile_stack = ExitStack()
        self.addCleanup(self.profile_stack.close)
        self.profile_stack.enter_context(profiles.scope(self.profile_name))
        (ROOT/self.test_domain/'task-2').mkdir(exist_ok=True)
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
        # The original fake omitted Gym from process.modules, hiding the missing
        # receipt gate. Mirror the four actual retained module facts instead.
        from scripts.neural import sustained_runtime
        receipt = self.api._read(ROOT/DOMAIN/'runtime/jobs/curriculum-qualification/output/job-receipt.json')
        stack.enter_context(patch.object(sustained_runtime, 'process_evidence', return_value=receipt['process']))
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
        self.assertEqual(argv[-2:], ['--profile', self.profile_name])
        schedule = self.api.CampaignSchedule(self.package, start=100.)
        schedule.accept_qualification()
        self.assertEqual(schedule.next(101.), {'kind': 'panel', 'slot': 'warm-start-fixed-003', 'arm': None})

    def test_actual_host_release_validation_and_downgrade_rejection(self):
        with blocked_optional():
            accepted = self.api.validate_release(ROOT, package_path=self.fixture/'package.json',
                source_snapshot=self.source_path, release_path=self.release_path)
        self.assertEqual(accepted['package']['campaign_id'], self.campaign_id)
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
        config = self.guest/self.test_domain/'inputs/training/fixed/config-0.toml'
        config.write_text(config.read_text().replace(REVISION, '533dad1c8aec86df624b3742b6b3ad1dbf1d0033'))
        binding = package.inventory(self.guest/self.test_domain/'inputs', ['training/fixed/config-0.toml'])
        manifest = self.guest/self.test_domain/'inputs/package.json'
        document = json.loads(manifest.read_text())
        document['inventory'].update(binding)
        document['training']['fixed'][0] = package._config_row(manifest.parent, 'training/fixed/config-0.toml')
        manifest.write_text(json.dumps(document))
        self.context['package_sha256'] = package.sha256(manifest)
        self.context['input_inventory'].update(binding)
        # Exercise producer's actual closed selected-config validation directly;
        # acyclic release chain would independently reject these rebound bytes.
        self.api.producer.validate_context(self.guest, self.context, clock=lambda: self.now)


class FreshProjectedTests(FixedProjectedTests):
    fixture_provider = staticmethod(fresh_fixture)
    test_domain = FRESH_DOMAIN
    profile_name = FRESH_PROFILE
    campaign_id = 'curriculum-20261101-c'


if __name__ == '__main__':
    unittest.main()
