"""Pure admission and accounting checks for the v5 continuation profile."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from scripts.neural import curriculum_runtime as producer

ROOT = Path(__file__).resolve().parents[2]
PROFILE_A = 'fresh-gym-afb51f5f'
PROFILE_B = 'fresh-gym-afb51f5f-b'
CURRENT_NATIVE = '1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac'
V4 = Path('artifacts/development/neural-combat/curriculum-campaign-v4')
PREDECESSOR_PATHS = {
    'audit': V4/'preflight-failure-audit.json',
    'result': V4/'runtime/campaign-result.json',
    'attempt': V4/'runtime/attempts/curriculum-qualification.json',
    'launcher_result': V4/'launcher-result.json',
    'launcher_os_exit': V4/'launcher-os-exit-observed.json',
}
FAILED_LEASE = {
    'environment_attempts': 512,
    'sample_rows': 512,
    'optimizer_attempts': 8,
    'reset_slots': 256,
    'fixed_random_choices': 0,
}
REMAINING_CAPS = {
    'environment_attempts': 1199488,
    'sample_rows': 1099488,
    'optimizer_attempts': 69992,
    'reset_slots': 1099744,
    'fixed_random_choices': 0,
}


def staged_predecessor_root(base: Path) -> Path:
    root = base / 'root'
    for name in PREDECESSOR_PATHS.values():
        source, destination = ROOT/name, root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    source = ROOT/'configs/neural/experiments/curriculum-v5/predecessor.json'
    destination = root/'configs/neural/experiments/curriculum-v5/predecessor.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return root


def staged_projection_root(base: Path) -> Path:
    root = base/'root'
    for version in ('curriculum-v4', 'curriculum-v5'):
        destination = root/'configs/neural/experiments'/version
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT/'configs/neural/experiments'/version, destination)
    source = ROOT/package.REUSABLE_PACKAGE
    destination = root/package.REUSABLE_PACKAGE
    destination.parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.parent, destination.parent)
    return root


def staged_caps_root(base: Path) -> Path:
    root = base/'root'
    for name in ('campaign.json', 'predecessor.json'):
        source = ROOT/'configs/neural/experiments/curriculum-v5'/name
        destination = root/'configs/neural/experiments/curriculum-v5'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root


def _runtime_pins(native_sha256):
    return {
        'profile_sha256': controls.cdi.PROFILE_SHA256,
        'native_sha256': native_sha256,
        'lock_sha256': controls.low.LOCK_SHA256,
        'versions': controls.PIN_VERSIONS,
        'python_sha256': controls.PYTHON_SHA256,
    }


class FreshGymContinuationProfileTests(unittest.TestCase):
    def test_b_profile_routes_v4_sources_and_v5_continuation_inputs(self):
        with profiles.scope(PROFILE_B):
            selected = profiles.current()
            self.assertEqual(selected.campaign_id, 'curriculum-20261102-b')
            self.assertEqual(selected.domain,
                             Path('artifacts/development/neural-combat/curriculum-campaign-v5'))
            self.assertEqual(selected.qualification_input,
                             Path('artifacts/development/neural-combat/curriculum-campaign-v5/qualification'))
            self.assertEqual(selected.template, profiles.FRESH_GYM.template)
            self.assertEqual(selected.parents, profiles.FRESH_GYM.parents)
            self.assertEqual(selected.overlay_manifest, profiles.FRESH_GYM.overlay_manifest)
            self.assertEqual(selected.runtime_profile, profiles.FRESH_GYM.runtime_profile)
            self.assertEqual(selected.revision, profiles.FRESH_GYM.revision)
            self.assertEqual(selected.native_sha256, CURRENT_NATIVE)
            self.assertEqual(selected.package_overlay, profiles.FRESH_GYM.package_overlay)
            self.assertEqual(selected.whole_seconds, 35400)
            self.assertTrue(profiles.is_fresh_gym_family())
            self.assertEqual(profiles.qualification_source(ROOT),
                             ROOT/'configs/neural/experiments/curriculum-v4/qualification')
            self.assertEqual(profiles.qualification_hashes(ROOT), package.read_json(
                ROOT/'configs/neural/experiments/curriculum-v4/qualification-identities.json'))
            profile_paths = set(profiles.source_paths(ROOT))
            self.assertTrue({
                'configs/neural/experiments/curriculum-v4/campaign.json',
                'configs/neural/experiments/curriculum-v4/template.toml',
                'configs/neural/experiments/curriculum-v4/parents.json',
                'configs/neural/experiments/curriculum-v4/qualification-identities.json',
                'configs/neural/experiments/curriculum-v5/campaign.json',
                'configs/neural/experiments/curriculum-v5/predecessor.json',
                'tests/unit/test_neural_curriculum_continuation.py',
            } <= profile_paths)
            self.assertIn('scripts/neural/curriculum_fault_capture.py', producer.source_paths(ROOT))
        with profiles.scope(PROFILE_A):
            self.assertTrue(profiles.is_fresh_gym_family())
            self.assertEqual(profiles.current().whole_seconds, 36000)
        with profiles.scope('fixed-gym-a86df922'):
            self.assertFalse(profiles.is_fresh_gym_family())

    def test_b_package_refresh_uses_current_contract_projection(self):
        import tomllib

        with tempfile.TemporaryDirectory() as temporary:
            root = staged_projection_root(Path(temporary))
            source = package.read_json(root/package.REUSABLE_PACKAGE)
            with profiles.scope(PROFILE_B):
                changed = package._refreshed_files(root, source)
                campaign = package.read_json(root/profiles.current().campaign_path)
                train_document = tomllib.loads(changed['training/fixed/config-0.toml'].decode())
                eval_document = tomllib.loads(changed['panel/config-0.toml'].decode())
                accepted_panel = tomllib.loads((root/Path(package.REUSABLE_PACKAGE).parent/
                                                'panel/config-0.toml').read_text())
                self.assertEqual(train_document['budgets'], campaign['training_budgets'])
                self.assertEqual(eval_document['budgets'], accepted_panel['budgets'])
                self.assertEqual(eval_document['device'], 'cpu')
                self.assertEqual(train_document['environment']['gym_revision'], profiles.FRESH_GYM.revision)
                self.assertEqual(train_document['environment']['common_view_fingerprint'],
                             '4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463')
                self.assertEqual(train_document['accepted_identities'][0]['accepted_raw_mode_mappings'], {
                    'standalone': 'spire-learn-standalone-common-v1@4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463',
                    'run': 'spire-learn-run-policy-v3-common-v1@28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64:4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463',
                })
                self.assertNotIn('training/fixed/shard-0.json', changed)
                self.assertNotIn('training/fixed/support.json', changed)

                from spire_learn.neural import config as neural_config
                artifact_root = root/'artifacts/development/neural-combat'
                destination = artifact_root/'curriculum-campaign-v5/preparation/package-b'
                destination.parent.mkdir(parents=True)
                with patch.object(neural_config, '__file__',
                                  str(root/'src/spire_learn/neural/config.py')):
                    refreshed = package.refresh_package(root, root/package.REUSABLE_PACKAGE, destination)
                    validated = package.validate_refreshed_package(root, destination/'package.json')
                self.assertEqual(refreshed, validated)
                for name in ('training/fixed/shard-0.json', 'training/mixed/support.json',
                             'panel/case-0.json'):
                    self.assertEqual((destination/name).read_bytes(),
                                     (root/Path(package.REUSABLE_PACKAGE).parent/name).read_bytes())


class ContinuationBudgetAndWatchdogTests(unittest.TestCase):
    def test_b_training_cutoff_preserves_evaluation_and_cleanup_reserve(self):
        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_B):
            root = staged_caps_root(Path(temporary))
            selected = profiles.current()
            release = controls.release_caps(root)
            self.assertEqual(selected.training_stop_second, 33000)
            self.assertEqual(release['training_stop_second'], 33000)
            self.assertEqual(33000 + release['evaluation_seconds'] + 600,
                             release['whole_seconds'])

            budget = controls._campaign_budget(root, start=1000., deadline=36400.,
                                               clock=lambda: 33500.)
            soft, whole = budget.job_deadlines('segment', arm_deadline=34000.)
            self.assertEqual(whole, 34000.)
            self.assertLess(soft, whole)

            slots = ['fresh-initial', 'matched-fixed', 'matched-randomized', 'matched-mixed',
                     'mixed-time-6000', 'mixed-time-12000', 'mixed-time-18000', 'final-mixed']
            schedule = controls.CampaignSchedule({'evaluation_slots': slots}, start=0.)
            schedule.accept_qualification()
            schedule.finish_panel('fresh-initial', status='accepted', reason='ok')
            before_cutoff = schedule.next(32999.)
            at_cutoff = schedule.next(33000.)
            self.assertEqual(before_cutoff['kind'], 'segment')
            self.assertEqual(before_cutoff['arm_deadline'], 33000.)
            self.assertEqual(at_cutoff, {'kind': 'panel', 'slot': 'matched-fixed', 'arm': 'fixed'})

        with profiles.scope(PROFILE_A):
            self.assertEqual(controls.release_caps(ROOT)['training_stop_second'], 33600)
            schedule = controls.CampaignSchedule({'evaluation_slots': slots}, start=0.)
            schedule.accept_qualification()
            schedule.finish_panel('fresh-initial', status='accepted', reason='ok')
            self.assertEqual(schedule.next(33599.)['arm_deadline'], 33600.)
            self.assertEqual(schedule.next(33600.),
                             {'kind': 'panel', 'slot': 'matched-fixed', 'arm': 'fixed'})

        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_B):
            root = staged_caps_root(Path(temporary))
            campaign_path = root/profiles.current().campaign_path
            campaign = json.loads(campaign_path.read_text())
            campaign['training_stop_second'] = 33600
            campaign_path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError, 'continuation campaign budget differs'):
                controls.release_caps(root)

    def test_b_remaining_caps_flow_into_release_and_campaign_budget(self):
        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_B):
            root = staged_caps_root(Path(temporary))
            caps = controls.release_caps(root)['global_effect_caps']
            self.assertEqual(caps, REMAINING_CAPS)
            self.assertEqual({key: caps[key] + FAILED_LEASE[key] for key in controls.CAPS}, controls.CAPS)
            budget = controls._campaign_budget(root, start=1000., deadline=36400., clock=lambda: 1000.)
            self.assertEqual(budget.caps, REMAINING_CAPS)
            self.assertEqual(budget.deadline-budget.start, 35400.)
        for legacy in ('historical', 'fixed-gym-a86df922', PROFILE_A):
            with self.subTest(profile=legacy), profiles.scope(legacy):
                caps = controls.release_caps(ROOT)
                self.assertEqual(caps['global_effect_caps'], controls.CAPS)
                self.assertEqual(caps['whole_seconds'], 36000)

    def test_b_watchdog_uses_shorter_whole_deadline_and_preserves_termination_margin(self):
        now = [5.]
        calls = []

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            now[0] = 6.
            return subprocess.CompletedProcess(argv, 17)

        with profiles.scope(PROFILE_B), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            result = controls.launch_guard(['run', '--root', str(ROOT)],
                result_path=Path(temporary)/'outer.json', clock=lambda: now[0], run=run,
                parent_start=5.)
            retained = json.loads((Path(temporary)/'outer.json').read_text())
        argv = calls[0][0]
        self.assertEqual(result, 17)
        self.assertEqual(argv[:5], ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', '35299s', sys.executable])
        self.assertEqual(calls[0][1]['env']['SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE'], '35405.0')
        self.assertEqual(retained['whole_deadline'], 35405.)
        self.assertEqual(retained['term_after_seconds'], 35299)
        self.assertEqual(retained['kill_grace_seconds'], 90)

        api = controls
        with profiles.scope(PROFILE_B), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            result_path, binding_path = base/'result.json', base/'binding.json'
            result_path.touch()
            info = result_path.stat()
            inputs = dict(package_path=ROOT/'package.json', source_snapshot=ROOT/'source.json',
                          release_path=ROOT/'release.json')
            child_argv = api._controller_argv(ROOT, **inputs)
            watchdog_argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', '35298s', *child_argv]
            ticks = os.sysconf('SC_CLK_TCK')
            launcher_pid, watchdog_pid, child_pid = 55501, 55502, os.getpid()
            common = dict(uid=1000, state='S', cwd=str(ROOT), executable=str(Path(sys.executable).resolve()))
            rows = {
                launcher_pid: dict(common, pid=launcher_pid, ppid=1, pgid=launcher_pid, session=launcher_pid,
                    start_ticks=1000*ticks, argv=[sys.executable, '-B', child_argv[2], '_launch']),
                watchdog_pid: dict(common, pid=watchdog_pid, ppid=launcher_pid, pgid=watchdog_pid, session=watchdog_pid,
                    start_ticks=1000*ticks+1, argv=watchdog_argv, executable=str(Path('/usr/bin/timeout').resolve())),
                child_pid: dict(common, pid=child_pid, ppid=watchdog_pid, pgid=watchdog_pid, session=watchdog_pid,
                    start_ticks=1000*ticks+2, argv=child_argv),
            }
            binding = dict(schema='spire-learn-curriculum-supervision-v1', launcher_pid=launcher_pid,
                launcher_start_ticks=1000*ticks, parent_start=1000., whole_deadline=36400.,
                watchdog_argv=watchdog_argv, result_path=str(result_path), result_fd=3,
                result_device=info.st_dev, result_inode=info.st_ino,
                controller_source_sha256=api.package_api.sha256(Path(api.__file__)))
            descriptor = dict(path=str(result_path), device=info.st_dev, inode=info.st_ino,
                uid=1000, regular=True, writable=True)

            def verify(value=binding, facts=rows):
                binding_path.write_text(json.dumps(value))
                with patch.dict(os.environ, {
                    'SPIRE_LEARN_CURRICULUM_SUPERVISION': str(binding_path),
                    'SPIRE_LEARN_CURRICULUM_SUPERVISION_SHA256': api.package_api.sha256(binding_path),
                    'SPIRE_LEARN_CURRICULUM_PARENT_START': '1000.0',
                    'SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE': '36400.0'}), \
                     patch.object(api, '_proc_identity', side_effect=lambda pid: copy.deepcopy(facts[pid])), \
                     patch.object(api, '_result_descriptor', return_value=descriptor):
                    return api._verify_supervision(ROOT, **inputs, clock=lambda: 1001.)

            self.assertEqual(verify()['whole_deadline'], 36400.)
            old_timeout = copy.deepcopy(binding)
            old_timeout['watchdog_argv'][3] = '35898s'
            rows[watchdog_pid]['argv'][3] = '35898s'
            with self.assertRaisesRegex(ValueError, 'process chain differs'):
                verify(old_timeout, rows)


class GuestNativePinTests(unittest.TestCase):
    def test_selected_native_pin_is_used_by_the_guest_runtime_verifier(self):
        cases = ('historical', 'fixed-gym-a86df922', PROFILE_A, PROFILE_B)
        for name in cases:
            with self.subTest(profile=name), profiles.scope(name):
                if profiles.current().fixed:
                    expected = profiles.overlay_pins(ROOT)['native_sha256']
                else:
                    expected = controls.low.NATIVE_SHA256
                pins = _runtime_pins(expected)
                controls._verify_guest_runtime_pins(pins, expected_native_sha256=expected)
                changed = dict(pins, native_sha256='0'*64)
                with self.assertRaisesRegex(ValueError, 'guest runtime pins differ'):
                    controls._verify_guest_runtime_pins(changed, expected_native_sha256=expected)
        with profiles.scope(PROFILE_A):
            current = profiles.overlay_pins(ROOT)['native_sha256']
            self.assertNotEqual(current, profiles.NATIVE_SHA256)


class PredecessorAdmissionTests(unittest.TestCase):
    def test_host_validator_accepts_only_the_hashed_zero_effect_preflight_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_predecessor_root(Path(temporary))
            with profiles.scope(PROFILE_B):
                controls.validate_predecessor(root)

        for mutation in ('receipt', 'cleanup'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = staged_predecessor_root(Path(temporary))
                attempt_path = root/PREDECESSOR_PATHS['attempt']
                audit_path = root/PREDECESSOR_PATHS['audit']
                predecessor_path = root/'configs/neural/experiments/curriculum-v5/predecessor.json'
                attempt, audit, predecessor = (json.loads(attempt_path.read_text()),
                    json.loads(audit_path.read_text()), json.loads(predecessor_path.read_text()))
                result_path = root/PREDECESSOR_PATHS['result']
                result = json.loads(result_path.read_text())
                if mutation == 'receipt':
                    attempt['receipt_binding'] = {'path': 'runtime/jobs/curriculum-qualification/output/job-receipt.json'}
                else:
                    rm = next(row for row in attempt['commands']
                              if len(row.get('argv', [])) > 5 and row['argv'][5] == 'rm')
                    rm['argv'][-1] = 'unrelated-container'
                attempt_bytes = (json.dumps(attempt, sort_keys=True, separators=(',', ':'))+'\n').encode()
                attempt_path.write_bytes(attempt_bytes)
                attempt_sha = hashlib.sha256(attempt_bytes).hexdigest()
                reference = result['jobs'][0]['references']['attempt']
                reference['sha256'], reference['size_bytes'] = attempt_sha, len(attempt_bytes)
                result_bytes = (json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n').encode()
                result_path.write_bytes(result_bytes)
                result_sha = hashlib.sha256(result_bytes).hexdigest()
                audit['hashes']['runtime/attempts/curriculum-qualification.json'] = attempt_sha
                audit['hashes']['runtime/campaign-result.json'] = result_sha
                audit_bytes = (json.dumps(audit, sort_keys=True, separators=(',', ':'))+'\n').encode()
                audit_path.write_bytes(audit_bytes)
                predecessor['artifacts']['attempt']['sha256'] = attempt_sha
                predecessor['artifacts']['result']['sha256'] = result_sha
                predecessor['artifacts']['audit']['sha256'] = hashlib.sha256(audit_bytes).hexdigest()
                predecessor_path.write_text(json.dumps(predecessor, sort_keys=True, indent=2)+'\n')
                error = 'receipt' if mutation == 'receipt' else 'cleanup'
                with profiles.scope(PROFILE_B), self.assertRaisesRegex(ValueError, error):
                    controls.validate_predecessor(root)

    def test_rebound_audit_still_requires_failed_preflight_zero_effects_and_os_exit_one(self):
        cases = ('status', 'root_cause', 'effect_count', 'launcher_os_exit')
        for mutation in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = staged_predecessor_root(Path(temporary))
                audit_path = root/PREDECESSOR_PATHS['audit']
                result_path = root/PREDECESSOR_PATHS['result']
                os_path = root/PREDECESSOR_PATHS['launcher_os_exit']
                predecessor_path = root/'configs/neural/experiments/curriculum-v5/predecessor.json'
                audit, result, os_exit = (json.loads(audit_path.read_text()),
                    json.loads(result_path.read_text()), json.loads(os_path.read_text()))
                if mutation == 'status':
                    audit['status'] = 'complete'
                elif mutation == 'root_cause':
                    audit['root_cause'] = 'different failure'
                elif mutation == 'effect_count':
                    audit['optimizer_steps'] = 1
                else:
                    os_exit['exit_code'] = 0
                    os_bytes = (json.dumps(os_exit, sort_keys=True, separators=(',', ':'))+'\n').encode()
                    os_path.write_bytes(os_bytes)
                    audit['hashes']['launcher-os-exit-observed.json'] = hashlib.sha256(os_bytes).hexdigest()
                audit_bytes = (json.dumps(audit, sort_keys=True, separators=(',', ':'))+'\n').encode()
                audit_path.write_bytes(audit_bytes)
                predecessor = json.loads(predecessor_path.read_text())
                predecessor['artifacts']['audit']['sha256'] = hashlib.sha256(audit_bytes).hexdigest()
                if mutation == 'launcher_os_exit':
                    predecessor['artifacts']['launcher_os_exit']['sha256'] = hashlib.sha256(os_bytes).hexdigest()
                predecessor_path.write_text(json.dumps(predecessor, sort_keys=True, indent=2)+'\n')
                expected_error = 'launcher exit' if mutation == 'launcher_os_exit' else 'predecessor audit'
                with profiles.scope(PROFILE_B), self.assertRaisesRegex(ValueError, expected_error):
                    controls.validate_predecessor(root)

    def test_release_validation_runs_the_host_predecessor_gate_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_predecessor_root(Path(temporary))
            result_path = root/'invalid-release.json'
            result_path.write_text('{}\n')
            snapshot_path = root/'source.json'
            snapshot_path.write_text(json.dumps({
                'schema': controls.SOURCE_SCHEMA,
                'state': 'source_only',
                'learner_head': 'a'*40,
                'dirty': {},
                'files': {'scripts/neural/curriculum_controls.py': {'sha256': '0'*64, 'size_bytes': 1}},
                'modes': {'scripts/neural/curriculum_controls.py': 0o644},
            }))
            altered = root/PREDECESSOR_PATHS['result']
            altered.write_bytes(altered.read_bytes()+b'\n')
            with profiles.scope(PROFILE_B), self.assertRaisesRegex(ValueError, 'raw artifact changed: result'):
                controls.validate_release(root, package_path=root/'unused-package.json',
                    source_snapshot=snapshot_path, release_path=result_path)


if __name__ == '__main__':
    unittest.main()
