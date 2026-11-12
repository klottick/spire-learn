"""Pure source, debit, and package checks for the second fresh continuation."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from scripts.neural import curriculum_runtime as producer

ROOT = Path(__file__).resolve().parents[2]
PROFILE_A = 'fresh-gym-afb51f5f'
PROFILE_B = 'fresh-gym-afb51f5f-b'
PROFILE_C = 'fresh-gym-afb51f5f-c'
CURRENT_NATIVE = '1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac'
V4 = Path('artifacts/development/neural-combat/curriculum-campaign-v4')
V5 = Path('artifacts/development/neural-combat/curriculum-campaign-v5')
B_DEBIT = {
    'environment_attempts': 16512,
    'sample_rows': 128,
    'optimizer_attempts': 8,
    'reset_slots': 74,
    'fixed_random_choices': 0,
}
C_CAPS = {
    'environment_attempts': 1182976,
    'sample_rows': 1099360,
    'optimizer_attempts': 69984,
    'reset_slots': 1099670,
    'fixed_random_choices': 0,
}
B_ARTIFACT_PATHS = {
    'audit': V5/'evaluation-failure-audit.json',
    'result': V5/'runtime/campaign-result.json',
    'qualification_attempt': V5/'runtime/attempts/curriculum-qualification.json',
    'qualification_receipt': V5/'runtime/jobs/curriculum-qualification/output/job-receipt.json',
    'panel_attempt': V5/'runtime/attempts/curriculum-panel-fresh-initial.json',
    'panel_receipt': V5/'runtime/jobs/curriculum-panel-fresh-initial/output/job-receipt.json',
    'launcher_result': V5/'launcher-result.json',
    'launcher_os_exit': V5/'launcher-os-exit-observed.json',
}


def staged_c_config_root(base: Path) -> Path:
    root = base/'root'
    for version in ('curriculum-v4', 'curriculum-v5', 'curriculum-v6'):
        destination = root/'configs/neural/experiments'/version
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT/'configs/neural/experiments'/version, destination)
    return root


def staged_c_package_root(base: Path) -> Path:
    root = staged_c_config_root(base)
    source = ROOT/package.REUSABLE_PACKAGE
    destination = root/package.REUSABLE_PACKAGE
    destination.parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.parent, destination.parent)
    return root


def staged_c_caps_root(base: Path) -> Path:
    root = base/'root'
    for version, names in (
        ('curriculum-v4', ('campaign.json',)),
        ('curriculum-v5', ('campaign.json', 'predecessor.json')),
        ('curriculum-v6', ('campaign.json', 'predecessor-b.json')),
    ):
        for name in names:
            source = ROOT/'configs/neural/experiments'/version/name
            destination = root/'configs/neural/experiments'/version/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return root


def staged_c_predecessor_root(base: Path) -> Path:
    root = base/'root'
    for name in controls.PREDECESSOR_ARTIFACT_PATHS.values():
        source, destination = ROOT/name, root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in B_ARTIFACT_PATHS.values():
        source, destination = ROOT/name, root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    configs = (
        ('curriculum-v5', 'predecessor.json'),
        ('curriculum-v6', 'predecessor-b.json'),
    )
    for version, name in configs:
        source = ROOT/'configs/neural/experiments'/version/name
        destination = root/'configs/neural/experiments'/version/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root


class FreshGymCProfileTests(unittest.TestCase):
    def test_c_profile_routes_ancestors_and_v6_inputs(self):
        with profiles.scope(PROFILE_C):
            selected = profiles.current()
            self.assertEqual(selected.campaign_id, 'curriculum-20261102-c')
            self.assertEqual(selected.domain,
                             Path('artifacts/development/neural-combat/curriculum-campaign-v6'))
            self.assertEqual(selected.qualification_input,
                             Path('artifacts/development/neural-combat/curriculum-campaign-v6/qualification'))
            self.assertEqual(selected.template, profiles.FRESH_GYM.template)
            self.assertEqual(selected.parents, profiles.FRESH_GYM.parents)
            self.assertEqual(selected.runtime_profile, profiles.FRESH_GYM.runtime_profile)
            self.assertEqual(selected.overlay_manifest, profiles.FRESH_GYM.overlay_manifest)
            self.assertEqual(selected.revision, profiles.FRESH_GYM.revision)
            self.assertEqual(selected.native_sha256, CURRENT_NATIVE)
            self.assertEqual(selected.predecessor_path, profiles.FRESH_GYM_B.predecessor_path)
            self.assertEqual(selected.additional_predecessor_path,
                             'configs/neural/experiments/curriculum-v6/predecessor-b.json')
            self.assertEqual(selected.whole_seconds, 34200)
            self.assertEqual(selected.training_stop_second, 31800)
            self.assertTrue(profiles.is_fresh_gym_family())
            self.assertEqual(profiles.qualification_source(ROOT),
                             ROOT/'configs/neural/experiments/curriculum-v4/qualification')
            self.assertEqual(profiles.qualification_hashes(ROOT), package.read_json(
                ROOT/'configs/neural/experiments/curriculum-v4/qualification-identities.json'))
            paths = set(profiles.source_paths(ROOT))
            self.assertTrue({
                'configs/neural/experiments/curriculum-v4/campaign.json',
                'configs/neural/experiments/curriculum-v5/campaign.json',
                'configs/neural/experiments/curriculum-v5/predecessor.json',
                'configs/neural/experiments/curriculum-v6/campaign.json',
                'configs/neural/experiments/curriculum-v6/predecessor-b.json',
                'tests/unit/test_neural_curriculum_continuation.py',
                'tests/unit/test_neural_curriculum_continuation_c.py',
            } <= paths)
            self.assertIn('scripts/neural/curriculum_fault_capture.py', producer.source_paths(ROOT))

        for name, whole, stop in ((PROFILE_A, 36000, 33600), (PROFILE_B, 35400, 33000)):
            with self.subTest(profile=name), profiles.scope(name):
                self.assertTrue(profiles.is_fresh_gym_family())
                self.assertEqual(profiles.current().whole_seconds, whole)
                self.assertEqual(profiles.current().training_stop_second, stop)

    def test_c_selected_native_pin_uses_shared_guest_runtime_verifier(self):
        with profiles.scope(PROFILE_C):
            expected = profiles.overlay_pins(ROOT)['native_sha256']
            self.assertEqual(expected, CURRENT_NATIVE)
            pins = {
                'profile_sha256': controls.cdi.PROFILE_SHA256,
                'native_sha256': expected,
                'lock_sha256': controls.low.LOCK_SHA256,
                'versions': controls.PIN_VERSIONS,
                'python_sha256': controls.PYTHON_SHA256,
            }
            controls._verify_guest_runtime_pins(pins, expected_native_sha256=expected)


class FreshGymCPackageTests(unittest.TestCase):
    def test_c_refresh_projects_budget_and_validates_reused_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_c_package_root(Path(temporary))
            source = package.read_json(root/package.REUSABLE_PACKAGE)
            with profiles.scope(PROFILE_C):
                changed = package._refreshed_files(root, source)
                campaign = package.read_json(root/profiles.current().campaign_path)
                train = tomllib.loads(changed['training/fixed/config-0.toml'].decode())
                panel = tomllib.loads(changed['panel/config-0.toml'].decode())
                accepted_panel = tomllib.loads((root/Path(package.REUSABLE_PACKAGE).parent/
                                                'panel/config-0.toml').read_text())
                self.assertEqual(campaign['training_budgets']['max_wall_seconds'], 34200.0)
                self.assertEqual(train['budgets'], campaign['training_budgets'])
                self.assertEqual(panel['budgets'], accepted_panel['budgets'])
                self.assertEqual(panel['device'], 'cpu')
                self.assertNotIn('training/fixed/shard-0.json', changed)
                self.assertNotIn('training/fixed/support.json', changed)

                from spire_learn.neural import config as neural_config
                destination = (root/'artifacts/development/neural-combat/'
                               'curriculum-campaign-v6/preparation/package-c')
                destination.parent.mkdir(parents=True)
                with patch.object(neural_config, '__file__',
                                  str(root/'src/spire_learn/neural/config.py')):
                    refreshed = package.refresh_package(root, root/package.REUSABLE_PACKAGE, destination)
                    validated = package.validate_refreshed_package(root, destination/'package.json')
                self.assertEqual(refreshed, validated)
                for name in ('training/fixed/shard-0.json', 'training/mixed/support.json',
                             'panel/case-0.json'):
                    source_path = root/Path(package.REUSABLE_PACKAGE).parent/name
                    self.assertEqual((destination/name).read_bytes(), source_path.read_bytes())


class FreshGymCDebitTests(unittest.TestCase):
    def test_c_caps_subtract_both_predecessors_and_preserve_b_accounting(self):
        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_C):
            root = staged_c_caps_root(Path(temporary))
            release = controls.release_caps(root)
            self.assertEqual(release['global_effect_caps'], C_CAPS)
            self.assertEqual(release['whole_seconds'], 34200)
            self.assertEqual(release['training_stop_second'], 31800)
            self.assertEqual(release['evaluation_seconds'], 1800)
            self.assertEqual(31800 + 1800 + 600, 34200)
            self.assertEqual(34200 + (36000-35400) + 600 + 600, 36000)
            self.assertEqual({
                key: release['global_effect_caps'][key] +
                controls.FAILED_QUALIFICATION_LEASE[key] + B_DEBIT[key]
                for key in controls.CAPS
            }, controls.CAPS)
            budget = controls._campaign_budget(root, start=100., deadline=34300.,
                                               clock=lambda: 100.)
            self.assertEqual(budget.caps, C_CAPS)
            self.assertEqual(budget.deadline-budget.start, 34200.)
            segment_budget = controls._campaign_budget(root, start=100., deadline=34300.,
                                                       clock=lambda: 31700.)
            _, segment_whole = segment_budget.job_deadlines('segment', arm_deadline=31900.)
            self.assertEqual(segment_whole, 31900.)

            slots = ['fresh-initial', 'matched-fixed', 'matched-randomized', 'matched-mixed',
                     'mixed-time-6000', 'mixed-time-12000', 'mixed-time-18000', 'final-mixed']
            schedule = controls.CampaignSchedule({'evaluation_slots': slots}, start=0.)
            schedule.accept_qualification()
            schedule.finish_panel('fresh-initial', status='accepted', reason='ok')
            self.assertEqual(schedule.next(31799.)['arm_deadline'], 31800.)
            self.assertEqual(schedule.next(31800.),
                             {'kind': 'panel', 'slot': 'matched-fixed', 'arm': 'fixed'})

        with profiles.scope(PROFILE_B):
            release = controls.release_caps(ROOT)
            self.assertEqual(release['whole_seconds'], 35400)
            self.assertEqual(release['training_stop_second'], 33000)
            self.assertEqual(release['global_effect_caps'], {
                'environment_attempts': 1199488,
                'sample_rows': 1099488,
                'optimizer_attempts': 69992,
                'reset_slots': 1099744,
                'fixed_random_choices': 0,
            })
        with profiles.scope(PROFILE_A):
            release = controls.release_caps(ROOT)
            self.assertEqual(release['whole_seconds'], 36000)
            self.assertEqual(release['training_stop_second'], 33600)
            self.assertEqual(release['global_effect_caps'], controls.CAPS)

    def test_c_release_rejects_campaign_or_manifest_debit_drift(self):
        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_C):
            root = staged_c_caps_root(Path(temporary))
            campaign_path = root/profiles.current().campaign_path
            campaign = json.loads(campaign_path.read_text())
            campaign['training_stop_second'] = 31801
            campaign_path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError, 'continuation campaign budget differs'):
                controls.release_caps(root)

        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_C):
            root = staged_c_caps_root(Path(temporary))
            manifest_path = root/profiles.current().additional_predecessor_path
            manifest = json.loads(manifest_path.read_text())
            manifest['charged_or_reserved']['sample_rows'] -= 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'additional campaign predecessor differs'):
                controls.release_caps(root)

        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_C):
            root = staged_c_caps_root(Path(temporary))
            manifest_path = root/profiles.current().additional_predecessor_path
            manifest = json.loads(manifest_path.read_text())
            manifest['reserved_time_seconds']['panel'] -= 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'additional campaign predecessor differs'):
                controls.release_caps(root)

        with tempfile.TemporaryDirectory() as temporary, profiles.scope(PROFILE_C):
            root = staged_c_caps_root(Path(temporary))
            prior_campaign_path = profiles.FRESH_GYM_B.campaign_path
            prior_campaign = json.loads((root/prior_campaign_path).read_text())
            prior_campaign['qualification']['whole_seconds'] = 599
            (root/prior_campaign_path).write_text(json.dumps(prior_campaign))
            with self.assertRaisesRegex(ValueError, 'continuation campaign time debit differs'):
                controls.release_caps(root)


class FreshGymCPredecessorTests(unittest.TestCase):
    def test_host_accepts_hashed_b_failure_and_rejects_rebound_contract_or_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_c_predecessor_root(Path(temporary))
            with profiles.scope(PROFILE_C):
                controls.validate_predecessor(root)

        for mutation in ('panel_contract', 'cleanup'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = staged_c_predecessor_root(Path(temporary))
                manifest_path = root/'configs/neural/experiments/curriculum-v6/predecessor-b.json'
                manifest = json.loads(manifest_path.read_text())
                audit_path = root/B_ARTIFACT_PATHS['audit']
                audit = json.loads(audit_path.read_text())
                if mutation == 'panel_contract':
                    receipt_path = root/B_ARTIFACT_PATHS['panel_receipt']
                    receipt = json.loads(receipt_path.read_text())
                    report = receipt['reports'][0]['report']
                    report['optimizer']['policy_version'] = 1
                    raw = (json.dumps(receipt, sort_keys=True, separators=(',', ':'))+'\n').encode()
                    receipt_path.write_bytes(raw)
                    receipt_hash = hashlib.sha256(raw).hexdigest()
                    manifest['artifacts']['panel_receipt']['sha256'] = receipt_hash
                    audit['hashes'][str(B_ARTIFACT_PATHS['panel_receipt']).removeprefix(str(V5)+'/')] = receipt_hash
                else:
                    audit['cleanup'][0]['absent'] = False
                audit_raw = (json.dumps(audit, sort_keys=True, separators=(',', ':'))+'\n').encode()
                audit_path.write_bytes(audit_raw)
                manifest['artifacts']['audit']['sha256'] = hashlib.sha256(audit_raw).hexdigest()
                manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
                expected = 'panel failure evidence' if mutation == 'panel_contract' else 'container cleanup'
                with profiles.scope(PROFILE_C), self.assertRaisesRegex(ValueError, expected):
                    controls.validate_predecessor(root)

    def test_raw_b_artifact_change_fails_before_semantic_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = staged_c_predecessor_root(Path(temporary))
            path = root/B_ARTIFACT_PATHS['panel_attempt']
            path.write_bytes(path.read_bytes()+b' ')
            with profiles.scope(PROFILE_C), self.assertRaisesRegex(
                    ValueError, 'raw artifact changed: panel_attempt'):
                controls.validate_predecessor(root)


if __name__ == '__main__':
    unittest.main()
