"""Projected monitor-bootstrap and controller-admission contracts for continuation 2."""
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from tests.unit.test_neural_curriculum_package import ROOT


PROFILE = 'ironclad-twenty-hour-continuation-20261103-corrected'
AUTHORITY = 'configs/neural/experiments/twenty-hour-20261103-continuation-2/launch-authority.json'


class CorrectedContinuationBootstrapTests(unittest.TestCase):
    def test_profile_uses_a_distinct_campaign_and_fresh_output_domain(self):
        self.assertIn(PROFILE, profiles.NAMES)
        with profiles.scope(PROFILE):
            selected = profiles.current()
            self.assertEqual(selected.campaign_id,
                'curriculum-20261103-twenty-hour-continuation-2')
            self.assertEqual(selected.domain,
                Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-2'))
            self.assertNotEqual(selected.domain, profiles.TWENTY_HOUR_CONTINUATION.domain)
            self.assertEqual((selected.whole_seconds, selected.training_stop_second),
                (65998, 64198))
            paths = profiles.source_paths(ROOT)
            self.assertIn('scripts/neural/campaign_monitor.py', paths)
            self.assertIn('configs/neural/experiments/twenty-hour-20261103-continuation-2/campaign.json', paths)
            self.assertIn('tests/unit/test_neural_curriculum_continuation_2.py', paths)

    def test_actual_monitor_bootstrap_precedes_controller_preflight_and_claim(self):
        self.assertIn(PROFILE, profiles.NAMES)
        self.assertIsNotNone(find_spec('scripts.neural.campaign_monitor'))
        from scripts.neural import campaign_monitor

        profile = next(row for row in profiles._PROFILES if row.name == PROFILE)
        original_domain = profile.domain
        task_root = ROOT/original_domain/'task-2'
        task_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='bootstrap-preflight-', dir=task_root) as temp:
            domain = Path(temp)
            runtime = domain/'runtime'
            control = domain/'control'
            package_dir = domain/'inputs'/'package-a'
            control.mkdir()
            package_dir.parent.mkdir()
            with profiles.scope(PROFILE):
                # Keep the canonical profile object so all closed-family predicates
                # exercise the actual continuation-2 branches during this projection.
                object.__setattr__(profile, 'domain', domain.relative_to(ROOT))
                try:
                    self._exercise_actual_bootstrap_and_preflight(
                        campaign_monitor, profile, domain, runtime, control, package_dir)
                finally:
                    object.__setattr__(profile, 'domain', original_domain)

    def _exercise_actual_bootstrap_and_preflight(
        self, campaign_monitor, profile, domain, runtime, control, package_dir
    ):
        with profiles.scope(PROFILE):
            package_doc = package.author_package(ROOT, package_dir,
                campaign_id=profile.campaign_id)
            package_path = package_dir/'package.json'
            source_path = control/'source-snapshot.json'
            controls.capture_source(ROOT, source_path, learner_head='a'*40,
                dirty={'test_projection': True})
            release = controls.release_template(ROOT, package_path=package_path,
                source_snapshot=source_path, user_approval=ROOT/AUTHORITY)
            release.update(state='approved', review={'approved': True,
                'reports': ['projected-source-review', 'projected-independent-review']})
            release_path = control/'campaign-release.json'
            controls._write(release_path, release)

            self.assertFalse(runtime.exists())
            prepared = campaign_monitor.bootstrap_monitor(
                root=ROOT, campaign_domain=domain, profile_name=PROFILE,
                campaign_id=profile.campaign_id, owner_exec_session_id=424242,
                launcher_pid=os.getpid(), launcher_start_ticks=1,
                monotonic_start=100.0, whole_seconds=profile.whole_seconds,
                training_stop_second=profile.training_stop_second,
                launch_utc='2026-11-03T00:00:00Z',
                control_paths=(package_path, source_path, release_path))
            self.assertFalse(runtime.exists())
            self.assertTrue(prepared['launch_session_path'].is_file())
            self.assertTrue(prepared['heartbeat_state_path'].is_file())
            self.assertFalse(prepared['launch_session_path'].is_relative_to(runtime))
            self.assertFalse(prepared['heartbeat_state_path'].is_relative_to(runtime))
            self.assertEqual(json.loads(prepared['launch_session_path'].read_text())['campaign_id'],
                package_doc['campaign_id'])

            events = []
            real_storage = controls.storage_admission
            real_write = controls._write
            class StopBeforeFirstJob(Exception):
                pass

            def storage(root, host, deps, *, reservation=0, live=False):
                if reservation:
                    self.assertFalse(runtime.exists())
                    events.append('static-admission')
                return real_storage(root, host, deps, reservation=reservation, live=live)

            def write(path, value):
                if Path(path) == runtime/'campaign-claim.json':
                    self.assertEqual(events[-1], 'static-admission')
                    events.append('campaign-claim')
                return real_write(path, value)

            def stop_after_claim(root, host):
                self.assertTrue((runtime/'campaign-claim.json').is_file())
                events.append('post-claim-boundary')
                raise StopBeforeFirstJob('projected stop after admission')

            deps = controls.HostDependencies(clock=lambda: 100.0,
                disk_usage=lambda _: SimpleNamespace(free=64*controls.GIB),
                uid=lambda: 1000, gid=lambda: 1000)
            package_sha = package.sha256(package_path)
            source_sha = package.sha256(source_path)
            release_sha = package.sha256(release_path)
            supervision = {'parent_start': 100.0,
                'whole_deadline': 100.0+profile.whole_seconds,
                'binding_sha256': 'b'*64, 'launcher_pid': os.getpid(),
                'watchdog_pid': os.getpid(), 'result_path': str(domain/'launcher-result.json')}

            def projected_run(argv, *, env, check, timeout):
                self.assertEqual(argv[-2:], ['--profile', PROFILE])
                with patch.dict(os.environ, env), \
                        patch.object(controls, '_verify_supervision', return_value=supervision), \
                        patch.object(controls, 'storage_admission', side_effect=storage), \
                        patch.object(controls, '_write', side_effect=write), \
                        patch.object(controls.transport, '_sync_claim_ancestors', side_effect=stop_after_claim):
                    try:
                        controls.run_campaign(ROOT, package_path=package_path,
                            source_snapshot=source_path, release_path=release_path,
                            dependencies=deps, clock=lambda: 100.0)
                    except StopBeforeFirstJob:
                        pass
                return subprocess.CompletedProcess(argv, 0, '', '')

            argv = ['run', '--root', str(ROOT), '--package', str(package_path),
                '--source', str(source_path), '--release', str(release_path),
                '--profile', PROFILE]
            result_path = domain/'launcher-result.json'
            status = controls.launch_guard(argv, result_path=result_path,
                parent_start=100.0, clock=lambda: 100.0, run=projected_run)

            self.assertEqual(status, 0)
            self.assertEqual(events, ['static-admission', 'campaign-claim', 'post-claim-boundary'])
            claims = list(runtime.rglob('campaign-claim.json'))
            self.assertEqual(claims, [runtime/'campaign-claim.json'])
            self.assertFalse((runtime/'job-claims').exists())
            claim = json.loads(claims[0].read_text())
            self.assertEqual(claim['campaign_id'], package_doc['campaign_id'])
            self.assertEqual(claim['package_sha256'], package_sha)
            self.assertEqual(claim['source_sha256'], source_sha)
            self.assertEqual(claim['release_sha256'], release_sha)
            self.assertEqual(release['campaign_id'], package_doc['campaign_id'])
            self.assertEqual(release['package_sha256'], package_sha)
            self.assertEqual(release['source_snapshot_sha256'], source_sha)
            self.assertTrue((runtime/'campaign-result.json').is_file())


if __name__ == '__main__':
    unittest.main()
