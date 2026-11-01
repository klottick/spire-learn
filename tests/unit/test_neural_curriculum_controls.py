"""Pure campaign host/guest transport tests; optional runtimes stay blocked."""
from dataclasses import replace
import importlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import ExitStack, redirect_stderr
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit.test_neural_curriculum_package import ROOT, package_fixture
from tests.unit.test_neural_curriculum_runtime import ProducerHarness
from tests.unit.test_neural_study_cdi import cdi_facts

from tests.unit.test_neural_curriculum_integration import blocked_optional


class ControlContractTests(unittest.TestCase):
    def controls(self):
        return importlib.import_module('scripts.neural.curriculum_controls')

    def test_shared_pid_limit_preserves_default_and_rejects_nonprimitive(self):
        from scripts.neural import task9_study_controls as low
        spec = low.PhaseSpec('curriculum-qualification', 'curriculum-qualification',
            'curriculum-20261101-a', '1'*64, '2'*64, '3'*64, 120., 600.,
            '/learner/output', ('-B', 'entry'), (), (), (), runtime_class='gpu')
        self.assertEqual(getattr(spec, 'pids_limit', None), 512)
        for value in (True, False, 0, 127, 129, 512., '128', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(spec, pids_limit=value)
        qualified = replace(spec, pids_limit=128)
        self.assertEqual(low._inspect_projection(qualified, 'a'*64)['HostConfig']['PidsLimit'], 128)
        with tempfile.TemporaryDirectory() as directory:
            argv = low.build_create_argv(qualified, '/fake/docker', Path(directory))
        self.assertEqual(argv[argv.index('--pids-limit')+1], '128')
        wrong = low._inspect_projection(qualified, 'a'*64)
        wrong['HostConfig']['PidsLimit'] = 512
        with self.assertRaises(ValueError):
            low.assert_phase_inspect(json.dumps([wrong]), spec=qualified, container_id='a'*64)

    def test_budget_reserves_before_dispatch_and_never_refunds_uncertainty(self):
        api = self.controls()
        now = [10.]
        budget = api.CampaignBudget(api.CAPS, start=10., deadline=36010., clock=lambda: now[0])
        lease = dict(environment_attempts=512, sample_rows=512, optimizer_attempts=8,
                     reset_slots=256, fixed_random_choices=0)
        budget.reserve('curriculum-qualification', lease)
        self.assertEqual(budget.reserved, lease)
        with self.assertRaises(ValueError):
            budget.reserve('curriculum-qualification', lease)
        with self.assertRaises(ValueError):
            budget.settle('curriculum-qualification', {key: dict(reserved=value, attempted=value,
                delivered=value, uncertain=1 if value else 0) for key, value in lease.items()})
        self.assertEqual(budget.reserved, lease)
        now[0] = 33610.
        with self.assertRaises(TimeoutError):
            budget.job_deadlines('segment')
        now[0] = 35960.
        with self.assertRaises(TimeoutError):
            budget.job_deadlines('panel')

    def test_help_and_import_do_not_resolve_optional_packages(self):
        with blocked_optional() as attempts:
            api = self.controls()
            with self.assertRaises(SystemExit) as stopped:
                api.main(['--help'])
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(attempts, [])

    def test_source_capture_is_closed_and_staging_copies_bytes(self):
        api = self.controls()
        with tempfile.TemporaryDirectory() as directory, blocked_optional():
            destination = Path(directory) / 'snapshot.json'
            snapshot = api.capture_source(ROOT, destination, learner_head='a'*40,
                                          dirty={'dirty': True, 'status': 'preserved user changes'})
            self.assertEqual(snapshot['state'], 'source_only')
            self.assertEqual(snapshot['files']['tests/__init__.py']['size_bytes'], 0)
            stage = Path(directory) / 'stage'
            api.stage_source(ROOT, stage, snapshot)
            name = 'scripts/neural/curriculum_controls.py'
            self.assertNotEqual((ROOT / name).stat().st_ino, (stage / name).stat().st_ino)
            api.verify_source(stage, snapshot)
            (stage / 'src/extra.py').write_text('unreviewed')
            with self.assertRaisesRegex(ValueError, 'closed'):
                api.verify_source(stage, snapshot)
            (stage / 'src/extra.py').unlink()
            (stage / name).chmod(0o777)
            with self.assertRaisesRegex(ValueError, 'mode'):
                api.verify_source(stage, snapshot)
            with self.assertRaises(FileExistsError):
                api.capture_source(ROOT, destination, learner_head='a'*40, dirty={})

    def test_review_release_has_own_schema_pins_and_requires_user_binding(self):
        api = self.controls()
        fixture, package = package_fixture()
        with tempfile.TemporaryDirectory() as directory, blocked_optional():
            base = Path(directory)
            snapshot_path = base / 'source.json'
            api.capture_source(ROOT, snapshot_path, learner_head='a'*40, dirty={'dirty': True})
            approval = ROOT / api.DOMAIN / 'preparation/user-approval.json'
            document = api.release_template(ROOT, package_path=fixture/'package.json',
                source_snapshot=snapshot_path, user_approval=approval)
            document.update(state='approved', review={'approved': True, 'reports': ['source-review-a', 'source-review-b']})
            release = base / 'release.json'
            release.write_text(json.dumps(document))
            # Task 1 already tests the expensive full parser/cycle validator.
            # Here verify the host forwards to that exact boundary once.
            with patch.object(api.package_api, 'validate_package', return_value=package) as validator:
                accepted = api.validate_release(ROOT, package_path=fixture/'package.json',
                    source_snapshot=snapshot_path, release_path=release)
                validator.assert_called_once_with(ROOT, fixture/'package.json')
            self.assertEqual(accepted['package'], package)
            for key, value in [('schema', 'spire-learn-sustained-stage-release-v1'),
                               ('single_use', False), ('caps', {**document['caps'], 'whole_seconds': 36001})]:
                changed = {**document, key: value}
                release.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    api.validate_release(ROOT, package_path=fixture/'package.json',
                        source_snapshot=snapshot_path, release_path=release)
            release.write_text(json.dumps({**document, 'user_approval': {**document['user_approval'], 'sha256': '0'*64}}))
            with self.assertRaises(ValueError):
                api.validate_release(ROOT, package_path=fixture/'package.json',
                    source_snapshot=snapshot_path, release_path=release)
            malformed = copy.deepcopy(document)
            malformed['caps']['global_effect_caps']['fixed_random_choices'] = False
            release.write_text(json.dumps(malformed))
            with patch.object(api.package_api, 'validate_package', return_value=package):
                with self.assertRaises(ValueError):
                    api.validate_release(ROOT, package_path=fixture/'package.json',
                        source_snapshot=snapshot_path, release_path=release)

    def test_snapshot_schema_cannot_be_approved_by_rebinding_its_raw_hash(self):
        api = self.controls()
        fixture, package = package_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source, release = Path(directory)/'source.json', Path(directory)/'release.json'
            value = api.capture_source(ROOT, source, learner_head='a'*40, dirty={})
            value['state'] = 'approved-by-test'
            source.write_text(json.dumps(value))
            document = api.release_template(ROOT, package_path=fixture/'package.json',
                source_snapshot=source, user_approval=ROOT/api.DOMAIN/'preparation/user-approval.json')
            document.update(state='approved', review={'approved': True, 'reports': ['one', 'two']})
            release.write_text(json.dumps(document))
            with patch.object(api.package_api, 'validate_package', return_value=package):
                with self.assertRaises(ValueError):
                    api.validate_release(ROOT, package_path=fixture/'package.json', source_snapshot=source, release_path=release)

    def test_qualified_python_symlink_is_checked_as_container_link_without_execution(self):
        api = self.controls()
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            (venv/'bin').mkdir()
            (venv/'bin/python3.12').symlink_to('/usr/local/bin/python3.12')
            api.verify_python_link(venv)
            (venv/'bin/python3.12').unlink()
            (venv/'bin/python3.12').symlink_to('/unreviewed/python')
            with self.assertRaises(ValueError):
                api.verify_python_link(venv)

    def test_launcher_watchdog_uses_parent_start_and_retains_actual_child_exit(self):
        api = self.controls()
        now = [5.]
        calls = []
        def execute(argv, **kwargs):
            calls.append((argv, kwargs))
            now[0] = 6.
            return subprocess.CompletedProcess(argv, 17)
        with tempfile.TemporaryDirectory() as directory:
            result = api.launch_guard(['run', '--root', str(ROOT)],
                result_path=Path(directory)/'outer.json', clock=lambda: now[0], run=execute)
            retained = json.loads((Path(directory)/'outer.json').read_text())
        self.assertEqual(result, 17)
        self.assertEqual(retained['watchdog_exit_status'], 17)
        self.assertEqual(retained['controller_os_exit_status']['availability'], 'unavailable')
        self.assertEqual(calls[0][0][:5], ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', '35899s', sys.executable])
        self.assertEqual(calls[0][1]['env']['SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE'], '36005.0')
        self.assertEqual(retained['elapsed_seconds'], 1.)

    def test_launcher_retains_failed_spawn_without_inventing_a_child_exit(self):
        api = self.controls()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'outer.json'
            with self.assertRaises(OSError):
                api.launch_guard(['run'], result_path=path, clock=lambda: 5.,
                    run=lambda *a, **k: (_ for _ in ()).throw(OSError('spawn failed')))
            retained = json.loads(path.read_text())
        self.assertIsNone(retained['watchdog_exit_status'])
        self.assertEqual(retained['error']['type'], 'OSError')

    def test_launcher_sidecar_failure_retains_failed_result_without_spawning(self):
        api = self.controls()
        calls = []
        original = api._write
        def write(path, value):
            if str(path).endswith('.supervision.json'):
                raise OSError('owned supervision sidecar failed')
            return original(path, value)
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory)/'outer.json'
            with patch.object(api, '_write', side_effect=write):
                with self.assertRaises(OSError):
                    api.launch_guard(['run'], result_path=result_path, clock=lambda: 5.,
                        run=lambda *args, **kwargs: calls.append(args))
            record = json.loads(result_path.read_text())
            self.assertIsNone(record['watchdog_exit_status'])
            self.assertEqual(record['error']['type'], 'OSError')
            self.assertFalse(record['whole_run_complete'])
            self.assertTrue(result_path.with_name(result_path.name+'.failure.json').exists())
        self.assertEqual(calls, [])

    def test_launcher_retains_first_wait_control_through_result_write_and_fsync(self):
        api = self.controls()
        for stage in ('write', 'fsync', 'veto_failure'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'outer.json'
                sent, entered, injected = [], [False], [False]
                process = SimpleNamespace(pid=43210, returncode=None)
                def wait(**kwargs):
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    process.returncode = 130
                    return 130
                process.wait = wait
                original_dump, original_fsync, original_write = json.dump, os.fsync, api._write
                def late_control():
                    handler = signal.getsignal(signal.SIGTERM)
                    self.assertTrue(callable(handler), 'launcher publication lost its signal scope')
                    handler(signal.SIGTERM, None)
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    injected[0] = True
                def dump(value, stream, **kwargs):
                    entered[0] = True
                    if stage != 'fsync' and not injected[0]:
                        late_control()
                    return original_dump(value, stream, **kwargs)
                def fsync(descriptor):
                    if entered[0] and stage == 'fsync' and not injected[0]:
                        late_control()
                    return original_fsync(descriptor)
                def write(file, value):
                    if str(file).endswith('.failure.json') and stage == 'veto_failure':
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                        raise OSError('later launcher veto failed')
                    return original_write(file, value)
                previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
                with patch.object(api.subprocess, 'Popen', return_value=process), \
                     patch.object(os, 'killpg', side_effect=lambda pid, number: sent.append((pid, number))), \
                     patch.object(json, 'dump', side_effect=dump), patch.object(os, 'fsync', side_effect=fsync), \
                     patch.object(api, '_write', side_effect=write):
                    with self.assertRaises(api.low._Termination) as stopped:
                        api.launch_guard(['run'], result_path=path, clock=lambda: 5.)
                self.assertEqual(str(stopped.exception), 'signal '+str(signal.SIGINT))
                self.assertEqual(sent, [(43210, signal.SIGINT)])
                self.assertEqual({number: signal.getsignal(number) for number in previous}, previous)
                record = json.loads(path.read_text())
                self.assertEqual(record['watchdog_exit_status'], 130)
                self.assertEqual(record['parent_control'], signal.SIGINT)
                self.assertFalse(record['whole_run_complete'])

    def test_first_parent_control_only_during_postwait_publication_never_signals_reaped_group(self):
        api = self.controls()
        sent = []
        process = SimpleNamespace(pid=43211, returncode=None)
        def wait(**kwargs):
            process.returncode = 0
            return 0
        process.wait = wait
        original_dump = json.dump
        injected = [False]
        def dump(value, stream, **kwargs):
            if not injected[0]:
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler), 'postwait publication handler missing')
                handler(signal.SIGTERM, None)
                injected[0] = True
            return original_dump(value, stream, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'outer.json'
            with patch.object(api.subprocess, 'Popen', return_value=process), \
                 patch.object(os, 'killpg', side_effect=lambda pid, number: sent.append((pid, number))), \
                 patch.object(api, '_controller_logical_result', return_value={'status': 'complete', 'finalization_veto': False}), \
                 patch.object(json, 'dump', side_effect=dump):
                with self.assertRaises(api.low._Termination):
                    api.launch_guard(['run'], result_path=path, clock=lambda: 5.)
            self.assertFalse(json.loads(path.read_text())['whole_run_complete'])
            self.assertTrue(path.with_name(path.name+'.failure.json').exists())
        self.assertEqual(sent, [])

    def test_parent_result_fsync_failure_keeps_actual_wait_and_failure_veto(self):
        api = self.controls()
        original_dump, original_fsync = json.dump, os.fsync
        publication, failed = [False], [False]
        def dump(value, stream, **kwargs):
            publication[0] = True
            return original_dump(value, stream, **kwargs)
        def fsync(descriptor):
            if publication[0] and not failed[0]:
                failed[0] = True
                raise OSError('parent result fsync failed')
            return original_fsync(descriptor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'outer.json'
            with patch.object(json, 'dump', side_effect=dump), patch.object(os, 'fsync', side_effect=fsync), \
                 patch.object(api, '_controller_logical_result', return_value={'status': 'complete', 'finalization_veto': False}):
                with self.assertRaises(OSError):
                    api.launch_guard(['run'], result_path=path, clock=lambda: 5.,
                        run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0))
            result = json.loads(path.read_text())
            veto = json.loads(path.with_name(path.name+'.failure.json').read_text())
            self.assertEqual(result['watchdog_exit_status'], 0)
            self.assertFalse(result['whole_run_complete'])
            self.assertEqual(result['error']['type'], 'OSError')
            self.assertTrue(veto['publication_candidate']['whole_run_complete'])
            self.assertFalse(veto['whole_run_complete'])

    def test_failed_dispatch_references_are_explicitly_unavailable_without_valid_bytes(self):
        api = self.controls()
        with tempfile.TemporaryDirectory() as directory:
            host = Path(directory)
            missing = api._failed_dispatch_references(host, 'curriculum-missing')
            self.assertTrue(all(row['availability'] == 'unavailable' for row in missing['references'].values()))
            self.assertEqual(missing['partial_report_availability']['availability'], 'unavailable')
            context = host/'jobs/curriculum-missing/control/job.json'
            context.parent.mkdir(parents=True)
            context.write_text('{invalid JSON')
            unreadable = api._failed_dispatch_references(host, 'curriculum-missing')['references']['context']
            self.assertEqual(unreadable['availability'], 'unavailable')
            self.assertEqual(unreadable['raw']['sha256'], api.package_api.sha256(context))

    def test_schedule_is_fixed_and_availability_preserves_missing_prefixes(self):
        api = self.controls()
        package = package_fixture()[1]
        plan = api.CampaignSchedule(package, start=10.)
        self.assertEqual(plan.next(11.)['kind'], 'qualification')
        plan.accept_qualification()
        self.assertEqual(plan.next(12.)['slot'], 'original-b')
        plan.finish_panel('original-b', status='not_started', reason='time')
        request = plan.next(13.)
        self.assertEqual((request['arm'], request['index']), ('fixed', 0))
        plan.stop_arm('fixed', reason='zero_progress')
        self.assertEqual(plan.next(14.)['slot'], 'matched-fixed')
        plan.finish_panel('matched-fixed', status='unavailable', reason='no checkpoint')
        self.assertEqual(plan.next(15.)['arm'], 'randomized')
        plan.stop_arm('randomized', reason='time')
        plan.finish_panel('matched-randomized', status='unavailable', reason='no checkpoint')
        self.assertEqual(plan.next(16.)['arm'], 'mixed')
        self.assertEqual(plan.availability['matched-fixed']['status'], 'unavailable')
        self.assertEqual(plan.prefixes['fixed'], {})

    def test_mixed_milestones_include_matched_charges_and_exclude_cpu_or_other_arms(self):
        api = self.controls()
        plan = api.CampaignSchedule(package_fixture()[1], start=0.)
        plan.qualified, plan.arm_index = True, 3
        plan.panels_done.update(('original-b', 'matched-fixed', 'matched-randomized', 'matched-mixed'))
        # Matched mixed already consumed 2,900 seconds before long mixed starts.
        plan.training_seconds = {'fixed': 2900., 'randomized': 2900., 'mixed': 2900.}
        self.assertEqual(plan.next(10000.)['kind'], 'segment')
        plan.training_seconds['mixed'] += 3000.
        self.assertEqual(plan.next(13000.)['kind'], 'segment')
        plan.training_seconds['mixed'] += 100.
        self.assertEqual(plan.next(13100.)['slot'], 'mixed-time-6000')
        plan.finish_panel('mixed-time-6000', status='not_started', reason='pure scheduling fixture')
        # CPU wall and unrelated arm charges cannot move the 12K threshold.
        plan.training_seconds['fixed'] += 500.
        self.assertEqual(plan.next(13700.)['kind'], 'segment')
        plan.training_seconds['mixed'] = 11999.
        self.assertEqual(plan.next(20000.)['kind'], 'segment')
        plan.training_seconds['mixed'] = 12000.
        self.assertEqual(plan.next(20001.)['slot'], 'mixed-time-12000')

    def test_outer_supervisor_forwards_only_to_its_owned_group_and_records_first_control(self):
        api = self.controls()
        handlers, sent = {}, []
        def register(number, handler):
            previous = handlers.get(number, signal.SIG_DFL)
            handlers[number] = handler
            return previous
        def wait(**kwargs):
            handlers[signal.SIGINT](signal.SIGINT, None)
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 130
        process = SimpleNamespace(pid=43210, wait=wait)
        with patch.object(api.subprocess, 'Popen', return_value=process) as popen, \
             patch.object(signal, 'signal', side_effect=register), \
             patch.object(os, 'killpg', side_effect=lambda pid, number: sent.append((pid, number))):
            result = api.supervise(['/fake/timeout'], env={}, deadline=100., clock=lambda: 1.)
        self.assertEqual(result.returncode, 130)
        self.assertEqual(result.parent_control, signal.SIGINT)
        self.assertEqual(sent, [(43210, signal.SIGINT)])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])

    def test_outer_supervisor_captures_control_during_owned_child_creation(self):
        api = self.controls()
        handlers, sent = {}, []
        def register(number, handler):
            previous = handlers.get(number, signal.SIG_DFL)
            handlers[number] = handler
            return previous
        def spawn(*args, **kwargs):
            # The parent already has handlers before ownership is returned.
            self.assertTrue(callable(handlers.get(signal.SIGTERM)))
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return SimpleNamespace(pid=43211, wait=lambda **kwargs: 143)
        with patch.object(api.subprocess, 'Popen', side_effect=spawn), \
             patch.object(signal, 'signal', side_effect=register), \
             patch.object(os, 'killpg', side_effect=lambda pid, number: sent.append((pid, number))):
            result = api.supervise(['/fake/timeout'], env={}, deadline=100., clock=lambda: 1.)
        self.assertEqual(sent, [(43211, signal.SIGTERM)])
        self.assertEqual(result.parent_control, signal.SIGTERM)

    def test_fresh_help_stale_validation_and_discovery_block_all_six_prefixes(self):
        code = '''
from contextlib import redirect_stdout
import io, json, pathlib, tempfile, unittest
from tests.unit.test_neural_curriculum_integration import blocked_optional
with blocked_optional() as attempts:
    from scripts.neural import curriculum_controls as api
    with redirect_stdout(io.StringIO()):
        try: api.main(['--help'])
        except SystemExit as e: assert e.code == 0
    with tempfile.TemporaryDirectory() as directory:
        stale = pathlib.Path(directory)/'release.json'
        stale.write_text(json.dumps({'schema': 'spire-learn-sustained-stage-release-v1'}))
        try: api.validate_release(pathlib.Path.cwd(), package_path=stale, source_snapshot=stale, release_path=stale)
        except ValueError: pass
        else: raise AssertionError('stale release accepted')
    suite = unittest.defaultTestLoader.discover('tests')
    assert suite.countTestCases() > 100
    assert not attempts
print('fresh import/help/stale-validation/discovery: six optional prefixes blocked; zero attempts')
'''
        environment = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
        environment['PYTHONPATH'] = 'src'
        child = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=environment,
            capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('zero attempts', child.stdout)

    def test_run_rejects_forged_parent_timestamps_before_static_admission(self):
        api = self.controls()
        with patch.dict(os.environ, {'SPIRE_LEARN_CURRICULUM_PARENT_START': '100.0',
                'SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE': '36100.0'}), \
             patch.object(api, 'validate_release', side_effect=AssertionError('unowned static admission')) as validate:
            with self.assertRaisesRegex(ValueError, 'supervis'):
                api.run_campaign(ROOT, package_path=Path('unused'), source_snapshot=Path('unused'),
                    release_path=Path('unused'), dependencies=api.HostDependencies(clock=lambda: 100.))
        validate.assert_not_called()

    def test_watchdog_timeout_and_control_status_never_become_controller_os_exit(self):
        api = self.controls()
        for code in (0, 124, 137, 143, -signal.SIGTERM):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'outer.json'
                api.launch_guard(['run'], result_path=path, clock=lambda: 5.,
                    run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, code))
                result = json.loads(path.read_text())
                self.assertEqual(result['watchdog_exit_status'], code)
                self.assertEqual(result['controller_os_exit_status'], {
                    'availability': 'unavailable', 'reason': 'only GNU timeout was directly waited on'})
                self.assertFalse(result['whole_run_complete'])

    def test_capture_cli_rejects_provenance_overrides(self):
        api = self.controls()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'never.json'
            with self.assertRaises(SystemExit) as stopped:
                api.main(['capture', '--root', str(ROOT), '--destination', str(destination), '--head', 'a'*40])
            self.assertEqual(stopped.exception.code, 2)
            self.assertFalse(destination.exists())

    def test_actual_owned_launch_proc_access_reaches_only_rejected_static_admission(self):
        api = self.controls()
        runtime = ROOT/api.RUNTIME
        before = (runtime.stat(), sorted(path.name for path in runtime.iterdir())) if runtime.exists() else None
        with tempfile.TemporaryDirectory(dir=ROOT/api.DOMAIN/'task-2') as directory:
            base = Path(directory)
            rejected, result_path = base/'invalid.json', base/'outer.json'
            rejected.write_text('{}')
            environment = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
            environment['PYTHONPATH'] = 'src'
            command = [sys.executable, '-B', str(ROOT/'scripts/neural/curriculum_controls.py'), '_launch',
                '--root', str(ROOT), '--package', str(package_fixture()[0]/'package.json'),
                '--source', str(rejected), '--release', str(rejected), '--result', str(result_path)]
            child = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 1, child.stderr)
            self.assertIn('source snapshot schema differs', child.stderr)
            result = json.loads(result_path.read_text())
            self.assertEqual(result['watchdog_exit_status'], 1)
            self.assertFalse(result['whole_run_complete'])
            self.assertEqual(result['controller_os_exit_status']['availability'], 'unavailable')
            after = (runtime.stat(), sorted(path.name for path in runtime.iterdir())) if runtime.exists() else None
            self.assertEqual(after, before)

    def test_supervisor_policy_checks_chain_limits_open_result_and_source_binding(self):
        api = self.controls()
        with tempfile.TemporaryDirectory(dir=ROOT/api.DOMAIN/'task-2') as directory:
            base = Path(directory)
            result, binding_path = base/'result.json', base/'binding.json'
            result.touch()
            info = result.stat()
            inputs = dict(package_path=package_fixture()[0]/'package.json', source_snapshot=base/'source.json',
                          release_path=base/'release.json')
            child_argv = api._controller_argv(ROOT, **inputs)
            watchdog_argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', '35898s', *child_argv]
            ticks = os.sysconf('SC_CLK_TCK')
            launcher_pid, watchdog_pid, child_pid = 55501, 55502, os.getpid()
            common = dict(uid=1000, state='S', cwd=str(ROOT), executable=str(Path(sys.executable).resolve()))
            facts = {
                launcher_pid: dict(common, pid=launcher_pid, ppid=1, pgid=launcher_pid, session=launcher_pid,
                    start_ticks=1000*ticks, argv=[sys.executable, '-B', child_argv[2], '_launch']),
                watchdog_pid: dict(common, pid=watchdog_pid, ppid=launcher_pid, pgid=watchdog_pid, session=watchdog_pid,
                    start_ticks=1000*ticks+1, argv=watchdog_argv, executable=str(Path('/usr/bin/timeout').resolve())),
                child_pid: dict(common, pid=child_pid, ppid=watchdog_pid, pgid=watchdog_pid, session=watchdog_pid,
                    start_ticks=1000*ticks+2, argv=child_argv)}
            binding = dict(schema='spire-learn-curriculum-supervision-v1', launcher_pid=launcher_pid,
                launcher_start_ticks=1000*ticks, parent_start=1000., whole_deadline=37000.,
                watchdog_argv=watchdog_argv, result_path=str(result), result_fd=3,
                result_device=info.st_dev, result_inode=info.st_ino, controller_source_sha256=api.package_api.sha256(Path(api.__file__)))
            descriptor = dict(path=str(result), device=info.st_dev, inode=info.st_ino, uid=1000, regular=True, writable=True)
            def verify(rows=facts, link=descriptor, value=binding):
                binding_path.write_text(json.dumps(value))
                with blocked_optional(), patch.dict(os.environ, {
                    'SPIRE_LEARN_CURRICULUM_SUPERVISION': str(binding_path),
                    'SPIRE_LEARN_CURRICULUM_SUPERVISION_SHA256': api.package_api.sha256(binding_path),
                    'SPIRE_LEARN_CURRICULUM_PARENT_START': '1000.0',
                    'SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE': '37000.0'}), \
                     patch.object(api, '_proc_identity', side_effect=lambda pid: copy.deepcopy(rows[pid])), \
                     patch.object(api, '_result_descriptor', return_value=link):
                    return api._verify_supervision(ROOT, **inputs, clock=lambda: 1001.)
            self.assertEqual(verify()['watchdog_pid'], watchdog_pid)
            for mutation in ('missing_parent', 'wrong_timeout', 'wrong_launcher', 'wrong_fd', 'wrong_source'):
                with self.subTest(mutation=mutation):
                    rows, link, value = copy.deepcopy(facts), dict(descriptor), copy.deepcopy(binding)
                    if mutation == 'missing_parent':
                        del rows[launcher_pid]
                    elif mutation == 'wrong_timeout':
                        rows[watchdog_pid]['argv'][2] = '--kill-after=91s'
                        value['watchdog_argv'][2] = '--kill-after=91s'
                    elif mutation == 'wrong_launcher':
                        rows[launcher_pid]['argv'][3] = 'run'
                    elif mutation == 'wrong_fd':
                        link['writable'] = False
                    else:
                        value['controller_source_sha256'] = '0'*64
                    with self.assertRaises(ValueError):
                        verify(rows, link, value)


class ProjectedHostTests(unittest.TestCase):
    """Actual host layout, verifier, entry and producer; fake OS/runtime edges."""
    fixture_provider = staticmethod(package_fixture)
    test_domain = None

    def setUp(self):
        self.api = importlib.import_module('scripts.neural.curriculum_controls')
        self.fixture, self.package = self.fixture_provider()
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/(self.test_domain or self.api.DOMAIN)/'task-2')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.h = self.base/'host'
        self.h.mkdir()
        self.guest = self.base/'guest'
        self.control = self.base/'control'
        self.now = 100.
        self.budget = self.api.CampaignBudget(self.api.CAPS, start=100., deadline=36100., clock=lambda: self.now)
        self.source_path = self.h/'source.json'
        snapshot = self.api.capture_source(ROOT, self.source_path, learner_head='a'*40, dirty={'dirty': True})
        release = self.api.release_template(ROOT, package_path=self.fixture/'package.json',
            source_snapshot=self.source_path, user_approval=ROOT/(self.test_domain or self.api.DOMAIN)/'preparation/user-approval.json')
        release.update(state='approved', review={'approved': True, 'reports': ['first', 'second']})
        self.release_path = self.h/'release.json'
        self.api._write(self.release_path, release)
        self.claim_path = self.h/'campaign-claim.json'
        self.api._write(self.claim_path, {'schema': self.api.CLAIM_SCHEMA, 'campaign_id': self.package['campaign_id'],
            'release_sha256': self.api.package_api.sha256(self.release_path),
            'source_sha256': self.api.package_api.sha256(self.source_path),
            'package_sha256': self.api.package_api.sha256(self.fixture/'package.json')})
        self.verified = dict(snapshot=snapshot, release=release, package=self.package,
            package_path=self.fixture/'package.json', source_snapshot=self.source_path,
            release_path=self.release_path, claim_path=self.claim_path)
        self.api.initialize_stage(ROOT, self.h, self.verified)
        self.mapping = {'/learner': self.guest, '/control': self.control,
            '/inputs/parent': self.base/'parent', '/runtime/venv-cuda': ROOT/self.api.low.VENV,
            '/runtime/requirements-cuda.lock': ROOT/self.api.low.LOCK}
        def mapped(value):
            for name in sorted(self.mapping, key=len, reverse=True):
                path = Path(value)
                if path == Path(name) or path.is_relative_to(name):
                    return self.mapping[name]/path.relative_to(name)
            return Path(value)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.api.low, '_guest_path', side_effect=mapped))
        self.calls, self.specs, self.removed = [], {}, []
        self.active, self.finished, self.collision, self.ambiguous, self.bad_mount = None, False, False, False, False
        self.guest_mutation = None
        self.factory_limit = None
        self.harness = ProducerHarness(self, self.guest)
        self.original = ROOT/self.package['original_parent']['checkpoint_directory']
        self.harness.original(self.original)
        self.logs = []
        self.client = self.h/'docker-client'
        self.client.mkdir()
        self.deps = self.api.HostDependencies(clock=lambda: self.now, run=self.docker,
            disk_usage=lambda path: SimpleNamespace(free=200*self.api.GIB), uid=lambda: 1000, gid=lambda: 1000)

    def prepare(self, kind='segment', *, job='first', **kwargs):
        parent = kwargs.pop('parent', {'path': str(self.original), 'inventory': self.package['original_parent']['inventory']})
        return self.api.prepare_job(ROOT, self.h, self.verified, self.budget,
            kind=kind, job_id='curriculum-'+job, parent=parent, **kwargs)

    def docker(self, argv, **kwargs):
        self.calls.append(list(argv))
        low = self.api.low
        if argv[0] == 'nvidia-smi':
            if 'compute-apps' in argv[1]:
                return subprocess.CompletedProcess(argv, 0, '3772, 498, '+low.GPU_UUID+'\n', '')
            return subprocess.CompletedProcess(argv, 0, low.GPU_UUID+', NVIDIA GeForce RTX 5090, 595.91.07, 32607, 31000\n', '')
        verb = argv[5] if argv[0] == '/fake/docker' else ''
        stdout, stderr = '', ''
        if verb == 'ps':
            if self.collision and any('name=' in a for a in argv):
                stdout = 'existing-other-container\n'
        elif verb == 'create':
            bindings = [argv[n+1] for n, value in enumerate(argv) if value == '--mount']
            control = next(value.split(',')[1][4:] for value in bindings
                if ',dst='+str(self.control)+',readonly' in value)
            self.context = json.loads((Path(control)/'job.json').read_text())
            layout = json.loads((Path(control)/'layout.json').read_text())
            self.spec = self.api._spec(self.context, Path(control)/'job.json', layout)
            env = [tuple(argv[n+1].split('=', 1)) for n, value in enumerate(argv) if value == '--env']
            labels = [tuple(argv[n+1].split('=', 1)) for n, value in enumerate(argv) if value == '--label']
            self.spec = replace(self.spec, environment=tuple(env), labels=tuple(labels),
                context_sha256=dict(env)[self.api.ENV_CONTEXT])
            self.finished = False
            self.output = Path(next(s for s, d, r in self.spec.mounts if not r))
            self.parent_host = {'path': next(s for s, d, r in self.spec.mounts
                if d == str(self.base/'parent') or d == str(self.guest/self.package['original_parent']['checkpoint_directory'])),
                'inventory': self.context['parent']['inventory']}
            if self.ambiguous:
                return subprocess.CompletedProcess(argv, 1, 'a'*64+'\n', 'uncertain create')
            self.active = 'a'*64
            self.specs[self.active] = self.spec
            self.assertEqual(argv, low.build_create_argv(self.spec, '/fake/docker', self.client))
            stdout = self.active+'\n'
        elif verb == 'inspect':
            self.assertEqual(argv[-1], self.active)
            record = low._inspect_projection(self.spec, self.active)
            if self.finished:
                record['State']['Status'] = 'exited'
            if self.bad_mount:
                record['Mounts'][0]['RW'] = True
            stdout = json.dumps([record])
        elif verb == 'start':
            self.project()
            error = io.StringIO()
            try:
                with redirect_stderr(error), self.guest_boundaries():
                    self.api.producer._guest_entry(self.guest, self.control/'job.json',
                        dependencies=self.harness.dependencies(), clock=lambda: self.now)
            except BaseException:
                # A strict fake Docker start reflects the guest's actual failure.
                stderr = error.getvalue()
                self.copy_output()
                self.finished = True
                raise
            stderr = error.getvalue()
            self.copy_output()
            self.finished = True
            self.now += 1.
        elif verb == 'rm':
            self.assertEqual(argv[-1], 'a'*64)
            self.removed.append(argv[-1])
        else:
            raise AssertionError('unexpected transport command '+repr(argv))
        return subprocess.CompletedProcess(argv, 0, stdout, stderr)

    def project(self):
        if not self.guest.exists():
            shutil.copytree(self.h/'source', self.guest)
        for source, destination, readonly in self.spec.mounts:
            path = Path(destination)
            if destination == str(self.guest) or destination.startswith(str(ROOT/self.api.low.VENV)) or destination == str(ROOT/self.api.low.LOCK):
                continue
            if path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).is_dir():
                shutil.copytree(source, path)
            else:
                shutil.copyfile(source, path)
        parent_wire = self.context['parent']['path']
        parent_actual = self.api.producer._external_path(self.guest, parent_wire, parent=True)
        self.harness.loaded[parent_actual.resolve()] = self.harness.loaded[Path(self.parent_host['path']).resolve()]

    def copy_output(self):
        guest_output = Path(self.context['output_root'])
        for path in guest_output.iterdir():
            target = self.output/path.name
            shutil.copytree(path, target) if path.is_dir() else shutil.copyfile(path, target)
        # Model/state tensors stay fake at the archive boundary only.
        for path, value in list(self.harness.loaded.items()):
            if path.is_relative_to(guest_output):
                self.harness.loaded[(self.output/path.relative_to(guest_output)).resolve()] = value

    def guest_boundaries(self):
        stack = ExitStack()
        stack.enter_context(blocked_optional())
        stack.enter_context(patch.dict(os.environ, dict(self.spec.environment), clear=True))
        facts = {'mounts': [('/', True)]+[(d, r) for s, d, r in self.spec.mounts],
            'environment': dict(self.spec.environment), 'uid': 1000, 'gid': 1000,
            'python': self.api.low.CONTAINER_PYTHON, 'device_nodes': [], 'supplementary_group_ids': [1000]}
        if self.spec.uses_gpu:
            facts = cdi_facts(facts)
        if self.guest_mutation:
            self.guest_mutation(facts)
        stack.enter_context(patch.object(self.api.low, '_guest_facts', return_value=facts))
        stack.enter_context(patch.object(self.api.low, '_verify_guest_runtime'))
        stack.enter_context(patch.object(self.api.low, '_runtime_metadata', return_value=self.api.PIN_VERSIONS))
        from scripts.neural import sustained_runtime
        process = dict(pid=999, python=self.api.low.CONTAINER_PYTHON,
            threads={'intraop': 4, 'interop': 1}, modules={'torch': dict(version='2.14.1+cu130',
                file='/runtime/venv-cuda/lib/python3.12/site-packages/torch/__init__.py')})
        stack.enter_context(patch.object(sustained_runtime, 'process_evidence', return_value=process))
        stack.enter_context(self.harness.producers())
        return stack

    def execute(self, prepared):
        self.context, self.spec, self.output = prepared['context'], prepared['spec'], prepared['output']
        self.parent_host = prepared['parent_host']
        self.finished = False
        lifecycle = self.api.transport.execute_job(self.spec, deps=self.deps, client=self.client,
            docker='/fake/docker', logs=self.logs, progress=lambda: None)
        return self.api.reconcile_job(ROOT, self.verified, self.context, self.output,
            lifecycle=lifecycle, budget=self.budget, clock=lambda: self.now)

    def test_projected_gate_warm_resume_cpu_panel_real_receipts(self):
        qualification = self.prepare('qualification', job='qualification')
        q = self.execute(qualification)
        self.assertEqual(q['receipt']['qualification_attempts']['adam'], 8)
        first = self.execute(self.prepare(arm='mixed', index=0, job='mixed-0', max_updates=1))
        self.assertEqual(first['checkpoint']['counters']['update_count'], 1)
        second = self.execute(self.prepare(arm='mixed', index=1, job='mixed-1', max_updates=2,
            prefix=first['checkpoint']['counters'], parent=first['checkpoint'], predecessor=first['receipt_binding']))
        self.assertEqual(second['checkpoint']['counters']['update_count'], 2)
        panel = self.execute(self.prepare('panel', job='panel', slot='final-mixed', parent=second['checkpoint']))
        self.assertEqual(len(panel['receipt']['reports']), 64)
        self.assertEqual(panel['receipt']['effects']['sample_rows']['attempted'], 0)
        self.assertEqual(len(self.removed), 4)
        self.assertTrue(all(driver.close_calls == 1 for driver in self.harness.drivers))

    def test_collision_and_uncertain_create_never_clean_by_name(self):
        prepared = self.prepare(arm='fixed', index=0)
        self.collision = True
        with self.assertRaises(RuntimeError):
            self.execute(prepared)
        self.assertEqual(self.removed, [])
        self.collision, self.ambiguous = False, True
        with self.assertRaises(ValueError):
            self.execute(prepared)
        self.assertEqual(self.removed, [])
        self.assertFalse(any('rm' in argv for argv in self.calls))

    def test_actual_guest_mount_and_uid_reject_before_factory(self):
        self.guest_mutation = lambda facts: facts.update(uid=0)
        with self.assertRaises(ValueError):
            self.execute(self.prepare(arm='fixed', index=0))
        self.assertEqual(self.harness.factory_calls, [])
        self.assertEqual(self.removed, ['a'*64])

    def test_raw_context_and_staged_source_reject_before_factory(self):
        prepared = self.prepare(arm='fixed', index=0)
        prepared['context_path'].write_text(prepared['context_path'].read_text()+'\n')
        with self.assertRaises(ValueError):
            self.execute(prepared)
        self.assertEqual(self.harness.factory_calls, [])
        with self.assertRaises(ValueError):
            self.api.prepare_job(ROOT, self.h, self.verified, self.budget, kind='segment',
                job_id='curriculum-first', parent=prepared['parent_host'], arm='fixed', index=0)

    def test_storage_reservation_enforces_new_aggregate_and_free_floors(self):
        with patch.object(self.api.low, 'tree_bytes', side_effect=[12*self.api.GIB-1, 20*self.api.GIB]):
            with self.assertRaises(RuntimeError):
                self.api.storage_admission(ROOT, self.h, self.deps, reservation=2)
        with patch.object(self.api.low, 'tree_bytes', side_effect=[1, 40*self.api.GIB]):
            with self.assertRaises(RuntimeError):
                self.api.storage_admission(ROOT, self.h, self.deps, reservation=1)

    def test_saved_prefix_or_nested_cpu_fault_cannot_settle_reserved_attempt(self):
        first = self.prepare(arm='mixed', index=0, max_updates=1)
        result = self.execute(first)
        raw = self.output/'job-receipt.json'
        receipt = json.loads(raw.read_text())
        receipt['checkpoint']['counters']['update_count'] += 1
        raw.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            self.api.reconcile_job(ROOT, self.verified, self.context, self.output,
                lifecycle={'container_id': 'a'*64, 'removal_entered': True,
                    'admission': {'job_id': self.context['job_id']}}, budget=self.budget, clock=lambda: self.now)
        panel = self.prepare('panel', job='panel-fault', slot='final-mixed', parent=result['checkpoint'])
        self.harness.step_error = RuntimeError('actual native boundary fault')
        with self.assertRaises(RuntimeError):
            self.execute(panel)
        self.assertEqual(self.budget.jobs['curriculum-panel-fault']['state'], 'reserved')

    def test_host_rechecks_held_saved_prefix_and_nested_panel_exit(self):
        for kind in ('qualification', 'panel'):
            with self.subTest(kind=kind):
                prepared = self.prepare(kind, job=kind+'-tamper', **({'slot': 'original-b'} if kind == 'panel' else {}))
                result = self.execute(prepared)
                path = self.output/'job-receipt.json'
                receipt = json.loads(path.read_text())
                if kind == 'qualification':
                    receipt['qualification_segments'][0]['saved_prefix']['update_count'] += 1
                else:
                    receipt['reports'][0]['exit_code'] = 1
                path.write_text(json.dumps(receipt))
                budget = self.api.CampaignBudget(self.api.CAPS, start=100., deadline=36100., clock=lambda: self.now)
                budget.reserve(self.context['job_id'], self.context['reservation'])
                with self.assertRaises(ValueError):
                    self.api.reconcile_job(ROOT, self.verified, self.context, self.output,
                        lifecycle={'container_id': 'a'*64, 'removal_entered': True,
                            'admission': {'job_id': self.context['job_id']}}, budget=budget, clock=lambda: self.now)

    def test_qualification_actual_effects_and_one_update_progression_cross_reconcile(self):
        self.execute(self.prepare('qualification', job='qualification-crosscheck'))
        originals = {path: path.read_bytes() for path in self.output.rglob('*.json')}
        for mutation in ('actions', 'samples', 'adam', 'resets', 'report_actions', 'update_count'):
            with self.subTest(mutation=mutation):
                for path, raw in originals.items():
                    path.write_bytes(raw)
                receipt_path = self.output/'job-receipt.json'
                receipt = json.loads(receipt_path.read_text())
                if mutation == 'report_actions':
                    segment = receipt['qualification_segments'][0]
                    checkpoint = self.api._mapped(self.context, self.output, segment['checkpoint']['path'])
                    summary_path = checkpoint.parent/'summary.json'
                    envelope = json.loads(summary_path.read_text())
                    segment['report']['collection']['attempted_environment_slots'] += 4
                    envelope['summary']['collection']['attempted_environment_slots'] += 4
                    envelope['summary_sha256'] = self.api.transport.canonical_sha256(envelope['summary'])
                    summary_path.write_text(json.dumps(envelope))
                elif mutation == 'update_count':
                    segment = receipt['qualification_segments'][0]
                    checkpoint = self.api._mapped(self.context, self.output, segment['checkpoint']['path'])
                    metadata_path = checkpoint/'metadata.json'
                    metadata = json.loads(metadata_path.read_text())
                    metadata['training_state']['counters']['update_count'] = 2
                    metadata['run_metadata']['cumulative_prefix']['update_count'] = 2
                    metadata_path.write_text(json.dumps(metadata))
                    segment['saved_prefix']['update_count'] = 2
                    segment['checkpoint']['inventory']['metadata.json'] = {
                        'sha256': self.api.package_api.sha256(metadata_path), 'size_bytes': metadata_path.stat().st_size}
                else:
                    kind = dict(actions='environment_attempts', samples='sample_rows', adam='optimizer_attempts', resets='reset_slots')[mutation]
                    difference = -1 if mutation == 'adam' else 1
                    control = self.output/'campaign-control'
                    attempts_path = control/'attempts.json'
                    attempts = json.loads(attempts_path.read_text())
                    receipt['qualification_attempts'][mutation] += difference
                    attempts['qualification_attempts'][mutation] += difference
                    for key in ('reserved', 'attempted', 'delivered'):
                        receipt['effects'][kind][key] += difference
                        attempts['effects'][kind][key] += difference
                        attempts['categories']['qualification-'+mutation][key] += difference
                    selected = None
                    for part in attempts['journal']:
                        path = control/part['path']
                        journal = json.loads(path.read_text())
                        for row in journal['records']:
                            if selected is None and row['kind'] == kind:
                                selected = row['operation_id']
                            if row['operation_id'] == selected:
                                row['reserved'] += difference
                                if row['event'] == 'settle':
                                    row['attempted'] += difference
                                    row['delivered'] += difference
                        path.write_text(json.dumps(journal))
                        part.update(sha256=self.api.package_api.sha256(path), size_bytes=path.stat().st_size)
                    attempts_path.write_text(json.dumps(attempts))
                receipt_path.write_text(json.dumps(receipt))
                budget = self.api.CampaignBudget(self.api.CAPS, start=100., deadline=36100., clock=lambda: self.now)
                budget.reserve(self.context['job_id'], self.context['reservation'])
                with self.assertRaises(ValueError):
                    self.api.reconcile_job(ROOT, self.verified, self.context, self.output,
                        lifecycle={'container_id': 'a'*64, 'removal_entered': True,
                            'admission': {'job_id': self.context['job_id']}}, budget=budget, clock=lambda: self.now)

    def test_preflight_interrupt_keeps_lease_and_owned_cleanup_runs_once(self):
        prepared = self.prepare(arm='fixed', index=0)
        self.harness.close_error = SystemExit('close control')
        self.harness.wrong = 'model'
        with self.assertRaises(SystemExit):
            self.execute(prepared)
        self.assertEqual(self.removed, ['a'*64])
        self.assertEqual(self.budget.jobs['curriculum-first']['state'], 'reserved')
        self.assertTrue(all(driver.close_calls == 1 for driver in self.harness.drivers))

    def test_repeated_cleanup_signals_defer_until_owned_full_id_removal(self):
        prepared = self.prepare(arm='fixed', index=0, max_updates=1)
        original = self.deps.run
        def interrupted(argv, **kwargs):
            if 'rm' in argv:
                handler = signal.getsignal(signal.SIGTERM)
                handler(signal.SIGTERM, None)
                handler(signal.SIGTERM, None)
            return original(argv, **kwargs)
        self.deps = replace(self.deps, run=interrupted)
        with self.api.transport.signal_scope(), self.assertRaises(self.api.low._Termination):
            self.execute(prepared)
        self.assertEqual(self.removed, ['a'*64])
        self.assertEqual(self.budget.jobs['curriculum-first']['state'], 'reserved')

    def test_inspect_mount_drift_still_requires_exact_owner_for_cleanup(self):
        self.bad_mount = True
        with self.assertRaises(ValueError):
            self.execute(self.prepare(arm='fixed', index=0))
        self.assertEqual(self.harness.factory_calls, [])
        self.assertEqual(self.removed, [])

    def test_failed_host_attempt_publication_stops_after_cleanup_and_preserves_control(self):
        original = self.api._write
        def interrupted(path, value):
            if Path(path).parent.name == 'attempts':
                raise KeyboardInterrupt('host publication control')
            return original(path, value)
        with patch.object(self.api, '_write', side_effect=interrupted), \
             patch.object(self.api.cdi, 'verify_host_profile'), self.assertRaises(KeyboardInterrupt):
            self.api.dispatch_job(ROOT, self.h, self.verified, self.budget, deps=self.deps,
                docker='/fake/docker', client=self.client, kind='segment', job_id='curriculum-publication',
                parent={'path': str(self.original), 'inventory': self.package['original_parent']['inventory']},
                arm='fixed', index=0, max_updates=1)
        self.assertEqual(self.removed, ['a'*64])

    def test_first_different_cleanup_control_supersedes_ordinary_guest_fault(self):
        original = self.deps.run
        def interrupted(argv, **kwargs):
            if 'rm' in argv:
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return original(argv, **kwargs)
        self.deps = replace(self.deps, run=interrupted)
        self.harness.wrong = 'model'
        with self.api.transport.signal_scope(), patch.object(self.api.cdi, 'verify_host_profile'):
            with self.assertRaises(self.api.low._Termination) as stopped:
                self.api.dispatch_job(ROOT, self.h, self.verified, self.budget, deps=self.deps,
                    docker='/fake/docker', client=self.client, kind='segment', job_id='curriculum-first-control',
                    parent={'path': str(self.original), 'inventory': self.package['original_parent']['inventory']},
                    arm='fixed', index=0, max_updates=1)
        self.assertEqual(str(stopped.exception), 'signal '+str(signal.SIGINT))
        self.assertEqual(self.removed, ['a'*64])

    def test_cpu_evaluation_lease_is_shared_across_opportunities(self):
        self.budget.evaluation_seconds = 1750.
        with self.assertRaises(TimeoutError):
            self.prepare('panel', slot='original-b')
        self.assertEqual(self.harness.factory_calls, [])

    def test_staged_source_drift_and_short_shared_lease_refuse_new_job(self):
        (self.h/'source/src/spire_learn/__init__.py').write_text('unreviewed')
        with self.assertRaises(ValueError):
            self.prepare(arm='fixed', index=0)
        self.assertEqual(self.budget.jobs, {})
        self.api.stage_source(ROOT, self.h/'replacement', self.verified['snapshot'])
        shutil.rmtree(self.h/'source')
        (self.h/'replacement').rename(self.h/'source')
        self.now = self.budget.deadline-150.
        with self.assertRaises(TimeoutError):
            self.prepare(arm='fixed', index=0)

    def test_new_host_job_uses_actual_transport_and_global_preflight(self):
        with patch.object(self.api.cdi, 'verify_host_profile'):
            result = self.api.dispatch_job(ROOT, self.h, self.verified, self.budget, deps=self.deps,
                docker='/fake/docker', client=self.client, kind='segment', job_id='curriculum-dispatch',
                parent={'path': str(self.original), 'inventory': self.package['original_parent']['inventory']},
                arm='fixed', index=0, max_updates=1)
        self.assertEqual(result['checkpoint']['counters']['update_count'], 1)
        attempt = json.loads((self.h/'attempts/curriculum-dispatch.json').read_text())
        self.assertEqual(attempt['status'], 'complete')
        self.assertEqual(result['unrelated_gpu_process_observations']['before'][0]['pid'], 3772)

    def test_all_scratch_paths_exist_before_tensor_checkpoint_loading(self):
        original = self.harness.load
        observed = []
        def load(*args, **kwargs):
            for key in self.api.low.SCRATCH:
                self.assertTrue(Path(os.environ[key]).is_dir(), key)
                self.assertEqual(Path(os.environ[key]), Path(self.context['output_root']))
                observed.append(key)
            # Strict fake tensor import: only after the real verifier and
            # producer have accepted the empty output may a library write.
            self.assertEqual(list(Path(self.context['output_root']).iterdir()), [])
            (Path(os.environ['TMPDIR'])/'fake-library-cache').mkdir()
            return original(*args, **kwargs)
        self.harness.load = load
        result = self.execute(self.prepare(arm='fixed', index=0, max_updates=1))
        self.assertEqual(result['checkpoint']['counters']['update_count'], 1)
        self.assertEqual(set(observed), set(self.api.low.SCRATCH))

    def test_actual_mixed_job_charge_includes_host_attempt_publication(self):
        original = self.api._write
        def publication(path, value):
            result = original(path, value)
            if Path(path).parent.name == 'attempts':
                self.now += 17.5
            return result
        with patch.object(self.api, '_write', side_effect=publication), patch.object(self.api.cdi, 'verify_host_profile'):
            result = self.api.dispatch_job(ROOT, self.h, self.verified, self.budget, deps=self.deps,
                docker='/fake/docker', client=self.client, kind='segment', job_id='curriculum-mixed-charge',
                parent={'path': str(self.original), 'inventory': self.package['original_parent']['inventory']},
                arm='mixed', index=0, max_updates=1)
        self.assertEqual(result['host_elapsed_seconds'], 18.5)
        self.assertEqual(self.budget.training_seconds, {'fixed': 0., 'randomized': 0., 'mixed': 18.5})
        self.assertEqual(self.budget.evaluation_seconds, 0.)

    def test_late_host_publication_cannot_return_a_passing_successor(self):
        original = self.api._write
        def publication(path, value):
            result = original(path, value)
            if Path(path).parent.name == 'attempts':
                self.now += 601.
            return result
        with patch.object(self.api, '_write', side_effect=publication), patch.object(self.api.cdi, 'verify_host_profile'):
            with self.assertRaises(TimeoutError):
                self.api.dispatch_job(ROOT, self.h, self.verified, self.budget, deps=self.deps,
                    docker='/fake/docker', client=self.client, kind='segment', job_id='curriculum-late-publication',
                    parent={'path': str(self.original), 'inventory': self.package['original_parent']['inventory']},
                    arm='mixed', index=0, max_updates=1)
        self.assertEqual(self.removed, ['a'*64])

    def test_complete_controller_projects_real_gate_baseline_training_and_time_cutoff(self):
        api = self.api
        self.h = self.base/'campaign'
        self.client = self.h/'docker-client'
        real_next = api.CampaignSchedule.next
        def next_request(plan, now):
            if plan.indices['fixed'] == 1:
                self.now = max(self.now, 33701.)
            return real_next(plan, self.now)
        with blocked_optional(), patch.dict(os.environ, {
                'SPIRE_LEARN_CURRICULUM_PARENT_START': '100.0',
                'SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE': '36100.0'}), \
             patch.object(api, 'RUNTIME', self.h.relative_to(ROOT)), \
             patch.object(api, '_verify_supervision', return_value={'parent_start': 100., 'whole_deadline': 36100.}), \
             patch.object(api, 'validate_release', return_value=self.verified), \
             patch.object(api.low, '_docker_binary', return_value='/fake/docker'), \
             patch.object(api.cdi, 'verify_host_profile'), \
             patch.object(api.CampaignSchedule, 'next', next_request):
            seal = api.run_campaign(ROOT, package_path=self.fixture/'package.json',
                source_snapshot=self.source_path, release_path=self.release_path,
                dependencies=self.deps, clock=lambda: self.now)
            normal = self.base/'normal-watchdog-result.json'
            api.launch_guard(['run', '--root', str(ROOT)], result_path=normal, parent_start=100.,
                clock=lambda: self.now, run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0))
            observed = json.loads(normal.read_text())
            self.assertTrue(observed['whole_run_complete'])
            self.assertEqual(observed['controller_logical_result']['status'], 'complete')
            self.assertEqual(observed['controller_os_exit_status']['availability'], 'unavailable')
            api._write(self.h/'campaign-seal-failure.json', {'status': 'failed'})
            veto = self.base/'veto-watchdog-result.json'
            api.launch_guard(['run', '--root', str(ROOT)], result_path=veto, parent_start=100.,
                clock=lambda: self.now, run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0))
            self.assertFalse(json.loads(veto.read_text())['whole_run_complete'])
        self.assertEqual(seal['status'], 'complete')
        self.assertEqual([row['job_id'] for row in seal['jobs']], [
            'curriculum-qualification', 'curriculum-panel-original-b', 'curriculum-fixed-000',
            'curriculum-panel-matched-fixed'])
        self.assertEqual(seal['prefixes']['fixed']['attempted_environment_slots'], 4096)
        self.assertEqual(seal['prefixes']['randomized'], {})
        self.assertEqual(seal['availability']['matched-randomized']['status'], 'unavailable')
        self.assertEqual(seal['availability']['mixed-time-6000']['status'], 'not_started')
        self.assertEqual(len(self.removed), 4)
        self.assertEqual(seal['training_seconds'], {'fixed': 1., 'randomized': 0., 'mixed': 0.})
        self.assertEqual(seal['evaluation_seconds'], 2.)
        self.assertEqual(seal['qualification_seconds'], 1.)
        with self.assertRaises(FileExistsError):
            api._write(self.h/'campaign-claim.json', {})
        self.deps = replace(self.deps, disk_usage=lambda p: SimpleNamespace(free=30*self.api.GIB))
        with patch.object(self.api.low, 'tree_bytes', return_value=0):
            with self.assertRaises(RuntimeError):
                self.api.storage_admission(ROOT, self.h, self.deps, reservation=1)

    def test_actual_failed_panel_is_failed_in_sealed_availability_and_job_history(self):
        api = self.api
        self.h = self.base/'failed-panel-campaign'
        self.client = self.h/'docker-client'
        original_factory = self.harness.factory
        evaluations = []
        def factory(config, **kwargs):
            if kwargs['purpose'] == 'evaluate':
                evaluations.append(config.cases.cases[0].case_id)
                self.harness.wrong = 'eval-policy-version' if len(evaluations) == 2 else None
            return original_factory(config, **kwargs)
        self.harness.factory = factory
        with blocked_optional(), patch.object(api, 'RUNTIME', self.h.relative_to(ROOT)), \
             patch.object(api, '_verify_supervision', return_value={'parent_start': 100., 'whole_deadline': 36100.}), \
             patch.object(api, 'validate_release', return_value=self.verified), \
             patch.object(api.low, '_docker_binary', return_value='/fake/docker'), \
             patch.object(api.cdi, 'verify_host_profile'):
            with self.assertRaisesRegex(RuntimeError, 'operational_failure'):
                api.run_campaign(ROOT, package_path=self.fixture/'package.json', source_snapshot=self.source_path,
                    release_path=self.release_path, dependencies=self.deps, clock=lambda: self.now)
        seal = json.loads((self.h/'campaign-result.json').read_text())
        self.assertEqual(seal['status'], 'failed')
        self.assertEqual(seal['availability']['original-b']['status'], 'failed')
        self.assertIn('operational_failure', seal['availability']['original-b']['reason'])
        self.assertTrue(all(row['status'] == 'not_started' for slot, row in seal['availability'].items() if slot != 'original-b'))
        failed = seal['jobs'][-1]
        self.assertEqual(failed['job_id'], 'curriculum-panel-original-b')
        self.assertEqual(failed['status'], 'failed')
        self.assertFalse(failed['receipt_reconciled'])
        for name in ('claim', 'attempt', 'context', 'receipt', 'guest_failure'):
            reference = failed['references'][name]
            self.assertEqual(reference['availability'], 'retained')
            self.assertEqual(api.package_api.sha256(Path(reference['path'])), reference['sha256'])
        receipt = json.loads(Path(failed['references']['receipt']['path']).read_text())
        self.assertEqual(receipt['status'], 'operational_failure')
        self.assertEqual(len(receipt['reports']), 2)
        self.assertEqual(len(failed['partial_reports']), 2)
        self.assertEqual(seal['attempts'][failed['job_id']]['state'], 'reserved')
        self.assertEqual(len(evaluations), 2)
        self.assertFalse(any(row[0] == 'warm_start' and row[1] == self.base/'parent' for row in self.harness.factory_calls))
        self.assertEqual(len(self.removed), 2)

    def test_failed_panel_reference_inspection_finishes_under_repeated_controls(self):
        api = self.api
        host = self.base/'interrupted-failure-evidence'
        original_reference, injected = api._retained_reference, [False]
        def reference(*args, **kwargs):
            if not injected[0]:
                injected[0] = True
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return original_reference(*args, **kwargs)
        with blocked_optional(), patch.object(api, 'RUNTIME', host.relative_to(ROOT)), \
             patch.object(api, '_verify_supervision', return_value={'parent_start': 100., 'whole_deadline': 36100.}), \
             patch.object(api, 'validate_release', return_value=self.verified), \
             patch.object(api.low, '_docker_binary', return_value='/fake/docker'), \
             patch.object(api.CampaignSchedule, 'next', return_value={'kind': 'panel', 'slot': 'original-b', 'arm': None}), \
             patch.object(api, 'dispatch_job', side_effect=RuntimeError('earlier panel dispatch failure')), \
             patch.object(api, '_retained_reference', side_effect=reference):
            with self.assertRaises(api.low._Termination) as stopped:
                api.run_campaign(ROOT, package_path=self.fixture/'package.json', source_snapshot=self.source_path,
                    release_path=self.release_path, dependencies=self.deps, clock=lambda: self.now)
        self.assertEqual(str(stopped.exception), 'signal '+str(signal.SIGINT))
        seal = json.loads((host/'campaign-result.json').read_text())
        self.assertEqual(seal['availability']['original-b']['status'], 'failed')
        self.assertIn('earlier panel dispatch failure', seal['availability']['original-b']['reason'])
        self.assertEqual(len(seal['jobs']), 1)
        self.assertTrue(all(value['availability'] == 'unavailable' and 'expected_path' in value
                            for value in seal['jobs'][0]['references'].values()))
        self.assertEqual(self.calls, [])

    def test_signal_scope_covers_seal_write_fsync_and_failure_veto_publication(self):
        api = self.api
        original_write, original_fsync = api._write, os.fsync
        for index, stage in enumerate(('write', 'fsync', 'veto_failure')):
            with self.subTest(stage=stage):
                host = self.base/f'seal-{index}'
                active = [False]
                def deliver():
                    self.assertIsNotNone(api._control_state.get(), 'sealing lost its owned signal scope')
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                def write(path, value):
                    if Path(path).name == 'campaign-result.json':
                        if stage in ('write', 'veto_failure'):
                            deliver()
                        active[0] = True
                        try:
                            if stage == 'veto_failure':
                                raise OSError('seal publication failed')
                            return original_write(path, value)
                        finally:
                            active[0] = False
                    if Path(path).name == 'campaign-seal-failure.json' and stage == 'veto_failure':
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                        raise SystemExit('later veto publication control')
                    return original_write(path, value)
                def fsync(descriptor):
                    if active[0] and stage == 'fsync':
                        deliver()
                    return original_fsync(descriptor)
                previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
                with blocked_optional(), patch.object(api, 'RUNTIME', host.relative_to(ROOT)), \
                     patch.object(api, '_verify_supervision', return_value={'parent_start': 100., 'whole_deadline': 36100.}), \
                     patch.object(api, 'validate_release', return_value=self.verified), \
                     patch.object(api.low, '_docker_binary', return_value='/fake/docker'), \
                     patch.object(api.CampaignSchedule, 'next', return_value=None), \
                     patch.object(api, '_write', side_effect=write), patch.object(os, 'fsync', side_effect=fsync):
                    with self.assertRaises(api.low._Termination) as stopped:
                        api.run_campaign(ROOT, package_path=self.fixture/'package.json', source_snapshot=self.source_path,
                            release_path=self.release_path, dependencies=self.deps, clock=lambda: self.now)
                self.assertEqual(str(stopped.exception), 'signal '+str(signal.SIGINT))
                self.assertEqual({number: signal.getsignal(number) for number in previous}, previous)
                if (host/'campaign-result.json').exists():
                    self.assertFalse(json.loads((host/'campaign-result.json').read_text())['whole_run_claim'])
                if stage != 'veto_failure':
                    self.assertEqual(json.loads((host/'campaign-seal-failure.json').read_text())['status'], 'failed')
                self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
