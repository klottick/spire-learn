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
        self.assertEqual(retained['child_exit_code'], 17)
        self.assertEqual(calls[0][0][:5], ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', '35900s', sys.executable])
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
        self.assertIsNone(retained['child_exit_code'])
        self.assertEqual(retained['error']['type'], 'OSError')

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


class ProjectedHostTests(unittest.TestCase):
    """Actual host layout, verifier, entry and producer; fake OS/runtime edges."""
    def setUp(self):
        self.api = importlib.import_module('scripts.neural.curriculum_controls')
        self.fixture, self.package = package_fixture()
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/self.api.DOMAIN/'task-2')
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
            source_snapshot=self.source_path, user_approval=ROOT/self.api.DOMAIN/'preparation/user-approval.json')
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
             patch.object(api, 'validate_release', return_value=self.verified), \
             patch.object(api.low, '_docker_binary', return_value='/fake/docker'), \
             patch.object(api.cdi, 'verify_host_profile'), \
             patch.object(api.CampaignSchedule, 'next', next_request):
            seal = api.run_campaign(ROOT, package_path=self.fixture/'package.json',
                source_snapshot=self.source_path, release_path=self.release_path,
                dependencies=self.deps, clock=lambda: self.now)
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


if __name__ == '__main__':
    unittest.main()
