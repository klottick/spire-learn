"""Task 9 controls: no Docker/native/backend calls; actual B CLI is projected."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.abc
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def controls():
    return importlib.import_module('scripts.neural.task9_study_controls')


class ControlFixture:
    def __init__(self, directory):
        self.c = controls()
        self.root = Path(directory) / 'checkout'
        shutil.copytree(ROOT / 'src', self.root / 'src')
        shutil.copytree(ROOT / 'configs', self.root / 'configs')
        for name in self.c.SNAPSHOT_PATHS:
            p = ROOT / name
            if p.is_file():
                q = self.root / name
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
        self.records = self.root / 'artifacts/development/neural-combat/task-9/controls-fixture'
        self.records.mkdir(parents=True)
        self.snapshot = self.records / 'snapshot.json'
        from spire_learn.neural import config as config_module
        with patch.object(config_module,'__file__',str(self.root/'src/spire_learn/neural/config.py')):
            self.c.capture_source_snapshot(self.root, self.snapshot, learner_head='a'*40,
                                           dirty={}, state='reviewed')
            self.study=self.c._study(self.root)
        self.review = self.records / 'review.md'
        self.review.write_text('Independent review fixture only.\n')
        self.approval = self.records / 'approval.json'
        self.c.write_json_exclusive(self.approval, {
            'schema': self.c.APPROVAL_SCHEMA, 'state': 'approved', 'single_use': True,
            'campaign_id': 'fixture-a', 'source_snapshot_sha256': self.c.sha256_file(self.snapshot),
            'study_raw_sha256': self.c.STUDY_SHA256,
            'review_path': self.review.relative_to(self.root).as_posix(),
            'review_sha256': self.c.sha256_file(self.review),
            'phase_order': list(self.c.PHASES), 'runtime_pins': self.study['runtime_pins'],
            'budgets': self.study['budgets'],
            'checkpoint_binding': 'actual-post-training-final-inventory-once',
        })
        self.h = self.records / 'campaign-fixture-a'
        self.now = 100.1234567890123
        self.gpu_csv=self.c.GPU_UUID+', NVIDIA GeForce RTX 5090, 595.91.07, 32607, 20000\n'
        self.commands = []
        self.instances = {}
        self.started=set()
        self.failure = None
        self.context_calls = []
        self.children = []
        self.factory_calls = []
        self.guest_mutation = None
        self.thread_evidence=True
        self.fresh_reference=False
        self.reference_child_pid=None
        self.last_guest_stderr=''
        self.selection_expiry=None
        self.selection_expiry_operation='selection-000'
        self.evaluation_calls=[]
        self.reservation_fault=None
        self.child_timeouts=[]

    def dependencies(self):
        return self.c.HostDependencies(clock=lambda:self.now, run=self.run,
            disk_usage=lambda p: SimpleNamespace(free=100*1024**3), uid=lambda:1000, gid=lambda:1000)

    def runtime(self, root, approval):
        return {'lock_sha256': self.c.LOCK_SHA256, 'native_sha256': self.c.NATIVE_SHA256,
                'torch': '2.14.1+cu130', 'torch_cuda': '13.0', 'venv': str(root/self.c.VENV)}

    def run(self, argv, **kw):
        self.commands.append((list(argv), kw))
        self.assert_bounded(argv, kw)
        if argv[0] == 'git':
            out = 'a'*40+'\n' if 'rev-parse' in argv else ''
            return subprocess.CompletedProcess(argv,0,out,'')
        if argv[0] == 'nvidia-smi':
            out='777, 1, '+self.c.GPU_UUID+'\n' if any('compute-apps' in a for a in argv) else self.gpu_csv
            return subprocess.CompletedProcess(argv,0,out,'')
        # Every fake Docker call requires the actual explicit CLI boundary.
        if argv[1:3] != ['--host', 'unix:///var/run/docker.sock'] or argv[3] != '--config':
            raise AssertionError(argv)
        if any(Path(argv[4]).iterdir()):
            raise AssertionError('nonempty Docker config')
        args=argv[5:]
        if args[:2] == ['image','inspect']:
            return subprocess.CompletedProcess(argv,0,json.dumps([{
                'Id':self.c.IMAGE_ID,'Os':'linux','Architecture':'amd64',
                'RepoDigests':['python@sha256:'+self.c.IMAGE_ID.split(':')[1]]}]),'')
        if args[0] == 'ps':
            if self.failure == 'collision':
                return subprocess.CompletedProcess(argv,0,'b'*64+'\n','')
            return subprocess.CompletedProcess(argv,0,'','')
        if args[0] == 'create':
            if self.failure == 'create125':
                return subprocess.CompletedProcess(argv,125,'','create failed')
            if self.failure == 'create-timeout':
                return SimpleNamespace(returncode=0,stdout='c'*64,stderr='',timed_out=True)
            if self.failure == 'malformed':
                return subprocess.CompletedProcess(argv,0,'abc\n','')
            context_path = next(v.split('=',1)[1] for i,v in enumerate(args)
                                if i and args[i-1]=='--label' and v.startswith('task9.context='))
            context = next(p for p in self.h.rglob('phase.json')
                           if self.c.sha256_file(p)==context_path)
            spec = self.c._spec_from_context(self.root, context)
            if list(argv) != self.c.build_create_argv(spec, argv[0], Path(argv[4])):
                raise AssertionError('create argv differs from bound actual spec')
            cid = f'{len(self.instances)+1:064x}'
            self.instances[cid] = (spec,context)
            return subprocess.CompletedProcess(argv,0,cid+'\n','')
        if args[0] == 'inspect':
            spec,ctx=self.instances[args[-1]]
            value=self.c._inspect_projection(spec,args[-1])
            if args[-1] in self.started:value['State']['Status']='exited'
            if self.failure == 'inspect': value['Config']['User']='0:0'
            return subprocess.CompletedProcess(argv,0,json.dumps([value]),'')
        if args[0] == 'start':
            self.started.add(args[-1])
            spec,ctx=self.instances[args[-1]]
            if self.failure == 'start-control': raise KeyboardInterrupt('fake start control')
            if self.failure == 'start-timeout':
                return SimpleNamespace(returncode=137,stdout='prefix',stderr='',timed_out=True,
                                       stdout_truncated=False,stderr_truncated=False)
            if self.failure == 'clock': self.now += 901
            with redirect_stderr(io.StringIO()) as captured:
                try: status=self.guest(spec,ctx)
                except SystemExit as error: status=int(error.code)
            self.last_guest_stderr=captured.getvalue()
            return subprocess.CompletedProcess(argv,status,'guest completed\n','')
        if args[0] == 'rm':
            if self.failure == 'cleanup': return subprocess.CompletedProcess(argv,1,'','fake cleanup failure')
            return subprocess.CompletedProcess(argv,0,args[-1]+'\n','')
        raise AssertionError(argv)

    def assert_bounded(self, argv, kw):
        assert kw['capture_output'] and kw['text'] and not kw['check']
        assert 0 < kw['timeout'] <= 900
        assert kw['output_limit_bytes'] == 65536
        assert kw['env'].get('HOME') == os.environ.get('HOME')
        assert not any(x.startswith('DOCKER_') for x in kw['env'])

    def guest(self, spec, context):
        from scripts.neural import task9_study_runtime as b
        from spire_learn.neural import config as config_module, runner
        from tests.unit.test_neural_study_runtime import (
            _StudyRuntime,_StudyCollector,_training_rollout,_evaluation_rollout)
        original_train, original_schedule, original_operation = (
            b.run_training_operation,b.run_selection_schedule,b.run_selection_operation)
        real_verify=self.c.verify_guest_context
        def mapper(value):
            s=str(value)
            for prefix,target in [('/learner',self.root),('/control',context.parent),
                                   ('/runtime/venv-cuda',self.root/self.c.VENV)]:
                if s==prefix or s.startswith(prefix+'/'):
                    return Path(str(target)+s[len(prefix):])
            return Path(value)
        def verify(**kwargs):
            self.context_calls.append(kwargs)
            return real_verify(**kwargs)
        def factory(config, **kwargs):
            self.factory_calls.append((spec.phase,kwargs))
            learned=kwargs['policy_kind']=='checkpoint'
            rollout=_training_rollout(config) if kwargs['purpose']=='train' else _evaluation_rollout(config,learned=learned)
            r=_StudyRuntime(config,_StudyCollector(config,[rollout]))
            r.collector.learned_combat_policy=learned
            r.policy_kind=kwargs['policy_kind'];r.action_selection=kwargs['action_selection']
            r.parent_checkpoint=kwargs['checkpoint']
            if self.thread_evidence and learned:
                r.task9_thread_settings={'intraop_threads':4,'interop_threads':1,'process_id':os.getpid(),
                    'verified_before_runtime_assembly':True,'verified_after_runtime_assembly':True}
            return r
        def op(*args,**kwargs):
            expire_here=args[2]==getattr(self,'selection_expiry_operation','selection-000')
            reserve=b._reserve_selection_operation
            def enter(*a,**k):
                result=reserve(*a,**k)
                if expire_here and getattr(self,'selection_expiry',None)=='after_entry':self.now=kwargs['deadline']
                return result
            def preflight(*a,**k):
                result=b._default_checkpoint_preflight(*a,**k)
                if expire_here and getattr(self,'selection_expiry',None)=='after_preflight':self.now=kwargs['deadline']
                return result
            def evaluate(*a,**k):
                getattr(self,'evaluation_calls',[]).append(a[0].mode)
                return runner.evaluate_policy(*a,**k)
            with patch.object(b,'_reserve_selection_operation',side_effect=enter):
                return original_operation(*args,**kwargs, clock=lambda:self.now,
                    dependencies=b.RuntimeDependencies(factory=factory,evaluate=evaluate,
                                                       checkpoint_preflight=preflight))
        def child(argv,**kw):
            self.children.append(argv)
            if getattr(self,'selection_expiry',None) is not None:
                target=next(r for r in self.study['selection']['schedule']
                            if r['operation_id']==self.selection_expiry_operation)
                if len(self.children)>target['index']+1:
                    raise AssertionError('deadline fixture exceeded its bounded operation prefix')
            getattr(self,'child_timeouts',[]).append(kw['timeout'])
            operation=argv[argv.index('--operation-id')+1]
            row=next(r for r in self.study['selection']['schedule'] if r['operation_id']==operation)
            if self.fresh_reference and self.reference_child_pid is None and row['classification']=='reference':
                # One real fresh process runs actual B main and the actual C
                # verifier. Only runtime and filesystem facts remain injected.
                harness='''import json,sys,os
from pathlib import Path
from dataclasses import replace
from tests.unit.test_neural_study_controls import ControlFixture,controls
f=ControlFixture.__new__(ControlFixture)
f.c=controls();f.root=Path(sys.argv[1]);f.now=float(sys.argv[3])
f.factory_calls=[];f.context_calls=[];f.children=[]
f.thread_evidence=True;f.guest_mutation=None
spec=f.c._spec_from_context(f.root,Path(sys.argv[2]))
spec=replace(spec,command=tuple(json.loads(sys.argv[4])))
code=f.guest(spec,Path(sys.argv[2]))
assert not any(n.split('.')[0] in {'torch','tensordict','torchrl'} for n in sys.modules)
print(json.dumps({'pid':os.getpid(),'calls':f.context_calls},default=str))
raise SystemExit(code)
'''
                process=self.c._bounded_subprocess_run([sys.executable,'-B','-c',harness,
                    str(self.root),str(context),repr(self.now),json.dumps(['-B',argv[1],*argv[2:]])],
                    capture_output=True,text=True,check=False,timeout=15,output_limit_bytes=65536,
                    env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),
                         'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
                if process.returncode!=0:raise AssertionError(process.stderr)
                observed=json.loads(process.stdout)
                self.reference_child_pid=observed['pid'];self.context_calls.extend(observed['calls'])
                return subprocess.CompletedProcess(argv,process.returncode,process.stdout,process.stderr)
            code=b.main(argv[2:])
            return subprocess.CompletedProcess(argv,code,'','')
        def schedule(*args,**kwargs):
            deps=kwargs.pop('dependencies');reserve=b._reserve_scheduled_operation;sync=b._fsync_directory
            def reserve_parent(*a,**k):
                result=reserve(*a,**k)
                if getattr(self,'selection_expiry',None)=='bookkeeping':self.now=kwargs['deadline']+20
                return result
            def fsync(directory):
                if (getattr(self,'reservation_fault',None)=='directory'
                    and Path(directory)==self.root/self.c.SELECTION/'control/operations'):
                    raise OSError('real reservation directory fsync failed')
                return sync(directory)
            with patch.object(b,'_reserve_scheduled_operation',side_effect=reserve_parent), \
                 patch.object(b,'_fsync_directory',side_effect=fsync):
                return original_schedule(*args,**kwargs,clock=lambda:self.now,
                    dependencies=replace(deps,child_process=child))
        def train(*args,**kwargs):
            return original_train(*args,**kwargs,clock=lambda:self.now,
                dependencies=b.RuntimeDependencies(factory=factory,train=runner.train))
        cmd=[str(mapper(arg)) if arg.startswith('/') else arg for arg in spec.command[2:]]
        facts={'mounts':[('/',True)]+[(dest,ro) for _,dest,ro in spec.mounts],
               'environment':dict(spec.environment), 'uid':1000,'gid':1000,
               'python':self.c.CONTAINER_PYTHON,'monotonic':self.now,'device_nodes':[] if spec.phase=='selection' else ['/dev/nvidia0']}
        if self.guest_mutation:self.guest_mutation(facts)
        with patch.object(config_module,'__file__',str(self.root/'src/spire_learn/neural/config.py')), \
             patch.object(self.c,'_guest_path',side_effect=mapper), \
             patch.object(self.c,'_guest_facts',return_value=facts), \
             patch.object(self.c,'_runtime_metadata',return_value={'torch':'2.14.1+cu130'}), \
             patch.object(self.c,'_verify_guest_runtime',return_value=None), \
             patch.object(self.c,'verify_guest_context',side_effect=verify), \
             patch.object(b,'run_training_operation',side_effect=train), \
             patch.object(b,'run_selection_schedule',side_effect=schedule), \
             patch.object(b,'run_selection_operation',side_effect=op):
            return b.main(cmd)

    def campaign(self, *, via_cli=False):
        from spire_learn.neural import config as config_module
        with patch.object(self.c,'_verify_runtime',side_effect=self.runtime), \
             patch.object(self.c,'_docker_binary',return_value='/usr/bin/docker-fixture'), \
             patch.object(self.c,'_recorded_path',side_effect=lambda p:str(self.root/p)), \
             patch.object(config_module,'__file__',str(self.root/'src/spire_learn/neural/config.py')):
            if via_cli:
                with redirect_stdout(io.StringIO()) as captured, patch.dict(os.environ,{'SPIRE_LEARN_TASK9_WATCHDOG':'4470+30'}):
                    status=self.c.main(['run','--root',str(self.root),'--snapshot',str(self.snapshot),
                        '--approval',str(self.approval),'--campaign-id','fixture-a','--control-root',str(self.h)],
                        dependencies=self.dependencies())
                result=json.loads((self.h/'campaign-result.json').read_text())
                assert status==(0 if result['status']=='complete' else 1),captured.getvalue()
                return result
            return self.c.run_campaign(self.root,snapshot_path=self.snapshot,
                approval_path=self.approval,campaign_id='fixture-a',control_root=self.h,
                dependencies=self.dependencies())


class NeuralStudyControlTests(unittest.TestCase):
    def test_public_api_and_exact_verifier_signature(self):
        import inspect
        c=controls()
        self.assertEqual(list(inspect.signature(c.verify_guest_context).parameters),
            ['root','study_path','phase_context','phase','deadline','policy_id',
             'operation_id','bindings_path','receipt_path','output_dir'])
        for name in ['capture_source_snapshot','validate_approval','run_campaign',
                     'build_create_argv','assert_phase_inspect','verify_training_bindings','main']:
            self.assertTrue(callable(getattr(c,name)))

    def test_full_five_phase_actual_guest_cli_and_192_children(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.fresh_reference=True;result=f.campaign(via_cli=True)
            self.assertEqual(result['status'],'complete',result)
            self.assertEqual([p['phase'] for p in result['phases']],list(f.c.PHASES))
            self.assertEqual([p['status'] for p in result['phases']],['complete']*5)
            self.assertEqual(len(f.children),192)
            self.assertIsNotNone(f.reference_child_pid)
            self.assertNotEqual(f.reference_child_pid,os.getpid())
            self.assertEqual(len(f.context_calls),197)
            self.assertEqual(len([a for a,k in f.commands if 'rm' in a]),5)
            selection=json.loads((f.root/f.c.SELECTION/'control/schedule-result.json').read_text())
            self.assertEqual(selection['reserved_step_slots'],49152)
            self.assertGreater(selection['progress_totals']['learned_decisions']['known_sum'],0)
            self.assertTrue(all(len(x)==10 for x in f.context_calls))
            deadline=f.context_calls[-1]['deadline']
            self.assertTrue(all(float(a[a.index('--deadline')+1])==deadline for a in f.children))
            bindings=json.loads((f.root/f.c.BINDINGS).read_text())
            self.assertEqual(len([p for p in bindings['policies'] if p['status']=='available']),4)
            f.c.verify_closed_inventory(f.h)
            self.assertTrue(all(k['env'].get('HOME')==os.environ.get('HOME') for _,k in f.commands))

    def test_invalid_approval_or_source_consumes_no_claim(self):
        for mutation in ('awaiting_approval','study','source','pin','review','snapshot-state'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d); a=json.loads(f.approval.read_text())
                if mutation=='awaiting_approval':a['state']=mutation
                if mutation=='study':a['study_raw_sha256']='0'*64
                if mutation=='source':(f.root/'src/spire_learn/__init__.py').write_text('changed\n')
                if mutation=='pin':a['runtime_pins']['torch']='2.14.1+cpu'
                if mutation=='review':f.review.write_text('changed')
                if mutation=='snapshot-state':
                    s=json.loads(f.snapshot.read_text());s['state']='source_only'
                    f.snapshot.write_text(json.dumps(s));a['source_snapshot_sha256']=f.c.sha256_file(f.snapshot)
                f.approval.write_text(json.dumps(a))
                with self.assertRaises(Exception):f.campaign()
                self.assertFalse((f.h/'campaign.claim.json').exists())
                self.assertFalse(any('create' in a for a,k in f.commands))

    def test_unowned_collision_create_failure_and_timeout_never_remove(self):
        for failure in ('collision','create125','create-timeout','malformed'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.failure=failure
                try: result=f.campaign()
                except Exception: result=None
                self.assertFalse(any('rm' in a for a,k in f.commands))
                if result:self.assertNotEqual(result['status'],'complete')

    def test_owned_failure_and_first_control_cleanup(self):
        for failure in ('inspect','start-timeout','cleanup','clock','start-control'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.failure=failure
                if failure=='start-control':
                    with self.assertRaises(KeyboardInterrupt):f.campaign()
                    result=json.loads((f.h/'campaign-result.json').read_text())
                else:result=f.campaign()
                self.assertNotEqual(result['status'],'complete')
                self.assertEqual(result['phases'][1]['status'],'not_started')
                # Wrong full spec still has matching ID/name/labels, permitting owned cleanup.
                self.assertLessEqual(len([a for a,k in f.commands if 'rm' in a]),1)
                self.assertTrue((f.h/'campaign-result.json').exists())

    def test_inspect_rejects_each_load_bearing_field(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout';f.campaign()
            cid,(spec,ctx)=next(iter(f.instances.items()))
            good=f.c._inspect_projection(spec,cid)
            mutations=[('Config','User','0:0'),('Config','Cmd',['false']),
                ('Config','Entrypoint',['/bin/sh']),('Config','Env',[]),
                ('HostConfig','Memory',1),('HostConfig','MemorySwap',-1),
                ('HostConfig','PidsLimit',0),('HostConfig','NanoCpus',1),
                ('HostConfig','Privileged',True),('HostConfig','ReadonlyRootfs',False),
                ('HostConfig','NetworkMode','host'),('HostConfig','CapDrop',[]),
                ('HostConfig','SecurityOpt',[]),('HostConfig','DeviceRequests',[]),
                ('HostConfig','Tmpfs',{'/tmp':'rw'}),('HostConfig','Devices',[{}]),
                ('HostConfig','DeviceCgroupRules',['c *:* rwm']),('HostConfig','GroupAdd',['0']),
                ('State','Running',True),('State','Status','running')]
            for obj,key,value in mutations:
                with self.subTest(field=key):
                    bad=deepcopy(good);bad[obj][key]=value
                    with self.assertRaises(ValueError):f.c.assert_phase_inspect(json.dumps([bad]),spec=spec,container_id=cid)
            for kind in ('missing','duplicate','extra','rw','source','destination','type'):
                bad=deepcopy(good);m=bad['Mounts']
                if kind=='missing':m.pop()
                if kind=='duplicate':m.append(deepcopy(m[0]))
                if kind=='extra':m.append({'Type':'bind','Source':'/secret','Destination':'/secret','RW':False})
                if kind=='rw':m[0]['RW']=True
                if kind=='source':m[0]['Source']='/wrong'
                if kind=='destination':m[0]['Destination']='/wrong'
                if kind=='type':m[0]['Type']='volume'
                with self.subTest(mount=kind),self.assertRaises(ValueError):
                    f.c.assert_phase_inspect(json.dumps([bad]),spec=spec,container_id=cid)

    def test_storage_symlink_accounting_and_strict_phase_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'file').write_bytes(b'x'*19);(p/'link').symlink_to('file')
            self.assertEqual(controls().tree_bytes(p,strict=False),23)
            with self.assertRaises(ValueError):controls().tree_bytes(p,strict=True)

    def test_strict_json_and_deadlines(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'j'
            for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
                p.write_text(raw)
                with self.assertRaises(ValueError):controls()._json(p)
            for value in (True,float('nan'),float('inf'),-1):
                with self.assertRaises(ValueError):controls()._finite_time(value)

    def test_approval_change_after_create_stops_and_records_failure(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);original=f.guest
            def guest(*args):
                result=original(*args);f.approval.write_text('{}');return result
            f.guest=guest
            result=f.campaign()
            self.assertEqual(result['phases'][0]['status'],'operational_failure')
            self.assertEqual(result['phases'][1]['status'],'not_started')

    def test_closed_inventory_binds_actual_phase_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout';f.campaign()
            output=f.root/f.c._phase_output('standalone-a')
            (output/'late-file').write_text('unrecorded phase output')
            with self.assertRaises(ValueError):f.c.verify_closed_inventory(f.h)

    def test_closure_publication_failure_cannot_return_success(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout'
            with patch.object(f.c,'_close_inventory',side_effect=OSError('final inventory failed')):
                with self.assertRaises(f.c.StudyControlError) as caught:f.campaign()
            self.assertTrue(any('final inventory failed' in n for n in caught.exception.__notes__))

    def test_resource_admission_and_gpu_floor_before_factory(self):
        for kind in ('disk','gpu'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d)
                original=f.dependencies
                if kind=='disk':
                    f.dependencies=lambda:replace(original(),disk_usage=lambda p:SimpleNamespace(free=29*1024**3))
                    with self.assertRaises(ValueError):f.campaign()
                    self.assertTrue((f.h/'preflight-commands.json').exists())
                    self.assertTrue((f.h/'campaign-result.json').exists())
                    self.assertFalse((f.h/'campaign.claim.json').exists())
                else:
                    original_run=f.run
                    def run(argv,**kw):
                        if argv[0]=='nvidia-smi':
                            return subprocess.CompletedProcess(argv,0,f.gpu_csv.rsplit(',',1)[0]+', 1\n','')
                        return original_run(argv,**kw)
                    f.run=run
                    self.assertEqual(f.campaign()['status'],'stopped')
                self.assertFalse(f.factory_calls)
                self.assertFalse(any('create' in a for a,k in f.commands))

    def test_actual_guest_verifier_rejects_environment_mount_and_device_mutations(self):
        mutations=[lambda f:f['environment'].__setitem__('HOME','/tmp'),
                   lambda f:f['environment'].__setitem__('DOCKER_HOST','tcp://remote'),
                   lambda f:f['mounts'].__setitem__(0,(f['mounts'][0][0],False)),
                   lambda f:f['mounts'].append(('/learner/rogue',True)),
                   lambda f:f['mounts'].append(('/secret',False)),
                   lambda f:f['mounts'].__setitem__(0,('/',False)),
                   lambda f:f.__setitem__('monotonic',10000.0)]
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.guest_mutation=mutate
                result=f.campaign()
                self.assertIn('phase-context verification failed',f.last_guest_stderr)
                self.assertEqual(result['status'],'stopped')
                self.assertFalse(f.factory_calls)
                self.assertEqual(result['phases'][1]['status'],'not_started')

    def test_later_writer_control_outranks_ordinary_operation_fault(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout';writer=f.c.write_json_exclusive
            def write(path,value):
                if Path(path).name=='standalone-a-commands.json':raise SystemExit('writer control')
                return writer(path,value)
            with patch.object(f.c,'write_json_exclusive',side_effect=write):
                with self.assertRaises(SystemExit) as caught:f.campaign()
            self.assertTrue(any('earlier' in x for x in caught.exception.__notes__))

    def test_missing_actual_neural_thread_evidence_stops_before_next_phase(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.thread_evidence=False;guest=f.guest
            def first(*args):
                code=guest(*args);f.failure='start-timeout';return code
            f.guest=first
            result=f.campaign()
            self.assertEqual(result['phases'][0]['status'],'operational_failure')
            self.assertEqual(result['phases'][1]['status'],'not_started')
            self.assertEqual(len(f.factory_calls),1)

    def test_approval_types_are_exact_json_not_python_equal(self):
        from spire_learn.neural import config as config_module
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);a=json.loads(f.approval.read_text())
            a['runtime_pins']['interop_threads']=True
            f.approval.write_text(json.dumps(a))
            with patch.object(config_module,'__file__',str(f.root/'src/spire_learn/neural/config.py')):
                with self.assertRaises(ValueError):f.c.validate_approval(f.root,f.snapshot,f.approval)

    def test_transport_success_cannot_omit_selection_schedule_identities(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);guest=f.guest
            def corrupt(spec,context):
                if spec.phase!='selection':return guest(spec,context)
                path=f.root/f.c.SELECTION/'control/schedule-result.json'
                path.write_text(json.dumps({'transport_exit_code':0,'operational_failure_count':0,
                    'unresolved_operation_count':0,'reserved_step_slots':0,'rows':[]}))
                return 0
            f.guest=corrupt;result=f.campaign()
            self.assertEqual(result['phases'][-1]['status'],'operational_failure')

    def test_cli_requires_launcher_watchdog_before_live_dispatch(self):
        with patch.dict(os.environ,clear=False):
            os.environ.pop('SPIRE_LEARN_TASK9_WATCHDOG',None)
            with patch.object(controls(),'run_campaign',return_value={'status':'stopped'}) as run:
                with redirect_stderr(io.StringIO()) as captured:
                    status=controls().main(['run','--root','/unused','--snapshot','/unused',
                        '--approval','/unused','--campaign-id','fixture-a','--control-root','/unused'])
                self.assertIn('requires the reviewed launcher watchdog',captured.getvalue())
                self.assertEqual(status,1)
                run.assert_not_called()

    def test_duplicate_campaign_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout';f.campaign()
            count=len(f.commands)
            with self.assertRaises(FileExistsError):f.campaign()
            self.assertEqual(len(f.commands),count)

    def test_stable_claim_blocks_alternate_evidence_directory_after_claimed_fault(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);run=f.run
            def low_gpu(argv,**kw):
                if argv[0]=='nvidia-smi':
                    return subprocess.CompletedProcess(argv,0,f.gpu_csv.rsplit(',',1)[0]+', 1\n','')
                return run(argv,**kw)
            f.run=low_gpu
            self.assertEqual(f.campaign()['status'],'stopped')
            slot=f.root/'artifacts/development/neural-combat/task-9/study-campaign-claims/fixture-a/campaign.claim.json'
            self.assertTrue(slot.is_file())
            frozen=slot.read_bytes();before=len(f.commands)
            self.assertEqual((f.h/'campaign.claim.json').read_bytes(),frozen)
            inventory=json.loads((f.h/'closed-inventory.json').read_text())
            self.assertIn(str(slot.parent),inventory['roots'])
            self.assertEqual(inventory['roots'].count(str(slot.parent)),1)
            f.h=f.records/'alternate-host-records'
            result=f.campaign()
            self.assertEqual(result['status'],'stopped')
            self.assertEqual(slot.read_bytes(),frozen)
            self.assertEqual(len(f.commands),before)
            self.assertFalse(f.factory_calls)
            self.assertFalse(any('create' in a for a,k in f.commands))
            self.assertEqual(result['campaign_claim_state'],'unclaimed')

    def test_stable_claim_concurrent_contenders_have_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);barrier=Barrier(2)
            def claim(i):
                barrier.wait()
                try:
                    return f.c._acquire_campaign_claim(f.root,'fixture-a',{'contender':i})
                except FileExistsError:return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(claim,(1,2)))
            self.assertEqual(sum(r is not None for r in results),1)
            slot=f.c._campaign_claim_path(f.root,'fixture-a')
            self.assertIn(json.loads(slot.read_text())['contender'],(1,2))

    def test_claim_fsync_order_covers_fresh_and_existing_domain(self):
        for preexisting in (False,True):
            with self.subTest(preexisting=preexisting),tempfile.TemporaryDirectory() as d:
                c=controls();root=Path(d);domain=root/c.CLAIMS_DOMAIN
                if preexisting:domain.mkdir(parents=True)
                slot=domain/'fixture-a/campaign.claim.json'
                parts=c.CLAIMS_DOMAIN.parts
                expected=[root.joinpath(*parts[:i]) for i in range(len(parts))]+[domain,slot,slot.parent]
                events=[];sync=os.fsync
                def fsync(fd):
                    events.append(Path(os.readlink(f'/proc/self/fd/{fd}')))
                    return sync(fd)
                with patch.object(c.os,'fsync',side_effect=fsync):
                    actual=c._acquire_campaign_claim(root,'fixture-a',{'fixture':'durable'})
                self.assertEqual(actual,slot)
                self.assertEqual(events,expected)
                self.assertEqual(json.loads(slot.read_text()),{'fixture':'durable'})

    def test_each_claim_directory_or_file_fsync_failure_retains_consumed_slot(self):
        c=controls();parts=c.CLAIMS_DOMAIN.parts
        for failure_index in range(len(parts)+3):
            with self.subTest(failure_index=failure_index),tempfile.TemporaryDirectory() as d:
                root=Path(d);domain=root/c.CLAIMS_DOMAIN;slot=domain/'fixture-a/campaign.claim.json'
                sequence=[root.joinpath(*parts[:i]) for i in range(len(parts))]+[domain,slot,slot.parent]
                fail_at=sequence[failure_index];events=[];sync=os.fsync
                def fsync(fd):
                    path=Path(os.readlink(f'/proc/self/fd/{fd}'));events.append(path)
                    if path==fail_at:raise OSError('injected claim fsync failure')
                    return sync(fd)
                with patch.object(c.os,'fsync',side_effect=fsync):
                    with self.assertRaisesRegex(OSError,'claim fsync failure'):
                        c._acquire_campaign_claim(root,'fixture-a',{'fixture':'never forwarded'})
                self.assertEqual(events,sequence[:failure_index+1])
                self.assertTrue(slot.parent.is_dir())
                self.assertEqual(slot.exists(),failure_index>=len(parts)+1)
                before=list(slot.parent.iterdir())
                with self.assertRaises(FileExistsError):
                    c._acquire_campaign_claim(root,'fixture-a',{'fixture':'replay'})
                self.assertEqual(list(slot.parent.iterdir()),before)

    def test_claim_ancestor_and_domain_sync_faults_block_phase_dispatch(self):
        for component in ('ancestor','domain'):
            with self.subTest(component=component),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.failure='start-timeout'
                domain=f.root/f.c.CLAIMS_DOMAIN
                target=domain if component=='domain' else domain.parent
                sync=os.fsync
                def fsync(fd):
                    if Path(os.readlink(f'/proc/self/fd/{fd}'))==target:
                        raise OSError('claim path durability failed')
                    return sync(fd)
                with patch.object(f.c.os,'fsync',side_effect=fsync):result=f.campaign()
                self.assertEqual(result['status'],'stopped')
                self.assertEqual(result['campaign_claim_state'],'publication_attempted')
                self.assertFalse(f.factory_calls)
                self.assertFalse(any('create' in argv or 'rm' in argv or argv[0]=='nvidia-smi' for argv,_ in f.commands))
                self.assertTrue((domain/'fixture-a').is_dir())
                self.assertFalse((domain/'fixture-a/campaign.claim.json').exists())
                count=len(f.commands);f.h=f.records/'retry-after-sync-fault'
                self.assertEqual(f.campaign()['campaign_claim_state'],'unclaimed')
                self.assertEqual(len(f.commands),count)

    def test_concurrent_claim_domain_initializers_each_sync_the_ancestor_chain(self):
        with tempfile.TemporaryDirectory() as d:
            c=controls();root=Path(d);barrier=Barrier(2);events={};sync=os.fsync
            def fsync(fd):
                events.setdefault(get_ident(),[]).append(Path(os.readlink(f'/proc/self/fd/{fd}')))
                return sync(fd)
            def claim(campaign_id):
                barrier.wait();path=c._acquire_campaign_claim(root,campaign_id,{'campaign':campaign_id})
                return get_ident(),path
            with patch.object(c.os,'fsync',side_effect=fsync),ThreadPoolExecutor(max_workers=2) as pool:
                returned=list(pool.map(claim,('fixture-a','fixture-b')))
            parts=c.CLAIMS_DOMAIN.parts
            prefix=[root.joinpath(*parts[:i]) for i in range(len(parts))]+[root/c.CLAIMS_DOMAIN]
            self.assertEqual(len(returned),2)
            for tid,path in returned:
                self.assertEqual(events[tid],prefix+[path,path.parent])

    def test_initial_preflight_faults_retain_plan_logs_and_unowned_state(self):
        for kind in ('collision','query_failure','query_timeout','git_failure','staging','qualification'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);run=f.run
                def fail(argv,**kw):
                    if (kind=='git_failure' and argv[0]=='git') or (kind in ('query_failure','query_timeout') and 'ps' in argv):
                        return SimpleNamespace(returncode=125 if kind!='query_timeout' else 0,
                            stdout='bounded preflight prefix',stderr='preflight diagnostic',
                            timed_out=kind=='query_timeout')
                    return run(argv,**kw)
                f.run=fail
                if kind=='collision':f.failure='collision'
                def staging(*args):raise OSError('staging failure')
                def qualification(*args):raise OSError('qualified input failure')
                with patch.object(f.c,'_stage',side_effect=staging) if kind=='staging' else patch.object(f,'runtime',side_effect=qualification) if kind=='qualification' else patch.object(f,'now',f.now):
                    result=f.campaign()
                self.assertEqual(result['status'],'stopped')
                self.assertEqual(len(result['phases']),5)
                self.assertEqual(len(result['selection_rows']),192)
                self.assertTrue(all(p['status']=='not_started' and p['cleanup']=='not_owned' for p in result['phases']))
                self.assertEqual(result['campaign_claim_state'],'unclaimed')
                self.assertFalse(f.factory_calls)
                self.assertFalse(any('create' in a or 'rm' in a for a,k in f.commands))
                log=json.loads((f.h/'preflight-commands.json').read_text())
                self.assertTrue(log['commands'])
                if kind.startswith('query') or kind=='git_failure':
                    self.assertIn('bounded preflight prefix',json.dumps(log))
                f.c.verify_closed_inventory(f.h)

    def test_final_publication_clock_and_storage_are_included_in_closure(self):
        for kind in ('writer_clock','inventory_clock','storage'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.failure='start-timeout'
                writer=f.c.write_json_exclusive;close=f.c._close_inventory;tree=f.c.tree_bytes
                def write(path,value):
                    result=writer(path,value)
                    if kind=='writer_clock' and Path(path).name=='campaign-result.json':f.now+=4501
                    return result
                def inventory(h):
                    result=close(h)
                    if kind=='inventory_clock':f.now+=4501
                    return result
                def size(path,*,strict):
                    if kind=='storage' and Path(path)==f.h and (f.h/'closed-inventory.json').exists():return 7*1024**3
                    return tree(path,strict=strict)
                with patch.object(f.c,'write_json_exclusive',side_effect=write), \
                     patch.object(f.c,'_close_inventory',side_effect=inventory), \
                     patch.object(f.c,'tree_bytes',side_effect=size):
                    with self.assertRaises(f.c.StudyControlError) as caught:f.campaign()
                self.assertIn('timeout/output limit',str(caught.exception))
                marker=json.loads((f.h/'closure-failure.json').read_text())
                self.assertEqual(marker['status'],'stopped')
                self.assertTrue(marker['error']['notes'])
                with self.assertRaises(ValueError):f.c.verify_closed_inventory(f.h)

    def test_reservation_audit_does_not_resolve_partial_receipt_or_lose_unreadable_prefix(self):
        for kind in ('partial_receipt','unreadable_reservation'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);row=f.study['selection']['schedule'][0]
                base=f.root/f.c.SELECTION/'control/operations';base.mkdir(parents=True)
                reservation=base/(row['operation_id']+'.reservation.json')
                reservation.write_text(json.dumps({'operation_id':row['operation_id'],'reserved_step_slots':256}))
                (base/(row['operation_id']+'.json')).write_text('{partial')
                sha=f.c.sha256_file
                def digest(path):
                    if kind=='unreadable_reservation' and Path(path)==reservation:raise PermissionError('unreadable reservation')
                    return sha(path)
                with patch.object(f.c,'sha256_file',side_effect=digest):audit=f.c._reservation_audit(f.root,f.study)
                self.assertEqual(len(audit),1)
                self.assertTrue(audit[0]['unresolved'])
                self.assertEqual(audit[0]['reserved_step_slots_upper_bound'],256)
                self.assertTrue(audit[0]['diagnostic'])

    def test_final_rows_use_conservative_audit_for_corrupt_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.failure='start-timeout';run=f.run
            def corrupt(argv,**kw):
                if 'start' in argv:
                    base=f.root/f.c.SELECTION/'control/operations';base.mkdir(parents=True)
                    (base/'selection-000.reservation.json').write_text('{}')
                    (base/'selection-000.json').write_text('{broken')
                return run(argv,**kw)
            f.run=corrupt;result=f.campaign()
            self.assertEqual(result['selection_rows'][0]['status'],'unresolved')
            self.assertEqual(result['selection_reservation_audit'][0]['reserved_step_slots_upper_bound'],256)
            self.assertEqual(result['phases'][0]['error']['type'],'StudyControlError')

    def test_training_driver_files_are_narrow_regular_and_readonly(self):
        destinations=('/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07',
                      '/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1','/usr/bin/nvidia-smi')
        for dest in destinations:
            with self.subTest(destination=dest),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);guest=f.guest
                def mutation(facts):
                    facts['mounts'].append((dest,True));facts['regular_mount_files']={dest}
                f.guest_mutation=mutation
                def first(*args):
                    result=guest(*args);f.failure='start-timeout';return result
                f.guest=first;result=f.campaign()
                self.assertEqual(result['phases'][0]['status'],'complete',result)
        c=controls()
        for phase,dest,readonly,regular in (
            ('selection',destinations[0],True,True),('standalone-a',destinations[0],False,True),
            ('standalone-a',destinations[0],True,False),
            ('standalone-a','/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.08',True,True),
            ('standalone-a','/usr/bin/unrelated',True,True),('standalone-a','/usr',True,True)):
            with self.subTest(phase=phase,destination=dest):
                self.assertFalse(c._driver_mount_allowed(phase,dest,readonly,{dest} if regular else set()))

    def test_production_guest_facts_establish_regular_driver_file_type(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);file=p/'libcuda.so.595.91.07';file.write_text('synthetic regular file')
            link=p/'alias';link.symlink_to(file);directory=p/'directory';directory.mkdir()
            raw=''.join(f'1 0 0:1 / {path} ro - ext4 root ro\n' for path in (file,link,directory))
            read=Path.read_text
            def mountinfo(path,*args,**kwargs):
                return raw if str(path)=='/proc/self/mountinfo' else read(path,*args,**kwargs)
            with patch.object(Path,'read_text',new=mountinfo):facts=controls()._guest_facts()
            self.assertIn(str(file),facts['regular_mount_files'])
            self.assertNotIn(str(link),facts['regular_mount_files'])
            self.assertNotIn(str(directory),facts['regular_mount_files'])

    def test_real_deadline_receipts_close_healthy_partial_campaign_without_factory(self):
        cases=[(mode,'learned',boundary) for mode in ('standalone','run')
               for boundary in ('after_entry','after_preflight')]
        cases += [(mode,'reference','after_entry') for mode in ('standalone','run')]
        cases += [('standalone','learned','bookkeeping')]
        for mode,kind,boundary in cases:
            with self.subTest(mode=mode,kind=kind,boundary=boundary),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);f.selection_expiry=boundary
                target=next(row for row in f.study['selection']['schedule']
                    if row['mode']==mode and (row['classification']=='reference')==(kind=='reference'))
                f.selection_expiry_operation=target['operation_id'];index=target['index']
                result=f.campaign()
                self.assertEqual(result['status'],'complete',result.get('error'))
                self.assertEqual(result['closure_state'],'verified')
                self.assertEqual(len(f.children),index+1)
                self.assertEqual(len(f.factory_calls),4+index)
                self.assertEqual(f.evaluation_calls,[row['mode'] for row in f.study['selection']['schedule'][:index]])
                self.assertEqual(f.child_timeouts,[40.0] if boundary=='bookkeeping' else [660.0]*(index+1))
                output=f.root/f.c.SELECTION
                schedule=json.loads((output/'control/schedule-result.json').read_text())
                receipt_path=output/'control/operations'/f"{target['operation_id']}.json"
                receipt=json.loads(receipt_path.read_text())
                self.assertEqual(schedule['reserved_step_slots'],256*(index+1))
                self.assertEqual(schedule['complete_operation_count'],index)
                self.assertEqual(schedule['not_started_operation_count'],192-index)
                self.assertEqual(schedule['transport_exit_code'],0)
                self.assertTrue(all(row['status']=='complete' for row in schedule['rows'][:index]))
                self.assertTrue(all(row['status']=='not_started' for row in schedule['rows'][index:]))
                if mode=='run':
                    self.assertEqual(schedule['rows'][index-1]['mode'],'standalone')
                    self.assertEqual(schedule['rows'][index-1]['status'],'complete')
                self.assertEqual(receipt['reason'],'shared_soft_deadline_exhausted_before_factory')
                self.assertIs(receipt['attempted_effects'],False)
                self.assertTrue(all(v is None for v in receipt['counts'].values()))
                for name in ('report','lifecycle','summary_path','lifecycle_path','runner_exit_code','run_id'):
                    self.assertIsNone(receipt[name])
                self.assertEqual(receipt['close_status'],'runner_owned_or_unknown')
                audit=result['selection_reservation_audit'][index]
                self.assertEqual(audit['operation_id'],target['operation_id'])
                self.assertEqual(audit['reserved_step_slots_upper_bound'],256)
                self.assertFalse(audit['unresolved'])
                self.assertEqual(result['selection_rows'][index]['status'],'not_started')
                self.assertEqual(receipt['checkpoint']['before'] is not None,boundary=='after_preflight')
                raw_case=json.loads((f.root/target['one_case_manifest_path']).read_text())['cases'][0]
                if mode=='run':
                    self.assertNotIn('entry_id',raw_case)
                    self.assertIsNone(receipt['seed_evidence']['entry_id'])
                else:
                    self.assertEqual(receipt['seed_evidence']['entry_id'],raw_case['entry_id'])
                f.c.verify_closed_inventory(f.h)
                if mode=='run':
                    bad=deepcopy(receipt);bad['seed_evidence']['entry_id']='invented-run-entry'
                    receipt_path.write_text(json.dumps(bad));returned=deepcopy(schedule)
                    returned['rows'][index]['receipt_raw_sha256']=f.c.sha256_file(receipt_path)
                    bad_audit=f.c._reservation_audit(f.root,f.study,reconciled_schedule=returned)[index]
                    self.assertTrue(bad_audit['unresolved'])
                    self.assertEqual(bad_audit['reserved_step_slots_upper_bound'],256)

    def test_standalone_deadline_receipt_keeps_required_entry_validation(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.selection_expiry='after_entry';result=f.campaign()
            self.assertEqual(result['status'],'complete')
            output=f.root/f.c.SELECTION;target=f.study['selection']['schedule'][0]
            receipt_path=output/'control/operations/selection-000.json'
            receipt=json.loads(receipt_path.read_text())
            schedule=json.loads((output/'control/schedule-result.json').read_text())
            manifest=f.root/target['one_case_manifest_path'];original=manifest.read_bytes()
            for change in ('missing_required_raw','incorrect_required_raw','missing_receipt','incorrect_receipt'):
                with self.subTest(change=change):
                    raw=json.loads(original);r=deepcopy(receipt);returned=deepcopy(schedule)
                    if change=='missing_required_raw':del raw['cases'][0]['entry_id']
                    if change=='incorrect_required_raw':
                        raw['cases'][0]['entry_id']='wrong-entry-id'
                        r['seed_evidence']['entry_id']='wrong-entry-id'
                    if change=='missing_receipt':del r['seed_evidence']['entry_id']
                    if change=='incorrect_receipt':r['seed_evidence']['entry_id']=None
                    manifest.write_text(json.dumps(raw));receipt_path.write_text(json.dumps(r))
                    returned['rows'][0]['receipt_raw_sha256']=f.c.sha256_file(receipt_path)
                    audit=f.c._reservation_audit(f.root,f.study,reconciled_schedule=returned)
                    self.assertTrue(audit[0]['unresolved'])
                    self.assertEqual(audit[0]['reserved_step_slots_upper_bound'],256)

    def test_reserved_deadline_receipt_rejects_tampered_or_untrusted_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.selection_expiry='after_entry';f.campaign()
            output=f.root/f.c.SELECTION
            context=f.h/'selection/phase.json';spec=f.c._spec_from_context(f.root,context)
            schedule_path=output/'control/schedule-result.json'
            receipt_path=output/'control/operations/selection-000.json'
            entry_path=output/'control/operations/selection-000.entered.json'
            schedule=json.loads(schedule_path.read_text());receipt=json.loads(receipt_path.read_text())
            entry=entry_path.read_bytes()
            reservation_path=output/'control/operations/selection-000.reservation.json'
            original_reservation=reservation_path.read_bytes()
            for change in ('missing_entry','mismatched_entry','effects_none','effects_true','effects_zero',
                           'wrong_reason','wrong_binding','wrong_config','wrong_reservation_binding',
                           'wrong_reservation_mode','fabricated_counts','runner_artifact','corrupt'):
                with self.subTest(change=change):
                    r=deepcopy(receipt);plan=deepcopy(schedule);entry_path.write_bytes(entry)
                    reservation_path.write_bytes(original_reservation)
                    if change=='missing_entry':entry_path.unlink()
                    if change=='mismatched_entry':
                        e=json.loads(entry);e['reservation_raw_sha256']='0'*64;entry_path.write_text(json.dumps(e))
                    if change.startswith('effects_'):r['attempted_effects']={'effects_none':None,'effects_true':True,'effects_zero':0}[change]
                    if change=='wrong_reason':r['reason']='arbitrary_not_started'
                    if change=='wrong_binding':r['binding_raw_sha256']='0'*64
                    if change=='wrong_config':r['config']['canonical_sha256']='0'*64
                    if change.startswith('wrong_reservation_'):
                        reserve=json.loads(original_reservation)
                        reserve['binding_raw_sha256' if change.endswith('binding') else 'mode']='0'*64 if change.endswith('binding') else 'run'
                        reservation_path.write_text(json.dumps(reserve));digest=f.c.sha256_file(reservation_path)
                        e=json.loads(entry);e['reservation_raw_sha256']=digest;entry_path.write_text(json.dumps(e))
                        r['reservation']['raw_sha256']=digest;plan['rows'][0]['reservation_raw_sha256']=digest
                    if change=='fabricated_counts':r['counts']['attempted_environment_slots']=0
                    if change=='runner_artifact':r['report']={}
                    receipt_path.write_text('{partial' if change=='corrupt' else json.dumps(r))
                    plan['rows'][0]['receipt_raw_sha256']=f.c.sha256_file(receipt_path)
                    schedule_path.write_text(json.dumps(plan))
                    try:verified=f.c._verify_schedule(f.root,f.study,spec)
                    except (ValueError,KeyError,TypeError):continue
                    audit=f.c._reservation_audit(f.root,f.study,reconciled_schedule=verified)
                    self.assertTrue(audit[0]['unresolved'])
                    self.assertEqual(audit[0]['reserved_step_slots_upper_bound'],256)

    def test_real_ambiguous_reservation_producer_and_campaign_audit_agree(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);f.reservation_fault='directory';result=f.campaign()
            schedule=json.loads((f.root/f.c.SELECTION/'control/schedule-result.json').read_text())
            self.assertEqual(result['status'],'stopped')
            self.assertFalse(f.children)
            self.assertEqual(len(f.factory_calls),4)
            self.assertEqual(schedule['reserved_step_slots'],256)
            self.assertEqual(schedule['attempted_operation_count'],0)
            self.assertEqual(schedule['rows'][0]['dispatch_state'],'not_dispatched')
            self.assertIs(schedule['rows'][0]['attempted_effects'],False)
            self.assertEqual(schedule['error']['message'],'real reservation directory fsync failed')
            self.assertEqual(result['selection_reservation_audit'][0]['reserved_step_slots_upper_bound'],256)
            self.assertTrue(result['selection_reservation_audit'][0]['unresolved'])
            self.assertEqual(result['selection_rows'][0]['status'],'unresolved')
            f.c.verify_closed_inventory(f.h)

    def test_actual_direct_cli_verifies_clean_copy_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d);env=dict(os.environ);env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,'-B',str(f.root/'scripts/neural/task9_study_controls.py'),
                'verify','--root',str(f.root),'--snapshot',str(f.snapshot),'--approval',str(f.approval)],
                cwd=d,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_shell_syntax_and_fresh_optional_import_guard_in_clean_copy(self):
        with tempfile.TemporaryDirectory() as d:
            f=ControlFixture(d)
            result=subprocess.run(['bash','-n',str(f.root/'scripts/neural/run-task9-study.sh')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            guard='''import importlib.abc,sys,runpy
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','tensordict','torchrl','gymnasium','spire_lab'}:
   attempts.append(fullname);raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
sys.argv=['task9_study_controls.py','describe']
try:runpy.run_path('scripts/neural/task9_study_controls.py',run_name='__main__')
except SystemExit as e:assert e.code==0
from scripts.neural.task9_study_controls import _study
from pathlib import Path
assert len(_study(Path.cwd())['selection']['schedule'])==192
assert not attempts
assert not any(n.split('.')[0] in {'torch','numpy','tensordict','torchrl','gymnasium','spire_lab'} for n in sys.modules)
'''
            result=subprocess.run([sys.executable,'-B','-c',guard],cwd=f.root,
                env={**os.environ,'PYTHONPATH':'src:.'},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('prepared_not_released',result.stdout)


if __name__=='__main__':unittest.main()
