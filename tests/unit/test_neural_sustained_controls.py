"""Sustained strict containment and scheduler tests use fake boundaries only."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


class FakeDocker:
    def __init__(self,spec,*,failure=None):
        self.spec=spec;self.failure=failure;self.calls=[];self.cid='d'*64;self.started=False
    def run(self,argv,**kwargs):
        from scripts.neural import task9_study_controls as low
        self.calls.append(list(argv));args=argv[5:];command=args[0];stdout='';stderr=''
        if command=='ps' and self.failure=='existing':stdout='preexisting\n'
        if command=='create':
            if self.failure=='create-timeout':raise TimeoutError('create timeout')
            stdout='short\n' if self.failure=='malformed' else self.cid+'\n'
        if command=='inspect':
            projection=low._inspect_projection(self.spec,self.cid)
            if self.started:projection['State']['Status']='exited'
            if self.failure=='wrong-owner':projection['Config']['Labels']={'unexpected':'owner'}
            stdout=json.dumps([projection])
        if command=='start':
            self.started=True
            if self.failure in ('start','cleanup-term'):raise RuntimeError('primary start fault')
            from scripts.neural.sustained_controls import MARKER
            stderr=MARKER+json.dumps({'context_sha256':self.spec.context_sha256,'job_id':self.spec.phase,'runtime_class':self.spec.runtime_class})+'\n'
        if command=='rm' and self.failure=='cleanup-term':raise KeyboardInterrupt('TERM cleanup')
        return SimpleNamespace(returncode=0,stdout=stdout,stderr=stderr)


def spec():
    from scripts.neural.task9_study_controls import PhaseSpec
    return PhaseSpec('run-1-segment-000','new-job','sustained','a'*64,'b'*64,'c'*64,
        20.,60.,'output',('-B','entry.py'),(),(('/tmp/owned','/output',False),),(),runtime_class='cpu')


class ControlTests(unittest.TestCase):
    def test_truthful_runtime_class_controls_gpu_request(self):
        from scripts.neural import task9_study_controls as old
        spec=old.PhaseSpec('run-1-segment-000','new-job','sustained','a'*64,'b'*64,'c'*64,
            20.,30.,'output',('-B','entry.py'),(),(('/tmp/owned','/output',False),),(),runtime_class='cpu')
        with tempfile.TemporaryDirectory() as directory:
            argv=old.build_create_argv(spec,'docker',Path(directory))
            self.assertNotIn('--gpus',argv)
            projection=old._inspect_projection(spec,'d'*64)
            old.assert_phase_inspect(json.dumps([projection]),spec=spec,container_id='d'*64)
            gpu=replace(spec,runtime_class='gpu')
            self.assertIn('--gpus',old.build_create_argv(gpu,'docker',Path(directory)))
            with self.assertRaises(ValueError):replace(spec,runtime_class='other')
            with self.assertRaises(ValueError):replace(spec,runtime_class=None)

    def test_cpu_cdi_class_is_independent_of_truthful_job_id(self):
        from scripts.neural import task9_nvidia_cdi as cdi
        from tests.unit.test_neural_study_cdi import ROOT
        profile=cdi.load_runtime_profile(ROOT)
        env={'NVIDIA_VISIBLE_DEVICES':'none','CUDA_VISIBLE_DEVICES':'','NVIDIA_DRIVER_CAPABILITIES':''}
        facts={'mounts':[('/',True)],'device_nodes':[]}
        accepted=cdi.validate_runtime_class(profile,runtime_class='cpu',requested_environment=env,actual_environment=env,facts=facts)
        self.assertEqual(accepted['mode'],'legacy')
        with self.assertRaises(ValueError):cdi.validate_runtime_class(profile,runtime_class='cpu',requested_environment=env,actual_environment=env,facts={**facts,'device_nodes':['/dev/nvidia0']})

    def test_budget_charges_intervals_once_and_never_transfers_buckets(self):
        c=importlib.import_module('scripts.neural.sustained_controls')
        clock=[0.]
        b=c.BudgetAccount({k:0. for k in c.BUCKETS},clock=lambda:clock[0])
        clock[0]=1.;b.switch('run-1');clock[0]=601.;b.switch('shared')
        self.assertEqual(b.charged['shared'],1.)
        self.assertEqual(b.charged['run-1'],600.)
        clock[0]=602.;b.close();b.close()
        self.assertEqual(sum(b.charged.values()),602.)
        self.assertLessEqual(b.remaining('run-1'),5400.)

    def test_outer_stage_attachment_carries_only_nonoverlapping_residual_to_shared(self):
        from scripts.neural import sustained_controls as c
        incoming={k:(.09155025496147573 if k=='shared' else 0.) for k in c.BUCKETS}
        charged=dict(incoming);charged['shared']+=5.;charged['profile']=5.
        seal={'stage':'profile','campaign_id':'test','stage_elapsed_seconds':10.,'charged_seconds':charged,'incoming_charged_seconds':incoming}
        outer={'schema':c.OUTER_SCHEMA,'stage':'profile','campaign_id':'test','seal_sha256':'a'*64,'exit_code':0,
            'enclosing_elapsed_seconds':12.,'accounted_interval_seconds':10.,'residual_shared_seconds':2.}
        result=c.reconcile_outer_accounting(seal,outer)
        self.assertEqual(result['shared'],incoming['shared']+7.)
        self.assertEqual(result['profile'],5.)
        with self.assertRaises(ValueError):c.reconcile_outer_accounting(seal,{**outer,'residual_shared_seconds':12.})
        with self.assertRaises(ValueError):c.reconcile_outer_accounting(seal,{**outer,'enclosing_elapsed_seconds':9.})
        with self.assertRaises(ValueError):c.reconcile_outer_accounting(seal,{**outer,'exit_code':False})

    def test_failed_create_never_removes_by_name_or_partial_id(self):
        from scripts.neural import sustained_controls as c
        for failure in ('existing','malformed','create-timeout','wrong-owner'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                docker=FakeDocker(spec(),failure=failure)
                deps=SimpleNamespace(clock=lambda:0.,run=docker.run)
                with self.assertRaises((RuntimeError,ValueError,TimeoutError)):
                    c.execute_job(docker.spec,deps=deps,client=Path(directory),docker='docker',logs=[],progress=lambda:None)
                self.assertFalse(any('rm' in call for call in docker.calls))

    def test_cleanup_full_id_once_preserves_primary_on_term(self):
        from scripts.neural import sustained_controls as c
        for failure in ('start','cleanup-term'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                docker=FakeDocker(spec(),failure=failure)
                deps=SimpleNamespace(clock=lambda:0.,run=docker.run)
                with self.assertRaisesRegex(RuntimeError,'primary start fault'):
                    c.execute_job(docker.spec,deps=deps,client=Path(directory),docker='docker',logs=[],progress=lambda:None)
                remove=[call for call in docker.calls if 'rm' in call]
                self.assertEqual(len(remove),1)
                self.assertEqual(remove[0][-1],docker.cid)
                self.assertNotIn(docker.spec.name,remove[0])

    def test_strict_inspect_rejects_every_material_containment_drift(self):
        from copy import deepcopy
        from scripts.neural import task9_study_controls as low
        selected=spec();base=low._inspect_projection(selected,'d'*64)
        edits=[('Image','other'),('Config.Env',['LEAK=1']),('Config.Cmd',['other.py']),
               ('HostConfig.NetworkMode','host'),('HostConfig.ReadonlyRootfs',False),
               ('HostConfig.Memory',16),('HostConfig.PidsLimit',1024),
               ('HostConfig.DeviceRequests',[{'Capabilities':[['gpu']]}]),
               ('Mounts',[{'Type':'bind','Source':'/home','Destination':'/output','RW':True}])]
        for key,value in edits:
            with self.subTest(key=key):
                actual=deepcopy(base);parts=key.split('.');target=actual
                for component in parts[:-1]:target=target[component]
                target[parts[-1]]=value
                with self.assertRaises(ValueError):low.assert_phase_inspect(json.dumps([actual]),spec=selected,container_id='d'*64)

    def test_source_failure_precedes_any_stage_claim_even_with_retained_outputs(self):
        from scripts.neural import sustained_controls as c
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);retained=root/c.DOMAIN/'retained';retained.mkdir(parents=True)
            marker=retained/'closed-stage.json';marker.write_text('user evidence')
            before=list(root.rglob('*'))
            with self.assertRaises((ValueError,FileNotFoundError)):
                c.run_campaign(root,campaign_path=Path('bad'),source_snapshot=Path('bad'),release_path=Path('bad'),stage='profile')
            self.assertEqual(list(root.rglob('*')),before);self.assertEqual(marker.read_text(),'user evidence')

    def test_callback_validator_rejects_omitted_and_tampered_counters(self):
        from scripts.neural import sustained_controls as c,sustained_runtime as r
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);journal=r.Journal(output)
            ledger=r.EffectLedger('actual',journal,deadline=10.,clock=lambda:0.,limits={k:4 for k in r.EFFECT_KINDS})
            ledger.call('environment_attempts',2,'native-step',lambda:None);journal.flush()
            receipt={'journal':journal.parts,'effects':ledger.snapshot(),'effect_categories':ledger.categories}
            c.validate_effects({'reservation':{k:4 for k in r.EFFECT_KINDS}},receipt,output)
            receipt['effects']['environment_attempts']['delivered']=1
            with self.assertRaises(ValueError):c.validate_effects({'reservation':{k:4 for k in r.EFFECT_KINDS}},receipt,output)

    def test_mode_balanced_queue_preserves_all_704_fixed_operations(self):
        from scripts.neural import sustained_controls as c
        from tests.unit.test_neural_sustained_runtime import ROOT
        campaign=json.loads((ROOT/c.CAMPAIGN_PATH).read_text())
        queue=c.evaluation_dispatch(campaign,{})
        self.assertEqual(len(queue),704)
        self.assertEqual(len({r['operation_id'] for r in queue}),704)
        self.assertEqual([queue[i]['mode'] for i in (0,4,8,12)],['standalone','run','standalone','run'])
        self.assertTrue(all(r['checkpoint_slot'] in ('anchor','reference') for r in queue[:256]))

    def test_term_during_cleanup_defers_until_owned_removal_entered_once(self):
        import signal
        from scripts.neural import sustained_controls as c
        with tempfile.TemporaryDirectory() as directory,c.signal_scope():
            docker=FakeDocker(spec());original=docker.run
            def run(argv,**kwargs):
                if 'rm' in argv:signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                return original(argv,**kwargs)
            with self.assertRaises(BaseException):
                c.execute_job(docker.spec,deps=SimpleNamespace(clock=lambda:0.,run=run),client=Path(directory),docker='docker',logs=[],progress=lambda:None)
            self.assertEqual(len([call for call in docker.calls if 'rm' in call]),1)


class SchedulerTests(unittest.TestCase):
    def test_stage_ancestor_sync_failure_consumes_slot_before_any_docker_effect(self):
        from unittest.mock import Mock
        from scripts.neural import sustained_controls as c
        from tests.unit.test_neural_sustained_runtime import ROOT
        campaign=json.loads((ROOT/c.CAMPAIGN_PATH).read_text())
        verified={'campaign':campaign,'snapshot':{'files':{}},'release':{'campaign_id':'isolated','predecessor':None,
            'initial_accounting':{'path':c.INITIAL.as_posix(),'sha256':c.INITIAL_SHA256}},
            'source_sha256':'a'*64,'release_sha256':'b'*64,'charged':{k:0. for k in c.BUCKETS}}
        run=Mock(side_effect=RuntimeError('unexpected Docker dispatch'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(c,'validate_release',return_value=verified),patch.object(c.low,'_verify_runtime'),patch.object(c.low,'_docker_binary',return_value='docker'),patch.object(c.low,'_fsync_claim_directory',side_effect=OSError('claim ancestor sync failed')),patch.object(c,'_storage'):
                result=c.run_campaign(root,campaign_path=Path(c.CAMPAIGN_PATH),source_snapshot=Path('snapshot'),release_path=Path('release'),stage='profile',dependencies=SimpleNamespace(clock=lambda:0.,run=run))
            run.assert_not_called()
            self.assertEqual(result['status'],'failed')
            self.assertIn('claim ancestor sync failed',result['error']['message'])
            self.assertTrue((root/c.DOMAIN/'isolated/profile').is_dir())

    def test_host_builder_to_projected_guest_two_resumes_and_strict_receipts(self):
        from contextlib import redirect_stderr
        import io
        import os
        import shutil
        from scripts.neural import sustained_controls as c,sustained_runtime as r,sustained_package as package
        from tests.unit.test_neural_sustained_runtime import ROOT,SegmentFixture
        from tests.unit.test_neural_study_cdi import cdi_facts
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory);f.projected=True;f.learned=True;root=f.root
            for scope in c.SNAPSHOT_SCOPES:
                original=ROOT/scope
                if original.is_file():
                    target=root/scope;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target);target.chmod(original.stat().st_mode&0o777)
            target=root/package.PARENT_PATH;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/package.PARENT_PATH,target)
            for binding in (*f.campaign['parents'].values(),*f.campaign['diagnostic_checkpoints'].values()):
                shutil.copytree(ROOT/binding['path'],root/binding['path'])
            inventory=c._source_inventory(root,c._source_files(root,c.SNAPSHOT_SCOPES))
            snapshot=root/'snapshot.json';c.write_json_exclusive(snapshot,{'state':'reviewed','files':inventory})
            release=root/'release.json';c.write_json_exclusive(release,{'schema':c.RELEASE_SCHEMA,'state':'approved','single_use':True,
                'source_snapshot_sha256':c.sha256_file(snapshot),'campaign_id':'isolated','campaign_raw_sha256':c.sha256_file(root/c.CAMPAIGN_PATH),
                'runtime_pins':f.campaign['runtime_pins'],'budgets':f.campaign['budgets']})
            verified={'campaign':f.campaign,'snapshot':{'files':inventory},'release':{'campaign_id':'isolated'},
                'source_sha256':c.sha256_file(snapshot),'release_sha256':c.sha256_file(release),'snapshot_path':snapshot,'release_path':release}
            stage=root/'stage';stage.mkdir();source=stage/'source';c._stage(root,source,inventory)
            client=stage/'client';client.mkdir();created=[];removed=[];current={}
            def run(argv,**kwargs):
                if argv[0]=='nvidia-smi':
                    text='3772, 498, '+c.low.GPU_UUID+'\n' if 'compute-apps' in argv[1] else c.low.GPU_UUID+', NVIDIA GeForce RTX 5090, 595.91.07, 32607, 20000\n'
                    return SimpleNamespace(returncode=0,stdout=text,stderr='')
                command=argv[5];text='';error=''
                if command=='create':
                    control=next(Path(value.split('src=')[1].split(',')[0]) for value in argv if value.startswith('type=bind,src=') and ',dst=/control,' in value)
                    context=json.loads((control/'job.json').read_text());selected=c._spec(context,control/'job.json')
                    self.assertEqual(argv,c.low.build_create_argv(selected,'docker',client))
                    current.update(control=control,context=context,spec=selected,started=False)
                    created.append(context['job_id']);text='d'*64+'\n'
                elif command=='inspect':
                    projected=c.low._inspect_projection(current['spec'],'d'*64)
                    if current['started']:projected['State']['Status']='exited'
                    text=json.dumps([projected])
                elif command=='start':
                    context=current['context'];control=current['control'];selected=current['spec']
                    projected_output=source/Path(context['guest_output_root']).relative_to('/learner');projected_output.mkdir(parents=True)
                    def guest(value):
                        path=Path(value)
                        if path.is_relative_to('/learner'):return source/path.relative_to('/learner')
                        if path.is_relative_to('/control'):return control/path.relative_to('/control')
                        if path==Path('/inputs/parent'):return Path(next(m['source'] for m in context['mounts'] if m['destination']=='/inputs/parent'))
                        return path
                    facts=cdi_facts({'mounts':[('/',True)]+[(m['destination'],m['readonly']) for m in context['mounts']],
                        'environment':dict(selected.environment),'uid':1000,'gid':1000,'python':c.low.CONTAINER_PYTHON,'device_nodes':['/dev/nvidia0']})
                    torch=SimpleNamespace(__version__='2.14.1+cu130',__file__='/runtime/venv-cuda/lib/python3.12/site-packages/torch/__init__.py',get_num_threads=lambda:4,get_num_interop_threads=lambda:1)
                    stream=io.StringIO()
                    with patch.object(c.low,'_guest_path',guest),patch.object(c.low,'_guest_facts',return_value=facts),patch.object(c.low,'_verify_guest_runtime'),patch.object(c.low,'_runtime_metadata',return_value={'torch':'2.14.1+cu130','numpy':'2.5.4','gymnasium':'1.4.0','spire-lab':'0.1.0'}),patch.dict('os.environ',dict(selected.environment),clear=True),patch.dict('sys.modules',{'torch':torch}),patch('spire_learn.neural.config.__file__',str(source/'src/spire_learn/neural/config.py')),redirect_stderr(stream):
                        value=r._guest_entry(source,control/'job.json',dependencies={'factory':f.factory},clock=f.clock)
                    error=stream.getvalue();text=json.dumps(value)+'\n'
                    self.assertLess(len(text),1024)
                    os.rmdir(context['host_output']);projected_output.rename(context['host_output']);current['started']=True
                elif command=='rm':removed.append(argv[-1])
                return SimpleNamespace(returncode=0,stdout=text,stderr=error)
            deps=c.HostDependencies(clock=f.clock,run=run,disk_usage=lambda p:SimpleNamespace(free=100*1024**3))
            budget=c.BudgetAccount({k:0. for k in c.BUCKETS},clock=f.clock)
            parent=f.campaign['parents']['run-b'];prefix={};predecessor=None;outputs=[]
            with patch.object(c.cdi,'verify_host_profile'),patch('spire_learn.neural.config.__file__',str(root/'src/spire_learn/neural/config.py')):
                for index in range(3):
                    job=c._job(root,stage,source,verified,budget,kind='segment',job_id=f'run-1-segment-{index:03d}',
                        reservation=dict(environment_attempts=3,sample_rows=3,optimizer_attempts=8,reset_slots=19,fixed_random_choices=3),
                        output_root=f.run['output_root'],parent=parent,prefix=prefix,replica='run-1',index=index,predecessor=predecessor,
                        deps=deps,docker='docker',client=client)
                    self.assertEqual(job['delta']['optimizer_accepted_steps'],4)
                    self.assertEqual(job['checkpoint']['counters']['optimizer_accepted_steps'],4*(index+1))
                    parent={'path':job['checkpoint']['path'],'inventory':job['checkpoint']['inventory']};prefix=job['checkpoint']['counters'];predecessor=job['receipt']
                    outputs.append(str(Path(parent['path']).parents[1]))
            self.assertEqual(len(set(outputs)),3);self.assertEqual(created,['run-1-segment-000','run-1-segment-001','run-1-segment-002'])
            self.assertEqual(removed,['d'*64]*3);self.assertEqual(f.calls,['warm_start','resume','resume'])

    def test_real_release_validation_is_read_only_and_binds_all_package_inputs(self):
        import shutil
        from scripts.neural import sustained_controls as c,sustained_package as package
        from tests.unit.test_neural_sustained_runtime import ROOT,SegmentFixture
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory);root=f.root
            for scope in c.SNAPSHOT_SCOPES:
                original=ROOT/scope
                if original.is_file():
                    target=root/scope;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target);target.chmod(original.stat().st_mode&0o777)
            initial=json.loads((ROOT/c.INITIAL).read_text())
            for relative in (c.INITIAL,Path(package.PARENT_PATH),Path(initial['runtime_intervals'][0]['receipt'])):
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
            for binding in (*f.campaign['parents'].values(),*f.campaign['diagnostic_checkpoints'].values()):
                shutil.copytree(ROOT/binding['path'],root/binding['path'])
            snapshot=root/'snapshot.json';review=root/'review.txt';review.write_text('isolated review fixture')
            with patch('spire_learn.neural.config.__file__',str(root/'src/spire_learn/neural/config.py')):
                value=c.capture_source_snapshot(root,root/'source-only.json',learner_head='a'*40,dirty={})
                value['state']='reviewed';c.write_json_exclusive(snapshot,value)
                release={'schema':c.RELEASE_SCHEMA,'state':'approved','single_use':True,'stage':'profile','campaign_id':'isolated',
                    'source_snapshot_sha256':c.sha256_file(snapshot),'campaign_raw_sha256':c.sha256_file(root/c.CAMPAIGN_PATH),
                    'runtime_pins':f.campaign['runtime_pins'],'budgets':f.campaign['budgets'],'review':{'path':'review.txt','sha256':c.sha256_file(review)},
                    'predecessor':None,'initial_accounting':{'path':c.INITIAL.as_posix(),'sha256':c.INITIAL_SHA256}}
                c.write_json_exclusive(root/'release.json',release)
                result=c.validate_release(root,campaign_path=Path(c.CAMPAIGN_PATH),source_snapshot=snapshot,release_path=root/'release.json',stage='profile')
                self.assertEqual(result['charged']['shared'],.09155025496147573)
                self.assertFalse((root/c.DOMAIN).exists())
                (root/'src/spire_learn/__init__.py').write_text('changed source')
                with self.assertRaisesRegex(ValueError,'source snapshot differs'):
                    c.validate_release(root,campaign_path=Path(c.CAMPAIGN_PATH),source_snapshot=snapshot,release_path=root/'release.json',stage='profile')
                self.assertFalse((root/c.DOMAIN).exists())

    def test_storage_floor_rejects_before_any_job_reservation(self):
        from scripts.neural import sustained_controls as c
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);owned=root/'owned';owned.mkdir()
            with self.assertRaisesRegex(RuntimeError,'storage'):
                c._storage(root,owned,SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=29*1024**3)),None,reservation=1)
            self.assertEqual(list(owned.iterdir()),[])

    def test_first_effect_fault_preserves_full_schedule_and_does_not_admit_later_jobs(self):
        from scripts.neural import sustained_controls as c
        from tests.unit.test_neural_sustained_runtime import ROOT
        campaign=json.loads((ROOT/c.CAMPAIGN_PATH).read_text());calls=[]
        verified={'campaign':campaign,'snapshot':{'files':{}},'release':{'campaign_id':'isolated','predecessor':None},
            'source_sha256':'a'*64,'release_sha256':'b'*64,'charged':{k:(.09155025496147573 if k=='shared' else 0.) for k in c.BUCKETS}}
        verified['release']['initial_accounting']={'path':c.INITIAL.as_posix(),'sha256':c.INITIAL_SHA256}
        def fail(*args,**kwargs):calls.append(kwargs['job_id']);raise RuntimeError('accepted effect fault')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(c,'validate_release',return_value=verified),patch.object(c.low,'_verify_runtime'),patch.object(c.low,'_docker_binary',return_value='docker'),patch.object(c.low,'_command',return_value='[]'),patch('scripts.neural.task8_gpu_setup_helpers.verify_image_inspect',return_value={'image_id':c.low.IMAGE_ID}),patch.object(c,'_stage',side_effect=lambda root,target,files:target.mkdir()),patch.object(c,'_job',side_effect=fail),patch.object(c,'_storage'):
                result=c.run_campaign(root,campaign_path=Path(c.CAMPAIGN_PATH),source_snapshot=Path('snapshot'),release_path=Path('release'),stage='train-evaluate',dependencies=SimpleNamespace(clock=lambda:0.))
            self.assertEqual(result['status'],'failed');self.assertEqual(len(calls),1)
            self.assertEqual(len(result['schedule']),704)
            self.assertEqual(sum(row['state']=='failed' for row in result['schedule']),1)
            self.assertEqual(sum(row['state']=='not_started' for row in result['schedule']),703)
            self.assertTrue(all(row['state']=='not_started' for row in result['training_schedule']))
            self.assertEqual(result['charged_seconds']['shared'],.09155025496147573)
            retained=root/c.DOMAIN/'isolated/train-evaluate/closed-stage.json'
            self.assertEqual(json.loads(retained.read_text())['schedule'],result['schedule'])
            seal=json.loads((retained.parent/'closure.json').read_text())
            self.assertEqual(seal['result']['sha256'],c.sha256_file(retained))
            self.assertEqual(seal['charged_seconds'],result['charged_seconds'])
