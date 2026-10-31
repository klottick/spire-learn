"""Pure sustained guest effects and real trainer facade; no runtime imports."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def api():
    return importlib.import_module('scripts.neural.sustained_runtime')


class Journal:
    def __init__(self): self.reserved = {}; self.settled = {}
    def reserve(self, row):
        assert row['operation_id'] not in self.reserved
        self.reserved[row['operation_id']] = row
    def settle(self, row):
        assert row['operation_id'] not in self.settled
        self.settled[row['operation_id']] = row


class RuntimeTests(unittest.TestCase):
    def test_profile_publication_has_separate_size_and_node_allowance(self):
        r=api()
        from scripts.neural import sustained_controls as c
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            payload={'trace_envelope':'x'*(9*1024**2),'primitive_rows':[0]*100001}
            r.publish_profile_result(path,payload)
            self.assertGreater(path.stat().st_size,8*1024**2)
            self.assertEqual(c._profile_receipt(path),payload)
            rejected=Path(directory)/'oversized.json'
            with self.assertRaisesRegex(ValueError,'profile result'):
                r.publish_profile_result(rejected,{'trace_envelope':'x'*(32*1024**2)})
            self.assertFalse(rejected.exists())

    def test_ledger_materializes_zeros_and_latches_uncertainty(self):
        r = api(); journal = Journal()
        ledger = r.EffectLedger('segment-000', journal, deadline=20., clock=lambda: 0.,
            limits=dict(environment_attempts=3, sample_rows=3, optimizer_attempts=3,
                        reset_slots=4, fixed_random_choices=3))
        with self.assertRaisesRegex(RuntimeError, 'primary'):
            ledger.call('environment_attempts', 2, 'native-step',
                        lambda: (_ for _ in ()).throw(RuntimeError('primary')))
        self.assertEqual(ledger.snapshot()['environment_attempts']['uncertain'], 2)
        self.assertEqual(ledger.snapshot()['sample_rows']['attempted'], 0)
        with self.assertRaisesRegex(RuntimeError, 'primary'):
            ledger.call('sample_rows', 1, 'actor', lambda: None)
        self.assertEqual(len(journal.reserved), 1)

    def test_deadline_before_effect_and_once_close_preserves_primary(self):
        r = api(); journal = Journal()
        ledger = r.EffectLedger('expired', journal, deadline=1., clock=lambda: 2.,
                              limits={k: 2 for k in r.EFFECT_KINDS})
        with self.assertRaises(TimeoutError): ledger.call('reset_slots',1,'reset',lambda: self.fail('effect'))
        self.assertEqual(journal.reserved,{})
        calls=[]
        runtime=SimpleNamespace(close=lambda: (calls.append('close'), (_ for _ in ()).throw(KeyboardInterrupt('close'))))
        with self.assertRaisesRegex(RuntimeError,'first') as caught:
            with r.owned_runtime(runtime): raise RuntimeError('first')
        self.assertEqual(calls,['close'])
        self.assertIn('close',str(caught.exception.__notes__))

    def test_primitive_runtime_versions_and_threads_are_process_observations(self):
        r=api()
        version=type('Version',(str,),{})('2.14.1+cu130')
        torch=SimpleNamespace(__version__=version,get_num_threads=lambda:4,get_num_interop_threads=lambda:1)
        with patch.dict('sys.modules',{'torch':torch}):
            evidence=r.process_evidence()
        self.assertIs(type(evidence['modules']['torch']['version']),str)
        self.assertEqual(evidence['threads'],{'intraop':4,'interop':1})

    def test_thread_initialization_once_across_repeated_neural_factories(self):
        r=api();calls=[]
        torch=SimpleNamespace(set_num_threads=lambda n:calls.append(('intra',n)),
            set_num_interop_threads=lambda n:calls.append(('interop',n)),
            get_num_threads=lambda:4,get_num_interop_threads=lambda:1)
        with patch.dict('sys.modules',{'torch':torch}),patch('spire_learn.neural.device.plan_runtime_device',return_value=None),patch('spire_learn.neural.device.require_runtime_execution_gate'),patch('spire_learn.neural.runner.create_runtime',return_value=SimpleNamespace(close=lambda:None)):
            for _ in range(11):r.default_factory(None,purpose='warm_start',checkpoint=Path('parent'))
        self.assertEqual(calls,[('intra',4),('interop',1)])

    def test_profile_bridge_exact_payload_and_compact_stdout_contract(self):
        r=api()
        payload=dict(phase_id='profile',source_sha256='a'*64,deadline=100.,input_bindings={},job_reservation={})
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            context=dict(kind='profile',job_id='profile',guest_output_root=str(output),profile_payload=payload)
            seen=[]
            def producer(root,campaign,context,**kwargs):
                self.assertEqual(set(context),set(payload)|{'effect_ledger'})
                self.assertEqual({k:context[k] for k in payload},payload)
                seen.append(context['effect_ledger']);return {'trace':'x'*100000}
            with patch('scripts.neural.sustained_controls.verify_guest_context',return_value=(context,{})),patch('scripts.neural.sustained_profile.run_profile',side_effect=producer):
                result=r._guest_entry(output,output/'job.json',dependencies={'factory':lambda:None})
            self.assertLess(len(json.dumps(result)),1024)
            self.assertGreater((output/'job-receipt.json').stat().st_size,65536)
            self.assertEqual(len(seen),1)


class SegmentFixture:
    """Exact authored config, real train, immutable metadata and fake native facade."""
    def __init__(self, directory):
        from scripts.neural import sustained_package as package
        self.root=Path(directory); self.campaign=json.loads((ROOT/package.CAMPAIGN_PATH).read_text())
        for scope in ('src','configs'):
            shutil.copytree(ROOT/scope,self.root/scope)
        self.run=self.campaign['training']['runs'][1]  # Run B1 permits a fixed-only real collector facade.
        self.journal=Journal(); self.now=0.; self.calls=[]; self.closed=[]
        self.parent=self.root/'inputs/parent'; self.parent.mkdir(parents=True)
        original=ROOT/self.campaign['parents']['run-b']['path']
        for name in ('metadata.json','model.pt','training.pt'): shutil.copyfile(original/name,self.parent/name)
        self.context=None
    def clock(self): self.now+=.001; return self.now
    def context_for(self,index):
        r=api(); output=self.root/self.run['output_root'].removeprefix('/learner/')
        output.mkdir(parents=True,exist_ok=True)
        return dict(schema=r.CONTEXT_SCHEMA,job_id=f'run-1-segment-{index:03}',kind='segment',
            replica='run-1',segment_index=index,source_sha256='a'*64,deadline=self.now+100.,
            whole_deadline=self.now+130.,guest_output_root=str(output),parent=dict(path=str(self.parent),
                inventory=r.checkpoint_inventory(self.parent)),prefix={} if index==0 else deepcopy(self.prefix),
            predecessor=None if index==0 else dict(receipt_sha256='b'*64),
            reservation=dict(environment_attempts=3,sample_rows=3,optimizer_attempts=8,reset_slots=19,fixed_random_choices=3),
            effect_ledger=self.journal)
    def factory(self,config,*,purpose,checkpoint,**kwargs):
        from spire_learn.neural import runner
        from tests.unit.test_neural_runner import FakeRuntime, FakeCollector, _record, _rollout
        r=api(); self.calls.append(purpose)
        prefix={} if purpose=='warm_start' else deepcopy(self.prefix)
        learned=getattr(self,'learned',False)
        records=[_record(config,slot=0,decision=i,version=0,terminal=False,learned=learned) for i in range(3)]
        rollout=_rollout(config,records,stop_reason='inactive_reset_boundary')
        collector=FakeCollector(config,[rollout]); collector.controller_rng=SimpleNamespace(choice=lambda rows:rows[0])
        collector.driver.reset=lambda **kw: ()
        collector.driver.step=lambda actions: ()
        collector.policy.act=lambda frames,**kw: ()
        original=collector.collect
        def collect(**kw):
            self.assert_cap=kw['max_env_actions']
            if learned:collector.policy.act(tuple(record.source for record in records))
            collector.driver.step((0,0,0))
            if not learned:
                for _ in range(3):collector.controller_rng.choice((0,))
            result=original(**kw); collector.diagnostics=result.collection
            return result
        collector.collect=collect
        runtime=FakeRuntime(config,collector,prefix=prefix); runtime.optimizer=SimpleNamespace(step=lambda:None) if learned else None
        if learned:
            original_update=runtime.update
            def update(rollout):
                for _ in range(config.ppo.epochs):runtime.optimizer.step()
                return original_update(rollout)
            runtime.update=update
        runtime.close=lambda:self.closed.append(purpose)
        def save(path,purpose):
            # Preserve actual payload archive/inventory; only authored metadata is fake.
            path.mkdir()
            for name in ('model.pt','training.pt'):shutil.copyfile(self.parent/name,path/name)
            meta=json.loads((self.parent/'metadata.json').read_text())
            diagnostics=collector.partial_rollout.collection if collector.partial_rollout is not None else None
            progress={k:prefix.get(k,0)+getattr(diagnostics,k,0) for k in runner._PROGRESS_FIELDS}
            for k in ('optimizer_attempted_steps','optimizer_accepted_steps','optimizer_uncertain_steps','update_count'):
                progress[k]=getattr(runtime,k)
            progress['elapsed_seconds']=prefix.get('elapsed_seconds',0.)+runtime.budget.elapsed()
            meta['identity']=__import__('spire_learn.neural.checkpoint',fromlist=['identity_payload']).identity_payload(runner._selected_identity(config))
            recovery=dict(next_seed_ordinal=progress['delivered_reset_slots'],next_decision_sequence=progress['delivered_environment_slots'],next_episode_ordinals=[progress['delivered_reset_slots']])
            meta['training_state'].update(counters=progress,optimizer_step=progress['optimizer_accepted_steps'],next_environment_seed_ordinal=recovery['next_seed_ordinal'],next_decision_sequence=recovery['next_decision_sequence'])
            meta['run_metadata'].update(config_sha256=config.config_sha256,manifest_sha256=config.manifest_sha256,
                mode=config.mode,cumulative_prefix=progress,collector_recovery=recovery)
            meta['parent_checkpoint']={'path':'/inputs/parent' if getattr(self,'projected',False) else str(checkpoint)}
            (path/'metadata.json').write_text(json.dumps(meta))
            self.prefix=progress
            self.now+=.1  # save/close time must not change exact checkpoint prefix.
        runtime.save=save
        return runtime


class SegmentTests(unittest.TestCase):
    def test_factory_consuming_action_allowance_can_save_clean_zero_update_checkpoint(self):
        r=api()
        from scripts.neural import sustained_controls as c
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory);context=f.context_for(0)
            def factory(*args,**kwargs):
                runtime=f.factory(*args,**kwargs);f.now=context['deadline']+.1;return runtime
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                result=r.run_segment(f.root,f.campaign,context,dependencies={'factory':factory},clock=f.clock)
                c.validate_segment_receipt(f.root,f.campaign,context,result)
            self.assertEqual(result['delta']['attempted_environment_slots'],0)
            self.assertEqual(result['delta']['optimizer_accepted_steps'],0)
            self.assertEqual(result['effects']['environment_attempts']['attempted'],0)
            self.assertEqual(len(f.closed),1)

    def test_evaluation_child_timeout_is_retained_and_never_retried(self):
        r=api();calls=[]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            context={'deadline':10.,'guest_output_root':str(output),'operation':{'availability':'reference'}}
            def child(*args,**kwargs):calls.append(args[0]);raise TimeoutError('child timeout')
            with patch('scripts.neural.gpu_runtime_proof_helpers._bounded_subprocess_run',side_effect=child):
                with self.assertRaisesRegex(TimeoutError,'child timeout'):r.run_evaluation(output,{},context,clock=lambda:0.)
            self.assertEqual(len(calls),1)
            transport=json.loads((output/'reference-child-transport.json').read_text())
            self.assertEqual(transport['status'],'unreturned')
            self.assertIn('_reference',transport['argv'])

    def test_bounded_evaluation_incomplete_is_retained_without_fabricating_a_loss(self):
        r=api()
        from scripts.neural.sustained_package import evaluation_schedule
        from tests.unit.test_neural_runner import FakeRuntime,FakeCollector,_record,_rollout
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory)
            operation=next(row for row in evaluation_schedule(f.campaign,{}) if row['policy_id']=='end-turn-last-v1')
            work=f.campaign['selection']['case_bindings'][operation['case_index']]
            output=f.root/work['output_root'].removeprefix('/learner/');output.mkdir(parents=True)
            context=dict(schema=r.CONTEXT_SCHEMA,kind='evaluation',job_id='evaluation-case0',source_sha256='a'*64,
                operation=operation,parent=None,guest_output_root=str(output),deadline=100.,whole_deadline=130.,
                reservation=dict(environment_attempts=256,sample_rows=256,optimizer_attempts=0,reset_slots=1,fixed_random_choices=256),effect_ledger=Journal())
            def factory(config,**kwargs):
                records=(_record(config,slot=0,decision=0,version=0,terminal=False,learned=False),)
                collector=FakeCollector(config,[_rollout(config,records,stop_reason='environment_action_budget')])
                collector.controller_rng=SimpleNamespace(choice=lambda rows:rows[0]);collector.driver.reset=lambda **kw:()
                collector.driver.step=lambda actions:();collector.policy.act=lambda *args,**kwargs:()
                original=collector.collect
                def collect(**kw):collector.driver.step((0,));return original(**kw)
                collector.collect=collect
                runtime=FakeRuntime(config,collector);runtime.optimizer=None;runtime.policy_kind=kwargs['policy_kind']
                return runtime
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                result=r.run_evaluation(f.root,f.campaign,context,dependencies={'factory':factory},clock=f.clock)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['report']['status'],'incomplete')
            self.assertEqual(result['report']['collection']['deaths'],0)
            self.assertEqual(result['effects']['optimizer_attempts']['attempted'],0)
            self.assertEqual(result['reference_import_guard']['attempts'],[])

    def test_complete_projected_entry_uses_real_trainer_and_strict_host_validator(self):
        from scripts.neural import sustained_controls as c,task9_study_controls as low
        from scripts.neural.gpu_runtime_proof_helpers import _source_files,_source_inventory,sha256_file,write_json_exclusive
        from tests.unit.test_neural_study_cdi import cdi_facts
        r=api()
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory);f.projected=True
            for helper in c.HELPERS:
                target=f.root/helper;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/helper,target)
            inventory=_source_inventory(f.root,_source_files(f.root,c.GUEST_SCOPES))
            output=f.root/f.run['output_root'].removeprefix('/learner/');output.mkdir(parents=True)
            control=f.root/'control';control.mkdir()
            release={'schema':c.RELEASE_SCHEMA,'state':'approved','single_use':True,'source_snapshot_sha256':'placeholder',
                'campaign_id':'fake','campaign_raw_sha256':sha256_file(f.root/c.CAMPAIGN_PATH),
                'runtime_pins':f.campaign['runtime_pins'],'budgets':f.campaign['budgets']}
            write_json_exclusive(control/'snapshot.json',{'state':'reviewed','files':inventory})
            release['source_snapshot_sha256']=sha256_file(control/'snapshot.json')
            write_json_exclusive(control/'release.json',release)
            context=f.context_for(0);context.pop('effect_ledger');context.update(
                guest_output_root=f.run['output_root'],host_output=str(output),campaign_id='fake',container_name='truthful-run-1-segment-000',
                runtime_class='gpu',source_files=inventory,source_sha256=release['source_snapshot_sha256'],
                release_sha256=sha256_file(control/'release.json'))
            context['parent']['path']='/inputs/parent'
            claim={'job_id':context['job_id'],'release_sha256':context['release_sha256'],'reservation':context['reservation'],'predecessor':None}
            write_json_exclusive(control/'claim.json',claim);context['claim_sha256']=sha256_file(control/'claim.json')
            mounts=[{'source':str(f.root/scope),'destination':'/learner/'+scope,'readonly':True} for scope in c.GUEST_SCOPES]
            mounts += [{'source':'/fake/runtime','destination':'/runtime/venv-cuda','readonly':True},
                {'source':'/fake/lock','destination':'/runtime/requirements-cuda.lock','readonly':True},
                {'source':str(control),'destination':'/control','readonly':True},
                {'source':str(f.parent),'destination':'/inputs/parent','readonly':True},
                {'source':str(output),'destination':f.run['output_root'],'readonly':False}]
            context['mounts']=mounts;write_json_exclusive(control/'job.json',context)
            selected=c._spec(context,control/'job.json')
            facts=cdi_facts({'mounts':[('/',True)]+[(m['destination'],m['readonly']) for m in mounts],
                'environment':dict(selected.environment),'uid':1000,'gid':1000,'python':low.CONTAINER_PYTHON,
                'device_nodes':['/dev/nvidia0']})
            def guest(value):
                path=Path(value)
                if path==Path('/learner') or path.is_relative_to('/learner'):return f.root/path.relative_to('/learner')
                if path.is_relative_to('/control'):return control/path.relative_to('/control')
                if path==Path('/inputs/parent'):return f.parent
                return path
            with patch.object(low,'_guest_path',guest),patch.object(low,'_guest_facts',return_value=facts),patch.object(low,'_verify_guest_runtime'),patch.object(low,'_runtime_metadata',return_value={'torch':'2.14.1+cu130','numpy':'2.5.4','gymnasium':'1.4.0','spire-lab':'0.1.0'}),patch.dict('os.environ',dict(selected.environment),clear=True),patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                result=r._guest_entry(f.root,control/'job.json',dependencies={'factory':f.factory},clock=f.clock)
                receipt=json.loads((output/'job-receipt.json').read_text())
                c.validate_segment_receipt(f.root,f.campaign,context,receipt)
            self.assertLess(len(json.dumps(result)),1024)
            self.assertEqual(f.calls,['warm_start'])
            self.assertEqual(receipt['effects']['environment_attempts']['delivered'],3)
            self.assertTrue(receipt['journal'])

    def test_warm_start_then_two_resumes_preserve_save_prefix_and_local_deltas(self):
        r=api()
        from scripts.neural import sustained_controls as controls
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory)
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                for index in range(3):
                    context=f.context_for(index)
                    receipt=r.run_segment(f.root,f.campaign,context,dependencies={'factory':f.factory},clock=f.clock)
                    validated=controls.validate_segment_receipt(f.root,f.campaign,context,receipt)
                    self.assertEqual(validated['delta']['attempted_environment_slots'],3)
                    self.assertEqual(validated['delta']['optimizer_accepted_steps'],0)
                    self.assertEqual(receipt['checkpoint']['counters'],f.prefix)
                    self.assertEqual(f.assert_cap,3)
                    source=Path(receipt['checkpoint']['path'])
                    shutil.rmtree(f.parent);shutil.copytree(source,f.parent)
                    physical=f.root/f'finished-{index}';source.parents[1].rename(physical)
            self.assertEqual(f.calls,['warm_start','resume','resume'])
            self.assertEqual(len(f.closed),3)

    def test_stale_prefix_or_missing_predecessor_rejects_before_factory(self):
        r=api()
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory)
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                context=f.context_for(0);context['prefix']={'attempted_environment_slots':1}
                with self.assertRaisesRegex(ValueError,'prefix'):r.run_segment(f.root,f.campaign,context,dependencies={'factory':f.factory},clock=f.clock)
                f.prefix={};context=f.context_for(1);context['predecessor']=None
                with self.assertRaisesRegex(ValueError,'preceding receipt'):r.run_segment(f.root,f.campaign,context,dependencies={'factory':f.factory},clock=f.clock)
            self.assertEqual(f.calls,[])

    def test_two_resumed_optimizer_prefixes_are_cumulative_not_segment_work(self):
        r=api()
        from scripts.neural import sustained_controls as controls
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory);f.learned=True
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                for index in range(3):
                    context=f.context_for(index)
                    receipt=r.run_segment(f.root,f.campaign,context,dependencies={'factory':f.factory},clock=f.clock)
                    validated=controls.validate_segment_receipt(f.root,f.campaign,context,receipt)
                    self.assertEqual(validated['delta']['optimizer_accepted_steps'],4)
                    self.assertEqual(receipt['report']['optimizer']['accepted_steps'],4*(index+1))
                    self.assertEqual(receipt['effects']['optimizer_attempts']['delivered'],4)
                    source=Path(receipt['checkpoint']['path']);shutil.rmtree(f.parent);shutil.copytree(source,f.parent)
                    source.parents[1].rename(f.root/f'finished-{index}')

    def test_save_failure_closes_once_and_never_produces_success_receipt(self):
        r=api()
        with tempfile.TemporaryDirectory() as directory:
            f=SegmentFixture(directory)
            def factory(*args,**kwargs):
                runtime=f.factory(*args,**kwargs)
                runtime.save=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('save failed'))
                return runtime
            with patch('spire_learn.neural.config.__file__',str(f.root/'src/spire_learn/neural/config.py')):
                with self.assertRaisesRegex(RuntimeError,'did not complete'):
                    r.run_segment(f.root,f.campaign,f.context_for(0),dependencies={'factory':factory},clock=f.clock)
            self.assertEqual(len(f.closed),1)
