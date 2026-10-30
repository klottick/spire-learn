"""Pure producers: exercise real profiling wrappers with bounded fake effects."""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit.test_neural_schema import _api, _synthetic_schema, _frame
from spire_learn.neural.contracts import PolicyChoice

ROOT = Path(__file__).resolve().parents[2]


def api(test):
    try:
        return importlib.import_module('scripts.neural.sustained_profile')
    except ImportError as error:
        test.fail(f'profile producer missing: {error}')


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self):
        if getattr(self,'fail_next',False):
            self.fail_next=False
            raise RuntimeError('factory clock failure')
        self.now += .001
        return self.now


class DurableLedger:
    def __init__(self): self.reservations = {}; self.receipts = {}
    def reserve(self, record):
        assert record['operation_id'] not in self.reservations
        self.reservations[record['operation_id']] = dict(record)
    def settle(self, record):
        identifier = record['operation_id']
        assert identifier in self.reservations and identifier not in self.receipts
        self.receipts[identifier] = dict(record)


class ProfileTests(unittest.TestCase):
    def test_inputs_roundtrip_and_exact_work_inventory(self):
        p = api(self)
        campaign = json.loads((ROOT/'configs/neural/experiments/sustained-v1/campaign.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Author in isolated tree, retaining exact historical source inputs.
            for relative in p.profile_source_paths(campaign):
                target = root/relative; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            files = p.author_profile_inputs(root, campaign)
            self.assertEqual(files, p.author_profile_inputs(root, campaign))
            with patch('spire_learn.neural.config.__file__',str(root/'src/spire_learn/neural/config.py')):
                plan = p.profile_plan(root, campaign)
            self.assertEqual(len(plan['catalog_case_ids']), 78)
            self.assertEqual([row['old_case_index'] for row in plan['traces']], [0,0,4,4,1,1,3,3])
            self.assertEqual([row['checkpoint_id'] for row in plan['traces']], ['standalone-a','standalone-b']*4)
            self.assertEqual(plan['reservation_ceiling']['environment_attempts'], 1614)
            self.assertEqual(plan['reservation_ceiling']['optimizer_attempts'], 80)
            self.assertLessEqual(plan['reservation_ceiling']['sample_rows'], 16384)
            for row in plan['training']:
                from spire_learn.neural.config import load_neural_config
                with patch('spire_learn.neural.config.__file__',str(root/'src/spire_learn/neural/config.py')):
                    config = load_neural_config(root/row['config'])
                self.assertEqual(config.rollout_rows, 16 if row['mode']=='standalone' else 1)
                self.assertEqual(config.budgets.max_environment_actions, 1024 if row['mode']=='standalone' else 256)
            target = root/next(iter(files)); target.write_bytes(b'collision')
            with self.assertRaisesRegex(ValueError, 'differing'):
                p.author_profile_inputs(root, campaign)

    def test_semantic_projection_preserves_unknown_and_original_indices(self):
        p = api(self); a = _api(self)
        with _synthetic_schema(a) as schema:
            frame = _frame(a, schema, scalar0_known=False, target_group_known=True, target_row_known=False)
            choice = PolicyChoice(9, -.5, 1., 0)
            trace = p.public_decision_trace(schema, frame, choice,
                legal_scores={2:{'logit':1., 'probability':.4},9:{'logit':2., 'probability':.6}},
                successor=frame, result={'native_reward':0.,'terminated':False,'truncated':False,'last_exit':{'secret':1}})
            self.assertEqual([row['index'] for row in trace['legal_candidates']], [2,9])
            self.assertIsNone(trace['legal_candidates'][0]['scalar0'])
            self.assertFalse(trace['legal_candidates'][0]['scalar0_known'])
            self.assertIsNone(trace['legal_candidates'][0]['target']['row'])
            self.assertFalse(trace['legal_candidates'][0]['target']['known'])
            self.assertEqual(len(trace['source']['enemies']), 2)
            self.assertFalse(trace['source']['enemies'][1]['occupied'])
            self.assertIsNone(trace['source']['enemies'][1]['fields']['enemy_kind']['value'])
            self.assertNotIn('last_exit', json.dumps(trace))
            self.assertEqual(set(trace['source']),{'player','enemies'})
            self.assertLessEqual(len(p.json_bytes(trace)), 65536)
            with self.assertRaisesRegex(ValueError, 'legal'):
                p.public_decision_trace(schema, frame, choice, legal_scores={0:{'logit':0.,'probability':1.}}, successor=frame,result={})

    def test_effect_reserves_precede_calls_and_uncertainty_is_not_retried(self):
        p = api(self); external = DurableLedger(); clock = Clock()
        ledger = p.EffectLedger('test', external, deadline=100., clock=clock)
        def effect():
            self.assertEqual(len(external.reservations), 1)
            raise RuntimeError('accepted effect failed')
        with self.assertRaisesRegex(RuntimeError, 'accepted effect failed'):
            ledger.call('environment_attempts', 3, 'native-step', effect)
        self.assertEqual(ledger.snapshot()['environment_attempts'], {'reserved':3,'attempted':3,'delivered':0,'uncertain':3})
        self.assertEqual(len(external.receipts), 1)
        ledger=p.EffectLedger('second', external,deadline=100.,clock=clock)
        external.settle = lambda record: (_ for _ in ()).throw(KeyboardInterrupt('settle failed'))
        with self.assertRaisesRegex(RuntimeError, 'first cause') as caught:
            ledger.call('sample_rows',1,'actor',lambda: (_ for _ in ()).throw(RuntimeError('first cause')))
        self.assertTrue(any('settle failed' in note for note in caught.exception.__notes__))

    def test_timing_nested_exclusive_sync_and_inactive_hooks(self):
        p = api(self)
        from spire_learn.neural.profiling import observe, span, observed
        clock = Clock(); sync = []
        recorder = p.TimingRecorder(clock=clock, synchronize=lambda: sync.append('sync'))
        with observe(recorder):
            with span('outer'):
                with span('inner'): clock()
        rows = recorder.snapshot()
        self.assertEqual(rows['outer']['count'], 1)
        self.assertAlmostEqual(rows['outer']['inclusive_wall_seconds'], rows['outer']['exclusive_wall_seconds']+rows['inner']['inclusive_wall_seconds'])
        self.assertEqual(rows['inner']['synchronization'], 'device-synchronized-wall')
        self.assertGreater(rows['outer']['synchronization_seconds'], 0.)
        self.assertEqual(len(sync),4)
        @observed('unused')
        def identity(value): return value
        with patch('time.monotonic', side_effect=AssertionError('inactive clock')):
            self.assertEqual(identity(4),4)
        class CallableAlias:
            method = staticmethod(identity)
        self.assertEqual(CallableAlias().method(5), 5)

    def test_unique_frame_payload_across_retained_rollouts(self):
        p = api(self); a = _api(self)
        with _synthetic_schema(a) as schema:
            first = _frame(a,schema); second = _frame(a,schema)
            counter = p.FramePayloadCounter(max_buffers=5000)
            record = SimpleNamespace(source=first,successor=second)
            counter.add((record,))
            expected = sum(len(value) for value in {id(b.data):b.data for f in (first,second) for b in f.tensors.values()}.values())
            self.assertEqual(counter.bytes,expected)
            counter.add((SimpleNamespace(source=second,successor=first),))
            self.assertEqual(counter.bytes,expected)
            with self.assertRaisesRegex(ValueError,'capacity'):
                p.FramePayloadCounter(max_buffers=1).add((record,))

    def test_real_runtime_wrappers_drive_fake_effects_and_restore(self):
        p = api(self); clock=Clock(); external=DurableLedger()
        ledger=p.EffectLedger('test',external,deadline=100.,clock=clock)
        timing=p.TimingRecorder(clock=clock)
        a=_api(self)
        with _synthetic_schema(a) as schema:
            frame=_frame(a,schema)
            class Policy:
                def act(self,frames,**kwargs):
                    self_test.assertEqual(ledger.snapshot()['sample_rows']['reserved'],len(frames))
                    return tuple(PolicyChoice(9,-.5,1.,0) for _ in frames)
                def values(self,frames): return tuple(0. for _ in frames)
            class Driver:
                def reset(self,*,seeds,entry_ids,reset_mask=None): return ('reset',)
                def step(self,actions): return ('step',)
            class Optimizer:
                def step(self):
                    self_test.assertEqual(ledger.snapshot()['optimizer_attempts']['reserved'],4)
            self_test=self
            policy=Policy(); driver=Driver(); optimizer=Optimizer()
            rollout=SimpleNamespace(records=(SimpleNamespace(source=frame,successor=frame,actor_training_mask=True),))
            collector=SimpleNamespace(driver=driver,policy=policy,controller_rng=SimpleNamespace(choice=lambda rows:rows[0]),collect=lambda **kwargs:rollout)
            closed=[]
            def update(value): optimizer.step(); return 'updated'
            runtime=SimpleNamespace(collector=collector,policy=policy,optimizer=optimizer,update=update,save=lambda *a:None,close=lambda:closed.append(1),torch=None)
            config=SimpleNamespace(ppo=SimpleNamespace(epochs=4,minibatch_rows=64),device='cpu')
            original=collector.collect
            with p.instrument_runtime(runtime,config,ledger,timing,operation='training') as metrics:
                driver.reset(seeds=(1,),entry_ids=('one',))
                policy.act((frame,),generator=object())
                driver.step((9,))
                got=collector.collect(); self.assertIs(got,rollout)
                self.assertEqual(runtime.update(got),'updated')
                runtime.save(Path('unused'),'final')
            self.assertIs(collector.collect,original)
            self.assertEqual(closed,[1])
            self.assertEqual(metrics['retained_frame_payload_bytes'],sum(len(v) for v in {id(b.data):b.data for b in frame.tensors.values()}.values()))
            self.assertEqual(ledger.snapshot()['optimizer_attempts'],{'reserved':4,'attempted':1,'delivered':1,'uncertain':0})
            self.assertIn('collection',timing.snapshot()); self.assertIn('ppo_update',timing.snapshot())

    def test_close_error_preserves_control_and_instrumentation_restores(self):
        p=api(self); clock=Clock(); ledger=p.EffectLedger('test',DurableLedger(),deadline=100.,clock=clock)
        collector=SimpleNamespace(driver=SimpleNamespace(reset=lambda **k:(),step=lambda a:()),policy=SimpleNamespace(act=lambda *a,**k:(),values=lambda a:()),controller_rng=SimpleNamespace(choice=lambda a:a[0]),collect=lambda **k:None)
        def close(): raise RuntimeError('close failed')
        runtime=SimpleNamespace(collector=collector,policy=collector.policy,optimizer=None,close=close,torch=None)
        original=collector.collect
        with self.assertRaisesRegex(KeyboardInterrupt,'primary') as caught:
            with p.instrument_runtime(runtime,SimpleNamespace(device='cpu'),ledger,p.TimingRecorder(clock=clock),operation='trace'):
                raise KeyboardInterrupt('primary')
        self.assertIs(collector.collect,original)
        self.assertTrue(any('close failed' in note for note in caught.exception.__notes__))

    def test_trace_store_bounds_reject_before_retention(self):
        p=api(self)
        store=p.TraceStore(max_actions=32,max_decision_bytes=65536,max_total_bytes=16777216)
        for i in range(32): store.add('episode',{'index':i})
        with self.assertRaisesRegex(ValueError,'action'): store.add('episode',{})
        with self.assertRaisesRegex(ValueError,'decision'): store.add('other',{'data':'a'*65536})
        tiny=p.TraceStore(max_actions=32,max_decision_bytes=65536,max_total_bytes=12)
        with self.assertRaisesRegex(ValueError,'total'): tiny.add('episode',{'data':'a'*12})
        self.assertEqual(tiny.bytes,2)

    def test_imports_are_optional_runtime_free_in_fresh_process(self):
        api(self)
        code='''import importlib.abc,sys\nclass Guard(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in {'torch','numpy','gymnasium','spire_lab','torchrl','tensordict'}: raise AssertionError(fullname)\nsys.meta_path.insert(0,Guard())\nfrom scripts.neural import sustained_profile as p\nfrom spire_learn.neural import profiling,runner,features,model,ppo,envs,artifacts\nassert callable(p.run_profile) and callable(p.public_decision_trace)\nprint('optional-import guard PASS')\n'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_complete_fake_driven_profile_strict_accepted_boundaries(self):
        """Catches wrong real signatures, raw trace construction, and bypassed effects."""
        p=api(self); campaign=json.loads((ROOT/'configs/neural/experiments/sustained-v1/campaign.json').read_text())
        a=_api(self); clock=Clock(); external=DurableLedger(); opened=[]; saves=[]
        from spire_learn.neural.contracts import Lifecycle,StepRecord,TaskSignal
        from spire_learn.neural.envs import DriverRow,DriverTransition
        from spire_learn.neural.schema import candidates
        from spire_learn.neural.runner import NeuralReport
        import shutil
        with tempfile.TemporaryDirectory() as directory, _synthetic_schema(a) as schema:
            root=Path(directory)
            for relative in p.profile_source_paths(campaign):
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/relative,target)
            for binding in (*campaign['parents'].values(),campaign['diagnostic_checkpoints']['standalone-a']):
                for name in binding['inventory']:
                    relative=binding['path']+'/'+name; target=root/relative
                    target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/relative,target)
            p.author_profile_inputs(root,campaign)
            frame=_frame(a,schema)
            life=Lifecycle(True,'combat',None,0,1,0,None)
            row=DriverRow(frame,life,candidates(frame,schema),())
            class Driver:
                def __init__(self,width): self.schema=schema; self.width=width
                def reset(self,*,seeds,entry_ids,reset_mask=None):
                    assert len(seeds)==self.width and len(entry_ids)==self.width
                    return (row,)*self.width
                def step(self,actions):
                    assert len(actions)==self.width
                    return tuple(DriverTransition(row,row,0.,False,False) for _ in actions)
            class Policy:
                def act(self,frames,*,generator,deterministic=False):
                    return tuple(PolicyChoice(9,-.5,1.,0) for _ in frames)
                def values(self,frames): return (0.,)*len(frames)
            class Optimizer:
                def __init__(self): self.attempted=0; self.accepted=0
                def step(self): self.attempted+=1; self.accepted+=1
            class Rng:
                def choice(self,items): return items[0]
            class Collector:
                def __init__(self,config,driver,policy):
                    self.config=config; self.driver=driver; self.policy=policy; self.controller_rng=Rng(); self.sequence=0
                def collect(self,*,max_learned_decisions,max_env_actions,policy_version):
                    width=self.config.rollout_rows
                    self.driver.reset(seeds=(0,)*width,entry_ids=('one',)*width)
                    choices=self.policy.act((frame,)*width,generator=object())
                    self.driver.step(tuple(choice.original_action_index for choice in choices))
                    records=tuple(StepRecord('stream','task',f'decision-{self.sequence}-{i}',frame,frame,0.,False,False,
                        TaskSignal(0.,False,True,True),choice,True,0.) for i,choice in enumerate(choices))
                    self.sequence+=1
                    return SimpleNamespace(records=records)
            class Runtime:
                def __init__(self,config):
                    self.config=config; self.policy=Policy(); self.optimizer=Optimizer() if config.operation=='train' else None
                    self.collector=Collector(config,Driver(config.rollout_rows),self.policy); self.closed=False
                    self.torch=SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda device:None))
                def update(self,rollout): self.optimizer.step(); return self.optimizer.accepted
                def save(self,path,purpose): saves.append((path,purpose))
                def close(self): self.closed=True
            def factory(config,*,purpose,checkpoint=None,policy_kind='checkpoint',action_selection='seeded-sampling'):
                assert policy_kind=='checkpoint' and checkpoint.is_dir()
                if purpose=='evaluate': assert action_selection=='deterministic' and config.device=='cpu'
                else: assert purpose=='warm_start'
                runtime=Runtime(config); opened.append(runtime); return runtime
            def train(config,*,runtime,clock):
                assert runtime.operation_limits.max_updates==2
                for _ in range(2):
                    rollout=runtime.collector.collect(max_learned_decisions=config.budgets.max_training_decisions,
                        max_env_actions=runtime.operation_limits.max_environment_actions,policy_version=0)
                    runtime.update(rollout)
                runtime.save(Path('fake-final'),'final'); runtime.close()
                return NeuralReport({'status':'complete'},0,None)
            def evaluate(config,*,runtime,clock,allow_final=False):
                assert not allow_final and config.budgets.max_environment_actions==32
                runtime.collector.collect(max_learned_decisions=32,max_env_actions=32,policy_version=0)
                runtime.close(); return NeuralReport({'status':'complete'},0,None)
            def memory(runtime): return {'rss_bytes':100}
            def scores(runtime,source):
                assert source is frame
                return {2:{'logit':1.,'probability':.4},9:{'logit':2.,'probability':.6}}
            with patch('spire_learn.neural.config.__file__',str(root/'src/spire_learn/neural/config.py')):
                inventory=p.profile_input_inventory(root,campaign)
                context={'phase_id':'profile-fake','source_sha256':'a'*64,'deadline':100.,'effect_ledger':external,
                    'input_bindings':{relative:{**item,'path':str(root/relative)} for relative,item in inventory.items()},
                    'job_reservation':{'environment_attempts':1614,'sample_rows':1792,'optimizer_attempts':80,'reset_slots':1383,'fixed_random_choices':256}}
                with patch.object(p,'_legal_scores',scores):
                    report=p.run_profile(root,campaign,context,dependencies={'factory':factory,'train':train,'evaluate':evaluate,
                        'clock':clock,'memory':memory,'synchronize':lambda runtime:None})
                p.validate_profile_report(report,p.profile_plan(root,campaign))
                self.assertEqual(report['effects']['environment_attempts']['attempted'],120)
                self.assertEqual(report['effects']['sample_rows']['attempted'],50)
                self.assertEqual(report['effects']['optimizer_attempts']['attempted'],4)
                self.assertEqual(report['randomness']['deterministic_trace_policy_rng_draws'],0)
                self.assertEqual(len(report['traces']),8)
                self.assertEqual(len(report['catalog']),78)
                self.assertEqual(len(saves),2)
                self.assertTrue(all(runtime.closed for runtime in opened))
                self.assertEqual(len(external.reservations),len(external.receipts))
                before=len(opened)
                incomplete={**context,'input_bindings':dict(context['input_bindings'])}
                incomplete['input_bindings'].pop(next(iter(incomplete['input_bindings'])))
                with self.assertRaisesRegex(ValueError,'bindings incomplete'):
                    p.run_profile(root,campaign,incomplete,dependencies={'factory':factory,'clock':clock})
                self.assertEqual(len(opened),before)
                def clock_fault_factory(config,*,purpose,checkpoint=None,policy_kind='checkpoint',action_selection='seeded-sampling'):
                    created=factory(config,purpose=purpose,checkpoint=checkpoint,policy_kind=policy_kind,action_selection=action_selection)
                    clock.fail_next=True
                    return created
                retry_context={**context,'phase_id':'fault-profile','effect_ledger':DurableLedger()}
                with self.assertRaisesRegex(RuntimeError,'factory clock failure'):
                    p.run_profile(root,campaign,retry_context,dependencies={'factory':clock_fault_factory,'clock':clock,'memory':memory})
                self.assertTrue(opened[-1].closed)
                broken=json.loads(json.dumps(report)); broken['trace_bytes']+=1
                with self.assertRaisesRegex(ValueError,'byte accounting'):
                    p.validate_profile_report(broken,p.profile_plan(root,campaign))
                broken=json.loads(json.dumps(report)); broken['effects']['environment_attempts']['attempted']+=1
                broken['effects']['environment_attempts']['delivered']+=1; broken['effects']['environment_attempts']['reserved']+=1
                with self.assertRaisesRegex(ValueError,'retained records'):
                    p.validate_profile_report(broken,p.profile_plan(root,campaign))
                broken=json.loads(json.dumps(report)); broken['effect_categories']['diagnostic_score_rows']['attempted']+=1
                with self.assertRaisesRegex(ValueError,'categories'):
                    p.validate_profile_report(broken,p.profile_plan(root,campaign))


    def test_memory_observer_failure_cannot_replace_optimizer_cause(self):
        p=api(self); clock=Clock(); ledger=p.EffectLedger('test',DurableLedger(),deadline=100.,clock=clock)
        collector=SimpleNamespace(driver=SimpleNamespace(reset=lambda **k:(),step=lambda a:()),policy=SimpleNamespace(act=lambda *a,**k:(),values=lambda a:()),controller_rng=SimpleNamespace(choice=lambda a:a[0]),collect=lambda **k:None)
        def update(rollout): raise RuntimeError('optimizer primary')
        runtime=SimpleNamespace(collector=collector,policy=collector.policy,optimizer=SimpleNamespace(step=lambda:None),update=update,close=lambda:None,torch=None)
        def memory(runtime): raise KeyboardInterrupt('memory observer')
        with self.assertRaisesRegex(RuntimeError,'optimizer primary') as caught:
            with p.instrument_runtime(runtime,SimpleNamespace(device='cpu',ppo=SimpleNamespace(epochs=4,minibatch_rows=64)),ledger,p.TimingRecorder(clock=clock),operation='training',memory=memory):
                runtime.update(SimpleNamespace(records=(SimpleNamespace(actor_training_mask=True),)))
        self.assertTrue(any('memory observer' in note for note in caught.exception.__notes__))

    def test_context_refuses_widened_or_missing_binding_before_any_factory(self):
        p=api(self)
        campaign=json.loads((ROOT/'configs/neural/experiments/sustained-v1/campaign.json').read_text())
        called=[]
        with self.assertRaisesRegex(ValueError,'context'):
            p.run_profile(ROOT,campaign,{},dependencies={'factory':lambda:called.append(1)})
        self.assertEqual(called,[])


    def test_effect_ceiling_and_deadline_stop_before_forwarding(self):
        p=api(self); clock=Clock(); external=DurableLedger(); calls=[]
        limits={'environment_attempts':1,'sample_rows':0,'optimizer_attempts':0,'reset_slots':1,'fixed_random_choices':0}
        ledger=p.EffectLedger('ceiling',external,deadline=1.,clock=clock,limits=limits)
        ledger.call('environment_attempts',1,'step',lambda:calls.append(1))
        with self.assertRaisesRegex(ValueError,'ceiling'):
            ledger.call('environment_attempts',1,'step',lambda:calls.append(2))
        clock.now=2.
        with self.assertRaises(TimeoutError):
            ledger.call('reset_slots',1,'reset',lambda:calls.append(3))
        self.assertEqual(calls,[1]); self.assertEqual(len(external.reservations),1)

    def test_timing_post_sync_error_preserves_native_exception(self):
        p=api(self); clock=Clock(); calls=[]
        def sync():
            calls.append(1)
            if len(calls)==2: raise KeyboardInterrupt('sync error')
        recorder=p.TimingRecorder(clock=clock,synchronize=sync)
        with self.assertRaisesRegex(RuntimeError,'native primary') as caught:
            with recorder.span('native'): raise RuntimeError('native primary')
        self.assertEqual(recorder.stack,[])
        self.assertTrue(any('sync error' in note for note in caught.exception.__notes__))


    def test_profile_authoring_rejects_changed_old_trace_case(self):
        p=api(self); campaign=json.loads((ROOT/'configs/neural/experiments/sustained-v1/campaign.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for relative in p.profile_source_paths(campaign):
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            target=root/'configs/cases/neural/task9-selection-case-04-standalone.json'
            target.write_bytes(target.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'historical'):
                p.author_profile_inputs(root,campaign)
            self.assertFalse((root/p.PLAN_PATH).exists())

    def test_trace_total_counts_serialized_episode_envelope(self):
        p=api(self)
        store=p.TraceStore(max_total_bytes=10)
        with self.assertRaisesRegex(ValueError,'total'):
            store.add('episode',{})
        self.assertEqual(store.episodes,{})

    def test_failed_timing_clock_cannot_prevent_resource_close(self):
        p=api(self); closed=[]
        def dead_clock(): raise RuntimeError('clock unavailable')
        collector=SimpleNamespace(driver=SimpleNamespace(reset=lambda **k:(),step=lambda a:()),policy=SimpleNamespace(act=lambda *a,**k:(),values=lambda a:()),controller_rng=SimpleNamespace(choice=lambda a:a[0]),collect=lambda **k:None)
        runtime=SimpleNamespace(collector=collector,policy=collector.policy,optimizer=None,close=lambda:closed.append(1),torch=None)
        ledger=p.EffectLedger('test',DurableLedger(),deadline=100.,clock=dead_clock)
        with self.assertRaisesRegex(KeyboardInterrupt,'primary') as caught:
            with p.instrument_runtime(runtime,SimpleNamespace(device='cpu'),ledger,p.TimingRecorder(clock=dead_clock),operation='trace'):
                raise KeyboardInterrupt('primary')
        self.assertEqual(closed,[1])
        self.assertTrue(any('clock unavailable' in note for note in caught.exception.__notes__))
