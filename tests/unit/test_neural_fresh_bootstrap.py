"""Fresh parent ownership and no-training boundary checks using strict fakes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import importlib.abc
import os
import sys
import json
from unittest.mock import patch

from spire_learn.neural.config import load_neural_config


class FreshBootstrapTests(unittest.TestCase):
    def test_correction_requires_preserved_zero_effect_failure_and_cleanup(self):
        from scripts.neural.fresh_curriculum_bootstrap import verify_initial_fixture_failure, DOMAIN
        import hashlib
        process={'source_postcheck':True,'protected_postcheck':True,'commands':[
            {'argv':['docker','--config','c','--host','h','start'],'stderr':'ValueError: fresh model forward output invalid'},
            {'argv':['docker','--config','c','--host','h','rm','a'*64],'exit_code':0},
            {'argv':['docker','--config','c','--host','h','ps','id='+'a'*64],'exit_code':0,'stdout':''}]}
        proof={'status':'failed_probe_fixture','tool_os_exit':1,'environment_actions':0,'resets':0,
               'samples':0,'optimizer_steps':0,'gpu_work':False,'checkpoint_published':False}
        with tempfile.TemporaryDirectory() as temp:
            domain=Path(temp)/DOMAIN;domain.mkdir(parents=True)
            process_path=domain/'bootstrap-process.json';process_path.write_text(json.dumps(process))
            proof['process_sha256']=hashlib.sha256(process_path.read_bytes()).hexdigest()
            proof_path=domain/'bootstrap-initial-failure.json';proof_path.write_text(json.dumps(proof))
            self.assertEqual(verify_initial_fixture_failure(temp),proof)
            for key,value in (('samples',1),('tool_os_exit',0),('gpu_work',True)):
                proof_path.write_text(json.dumps(dict(proof,**{key:value})))
                with self.assertRaises(ValueError):verify_initial_fixture_failure(temp)
            proof_path.write_text(json.dumps(proof));process_path.write_text('{}')
            with self.assertRaises(ValueError):verify_initial_fixture_failure(temp)

    def test_actual_probe_fixture_has_two_distinct_legal_actions(self):
        from scripts.neural.fresh_curriculum_bootstrap import initial_probe_frame
        from spire_learn.neural.schema import parse_common_descriptor,candidates
        schema=parse_common_descriptor(json.loads(Path('tests/fixtures/neural/common-afb51f5.json').read_text()),
            revision='afb51f5f222b4c4331ed34067c5eebeda606db91')
        frame=initial_probe_frame(schema)
        self.assertEqual([(row.index,row.verb) for row in candidates(frame,schema)],
                         [(2,'play_card'),(1023,'end_turn')])

    def test_parent_clock_bounds_preflight_and_preserves_cleanup_reserve(self):
        try:
            from scripts.neural.fresh_curriculum_bootstrap import bootstrap_window
        except ImportError as error:self.fail(f'parent bootstrap clock API missing: {error}')
        self.assertEqual(bootstrap_window('100.0',120.0),(100.0,180.0))
        for epoch,now in ((None,100.),('nan',100.),('200',100.),('100',141.),('100',181.)):
            with self.subTest(epoch=epoch,now=now),self.assertRaises((ValueError,TimeoutError)):
                bootstrap_window(epoch,now)

    def test_bad_guest_controls_refuse_before_optional_imports(self):
        from scripts.neural.fresh_curriculum_bootstrap import _guest
        attempts=[]
        class Guard(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split('.')[0] in {'torch','torchrl','tensordict','numpy','gymnasium','spire_lab'}:
                    attempts.append(fullname)
                    raise AssertionError('optional import before guest admission')
        guard=Guard();sys.meta_path.insert(0,guard)
        try:
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'bad','NVIDIA_VISIBLE_DEVICES':'all','FRESH_BOOTSTRAP_CONTEXT':'bad'}):
                with self.assertRaisesRegex(ValueError,'controls'):_guest()
        finally:sys.meta_path.remove(guard)
        self.assertEqual(attempts,[])

    def test_real_model_probe_routes_through_collector_owned_driver(self):
        from scripts.neural.fresh_curriculum_bootstrap import _check_cpu_model
        calls=[]
        class Value:
            def masked_select(self,mask):return self
            def numel(self):return 2
            def all(self):return self
            def item(self):return 1
            def sum(self):return self
            def __add__(self,other):return self
            def backward(self):calls.append('backward')
        class Policy:
            def __call__(self,batch):calls.append('forward');return SimpleNamespace(logits=Value(),values=Value())
            def parameters(self):return [SimpleNamespace(grad=Value(),numel=lambda:7)]
            def zero_grad(self,**kwargs):calls.append(('zero_grad',kwargs))
        expected=object()
        runtime=SimpleNamespace(collector=SimpleNamespace(driver=SimpleNamespace(schema=expected)),policy=Policy())
        frame=SimpleNamespace(tensors={'values':SimpleNamespace(data=b'123')})
        def authored(schema):self.assertIs(schema,expected);return frame
        def encode(frames,schema,**kwargs):
            self.assertEqual(frames,(frame,));self.assertIs(schema,expected)
            self.assertEqual(kwargs,{'device':'cpu'});return SimpleNamespace(action_mask=object())
        result=_check_cpu_model(runtime,torch=SimpleNamespace(isfinite=lambda v:Value(),count_nonzero=lambda g:Value()),
                               encode_frames=encode,authored_frame=authored)
        self.assertEqual(result['parameter_count'],7)
        self.assertEqual(result['frame_bytes'],3)
        self.assertEqual(calls,['forward','backward',('zero_grad',{'set_to_none':True})])

    def api(self):
        try:
            from scripts.neural.fresh_curriculum_bootstrap import initialize_parent
        except ImportError as error:
            self.fail(f"fresh bootstrap API missing: {error}")
        return initialize_parent

    def fixture(self, *, dirty=False, save_error=None, close_error=None, mismatch=False):
        config = load_neural_config(Path('configs/neural/experiments/curriculum-v4/bootstrap.toml'))
        calls = []
        policy = SimpleNamespace(_behavior_policy_version=0, state_dict=lambda: {'weight': 0.25})
        runtime = SimpleNamespace(policy=policy, recovered_progress={}, parent_checkpoint=None,
            optimizer=SimpleNamespace(state_dict=lambda: {'state': {1: 'old'} if dirty else {}}),
            collector=SimpleNamespace(recovery_state=SimpleNamespace(next_seed_ordinal=0,
                next_decision_sequence=0, next_episode_ordinals=(0, 0, 0, 0))),
            _latest_counters=lambda: {'learned_decisions': 0, 'optimizer_accepted_steps': 0,
                                     'update_count': 0, 'elapsed_seconds': .1})
        def save(path, purpose):
            calls.append(('save', path, purpose))
            if save_error: raise save_error
        def close():
            calls.append(('close',))
            if close_error: raise close_error
        runtime.save, runtime.close = save, close
        def factory(actual, **kwargs):
            self.assertIs(actual, config)
            self.assertEqual(kwargs, {'purpose': 'train'})
            calls.append(('factory',))
            return runtime
        def loader(path, identity, **kwargs):
            self.assertEqual(identity, config.accepted_identities[0])
            self.assertEqual(kwargs, {'purpose': 'resume', 'device': 'cpu'})
            calls.append(('load',))
            return SimpleNamespace(model_tensors={'weight': 0.5 if mismatch else .25},
                metadata={'training_state': {'optimizer_step': 0, 'behavior_policy_version': 0,
                    'counters': {'learned_decisions': 0, 'update_count': 0, 'elapsed_seconds': .1}}})
        dependencies = dict(create_runtime=factory, load_checkpoint=loader,
            equal=lambda a,b:a==b, check_model=lambda runtime: {'forward_backward': True},
            inventory=lambda path: {'metadata.json': {'sha256': 'a'*64, 'size_bytes': 1}})
        return config, calls, dependencies

    def test_fresh_initialization_saves_zero_progress_and_closes_once(self):
        initialize = self.api()
        config, calls, deps = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'initial-policy'
            result = initialize(config, path, **deps)
        self.assertEqual([call[0] for call in calls], ['factory','save','load','close'])
        self.assertEqual(result['status'], 'fresh_untrained_parent')
        self.assertEqual(result['training_exposure'], 0)
        self.assertEqual(result['model_checks'], {'forward_backward': True})

    def test_loader_immutable_counter_mapping_is_validated_without_mutation(self):
        from types import MappingProxyType
        initialize=self.api();config,calls,deps=self.fixture()
        loader=deps['load_checkpoint']
        def immutable_loader(*args,**kwargs):
            result=loader(*args,**kwargs)
            result.metadata['training_state']['counters']=MappingProxyType(result.metadata['training_state']['counters'])
            return result
        deps['load_checkpoint']=immutable_loader
        with tempfile.TemporaryDirectory() as directory:
            result=initialize(config,Path(directory)/'initial-policy',**deps)
        self.assertEqual(result['training_exposure'],0)
        self.assertEqual([c[0] for c in calls],['factory','save','load','close'])

    def test_stale_optimizer_refuses_save_and_still_closes_once(self):
        initialize = self.api()
        config,calls,deps=self.fixture(dirty=True)
        with self.assertRaisesRegex(ValueError,'fresh'):
            initialize(config,Path('/owned/initial'),**deps)
        self.assertEqual([c[0] for c in calls],['factory','close'])

    def test_primary_control_survives_cleanup_control_without_retry(self):
        initialize=self.api()
        primary=KeyboardInterrupt('primary save control')
        config,calls,deps=self.fixture(save_error=primary,close_error=SystemExit('close control'))
        with self.assertRaises(KeyboardInterrupt) as raised:
            initialize(config,Path('/owned/initial'),**deps)
        self.assertIs(raised.exception,primary)
        self.assertEqual([c[0] for c in calls],['factory','save','close'])

    def test_loaded_weight_mismatch_is_fatal_and_closes_once(self):
        initialize=self.api();config,calls,deps=self.fixture(mismatch=True)
        with self.assertRaisesRegex(ValueError,'weights'):
            initialize(config,Path('/owned/initial'),**deps)
        self.assertEqual([c[0] for c in calls],['factory','save','load','close'])

    def test_wrong_release_and_gpu_refuse_before_factory(self):
        initialize=self.api();config,calls,deps=self.fixture()
        for changed in (replace(config,device='cuda:0'),replace(config,environment_revision='0'*40)):
            with self.subTest(changed=changed.device),self.assertRaises(ValueError):
                initialize(changed,Path('/owned/initial'),**deps)
        self.assertEqual(calls,[])
