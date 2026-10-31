"""Lazy, single-job guest producer for the reviewed sustained campaign.

The host owns authority and durable whole-job reservations. This module only
consumes a verified context; importing it never resolves an optional backend.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager, ExitStack, nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    _root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_root/'src'), str(_root)]

from scripts.neural.gpu_runtime_proof_helpers import sha256_file, write_json_exclusive
from scripts.neural.sustained_profile import _patch, _note

CONTEXT_SCHEMA = 'spire-learn-sustained-job-context-v1'
RECEIPT_SCHEMA = 'spire-learn-sustained-job-receipt-v1'
EFFECT_KINDS = ('environment_attempts','sample_rows','optimizer_attempts','reset_slots','fixed_random_choices')
FILES = ('metadata.json','model.pt','training.pt')
OPTIONAL = ('torch','numpy','tensordict','torchrl','gymnasium','spire_lab')


def json_value(value):
    from spire_learn.neural.artifacts import _json_value
    return _json_value(value)


def integer(value, label):
    if type(value) is not int or value < 0: raise ValueError(label+' must be a nonnegative integer')
    return value


def finite(value, label):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError(label+' must be finite and nonnegative')
    return float(value)


def guest_path(value):
    # The production namespace is literal. Pure tests project only filesystem
    # access through the same existing boundary as containment verification.
    from scripts.neural.task9_study_controls import _guest_path
    return _guest_path(value)


def logical_path(root,value):
    path=Path(value);root=Path(root)
    if not path.is_relative_to(root):raise ValueError('runner path escaped projected source root')
    return str(Path('/learner')/path.relative_to(root))


def publish_profile_result(path,result):
    """Separate large result envelope; never widen small control/artifact codecs."""
    raw=json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)+1>32*1024**2:raise ValueError('profile result exceeds separate 32MiB publication allowance')
    write_json_exclusive(path,result)


def checkpoint_inventory(path):
    path=Path(path)
    if path.is_symlink() or not path.is_dir() or {p.name for p in path.iterdir()} != set(FILES):
        raise ValueError('checkpoint requires exactly three regular files')
    result={}
    for name in FILES:
        file=path/name
        if file.is_symlink() or not file.is_file() or not 0 < file.stat().st_size <= 256*1024**2:
            raise ValueError('unsafe checkpoint file')
        result[name]={'sha256':sha256_file(file),'size_bytes':file.stat().st_size}
    return result


class EffectLedger:
    """Reservations precede each real call; a failed boundary permanently halts."""
    def __init__(self, job_id, external, *, deadline, clock=time.monotonic, limits):
        if set(limits)!=set(EFFECT_KINDS): raise ValueError('effect limits incomplete')
        self.limits={k:integer(v,k) for k,v in limits.items()}
        if any(not callable(getattr(external,k,None)) for k in ('reserve','settle')):
            raise TypeError('effect observer requires plain reserve/settle callables')
        self.external=external; self.job_id=job_id; self.deadline=finite(deadline,'deadline');self.clock=clock
        self.sequence=0;self.first_error=None
        self.totals={k:dict(reserved=0,attempted=0,delivered=0,uncertain=0) for k in EFFECT_KINDS}
        self.categories={}
    def check(self):
        if self.first_error is not None: raise self.first_error
        if finite(self.clock(),'clock')>=self.deadline: raise TimeoutError('job deadline exhausted before effect')
    def halt(self,error):
        if self.first_error is None:self.first_error=error
    @contextmanager
    def reservation(self,kind,count,label):
        self.check();count=integer(count,'reservation');total=self.totals[kind]
        if total['reserved']+count>self.limits[kind]: raise RuntimeError('effect reservation ceiling: '+kind)
        self.sequence+=1;total['reserved']+=count
        row=dict(schema='spire-learn-profile-effect-v1',operation_id=f'{self.job_id}:{self.sequence:06d}',kind=kind,label=label,reserved=count)
        lease=dict(attempted=0,delivered=0,uncertain=0,_reserved=count);primary=None
        try:
            self.external.reserve(dict(row));yield lease
        except BaseException as error:
            primary=error;self.halt(error);raise
        finally:
            cat=self.categories.setdefault(label,dict(reserved=0,attempted=0,delivered=0,uncertain=0));cat['reserved']+=count
            for k in ('attempted','delivered','uncertain'):total[k]+=lease[k];cat[k]+=lease[k]
            try:self.external.settle({**row,**{k:lease[k] for k in ('attempted','delivered','uncertain')},'unused':count-lease['attempted']})
            except BaseException as error:
                self.halt(error)
                if primary is not None:_note(primary,'effect settlement also failed',error)
                else:raise
    def invoke(self,lease,count,function):
        self.check();count=integer(count,'attempt')
        if lease['attempted']+count>lease['_reserved']:raise RuntimeError('effect lease exhausted')
        lease['attempted']+=count
        try:result=function()
        except BaseException as error:lease['uncertain']+=count;self.halt(error);raise
        lease['delivered']+=count;return result
    def call(self,kind,count,label,function):
        with self.reservation(kind,count,label) as lease:return self.invoke(lease,count,function)
    def snapshot(self):return {k:dict(v) for k,v in self.totals.items()}


class Journal:
    """Bounded append-only batches; crash uncertainty is covered by host lease."""
    def __init__(self,output):self.output=Path(output);self.pending=[];self.open={};self.count=0;self.parts=[]
    def append(self,record):
        self.count+=1
        if self.count>65536:raise RuntimeError('effect journal record guard')
        self.pending.append(json_value(record))
        if len(self.pending)>=512:self.flush()
    def reserve(self,record):
        key=record['operation_id']
        if key in self.open:raise RuntimeError('duplicate effect ID')
        self.open[key]=record;self.append({'event':'reserve',**record})
    def settle(self,record):
        key=record['operation_id']
        original=self.open.pop(key)
        if any(record[k]!=v for k,v in original.items()):raise RuntimeError('effect identity changed')
        self.append({'event':'settle',**record})
    def flush(self):
        if not self.pending:return
        path=self.output/f'effects-{len(self.parts):04d}.json'
        write_json_exclusive(path,{'records':self.pending})
        self.parts.append({'path':path.name,'sha256':sha256_file(path),'size_bytes':path.stat().st_size});self.pending=[]


@contextmanager
def owned_runtime(runtime):
    """Close exactly once even after runner/setup/control failure; retain cause."""
    original=runtime.close;closed=False;primary=None
    def close():
        nonlocal closed
        if not closed:closed=True;return original()
    runtime.close=close
    try:yield runtime
    except BaseException as error:primary=error;raise
    finally:
        try:close()
        except BaseException as error:
            if primary is not None:_note(primary,'runtime close also failed',error)
            else:raise
        finally:runtime.close=original


@contextmanager
def effects(runtime,config,ledger,*,training):
    """Only operational effect wrappers; no synchronized timing observer."""
    with ExitStack() as stack:
        driver=runtime.collector.driver;policy=runtime.collector.policy;rng=runtime.collector.controller_rng
        def reset(*args,**kwargs):
            n=sum(kwargs['reset_mask']) if kwargs.get('reset_mask') is not None else len(kwargs['seeds'])
            return ledger.call('reset_slots',n,'reset',lambda:original_reset(*args,**kwargs))
        original_reset=_patch(stack,driver,'reset',reset)
        def step(actions):return ledger.call('environment_attempts',len(actions),'native-step',lambda:original_step(actions))
        original_step=_patch(stack,driver,'step',step)
        def act(frames,**kwargs):
            kind=getattr(runtime,'policy_kind','checkpoint')
            label='stochastic_actor_rows' if training else 'stochastic_reference_rows' if kind=='legal-random' else 'deterministic_reference_rows' if kind=='end-turn-last-v1' else 'deterministic_actor_rows'
            return ledger.call('sample_rows',len(frames),label,lambda:original_act(frames,**kwargs))
        original_act=_patch(stack,policy,'act',act)
        def choice(rows):return ledger.call('fixed_random_choices',1,'fixed_controller_random_choice',lambda:original_choice(rows))
        original_choice=_patch(stack,rng,'choice',choice)
        lease=None
        if getattr(runtime,'optimizer',None) is not None:
            def optimizer_step(*args,**kwargs):
                if lease is None:raise RuntimeError('optimizer lacks reservation')
                return ledger.invoke(lease,1,lambda:original_step_optimizer(*args,**kwargs))
            original_step_optimizer=_patch(stack,runtime.optimizer,'step',optimizer_step)
        if training:
            def update(rollout):
                nonlocal lease
                n=sum(record.actor_training_mask for record in rollout.records)
                cap=config.ppo.epochs*math.ceil(n/config.ppo.minibatch_rows)
                with ledger.reservation('optimizer_attempts',cap,'training_update') as reservation:
                    lease=reservation
                    try:return original_update(rollout)
                    finally:lease=None
            original_update=_patch(stack,runtime,'update',update)
        yield


def process_evidence():
    """Inspect this process, never a parent shell's module/thread settings."""
    modules={}
    for name in OPTIONAL:
        module=sys.modules.get(name)
        if module is not None:
            version=getattr(module,'__version__',None)
            modules[name]={'version':None if version is None else str(version),'file':str(getattr(module,'__file__',''))}
    torch=sys.modules.get('torch')
    threads=None if torch is None else {'intraop':int(torch.get_num_threads()),'interop':int(torch.get_num_interop_threads())}
    return {'pid':os.getpid(),'python':sys.executable,'modules':modules,'threads':threads}


_thread_module=None


def default_factory(config,*,purpose,checkpoint=None,policy_kind='checkpoint',action_selection='seeded-sampling'):
    """Initialize neural process threads once; stateless references stay Torch-free."""
    from spire_learn.neural.runner import create_runtime
    if purpose=='evaluate' and policy_kind in ('legal-random','end-turn-last-v1'):
        return create_runtime(config,purpose=purpose,checkpoint=checkpoint,policy_kind=policy_kind,action_selection=action_selection)
    from spire_learn.neural.device import plan_runtime_device,require_runtime_execution_gate
    require_runtime_execution_gate(plan_runtime_device(config))
    import torch
    global _thread_module
    if _thread_module is not torch:
        torch.set_num_threads(4);torch.set_num_interop_threads(1);_thread_module=torch
    if torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=1:raise RuntimeError('executing Torch threads differ before factory')
    runtime=create_runtime(config,purpose=purpose,checkpoint=checkpoint,policy_kind=policy_kind,action_selection=action_selection)
    try:
        if torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=1:raise RuntimeError('executing Torch threads differ after factory')
    except BaseException as error:
        try:runtime.close()
        except BaseException as close_error:_note(error,'factory thread verification close also failed',close_error)
        raise
    return runtime


def _dependencies(supplied):
    from spire_learn.neural.runner import train,evaluate_policy
    value={'factory':default_factory,'train':train,'evaluate':evaluate_policy}
    supplied=dict(supplied or {})
    if set(supplied)-set(value):raise ValueError('unknown runtime dependency')
    value.update(supplied)
    if any(not callable(v) for v in value.values()):raise TypeError('runtime dependencies must be plain callables')
    return value


def _selected_config(root,campaign,context):
    from spire_learn.neural.config import load_neural_config
    if context.get('schema')!=CONTEXT_SCHEMA:raise ValueError('job context schema differs')
    if context['kind']=='segment':
        work=next(r for r in campaign['training']['runs'] if r['policy_id']==context['replica'])
        binding=work['config'];relative=binding['path'];raw_hash=binding['raw_sha256']
    else:
        work=next(r for r in campaign['selection']['case_bindings'] if r['case_id']==context['operation']['case_id'])
        relative=work['config_path'];raw_hash=work['config_raw_sha256'];binding={'config_sha256':work['config_sha256']}
    if sha256_file(root/relative)!=raw_hash:raise ValueError('selected raw config changed')
    config=load_neural_config(root/relative)
    if config.config_sha256!=binding['config_sha256']:raise ValueError('selected canonical config changed')
    expected_output=root/work['output_root'].removeprefix('/learner/')
    if str(config.output_root)!=str(expected_output) or guest_path(context['guest_output_root'])!=expected_output:
        raise ValueError('fixed logical output root differs')
    return work,config


def _parent(context,config,*,resume):
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.runner import _selected_identity,_checkpoint_recovery
    binding=context['parent'];path=guest_path(binding['path'])
    if checkpoint_inventory(path)!=json_value(binding['inventory']):raise ValueError('parent checkpoint inventory changed')
    if resume and context.get('predecessor') is None:raise ValueError('resume requires preceding receipt binding')
    loaded=preflight_checkpoint(path,_selected_identity(config),purpose='resume' if resume else 'warm_start')
    prefix={}
    if resume:
        _,prefix=_checkpoint_recovery(loaded,config)
    if prefix!=json_value(context['prefix']):raise ValueError('incoming checkpoint prefix differs')
    return path,prefix


def report_files(report,output):
    path=Path(report.run_dir)
    if path.parent!=Path(output) or path.is_symlink() or not path.is_dir():raise ValueError('runner output escaped fixed logical root')
    result={}
    for name in ('summary.json','lifecycle.json','events.jsonl'):
        file=path/name
        if file.is_symlink() or not file.is_file():raise ValueError('runner report file missing/unsafe')
        result[name]={'sha256':sha256_file(file),'size_bytes':file.stat().st_size}
    return {'path':str(path),'inventory':result}


def reconcile_segment(config,context,summary,metadata,effect_totals,*,elapsed):
    """Validate save-boundary state, local collection, cumulative optimizer once."""
    from spire_learn.neural.runner import _checkpoint_recovery,_PROGRESS_FIELDS
    from types import SimpleNamespace
    _,saved=_checkpoint_recovery(SimpleNamespace(metadata=metadata),config)
    prefix=context['prefix'];delta={}
    for key in _PROGRESS_FIELDS:
        previous=integer(prefix.get(key,0),'prefix '+key);current=integer(saved[key],'saved '+key)
        if current<previous:raise ValueError('checkpoint counter moved backwards')
        delta[key]=current-previous
    if not 0<=saved['elapsed_seconds']-prefix.get('elapsed_seconds',0.)<=elapsed:
        raise ValueError('checkpoint elapsed prefix exceeds enclosing invocation')
    if metadata.get('parent_checkpoint')!={'path':context['parent']['path']}:raise ValueError('checkpoint parent mount differs')
    expected_prefix={'environment_slots':prefix.get('attempted_environment_slots',0),'learned_decisions':prefix.get('learned_decisions',0),
        'optimizer_attempted_steps':prefix.get('optimizer_attempted_steps',0),'optimizer_accepted_steps':prefix.get('optimizer_accepted_steps',0),'elapsed_seconds':prefix.get('elapsed_seconds',0.)}
    if summary.get('cumulative_prefix')!=expected_prefix:raise ValueError('report incoming prefix differs')
    for key in _PROGRESS_FIELDS-{'optimizer_attempted_steps','optimizer_accepted_steps','optimizer_uncertain_steps','update_count'}:
        if summary['collection'].get(key)!=delta[key]:raise ValueError('local collection disagrees with saved delta: '+key)
    for source,target in (('attempted_steps','optimizer_attempted_steps'),('accepted_steps','optimizer_accepted_steps'),('uncertain_steps','optimizer_uncertain_steps'),('updates','update_count')):
        if summary['optimizer'].get(source)!=saved[target]:raise ValueError('cumulative optimizer differs')
    for kind,attempted,delivered in (('environment_attempts','attempted_environment_slots','delivered_environment_slots'),('reset_slots','attempted_reset_slots','delivered_reset_slots'),('optimizer_attempts','optimizer_attempted_steps','optimizer_accepted_steps')):
        row=effect_totals[kind]
        if row['attempted']!=delta[attempted] or row['delivered']!=delta[delivered]:raise ValueError('effect delivery disagrees with real runner: '+kind)
    if (delta['uncertain_environment_slots'] or delta['uncertain_reset_slots'] or delta['optimizer_uncertain_steps']
        or any(v['uncertain'] for v in effect_totals.values()) or summary['collection'].get('faults') or summary.get('fault')):
        raise RuntimeError('accepted or uncertain effect fault')
    if effect_totals['sample_rows']['attempted']<delta['learned_decisions']:raise ValueError('actor effect rows below learned decisions')
    if delta['attempted_reset_slots']>delta['attempted_environment_slots']+16:raise ValueError('reset slots exceed actual action delta plus initial slots')
    return saved,delta


def run_segment(root: Path,campaign: Mapping[str,object],context: Mapping[str,object],*,dependencies=None,clock=time.monotonic)->dict:
    from spire_learn.neural.runner import RuntimeOperationLimits,_selected_identity
    from spire_learn.neural.checkpoint import preflight_checkpoint
    root=Path(root);started=finite(clock(),'start');work,config=_selected_config(root,campaign,context)
    index=integer(context['segment_index'],'segment index')
    parent,prefix=_parent(context,config,resume=index>0)
    reservation=context['reservation'];ledger=EffectLedger(context['job_id'],context['effect_ledger'],deadline=context['deadline'],clock=clock,limits=reservation)
    deps=_dependencies(dependencies);ledger.check()
    runtime=deps['factory'](config,purpose='resume' if index else 'warm_start',checkpoint=parent)
    with owned_runtime(runtime),effects(runtime,config,ledger,training=True):
        if json_value(runtime.recovered_progress)!=json_value(prefix):raise ValueError('factory restored prefix differs')
        max_updates=campaign['budgets']['segment'][config.mode+'_updates']
        runtime.operation_limits=RuntimeOperationLimits(max_updates=prefix.get('update_count',0)+max_updates,
            monotonic_deadline=context['deadline'],max_environment_actions=prefix.get('attempted_environment_slots',0)+reservation['environment_attempts'])
        report=deps['train'](config,runtime=runtime,clock=clock)
    if ledger.first_error is not None:raise ledger.first_error
    elapsed=finite(clock()-started,'segment elapsed')
    if report.exit_code!=0 or report.summary.get('status')!='complete':raise RuntimeError('accepted trainer did not complete: '+str(report.summary.get('stop_reason')))
    if clock()>context['whole_deadline']:raise TimeoutError('segment whole deadline exceeded')
    files=report_files(report,guest_path(context['guest_output_root']));path=Path(report.run_dir)/'checkpoint-final'
    loaded=preflight_checkpoint(path,_selected_identity(config),purpose='resume')
    saved,delta=reconcile_segment(config,context,report.summary,loaded.metadata,ledger.snapshot(),elapsed=elapsed)
    return {'schema':RECEIPT_SCHEMA,'kind':'segment','job_id':context['job_id'],'source_sha256':context['source_sha256'],
        'status':'complete','elapsed_seconds':elapsed,'report':json_value(report.summary),'report_files':files,
        'checkpoint':{'path':str(path),'inventory':checkpoint_inventory(path),'counters':saved},
        'parent':json_value(context['parent']),'predecessor':json_value(context.get('predecessor')),'delta':delta,
        'effects':ledger.snapshot(),'effect_categories':ledger.categories,'process':process_evidence()}


def _evaluate_one(root,campaign,context,*,dependencies=None,clock=time.monotonic):
    from spire_learn.neural.runner import RuntimeOperationLimits
    from scripts.neural.task9_study_runtime import _reference_import_guard
    root=Path(root);started=clock();_,config=_selected_config(root,campaign,context)
    row=context['operation'];reference=row['availability']=='reference'
    guard=_reference_import_guard() if reference else nullcontext()
    with guard as imports:
        deps=_dependencies(dependencies);parent=None
        if not reference:parent,_=_parent({**context,'prefix':{}},config,resume=False)
        ledger=EffectLedger(row['operation_id'],context['effect_ledger'],deadline=context['deadline'],clock=clock,
            limits=context['reservation']);ledger.check()
        kind=row['policy_id'] if reference else 'checkpoint'
        runtime=deps['factory'](config,purpose='evaluate',checkpoint=parent,policy_kind=kind,
            action_selection='seeded-sampling' if kind=='legal-random' else 'deterministic')
        with owned_runtime(runtime),effects(runtime,config,ledger,training=False):
            runtime.operation_limits=RuntimeOperationLimits(monotonic_deadline=context['deadline'],max_environment_actions=256)
            report=deps['evaluate'](config,runtime=runtime,clock=clock,allow_final=False)
        if ledger.first_error is not None:raise ledger.first_error
        status=report.summary.get('status')
        if status not in ('complete','incomplete') or report.exit_code!=(0 if status=='complete' else 1) or report.summary.get('fault'):
            raise RuntimeError('evaluation runner failed')
        files=report_files(report,guest_path(context['guest_output_root']))
        return {'schema':RECEIPT_SCHEMA,'kind':'evaluation','job_id':context['job_id'],'operation_id':row['operation_id'],
            'source_sha256':context['source_sha256'],'status':'complete','report':json_value(report.summary),'report_files':files,
            'effects':ledger.snapshot(),'effect_categories':ledger.categories,'elapsed_seconds':finite(clock()-started,'eval elapsed'),
            'process':process_evidence(),'reference_import_guard':imports.check() if reference else None}


def run_evaluation(root: Path,campaign: Mapping[str,object],context: Mapping[str,object],*,dependencies=None,clock=time.monotonic)->dict:
    """One fixed CPU batch (one scheduled operation); references use a fresh child."""
    # Every host job is a fresh container. Even within that fresh process use an
    # import-guarded child for stateless references; checkpoint jobs run locally.
    if context['operation']['availability']!='reference' or dependencies is not None:
        return _evaluate_one(root,campaign,context,dependencies=dependencies,clock=clock)
    from scripts.neural.gpu_runtime_proof_helpers import _bounded_subprocess_run
    deadline=finite(context['deadline'],'deadline');remaining=deadline-clock()
    if remaining<=0:raise TimeoutError('reference child deadline exhausted')
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'_reference','--root',str(root),'--context','/control/job.json']
    transport={'argv':argv,'timeout_seconds':remaining,'status':'attempted'};primary=None
    try:
        child=_bounded_subprocess_run(argv,capture_output=True,text=True,timeout=remaining,check=False,env=dict(os.environ),output_limit_bytes=65536,
            progress_check=lambda: (_ for _ in ()).throw(TimeoutError('reference deadline')) if clock()>=deadline else None)
        transport.update(status='returned',exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,
            timed_out=getattr(child,'timed_out',False),stdout_truncated=getattr(child,'stdout_truncated',False),stderr_truncated=getattr(child,'stderr_truncated',False))
    except BaseException as error:
        primary=error;transport.update(status='unreturned',error={'type':type(error).__name__,'message':str(error)[:512]});raise
    finally:
        try:write_json_exclusive(guest_path(context['guest_output_root'])/'reference-child-transport.json',transport)
        except BaseException as error:
            if primary is None:raise
            _note(primary,'reference transport retention also failed',error)
    if child.returncode or getattr(child,'timed_out',False) or getattr(child,'stdout_truncated',False) or getattr(child,'stderr_truncated',False):
        raise RuntimeError('reference child failed or timed out')
    from scripts.neural.task9_study_controls import _json
    return _json(guest_path(context['guest_output_root'])/'job-receipt.json')


def _guest_entry(root,context_path,*,dependencies=None,clock=time.monotonic,reference_child=False):
    from scripts.neural.sustained_controls import verify_guest_context
    from scripts.neural.task9_study_controls import _json
    context,campaign=verify_guest_context(Path(root),Path(context_path),clock=clock)
    output=guest_path(context['guest_output_root'])
    if not reference_child:
        scratch=output/'.runtime';scratch.mkdir()
        from scripts.neural.task9_study_controls import SCRATCH
        for name in SCRATCH.values():(scratch/name).mkdir()
    journal=Journal(output);transient={**context,'effect_ledger':journal};primary=None
    try:
        if context['kind']=='profile':
            from scripts.neural.sustained_profile import run_profile
            # Exact five raw keys plus transient observer, never the outer context.
            payload={**context['profile_payload'],'effect_ledger':journal}
            profile_deps={'factory':default_factory};profile_deps.update(dependencies or {})
            result=run_profile(Path(root),campaign,payload,dependencies=profile_deps,clock=clock)
        elif context['kind']=='segment':result=run_segment(Path(root),campaign,transient,dependencies=dependencies,clock=clock)
        elif reference_child:result=_evaluate_one(Path(root),campaign,transient,dependencies=dependencies,clock=clock)
        else:result=run_evaluation(Path(root),campaign,transient,dependencies=dependencies,clock=clock)
    except BaseException as error:primary=error;raise
    finally:
        try:journal.flush()
        except BaseException as error:
            if primary is not None:_note(primary,'effect journal finalization also failed',error)
            else:raise
        if primary is not None:
            try:write_json_exclusive(output/'guest-failure.json',{'job_id':context['job_id'],'error':{'type':type(primary).__name__,'message':str(primary)[:512]},
                'journal':journal.parts,'unresolved_effect_ids':sorted(journal.open)})
            except BaseException as error:_note(primary,'guest failure retention also failed',error)
    # The reference child already publishes this immutable output. Its parent
    # reads it without publishing a second copy or creating duplicate journals.
    path=output/'job-receipt.json'
    if not path.exists():
        # Production root is /learner, so these retain the literal runner paths.
        # The projection boundary gives pure tests the identical wire namespace.
        if 'report_files' in result:result['report_files']['path']=logical_path(root,result['report_files']['path'])
        if 'checkpoint' in result:result['checkpoint']['path']=logical_path(root,result['checkpoint']['path'])
        result={**result,'journal':journal.parts}
        if context['kind']=='profile':result['process']=process_evidence()
        if context['kind']=='profile':publish_profile_result(path,result)
        else:write_json_exclusive(path,result)
    elif context['kind']!='evaluation' or context['operation']['availability']!='reference':
        raise FileExistsError('job receipt already exists')
    return {'status':'complete','job_id':context['job_id'],'receipt_path':str(path),'receipt_sha256':sha256_file(path)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('_guest','_reference'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--context',type=Path,required=True)
    args=parser.parse_args(argv)
    from scripts.neural.sustained_controls import signal_scope
    with signal_scope():result=_guest_entry(args.root,args.context,reference_child=args.command=='_reference')
    print(json.dumps(result,separators=(',',':'),allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
