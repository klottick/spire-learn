"""Bounded sustained-campaign profiling, with lazy accepted runtime boundaries.

Import and input preparation are stdlib-only. Execution requires an explicit
phase context and job reservation. This module never grants runtime release.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import time

from spire_learn.neural.profiling import observe

PROFILE_CONFIG = 'configs/neural/experiments/sustained-v1/profile'
PROFILE_CASES = 'configs/cases/neural/sustained-v1/profile'
PLAN_PATH = PROFILE_CONFIG + '/profile.json'
OUTPUT_PREFIX = 'artifacts/development/neural-combat/experiments/sustained-v1/profile'
OLD_CASES = (0, 4, 1, 3)
LIMITS = dict(environment_attempts=8192, sample_rows=16384, optimizer_attempts=256,
              reset_slots=8248, fixed_random_choices=8192)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _binding(raw):
    return {'sha256':hashlib.sha256(raw).hexdigest(), 'size_bytes':len(raw)}


def _finite(value, label):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return float(value)


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def _path(root, relative):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('input path must be a confined relative path')
    root=Path(root).resolve(); path=root/relative
    current=root
    for part in Path(relative).parts:
        current=current/part
        if current.is_symlink(): raise ValueError('symlink input is forbidden')
    if not path.is_relative_to(root): raise ValueError('input escaped root')
    return path


def _note(primary, label, secondary):
    primary.add_note(f'{label}: {type(secondary).__name__}: {secondary}')


def profile_source_paths(campaign):
    return tuple([campaign['training']['runs'][i]['config']['path'] for i in (0,1)] +
        ['configs/neural/experiments/task9-v1/study.json',
         'configs/neural/experiments/task9-v1/selection/case-00-standalone.toml',
         campaign['training']['manifest']['path']] +
        [f'configs/cases/neural/task9-selection-case-{index:02d}-standalone.json' for index in OLD_CASES])


def _plan_skeleton(campaign):
    training=[{'operation_id':'profile-train-'+mode, 'mode':mode,
        'config':f'{PROFILE_CONFIG}/train-{mode}.toml', 'checkpoint_id':mode+'-b',
        'max_updates':2, 'environment_attempts':cap} for mode,cap in [('standalone',1024),('run',256)]]
    traces=[{'operation_id':f'profile-trace-{index:02d}-{anchor[-1]}',
        'old_case_index':index, 'checkpoint_id':anchor,
        'config':f'{PROFILE_CONFIG}/trace-{index:02d}-{anchor[-1]}.toml',
        'action_selection':'deterministic', 'max_actions':32} for index in OLD_CASES
        for anchor in ('standalone-a','standalone-b')]
    return {'schema':'spire-learn-sustained-profile-plan-v1',
        'catalog_config':PROFILE_CONFIG+'/catalog.toml',
        'catalog_case_ids':[f'sustained-train-standalone-{index:02d}' for index in range(78)],
        'training':training, 'traces':traces,
        'reservation_ceiling':{'environment_attempts':1614,'sample_rows':1792,
            'optimizer_attempts':80,'reset_slots':1383,'fixed_random_choices':256},
        'whole_wall_seconds':900,
        'interpretation':{'updates':'disposable; never a campaign parent',
            'native_timing':'public Python provider boundary includes native export; no separate Rust engine claim',
            'trace_selection':'frozen greedy CPU; no policy-generator draws',
            'qualifications':['RNG-001','TIME-001','simulator only']}}


def author_profile_inputs(root: Path, campaign: Mapping[str,object]):
    """Create narrow diagnostic copies, refusing every differing existing byte."""
    root=Path(root); sources={path:_path(root,path).read_bytes() for path in profile_source_paths(campaign)}
    study_path='configs/neural/experiments/task9-v1/study.json'
    if _binding(sources[study_path])['sha256']!=campaign['sources'][study_path]:
        raise ValueError('historical study input hash differs')
    study=json.loads(sources[study_path])
    for path,raw in sources.items():
        expected=campaign['files'].get(path,{}).get('raw_sha256') or campaign['sources'].get(path)
        if expected is None:
            expected=next(row['manifest_raw_sha256'] for row in study['selection']['case_bindings'] if row['manifest_path']==path)
        if _binding(raw)['sha256']!=expected: raise ValueError('historical profile source hash differs: '+path)
    files={}; plan=_plan_skeleton(campaign)
    def config(raw, target, manifest, output, *, budgets, rows=None, steps=None, cpu=False):
        text=raw.decode()
        replacements={'manifest_path':str(Path(os.path.relpath(manifest,Path(target).parent))),
            'output_root':str(Path(os.path.relpath(OUTPUT_PREFIX+'/'+output,Path(target).parent)))}
        if cpu: replacements['device']='cpu'
        for key,value in replacements.items():
            text=re.sub(r'^'+key+r' = .*$',key+' = '+json.dumps(value),text,flags=re.M)
        for key,value in budgets.items():
            text=re.sub(r'^'+key+r' = [0-9.]+$',f'{key} = {value}',text,flags=re.M)
        if rows is not None: text=re.sub(r'^rows = \d+$',f'rows = {rows}',text,flags=re.M)
        if steps is not None: text=re.sub(r'^steps = \d+$',f'steps = {steps}',text,flags=re.M)
        if cpu: text=re.sub(r'\n\[runtime\]\n(?:gpu_uuid = .*\n)?','\n',text)
        files[target]=(text.rstrip()+'\n').encode()
    train_manifest=campaign['training']['manifest']['path']
    for row in plan['training']:
        source=next(run['config']['path'] for run in campaign['training']['runs'] if run['mode']==row['mode'])
        expected=campaign['files'][source]
        if _binding(sources[source]) != {'sha256':expected['raw_sha256'],'size_bytes':expected['size_bytes']}:
            raise ValueError('training source differs from validated campaign')
        config(sources[source],row['config'],train_manifest,row['operation_id'],budgets={
            'max_environment_actions':row['environment_attempts'], 'max_training_decisions':row['environment_attempts'],
            'max_optimizer_steps':64 if row['mode']=='standalone' else 16,'max_wall_seconds':900.0})
    standalone=sources[campaign['training']['runs'][0]['config']['path']]
    config(standalone,plan['catalog_config'],train_manifest,'catalog',rows=1,steps=1,cpu=True,
        budgets={'max_environment_actions':78,'max_training_decisions':78,'max_optimizer_steps':1,'max_wall_seconds':900.0})
    template=sources['configs/neural/experiments/task9-v1/selection/case-00-standalone.toml']
    for index in OLD_CASES:
        source=f'configs/cases/neural/task9-selection-case-{index:02d}-standalone.json'
        manifest=f'{PROFILE_CASES}/case-{index:02d}.json'; files[manifest]=sources[source]
        for row in (row for row in plan['traces'] if row['old_case_index']==index):
            config(template,row['config'],manifest,row['operation_id'],budgets={
                'max_environment_actions':32,'max_training_decisions':32,'max_optimizer_steps':1,'max_wall_seconds':900.0})
    plan['source_inventory']={path:_binding(raw) for path,raw in sources.items()}
    plan['files']={path:_binding(raw) for path,raw in files.items()}
    files[PLAN_PATH]=json_bytes(plan)+b'\n'
    for path,raw in files.items():
        target=_path(root,path)
        if target.exists() and target.read_bytes()!=raw: raise ValueError('differing profile input: '+path)
    for path,raw in files.items():
        target=_path(root,path)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(raw)
    return {path:_binding(raw) for path,raw in files.items()}


def profile_plan(root: Path, campaign: Mapping[str,object]):
    """Read the already authored plan; no historical broad-artifact mount needed."""
    plan=json.loads(_path(root,PLAN_PATH).read_bytes())
    skeleton=_plan_skeleton(campaign)
    if set(plan)!=set(skeleton)|{'source_inventory','files'} or any(plan[key]!=value for key,value in skeleton.items()):
        raise ValueError('profile workload differs from fixed plan')
    expected_paths={plan['catalog_config'],*(row['config'] for row in plan['training']),
        *(row['config'] for row in plan['traces']),*(f'{PROFILE_CASES}/case-{i:02d}.json' for i in OLD_CASES)}
    if set(plan['files']) != expected_paths: raise ValueError('profile file inventory differs')
    for path,binding in plan['files'].items():
        if _binding(_path(root,path).read_bytes())!=binding: raise ValueError('profile input hash mismatch: '+path)
    from spire_learn.neural.config import load_neural_config
    configs={path:load_neural_config(_path(root,path)) for path in expected_paths if path.endswith('.toml')}
    for row in plan['training']:
        config=configs[row['config']]
        if config.mode!=row['mode'] or config.operation!='train' or config.device!='cuda:0' or config.budgets.max_environment_actions!=row['environment_attempts']:
            raise ValueError('profile training config differs')
    for row in plan['traces']:
        config=configs[row['config']]
        if config.device!='cpu' or config.operation!='evaluate' or config.budgets.max_environment_actions!=32 or config.rollout_rows!=1 or config.rollout_steps!=1:
            raise ValueError('profile trace config differs')
    return plan


def profile_input_inventory(root: Path, campaign: Mapping[str,object]):
    """Exact guest input files. Map these individual paths read-only, never artifacts/."""
    plan=profile_plan(root,campaign)
    inventory={**plan['files'],PLAN_PATH:_binding(_path(root,PLAN_PATH).read_bytes())}
    manifest=campaign['training']['manifest']['path']
    expected=campaign['files'][manifest]
    inventory[manifest]={'sha256':expected['raw_sha256'],'size_bytes':expected['size_bytes']}
    for binding in (*campaign['parents'].values(),campaign['diagnostic_checkpoints']['standalone-a']):
        for name,item in binding['inventory'].items(): inventory[binding['path']+'/'+name]=dict(item)
    for path,binding in inventory.items():
        if _binding(_path(root,path).read_bytes())!=binding: raise ValueError('profile input inventory mismatch: '+path)
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.checkpoint import preflight_checkpoint
    identity=load_neural_config(_path(root,plan['catalog_config'])).accepted_identities[0]
    for identifier,binding in {**campaign['parents'],**campaign['diagnostic_checkpoints']}.items():
        preflight_checkpoint(_path(root,binding['path']),identity,purpose='evaluate' if identifier=='standalone-a' else 'warm_start')
    return inventory


class TimingRecorder:
    """Nested wall/CPU spans; inclusive parents are never added to children."""
    def __init__(self, *, clock=time.monotonic, synchronize=None):
        self.clock=clock; self.synchronize=synchronize; self.rows={}; self.stack=[]
    @contextmanager
    def span(self,name):
        if len(self.rows)>=64 and name not in self.rows: raise ValueError('timing phase capacity reached')
        start=self.clock(); cpu=time.process_time(); state=[0.,0.]; sync_seconds=0.; primary=None
        self.stack.append(state)
        try:
            if self.synchronize:
                before=self.clock(); self.synchronize(); sync_seconds+=self.clock()-before
            yield
        except BaseException as error:
            primary=error; raise
        finally:
            try:
                if self.synchronize:
                    before=self.clock(); self.synchronize(); sync_seconds+=self.clock()-before
                wall=_finite(self.clock()-start,'span elapsed'); cpu_elapsed=_finite(time.process_time()-cpu,'CPU elapsed')
                row=self.rows.setdefault(name,{'count':0,'inclusive_wall_seconds':0.,'exclusive_wall_seconds':0.,
                    'inclusive_cpu_seconds':0.,'exclusive_cpu_seconds':0.,'synchronization_seconds':0.,
                    'synchronization':'device-synchronized-wall' if self.synchronize else 'CPU-wall-no-device-sync'})
                row['count']+=1; row['inclusive_wall_seconds']+=wall
                row['exclusive_wall_seconds']+=max(0.,wall-state[0]); row['inclusive_cpu_seconds']+=cpu_elapsed
                row['exclusive_cpu_seconds']+=max(0.,cpu_elapsed-state[1]); row['synchronization_seconds']+=sync_seconds
                if len(self.stack)>1:
                    self.stack[-2][0]+=wall; self.stack[-2][1]+=cpu_elapsed
            except BaseException as error:
                if primary is not None: _note(primary,'profile timing also failed',error)
                else: raise
            finally: self.stack.pop()
    def snapshot(self): return {key:dict(value) for key,value in self.rows.items()}


class EffectLedger:
    """Enforce guest effects inside a host-reserved job; callbacks may batch I/O.

    reserve(record) and settle(record) must raise on failure. Each unique ID is
    claimed before effects. settle includes unused reservation and conservative
    uncertainty. Callback failure halts dispatch. Failed effects never retry.
    """
    def __init__(self, phase_id, external, *, deadline, clock=time.monotonic, limits=None):
        if not isinstance(phase_id,str) or not phase_id: raise ValueError('phase_id is required')
        for name in ('reserve','settle'):
            if not callable(getattr(external,name,None)): raise TypeError('effect ledger needs reserve/settle callbacks')
        self.phase_id=phase_id; self.external=external; self.deadline=_finite(deadline,'deadline'); self.clock=clock
        self.limits=dict(LIMITS if limits is None else limits)
        if set(self.limits)!=set(LIMITS) or any(_integer(value,'limit')>LIMITS[key] for key,value in self.limits.items()):
            raise ValueError('context cannot widen profile limits')
        self.totals={kind:dict(reserved=0,attempted=0,delivered=0,uncertain=0) for kind in LIMITS}
        self.sequence=0; self.failed=False; self.categories={}; self.first_error=None
    def halt(self,error):
        self.failed=True
        if self.first_error is None: self.first_error=error
    def check(self):
        if self.failed: raise self.first_error
        if self.clock()>=self.deadline: raise TimeoutError('profile deadline exhausted before effect')
    @contextmanager
    def reservation(self,kind,count,label):
        self.check(); count=_integer(count,'reservation')
        if kind not in self.limits: raise ValueError('unknown effect kind')
        total=self.totals[kind]
        if total['reserved']+count>self.limits[kind]: raise ValueError('profile effect ceiling exhausted: '+kind)
        self.sequence+=1
        record={'schema':'spire-learn-profile-effect-v1','operation_id':f'{self.phase_id}:{self.sequence:06d}',
                'kind':kind,'label':label,'reserved':count}
        # Local charge survives uncertain callback delivery; never recycle IDs.
        total['reserved']+=count
        lease={'attempted':0,'delivered':0,'uncertain':0,'_reserved':count}
        primary=None
        try:
            self.external.reserve(dict(record))
            yield lease
        except BaseException as error:
            self.halt(error); primary=error; raise
        finally:
            for key in ('attempted','delivered','uncertain'): total[key]+=lease[key]
            category=self.categories.setdefault(label,dict(reserved=0,attempted=0,delivered=0,uncertain=0))
            category['reserved']+=count
            for key in ('attempted','delivered','uncertain'): category[key]+=lease[key]
            try: self.external.settle({**record,**{key:lease[key] for key in ('attempted','delivered','uncertain')},'unused':count-lease['attempted']})
            except BaseException as error:
                self.halt(error)
                if primary is not None: _note(primary,'effect settle also failed',error)
                else: raise
    def invoke(self,lease,count,function):
        self.check(); count=_integer(count,'effect invocation')
        if lease['attempted']+count>lease['_reserved']: raise ValueError('effect invocation exceeds reservation')
        lease['attempted']+=count
        try: result=function()
        except BaseException as error:
            lease['uncertain']+=count; self.halt(error); raise
        lease['delivered']+=count
        return result
    def call(self,kind,count,label,function):
        with self.reservation(kind,count,label) as lease:
            return self.invoke(lease,count,function)
    def snapshot(self): return {key:dict(value) for key,value in self.totals.items()}


class FramePayloadCounter:
    """Unique backing-byte identities among records retained by one trainer.

    Only identities are kept, not frame/byte references. Reset per runtime so
    released runtime allocations cannot cause cross-run identity reuse errors.
    """
    def __init__(self,*,max_buffers=1000000):
        self.seen=set(); self.bytes=0; self.max_buffers=max_buffers
    def add(self,records):
        for record in records:
            for frame in (record.source,record.successor):
                for buffer in frame.tensors.values():
                    identity=id(buffer.data)
                    if identity not in self.seen:
                        if len(self.seen)>=self.max_buffers: raise ValueError('frame byte identity capacity reached')
                        self.seen.add(identity); self.bytes+=len(buffer.data)


def _public_frame(schema,frame):
    from spire_learn.neural.schema import validate_frame
    validate_frame(frame,schema); result={}
    # The approved trace projection contains only public player/enemy fields.
    for name,spec in schema.tensors.items():
        if not spec.columns or name not in {'player','enemies'}: continue
        raw=frame.tensors[name].data; known=frame.tensors[spec.known_key].data
        rows=spec.shape[0] if len(spec.shape)==2 else 1; width=len(spec.columns)
        occupied=frame.tensors[spec.occupied_key].data if spec.occupied_key else bytes([1])*rows
        projected=[]
        for row in range(rows):
            fields={}
            for column_index,column in enumerate(spec.columns):
                offset=row*width+column_index; is_known=bool(occupied[row] and known[offset]); value=None
                if is_known:
                    value=struct.unpack_from('=q',raw,offset*8)[0]
                    if column.vocabulary is not None: value=schema.vocabularies[column.vocabulary][value]
                    elif column.reference is not None:
                        value={'row':value,'reference':column.reference}
                fields[column.field]={'known':is_known,'value':value,'unit':column.unit}
            projected.append({'index':row,'occupied':bool(occupied[row]),'fields':fields})
        result[name]=projected
    return result


def _public_referenced_rows(schema,frame,legal):
    """Decode only source-frame rows reachable from legal public references.

    Descriptor-named selector columns and fixed-group references use the same
    meaning as validated actor tensors. Known reference paths can resolve a
    selection through to its public card. Deduplication terminates cycles; a
    hard row limit and the final decision-byte limit bound all retained data.
    """
    pending=[]
    seen=set()
    result={}
    def cell(spec,row,column_index):
        return row*len(spec.columns)+column_index
    def number(spec,row,column_index):
        return struct.unpack_from('=q',frame.tensors[spec.key].data,
                                  cell(spec,row,column_index)*8)[0]
    def reference(spec,row,column_index):
        column=spec.columns[column_index]
        selector=next((i for i,item in enumerate(spec.columns)
                       if item.field==column.reference),None)
        if selector is None:
            group=column.reference
        else:
            if not frame.tensors[spec.known_key].data[cell(spec,row,selector)]:
                return None
            vocabulary=spec.columns[selector].vocabulary
            group=schema.vocabularies[vocabulary][number(spec,row,selector)]
        if group is None or group=='none': return None
        return {'group':group,'row':number(spec,row,column_index)}
    def retain(target):
        if target is None: return
        identity=(target['group'],target['row'])
        if identity in seen: return
        if len(seen)>=128: raise ValueError('trace referenced-row ceiling exceeded')
        seen.add(identity)
        pending.append(identity)
    actions=schema.tensors['actions']
    for candidate in legal:
        for column_index in actions.reference_columns:
            if frame.tensors[actions.known_key].data[cell(actions,candidate.index,column_index)]:
                retain(reference(actions,candidate.index,column_index))
    for group,row in pending:
        spec=schema.tensors[group]
        occupied=spec.occupied_key is None or bool(frame.tensors[spec.occupied_key].data[row])
        fields={}
        for column_index,column in enumerate(spec.columns):
            known=occupied and bool(frame.tensors[spec.known_key].data[cell(spec,row,column_index)])
            value=None
            if known:
                if column.reference is not None:
                    value=reference(spec,row,column_index)
                    retain(value)
                elif column.vocabulary is not None:
                    value=schema.vocabularies[column.vocabulary][number(spec,row,column_index)]
                else:
                    value=number(spec,row,column_index)
            fields[column.field]={'known':known,'value':value,'unit':column.unit}
        result.setdefault(group,[]).append({'index':row,'occupied':occupied,'fields':fields})
    return {group:sorted(rows,key=lambda item:item['index']) for group,rows in sorted(result.items())}


def public_decision_trace(schema,frame,choice,*,legal_scores,successor,result):
    from dataclasses import asdict
    from spire_learn.neural.schema import candidates
    legal=candidates(frame,schema); indices={row.index for row in legal}
    if not isinstance(legal_scores,Mapping) or any(type(index) is not int for index in legal_scores) or set(legal_scores)!=indices or choice.original_action_index not in indices:
        raise ValueError('trace scores/choice must match exactly the original legal indices')
    rows=[]
    for candidate in legal:
        score=legal_scores[candidate.index]
        if not isinstance(score,Mapping) or set(score)!={'logit','probability'}: raise ValueError('invalid legal score')
        logit=score['logit']; probability=score['probability']
        if type(logit) not in (int,float) or not math.isfinite(logit): raise ValueError('legal logit must be finite')
        if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('legal probability must be in [0,1]')
        rows.append({**asdict(candidate),'source_applicable':candidate.source_applicable,
            'source_known':candidate.source_known,'target_applicable':candidate.target_applicable,
            'target_known':candidate.target_known,'logit':float(logit),'probability':float(probability)})
    if abs(sum(row['probability'] for row in rows)-1.)>1e-5: raise ValueError('legal probabilities must sum to one')
    if not isinstance(result,Mapping) or set(result)&{'native_reward','terminated','truncated'}!={'native_reward','terminated','truncated'}:
        raise ValueError('trace result requires reward and ending flags')
    if type(result['native_reward']) not in (int,float) or not math.isfinite(result['native_reward']):
        raise ValueError('trace reward must be finite')
    if any(type(result[key]) is not bool for key in ('terminated','truncated')) or (result['terminated'] and result['truncated']):
        raise ValueError('trace ending flags are invalid')
    output={'schema':'spire-learn-public-decision-trace-v1','schema_fingerprint':schema.schema_fingerprint,
        'source':_public_frame(schema,frame),'successor':_public_frame(schema,successor),
        'referenced_rows':_public_referenced_rows(schema,frame,legal),
        'choice':{'original_action_index':choice.original_action_index,'log_probability':choice.log_probability,'value':choice.value},
        'legal_candidates':rows,'result':{key:result[key] for key in ('native_reward','terminated','truncated') if key in result}}
    if len(json_bytes(output))>65536: raise ValueError('trace decision byte ceiling exceeded')
    return output


class TraceStore:
    def __init__(self,*,max_actions=32,max_decision_bytes=65536,max_total_bytes=16777216):
        if max_actions>32 or max_decision_bytes>65536 or max_total_bytes>16777216: raise ValueError('trace limits cannot widen')
        self.max_actions=max_actions; self.max_decision_bytes=max_decision_bytes; self.max_total_bytes=max_total_bytes
        self.episodes={}; self.bytes=2  # The empty JSON object is part of retained trace bytes.
    def add(self,episode,trace):
        if len(self.episodes)>=8 and episode not in self.episodes: raise ValueError('trace episode ceiling exceeded')
        if len(self.episodes.get(episode,()))>=self.max_actions: raise ValueError('trace action ceiling exceeded')
        raw=json_bytes(trace)
        if len(raw)>self.max_decision_bytes: raise ValueError('trace decision byte ceiling exceeded')
        if episode in self.episodes:
            new_bytes=self.bytes+len(raw)+(1 if self.episodes[episode] else 0)
        else:
            # Encoded key, colon, array brackets and any preceding comma.
            new_bytes=self.bytes+len(json_bytes(episode))+3+len(raw)+(1 if self.episodes else 0)
        if new_bytes>self.max_total_bytes: raise ValueError('trace total byte ceiling exceeded')
        self.episodes.setdefault(episode,[]).append(json.loads(raw)); self.bytes=new_bytes


def memory_snapshot(runtime=None):
    """RSS/HWM/cgroup and loaded Torch allocator peaks; missing fields stay null."""
    result={'rss_bytes':None,'rss_hwm_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        'cgroup_current_bytes':None,'cgroup_peak_bytes':None,'cgroup_anon_bytes':None,'cgroup_file_bytes':None,'torch_allocated_peak_bytes':None,'torch_reserved_peak_bytes':None}
    try:
        status=Path('/proc/self/status').read_text()
        for field,key in [('VmRSS','rss_bytes'),('VmHWM','rss_hwm_bytes')]:
            match=re.search(r'^'+field+r':\s+(\d+) kB$',status,re.M)
            if match: result[key]=int(match[1])*1024
        group=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        if '..' not in Path(group).parts:
            cgroup=Path('/sys/fs/cgroup')/group.lstrip('/')
            for file,key in [('memory.current','cgroup_current_bytes'),('memory.peak','cgroup_peak_bytes')]:
                try: result[key]=int((cgroup/file).read_text().strip())
                except (OSError,ValueError): pass
            try:
                stats=dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
                for field in ('anon','file'): result['cgroup_'+field+'_bytes']=int(stats[field])
            except (OSError,ValueError,KeyError): pass
    except (OSError,StopIteration): pass
    if runtime is not None and getattr(runtime,'torch',None) is not None and str(runtime.config.device).startswith('cuda'):
        cuda=runtime.torch.cuda
        result['torch_allocated_peak_bytes']=int(cuda.max_memory_allocated(runtime.config.device))
        result['torch_reserved_peak_bytes']=int(cuda.max_memory_reserved(runtime.config.device))
    return result


def _patch(stack,obj,name,replacement):
    own=name in vars(obj); original=getattr(obj,name)
    setattr(obj,name,replacement)
    def restore():
        if own: setattr(obj,name,original)
        else: delattr(obj,name)
    stack.callback(restore)
    return original


@contextmanager
def _after_preserving(label,callback):
    primary=None
    try:
        yield
    except BaseException as error:
        primary=error
        raise
    finally:
        try: callback()
        except BaseException as error:
            if primary is not None: _note(primary,label,error)
            else: raise


@contextmanager
def instrument_runtime(runtime,config,ledger,timing,*,operation,trace_callback=None,memory=memory_snapshot):
    """Scoped wrappers around the real accepted collector, policy and optimizer."""
    metrics={'retained_frame_payload_bytes':0,'retained_record_count':0,
             'collection_memory':[],'update_memory':[],'event_log_bytes':0}
    payload=FramePayloadCounter()
    closed=False
    optimizer_lease=None
    optimizer_allowance=0
    collector=runtime.collector
    driver=collector.driver
    original_close=runtime.close
    def timed(label,function):
        with timing.span(label): return function()
    def close():
        nonlocal closed
        if closed: return
        closed=True
        called=False
        def invoke_close():
            nonlocal called
            called=True
            return original_close()
        try: return timed('runtime_close',invoke_close)
        except BaseException as error:
            ledger.halt(error)
            # A pre-span clock/sync failure must not skip resource cleanup.
            # Once close itself was entered, its uncertain effects never retry.
            if not called:
                try: invoke_close()
                except BaseException as close_error: _note(error,'untimed resource close also failed',close_error)
            raise
    def reset_peaks():
        if str(config.device).startswith('cuda') and runtime.torch is not None:
            runtime.torch.cuda.reset_peak_memory_stats(config.device)
    def capture_memory(key):
        limit=2 if key=='update_memory' else 1026
        if len(metrics[key])>=limit: raise ValueError('memory record capacity reached')
        metrics[key].append(memory(runtime))
    # Close even if a wrapper cannot be installed or an observer fails at setup.
    with _after_preserving('runtime close also failed',close), ExitStack() as stack:
        def reset(*args,**kwargs):
            count=sum(kwargs['reset_mask']) if kwargs.get('reset_mask') is not None else len(kwargs['seeds'])
            return ledger.call('reset_slots',count,'reset',lambda:timed('driver_reset_inclusive',lambda:original_reset(*args,**kwargs)))
        original_reset=_patch(stack,driver,'reset',reset)
        def step(actions):
            return ledger.call('environment_attempts',len(actions),'native-step',lambda:timed('driver_step_inclusive',lambda:original_step(actions)))
        original_step=_patch(stack,driver,'step',step)
        policy=collector.policy
        def act(frames,**kwargs):
            label='stochastic_actor_rows' if operation=='training' else 'deterministic_actor_rows'
            return ledger.call('sample_rows',len(frames),label,lambda:timed('policy_act',lambda:original_act(frames,**kwargs)))
        original_act=_patch(stack,policy,'act',act)
        def values(frames): return timed('bootstrap_values',lambda:original_values(frames))
        original_values=_patch(stack,policy,'values',values)
        rng=collector.controller_rng
        def fixed_choice(rows):
            return ledger.call('fixed_random_choices',1,'fixed_controller_random_choice',lambda:original_choice(rows))
        original_choice=_patch(stack,rng,'choice',fixed_choice)
        def capture_payload(records):
            payload.add(records)
            metrics['retained_frame_payload_bytes']=payload.bytes
            metrics['retained_record_count']+=len(records)
        def collect(**kwargs):
            try:
                reset_peaks()
                try: result=timed('collection',lambda:original_collect(**kwargs))
                except BaseException as error:
                    partial=getattr(collector,'partial_rollout',None)
                    if partial is not None:
                        try: capture_payload(partial.records)
                        except BaseException as observer: _note(error,'partial payload observation also failed',observer)
                    raise
                capture_payload(result.records)
                capture_memory('collection_memory')
                if trace_callback:
                    for record in result.records: trace_callback(record)
                return result
            except BaseException as error:
                ledger.halt(error)
                raise
        original_collect=_patch(stack,collector,'collect',collect)
        if getattr(runtime,'optimizer',None) is not None:
            def optimizer_step(*args,**kwargs):
                if optimizer_lease is None or optimizer_lease['attempted']>=optimizer_allowance:
                    raise ValueError('optimizer call lacks update reservation')
                return ledger.invoke(optimizer_lease,1,lambda:timed('adam_step',lambda:original_optimizer(*args,**kwargs)))
            original_optimizer=_patch(stack,runtime.optimizer,'step',optimizer_step)
        if callable(getattr(runtime,'update',None)):
            def update(rollout):
                nonlocal optimizer_lease,optimizer_allowance
                rows=sum(record.actor_training_mask for record in rollout.records)
                optimizer_allowance=config.ppo.epochs*math.ceil(rows/config.ppo.minibatch_rows)
                try:
                    reset_peaks()
                    with ledger.reservation('optimizer_attempts',optimizer_allowance,'disposable_update') as lease:
                        optimizer_lease=lease
                        try:
                            with _after_preserving('update memory observer also failed',lambda:capture_memory('update_memory')):
                                return timed('ppo_update',lambda:original_update(rollout))
                        finally: optimizer_lease=None
                except BaseException as error:
                    ledger.halt(error)
                    raise
            original_update=_patch(stack,runtime,'update',update)
        if callable(getattr(runtime,'save',None)):
            def save(*args,**kwargs): return timed('checkpoint_save',lambda:original_save(*args,**kwargs))
            original_save=_patch(stack,runtime,'save',save)
        _patch(stack,runtime,'close',close)
        with observe(timing):
            yield metrics


def _legal_scores(runtime,frame):
    """No-grad public logits/probabilities, with no generator or sampling call."""
    from spire_learn.neural.features import encode_frames
    from spire_learn.neural.distribution import masked_categorical
    from spire_learn.neural.schema import candidates
    with runtime.torch.no_grad():
        batch=encode_frames((frame,),runtime.collector.driver.schema,device='cpu')
        output=runtime.policy(batch)
        distribution=masked_categorical(output.logits,batch.action_mask)
        return {candidate.index:{'logit':float(output.logits[0,candidate.index].item()),
            'probability':float(distribution.probabilities[0,candidate.index].item())}
            for candidate in candidates(frame,runtime.collector.driver.schema)}


def _default_dependencies():
    # Namespace functions remain ordinary callables, never class attributes.
    from spire_learn.neural.runner import create_runtime,train,evaluate_policy
    return {'factory':create_runtime,'train':train,'evaluate':evaluate_policy,
            'memory':memory_snapshot,'synchronize':None}


def _runtime_timing(runtime,clock,synchronize):
    sync=None
    if str(runtime.config.device).startswith('cuda'):
        if synchronize is not None: sync=lambda:synchronize(runtime)
        elif runtime.torch is not None: sync=lambda:runtime.torch.cuda.synchronize(runtime.config.device)
        else: raise ValueError('CUDA profiling requires device synchronization')
        if runtime.torch is not None: runtime.torch.cuda.reset_peak_memory_stats(runtime.config.device)
    return TimingRecorder(clock=clock,synchronize=sync)


def _context(root,campaign,context,clock):
    required={'phase_id','source_sha256','deadline','effect_ledger','input_bindings','job_reservation'}
    if not isinstance(context,Mapping) or set(context)!=required: raise ValueError('profile context requires exact phase/source/deadline/ledger/input/job bindings')
    if not isinstance(context['source_sha256'],str) or re.fullmatch('[0-9a-f]{64}',context['source_sha256']) is None:
        raise ValueError('profile source_sha256 must be a lowercase SHA256')
    from scripts.neural.sustained_package import BUDGETS
    if campaign['budgets']['profile']!=BUDGETS['profile']: raise ValueError('campaign profile budgets differ')
    now=clock(); deadline=_finite(context['deadline'],'profile deadline')
    if not now<deadline<=now+900.: raise ValueError('profile deadline expired or widened')
    plan=profile_plan(root,campaign)
    if context['job_reservation']!=plan['reservation_ceiling']: raise ValueError('host job reservation must match fixed workload ceilings')
    inventory=profile_input_inventory(root,campaign); bindings=context['input_bindings']
    if not isinstance(bindings,Mapping) or set(bindings)!=set(inventory): raise ValueError('explicit profile input bindings incomplete')
    for relative,item in inventory.items():
        binding=bindings[relative]
        if not isinstance(binding,Mapping) or dict(binding)!={**item,'path':str(_path(root,relative))}:
            raise ValueError('explicit profile input binding differs: '+relative)
    return plan,EffectLedger(context['phase_id'],context['effect_ledger'],deadline=deadline,clock=clock,limits=context['job_reservation'])


def run_profile(root: Path, campaign: Mapping[str,object], context: Mapping[str,object], *,
                dependencies: Mapping[str,object] | None=None, clock: Callable[[],float]=time.monotonic) -> dict:
    """Execute the fixed, once-only profile after the caller's reviewed release.

    Dependencies are factory/create_runtime, train, evaluate/evaluate_policy,
    clock/monotonic, memory(runtime)->JSON metrics, synchronize(runtime)->None.
    All provided functions are used as plain callables. Failure raises its first
    cause; the caller retains the effect callbacks and never retries the job.
    """
    dependencies=dict(dependencies or {})
    if set(dependencies)-{'factory','train','evaluate','clock','memory','synchronize'}: raise ValueError('unknown profile dependency')
    if 'clock' in dependencies: clock=dependencies.pop('clock')
    for name,function in dependencies.items():
        if function is not None and not callable(function): raise TypeError('profile dependency must be callable: '+name)
    root=Path(root).resolve(); started=clock()
    plan,ledger=_context(root,campaign,context,clock)
    deps=_default_dependencies(); deps.update(dependencies)
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.runner import RuntimeOperationLimits
    from spire_learn.neural.schema import validate_frame
    store=TraceStore(); rows=[]; catalog=[]; factory_timing=TimingRecorder(clock=clock)
    initial_memory=deps['memory'](None)
    def checkpoint(identifier):
        binding=campaign['diagnostic_checkpoints'][identifier] if identifier=='standalone-a' else campaign['parents'][identifier]
        return _path(root,binding['path'])
    def make(config,identifier,purpose,action_selection='seeded-sampling'):
        ledger.check()
        runtime=None
        try:
            with factory_timing.span('runtime_factory'):
                runtime=deps['factory'](config,purpose=purpose,checkpoint=checkpoint(identifier),action_selection=action_selection)
            timing=_runtime_timing(runtime,clock,deps['synchronize'])
        except BaseException as error:
            if runtime is not None:
                try: runtime.close()
                except BaseException as close_error: _note(error,'runtime setup close also failed',close_error)
            raise
        return runtime,timing
    def result_row(operation,report,metrics,timing):
        if ledger.first_error is not None: raise ledger.first_error
        if report.run_dir is not None:
            event=Path(report.run_dir)/'events.jsonl'
            if not event.resolve().is_relative_to(root/OUTPUT_PREFIX): raise ValueError('profile report escaped output root')
            if event.is_file(): metrics['event_log_bytes']=event.stat().st_size
        if report.exit_code!=0 or report.summary.get('status')!='complete':
            raise RuntimeError('profile accepted runner failed: '+operation)
        return {'operation_id':operation,'runner_status':'complete','metrics':metrics,'timings':timing.snapshot()}
    # A B1 authenticated runtime tests every authored catalog entry once.
    config=load_neural_config(_path(root,plan['catalog_config']))
    runtime,timing=make(config,'standalone-b','warm_start')
    with instrument_runtime(runtime,config,ledger,timing,operation='catalog',memory=deps['memory']) as metrics:
        for case in config.cases.for_mode('standalone'):
            ledger.check()
            driver=runtime.collector.driver
            boundary=driver.reset(seeds=(case.environment_seed,),entry_ids=(case.entry_id,))
            if len(boundary)!=1: raise ValueError('catalog requires exactly one native slot')
            validate_frame(boundary[0].frame,driver.schema)
            if not boundary[0].candidates: raise ValueError('catalog reset has no legal candidate')
            action=boundary[0].candidates[0].index
            transition=driver.step((action,))
            if len(transition)!=1: raise ValueError('catalog step requires exactly one native slot')
            validate_frame(transition[0].successor.frame,driver.schema)
            catalog.append({'case_id':case.case_id,'state':'validated','reset_slots':1,'action_attempts':1})
    rows.append({'operation_id':'profile-catalog','runner_status':'complete','metrics':metrics,'timings':timing.snapshot()})
    for work in plan['training']:
        config=load_neural_config(_path(root,work['config']))
        runtime,timing=make(config,work['checkpoint_id'],'warm_start')
        with instrument_runtime(runtime,config,ledger,timing,operation='training',memory=deps['memory']) as metrics:
            runtime.operation_limits=RuntimeOperationLimits(max_updates=2,monotonic_deadline=context['deadline'],max_environment_actions=work['environment_attempts'])
            with timing.span('training_inclusive'):
                report=deps['train'](config,runtime=runtime,clock=clock)
        rows.append(result_row(work['operation_id'],report,metrics,timing))
    for work in plan['traces']:
        config=load_neural_config(_path(root,work['config']))
        runtime,timing=make(config,work['checkpoint_id'],'evaluate',action_selection='deterministic')
        def trace(record):
            scores=ledger.call('sample_rows',1,'diagnostic_score_rows',lambda:_legal_scores(runtime,record.source))
            store.add(work['operation_id'],public_decision_trace(runtime.collector.driver.schema,record.source,record.choice,
                legal_scores=scores,successor=record.successor,
                result={'native_reward':record.native_reward,'terminated':record.terminated,'truncated':record.truncated}))
        with instrument_runtime(runtime,config,ledger,timing,operation='trace',trace_callback=trace,memory=deps['memory']) as metrics:
            runtime.operation_limits=RuntimeOperationLimits(monotonic_deadline=context['deadline'],max_environment_actions=32)
            with timing.span('evaluation_inclusive'):
                report=deps['evaluate'](config,runtime=runtime,clock=clock)
        rows.append(result_row(work['operation_id'],report,metrics,timing))
    final_memory=deps['memory'](None)
    elapsed=_finite(clock()-started,'profile elapsed')
    if clock()>context['deadline']: raise TimeoutError('profile deadline exceeded including close')
    report={'schema':'spire-learn-sustained-profile-v1','status':'complete','phase_id':context['phase_id'],
        'source_sha256':context['source_sha256'],'elapsed_seconds':elapsed,'plan_sha256':_binding(_path(root,PLAN_PATH).read_bytes())['sha256'],
        'effects':ledger.snapshot(),'effect_categories':ledger.categories,'catalog':catalog,'operations':rows,
        'factory_timings':factory_timing.snapshot(),'initial_memory':initial_memory,'final_memory':final_memory,
        'traces':store.episodes,'trace_bytes':store.bytes,
        'timing_interpretation':'inclusive wall includes synchronization; sum only exclusive spans within each runtime; factory CPU wall separately; no CUDA-event or inaccessible Rust engine claim',
        'frame_payload_interpretation':'unique BufferArray.data identities retained by each trainer, cumulative across rollouts; excludes Python objects and temporary tensors',
        'memory_interpretation':'RSS/cgroup high-water marks are lifetime peaks; Torch allocator peaks reset before each collection/update; allocator reserved is not total VRAM',
        'timing_boundary':'after all runtime closes and final memory capture; before report validation/serialization; parent outer lifecycle receipt charges publication/cleanup',
        'randomness':{'stochastic_actor_rows':ledger.categories.get('stochastic_actor_rows',{}).get('attempted',0),
            'deterministic_trace_policy_rng_draws':0,'diagnostic_score_rng_draws':0,
            'fixed_controller_random_choices':ledger.totals['fixed_random_choices']['attempted']}}
    validate_profile_report(report,plan)
    return report


def validate_profile_report(report,plan):
    """Strict shape/count/byte reconciliation for producer output, never a release."""
    required={'schema','status','phase_id','source_sha256','elapsed_seconds','plan_sha256','effects','effect_categories',
        'catalog','operations','factory_timings','initial_memory','final_memory','traces','trace_bytes',
        'timing_interpretation','frame_payload_interpretation','memory_interpretation','timing_boundary','randomness'}
    if not isinstance(report,Mapping) or set(report)!=required or report['schema']!='spire-learn-sustained-profile-v1' or report['status']!='complete':
        raise ValueError('invalid complete profile report shape')
    if _finite(report['elapsed_seconds'],'elapsed')>900.: raise ValueError('profile elapsed exceeds ceiling')
    if [row.get('case_id') for row in report['catalog']]!=plan['catalog_case_ids'] or any(row!={'case_id':case,'state':'validated','reset_slots':1,'action_attempts':1} for row,case in zip(report['catalog'],plan['catalog_case_ids'])):
        raise ValueError('profile catalog validation incomplete')
    expected=['profile-catalog']+[row['operation_id'] for row in (*plan['training'],*plan['traces'])]
    if [row.get('operation_id') for row in report['operations']]!=expected or any(row.get('runner_status')!='complete' for row in report['operations']):
        raise ValueError('profile operation inventory incomplete')
    if set(report['effects'])!=set(LIMITS): raise ValueError('effect counters incomplete')
    for kind,values in report['effects'].items():
        if set(values)!={'reserved','attempted','delivered','uncertain'}: raise ValueError('invalid effect counters')
        for key,value in values.items(): _integer(value,'effect '+key)
        if values['reserved']>plan['reservation_ceiling'][kind] or values['attempted']>values['reserved'] or values['delivered']+values['uncertain']!=values['attempted'] or values['uncertain']:
            raise ValueError('effect accounting does not reconcile')
    record_count=0
    for operation in report['operations']:
        metrics=operation.get('metrics')
        if not isinstance(metrics,Mapping) or set(metrics)!={'retained_frame_payload_bytes','retained_record_count','collection_memory','update_memory','event_log_bytes'}:
            raise ValueError('profile metrics shape differs')
        record_count+=_integer(metrics['retained_record_count'],'retained records')
        _integer(metrics['retained_frame_payload_bytes'],'frame payload bytes')
        _integer(metrics['event_log_bytes'],'event log bytes')
        if metrics['event_log_bytes']>64*1024**2: raise ValueError('event log guard exceeded')
        if not isinstance(metrics['collection_memory'],list) or len(metrics['collection_memory'])>1026 or not isinstance(metrics['update_memory'],list) or len(metrics['update_memory'])>2:
            raise ValueError('memory observation capacity exceeded')
    if report['effects']['environment_attempts']['delivered']!=78+record_count:
        raise ValueError('environment effects differ from retained records')
    label_kinds={'reset':'reset_slots','native-step':'environment_attempts','stochastic_actor_rows':'sample_rows',
        'deterministic_actor_rows':'sample_rows','diagnostic_score_rows':'sample_rows',
        'fixed_controller_random_choice':'fixed_random_choices','disposable_update':'optimizer_attempts'}
    sums={kind:dict(reserved=0,attempted=0,delivered=0,uncertain=0) for kind in LIMITS}
    for label,values in report['effect_categories'].items():
        if label not in label_kinds or set(values)!={'reserved','attempted','delivered','uncertain'}:
            raise ValueError('profile effect categories shape differs')
        for key,value in values.items(): sums[label_kinds[label]][key]+=_integer(value,'effect category')
    if sums!=report['effects']: raise ValueError('profile effect categories do not reconcile')
    if set(report['traces'])!={row['operation_id'] for row in plan['traces']}: raise ValueError('eight trace episodes required')
    total=0
    trace_actions=0
    for decisions in report['traces'].values():
        if not isinstance(decisions,list) or not 1<=len(decisions)<=32: raise ValueError('trace action bound violated')
        trace_actions+=len(decisions)
        for trace in decisions:
            if set(trace)!={'schema','schema_fingerprint','source','successor','referenced_rows','choice','legal_candidates','result'} or trace['schema']!='spire-learn-public-decision-trace-v1':
                raise ValueError('public trace shape differs')
            if set(trace['source'])-{'player','enemies'} or set(trace['successor'])-{'player','enemies'}:
                raise ValueError('public trace feature groups differ')
            if not isinstance(trace['referenced_rows'],Mapping) or sum(len(rows) for rows in trace['referenced_rows'].values())>128:
                raise ValueError('public trace referenced-row ceiling exceeded')
            for group,rows in trace['referenced_rows'].items():
                if not isinstance(group,str) or not isinstance(rows,list) or any(
                    set(row)!={'index','occupied','fields'} for row in rows):
                    raise ValueError('public trace referenced-row shape differs')
            if set(trace['result'])!={'native_reward','terminated','truncated'}:
                raise ValueError('public trace result shape differs')
            raw=json_bytes(trace)
            if len(raw)>65536: raise ValueError('trace decision bound violated')
            total+=len(raw)
    if report['effect_categories'].get('diagnostic_score_rows',{}).get('delivered',0)!=trace_actions:
        raise ValueError('diagnostic forward count differs from traces')
    expected_randomness={'stochastic_actor_rows':report['effect_categories'].get('stochastic_actor_rows',{}).get('attempted',0),
        'deterministic_trace_policy_rng_draws':0,'diagnostic_score_rng_draws':0,
        'fixed_controller_random_choices':report['effects']['fixed_random_choices']['attempted']}
    if report['randomness']!=expected_randomness: raise ValueError('profile randomness labels differ')
    if len(json_bytes(report['traces']))!=report['trace_bytes'] or report['trace_bytes']>16777216: raise ValueError('trace byte accounting differs')
    for operation in report['operations']:
        if set(operation)!={'operation_id','runner_status','metrics','timings'}: raise ValueError('invalid profile operation shape')
        for row in operation['timings'].values():
            if set(row)!={'count','inclusive_wall_seconds','exclusive_wall_seconds','inclusive_cpu_seconds','exclusive_cpu_seconds','synchronization_seconds','synchronization'}:
                raise ValueError('invalid phase timing shape')
            _integer(row['count'],'phase count')
            for key,value in row.items():
                if key.endswith('_seconds'): _finite(value,key)
            if row['exclusive_wall_seconds']>row['inclusive_wall_seconds']+1e-9: raise ValueError('phase timing does not reconcile')
    json_bytes(report)
    return report


__all__=['run_profile','public_decision_trace','author_profile_inputs','profile_plan','profile_input_inventory',
         'profile_source_paths','validate_profile_report']
