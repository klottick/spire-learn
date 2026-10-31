"""Reviewed, one-use sustained campaign host controls (stdlib only)."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

if __package__ in (None, ''):
    _root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(_root/'src'),str(_root)]

from scripts.neural import task9_study_controls as low, task9_nvidia_cdi as cdi
from scripts.neural.gpu_runtime_proof_helpers import (
    HostDependencies,_source_files,_source_inventory,canonical_sha256,sha256_file,write_json_exclusive,
)
from scripts.neural.sustained_package import CAMPAIGN_PATH,BUDGETS,evaluation_schedule,validate_campaign
from scripts.neural.sustained_runtime import (
    CONTEXT_SCHEMA,RECEIPT_SCHEMA,EFFECT_KINDS,checkpoint_inventory,finite,integer,json_value,
    reconcile_segment,_selected_config,
)

DOMAIN=Path('artifacts/development/neural-combat/sustained-v1/runtime')
INITIAL=Path('artifacts/development/neural-combat/sustained-v1/preparation/runtime-budget-initial.json')
INITIAL_SHA256='cfb5595c860dc9f753cac2f58477a6bc9e109b6c44a7ce347bb211fbe590ffd5'
BUCKETS={'shared':1200.,'profile':900.,'evaluation':2700.,
         'standalone-1':6000.,'run-1':6000.,'standalone-2':6000.,'run-2':6000.}
HELPERS=tuple('scripts/neural/'+name for name in (
    'sustained_package.py','sustained_profile.py','sustained_runtime.py','sustained_controls.py',
    'task9_study_controls.py','task9_study_runtime.py','task9_study_package.py',
    'gpu_runtime_proof_helpers.py','task8_gpu_setup_helpers.py','task9_nvidia_cdi.py'))
GUEST_SCOPES=('src','configs',*HELPERS)
SNAPSHOT_SCOPES=(*GUEST_SCOPES,'scripts/neural/run-sustained-training.sh',
    'scripts/neural/sustained-training-runbook.md','tests/unit/test_neural_sustained_runtime.py',
    'tests/unit/test_neural_sustained_controls.py')
SNAPSHOT_SCHEMA='spire-learn-sustained-source-snapshot-v1'
RELEASE_SCHEMA='spire-learn-sustained-stage-release-v1'
CLOSED_SCHEMA='spire-learn-sustained-closed-stage-v1'
SEAL_SCHEMA='spire-learn-sustained-stage-seal-v1'
OUTER_SCHEMA='spire-learn-sustained-outer-accounting-v1'
MARKER='SPIRE_LEARN_SUSTAINED_ADMISSION_V1 '
STAGES=('profile','train-evaluate')
_ID=re.compile(r'[a-z0-9][a-z0-9-]{0,23}\Z')
_signals=ContextVar('sustained_signals',default=None)


@contextmanager
def signal_scope():
    """First control signal unwinds; later cleanup signals cannot skip owned rm."""
    state={'cleanup':False,'deferred':None}
    def handler(number,frame):
        error=low._Termination('signal '+str(number))
        if state['cleanup']:state['deferred']=error
        else:raise error
    old={number:signal.signal(number,handler) for number in (signal.SIGINT,signal.SIGTERM)}
    token=_signals.set(state)
    try:yield
    finally:
        _signals.reset(token)
        for number,value in old.items():signal.signal(number,value)


def _read(path):return low._json(Path(path))
def _path(root,value,*,exists=False):return low._safe(Path(root),value,exists=exists)
def _error(error):return low._error(error)


def _profile_receipt(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=32*1024**2:raise ValueError('bounded profile receipt size/type differs')
    value=json.loads(path.read_text(),object_pairs_hook=low._unique,parse_constant=low._bad_constant)
    if not isinstance(value,dict):raise ValueError('profile receipt must be object')
    return value


def evaluation_dispatch(campaign,slots):
    """Preregistered mode-balanced queue; scores never enter this ordering."""
    cases=campaign['selection']['case_bindings'];modes={mode:[row['case_index'] for row in cases if row['mode']==mode] for mode in ('standalone','run')}
    ordered=[]
    for index in range(max(map(len,modes.values()))):
        for mode in ('standalone','run'):
            if index<len(modes[mode]):ordered.append(modes[mode][index])
    rank={case:index for index,case in enumerate(ordered)}
    slots_order={'anchor':0,'reference':0,'25':1,'50':2,'75':3,'final':4}
    policies={name:index for index,name in enumerate((*campaign['training']['order'],'standalone-b','run-b','legal-random','end-turn-last-v1'))}
    return tuple(sorted(evaluation_schedule(campaign,slots),key=lambda row:(slots_order[row['checkpoint_slot']],rank[row['case_index']],policies[row['policy_id']])))


class BudgetAccount:
    """Disjoint outer intervals; job setup, publication and cleanup share its bucket."""
    def __init__(self,charged,*,clock=time.monotonic):
        if set(charged)!=set(BUCKETS):raise ValueError('budget buckets incomplete')
        self.charged={k:finite(v,k) for k,v in charged.items()};self.clock=clock
        self.bucket='shared';self.mark=finite(clock(),'clock');self.closed=False;self.intervals=[]
        self.check()
    def _charge(self):
        now=finite(self.clock(),'clock')
        if now<self.mark:raise ValueError('monotonic clock moved backwards')
        seconds=now-self.mark;self.charged[self.bucket]+=seconds
        self.intervals.append({'bucket':self.bucket,'seconds':seconds});self.mark=now
    def switch(self,bucket):
        if self.closed or bucket not in BUCKETS:raise ValueError('closed or unknown budget bucket')
        self._charge();self.bucket=bucket;self.check()
    def check(self):
        totals=dict(self.charged)
        if not self.closed:totals[self.bucket]+=max(0.,finite(self.clock(),'clock')-self.mark)
        if any(totals[k]>BUCKETS[k] for k in BUCKETS) or sum(totals.values())>28800.:
            raise TimeoutError('campaign resource wall ceiling exceeded')
    def remaining(self,bucket):
        active=0. if self.closed or self.bucket!=bucket else max(0.,self.clock()-self.mark)
        total_active=0. if self.closed else max(0.,self.clock()-self.mark)
        return max(0.,min(BUCKETS[bucket]-self.charged[bucket]-active,28800.-sum(self.charged.values())-total_active))
    def close(self):
        if not self.closed:self._charge();self.closed=True;self.check()


def capture_source_snapshot(root,destination,*,learner_head,dirty):
    root=Path(root).resolve(strict=True)
    if re.fullmatch('[0-9a-f]{40}',learner_head) is None:raise ValueError('full learner head required')
    validate_campaign(root,root/CAMPAIGN_PATH)
    files=_source_inventory(root,_source_files(root,SNAPSHOT_SCOPES))
    value={'schema':SNAPSHOT_SCHEMA,'state':'source_only','learner_head':learner_head,'dirty':json_value(dirty),
        'files':files,'source_inventory_sha256':canonical_sha256(files),'campaign_raw_sha256':sha256_file(root/CAMPAIGN_PATH)}
    write_json_exclusive(_path(root,destination),value);return value


def _initial(root):
    path=_path(root,INITIAL,exists=True)
    if sha256_file(path)!=INITIAL_SHA256:raise ValueError('initial resource accounting raw binding changed')
    initial=_read(path)
    if initial.get('schema')!='spire-learn-sustained-initial-runtime-accounting-v1' or initial.get('effect_attempts')!=0 or initial.get('stage_claims_consumed')!=0:
        raise ValueError('initial accounting identity differs')
    for row in initial['runtime_intervals']:
        if sha256_file(_path(root,row['receipt'],exists=True))!=row['raw_sha256']:raise ValueError('initial headroom receipt changed')
    if initial['charged_seconds']!={k:(.09155025496147573 if k=='shared' else 0.) for k in BUCKETS}:
        raise ValueError('initial accounting totals changed')
    return initial


def reconcile_outer_accounting(seal,outer):
    """Carry the enclosing process's uncharged startup/exit tail into shared."""
    expected={'schema','stage','campaign_id','seal_sha256','exit_code','enclosing_elapsed_seconds',
              'accounted_interval_seconds','residual_shared_seconds'}
    if (set(outer)!=expected or outer['schema']!=OUTER_SCHEMA or outer['stage']!=seal['stage']
        or outer['campaign_id']!=seal['campaign_id'] or type(outer['exit_code']) is not int or outer['exit_code']!=0):
        raise ValueError('outer stage accounting identity/exit differs')
    elapsed=finite(outer['enclosing_elapsed_seconds'],'outer elapsed')
    accounted=finite(seal['stage_elapsed_seconds'],'sealed elapsed')
    if outer['accounted_interval_seconds']!=accounted or elapsed<accounted:
        raise ValueError('outer enclosing/accounted intervals do not reconcile')
    residual=elapsed-accounted
    if outer['residual_shared_seconds']!=residual:raise ValueError('outer shared residual differs')
    charged={k:finite(v,k) for k,v in seal['charged_seconds'].items()}
    if set(charged)!=set(BUCKETS):raise ValueError('sealed buckets incomplete')
    incoming=seal['incoming_charged_seconds']
    if set(incoming)!=set(BUCKETS) or any(charged[k]<finite(incoming[k],'incoming '+k) for k in BUCKETS):raise ValueError('sealed charged prefix moved backwards')
    if not math.isclose(sum(charged.values())-sum(incoming.values()),accounted,rel_tol=0.,abs_tol=1e-8):
        raise ValueError('sealed elapsed does not equal disjoint charged delta')
    charged['shared']+=residual
    BudgetAccount(charged,clock=lambda:0.).close()
    return charged


def validate_release(root,*,campaign_path,source_snapshot,release_path,stage):
    """Host-only full package validation before claims or output mutation."""
    root=Path(root).resolve(strict=True)
    if stage not in STAGES or _path(root,campaign_path,exists=True)!=root/CAMPAIGN_PATH:raise ValueError('stage/campaign path differs')
    campaign=validate_campaign(root,root/CAMPAIGN_PATH)
    snapshot_path=_path(root,source_snapshot,exists=True);release_path=_path(root,release_path,exists=True)
    snapshot=_read(snapshot_path);release=_read(release_path)
    expected={'schema','state','single_use','stage','campaign_id','source_snapshot_sha256','campaign_raw_sha256',
              'runtime_pins','budgets','review','predecessor','initial_accounting'}
    if (set(release)!=expected or release['schema']!=RELEASE_SCHEMA or release['state']!='approved'
        or release['single_use'] is not True or release['stage']!=stage or not isinstance(release['campaign_id'],str)
        or _ID.fullmatch(release['campaign_id']) is None or release['source_snapshot_sha256']!=sha256_file(snapshot_path)
        or release['campaign_raw_sha256']!=sha256_file(root/CAMPAIGN_PATH)
        or canonical_sha256(release['runtime_pins'])!=canonical_sha256(campaign['runtime_pins'])
        or canonical_sha256(release['budgets'])!=canonical_sha256(BUDGETS)):
        raise ValueError('release does not bind exact reviewed source/package/pins/budgets/stage')
    inventory=_source_inventory(root,_source_files(root,SNAPSHOT_SCOPES))
    if (snapshot.get('schema')!=SNAPSHOT_SCHEMA or snapshot.get('state')!='reviewed'
        or snapshot.get('files')!=inventory or snapshot.get('source_inventory_sha256')!=canonical_sha256(inventory)
        or snapshot.get('campaign_raw_sha256')!=release['campaign_raw_sha256']):raise ValueError('reviewed source snapshot differs')
    review=release['review']
    if set(review)!={'path','sha256'} or sha256_file(_path(root,review['path'],exists=True))!=review['sha256']:
        raise ValueError('review binding changed')
    initial=_initial(root)
    if release['initial_accounting']!={'path':INITIAL.as_posix(),'sha256':INITIAL_SHA256}:raise ValueError('release initial accounting differs')
    predecessor=release['predecessor'];previous=None
    if stage=='profile':
        if predecessor is not None:raise ValueError('profile must consume only initial accounting')
    else:
        if not isinstance(predecessor,dict) or set(predecessor)!={'path','sha256','outer_accounting'}:raise ValueError('sustained stage needs sealed profile and enclosing accounting predecessor')
        path=_path(root,predecessor['path'],exists=True)
        if sha256_file(path)!=predecessor['sha256']:raise ValueError('profile predecessor changed')
        seal=_read(path)
        if (seal.get('schema')!=SEAL_SCHEMA or seal.get('status')!='complete' or seal.get('stage')!='profile'
            or seal.get('campaign_id')!=release['campaign_id'] or seal.get('source_sha256')!=release['source_snapshot_sha256']
            or (path.parent/'terminal-closure-failure.json').exists()):raise ValueError('profile seal is not accepted')
        result_path=_path(root,seal['result']['path'],exists=True)
        if sha256_file(result_path)!=seal['result']['sha256']:raise ValueError('sealed profile result changed')
        previous=_read(result_path)
        if (previous.get('schema')!=CLOSED_SCHEMA or previous.get('status')!='complete' or previous.get('stage')!='profile'
            or previous.get('campaign_id')!=release['campaign_id'] or previous.get('source_sha256')!=release['source_snapshot_sha256']
            or len(previous.get('jobs',[]))!=1 or previous['jobs'][0].get('status')!='complete'):
            raise ValueError('profile predecessor is not an accepted closed stage')
        verify_closed_stage(root,previous)
        outer_binding=predecessor['outer_accounting']
        if not isinstance(outer_binding,dict) or set(outer_binding)!={'path','sha256'}:raise ValueError('outer accounting raw binding required')
        outer_path=_path(root,outer_binding['path'],exists=True)
        if sha256_file(outer_path)!=outer_binding['sha256']:raise ValueError('outer accounting receipt changed')
        outer=_read(outer_path)
        if outer.get('seal_sha256')!=predecessor['sha256']:raise ValueError('outer accounting does not bind seal raw hash')
        charged=reconcile_outer_accounting(seal,outer)
    return {'campaign':campaign,'snapshot':snapshot,'release':release,'previous':previous,
        'source_sha256':sha256_file(snapshot_path),'release_sha256':sha256_file(release_path),
        'charged':dict(initial['charged_seconds'] if previous is None else charged),
        'initial':initial,'snapshot_path':snapshot_path,'release_path':release_path}


def _stage(root,destination,files):
    destination.mkdir()
    for relative,binding in files.items():
        source=_path(root,relative,exists=True);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);target.chmod(source.stat().st_mode&0o777)
    if _source_inventory(destination,tuple(files))!=files:raise ValueError('staged source bytes/modes differ')


def _sync_claim_ancestors(root,directory):
    """Persist every exclusive directory entry before any workload dispatch."""
    root=Path(root).resolve(strict=True);directory=_path(root,directory,exists=True)
    cursor=root
    low._fsync_claim_directory(root,cursor)
    for component in directory.relative_to(root).parts:
        cursor=cursor/component;low._fsync_claim_directory(root,cursor)


def _environment(runtime_class,output,deadline):
    scratch=output+'/.runtime/'
    return {**low.IMAGE_ENV,**{k:scratch+v for k,v in low.SCRATCH.items()},
        'PYTHONPATH':'/learner/src:/learner','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4',
        'SPIRE_LEARN_RUN_NEURAL_RUNTIME':'1','SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME':'1' if runtime_class=='gpu' else '0',
        'SPIRE_LEARN_NEURAL_GPU_UUID':low.GPU_UUID if runtime_class=='gpu' else '',
        'NVIDIA_VISIBLE_DEVICES':low.GPU_UUID if runtime_class=='gpu' else 'none',
        'CUDA_VISIBLE_DEVICES':'0' if runtime_class=='gpu' else '',
        'NVIDIA_DRIVER_CAPABILITIES':'compute,utility' if runtime_class=='gpu' else ''}


def _command():return ('-B','/learner/scripts/neural/sustained_runtime.py','_guest','--root','/learner','--context','/control/job.json')


def _spec(context,context_path):
    digest=sha256_file(context_path);env=_environment(context['runtime_class'],context['guest_output_root'],context['deadline'])
    env['SPIRE_LEARN_SUSTAINED_CONTEXT_SHA256']=digest
    labels={'sustained.campaign':context['campaign_id'],'sustained.job':context['job_id'],
        'sustained.source':context['source_sha256'],'sustained.release':context['release_sha256'],
        'sustained.claim':context['claim_sha256'],'sustained.context':digest}
    return low.PhaseSpec(context['job_id'],context['container_name'],context['campaign_id'],context['release_sha256'],digest,
        context['claim_sha256'],context['deadline'],context['whole_deadline'],context['guest_output_root'],_command(),
        tuple(sorted(env.items())),tuple((m['source'],m['destination'],m['readonly']) for m in context['mounts']),
        tuple(sorted(labels.items())),runtime_class=context['runtime_class'])


def verify_guest_context(root,context_path,*,clock=time.monotonic):
    """Verify selected guest scope without invoking the host full-package validator."""
    if root!=low._guest_path('/learner') or context_path!=low._guest_path('/control/job.json'):raise ValueError('guest entry namespace differs')
    context=_read(context_path);digest=sha256_file(context_path)
    if context.get('schema')!=CONTEXT_SCHEMA or os.environ.get('SPIRE_LEARN_SUSTAINED_CONTEXT_SHA256')!=digest:raise ValueError('guest context raw binding differs')
    release=_read(context_path.parent/'release.json');snapshot=_read(context_path.parent/'snapshot.json');claim=_read(context_path.parent/'claim.json')
    if (sha256_file(context_path.parent/'release.json')!=context['release_sha256']
        or sha256_file(context_path.parent/'snapshot.json')!=context['source_sha256']
        or sha256_file(context_path.parent/'claim.json')!=context['claim_sha256']
        or release.get('schema')!=RELEASE_SCHEMA or release.get('state')!='approved' or release.get('single_use') is not True
        or release.get('source_snapshot_sha256')!=context['source_sha256'] or release.get('campaign_id')!=context['campaign_id']
        or snapshot.get('state')!='reviewed' or claim.get('job_id')!=context['job_id'] or claim.get('reservation')!=context['reservation']
        or claim.get('release_sha256')!=context['release_sha256'] or claim.get('predecessor')!=context['predecessor']):
        raise ValueError('guest acyclic release/source/claim chain differs')
    if sha256_file(root/CAMPAIGN_PATH)!=release['campaign_raw_sha256']:raise ValueError('guest campaign raw binding differs')
    campaign=_read(root/CAMPAIGN_PATH)
    if canonical_sha256(release['runtime_pins'])!=canonical_sha256(campaign['runtime_pins']) or canonical_sha256(release['budgets'])!=canonical_sha256(campaign['budgets']):raise ValueError('guest release pins/budgets differ')
    inventory=_source_inventory(root,_source_files(root,GUEST_SCOPES))
    if inventory!=context['source_files'] or any(snapshot['files'].get(k)!=v for k,v in inventory.items()):raise ValueError('guest projected source differs')
    if not clock()<context['deadline']<context['whole_deadline'] or context['whole_deadline']-clock()>900.:
        raise TimeoutError('guest deadline invalid or expired')
    kind=context['kind'];gpu=kind in ('profile','segment')
    if kind not in ('profile','segment','evaluation') or context['runtime_class']!=('gpu' if gpu else 'cpu'):
        raise ValueError('job kind/runtime class differs')
    allowed={('/learner/'+scope,True) for scope in GUEST_SCOPES}
    allowed.update({('/runtime/venv-cuda',True),('/runtime/requirements-cuda.lock',True),('/control',True),
                    (context['guest_output_root'],False)})
    if kind=='profile':
        from scripts.neural.sustained_profile import OUTPUT_PREFIX
        if context['guest_output_root']!='/learner/'+OUTPUT_PREFIX:raise ValueError('profile logical output differs')
        allowed.update(('/learner/'+p['path'],True) for p in (*campaign['parents'].values(),*campaign['diagnostic_checkpoints'].values()))
    else:
        work,config=_selected_config(root,campaign,context)
        if context['parent'] is not None:
            if context['parent']['path']!='/inputs/parent':raise ValueError('selected parent mount differs')
            allowed.add(('/inputs/parent',True))
        if kind=='segment':
            if config.manifest_sha256!=campaign['training']['manifest']['canonical_sha256']:raise ValueError('training cases changed')
        else:
            scheduled=next((r for r in evaluation_schedule(campaign,{}) if r['operation_id']==context['operation']['operation_id']),None)
            if scheduled is None or any(scheduled[k]!=context['operation'][k] for k in scheduled if k not in ('checkpoint','availability')):
                raise ValueError('evaluation operation differs from fixed schedule')
            if config.manifest_sha256!=work['manifest_sha256']:raise ValueError('evaluation cases changed')
    projected=[(m['destination'],m['readonly']) for m in context['mounts']]
    if len(projected)!=len(allowed) or set(projected)!=allowed:raise ValueError('job application mount scope differs')
    spec=_spec(context,context_path);facts=low._guest_facts()
    for k,wanted in (('uid',1000),('gid',1000),('python',low.CONTAINER_PYTHON)):
        if facts.get(k)!=wanted:raise ValueError('executing process '+k+' differs')
    observed=cdi.validate_runtime_class(cdi.load_runtime_profile(root),runtime_class=context['runtime_class'],
        requested_environment=dict(spec.environment),actual_environment=facts['environment'],facts=facts)
    expected={destination:ro for _,destination,ro in spec.mounts};table=dict(facts['mounts'])
    if len(table)!=len(facts['mounts']) or table.get('/') is not True or any(table.get(k) is not v for k,v in expected.items()):
        raise ValueError('actual application mounts/root differ')
    admitted=set(observed['admitted_mount_destinations'])
    for destination,ro in facts['mounts']:
        auto=any(destination==p or destination.startswith(p+'/') for p in ('/proc','/dev','/sys'))
        if destination in expected or destination=='/' or destination in ('/etc/hosts','/etc/hostname','/etc/resolv.conf') or destination in admitted:continue
        if destination.startswith(('/learner','/control','/runtime')) or not auto:
            raise ValueError('unexpected executing guest mount: '+destination)
    low._verify_guest_runtime()
    packages=low._runtime_metadata(low._guest_path('/runtime/venv-cuda/lib/python3.12/site-packages'))
    for name,version in {'torch':'2.14.1+cu130','numpy':'2.5.4','gymnasium':'1.4.0','spire-lab':'0.1.0'}.items():
        if packages.get(name)!=version:raise ValueError('executing installed distribution differs: '+name)
    record={'schema':'spire-learn-sustained-admission-v1','job_id':context['job_id'],'context_sha256':digest,
        'runtime_class':context['runtime_class'],'process':{k:facts[k] for k in ('uid','gid','python','supplementary_group_ids')},
        'runtime_profile':observed,'installed_distributions':packages}
    encoded=MARKER+json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>16384:raise ValueError('guest admission evidence exceeds bound')
    if sys.stderr.write(encoded)!=len(encoded):raise OSError('short admission write')
    sys.stderr.flush()
    return context,campaign


def _mapped(context,value):
    guest=Path(context['guest_output_root']);raw=Path(value)
    if '..' in raw.parts or not raw.is_relative_to(guest):raise ValueError('receipt path outside fixed guest output namespace')
    return _path(Path(context['host_output']),raw.relative_to(guest),exists=True)


def validate_effects(context,receipt,output):
    """Reconstruct actual sparse callback delivery, explicitly including zeros."""
    totals={k:dict(reserved=0,attempted=0,delivered=0,uncertain=0) for k in EFFECT_KINDS};pending={};seen=set();categories={}
    count=0;prefix=context['operation']['operation_id'] if context.get('kind')=='evaluation' else context.get('job_id')
    labels={'native-step':'environment_attempts','reset':'reset_slots','training_update':'optimizer_attempts','disposable_update':'optimizer_attempts',
        'fixed_controller_random_choice':'fixed_random_choices','stochastic_actor_rows':'sample_rows','deterministic_actor_rows':'sample_rows',
        'diagnostic_score_rows':'sample_rows','stochastic_reference_rows':'sample_rows','deterministic_reference_rows':'sample_rows'}
    for part in receipt['journal']:
        if set(part)!={'path','sha256','size_bytes'} or '/' in part['path'] or not re.fullmatch(r'effects-[0-9]{4}\.json',part['path']):raise ValueError('journal part shape differs')
        file=_path(output,part['path'],exists=True)
        if sha256_file(file)!=part['sha256'] or file.stat().st_size!=part['size_bytes']:raise ValueError('effect journal changed')
        for row in _read(file)['records']:
            count+=1
            if count>65536:raise ValueError('effect journal record guard')
            identifier=row['operation_id'];kind=row['kind'];integer(row['reserved'],'reserved')
            if kind not in totals or row['schema']!='spire-learn-profile-effect-v1':raise ValueError('effect kind/schema differs')
            if labels.get(row['label'])!=kind:raise ValueError('effect kind/label differs')
            if row['event']=='reserve':
                if set(row)!={'event','schema','operation_id','kind','label','reserved'} or identifier in seen:raise ValueError('duplicate/malformed effect reservation')
                if prefix is not None and identifier!=f'{prefix}:{len(seen)+1:06d}':raise ValueError('effect ID prefix/sequence differs')
                seen.add(identifier);pending[identifier]=row
                totals[kind]['reserved']+=row['reserved']
            elif row['event']=='settle':
                original=pending.pop(identifier)
                if set(row)!={*original,'attempted','delivered','uncertain','unused'} or any(row[k]!=v for k,v in original.items() if k!='event'):
                    raise ValueError('effect settlement identity differs')
                for k in ('attempted','delivered','uncertain','unused'):integer(row[k],k)
                if row['attempted']+row['unused']!=row['reserved'] or row['delivered']+row['uncertain']!=row['attempted']:raise ValueError('effect settlement arithmetic differs')
                category=categories.setdefault(row['label'],dict(reserved=0,attempted=0,delivered=0,uncertain=0));category['reserved']+=row['reserved']
                for k in ('attempted','delivered','uncertain'):totals[kind][k]+=row[k];category[k]+=row[k]
            else:raise ValueError('unknown effect journal event')
    if pending:raise RuntimeError('unresolved accepted effect reservations')
    if totals!=receipt['effects'] or categories!=receipt['effect_categories']:raise ValueError('callback totals/categories disagree with produced receipt')
    if any(v['reserved']>context['reservation'][k] or v['uncertain'] for k,v in totals.items()):raise RuntimeError('job reservation exceeded or uncertain')
    return totals


def _validate_reports(context,receipt,*,allow_incomplete=False):
    files=receipt['report_files'];directory=_mapped(context,files['path'])
    if directory.parent!=Path(context['host_output']) or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}',directory.name):
        raise ValueError('actual runner UUID/output root differs')
    if set(files['inventory'])!={'summary.json','lifecycle.json','events.jsonl'}:raise ValueError('report inventory incomplete')
    for name,binding in files['inventory'].items():
        file=_path(directory,name,exists=True)
        if sha256_file(file)!=binding['sha256'] or file.stat().st_size!=binding['size_bytes']:raise ValueError('raw runner report changed')
    lifecycle=_read(directory/'lifecycle.json');summary=receipt['report']
    statuses=('complete','incomplete') if allow_incomplete else ('complete',)
    expected_exit=0 if lifecycle.get('status')=='complete' else 1
    if lifecycle!=summary.get('lifecycle') or lifecycle.get('status') not in statuses or lifecycle.get('exit_code')!=expected_exit or lifecycle.get('close_errors'):
        raise ValueError('actual lifecycle receipt does not establish clean completion')
    if summary.get('status') not in statuses or summary.get('fault') or summary.get('lifecycle_receipt_error'):raise RuntimeError('runner reported failure')
    envelope=_read(directory/'summary.json');disk=envelope.get('summary')
    if not isinstance(disk,dict) or envelope.get('record_type')!='summary' or envelope.get('summary_sha256')!=canonical_sha256(disk):raise ValueError('raw summary envelope/hash differs')
    final_fields={'status','exit_code','elapsed_seconds','total_elapsed_seconds_with_resume_prefix','wall_overshoot_seconds','lifecycle'}
    if any(summary.get(k)!=v for k,v in disk.items() if k not in final_fields):raise ValueError('raw runner summary differs from receipt projection')
    return directory


def validate_segment_receipt(root,campaign,context,receipt):
    from spire_learn.neural.checkpoint import preflight_checkpoint
    from spire_learn.neural.runner import _selected_identity
    if (receipt.get('schema')!=RECEIPT_SCHEMA or receipt.get('kind')!='segment' or receipt.get('job_id')!=context['job_id']
        or receipt.get('source_sha256')!=context['source_sha256'] or receipt.get('status')!='complete'
        or receipt.get('parent')!=json_value(context['parent']) or receipt.get('predecessor')!=json_value(context['predecessor'])):
        raise ValueError('segment receipt source/parent/predecessor identity differs')
    # Standalone unit calls use the same namespace. A host invocation supplies a
    # distinct physical root, never rewriting the raw guest report/checkpoint.
    host_context={**context,'host_output':context.get('host_output',context['guest_output_root'])}
    directory=_validate_reports(host_context,receipt);path=_mapped(host_context,receipt['checkpoint']['path'])
    if path!=directory/'checkpoint-final' or checkpoint_inventory(path)!=receipt['checkpoint']['inventory']:raise ValueError('published checkpoint inventory differs')
    # Parse host config with its unchanged logical raw bytes; output namespace is
    # checked in the guest, while the host maps only the bound output prefix.
    work=next(row for row in campaign['training']['runs'] if row['policy_id']==context['replica'])
    from spire_learn.neural.config import load_neural_config
    config=load_neural_config(Path(root)/work['config']['path'])
    loaded=preflight_checkpoint(path,_selected_identity(config),purpose='resume')
    from spire_learn.neural.device import plan_runtime_device,validate_resume_device
    validate_resume_device(loaded.metadata['run_metadata'],plan_runtime_device(config))
    totals=receipt['effects'] if 'journal' not in receipt else validate_effects(context,receipt,Path(host_context['host_output']))
    saved,delta=reconcile_segment(config,context,receipt['report'],loaded.metadata,totals,elapsed=finite(receipt['elapsed_seconds'],'elapsed'))
    if saved!=receipt['checkpoint']['counters'] or delta!=receipt['delta']:raise ValueError('checkpoint prefix/delta receipt changed')
    return {'path':str(path),'inventory':checkpoint_inventory(path),'counters':saved,'delta':delta}


def _storage(root,h,deps,baseline,*,reservation=0):
    new=low.tree_bytes(h,strict=True);total=low.tree_bytes(root/'artifacts/development/neural-combat',strict=False)
    free=deps.disk_usage(root).free
    if new+reservation>BUDGETS['storage']['new_bytes'] or total+reservation>BUDGETS['storage']['cumulative_bytes'] or free-reservation<BUDGETS['storage']['minimum_free_bytes']:
        raise RuntimeError('storage admission ceiling/floor exhausted')
    return {'new_bytes':new,'cumulative_bytes':total,'free_bytes':free,'reservation_bytes':reservation}


def execute_job(spec,*,deps,client,docker,logs,progress):
    """Failed create never establishes ownership; cleanup uses only a full ID."""
    cid=None;primary=None;removed=False;admission=None
    def call(*args,maximum=15.):
        return low._command(deps,low._docker(docker,client,*args),deadline=spec.whole_deadline,logs=logs,client=client,maximum=maximum,progress=progress)
    try:
        existing=call('ps','-a','--filter','name=^/'+spec.name+'$','--format','{{.ID}}')
        if existing.strip():raise RuntimeError('exact container name already exists')
        output=low._command(deps,low.build_create_argv(spec,docker,client),deadline=spec.whole_deadline,logs=logs,client=client,progress=progress)
        if re.fullmatch(r'[0-9a-f]{64}\n?',output) is None:raise RuntimeError('create did not return a full immutable container ID')
        cid=output.strip()
        low.assert_phase_inspect(call('inspect',cid),spec=spec,container_id=cid)
        attached=spec.whole_deadline-deps.clock()-40.
        if attached<=0:raise TimeoutError('cleanup reservation prevents container start')
        call('start','--attach',cid,maximum=attached)
        final=low.assert_phase_inspect(call('inspect',cid,maximum=10.),spec=spec,container_id=cid,lifecycle='finished')
        state=final['State']
        if state.get('ExitCode')!=0 or state.get('OOMKilled') is not False or state.get('Error')!='':raise RuntimeError('container failed/OOM/error')
        start=next(row for row in reversed(logs) if 'start' in row['argv'])
        lines=[line[len(MARKER):] for line in start['stderr'].splitlines() if line.startswith(MARKER)]
        if len(lines)!=1 or len(lines[0].encode())>16384:raise ValueError('bounded guest admission evidence missing/duplicate')
        admission=json.loads(lines[0],object_pairs_hook=low._unique,parse_constant=low._bad_constant)
        if admission.get('context_sha256')!=spec.context_sha256 or admission.get('job_id')!=spec.phase or admission.get('runtime_class')!=spec.runtime_class:
            raise ValueError('guest admission identity differs')
    except BaseException as error:primary=error
    finally:
        signals=_signals.get()
        if signals is not None:signals['cleanup']=True
        if cid is not None:
            try:
                low.assert_phase_inspect(call('inspect',cid,maximum=10.),spec=spec,container_id=cid,lifecycle='cleanup')
                # Mark before dispatch. Even an interrupted/uncertain rm is never retried.
                removed=True;call('rm','--force',cid,maximum=10.)
                if call('ps','-a','--filter','id='+cid,'--format','{{.ID}}',maximum=10.).strip():raise RuntimeError('owned full ID still present after removal')
            except BaseException as error:
                if primary is None:primary=error
                else:primary.add_note('owned cleanup also failed: '+str(_error(error)))
        if signals is not None:
            signals['cleanup']=False
            if signals['deferred'] is not None:
                if primary is None:primary=signals['deferred']
                else:primary.add_note('control signal deferred during owned cleanup')
                signals['deferred']=None
    if primary is not None:raise primary
    return {'container_id':cid,'removal_entered':removed,'admission':admission}


def _job(root,h,stage_tree,verified,budget,*,kind,job_id,reservation,output_root,parent=None,prefix=None,replica=None,index=None,operation=None,predecessor=None,deps,docker,client):
    """One immutable host claim/context and fresh physical output bind."""
    # All full-package/source/storage checks precede exclusive claims.
    validate_campaign(root,root/CAMPAIGN_PATH)
    if _source_inventory(root,tuple(verified['snapshot']['files']))!=verified['snapshot']['files']:raise ValueError('source changed before admission')
    budget.check();bucket='profile' if kind=='profile' else replica if kind=='segment' else 'evaluation'
    allowance=min(900. if kind=='profile' else 600.,budget.remaining(bucket))
    if allowance<=75.:return None
    _storage(root,h.parent,deps,None,reservation=(1024 if kind=='profile' else 512 if kind=='segment' else 128)*1024**2)
    if kind=='segment' and index:
        if not isinstance(predecessor,dict) or set(predecessor)!={'path','sha256'}:raise ValueError('resumed host job requires actual predecessor receipt')
        previous_path=_path(root,predecessor['path'],exists=True)
        if sha256_file(previous_path)!=predecessor['sha256']:raise ValueError('actual predecessor receipt changed')
        previous=_read(previous_path)
        raw_path=Path(previous['checkpoint']['path']);logical=Path(output_root)
        if (not raw_path.is_relative_to(logical) or previous['checkpoint']['inventory']!=parent['inventory']
            or previous['checkpoint']['counters']!=prefix or previous.get('status')!='complete'
            or previous.get('source_sha256')!=verified['source_sha256']
            or previous_path.parent/raw_path.relative_to(logical)!=_path(root,parent['path'],exists=True)):
            raise ValueError('actual parent/predecessor/prefix chain differs')
    budget.switch(bucket);allowance=min(allowance,budget.remaining(bucket));started=deps.clock();whole=started+allowance;deadline=whole-60.
    jobdir=h/'jobs'/job_id;jobdir.mkdir(parents=True,exist_ok=False)
    output=jobdir/'output';output.mkdir();control=jobdir/'control';control.mkdir()
    _sync_claim_ancestors(root,control)
    campaign=verified['campaign'];runtime_class='cpu' if kind=='evaluation' else 'gpu'
    claim={'schema':'spire-learn-sustained-job-claim-v1','job_id':job_id,'release_sha256':verified['release_sha256'],
        'predecessor':predecessor,'reservation':reservation,'deadline':deadline,'whole_deadline':whole}
    claimpath=control/'claim.json';write_json_exclusive(claimpath,claim)
    shutil.copyfile(verified['release_path'],control/'release.json');shutil.copyfile(verified['snapshot_path'],control/'snapshot.json')
    mounts=[{'source':str(stage_tree/scope),'destination':'/learner/'+scope,'readonly':True} for scope in GUEST_SCOPES]
    mounts += [{'source':str(root/low.VENV),'destination':'/runtime/venv-cuda','readonly':True},
        {'source':str(root/low.LOCK),'destination':'/runtime/requirements-cuda.lock','readonly':True},
        {'source':str(control),'destination':'/control','readonly':True},
        {'source':str(output),'destination':output_root,'readonly':False}]
    guest_parent=None
    if parent is not None:
        actual=_path(root,parent['path'],exists=True)
        if checkpoint_inventory(actual)!=parent['inventory']:raise ValueError('current parent actual inventory changed')
        guest_parent={'path':'/inputs/parent','inventory':parent['inventory']}
        mounts.append({'source':str(actual),'destination':'/inputs/parent','readonly':True})
    context={'schema':CONTEXT_SCHEMA,'kind':kind,'job_id':job_id,'campaign_id':verified['release']['campaign_id'],
        'container_name':'spire-sustained-'+verified['release']['campaign_id']+'-'+job_id,'runtime_class':runtime_class,
        'source_sha256':verified['source_sha256'],'source_files':_source_inventory(stage_tree,_source_files(stage_tree,GUEST_SCOPES)),
        'release_sha256':verified['release_sha256'],'claim_sha256':sha256_file(claimpath),'deadline':deadline,'whole_deadline':whole,
        'guest_output_root':output_root,'host_output':str(output),'mounts':mounts,'reservation':reservation,
        'parent':guest_parent,'parent_host_binding':parent,'prefix':prefix or {},'predecessor':predecessor,'replica':replica,'segment_index':index,'operation':operation}
    if kind=='profile':
        from scripts.neural.sustained_profile import profile_input_inventory
        inputs=profile_input_inventory(root,campaign)
        bindings={relative:{**binding,'path':'/learner/'+relative} for relative,binding in inputs.items()}
        for binding in (*campaign['parents'].values(),*campaign['diagnostic_checkpoints'].values()):
            mounts.append({'source':str(root/binding['path']),'destination':'/learner/'+binding['path'],'readonly':True})
        context['profile_payload']={'phase_id':job_id,'source_sha256':verified['source_sha256'],'deadline':deadline,
            'input_bindings':bindings,'job_reservation':reservation}
    path=control/'job.json';write_json_exclusive(path,context);spec=_spec(context,path);logs=[];primary=None;result=None;gpu_before=None;gpu_after=None
    try:
        if runtime_class=='gpu':
            cdi.verify_host_profile(cdi.load_runtime_profile(root))
            raw=low._command(deps,['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.free','--format=csv,noheader,nounits'],
                deadline=whole,logs=logs,client=client,progress=budget.check)
            from scripts.neural.task8_gpu_setup_helpers import verify_gpu_inventory
            verify_gpu_inventory(raw,minimum_free_mib=12288)
            gpu_before=low._gpu_processes(deps,whole,logs,client)
        lifecycle=execute_job(spec,deps=deps,client=client,docker=docker,logs=logs,progress=budget.check)
        if runtime_class=='gpu':gpu_after=low._gpu_processes(deps,whole,logs,client)
        receipt_path=output/'job-receipt.json';receipt=_profile_receipt(receipt_path) if kind=='profile' else _read(receipt_path)
        if kind=='profile':
            from scripts.neural.sustained_profile import validate_profile_report,profile_plan
            report={k:v for k,v in receipt.items() if k not in ('journal','process')}
            if report['source_sha256']!=verified['source_sha256'] or report['phase_id']!=job_id:raise ValueError('profile source/job identity differs')
            validate_profile_report(report,profile_plan(root,campaign))
            validate_effects(context,receipt,output)
        elif kind=='segment':
            checkpoint=validate_segment_receipt(root,campaign,context,receipt)
        else:
            if receipt.get('schema')!=RECEIPT_SCHEMA or receipt.get('operation_id')!=operation['operation_id'] or receipt.get('source_sha256')!=verified['source_sha256']:
                raise ValueError('evaluation receipt identity differs')
            _validate_reports(context,receipt,allow_incomplete=True);totals=validate_effects(context,receipt,output)
            if totals['optimizer_attempts']['attempted'] or totals['reset_slots']['attempted']>1 or totals['environment_attempts']['attempted']>256:raise ValueError('evaluation effects differ')
            summary=receipt['report'];collection=summary['collection']
            if summary.get('config_sha256')!=operation['config']['config_sha256']:raise ValueError('evaluation config identity differs')
            for effect,attempted,delivered in (('environment_attempts','attempted_environment_slots','delivered_environment_slots'),('reset_slots','attempted_reset_slots','delivered_reset_slots')):
                if totals[effect]['attempted']!=collection[attempted] or totals[effect]['delivered']!=collection[delivered]:raise ValueError('evaluation delivery counters differ')
            if collection['faults'] or collection['uncertain_environment_slots'] or collection['uncertain_reset_slots'] or any(summary['optimizer'][key] for key in ('attempted_steps','accepted_steps','uncertain_steps','updates')):
                raise RuntimeError('evaluation effect/optimizer fault')
            if operation['availability']=='reference':
                imports=receipt['reference_import_guard']
                if not imports or imports.get('attempts') or imports.get('loaded_modules') or imports.get('installed') is not True:raise ValueError('reference import evidence differs')
        process=receipt.get('process')
        if not isinstance(process,dict):raise ValueError('executing process evidence missing')
        modules=process.get('modules',{});threads=process.get('threads')
        neural=kind!='evaluation' or operation['availability']!='reference'
        if neural and ('torch' not in modules or modules['torch']['version']!='2.14.1+cu130' or threads!={'intraop':4,'interop':1}):raise ValueError('executing Torch/thread evidence differs')
        if any(not item.get('file','').startswith('/runtime/venv-cuda/lib/python3.12/site-packages/') for item in modules.values()):
            raise ValueError('executing module escaped qualified runtime')
        result={'job_id':job_id,'kind':kind,'status':'complete','reservation':reservation,'lifecycle':lifecycle,
            'context':{'path':str(path.relative_to(root)),'sha256':sha256_file(path)},
            'receipt':{'path':str(receipt_path.relative_to(root)),'sha256':sha256_file(receipt_path)},'effects':receipt['effects'],
            'unrelated_gpu_process_observations':{'before':gpu_before,'after':gpu_after}}
        if kind=='segment':result['checkpoint']={k:v for k,v in checkpoint.items() if k!='delta'};result['delta']=checkpoint['delta']
        if kind=='evaluation':result['operation_id']=operation['operation_id'];result['outcome_state']=receipt['report']['status']
    except BaseException as error:primary=error
    finally:
        try:write_json_exclusive(jobdir/'commands.json',{'commands':logs})
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note('command retention also failed: '+str(_error(error)))
    try:
        _storage(root,h.parent,deps,None)
        if deps.clock()>whole:raise TimeoutError('whole job deadline exceeded after cleanup/postcheck')
        write_json_exclusive(jobdir/'host-job-result.json',result if primary is None else
            {'job_id':job_id,'kind':kind,'status':'failed','reservation':reservation,'error':_error(primary),
             'context':{'path':str(path.relative_to(root)),'sha256':sha256_file(path)},
             'accounting':{'actual_effects':None,'maximum_uncertain_effects':reservation,'reason':'no complete reconciled guest receipt'},
             'retained_output_files':_source_inventory(output,_source_files(output,tuple(p.name for p in output.iterdir() if p.is_file())) )})
    except BaseException as error:
        if primary is None:primary=error
        else:primary.add_note('host job result publication also failed: '+str(_error(error)))
    try:budget.switch('shared')
    except BaseException as error:
        if primary is None:primary=error
        else:primary.add_note('job interval charge also failed: '+str(_error(error)))
    if primary is not None:raise primary
    return result


def verify_closed_stage(root,receipt):
    if receipt.get('schema')!=CLOSED_SCHEMA or receipt.get('status')!='complete':raise ValueError('stage is not closed/complete')
    for job in receipt['jobs']:
        for field in ('context','receipt'):
            item=job[field]
            if sha256_file(_path(root,item['path'],exists=True))!=item['sha256']:raise ValueError('closed stage artifact changed')
    if len(receipt['schedule'])!=704 or len({r['operation_id'] for r in receipt['schedule']})!=704:raise ValueError('closed stage schedule differs')
    BudgetAccount(receipt['charged_seconds'],clock=lambda:0.).close()


def run_campaign(root: Path,*,campaign_path: Path,source_snapshot: Path,release_path: Path,stage: str,dependencies=None)->dict:
    """Consume externally reviewed stage authority once; never mint a release."""
    root=Path(root).resolve(strict=True);deps=dependencies or HostDependencies();started=finite(deps.clock(),'stage start')
    verified=validate_release(root,campaign_path=campaign_path,source_snapshot=source_snapshot,release_path=release_path,stage=stage)
    # Metadata and read-only pin checks are preflight, not workload qualification.
    low._verify_runtime(root,verified['release'])
    campaign=verified['campaign'];campaign_id=verified['release']['campaign_id'];h=_path(root,DOMAIN/campaign_id)
    budget=BudgetAccount(verified['charged'],clock=deps.clock);budget.mark=started;budget.check()
    jobs=[];slots={};primary=None;closed=None
    schedule=list(evaluation_schedule(campaign,slots));states={r['operation_id']:dict(r) for r in schedule}
    training_states={replica+'-segment-'+f'{index:03d}':{'job_id':replica+'-segment-'+f'{index:03d}',
        'replica':replica,'segment_index':index,'state':'not_started'} for replica in campaign['training']['order'] for index in range(256)}
    stage_dir=h/stage
    try:
        h.mkdir(parents=True,exist_ok=True)
        stage_dir.mkdir(exist_ok=False)  # Exclusive stage claim, retained on all failures.
        _sync_claim_ancestors(root,stage_dir)
        write_json_exclusive(stage_dir/'stage-claim.json',{'schema':'spire-learn-sustained-stage-claim-v1','stage':stage,
            'release_sha256':verified['release_sha256'],'source_sha256':verified['source_sha256'],'predecessor':verified['release']['predecessor']})
        write_json_exclusive(stage_dir/'schedule-plan.json',{'evaluation':schedule,'training':list(training_states.values()),
            'dispatch_order':[row['operation_id'] for row in evaluation_dispatch(campaign,{})]})
        client=stage_dir/'docker-client';client.mkdir();docker=low._docker_binary();logs=[]
        from scripts.neural.task8_gpu_setup_helpers import verify_image_inspect
        image=verify_image_inspect(low._command(deps,low._docker(docker,client,'image','inspect',low.IMAGE),
            deadline=deps.clock()+min(30.,budget.remaining('shared')),logs=logs,client=client,progress=budget.check))
        if image['image_id']!=low.IMAGE_ID:raise ValueError('immutable image ID differs')
        write_json_exclusive(stage_dir/'image-commands.json',{'commands':logs})
        source=stage_dir/'source';_stage(root,source,verified['snapshot']['files'])
        if stage=='profile':
            from scripts.neural.sustained_profile import profile_plan,OUTPUT_PREFIX
            plan=profile_plan(root,campaign)
            job=_job(root,stage_dir,source,verified,budget,kind='profile',job_id='profile',reservation=plan['reservation_ceiling'],
                output_root='/learner/'+OUTPUT_PREFIX,deps=deps,docker=docker,client=client)
            if job is None:raise TimeoutError('profile admission budget exhausted')
            jobs.append(job)
        else:
            completed=set();parents={run['policy_id']:json_value(campaign['parents'][run['parent_id']]) for run in campaign['training']['runs']}
            prefixes={k:{} for k in parents};predecessors={k:None for k in parents};segments={k:0 for k in parents};effects={k:{kind:0 for kind in EFFECT_KINDS} for k in parents}
            def evaluate_available(*,anchors=False):
                nonlocal schedule
                schedule=list(evaluation_dispatch(campaign,slots))
                for operation in schedule:
                    identifier=operation['operation_id']
                    if identifier in completed or operation['availability']=='unavailable':continue
                    if (operation['checkpoint_slot']=='anchor' or operation['availability']=='reference') is not anchors:continue
                    if budget.remaining('evaluation')<=60.:return
                    jobid='eval-'+hashlib.sha256(identifier.encode()).hexdigest()[:24]
                    reservation=dict(environment_attempts=256,sample_rows=256,optimizer_attempts=0,reset_slots=1,fixed_random_choices=256)
                    binding=campaign['selection']['case_bindings'][operation['case_index']]
                    checkpoint=operation['checkpoint'];job=None
                    try:
                        job=_job(root,stage_dir,source,verified,budget,kind='evaluation',job_id=jobid,reservation=reservation,
                            output_root=binding['output_root'],parent=checkpoint,operation=operation,deps=deps,docker=docker,client=client)
                    except BaseException:
                        states[identifier]={**operation,'state':'failed','reason':'accepted_or_uncertain_job_failure'};raise
                    if job is None:return
                    jobs.append(job);completed.add(identifier);states[identifier]={**operation,'state':job['outcome_state'],'job_id':jobid,'receipt':job['receipt']}
            evaluate_available(anchors=True)
            active=set(parents)
            while active:
                progressed=False
                for run in campaign['training']['runs']:
                    replica=run['policy_id']
                    if replica not in active:continue
                    prefix=prefixes[replica];remaining_env=1000000-prefix.get('attempted_environment_slots',0)
                    remaining_sample=500000-effects[replica]['sample_rows'];remaining_opt=32768-prefix.get('optimizer_attempted_steps',0)
                    width=run['rollout']['rows']
                    if min(remaining_env,remaining_sample)<width or remaining_opt<=0 or budget.remaining(replica)<=60. or segments[replica]>=256:
                        active.remove(replica)
                        if segments[replica]:slots[replica+':final']=parents[replica]
                        continue
                    environment_cap=min(4096,remaining_env,remaining_sample)
                    reservation=dict(environment_attempts=environment_cap,sample_rows=min(4096,remaining_sample),
                        optimizer_attempts=min(remaining_opt,1024 if run['mode']=='standalone' else 256),
                        reset_slots=environment_cap+16,fixed_random_choices=environment_cap)
                    index=segments[replica];jobid=replica+'-segment-'+f'{index:03d}'
                    try:
                        job=_job(root,stage_dir,source,verified,budget,kind='segment',job_id=jobid,reservation=reservation,
                            output_root=run['output_root'],parent=parents[replica],prefix=prefix,replica=replica,index=index,
                            predecessor=predecessors[replica],deps=deps,docker=docker,client=client)
                    except BaseException:
                        training_states[jobid]['state']='failed';raise
                    if job is None:active.remove(replica);continue
                    training_states[jobid]['state']='complete'
                    jobs.append(job);progressed=True;segments[replica]+=1;prefixes[replica]=job['checkpoint']['counters']
                    for kind in EFFECT_KINDS:effects[replica][kind]+=job['effects'][kind]['attempted']
                    parents[replica]={'path':str(Path(job['checkpoint']['path']).relative_to(root)),'inventory':job['checkpoint']['inventory']}
                    predecessors[replica]=job['receipt']
                    for milestone,threshold in campaign['selection']['milestones'].items():
                        if replica+':'+milestone not in slots and budget.charged[replica]>=threshold:slots[replica+':'+milestone]=parents[replica]
                    if not job['delta']['attempted_environment_slots']:active.remove(replica);slots[replica+':final']=parents[replica]
                evaluate_available(anchors=False)
                if not progressed:break
            for replica in parents:
                if segments[replica]:slots.setdefault(replica+':final',parents[replica])
            evaluate_available(anchors=False)
        _storage(root,h,deps,None)
    except BaseException as error:primary=error
    finally:
        final_schedule=[]
        for row in evaluation_schedule(campaign,slots):
            existing=states[row['operation_id']]
            final_schedule.append(existing if existing['state']!='not_started' else {**row,'reason':'checkpoint_unavailable' if row['availability']=='unavailable' else 'stage_not_started_or_budget_exhausted'})
        # Include failed/uncertain consumed job claims without converting their
        # reservations into invented delivered counts or silently losing them.
        known={job['job_id'] for job in jobs}
        jobs_root=stage_dir/'jobs'
        if jobs_root.is_dir():
            for directory in sorted(jobs_root.iterdir()):
                if directory.name in known:continue
                try:
                    result_path=directory/'host-job-result.json'
                    if result_path.is_file():jobs.append(_read(result_path))
                    else:jobs.append({'job_id':directory.name,'status':'failed','accounting':{'actual_effects':None,'reason':'setup/control publication incomplete'}})
                except BaseException as error:
                    if primary is None:primary=error
                    else:primary.add_note('failed job inventory also failed: '+str(_error(error)))
        try:budget.switch('shared')
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note('outer accounting also failed: '+str(_error(error)))
        closed={'schema':CLOSED_SCHEMA,'stage':stage,'campaign_id':campaign_id,'source_sha256':verified['source_sha256'],
            'release_sha256':verified['release_sha256'],'status':'complete' if primary is None else 'failed','error':None if primary is None else _error(primary),
            'jobs':jobs,'schedule':final_schedule,'training_schedule':list(training_states.values()),'checkpoints':slots,'charged_seconds':dict(budget.charged),'intervals':list(budget.intervals),
            'predecessor':verified['release']['predecessor'],'initial_accounting':verified['release']['initial_accounting']}
        try:
            result_path=stage_dir/'closed-stage.json';write_json_exclusive(result_path,closed)
            digest=sha256_file(result_path)
            _storage(root,h,deps,None)
            if primary is None:verify_closed_stage(root,closed)
            budget.close()
            seal={'schema':SEAL_SCHEMA,'stage':stage,'campaign_id':campaign_id,'source_sha256':verified['source_sha256'],
                'release_sha256':verified['release_sha256'],'status':'complete' if primary is None else 'failed',
                'result':{'path':str(result_path.relative_to(root)),'sha256':digest},
                'incoming_charged_seconds':verified['charged'],'charged_seconds':dict(budget.charged),
                'stage_elapsed_seconds':sum(row['seconds'] for row in budget.intervals),
                'timing_boundary':'after stage result publication/hash/storage/inventory checks; before seal serialization and process exit; enclosing receipt carries residual to shared'}
            seal_path=stage_dir/'closure.json';write_json_exclusive(seal_path,seal)
            # Tail time belongs to the externally captured process interval. It
            # remains bounded here and is debited once by the next release.
            tail=finite(deps.clock()-budget.mark,'seal publication tail')
            if tail>min(BUCKETS['shared']-budget.charged['shared'],28800.-sum(budget.charged.values())):
                raise TimeoutError('seal publication exhausted remaining shared/global allowance')
            closed['closure']={'path':str(seal_path.relative_to(root)),'sha256':sha256_file(seal_path)}
            closed['charged_seconds']=dict(budget.charged)
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note('closed stage publication also failed: '+str(_error(error)))
            closed['status']='failed';closed['error']=_error(primary)
            try:write_json_exclusive(stage_dir/'terminal-closure-failure.json',{'error':_error(primary)})
            except BaseException as secondary:primary.add_note('terminal closure failure retention also failed: '+str(_error(secondary)))
    if primary is not None and not isinstance(primary,Exception):raise primary
    return closed


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('capture-source');capture.add_argument('--destination',type=Path,required=True)
    capture.add_argument('--learner-head',required=True);capture.add_argument('--dirty-json',type=Path,required=True)
    validate=sub.add_parser('validate');validate.add_argument('--campaign',type=Path,default=Path(CAMPAIGN_PATH))
    run=sub.add_parser('run');run.add_argument('--campaign',type=Path,default=Path(CAMPAIGN_PATH))
    run.add_argument('--source-snapshot',type=Path,required=True);run.add_argument('--release',type=Path,required=True);run.add_argument('--stage',choices=STAGES,required=True)
    run.add_argument('--watchdog-seconds',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.command=='capture-source':result=capture_source_snapshot(args.root,args.destination,learner_head=args.learner_head,dirty=_read(args.dirty_json));status='captured'
    elif args.command=='validate':result=validate_campaign(args.root,args.root/args.campaign);status='valid'
    else:
        if args.watchdog_seconds:
            verified=validate_release(args.root,campaign_path=args.campaign,source_snapshot=args.source_snapshot,release_path=args.release,stage=args.stage)
            print(max(0,math.floor(28800.-sum(verified['charged'].values()))-10));return 0
        if os.environ.get('SPIRE_LEARN_SUSTAINED_WATCHDOG')!='remaining+10':raise ValueError('run requires the sustained whole-stage watchdog wrapper')
        with signal_scope():result=run_campaign(args.root,campaign_path=args.campaign,source_snapshot=args.source_snapshot,release_path=args.release,stage=args.stage)
        status=result['status']
    print(json.dumps({'status':status,'stage':getattr(args,'stage',None)},separators=(',',':')))
    return 0 if status!='failed' else 1


if __name__=='__main__':raise SystemExit(main())
