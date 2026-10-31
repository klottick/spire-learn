"""Protected Task 9 five-phase study controls; importing this module has no effects.

Only an approved, reviewed, one-use campaign may enter ``run``. Preparation
commands never mint an approval, claim, context, or workload. See the runbook.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import tomllib

# Direct script and namespace-package imports share the same stdlib-only helpers.
if __package__ in (None, ''):
    module_root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(module_root/'src'),str(module_root)]
from scripts.neural.gpu_runtime_proof_helpers import (
    HostDependencies, _bounded_subprocess_run, _source_files, _source_inventory,
    canonical_sha256, load_json, sha256_file, write_json_exclusive,
)
from scripts.neural.task8_gpu_setup_helpers import (
    verify_image_inspect, verify_gpu_inventory, _inventory_entries,
)

from scripts.neural import task9_nvidia_cdi as cdi

PHASES = ('standalone-a', 'run-a', 'standalone-b', 'run-b', 'selection')
STUDY = Path('configs/neural/experiments/task9-v1/study.json')
STUDY_SHA256 = 'c4939fc7d882fadb265fbc5a4ced42c7809e6482720dc4bac7698188aa14f555'
EXPERIMENT = Path('artifacts/development/neural-combat/experiments/task9-v1-attempt-b')
SELECTION = EXPERIMENT / 'selection'
BINDINGS = SELECTION / 'control/checkpoint-bindings.json'
RUNTIME = Path('artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a')
VENV = RUNTIME / 'install-work/venv-cuda'
LOCK = RUNTIME / 'resolve-work/requirements-cuda.lock'
NATIVE = VENV / 'lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so'
LOCK_SHA256 = '92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45'
NATIVE_SHA256 = '071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9'
IMAGE_ID = 'sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0'
IMAGE = 'docker.io/library/python@' + IMAGE_ID
GPU_UUID = 'GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d'
CONTAINER_PYTHON = '/runtime/venv-cuda/bin/python3.12'
GIB = 1024**3
STREAM_LIMIT = 65536
ADMISSION_SCHEMA = 'spire-learn-task9-guest-admission-v1'
ADMISSION_MARKER = 'SPIRE_LEARN_TASK9_ADMISSION_V1 '
ADMISSION_LIMIT = 16 * 1024
CLAIMS_DOMAIN = Path('artifacts/development/neural-combat/task-9/study-campaign-claims')
FINAL_RECORD_RESERVATION = 32 * 1024**2
CONTROL_RECORD_LIMIT = 8 * 1024**2
APPROVAL_SCHEMA = 'spire-learn-task9-campaign-approval-v1'
SNAPSHOT_SCHEMA = 'spire-learn-task9-source-snapshot-v1'
CLAIM_SCHEMA = 'spire-learn-task9-campaign-claim-v1'
CONTEXT_SCHEMA = 'spire-learn-task9-phase-context-v1'
INVENTORY_SCHEMA = 'spire-learn-task9-closed-inventory-v1'
HELPERS = tuple('scripts/neural/' + x for x in (
    'task9_study_package.py', 'task9_study_runtime.py', 'task9_study_controls.py',
    'gpu_runtime_proof_helpers.py', 'task8_gpu_setup_helpers.py', 'task9_nvidia_cdi.py'))
GUEST_SCOPES = ('src', 'configs', *HELPERS)
SNAPSHOT_PATHS = (*GUEST_SCOPES, 'scripts/neural/run-task9-study.sh',
    'scripts/neural/task9-study-runbook.md', 'tests/unit/test_neural_study_controls.py',
    'docs/neural-study.md', 'docs/neural-combat.md', 'tests/unit/test_neural_study_cdi.py')
IMAGE_ENV = {
    'PATH':'/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    'LANG':'C.UTF-8', 'GPG_KEY':'7169605F62C751356D054A26A821E680E5FA6305',
    'PYTHON_VERSION':'3.12.15',
    'PYTHON_SHA256':'c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a',
}
SCRATCH = {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','TORCH_HOME':'torch',
    'TORCH_EXTENSIONS_DIR':'torch-extensions','CUDA_CACHE_PATH':'cuda',
    'TRITON_CACHE_DIR':'triton','HOME':'home'}
_ID = re.compile(r'[a-z0-9][a-z0-9-]{0,39}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')


class StudyControlError(ValueError):
    pass


class _Termination(BaseException):
    pass


def _finite_time(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise StudyControlError('monotonic time must be finite and nonnegative')
    return float(value)


def _now(deps):
    return _finite_time(deps.clock())


def _json(path: Path) -> dict:
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 8*1024**2:
        raise StudyControlError(f'unsafe or oversized JSON input: {path}')
    result=load_json(path)
    # The generic reader rejects duplicates; canonical JSON rejects NaN/Infinity.
    canonical_sha256(result)
    return result


def _safe(root: Path, value, *, exists=False) -> Path:
    root=Path(root).resolve(strict=True)
    raw=Path(value)
    if '..' in raw.parts:
        raise StudyControlError('parent traversal is forbidden')
    path=raw if raw.is_absolute() else root/raw
    if any(ch in str(path) for ch in (',','\n','\r','\x00')):
        raise StudyControlError('unsafe bind/control path spelling')
    if not path.is_relative_to(root):
        raise StudyControlError(f'path outside owned domain: {path}')
    current=root
    for component in path.relative_to(root).parts:
        current=current/component
        if current.is_symlink():raise StudyControlError(f'symlink in protected path: {current}')
    if exists and not path.exists():raise StudyControlError(f'missing input: {path}')
    return path


def _study(root: Path) -> dict:
    from scripts.neural.task9_study_package import validate_study
    if sha256_file(_safe(root,STUDY,exists=True)) != STUDY_SHA256:
        raise StudyControlError('study raw hash changed')
    return validate_study(root,root/STUDY)


def capture_source_snapshot(root: Path, destination: Path, *, learner_head: str,
                            dirty: Mapping, state='source_only') -> dict:
    root=Path(root).resolve(strict=True)
    if not re.fullmatch('[0-9a-f]{40}',learner_head) or state not in ('source_only','reviewed'):
        raise StudyControlError('invalid source snapshot identity/state')
    _study(root)
    paths=_source_files(root,SNAPSHOT_PATHS)
    source=_source_inventory(root,paths)
    value={'schema':SNAPSHOT_SCHEMA,'state':state,'learner_head':learner_head,
        'dirty':dict(dirty),'files':source,'study_raw_sha256':STUDY_SHA256}
    write_json_exclusive(destination,value)
    return value


def validate_approval(root: Path, snapshot_path: Path, approval_path: Path) -> dict:
    root=Path(root).resolve(strict=True)
    snapshot=_json(_safe(root,snapshot_path,exists=True))
    approval=_json(_safe(root,approval_path,exists=True))
    expected={'schema','state','single_use','campaign_id','source_snapshot_sha256',
        'study_raw_sha256','review_path','review_sha256','phase_order','runtime_pins',
        'budgets','checkpoint_binding'}
    study=_study(root)
    if (set(approval)!=expected or approval['schema']!=APPROVAL_SCHEMA
        or approval['state']!='approved' or approval['single_use'] is not True
        or not isinstance(approval['campaign_id'],str) or not _ID.fullmatch(approval['campaign_id'])
        or approval['phase_order']!=list(PHASES)
        or approval['source_snapshot_sha256']!=sha256_file(snapshot_path)
        or approval['study_raw_sha256']!=STUDY_SHA256
        or canonical_sha256(approval['runtime_pins'])!=canonical_sha256(study['runtime_pins'])
        or canonical_sha256(approval['budgets'])!=canonical_sha256(study['budgets'])
        or approval['checkpoint_binding']!='actual-post-training-final-inventory-once'):
        raise StudyControlError('campaign approval does not bind exact reviewed study and limits')
    if (snapshot.get('schema')!=SNAPSHOT_SCHEMA or snapshot.get('state')!='reviewed'
        or snapshot.get('study_raw_sha256')!=STUDY_SHA256
        or snapshot.get('files')!=_source_inventory(root,_source_files(root,SNAPSHOT_PATHS))):
        raise StudyControlError('reviewed source snapshot differs from actual public inputs')
    review=_safe(root,approval['review_path'],exists=True)
    if sha256_file(review)!=approval['review_sha256']:
        raise StudyControlError('final review hash changed')
    return {'approval':approval,'snapshot':snapshot,'study':study,
            'approval_raw_sha256':sha256_file(approval_path)}


def tree_bytes(root: Path, *, strict: bool) -> int:
    root=Path(root)
    if not root.exists():return 0
    if root.is_symlink():
        if strict:raise StudyControlError('linked accounting root')
        return root.lstat().st_size
    total=0
    for directory,names,files in os.walk(root,followlinks=False):
        for name in list(names)+files:
            path=Path(directory)/name
            info=path.lstat()
            if stat.S_ISLNK(info.st_mode):
                if strict:raise StudyControlError(f'linked phase output: {path}')
                total+=info.st_size
                if name in names:names.remove(name)
            elif stat.S_ISREG(info.st_mode):total+=info.st_size
            elif not stat.S_ISDIR(info.st_mode):raise StudyControlError('nonregular output entry')
    return total


def _runtime_metadata(site: Path) -> dict:
    return {d.metadata['Name'].lower().replace('_','-'):d.version
            for d in importlib.metadata.distributions(path=[str(site)])}


def _verify_runtime(root: Path, approval: Mapping) -> dict:
    profile=cdi.verify_host_profile(cdi.load_runtime_profile(root))
    for path,digest in ((LOCK,LOCK_SHA256),(NATIVE,NATIVE_SHA256)):
        if sha256_file(_safe(root,path,exists=True))!=digest:
            raise StudyControlError('qualified lock/native input changed')
    packages=_runtime_metadata(root/VENV/'lib/python3.12/site-packages')
    for name,version in {'torch':'2.14.1+cu130','numpy':'2.5.4','gymnasium':'1.4.0',
                         'spire-lab':'0.1.0'}.items():
        if packages.get(name)!=version:raise StudyControlError(f'qualified distribution differs: {name}')
    # Existing qualification is read as evidence, never replayed or rearmed.
    qualification=_json(_safe(root,RUNTIME/'device-check-work/device-check.json',exists=True))
    for name,value in {'status':'qualified','gpu_uuid':GPU_UUID,'visible_gpu_uuid':GPU_UUID,
                       'torch':'2.14.1+cu130','torch_cuda':'13.0','compute_capability':[12,0]}.items():
        if qualification.get(name)!=value:raise StudyControlError('qualified device evidence changed')
    return {'lock_sha256':LOCK_SHA256,'native_sha256':NATIVE_SHA256,'nvidia_cdi_profile':profile,
        'torch':packages['torch'],'torch_cuda':'13.0', 'venv':str(root/VENV),
        'qualification_sha256':sha256_file(root/RUNTIME/'device-check-work/device-check.json')}


def _recorded_path(relative: str) -> str:
    return '/learner/'+relative


def _verify_training_receipt(root: Path, record: Mapping) -> dict:
    from scripts.neural import task9_study_runtime as b
    from spire_learn.neural.config import load_neural_config
    path=_safe(root,Path(record['output_root'])/'operation-receipt.json',exists=True)
    r=_json(path)
    config=load_neural_config(root/record['config']['path'])
    if (r.get('schema_id')!='spire-learn-task9-training-operation-v1'
        or r.get('policy_id')!=record['policy_id'] or r.get('mode')!=record['mode']
        or r.get('study_raw_sha256')!=STUDY_SHA256
        or r.get('config')!={'path':record['config']['path'],
            'canonical_sha256':record['config']['config_sha256'],
            'raw_sha256':record['config']['raw_sha256']}
        or r.get('seed_evidence')!=record['seeds']
        or r.get('transport_exit_code')!=0 or r.get('status') not in ('eligible','unavailable')):
        raise StudyControlError('prior training receipt is not a healthy authored operation')
    _verify_threads(r.get('runtime_thread_settings'))
    manifest=_study(root)['training']['manifest']
    if (r.get('manifest')!={'path':manifest['path'],'canonical_sha256':manifest['canonical_sha256'],
                          'raw_sha256':manifest['raw_sha256']}
        or r.get('environment_revision')!=config.environment_revision or r.get('rollout')!=record['rollout']):
        raise StudyControlError('training manifest/environment/rollout mismatch')
    run=_safe(root,r.get('run_directory'),exists=True)
    if run.parent!=root/record['output_root'] or run.name!=r.get('run_id'):
        raise StudyControlError('training writer UUID/path mismatch')
    report=r.get('report');lifecycle=r.get('lifecycle')
    if not isinstance(report,dict) or not isinstance(lifecycle,dict):raise StudyControlError('missing training report/lifecycle')
    for key,name in (('summary','summary.json'),('lifecycle','lifecycle.json')):
        p=_safe(root,r.get(key+'_path'),exists=True)
        if p!=run/name or sha256_file(p)!=r.get(key+'_sha256'):
            raise StudyControlError('prior training report raw hash/path changed')
    disk_lifecycle=_json(run/'lifecycle.json')
    summary=_json(run/'summary.json');provisional=summary.get('summary')
    if (disk_lifecycle!=lifecycle or report.get('lifecycle')!=lifecycle
        or lifecycle.get('exit_code')!=r.get('runner_exit_code')
        or report.get('run_directory')!=_recorded_path(r['run_directory'])
        or not isinstance(provisional,dict)
        or summary.get('summary_sha256')!=canonical_sha256(provisional)
        or provisional.get('lifecycle',{}).get('status')!='receipt_pending'):
        raise StudyControlError('authoritative training lifecycle or provisional summary mismatch')
    for name,value in {'operation':'train','mode':config.mode,'split':'train',
        'config_sha256':config.config_sha256,'manifest_sha256':config.manifest_sha256,
        'environment_revision':config.environment_revision}.items():
        if report.get(name)!=value or provisional.get(name)!=value:
            raise StudyControlError('prior training report input identity mismatch')
    cp=r.get('checkpoint')
    if cp is not None:
        cp_path=_safe(root,cp['directory'],exists=True)
        if cp_path!=run/'checkpoint-final':raise StudyControlError('only final training checkpoint is eligible')
        current=b._checkpoint_inventory(root,cp_path,config,b._default_checkpoint_preflight)
        if current!=cp:raise StudyControlError('checkpoint metadata/payloads changed')
        b._validate_training_checkpoint_prefix(current,report,lifecycle,config)
    status,_=b._training_status(r,cp)
    if status!=r['status']:raise StudyControlError('training eligibility disagrees with actual accepted update evidence')
    return {'path':path.relative_to(root).as_posix(),'raw_sha256':sha256_file(path),
            'run_id':r['run_id'],'receipt':r}


def _verify_threads(value):
    if (not isinstance(value,dict) or type(value.get('process_id')) is not int
        or value['process_id']<=0 or type(value.get('intraop_threads')) is not int
        or value.get('intraop_threads')!=4 or type(value.get('interop_threads')) is not int
        or value.get('interop_threads')!=1 or value.get('verified_before_runtime_assembly') is not True
        or value.get('verified_after_runtime_assembly') is not True):
        raise StudyControlError('missing actual-process neural 4/1 thread evidence')


def verify_training_bindings(root: Path, study_path: Path, bindings_path: Path,
                             training_receipts: Mapping[str, Path | None]) -> dict:
    from scripts.neural import task9_study_runtime as b
    study=_study(root)
    if Path(study_path) not in (STUDY,root/STUDY) or set(training_receipts)!=set(PHASES[:-1]):
        raise StudyControlError('wrong training/study binding domain')
    bindings,raw=b._binding_file(root,(root/STUDY).read_bytes(),bindings_path)
    if bindings['stop_required_policy_ids']:raise StudyControlError('prior training fault blocks selection')
    for record,bound in zip(study['training']['runs'],bindings['policies'],strict=True):
        expected=root/record['output_root']/'operation-receipt.json'
        if training_receipts[record['policy_id']]!=expected:
            raise StudyControlError('missing actual prior training receipt')
        current=_verify_training_receipt(root,record)
        if bound.get('training_receipt')!={key:current[key] for key in ('path','raw_sha256','run_id')}:
            raise StudyControlError('frozen prior training receipt hash or UUID mismatch')
        r=current['receipt']
        if r['status']=='eligible':
            if (bound['status']!='available' or bound['checkpoint_directory']!=r['checkpoint']['directory']
                or bound['inventory']!=r['checkpoint']['inventory']
                or bound['model_tensors']!=r['checkpoint']['model_tensors']):
                raise StudyControlError('frozen checkpoint differs from actual training output')
        elif bound['status']!='unavailable':raise StudyControlError('unavailable policy substituted')
    return bindings


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    name: str
    campaign_id: str
    approval_sha256: str
    context_sha256: str
    phase_claim_sha256: str
    soft_deadline: float
    whole_deadline: float
    output: str
    command: tuple[str,...]
    environment: tuple[tuple[str,str],...]
    mounts: tuple[tuple[str,str,bool],...]
    labels: tuple[tuple[str,str],...]
    runtime_class: str | None = None

    def __post_init__(self):
        if self.runtime_class not in (None, 'cpu', 'gpu'):
            raise StudyControlError('runtime class must be cpu or gpu')
        if self.runtime_class is None and self.phase not in PHASES:
            raise StudyControlError('new job identity requires an explicit runtime class')

    @property
    def uses_gpu(self):
        return self.runtime_class == 'gpu' if self.runtime_class is not None else self.phase != 'selection'


def _phase_output(phase):
    return SELECTION if phase=='selection' else EXPERIMENT/'training'/phase


def _base_env(phase, output, deadline, approval_hash, claim_hash):
    scratch='/learner/'+output+'/.runtime/'
    env={**IMAGE_ENV,**{k:scratch+v for k,v in SCRATCH.items()},
        'PYTHONPATH':'/learner/src:/learner','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4',
        'SPIRE_LEARN_TASK9_APPROVAL_SHA256':approval_hash,
        'SPIRE_LEARN_TASK9_PHASE_CLAIM_SHA256':claim_hash,
        'SPIRE_LEARN_TASK9_DEADLINE':repr(deadline),
        'NVIDIA_VISIBLE_DEVICES':'none' if phase=='selection' else GPU_UUID,
        'CUDA_VISIBLE_DEVICES':'' if phase=='selection' else '0',
        'NVIDIA_DRIVER_CAPABILITIES':'compute,utility' if phase!='selection' else '',
    }
    if phase!='selection':
        env.update(SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME='1',SPIRE_LEARN_NEURAL_GPU_UUID=GPU_UUID)
    return env


def _base_command(phase,deadline):
    result=['-B','/learner/scripts/neural/task9_study_runtime.py',
        'selection' if phase=='selection' else 'train','--root','/learner',
        '--study',STUDY.as_posix(),'--phase-context','/control/phase.json',
        '--deadline',repr(deadline)]
    if phase=='selection':result+=['--bindings','/learner/'+BINDINGS.as_posix(),
                                  '--output-dir','/learner/'+SELECTION.as_posix()]
    else:result+=['--policy-id',phase,'--receipt','/learner/'+(_phase_output(phase)/'operation-receipt.json').as_posix()]
    return result


def _spec_from_context(root, context_path):
    context=_json(context_path)
    env=dict(context['base_environment'])
    digest=sha256_file(context_path)
    env['SPIRE_LEARN_TASK9_CONTEXT_SHA256']=digest
    labels={'task9.campaign':context['campaign_id'],'task9.phase':context['phase'],
        'task9.approval':context['approval_raw_sha256'],'task9.context':digest,
        'task9.claim':context['phase_claim_sha256']}
    return PhaseSpec(context['phase'],context['container_name'],context['campaign_id'],
        context['approval_raw_sha256'],digest,context['phase_claim_sha256'],
        context['soft_deadline'],context['whole_deadline'],context['output'],
        tuple(context['command']),tuple(sorted(env.items())),
        tuple((m['source'],m['destination'],m['readonly']) for m in context['mounts']),
        tuple(sorted(labels.items())))


def build_phase_spec(root: Path, *, campaign: Mapping, phase: str, phase_start: float,
                     source_tree: Path, context_path: Path,
                     bindings_path: Path | None) -> PhaseSpec:
    """Publish one immutable acyclic phase claim/context after host preflight."""
    root=Path(root);start=_finite_time(phase_start)
    if phase not in PHASES:raise StudyControlError('unknown phase')
    deadline=start+600.0;whole=min(start+900.0,campaign['whole_deadline'])
    claim={'schema':'spire-learn-task9-phase-claim-v1','campaign_id':campaign['campaign_id'],
        'phase':phase,'phase_index':PHASES.index(phase),
        'approval_raw_sha256':campaign['approval_raw_sha256'],
        'campaign_claim_sha256':campaign['claim_sha256'],
        'predecessors':campaign['predecessors'],'soft_deadline':deadline,'whole_deadline':whole}
    parent=context_path.parent
    parent.mkdir(parents=True,exist_ok=False)
    claim_path=parent/'phase-claim.json';write_json_exclusive(claim_path,claim)
    phase_hash=sha256_file(claim_path)
    output=_phase_output(phase).as_posix()
    mounts=[(str(source_tree/scope),'/learner/'+scope,True) for scope in GUEST_SCOPES]
    mounts += [(str(root/VENV),'/runtime/venv-cuda',True),
        (str(root/LOCK),'/runtime/requirements-cuda.lock',True),
        (str(parent),'/control',True),(str(root/output),'/learner/'+output,False)]
    bindings=None
    if phase=='selection':
        if bindings_path!=root/BINDINGS:raise StudyControlError('wrong immutable binding path')
        bindings=_json(bindings_path)
        mounts.append((str(bindings_path),'/learner/'+BINDINGS.as_posix(),True))
        for bound in bindings['policies']:
            if bound['status']=='available':
                relative=bound['checkpoint_directory']
                mounts.append((str(_safe(root,relative,exists=True)),'/learner/'+relative,True))
    context={'schema':CONTEXT_SCHEMA,'campaign_id':campaign['campaign_id'],'phase':phase,
        'container_name':'spire-learn-task9-'+campaign['campaign_id']+'-'+phase,
        'approval_raw_sha256':campaign['approval_raw_sha256'],'phase_claim_sha256':phase_hash,
        'campaign_claim_sha256':campaign['claim_sha256'],
        'source_snapshot_sha256':campaign['source_snapshot_sha256'],
        'source_files':campaign['guest_source_files'],'study_raw_sha256':STUDY_SHA256,
        'soft_deadline':deadline,'whole_deadline':whole,'output':output,
        'bindings_sha256':sha256_file(bindings_path) if bindings_path else None,
        'runtime_profile':{'path':cdi.PROFILE_PATH,'raw_sha256':cdi.PROFILE_SHA256,'profile_id':cdi.PROFILE_ID},
        'bindings':bindings,'command':_base_command(phase,deadline),
        'base_environment':_base_env(phase,output,deadline,campaign['approval_raw_sha256'],phase_hash),
        'mounts':[{'source':s,'destination':d,'readonly':r} for s,d,r in mounts],
        'predecessors':campaign['predecessors']}
    # Inputs are complete before create; none hashes its later containing record.
    for name,src in [('approval.json',campaign['approval_path']),('snapshot.json',campaign['snapshot_path']),
                     ('campaign-claim.json',campaign['claim_path'])]:shutil.copyfile(src,parent/name)
    write_json_exclusive(context_path,context)
    return _spec_from_context(root,context_path)


def _docker_binary():
    candidate=shutil.which('docker')
    if candidate is None:raise StudyControlError('local Docker CLI executable is unavailable')
    path=Path(candidate).resolve(strict=True)
    if not path.is_file() or not os.access(path,os.X_OK):raise StudyControlError('unsafe local Docker CLI')
    return str(path)


def _docker_env(client):
    env={'PATH':os.environ.get('PATH',os.defpath),'LC_ALL':'C','XDG_CONFIG_HOME':str(client)}
    if 'HOME' in os.environ:env['HOME']=os.environ['HOME']
    return env


def _docker(docker,client,*args):
    if client.is_symlink() or not client.is_dir() or any(client.iterdir()):
        raise StudyControlError('Docker config must remain fresh and empty')
    return [docker,'--host','unix:///var/run/docker.sock','--config',str(client),*args]


def build_create_argv(spec: PhaseSpec, docker: str, client: Path) -> list[str]:
    args=_docker(docker,client,'create','--name',spec.name,'--pull','never',
        '--platform','linux/amd64','--entrypoint',CONTAINER_PYTHON,
        '--user','1000:1000','--workdir','/learner','--network','none','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges:true',
        '--cpus','4','--memory',str(8*GIB),'--memory-swap',str(8*GIB),'--pids-limit','512')
    if spec.uses_gpu:args+=['--gpus','device='+GPU_UUID]
    for k,v in spec.environment:args+=['--env',k+'='+v]
    for k,v in spec.labels:args+=['--label',k+'='+v]
    for src,dst,ro in spec.mounts:
        if ',' in src or ',' in dst:raise StudyControlError('comma in protected bind path')
        args+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    return args+[IMAGE,*spec.command]


def _inspect_projection(spec, container_id):
    """Expected fields, also used by strict fake Docker; never normalizes inspect."""
    devices=[] if not spec.uses_gpu else [{'Driver':'','Count':0,
        'DeviceIDs':[GPU_UUID],'Capabilities':[['gpu']],'Options':{}}]
    return {'Id':container_id,'Name':'/'+spec.name,'Image':IMAGE_ID,
        'Platform':'linux','Config':{'Image':IMAGE,'User':'1000:1000','WorkingDir':'/learner',
            'Entrypoint':[CONTAINER_PYTHON],'Cmd':list(spec.command),
            'Env':[k+'='+v for k,v in spec.environment],'Labels':dict(spec.labels)},
        'HostConfig':{'ReadonlyRootfs':True,'NetworkMode':'none','CapDrop':['ALL'],
            'CapAdd':None,'SecurityOpt':['no-new-privileges:true'],'Privileged':False,
            'Memory':8*GIB,'MemorySwap':8*GIB,'NanoCpus':4*10**9,'PidsLimit':512,
            'DeviceRequests':devices,'Devices':[],'DeviceCgroupRules':None,'GroupAdd':None,
            'Tmpfs':{},'Binds':None,
            'PidMode':'','IpcMode':'private','UTSMode':'','UsernsMode':'',
            'VolumesFrom':None,'ExtraHosts':None,'PortBindings':{},'PublishAllPorts':False,
            'AutoRemove':False,'RestartPolicy':{'Name':'no','MaximumRetryCount':0}},
        'Mounts':[{'Type':'bind','Source':s,'Destination':d,'RW':not r} for s,d,r in spec.mounts],
        'State':{'Status':'created','Running':False,'ExitCode':0,'OOMKilled':False,'Error':''}}


def _inspect_value(raw):
    value=json.loads(raw,object_pairs_hook=_unique,parse_constant=_bad_constant)
    if not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],dict):
        raise StudyControlError('inspect must describe exactly one object')
    return value[0]


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise StudyControlError('duplicate JSON key')
        result[key]=value
    return result


def _bad_constant(value):
    raise StudyControlError('nonfinite JSON constant')


def _assert_owner(value,spec,cid):
    if (value.get('Id')!=cid or value.get('Name')!='/'+spec.name
        or value.get('Config',{}).get('Labels')!=dict(spec.labels)):
        raise StudyControlError('container cleanup ownership mismatch')


def assert_phase_inspect(raw: str | bytes, *, spec: PhaseSpec, container_id: str,
                         lifecycle: str = 'prestart') -> dict:
    actual=_inspect_value(raw);_assert_owner(actual,spec,container_id)
    expected=_inspect_projection(spec,container_id)
    for key in ('Image','Platform'):
        if actual.get(key)!=expected[key]:raise StudyControlError('container image/platform differs')
    for key,value in expected['Config'].items():
        observed=actual.get('Config',{}).get(key)
        if key=='Env':
            if (not isinstance(observed,list) or len(observed)!=len(value)
                or sorted(observed)!=sorted(value)):raise StudyControlError('container exact environment differs')
        elif observed!=value:raise StudyControlError(f'container Config.{key} differs')
    host=actual.get('HostConfig',{})
    for key,value in expected['HostConfig'].items():
        observed=host.get(key)
        # Docker encodes unused list/map fields as null or their empty value.
        if value in (None,[],{}) and observed in (None,[],{}):continue
        if type(observed)!=type(value) or observed!=value:
            raise StudyControlError(f'container HostConfig.{key} differs')
    mounts=actual.get('Mounts')
    if not isinstance(mounts,list):raise StudyControlError('missing bind table')
    actual_rows=[]
    for mount in mounts:
        if (not isinstance(mount,dict) or mount.get('Type')!='bind'
            or type(mount.get('RW')) is not bool):raise StudyControlError('unsafe mount type/access')
        actual_rows.append((mount.get('Source'),mount.get('Destination'),not mount['RW']))
    if (sorted(actual_rows)!=sorted(spec.mounts)
        or len({r[1] for r in actual_rows})!=len(actual_rows)
        or sum(not r for _,_,r in actual_rows)!=1):raise StudyControlError('exact bind mount table differs')
    state=actual.get('State',{})
    if lifecycle=='prestart':
        valid=state.get('Running') is False and state.get('Status')=='created'
    elif lifecycle=='finished':
        valid=state.get('Running') is False and state.get('Status')=='exited'
    elif lifecycle=='cleanup':
        valid=(state.get('Status') in ('created','running','exited','dead')
               and state.get('Running') is (state.get('Status')=='running'))
    else:raise StudyControlError('unknown inspect lifecycle')
    if not valid:raise StudyControlError('container inspect lifecycle differs')
    return actual


def _guest_path(value) -> Path:
    """Private filesystem namespace boundary; production uses actual guest paths."""
    return Path(value)


def _guest_facts() -> dict:
    mountinfo=Path('/proc/self/mountinfo').read_text()
    runtime=cdi.collect_runtime_mount_facts(mountinfo)
    mounts=[]
    for line in mountinfo.splitlines():
        fields=line.split();mounts.append((fields[4].replace('\\040',' '),'ro' in fields[5].split(',')))
    regular=set()
    for destination,_ in mounts:
        try:
            if stat.S_ISREG(Path(destination).lstat().st_mode):regular.add(destination)
        except OSError:
            pass  # Missing/unreadable type evidence can never admit a driver file.
    return {**runtime,'mounts':mounts,'regular_mount_files':regular,'environment':dict(os.environ),'uid':os.getuid(),'gid':os.getgid(),
        'python':sys.executable,'monotonic':time.monotonic(),'device_nodes':[str(p) for p in Path('/dev').glob('nvidia*')]}


def _driver_mount_allowed(phase, destination, readonly, regular_files):
    """Only qualified compute/utility regular files; never a directory allowance."""
    if phase not in PHASES[:-1] or readonly is not True or destination not in regular_files:
        return False
    path=PurePosixPath(destination)
    if not path.is_absolute() or '..' in path.parts:return False
    utilities={'nvidia-smi','nvidia-debugdump','nvidia-persistenced',
               'nvidia-cuda-mps-control','nvidia-cuda-mps-server'}
    if path.parent.as_posix() in ('/usr/bin','/usr/local/nvidia/bin'):
        return path.name in utilities
    if path.parent.as_posix() not in ('/usr/lib','/usr/lib64','/usr/lib/x86_64-linux-gnu',
                                      '/usr/local/nvidia/lib','/usr/local/nvidia/lib64'):
        return False
    version='595.91.07'
    sonames={'libcuda':('1',version),'libnvidia-ml':('1',version),
        'libnvidia-ptxjitcompiler':('1',version),'libnvidia-nvvm':('4',version),
        'libnvidia-gpucomp':(version,),'libnvidia-compiler':(version,),
        'libnvidia-allocator':('1',version)}
    return any(path.name==family+'.so.'+suffix for family,versions in sonames.items() for suffix in versions)


def _verify_guest_runtime():
    native=_guest_path('/runtime/venv-cuda/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so')
    if sha256_file(native)!=NATIVE_SHA256 or sha256_file(_guest_path('/runtime/requirements-cuda.lock'))!=LOCK_SHA256:
        raise StudyControlError('guest qualified native/lock differs')


def _emit_guest_admission(record: Mapping) -> None:
    # stderr is already bounded/captured by the host transport. Never create an
    # application file before B's startup-empty check, or retry a partial write.
    encoded=ADMISSION_MARKER+json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode('utf-8'))>ADMISSION_LIMIT:
        raise StudyControlError('admission evidence exceeds 16KiB emission bound')
    if sys.stderr.write(encoded)!=len(encoded):
        raise StudyControlError('admission evidence short stderr write')
    sys.stderr.flush()


def verify_guest_context(*, root: Path, study_path: Path, phase_context: Path,
    phase: str, deadline: float, policy_id: str | None = None,
    operation_id: str | None = None, bindings_path: Path | None = None,
    receipt_path: Path | None = None, output_dir: Path | None = None) -> Mapping[str,object]:
    """Ten-keyword B admission with captured stderr evidence; B owns operation entries."""
    if root!=_guest_path('/learner') or phase_context!=_guest_path('/control/phase.json'):
        raise StudyControlError('guest root/context path mismatch')
    if Path(study_path) not in (STUDY,root/STUDY) or phase not in PHASES:
        raise StudyControlError('guest phase/study identity mismatch')
    c=_json(phase_context);deadline=_finite_time(deadline)
    phase_claim=_json(_guest_path('/control/phase-claim.json'))
    approval=_json(_guest_path('/control/approval.json'))
    snapshot=_json(_guest_path('/control/snapshot.json'))
    claim=_json(_guest_path('/control/campaign-claim.json'))
    if (c.get('schema')!=CONTEXT_SCHEMA or c.get('phase')!=phase or c.get('soft_deadline')!=deadline
        or c.get('output')!=_phase_output(phase).as_posix()
        or c.get('study_raw_sha256')!=STUDY_SHA256
        or c.get('approval_raw_sha256')!=sha256_file(_guest_path('/control/approval.json'))
        or c.get('phase_claim_sha256')!=sha256_file(_guest_path('/control/phase-claim.json'))
        or c.get('campaign_claim_sha256')!=sha256_file(_guest_path('/control/campaign-claim.json'))
        or c.get('source_snapshot_sha256')!=sha256_file(_guest_path('/control/snapshot.json'))
        or approval.get('schema')!=APPROVAL_SCHEMA or approval.get('state')!='approved'
        or approval.get('single_use') is not True or approval.get('phase_order')!=list(PHASES)
        or snapshot.get('state')!='reviewed' or claim.get('schema')!=CLAIM_SCHEMA
        or approval.get('source_snapshot_sha256')!=c['source_snapshot_sha256']
        or approval.get('study_raw_sha256')!=STUDY_SHA256
        or claim.get('source_snapshot_sha256')!=c['source_snapshot_sha256']
        or claim.get('study_raw_sha256')!=STUDY_SHA256
        or claim.get('phase_order')!=list(PHASES)
        or phase_claim.get('schema')!='spire-learn-task9-phase-claim-v1'
        or phase_claim.get('phase_index')!=PHASES.index(phase)
        or phase_claim.get('phase')!=phase or phase_claim.get('soft_deadline')!=deadline
        or phase_claim.get('whole_deadline')!=c.get('whole_deadline')
        or phase_claim.get('predecessors')!=c.get('predecessors')
        or phase_claim.get('campaign_claim_sha256')!=c['campaign_claim_sha256']
        or claim.get('approval_raw_sha256')!=c['approval_raw_sha256']
        or phase_claim.get('approval_raw_sha256')!=c['approval_raw_sha256']
        or any(x.get('campaign_id')!=c.get('campaign_id') for x in (approval,claim,phase_claim))):
        raise StudyControlError('guest source/approval/claim/context chain mismatch')
    study=_study(root)
    if (canonical_sha256(approval.get('runtime_pins'))!=canonical_sha256(study['runtime_pins'])
        or canonical_sha256(approval.get('budgets'))!=canonical_sha256(study['budgets'])):
        raise StudyControlError('guest runtime pins/budgets mismatch')
    inventory=_source_inventory(root,_source_files(root,GUEST_SCOPES))
    if inventory!=c.get('source_files') or any(snapshot['files'].get(k)!=v for k,v in inventory.items()):
        raise StudyControlError('guest released source bytes/modes differ')
    output=root/_phase_output(phase)
    if phase!='selection':
        if (policy_id!=phase or operation_id is not None or bindings_path is not None
            or output_dir is not None or receipt_path!=output/'operation-receipt.json'):
            raise StudyControlError('training CLI keyword/path mismatch')
    else:
        if policy_id is not None or bindings_path!=root/BINDINGS:
            raise StudyControlError('selection CLI policy/binding mismatch')
        if sha256_file(bindings_path)!=c.get('bindings_sha256') or _json(bindings_path)!=c.get('bindings'):
            raise StudyControlError('frozen selection bindings changed')
        if operation_id is None:
            if receipt_path is not None or output_dir!=output:raise StudyControlError('selection parent CLI mismatch')
        else:
            if (operation_id not in {r['operation_id'] for r in study['selection']['schedule']}
                or output_dir is not None
                or receipt_path!=output/'control/operations'/f'{operation_id}.json'):
                raise StudyControlError('selection child CLI mismatch')
        _verify_guest_checkpoints(root,c['bindings'])
    spec=_spec_from_context(root,phase_context)
    expected_env=_base_env(phase,c['output'],deadline,c['approval_raw_sha256'],c['phase_claim_sha256'])
    expected_env['SPIRE_LEARN_TASK9_CONTEXT_SHA256']=sha256_file(phase_context)
    if dict(spec.environment)!=expected_env or list(spec.command)!=_base_command(phase,deadline):
        raise StudyControlError('context argv/environment differs from phase protocol')
    expected_mounts={('/learner/'+p,True) for p in GUEST_SCOPES}
    expected_mounts.update({('/runtime/venv-cuda',True),('/runtime/requirements-cuda.lock',True),
        ('/control',True),('/learner/'+c['output'],False)})
    if phase=='selection':
        expected_mounts.add(('/learner/'+BINDINGS.as_posix(),True))
        expected_mounts.update(('/learner/'+p['checkpoint_directory'],True)
            for p in c['bindings']['policies'] if p['status']=='available')
    if (len(spec.mounts)!=len(expected_mounts)
        or {(d,r) for _,d,r in spec.mounts}!=expected_mounts):
        raise StudyControlError('context mount destinations/access differ from fixed phase contract')
    facts=_guest_facts();env=facts['environment']
    now=_finite_time(facts['monotonic']);whole=_finite_time(c['whole_deadline'])
    if now>=whole or whole>deadline+300.0:
        raise StudyControlError('guest whole phase deadline exhausted or invalid')
    for key,wanted in (('uid',1000),('gid',1000),('python',CONTAINER_PYTHON)):
        if facts.get(key)!=wanted:
            raise StudyControlError('guest '+key+' differs: '+repr(facts.get(key))[:256])
    profile=cdi.load_runtime_profile(root)
    expected_profile={'path':cdi.PROFILE_PATH,'raw_sha256':cdi.PROFILE_SHA256,'profile_id':cdi.PROFILE_ID}
    if c.get('runtime_profile')!=expected_profile or study['runtime_pins'].get('nvidia_cdi_profile')!=expected_profile:
        raise StudyControlError('guest runtime profile binding differs')
    runtime_profile=cdi.validate_guest_profile(profile,phase=phase,requested_environment=expected_env,
        actual_environment=env,facts=facts)
    admitted=set(runtime_profile['admitted_mount_destinations'])
    allowed_auto={'HOSTNAME','PWD','SHLVL','_','LC_ALL','NVIDIA_REQUIRE_CUDA'}
    mounts=dict(facts['mounts'])
    if mounts.get('/') is not True:
        raise StudyControlError('guest root filesystem must be read-only')
    application={d for _,d,_ in spec.mounts}
    automatic_files={'/etc/hosts','/etc/hostname','/etc/resolv.conf'}
    for dest,readonly in facts['mounts']:
        if (dest not in application and dest!='/' and dest not in automatic_files
            and not any(dest==p or dest.startswith(p+'/') for p in ('/proc','/dev','/sys'))
            and dest not in admitted
            and not (runtime_profile['mode']=='legacy' and _driver_mount_allowed(phase,dest,readonly,facts.get('regular_mount_files',set())))):
            raise StudyControlError('unexpected guest mount outside OCI/application allowlist: '+repr((dest,readonly))[:256])
    for _,dest,ro in spec.mounts:
        if mounts.get(dest) is not ro:raise StudyControlError('guest application mount flags differ: '+repr((dest,mounts.get(dest),ro))[:256])
    # Check every application mount, while retaining normal OCI /proc,/dev,/sys mounts.
    allowed={d for _,d,_ in spec.mounts}
    if any((d.startswith('/learner') or d.startswith('/control') or d.startswith('/runtime'))
           and d not in allowed for d,_ in facts['mounts']):
        raise StudyControlError('extra guest application mount')
    if phase=='selection' and facts['device_nodes']:raise StudyControlError('selection exposes CUDA devices')
    packages=_runtime_metadata(_guest_path('/runtime/venv-cuda/lib/python3.12/site-packages'))
    if packages.get('torch')!='2.14.1+cu130':raise StudyControlError('guest is not qualified CUDA-wheel stack')
    _verify_guest_runtime()
    if operation_id is None:
        _emit_guest_admission({'schema':ADMISSION_SCHEMA,'phase':phase,'campaign_id':c['campaign_id'],
            'context_raw_sha256':sha256_file(phase_context),'study_raw_sha256':c['study_raw_sha256'],
            'profile_binding':c['runtime_profile'],
            'process':{key:facts[key] for key in ('uid','gid','python','supplementary_group_ids')},
            'observed_environment':{key:env.get(key) for key in ('NVIDIA_VISIBLE_DEVICES','NVIDIA_CTK_LIBCUDA_DIR')},
            'runtime_profile':runtime_profile})
    return {'verified':True,'phase':phase,'context_raw_sha256':sha256_file(phase_context),
            'automatic_environment':{k:env[k] for k in allowed_auto if k in env},'runtime_profile':runtime_profile}


def _verify_guest_checkpoints(root, bindings):
    if bindings.get('stop_required_policy_ids'):raise StudyControlError('prior training fault')
    for policy in bindings['policies']:
        if policy['status']!='available':continue
        path=_safe(root,policy['checkpoint_directory'],exists=True)
        if path.name!='checkpoint-final' or set(p.name for p in path.iterdir())!=set(policy['inventory']):
            raise StudyControlError('checkpoint exact inventory differs')
        for name,item in policy['inventory'].items():
            file=_safe(root,path/name,exists=True)
            if not file.is_file() or file.stat().st_size!=item['size_bytes'] or sha256_file(file)!=item['sha256']:
                raise StudyControlError('checkpoint bytes changed')


def _promote(primary,error):
    control=lambda e:isinstance(e,(KeyboardInterrupt,SystemExit,_Termination)) or not isinstance(e,Exception)
    if primary is None:return error
    if control(error) and not control(primary):
        error.add_note(f'earlier operation failed: {type(primary).__name__}: {primary}');return error
    primary.add_note(f'later failure: {type(error).__name__}: {error}');return primary


def _error(error):
    return {'type':type(error).__name__,'message':str(error)[:2048],
            'notes':[str(x)[:2048] for x in getattr(error,'__notes__',())]}


def _command(deps, argv, *, deadline, logs, client, maximum=15.0, progress=None):
    remaining=deadline-_now(deps)
    if remaining<=0:raise StudyControlError('whole phase/campaign command deadline exhausted')
    value={'argv':list(argv),'status':'attempted','exit_code':None,'stdout':None,'stderr':None}
    logs.append(value)
    try:
        result=deps.run(argv,capture_output=True,text=True,timeout=min(maximum,remaining),check=False,
            env=_docker_env(client),output_limit_bytes=STREAM_LIMIT,progress_check=progress)
    except BaseException as error:
        value['status']='unreturned';value['error']=_error(error)
        raise
    value.update({'status':'returned','exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
           'timed_out':getattr(result,'timed_out',False),
           'stdout_truncated':getattr(result,'stdout_truncated',False),
           'stderr_truncated':getattr(result,'stderr_truncated',False)})
    if (value['timed_out'] or value['stdout_truncated'] or value['stderr_truncated']
        or len(str(result.stdout).encode())>STREAM_LIMIT or len(str(result.stderr).encode())>STREAM_LIMIT):
        raise StudyControlError('bounded command timeout/output limit; retained prefix is not success')
    if result.returncode!=0:raise StudyControlError(f'command failed exit {result.returncode}: {argv[0]}')
    return result.stdout


def _gpu_processes(deps,deadline,logs,client):
    raw=_command(deps,['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
        '--format=csv,noheader,nounits'],deadline=deadline,logs=logs,client=client,maximum=10)
    rows=[]
    for line in raw.splitlines():
        if not line.strip():continue
        values=[v.strip() for v in line.split(',')]
        if len(values)!=3 or not values[0].isdigit() or not values[1].isdigit():
            raise StudyControlError('malformed current GPU process memory inventory')
        rows.append({'pid':int(values[0]),'used_gpu_memory_mib':int(values[1]),'gpu_uuid':values[2]})
    return rows


def _campaign_claim_path(root: Path, campaign_id: str) -> Path:
    if not isinstance(campaign_id,str) or not _ID.fullmatch(campaign_id):
        raise StudyControlError('invalid campaign claim identity')
    return _safe(root,CLAIMS_DOMAIN/campaign_id/'campaign.claim.json')


def _fsync_claim_directory(root: Path, directory: Path) -> None:
    directory=_safe(root,directory,exists=True)
    if not directory.is_dir():raise StudyControlError('claim ancestor must be a directory')
    flags=os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(directory,flags)
    try:os.fsync(fd)
    finally:os.close(fd)


def _acquire_campaign_claim(root: Path, campaign_id: str, claim: Mapping) -> Path:
    """Persist the stable slot's entire path before returning claim authority."""
    root=Path(root).resolve(strict=True)
    path=_campaign_claim_path(root,campaign_id)
    directory=root
    entry_parents=[]
    for component in CLAIMS_DOMAIN.parts:
        child=_safe(root,directory/component)
        try:child.mkdir()
        except FileExistsError:
            if not child.is_dir() or child.is_symlink():
                raise StudyControlError('claim ancestor must be a regular directory')
        entry_parents.append(directory)
        directory=child
    path.parent.mkdir()  # Atomic reservation; failures below never delete it.
    # Always sync the chain, including existing entries: a concurrent creator
    # may not have persisted them. Reserve first so any sync failure leaves
    # this campaign's occupied slot consumed in the current filesystem.
    for parent in entry_parents:_fsync_claim_directory(root,parent)
    _fsync_claim_directory(root,directory)  # Persist the new per-ID entry.
    write_json_exclusive(path,claim)  # Existing file + immediate-parent fsync.
    return path


def _bounded_control_write(path: Path, value: Mapping) -> None:
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(raw)+1>CONTROL_RECORD_LIMIT:
        raise StudyControlError('control record exceeds its 8 MiB publication bound')
    write_json_exclusive(path,value)


def _admission(root,h,deps,*,reservation=0,claim_path=None,reserve_final=True):
    retained=root/'artifacts/development/neural-combat'
    outputs=[root/EXPERIMENT/'training'/p for p in PHASES[:-1]]+[root/SELECTION]
    claim_bytes=tree_bytes(claim_path.parent,strict=True) if claim_path is not None else 0
    used=tree_bytes(h,strict=True)+sum(tree_bytes(p,strict=True) for p in outputs)+claim_bytes
    reserved=reservation+(FINAL_RECORD_RESERVATION if reserve_final else 0)
    if used+reserved>6*GIB:raise StudyControlError('additional campaign 6 GiB storage reservation exhausted')
    if tree_bytes(retained,strict=False)+reserved>40*GIB:raise StudyControlError('retained 40 GiB ceiling exhausted')
    if deps.disk_usage(root).free-reserved<30*GIB:raise StudyControlError('30 GiB free admission floor exhausted')
    for path in outputs:
        limit=2*GIB if path==root/SELECTION else GIB
        if tree_bytes(path,strict=True)>limit:raise StudyControlError('phase output accounting ceiling exceeded')
    return {'additional_bytes':used,'reserved_next_phase_bytes':reservation,
            'reserved_final_record_bytes':FINAL_RECORD_RESERVATION if reserve_final else 0}


def _stage(root,h,files):
    stage=h/'source';stage.mkdir()
    guest=_source_inventory(root,_source_files(root,GUEST_SCOPES))
    for name in guest:
        src=root/name;dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);dst.chmod(0o555 if guest[name]['git_mode']=='100755' else 0o444)
    if _source_inventory(stage,_source_files(stage,GUEST_SCOPES))!=guest:
        raise StudyControlError('staged source inventory mismatch')
    return stage,guest


def _verify_sources(root,stage,snapshot,guest):
    if _source_inventory(root,_source_files(root,SNAPSHOT_PATHS))!=snapshot['files']:
        raise StudyControlError('source snapshot changed during campaign')
    if _source_inventory(stage,_source_files(stage,GUEST_SCOPES))!=guest:
        raise StudyControlError('staged guest source changed during campaign')


def _phase_inventory(root,h,phase):
    target=h/(phase+'-output-inventory.json')
    output=root/_phase_output(phase)
    value={'schema':'spire-learn-task9-phase-output-inventory-v1','phase':phase,
           'output':str(output),'entries':_inventory_entries([output],target) if output.exists() else []}
    _bounded_control_write(target,value)
    return sha256_file(target)


def _verify_predecessors(root,h,predecessors):
    for item in predecessors:
        phase=item['phase'];receipt=h/(phase+'-result.json')
        inventory=h/(phase+'-output-inventory.json')
        if sha256_file(receipt)!=item['receipt_sha256'] or sha256_file(inventory)!=item['inventory_sha256']:
            raise StudyControlError('predecessor receipt/inventory raw hash changed')
        frozen=_json(inventory)
        output=root/_phase_output(phase)
        if frozen.get('entries')!=_inventory_entries([output],inventory):
            raise StudyControlError('closed predecessor output bytes/modes changed')


def _verify_schedule(root,study,spec):
    schedule=_json(root/SELECTION/'control/schedule-result.json')
    rows=schedule.get('rows')
    bindings=_json(root/BINDINGS)
    if (schedule.get('schema_id')!='spire-learn-task9-selection-schedule-result-v1'
        or schedule.get('study_raw_sha256')!=STUDY_SHA256
        or schedule.get('binding_raw_sha256')!=sha256_file(root/BINDINGS)
        or schedule.get('shared_soft_deadline_monotonic')!=spec.soft_deadline
        or schedule.get('transport_exit_code')!=0
        or schedule.get('operational_failure_count')!=0 or schedule.get('unresolved_operation_count')!=0
        or not isinstance(rows,list) or len(rows)!=192):
        raise StudyControlError('selection schedule identity/status/count inventory differs')
    for expected,row in zip(study['selection']['schedule'],rows,strict=True):
        if any(row.get(k)!=v for k,v in expected.items()):
            raise StudyControlError('selection schedule reordered or substituted an identity')
        if row.get('status') not in ('complete','incomplete','not_started'):
            raise StudyControlError('selection schedule contains an unresolved/faulted operation')
        if row.get('receipt_path') is None:
            if row['status']!='not_started' or row.get('reserved_step_slots')!=0:
                raise StudyControlError('unreturned reserved operation is not a healthy result')
            continue
        receipt_path=root/SELECTION/'control/operations'/f"{expected['operation_id']}.json"
        if (row['receipt_path']!=receipt_path.relative_to(root).as_posix()
            or sha256_file(receipt_path)!=row.get('receipt_raw_sha256')):
            raise StudyControlError('selection receipt path/raw hash changed')
        receipt=_json(receipt_path)
        if (receipt.get('status')!=row['status'] or receipt.get('operation_id')!=expected['operation_id']
            or receipt.get('policy_id')!=expected['policy_id'] or receipt.get('case_id')!=expected['case_id']
            or receipt.get('study_raw_sha256')!=STUDY_SHA256
            or receipt.get('binding_raw_sha256')!=sha256_file(root/BINDINGS)):
            raise StudyControlError('selection receipt identity differs from scheduled input')
        if row['status']=='not_started':continue
        for key in ('summary','lifecycle'):
            path=_safe(root,receipt[key+'_path'],exists=True)
            if sha256_file(path)!=receipt[key+'_sha256']:
                raise StudyControlError('selection report/lifecycle raw hash changed')
        lifecycle=_json(root/receipt['lifecycle_path'])
        if lifecycle!=receipt.get('lifecycle') or lifecycle.get('close_errors')!=[]:
            raise StudyControlError('selection authoritative lifecycle differs')
        optimizer=receipt.get('optimizer',{})
        if any(type(optimizer.get(k)) is not int or optimizer[k]!=0
               for k in ('attempted_steps','accepted_steps','uncertain_steps','updates')):
            raise StudyControlError('selection receipt reports optimizer effects')
        learned=expected['policy_id'] in PHASES[:-1]
        if learned:
            _verify_threads(receipt.get('runtime_thread_settings'))
            frozen=next(p for p in bindings['policies'] if p['policy_id']==expected['policy_id'])
            cp=receipt.get('checkpoint',{})
            if (cp.get('training_receipt')!=frozen['training_receipt']
                or cp.get('before')!=frozen['inventory'] or cp.get('after')!=frozen['inventory']
                or cp.get('factory_checkpoint_path')!=frozen['checkpoint_directory']
                or cp.get('runtime_parent_checkpoint')!=frozen['checkpoint_directory']):
                raise StudyControlError('selection checkpoint receipt differs from frozen files')
        else:
            guard=receipt.get('reference_import_guard')
            if (not isinstance(guard,dict) or guard.get('installed') is not True
                or type(guard.get('process_id')) is not int or guard['process_id']<=0
                or guard.get('attempts')!=[] or guard.get('loaded_modules')!=[]
                or receipt.get('counts',{}).get('learned_decisions')!=0):
                raise StudyControlError('reference attempted/imported Torch or reported learned choices')
    return schedule


def _footprint(path: Path) -> bool:
    try:path.lstat();return True
    except FileNotFoundError:return False


def _verify_reserved_deadline_receipt(root,expected,r,returned,reservation_value,entry,deadline):
    """The one supported reserved/entered no-runner outcome, never a refund."""
    from scripts.neural.task9_study_runtime import _selection_count_projection
    from spire_learn.neural.cases import load_neural_cases
    deadline=_finite_time(deadline)
    binding_path=_safe(root,BINDINGS,exists=True)
    binding_hash=sha256_file(binding_path)
    identities=('index','operation_id','case_index','case_id','mode','cohort',
                'policy_id','source_policy_mode','classification')
    if canonical_sha256({k:r.get(k) for k in identities})!=canonical_sha256({k:expected[k] for k in identities}):
        raise StudyControlError('reserved deadline operation identity mismatch')
    config={'path':expected['config_path'],'canonical_sha256':expected['config_sha256'],
            'raw_sha256':expected['config_raw_sha256']}
    manifest={'path':expected['one_case_manifest_path'],
              'canonical_sha256':expected['one_case_manifest_sha256'],
              'raw_sha256':expected['one_case_manifest_raw_sha256'],
              'parent_selection_sha256':expected['parent_selection_sha256']}
    study=_json(root/STUDY)
    case,=load_neural_cases(_safe(root,expected['one_case_manifest_path'],exists=True)).for_mode(expected['mode'])
    seeds={k:getattr(case,k) for k in ('environment_seed','lineage_id','entry_id')}
    seeds['streams']=study['budgets']['selection']['eval_seeds']
    config_path=_safe(root,expected['config_path'],exists=True)
    if sha256_file(config_path)!=expected['config_raw_sha256']:
        raise StudyControlError('reserved deadline authored config bytes changed')
    authored_budgets=tomllib.loads(config_path.read_text(encoding='utf-8'))['budgets']
    budgets={'reserved_step_slots':256,'learned_decision_cap':authored_budgets['max_training_decisions'],
             'environment_action_cap':authored_budgets['max_environment_actions'],
             'optimizer_call_cap':authored_budgets['max_optimizer_steps'],'soft_deadline_monotonic':deadline}
    exact_reservation_identity={'schema_id':'spire-learn-task9-selection-reservation-v1',
        'operation_id':expected['operation_id'],'index':expected['index'],'case_id':expected['case_id'],
        'policy_id':expected['policy_id'],'mode':expected['mode'],'binding_raw_sha256':binding_hash,
        'reserved_step_slots':256,'deadline_monotonic':deadline,'state':'reserved_once'}
    if canonical_sha256({k:reservation_value.get(k) for k in exact_reservation_identity})!=canonical_sha256(exact_reservation_identity):
        raise StudyControlError('reserved deadline reservation identity differs from its operation')
    if (r.get('binding_raw_sha256')!=binding_hash
        or canonical_sha256(r.get('config'))!=canonical_sha256(config)
        or canonical_sha256(r.get('manifest'))!=canonical_sha256(manifest)
        or canonical_sha256(r.get('seed_evidence'))!=canonical_sha256(seeds)
        or canonical_sha256(r.get('budgets'))!=canonical_sha256(budgets)
        or r.get('environment_revision')!=study['runtime_pins']['gym_revision']
        or reservation_value.get('schema_id')!='spire-learn-task9-selection-reservation-v1'
        or reservation_value.get('binding_raw_sha256')!=binding_hash
        or reservation_value.get('deadline_monotonic')!=deadline
        or entry.get('schema_id')!='spire-learn-task9-selection-entry-v1'):
        raise StudyControlError('reserved deadline config/binding/reservation identity mismatch')
    reservation_path=(SELECTION/'control/operations'/f"{expected['operation_id']}.reservation.json").as_posix()
    exact_reservation={'path':reservation_path,'raw_sha256':returned['reservation_raw_sha256'],
                       'reserved_step_slots':256}
    if canonical_sha256(r.get('reservation'))!=canonical_sha256(exact_reservation):
        raise StudyControlError('reserved deadline receipt reservation linkage mismatch')
    reason='shared_soft_deadline_exhausted_before_factory'
    if (r.get('status')!='not_started' or r.get('reason')!=reason or returned.get('reason')!=reason
        or r.get('attempted_effects') is not False or returned.get('attempted_effects') is not False
        or type(r.get('transport_exit_code')) is not int or r['transport_exit_code']!=0
        or type(returned.get('transport_exit_code')) is not int or returned['transport_exit_code']!=0
        or type(returned.get('child_exit_code')) is not int or returned['child_exit_code']!=0
        or returned.get('dispatch_state')!='receipt_reconciled'
        or returned.get('runner_exit_code') is not None
        or r.get('close_status')!='runner_owned_or_unknown'
        or r.get('error') is not None or r.get('final_clock_error') is not None):
        raise StudyControlError('unsupported reserved not-started reason/effect/return shape')
    absent=('run_id','run_directory','runner_exit_code','report','lifecycle','summary_path',
            'summary_sha256','lifecycle_path','lifecycle_sha256','case_accounting','fight_summary',
            'optimizer','runtime_thread_settings','reference_import_guard')
    unknown=_selection_count_projection(None)
    if (any(k not in r or r[k] is not None for k in absent)
        or canonical_sha256(r.get('counts'))!=canonical_sha256(unknown)
        or canonical_sha256(returned.get('counts'))!=canonical_sha256(unknown)):
        raise StudyControlError('reserved deadline receipt fabricated runner artifacts/counters')
    checkpoint=r.get('checkpoint')
    if not isinstance(checkpoint,dict) or set(checkpoint)!={'training_receipt','before','after'}:
        raise StudyControlError('reserved deadline checkpoint shape mismatch')
    frozen=next((p for p in _json(binding_path)['policies'] if p['policy_id']==expected['policy_id']),None)
    if expected['policy_id'] in PHASES[:-1]:
        if frozen is None or frozen['status']!='available' or checkpoint['training_receipt']!=frozen['training_receipt']:
            raise StudyControlError('reserved deadline policy is not bound to its actual training receipt')
        if checkpoint['before'] is not None and checkpoint['before']!=frozen['inventory']:
            raise StudyControlError('reserved deadline preflight checkpoint inventory differs')
    elif checkpoint['training_receipt'] is not None or checkpoint['before'] is not None:
        raise StudyControlError('reference deadline receipt claims a checkpoint')
    if checkpoint['after'] is not None:
        raise StudyControlError('no-runner deadline receipt claims postflight effects')
    timing=r.get('outer_timing',{})
    start=_finite_time(timing.get('start_monotonic'));finish=_finite_time(timing.get('finish_monotonic'))
    entered_at=_finite_time(entry.get('entered_at_monotonic'))
    _finite_time(reservation_value.get('reserved_at_monotonic'))
    if finish<start or finish<deadline or entered_at<start or entered_at>finish:
        raise StudyControlError('reserved deadline outcome lacks observed deadline exhaustion')


def _audit_receipt(root,expected,item,reservation,entered,receipt,returned,*,deadline=None):
    """Resolution requires a matched returned schedule row and authoritative files."""
    if returned is None:raise StudyControlError('no trustworthy schedule return reconciliation')
    if any(returned.get(k)!=v for k,v in expected.items()):
        raise StudyControlError('returned schedule operation identity mismatch')
    r=_json(_safe(root,receipt,exists=True))
    digest=sha256_file(receipt)
    if (returned.get('receipt_raw_sha256')!=digest
        or returned.get('receipt_path')!=receipt.relative_to(root).as_posix()
        or r.get('schema_id')!='spire-learn-task9-selection-operation-v1'
        or r.get('operation_id')!=expected['operation_id'] or r.get('policy_id')!=expected['policy_id']
        or r.get('case_id')!=expected['case_id'] or r.get('mode')!=expected['mode']
        or r.get('study_raw_sha256')!=STUDY_SHA256
        or r.get('status')!=returned.get('status') or r.get('transport_exit_code')!=0):
        raise StudyControlError('operation receipt identity/hash/status is not reconciled')
    if (not item['reservation_present'] and not item['entered_present']
        and r['status']=='not_started' and returned.get('reserved_step_slots')==0
        and r.get('attempted_effects') is False and r.get('reservation') is None):
        item.update(reserved_step_slots_upper_bound=0,status='not_started',unresolved=False,
                    receipt_raw_sha256=digest,diagnostic=None)
        return
    if not item['reservation_present'] or not item['entered_present']:
        raise StudyControlError('reserved operation lacks reservation/entry evidence')
    reservation_value=_json(_safe(root,reservation,exists=True))
    entry=_json(_safe(root,entered,exists=True))
    res_hash=item['reservation_raw_sha256']
    if (reservation_value.get('operation_id')!=expected['operation_id']
        or reservation_value.get('policy_id')!=expected['policy_id']
        or reservation_value.get('case_id')!=expected['case_id']
        or reservation_value.get('reserved_step_slots')!=256
        or reservation_value.get('state')!='reserved_once'
        or entry.get('operation_id')!=expected['operation_id'] or entry.get('state')!='attempted_once'
        or entry.get('reservation_raw_sha256')!=res_hash
        or returned.get('reservation_raw_sha256')!=res_hash or returned.get('reserved_step_slots')!=256
        or r.get('reservation',{}).get('raw_sha256')!=res_hash):
        raise StudyControlError('reservation/entry/return identity mismatch')
    if r.get('status')=='not_started':
        _verify_reserved_deadline_receipt(root,expected,r,returned,reservation_value,entry,deadline)
        item.update(status='not_started',unresolved=False,receipt_raw_sha256=digest,diagnostic=None)
        return
    if r.get('status') not in ('complete','incomplete'):
        raise StudyControlError('unsupported reserved operation outcome')
    lifecycle=_json(_safe(root,r.get('lifecycle_path'),exists=True))
    summary_path=_safe(root,r.get('summary_path'),exists=True)
    summary=_json(summary_path)
    report=r.get('report')
    if (sha256_file(root/r['lifecycle_path'])!=r.get('lifecycle_sha256')
        or sha256_file(summary_path)!=r.get('summary_sha256')
        or lifecycle!=r.get('lifecycle') or lifecycle.get('status')!=r['status']
        or lifecycle.get('close_errors')!=[] or lifecycle.get('exit_code')!=r.get('runner_exit_code')
        or r.get('close_status')!='clean' or not isinstance(report,dict)
        or report.get('lifecycle')!=lifecycle
        or summary.get('summary_sha256')!=canonical_sha256(summary.get('summary'))):
        raise StudyControlError('operation authoritative lifecycle/summary evidence differs')
    item.update(status=r['status'],unresolved=False,receipt_raw_sha256=digest,diagnostic=None)


def _reservation_audit(root,study,*,reconciled_schedule=None):
    """Read failures are diagnostics, never proof that a reservation was refunded."""
    result=[]
    returned={r['operation_id']:r for r in reconciled_schedule['rows']} if reconciled_schedule else {}
    for row in study['selection']['schedule']:
        base=root/SELECTION/'control/operations'/row['operation_id']
        reservation=base.with_name(base.name+'.reservation.json')
        entered=base.with_name(base.name+'.entered.json')
        receipt=base.with_suffix('.json')
        item={'operation_id':row['operation_id'],'reserved_step_slots_upper_bound':256,
              'status':'unresolved','unresolved':True,'diagnostic':None}
        try:
            for key,path in (('reservation_present',reservation),('entered_present',entered),('receipt_present',receipt)):
                item[key]=_footprint(path)
            if not any(item[k] for k in ('reservation_present','entered_present','receipt_present')):continue
            if item['reservation_present']:
                _safe(root,reservation,exists=True)
                item['reservation_raw_sha256']=sha256_file(reservation)
            _audit_receipt(root,row,item,reservation,entered,receipt,returned.get(row['operation_id']),
                deadline=reconciled_schedule['shared_soft_deadline_monotonic'] if reconciled_schedule else None)
        except Exception as error:
            for key in ('reservation_present','entered_present','receipt_present'):item.setdefault(key,None)
            item['diagnostic']=(type(error).__name__+': '+str(error))[:2048]
        result.append(item)
    return result


def _unknown_reservation_audit(study,error):
    return [{'operation_id':r['operation_id'],'reserved_step_slots_upper_bound':256,
             'unresolved':True,'status':'unresolved','diagnostic':
             ('audit interrupted: '+type(error).__name__+': '+str(error))[:2048]}
            for r in study['selection']['schedule']]


def _close_inventory(h):
    # Both host records and actual outputs are closed; the inventory excludes
    # only itself. Persisted roots are exact, disjoint owned phase domains.
    root=next(p for p in h.parents if (p/STUDY).is_file())
    roots=[h]+[root/_phase_output(p) for p in PHASES if (root/_phase_output(p)).exists()]
    plan=_json(h/'campaign-plan.json')
    claim=_campaign_claim_path(root,plan['campaign_id'])
    if claim.parent.exists():roots.append(claim.parent)
    target=h/'closed-inventory.json'
    value={'schema':INVENTORY_SCHEMA,'roots':[str(p) for p in roots],
           'entries':_inventory_entries(roots,target)}
    _bounded_control_write(target,value)
    return value


def verify_closed_inventory(h):
    if (h/'closure-failure.json').exists():
        raise StudyControlError('closure failure invalidates the provisional closed inventory')
    frozen=_json(h/'closed-inventory.json')
    roots=[Path(p) for p in frozen['roots']]
    if (frozen.get('schema')!=INVENTORY_SCHEMA or not roots or roots[0]!=h
        or _inventory_entries(roots,h/'closed-inventory.json')!=frozen.get('entries')):
        raise StudyControlError('closed evidence inventory changed')
    return frozen


def run_campaign(root: Path, *, snapshot_path: Path, approval_path: Path,
                 campaign_id: str, control_root: Path,
                 dependencies: HostDependencies | None = None) -> dict:
    """One approved five-phase invocation. No retry/resume/replacement path exists."""
    deps=dependencies or HostDependencies();started=_now(deps)
    root=Path(root).resolve(strict=True);h=_safe(root,control_root)
    retained=root/'artifacts/development/neural-combat'
    if not h.is_relative_to(retained) or h.is_relative_to(root/EXPERIMENT):
        raise StudyControlError('host control records must be outside writable experiment outputs')
    if h.exists():raise FileExistsError('campaign control directory already exists')
    binding=validate_approval(root,snapshot_path,approval_path)
    if campaign_id!=binding['approval']['campaign_id']:raise StudyControlError('wrong approved campaign ID')
    if deps.uid()!=1000 or deps.gid()!=1000:raise StudyControlError('protected launcher requires host 1000:1000')
    snapshot=binding['snapshot'];study=binding['study'];approval_hash=binding['approval_raw_sha256']
    # All authored output slots must be fresh; no unknown prior data is removed.
    for phase in PHASES:
        output=_safe(root,_phase_output(phase))
        if output.exists():raise FileExistsError('authored phase output already exists')
    claim_path=_campaign_claim_path(root,campaign_id)
    if h.is_relative_to(root/CLAIMS_DOMAIN) or (root/CLAIMS_DOMAIN).is_relative_to(h):
        raise StudyControlError('evidence must be separate from the fixed campaign claim domain')
    h.mkdir(parents=True)
    client=h/'client-empty'
    logs=[];whole=started+4500.0;work_deadline=whole-30.0
    rows=[{'phase':p,'status':'not_started','reason':'prior_phase_not_completed',
           'cleanup':'not_owned','container_id':None,'attempted_effects':False} for p in PHASES]
    primary=None;receipts={};runtime=None;campaign=None;stage=None;guest=None
    claim_state='unclaimed';reconciled_schedule=None
    try:
        _bounded_control_write(h/'campaign-plan.json',{'schema':'spire-learn-task9-campaign-plan-v1',
            'campaign_id':campaign_id,'approval_raw_sha256':approval_hash,
            'claim_slot':claim_path.relative_to(root).as_posix(),
            'phases':rows,'selection_rows':study['selection']['schedule']})
        client.mkdir()
        if claim_path.parent.exists():
            raise FileExistsError('approved campaign identity was already claimed or reserved')
        docker=_docker_binary()
        revision=_command(deps,['git','--no-optional-locks','-C',str(root),'rev-parse','HEAD'],
                          deadline=work_deadline,logs=logs,client=client).strip()
        dirty=_command(deps,['git','--no-optional-locks','-C',str(root),'status','--porcelain=v1','--untracked-files=all'],
                       deadline=work_deadline,logs=logs,client=client)
        if revision!=snapshot['learner_head'] or {'porcelain':dirty}!=snapshot['dirty'] and not (dirty=='' and snapshot['dirty']=={}):
            raise StudyControlError('current learner HEAD/dirty state differs from reviewed snapshot')
        runtime=_verify_runtime(root,binding['approval'])
        raw_image=_command(deps,_docker(docker,client,'image','inspect',IMAGE),deadline=work_deadline,logs=logs,client=client)
        image=verify_image_inspect(raw_image)
        if image['image_id']!=IMAGE_ID:raise StudyControlError('immutable local image ID differs')
        _admission(root,h,deps,reservation=GIB,claim_path=claim_path)
        stage,guest=_stage(root,h,snapshot['files'])
        for phase in PHASES:
            name='spire-learn-task9-'+campaign_id+'-'+phase
            raw=_command(deps,_docker(docker,client,'ps','-a','--no-trunc','--filter','name=^/'+name+'$','--format','{{.ID}}'),
                         deadline=work_deadline,logs=logs,client=client)
            if raw.strip():raise StudyControlError('exact container name already exists')
        _verify_sources(root,stage,snapshot,guest)
        claim={'schema':CLAIM_SCHEMA,'campaign_id':campaign_id,'approval_raw_sha256':approval_hash,
            'source_snapshot_sha256':sha256_file(snapshot_path),'study_raw_sha256':STUDY_SHA256,
            'phase_order':list(PHASES),'start_monotonic':started,'whole_deadline':whole,
            'claim_slot':claim_path.relative_to(root).as_posix(),'evidence_root':h.relative_to(root).as_posix()}
        claim_state='publication_attempted'
        _acquire_campaign_claim(root,campaign_id,claim)
        claim_state='owned'
        write_json_exclusive(h/'campaign.claim.json',claim)
        digest=sha256_file(claim_path)
        if sha256_file(h/'campaign.claim.json')!=digest:raise StudyControlError('campaign claim evidence copy differs')
        campaign={**claim,'claim_sha256':digest,'claim_path':str(claim_path),
            'approval_path':str(approval_path),'snapshot_path':str(snapshot_path),'guest_source_files':guest,
            'predecessors':[]}
    except BaseException as error:
        primary=_promote(primary,error)
    for index,phase in enumerate(PHASES):
        if primary is not None:break
        try:phase_start=_now(deps)
        except BaseException as error:
            primary=_promote(primary,error);break
        phase_whole=min(phase_start+900,whole)
        active=min(phase_whole-30,work_deadline);cid=None;spec=None;phase_error=None
        row={'phase':phase,'status':'unresolved','start_monotonic':phase_start,
            'soft_deadline':phase_start+600,'whole_deadline':phase_whole,'container_id':None,
            'cleanup':'not_owned','attempted_effects':None,'approval_raw_sha256':approval_hash,
            'campaign_claim_sha256':campaign['claim_sha256']}
        phase_logs=[]
        try:
            if phase_start>=active:raise StudyControlError('phase setup exhausted whole deadline')
            _admission(root,h,deps,reservation=2*GIB if phase=='selection' else GIB,claim_path=claim_path)
            _verify_sources(root,stage,snapshot,guest)
            _verify_predecessors(root,h,campaign['predecessors'])
            if sha256_file(approval_path)!=approval_hash:raise StudyControlError('approval changed after claim')
            if _verify_runtime(root,binding['approval'])!=runtime:raise StudyControlError('runtime input changed')
            if phase!='selection':
                raw=_command(deps,['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.free','--format=csv,noheader,nounits'],
                    deadline=active,logs=phase_logs,client=client,maximum=10)
                row['gpu_inventory']=verify_gpu_inventory(raw)
                row['gpu_process_inventory_before']=_gpu_processes(deps,active,phase_logs,client)
            output=root/_phase_output(phase);output.mkdir(parents=True)
            bindings_path=None
            if phase=='selection':
                for name in SCRATCH.values():(output/'.runtime'/name).mkdir(parents=True,exist_ok=True)
                from scripts.neural.task9_study_runtime import freeze_checkpoint_bindings
                bindings_path=root/BINDINGS
                freeze_checkpoint_bindings(root,STUDY,receipts,bindings_path)
                verify_training_bindings(root,STUDY,bindings_path,receipts)
            name='spire-learn-task9-'+campaign_id+'-'+phase
            raw=_command(deps,_docker(docker,client,'ps','-a','--no-trunc','--filter','name=^/'+name+'$','--format','{{.ID}}'),
                         deadline=active,logs=phase_logs,client=client)
            if raw.strip():raise StudyControlError('exact container name collision')
            spec=build_phase_spec(root,campaign=campaign,phase=phase,phase_start=phase_start,
                source_tree=stage,context_path=h/phase/'phase.json',bindings_path=bindings_path)
            row['context_raw_sha256']=spec.context_sha256
            row['phase_claim_sha256']=spec.phase_claim_sha256
            # No exception between verified create output and saving ownership locally.
            raw=_command(deps,build_create_argv(spec,docker,client),deadline=active,logs=phase_logs,client=client)
            candidate=raw.strip()
            if not _HASH.fullmatch(candidate):raise StudyControlError('create did not return one full64hex ID')
            cid=candidate;row['container_id']=cid;row['cleanup']='owned_pending'
            raw=_command(deps,_docker(docker,client,'inspect',cid),deadline=active,logs=phase_logs,client=client)
            assert_phase_inspect(raw,spec=spec,container_id=cid)
            def progress():
                if _now(deps)>=active:raise StudyControlError('phase work deadline exhausted')
                _admission(root,h,deps,claim_path=claim_path)
            _command(deps,_docker(docker,client,'start','--attach',cid),deadline=active,
                     logs=phase_logs,client=client,maximum=max(0,active-_now(deps)),progress=progress)
            raw=_command(deps,_docker(docker,client,'inspect',cid),deadline=active,logs=phase_logs,client=client)
            inspected=assert_phase_inspect(raw,spec=spec,container_id=cid,lifecycle='finished')
            state=inspected.get('State',{})
            if (state.get('Running') is not False or state.get('ExitCode')!=0
                or state.get('OOMKilled') is not False or state.get('Error')):
                raise StudyControlError('container did not finish healthy')
            if phase=='selection':
                verify_training_bindings(root,STUDY,root/BINDINGS,receipts)
                schedule=_verify_schedule(root,study,spec)
                reconciled_schedule=schedule
                audit=_reservation_audit(root,study,reconciled_schedule=schedule)
                if any(r['unresolved'] for r in audit) or sum(r['reserved_step_slots_upper_bound'] for r in audit)!=schedule.get('reserved_step_slots'):
                    raise StudyControlError('selection reservation footprints disagree or remain unresolved')
                row['schedule_result_sha256']=sha256_file(output/'control/schedule-result.json')
            else:
                record=study['training']['runs'][index]
                verified=_verify_training_receipt(root,record)
                if verified['receipt']['budgets']['shared_deadline_monotonic']!=spec.soft_deadline:
                    raise StudyControlError('training receipt uses a different phase deadline')
                receipts[phase]=root/record['output_root']/'operation-receipt.json'
                row['training_receipt']=verified['path'];row['training_receipt_raw_sha256']=verified['raw_sha256']
                row['run_id']=verified['run_id'];row['attempted_effects']=verified['receipt']['attempted_effects']
                row['gpu_process_inventory_after']=_gpu_processes(deps,active,phase_logs,client)
            row['status']='complete'
        except BaseException as error:
            phase_error=error;row['status']='operational_failure';row['error']=_error(error)
        finally:
            if cid is not None:
                try:
                    raw=_command(deps,_docker(docker,client,'inspect',cid),deadline=phase_whole,logs=phase_logs,client=client)
                    _assert_owner(_inspect_value(raw),spec,cid)
                    try:assert_phase_inspect(raw,spec=spec,container_id=cid,lifecycle='cleanup')
                    except BaseException as error:
                        # Retain full-spec faults while still removing this exact
                        # successfully-created, re-owned ID; never a guessed name.
                        phase_error=_promote(phase_error,error)
                    write_json_exclusive(h/(phase+'-remove.claim.json'),{'container_id':cid,
                        'phase':phase,'approval_raw_sha256':approval_hash,'context_raw_sha256':spec.context_sha256})
                    _command(deps,_docker(docker,client,'rm','--force',cid),deadline=phase_whole,logs=phase_logs,client=client)
                    row['cleanup']='removed_owned_id'
                except BaseException as error:
                    phase_error=_promote(phase_error,error);row['cleanup']='failed_or_unknown'
            else:
                try:
                    _command(deps,_docker(docker,client,'ps','-a','--no-trunc','--filter',
                        'name=^/spire-learn-task9-'+campaign_id+'-'+phase+'$','--format','{{.ID}}'),
                        deadline=phase_whole,logs=phase_logs,client=client)
                except BaseException as error:phase_error=_promote(phase_error,error)
            try:
                finished=_now(deps)
                if finished<phase_start or finished>phase_whole:raise StudyControlError('final phase clock outside whole budget')
                row['finish_monotonic']=finished
                _verify_sources(root,stage,snapshot,guest)
                _verify_predecessors(root,h,campaign['predecessors'])
                if sha256_file(approval_path)!=approval_hash:raise StudyControlError('approval changed during phase')
                if _verify_runtime(root,binding['approval'])!=runtime:raise StudyControlError('runtime inputs changed during phase')
                _admission(root,h,deps,claim_path=claim_path)
            except BaseException as error:phase_error=_promote(phase_error,error)
            if phase_error is not None:
                row['status']='operational_failure';row['error']=_error(phase_error)
                primary=_promote(primary,phase_error)
            rows[index]=row
            try:
                inventory_hash=_phase_inventory(root,h,phase)
                write_json_exclusive(h/(phase+'-commands.json'),{'commands':phase_logs})
                row['finish_monotonic']=_now(deps)
                row['timing_boundary']='before host receipt publication; whole deadline rechecked afterward'
                if row['finish_monotonic']<phase_start or row['finish_monotonic']>phase_whole:
                    raise StudyControlError('phase finalization exhausted whole budget')
                write_json_exclusive(h/(phase+'-result.json'),row)
                if _now(deps)>phase_whole:raise StudyControlError('phase receipt publication exhausted whole budget')
                campaign['predecessors'].append({'phase':phase,'receipt_sha256':sha256_file(h/(phase+'-result.json')),
                                                'inventory_sha256':inventory_hash})
            except BaseException as error:
                primary=_promote(primary,error);row['status']='operational_failure';row['error']=_error(primary)
    try:
        if runtime is not None and _verify_runtime(root,binding['approval'])!=runtime:
            raise StudyControlError('runtime inputs changed before closure')
    except BaseException as error:primary=_promote(primary,error)
    try:
        audit=_reservation_audit(root,study,reconciled_schedule=reconciled_schedule)
    except BaseException as error:
        primary=_promote(primary,error);audit=_unknown_reservation_audit(study,error)
    if any(item['unresolved'] for item in audit):
        primary=_promote(primary,StudyControlError('selection reservation audit retains unresolved effects'))
    audited={item['operation_id']:item for item in audit}
    result={'schema':'spire-learn-task9-campaign-result-v1','campaign_id':campaign_id,
        'approval_raw_sha256':approval_hash,'runtime_inputs':runtime,
        'campaign_claim_state':claim_state,'campaign_claim_path':claim_path.relative_to(root).as_posix(),
        'campaign_claim_raw_sha256':campaign['claim_sha256'] if campaign else None,
        'status':'complete' if primary is None else 'stopped',
        'closure_state':'provisional_until_closed_inventory_and_successful_return',
        'phases':rows,'selection_reservation_audit':audit,
        'selection_rows':[{'operation_id':r['operation_id'],'policy_id':r['policy_id'],
            'case_id':r['case_id'],'status':audited[r['operation_id']]['status']
                if r['operation_id'] in audited else 'not_started'} for r in study['selection']['schedule']]}
    try:
        finish=_now(deps)
        if finish<started or finish>whole:raise StudyControlError('final campaign clock outside whole budget')
        result['finish_monotonic']=finish
    except BaseException as error:primary=_promote(primary,error)
    if primary:result['status']='stopped';result['error']=_error(primary)
    try:
        # Reserve bounded record/log/inventory space, then validate actual
        # bytes and elapsed work after every final publication and hashing.
        try:_admission(root,h,deps,claim_path=claim_path)
        except BaseException as error:
            # Failure evidence is still attempted within the bounded emergency
            # record budget; this cannot restore success or admit a workload.
            primary=_promote(primary,error)
            result['status']='stopped';result['error']=_error(primary)
        _bounded_control_write(h/'preflight-commands.json',{'commands':logs})
        _bounded_control_write(h/'campaign-result.json',result)
        _close_inventory(h)
        verify_closed_inventory(h)
        if runtime is not None and _verify_runtime(root,binding['approval'])!=runtime:
            raise StudyControlError('runtime inputs changed during final closure')
        accounting=_admission(root,h,deps,claim_path=claim_path,reserve_final=False)
        finished=_now(deps)
        if finished<started or finished>whole:
            raise StudyControlError('final publication/closed-inventory clock outside whole budget')
        result['closure_state']='verified'
        result['closed_at_monotonic']=finished
        result['final_accounting']=accounting
    except BaseException as error:
        primary=_promote(primary,error)
        # This new terminal failure invalidates any earlier complete result or
        # inventory. Never overwrite those historical provisional bytes.
        try:_bounded_control_write(h/'closure-failure.json',{'status':'stopped','error':_error(primary),
            'invalidates_provisional_inventory':True})
        except BaseException as later:primary=_promote(primary,later)
        raise primary
    if primary is not None and (not isinstance(primary,Exception) or not (h/'campaign-result.json').exists()):raise primary
    return result


def main(argv: Sequence[str] | None=None, *, dependencies: HostDependencies | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('describe')
    for name in ('verify','run','snapshot'):
        p=commands.add_parser(name);p.add_argument('--root',type=Path,required=True)
        p.add_argument('--snapshot',type=Path,required=True)
        if name!='snapshot':p.add_argument('--approval',type=Path,required=True)
        if name=='run':
            p.add_argument('--campaign-id',required=True);p.add_argument('--control-root',type=Path,required=True)
        if name=='snapshot':
            p.add_argument('--learner-head',required=True);p.add_argument('--dirty-json',type=Path,required=True)
            p.add_argument('--state',choices=('source_only','reviewed'),default='source_only')
    args=parser.parse_args(argv)
    if args.command=='describe':
        print(json.dumps({'state':'prepared_not_released','phases':list(PHASES),
            'verifier_keywords':['root','study_path','phase_context','phase','deadline','policy_id',
                'operation_id','bindings_path','receipt_path','output_dir']}));return 0
    try:
        if args.command=='verify':value=validate_approval(args.root,args.snapshot,args.approval)
        elif args.command=='snapshot':
            value=capture_source_snapshot(args.root,args.snapshot,learner_head=args.learner_head,
                dirty=_json(args.dirty_json),state=args.state)
        else:
            if os.environ.get('SPIRE_LEARN_TASK9_WATCHDOG')!='4470+30':
                raise StudyControlError('live CLI requires the reviewed launcher watchdog')
            old={}
            def terminate(signum,frame):raise _Termination(f'outer watchdog signal {signum}')
            try:
                for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,terminate)
                value=run_campaign(args.root,snapshot_path=args.snapshot,approval_path=args.approval,
                    campaign_id=args.campaign_id,control_root=args.control_root,dependencies=dependencies)
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
        print(json.dumps(value,allow_nan=False,sort_keys=True))
        return 0 if value.get('status') in (None,'complete') else 1
    except Exception as error:
        print(f'Task 9 controls refused: {type(error).__name__}: {error}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
