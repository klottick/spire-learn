"""Create the shared, untrained current-schema parent in one owned CPU job."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo/'src'), str(_repo)]

from spire_learn.neural.gym_contracts import CURRENT_CONTRACT
from spire_learn.neural.runner import _close_preserving

DOMAIN = Path('artifacts/development/neural-combat/curriculum-campaign-v4')
CONFIG = Path('configs/neural/experiments/curriculum-v4/bootstrap-corrected.toml')
BOOTSTRAP = DOMAIN/'bootstrap-corrected-1'


def verify_initial_fixture_failure(root):
    """Admit only the explicitly reviewed zero-gameplay setup correction."""
    campaign = Path(root)/DOMAIN
    proof = json.loads((campaign/'bootstrap-initial-failure.json').read_text())
    process_path = campaign/'bootstrap-process.json'
    process = json.loads(process_path.read_text())
    if (proof.get('status') != 'failed_probe_fixture' or proof.get('tool_os_exit') != 1
            or proof.get('process_sha256') != hashlib.sha256(process_path.read_bytes()).hexdigest()
            or any(proof.get(k) != 0 for k in ('environment_actions','resets','samples','optimizer_steps'))
            or proof.get('gpu_work') is not False or proof.get('checkpoint_published') is not False
            or process.get('source_postcheck') is not True or process.get('protected_postcheck') is not True
            or (campaign/'bootstrap-output/initial-policy').exists()):
        raise ValueError('original CPU setup failure does not authorize this correction')
    commands = process['commands']
    starts = [row for row in commands if len(row['argv'])>5 and row['argv'][5]=='start']
    removes = [row for row in commands if len(row['argv'])>5 and row['argv'][5]=='rm']
    if (len(starts)!=1 or 'ValueError: fresh model forward output invalid' not in starts[0]['stderr']
            or len(removes)!=1 or commands[-1]['stdout'].strip()
            or commands[-1]['exit_code']!=0 or removes[0]['exit_code']!=0
            or 'id='+removes[0]['argv'][-1] not in commands[-1]['argv']):
        raise ValueError('original CPU setup failure/cleanup evidence differs')
    return proof


def initial_probe_frame(schema):
    from tests.neural._support import CandidateInput, authored_frame
    return authored_frame(schema,candidates=(
        CandidateInput(2,'play_card',source_group='combat_cards',source_row=0,
                       target_group='enemies',target_row=0),
        CandidateInput(schema.action_space.n-1,'end_turn'),
    ))


def bootstrap_window(value, now):
    """The 80s transport total includes its 40s cleanup reserve, from parent start."""
    try:
        start = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('bootstrap parent clock missing') from error
    if not math.isfinite(start) or start < 0 or start > now:
        raise ValueError('bootstrap parent clock invalid')
    deadline = start + 80.
    if now >= deadline - 40.:
        raise TimeoutError('bootstrap preflight exhausted active budget; cleanup reserve untouched')
    return start, deadline


def _zero_progress(counters):
    if not isinstance(counters, Mapping) or not counters:
        raise ValueError('fresh parent counters missing')
    for name, value in counters.items():
        if name == 'elapsed_seconds':
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('fresh parent elapsed time invalid')
        elif type(value) is not int or value != 0:
            raise ValueError('fresh parent has nonzero training progress: '+name)


def initialize_parent(config, path, *, create_runtime, load_checkpoint, equal,
                      check_model, inventory):
    """Save fresh weights once; injected boundaries let pure tests check ownership."""
    if (config.device != 'cpu' or config.environment_revision != CURRENT_CONTRACT.source_revision
            or config.operation != 'train' or config.split != 'train' or config.mode != 'standalone'):
        raise ValueError('fresh bootstrap requires current release CPU training config')
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('fresh parent output already exists')
    runtime = None
    primary = None
    result = None
    try:
        runtime = create_runtime(config, purpose='train')
        if (runtime.recovered_progress != {} or runtime.parent_checkpoint is not None
                or runtime.policy._behavior_policy_version != 0
                or runtime.optimizer.state_dict()['state'] != {}):
            raise ValueError('fresh initialization inherited policy/optimizer/progress')
        recovery = runtime.collector.recovery_state
        if (recovery.next_seed_ordinal != 0 or recovery.next_decision_sequence != 0
                or any(recovery.next_episode_ordinals)):
            raise ValueError('fresh initialization inherited collector progress')
        _zero_progress(runtime._latest_counters())
        before = copy.deepcopy(dict(runtime.policy.state_dict()))
        model_checks = check_model(runtime)
        if not equal(before, dict(runtime.policy.state_dict())):
            raise ValueError('fresh model checks changed weights')
        _zero_progress(runtime._latest_counters())
        if runtime.optimizer.state_dict()['state'] != {}:
            raise ValueError('fresh model checks changed optimizer')
        runtime.save(path, 'final')
        loaded = load_checkpoint(path, config.accepted_identities[0], purpose='resume', device='cpu')
        if not equal(before, dict(loaded.model_tensors)):
            raise ValueError('saved fresh weights differ from initialization')
        state = loaded.metadata['training_state']
        if state['optimizer_step'] != 0 or state['behavior_policy_version'] != 0:
            raise ValueError('fresh checkpoint contains training state')
        _zero_progress(state['counters'])
        result = dict(status='fresh_untrained_parent', training_exposure=0,
            environment_revision=config.environment_revision, config_sha256=config.config_sha256,
            common_fingerprint=config.common_view_fingerprint,
            checkpoint=str(path), inventory=inventory(path), model_checks=model_checks)
    except BaseException as error:
        primary = error
    finally:
        if runtime is not None:
            primary = _close_preserving(primary, runtime, 'fresh bootstrap')
    if primary is not None:
        raise primary
    return result


def _check_cpu_model(runtime, *, torch, encode_frames, authored_frame):
    frame = authored_frame(runtime.collector.driver.schema)
    batch = encode_frames((frame,), runtime.collector.driver.schema, device='cpu')
    output = runtime.policy(batch)
    legal = output.logits.masked_select(batch.action_mask)
    if legal.numel() < 2 or not torch.isfinite(legal).all().item() or not torch.isfinite(output.values).all().item():
        raise ValueError('fresh model forward output invalid')
    (legal.sum()+output.values.sum()).backward()
    gradients = [p.grad for p in runtime.policy.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise ValueError('fresh model backward gradient invalid')
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise ValueError('fresh model backward has zero gradients')
    runtime.policy.zero_grad(set_to_none=True)
    return dict(forward_backward=True, device='cpu', parameter_count=sum(p.numel() for p in runtime.policy.parameters()),
        frame_bytes=sum(len(buffer.data) for buffer in frame.tensors.values()), actor_samples=0, optimizer_steps=0)

def _guest():
    context = os.environ.get('FRESH_BOOTSTRAP_CONTEXT', '')
    if (os.getuid() != 1000 or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'none'
            or len(context) != 64 or any(c not in '0123456789abcdef' for c in context)):
        raise ValueError('bootstrap guest execution controls differ')
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    from spire_lab import _native
    from spire_lab.gymnasium import IroncladCombatVectorEnv, RunEnv
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.runner import create_runtime
    from spire_learn.neural.checkpoint import load_checkpoint
    from spire_learn.neural.features import encode_frames
    from scripts.neural.sustained_runtime import checkpoint_inventory

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    runtime_profile = json.loads(Path('/learner/configs/neural/gym-20261102/runtime.json').read_text())
    native_path = Path(_native.__file__).resolve()
    if (not native_path.is_relative_to('/runtime/gym-overlay/spire_lab')
            or hashlib.sha256(native_path.read_bytes()).hexdigest() != runtime_profile['native_sha256']
            or _native.build_info()['revision'] != CURRENT_CONTRACT.source_revision):
        raise ValueError('bootstrap imported native identity differs')
    denied = []
    def no_effect(*args, **kwargs):
        denied.append('forbidden effect')
        raise AssertionError('bootstrap cannot reset, step, sample or optimize')
    def equal(left, right):
        return left.keys() == right.keys() and all(torch.equal(left[k].cpu(), right[k].cpu()) for k in left)
    def check_model(runtime):
        return _check_cpu_model(runtime, torch=torch, encode_frames=encode_frames, authored_frame=initial_probe_frame)
    config = load_neural_config(Path('/learner')/CONFIG)
    with ExitStack() as stack:
        for cls in (IroncladCombatVectorEnv, RunEnv):
            for name in ('reset', 'step'):
                stack.enter_context(patch.object(cls, name, side_effect=no_effect))
        stack.enter_context(patch.object(torch, 'multinomial', side_effect=no_effect))
        stack.enter_context(patch.object(torch.optim.Adam, 'step', new=no_effect))
        result = initialize_parent(config, Path('/output/initial-policy'), create_runtime=create_runtime,
            load_checkpoint=load_checkpoint, equal=equal, check_model=check_model, inventory=checkpoint_inventory)
    if denied or torch.cuda.is_initialized():
        raise ValueError('bootstrap performed forbidden effects or initialized CUDA')
    result.update(reset_calls=0, step_calls=0, samples=0, optimizer_steps=0,
                  cuda_initialized=False, uid=os.getuid(), native_file=str(native_path))
    with Path('/output/bootstrap.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,sort_keys=True))
    print('SPIRE_LEARN_SUSTAINED_ADMISSION_V1 '+json.dumps({
        'context_sha256':os.environ['FRESH_BOOTSTRAP_CONTEXT'], 'job_id':'fresh-initial-policy-corrected-1', 'runtime_class':'cpu'}),file=sys.stderr)


def _host():
    from scripts.neural import task9_study_controls as low, sustained_controls as transport
    from scripts.neural.gpu_runtime_proof_helpers import HostDependencies
    from scripts.neural.task8_gpu_setup_helpers import verify_image_inspect
    from scripts.neural.qualify_gym_release import source_inventory
    deps=HostDependencies()
    start,deadline=bootstrap_window(os.environ.get('FRESH_BOOTSTRAP_PARENT_START'),deps.clock())
    root=Path.cwd(); campaign=root/DOMAIN; domain=root/BOOTSTRAP; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    verify_initial_fixture_failure(root)
    release=json.loads((domain/'bootstrap-release.json').read_text())
    source=source_inventory(); digest=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (release['state']!='reviewed' or release['source_digest']!=digest
            or release['script_sha256']!=sha(Path(__file__))
            or release['launcher_sha256']!=sha(root/'scripts/neural/run-fresh-curriculum-bootstrap.sh')):
        raise ValueError('fresh bootstrap reviewed source binding differs')
    runtime=json.loads((root/'configs/neural/gym-20261102/runtime.json').read_text())
    if sha(root/'configs/neural/gym-20261102/runtime.json')!=release['runtime_profile_sha256']:
        raise ValueError('bootstrap runtime profile differs')
    from scripts.neural.curriculum_profiles import verify_overlay
    manifest=json.loads((root/'configs/neural/experiments/curriculum-v4/overlay.json').read_text())
    verify_overlay(root/runtime['package_overlay'],manifest['inventory'])
    protected=json.loads((campaign/'preparation/protected-files.json').read_text())
    if any(sha(root/name)!=expected for name,expected in protected.items()):
        raise ValueError('bootstrap protected files changed')
    if sha(root/low.LOCK)!=low.LOCK_SHA256 or sha(root/low.NATIVE)!=low.NATIVE_SHA256:
        raise ValueError('bootstrap qualified dependencies differ')
    client=domain/'bootstrap-client';output=domain/'bootstrap-output'
    if client.exists() or output.exists() or (domain/'bootstrap-started.json').exists():
        raise ValueError('bootstrap already consumed')
    bootstrap_window(os.environ.get('FRESH_BOOTSTRAP_PARENT_START'),deps.clock())
    environment=dict(low.IMAGE_ENV)
    environment.update(PYTHONPATH='/runtime/gym-overlay:/learner/src:/learner',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',TMPDIR='/output',XDG_CACHE_HOME='/output',
        FRESH_BOOTSTRAP_CONTEXT=digest)
    spec=low.PhaseSpec('fresh-initial-policy-corrected-1','spirelearn-fresh-initial-corrected-1-20261102-a','curriculum-20261102-a',
        digest,digest,digest,start+40.,deadline,'/output',
        ('-B','/learner/scripts/neural/fresh_curriculum_bootstrap.py','--guest'),tuple(sorted(environment.items())),
        ((str(root/low.VENV),'/runtime/venv-cuda',True),
         (str(root/runtime['package_overlay']),'/runtime/gym-overlay',True),
         (str(root/'src'),'/learner/src',True),(str(root/'configs'),'/learner/configs',True),
         (str(root/'scripts'),'/learner/scripts',True),(str(root/'tests'),'/learner/tests',True),
         (str(output),'/output',False)),(('learn.fresh-bootstrap',digest),),runtime_class='cpu',pids_limit=128)
    with (domain/'bootstrap-started.json').open('x') as f:
        json.dump(dict(source=source,spec=asdict(spec),release_sha256=sha(domain/'bootstrap-release.json')),f,indent=2)
    client.mkdir();output.mkdir();logs=[];result=None;primary=None
    try:
        docker=low._docker_binary()
        raw=low._command(deps,low._docker(docker,client,'image','inspect',low.IMAGE),deadline=deadline,logs=logs,client=client)
        verify_image_inspect(raw)
        with transport.signal_scope():
            result=transport.execute_job(spec,deps=deps,client=client,docker=docker,logs=logs,progress=None)
    except BaseException as error:
        primary=error
        raise
    finally:
        record=dict(result=result,failure=None if primary is None else {'type':type(primary).__name__,'message':str(primary)},
            elapsed_seconds=deps.clock()-start,source_postcheck=source_inventory()==source,
            protected_postcheck=all(sha(root/name)==expected for name,expected in protected.items()),commands=logs)
        try:
            with (domain/'bootstrap-process.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        except BaseException as final_error:
            if primary is None:raise
            primary.add_note('bootstrap evidence write failed: '+type(final_error).__name__)
    if not record['source_postcheck'] or not record['protected_postcheck'] or record['elapsed_seconds']>120:
        raise ValueError('bootstrap final accounting/preservation failed')
    print(json.dumps(dict(status='complete',elapsed_seconds=record['elapsed_seconds'],container_id=result['container_id'])))


if __name__=='__main__':
    if sys.argv[1:]==['--guest']:
        _guest()
    elif sys.argv[1:]==[]:
        _host()
    else:
        raise SystemExit('unsupported bootstrap arguments')
