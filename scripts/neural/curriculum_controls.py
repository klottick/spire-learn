"""Single-use curriculum host controls. Import and validation are stdlib only."""
from __future__ import annotations

import time
_ENTRY_MONOTONIC = time.monotonic()

import argparse
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys

if __package__ in (None, ''):
    _root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_root / 'src'), str(_root)]

from scripts.neural import curriculum_package as package_api, curriculum_runtime as producer
from scripts.neural import sustained_controls as transport, task9_study_controls as low
from scripts.neural import task9_nvidia_cdi as cdi
from scripts.neural.gpu_runtime_proof_helpers import HostDependencies, write_json_exclusive

DOMAIN = Path('artifacts/development/neural-combat/curriculum-campaign-v1')
RUNTIME = DOMAIN / 'runtime'
QUAL_INPUT = Path('artifacts/development/neural-combat/curriculum-v1/task-5/prepared-mixed')
PACKAGE_INPUT = DOMAIN / 'inputs'
SOURCE_SCHEMA = 'spire-learn-curriculum-source-snapshot-v1'
RELEASE_SCHEMA = 'spire-learn-curriculum-release-v1'
CLAIM_SCHEMA = 'spire-learn-curriculum-claim-v1'
ENV_CONTEXT = 'SPIRE_LEARN_CURRICULUM_CONTEXT_SHA256'
CAPS = dict(environment_attempts=1200000, sample_rows=1100000, optimizer_attempts=70000,
            reset_slots=1100000, fixed_random_choices=0)
GIB = 1024**3
PIN_VERSIONS = {'torch': '2.14.1+cu130', 'numpy': '2.5.4', 'gymnasium': '1.4.0', 'spire-lab': '0.1.0'}
PYTHON_SHA256 = '4dbf3143240288fb2170257ffaa7bd030cdda5d2703d1f5f30b627042267e2e3'
EXTRA_SOURCE = ('scripts/neural/curriculum_controls.py', 'scripts/neural/run-curriculum-campaign.sh',
    'scripts/neural/curriculum-campaign-runbook.md', 'tests/unit/test_neural_curriculum_controls.py',
    package_api.CAMPAIGN_PATH, 'configs/neural/curriculum-v1/pilot.toml',
    'configs/neural/curriculum-v1/recipe.json', 'configs/neural/curriculum-v1/support.json',
    'configs/neural/curriculum-v1/pilot-parents.json', cdi.PROFILE_PATH)
SOURCE_ROOTS = ('src', 'configs', 'scripts', 'tests')
RELEASE_KEYS = {'schema', 'state', 'campaign_id', 'single_use', 'source_snapshot_sha256',
    'package_sha256', 'campaign_raw_sha256', 'qualification_input_inventory', 'runtime_pins',
    'caps', 'user_approval', 'review'}
_control_state = ContextVar('curriculum_first_control', default=None)


@contextmanager
def campaign_signal_scope():
    """Reuse owned cleanup deferral, retaining the actual first signal object."""
    with transport.signal_scope():
        deferred = transport._signals.get()
        state = {'first': None}
        token = _control_state.set(state)
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        def handler(number, frame):
            # The generic handler replaces deferred controls. Stop that
            # replacement before it happens, including within one command.
            if deferred['cleanup'] and deferred['deferred'] is not None:
                if state['first'] is None:
                    state['first'] = deferred['deferred']
                return
            try:
                previous[number](number, frame)
            except BaseException as error:
                if state['first'] is None:
                    state['first'] = error
                raise state['first']
            finally:
                if state['first'] is None and deferred['deferred'] is not None:
                    state['first'] = deferred['deferred']
        for number in previous:
            signal.signal(number, handler)
        try:
            yield
        except BaseException as error:
            first = state['first']
            if first is not None and first is not error:
                first.add_note('other campaign failure: '+str(low._error(error)))
                raise first
            raise
        finally:
            for number, original in previous.items():
                signal.signal(number, original)
            _control_state.reset(token)


def execute_transaction(spec, *, deps, **kwargs):
    """The unchanged generic transaction plus local first-control promotion."""
    scope = campaign_signal_scope() if _control_state.get() is None else nullcontext()
    with scope:
        state = _control_state.get()
        def run(*args, **options):
            try:
                return deps.run(*args, **options)
            except BaseException as error:
                if not isinstance(error, Exception) and state['first'] is None:
                    state['first'] = error
                raise
        try:
            result = transport.execute_job(spec, deps=replace(deps, run=run), **kwargs)
        except BaseException as error:
            if state['first'] is not None and state['first'] is not error:
                state['first'].add_note('other transaction failure: '+str(low._error(error)))
                raise state['first']
            raise
        if state['first'] is not None:
            raise state['first']
        return result


def _read(path):
    return package_api.read_json(Path(path), maximum=32*1024**2)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(path, value)


def _same(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _snapshot_schema(snapshot):
    if (not isinstance(snapshot, Mapping)
            or set(snapshot) != {'schema', 'state', 'learner_head', 'dirty', 'files', 'modes'}
            or snapshot['schema'] != SOURCE_SCHEMA or snapshot['state'] != 'source_only'
            or not isinstance(snapshot['learner_head'], str)
            or re.fullmatch('[0-9a-f]{40}', snapshot['learner_head']) is None
            or not isinstance(snapshot['dirty'], Mapping)
            or not isinstance(snapshot['files'], Mapping) or not snapshot['files']
            or not isinstance(snapshot['modes'], Mapping)
            or set(snapshot['files']) != set(snapshot['modes'])):
        raise ValueError('source snapshot schema differs')


def _files(root, scopes=SOURCE_ROOTS):
    found = []
    for scope in scopes:
        directory = Path(root) / scope
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('symlink in closed source')
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError('nonregular source file')
            found.append(str(path.relative_to(root)))
    return sorted(found)


def capture_source(root: Path, destination: Path, *, learner_head: str, dirty: Mapping) -> Mapping:
    root = Path(root).resolve(strict=True)
    if not isinstance(learner_head, str) or re.fullmatch('[0-9a-f]{40}', learner_head) is None or not isinstance(dirty, Mapping):
        raise ValueError('full source revision and honest dirty mapping required')
    names = sorted(set(producer.source_paths(root)) | set(EXTRA_SOURCE))
    files = package_api.inventory(root, names)
    modes = {name: stat.S_IMODE((root/name).stat().st_mode) for name in names}
    value = dict(schema=SOURCE_SCHEMA, state='source_only', learner_head=learner_head,
        dirty=dict(dirty), files=files, modes=modes)
    _write(destination, value)
    return value


def verify_source(root, snapshot):
    _snapshot_schema(snapshot)
    if set(_files(root)) != set(snapshot['files']):
        raise ValueError('source inventory is not closed')
    package_api.verify_inventory(root, snapshot['files'])
    if any(type(mode) is not int or mode not in (0o644, 0o664, 0o755, 0o775)
           or stat.S_IMODE((Path(root)/name).stat().st_mode) != mode for name, mode in snapshot['modes'].items()):
        raise ValueError('source mode differs')


def stage_source(root, destination, snapshot):
    destination = Path(destination)
    destination.mkdir()
    package_api.verify_inventory(root, snapshot['files'])
    for name in snapshot['files']:
        source = package_api.relative_file(root, name)
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(snapshot['modes'][name])
    verify_source(destination, snapshot)


def verify_python_link(venv):
    """The qualified interpreter is an OCI path, possibly dangling on the host."""
    entry = Path(venv)/'bin/python3.12'
    if entry.is_symlink():
        if os.readlink(entry) != '/usr/local/bin/python3.12':
            raise ValueError('qualified container Python symlink differs')
    elif not entry.is_file() or package_api.sha256(entry) != PYTHON_SHA256:
        raise ValueError('qualified copied container Python differs')


def runtime_pins(root):
    root = Path(root)
    verify_python_link(root/low.VENV)
    for name, digest in ((low.LOCK, low.LOCK_SHA256), (low.NATIVE, low.NATIVE_SHA256)):
        if package_api.sha256(low._safe(root, name, exists=True)) != digest:
            raise ValueError('qualified runtime bytes differ')
    versions = low._runtime_metadata(root/low.VENV/'lib/python3.12/site-packages')
    if any(versions.get(name) != value for name, value in PIN_VERSIONS.items()):
        raise ValueError('qualified installed metadata differs')
    qualification = root/low.RUNTIME/'device-check-work/device-check.json'
    evidence = _read(qualification)
    expected = dict(status='qualified', gpu_uuid=low.GPU_UUID, visible_gpu_uuid=low.GPU_UUID,
        torch=PIN_VERSIONS['torch'], torch_cuda='13.0', compute_capability=[12, 0])
    if any(evidence.get(k) != v for k, v in expected.items()):
        raise ValueError('qualified device evidence differs')
    cdi.load_runtime_profile(root)
    return dict(image=low.IMAGE, image_id=low.IMAGE_ID, gpu_uuid=low.GPU_UUID,
        python_sha256=PYTHON_SHA256,
        lock_sha256=low.LOCK_SHA256, native_sha256=low.NATIVE_SHA256,
        profile_sha256=package_api.sha256(root/cdi.PROFILE_PATH),
        qualification_sha256=package_api.sha256(qualification), versions=PIN_VERSIONS,
        threads={'intraop': 4, 'interop': 1}, cpu=4, memory_bytes=8*GIB,
        qualification_pids=128, campaign_pids=512)


def release_caps(root):
    campaign = _read(Path(root)/package_api.CAMPAIGN_PATH)
    return dict(global_effect_caps=CAPS, whole_seconds=36000, training_stop_second=33600,
        evaluation_seconds=1800, storage=campaign['storage'], matched_arm=campaign['matched_arm'],
        qualification=campaign['qualification'], segment=campaign['segment'])


def qualification_inventory(root):
    directory = Path(root)/QUAL_INPUT
    names = sorted(p.name for p in directory.iterdir() if p.is_file())
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError('held qualification inputs are not closed regular files')
    return package_api.inventory(directory, names)


def release_template(root, *, package_path, source_snapshot, user_approval):
    root = Path(root).resolve(strict=True)
    approval = low._safe(root, user_approval, exists=True)
    return dict(schema=RELEASE_SCHEMA, state='pending_review', campaign_id=package_api.CAMPAIGN_ID,
        single_use=True, source_snapshot_sha256=package_api.sha256(source_snapshot),
        package_sha256=package_api.sha256(package_path),
        campaign_raw_sha256=package_api.sha256(root/package_api.CAMPAIGN_PATH),
        qualification_input_inventory=qualification_inventory(root), runtime_pins=runtime_pins(root),
        caps=release_caps(root), user_approval=dict(path=str(approval.relative_to(root)), sha256=package_api.sha256(approval)),
        review={'approved': False, 'reports': []})


def validate_release(root, *, package_path, source_snapshot, release_path):
    root = Path(root).resolve(strict=True)
    release, snapshot = _read(release_path), _read(source_snapshot)
    _snapshot_schema(snapshot)
    if (set(release) != RELEASE_KEYS or release['schema'] != RELEASE_SCHEMA
            or release['state'] != 'approved' or release['campaign_id'] != package_api.CAMPAIGN_ID
            or release['single_use'] is not True or not isinstance(release['review'], dict)
            or set(release['review']) != {'approved', 'reports'} or release['review']['approved'] is not True
            or not isinstance(release['review']['reports'], list) or len(release['review']['reports']) < 2
            or any(not isinstance(r, str) or not r for r in release['review']['reports'])):
        raise ValueError('new reviewed single-use release required')
    approval = release['user_approval']
    if not isinstance(approval, dict) or set(approval) != {'path', 'sha256'}:
        raise ValueError('user approval binding required')
    expected = release_template(root, package_path=package_path, source_snapshot=source_snapshot,
                                user_approval=approval['path'])
    if any(not _same(release[k], expected[k]) for k in RELEASE_KEYS - {'state', 'review'}):
        raise ValueError('release source/package/approval/runtime/cap binding differs')
    authority = _read(low._safe(root, approval['path'], exists=True))
    if (authority.get('campaign_id') != package_api.CAMPAIGN_ID or authority.get('new_runtime_authorized') is not True
            or authority.get('whole_runtime_wall_seconds') != 36000 or authority.get('standalone_only') is not True):
        raise ValueError('user authority scope differs')
    package_api.verify_inventory(root, snapshot['files'])
    if set(snapshot['files']) != set(producer.source_paths(root)) | set(EXTRA_SOURCE):
        raise ValueError('reviewed source inventory differs')
    if any(stat.S_IMODE((root/name).stat().st_mode) != mode for name, mode in snapshot['modes'].items()):
        raise ValueError('reviewed checkout source mode differs')
    # Only the host sees the full 264-shard package. Guests verify selected bytes.
    package = package_api.validate_package(root, Path(package_path))
    from scripts.neural.sustained_runtime import checkpoint_inventory
    original = package['original_parent']
    if checkpoint_inventory(root/original['checkpoint_directory']) != original['inventory']:
        raise ValueError('original-B checkpoint differs')
    return dict(release=release, snapshot=snapshot, package=package)


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid finite ' + name)
    return float(value)


def _counts(value):
    if not isinstance(value, Mapping) or set(value) != set(CAPS):
        raise ValueError('effect keys differ')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('effect values must be nonnegative primitive integers')
    return dict(value)


class CampaignBudget:
    """Shared deadline and effect leases; unknown dispatches retain their full lease."""
    def __init__(self, caps, *, start, deadline, clock=time.monotonic):
        self.caps = _counts(caps)
        self.start, self.deadline = _number(start, 'start'), _number(deadline, 'deadline')
        if not 0 < self.deadline - self.start <= 36000:
            raise ValueError('whole campaign deadline differs')
        self.clock = clock
        self.reserved = dict.fromkeys(CAPS, 0)
        self.attempted = dict.fromkeys(CAPS, 0)
        self.jobs = {}
        self.evaluation_seconds = 0.
        self.training_seconds = dict.fromkeys(('fixed', 'randomized', 'mixed'), 0.)
        self.qualification_seconds = 0.

    def reserve(self, job_id, lease):
        lease = _counts(lease)
        if job_id in self.jobs or any(self.reserved[k] + lease[k] > self.caps[k] for k in CAPS):
            raise ValueError('duplicate job or global effect reservation exhausted')
        self.reserved = {k: self.reserved[k] + lease[k] for k in CAPS}
        self.jobs[job_id] = dict(state='reserved', lease=lease)

    def settle(self, job_id, effects):
        item = self.jobs[job_id]
        if item['state'] != 'reserved' or set(effects) != set(CAPS):
            raise ValueError('effect settlement identity differs')
        actual = {}
        for key, lease in item['lease'].items():
            row = effects[key]
            if (not isinstance(row, Mapping) or set(row) != {'reserved', 'attempted', 'delivered', 'uncertain'}
                    or any(type(n) is not int or n < 0 for n in row.values())
                    or not row['delivered'] <= row['attempted'] <= row['reserved'] <= lease
                    or row['uncertain'] != 0 or row['attempted'] != row['delivered']):
                raise ValueError('uncertain or inconsistent effect settlement')
            actual[key] = row['attempted']
        self.reserved = {k: self.reserved[k] - item['lease'][k] + actual[k] for k in CAPS}
        self.attempted = {k: self.attempted[k] + actual[k] for k in CAPS}
        item.update(state='settled', actual=actual)

    def job_deadlines(self, kind, *, arm_deadline=None):
        now = _number(self.clock(), 'clock')
        if now < self.start:
            raise ValueError('monotonic clock moved backwards')
        whole = min(now + 600., self.deadline - 110.)
        if kind == 'segment':
            whole = min(whole, self.start + 33600., arm_deadline or self.deadline)
        elif kind == 'panel':
            whole = min(whole, now + 1800. - self.evaluation_seconds)
        soft = min(now + (120. if kind == 'qualification' else 480.), whole - 100.)
        if soft <= now:
            raise TimeoutError('shared job/cleanup/training/evaluation time exhausted')
        return soft, whole


def initialize_stage(root, host, verified):
    """Freeze source once; all nested RO bind targets exist before OCI setup."""
    stage = Path(host)/'source'
    stage_source(root, stage, verified['snapshot'])
    for name in (PACKAGE_INPUT, QUAL_INPUT, Path(verified['package']['original_parent']['checkpoint_directory'])):
        (stage/name).mkdir(parents=True, exist_ok=True)
    (stage/'inputs').mkdir()
    (stage/'inputs/predecessor.json').touch()
    return stage


def _copy_inputs(source, target, bindings):
    target.mkdir(parents=True)
    if not bindings:
        return
    package_api.verify_inventory(source, bindings)
    for name in bindings:
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_api.relative_file(source, name), path)
    package_api.verify_inventory(target, bindings)


def _guest_root():
    return low._guest_path('/learner')


def _environment(context, digest):
    value = transport._environment(context['runtime_class'], context['output_root'], context['deadline'])
    # Producer admission checks that this existing owned directory is empty.
    # All cache/HOME/temp roots therefore exist before optional imports without
    # creating a child that would invalidate the producer's empty-root check.
    value.update({name: context['output_root'] for name in low.SCRATCH})
    value[ENV_CONTEXT] = digest
    if context['kind'] == 'qualification':
        value['SPIRE_LEARN_RUN_NEURAL_CURRICULUM_INTEGRATION'] = '1'
    return value


def _spec(context, context_path, layout):
    digest = package_api.sha256(context_path)
    labels = {f'curriculum.{name}': context[key] for name, key in (
        ('campaign', 'campaign_id'), ('job', 'job_id'), ('source', 'source_sha256'),
        ('release', 'release_sha256'), ('claim', 'claim_sha256'))}
    labels['curriculum.context'] = digest
    command = ('-B', '/learner/scripts/neural/curriculum_runtime.py', '_guest', '--root', '/learner',
               '--context', '/control/job.json')
    return low.PhaseSpec(context['job_id'], layout['name'], context['campaign_id'],
        context['release_sha256'], digest, context['claim_sha256'], context['deadline'],
        context['whole_deadline'], context['output_root'], command,
        tuple(sorted(_environment(context, digest).items())), tuple(tuple(row) for row in layout['mounts']),
        tuple(sorted(labels.items())), runtime_class=context['runtime_class'],
        pids_limit=128 if context['kind'] == 'qualification' else 512)


def prepare_job(root, host, verified, budget, *, kind, job_id, parent, arm=None, index=0,
                prefix=None, predecessor=None, slot=None, arm_deadline=None, max_updates=None):
    """Create a new immutable lease/context and selected RO input projection."""
    root, host = Path(root), Path(host)
    package = verified['package']
    if re.fullmatch(r'curriculum-[a-z0-9-]{1,100}', job_id) is None or kind not in producer.KIND_KEYS:
        raise ValueError('new job identity/kind differs')
    verify_source(host/'source', verified['snapshot'])
    if package_api.sha256(verified['package_path']) != verified['release']['package_sha256']:
        raise ValueError('immutable package changed')
    soft, whole = budget.job_deadlines(kind, arm_deadline=arm_deadline)
    job = host/'jobs'/job_id
    if job.exists() or job_id in budget.jobs:
        raise ValueError('job identity already consumed')
    job.mkdir(parents=True)
    control, output = job/'control', job/'output'
    control.mkdir(); output.mkdir()
    guest = _guest_root()
    package_stage = job/'selected-package'
    original = package['original_parent']
    parent_host = {key: parent[key] for key in ('path', 'inventory')}
    from scripts.neural.sustained_runtime import checkpoint_inventory
    parent_path = low._safe(root, parent_host['path'], exists=True) if Path(parent_host['path']).is_relative_to(root) else Path(parent_host['path'])
    # Outputs may be in the fresh host tree; no other foreign host input is valid.
    if not parent_path.is_relative_to(root) and not parent_path.is_relative_to(host):
        raise ValueError('parent outside retained input or owned output tree')
    if checkpoint_inventory(parent_path) != parent_host['inventory']:
        raise ValueError('parent bytes differ before dispatch')
    prefix = dict(prefix or {})
    if kind == 'segment':
        if arm not in package['training'] or type(index) is not int or not 0 <= index < len(package['training'][arm]):
            raise ValueError('training arm/index differs')
        names = package['training'][arm][index]['inputs']
        bindings = {name: package['inventory'][name] for name in names}
        input_relative = PACKAGE_INPUT
        output_relative = Path('training')/arm/'runs'
        reservation = dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
                           reset_slots=4112, fixed_random_choices=0)
        ceiling = 1000000 if arm == 'mixed' else 16384
        adam_ceiling = 65536 if arm == 'mixed' else 1024
        remaining_actions = ceiling-prefix.get('attempted_environment_slots', 0)
        remaining_adam = adam_ceiling-prefix.get('optimizer_attempted_steps', 0)
        updates = min(8, remaining_adam//32)
        if remaining_actions <= 0 or updates <= 0:
            raise TimeoutError('arm cumulative effect ceiling exhausted')
        reservation['environment_attempts'] = min(4096, remaining_actions)
        reservation['sample_rows'] = reservation['environment_attempts']
        reservation['optimizer_attempts'] = min(256, remaining_adam)
        reservation['reset_slots'] = min(4112, reservation['environment_attempts']+16)
        extra = dict(arm=arm, shard_index=index, prefix=prefix,
            max_updates=prefix.get('update_count', 0)+updates if max_updates is None else max_updates,
            max_environment_actions=prefix.get('attempted_environment_slots', 0)+reservation['environment_attempts'],
            predecessor=None)
    elif kind == 'panel':
        names = sorted({package['panel']['path']} | {name for row in package['panel']['cases'] for name in row['inputs']})
        bindings = {name: package['inventory'][name] for name in names}
        input_relative, output_relative = PACKAGE_INPUT, Path('panel/runs')
        reservation = dict(environment_attempts=16384, sample_rows=0, optimizer_attempts=0, reset_slots=64, fixed_random_choices=0)
        extra = dict(slot=slot, case_ids=[row['case_id'] for row in package['panel']['cases']])
    else:
        bindings = verified['release']['qualification_input_inventory']
        input_relative, output_relative = QUAL_INPUT, Path('runs')
        reservation = {key: verified['release']['caps']['qualification'][key] for key in CAPS}
        extra = {}
    _copy_inputs(Path(verified['package_path']).parent, package_stage,
                 bindings if kind != 'qualification' else {})
    shutil.copyfile(verified['package_path'], package_stage/'package.json')
    if kind == 'qualification':
        input_stage = job/'selected-qualification'
        _copy_inputs(root/QUAL_INPUT, input_stage, bindings)
    else:
        input_stage = package_stage
    (input_stage/output_relative).mkdir(parents=True)
    guest_output = str(guest/input_relative/output_relative)
    parent_guest = str(guest/original['checkpoint_directory']) if kind == 'qualification' else '/inputs/parent'
    mounts = [(str(host/'source'), str(guest), True),
        (str(package_stage), str(guest/PACKAGE_INPUT), True),
        (str(parent_path), parent_guest if kind == 'qualification' else str(low._guest_path(parent_guest)), True),
        (str(control), str(low._guest_path('/control')), True),
        (str(root/low.VENV), str(low._guest_path('/runtime/venv-cuda')), True),
        (str(root/low.LOCK), str(low._guest_path('/runtime/requirements-cuda.lock')), True),
        (str(output), guest_output, False)]
    if kind == 'qualification':
        mounts.insert(2, (str(input_stage), str(guest/QUAL_INPUT), True))
    if predecessor is not None:
        if kind != 'segment' or set(predecessor) != {'path', 'sha256'}:
            raise ValueError('predecessor binding differs')
        prior = Path(predecessor['path'])
        if not prior.is_relative_to(host) or package_api.sha256(prior) != predecessor['sha256']:
            raise ValueError('predecessor outside owned receipt chain or changed')
        shutil.copyfile(prior, control/'predecessor.json')
        extra['predecessor'] = {'path': 'inputs/predecessor.json', 'sha256': predecessor['sha256']}
        mounts.append((str(control/'predecessor.json'), str(guest/'inputs/predecessor.json'), True))
    # Lease reservation is retained if any publication, preflight or dispatch fails.
    budget.reserve(job_id, reservation)
    claim = dict(schema=CLAIM_SCHEMA, campaign_id=package['campaign_id'], job_id=job_id,
        kind=kind, release_sha256=package_api.sha256(verified['release_path']),
        source_sha256=package_api.sha256(verified['source_snapshot']),
        package_sha256=package_api.sha256(verified['package_path']),
        campaign_claim_sha256=package_api.sha256(verified['claim_path']), reservation=reservation,
        predecessor=extra.get('predecessor'))
    claim_path = host/'job-claims'/(job_id+'.json')
    _write(claim_path, claim)
    for name, path in (('snapshot.json', verified['source_snapshot']), ('release.json', verified['release_path']),
                       ('campaign-claim.json', verified['claim_path']), ('claim.json', claim_path)):
        shutil.copyfile(path, control/name)
    context = dict(schema=producer.CONTEXT_SCHEMA, campaign_id=package['campaign_id'], job_id=job_id,
        kind=kind, runtime_class='cpu' if kind == 'panel' else 'gpu',
        source_inventory=verified['snapshot']['files'], source_sha256=claim['source_sha256'],
        release_sha256=claim['release_sha256'], claim_sha256=package_api.sha256(claim_path),
        package_path=str(PACKAGE_INPUT/'package.json'), package_sha256=claim['package_sha256'],
        input_root=str(input_relative), input_inventory=bindings,
        parent=dict(path=parent_guest, inventory=parent_host['inventory']), output_root=guest_output,
        deadline=soft, whole_deadline=whole, reservation=reservation, **extra)
    context_path = control/'job.json'
    _write(context_path, context)
    layout = dict(schema='spire-learn-curriculum-layout-v1', name=job_id, mounts=mounts)
    _write(control/'layout.json', layout)
    return dict(context=context, context_path=context_path, spec=_spec(context, context_path, layout),
                output=output, parent_host=parent_host, input_stage=input_stage)


def verify_guest_context(root, context_path, *, clock=time.monotonic):
    """Admission in the executing process, before resolving optional packages."""
    root, context_path = Path(root), Path(context_path)
    if root != _guest_root() or context_path != low._guest_path('/control/job.json'):
        raise ValueError('guest entry namespace differs')
    context = _read(context_path)
    digest = package_api.sha256(context_path)
    if os.environ.get(ENV_CONTEXT) != digest:
        raise ValueError('guest raw context binding differs')
    control = context_path.parent
    release, snapshot, claim, campaign_claim, layout = [_read(control/name) for name in (
        'release.json', 'snapshot.json', 'claim.json', 'campaign-claim.json', 'layout.json')]
    for name, key in (('release.json', 'release_sha256'), ('snapshot.json', 'source_sha256'), ('claim.json', 'claim_sha256')):
        if package_api.sha256(control/name) != context[key]:
            raise ValueError('guest source/release/claim raw chain differs')
    if (set(release) != RELEASE_KEYS or release['schema'] != RELEASE_SCHEMA or release['state'] != 'approved'
            or release['single_use'] is not True or release['campaign_id'] != context['campaign_id']
            or release['source_snapshot_sha256'] != context['source_sha256']
            or release['package_sha256'] != context['package_sha256'] or release['review']['approved'] is not True
            or not _same(release['caps'], release_caps(root))
            or release['campaign_raw_sha256'] != package_api.sha256(root/package_api.CAMPAIGN_PATH)
            or claim.get('schema') != CLAIM_SCHEMA or claim.get('campaign_id') != context['campaign_id']
            or any(claim.get(k) != context[k] for k in ('job_id', 'kind', 'release_sha256', 'source_sha256', 'package_sha256', 'reservation'))
            or claim.get('predecessor') != context.get('predecessor')
            or claim.get('campaign_claim_sha256') != package_api.sha256(control/'campaign-claim.json')
            or campaign_claim.get('schema') != CLAIM_SCHEMA
            or any(campaign_claim.get(k) != context[k] for k in ('campaign_id', 'source_sha256', 'release_sha256', 'package_sha256'))):
        raise ValueError('guest acyclic authority chain differs')
    if snapshot['files'] != context['source_inventory']:
        raise ValueError('guest source inventory differs')
    verify_source(root, snapshot)
    package, input_root, parent, output, config = producer.validate_context(root, context, clock=clock)
    if context['kind'] == 'qualification' and context['input_inventory'] != release['qualification_input_inventory']:
        raise ValueError('guest held qualification input differs')
    expected = {(str(root), True), (str(root/PACKAGE_INPUT), True),
        (str(low._guest_path('/control')), True), (str(low._guest_path('/runtime/venv-cuda')), True),
        (str(low._guest_path('/runtime/requirements-cuda.lock')), True), (context['output_root'], False),
        (str(parent), True)}
    if context['kind'] == 'qualification':
        expected.add((str(root/QUAL_INPUT), True))
    if context.get('predecessor'):
        expected.add((str(root/'inputs/predecessor.json'), True))
    if (set(layout) != {'schema', 'name', 'mounts'} or layout['schema'] != 'spire-learn-curriculum-layout-v1'
            or layout['name'] != context['job_id'] or not isinstance(layout['mounts'], list)
            or any(not isinstance(row, list) or len(row) != 3 or type(row[2]) is not bool for row in layout['mounts'])
            or len(layout['mounts']) != len(expected) or {(d, r) for s, d, r in layout['mounts']} != expected):
        raise ValueError('guest selected mount layout differs')
    # Close selected file inventories, excluding the one empty writable overlay.
    selected_names = set(context['input_inventory'])
    actual = {str(p.relative_to(input_root)) for p in input_root.rglob('*') if p.is_file()}
    if input_root == root/PACKAGE_INPUT:
        selected_names.add('package.json')
    if actual != selected_names or any(p.is_symlink() for p in input_root.rglob('*')):
        raise ValueError('guest selected input tree is not closed')
    spec, facts = _spec(context, context_path, layout), low._guest_facts()
    if any(facts.get(k) != v for k, v in (('uid', 1000), ('gid', 1000), ('python', low.CONTAINER_PYTHON))):
        raise ValueError('executing guest UID/GID/Python differs')
    profile = cdi.validate_runtime_class(cdi.load_runtime_profile(root), runtime_class=context['runtime_class'],
        requested_environment=dict(spec.environment), actual_environment=facts['environment'], facts=facts)
    if context['runtime_class'] == 'gpu' and profile['mode'] != 'nvidia-cdi':
        raise ValueError('qualified CDI execution profile required')
    table = dict(facts['mounts'])
    if len(table) != len(facts['mounts']) or table.get('/') is not True or any(table.get(d) is not r for d, r in expected):
        raise ValueError('actual guest root/application mounts differ')
    admitted = set(profile['admitted_mount_destinations'])
    for destination, readonly in facts['mounts']:
        automatic = destination == '/' or destination in ('/etc/hosts', '/etc/hostname', '/etc/resolv.conf') or any(
            destination == path or destination.startswith(path+'/') for path in ('/proc', '/dev', '/sys'))
        if destination in dict(expected) or destination in admitted:
            continue
        if not automatic or destination.startswith(('/learner', '/inputs', '/control', '/runtime')):
            raise ValueError('unexpected guest mount '+destination)
    low._verify_guest_runtime()
    verify_python_link(low._guest_path('/runtime/venv-cuda'))
    versions = low._runtime_metadata(low._guest_path('/runtime/venv-cuda/lib/python3.12/site-packages'))
    if any(versions.get(k) != v for k, v in PIN_VERSIONS.items()):
        raise ValueError('executing installed versions differ')
    if (release['runtime_pins']['profile_sha256'] != cdi.PROFILE_SHA256
            or release['runtime_pins']['native_sha256'] != low.NATIVE_SHA256
            or release['runtime_pins']['lock_sha256'] != low.LOCK_SHA256
            or release['runtime_pins']['versions'] != PIN_VERSIONS
            or release['runtime_pins']['python_sha256'] != PYTHON_SHA256):
        raise ValueError('guest runtime pins differ')
    record = dict(schema='spire-learn-curriculum-admission-v1', job_id=context['job_id'],
        context_sha256=digest, runtime_class=context['runtime_class'],
        process={k: facts[k] for k in ('uid', 'gid', 'python', 'supplementary_group_ids')},
        runtime_profile=profile, installed_distributions=versions)
    encoded = transport.MARKER+json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > 16384:
        raise ValueError('guest admission record too large')
    if sys.stderr.write(encoded) != len(encoded):
        raise OSError('short guest admission write')
    sys.stderr.flush()
    return context


def storage_admission(root, host, deps, *, reservation=0):
    if type(reservation) is not int or reservation < 0:
        raise ValueError('invalid storage reservation')
    new = low.tree_bytes(Path(host), strict=True)
    aggregate = low.tree_bytes(Path(root)/'artifacts/development/neural-combat', strict=False)
    free = deps.disk_usage(root).free
    if new+reservation > 12*GIB or aggregate+reservation > 40*GIB or free-reservation < 30*GIB:
        raise RuntimeError('storage admission ceiling/floor exhausted')
    return dict(new_bytes=new, aggregate_bytes=aggregate, free_bytes=free, reservation_bytes=reservation)


def _mapped(context, output, value):
    path = Path(value)
    base = Path(context['output_root'])
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(base):
        raise ValueError('receipt path outside exact job output overlay')
    return package_api.relative_file(output, str(path.relative_to(base)))


def _journal(context, receipt, output):
    control = Path(output)/'campaign-control'
    attempts = _read(control/'attempts.json')
    if (attempts.get('error') is not None or attempts.get('unresolved_effect_ids') != []
            or attempts.get('effects') != receipt['effects']):
        raise ValueError('actual retained effect attempts differ or remain unresolved')
    totals = {k: dict.fromkeys(('reserved', 'attempted', 'delivered', 'uncertain'), 0) for k in CAPS}
    pending, seen, categories = {}, set(), {}
    labels = ({'qualification-'+name: target for name, target in dict(actions='environment_attempts',
        samples='sample_rows', adam='optimizer_attempts', resets='reset_slots').items()}
        if context['kind'] == 'qualification' else {'panel-step': 'environment_attempts', 'panel-reset': 'reset_slots'}
        if context['kind'] == 'panel' else {'native-step': 'environment_attempts', 'reset': 'reset_slots',
            'training_update': 'optimizer_attempts', 'stochastic_actor_rows': 'sample_rows'})
    for index, part in enumerate(attempts['journal']):
        if set(part) != {'path', 'sha256', 'size_bytes'} or part['path'] != f'effects-{index:04d}.json':
            raise ValueError('journal part identity differs')
        file = package_api.relative_file(control, part['path'])
        if package_api.sha256(file) != part['sha256'] or file.stat().st_size != part['size_bytes']:
            raise ValueError('journal bytes differ')
        for row in _read(file)['records']:
            identifier, kind = row['operation_id'], row['kind']
            if kind not in totals or row['schema'] != 'spire-learn-profile-effect-v1' or labels.get(row['label']) != kind:
                raise ValueError('effect journal schema/kind differs')
            if row['event'] == 'reserve':
                if (set(row) != {'event', 'schema', 'operation_id', 'kind', 'label', 'reserved'}
                        or identifier in seen or len(seen) >= 65536
                        or identifier != f"{context['job_id']}:{len(seen)+1:06d}"
                        or type(row['reserved']) is not int or row['reserved'] < 0):
                    raise ValueError('effect reservation identity differs')
                seen.add(identifier); pending[identifier] = row
                totals[kind]['reserved'] += row['reserved']
            elif row['event'] == 'settle':
                original = pending.pop(identifier)
                if (set(row) != set(original) | {'attempted', 'delivered', 'uncertain', 'unused'}
                        or any(row[k] != v for k, v in original.items() if k != 'event')
                        or any(type(row[k]) is not int or row[k] < 0 for k in ('attempted', 'delivered', 'uncertain', 'unused'))
                        or row['attempted']+row['unused'] != row['reserved']
                        or row['delivered']+row['uncertain'] != row['attempted']):
                    raise ValueError('effect settlement differs')
                cat = categories.setdefault(row['label'], dict.fromkeys(totals[kind], 0))
                cat['reserved'] += row['reserved']
                for key in ('attempted', 'delivered', 'uncertain'):
                    totals[kind][key] += row[key]; cat[key] += row[key]
            else:
                raise ValueError('unknown effect journal event')
    if pending or totals != receipt['effects'] or categories != attempts['categories']:
        raise ValueError('effect journal totals differ')
    if context['kind'] == 'qualification':
        if attempts.get('qualification_attempts') != receipt.get('qualification_attempts'):
            raise ValueError('actual held qualification counters differ')
    elif receipt['effect_categories'] != categories:
        raise ValueError('effect categories differ')


def _report(context, output, report, files, *, incomplete=False):
    mapped = dict(guest_output_root=context['output_root'], host_output=str(output))
    return transport._validate_reports(mapped, dict(report=report, report_files=files), allow_incomplete=incomplete)


def reconcile_job(root, verified, context, output, *, lifecycle, budget, clock=time.monotonic):
    """A receipt becomes a successor only after full-ID cleanup and byte checks."""
    from scripts.neural.sustained_runtime import checkpoint_inventory, reconcile_segment
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.checkpoint import preflight_checkpoint
    output = Path(output)
    if (lifecycle.get('removal_entered') is not True or re.fullmatch('[0-9a-f]{64}', lifecycle.get('container_id', '')) is None
            or lifecycle.get('admission', {}).get('job_id') != context['job_id']):
        raise ValueError('successful owned cleanup/admission missing')
    path = output/'job-receipt.json'
    receipt = _read(path)
    keys = ('campaign_id', 'job_id', 'kind', 'runtime_class', 'source_sha256', 'package_sha256',
        'release_sha256', 'claim_sha256', 'parent', 'input_inventory', 'deadline', 'whole_deadline')
    if (receipt.get('schema') != producer.RECEIPT_SCHEMA or receipt.get('status') != 'complete'
            or any(receipt.get(k) != context[k] for k in keys)
            or clock() >= context['whole_deadline'] or _number(receipt['elapsed_seconds'], 'elapsed') > 600):
        raise ValueError('job receipt authority/status/deadline differs')
    transport.validate_process_evidence(receipt, reference=False)
    _journal(context, receipt, output)
    checkpoint = None
    package, kind = verified['package'], context['kind']
    if kind == 'segment':
        binding = package['training'][context['arm']][context['shard_index']]
        if any(receipt.get(k) != context[k] for k in ('arm', 'shard_index', 'prefix', 'predecessor')) or any(
                receipt.get(k) != binding[k] for k in ('config_sha256', 'manifest_sha256')):
            raise ValueError('segment config/stream/predecessor differs')
        directory = _report(context, output, receipt['report'], receipt['report_files'])
        physical = _mapped(context, output, receipt['checkpoint']['path'])
        if physical != directory/'checkpoint-final' or checkpoint_inventory(physical) != receipt['checkpoint']['inventory']:
            raise ValueError('published checkpoint byte inventory differs')
        config = load_neural_config(Path(verified['package_path']).parent/binding['config_path'])
        loaded = preflight_checkpoint(physical, config.accepted_identities[0], purpose='resume')
        physical_context = {**context, 'parent': {**context['parent'],
            'path': str(low._guest_path(context['parent']['path']))}}
        saved, delta = reconcile_segment(config, physical_context, receipt['report'], loaded.metadata,
                                         receipt['effects'], elapsed=receipt['elapsed_seconds'])
        from spire_learn.neural.device import plan_runtime_device, validate_resume_device
        validate_resume_device(loaded.metadata['run_metadata'], plan_runtime_device(config))
        if (saved != receipt['checkpoint']['counters'] or delta != receipt['delta']
                or saved['update_count'] > context['max_updates']
                or saved['attempted_environment_slots'] > context['max_environment_actions']
                or receipt['next_shard'].get('shard_index') != context['shard_index']+1):
            raise ValueError('saved prefix/delta/successor differs')
        checkpoint = dict(path=str(physical), inventory=checkpoint_inventory(physical), counters=saved)
    elif kind == 'qualification':
        actual = receipt['qualification_attempts']
        names = dict(actions='environment_attempts', samples='sample_rows', adam='optimizer_attempts', resets='reset_slots')
        if (receipt.get('healthy_harness') is not True or set(actual) != set(names)
                or receipt.get('qualification_method') != 'tests.neural.test_curriculum.CurriculumIntegrationTests.test_public_catalog_tensor_resume_and_tiny_healthy_segments'
                or receipt.get('threads') != {'intraop': 4, 'interop': 1}
                or len(receipt['qualification_segments']) != 2
                or any(type(actual[k]) is not int or not 0 < actual[k] <= context['reservation'][v]
                       or actual[k] != receipt['effects'][v]['attempted'] for k, v in names.items())):
            raise ValueError('exact healthy held qualification evidence differs')
        if actual['adam'] != 8:
            raise ValueError('held qualification did not complete its eight Adam attempts')
        from spire_learn.neural.runner import _checkpoint_recovery
        for index, segment in enumerate(receipt['qualification_segments']):
            checkpoint_path = _mapped(context, output, segment['checkpoint']['path'])
            if checkpoint_inventory(checkpoint_path) != segment['checkpoint']['inventory']:
                raise ValueError('qualification saved checkpoint changed')
            report = segment['report']
            config = load_neural_config(Path(root)/QUAL_INPUT/f'qualification-{index}.toml')
            saved = preflight_checkpoint(checkpoint_path, config.accepted_identities[0], purpose='resume')
            saved_prefix = _checkpoint_recovery(saved, config)[1]
            # The unchanged held method passes its frozen root-relative pilot
            # parent verbatim; campaign producers use /inputs/parent instead.
            expected_parent = (package['original_parent']['checkpoint_directory'] if index == 0 else
                receipt['qualification_segments'][index-1]['checkpoint']['path'])
            if (saved_prefix != segment['saved_prefix'] or segment['next_shard'].get('shard_index') != index+1
                    or saved.metadata['run_metadata']['scenario_stream']['current']['shard_index'] != index
                    or saved.metadata.get('parent_checkpoint') != {'path': expected_parent}
                    or any(saved_prefix[k] for k in ('uncertain_environment_slots', 'uncertain_reset_slots', 'optimizer_uncertain_steps'))):
                raise ValueError('held saved prefix/parent/successor differs')
            files = dict(path=str(Path(segment['checkpoint']['path']).parent), inventory=package_api.inventory(
                checkpoint_path.parent, ('summary.json', 'lifecycle.json', 'events.jsonl')))
            _report(context, output, report, files)
            if report.get('status') != 'complete' or report.get('fault') or report.get('lifecycle', {}).get('close_errors'):
                raise ValueError('qualification segment failed lifecycle')
    else:
        rows = package['panel']['cases']
        by_id = {row['case_id']: row for row in rows}
        reports = receipt['reports']
        ids = [row['case_id'] for row in reports]
        if (receipt.get('slot') != context['slot'] or len(set(ids)) != len(ids)
                or ids != [key for key in context['case_ids'] if key in ids]
                or receipt['missing_case_ids'] != [row['case_id'] for row in rows if row['case_id'] not in ids]
                or receipt['summary'].get('status') not in ('complete', 'incomplete')):
            raise ValueError('CPU panel availability/summary differs')
        for row in reports:
            binding = by_id[row['case_id']]
            if any(row.get(k) != binding[k] for k in ('config_sha256', 'manifest_sha256', 'environment_seed')):
                raise ValueError('CPU panel config/seed differs')
            if row['exit_code'] != 0 or row['report'].get('status') != 'complete' or row['report'].get('fault'):
                raise ValueError('CPU panel nested operation failed')
            if row['report_files'] is None:
                raise ValueError('healthy CPU report is missing actual files')
            _report(context, output, row['report'], row['report_files'])
        from spire_learn.neural.runner import NeuralReport
        from spire_learn.neural.stream_runner import summarize_development_reports
        panel = _read(Path(verified['package_path']).parent/package['panel']['path'])
        aggregate = summarize_development_reports(panel, {row['case_id']: NeuralReport(row['report'], row['exit_code'],
            _mapped(context, output, row['report_files']['path'])) for row in reports})
        if not _same(aggregate, receipt['summary']):
            raise ValueError('actual CPU panel summary differs from retained rows')
        if any(receipt['effects'][key]['attempted'] for key in ('sample_rows', 'optimizer_attempts', 'fixed_random_choices')):
            raise ValueError('deterministic CPU panel used stochastic/optimizer effects')
    budget.settle(context['job_id'], receipt['effects'])
    return dict(job_id=context['job_id'], receipt=receipt,
        receipt_binding={'path': str(path), 'sha256': package_api.sha256(path)}, checkpoint=checkpoint)


class CampaignSchedule:
    """Preregistered order. No metric or score is an input to this state machine."""
    def __init__(self, package, *, start, training_seconds=None):
        self.package, self.start = package, start
        self.qualified = False
        self.arm_index = 0
        self.indices = dict.fromkeys(('fixed', 'randomized', 'mixed'), 0)
        self.prefixes = {arm: {} for arm in self.indices}
        self.checkpoints = dict.fromkeys(self.indices, None)
        self.predecessors = dict.fromkeys(self.indices, None)
        self.stopped = {}
        self.arm_started = {}
        self.long_entered = False
        # The live controller shares its actual charged job intervals here;
        # CPU panels and other arms never advance mixed milestones.
        self.training_seconds = (training_seconds if training_seconds is not None else
                                 dict.fromkeys(('fixed', 'randomized', 'mixed'), 0.))
        self.availability = {slot: dict(status='not_started', reason='not reached') for slot in package['evaluation_slots']}
        self.panels_done = set()

    def accept_qualification(self):
        if self.qualified:
            raise ValueError('qualification cannot run twice')
        self.qualified = True

    def finish_panel(self, slot, *, status, reason, result=None):
        if slot not in self.availability or slot in self.panels_done:
            raise ValueError('panel slot missing or consumed')
        self.availability[slot] = dict(status=status, reason=reason, result=result)
        self.panels_done.add(slot)

    def stop_arm(self, arm, *, reason):
        self.stopped[arm] = reason

    def accept_segment(self, arm, result):
        before = self.prefixes[arm]
        after = result['checkpoint']['counters']
        # A save without new actions cannot spin through the remaining shards.
        if after['attempted_environment_slots'] <= before.get('attempted_environment_slots', 0):
            self.stop_arm(arm, reason='zero_progress')
        self.prefixes[arm] = after
        self.checkpoints[arm] = result['checkpoint']
        self.predecessors[arm] = result['receipt_binding']
        self.indices[arm] += 1

    def next(self, now):
        if not self.qualified:
            return dict(kind='qualification')
        if 'original-b' not in self.panels_done:
            return dict(kind='panel', slot='original-b', arm=None)
        arms = ('fixed', 'randomized', 'mixed')
        while self.arm_index < 3:
            arm = arms[self.arm_index]
            if arm not in self.arm_started:
                self.arm_started[arm] = now
            exhausted = (arm in self.stopped or self.indices[arm] >= 4
                or now >= self.arm_started[arm]+3000. or now >= self.start+33600.)
            if not exhausted:
                return dict(kind='segment', arm=arm, index=self.indices[arm],
                    arm_deadline=min(self.arm_started[arm]+3000., self.start+33600.))
            slot = 'matched-'+arm
            if slot not in self.panels_done:
                return dict(kind='panel', slot=slot, arm=arm)
            self.arm_index += 1
        if not self.long_entered:
            self.long_entered = True
            # A short matched time lease does not prohibit the planned mixed
            # continuation; zero progress or a cumulative budget does.
            if self.stopped.get('mixed') == 'time':
                del self.stopped['mixed']
        for seconds in (6000, 12000, 18000):
            slot = f'mixed-time-{seconds}'
            if slot not in self.panels_done and _number(self.training_seconds['mixed'], 'mixed charged seconds') >= seconds:
                return dict(kind='panel', slot=slot, arm='mixed')
        prefix = self.prefixes['mixed']
        if ('mixed' not in self.stopped and self.indices['mixed'] < 256 and now < self.start+33600.
                and prefix.get('attempted_environment_slots', 0) < 1000000
                and prefix.get('optimizer_attempted_steps', 0) < 65536):
            return dict(kind='segment', arm='mixed', index=self.indices['mixed'], arm_deadline=self.start+33600.)
        if 'final-mixed' not in self.panels_done:
            return dict(kind='panel', slot='final-mixed', arm='mixed')
        return None


def dispatch_job(root, host, verified, budget, *, deps, docker, client, **request):
    """One attempt, one immutable context, one owned transaction, no retry."""
    started = deps.clock()
    logs, result, primary, prepared = [], None, None, None
    limit = GIB
    try:
        storage_admission(root, host, deps, reservation=limit)
        prepared = prepare_job(root, host, verified, budget, **request)
        context, output, spec = prepared['context'], prepared['output'], prepared['spec']
        next_storage_check = deps.clock()
        def progress():
            nonlocal next_storage_check
            signals = transport._signals.get()
            if signals is not None and signals['cleanup']:
                return  # The transaction already reserved its owned cleanup tail.
            if deps.clock() >= spec.whole_deadline-35.:
                raise TimeoutError('job whole deadline cleanup reserve exhausted')
            size = low.tree_bytes(output, strict=True)
            if size > limit:
                raise RuntimeError('job output storage lease exceeded')
            if deps.clock() >= next_storage_check:
                storage_admission(root, host, deps, reservation=limit-size)
                next_storage_check = deps.clock()+5.
        verify_source(Path(host)/'source', verified['snapshot'])
        package_api.verify_inventory(prepared['input_stage'], context['input_inventory'])
        progress()
        before = after = None
        if spec.uses_gpu:
            cdi.verify_host_profile(cdi.load_runtime_profile(root))
            raw = low._command(deps, ['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,memory.free',
                '--format=csv,noheader,nounits'], deadline=spec.whole_deadline, logs=logs, client=client, progress=progress)
            from scripts.neural.task8_gpu_setup_helpers import verify_gpu_inventory
            verify_gpu_inventory(raw, minimum_free_mib=12288)
            before = low._gpu_processes(deps, spec.whole_deadline, logs, client)
        if deps.clock() >= context['deadline']:
            raise TimeoutError('setup exhausted shared soft deadline')
        lifecycle = execute_transaction(spec, deps=deps, docker=docker, client=client, logs=logs, progress=progress)
        if spec.uses_gpu:
            after = low._gpu_processes(deps, spec.whole_deadline, logs, client)
        storage_admission(root, host, deps)
        verify_source(Path(host)/'source', verified['snapshot'])
        package_api.verify_inventory(prepared['input_stage'], context['input_inventory'])
        result = reconcile_job(root, verified, context, output, lifecycle=lifecycle, budget=budget, clock=deps.clock)
        result.update(lifecycle=lifecycle, unrelated_gpu_process_observations={'before': before, 'after': after})
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        record = dict(job_id=request['job_id'], status='complete' if result is not None and primary is None else 'failed',
            elapsed_seconds=max(0., deps.clock()-started), commands=logs,
            measurement_boundary='producer/cleanup outcome before host attempt publication and final closure audit',
            error=None if primary is None else low._error(primary),
            reservation=budget.jobs.get(request['job_id']),
            receipt_binding=None if result is None else result['receipt_binding'])
        final = primary
        try:
            _write(Path(host)/'attempts'/(request['job_id']+'.json'), record)
        except BaseException as retention:
            final = low._promote(final, retention)
        try:
            if prepared is not None:
                storage_admission(root, host, deps)
        except BaseException as closure:
            final = low._promote(final, closure)
        try:
            finished = _number(deps.clock(), 'host job closure clock')
            elapsed = _number(finished-started, 'actual enclosing job elapsed')
            if request['job_id'] in budget.jobs:
                budget.jobs[request['job_id']]['host_elapsed_seconds'] = elapsed
            if request['kind'] == 'panel':
                budget.evaluation_seconds += elapsed
            elif request['kind'] == 'segment':
                budget.training_seconds[request['arm']] += elapsed
            else:
                budget.qualification_seconds += elapsed
            if result is not None:
                result['host_elapsed_seconds'] = elapsed
            if prepared is not None and finished >= prepared['context']['whole_deadline']:
                raise TimeoutError('host attempt publication/closure exhausted job whole lease')
            if budget.evaluation_seconds > 1800.:
                raise TimeoutError('actual enclosing CPU evaluation lease exhausted')
        except BaseException as closure:
            final = low._promote(final, closure)
        if final is not None and final is not primary:
            raise final


def run_campaign(root: Path, *, package_path: Path, source_snapshot: Path, release_path: Path,
                 dependencies=None, clock=time.monotonic) -> Mapping:
    """Fresh campaign only, admitted by an independently supervised launcher."""
    root = Path(root).resolve(strict=True)
    deps = dependencies or HostDependencies(clock=clock)
    deadline = _number(float(os.environ.get('SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE', 'nan')), 'parent deadline')
    start = _number(float(os.environ.get('SPIRE_LEARN_CURRICULUM_PARENT_START', 'nan')), 'parent start')
    if deadline != start+36000. or not start <= deps.clock() < deadline-110.:
        raise ValueError('independent parent whole deadline required')
    budget = CampaignBudget(CAPS, start=start, deadline=deadline, clock=deps.clock)
    verified = dict(validate_release(root, package_path=package_path, source_snapshot=source_snapshot, release_path=release_path),
        package_path=Path(package_path), source_snapshot=Path(source_snapshot), release_path=Path(release_path))
    if deps.uid() != 1000 or deps.gid() != 1000:
        raise ValueError('campaign host UID/GID differs')
    host = low._safe(root, RUNTIME)
    if host.exists():
        raise ValueError('fresh campaign runtime domain already consumed')
    storage_admission(root, root/DOMAIN, deps, reservation=GIB)
    host.mkdir()
    # A durable campaign claim is acquired only after complete static admission.
    verified['claim_path'] = host/'campaign-claim.json'
    claim = dict(schema=CLAIM_SCHEMA, campaign_id=package_api.CAMPAIGN_ID,
        source_sha256=package_api.sha256(source_snapshot), release_sha256=package_api.sha256(release_path),
        package_sha256=package_api.sha256(package_path), parent_start=start, whole_deadline=deadline)
    plan = CampaignSchedule(verified['package'], start=start, training_seconds=budget.training_seconds)
    original = verified['package']['original_parent']
    original_parent = dict(path=str(root/original['checkpoint_directory']), inventory=original['inventory'])
    results, primary, status = [], None, 'running'
    try:
        with campaign_signal_scope():
            _write(verified['claim_path'], claim)
            transport._sync_claim_ancestors(root, host)
            client = host/'docker-client'; client.mkdir()
            initialize_stage(root, host, verified)
            docker = low._docker_binary()
            while True:
                request = plan.next(deps.clock())
                if request is None:
                    status = 'complete'
                    break
                kind = request['kind']
                arm = request.get('arm')
                if kind == 'panel':
                    slot = request['slot']
                    parent = original_parent if arm is None else plan.checkpoints[arm]
                    if parent is None:
                        plan.finish_panel(slot, status='unavailable', reason='no healthy arm checkpoint')
                        continue
                    try:
                        budget.job_deadlines('panel')
                    except TimeoutError as error:
                        plan.finish_panel(slot, status='not_started', reason=str(error))
                        continue
                    job_id = 'curriculum-panel-'+slot
                    parameters = dict(kind=kind, slot=slot, parent=parent)
                elif kind == 'segment':
                    try:
                        budget.job_deadlines('segment', arm_deadline=request['arm_deadline'])
                    except TimeoutError:
                        plan.stop_arm(arm, reason='time')
                        continue
                    job_id = f"curriculum-{arm}-{request['index']:03d}"
                    parameters = dict(request, parent=plan.checkpoints[arm] or original_parent,
                        prefix=plan.prefixes[arm], predecessor=plan.predecessors[arm])
                else:
                    job_id = 'curriculum-qualification'
                    parameters = dict(kind=kind, parent=original_parent)
                result = dispatch_job(root, host, verified, budget, deps=deps, docker=docker,
                    client=client, job_id=job_id, **parameters)
                results.append(dict(job_id=job_id, receipt=result['receipt_binding'], checkpoint=result['checkpoint'],
                                    host_elapsed_seconds=result['host_elapsed_seconds']))
                if kind == 'qualification':
                    plan.accept_qualification()
                elif kind == 'segment':
                    plan.accept_segment(arm, result)
                else:
                    complete = not result['receipt']['missing_case_ids'] and result['receipt']['summary']['status'] == 'complete'
                    plan.finish_panel(slot, status='complete' if complete else 'incomplete', reason='actual CPU panel receipt',
                        result=result['receipt_binding'])
                _write(host/'progress'/f'{len(results):04d}.json', dict(jobs=results, availability=plan.availability,
                    prefixes=plan.prefixes, effects=budget.attempted, reserved=budget.reserved,
                    evaluation_seconds=budget.evaluation_seconds, training_seconds=budget.training_seconds,
                    qualification_seconds=budget.qualification_seconds, elapsed_seconds=deps.clock()-start))
    except BaseException as error:
        primary, status = error, 'failed'
    seal = dict(schema='spire-learn-curriculum-campaign-result-v1', campaign_id=package_api.CAMPAIGN_ID,
        status=status, error=None if primary is None else low._error(primary), jobs=results,
        availability=plan.availability, prefixes=plan.prefixes, stopped_arms=plan.stopped,
        effects=budget.attempted, charged_or_reserved=budget.reserved, attempts=budget.jobs,
        evaluation_seconds=budget.evaluation_seconds, training_seconds=budget.training_seconds,
        qualification_seconds=budget.qualification_seconds, elapsed_seconds=deps.clock()-start,
        parent_start=start, whole_deadline=deadline, source_sha256=claim['source_sha256'],
        release_sha256=claim['release_sha256'], qualifications=verified['package']['campaign']['qualifications'])
    try:
        storage_admission(root, host, deps)
        _write(host/'campaign-result.json', seal)
    except BaseException as error:
        primary = low._promote(primary, error)
    if primary is not None:
        raise primary
    return seal


def supervise(command, *, env, deadline, clock=time.monotonic):
    """Wait for an owned timeout process; repeated parent controls cannot skip it."""
    process, first, sent = None, None, False
    def forward(number, frame):
        nonlocal first, sent
        if first is None:
            first = number
        if process is not None and not sent:
            sent = True
            try:
                os.killpg(process.pid, first)
            except ProcessLookupError:
                pass
    previous = {number: signal.signal(number, forward) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        if first is not None:
            forward(first, None)
        try:
            code = process.wait(timeout=max(.1, deadline-clock()-5.))
        except subprocess.TimeoutExpired:
            # Only this freshly spawned process group is controlled. GNU
            # timeout normally completes its independent KILL before this.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(timeout=max(.1, deadline-clock()-1.))
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    result = subprocess.CompletedProcess(command, code)
    result.parent_control = first
    return result


def launch_guard(arguments, *, result_path, clock=time.monotonic, run=None, parent_start=None):
    """Independent supervisor retains the child exit after TERM/KILL completion."""
    start = _number(clock() if parent_start is None else parent_start, 'launcher start')
    if start > clock():
        raise ValueError('launcher start cannot renew time in the future')
    deadline = start+36000.
    env = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
    env.update(SPIRE_LEARN_CURRICULUM_PARENT_START=str(start), SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE=str(deadline))
    if not Path('/usr/bin/timeout').is_file():
        raise ValueError('GNU timeout is required')
    term_seconds = max(1, math.floor(start+35900.-clock()))
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', str(term_seconds)+'s', sys.executable,
        '-B', str(Path(__file__).resolve()), *arguments]
    # Reserve the outer record pathname before dispatch; never overwrite evidence.
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open('x') as stream:
        result, primary = None, None
        # File reservation/setup consumes this same parent allowance.
        term_seconds = max(1, math.floor(start+35900.-clock()))
        command[3] = str(term_seconds)+'s'
        try:
            result = (supervise(command, env=env, deadline=deadline, clock=clock) if run is None
                else run(command, env=env, check=False, timeout=max(1., deadline-clock()-5.)))
        except BaseException as error:
            primary = error
        record = dict(schema='spire-learn-curriculum-launcher-result-v1', parent_start=start,
            whole_deadline=deadline, child_exit_code=None if result is None else result.returncode,
            error=None if primary is None else low._error(primary), elapsed_seconds=clock()-start,
            command=command, term_after_seconds=term_seconds, kill_grace_seconds=90,
            parent_control=getattr(result, 'parent_control', None),
            measurement_boundary='actual child exit before parent result publication; final 10 seconds reserved')
        try:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        except BaseException as retention:
            primary = low._promote(primary, retention)
        if primary is not None:
            raise primary
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('capture', 'release-template', 'validate', 'run', '_launch'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--package', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--user-approval', type=Path)
    parser.add_argument('--head')
    parser.add_argument('--dirty-json', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    def required(*names):
        if any(getattr(args, name) is None for name in names):
            parser.error('required: '+', '.join('--'+name.replace('_', '-') for name in names))
    def path(value):
        return low._safe(root, value)
    if args.command == 'capture':
        required('destination')
        from scripts.neural.gpu_runtime_proof_helpers import _git_state
        actual_head, actual_dirty = _git_state(root, timeout=10.)
        value = capture_source(root, path(args.destination), learner_head=args.head or actual_head,
            dirty=_read(path(args.dirty_json)) if args.dirty_json else actual_dirty)
    elif args.command == 'release-template':
        required('package', 'source', 'destination', 'user_approval')
        value = release_template(root, package_path=path(args.package), source_snapshot=path(args.source),
            user_approval=path(args.user_approval))
        _write(path(args.destination), value)
    elif args.command == 'validate':
        required('package', 'source', 'release')
        verified = validate_release(root, package_path=path(args.package), source_snapshot=path(args.source),
            release_path=path(args.release))
        value = dict(status='validated', campaign_id=package_api.CAMPAIGN_ID,
            source_sha256=package_api.sha256(path(args.source)), package_sha256=package_api.sha256(path(args.package)),
            release_sha256=package_api.sha256(path(args.release)), full_package_shards=264)
    elif args.command == 'run':
        required('package', 'source', 'release')
        value = run_campaign(root, package_path=path(args.package), source_snapshot=path(args.source), release_path=path(args.release))
    else:
        required('package', 'source', 'release')
        arguments = ['run', '--root', str(root), '--package', str(path(args.package)),
            '--source', str(path(args.source)), '--release', str(path(args.release))]
        return launch_guard(arguments, result_path=path(args.result or DOMAIN/'launcher-result.json'),
                            parent_start=_ENTRY_MONOTONIC)
    print(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
