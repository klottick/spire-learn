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
from scripts.neural import curriculum_profiles as profiles
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
PREDECESSOR_ARTIFACT_PATHS = {
    'audit': 'artifacts/development/neural-combat/curriculum-campaign-v4/preflight-failure-audit.json',
    'result': 'artifacts/development/neural-combat/curriculum-campaign-v4/runtime/campaign-result.json',
    'attempt': 'artifacts/development/neural-combat/curriculum-campaign-v4/runtime/attempts/curriculum-qualification.json',
    'launcher_result': 'artifacts/development/neural-combat/curriculum-campaign-v4/launcher-result.json',
    'launcher_os_exit': 'artifacts/development/neural-combat/curriculum-campaign-v4/launcher-os-exit-observed.json',
}
PREDECESSOR_AUDIT_HASH_PATHS = {
    'result': 'runtime/campaign-result.json',
    'attempt': 'runtime/attempts/curriculum-qualification.json',
    'launcher_result': 'launcher-result.json',
    'launcher_os_exit': 'launcher-os-exit-observed.json',
}
FAILED_QUALIFICATION_LEASE = dict(environment_attempts=512, sample_rows=512,
    optimizer_attempts=8, reset_slots=256, fixed_random_choices=0)
ADDITIONAL_PREDECESSOR_B_ARTIFACT_PATHS = {
    'audit': 'artifacts/development/neural-combat/curriculum-campaign-v5/evaluation-failure-audit.json',
    'result': 'artifacts/development/neural-combat/curriculum-campaign-v5/runtime/campaign-result.json',
    'qualification_attempt': 'artifacts/development/neural-combat/curriculum-campaign-v5/runtime/attempts/curriculum-qualification.json',
    'qualification_receipt': 'artifacts/development/neural-combat/curriculum-campaign-v5/runtime/jobs/curriculum-qualification/output/job-receipt.json',
    'panel_attempt': 'artifacts/development/neural-combat/curriculum-campaign-v5/runtime/attempts/curriculum-panel-fresh-initial.json',
    'panel_receipt': 'artifacts/development/neural-combat/curriculum-campaign-v5/runtime/jobs/curriculum-panel-fresh-initial/output/job-receipt.json',
    'launcher_result': 'artifacts/development/neural-combat/curriculum-campaign-v5/launcher-result.json',
    'launcher_os_exit': 'artifacts/development/neural-combat/curriculum-campaign-v5/launcher-os-exit-observed.json',
}
ADDITIONAL_PREDECESSOR_B_AUDIT_HASH_PATHS = {
    'runtime/campaign-result.json': 'result',
    'runtime/attempts/curriculum-qualification.json': 'qualification_attempt',
    'runtime/jobs/curriculum-qualification/output/job-receipt.json': 'qualification_receipt',
    'runtime/attempts/curriculum-panel-fresh-initial.json': 'panel_attempt',
    'runtime/jobs/curriculum-panel-fresh-initial/output/job-receipt.json': 'panel_receipt',
    'launcher-result.json': 'launcher_result',
    'launcher-os-exit-observed.json': 'launcher_os_exit',
}
FAILED_B_QUALIFICATION_LEASE = dict(environment_attempts=512, sample_rows=512,
    optimizer_attempts=8, reset_slots=256, fixed_random_choices=0)
FAILED_B_QUALIFICATION_EFFECTS = dict(environment_attempts=128, sample_rows=128,
    optimizer_attempts=8, reset_slots=10, fixed_random_choices=0)
FAILED_B_PANEL_LEASE = dict(environment_attempts=16384, sample_rows=0,
    optimizer_attempts=0, reset_slots=64, fixed_random_choices=0)
FAILED_B_CHARGED_OR_RESERVED = dict(environment_attempts=16512, sample_rows=128,
    optimizer_attempts=8, reset_slots=74, fixed_random_choices=0)
FAILED_B_RESERVED_TIME_SECONDS = {'qualification': 600, 'panel': 600}
FAILED_B_PANEL_CASE = 'dev-7c462a6b7d57517290f62cd156a688d88414c3a2e9868724'
FAILED_B_CONTRACT_MESSAGE = 'learned choice statistics/version disagree with the requested policy'
FAILED_B_CAUSE = ('evaluate_policy uses _runtime_policy_version(runtime) or reference-policy-v1; '
    'valid integer0 becomes reference string, conflicting with learned choice behavior version0')
FAILED_PRECHECK_STAGE = 'verify_guest_context before admitted context/held qualification/env construction or Torch workload'
FAILED_PRECHECK_CAUSE = 'Learn guest verifier still compares against legacy profiles.NATIVE_SHA256 rather than selected profile native hash'
TWENTY_HOUR_LIFETIME_CAPS = dict(environment_attempts=4899328, sample_rows=4719104,
    optimizer_attempts=294920, reset_slots=4737984, fixed_random_choices=0)
TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS = {
    'campaign_release_approved': 'artifacts/development/neural-combat/twenty-hour-20261103/campaign-release-approved.json',
    'campaign_claim': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/campaign-claim.json',
    'campaign_result': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/campaign-result.json',
    'seal_failure': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/campaign-seal-failure.json',
    'launcher_result': 'artifacts/development/neural-combat/twenty-hour-20261103/launcher-result.json',
    'launcher_supervision': 'artifacts/development/neural-combat/twenty-hour-20261103/launcher-result.json.supervision.json',
    'owner_exit': 'artifacts/development/neural-combat/twenty-hour-20261103/launcher-os-exit-observed.json',
    'progress_0084': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/progress/0084.json',
    'shard_337_attempt': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/attempts/curriculum-mixed-337.json',
    'shard_338_attempt': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/attempts/curriculum-mixed-338.json',
    'shard_338_claim': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/job-claims/curriculum-mixed-338.json',
    'shard_337_receipt': 'artifacts/development/neural-combat/twenty-hour-20261103/runtime/jobs/curriculum-mixed-337/output/job-receipt.json',
    'shard_337_checkpoint_metadata': ('artifacts/development/neural-combat/twenty-hour-20261103/runtime/jobs/'
        'curriculum-mixed-337/output/20261103T061627Z-c3c9a1f87bc548a98c6c368999b29c91/checkpoint-final/metadata.json'),
    'restart_authority': 'artifacts/development/neural-combat/twenty-hour-20261103/race-fix/restart-authority.json',
}
TWENTY_HOUR_RESTART_AUTHORITY = TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['restart_authority']
_control_state = ContextVar('curriculum_first_control', default=None)
_launcher_state = ContextVar('curriculum_launcher_control', default=None)


def _extra_source(root):
    return tuple(sorted(set(EXTRA_SOURCE) | set(profiles.source_paths(root))))


@contextmanager
def campaign_signal_scope():
    """Reuse owned cleanup deferral, retaining the actual first signal object."""
    with transport.signal_scope():
        deferred = transport._signals.get()
        state = {'first': None, 'sealing': False}
        token = _control_state.set(state)
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        def handler(number, frame):
            if state['sealing']:
                if state['first'] is None:
                    state['first'] = low._Termination('signal '+str(number))
                return  # Finish the owned seal/veto write before unwinding.
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
            if state['first'] is not None:
                raise state['first']
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
    names = sorted(set(producer.source_paths(root)) | set(_extra_source(root)))
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
    result = dict(image=low.IMAGE, image_id=low.IMAGE_ID, gpu_uuid=low.GPU_UUID,
        python_sha256=PYTHON_SHA256,
        lock_sha256=low.LOCK_SHA256, native_sha256=low.NATIVE_SHA256,
        profile_sha256=package_api.sha256(root/cdi.PROFILE_PATH),
        qualification_sha256=package_api.sha256(qualification), versions=PIN_VERSIONS,
        threads={'intraop': 4, 'interop': 1}, cpu=4, memory_bytes=8*GIB,
        qualification_pids=128, campaign_pids=512)
    if profiles.current().fixed:
        overlay = profiles.overlay_pins(root)
        profiles.verify_overlay(root/overlay['package_overlay']['path'], overlay['package_overlay']['inventory'])
        result.update(overlay)
    return result


def _predecessor_document(root):
    selected = profiles.current()
    if selected.predecessor_path is None:
        return None
    document = package_api.read_json(Path(root)/selected.predecessor_path)
    if not isinstance(document, Mapping):
        raise ValueError('source-bound campaign predecessor differs')
    retained_lease = document.get('retained_lease')
    if (set(document) != {'schema', 'campaign_id', 'profile', 'artifacts', 'retained_lease', 'debit_reason'}
            or document['schema'] != 'spire-learn-curriculum-predecessor-v1'
            or document['campaign_id'] != profiles.FRESH_GYM.campaign_id
            or document['profile'] != profiles.FRESH_GYM.name
            or not isinstance(retained_lease, Mapping) or set(retained_lease) != set(CAPS)
            or any(type(value) is not int or value < 0 for value in retained_lease.values())
            or retained_lease != FAILED_QUALIFICATION_LEASE
            or not isinstance(document['debit_reason'], str) or not document['debit_reason']):
        raise ValueError('source-bound campaign predecessor differs')
    artifacts = document['artifacts']
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(PREDECESSOR_ARTIFACT_PATHS):
        raise ValueError('campaign predecessor artifact set differs')
    for name, expected_path in PREDECESSOR_ARTIFACT_PATHS.items():
        binding = artifacts[name]
        if (not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}
                or binding['path'] != expected_path
                or not isinstance(binding['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', binding['sha256']) is None):
            raise ValueError('campaign predecessor artifact binding differs: '+name)
    return document


def _additional_predecessor_document(root):
    selected = profiles.current()
    if selected.additional_predecessor_path is None:
        return None
    document = package_api.read_json(Path(root)/selected.additional_predecessor_path)
    charged = document.get('charged_or_reserved') if isinstance(document, Mapping) else None
    if (not isinstance(document, Mapping)
            or set(document) != {'schema', 'campaign_id', 'profile', 'artifacts',
                                 'charged_or_reserved', 'reserved_time_seconds', 'debit_reason'}
            or document['schema'] != 'spire-learn-curriculum-additional-predecessor-v1'
            or document['campaign_id'] != profiles.FRESH_GYM_B.campaign_id
            or document['profile'] != profiles.FRESH_GYM_B.name
            or not isinstance(charged, Mapping) or set(charged) != set(CAPS)
            or any(type(value) is not int or value < 0 for value in charged.values())
            or charged != FAILED_B_CHARGED_OR_RESERVED
            or document.get('reserved_time_seconds') != FAILED_B_RESERVED_TIME_SECONDS
            or not isinstance(document['debit_reason'], str) or not document['debit_reason']):
        raise ValueError('additional campaign predecessor differs')
    artifacts = document['artifacts']
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ADDITIONAL_PREDECESSOR_B_ARTIFACT_PATHS):
        raise ValueError('additional campaign predecessor artifact set differs')
    for name, expected_path in ADDITIONAL_PREDECESSOR_B_ARTIFACT_PATHS.items():
        binding = artifacts[name]
        if (not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}
                or binding['path'] != expected_path
                or not isinstance(binding['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', binding['sha256']) is None):
            raise ValueError('additional campaign predecessor artifact binding differs: '+name)
    return document


def _validate_corrected_continuation_authority(root, predecessor):
    """Bind the one corrected zero-job launch to its retained failure and parent."""
    if not profiles.is_corrected_twenty_hour_continuation():
        raise ValueError('corrected continuation authority is outside its profile')
    root = Path(root).resolve(strict=True)
    path = profiles.current().user_approval_path
    if not isinstance(path, str) or not path:
        raise ValueError('corrected continuation authority path is absent')
    authority = package_api.read_json(root/path, maximum=65536)
    expected_bindings = {
        'launch_session': ('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1/launch-session.json',
            '76b78bafadae568863fdf33bf2d2fb3563f27ef3041e6cc793907289aa598372'),
        'launcher_result': ('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1/launcher-result.json',
            '602438da6caaba6ad49084fedc8da862d20ecad726e1e55b7bdc3c2c04ddb19a'),
        'launcher_supervision': ('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1/launcher-result.json.supervision.json',
            '6b5e457fe4cac7d645c116ce35d18c9c9845de146482778dc1414bd2dcfef557'),
        'owner_exit': ('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1/launcher-os-exit-observed.json',
            'df14ffa075552d4ae0e73c4effb260c9687e97eda12474fbcb24b59f3bb99314'),
    }
    expected_parent = {
        'campaign_id': profiles.TWENTY_HOUR.campaign_id,
        'healthy_shard': 337,
        'receipt_sha256': 'b2afb649a3887bbcc101c397831cbe6b5c0e639476c377cbace20f4f1e333fd0',
        'checkpoint_inventory_sha256': '071a05b8bdec51f38b9f7591dec553a0d7f5547cc2dd2131094d678fe617054e',
    }
    lease = dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
        reset_slots=4112, fixed_random_choices=0)
    expected_user_authority = {
        'instruction': 'Fix that, then start more training',
        'timing_steering': 'it can take 24 hours for all I care, timing is loose and the gpu is free',
        'one_corrected_launch': True,
        'preserve_all_failed_history': True,
        'no_accepted_effect_retry': True,
        'source_message_ids': [
            '8a408503-5700-4bf2-aa90-15f613e34ec9',
            'ccacdc24-cc28-4e29-ac2d-b6028049a4e8',
            '7215e592-d6dc-40ee-a3a3-ac159d8566a4',
        ],
    }
    expected_prior = {
        'campaign_id': profiles.TWENTY_HOUR_CONTINUATION.campaign_id,
        'owner_exec_session_id': 19939,
        'write_stdin_exit_code': 1,
        'launcher_elapsed_seconds': 304.97056149004493,
        'watchdog_exit_status': 1,
        'campaign_claim_published': False,
        'job_dispatched': False,
        'artifact_bindings': {
            key: {'path': relative, 'sha256': digest}
            for key, (relative, digest) in expected_bindings.items()
        },
    }
    expected_skip = dict(shard_index=338, actual=None, receipt_binding=None,
        lease=lease, first_new_shard_index=339)
    expected_limits = dict(maximum_corrected_campaign_seconds=86400,
        selected_whole_seconds=profiles.current().whole_seconds,
        training_stop_second=profiles.current().training_stop_second,
        no_new_effect_lease_for_zero_job_attempt=True)
    if (set(authority) != {'schema', 'campaign_id', 'failed_predecessor', 'prior_zero_job_attempt',
            'parent', 'skip', 'limits', 'user_authority'}
            or authority.get('schema') != 'spire-learn-curriculum-corrected-continuation-authority-v1'
            or authority.get('campaign_id') != profiles.current().campaign_id
            or authority.get('failed_predecessor') != profiles.TWENTY_HOUR.campaign_id
            or authority.get('prior_zero_job_attempt') != expected_prior
            or authority.get('parent') != expected_parent
            or authority.get('skip') != expected_skip
            or authority.get('limits') != expected_limits
            or authority.get('user_authority') != expected_user_authority
            or profiles.current().whole_seconds > authority['limits']['maximum_corrected_campaign_seconds']
            or predecessor.get('healthy_parent', {}).get('index') != 337
            or predecessor.get('failed_shard', {}).get('index') != 338
            or predecessor.get('failed_shard', {}).get('lease') != lease):
        raise ValueError('corrected continuation authority scope differs')
    for key, (relative, digest) in expected_bindings.items():
        if package_api.sha256(package_api.relative_file(root, relative)) != digest:
            raise ValueError('corrected continuation prior-attempt binding changed: '+key)
    prior_domain = root/profiles.TWENTY_HOUR_CONTINUATION.domain
    for relative in ('runtime/campaign-claim.json', 'runtime/campaign-result.json',
                     'runtime/campaign-seal-failure.json', 'runtime/job-claims'):
        if (prior_domain/relative).exists():
            raise ValueError('corrected continuation prior attempt published campaign/job effects')
    if (prior_domain/'runtime/progress').exists():
        raise ValueError('corrected continuation prior attempt published campaign progress')
    launch_session = package_api.read_json(root/expected_bindings['launch_session'][0])
    launcher = package_api.read_json(root/expected_bindings['launcher_result'][0])
    supervision = package_api.read_json(root/expected_bindings['launcher_supervision'][0])
    owner_exit = package_api.read_json(root/expected_bindings['owner_exit'][0])
    if (launch_session.get('campaign_id') != expected_prior['campaign_id']
            or launch_session.get('owner_exec_session_id') != 19939
            or launch_session.get('launcher_pid') != supervision.get('launcher_pid')
            or launch_session.get('launcher_start_ticks') != supervision.get('launcher_start_ticks')
            or launcher.get('watchdog_exit_status') != 1
            or launcher.get('elapsed_seconds') != expected_prior['launcher_elapsed_seconds']
            or launcher.get('whole_run_complete') is not False
            or supervision.get('result_path') != str(root/expected_bindings['launcher_result'][0])
            or supervision.get('parent_start') != launcher.get('parent_start')
            or supervision.get('whole_deadline') != launcher.get('whole_deadline')
            or owner_exit.get('owner_exec_session_id') != 19939
            or owner_exit.get('write_stdin_exit_code') != 1):
        raise ValueError('corrected continuation prior startup records disagree')
    return authority


def _twenty_hour_continuation_document(root):
    """Validate staged source-bound debit data without opening host history."""
    if not profiles.is_twenty_hour_continuation():
        raise ValueError('twenty-hour continuation predecessor is outside its profile')
    root = Path(root).resolve(strict=True)
    document = package_api.read_json(root/profiles.current().predecessor_path)
    if (not isinstance(document, Mapping)
            or set(document) != {'schema', 'campaign_id', 'profile', 'source_sha256',
                'package_sha256', 'release_sha256', 'original_whole_seconds',
                'actual_launcher_elapsed_seconds', 'remaining_whole_seconds',
                'settled_effects', 'charged_or_reserved', 'artifacts', 'failed_shard', 'healthy_parent'}
            or document['schema'] != 'spire-learn-curriculum-twenty-hour-predecessor-v1'
            or document['campaign_id'] != profiles.TWENTY_HOUR.campaign_id
            or document['profile'] != profiles.TWENTY_HOUR.name
            or document['source_sha256'] != 'd363f588bdafb4f2013828c149edeaff819eb422f6f48a6ff34493a257f7eb3a'
            or document['package_sha256'] != '72dfb1fda0233de0fb867ab5b9dc0c869f6ebf6166b0585b43126fd0a17600d4'
            or document['release_sha256'] != '59e070ccf2b06bc7c9eed7520fe25830a3e3994134d505352e3a321f02ef6857'
            or document['original_whole_seconds'] != 72000
            or document['actual_launcher_elapsed_seconds'] != 6001.25571680197
            or document['remaining_whole_seconds'] != 65998
            or math.floor(document['original_whole_seconds']-document['actual_launcher_elapsed_seconds'])
                != document['remaining_whole_seconds']):
        raise ValueError('twenty-hour continuation predecessor identity/time debit differs')
    expected_settled = dict(environment_attempts=337471, sample_rows=336000,
        optimizer_attempts=21000, reset_slots=15431, fixed_random_choices=0)
    expected_charged = dict(environment_attempts=341567, sample_rows=340096,
        optimizer_attempts=21256, reset_slots=19543, fixed_random_choices=0)
    if (document['settled_effects'] != expected_settled
            or document['charged_or_reserved'] != expected_charged
            or not isinstance(document['artifacts'], Mapping)
            or set(document['artifacts']) != set(TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS)):
        raise ValueError('twenty-hour continuation predecessor debit/evidence set differs')
    for name, expected_path in TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS.items():
        binding = document['artifacts'][name]
        if (not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}
                or binding['path'] != expected_path
                or re.fullmatch('[0-9a-f]{64}', str(binding['sha256'])) is None):
            raise ValueError('twenty-hour predecessor artifact binding differs: '+name)
    lease = dict(environment_attempts=4096, sample_rows=4096,
        optimizer_attempts=256, reset_slots=4112, fixed_random_choices=0)
    if (document['failed_shard'] != dict(job_id='curriculum-mixed-338', index=338,
            actual=None, receipt_binding=None, lease=lease)
            or document.get('healthy_parent', {}).get('index') != 337):
        raise ValueError('twenty-hour predecessor healthy parent/full unknown lease differs')
    return document


TWENTY_HOUR_FAILED_ADMISSION_SHA256 = '8aa8550d679c198b4a5dcb61badabf5cb75c21dc53e751884e5ad7d0c01fcc18'


def _twenty_hour_failed_admission_document(root):
    """Closed staged debit for continuation-2; actual effects remain unknown."""
    if not profiles.is_repaired_twenty_hour_continuation():
        raise ValueError('failed admission debit is outside its profile')
    path = Path(root)/profiles.current().additional_predecessor_path
    document = package_api.read_json(path)
    lease = dict(environment_attempts=512, sample_rows=512, optimizer_attempts=8,
        reset_slots=256, fixed_random_choices=0)
    if (package_api.sha256(path) != TWENTY_HOUR_FAILED_ADMISSION_SHA256
            or document.get('schema') != 'spire-learn-curriculum-failed-admission-predecessor-v1'
            or document.get('campaign_id') != profiles.TWENTY_HOUR_CONTINUATION_2.campaign_id
            or document.get('charged_or_reserved') != lease
            or document.get('settled_effects') != dict.fromkeys(CAPS, 0)
            or document.get('failed_job') != dict(job_id='curriculum-qualification',
                actual=None, receipt_binding=None, lease=lease)):
        raise ValueError('failed admission source-bound full lease debit differs')
    return document


def _validate_twenty_hour_failed_admission(root):
    """Host-only retained failed-attempt identity, terminal state and cleanup."""
    root = Path(root).resolve(strict=True)
    document = _twenty_hour_failed_admission_document(root)
    artifacts = document['artifacts']
    for name, binding in artifacts.items():
        if package_api.sha256(package_api.relative_file(root, binding['path'])) != binding['sha256']:
            raise ValueError('failed admission raw artifact changed: '+name)
    def read(name):
        return _read(root/artifacts[name]['path'])
    claim, job_claim, attempt = read('campaign_claim'), read('job_claim'), read('attempt')
    result, failure, launcher = read('campaign_result'), read('seal_failure'), read('launcher_result')
    supervision, session, owner = read('launcher_supervision'), read('launch_session'), read('owner_exit')
    release = read('release')
    lease = document['charged_or_reserved']
    identities = dict(campaign_id=document['campaign_id'], source_sha256=document['source_sha256'],
        package_sha256=document['package_sha256'], release_sha256=document['release_sha256'])
    if (any(claim.get(key) != value or job_claim.get(key) != value for key, value in identities.items())
            or claim.get('schema') != CLAIM_SCHEMA or job_claim.get('schema') != CLAIM_SCHEMA
            or job_claim.get('campaign_claim_sha256') != artifacts['campaign_claim']['sha256']
            or job_claim.get('job_id') != 'curriculum-qualification'
            or job_claim.get('kind') != 'qualification' or job_claim.get('reservation') != lease
            or attempt.get('reservation') != dict(state='reserved', lease=lease)
            or attempt.get('receipt_binding') is not None
            or release.get('state') != 'approved' or release.get('campaign_id') != document['campaign_id']
            or release.get('source_snapshot_sha256') != document['source_sha256']
            or release.get('package_sha256') != document['package_sha256']
            or release.get('runtime_pins') != {**runtime_pins(root), 'profile': document['profile']}
            or result.get('campaign_id') != document['campaign_id']
            or result.get('source_sha256') != document['source_sha256']
            or result.get('status') != 'failed' or result.get('whole_run_claim') is not False
            or result.get('effects') != document['settled_effects']
            or result.get('charged_or_reserved') != lease
            or len(result.get('jobs', [])) != 1
            or result['jobs'][0].get('job_id') != 'curriculum-qualification'
            or result['jobs'][0].get('receipt_reconciled') is not False
            or failure.get('status') != 'failed' or failure.get('whole_run_claim') is not False
            or launcher.get('watchdog_exit_status') != 1 or launcher.get('whole_run_complete') is not False
            or launcher.get('controller_logical_result', {}).get('finalization_veto') is not True
            or launcher.get('controller_logical_result', {}).get('sha256') != artifacts['campaign_result']['sha256']
            or supervision.get('parent_start') != claim.get('parent_start')
            or supervision.get('whole_deadline') != claim.get('whole_deadline')
            or claim.get('supervision', {}).get('binding_sha256') != artifacts['launcher_supervision']['sha256']
            or session.get('owner_exec_session_id') != 38079 or owner.get('owner_exec_session_id') != 38079
            or owner.get('write_stdin_exit_code') != 1
            or owner.get('launcher_pid') != session.get('launcher_pid')
            or owner.get('launcher_start_ticks') != session.get('launcher_start_ticks')):
        raise ValueError('failed admission terminal/identity/full lease evidence differs')
    _verify_retained_container_cleanup(attempt, 'curriculum-qualification',
        expected_status='failed', expected_receipt=None)
    prior = root/profiles.TWENTY_HOUR_CONTINUATION_2.domain/'runtime'
    if ((prior/'progress').exists()
            or (prior/'jobs/curriculum-qualification/output/job-receipt.json').exists()
            or {p.name for p in (prior/'job-claims').iterdir()} != {'curriculum-qualification.json'}):
        raise ValueError('failed admission unexpectedly published progress/receipt or another claim')
    return document


def _validate_repaired_continuation_authority(root):
    """Authenticate the recorded standing user instruction and execution scope."""
    if not profiles.is_repaired_twenty_hour_continuation():
        raise ValueError('repaired continuation authority is outside its profile')
    path = Path(root)/profiles.current().user_approval_path
    authority = package_api.read_json(path, maximum=65536)
    if (package_api.sha256(path) != producer.REPAIRED_CONTINUATION_AUTHORITY_SHA256
            or authority.get('schema') != 'spire-learn-curriculum-repaired-continuation-authority-v1'
            or authority.get('campaign_id') != profiles.current().campaign_id
            or authority.get('failed_admission', {}).get('predecessor_sha256') != TWENTY_HOUR_FAILED_ADMISSION_SHA256
            or profiles.current().whole_seconds > authority['limits']['maximum_campaign_seconds']):
        raise ValueError('repaired continuation standing authority differs')
    return authority


def _twenty_hour_continuation_predecessor(root):
    """Host-only physical history, exact337 checkpoint, cleanup and authority."""
    root = Path(root).resolve(strict=True)
    document = _twenty_hour_continuation_document(root)
    expected_settled = document['settled_effects']
    expected_charged = document['charged_or_reserved']
    for name, expected_path in TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS.items():
        if package_api.sha256(package_api.relative_file(root, expected_path)) != document['artifacts'][name]['sha256']:
            raise ValueError('twenty-hour predecessor raw artifact changed: '+name)

    claim = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['campaign_claim'])
    prior_release = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['campaign_release_approved'])
    result = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['campaign_result'])
    failure = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['seal_failure'])
    launcher = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['launcher_result'])
    supervision = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['launcher_supervision'])
    owner_exit = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['owner_exit'])
    progress = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['progress_0084'])
    parent_attempt = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['shard_337_attempt'])
    attempt = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['shard_338_attempt'])
    job_claim = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['shard_338_claim'])
    receipt = package_api.read_json(root/TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['shard_337_receipt'],
        maximum=32*1024**2)
    authority = package_api.read_json(root/TWENTY_HOUR_RESTART_AUTHORITY)
    healthy, failed = document['healthy_parent'], document['failed_shard']
    lease = dict(environment_attempts=4096, sample_rows=4096,
        optimizer_attempts=256, reset_slots=4112, fixed_random_choices=0)
    if (claim.get('campaign_id') != document['campaign_id']
            or claim.get('source_sha256') != document['source_sha256']
            or claim.get('package_sha256') != document['package_sha256']
            or claim.get('release_sha256') != document['release_sha256']
            or prior_release.get('schema') != RELEASE_SCHEMA
            or prior_release.get('state') != 'approved'
            or prior_release.get('campaign_id') != document['campaign_id']
            or prior_release.get('source_snapshot_sha256') != document['source_sha256']
            or prior_release.get('package_sha256') != document['package_sha256']
            or prior_release.get('runtime_pins') != {
                **runtime_pins(root), 'profile': profiles.TWENTY_HOUR.name}
            or result.get('status') != 'failed' or result.get('whole_run_claim') is not False
            or result.get('campaign_id') != document['campaign_id']
            or result.get('source_sha256') != document['source_sha256']
            or result.get('effects') != expected_settled
            or result.get('charged_or_reserved') != expected_charged
            or len(result.get('jobs', [])) != 85
            or result['jobs'][-1].get('job_id') != 'curriculum-mixed-338'
            or result['jobs'][-1].get('receipt_reconciled') is not False
            or failure.get('status') != 'failed' or failure.get('whole_run_claim') is not False
            or launcher.get('watchdog_exit_status') != 1 or launcher.get('whole_run_complete') is not False
            or launcher.get('elapsed_seconds') != document['actual_launcher_elapsed_seconds']
            or launcher.get('controller_logical_result', {}).get('finalization_veto') is not True
            or launcher.get('controller_logical_result', {}).get('sha256')
                != document['artifacts']['campaign_result']['sha256']
            or supervision.get('parent_start') != claim.get('parent_start')
            or supervision.get('whole_deadline') != claim.get('whole_deadline')
            or owner_exit.get('campaign_id') != document['campaign_id']
            or owner_exit.get('owner_exec_session_id') != 91592
            or owner_exit.get('write_stdin_exit_code') != 1
            or attempt.get('job_id') != 'curriculum-mixed-338'
            or attempt.get('status') != 'failed'
            or attempt.get('reservation') != {'state': 'reserved', 'lease': lease}
            or attempt.get('actual') is not None or attempt.get('receipt_binding') is not None
            or job_claim.get('campaign_id') != document['campaign_id']
            or job_claim.get('job_id') != 'curriculum-mixed-338'
            or job_claim.get('reservation') != lease
            or failed != {'job_id': 'curriculum-mixed-338', 'index': 338,
                'actual': None, 'receipt_binding': None, 'lease': lease}
            or authority.get('failed_predecessor') != document['campaign_id']
            or authority.get('healthy_parent_shard') != 337
            or authority.get('unreconciled_shard_skipped') != 338
            or authority.get('first_new_shard_minimum') != 339
            or authority.get('new_whole_seconds_upper_bound') != 65998
            or authority.get('failed_effect_lease_retained') != lease):
        raise ValueError('twenty-hour predecessor terminal/unknown-effect evidence differs')
    receipt_binding = document['artifacts']['shard_337_receipt']
    _verify_retained_container_cleanup(parent_attempt, 'curriculum-mixed-337',
        expected_status='complete', expected_receipt={'path': str(root/receipt_binding['path']),
            'sha256': receipt_binding['sha256']})
    _verify_retained_container_cleanup(attempt, 'curriculum-mixed-338',
        expected_status='failed', expected_receipt=None)
    if (receipt.get('status') != 'complete' or receipt.get('campaign_id') != document['campaign_id']
            or receipt.get('source_sha256') != document['source_sha256']
            or receipt.get('package_sha256') != document['package_sha256']
            or receipt.get('job_id') != 'curriculum-mixed-337'
            or receipt.get('shard_index') != 337
            or receipt.get('next_shard') != {'shard_index': 338,
                'manifest_sha256': healthy.get('next_manifest_sha256')}
            or receipt.get('report', {}).get('status') != 'complete'
            or receipt.get('report', {}).get('fault') is not None
            or receipt.get('checkpoint', {}).get('counters') != healthy.get('checkpoint_counters')
            or receipt.get('checkpoint', {}).get('inventory') != healthy.get('checkpoint_inventory')):
        raise ValueError('healthy shard337 retained receipt differs')
    tail = progress.get('jobs')
    if (not isinstance(tail, list) or len(tail) != 84
            or tail[-1].get('job_id') != 'curriculum-mixed-337'
            or tail[-1].get('receipt', {}).get('sha256')
                != document['artifacts']['shard_337_receipt']['sha256']
            or tail[-1].get('checkpoint', {}).get('inventory') != healthy.get('checkpoint_inventory')):
        raise ValueError('healthy shard337 completed progress binding differs')
    checkpoint_path = package_api.relative_file(root, healthy['checkpoint_path'])
    from scripts.neural.sustained_runtime import checkpoint_inventory
    if checkpoint_inventory(checkpoint_path) != healthy.get('checkpoint_inventory'):
        raise ValueError('healthy shard337 checkpoint inventory changed')
    metadata = package_api.read_json(checkpoint_path/'metadata.json', maximum=2*1024**2)
    stream = metadata.get('run_metadata', {}).get('scenario_stream', {})
    training_state = metadata.get('training_state', {})
    if (metadata.get('schema_id') != 'spire-learn-neural-checkpoint-v1'
            or stream.get('current', {}).get('shard_index') != 337
            or stream.get('next', {}).get('shard_index') != 338
            or training_state.get('counters') != healthy.get('checkpoint_counters')
            or training_state.get('optimizer_step') != 20992
            or training_state.get('rng_domains') != ['policy', 'minibatch', 'curriculum', 'controller']):
        raise ValueError('healthy shard337 exact resume metadata differs')
    return document


def _verify_retained_container_cleanup(attempt, job_id, *, expected_status, expected_receipt):
    """Verify the recorded full-ID cleanup/absence proof without runtime queries."""
    commands = attempt.get('commands') if isinstance(attempt, Mapping) else None
    if (attempt.get('job_id') != job_id or attempt.get('status') != expected_status
            or not isinstance(commands, list) or not commands):
        raise ValueError('retained predecessor container attempt identity differs')
    if expected_receipt is not None:
        binding = attempt.get('receipt_binding')
        if not isinstance(binding, Mapping) or dict(binding) != expected_receipt:
            raise ValueError('retained healthy-parent attempt receipt differs')
    elif attempt.get('receipt_binding') is not None:
        raise ValueError('unreconciled predecessor attempt unexpectedly has a receipt')
    removals = []
    for position, row in enumerate(commands):
        argv = row.get('argv') if isinstance(row, Mapping) else None
        if (isinstance(argv, list) and len(argv) >= 3
                and argv[-3:-1] == ['rm', '--force']):
            removals.append((position, row, argv[-1]))
    if (len(removals) != 1 or not isinstance(removals[0][2], str)
            or re.fullmatch('[0-9a-f]{64}', removals[0][2]) is None):
        raise ValueError('retained predecessor must remove one exact full container ID')
    position, removal, container_id = removals[0]
    if (removal.get('status') != 'returned' or removal.get('exit_code') != 0
            or removal.get('timed_out') is not False or removal.get('stdout', '').strip() != container_id):
        raise ValueError('retained predecessor full-ID removal did not succeed')
    checks = []
    for row in commands[position+1:]:
        argv = row.get('argv') if isinstance(row, Mapping) else None
        if isinstance(argv, list) and 'ps' in argv and '--filter' in argv:
            filter_index = argv.index('--filter')
            if filter_index+1 < len(argv) and argv[filter_index+1] == 'id='+container_id:
                checks.append(row)
    if (len(checks) != 1 or checks[0].get('status') != 'returned'
            or checks[0].get('exit_code') != 0 or checks[0].get('timed_out') is not False
            or checks[0].get('stdout', '').strip()):
        raise ValueError('retained predecessor exact container absence proof differs')


def _effective_global_effect_caps(root, campaign):
    if profiles.is_twenty_hour_continuation():
        predecessor = _twenty_hour_continuation_document(root)
        training = _counts(campaign.get('training_caps'))
        qualification_raw = campaign.get('qualification', {})
        qualification = _counts({key: qualification_raw.get(key) for key in CAPS})
        evaluation = campaign.get('evaluation', {})
        slots = campaign.get('evaluation_slots')
        milestones = [7200 * index for index in range(1, 9)]
        expected_slots = ['continuation-parent-shard337'] + [
            f'mixed-time-{value}' for value in milestones] + ['final-mixed']
        segment = campaign.get('segment', {})
        lease = predecessor['failed_shard']['lease']
        lifetime = {key: TWENTY_HOUR_LIFETIME_CAPS[key]
            - predecessor['charged_or_reserved'][key] for key in CAPS}
        if profiles.is_repaired_twenty_hour_continuation():
            additional = _twenty_hour_failed_admission_document(root)
            lifetime = {key: lifetime[key]-additional['charged_or_reserved'][key] for key in CAPS}
        expected_training = dict(environment_attempts=335872+1068*4096,
            sample_rows=335872+1068*4096, optimizer_attempts=20992+1068*256,
            reset_slots=15356+1068*4112, fixed_random_choices=0)
        training_delta = dict(environment_attempts=training['environment_attempts']-335872,
            sample_rows=training['sample_rows']-335872,
            optimizer_attempts=training['optimizer_attempts']-20992,
            reset_slots=training['reset_slots']-15356, fixed_random_choices=0)
        qualification_contract = dict(environment_attempts=512, sample_rows=512,
            optimizer_attempts=8, reset_slots=256, fixed_random_choices=0,
            soft_seconds=120, whole_seconds=600, output_bytes=GIB,
            stream_start_index=1410, seeds=dict(policy=20261201, optimizer=20261207,
                curriculum=20261211, noncombat=20261213))
        if profiles.is_repaired_twenty_hour_continuation():
            qualification_contract.update(stream_start_index=1412,
                seeds=dict(policy=20261217, optimizer=20261219, curriculum=20261223, noncombat=20261229))
        storage_contract = dict(package_bytes=512*1024**2,
            new_output_bytes=12*GIB, aggregate_bytes=40*GIB,
            free_bytes=30*GIB, vram_free_bytes=12*GIB)
        panel_count = campaign.get('development_panel', {}).get('case_count')
        if (campaign.get('campaign_id') != profiles.current().campaign_id
                or campaign.get('whole_seconds') != 65998
                or campaign.get('training_stop_second') != 64198
                or campaign.get('training_budgets') != dict(max_training_decisions=4718592,
                    max_environment_actions=4718592, max_optimizer_steps=294912,
                    max_wall_seconds=72000.0)
                or campaign.get('training_start_indices') != {'mixed': 339}
                or campaign.get('training_shards') != {'mixed': 1068}
                or campaign.get('mixture') != dict(arm='mixed', run_like_percent=70,
                    targeted_lesson_percent=20, unusual_percent=10)
                or segment != dict(updates=8, environment_attempts=4096, sample_rows=4096,
                    optimizer_attempts=256, reset_slots=4112, soft_seconds=480, whole_seconds=600)
                or campaign.get('qualification') != qualification_contract
                or evaluation.get('total_seconds') != 6600
                or evaluation.get('case_actions') != 256
                or evaluation.get('case_seconds') != 30
                or evaluation.get('job_soft_seconds') != 480
                or evaluation.get('job_whole_seconds') != 600
                or evaluation.get('milestone_training_seconds') != milestones
                or evaluation.get('final_panel_reserve_seconds') != 600
                or evaluation.get('pre_final_panel_reserve_seconds') != 6000
                or slots != expected_slots or panel_count != 64
                or campaign.get('development_panel', {}).get('repeated_exposure') is not True
                or campaign.get('development_panel', {}).get('protected_final_split') != 'untouched'
                or campaign.get('runtime', {}).get('gpu_uuid') != low.GPU_UUID
                or campaign.get('runtime', {}).get('threads') != 4
                or campaign.get('runtime', {}).get('interop_threads') != 1
                or campaign.get('storage') != storage_contract
                or training != expected_training
                or any(training_delta[key] > lifetime[key] for key in CAPS)
                or 64198*62 + 394252 != training_delta['environment_attempts']
                or 64198 + 600 + 600 + 600 != 65998
                or lifetime['environment_attempts'] < 4538880
                or lifetime['sample_rows'] < 4375040
                or lifetime['optimizer_attempts'] < 273416
                or lifetime['reset_slots'] < 4392512
                or lease != dict(environment_attempts=4096, sample_rows=4096,
                    optimizer_attempts=256, reset_slots=4112, fixed_random_choices=0)):
            raise ValueError('twenty-hour continuation time/effect contract differs')
        expected = dict(environment_attempts=4538880, sample_rows=4375040,
            optimizer_attempts=273416, reset_slots=4392512, fixed_random_choices=0)
        caps = _counts(campaign.get('global_effect_caps'))
        if caps != expected or any(caps[key] > lifetime[key] for key in CAPS):
            raise ValueError('continuation effect caps exceed lifetime residual after full lease debit')
        return caps
    if profiles.is_twenty_hour_campaign():
        training = _counts(campaign.get('training_caps'))
        qualification_raw = campaign.get('qualification', {})
        qualification = _counts({key: qualification_raw.get(key) for key in CAPS})
        slots = campaign.get('evaluation_slots')
        evaluation = campaign.get('evaluation', {})
        panel_count = campaign.get('development_panel', {}).get('case_count')
        milestones = [7200 * index for index in range(1, 10)]
        expected_slots = ['warm-start-baseline'] + [f'mixed-time-{value}' for value in milestones] + ['final-mixed']
        segment = campaign.get('segment', {})
        qualification_contract = dict(environment_attempts=512, sample_rows=512,
            optimizer_attempts=8, reset_slots=256, fixed_random_choices=0,
            soft_seconds=120, whole_seconds=600, output_bytes=GIB,
            stream_start_index=1408)
        storage_contract = dict(package_bytes=512*1024**2,
            new_output_bytes=12*GIB, aggregate_bytes=40*GIB,
            free_bytes=30*GIB, vram_free_bytes=12*GIB)
        if (not isinstance(slots, list) or not slots
                or slots != expected_slots
                or campaign.get('whole_seconds') != profiles.current().whole_seconds
                or campaign.get('training_stop_second') != profiles.current().training_stop_second
                or campaign.get('training_budgets', {}).get('max_training_decisions') != training['sample_rows']
                or campaign.get('training_budgets', {}).get('max_environment_actions') != training['environment_attempts']
                or campaign.get('training_budgets', {}).get('max_optimizer_steps') != training['optimizer_attempts']
                or campaign.get('training_budgets', {}).get('max_wall_seconds') != profiles.current().whole_seconds
                or campaign.get('training_start_indices') != {'mixed': 256}
                or campaign.get('training_shards') != {'mixed': 1152}
                or campaign.get('mixture') != dict(arm='mixed', run_like_percent=70,
                    targeted_lesson_percent=20, unusual_percent=10)
                or segment != dict(updates=8, environment_attempts=4096, sample_rows=4096,
                    optimizer_attempts=256, reset_slots=4112, soft_seconds=480, whole_seconds=600)
                or campaign.get('qualification') != qualification_contract
                or evaluation.get('total_seconds') != 7200
                or evaluation.get('case_actions') != 256
                or evaluation.get('case_seconds') != 30
                or evaluation.get('job_soft_seconds') != 480
                or evaluation.get('job_whole_seconds') != 600
                or evaluation.get('milestone_training_seconds') != milestones
                or panel_count != 64
                or campaign.get('development_panel', {}).get('repeated_exposure') is not True
                or campaign.get('development_panel', {}).get('protected_final_split') != 'untouched'
                or evaluation.get('final_panel_reserve_seconds') != 600
                or evaluation.get('pre_final_panel_reserve_seconds') != 6000):
            raise ValueError('twenty-hour campaign time/effect contract differs')
        if (campaign.get('storage') != storage_contract
                or campaign.get('runtime', {}).get('gpu_uuid') != low.GPU_UUID
                or campaign.get('runtime', {}).get('threads') != 4
                or campaign.get('runtime', {}).get('interop_threads') != 1):
            raise ValueError('twenty-hour storage/runtime pin projection differs')
        expected_decisions = profiles.current().training_stop_second * 62
        if (training['environment_attempts']-expected_decisions != 366192
                or profiles.current().training_stop_second + 600 + 600 + 600 != profiles.current().whole_seconds):
            raise ValueError('twenty-hour throughput margin or reserved-tail arithmetic differs')
        expected_training = dict(environment_attempts=1152*segment.get('environment_attempts', 0),
            sample_rows=1152*segment.get('sample_rows', 0),
            optimizer_attempts=1152*segment.get('optimizer_attempts', 0),
            reset_slots=1152*segment.get('reset_slots', 0), fixed_random_choices=0)
        if training != expected_training:
            raise ValueError('twenty-hour finite training effect projection differs')
        panel_cases = len(slots) * panel_count
        expected = dict(
            environment_attempts=training['environment_attempts'] + qualification['environment_attempts'] + panel_cases * evaluation['case_actions'],
            sample_rows=training['sample_rows'] + qualification['sample_rows'],
            optimizer_attempts=training['optimizer_attempts'] + qualification['optimizer_attempts'],
            reset_slots=training['reset_slots'] + qualification['reset_slots'] + len(slots) * 64,
            fixed_random_choices=0)
        caps = _counts(campaign.get('global_effect_caps'))
        if caps != expected:
            raise ValueError('twenty-hour global effect cap projection differs')
        return caps
    predecessor = _predecessor_document(root)
    additional = _additional_predecessor_document(root)
    if predecessor is None and additional is None:
        return dict(CAPS)
    debits = []
    if predecessor is not None:
        debits.append(predecessor['retained_lease'])
    if additional is not None:
        debits.append(additional['charged_or_reserved'])
    remaining = {key: CAPS[key] - sum(debit[key] for debit in debits) for key in CAPS}
    selected = profiles.current()
    if additional is not None:
        base_campaign = _read(Path(root)/profiles.FRESH_GYM.campaign_path)
        b_campaign = _read(Path(root)/profiles.FRESH_GYM_B.campaign_path)
        base_qualification = base_campaign.get('qualification', {}).get('whole_seconds')
        b_qualification = b_campaign.get('qualification', {}).get('whole_seconds')
        b_panel = b_campaign.get('evaluation', {}).get('job_whole_seconds')
        if (base_campaign.get('campaign_id') != profiles.FRESH_GYM.campaign_id
                or b_campaign.get('campaign_id') != profiles.FRESH_GYM_B.campaign_id
                or base_campaign.get('whole_seconds') != profiles.FRESH_GYM.whole_seconds
                or b_campaign.get('whole_seconds') != profiles.FRESH_GYM_B.whole_seconds
                or any(type(value) is not int or value <= 0
                       for value in (base_qualification, b_qualification, b_panel))):
            raise ValueError('continuation campaign time debit differs')
        b_time = additional['reserved_time_seconds']
        if b_qualification != b_time['qualification'] or b_panel != b_time['panel']:
            raise ValueError('continuation campaign time debit differs')
        b_prior_time = base_qualification + sum(b_time.values())
        original_whole = base_campaign['whole_seconds']
        expected_whole = original_whole - b_prior_time
        if (expected_whole != selected.whole_seconds
                or campaign.get('whole_seconds') != expected_whole
                or selected.training_stop_second + campaign.get('evaluation', {}).get('total_seconds', -1) + 600
                    != expected_whole
                or base_campaign.get('training_stop_second')
                    + base_campaign.get('evaluation', {}).get('total_seconds', -1) + 600
                    != base_campaign.get('whole_seconds')
                or b_campaign.get('training_stop_second')
                    + b_campaign.get('evaluation', {}).get('total_seconds', -1) + 600
                    != b_campaign.get('whole_seconds')):
            raise ValueError('continuation campaign time debit differs')
    if (any(value < 0 for value in remaining.values())
            or _counts(campaign.get('global_effect_caps')) != remaining
            or campaign.get('whole_seconds') != selected.whole_seconds
            or campaign.get('training_stop_second') != selected.training_stop_second):
        raise ValueError('continuation campaign budget differs from retained predecessor debit')
    return remaining


def _validate_initial_predecessor(root):
    """Host-only validation of the retained terminal preflight and full lease debit."""
    document = _predecessor_document(root)
    if document is None:
        return None
    root = Path(root).resolve(strict=True)
    actual_hashes, evidence = {}, {}
    for name, relative in PREDECESSOR_ARTIFACT_PATHS.items():
        path = package_api.relative_file(root, relative)
        actual_hashes[name] = package_api.sha256(path)
        if actual_hashes[name] != document['artifacts'][name]['sha256']:
            raise ValueError('campaign predecessor raw artifact changed: '+name)
        evidence[name] = _read(path)
    audit = evidence['audit']
    expected_audit_hashes = {
        PREDECESSOR_AUDIT_HASH_PATHS[name]: actual_hashes[name]
        for name in PREDECESSOR_AUDIT_HASH_PATHS
    }
    if audit.get('hashes') != expected_audit_hashes:
        raise ValueError('predecessor audit raw hashes differ')
    if (audit.get('status') != 'failed_guest_preflight'
            or audit.get('actual_launcher_os_exit') != 1
            or audit.get('watchdog_exit_status') != 1
            or audit.get('whole_run_complete') is not False
            or audit.get('stage') != FAILED_PRECHECK_STAGE
            or audit.get('root_cause') != FAILED_PRECHECK_CAUSE
            or audit.get('native_gameplay_calls') != 0
            or audit.get('tensor_training_calls') != 0
            or audit.get('optimizer_steps') != 0
            or audit.get('full_failed_lease_retained') != FAILED_QUALIFICATION_LEASE
            or audit.get('source_postcheck') is not True
            or audit.get('protected_postcheck') is not True
            or audit.get('absence_lookup_exit') != 0
            or not isinstance(audit.get('removed_container'), str)
            or not audit['removed_container']):
        raise ValueError('predecessor audit does not describe the accepted preflight failure')

    attempt = evidence['attempt']
    if (attempt.get('status') != 'failed'
            or attempt.get('job_id') != 'curriculum-qualification'
            or attempt.get('receipt_binding') is not None
            or attempt.get('reservation') != {'state': 'reserved', 'lease': FAILED_QUALIFICATION_LEASE}
            or not isinstance(attempt.get('error'), Mapping)
            or attempt['error'].get('type') != 'StudyControlError'
            or attempt['error'].get('message') != 'command failed exit 1: /usr/bin/docker'):
        raise ValueError('predecessor failed qualification or receipt evidence differs')
    commands = attempt.get('commands')
    if not isinstance(commands, list):
        raise ValueError('predecessor command audit is malformed')
    def docker_commands(name):
        return [row for row in commands if isinstance(row, Mapping)
                and isinstance(row.get('argv'), list) and len(row['argv']) > 5
                and row['argv'][0] == '/usr/bin/docker' and row['argv'][5] == name]
    created = docker_commands('create')
    started = docker_commands('start')
    removed = docker_commands('rm')
    absent = [row for row in docker_commands('ps')
        if row['argv'][5:] == ['ps', '-a', '--filter', 'id='+audit['removed_container'],
                               '--format', '{{.ID}}']]
    container = created[0].get('stdout', '').strip() if len(created) == 1 else None
    if (container != audit['removed_container'] or created[0].get('exit_code') != 0
            or created[0].get('status') != 'returned' or len(started) != 1
            or started[0].get('argv', [])[-1] != container or started[0].get('exit_code') != 1
            or started[0].get('status') != 'returned' or len(removed) != 1
            or removed[0].get('argv', [])[-2:] != ['--force', container]
            or removed[0].get('exit_code') != 0 or removed[0].get('status') != 'returned'
            or len(absent) != 1 or absent[0].get('exit_code') != 0
            or absent[0].get('status') != 'returned' or absent[0].get('stdout', '').strip()):
        raise ValueError('predecessor container cleanup evidence differs')

    result = evidence['result']
    zero_effects = dict.fromkeys(CAPS, 0)
    jobs = result.get('jobs')
    if (result.get('campaign_id') != profiles.FRESH_GYM.campaign_id
            or result.get('status') != 'failed' or result.get('whole_run_claim') is not False
            or result.get('effects') != zero_effects
            or result.get('charged_or_reserved') != FAILED_QUALIFICATION_LEASE
            or not isinstance(jobs, list) or len(jobs) != 1
            or jobs[0].get('job_id') != 'curriculum-qualification'
            or jobs[0].get('status') != 'failed' or jobs[0].get('receipt_reconciled') is not False
            or jobs[0].get('references', {}).get('receipt', {}).get('availability') != 'unavailable'
            or jobs[0].get('references', {}).get('receipt', {}).get('raw') is not None
            or jobs[0].get('references', {}).get('guest_failure', {}).get('availability') != 'unavailable'
            or jobs[0].get('references', {}).get('guest_failure', {}).get('raw') is not None
            or jobs[0].get('references', {}).get('attempt', {}).get('sha256') != actual_hashes['attempt']):
        raise ValueError('predecessor campaign result or no-receipt evidence differs')
    if (evidence['launcher_result'].get('watchdog_exit_status') != 1
            or evidence['launcher_result'].get('whole_run_complete') is not False
            or evidence['launcher_result'].get('controller_logical_result', {}).get('status') != 'failed'
            or evidence['launcher_result'].get('controller_logical_result', {}).get('finalization_veto') is not True
            or evidence['launcher_os_exit'].get('exit_code') != 1):
        raise ValueError('predecessor launcher exit evidence differs')
    return document


def _receipt_effect_rows(receipt, expected, *, name):
    rows = receipt.get('effects')
    if not isinstance(rows, Mapping) or set(rows) != set(CAPS):
        raise ValueError('additional predecessor '+name+' effect rows differ')
    for key, count in expected.items():
        if rows.get(key) != dict(reserved=count, attempted=count, delivered=count, uncertain=0):
            raise ValueError('additional predecessor '+name+' effect counts differ')


def _validate_b_container_attempt(attempt, *, job_id, container_id, start_exit):
    commands = attempt.get('commands')
    if not isinstance(commands, list):
        raise ValueError('additional predecessor container command audit is malformed')
    def docker_commands(name):
        return [row for row in commands if isinstance(row, Mapping)
                and isinstance(row.get('argv'), list) and len(row['argv']) > 5
                and row['argv'][0] == '/usr/bin/docker' and row['argv'][5] == name]
    created, started = docker_commands('create'), docker_commands('start')
    removed = docker_commands('rm')
    absent = [row for row in docker_commands('ps')
        if row['argv'][5:] == ['ps', '-a', '--filter', 'id='+container_id,
                               '--format', '{{.ID}}']]
    actual_id = created[0].get('stdout', '').strip() if len(created) == 1 else None
    if (actual_id != container_id or created[0].get('status') != 'returned'
            or created[0].get('exit_code') != 0
            or created[0].get('argv', [])[5:8] != ['create', '--name', job_id]
            or len(started) != 1 or started[0].get('status') != 'returned'
            or started[0].get('exit_code') != start_exit
            or started[0].get('argv', [])[-1] != container_id
            or len(removed) != 1 or removed[0].get('status') != 'returned'
            or removed[0].get('exit_code') != 0
            or removed[0].get('argv', [])[-2:] != ['--force', container_id]
            or len(absent) != 1 or absent[0].get('status') != 'returned'
            or absent[0].get('exit_code') != 0 or absent[0].get('stdout', '').strip()):
        raise ValueError('additional predecessor container cleanup differs')


def _validate_additional_predecessor(root, document):
    root = Path(root).resolve(strict=True)
    actual_hashes, evidence = {}, {}
    for name, relative in ADDITIONAL_PREDECESSOR_B_ARTIFACT_PATHS.items():
        path = package_api.relative_file(root, relative)
        actual_hashes[name] = package_api.sha256(path)
        if actual_hashes[name] != document['artifacts'][name]['sha256']:
            raise ValueError('campaign predecessor raw artifact changed: '+name)
        evidence[name] = _read(path)

    audit = evidence['audit']
    expected_audit_hashes = {
        relative: actual_hashes[name]
        for relative, name in ADDITIONAL_PREDECESSOR_B_AUDIT_HASH_PATHS.items()
    }
    if audit.get('hashes') != expected_audit_hashes:
        raise ValueError('additional predecessor audit raw hashes differ')
    if (audit.get('schema') != 'spire-learn-curriculum-zero-version-failure-audit-v1'
            or audit.get('status') != 'failed_evaluation_contract'
            or audit.get('cause') != FAILED_B_CAUSE
            or audit.get('actual_launcher_os_exit') != 1
            or audit.get('watchdog_exit_status') != 1
            or audit.get('whole_run_complete') is not False
            or audit.get('training_arms_started') is not False
            or audit.get('qualification_healthy') is not True
            or audit.get('qualification_attempts') != {'actions': 128, 'adam': 8, 'resets': 10, 'samples': 128}
            or audit.get('qualified_effects') != FAILED_B_QUALIFICATION_EFFECTS
            or audit.get('full_charged_or_reserved') != FAILED_B_CHARGED_OR_RESERVED
            or audit.get('panel_partial') != 'one accepted reset; zero env actions/samples/Adam; zero uncertainty; panel receipt unreconciled'
            or audit.get('failing_case') != FAILED_B_PANEL_CASE
            or audit.get('source_postcheck') is not True
            or audit.get('protected_postcheck') is not True):
        raise ValueError('additional predecessor audit does not describe the accepted B failure')

    qualification_attempt = evidence['qualification_attempt']
    qualification_actual = FAILED_B_QUALIFICATION_EFFECTS
    qualification_reservation = {
        'state': 'settled', 'lease': FAILED_B_QUALIFICATION_LEASE,
        'actual': qualification_actual,
    }
    qualification_receipt_binding = qualification_attempt.get('receipt_binding')
    if (qualification_attempt.get('job_id') != 'curriculum-qualification'
            or qualification_attempt.get('status') != 'complete'
            or qualification_attempt.get('error') is not None
            or qualification_attempt.get('reservation') != qualification_reservation
            or not isinstance(qualification_receipt_binding, Mapping)
            or qualification_receipt_binding.get('sha256') != actual_hashes['qualification_receipt']
            or not qualification_receipt_binding.get('path', '').endswith(
                '/runtime/jobs/curriculum-qualification/output/job-receipt.json')):
        raise ValueError('additional predecessor healthy qualification attempt differs')
    qualification_receipt = evidence['qualification_receipt']
    if (qualification_receipt.get('campaign_id') != profiles.FRESH_GYM_B.campaign_id
            or qualification_receipt.get('job_id') != 'curriculum-qualification'
            or qualification_receipt.get('kind') != 'qualification'
            or qualification_receipt.get('status') != 'complete'
            or qualification_receipt.get('healthy_harness') is not True
            or qualification_receipt.get('qualification_method') != 'tests.neural.test_curriculum.CurriculumIntegrationTests.test_public_catalog_tensor_resume_and_tiny_healthy_segments'
            or qualification_receipt.get('qualification_attempts') != {'actions': 128, 'adam': 8, 'resets': 10, 'samples': 128}
            or qualification_receipt.get('effect_delivery_basis') != 'successful exact held harness'):
        raise ValueError('additional predecessor healthy qualification receipt differs')
    _receipt_effect_rows(qualification_receipt, qualification_actual, name='qualification')

    panel_attempt = evidence['panel_attempt']
    if (panel_attempt.get('job_id') != 'curriculum-panel-fresh-initial'
            or panel_attempt.get('status') != 'failed'
            or panel_attempt.get('receipt_binding') is not None
            or panel_attempt.get('reservation') != {'state': 'reserved', 'lease': FAILED_B_PANEL_LEASE}
            or panel_attempt.get('error') != {
                'type': 'StudyControlError', 'message': 'command failed exit 1: /usr/bin/docker', 'notes': []}):
        raise ValueError('additional predecessor panel attempt differs')
    panel_receipt = evidence['panel_receipt']
    if (panel_receipt.get('campaign_id') != profiles.FRESH_GYM_B.campaign_id
            or panel_receipt.get('job_id') != 'curriculum-panel-fresh-initial'
            or panel_receipt.get('kind') != 'panel'
            or panel_receipt.get('slot') != 'fresh-initial'
            or panel_receipt.get('status') != 'operational_failure'):
        raise ValueError('additional predecessor panel receipt identity differs')
    _receipt_effect_rows(panel_receipt, dict.fromkeys(CAPS, 0) | {'reset_slots': 1}, name='panel')
    summary = panel_receipt.get('summary')
    reports = panel_receipt.get('reports')
    if (not isinstance(summary, Mapping) or summary.get('status') != 'operational_failure'
            or not isinstance(summary.get('failures'), list) or len(summary['failures']) != 1
            or not isinstance(reports, list) or len(reports) != 1):
        raise ValueError('additional predecessor panel failure summary differs')
    summary_failure = summary['failures'][0]
    panel_row = reports[0] if isinstance(reports[0], Mapping) else {}
    panel_report = panel_row.get('report')
    if (summary_failure.get('case_id') != FAILED_B_PANEL_CASE
            or summary_failure.get('status') != 'operational_failure'
            or summary_failure.get('reason') != 'collection_fault'
            or summary_failure.get('exit_code') != 1
            or summary_failure.get('fault') != {
                'exception_type': 'ContractError', 'message': FAILED_B_CONTRACT_MESSAGE}
            or panel_row.get('case_id') != FAILED_B_PANEL_CASE
            or panel_row.get('exit_code') != 1
            or not isinstance(panel_report, Mapping)
            or panel_report.get('status') != 'operational_failure'
            or panel_report.get('operation') != 'evaluate'
            or panel_report.get('mode') != 'standalone'
            or panel_report.get('exit_code') != 1
            or panel_report.get('fault') != {
                'exception_type': 'ContractError', 'message': FAILED_B_CONTRACT_MESSAGE}
            or panel_report.get('optimizer') != {
                'accepted_steps': 0, 'attempted_steps': 0,
                'policy_version': 0, 'uncertain_steps': 0, 'updates': 0}):
        raise ValueError('additional predecessor panel failure evidence differs')
    collection = panel_report.get('collection')
    expected_collection = {
        'attempted_environment_slots': 0, 'attempted_reset_slots': 1,
        'delivered_environment_slots': 0, 'delivered_reset_slots': 1,
        'forwarded_environment_slots': 0, 'forwarded_reset_slots': 1,
        'known_accepted_environment_slots': 0, 'known_accepted_reset_slots': 1,
        'uncertain_environment_slots': 0, 'uncertain_reset_slots': 0,
        'reset_operations': 1, 'reset_slots': 1,
    }
    if (not isinstance(collection, Mapping)
            or any(collection.get(key) != value for key, value in expected_collection.items())
            or collection.get('fault') != {
                'exception_type': 'ContractError', 'known_accepted_slots': 0,
                'message': FAILED_B_CONTRACT_MESSAGE, 'operation_slots': 0,
                'phase': 'ready', 'uncertain_slots': 0}
            or panel_report.get('fights', {}).get('environment_actions') != 0
            or panel_report.get('lifecycle', {}).get('operation_error') != {
                'exception_type': 'ContractError', 'message': FAILED_B_CONTRACT_MESSAGE}):
        raise ValueError('additional predecessor panel contract counters differ')

    result = evidence['result']
    result_attempts = result.get('attempts')
    jobs = result.get('jobs')
    if (result.get('campaign_id') != profiles.FRESH_GYM_B.campaign_id
            or result.get('status') != 'failed' or result.get('whole_run_claim') is not False
            or result.get('charged_or_reserved') != FAILED_B_CHARGED_OR_RESERVED
            or result.get('effects') != qualification_actual
            or not isinstance(result_attempts, Mapping)
            or set(result_attempts) != {'curriculum-qualification', 'curriculum-panel-fresh-initial'}
            or result_attempts.get('curriculum-qualification', {}).get('actual') != qualification_actual
            or result_attempts.get('curriculum-qualification', {}).get('lease') != FAILED_B_QUALIFICATION_LEASE
            or result_attempts.get('curriculum-qualification', {}).get('state') != 'settled'
            or result_attempts.get('curriculum-panel-fresh-initial', {}).get('lease') != FAILED_B_PANEL_LEASE
            or result_attempts.get('curriculum-panel-fresh-initial', {}).get('state') != 'reserved'
            or result.get('training_seconds') != dict.fromkeys(('fixed', 'randomized', 'mixed'), 0.)
            or result.get('prefixes') != dict.fromkeys(('fixed', 'randomized', 'mixed'), {})
            or result.get('stopped_arms') != {}
            or not isinstance(jobs, list) or len(jobs) != 2):
        raise ValueError('additional predecessor B campaign accounting differs')
    qualification_job, panel_job = jobs
    panel_refs = panel_job.get('references', {}) if isinstance(panel_job, Mapping) else {}
    if (qualification_job.get('job_id') != 'curriculum-qualification'
            or qualification_job.get('receipt', {}).get('sha256') != actual_hashes['qualification_receipt']
            or panel_job.get('job_id') != 'curriculum-panel-fresh-initial'
            or panel_job.get('status') != 'failed' or panel_job.get('receipt_reconciled') is not False
            or panel_job.get('error', {}).get('type') != 'StudyControlError'
            or panel_refs.get('attempt', {}).get('sha256') != actual_hashes['panel_attempt']
            or panel_refs.get('receipt', {}).get('sha256') != actual_hashes['panel_receipt']):
        raise ValueError('additional predecessor campaign receipt accounting differs')

    cleanup = audit.get('cleanup')
    if not isinstance(cleanup, list) or len(cleanup) != 2:
        raise ValueError('additional predecessor cleanup audit differs')
    cleanup_by_job = {row.get('job'): row for row in cleanup if isinstance(row, Mapping)}
    expected_jobs = {
        'curriculum-qualification': (qualification_attempt, 0),
        'curriculum-panel-fresh-initial': (panel_attempt, 1),
    }
    if set(cleanup_by_job) != set(expected_jobs):
        raise ValueError('additional predecessor cleanup job set differs')
    for job_id, (attempt, start_exit) in expected_jobs.items():
        cleanup_row = cleanup_by_job[job_id]
        container_id = cleanup_row.get('removed')
        if (cleanup_row.get('absent') is not True
                or not isinstance(container_id, str)
                or re.fullmatch('[0-9a-f]{64}', container_id) is None):
            raise ValueError('additional predecessor container cleanup differs')
        _validate_b_container_attempt(attempt, job_id=job_id,
                                      container_id=container_id, start_exit=start_exit)

    launcher_result = evidence['launcher_result']
    logical = launcher_result.get('controller_logical_result')
    if (launcher_result.get('watchdog_exit_status') != 1
            or launcher_result.get('whole_run_complete') is not False
            or not isinstance(logical, Mapping) or logical.get('status') != 'failed'
            or logical.get('finalization_veto') is not True
            or evidence['launcher_os_exit'].get('exit_code') != 1
            or evidence['launcher_os_exit'].get('session_id') != audit.get('session_id')):
        raise ValueError('additional predecessor launcher seal or exit differs')


def validate_predecessor(root):
    """Host-only validation of all source-bound predecessor debits."""
    if profiles.is_twenty_hour_continuation():
        document = _twenty_hour_continuation_predecessor(root)
        if profiles.is_repaired_twenty_hour_continuation():
            _validate_twenty_hour_failed_admission(root)
        return document
    initial = _validate_initial_predecessor(root)
    additional = _additional_predecessor_document(root)
    if additional is not None:
        _validate_additional_predecessor(root, additional)
    if additional is not None:
        return dict(initial=initial, additional=additional)
    return initial


def release_caps(root):
    campaign = _read(Path(root)/profiles.current().campaign_path)
    return dict(global_effect_caps=_effective_global_effect_caps(root, campaign),
        whole_seconds=profiles.current().whole_seconds,
        training_stop_second=profiles.current().training_stop_second,
        evaluation_seconds=campaign.get('evaluation', {}).get('total_seconds', 1800),
        storage=campaign['storage'], matched_arm=campaign.get('matched_arm'),
        qualification=campaign['qualification'], segment=campaign['segment'])


def qualification_inventory(root):
    directory = Path(root)/profiles.current().qualification_input
    names = sorted(p.name for p in directory.iterdir() if p.is_file())
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError('held qualification inputs are not closed regular files')
    result = package_api.inventory(directory, names)
    if profiles.current().fixed:
        authored = profiles.qualification_source(root)
        expected = package_api.inventory(authored, sorted(p.name for p in authored.iterdir()))
        if result != expected:
            raise ValueError('fixed qualification differs from source-bound authored inputs')
    return result


def release_template(root, *, package_path, source_snapshot, user_approval):
    root = Path(root).resolve(strict=True)
    approval = low._safe(root, user_approval, exists=True)
    if profiles.is_repaired_twenty_hour_continuation():
        validate_predecessor(root)
        _validate_repaired_continuation_authority(root)
        if str(approval.relative_to(root)) != profiles.current().user_approval_path:
            raise ValueError('repaired continuation authority path differs')
    elif profiles.is_corrected_twenty_hour_continuation():
        predecessor = _twenty_hour_continuation_predecessor(root)
        _validate_corrected_continuation_authority(root, predecessor)
        if str(approval.relative_to(root)) != profiles.current().user_approval_path:
            raise ValueError('corrected continuation authority path differs')
    elif profiles.is_twenty_hour_continuation():
        predecessor = _twenty_hour_continuation_predecessor(root)
        binding = predecessor['artifacts']['restart_authority']
        if (str(approval.relative_to(root)) != TWENTY_HOUR_RESTART_AUTHORITY
                or package_api.sha256(approval) != binding['sha256']):
            raise ValueError('continuation restart authority binding differs')
    return dict(schema=RELEASE_SCHEMA, state='pending_review', campaign_id=profiles.current().campaign_id,
        single_use=True, source_snapshot_sha256=package_api.sha256(source_snapshot),
        package_sha256=package_api.sha256(package_path),
        campaign_raw_sha256=package_api.sha256(root/profiles.current().campaign_path),
        qualification_input_inventory=qualification_inventory(root), runtime_pins=runtime_pins(root),
        caps=release_caps(root), user_approval=dict(path=str(approval.relative_to(root)), sha256=package_api.sha256(approval)),
        review={'approved': False, 'reports': []})


def validate_release(root, *, package_path, source_snapshot, release_path):
    root = Path(root).resolve(strict=True)
    release, snapshot = _read(release_path), _read(source_snapshot)
    _snapshot_schema(snapshot)
    if (profiles.current().predecessor_path is not None
            or profiles.current().additional_predecessor_path is not None):
        validate_predecessor(root)
    if (set(release) != RELEASE_KEYS or release['schema'] != RELEASE_SCHEMA
            or release['state'] != 'approved' or release['campaign_id'] != profiles.current().campaign_id
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
    if profiles.is_repaired_twenty_hour_continuation():
        expected_authority = _validate_repaired_continuation_authority(root)
        if (approval['path'] != profiles.current().user_approval_path
                or approval['sha256'] != producer.REPAIRED_CONTINUATION_AUTHORITY_SHA256
                or authority != expected_authority):
            raise ValueError('repaired continuation launch authority binding differs')
    elif profiles.is_corrected_twenty_hour_continuation():
        predecessor = _twenty_hour_continuation_predecessor(root)
        corrected_authority = _validate_corrected_continuation_authority(root, predecessor)
        if (approval['path'] != profiles.current().user_approval_path
                or approval['sha256'] != package_api.sha256(root/approval['path'])
                or authority != corrected_authority):
            raise ValueError('corrected continuation launch authority binding differs')
    elif profiles.is_twenty_hour_continuation():
        predecessor = _twenty_hour_continuation_predecessor(root)
        expected_authority = predecessor['artifacts']['restart_authority']
        if (approval['path'] != TWENTY_HOUR_RESTART_AUTHORITY
                or approval['sha256'] != expected_authority['sha256']
                or authority.get('failed_predecessor') != profiles.TWENTY_HOUR.campaign_id
                or authority.get('healthy_parent_shard') != 337
                or authority.get('unreconciled_shard_skipped') != 338
                or authority.get('first_new_shard_minimum') != 339
                or authority.get('new_whole_seconds_upper_bound') != 65998
                or authority.get('failed_effect_lease_retained') != predecessor['failed_shard']['lease']):
            raise ValueError('continuation restart authority scope differs')
    elif (authority.get('campaign_id') != profiles.current().campaign_id or authority.get('new_runtime_authorized') is not True
            or authority.get('whole_runtime_wall_seconds') != profiles.current().whole_seconds
            or authority.get('standalone_only') is not True):
        raise ValueError('user authority scope differs')
    package_api.verify_inventory(root, snapshot['files'])
    if set(snapshot['files']) != set(producer.source_paths(root)) | set(_extra_source(root)):
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


def _training_binding(package, arm, index):
    """Resolve an absolute scenario shard index into its closed package row."""
    rows = package['training'][arm]
    start = package.get('campaign', {}).get('training_start_indices', {}).get(arm, 0)
    if type(index) is not int or index < start or index >= start + len(rows):
        raise ValueError('training shard index outside the finite prepared stream')
    return rows[index - start]


def _proc_identity(pid):
    """Public Linux metadata for one known controller/owned ancestor only."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid supervisor process identity')
    directory = Path('/proc')/str(pid)
    fields = (directory/'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), session=int(fields[3]),
        start_ticks=int(fields[19]), state=fields[0], uid=directory.stat().st_uid,
        executable=os.readlink(directory/'exe'), cwd=os.readlink(directory/'cwd'),
        argv=[part.decode() for part in (directory/'cmdline').read_bytes().split(b'\0') if part])


def _result_descriptor(pid, descriptor):
    if type(descriptor) is not int or not 0 <= descriptor < 1048576:
        raise ValueError('invalid supervisor result descriptor')
    path = Path('/proc')/str(pid)/'fd'/str(descriptor)
    info = path.stat()
    flags = next(line.split()[1] for line in (path.parent.parent/'fdinfo'/str(descriptor)).read_text().splitlines()
                 if line.startswith('flags:'))
    return dict(path=os.readlink(path), device=info.st_dev, inode=info.st_ino, uid=info.st_uid,
        regular=stat.S_ISREG(info.st_mode), writable=int(flags, 8)&os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR))


def _controller_argv(root, package_path, source_snapshot, release_path):
    return [sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--root', str(root),
        '--package', str(Path(package_path).absolute()), '--source', str(Path(source_snapshot).absolute()),
        '--release', str(Path(release_path).absolute()), *profiles.arguments()]


def _verify_supervision(root, *, package_path, source_snapshot, release_path, clock=time.monotonic):
    """Require the live owned timeout→launcher chain and its open result file."""
    try:
        binding_path = low._safe(root, os.environ['SPIRE_LEARN_CURRICULUM_SUPERVISION'], exists=True)
        binding = package_api.read_json(binding_path, maximum=65536)
        if package_api.sha256(binding_path) != os.environ['SPIRE_LEARN_CURRICULUM_SUPERVISION_SHA256']:
            raise ValueError('supervision binding raw bytes differ')
        expected_keys = {'schema', 'launcher_pid', 'launcher_start_ticks', 'parent_start', 'whole_deadline',
            'watchdog_argv', 'result_path', 'result_fd', 'result_device', 'result_inode', 'controller_source_sha256'}
        if set(binding) != expected_keys or binding['schema'] != 'spire-learn-curriculum-supervision-v1':
            raise ValueError('supervision binding schema differs')
        start = _number(binding['parent_start'], 'supervisor start')
        deadline = _number(binding['whole_deadline'], 'supervisor deadline')
        whole_seconds = profiles.current().whole_seconds
        if (deadline != start+whole_seconds or not start <= clock() < deadline-110.
                or float(os.environ['SPIRE_LEARN_CURRICULUM_PARENT_START']) != start
                or float(os.environ['SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE']) != deadline
                or binding['controller_source_sha256'] != package_api.sha256(Path(__file__))):
            raise ValueError('supervision whole deadline/source differs')
        expected_child = _controller_argv(root, package_path, source_snapshot, release_path)
        command = binding['watchdog_argv']
        child = _proc_identity(os.getpid())
        if child['argv'] != expected_child:
            raise ValueError('actual supervised controller command differs')
        watchdog = _proc_identity(child['ppid'])
        if (watchdog['argv'] != command or watchdog['executable'] != str(Path('/usr/bin/timeout').resolve())
                or watchdog['ppid'] != binding['launcher_pid']):
            raise ValueError('actual owned supervision watchdog differs')
        launcher = _proc_identity(watchdog['ppid'])
        if (not isinstance(command, list) or len(command) != 4+len(expected_child)
                or command[:3] != ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s']
                or re.fullmatch(r'[1-9][0-9]*s', command[3]) is None
                or not 1 <= int(command[3][:-1]) <= whole_seconds-100 or command[4:] != expected_child
                or child['argv'] != expected_child or watchdog['argv'] != command
                or watchdog['executable'] != str(Path('/usr/bin/timeout').resolve())
                or launcher['pid'] != binding['launcher_pid'] or launcher['start_ticks'] != binding['launcher_start_ticks']
                or launcher['executable'] != str(Path(sys.executable).resolve())
                or len(launcher['argv']) < 4 or launcher['argv'][1] != '-B'
                or str((Path(launcher['cwd'])/launcher['argv'][2]).resolve()) != str(Path(__file__).resolve())
                or launcher['argv'][3] != '_launch'
                or any(row['uid'] != 1000 or row['state'] in ('Z', 'X') for row in (child, watchdog, launcher))
                or watchdog['session'] != watchdog['pid'] or watchdog['pgid'] != watchdog['pid']
                or child['session'] != watchdog['session'] or child['pgid'] != watchdog['pgid']
                or not launcher['start_ticks'] <= watchdog['start_ticks'] <= child['start_ticks']):
            raise ValueError('actual owned supervision process chain differs')
        ticks = os.sysconf('SC_CLK_TCK')
        if (not -.1 <= start-launcher['start_ticks']/ticks <= 30.
                or watchdog['start_ticks']/ticks+int(command[3][:-1])+90. > deadline-9.9):
            raise ValueError('actual watchdog hard-stop limit differs')
        result_path = low._safe(root, binding['result_path'], exists=True)
        info = result_path.lstat()
        observed = _result_descriptor(launcher['pid'], binding['result_fd'])
        expected = dict(path=str(result_path), device=binding['result_device'], inode=binding['result_inode'],
            uid=1000, regular=True, writable=True)
        if (observed != expected or not stat.S_ISREG(info.st_mode) or info.st_uid != 1000
                or (info.st_dev, info.st_ino) != (binding['result_device'], binding['result_inode'])
                or info.st_size != 0 or info.st_mode & 0o002):
            raise ValueError('owned supervisor result descriptor differs')
        if any(_proc_identity(row['pid'])['start_ticks'] != row['start_ticks'] for row in (launcher, watchdog)):
            raise ValueError('supervision identity changed during admission')
        return dict(parent_start=start, whole_deadline=deadline, binding_sha256=package_api.sha256(binding_path),
            launcher_pid=launcher['pid'], watchdog_pid=watchdog['pid'], result_path=str(result_path))
    except (OSError, KeyError, IndexError, StopIteration, TypeError) as error:
        raise ValueError('independent owned supervision is missing or unavailable') from error


class CampaignBudget:
    """Shared deadline and effect leases; unknown dispatches retain their full lease."""
    def __init__(self, caps, *, start, deadline, clock=time.monotonic,
                 evaluation_cap=1800, final_panel_reserve=0):
        self.caps = _counts(caps)
        self.start, self.deadline = _number(start, 'start'), _number(deadline, 'deadline')
        if not 0 < self.deadline - self.start <= profiles.current().whole_seconds:
            raise ValueError('whole campaign deadline differs')
        if (type(evaluation_cap) is not int or evaluation_cap <= 0
                or type(final_panel_reserve) is not int or final_panel_reserve < 0
                or final_panel_reserve > evaluation_cap):
            raise ValueError('evaluation lease bounds differ')
        self.clock = clock
        self.evaluation_cap = evaluation_cap
        self.final_panel_reserve = final_panel_reserve
        self.training_stop_second = profiles.current().training_stop_second
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

    def job_deadlines(self, kind, *, arm_deadline=None, final_panel=False):
        now = _number(self.clock(), 'clock')
        if now < self.start:
            raise ValueError('monotonic clock moved backwards')
        whole = min(now + 600., self.deadline - 110.)
        if kind == 'segment':
            whole = min(whole, self.start + self.training_stop_second, arm_deadline or self.deadline)
        elif kind == 'panel':
            if profiles.is_twenty_hour_campaign() or profiles.is_twenty_hour_continuation():
                reserve = 0 if final_panel else self.final_panel_reserve
                available = self.evaluation_cap - self.evaluation_seconds - reserve
                whole = min(whole, now + min(600., available))
                if not final_panel:
                    whole = min(whole, self.start + self.training_stop_second)
                elif profiles.is_twenty_hour_continuation():
                    whole = min(whole, self.start + self.training_stop_second + self.final_panel_reserve)
            else:
                whole = min(whole, now + 1800. - self.evaluation_seconds)
        soft = min(now + (120. if kind == 'qualification' else 480.), whole - 100.)
        if soft <= now:
            raise TimeoutError('shared job/cleanup/training/evaluation time exhausted')
        return soft, whole


def _campaign_budget(root, *, start, deadline, clock=time.monotonic):
    """Construct the live lease ledger from the same release caps guests verify."""
    caps = release_caps(root)
    evaluation = _read(Path(root)/profiles.current().campaign_path).get('evaluation', {})
    return CampaignBudget(caps['global_effect_caps'], start=start, deadline=deadline, clock=clock,
        evaluation_cap=caps['evaluation_seconds'],
        final_panel_reserve=evaluation.get('final_panel_reserve_seconds', 0))


def initialize_stage(root, host, verified):
    """Freeze source once; all nested RO bind targets exist before OCI setup."""
    stage = Path(host)/'source'
    stage_source(root, stage, verified['snapshot'])
    for name in ((profiles.current().domain / 'inputs'), profiles.current().qualification_input, Path(verified['package']['original_parent']['checkpoint_directory'])):
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
    if profiles.current().fixed:
        value['PYTHONPATH'] = profiles.OVERLAY_DESTINATION+':'+value['PYTHONPATH']
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
               '--context', '/control/job.json', *profiles.arguments())
    return low.PhaseSpec(context['job_id'], layout['name'], context['campaign_id'],
        context['release_sha256'], digest, context['claim_sha256'], context['deadline'],
        context['whole_deadline'], context['output_root'], command,
        tuple(sorted(_environment(context, digest).items())), tuple(tuple(row) for row in layout['mounts']),
        tuple(sorted(labels.items())), runtime_class=context['runtime_class'],
        pids_limit=128 if context['kind'] == 'qualification' else 512)


def prepare_job(root, host, verified, budget, *, kind, job_id, parent, arm=None, index=0,
                prefix=None, predecessor=None, slot=None, arm_deadline=None, max_updates=None,
                source_transition=None):
    """Create a new immutable lease/context and selected RO input projection."""
    root, host = Path(root), Path(host)
    package = verified['package']
    if source_transition is not None:
        producer.validate_source_transition(dict(source_transition=source_transition, kind=kind,
            campaign_id=package['campaign_id'], arm=arm, shard_index=index,
            source_sha256=package_api.sha256(verified['source_snapshot']),
            predecessor=predecessor, parent=parent),
            authority=verified['release']['user_approval']['sha256'])
    if re.fullmatch(r'curriculum-[a-z0-9-]{1,100}', job_id) is None or kind not in producer.KIND_KEYS:
        raise ValueError('new job identity/kind differs')
    verify_source(host/'source', verified['snapshot'])
    if package_api.sha256(verified['package_path']) != verified['release']['package_sha256']:
        raise ValueError('immutable package changed')
    soft, whole = budget.job_deadlines(kind, arm_deadline=arm_deadline,
        final_panel=kind == 'panel' and slot == 'final-mixed')
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
        if arm not in package['training']:
            raise ValueError('training arm/index differs')
        binding = _training_binding(package, arm, index)
        names = binding['inputs']
        bindings = {name: package['inventory'][name] for name in names}
        input_relative = (profiles.current().domain / 'inputs')
        output_relative = Path('training')/arm/'runs'
        if profiles.is_twenty_hour_campaign() or profiles.is_twenty_hour_continuation():
            segment_caps = package['campaign']['segment']
            reservation = dict(environment_attempts=segment_caps['environment_attempts'],
                sample_rows=segment_caps['sample_rows'], optimizer_attempts=segment_caps['optimizer_attempts'],
                reset_slots=segment_caps['reset_slots'], fixed_random_choices=0)
            ceiling = package['campaign']['training_caps']['environment_attempts']
            adam_ceiling = package['campaign']['training_caps']['optimizer_attempts']
            maximum_updates = segment_caps['updates']
        else:
            reservation = dict(environment_attempts=4096, sample_rows=4096, optimizer_attempts=256,
                               reset_slots=4112, fixed_random_choices=0)
            ceiling = 1000000 if arm == 'mixed' else 16384
            adam_ceiling = 65536 if arm == 'mixed' else 1024
            maximum_updates = 8
        remaining_actions = ceiling-prefix.get('attempted_environment_slots', 0)
        remaining_adam = adam_ceiling-prefix.get('optimizer_attempted_steps', 0)
        updates = min(maximum_updates, remaining_adam//32)
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
        input_relative, output_relative = (profiles.current().domain / 'inputs'), Path('panel/runs')
        reservation = dict(environment_attempts=16384, sample_rows=0, optimizer_attempts=0, reset_slots=64, fixed_random_choices=0)
        extra = dict(slot=slot, case_ids=[row['case_id'] for row in package['panel']['cases']])
    else:
        bindings = verified['release']['qualification_input_inventory']
        input_relative, output_relative = profiles.current().qualification_input, Path('runs')
        reservation = {key: verified['release']['caps']['qualification'][key] for key in CAPS}
        extra = {}
    _copy_inputs(Path(verified['package_path']).parent, package_stage,
                 bindings if kind != 'qualification' else {})
    shutil.copyfile(verified['package_path'], package_stage/'package.json')
    if kind == 'qualification':
        input_stage = job/'selected-qualification'
        _copy_inputs(root/profiles.current().qualification_input, input_stage, bindings)
    else:
        input_stage = package_stage
    (input_stage/output_relative).mkdir(parents=True)
    guest_output = str(guest/input_relative/output_relative)
    parent_guest = str(guest/original['checkpoint_directory']) if kind == 'qualification' else '/inputs/parent'
    mounts = [(str(host/'source'), str(guest), True),
        (str(package_stage), str(guest/(profiles.current().domain / 'inputs')), True),
        (str(parent_path), parent_guest if kind == 'qualification' else str(low._guest_path(parent_guest)), True),
        (str(control), str(low._guest_path('/control')), True),
        (str(root/low.VENV), str(low._guest_path('/runtime/venv-cuda')), True),
        (str(root/low.LOCK), str(low._guest_path('/runtime/requirements-cuda.lock')), True),
        (str(output), guest_output, False)]
    if kind == 'qualification':
        mounts.insert(2, (str(input_stage), str(guest/profiles.current().qualification_input), True))
    if profiles.current().fixed:
        overlay = profiles.overlay_pins(root)
        if any(not _same(verified['release']['runtime_pins'].get(key), value) for key, value in overlay.items()):
            raise ValueError('released package overlay differs')
        profiles.verify_overlay(root/overlay['package_overlay']['path'], overlay['package_overlay']['inventory'])
        mounts.append((str(root/overlay['package_overlay']['path']), str(low._guest_path(profiles.OVERLAY_DESTINATION)), True))
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
    if source_transition is not None:
        claim['source_transition'] = dict(source_transition)
        extra['source_transition'] = dict(source_transition)
    claim_path = host/'job-claims'/(job_id+'.json')
    _write(claim_path, claim)
    for name, path in (('snapshot.json', verified['source_snapshot']), ('release.json', verified['release_path']),
                       ('campaign-claim.json', verified['claim_path']), ('claim.json', claim_path)):
        shutil.copyfile(path, control/name)
    context = dict(schema=producer.CONTEXT_SCHEMA, campaign_id=package['campaign_id'], job_id=job_id,
        kind=kind, runtime_class='cpu' if kind == 'panel' else 'gpu',
        source_inventory=verified['snapshot']['files'], source_sha256=claim['source_sha256'],
        release_sha256=claim['release_sha256'], claim_sha256=package_api.sha256(claim_path),
        package_path=str((profiles.current().domain / 'inputs')/'package.json'), package_sha256=claim['package_sha256'],
        input_root=str(input_relative), input_inventory=bindings,
        parent=dict(path=parent_guest, inventory=parent_host['inventory']), output_root=guest_output,
        deadline=soft, whole_deadline=whole, reservation=reservation, **extra)
    context_path = control/'job.json'
    _write(context_path, context)
    layout = dict(schema='spire-learn-curriculum-layout-v1', name=job_id, mounts=mounts)
    _write(control/'layout.json', layout)
    return dict(context=context, context_path=context_path, spec=_spec(context, context_path, layout),
                output=output, parent_host=parent_host, input_stage=input_stage)


def _verify_guest_runtime_pins(runtime_pins, *, expected_native_sha256):
    """Check runtime pins against the native admitted from the selected overlay."""
    if (runtime_pins['profile_sha256'] != cdi.PROFILE_SHA256
            or runtime_pins['native_sha256'] != expected_native_sha256
            or runtime_pins['lock_sha256'] != low.LOCK_SHA256
            or runtime_pins['versions'] != PIN_VERSIONS
            or runtime_pins['python_sha256'] != PYTHON_SHA256):
        raise ValueError('guest runtime pins differ')


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
            or release['campaign_raw_sha256'] != package_api.sha256(root/profiles.current().campaign_path)
            or claim.get('schema') != CLAIM_SCHEMA or claim.get('campaign_id') != context['campaign_id']
            or any(claim.get(k) != context[k] for k in ('job_id', 'kind', 'release_sha256', 'source_sha256', 'package_sha256', 'reservation'))
            or claim.get('predecessor') != context.get('predecessor')
            or claim.get('campaign_claim_sha256') != package_api.sha256(control/'campaign-claim.json')
            or campaign_claim.get('schema') != CLAIM_SCHEMA
            or any(campaign_claim.get(k) != context[k] for k in ('campaign_id', 'source_sha256', 'release_sha256', 'package_sha256'))):
        raise ValueError('guest acyclic authority chain differs')
    if (('source_transition' in claim) != ('source_transition' in context)
            or claim.get('source_transition') != context.get('source_transition')):
        raise ValueError('guest source transition claim differs')
    producer.validate_source_transition(context, authority=release['user_approval']['sha256'])
    if snapshot['files'] != context['source_inventory']:
        raise ValueError('guest source inventory differs')
    verify_source(root, snapshot)
    package, input_root, parent, output, config = producer.validate_context(root, context, clock=clock)
    if context['kind'] == 'qualification' and context['input_inventory'] != release['qualification_input_inventory']:
        raise ValueError('guest held qualification input differs')
    expected = {(str(root), True), (str(root/(profiles.current().domain / 'inputs')), True),
        (str(low._guest_path('/control')), True), (str(low._guest_path('/runtime/venv-cuda')), True),
        (str(low._guest_path('/runtime/requirements-cuda.lock')), True), (context['output_root'], False),
        (str(parent), True)}
    if context['kind'] == 'qualification':
        expected.add((str(root/profiles.current().qualification_input), True))
    if profiles.current().fixed:
        expected.add((str(low._guest_path(profiles.OVERLAY_DESTINATION)), True))
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
    if input_root == root/(profiles.current().domain / 'inputs'):
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
    imported_gym = None
    expected_native_sha256 = low.NATIVE_SHA256
    if profiles.current().fixed:
        overlay = profiles.overlay_pins(root)
        if any(not _same(release['runtime_pins'].get(key), value) for key, value in overlay.items()):
            raise ValueError('guest released package overlay differs')
        expected_native_sha256 = overlay['native_sha256']
        directory = low._guest_path(profiles.OVERLAY_DESTINATION)
        profiles.verify_overlay(directory, overlay['package_overlay']['inventory'])
        if package_api.sha256(low._guest_path('/runtime/requirements-cuda.lock')) != low.LOCK_SHA256:
            raise ValueError('guest dependency lock differs')
        # Structural/source/mount/runtime-class checks have completed. Resolve
        # only the actual selected package, then authenticate its import origin.
        imported_gym = profiles.verify_imported_overlay(directory, overlay)
    else:
        low._verify_guest_runtime()
    verify_python_link(low._guest_path('/runtime/venv-cuda'))
    versions = low._runtime_metadata(low._guest_path('/runtime/venv-cuda/lib/python3.12/site-packages'))
    if any(versions.get(k) != v for k, v in PIN_VERSIONS.items()):
        raise ValueError('executing installed versions differ')
    _verify_guest_runtime_pins(release['runtime_pins'], expected_native_sha256=expected_native_sha256)
    record = dict(schema='spire-learn-curriculum-admission-v1', job_id=context['job_id'],
        context_sha256=digest, runtime_class=context['runtime_class'],
        process={k: facts[k] for k in ('uid', 'gid', 'python', 'supplementary_group_ids')},
        runtime_profile=profile, installed_distributions=versions)
    if imported_gym is not None:
        record['imported_gym'] = imported_gym
    encoded = transport.MARKER+json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > 16384:
        raise ValueError('guest admission record too large')
    if sys.stderr.write(encoded) != len(encoded):
        raise OSError('short guest admission write')
    sys.stderr.flush()
    return context


def storage_admission(root, host, deps, *, reservation=0, live=False):
    if type(reservation) is not int or reservation < 0:
        raise ValueError('invalid storage reservation')
    new = low.tree_bytes(Path(host), strict=True, live=live)
    aggregate = low.tree_bytes(Path(root)/'artifacts/development/neural-combat', strict=False, live=True)
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
            or ('source_transition' in receipt) != ('source_transition' in context)
            or receipt.get('source_transition') != context.get('source_transition')
            or clock() >= context['whole_deadline'] or _number(receipt['elapsed_seconds'], 'elapsed') > 600):
        raise ValueError('job receipt authority/status/deadline differs')
    profiles.validate_process_evidence(root, receipt, runtime_pins=verified['release']['runtime_pins'])
    _journal(context, receipt, output)
    checkpoint = None
    package, kind = verified['package'], context['kind']
    if kind == 'segment':
        binding = _training_binding(package, context['arm'], context['shard_index'])
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
        from spire_learn.neural.runner import _checkpoint_recovery
        previous_prefix = {}
        expected_attempts = dict.fromkeys(names, 0)
        for index, segment in enumerate(receipt['qualification_segments']):
            checkpoint_path = _mapped(context, output, segment['checkpoint']['path'])
            if checkpoint_inventory(checkpoint_path) != segment['checkpoint']['inventory']:
                raise ValueError('qualification saved checkpoint changed')
            report = segment['report']
            config = load_neural_config(Path(root)/profiles.current().qualification_input/f'qualification-{index}.toml')
            shard_index = config.scenario_stream['shard_index']
            saved = preflight_checkpoint(checkpoint_path, config.accepted_identities[0], purpose='resume')
            saved_prefix = _checkpoint_recovery(saved, config)[1]
            # The unchanged held method passes its frozen root-relative pilot
            # parent verbatim; campaign producers use /inputs/parent instead.
            expected_parent = (package['original_parent']['checkpoint_directory'] if index == 0 else
                receipt['qualification_segments'][index-1]['checkpoint']['path'])
            if (saved_prefix != segment['saved_prefix'] or saved_prefix['update_count'] != index+1
                    or segment['next_shard'].get('shard_index') != shard_index+1
                    or saved.metadata['run_metadata']['scenario_stream']['current']['shard_index'] != shard_index
                    or saved.metadata.get('parent_checkpoint') != {'path': expected_parent}
                    or any(saved_prefix[k] for k in ('uncertain_environment_slots', 'uncertain_reset_slots', 'optimizer_uncertain_steps'))):
                raise ValueError('held saved prefix/parent/successor differs')
            files = dict(path=str(Path(segment['checkpoint']['path']).parent), inventory=package_api.inventory(
                checkpoint_path.parent, ('summary.json', 'lifecycle.json', 'events.jsonl')))
            _report(context, output, report, files)
            if report.get('status') != 'complete' or report.get('fault') or report.get('lifecycle', {}).get('close_errors'):
                raise ValueError('qualification segment failed lifecycle')
            collection = report['collection']
            local = {name: producer._integer(collection[field], 'held report '+field) for name, field in (
                ('actions', 'attempted_environment_slots'), ('resets', 'attempted_reset_slots'),
                ('samples', 'learned_decisions'))}
            local['samples'] += producer._integer(collection['undispatched_actor_decisions'], 'held undispatched samples')
            local['adam'] = saved_prefix['optimizer_attempted_steps']-previous_prefix.get('optimizer_attempted_steps', 0)
            expected_effects = {}
            for name, kind in names.items():
                count = producer._integer(local[name], 'held actual '+name)
                expected_attempts[name] += count
                delivered = (producer._integer(collection['delivered_environment_slots'], 'held delivered actions')
                    if name == 'actions' else producer._integer(collection['delivered_reset_slots'], 'held delivered resets')
                    if name == 'resets' else count)
                expected_effects[kind] = dict(reserved=count, attempted=count, delivered=delivered, uncertain=0)
            expected_effects['fixed_random_choices'] = dict.fromkeys(('reserved', 'attempted', 'delivered', 'uncertain'), 0)
            reconciled, _ = reconcile_segment(config, {'prefix': previous_prefix, 'parent': {'path': expected_parent}},
                report, saved.metadata, expected_effects, elapsed=receipt['elapsed_seconds'])
            if reconciled != saved_prefix:
                raise ValueError('held report and saved prefix disagree')
            previous_prefix = saved_prefix
        if actual != expected_attempts:
            raise ValueError('held actual effects disagree with the two reports/saved prefixes')
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
        self.training_stop_second = profiles.current().training_stop_second
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
        if self.package['evaluation_slots'][0] not in self.panels_done:
            return dict(kind='panel', slot=self.package['evaluation_slots'][0], arm=None)
        arms = ('fixed', 'randomized', 'mixed')
        while self.arm_index < 3:
            arm = arms[self.arm_index]
            if arm not in self.arm_started:
                self.arm_started[arm] = now
            exhausted = (arm in self.stopped or self.indices[arm] >= 4
                or now >= self.arm_started[arm]+3000.
                or now >= self.start+self.training_stop_second)
            if not exhausted:
                return dict(kind='segment', arm=arm, index=self.indices[arm],
                    arm_deadline=min(self.arm_started[arm]+3000., self.start+self.training_stop_second))
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
        if ('mixed' not in self.stopped and self.indices['mixed'] < 256
                and now < self.start+self.training_stop_second
                and prefix.get('attempted_environment_slots', 0) < 1000000
                and prefix.get('optimizer_attempted_steps', 0) < 65536):
            return dict(kind='segment', arm='mixed', index=self.indices['mixed'],
                        arm_deadline=self.start+self.training_stop_second)
        if 'final-mixed' not in self.panels_done:
            return dict(kind='panel', slot='final-mixed', arm='mixed')
        return None


class TwentyHourSchedule:
    """One preregistered B16 mixed stream with time-triggered frozen panels."""
    def __init__(self, package, *, start, training_seconds=None,
                 continuation_parent_receipt=None):
        self.package, self.start = package, start
        self.training_stop_second = profiles.current().training_stop_second
        self.training_caps = package['campaign']['training_caps']
        self.stream_start = package['campaign']['training_start_indices']['mixed']
        self.stream_count = package['campaign']['training_shards']['mixed']
        self.qualified = False
        self.indices = {'mixed': self.stream_start}
        self.prefixes = {'mixed': {}}
        self.checkpoints = {'mixed': None}
        self.predecessors = {'mixed': None}
        if profiles.is_twenty_hour_continuation():
            continuation = package.get('campaign', {}).get('continuation', {})
            if (not isinstance(continuation_parent_receipt, Mapping)
                    or continuation_parent_receipt.get('sha256') != continuation.get('parent_receipt_sha256')
                    or not continuation.get('parent_prefix')):
                raise ValueError('continuation schedule needs its bound healthy shard337 parent')
            self.prefixes['mixed'] = dict(continuation['parent_prefix'])
            self.indices['mixed'] = continuation['first_new_shard_index']
            self.predecessors['mixed'] = dict(continuation_parent_receipt)
        self.stopped = {}
        self.training_seconds = (training_seconds if training_seconds is not None else
                                 dict.fromkeys(('fixed', 'randomized', 'mixed'), 0.))
        self.slots = package['evaluation_slots']
        self.milestones = package['campaign']['evaluation']['milestone_training_seconds']
        if len(self.slots) != len(self.milestones) + 2:
            raise ValueError('twenty-hour milestone/panel schedule differs')
        self.availability = {slot: dict(status='not_started', reason='not reached') for slot in self.slots}
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
        if arm != 'mixed':
            raise ValueError('twenty-hour training arm differs')
        self.stopped[arm] = reason

    def accept_segment(self, arm, result):
        if arm != 'mixed':
            raise ValueError('twenty-hour training arm differs')
        before = self.prefixes['mixed']
        after = result['checkpoint']['counters']
        if after['attempted_environment_slots'] <= before.get('attempted_environment_slots', 0):
            self.stopped['mixed'] = 'zero_progress'
        self.prefixes['mixed'] = after
        self.checkpoints['mixed'] = result['checkpoint']
        self.predecessors['mixed'] = result['receipt_binding']
        self.indices['mixed'] += 1

    def next(self, now):
        if not self.qualified:
            return dict(kind='qualification')
        initial = self.slots[0]
        if initial not in self.panels_done:
            return dict(kind='panel', slot=initial,
                arm='mixed' if profiles.is_twenty_hour_continuation() else None)
        charged = _number(self.training_seconds['mixed'], 'mixed charged seconds')
        cutoff = self.start + self.training_stop_second
        if now < cutoff:
            for threshold, slot in zip(self.milestones, self.slots[1:-1]):
                if slot not in self.panels_done and charged >= threshold:
                    return dict(kind='panel', slot=slot, arm='mixed')
        else:
            for slot in self.slots[1:-1]:
                if slot not in self.panels_done:
                    self.finish_panel(slot, status='not_started',
                        reason='training cutoff reached before milestone panel')
        prefix = self.prefixes['mixed']
        exhausted = (bool(self.stopped)
            or self.indices['mixed'] >= self.stream_start + self.stream_count
            or now >= cutoff
            or prefix.get('attempted_environment_slots', 0) >= self.training_caps['environment_attempts']
            or prefix.get('optimizer_attempted_steps', 0) >= self.training_caps['optimizer_attempts'])
        if not exhausted:
            return dict(kind='segment', arm='mixed', index=self.indices['mixed'],
                arm_deadline=self.start + self.training_stop_second)
        final = self.slots[-1]
        if final not in self.panels_done:
            return dict(kind='panel', slot=final, arm='mixed')
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
            size = low.tree_bytes(output, strict=True, live=True)
            if size > limit:
                raise RuntimeError('job output storage lease exceeded')
            if deps.clock() >= next_storage_check:
                storage_admission(root, host, deps, reservation=limit-size, live=True)
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
            if budget.evaluation_seconds > budget.evaluation_cap:
                raise TimeoutError('actual enclosing CPU evaluation lease exhausted')
        except BaseException as closure:
            final = low._promote(final, closure)
        if final is not None and final is not primary:
            raise final


def run_campaign(root: Path, *, package_path: Path, source_snapshot: Path, release_path: Path,
                 dependencies=None, clock=time.monotonic) -> Mapping:
    """Keep ownership and first-control protection through final seal publication."""
    with campaign_signal_scope():
        return _run_campaign(root, package_path=package_path, source_snapshot=source_snapshot,
            release_path=release_path, dependencies=dependencies, clock=clock)


def _retained_reference(host, path, *, json_document=True):
    """A failed job's bytes are evidence, never a reconciled success receipt."""
    raw = None
    try:
        path = low._safe(host, path, exists=True)
        if not path.is_file():
            raise ValueError('retained reference is not a regular file')
        raw = dict(path=str(path), sha256=package_api.sha256(path), size_bytes=path.stat().st_size)
        if json_document:
            _read(path)
        return dict(availability='retained', **raw)
    except Exception as error:
        return dict(availability='unavailable', expected_path=str(path), error=low._error(error), raw=raw)


def _failed_dispatch_references(host, job_id):
    host = Path(host)
    job, output = host/'jobs'/job_id, host/'jobs'/job_id/'output'
    references = {name: _retained_reference(host, path) for name, path in (
        ('claim', host/'job-claims'/(job_id+'.json')), ('attempt', host/'attempts'/(job_id+'.json')),
        ('context', job/'control/job.json'), ('receipt', output/'job-receipt.json'),
        ('guest_failure', output/'guest-failure.json'))}
    partial = []
    try:
        directories = sorted(path for path in output.iterdir()
            if re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}', path.name))
        if len(directories) > 64:
            raise ValueError('partial report directory bound exceeded')
        for directory in directories:
            partial.append(dict(directory=str(directory), receipt_reconciled=False,
                summary=_retained_reference(host, directory/'summary.json'),
                lifecycle=_retained_reference(host, directory/'lifecycle.json'),
                events=_retained_reference(host, directory/'events.jsonl', json_document=False)))
        availability = dict(availability='retained', interpretation='unreconciled raw reports only')
    except Exception as error:
        availability = dict(availability='unavailable', error=low._error(error))
    return dict(references=references, partial_reports=partial, partial_report_availability=availability)


def _run_campaign(root: Path, *, package_path: Path, source_snapshot: Path, release_path: Path,
                 dependencies=None, clock=time.monotonic) -> Mapping:
    """Fresh campaign only, admitted by an independently supervised launcher."""
    root = Path(root).resolve(strict=True)
    deps = dependencies or HostDependencies(clock=clock)
    supervision = _verify_supervision(root, package_path=package_path, source_snapshot=source_snapshot,
                                     release_path=release_path, clock=deps.clock)
    start, deadline = supervision['parent_start'], supervision['whole_deadline']
    budget = _campaign_budget(root, start=start, deadline=deadline, clock=deps.clock)
    verified = dict(validate_release(root, package_path=package_path, source_snapshot=source_snapshot, release_path=release_path),
        package_path=Path(package_path), source_snapshot=Path(source_snapshot), release_path=Path(release_path))
    if deps.uid() != 1000 or deps.gid() != 1000:
        raise ValueError('campaign host UID/GID differs')
    host = low._safe(root, (profiles.current().domain / 'runtime'))
    if host.exists():
        raise ValueError('fresh campaign runtime domain already consumed')
    storage_admission(root, root/profiles.current().domain, deps, reservation=GIB)
    host.mkdir()
    # A durable campaign claim is acquired only after complete static admission.
    verified['claim_path'] = host/'campaign-claim.json'
    claim = dict(schema=CLAIM_SCHEMA, campaign_id=profiles.current().campaign_id,
        source_sha256=package_api.sha256(source_snapshot), release_sha256=package_api.sha256(release_path),
        package_sha256=package_api.sha256(package_path), parent_start=start, whole_deadline=deadline,
        supervision=supervision)
    continuation_parent_receipt = None
    if profiles.is_twenty_hour_continuation():
        lineage = verified['package']['campaign']['continuation']
        source_receipt = package_api.relative_file(root, lineage['parent_receipt_path'])
        if package_api.sha256(source_receipt) != lineage['parent_receipt_sha256']:
            raise ValueError('continuation shard337 parent receipt changed before launch')
        lineage_dir = host/'lineage'
        lineage_dir.mkdir()
        owned_receipt = lineage_dir/'shard337-receipt.json'
        shutil.copyfile(source_receipt, owned_receipt)
        if package_api.sha256(owned_receipt) != lineage['parent_receipt_sha256']:
            raise ValueError('owned continuation parent receipt copy differs')
        continuation_parent_receipt = dict(path=str(owned_receipt),
            sha256=lineage['parent_receipt_sha256'])
    plan_type = (TwentyHourSchedule if profiles.is_twenty_hour_campaign()
        or profiles.is_twenty_hour_continuation() else CampaignSchedule)
    plan = (plan_type(verified['package'], start=start, training_seconds=budget.training_seconds,
            continuation_parent_receipt=continuation_parent_receipt)
        if profiles.is_twenty_hour_continuation()
        else plan_type(verified['package'], start=start, training_seconds=budget.training_seconds))
    original = verified['package']['original_parent']
    original_parent = dict(path=str(root/original['checkpoint_directory']), inventory=original['inventory'])
    results, primary, status = [], None, 'running'
    try:
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
                parent = original_parent if arm is None else plan.checkpoints.get(arm) or original_parent
                if parent is None:
                    plan.finish_panel(slot, status='unavailable', reason='no healthy arm checkpoint')
                    continue
                try:
                    budget.job_deadlines('panel', final_panel=slot == 'final-mixed')
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
                if (profiles.is_twenty_hour_continuation()
                        and request['index'] == verified['package']['campaign']['continuation']['first_new_shard_index']):
                    continuation = verified['package']['campaign']['continuation']
                    parameters['source_transition'] = dict(
                        schema=producer.SOURCE_TRANSITION_SCHEMA,
                        purpose='authorized-checkpoint-continuation',
                        from_source_sha256=continuation['parent_source_sha256'],
                        to_source_sha256=claim['source_sha256'],
                        predecessor_sha256=continuation_parent_receipt['sha256'],
                        checkpoint_inventory_sha256=continuation['parent_checkpoint_inventory_sha256'],
                        user_approval_sha256=verified['release']['user_approval']['sha256'])
            else:
                job_id = 'curriculum-qualification'
                parameters = dict(kind=kind, parent=original_parent)
            try:
                result = dispatch_job(root, host, verified, budget, deps=deps, docker=docker,
                    client=client, job_id=job_id, **parameters)
            except BaseException as error:
                # Dispatch has stopped permanently. Retain its bookkeeping
                # before repeated controls unwind the owned finalization path.
                state = _control_state.get()
                if state is not None:
                    state['sealing'] = True
                failed = dict(job_id=job_id, kind=kind, status='failed', dispatch_entered=True,
                    receipt_reconciled=False, error=low._error(error), references={name: dict(
                        availability='unavailable', reason='failure evidence inspection did not finish')
                        for name in ('claim', 'attempt', 'context', 'receipt', 'guest_failure')}, partial_reports=[])
                results.append(failed)
                if kind == 'panel':
                    failed['slot'] = slot
                    plan.finish_panel(slot, status='failed', reason=type(error).__name__+': '+str(error)[:2048],
                        result={'job_id': job_id, 'receipt_reconciled': False})
                try:
                    failed.update(_failed_dispatch_references(host, job_id))
                except BaseException as retention:
                    final = low._promote(error, retention)
                    if final is not error:
                        raise final
                raise
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
    seal = dict(schema='spire-learn-curriculum-campaign-result-v1', campaign_id=profiles.current().campaign_id,
        status=status, error=None if primary is None else low._error(primary), jobs=results,
        availability=plan.availability, prefixes=plan.prefixes, stopped_arms=plan.stopped,
        effects=budget.attempted, charged_or_reserved=budget.reserved, attempts=budget.jobs,
        evaluation_seconds=budget.evaluation_seconds, training_seconds=budget.training_seconds,
        qualification_seconds=budget.qualification_seconds, elapsed_seconds=deps.clock()-start,
        parent_start=start, whole_deadline=deadline, source_sha256=claim['source_sha256'],
        release_sha256=claim['release_sha256'], qualifications=verified['package']['campaign']['qualifications'])
    return _publish_campaign_result(root, host, deps, seal, primary)


def _publish_campaign_result(root, host, deps, seal, primary):
    state = _control_state.get()
    if state is None:
        raise RuntimeError('campaign sealing requires its owned signal scope')
    state['sealing'] = True
    def retain(error):
        nonlocal primary
        if not isinstance(error, Exception) and state['first'] is None:
            state['first'] = error
        if primary is not error:
            primary = low._promote(primary, error)
        if state['first'] is not None and primary is not state['first']:
            state['first'].add_note('other finalization failure: '+str(low._error(primary)))
            primary = state['first']
    if primary is not None:
        retain(primary)
    elif state['first'] is not None:
        retain(state['first'])
    seal['status'] = 'failed' if primary is not None else seal['status']
    seal['error'] = None if primary is None else low._error(primary)
    seal['whole_run_claim'] = False
    seal['outcome_scope'] = 'controller logical outcome; independent watchdog wait and absence of finalization veto required'
    try:
        storage_admission(root, host, deps)
        _write(host/'campaign-result.json', seal)
        if deps.clock() >= seal['whole_deadline']-10.:
            raise TimeoutError('campaign sealing exhausted independent parent exit tail')
    except BaseException as error:
        retain(error)
    if state['first'] is not None:
        retain(state['first'])
    if primary is not None:
        failure = dict(schema='spire-learn-curriculum-seal-failure-v1', status='failed', whole_run_claim=False,
            parent_start=seal['parent_start'], whole_deadline=seal['whole_deadline'],
            error=low._error(primary), source_sha256=seal['source_sha256'])
        try:
            _write(host/'campaign-seal-failure.json', failure)
        except BaseException as error:
            retain(error)
        if state['first'] is not None:
            retain(state['first'])
        raise primary
    return seal


def _forward_launcher_control(state):
    process = state['process']
    if (process is not None and not state['wait_finished'] and not state['sent']
            and getattr(process, 'returncode', None) is None):
        state['sent'] = True
        try:
            os.killpg(process.pid, state['number'])
        except ProcessLookupError:
            pass


@contextmanager
def _launcher_signal_scope():
    existing = _launcher_state.get()
    if existing is not None:
        yield existing
        return
    state = dict(first=None, number=None, process=None, wait_finished=False, sent=False, finish=None)
    token = _launcher_state.set(state)
    def handler(number, frame):
        if state['first'] is None:
            state['first'], state['number'] = low._Termination('signal '+str(number)), number
        _forward_launcher_control(state)
    previous = {number: signal.signal(number, handler) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
        if state['first'] is not None and state['finish'] is not None:
            try:
                state['finish'](state['first'])
            except BaseException as retention:
                state['first'].add_note('late launcher veto failed: '+str(low._error(retention)))
            raise state['first']
    finally:
        state['process'], state['wait_finished'] = None, True
        for number, handler in previous.items():
            signal.signal(number, handler)
        _launcher_state.reset(token)


def supervise(command, *, env, deadline, clock=time.monotonic):
    """Share the launcher's first control, and disarm its PGID after wait."""
    with _launcher_signal_scope() as state:
        try:
            process = subprocess.Popen(command, env=env, start_new_session=True)
            state['process'] = process
            if state['first'] is not None:
                _forward_launcher_control(state)
            try:
                code = process.wait(timeout=max(.1, deadline-clock()-5.))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait(timeout=max(.1, deadline-clock()-1.))
        finally:
            state['process'], state['wait_finished'] = None, True
        result = subprocess.CompletedProcess(command, code)
        result.parent_control = state['number']
        result.parent_control_error = state['first']
        return result


def _controller_logical_result(arguments, *, start, deadline):
    unavailable = dict(availability='unavailable', status=None, finalization_veto=True)
    if '--root' not in arguments:
        return unavailable
    try:
        root = Path(arguments[arguments.index('--root')+1]).resolve(strict=True)
        host = root/(profiles.current().domain / 'runtime')
        claim, seal = _read(host/'campaign-claim.json'), _read(host/'campaign-result.json')
        if any(value.get('parent_start') != start or value.get('whole_deadline') != deadline for value in (claim, seal)):
            return unavailable
        if any(seal.get(k) != claim.get(k) for k in ('campaign_id', 'source_sha256', 'release_sha256')):
            return unavailable
        veto = (host/'campaign-seal-failure.json').exists()
        return dict(availability='published', status='failed' if veto else seal['status'], finalization_veto=veto,
            path=str(host/'campaign-result.json'), sha256=package_api.sha256(host/'campaign-result.json'))
    except (OSError, ValueError, KeyError, IndexError):
        return unavailable


def launch_guard(arguments, *, result_path, clock=time.monotonic, run=None, parent_start=None):
    """Wait on the owned watchdog; controller OS exit is not directly observed."""
    with _launcher_signal_scope() as state:
        return _launch_guard(arguments, result_path=result_path, clock=clock, run=run,
                             parent_start=parent_start, state=state)


def _launch_guard(arguments, *, result_path, clock, run, parent_start, state):
    start = _number(clock() if parent_start is None else parent_start, 'launcher start')
    if start > clock():
        raise ValueError('launcher start cannot renew time in the future')
    whole_seconds = profiles.current().whole_seconds
    deadline = start+whole_seconds
    env = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
    env.update(SPIRE_LEARN_CURRICULUM_PARENT_START=str(start), SPIRE_LEARN_CURRICULUM_PARENT_DEADLINE=str(deadline))
    if not Path('/usr/bin/timeout').is_file():
        raise ValueError('GNU timeout is required')
    term_seconds = max(1, math.floor(start+whole_seconds-100.-clock()))
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=90s', str(term_seconds)+'s', sys.executable,
        '-B', str(Path(__file__).resolve()), *arguments]
    # Reserve the outer record pathname before dispatch; never overwrite evidence.
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open('x') as stream:
        result, primary = None, None
        veto_path = result_path.with_name(result_path.name+'.failure.json')
        def retain(error):
            nonlocal primary
            primary = low._promote(primary, error) if primary is not error else error
            if state['first'] is not None and primary is not state['first']:
                state['first'].add_note('other launcher failure: '+str(low._error(primary)))
                primary = state['first']
        try:
            # Sidecar failure is a recorded zero-child launch failure.
            term_seconds = max(1, math.floor(start+whole_seconds-100.-clock()-1.))
            command[3] = str(term_seconds)+'s'
            launcher = _proc_identity(os.getpid())
            result_info = os.fstat(stream.fileno())
            binding = dict(schema='spire-learn-curriculum-supervision-v1', launcher_pid=os.getpid(),
                launcher_start_ticks=launcher['start_ticks'], parent_start=start, whole_deadline=deadline,
                watchdog_argv=command, result_path=str(result_path.absolute()), result_fd=stream.fileno(),
                result_device=result_info.st_dev, result_inode=result_info.st_ino,
                controller_source_sha256=package_api.sha256(Path(__file__)))
            binding_path = result_path.with_name(result_path.name+'.supervision.json')
            _write(binding_path, binding)
            env['SPIRE_LEARN_CURRICULUM_SUPERVISION'] = str(binding_path.absolute())
            env['SPIRE_LEARN_CURRICULUM_SUPERVISION_SHA256'] = package_api.sha256(binding_path)
            if state['first'] is not None:
                raise state['first']
            result = (supervise(command, env=env, deadline=deadline, clock=clock) if run is None
                else run(command, env=env, check=False, timeout=max(1., deadline-clock()-5.)))
            earlier = getattr(result, 'parent_control', None)
            if earlier is not None and earlier != state['number']:
                state['number'] = earlier
                state['first'] = getattr(result, 'parent_control_error', None) or low._Termination('signal '+str(earlier))
        except BaseException as error:
            retain(error)
        finally:
            state['process'], state['wait_finished'] = None, True
        if state['first'] is not None:
            retain(state['first'])
        record = dict(schema='spire-learn-curriculum-launcher-result-v1', parent_start=start,
            whole_deadline=deadline, watchdog_exit_status=None if result is None else result.returncode,
            controller_os_exit_status=dict(availability='unavailable', reason='only GNU timeout was directly waited on'),
            error=None if primary is None else low._error(primary), elapsed_seconds=clock()-start,
            command=command, term_after_seconds=term_seconds, kill_grace_seconds=90,
            parent_control=state['number'], failure_veto_path=str(veto_path),
            measurement_boundary='actual owned GNU timeout wait completed before parent result publication; final 10 seconds reserved')
        try:
            logical = _controller_logical_result(arguments, start=start, deadline=deadline)
        except BaseException as error:
            retain(error)
            logical = dict(availability='unavailable', status=None, finalization_veto=True)
            record['error'] = low._error(primary)
        record['controller_logical_result'] = logical
        record['whole_run_complete'] = (primary is None and result is not None and result.returncode == 0
            and record['parent_control'] is None and clock() < deadline and logical.get('status') == 'complete'
            and logical.get('finalization_veto') is False)
        def failure_veto(error):
            _write(veto_path, dict(schema='spire-learn-curriculum-launcher-failure-v1', whole_run_complete=False,
                watchdog_exit_status=record['watchdog_exit_status'], parent_control=state['number'],
                error=low._error(error), publication_candidate=record))
        state['finish'] = failure_veto
        try:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        except BaseException as retention:
            retain(retention)
        if state['first'] is not None:
            retain(state['first'])
        if primary is not None:
            try:
                failure_veto(primary)
            except BaseException as retention:
                retain(retention)
            # This new reserved file has not successfully finalized. Retain
            # its initial candidate in the veto, then make one failure-only
            # correction; no child/effect or authority operation is retried.
            record = dict(record, whole_run_complete=False, error=low._error(primary), parent_control=state['number'])
            try:
                stream.seek(0); stream.truncate()
                json.dump(record, stream, sort_keys=True, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            except BaseException as retention:
                retain(retention)
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
    parser.add_argument('--profile', choices=profiles.NAMES, default=profiles.HISTORICAL.name)
    args = parser.parse_args(argv)
    with profiles.scope(args.profile):
        return _main(args, parser)


def _main(args, parser):
    root = args.root.resolve(strict=True)
    def required(*names):
        if any(getattr(args, name) is None for name in names):
            parser.error('required: '+', '.join('--'+name.replace('_', '-') for name in names))
    def path(value):
        return low._safe(root, value)
    if args.command == 'capture':
        required('destination')
        if args.head is not None or args.dirty_json is not None:
            parser.error('production capture always records actual Git provenance; overrides are fixture API only')
        from scripts.neural.gpu_runtime_proof_helpers import _git_state
        actual_head, actual_dirty = _git_state(root, timeout=10.)
        value = capture_source(root, path(args.destination), learner_head=actual_head, dirty=actual_dirty)
    elif args.command == 'release-template':
        required('package', 'source', 'destination', 'user_approval')
        value = release_template(root, package_path=path(args.package), source_snapshot=path(args.source),
            user_approval=path(args.user_approval))
        _write(path(args.destination), value)
    elif args.command == 'validate':
        required('package', 'source', 'release')
        verified = validate_release(root, package_path=path(args.package), source_snapshot=path(args.source),
            release_path=path(args.release))
        value = dict(status='validated', campaign_id=profiles.current().campaign_id,
            source_sha256=package_api.sha256(path(args.source)), package_sha256=package_api.sha256(path(args.package)),
            release_sha256=package_api.sha256(path(args.release)),
            full_package_shards=sum(len(rows) for rows in verified['package']['training'].values()))
    elif args.command == 'run':
        required('package', 'source', 'release')
        value = run_campaign(root, package_path=path(args.package), source_snapshot=path(args.source), release_path=path(args.release))
    else:
        required('package', 'source', 'release')
        arguments = ['run', '--root', str(root), '--package', str(path(args.package)),
            '--source', str(path(args.source)), '--release', str(path(args.release)), *profiles.arguments()]
        return launch_guard(arguments, result_path=path(args.result or profiles.current().domain/'launcher-result.json'),
                            parent_start=_ENTRY_MONOTONIC)
    print(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
