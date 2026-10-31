"""Pinned Task 9 NVIDIA CDI metadata contract; stdlib, no import-time effects."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat

PROFILE_PATH = 'configs/runtime/task9-nvidia-cdi-profile.json'
PROFILE_SHA256 = '83214eaa2b2568accc8350ac1cd05955d5cfdf1594a6234d383863d6ceabe937'
_PROFILE_CANONICAL = '458fbc72adb8fcee92a0ceb44eb2eec407040c0b1bf5fc6ad992b085151015d4'
PROFILE_ID = 'task9-nvidia-cdi-1.20.0-595.91.07-v1'
SOCKET = '/run/nvidia-persistenced/socket'
PARAMS = '/proc/driver/nvidia/params'
_HOOK_PREFIX = '/run/nvidia-ctk-hook'
_HOOK = re.compile(_HOOK_PREFIX + r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z')
_DRIVER_FILES = tuple('/usr/bin/' + n for n in (
    'nvidia-smi', 'nvidia-debugdump', 'nvidia-persistenced',
    'nvidia-cuda-mps-control', 'nvidia-cuda-mps-server')) + tuple(
    '/usr/lib/x86_64-linux-gnu/' + n for n in (
    'libcuda.so.595.91.07', 'libcudadebugger.so.595.91.07', 'libnvidia-cfg.so.595.91.07',
    'libnvidia-ml.so.595.91.07', 'libnvidia-nvvm.so.595.91.07', 'libnvidia-nvvm70.so.4',
    'libnvidia-opencl.so.595.91.07', 'libnvidia-pkcs11-openssl3.so.595.91.07',
    'libnvidia-pkcs11.so.595.91.07', 'libnvidia-ptxjitcompiler.so.595.91.07',
    'libnvidia-tileiras.so.595.91.07')) + (
    '/etc/OpenCL/vendors/nvidia.icd', '/usr/lib/firmware/nvidia/595.91.07/gsp_ga10x.bin',
    '/usr/lib/firmware/nvidia/595.91.07/gsp_tu10x.bin')
_AUTOMATIC = {'HOSTNAME', 'PWD', 'SHLVL', '_', 'LC_ALL', 'NVIDIA_REQUIRE_CUDA'}
_GPU_PHASES = {'standalone-a', 'run-a', 'standalone-b', 'run-b'}


def _fail(field, observed=None):
    # Unknown environment values are never included in a diagnostic.
    raise ValueError('CDI ' + str(field)[:180] + ' differs: ' + repr(observed)[:256])


def _checked(profile):
    if not isinstance(profile, Mapping):
        _fail('profile shape')
    raw = json.dumps(dict(profile), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != _PROFILE_CANONICAL:
        _fail('profile semantic hash')
    return profile


def load_runtime_profile(root: Path) -> dict:
    """Load only the exact authored profile; no host/runtime discovery."""
    root = Path(root).resolve(strict=True)
    path = root / PROFILE_PATH
    current = root
    for part in Path(PROFILE_PATH).parts:
        current = current / part
        if current.is_symlink():
            _fail('profile symlink')
    if not path.is_file() or not 0 < path.stat().st_size <= 65536:
        _fail('profile size/type')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROFILE_SHA256:
        _fail('profile raw hash')
    value = json.loads(raw)
    _checked(value)
    return value


def _host_file_sha256(path, maximum):
    path = Path(path)
    expected = {'/var/run/cdi/nvidia.yaml': '/run/cdi/nvidia.yaml',
                '/usr/bin/nvidia-cdi-hook': '/usr/bin/nvidia-cdi-hook'}
    if str(path) not in expected or str(path.resolve(strict=True)) != expected[str(path)]:
        _fail('host input canonical path', str(path))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        _fail('host input size/type', str(path))
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        _fail('host input size', str(path))
    return hashlib.sha256(raw).hexdigest()


def verify_host_profile(profile: Mapping) -> dict:
    """Hash pinned public files; never execute toolkit, YAML, GPU or native code."""
    _checked(profile)
    hashes = {}
    for name, item in profile['host_inputs'].items():
        digest = _host_file_sha256(item['path'], 65536 if name == 'cdi_spec' else 64*1024**2)
        if digest != item['sha256']:
            _fail('host ' + name + ' hash', digest)
        hashes[name + '_sha256'] = digest
    return {'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256, **hashes}


def _describe(path):
    info = path.lstat()
    return {'mode': info.st_mode, 'uid': info.st_uid, 'gid': info.st_gid,
            'device': info.st_dev, 'inode': info.st_ino, 'size_bytes': info.st_size,
            'effective_write_access': os.access(path, os.W_OK, effective_ids=True)}


def _unescape(path):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), path)


def collect_runtime_mount_facts(mountinfo_text: str) -> dict:
    """Bounded metadata only; no file contents, writes, socket or device opens."""
    if len(mountinfo_text.encode()) > 1024**2:
        _fail('mountinfo size')
    rows = []
    types = {}
    for line in mountinfo_text.splitlines():
        fields = line.split()
        try:
            split = fields.index('-')
            destination = _unescape(fields[4])
            data = {'fstype': fields[split+1], 'mount_options': fields[5].split(','),
                    'super_options': fields[split+3].split(',')}
        except (ValueError, IndexError):
            _fail('mountinfo shape')
        if destination in types:
            _fail('duplicate mount destination', destination)
        types[destination] = data
        rows.append((destination, 'ro' in data['mount_options']))
    targets = [p for p, _ in rows if p in _DRIVER_FILES or p in (SOCKET, PARAMS) or p.startswith(_HOOK_PREFIX)]
    if len(targets) > 64:
        _fail('runtime mount cardinality')
    metadata = {}
    for destination in targets:
        path = Path(destination)
        try:
            row = _describe(path)
            if stat.S_ISDIR(row['mode']):
                usage = os.statvfs(path)
                row.update(capacity_bytes=usage.f_frsize*usage.f_blocks,
                           available_bytes=usage.f_frsize*usage.f_bavail,
                           inode_capacity=usage.f_files, free_inodes=usage.f_favail)
                with os.scandir(path) as entries:
                    names = [entry.name for entry in islice(entries, 17)]
                row['children_truncated'] = len(names) > 16
                row['children'] = {n: _describe(path/n) for n in sorted(names[:16])}
            metadata[destination] = row
        except OSError as error:
            metadata[destination] = {'metadata_error': type(error).__name__}
    return {'runtime_mount_types': types, 'runtime_mount_metadata': metadata,
            'supplementary_group_ids': os.getgroups()}


def _integer(value, field, *, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(field, value)
    return value


def _kind(row, kind, field):
    if not isinstance(row, Mapping):
        _fail(field + ' metadata shape')
    mode = row.get('mode')
    for key in ('uid','gid'):
        _integer(row.get(key), field + ' ' + key)
    if type(mode) is not int or stat.S_IFMT(mode) != kind:
        _fail(field + ' type/mode', mode)
    return stat.S_IMODE(mode)


def _params_file(row, field, maximum):
    if (_kind(row, stat.S_IFREG, field) != 0o444 or row.get('uid') != 0 or row.get('gid') != 0
            or row.get('effective_write_access') is not False):
        _fail(field + ' owner/mode/access', row)
    for key in ('device', 'inode'):
        _integer(row.get(key), field + ' ' + key, minimum=1)
    size = _integer(row.get('size_bytes'), field + ' size', minimum=1)
    if size > maximum:
        _fail(field + ' size', size)


def validate_guest_profile(profile: Mapping, *, phase: str, requested_environment: Mapping,
                           actual_environment: Mapping, facts: Mapping) -> dict:
    """Validate real observations; return accepted deviations without altering them."""
    _checked(profile)
    if phase not in _GPU_PHASES | {'selection'}:
        _fail('phase', phase)
    return validate_runtime_class(profile, runtime_class='gpu' if phase in _GPU_PHASES else 'cpu',
        requested_environment=requested_environment, actual_environment=actual_environment, facts=facts)


def validate_runtime_class(profile: Mapping, *, runtime_class: str, requested_environment: Mapping,
                           actual_environment: Mapping, facts: Mapping) -> dict:
    """The same pinned OCI contract for a new truthful job identity."""
    _checked(profile)
    if runtime_class not in ('cpu', 'gpu'):
        _fail('runtime_class', runtime_class)
    requested = dict(requested_environment)
    actual = dict(actual_environment)
    gpu = runtime_class == 'gpu'
    if requested.get('NVIDIA_VISIBLE_DEVICES') != (profile['gpu_uuid'] if gpu else 'none'):
        _fail('requested NVIDIA_VISIBLE_DEVICES', requested.get('NVIDIA_VISIBLE_DEVICES'))
    if not gpu and (requested.get('CUDA_VISIBLE_DEVICES', '') != ''
                   or requested.get('NVIDIA_DRIVER_CAPABILITIES', '') != ''):
        _fail('CPU requested GPU environment')
    cdi = gpu and actual.get('NVIDIA_VISIBLE_DEVICES') == 'void'
    expected = {**requested, **(profile['post_oci_environment'] if cdi else {})}
    for key, value in expected.items():
        if actual.get(key) != value:
            _fail('environment ' + key, actual.get(key))
    unknown = sorted(set(actual) - set(expected) - _AUTOMATIC)
    if unknown:
        _fail('unexpected environment key ' + ','.join(unknown[:8]))
    mounts = facts.get('mounts', [])
    table = {}
    for destination, readonly in mounts:
        if type(readonly) is not bool or destination in table:
            _fail('mount shape/duplicate', destination)
        table[destination] = readonly
    special = [p for p in table if p in profile['driver_files'] or p in (SOCKET, PARAMS)
               or p.startswith(_HOOK_PREFIX)]
    if not gpu and (special or facts.get('device_nodes')):
        _fail('CPU GPU state', special[:4])
    if not cdi:
        # Legacy driver files retain C's existing exact predicate. No /run CDI
        # exceptions exist in this mode, even under otherwise normal OCI /proc.
        if any(p in (SOCKET, PARAMS) or p.startswith(_HOOK_PREFIX) for p in table):
            _fail('legacy CDI runtime mount')
        return {'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256, 'mode': 'legacy',
                'accepted_environment_changes': {}, 'admitted_mount_destinations': [], 'runtime_exceptions': {}}
    if sorted(facts.get('supplementary_group_ids', [])) != profile['supplementary_group_ids']:
        _fail('supplementary_group_ids', facts.get('supplementary_group_ids'))
    metadata = facts.get('runtime_mount_metadata', {})
    types = facts.get('runtime_mount_types', {})
    admitted = []
    for destination in profile['driver_files']:
        row = metadata.get(destination, {})
        mode = _kind(row, stat.S_IFREG, destination)
        if (table.get(destination) is not True or row.get('effective_write_access') is not False
                or row.get('uid') != 0 or mode & 0o022):
            _fail(destination + ' readonly/owner/access', row)
        admitted.append(destination)
    socket = metadata.get(SOCKET, {})
    wanted = profile['socket']
    if (_kind(socket, stat.S_IFSOCK, SOCKET) != wanted['mode'] or table.get(SOCKET) is not True
            or any(socket.get(k) != wanted[k] for k in ('uid','gid','effective_write_access'))
            or types.get(SOCKET,{}).get('fstype') != 'tmpfs'
            or set(types.get(SOCKET,{}).get('mount_options',[])) != {'ro','nosuid','nodev','noexec'}):
        _fail('persistence socket owner/mode/access', socket)
    admitted.append(SOCKET)
    hooks = [p for p in table if p.startswith(_HOOK_PREFIX)]
    if len(hooks) != 1 or not _HOOK.fullmatch(hooks[0]):
        _fail('hook mount cardinality/path', hooks)
    hook = hooks[0]
    if any(p.startswith(hook+'/') or p.startswith(SOCKET+'/') for p in table):
        _fail('nested runtime mount')
    row = metadata.get(hook, {})
    wanted = profile['scratch']
    if (_kind(row, stat.S_IFDIR, hook) != wanted['mode'] or row.get('uid') != 0 or row.get('gid') != 0
            or table[hook] is not False or row.get('effective_write_access') is not True
            or types.get(hook, {}).get('fstype') != 'tmpfs'):
        _fail('hook owner/mode/access/filesystem', row)
    capacity = _integer(row.get('capacity_bytes'), 'hook capacity', minimum=1)
    available = _integer(row.get('available_bytes'), 'hook available bytes')
    if capacity > wanted['capacity_max_bytes'] or available != 0:
        _fail('hook capacity/available', capacity)
    inode_capacity = _integer(row.get('inode_capacity'), 'hook inode capacity', minimum=1)
    free_inodes = _integer(row.get('free_inodes'), 'hook free inodes')
    if free_inodes > inode_capacity:
        _fail('hook inode accounting')
    children = row.get('children')
    if row.get('children_truncated') is not False or not isinstance(children, Mapping) or set(children) != {'nvct-params'}:
        _fail('hook child inventory', children)
    child = children['nvct-params']
    params = metadata.get(PARAMS, {})
    _params_file(child, 'hook child', capacity)
    _params_file(params, 'driver params', capacity)
    if (PARAMS not in table or types.get(PARAMS, {}).get('fstype') != 'tmpfs'
            or any(params.get(k) != child.get(k) for k in ('device','inode','size_bytes'))):
        _fail('driver params file identity')
    # The hook's bind is observed rw in mountinfo despite its requested RO bind;
    # immutable 0444 ownership/access and same-file identity are required above.
    admitted.extend((hook, PARAMS))
    # C still validates every application/OCI mount. Reject unknown runtime
    # paths here too, so a caller cannot treat this profile as a generic allowlist.
    for destination in table:
        if destination.startswith('/run/') and destination not in admitted:
            _fail('unrecognized runtime mount', destination)
        if destination.endswith('docker.sock') or destination.endswith('containerd.sock'):
            _fail('management socket', destination)
    scratch = {**row, 'inode_qualification': 'observed inode capacity; no separate hard inode quota'}
    return {'profile_id': PROFILE_ID, 'profile_sha256': PROFILE_SHA256, 'mode': 'nvidia-cdi',
            'accepted_environment_changes': dict(profile['post_oci_environment']),
            'admitted_mount_destinations': sorted(admitted),
            'runtime_exceptions': {'socket': socket, 'scratch_path': hook, 'scratch': scratch,
                                   'driver_params': params}}
