"""Standard-library bootstrap records for an owned curriculum campaign monitor."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _resolved(path: Path, root: Path) -> Path:
    path = path if path.is_absolute() else root/path
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError('campaign monitor path escapes repository root')
    return resolved


def bootstrap_monitor(*, root: Path, campaign_domain: Path, profile_name: str,
                      campaign_id: str, owner_exec_session_id: int, launcher_pid: int,
                      launcher_start_ticks: int, monotonic_start: float,
                      whole_seconds: int, training_stop_second: int, launch_utc: str,
                      control_paths: Iterable[Path]) -> dict[str, Path]:
    """Write owner/session and heartbeat state outside a still-absent runtime tree.

    The caller supplies the tool-owned exec session and the exact control inputs
    that the child controller will consume. This check runs before execing it.
    """
    root = Path(root).resolve(strict=True)
    domain = _resolved(Path(campaign_domain), root)
    runtime = domain/'runtime'
    if runtime.exists() or runtime.is_symlink():
        raise ValueError('fresh campaign runtime must be absent before monitor bootstrap')
    for value in (profile_name, campaign_id, launch_utc):
        if not isinstance(value, str) or not value:
            raise ValueError('campaign monitor identity/time labels are required')
    for name, value in (('owner session', owner_exec_session_id), ('launcher PID', launcher_pid),
                        ('launcher start ticks', launcher_start_ticks), ('whole seconds', whole_seconds),
                        ('training cutoff', training_stop_second)):
        if type(value) is not int or value <= 0:
            raise ValueError('campaign monitor '+name+' must be a positive integer')
    if not 0 < training_stop_second < whole_seconds:
        raise ValueError('campaign monitor training cutoff must precede its whole deadline')
    if isinstance(monotonic_start, bool) or not isinstance(monotonic_start, (int, float)) \
            or not math.isfinite(monotonic_start) or monotonic_start <= 0:
        raise ValueError('campaign monitor monotonic start must be positive and finite')

    runtime = _resolved(runtime, root)
    paths = tuple(_resolved(Path(path), root) for path in control_paths)
    if not paths or any(path.is_relative_to(runtime) for path in paths):
        raise ValueError('campaign package/source/release controls must be outside fresh runtime')
    monitor = _resolved(domain/'monitor', root)
    if monitor.is_relative_to(runtime):
        raise ValueError('campaign monitor state must be outside fresh runtime')
    launch_session_path = monitor/'launch-session.json'
    heartbeat_state_path = monitor/'heartbeat-state.json'
    if (launch_session_path.is_relative_to(runtime) or heartbeat_state_path.is_relative_to(runtime)
            or launch_session_path.exists() or heartbeat_state_path.exists()):
        raise ValueError('campaign monitor records overlap runtime or already exist')

    session = {
        'schema': 'spire-learn-curriculum-owned-launch-session-v1',
        'campaign_id': campaign_id,
        'profile_name': profile_name,
        'owner_exec_session_id': owner_exec_session_id,
        'launcher_pid': launcher_pid,
        'launcher_start_ticks': launcher_start_ticks,
        'launch_utc': launch_utc,
        'monotonic_start_before_exec': float(monotonic_start),
        'whole_seconds': whole_seconds,
        'training_stop_monotonic': float(monotonic_start)+training_stop_second,
        'whole_deadline_monotonic': float(monotonic_start)+whole_seconds,
        'runtime_path': str(runtime),
        'control_paths': [str(path) for path in paths],
        'state': 'launching',
    }
    state = {
        'schema': 'spire-learn-curriculum-heartbeat-state-v1',
        'campaign_id': campaign_id,
        'profile_name': profile_name,
        'owner_exec_session_id': owner_exec_session_id,
        'last_complete_progress_index': None,
        'first_healthy_mixed_training_reported': False,
        'development_panel_ids_reported': [],
        'last_observation_utc': launch_utc,
    }
    _write_exclusive(launch_session_path, session)
    _write_exclusive(heartbeat_state_path, state)
    if runtime.exists():
        raise ValueError('monitor bootstrap unexpectedly created fresh runtime')
    return {'launch_session_path': launch_session_path,
            'heartbeat_state_path': heartbeat_state_path,
            'runtime_path': runtime}
