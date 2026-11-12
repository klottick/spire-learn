"""Closed, source-bound deployment choices for the existing curriculum controller.

Selection changes configuration only. It never supplies observed runtime facts,
authority, release approval, or a replacement for an admission gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import sys


@dataclass(frozen=True)
class Profile:
    name: str
    campaign_id: str
    campaign_path: str
    domain: Path
    qualification_input: Path
    template: str
    parents: str
    revision: str
    baseline_slot: str
    runtime_profile: str | None = None
    overlay_manifest: str | None = None
    native_sha256: str | None = None
    package_overlay: str | None = None
    predecessor_path: str | None = None
    whole_seconds: int = 36000
    training_stop_second: int = 33600
    additional_predecessor_path: str | None = None
    user_approval_path: str | None = None

    @property
    def fixed(self):
        return self.runtime_profile is not None


HISTORICAL = Profile('historical', 'curriculum-20261101-a',
    'configs/neural/experiments/curriculum-v1/campaign.json',
    Path('artifacts/development/neural-combat/curriculum-campaign-v1'),
    Path('artifacts/development/neural-combat/curriculum-v1/task-5/prepared-mixed'),
    'configs/neural/curriculum-v1/pilot.toml', 'configs/neural/curriculum-v1/pilot-parents.json',
    '533dad1c8aec86df624b3742b6b3ad1dbf1d0033', 'original-b')
FIXED = Profile('fixed-gym-a86df922', 'curriculum-20261101-b',
    'configs/neural/experiments/curriculum-v2/campaign.json',
    Path('artifacts/development/neural-combat/curriculum-campaign-v2'),
    Path('artifacts/development/neural-combat/curriculum-campaign-v2/qualification'),
    'configs/neural/experiments/curriculum-v2/template.toml',
    'configs/neural/experiments/curriculum-v2/parents.json',
    'a86df922a9dc598645820cc6544df36aefb3d2cc', 'warm-start-fixed-003',
    'configs/neural/native-fixes-20261101/runtime.json',
    'configs/neural/experiments/curriculum-v2/overlay.json')
FIXED_C = replace(FIXED, name='fixed-gym-a86df922-c', campaign_id='curriculum-20261101-c',
    campaign_path='configs/neural/experiments/curriculum-v3/campaign.json',
    domain=Path('artifacts/development/neural-combat/curriculum-campaign-v3'),
    qualification_input=Path('artifacts/development/neural-combat/curriculum-campaign-v3/qualification'))
FRESH_GYM = Profile('fresh-gym-afb51f5f', 'curriculum-20261102-a',
    'configs/neural/experiments/curriculum-v4/campaign.json',
    Path('artifacts/development/neural-combat/curriculum-campaign-v4'),
    Path('artifacts/development/neural-combat/curriculum-campaign-v4/qualification'),
    'configs/neural/experiments/curriculum-v4/template.toml',
    'configs/neural/experiments/curriculum-v4/parents.json',
    'afb51f5f222b4c4331ed34067c5eebeda606db91', 'fresh-initial',
    'configs/neural/gym-20261102/runtime.json',
    'configs/neural/experiments/curriculum-v4/overlay.json',
    '1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac',
    'artifacts/development/neural-combat/gym-migration-20261102/site')
FRESH_GYM_B = Profile('fresh-gym-afb51f5f-b', 'curriculum-20261102-b',
    'configs/neural/experiments/curriculum-v5/campaign.json',
    Path('artifacts/development/neural-combat/curriculum-campaign-v5'),
    Path('artifacts/development/neural-combat/curriculum-campaign-v5/qualification'),
    FRESH_GYM.template, FRESH_GYM.parents, FRESH_GYM.revision, FRESH_GYM.baseline_slot,
    FRESH_GYM.runtime_profile, FRESH_GYM.overlay_manifest, FRESH_GYM.native_sha256,
    FRESH_GYM.package_overlay,
    predecessor_path='configs/neural/experiments/curriculum-v5/predecessor.json',
    whole_seconds=35400, training_stop_second=33000)
FRESH_GYM_C = replace(FRESH_GYM_B, name='fresh-gym-afb51f5f-c',
    campaign_id='curriculum-20261102-c',
    campaign_path='configs/neural/experiments/curriculum-v6/campaign.json',
    domain=Path('artifacts/development/neural-combat/curriculum-campaign-v6'),
    qualification_input=Path('artifacts/development/neural-combat/curriculum-campaign-v6/qualification'),
    additional_predecessor_path='configs/neural/experiments/curriculum-v6/predecessor-b.json',
    whole_seconds=34200, training_stop_second=31800)
TWENTY_HOUR = replace(FRESH_GYM, name='ironclad-twenty-hour-20261103',
    campaign_id='curriculum-20261103-twenty-hour',
    campaign_path='configs/neural/experiments/twenty-hour-20261103/campaign.json',
    domain=Path('artifacts/development/neural-combat/twenty-hour-20261103'),
    qualification_input=Path('artifacts/development/neural-combat/twenty-hour-20261103/qualification'),
    template='configs/neural/experiments/twenty-hour-20261103/template.toml',
    parents='configs/neural/experiments/twenty-hour-20261103/parents.json',
    baseline_slot='warm-start-baseline', whole_seconds=72000, training_stop_second=70200)
TWENTY_HOUR_CONTINUATION = replace(TWENTY_HOUR,
    name='ironclad-twenty-hour-continuation-20261103',
    campaign_id='curriculum-20261103-twenty-hour-continuation-1',
    campaign_path='configs/neural/experiments/twenty-hour-20261103-continuation-1/campaign.json',
    domain=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1'),
    qualification_input=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-1/qualification'),
    template='configs/neural/experiments/twenty-hour-20261103-continuation-1/template.toml',
    parents='configs/neural/experiments/twenty-hour-20261103-continuation-1/parents.json',
    baseline_slot='continuation-parent-shard337',
    predecessor_path='configs/neural/experiments/twenty-hour-20261103-continuation-1/predecessor.json',
    whole_seconds=65998, training_stop_second=64198)
TWENTY_HOUR_CONTINUATION_2 = replace(TWENTY_HOUR_CONTINUATION,
    name='ironclad-twenty-hour-continuation-20261103-corrected',
    campaign_id='curriculum-20261103-twenty-hour-continuation-2',
    campaign_path='configs/neural/experiments/twenty-hour-20261103-continuation-2/campaign.json',
    domain=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-2'),
    qualification_input=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-2/qualification'),
    template='configs/neural/experiments/twenty-hour-20261103-continuation-2/template.toml',
    parents='configs/neural/experiments/twenty-hour-20261103-continuation-2/parents.json',
    predecessor_path='configs/neural/experiments/twenty-hour-20261103-continuation-2/predecessor.json',
    user_approval_path='configs/neural/experiments/twenty-hour-20261103-continuation-2/launch-authority.json')
TWENTY_HOUR_CONTINUATION_3 = replace(TWENTY_HOUR_CONTINUATION_2,
    name='ironclad-twenty-hour-continuation-20261103-repaired',
    campaign_id='curriculum-20261103-twenty-hour-continuation-3',
    campaign_path='configs/neural/experiments/twenty-hour-20261103-continuation-3/campaign.json',
    domain=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-3'),
    qualification_input=Path('artifacts/development/neural-combat/twenty-hour-20261103-continuation-3/qualification'),
    template='configs/neural/experiments/twenty-hour-20261103-continuation-3/template.toml',
    parents='configs/neural/experiments/twenty-hour-20261103-continuation-3/parents.json',
    predecessor_path='configs/neural/experiments/twenty-hour-20261103-continuation-3/predecessor.json',
    additional_predecessor_path='configs/neural/experiments/twenty-hour-20261103-continuation-3/predecessor-continuation-2.json',
    user_approval_path='configs/neural/experiments/twenty-hour-20261103-continuation-3/launch-authority.json')
_PROFILES = (HISTORICAL, FIXED, FIXED_C, FRESH_GYM, FRESH_GYM_B, FRESH_GYM_C,
             TWENTY_HOUR, TWENTY_HOUR_CONTINUATION, TWENTY_HOUR_CONTINUATION_2,
             TWENTY_HOUR_CONTINUATION_3)
NAMES = tuple(profile.name for profile in _PROFILES)
OVERLAY_DESTINATION = '/runtime/gym-overlay'
NATIVE_SHA256 = '96d54621aa533f27ca720621dfa17664609e51e5b9311fed1636cffa8cd9d787'
LEGACY_PACKAGE_OVERLAY = 'artifacts/development/neural-combat/gym-refresh-20261101/site'
_selected = ContextVar('curriculum_deployment_profile', default=HISTORICAL)


def current():
    return _selected.get()


def is_fresh_gym_family():
    return current() in (FRESH_GYM, FRESH_GYM_B, FRESH_GYM_C, TWENTY_HOUR,
                         TWENTY_HOUR_CONTINUATION, TWENTY_HOUR_CONTINUATION_2,
                         TWENTY_HOUR_CONTINUATION_3)


def is_twenty_hour_campaign():
    return current() is TWENTY_HOUR


def is_twenty_hour_continuation():
    return current() in (TWENTY_HOUR_CONTINUATION, TWENTY_HOUR_CONTINUATION_2,
                         TWENTY_HOUR_CONTINUATION_3)


def is_corrected_twenty_hour_continuation():
    return current() is TWENTY_HOUR_CONTINUATION_2


def is_repaired_twenty_hour_continuation():
    return current() is TWENTY_HOUR_CONTINUATION_3


@contextmanager
def scope(name):
    if name not in NAMES:
        raise ValueError('unknown curriculum profile')
    token = _selected.set(next(profile for profile in _PROFILES if profile.name == name))
    try:
        yield current()
    finally:
        _selected.reset(token)


def arguments():
    return ['--profile', current().name] if current().fixed else []


def source_paths(root):
    paths = ['scripts/neural/curriculum_profiles.py']
    selected = current()
    if is_fresh_gym_family():
        directories = [Path(root) / 'configs/neural/experiments/curriculum-v4']
        if selected in (FRESH_GYM_B, FRESH_GYM_C):
            directories.append(Path(root) / 'configs/neural/experiments/curriculum-v5')
        if selected is FRESH_GYM_C:
            directories.append(Path(root) / 'configs/neural/experiments/curriculum-v6')
        if selected is TWENTY_HOUR:
            directories.append(Path(root) / 'configs/neural/experiments/twenty-hour-20261103')
        if selected is TWENTY_HOUR_CONTINUATION:
            directories.append(Path(root) / 'configs/neural/experiments/twenty-hour-20261103-continuation-1')
        if selected is TWENTY_HOUR_CONTINUATION_2:
            directories.append(Path(root) / 'configs/neural/experiments/twenty-hour-20261103-continuation-2')
        if selected is TWENTY_HOUR_CONTINUATION_3:
            directories.append(Path(root) / 'configs/neural/experiments/twenty-hour-20261103-continuation-3')
        for directory in directories:
            paths += [str(path.relative_to(root)) for path in directory.rglob('*') if path.is_file()]
        paths += [selected.runtime_profile, selected.overlay_manifest,
                  'tests/unit/test_neural_curriculum_fresh_profile.py']
        if selected is TWENTY_HOUR:
            paths.append('tests/unit/test_neural_twenty_hour_campaign.py')
        if selected is TWENTY_HOUR_CONTINUATION:
            paths.append('tests/unit/test_neural_curriculum_continuation_1.py')
        if selected is TWENTY_HOUR_CONTINUATION_2:
            paths += ['scripts/neural/campaign_monitor.py',
                      'tests/unit/test_neural_curriculum_continuation_2.py']
        if selected is TWENTY_HOUR_CONTINUATION_3:
            paths += ['scripts/neural/campaign_monitor.py',
                      'tests/unit/test_neural_curriculum_continuation_3.py',
                      'tests/unit/test_neural_curriculum_thin_guest.py']
        if selected in (FRESH_GYM_B, FRESH_GYM_C):
            paths.append('tests/unit/test_neural_curriculum_continuation.py')
        if selected is FRESH_GYM_C:
            paths.append('tests/unit/test_neural_curriculum_continuation_c.py')
    elif selected.fixed:
        directory = Path('configs/neural/experiments/curriculum-v2')
        paths += [str(path.relative_to(root)) for path in (Path(root)/directory).rglob('*') if path.is_file()]
        paths += [selected.campaign_path, selected.runtime_profile,
                  'tests/unit/test_neural_curriculum_fixed_profile.py']
    return paths


def qualification_source(root):
    if is_twenty_hour_campaign():
        return Path(root) / 'configs/neural/experiments/twenty-hour-20261103/qualification'
    if is_twenty_hour_continuation():
        return Path(root) / Path(current().campaign_path).parent / 'qualification'
    directory = ('configs/neural/experiments/curriculum-v4/qualification'
                 if is_fresh_gym_family() else 'configs/neural/experiments/curriculum-v2/qualification')
    return Path(root) / directory


def verify_overlay(directory, bindings):
    from scripts.neural.curriculum_package import verify_inventory
    directory = Path(directory)
    paths = list(directory.rglob('*'))
    if (directory.is_symlink() or any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in paths)
            or {str(p.relative_to(directory)) for p in paths if p.is_file()} != set(bindings)):
        raise ValueError('package overlay inventory is not closed')
    verify_inventory(directory, bindings)


def overlay_pins(root):
    """Validate source-bound metadata; never interpret it as observed import facts."""
    from scripts.neural.curriculum_package import read_json, sha256
    from scripts.neural import task9_study_controls as low
    selected = current()
    if not selected.fixed:
        raise ValueError('historical profile has no package overlay')
    root = Path(root)
    manifest = read_json(root/selected.overlay_manifest)
    runtime = read_json(root/selected.runtime_profile)
    native_sha256 = selected.native_sha256 or NATIVE_SHA256
    package_overlay = selected.package_overlay or LEGACY_PACKAGE_OVERLAY
    if (set(manifest) != {'schema', 'runtime_profile_sha256', 'inventory', 'reusable_package_sha256'}
            or manifest['schema'] != 'spire-learn-curriculum-overlay-v1'
            or sha256(root/selected.runtime_profile) != manifest['runtime_profile_sha256']
            or runtime['schema'] not in {
                'spire-learn-gym-runtime-profile-v1',
                'spire-learn-gym-runtime-profile-v2',
            }
            or runtime['gym_revision'] != selected.revision or runtime['native_sha256'] != native_sha256
            or runtime['dependency_venv'] != str(low.VENV)
            or runtime['dependency_lock_sha256'] != low.LOCK_SHA256
            or runtime['package_overlay'] != package_overlay):
        raise ValueError('fixed Gym runtime profile differs')
    native = 'spire_lab/_native.cpython-312-x86_64-linux-gnu.so'
    if manifest['inventory'][native]['sha256'] != native_sha256:
        raise ValueError('fixed Gym native inventory differs')
    return dict(profile=selected.name, runtime_profile_sha256=manifest['runtime_profile_sha256'],
        gym_revision=selected.revision, native_sha256=native_sha256,
        package_overlay={'path': package_overlay, 'inventory': manifest['inventory'],
            'guest_destination': OVERLAY_DESTINATION, 'readonly': True,
            'pythonpath': OVERLAY_DESTINATION+':/learner/src:/learner'})


def qualification_hashes(root):
    from scripts.neural.curriculum_package import read_json
    directory = (str(Path(current().campaign_path).parent)
                 if is_twenty_hour_campaign() or is_twenty_hour_continuation() else
                 'configs/neural/experiments/curriculum-v4'
                 if is_fresh_gym_family() else 'configs/neural/experiments/curriculum-v2')
    return read_json(Path(root)/directory/'qualification-identities.json')


def verify_imported_overlay(directory, pins):
    """Authenticate actual imported files/build info, without constructing an env."""
    from scripts.neural.curriculum_package import sha256
    directory = Path(directory).resolve(strict=True)
    package = importlib.import_module('spire_lab')
    native = importlib.import_module('spire_lab._native')
    for name, module in list(sys.modules.items()):
        if name == 'spire_lab' or name.startswith('spire_lab.'):
            path = Path(getattr(module, '__file__', '')).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('imported Gym package origin differs: '+name)
            relative = str(path.relative_to(directory))
            binding = pins['package_overlay']['inventory'].get(relative)
            if binding is None or sha256(path) != binding['sha256']:
                raise ValueError('imported Gym package inventory differs')
    native_path = Path(native.__file__).resolve()
    if native_path != directory/'spire_lab/_native.cpython-312-x86_64-linux-gnu.so' or sha256(native_path) != pins['native_sha256']:
        raise ValueError('imported Gym native identity differs')
    info = dict(native.build_info())
    if info.get('extension') != 'spire-python' or info.get('revision') != current().revision:
        raise ValueError('imported Gym build revision differs')
    return dict(native_file=str(native_path), package_file=str(Path(package.__file__).resolve()), build_info=info)


def validate_process_evidence(root, receipt, *, runtime_pins):
    """Admit only the authenticated Gym exception, then apply old dependency gates.

    The original receipt and process facts remain untouched. The generic gate
    receives every dependency entry unchanged after the one Gym entry has been
    checked here against the source-bound release and exact overlay origin.
    """
    from scripts.neural import sustained_controls as transport
    if not current().fixed:
        return transport.validate_process_evidence(receipt, reference=False)
    expected = overlay_pins(root)
    if any(runtime_pins.get(key) != value for key, value in expected.items()):
        raise ValueError('receipt released package overlay differs')
    overlay = expected['package_overlay']
    verify_overlay(Path(root)/overlay['path'], overlay['inventory'])
    process = receipt.get('process')
    modules = process.get('modules') if isinstance(process, dict) else None
    module = modules.get('spire_lab') if isinstance(modules, dict) else None
    if (not isinstance(module, dict)
            or module.get('file') != overlay['guest_destination']+'/spire_lab/__init__.py'
            or module.get('version') != runtime_pins['versions']['spire-lab']
            or 'spire_lab/__init__.py' not in overlay['inventory']):
        raise ValueError('receipt executing Gym overlay origin/version differs')
    dependencies = {name: value for name, value in modules.items() if name != 'spire_lab'}
    transport.validate_process_evidence({**receipt, 'process': {**process, 'modules': dependencies}}, reference=False)
