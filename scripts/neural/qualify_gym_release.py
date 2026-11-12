"""One-use, CPU-only metadata qualification using unchanged owned-ID transport."""
from pathlib import Path
import dataclasses
import hashlib
import json
import time
import sys

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo), str(_repo/'src')]

from scripts.neural import task9_study_controls as low, sustained_controls as transport
from scripts.neural.gpu_runtime_proof_helpers import HostDependencies
from scripts.neural.task8_gpu_setup_helpers import verify_image_inspect

ROOT = Path.cwd()
DOMAIN = ROOT/'artifacts/development/neural-combat/gym-migration-20261102'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory():
    return {str(p.relative_to(ROOT)): sha(p)
            for scope in ('src', 'configs', 'scripts', 'tests')
            for p in sorted((ROOT/scope).rglob('*'))
            if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}


def main():
    deps = HostDependencies()
    start = deps.clock()
    deadline = start + 80.
    prepared = json.loads((DOMAIN/'preparation.json').read_text())
    release = json.loads((DOMAIN/'qualification-release.json').read_text())
    source = source_inventory()
    digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert release['state'] == 'reviewed' and release['single_use'] is True
    assert release['source_digest'] == digest
    assert release['wrapper_sha256'] == sha(ROOT/'scripts/neural/qualify_gym_release.py')
    assert release['launcher_sha256'] == sha(ROOT/'scripts/neural/run-gym-release-qualification.sh')
    assert release['guest_sha256'] == sha(ROOT/'scripts/neural/probe_gym_release.py')
    assert release['preparation_sha256'] == sha(DOMAIN/'preparation.json')
    site = DOMAIN/'site'
    actual_files = {str(p.relative_to(site)):sha(p) for p in site.rglob('*') if p.is_file()}
    assert actual_files == prepared['overlay_files']
    assert not any(p.is_symlink() for p in site.rglob('*'))
    assert sha(ROOT/prepared['wheel_path']) == prepared['release']['wheel_sha256']
    for name, expected in prepared['preserved_files'].items():
        assert sha(ROOT/name) == expected, name
    assert sha(ROOT/low.LOCK) == low.LOCK_SHA256
    assert sha(ROOT/low.NATIVE) == low.NATIVE_SHA256
    client = DOMAIN/'docker-client'
    output = DOMAIN/'output'
    assert not client.exists() and not output.exists()
    assert not (DOMAIN/'qualification-started.json').exists()
    environment = dict(low.IMAGE_ENV)
    environment.update(PYTHONPATH='/release/site:/learner/src', PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', TMPDIR='/output', XDG_CACHE_HOME='/output',
        MIGRATION_CONTEXT_SHA256=digest)
    spec = low.PhaseSpec('gym-migration-metadata', 'spirelearn-gym-migration-20261102-a',
        'gym-migration-20261102', digest, digest, digest, start+80., deadline, '/output',
        ('-B', '/release/verify_guest.py'), tuple(sorted(environment.items())),
        ((str(ROOT/low.VENV), '/runtime/venv-cuda', True),
         (str(site), '/release/site', True),
         (str(ROOT/'scripts/neural/probe_gym_release.py'), '/release/verify_guest.py', True),
         (str(DOMAIN/'preparation.json'), '/release/preparation.json', True),
         (str(ROOT/'src'), '/learner/src', True),
         (str(ROOT/'configs'), '/learner/configs', True),
         (str(output), '/output', False)),
        (('learn.migration', digest),), runtime_class='cpu', pids_limit=128)
    with (DOMAIN/'qualification-started.json').open('x') as f:
        json.dump({'source': source, 'spec': dataclasses.asdict(spec),
                   'limits':prepared['qualification_caps'], 'release_sha256':sha(DOMAIN/'qualification-release.json')}, f, indent=2)
    client.mkdir(); output.mkdir()
    logs=[]; result=None; primary=None
    try:
        docker=low._docker_binary()
        image=low._command(deps, low._docker(docker,client,'image','inspect',low.IMAGE),
                           deadline=deadline,logs=logs,client=client)
        verify_image_inspect(image)
        with transport.signal_scope():
            result=transport.execute_job(spec,deps=deps,client=client,docker=docker,logs=logs,progress=None)
    except BaseException as error:
        primary=error
        raise
    finally:
        postcheck = source_inventory() == source
        protected = all(sha(ROOT/name) == expected for name,expected in prepared['preserved_files'].items())
        overlay = {str(p.relative_to(site)):sha(p) for p in site.rglob('*') if p.is_file()} == prepared['overlay_files']
        record={'result':result,'failure':None if primary is None else {
            'type':type(primary).__name__,'message':str(primary),'notes':getattr(primary,'__notes__',[])},
            'elapsed_seconds':deps.clock()-start,'source_postcheck':postcheck,
            'protected_postcheck':protected,'overlay_postcheck':overlay,'commands':logs}
        try:
            with (DOMAIN/'qualification-process.json').open('x') as f:
                json.dump(record,f,indent=2); f.write('\n')
        except BaseException as final_error:
            if primary is None: raise
            primary.add_note('qualification evidence write failed: '+type(final_error).__name__)
    assert postcheck and protected and overlay
    assert result and result['removal_entered'] is True
    assert record['elapsed_seconds'] <= 120.
    print(json.dumps({'status':'passed','elapsed_seconds':record['elapsed_seconds'],
                      'container_id':result['container_id']}))

if __name__ == '__main__':
    main()
