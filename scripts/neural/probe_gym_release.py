"""One bounded metadata and actual Learn factory check; no gameplay."""
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    attempted_imports = []
    class NoTrainingImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'torchrl', 'tensordict'}:
                attempted_imports.append(fullname)
                raise AssertionError('training import during metadata check: '+fullname)
    sys.meta_path.insert(0, NoTrainingImports())
    from spire_lab import _native, card_metadata
    from spire_lab.gymnasium import IroncladCombatVectorEnv, RunEnv
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.envs import make_driver
    from spire_learn.neural.features import feature_contract
    from spire_learn.neural.gym_contracts import CURRENT_CONTRACT

    prepared = json.loads(Path('/release/preparation.json').read_text())
    release = prepared['release']
    info = dict(_native.build_info())
    assert info['revision'] == release['revision'] == CURRENT_CONTRACT.source_revision
    assert info['extension'] == 'spire-python'
    native = Path(_native.__file__).resolve()
    assert native.is_relative_to('/release/site/spire_lab')
    assert hashlib.sha256(native.read_bytes()).hexdigest() == release['native_sha256']
    assert os.getuid() == 1000
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and os.environ['NVIDIA_VISIBLE_DEVICES'] == 'none'
    denied = []
    close_counts = {}
    def no_gameplay(*args, **kwargs):
        denied.append('unexpected reset/step')
        raise AssertionError('zero-gameplay check forwarded a reset/step')

    def close_once(original):
        def tracked(env):
            # Every environment remains referenced below, preventing id reuse.
            key = str(id(env))
            close_counts[key] = close_counts.get(key, 0) + 1
            return original(env)
        return tracked

    fingerprints = {}
    adapter_checks = []
    environments = []
    entry = {'id': 'case', 'encounter': 'cultist', 'hp': 80, 'max_hp': 80,
             'deck': [{'card': 'strike', 'upgrades': 0}], 'relics': [], 'potions': [None]*3}
    with (patch.object(IroncladCombatVectorEnv, 'reset', no_gameplay),
          patch.object(IroncladCombatVectorEnv, 'step', no_gameplay),
          patch.object(RunEnv, 'reset', no_gameplay), patch.object(RunEnv, 'step', no_gameplay),
          patch.object(IroncladCombatVectorEnv, 'close', close_once(IroncladCombatVectorEnv.close)),
          patch.object(RunEnv, 'close', close_once(RunEnv.close))):
        for name, profile in [('view-v1', None), ('view-v2', 'ironclad-combat-v2')]:
            catalog = {'entries': [entry]}
            if profile: catalog['profile_id'] = profile
            env = IroncladCombatVectorEnv(1, catalog)
            environments.append(env)
            try:
                descriptor = env.observation_spec()
                fingerprints[name] = descriptor['fingerprint']
                with Path('/output/'+name+'-descriptor.json').open('x') as f:
                    json.dump(descriptor, f, sort_keys=True, separators=(',', ':')); f.write('\n')
            finally: env.close()
        for name, profile in [('run-v3', 'm10-ironclad-policy'), ('m10-run-v2', 'm10-ironclad-heart'), ('m9-run-v1', 'm9-ironclad-act3')]:
            env = RunEnv('ironclad', 0, profile=profile)
            environments.append(env)
            try:
                descriptor = env.observation_spec()
                fingerprints[name] = descriptor['fingerprint']
                if name == 'run-v3':
                    assert env.combat_observation_spec()['fingerprint'] == release['fingerprints']['view-v1']
            finally: env.close()
        for mode in ('standalone', 'run'):
            config = load_neural_config(Path('/learner/configs/neural/gym-20261102')/(mode+'.toml'))
            assert config.environment_revision == release['revision']
            driver = make_driver(config, config.cases)
            environments.append(driver._provider._env)
            try:
                schema = driver.schema
                plan = feature_contract(schema)
                assert plan.schema_fingerprint == CURRENT_CONTRACT.common_fingerprint
                assert schema.tensors['combat_cards'].shape[0] == 1024
                assert schema.tensors['known_draw'].shape[0] == 1024
                state = {column.field: column for column in schema.tensors['combat_state'].columns}
                # Ironclad publishes phases in lifecycle metadata, not actor columns.
                assert set(state) == {'facing', 'surrounded', 'heart_damage_remaining', 'heart_cap'}
                assert schema.tensors['known_draw'].columns[-1].reference == 'combat_cards'
                adapter_checks.append({'mode': mode, 'config_sha256': config.config_sha256,
                    'batch_size': driver.batch_size, 'schema_revision': schema.gym_revision,
                    'fingerprint': schema.schema_fingerprint, 'feature_groups': len(plan.groups),
                    'frame_bytes': sum((8 if tensor.dtype == 'int64' else 1)*__import__('math').prod(tensor.shape) for tensor in schema.tensors.values())})
            finally: driver.close()

    expected = release['fingerprints']
    assert fingerprints == {name: expected[name] for name in fingerprints}
    metadata = card_metadata.load()
    assert metadata['schema_id'] == 'spire-lab-card-metadata-v1'
    assert metadata['fingerprint'] == CURRENT_CONTRACT.card_metadata_fingerprint
    body = {k: v for k, v in metadata.items() if k != 'fingerprint'}
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(b'spire-lab/card-metadata/v1\0'+canonical).hexdigest() == metadata['fingerprint']
    with Path('/output/card-metadata.json').open('x') as f:
        json.dump(metadata, f, sort_keys=True, separators=(',', ':')); f.write('\n')
    assert not denied and not attempted_imports
    assert len(close_counts) == 7 and set(close_counts.values()) == {1}
    assert not any(name.split('.')[0] in {'torch', 'torchrl', 'tensordict'} for name in sys.modules)
    versions = {name: importlib.metadata.version(name) for name in ('spire-lab', 'numpy', 'gymnasium', 'torch')}
    assert versions == {'spire-lab': '0.1.0', 'numpy': '2.5.4', 'gymnasium': '1.4.0', 'torch': '2.14.1+cu130'}
    record = {'status': 'qualified_zero_gameplay', 'build_info': info, 'native_file': str(native),
        'native_sha256': release['native_sha256'], 'versions': versions, 'fingerprints': fingerprints,
        'adapter_checks': adapter_checks, 'card_metadata_fingerprint': metadata['fingerprint'],
        'card_metadata_rows': len(metadata['cards']), 'close_counts': close_counts,
        'denied_gameplay': denied, 'training_import_attempts': attempted_imports,
        'reset_calls': 0, 'step_calls': 0, 'samples': 0, 'optimizer_steps': 0, 'uid': os.getuid()}
    with Path('/output/qualification.json').open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps(record, sort_keys=True))
    print('SPIRE_LEARN_SUSTAINED_ADMISSION_V1 '+json.dumps({
        'context_sha256': os.environ['MIGRATION_CONTEXT_SHA256'],
        'job_id': 'gym-migration-metadata', 'runtime_class': 'cpu'}), file=sys.stderr)

if __name__ == "__main__":
    main()
