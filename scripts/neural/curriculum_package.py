"""Pure immutable inputs for the approved standalone campaign; no runtime imports."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import shutil

if __package__ in (None, ''):
    _root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_root / 'src'), str(_root)]

CAMPAIGN_PATH = 'configs/neural/experiments/curriculum-v1/campaign.json'
SCHEMA = 'spire-learn-curriculum-package-v1'
CAMPAIGN_ID = 'curriculum-20261101-a'
MAX_BYTES = 256 * 1024**2
TWENTY_HOUR_MAX_BYTES = 512 * 1024**2
TWENTY_HOUR_PARENT = {
    'checkpoint_directory': (
        'artifacts/development/neural-combat/curriculum-campaign-v6/throughput-switch-1/runtime/jobs/'
        'curriculum-opt-mixed-099/output/20261103T023604Z-731b4fa6201f4ec6aac1cbb0d2981d6e/checkpoint-final'),
    'inventory': {
        'metadata.json': {'sha256': '0ef0363ad9742d1adf9da3cc1fb1590c07f6ab6c57db3ae0b98893a9f8c53ded',
                          'size_bytes': 62253},
        'model.pt': {'sha256': 'fcd172896fa2deae9ca5724ccd8a09242d3b4ea78eede92cfb7752e1ba3861a1',
                     'size_bytes': 1094976},
        'training.pt': {'sha256': '0a5b1c4b9e850d336a776e5aa5165561e729e94a38471ad0de45ad194e80264f',
                        'size_bytes': 2223285},
    },
    'warm_start_purpose': 'weights_only', 'exact_resume': False,
    'parent_campaign_id': 'curriculum-20261102-c',
    'parent_source_sha256': 'a05d631fe84f9327ee90fd1c0a287dfdf538d81e63e035eb46321cd5759e4ef6',
    'parent_package_sha256': '13a8377efa88f4850e1697450dc6a0b38949d7466487709929d76749beb1aa83',
    'inherited_exposure': {'mixed_decisions': 405504, 'accepted_adam_steps': 25344, 'ppo_updates': 792},
    'new_operation_state': {'adam': 'fresh',
        'rng_domains': ['policy', 'minibatch', 'curriculum', 'controller'],
        'operation_counters': 'zero'},
}

from scripts.neural import curriculum_profiles as profiles


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, maximum=4 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('unsafe or oversized JSON input')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def relative_file(base: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or str(Path(name)) != name):
        raise ValueError('input path must be normalized and relative')
    path = Path(base) / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink input path')
    return path


def inventory(base: Path, names) -> dict:
    result = {}
    for name in names:
        path = relative_file(base, name)
        if not path.is_file():
            raise ValueError('inventory input missing')
        result[name] = {'sha256': sha256(path), 'size_bytes': path.stat().st_size}
    return result


def verify_inventory(base: Path, bindings: Mapping) -> None:
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError('inventory must be nonempty')
    for name, binding in bindings.items():
        if (not isinstance(binding, Mapping) or set(binding) != {'sha256', 'size_bytes'}
                or type(binding['size_bytes']) is not int or binding['size_bytes'] < 0
                or not isinstance(binding['sha256'], str) or len(binding['sha256']) != 64):
            raise ValueError('inventory binding malformed')
        if inventory(base, [name])[name] != binding:
            raise ValueError('inventory input changed: ' + name)


def _put(base, name, value, *, raw=False):
    from spire_learn.neural.scenario_support import canonical_bytes
    path = relative_file(base, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value if raw else canonical_bytes(value) + b'\n')


def _config_row(base, name, *, config=None, stable_training_sha256=None):
    from spire_learn.neural.config import load_neural_config, _canonical_hash, _stream_runtime_projection
    from spire_learn.neural.scenario_support import content_sha256
    from spire_learn.neural.stream_resume import _stable_document
    raw = tomllib.loads((Path(base) / name).read_text())
    if config is None and stable_training_sha256 is None:
        config = load_neural_config(Path(base) / name)
    manifest_name = str(Path(name).parent / raw['manifest_path'])
    config_sha = config.config_sha256 if config is not None else _canonical_hash(raw)
    manifest_sha = (config.manifest_sha256 if config is not None else
                    content_sha256(read_json(Path(base) / manifest_name)))
    row = {'config_path': name, 'config_sha256': config_sha,
           'manifest_sha256': manifest_sha,
           'inputs': [name, manifest_name]}
    if 'scenario_stream' in raw:
        if config is not None:
            stable_training_sha256 = content_sha256({
                'source': _stable_document(config.scenario_stream['config_document']),
                'runtime': _stream_runtime_projection(config)})
        if not isinstance(stable_training_sha256, str):
            raise ValueError('scenario stream requires its stable loader identity')
        row.update(shard_index=raw['scenario_stream']['shard_index'],
                   stable_training_sha256=stable_training_sha256)
        row['inputs'] += [str(Path(name).parent / 'recipe.json'),
                          str(Path(name).parent / 'support.json')]
    return row


def _coverage_summary_from_cases(shards, indexes, recipe, *, require_aligned=True):
    """Compute the four-shard coverage receipt from the actual generated cases."""
    from spire_learn.neural.scenario_stream import _quotas
    if (len(shards) != 4 or len(indexes) != 4
            or indexes != list(range(indexes[0], indexes[0] + 4))
            or require_aligned and indexes[0] % 4):
        raise ValueError('coverage requires four aligned consecutive shards')
    entries, family_counts = [], Counter()
    for cases, index in zip(shards, indexes):
        if not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError('coverage shard must contain cases')
        entries.extend(case['entry'] for case in cases)
        run_like, lesson, unusual = _quotas(len(cases), index, 'mixed')
        family_counts.update(dict(run_like=run_like, lesson=lesson, unusual=unusual))
    forms = {(row['card'], row['upgrades']) for entry in entries for row in entry['deck']}
    required = {(card, level) for card in recipe.support.cards for level in (0, 1)}
    encounter_ids = {entry['encounter'] for entry in entries}
    broad = required <= forms and set(recipe.support.encounters) <= encounter_ids
    if not broad:
        raise ValueError('coverage cycle lacks supported base/upgraded cards or encounters')
    return dict(schema='spire-learn-scenario-coverage-v1', broad_coverage=True,
        arm='mixed', split='train', exposure='generated_only', entry_count=len(entries),
        card_count=len({card for card, _ in forms}), base_upgraded_count=len(required & forms),
        encounter_count=len(encounter_ids), family_counts=dict(sorted(family_counts.items())),
        pair_count=len({tuple(sorted((a, b))) for entry in entries
            for a in {row['card'] for row in entry['deck']}
            for b in {row['card'] for row in entry['deck']} if a != b}),
        composition_scope='novel_composition', historical_pretraining_exposure='unknown')


def _training_document(template, campaign, arm, index, distribution):
    document = json.loads(json.dumps(template))
    document.update(manifest_path=f'shard-{index}.json', output_root='runs', device='cuda:0')
    document['runtime'] = {'gpu_uuid': campaign['runtime']['gpu_uuid']}
    document['rollout'] = {'rows': 16, 'steps': 32}
    document['budgets'] = campaign['training_budgets']
    document['scenario_stream'] = {'schema': 'spire-learn-scenario-stream-v1',
        'recipe_path': 'recipe.json', 'arm': arm, 'shard_index': index,
        'distribution_sha256': distribution}
    if (profiles.is_twenty_hour_continuation()
            and index == campaign['training_start_indices']['mixed']):
        document['scenario_stream']['resume_skip'] = json.dumps(
            campaign['continuation']['resume_skip_authorization'], sort_keys=True,
            separators=(',', ':'), allow_nan=False)
    return document


def _package_byte_limit():
    return TWENTY_HOUR_MAX_BYTES if (profiles.is_twenty_hour_campaign()
        or profiles.is_twenty_hour_continuation()) else MAX_BYTES


def _author_twenty_hour_qualification(root: Path) -> Mapping:
    """Author a disjoint held GPU qualification stream and its input mirror."""
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.scenario_support import json_value
    from spire_learn.neural.scenario_stream import generate_scenario_shard, load_scenario_recipe
    from spire_learn.neural.stream_runner import _toml

    selected = profiles.current()
    campaign = read_json(root / selected.campaign_path)
    source = profiles.qualification_source(root)
    target = root / selected.qualification_input
    identity_path = (root / selected.campaign_path).parent / 'qualification-identities.json'
    if source.exists() or target.exists() or identity_path.exists():
        if (source.is_symlink() or target.is_symlink() or identity_path.is_symlink()
                or not source.is_dir() or not target.is_dir() or not identity_path.is_file()):
            raise ValueError('twenty-hour qualification source/input are partially authored')
        source_names = sorted(p.name for p in source.iterdir())
        target_names = sorted(p.name for p in target.iterdir())
        if (any(p.is_symlink() or not p.is_file() for p in source.iterdir())
                or any(p.is_symlink() or not p.is_file() for p in target.iterdir())
                or source_names != target_names):
            raise ValueError('twenty-hour qualification source/input inventories differ')
        bindings = inventory(source, source_names)
        verify_inventory(target, bindings)
        identities = read_json(identity_path)
        campaign = read_json(root / selected.campaign_path)
        expected = campaign['qualification']['stream_start_index']
        configs = [load_neural_config(target / f'qualification-{i}.toml') for i in (0, 1)]
        expected_seeds = campaign['qualification'].get('seeds',
            dict(policy=20261117, optimizer=20261119, curriculum=20261123, noncombat=20261129))
        if (identities != [config.config_sha256 for config in configs]
                or [config.scenario_stream['shard_index'] for config in configs] != [expected, expected + 1]
                or any(dict(policy=config.seed_streams.policy, optimizer=config.seed_streams.optimizer,
                    curriculum=config.seed_streams.curriculum, noncombat=config.seed_streams.noncombat)
                    != expected_seeds for config in configs)
                or any(config.environment_revision != selected.revision or
                       (config.rollout_rows, config.rollout_steps) != (4, 16) for config in configs)):
            raise ValueError('existing twenty-hour qualification identity/loader projection differs')
        return dict(source=str(source.relative_to(root)), input=str(target.relative_to(root)),
            identities=identities, inventory=bindings, stream_indices=[expected, expected + 1])
    source.mkdir(parents=True)
    parent = read_json(root / selected.parents)['standalone_b']
    recipe = load_scenario_recipe(root / 'configs/neural/curriculum-v1/recipe.json')
    recipe_document = json_value(recipe.payload)
    recipe_document['support_file'] = 'support.json'
    _put(source, 'recipe.json', recipe_document)
    _put(source, 'support.json', recipe.support.payload)
    _put(source, 'pilot-parents.json', dict(schema='spire-learn-curriculum-pilot-parents-v1',
        standalone_b=parent))

    template = tomllib.loads((root / selected.template).read_text())
    qualification_seeds = campaign['qualification'].get('seeds',
        dict(policy=20261117, optimizer=20261119, curriculum=20261123, noncombat=20261129))
    shard_indices = []
    start = campaign['qualification']['stream_start_index']
    for ordinal, shard_index in enumerate((start, start + 1)):
        shard = generate_scenario_shard(recipe, split='train', arm='mixed', shard_index=shard_index)
        _put(source, f'shard-{ordinal}.json', shard.manifest)
        document = json.loads(json.dumps(template))
        document.update(manifest_path=f'shard-{ordinal}.json', output_root='runs', device='cuda:0')
        document['seeds'] = qualification_seeds
        document['rollout'] = {'rows': 4, 'steps': 16}
        document['budgets'] = dict(max_optimizer_steps=8, max_training_decisions=512,
            max_environment_actions=512, max_wall_seconds=120.0)
        document['runtime'] = {'gpu_uuid': campaign['runtime']['gpu_uuid']}
        document['scenario_stream'] = dict(schema='spire-learn-scenario-stream-v1',
            recipe_path='recipe.json', arm='mixed', shard_index=shard_index,
            distribution_sha256=shard.distribution_sha256)
        name = f'qualification-{ordinal}.toml'
        _put(source, name, _toml(document).encode(), raw=True)
        shard_indices.append(shard_index)
    target.mkdir(parents=True)
    for path in sorted(p for p in source.iterdir() if p.is_file()):
        shutil.copyfile(path, target / path.name)
    identities = []
    for ordinal, shard_index in enumerate(shard_indices):
        config = load_neural_config(target / f'qualification-{ordinal}.toml')
        if (config.scenario_stream['shard_index'] != shard_index
                or config.environment_revision != selected.revision
                or config.rollout_rows != 4 or config.rollout_steps != 16):
            raise ValueError('authored qualification loader projection differs')
        identities.append(config.config_sha256)
    _put(identity_path.parent, identity_path.name, identities)
    bindings = inventory(source, sorted(p.name for p in source.iterdir() if p.is_file()))
    verify_inventory(target, bindings)
    return dict(source=str(source.relative_to(root)), input=str(target.relative_to(root)),
        identities=identities, inventory=bindings, stream_indices=[start, start + 1])


def _author_twenty_hour_package(root: Path, destination: Path) -> Mapping:
    """Publish the complete finite new-stream package for the authorized run."""
    from spire_learn.neural.scenario_support import content_sha256, json_value
    from spire_learn.neural.scenario_stream import (generate_scenario_shard,
        load_scenario_recipe)
    from spire_learn.neural.stream_runner import _toml

    selected = profiles.current()
    if destination.exists() or destination.is_symlink():
        raise ValueError('twenty-hour package destination must be absent')
    if not destination.parent.is_dir() or any(p.is_symlink() for p in destination.parents):
        raise ValueError('twenty-hour package parent must be regular and existing')
    campaign = read_json(root / selected.campaign_path)
    if campaign['campaign_id'] != selected.campaign_id:
        raise ValueError('twenty-hour campaign identity differs')
    qualification = _author_twenty_hour_qualification(root)
    recipe = load_scenario_recipe(root / 'configs/neural/curriculum-v1/recipe.json')
    template = tomllib.loads((root / selected.template).read_text())
    parent = read_json(root / selected.parents)['standalone_b']
    if profiles.is_twenty_hour_campaign() and parent != TWENTY_HOUR_PARENT:
        raise ValueError('twenty-hour campaign parent differs from the user-approved checkpoint')
    if profiles.is_twenty_hour_continuation():
        continuation = campaign.get('continuation', {})
        if (parent.get('exact_resume') is not True
                or parent.get('warm_start_purpose') != 'exact_resume'
                or parent.get('parent_campaign_id') != continuation.get('parent_campaign_id')
                or parent.get('parent_source_sha256') != continuation.get('parent_source_sha256')
                or parent.get('parent_package_sha256') != continuation.get('parent_package_sha256')
                or parent.get('parent_shard_index') != continuation.get('parent_shard_index')
                or parent.get('parent_next_shard_index') != continuation.get('parent_next_shard_index')
                or parent.get('parent_receipt_sha256') != continuation.get('parent_receipt_sha256')
                or content_sha256(parent.get('inventory')) != continuation.get('parent_checkpoint_inventory_sha256')
                or parent.get('inherited_exposure') != continuation.get('inherited_exposure')):
            raise ValueError('continuation exact-resume parent contract differs')
    continuity = campaign['development_panel']
    source_package_path = root / continuity['source_package_path']
    if sha256(source_package_path) != continuity['source_package_sha256']:
        raise ValueError('continuity development package source changed')
    source_package = read_json(source_package_path)
    panel_names = sorted(name for name in source_package['inventory'] if name.startswith('panel/'))
    if (source_package['panel']['path'] != continuity['manifest_path']
            or source_package['inventory'][continuity['manifest_path']]['sha256'] != continuity['manifest_sha256']
            or len(source_package['panel']['cases']) != continuity['case_count']):
        raise ValueError('continuity development panel source differs')
    verify_inventory(source_package_path.parent,
        {name: source_package['inventory'][name] for name in panel_names})

    destination.mkdir()
    _put(destination, 'campaign.json', campaign)
    _put(destination, 'original-parent.json', parent)
    arm, count = 'mixed', campaign['training_shards']['mixed']
    start = campaign['training_start_indices']['mixed']
    directory = f'training/{arm}'
    recipe_document = json_value(recipe.payload)
    recipe_document['support_file'] = 'support.json'
    _put(destination, directory + '/recipe.json', recipe_document)
    _put(destination, directory + '/support.json', recipe.support.payload)
    training, coverage = {arm: []}, {arm: []}
    stable_training_sha256 = None
    for offset in range(0, count, 4):
        cycle = [generate_scenario_shard(recipe, split='train', arm=arm,
            shard_index=index) for index in range(start + offset, start + offset + 4)]
        coverage[arm].append(_coverage_summary_from_cases(
            [shard.manifest['cases'] for shard in cycle], [shard.shard_index for shard in cycle], recipe,
            require_aligned=not profiles.is_twenty_hour_continuation()))
        for shard in cycle:
            index = shard.shard_index
            _put(destination, f'{directory}/shard-{index}.json', shard.manifest)
            document = _training_document(template, campaign, arm, index, shard.distribution_sha256)
            name = f'{directory}/config-{index}.toml'
            _put(destination, name, _toml(document).encode(), raw=True)
            row = _config_row(destination, name, stable_training_sha256=stable_training_sha256)
            stable_training_sha256 = row['stable_training_sha256']
            training[arm].append(row)

    for name in panel_names:
        target = relative_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative_file(source_package_path.parent, name), target)
    panel = dict(path=source_package['panel']['path'], cases=source_package['panel']['cases'],
        source_package_sha256=continuity['source_package_sha256'],
        source_manifest_sha256=continuity['manifest_sha256'], repeated_exposure=True)
    names = sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file())
    bindings = inventory(destination, names)
    size = sum(row['size_bytes'] for row in bindings.values())
    result = dict(schema=SCHEMA, campaign_id=selected.campaign_id, inventory=bindings,
        total_input_bytes=size, campaign=campaign, original_parent=parent,
        recipe_sha256=recipe.sha256, support_sha256=recipe.support.sha256,
        training=training, coverage=coverage, panel=panel,
        evaluation_slots=campaign['evaluation_slots'], qualification=qualification)
    if profiles.is_twenty_hour_continuation():
        result['continuation'] = campaign['continuation']
    _put(destination, 'package.json', result)
    if size + (destination / 'package.json').stat().st_size > _package_byte_limit():
        raise ValueError('twenty-hour package exceeds its byte cap')
    return result


def author_package(root: Path, destination: Path, *, campaign_id: str) -> Mapping:
    """Publish all reviewed inputs exclusively; actual loaders check every config."""
    from spire_learn.neural.scenario_stream import (load_scenario_recipe,
        generate_scenario_shard, validate_coverage_cycle)
    from spire_learn.neural.scenario_support import content_sha256, json_value
    from spire_learn.neural.stream_runner import _toml, build_development_panel
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if campaign_id != profiles.current().campaign_id:
        raise ValueError('campaign identity differs')
    if profiles.is_twenty_hour_campaign() or profiles.is_twenty_hour_continuation():
        return _author_twenty_hour_package(root, destination)
    if profiles.current().fixed:
        return refresh_package(root, Path(root)/REUSABLE_PACKAGE, destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('destination must be absent')
    if not destination.parent.is_dir() or any(p.is_symlink() for p in destination.parents):
        raise ValueError('destination parent must be regular and existing')
    campaign = read_json(root / CAMPAIGN_PATH)
    if campaign['campaign_id'] != campaign_id:
        raise ValueError('authored campaign identity differs')
    recipe = load_scenario_recipe(root / 'configs/neural/curriculum-v1/recipe.json')
    template = tomllib.loads((root / 'configs/neural/curriculum-v1/pilot.toml').read_text())
    destination.mkdir()
    _put(destination, 'campaign.json', campaign)
    parents = read_json(root / 'configs/neural/curriculum-v1/pilot-parents.json')['standalone_b']
    _put(destination, 'original-parent.json', parents)
    training, coverage = {}, {}
    for arm, count in campaign['training_shards'].items():
        directory = f'training/{arm}'
        recipe_document = json_value(recipe.payload)
        recipe_document['support_file'] = 'support.json'
        _put(destination, directory + '/recipe.json', recipe_document)
        _put(destination, directory + '/support.json', recipe.support.payload)
        training[arm], coverage[arm] = [], []
        for start in range(0, count, 4):
            cycle = [generate_scenario_shard(recipe, split='train', arm=arm, shard_index=i)
                     for i in range(start, start + 4)]
            coverage[arm].append(json_value(validate_coverage_cycle(cycle, recipe)))
            for shard in cycle:
                index = shard.shard_index
                _put(destination, f'{directory}/shard-{index}.json', shard.manifest)
                document = _training_document(template, campaign, arm, index, shard.distribution_sha256)
                name = f'{directory}/config-{index}.toml'
                _put(destination, name, _toml(document).encode(), raw=True)
                training[arm].append(_config_row(destination, name))
    panel = build_development_panel(recipe)
    _put(destination, 'panel/development-panel.json', panel)
    rows = []
    for ordinal, case in enumerate(panel['manifest']['cases']):
        document = json.loads(json.dumps(template))
        document.update(operation='evaluate', split='selection', partition='selection',
                        manifest_path=f'case-{ordinal}.json', output_root='runs', device='cpu')
        document['rollout'] = {'rows': 1, 'steps': 1}
        document['budgets'] = dict(max_training_decisions=256, max_environment_actions=256,
                                   max_optimizer_steps=1, max_wall_seconds=30.0)
        manifest = {**panel['manifest'], 'cases': [case]}
        _put(destination, f'panel/case-{ordinal}.json', manifest)
        name = f'panel/config-{ordinal}.toml'
        _put(destination, name, _toml(document).encode(), raw=True)
        rows.append({**_config_row(destination, name), 'case_id': case['case_id'],
                     'environment_seed': case['environment_seed']})
    names = sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file())
    bindings = inventory(destination, names)
    size = sum(row['size_bytes'] for row in bindings.values())
    if size > _package_byte_limit():
        raise ValueError('package exceeds its byte cap')
    result = dict(schema=SCHEMA, campaign_id=campaign_id, inventory=bindings,
        total_input_bytes=size, campaign=campaign, original_parent=parents,
        recipe_sha256=recipe.sha256, support_sha256=recipe.support.sha256,
        training=training, coverage=coverage,
        panel={'path': 'panel/development-panel.json', 'cases': rows},
        evaluation_slots=campaign['evaluation_slots'])
    _put(destination, 'package.json', result)
    if size + (destination / 'package.json').stat().st_size > _package_byte_limit():
        raise ValueError('package including manifest exceeds its byte cap')
    return result


def validate_twenty_hour_package(root: Path, package_path: Path) -> Mapping:
    """Rehash the full finite package and round-trip every actual config."""
    from spire_learn.neural.scenario_support import content_sha256, json_value
    from spire_learn.neural.scenario_stream import load_scenario_recipe
    from spire_learn.neural.stream_runner import _toml
    from spire_learn.neural.config import load_neural_config

    root, package_path = Path(root).resolve(), Path(package_path)
    selected = profiles.current()
    package = read_json(package_path)
    campaign = read_json(root / selected.campaign_path)
    continuation_run = profiles.is_twenty_hour_continuation()
    base = package_path.parent
    if (package.get('schema') != SCHEMA or package.get('campaign_id') != selected.campaign_id
            or package.get('campaign') != campaign
            or package.get('evaluation_slots') != campaign['evaluation_slots']
            or (continuation_run and package.get('continuation') != campaign.get('continuation'))):
        raise ValueError('twenty-hour package identity/campaign differs')
    verify_inventory(base, package['inventory'])
    emitted = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    if emitted != set(package['inventory']) | {'package.json'}:
        raise ValueError('twenty-hour package inventory is not closed')
    size = sum(row['size_bytes'] for row in package['inventory'].values())
    if (size != package['total_input_bytes']
            or size + package_path.stat().st_size > _package_byte_limit()):
        raise ValueError('twenty-hour package size differs')

    parent = read_json(root / selected.parents)['standalone_b']
    if continuation_run:
        continuation = campaign['continuation']
        expected_baseline = dict(label='continuation-parent-shard337',
            selection='last fully reconciled optimized mixed checkpoint before the unreconciled shard338 lease',
            initialization='exact resume model, Adam, RNG domains, sampler state, and operation counters; shard338 is skipped and its full lease remains charged',
            parent_campaign_id=continuation['parent_campaign_id'],
            parent_source_sha256=continuation['parent_source_sha256'],
            parent_package_sha256=continuation['parent_package_sha256'],
            parent_receipt_sha256=continuation['parent_receipt_sha256'],
            inherited_exposure=continuation['inherited_exposure'])
        parent_matches = (parent == read_json(root / selected.parents)['standalone_b']
            and parent.get('exact_resume') is True
            and content_sha256(parent.get('inventory')) == continuation.get('parent_checkpoint_inventory_sha256'))
    else:
        expected_baseline = dict(label='warm-start-mixed-099',
            selection='completed optimized mixed checkpoint, selected in the prior campaign',
            initialization='load parent weights only; create fresh Adam, behavior version, RNG domains, and zero operation counters; not exact resume',
            parent_campaign_id=TWENTY_HOUR_PARENT['parent_campaign_id'],
            parent_source_sha256=TWENTY_HOUR_PARENT['parent_source_sha256'],
            parent_package_sha256=TWENTY_HOUR_PARENT['parent_package_sha256'],
            inherited_exposure=TWENTY_HOUR_PARENT['inherited_exposure'])
        parent_matches = parent == TWENTY_HOUR_PARENT
    if (not parent_matches or campaign.get('baseline') != expected_baseline
            or package.get('original_parent') != parent
            or read_json(base / 'original-parent.json') != parent):
        raise ValueError('twenty-hour parent/lineage differs')
    recipe = load_scenario_recipe(root / 'configs/neural/curriculum-v1/recipe.json')
    if (package.get('recipe_sha256'), package.get('support_sha256')) != (recipe.sha256, recipe.support.sha256):
        raise ValueError('twenty-hour scenario recipe/support differs')
    if set(package['training']) != {'mixed'} or set(campaign['training_shards']) != {'mixed'}:
        raise ValueError('twenty-hour package includes an unapproved training arm')
    rows = package['training']['mixed']
    count, start = campaign['training_shards']['mixed'], campaign['training_start_indices']['mixed']
    if len(rows) != count or start <= 255 or start + count > 2**24 - 1:
        raise ValueError('twenty-hour finite new stream bounds differ')
    template = tomllib.loads((root / selected.template).read_text())
    if template.get('seeds') != dict(policy=20261103, optimizer=20261107,
            curriculum=20261111, noncombat=20261113):
        raise ValueError('twenty-hour training seed streams differ')
    training_recipe = load_scenario_recipe(base / 'training/mixed/recipe.json')
    if (training_recipe.sha256, training_recipe.support.sha256) != (recipe.sha256, recipe.support.sha256):
        raise ValueError('twenty-hour package recipe/support bytes differ')
    stable = set()
    for offset, row in enumerate(rows):
        index = start + offset
        name = f'training/mixed/config-{index}.toml'
        raw = tomllib.loads((base / name).read_text())
        config = load_neural_config(base / name)
        if (raw != _training_document(template, campaign, 'mixed', index,
                raw.get('scenario_stream', {}).get('distribution_sha256'))
                or row.get('shard_index') != index or row != _config_row(base, name, config=config)):
            raise ValueError('twenty-hour training config/index loader differs')
        if (config.scenario_stream['shard_index'] != index or config.scenario_stream['arm'] != 'mixed'
                or (config.rollout_rows, config.rollout_steps) != (16, 32)):
            raise ValueError('twenty-hour B16 training config differs')
        stable.add(row['stable_training_sha256'])
    if len(stable) != 1:
        raise ValueError('twenty-hour training stream stable identity differs')
    if continuation_run:
        prior_path = (root/'artifacts/development/neural-combat/twenty-hour-20261103/'
            'inputs/package-a/package.json')
        prior = read_json(prior_path)
        prior_start = prior['campaign']['training_start_indices']['mixed']
        prior_offset = 337-prior_start
        if (sha256(prior_path) != campaign['continuation']['parent_package_sha256']
                or prior_offset < 0 or prior_offset >= len(prior['training']['mixed'])
                or prior['training']['mixed'][prior_offset].get('shard_index') != 337
                or stable != {prior['training']['mixed'][prior_offset]['stable_training_sha256']}):
            raise ValueError('continuation training identity differs from healthy shard337')
    expected_coverage = []
    for offset in range(0, count, 4):
        indexes = list(range(start + offset, start + offset + 4))
        manifests = [read_json(base / f'training/mixed/shard-{index}.json')['cases'] for index in indexes]
        expected_coverage.append(_coverage_summary_from_cases(manifests, indexes, recipe,
            require_aligned=not continuation_run))
    if package.get('coverage', {}).get('mixed') != expected_coverage:
        raise ValueError('twenty-hour four-shard coverage cycles differ')

    continuity = campaign['development_panel']
    source_package_path = root / continuity['source_package_path']
    if sha256(source_package_path) != continuity['source_package_sha256']:
        raise ValueError('continuity development source package changed')
    source_package = read_json(source_package_path)
    panel_names = sorted(name for name in source_package['inventory'] if name.startswith('panel/'))
    expected_panel = dict(path=source_package['panel']['path'], cases=source_package['panel']['cases'],
        source_package_sha256=continuity['source_package_sha256'],
        source_manifest_sha256=continuity['manifest_sha256'], repeated_exposure=True)
    if (package.get('panel') != expected_panel
            or package['panel']['path'] != continuity['manifest_path']
            or len(package['panel']['cases']) != continuity['case_count']):
        raise ValueError('repeated development panel lineage differs')
    panel_bindings = {name: source_package['inventory'][name] for name in panel_names}
    verify_inventory(base, panel_bindings)
    panel = read_json(base / package['panel']['path'])
    if panel != read_json(source_package_path.parent / package['panel']['path']):
        raise ValueError('repeated development manifest changed')
    for ordinal, row in enumerate(package['panel']['cases']):
        case = panel['manifest']['cases'][ordinal]
        name = row['config_path']
        config = load_neural_config(base / name)
        if (row != {**_config_row(base, name), 'case_id': case['case_id'],
                    'environment_seed': case['environment_seed']}
                or config.environment_revision != selected.revision
                or (config.operation, config.split, config.device, config.rollout_rows, config.rollout_steps)
                   != ('evaluate', 'selection', 'cpu', 1, 1)
                or config.scenario_stream is not None):
            raise ValueError('repeated development case/config differs')
        if read_json(base / f'panel/case-{ordinal}.json') != {**panel['manifest'], 'cases': [case]}:
            raise ValueError('repeated development case manifest differs')

    qualification_source = profiles.qualification_source(root)
    qualification_input = root / selected.qualification_input
    qualification_names = sorted(p.name for p in qualification_source.iterdir() if p.is_file())
    qualification_inventory = inventory(qualification_source, qualification_names)
    verify_inventory(qualification_input, qualification_inventory)
    qual_configs = [load_neural_config(qualification_input / f'qualification-{i}.toml') for i in (0, 1)]
    qual_indices = [config.scenario_stream['shard_index'] for config in qual_configs]
    expected_start = campaign['qualification']['stream_start_index']
    expected_qualification_seeds = campaign['qualification'].get('seeds',
        dict(policy=20261117, optimizer=20261119, curriculum=20261123, noncombat=20261129))
    if (qual_indices != [expected_start, expected_start + 1]
            or [config.config_sha256 for config in qual_configs] != profiles.qualification_hashes(root)
            or any(dict(policy=config.seed_streams.policy, optimizer=config.seed_streams.optimizer,
                curriculum=config.seed_streams.curriculum, noncombat=config.seed_streams.noncombat)
                != expected_qualification_seeds
                for config in qual_configs)):
        raise ValueError('fresh qualification stream/config identities differ')
    if package.get('qualification') != dict(source=str(qualification_source.relative_to(root)),
            input=str(qualification_input.relative_to(root)), identities=profiles.qualification_hashes(root),
            inventory=qualification_inventory, stream_indices=qual_indices):
        raise ValueError('qualification package projection differs')
    return package


def validate_package(root: Path, package_path: Path) -> Mapping:
    """Host preparation check: rehash every input and load every actual config."""
    from spire_learn.neural.scenario_support import json_value
    from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard, validate_coverage_cycle
    from spire_learn.neural.stream_runner import build_development_panel
    if profiles.is_twenty_hour_campaign() or profiles.is_twenty_hour_continuation():
        return validate_twenty_hour_package(root, package_path)
    if profiles.current().fixed:
        return validate_refreshed_package(root, package_path)
    package_path = Path(package_path)
    package = read_json(package_path)
    if package.get('schema') != SCHEMA or package.get('campaign_id') != CAMPAIGN_ID:
        raise ValueError('package identity differs')
    if package['campaign'] != read_json(Path(root) / CAMPAIGN_PATH):
        raise ValueError('package campaign data differs')
    template = tomllib.loads((Path(root) / 'configs/neural/curriculum-v1/pilot.toml').read_text())
    reference_recipe = load_scenario_recipe(Path(root) / 'configs/neural/curriculum-v1/recipe.json')
    if (package['recipe_sha256'], package['support_sha256']) != (reference_recipe.sha256, reference_recipe.support.sha256):
        raise ValueError('package accepted recipe/support differs')
    base = package_path.parent
    verify_inventory(base, package['inventory'])
    emitted = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    if emitted != set(package['inventory']) | {'package.json'}:
        raise ValueError('package inventory is not closed')
    size = sum(v['size_bytes'] for v in package['inventory'].values())
    if size != package['total_input_bytes'] or size + package_path.stat().st_size > MAX_BYTES:
        raise ValueError('package size differs')
    original = read_json(Path(root) / 'configs/neural/curriculum-v1/pilot-parents.json')['standalone_b']
    if package['original_parent'] != original or read_json(base / 'original-parent.json') != original:
        raise ValueError('original parent differs')
    if package['evaluation_slots'] != package['campaign']['evaluation_slots']:
        raise ValueError('evaluation schedule differs')
    if set(package['training']) != {'fixed', 'randomized', 'mixed'}:
        raise ValueError('training arms differ')
    for arm, count in package['campaign']['training_shards'].items():
        rows = package['training'][arm]
        if len(rows) != count:
            raise ValueError('training shard count differs')
        recipe = load_scenario_recipe(base / f'training/{arm}/recipe.json')
        if recipe.sha256 != package['recipe_sha256'] or recipe.support.sha256 != package['support_sha256']:
            raise ValueError('recipe/support identity differs')
        for index, row in enumerate(rows):
            raw = tomllib.loads((base / f'training/{arm}/config-{index}.toml').read_text())
            if raw != _training_document(template, package['campaign'], arm, index,
                                          raw.get('scenario_stream', {}).get('distribution_sha256')):
                raise ValueError('campaign training settings differ')
            if row != _config_row(base, f'training/{arm}/config-{index}.toml'):
                raise ValueError('config loader identity differs')
        if len({row['stable_training_sha256'] for row in rows}) != 1:
            raise ValueError('training stable identity differs')
        expected = [json_value(validate_coverage_cycle([generate_scenario_shard(recipe,
            split='train', arm=arm, shard_index=i) for i in range(start, start + 4)], recipe))
            for start in range(0, count, 4)]
        if package['coverage'][arm] != expected:
            raise ValueError('coverage cycles differ')
    panel = read_json(base / package['panel']['path'])
    if panel != build_development_panel(recipe) or len(package['panel']['cases']) != 64:
        raise ValueError('complete panel identity differs')
    for index, row in enumerate(package['panel']['cases']):
        case = panel['manifest']['cases'][index]
        if row != {**_config_row(base, f'panel/config-{index}.toml'), 'case_id': case['case_id'],
                   'environment_seed': case['environment_seed']}:
            raise ValueError('panel config identity differs')
        if read_json(base / f'panel/case-{index}.json') != {**panel['manifest'], 'cases': [case]}:
            raise ValueError('single-case panel manifest differs')
    return package


REUSABLE_PACKAGE = ('artifacts/development/neural-combat/curriculum-campaign-v1/'
                    'task-1/package-fixture/package.json')


def _reusable(root, path):
    manifest = read_json(Path(root)/profiles.current().overlay_manifest)
    if not profiles.current().fixed or sha256(path) != manifest['reusable_package_sha256']:
        raise ValueError('accepted reusable package identity differs')
    value = read_json(path)
    if value['campaign_id'] != CAMPAIGN_ID:
        raise ValueError('reusable package is not historical campaign data')
    return value


def _project_gym_contract(document, revision):
    """Bind authored current-profile configs to one closed public contract."""
    from spire_learn.neural.gym_contracts import contract_for_revision
    contract = contract_for_revision(revision)
    environment = document['environment']
    environment['gym_revision'] = revision
    environment['common_view_fingerprint'] = contract.common_fingerprint
    if 'run_observation_fingerprint' in environment:
        environment['run_observation_fingerprint'] = contract.run_fingerprint
    for identity in document['accepted_identities']:
        identity['accepted_raw_mode_mappings'] = dict(contract.raw_mode_mappings)


def _refreshed_files(root, source):
    """Project the selected deployment and parent while reusing accepted scenarios."""
    from spire_learn.neural.stream_runner import _toml
    from spire_learn.neural.scenario_support import canonical_bytes
    selected = profiles.current()
    campaign = read_json(Path(root)/selected.campaign_path)
    parent = read_json(Path(root)/selected.parents)['standalone_b']
    result = {'campaign.json': canonical_bytes(campaign)+b'\n',
              'original-parent.json': canonical_bytes(parent)+b'\n'}
    for name in source['inventory']:
        if name.endswith('.toml'):
            directory = (Path(root)/REUSABLE_PACKAGE).parent
            verify_inventory(directory, {name: source['inventory'][name]})
            document = tomllib.loads(directory.joinpath(name).read_text())
            if document['environment']['gym_revision'] != profiles.HISTORICAL.revision:
                raise ValueError('reusable config runtime revision differs')
            if profiles.is_fresh_gym_family():
                _project_gym_contract(document, selected.revision)
                if document.get('operation') == 'train':
                    document['budgets'] = campaign['training_budgets']
            else:
                document['environment']['gym_revision'] = selected.revision
            result[name] = _toml(document).encode()
    return result


def _refreshed_document(root, base, source):
    from spire_learn.neural.scenario_support import content_sha256
    value = json.loads(json.dumps(source))
    value.update(campaign_id=profiles.current().campaign_id,
        campaign=read_json(Path(root)/profiles.current().campaign_path),
        original_parent=read_json(Path(root)/profiles.current().parents)['standalone_b'])
    value['evaluation_slots'] = value['campaign']['evaluation_slots']
    template = tomllib.loads((Path(root)/profiles.current().template).read_text())
    for arm, rows in value['training'].items():
        # One actual loader supplies each arm's stable runtime identity. Every
        # other document must be the exact accepted template transformation;
        # manifests are byte-identical accepted data. The real selected loader
        # also checks every shard again before its first runtime effect.
        stable = _config_row(base, rows[0]['config_path'])['stable_training_sha256']
        for index, row in enumerate(rows):
            document = tomllib.loads((Path(base)/row['config_path']).read_text())
            if document != _training_document(template, value['campaign'], arm, index,
                    document['scenario_stream']['distribution_sha256']):
                raise ValueError('refreshed training settings differ')
            row.update(config_sha256=content_sha256(document), stable_training_sha256=stable)
    for row in value['panel']['cases']:
        document = tomllib.loads((Path(base)/row['config_path']).read_text())
        row['config_sha256'] = content_sha256(document)
    if _config_row(base, value['panel']['cases'][0]['config_path'])['config_sha256'] != value['panel']['cases'][0]['config_sha256']:
        raise ValueError('refreshed panel loader differs')
    value['inventory'] = inventory(base, sorted(source['inventory']))
    value['total_input_bytes'] = sum(row['size_bytes'] for row in value['inventory'].values())
    return value


def refresh_package(root, source_package, destination):
    """Reuse accepted immutable scenario DATA with freshly loaded config identities."""
    root, destination, source_package = Path(root), Path(destination), Path(source_package)
    if not profiles.current().fixed:
        raise ValueError('refresh requires explicit fixed Gym profile')
    source = _reusable(root, source_package)
    verify_inventory(source_package.parent, source['inventory'])
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError('fresh destination with existing regular parent required')
    relative_file(destination.parent, destination.name)  # Reject symlink ancestors.
    destination.mkdir()
    changed = _refreshed_files(root, source)
    for name in source['inventory']:
        target = relative_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in changed:
            target.write_bytes(changed[name])
        else:
            shutil.copyfile(relative_file(source_package.parent, name), target)
    result = _refreshed_document(root, destination, source)
    _put(destination, 'package.json', result)
    if result['total_input_bytes']+(destination/'package.json').stat().st_size > MAX_BYTES:
        raise ValueError('refreshed package exceeds byte cap')
    return result


def validate_refreshed_package(root, package_path):
    """Check deterministic projection and every real loader; no scenario regeneration."""
    root, package_path = Path(root), Path(package_path)
    source = _reusable(root, root/REUSABLE_PACKAGE)
    base = package_path.parent
    package = read_json(package_path)
    verify_inventory(base, package['inventory'])
    if {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()} != set(source['inventory']) | {'package.json'}:
        raise ValueError('refreshed package inventory is not closed')
    changed = _refreshed_files(root, source)
    for name, binding in source['inventory'].items():
        if name in changed:
            if relative_file(base, name).read_bytes() != changed[name]:
                raise ValueError('refreshed config/campaign/parent settings differ')
        elif package['inventory'].get(name) != binding:
            raise ValueError('accepted scenario data identity differs')
    expected = _refreshed_document(root, base, source)
    if package != expected or package['total_input_bytes']+package_path.stat().st_size > MAX_BYTES:
        raise ValueError('refreshed package loader identity differs')
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('author', 'validate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--profile', choices=profiles.NAMES, default=profiles.HISTORICAL.name)
    args = parser.parse_args(argv)
    with profiles.scope(args.profile):
        result = (author_package(args.root, args.path, campaign_id=profiles.current().campaign_id)
                  if args.command == 'author' else validate_package(args.root, args.path))
    print(json.dumps({'schema': result['schema'], 'campaign_id': result['campaign_id'],
                      'total_input_bytes': result['total_input_bytes']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
