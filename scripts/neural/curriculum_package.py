"""Pure immutable inputs for the approved standalone campaign; no runtime imports."""
from __future__ import annotations

import argparse
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


def _config_row(base, name):
    from spire_learn.neural.config import load_neural_config
    from spire_learn.neural.stream_resume import stream_identity
    config = load_neural_config(base / name)
    row = {'config_path': name, 'config_sha256': config.config_sha256,
           'manifest_sha256': config.manifest_sha256,
           'inputs': [name, str(config.manifest_path.relative_to(base))]}
    if config.scenario_stream is not None:
        identity = stream_identity(config)
        row.update(shard_index=config.scenario_stream['shard_index'],
                   stable_training_sha256=identity['stable_training_sha256'])
        row['inputs'] += [str(Path(name).parent / 'recipe.json'),
                          str(Path(name).parent / 'support.json')]
    return row


def _training_document(template, campaign, arm, index, distribution):
    document = json.loads(json.dumps(template))
    document.update(manifest_path=f'shard-{index}.json', output_root='runs', device='cuda:0')
    document['runtime'] = {'gpu_uuid': campaign['runtime']['gpu_uuid']}
    document['rollout'] = {'rows': 16, 'steps': 32}
    document['budgets'] = campaign['training_budgets']
    document['scenario_stream'] = {'schema': 'spire-learn-scenario-stream-v1',
        'recipe_path': 'recipe.json', 'arm': arm, 'shard_index': index,
        'distribution_sha256': distribution}
    return document


def author_package(root: Path, destination: Path, *, campaign_id: str) -> Mapping:
    """Publish all reviewed inputs exclusively; actual loaders check every config."""
    from spire_learn.neural.scenario_stream import (load_scenario_recipe,
        generate_scenario_shard, validate_coverage_cycle)
    from spire_learn.neural.scenario_support import json_value
    from spire_learn.neural.stream_runner import _toml, build_development_panel
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if campaign_id != profiles.current().campaign_id:
        raise ValueError('campaign identity differs')
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
    if size > MAX_BYTES:
        raise ValueError('package exceeds 256 MiB')
    result = dict(schema=SCHEMA, campaign_id=campaign_id, inventory=bindings,
        total_input_bytes=size, campaign=campaign, original_parent=parents,
        recipe_sha256=recipe.sha256, support_sha256=recipe.support.sha256,
        training=training, coverage=coverage,
        panel={'path': 'panel/development-panel.json', 'cases': rows},
        evaluation_slots=campaign['evaluation_slots'])
    _put(destination, 'package.json', result)
    if size + (destination / 'package.json').stat().st_size > MAX_BYTES:
        raise ValueError('package including manifest exceeds 256 MiB')
    return result


def validate_package(root: Path, package_path: Path) -> Mapping:
    """Host preparation check: rehash every input and load every actual config."""
    from spire_learn.neural.scenario_support import json_value
    from spire_learn.neural.scenario_stream import load_scenario_recipe, generate_scenario_shard, validate_coverage_cycle
    from spire_learn.neural.stream_runner import build_development_panel
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


def _refreshed_files(root, source):
    """Only deployment revision, campaign and weight parent change."""
    from spire_learn.neural.stream_runner import _toml
    from spire_learn.neural.scenario_support import canonical_bytes
    campaign = read_json(Path(root)/profiles.current().campaign_path)
    parent = read_json(Path(root)/profiles.current().parents)['standalone_b']
    result = {'campaign.json': canonical_bytes(campaign)+b'\n',
              'original-parent.json': canonical_bytes(parent)+b'\n'}
    for name in source['inventory']:
        if name.endswith('.toml'):
            directory = (Path(root)/REUSABLE_PACKAGE).parent
            verify_inventory(directory, {name: source['inventory'][name]})
            document = tomllib.loads(directory.joinpath(name).read_text())
            if document['environment']['gym_revision'] != profiles.HISTORICAL.revision:
                raise ValueError('reusable config runtime revision differs')
            document['environment']['gym_revision'] = profiles.current().revision
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
