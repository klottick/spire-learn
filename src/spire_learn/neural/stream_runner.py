"""Pure curriculum preparation and explicitly invoked single-segment execution.

Preparation grants no execution authority. No runtime assembly happens on import.
Only the existing runner owns collection, PPO, checkpointing and lifecycle close.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import tomllib
from typing import TYPE_CHECKING

from .cases import _parse_case, environment_seed_for
from .scenario_stream import (
    ScenarioRecipe, _composition_split, generate_scenario_shard,
    load_scenario_recipe, validate_coverage_cycle,
)
from .scenario_support import canonical_bytes, content_sha256, json_value

if TYPE_CHECKING:
    from .config import NeuralConfig
    from .runner import RuntimeOperationLimits

_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = _ROOT / 'configs/neural/curriculum-v1'
_ARTIFACTS = _ROOT / 'artifacts/development/neural-combat'


def pilot_plan(recipe: ScenarioRecipe) -> Mapping[str, object]:
    """Bind authored original parents; do not read weights or imply qualification."""
    parents = json.loads((_CONFIGS / 'pilot-parents.json').read_text())
    template = tomllib.loads((_CONFIGS / 'pilot.toml').read_text())
    initialization = {'purpose': 'warm_start', 'optimizer': 'fresh',
        'behavior_policy_version': 0, 'progress': 'zero', 'rng': 'fresh',
        'seeds': template['seeds']}
    return {'schema': 'spire-learn-curriculum-pilot-v1',
        'execution_authorized': False, 'recipe_sha256': recipe.sha256,
        'parent_inventory_validation': 'required before any separately authorized execution',
        'arms': {arm: {'parent': json_value(parents['standalone_b']),
            'initialization': json_value(initialization), 'budgets': template['budgets']}
            for arm in ('fixed', 'randomized', 'mixed')},
        'frozen_run_reference': parents['run_b'],
        'historical_pretraining_exposure': 'unknown',
        'claim': 'paired initialization; no strength or execution claim'}


def build_development_panel(recipe: ScenarioRecipe) -> Mapping[str, object]:
    """Author 64 cases; reused compositions are explicitly diagnostic cohorts."""
    train = generate_scenario_shard(recipe, split='train', shard_index=0)
    selection = generate_scenario_shard(recipe, split='selection', shard_index=0)
    train_cases = json_value(train.manifest)['cases']
    novel = [c for c in json_value(selection.manifest)['cases']
        if not any(set(pair) <= {r['card'] for r in c['entry']['deck']}
                   for pair in recipe.reserved_pairs)]
    if len(train_cases) < 32 or len(novel) < 16 or not recipe.reserved_pairs:
        raise ValueError('development panel requires 32 training decks, 16 novel decks and reserved pairs')
    cases, annotations, compositions = [], {}, set()
    later = sorted(e['id'] for e in recipe.support.encounters.values() if e['act'] >= 2)
    for ordinal in range(64):
        cohort = ('familiar_new_seed', 'novel_deck', 'reserved_pair', 'resource_later_tier')[ordinal // 16]
        local = ordinal % 16
        if cohort == 'novel_deck':
            source = novel[local]
            family = selection.annotations[source['case_id']]['family']
        else:
            source = train_cases[local if cohort == 'familiar_new_seed' else 16 + local]
            family = train.annotations[source['case_id']]['family']
        entry = json_value(source['entry'])
        if cohort == 'reserved_pair':
            family = 'reserved_pair'
            pair = recipe.reserved_pairs[local % len(recipe.reserved_pairs)]
            rng = random.Random(int(content_sha256(['development-pair-v1', recipe.sha256, local]), 16))
            starter = [{'card': c, 'upgrades': 0} for c in
                       ['strike'] * 4 + ['defend'] * 4 + ['bash']]
            for _ in range(4096):
                deck = starter + [{'card': c, 'upgrades': rng.randrange(2)} for c in pair]
                deck += [{'card': rng.choice(sorted(recipe.support.cards)), 'upgrades': rng.randrange(2)}
                         for _ in range(rng.randint(1, 12))]
                deck.sort(key=lambda r: (r['card'], r['upgrades']))
                if _composition_split(deck) == 'selection' and content_sha256(deck) not in compositions:
                    entry['deck'] = deck
                    break
            else:
                raise ValueError('reserved development deck exhausted bounded construction')
        if cohort == 'resource_later_tier':
            entry.update(encounter=later[local % len(later)],
                         hp=24 if local % 2 == 0 else entry['max_hp'],
                         potions=[recipe.support.potions[local % len(recipe.support.potions)], None, None])
        composition = content_sha256(sorted(entry['deck'], key=lambda r: (r['card'], r['upgrades'])))
        if composition in compositions:
            raise ValueError('development panel composition collision')
        compositions.add(composition)
        case_id = 'dev-' + content_sha256([recipe.sha256, cohort, ordinal, entry])[:48]
        entry['id'] = case_id
        recipe.support.validate_entry(entry)
        case = {'case_id': case_id, 'lineage_id': f'development:{cohort}:{composition}:{ordinal}',
            'mode': 'standalone', 'environment_seed': environment_seed_for(
                'selection', replica=0, ordinal=2**32 + ordinal),
            'cohort': cohort, 'entry_id': case_id, 'entry': entry}
        _parse_case(case, split='selection')
        encounter = recipe.support.encounters[entry['encounter']]
        annotations[case_id] = {'cohort': cohort, 'composition_sha256': composition,
            'composition_scope': ('novel_composition' if cohort in ('novel_deck', 'reserved_pair')
                                  else 'seen_composition_diagnostic'),
            'family': family,
            'deck_stage': ('early' if len(entry['deck']) <= 12 else
                           'middle' if len(entry['deck']) <= 23 else 'developed'),
            'encounter': entry['encounter'],
            'encounter_tier': f"act{encounter['act']}:{encounter['kind']}",
            'cards': sorted({r['card'] for r in entry['deck']})}
        cases.append(case)
    manifest = {'schema': 'spire-learn-neural-cases-v1', 'split': 'selection',
                'partition': 'selection', 'cases': cases}
    return {'schema': 'spire-learn-development-panel-v1', 'recipe_sha256': recipe.sha256,
        'manifest': manifest, 'manifest_sha256': content_sha256(manifest),
        'annotations': annotations, 'historical_pretraining_exposure': 'unknown',
        'exclusion_scope': 'new experiment only; familiar/resource compositions deliberately seen',
        'claim': 'standalone diagnostics; no full-Run or original-game transfer claim'}


def _toml(document: Mapping[str, object]) -> str:
    """Serialize the small existing config schema without adding a dependency."""
    lines = []
    def table(value, prefix=''):
        for key, item in value.items():
            if not isinstance(item, (dict, list)):
                lines.append(f'{key} = {json.dumps(item, allow_nan=False)}')
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(item, dict):
                lines.extend(['', f'[{name}]'])
                table(item, name)
            elif isinstance(item, list):
                for row in item:
                    lines.extend(['', f'[[{name}]]'])
                    table(row, name)
    table(document)
    return '\n'.join(lines) + '\n'


def _preparation(recipe_path: Path, destination: Path, *, arm: str, split: str,
                 shard_index: int) -> tuple[dict[str, bytes], dict[str, object]]:
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('destination must be absent')
    if not destination.resolve().is_relative_to(_ARTIFACTS.resolve()):
        raise ValueError('destination must stay under artifacts/development/neural-combat')
    if not destination.parent.is_dir() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError('destination requires an existing regular parent path')
    if split not in ('train', 'selection'):
        raise ValueError('preparation supports train/selection; final remains frozen')
    if type(shard_index) is not int or not 0 <= shard_index < 2**24 - 1:
        raise ValueError('shard_index must have a representable successor')
    recipe = load_scenario_recipe(recipe_path)
    source_recipe = Path(recipe_path).read_bytes()
    source_support = (Path(recipe_path).resolve().parent /
                      json.loads(source_recipe)['support_file']).read_bytes()
    start = shard_index // 4 * 4
    shards = [generate_scenario_shard(recipe, split=split, shard_index=i, arm=arm)
              for i in range(start, start + 4)]
    coverage = validate_coverage_cycle(shards, recipe)
    shard = shards[shard_index - start]
    files = {}
    def put(name, value):
        files[name] = canonical_bytes(value) + b'\n'
    recipe_document = dict(json_value(recipe.payload))
    recipe_document['support_file'] = 'support.json'
    put('recipe.json', recipe_document)
    put('support.json', recipe.support.payload)
    for item in shards:
        put(f'shard-{item.shard_index}.json', item.manifest)
    put('catalog.json', shard.catalog)
    put('annotations.json', shard.annotations)
    panel = build_development_panel(recipe)
    put('development-panel.json', panel)
    put('development.json', panel['manifest'])
    put('pilot-plan.json', pilot_plan(recipe))
    document = tomllib.loads((_CONFIGS / 'pilot.toml').read_text())
    document.update(manifest_path=f'shard-{shard_index}.json', output_root='runs',
                    operation='train' if split == 'train' else 'evaluate', split=split, partition=split)
    if split == 'train':
        document['scenario_stream'] = {'schema': 'spire-learn-scenario-stream-v1',
            'recipe_path': 'recipe.json', 'arm': arm, 'shard_index': shard_index,
            'distribution_sha256': shard.distribution_sha256}
    else:
        document['rollout'] = {'rows': 1, 'steps': 1}
    files['config.toml'] = _toml(document).encode('utf-8')
    # Roundtrip the authored syntax before publication; actual loader follows below.
    if tomllib.loads(files['config.toml'].decode()) != document:
        raise ValueError('config syntax roundtrip failed')
    if len(files['config.toml']) > 32768:
        raise ValueError('prepared config exceeds 32 KiB')
    inventory = {name: {'raw_sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data),
        'canonical_sha256': content_sha256(document if name == 'config.toml' else json.loads(data))}
        for name, data in files.items()}
    result = {'schema': 'spire-learn-curriculum-preparation-v1', 'execution_authorized': False,
        'arm': arm, 'split': split, 'shard_index': shard_index,
        'recipe_sha256': recipe.sha256, 'support_sha256': recipe.support.sha256,
        'source_recipe_raw_sha256': hashlib.sha256(source_recipe).hexdigest(),
        'source_support_raw_sha256': hashlib.sha256(source_support).hexdigest(),
        'distribution_sha256': shard.distribution_sha256, 'manifest_sha256': shard.manifest_sha256,
        'config_sha256': content_sha256(document), 'config_path': str(destination / 'config.toml'),
        'output_root': str(destination / 'runs'), 'coverage': json_value(coverage),
        'cycle': [{'shard_index': s.shard_index, 'manifest_sha256': s.manifest_sha256,
                   'manifest_file': f'shard-{s.shard_index}.json'} for s in shards],
        'inventory': inventory, 'loader_roundtrip': False,
        'successor_config_eligible': split == 'train' and shard_index + 1 < 2**24 - 1}
    return files, result


def prepare_stream(recipe_path: Path, destination: Path, *, arm: str, split: str,
                   shard_index: int) -> Mapping[str, object]:
    """Exclusively publish bounded preparation under an absent artifact destination."""
    from .config import load_neural_config
    files, result = _preparation(recipe_path, destination, arm=arm, split=split, shard_index=shard_index)
    destination = Path(destination).absolute()
    destination.mkdir()
    for name, data in files.items():
        with (destination / name).open('xb') as stream:
            stream.write(data)
    config = load_neural_config(destination / 'config.toml')
    if (config.config_sha256 != result['config_sha256'] or
            config.manifest_sha256 != result['manifest_sha256']):
        raise ValueError('prepared loader roundtrip changed identity')
    result['loader_roundtrip'] = True
    with (destination / 'preparation.json').open('xb') as stream:
        stream.write(canonical_bytes(result) + b'\n')
    return result


def run_stream_segment(config: NeuralConfig, *, purpose: str, checkpoint: Path | None,
                       operation_limits: RuntimeOperationLimits,
                       dependencies: Mapping[str, object] | None = None) -> object:
    """Execute exactly once; offer a successor only from healthy published evidence."""
    from .config import NeuralConfig
    from .runner import NeuralReport, RuntimeOperationLimits, _checkpoint_recovery
    from .stream_resume import stream_identity
    if not isinstance(config, NeuralConfig):
        raise TypeError('config must be NeuralConfig')
    stream_identity(config)
    if purpose not in ('train', 'resume', 'warm_start'):
        raise ValueError('segment purpose must be train/resume/warm_start')
    if (purpose == 'train') != (checkpoint is None):
        raise ValueError('only fresh train omits checkpoint')
    if not isinstance(operation_limits, RuntimeOperationLimits):
        raise TypeError('operation_limits must be RuntimeOperationLimits')
    if dependencies is None:
        from .runner import create_runtime, train
        from time import monotonic
        dependencies = {'create_runtime': create_runtime, 'train': train, 'clock': monotonic}
    if (not isinstance(dependencies, Mapping) or set(dependencies) != {'create_runtime', 'train', 'clock'}
            or any(not callable(v) for v in dependencies.values())):
        raise TypeError('dependencies must be the three plain callables create_runtime/train/clock')
    runtime = dependencies['create_runtime'](config, purpose=purpose, checkpoint=checkpoint,
        policy_kind='checkpoint', action_selection='seeded-sampling')
    try:
        runtime.operation_limits = operation_limits
    except BaseException as error:
        try:
            runtime.close()
        except BaseException as close_error:
            error.add_note(f'runtime close also failed: {close_error}')
        raise
    # train owns close and preserves first faults/control exceptions; never retry or salvage here.
    report = dependencies['train'](config, runtime=runtime, clock=dependencies['clock'])
    if not isinstance(report, NeuralReport):
        raise TypeError('train must return NeuralReport')
    result = {'report': report, 'run_dir': report.run_dir, 'checkpoint': None,
              'saved_prefix': None, 'next_shard': None}
    summary = report.summary
    if report.exit_code != 0 or summary.get('status') != 'complete':
        return result
    lifecycle = summary.get('lifecycle', {})
    publication = summary.get('checkpoint_publication', {})
    if (summary.get('lifecycle_receipt_required') is not True or lifecycle.get('status') != 'complete'
            or lifecycle.get('exit_code') != 0 or lifecycle.get('close_errors') != []
            or lifecycle.get('clock_error') is not None or lifecycle.get('operation_error') is not None
            or summary.get('lifecycle_receipt_error') is not None
            or publication.get('publication_state') != 'published'
            or publication.get('purpose') != 'final'
            or lifecycle.get('checkpoint_publication') != publication):
        raise ValueError('segment lacks healthy finalized lifecycle/publication')
    if report.run_dir is None or Path(publication['checkpoint_path']) != report.run_dir / 'checkpoint-final':
        raise ValueError('checkpoint publication does not belong to actual run')
    from .checkpoint import preflight_checkpoint
    saved = preflight_checkpoint(Path(publication['checkpoint_path']), config.accepted_identities[0], purpose='resume')
    metadata = saved.metadata
    _, prefix = _checkpoint_recovery(saved, config)
    state = metadata['run_metadata']['scenario_stream']
    if any(prefix[name] for name in ('uncertain_environment_slots', 'uncertain_reset_slots',
                                     'optimizer_uncertain_steps')):
        raise ValueError('uncertain saved progress cannot authorize a successor')
    if state['current']['shard_index'] != config.scenario_stream['shard_index']:
        raise ValueError('published checkpoint is not the current shard')
    actual_parent = metadata['parent_checkpoint']
    if actual_parent != (None if checkpoint is None else {'path': str(checkpoint)}):
        raise ValueError('published checkpoint has a different current parent')
    result['checkpoint'] = Path(publication['checkpoint_path'])
    result['saved_prefix'] = prefix
    if state['next']['shard_index'] < 2**24 - 1:
        result['next_shard'] = state['next']
    return result


def summarize_development_reports(panel: Mapping[str, object], reports: Mapping[str, object]) -> Mapping[str, object]:
    """Aggregate one actual standalone report per case with explicit missing data."""
    from .runner import NeuralReport
    cases = panel['manifest']['cases']
    if (len(cases) != 64 or panel['manifest_sha256'] != content_sha256(panel['manifest'])
            or set(reports) - {c['case_id'] for c in cases}):
        raise ValueError('reports must bind cases in the 64-case development panel')
    strata = {name: {} for name in ('cohort', 'family', 'deck_stage', 'encounter', 'encounter_tier', 'card')}
    failures = []
    for case in cases:
        case_id = case['case_id']
        annotation = panel['annotations'][case_id]
        report = reports.get(case_id)
        started = completed = wins = deaths = escaped = failed = metric_available = 0
        hp_before, hp_after, retention = [], [], []
        if report is not None:
            if not isinstance(report, NeuralReport) or report.summary.get('mode') != 'standalone':
                raise ValueError('development inputs must be actual standalone NeuralReports')
            summary = report.summary
            failed = int(report.exit_code != 0 or summary.get('status') != 'complete'
                         or summary.get('collection', {}).get('uncertain_environment_slots', 0) > 0)
            rows = summary.get('case_accounting', {}).get('cases')
            if (rows is not None and (len(rows) != 1 or rows[0]['case_id'] != case_id)
                    or rows is None and not failed):
                raise ValueError('each report must account for exactly its supplied case')
            fight = summary.get('fights', {}).get('standalone_fights')
            if fight is not None:
                metric_available = 1
                started, completed, wins, deaths = (fight[k] for k in
                    ('reached_combat_denominator', 'completed', 'victories', 'deaths'))
                escaped = fight.get('escaped', 0)
                if any(type(v) is not int or v < 0 for v in (started, completed, wins, deaths, escaped)) or not wins + deaths + escaped == completed <= started <= 1:
                    raise ValueError('development fight denominators are inconsistent')
                if len(fight['fights']) != completed:
                    raise ValueError('resource rows must match actual completed fights')
            elif not failed:
                raise ValueError('healthy development report is missing fight metrics')
            if failed:
                failures.append({'case_id': case_id, 'status': summary.get('status'),
                    'exit_code': report.exit_code, 'run_dir': str(report.run_dir) if report.run_dir else None,
                    'fault': summary.get('fault', summary.get('fatal_error')),
                    'reason': summary.get('reason', summary.get('stop_reason'))})
            for row in fight['fights'] if fight is not None else ():
                for key, target in [('hp_before_healing', hp_before), ('hp_after_cleanup', hp_after), ('hp_retention', retention)]:
                    if row.get(key) is not None:
                        value = row[key]
                        # canonical serialization rejects nonfinite and non-JSON values.
                        canonical_bytes(value)
                        if type(value) not in (int, float):
                            raise ValueError('resource metric must be numeric')
                        target.append(value)
        for dimension, groups in strata.items():
            keys = annotation['cards'] if dimension == 'card' else [annotation[dimension]]
            for key in keys:
                row = groups.setdefault(key, {'scheduled': 0, 'reported': 0, 'started': 0,
                    'completed': 0, 'wins': 0, 'deaths': 0, 'escaped': 0, 'censored': 0,
                    'fight_reports': 0, 'fight_reports_missing': 0,
                    'operational_failures': 0, 'hp_samples': 0, 'hp_before_healing_sum': 0,
                    'hp_after_cleanup_sum': 0, 'hp_after_cleanup_samples': 0,
                    'hp_retention_sum': 0.0, 'hp_retention_samples': 0,
                    'potion_use': None, 'potion_use_samples': 0})
                for name, count in [('scheduled', 1), ('reported', int(report is not None)), ('started', started),
                    ('completed', completed), ('wins', wins), ('deaths', deaths), ('escaped', escaped), ('censored', started - completed),
                    ('fight_reports', metric_available),
                    ('fight_reports_missing', int(report is not None) - metric_available),
                    ('operational_failures', failed), ('hp_samples', len(hp_before)),
                    ('hp_before_healing_sum', sum(hp_before)), ('hp_after_cleanup_sum', sum(hp_after)),
                    ('hp_after_cleanup_samples', len(hp_after)), ('hp_retention_sum', sum(retention)),
                    ('hp_retention_samples', len(retention))]:
                    row[name] += count
    for groups in strata.values():
        for row in groups.values():
            row['win_rate'] = row['wins'] / row['completed'] if row['completed'] else None
            row['censored_fraction'] = row['censored'] / row['started'] if row['started'] else None
            row['coverage_fraction'] = row['reported'] / row['scheduled']
            for output, total, count in [('mean_hp_before_healing', 'hp_before_healing_sum', 'hp_samples'),
                ('mean_hp_after_cleanup', 'hp_after_cleanup_sum', 'hp_after_cleanup_samples'),
                ('mean_hp_retention', 'hp_retention_sum', 'hp_retention_samples')]:
                row[output] = row[total] / row[count] if row[count] else None
    worst = {}
    for dimension, groups in strata.items():
        observed = [(key, row) for key, row in groups.items() if row['completed']]
        if observed:
            key, row = min(observed, key=lambda item: (item[1]['win_rate'], item[0]))
            worst[dimension] = {'stratum': key, 'win_rate': row['win_rate'],
                'completed': row['completed'], 'scheduled': row['scheduled'], 'reported': row['reported']}
        else:
            worst[dimension] = None
    return {'schema': 'spire-learn-development-summary-v1',
        'panel_manifest_sha256': panel['manifest_sha256'], 'strata': strata,
        'worst_strata': worst,
        'status': ('operational_failure' if failures else 'complete'
                   if sum(row['completed'] for row in strata['cohort'].values()) == 64 else 'incomplete'),
        'failures': failures, 'potion_use_availability': 'unavailable in accepted reports',
        'denominator_scope': 'fight counts cover reports with supplied fight metrics only',
        'claim': 'standalone diagnostics; small strata; historical exposure unknown; no full-Run/transfer claim'}


__all__ = ['prepare_stream', 'build_development_panel', 'pilot_plan',
           'run_stream_segment', 'summarize_development_reports']
