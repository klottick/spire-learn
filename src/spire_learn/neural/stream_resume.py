"""Explicit pure scenario-stream provenance and exact same/next-shard admission.

The legacy checkpoint payload validator owns model/Adam/RNG validation. This
module only admits a concrete catalog transition and bounded sampler projection.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING

from .cases import _freeze_json
from .curriculum import CurriculumSampler
from .scenario_stream import ScenarioRecipe, ScenarioShard, generate_scenario_shard, load_scenario_recipe
from .scenario_support import content_sha256, json_value

if TYPE_CHECKING:
    from .config import NeuralConfig

_SCHEMA = 'spire-learn-scenario-stream-v1'
_FIELDS = {'schema', 'recipe_path', 'arm', 'shard_index', 'distribution_sha256'}
_MAX_CONFIG_BYTES = 32 * 1024


def _stable_document(raw: Mapping[str, object]) -> dict[str, object]:
    value = json_value(raw)
    if (not isinstance(value, dict) or 'manifest_path' not in value
            or not isinstance(value.get('scenario_stream'), dict)
            or set(value['scenario_stream']) != _FIELDS):
        raise ValueError('scenario_stream source document is malformed')
    value.pop('manifest_path')
    value['scenario_stream'].pop('shard_index')
    return value


def normalize_stream_config(raw: Mapping[str, object], source: Path,
                            manifest_path: Path, manifest_sha256: str) -> Mapping[str, object]:
    """Loader-only normalization; authored table fields remain exact."""
    config_source = source.read_bytes().decode('utf-8')
    if len(config_source.encode('utf-8')) > _MAX_CONFIG_BYTES:
        raise ValueError('scenario_stream config source exceeds 32 KiB metadata bound')
    table = raw['scenario_stream']
    if not isinstance(table, Mapping) or set(table) != _FIELDS or table['schema'] != _SCHEMA:
        raise ValueError('scenario_stream must have the exact supported fields/schema')
    if (raw['mode'], raw['operation'], raw['split'], raw['partition']) != (
            'standalone', 'train', 'train', 'train'):
        raise ValueError('scenario_stream requires standalone train')
    if table['shard_index'] == 2**24 - 1:
        raise ValueError('scenario_stream requires a representable successor shard')
    if not isinstance(table['recipe_path'], str) or not table['recipe_path']:
        raise ValueError('scenario_stream.recipe_path must be a path')
    recipe_path = (source.parent / table['recipe_path']).resolve()
    recipe = load_scenario_recipe(recipe_path)
    shard = generate_scenario_shard(recipe, split='train', shard_index=table['shard_index'],
                                    arm=table['arm'])
    if table['distribution_sha256'] != shard.distribution_sha256:
        raise ValueError('scenario_stream distribution does not match recipe/support/arm')
    if manifest_sha256 != shard.manifest_sha256:
        raise ValueError('scenario_stream manifest is not the actual derived shard')
    return _freeze_json({**table, 'resolved_recipe_path': str(recipe_path),
        'config_document': raw, 'config_source': config_source,
        'resolved_manifest_path': str(manifest_path),
        'resolved_output_root': str((source.parent / raw['output_root']).resolve()),
        'manifest_raw_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})


def _context(config: NeuralConfig) -> tuple[Mapping[str, object], ScenarioRecipe, ScenarioShard]:
    table = config.scenario_stream
    if table is None:
        raise ValueError('explicit scenario_stream config is required')
    recipe = load_scenario_recipe(Path(table['resolved_recipe_path']))
    shard = generate_scenario_shard(recipe, split='train', shard_index=table['shard_index'],
                                    arm=table['arm'])
    raw = table['config_document']
    if (str(config.manifest_path) != table['resolved_manifest_path']
            or str(config.output_root) != table['resolved_output_root']
            or content_sha256(raw) != config.config_sha256
            or content_sha256(tomllib.loads(table['config_source'])) != config.config_sha256
            or shard.manifest_sha256 != config.manifest_sha256
            or shard.manifest_sha256 != config.cases.sha256
            or shard.distribution_sha256 != table['distribution_sha256']):
        raise ValueError('scenario_stream actual config/manifest/distribution binding changed')
    return table, recipe, shard


def stream_identity(config: NeuralConfig) -> Mapping[str, object]:
    """Bind semantic source settings and actual runtime settings, independent of mount roots."""
    return _identity(config, *_context(config))


def _identity(config: NeuralConfig, table: Mapping[str, object],
              recipe: ScenarioRecipe, shard: ScenarioShard) -> dict[str, object]:
    # The source digest preserves logical path semantics (especially output_root).
    # The projection also catches stale dataclasses.replace settings at API boundaries.
    runtime = {name: getattr(config, name) for name in (
        'mode', 'operation', 'split', 'partition', 'environment_revision',
        'common_view_id', 'common_view_fingerprint', 'run_observation_fingerprint',
        'profile_id', 'observation_version', 'replay_version', 'rollout_rows',
        'rollout_steps', 'device', 'final_requires_opt_in', 'calibration_gpu_uuid',
        'runtime_gpu_uuid')}
    runtime.update({name: asdict(getattr(config, name)) for name in (
        'seed_streams', 'model', 'ppo', 'budgets')})
    runtime['accepted_identities'] = [{
        'model_id': identity.model_id, 'features_id': identity.features_id,
        'action_id': identity.action_id, 'objective_id': identity.objective_id,
        'accepted_raw_mode_mappings': dict(identity.accepted_raw_mode_mappings),
        'config_hashes': dict(identity.config_hashes),
    } for identity in config.accepted_identities]
    if (config.mode, config.operation, config.split, config.partition) != (
            'standalone', 'train', 'train', 'train'):
        raise ValueError('scenario_stream requires standalone train')
    return {'schema': _SCHEMA, 'distribution_sha256': shard.distribution_sha256,
            'recipe_sha256': recipe.sha256, 'support_sha256': recipe.support.sha256,
            'arm': shard.arm, 'stable_training_sha256': content_sha256({
                'source': _stable_document(table['config_document']), 'runtime': runtime}),
            'rng_domains': ['policy', 'minibatch', 'curriculum', 'controller']}


def _binding(config: NeuralConfig, table: Mapping[str, object],
             shard: ScenarioShard) -> dict[str, object]:
    source = table['config_source']
    return {'shard_index': shard.shard_index, 'config_sha256': config.config_sha256,
            'config_raw_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
            'config_source': source,
            'manifest_sha256': config.manifest_sha256,
            'manifest_raw_sha256': table['manifest_raw_sha256']}


def _sampler_state(recipe: ScenarioRecipe, shard: ScenarioShard, state: object) -> dict[str, object]:
    if shard.arm != 'mixed':
        if state != {}:
            raise ValueError('nonmixed streams must not carry adaptive sampler state')
        return {}
    if not isinstance(state, Mapping):
        raise ValueError('mixed stream resume requires explicit sampler state')
    return CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                             categories=recipe.category_keys, state=state).snapshot()


def stream_checkpoint_state(config: NeuralConfig,
                            sampler_state: Mapping[str, object]) -> Mapping[str, object]:
    """Capture the current real binding and healthy sampler state without RNG draws."""
    table, recipe, shard = _context(config)
    next_index = shard.shard_index + 1
    # The final representable shard cannot declare an unrepresentable successor.
    next_shard = generate_scenario_shard(recipe, split='train', shard_index=next_index, arm=shard.arm)
    return {'identity': _identity(config, table, recipe, shard),
            'current': _binding(config, table, shard),
            'next': {'shard_index': next_index, 'manifest_sha256': next_shard.manifest_sha256},
            'sampler_state': _sampler_state(recipe, shard, sampler_state)}


def _validate_binding(binding, recipe, arm, destination_raw) -> int:
    if not isinstance(binding, Mapping) or set(binding) != {
            'shard_index', 'config_sha256', 'config_raw_sha256', 'config_source',
            'manifest_sha256', 'manifest_raw_sha256'}:
        raise ValueError('resume scenario_stream binding is malformed')
    index = binding['shard_index']
    shard = generate_scenario_shard(recipe, split='train', shard_index=index, arm=arm)
    source = binding['config_source']
    if not isinstance(source, str) or len(source.encode('utf-8')) > _MAX_CONFIG_BYTES:
        raise ValueError('resume scenario_stream source must be bounded TOML')
    try:
        previous_raw = tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError('resume scenario_stream source must be TOML') from error
    previous_stable = _stable_document(previous_raw)
    if (content_sha256(previous_raw) != binding['config_sha256']
            or hashlib.sha256(source.encode('utf-8')).hexdigest() != binding['config_raw_sha256']
            or binding['manifest_sha256'] != shard.manifest_sha256
            or previous_raw.get('scenario_stream', {}).get('shard_index') != index
            or previous_stable != _stable_document(destination_raw)):
        raise ValueError('resume scenario_stream predecessor config/manifest binding changed')
    # A prior raw manifest hash is historical provenance from the authenticated
    # checkpoint; canonical content is regenerated above. Current bytes are
    # verified by the loader and exact same-shard binding comparison below.
    raw_hash = binding['manifest_raw_sha256']
    if not isinstance(raw_hash, str) or len(raw_hash) != 64 or any(c not in '0123456789abcdef' for c in raw_hash):
        raise ValueError('resume scenario_stream historical raw manifest hash is invalid')
    return index


def validate_stream_resume(metadata: Mapping[str, object],
                           config: NeuralConfig) -> Mapping[str, object]:
    """Validate a run_metadata mapping against the actual same or next derived shard."""
    table, recipe, destination = _context(config)
    identity = _identity(config, table, recipe, destination)
    destination_binding = _binding(config, table, destination)
    state = metadata.get('scenario_stream')
    if not isinstance(state, Mapping) or set(state) != {'identity', 'current', 'next', 'sampler_state'}:
        raise ValueError('resume checkpoint lacks explicit scenario_stream state; use weight-only warm start')
    if json_value(state['identity']) != identity:
        raise ValueError('resume scenario_stream stable training identity changed')
    current = state['current']
    index = _validate_binding(current, recipe, destination.arm, table['config_document'])
    if index not in {destination.shard_index, destination.shard_index - 1}:
        raise ValueError('resume scenario_stream permits only same or exactly next shard')
    successor = generate_scenario_shard(recipe, split='train', shard_index=index + 1, arm=destination.arm)
    if json_value(state['next']) != {'shard_index': index + 1, 'manifest_sha256': successor.manifest_sha256}:
        raise ValueError('resume scenario_stream declared successor is not derived from recipe')
    if (current['config_sha256'] != metadata.get('config_sha256')
            or current['manifest_sha256'] != metadata.get('manifest_sha256')):
        raise ValueError('resume scenario_stream current hashes disagree with checkpoint')
    transition = metadata.get('scenario_stream_transition')
    if metadata.get('controller_id') == 'neural-policy-resume-v1' and transition is None:
        raise ValueError('resumed scenario_stream checkpoint is missing its transition binding')
    if transition is not None:
        if (not isinstance(transition, Mapping) or set(transition) != {'identity', 'from', 'to'}
                or json_value(transition['identity']) != identity
                or json_value(transition['to']) != json_value(current)):
            raise ValueError('resume scenario_stream transition destination/identity changed')
        previous_index = _validate_binding(transition['from'], recipe, destination.arm,
                                           table['config_document'])
        if previous_index not in {index, index - 1}:
            raise ValueError('resume scenario_stream historical transition is not same/next')
    if index == destination.shard_index and dict(current) != destination_binding:
        raise ValueError('same-shard resume requires exact source and manifest bytes')
    return {'identity': identity, 'current': json_value(current),
            'destination': destination_binding, 'next': json_value(state['next']),
            'sampler_state': _sampler_state(recipe, destination, state['sampler_state'])}


__all__ = ['stream_identity', 'validate_stream_resume', 'stream_checkpoint_state']
