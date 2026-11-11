"""Explicit pure scenario-stream provenance and exact same/next-shard admission.

The legacy checkpoint payload validator owns model/Adam/RNG validation. This
module only admits a concrete catalog transition and bounded sampler projection.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
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
_SKIP_AUTHORIZATION = {
    'schema': 'spire-learn-scenario-stream-gap-v1',
    'campaign_id': 'curriculum-20261103-twenty-hour-continuation-1',
    'from_shard_index': 337,
    'expected_next_shard_index': 338,
    'skipped_shard_index': 338,
    'to_shard_index': 339,
    'parent_receipt_sha256': 'b2afb649a3887bbcc101c397831cbe6b5c0e639476c377cbace20f4f1e333fd0',
    'parent_checkpoint_inventory_sha256': '071a05b8bdec51f38b9f7591dec553a0d7f5547cc2dd2131094d678fe617054e',
    'skipped_attempt_sha256': '67b1adf91b865246f34d43ad30a1724b804aacccb8db6daa8e2fa55e7e0beea8',
    'skipped_claim_sha256': '37ad8526dce9e4431d6a0b0c43ee3f5a51bb7ac4ee530d11a186bef18de110f9',
    'skipped_lease': {
        'environment_attempts': 4096,
        'sample_rows': 4096,
        'optimizer_attempts': 256,
        'reset_slots': 4112,
        'fixed_random_choices': 0,
    },
    'actual': None,
    'receipt_binding': None,
    'user_authority_sha256': '1c1c1d85b858caf1900db84bfdec5ba9696cdd8d71823b9fbada0bebae2e35c4',
}
_SKIP_AUTHORIZATION_CORRECTED = {
    **_SKIP_AUTHORIZATION,
    'campaign_id': 'curriculum-20261103-twenty-hour-continuation-2',
    'user_authority_sha256': '161a5ed34029a62cd4229c0b4fb852cbca6d0a78fcc4813e75b966f7d1e7cd22',
}
_SKIP_AUTHORIZATION_REPAIRED = {
    **_SKIP_AUTHORIZATION,
    'campaign_id': 'curriculum-20261103-twenty-hour-continuation-3',
    'user_authority_sha256': '5f86c21f1d1321287f09a2307dcf16d5ebb16656984e038355264bb40da1d2bf',
}
_SKIP_AUTHORIZATIONS = (_SKIP_AUTHORIZATION, _SKIP_AUTHORIZATION_CORRECTED,
                        _SKIP_AUTHORIZATION_REPAIRED)
_MAX_CONFIG_BYTES = 32 * 1024


def validate_continuation_skip_authorization(value: Mapping[str, object]) -> Mapping[str, object]:
    """Accept only the recorded one-shard skip from healthy337 across unknown338."""
    if isinstance(value, str):
        def pairs(rows):
            result = {}
            for key, item in rows:
                if key in result:
                    raise ValueError('duplicate continuation skip authorization field')
                result[key] = item
            return result
        try:
            value = json.loads(value, object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite skip authorization')))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError('continuation skip authorization encoding differs') from error
    if (not isinstance(value, Mapping)
            or not any(content_sha256(json_value(value)) == content_sha256(authorized)
                       for authorized in _SKIP_AUTHORIZATIONS)):
        raise ValueError('continuation stream skip authorization differs')
    return json_value(value)


def _stable_document(raw: Mapping[str, object]) -> dict[str, object]:
    value = json_value(raw)
    table = value.get('scenario_stream') if isinstance(value, dict) else None
    if (not isinstance(value, dict) or 'manifest_path' not in value
            or not isinstance(table, dict)
            or set(table) not in (_FIELDS, _FIELDS | {'resume_skip'})):
        raise ValueError('scenario_stream source document is malformed')
    if 'resume_skip' in table:
        validate_continuation_skip_authorization(table['resume_skip'])
    value.pop('manifest_path')
    value['scenario_stream'].pop('shard_index')
    value['scenario_stream'].pop('resume_skip', None)
    return value


def normalize_stream_config(raw: Mapping[str, object], source: Path,
                            manifest_path: Path, manifest_sha256: str) -> Mapping[str, object]:
    """Loader-only normalization; authored table fields remain exact."""
    config_source = source.read_bytes().decode('utf-8')
    if len(config_source.encode('utf-8')) > _MAX_CONFIG_BYTES:
        raise ValueError('scenario_stream config source exceeds 32 KiB metadata bound')
    table = raw['scenario_stream']
    if (not isinstance(table, Mapping)
            or set(table) not in (_FIELDS, _FIELDS | {'resume_skip'})
            or table['schema'] != _SCHEMA):
        raise ValueError('scenario_stream must have the exact supported fields/schema')
    if 'resume_skip' in table:
        authorization = validate_continuation_skip_authorization(table['resume_skip'])
        if table.get('arm') != 'mixed' or table.get('shard_index') != 339:
            raise ValueError('continuation stream skip authorization is only valid for shard339')
    else:
        authorization = None
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
    normalized_table = dict(table)
    if authorization is not None:
        normalized_table['resume_skip'] = authorization
    return _freeze_json({**normalized_table, 'resolved_recipe_path': str(recipe_path),
        'config_document': raw, 'config_source': config_source,
        'resolved_manifest_path': str(manifest_path),
        'resolved_output_root': str((source.parent / raw['output_root']).resolve()),
        'manifest_raw_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})


def _context(config: NeuralConfig) -> tuple[Mapping[str, object], ScenarioRecipe, ScenarioShard]:
    table = config.scenario_stream
    if table is None:
        raise ValueError('explicit scenario_stream config is required')
    from .config import _stream_runtime_projection
    if content_sha256(_stream_runtime_projection(config)) != content_sha256(table.get('runtime_projection')):
        raise ValueError('scenario_stream runtime projection disagrees with loaded config')
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
    from .config import _stream_runtime_projection
    # Admission already compared these normalized settings with the loader binding.
    runtime = _stream_runtime_projection(config)
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
    authorization = table.get('resume_skip')
    if authorization is not None:
        authorization = validate_continuation_skip_authorization(authorization)
    next_state = state.get('next')
    gap_parent = (authorization is not None and isinstance(next_state, Mapping) and index == 337
        and destination.shard_index == 339
        and authorization['from_shard_index'] == index
        and authorization['expected_next_shard_index'] == index + 1
        and next_state.get('shard_index') == index + 1)
    if index not in {destination.shard_index, destination.shard_index - 1} and not gap_parent:
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
        if (not isinstance(transition, Mapping)
                or set(transition) not in ({'identity', 'from', 'to'},
                                           {'identity', 'from', 'to', 'resume_skip_authorization'})
                or json_value(transition['identity']) != identity
                or json_value(transition['to']) != json_value(current)):
            raise ValueError('resume scenario_stream transition destination/identity changed')
        recorded_skip = transition.get('resume_skip_authorization')
        if recorded_skip is not None:
            recorded_skip = validate_continuation_skip_authorization(recorded_skip)
        previous_index = _validate_binding(transition['from'], recipe, destination.arm,
                                           table['config_document'])
        recorded_gap = (recorded_skip is not None and previous_index == 337
            and index == 339 and transition['to']['shard_index'] == 339)
        if previous_index not in {index, index - 1} and not recorded_gap:
            raise ValueError('resume scenario_stream historical transition is not same/next')
    else:
        recorded_gap = False
    if index == destination.shard_index and dict(current) != destination_binding:
        raise ValueError('same-shard resume requires exact source and manifest bytes')
    return {'identity': identity, 'current': json_value(current),
            'destination': destination_binding, 'next': json_value(state['next']),
            'sampler_state': _sampler_state(recipe, destination, state['sampler_state']),
            **({'resume_skip_authorization': json_value(authorization if gap_parent else recorded_skip)}
               if gap_parent or recorded_gap else {})}


__all__ = ['stream_identity', 'validate_stream_resume', 'stream_checkpoint_state',
           'validate_continuation_skip_authorization']
