#!/usr/bin/env python3.12
"""Generate baalorlord-standard-v1 shards and panels with the Gym, and reset every entry.

    standard_stream_check.py --recipe PATH --output DIR [--shards 0 25 50 75] [--panels selection final]

Needs the Gym, so it runs in the offline runtime container
(``scripts/neural/runtime_container.py --entry standard_stream_check.py``). Each shard and
panel is generated with the Gym's catalog validator, then its whole catalog is validated
again and every entry is reset once at its case's environment seed. A Gym rejection or
reset fault is an error (exit 1), never a skipped case. Writes one JSON file per shard and
panel plus ``summary.json`` to the output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

from spire_learn.neural.scenario_support import canonical_bytes, content_sha256, json_value  # noqa: E402
from spire_learn.neural.recipes import gym_validator  # noqa: E402
from spire_learn.neural.standard_catalog import catalog, validate_entries  # noqa: E402
from spire_learn.neural.standard_stream import (  # noqa: E402
    build_standard_panel, generate_standard_shard, levels_for_shard, load_standard_recipe,
)

DEFAULT_RECIPE = Path(__file__).resolve().parents[2] / 'configs/neural/baalorlord-standard-v1/recipe.json'


def gym_check(manifest: dict, validate) -> dict:
    """Validate the catalog and reset each entry once; raises on any rejection or fault."""
    from spire_lab.gymnasium import StandardCombatEnv
    entries = [case['entry'] for case in manifest['cases']]
    _, rejected = validate_entries(entries, validate)
    if rejected:
        raise RuntimeError(f'Gym rejected {len(rejected)} generated entries: {rejected[0][1]}')
    resets = 0
    for start in range(0, len(manifest['cases']), 256):
        chunk = manifest['cases'][start:start + 256]
        env = StandardCombatEnv(catalog([case['entry'] for case in chunk]))
        try:
            for case in chunk:
                observation, _ = env.reset(seed=case['environment_seed'], options={'entry_id': case['entry']['id']})
                if not observation:
                    raise RuntimeError(f'empty observation for {case["entry"]["id"]}')
                resets += 1
        finally:
            env.close()
    return {'entries': len(entries), 'validated': len(entries), 'resets': resets}


def main(argv: list[str] | None = None, *, build_info=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--recipe', type=Path, default=DEFAULT_RECIPE)
    parser.add_argument('--output', type=Path, default=os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT'))
    parser.add_argument('--shards', type=int, nargs='*', default=[0, 25, 50, 75])
    parser.add_argument('--panels', nargs='*', default=['selection', 'final'], choices=['selection', 'final'])
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error('--output is required outside the runtime container')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = load_standard_recipe(args.recipe)
    # The checked path: refuses a Gym whose revision differs from the recipe's table revision.
    validate = gym_validator('stream check', recipe, build_info=build_info)
    summary = {'schema': 'spire-learn-standard-stream-check-v1', 'recipe_sha256': recipe.sha256,
               'table_sha256': recipe.table_sha256, 'shards': {}, 'panels': {},
               'split_states': {split: len(recipe.states_for(split)) for split in ('train', 'selection', 'final')}}
    for index in args.shards:
        shard = generate_standard_shard(recipe, split='train', shard_index=index, validate=validate)
        manifest = json_value(shard.manifest)
        levels = Counter(a['level'] for a in shard.annotations.values())
        summary['shards'][str(index)] = {
            'levels_enabled': list(levels_for_shard(recipe, index)), 'cases': len(manifest['cases']),
            'skipped': recipe.shard_size - len(manifest['cases']),
            'levels': {str(k): v for k, v in sorted(levels.items())}, 'manifest_sha256': shard.manifest_sha256,
            'gym': gym_check(manifest, validate)}
        (output / f'shard-train-{index}.json').write_bytes(canonical_bytes(
            {'manifest': manifest, 'annotations': json_value(shard.annotations)}))
        print(json.dumps({'shard': index, **summary['shards'][str(index)]}), flush=True)
    for split in args.panels:
        panel = build_standard_panel(recipe, split, validate=validate)
        document = json_value(panel)
        summary['panels'][split] = {
            'panel_sha256': content_sha256(document), 'manifest_sha256': document['manifest_sha256'],
            'panel_identity_sha256': document['panel_sha256'], 'panel_identity': document['panel_identity'],
            'cases': len(document['manifest']['cases']), 'strata': document['strata'],
            'available': document['available'], 'skipped': document['skipped'],
            'gym': gym_check(document['manifest'], validate)}
        (output / f'panel-{split}.json').write_bytes(canonical_bytes(document))
        print(json.dumps({'panel': split, 'panel_sha256': summary['panels'][split]['panel_sha256'],
                          'manifest_sha256': document['manifest_sha256'],
                          'panel_identity_sha256': document['panel_sha256'],
                          'skipped': document['skipped']}), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
