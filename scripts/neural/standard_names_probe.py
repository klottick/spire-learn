"""Probe: check every Standard name alias target against the Gym.

Runs in the offline runtime container (no training): the Gym's card metadata must
match the alias table's pinned fingerprint and card ids, and one-entry Standard
combat catalogs must accept every relic and potion target. Writes
``$SPIRE_LEARN_RUNNER_OUTPUT/standard-names-probe.json``.

    python3.12 scripts/neural/runtime_container.py --profile configs/neural/gym-1.1.1/runtime.json \\
        --entry standard_names_probe.py --output <abs dir> --authority ... -- configs/neural/standard/name-aliases.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'src'), str(_ROOT)]


def main(argv: list[str]) -> int:
    from spire_learn.neural.standard_names import gym_card_metadata, gym_rejections, load_aliases
    aliases = load_aliases(Path(argv[0]))
    metadata = gym_card_metadata()
    result = {'card_metadata_fingerprint': metadata['fingerprint'],
              'card_ids': sorted(row['id'] for row in metadata['cards']),
              'alias_sha256': aliases.sha256, 'rejections': gym_rejections(aliases)}
    (Path(os.environ['SPIRE_LEARN_RUNNER_OUTPUT']) / 'standard-names-probe.json').write_text(
        json.dumps(result, sort_keys=True))
    print(json.dumps({'fingerprint': result['card_metadata_fingerprint'],
                      'rejections': len(result['rejections'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
