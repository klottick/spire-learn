"""Write the Gym's static card metadata to a committable snapshot (runtime container; no training).

Calls ``standard_names.gym_card_metadata()`` (``spire_lab.card_metadata.load()``, the Gym's own
``spire-lab-card-metadata-v1`` descriptor: static, public, run-independent) and writes it unchanged,
as canonical JSON, to ``$SPIRE_LEARN_RUNNER_OUTPUT/card-metadata-gym-1.2.4.json``. The controller
copies that file to ``configs/neural/standard/card-metadata-gym-1.2.4.json``; S3
(``transformer.build_transformer_cardmeta_policy``) loads it and refuses any other fingerprint.

The entry exits 1 when the Gym's fingerprint is not ``CARD_METADATA_FINGERPRINT`` (the snapshot is
still written, for inspection, but the loader will refuse it).

    python3.12 scripts/neural/runtime_container.py --profile configs/neural/gym-1.2.5/runtime.json \\
        --entry card_metadata_snapshot_entry.py --output <abs dir> --authority <text>
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'src'), str(_ROOT)]

SNAPSHOT_NAME = 'card-metadata-gym-1.2.4.json'


def main(argv: list[str]) -> int:
    if argv:
        print('card_metadata_snapshot_entry.py takes no arguments', file=sys.stderr)
        return 2
    from spire_learn.neural.config import CARD_METADATA_FINGERPRINT
    from spire_learn.neural.standard_names import gym_card_metadata
    metadata = gym_card_metadata()
    text = json.dumps(metadata, sort_keys=True, separators=(',', ':')) + '\n'
    (Path(os.environ['SPIRE_LEARN_RUNNER_OUTPUT']) / SNAPSHOT_NAME).write_text(text)
    summary = {'snapshot': SNAPSHOT_NAME, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
               'fingerprint': metadata['fingerprint'], 'pinned_fingerprint': CARD_METADATA_FINGERPRINT,
               'matches_pin': metadata['fingerprint'] == CARD_METADATA_FINGERPRINT,
               'cards': len(metadata['cards']),
               'faces': sum(len(row['faces']) for row in metadata['cards'])}
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary['matches_pin'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
