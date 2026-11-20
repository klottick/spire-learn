#!/usr/bin/env python3.12
"""Run the Standard stream runtime tests, including the gated Gym smoke, in the runtime container.

    runtime_container.py --entry standard_stream_smoke_entry.py --env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 ...

Runs ``tests.unit.test_neural_standard_stream_runtime`` (host tests plus
``test_standard_stream_smoke``: two PPO updates at 16 rows on a synthetic table, the
second resumed on a fuzzed shard validated by the real Gym). Writes ``result.json`` to the
runner output. Exits 0 only if every test passed and the smoke test actually ran.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

MODULE = 'tests.unit.test_neural_standard_stream_runtime'
SMOKE = f'{MODULE}.StandardStreamSmokeTest.test_standard_stream_smoke'


def main() -> int:
    output = Path(os.environ['SPIRE_LEARN_RUNNER_OUTPUT'])
    if os.environ.get('SPIRE_LEARN_RUN_NATIVE_INTEGRATION') != '1':
        print('set --env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1', file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromName(MODULE)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    detail = lambda rows: [{'test': test.id(), 'message': str(message)[-4000:]} for test, message in rows]
    skipped = detail(result.skipped)
    smoke_ran = not any(row['test'] == SMOKE for row in skipped)
    document = {'schema': 'spire-learn-standard-stream-smoke-v1', 'tests_run': result.testsRun,
                'failures': detail(result.failures), 'errors': detail(result.errors), 'skipped': skipped,
                'smoke_ran': smoke_ran, 'successful': result.wasSuccessful() and smoke_ran,
                'evidence': [json.loads(path.read_text()) for path in sorted(output.glob(
                    'standard-stream-smoke-*/smoke-evidence.json'))]}
    with (output / 'result.json').open('x') as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + '\n')
    return 0 if document['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
