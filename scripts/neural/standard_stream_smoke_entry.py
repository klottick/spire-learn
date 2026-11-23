#!/usr/bin/env python3.12
"""Run the Standard stream runtime tests, including the gated Gym smoke, in the runtime container.

    runtime_container.py --entry standard_stream_smoke_entry.py --env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 ...

Runs ``tests.unit.test_neural_standard_stream_runtime`` and ``tests.unit.test_neural_standard_compact``
(host tests, the compact-view gated tests on v1.2.0, and the smoke of the
loaded Gym release: ``test_standard_stream_smoke`` on v1.1.1, ``test_standard_stream_smoke_gym_1_2_0``
(compact view) on v1.2.0; two PPO updates at 16 rows on a synthetic table, the second resumed on
a fuzzed shard validated by the real Gym). Writes ``result.json`` to the runner output. Exits 0
only if every test passed and the release's smoke test actually ran.
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
COMPACT = 'tests.unit.test_neural_standard_compact'
MODULES = (MODULE, COMPACT)
# The gated tests each Gym release must run (the others skip outside their release's runtime):
# the release's stream smoke and, on v1.2.0, the compact-view equivalence and overflow tests.
SMOKES = {
    'e3c13836bd1458be8a87b93160f5c350f78fcd80': (
        f'{MODULE}.StandardStreamSmokeTest.test_standard_stream_smoke',),
    '4d198064b3b568b6f0d08d6e3937d236fbc48e9b': (
        f'{MODULE}.StandardStreamSmokeTest.test_standard_stream_smoke_gym_1_2_0',
        f'{COMPACT}.GatedCompactTests.test_compact_and_dense_give_identical_policy_outputs',
        f'{COMPACT}.GatedCompactTests.test_capacity_overflow_is_a_driver_fault'),
}


def main() -> int:
    output = Path(os.environ['SPIRE_LEARN_RUNNER_OUTPUT'])
    if os.environ.get('SPIRE_LEARN_RUN_NATIVE_INTEGRATION') != '1':
        print('set --env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1', file=sys.stderr)
        return 2
    from spire_lab import _native
    revision = _native.build_info().get('revision')
    smoke = SMOKES.get(revision)
    if smoke is None:
        print(f'no Standard stream smoke for Gym revision {revision!r}', file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    detail = lambda rows: [{'test': test.id(), 'message': str(message)[-4000:]} for test, message in rows]
    skipped = detail(result.skipped)
    smoke_ran = not any(row['test'] in smoke for row in skipped)
    document = {'schema': 'spire-learn-standard-stream-smoke-v1', 'gym_revision': revision,
                'required_gated_tests': list(smoke), 'tests_run': result.testsRun,
                'failures': detail(result.failures), 'errors': detail(result.errors), 'skipped': skipped,
                'smoke_ran': smoke_ran, 'successful': result.wasSuccessful() and smoke_ran,
                'evidence': [json.loads(path.read_text()) for path in sorted(output.glob(
                    'standard-stream-smoke-*/smoke-evidence.json'))]
                + [json.loads(path.read_text()) for path in sorted(output.glob('compact-*.json'))]}
    with (output / 'result.json').open('x') as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + '\n')
    return 0 if document['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
