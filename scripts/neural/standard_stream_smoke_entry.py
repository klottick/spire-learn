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

from spire_learn.neural.gym_contracts import contract_for_revision

MODULES = ('tests.unit.test_neural_standard_stream_runtime', 'tests.unit.test_neural_standard_compact')


def smoke_tests_for_revision(revision: str) -> tuple[str, ...]:
    message = f'no Standard stream smoke for Gym revision {revision!r}'
    try:
        contract = contract_for_revision(revision)
    except ValueError as error:
        raise ValueError(message) from error
    if not contract.standard_stream_smoke_tests:
        raise ValueError(message)
    return contract.standard_stream_smoke_tests


def _test_ids(suite) -> list[str]:
    if isinstance(suite, unittest.TestSuite):
        return [test_id for child in suite for test_id in _test_ids(child)]
    return [suite.id()]


def main() -> int:
    output = Path(os.environ['SPIRE_LEARN_RUNNER_OUTPUT'])
    if os.environ.get('SPIRE_LEARN_RUN_NATIVE_INTEGRATION') != '1':
        print('set --env SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1', file=sys.stderr)
        return 2
    from spire_lab import _native
    revision = _native.build_info().get('revision')
    try:
        smoke = smoke_tests_for_revision(revision)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    collected = set(_test_ids(suite))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    detail = lambda rows: [{'test': test.id(), 'message': str(message)[-4000:]} for test, message in rows]
    skipped = detail(result.skipped)
    # Ran = collected (a renamed or missing test is not "not skipped") and not skipped.
    missing = [test for test in smoke if test not in collected]
    smoke_ran = not missing and not any(row['test'] in smoke for row in skipped)
    document = {'schema': 'spire-learn-standard-stream-smoke-v1', 'gym_revision': revision,
                'required_gated_tests': list(smoke), 'missing_required_tests': missing,
                'tests_run': result.testsRun,
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
