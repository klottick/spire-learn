import json
from pathlib import Path
import tempfile
import unittest

from spire_learn.experiments.cases import CaseManifest, load_cases, validate_splits
from spire_learn.contracts import CaseExecution


class CasesTests(unittest.TestCase):
    def write(self, root, split='selection', partition='native-a', env=1, policy=2, case='c', execution='e'):
        path = root / f'{split}-{execution}.json'
        path.write_text(json.dumps({'schema': 'spire-learn-cases-v1', 'split': split, 'cases': [
            {'case_id': case, 'scenario_id': 's', 'partition': partition,
             'executions': [{'execution_id': execution, 'environment_seed': env, 'policy_seed': policy}]}
        ]}))
        return path

    def test_manifest_preserves_explicit_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = load_cases(self.write(Path(d), env=0, policy=10**80))
            self.assertEqual((manifest.executions[0].environment_seed, manifest.executions[0].policy_seed), (0, 10**80))

    def test_authored_selection_smoke_cases_are_unique_and_exact(self):
        manifest = load_cases(Path("configs/cases/selection-smoke.json"))
        self.assertEqual(manifest.split, "selection")
        self.assertEqual(
            tuple(
                (item.case_id, item.scenario_id, item.partition,
                 item.environment_seed, item.policy_seed)
                for item in manifest.executions
            ),
            (
                ("starter-cultist-a0-v1", "starter-cultist-a0-v1", "selection", 7, 101),
                ("starter-jaw-worm-a0-v1", "starter-jaw-worm-a0-v1", "selection", 11, 103),
            ),
        )
        self.assertEqual(
            len({item.case_id for item in manifest.executions}),
            len(manifest.executions),
        )
        self.assertEqual(
            len({item.execution_id for item in manifest.executions}),
            len(manifest.executions),
        )

    def test_bool_and_negative_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(Path(d)); data = json.loads(path.read_text())
            data['cases'][0]['executions'][0]['environment_seed'] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_cases(path)

    def test_split_overlap_ignores_partition_and_policy_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = load_cases(self.write(root, 'train', 'one', 7, 1, 'a', 'x'))
            b = load_cases(self.write(root, 'final', 'two', 7, 999, 'b', 'y'))
            with self.assertRaises(ValueError): validate_splits((a, b))

    def test_strict_json_rejects_unknown_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.write(root)
            path.write_text(path.read_text().replace('"split": "selection"', '"split": "selection", "extra": 1'))
            with self.assertRaises(ValueError):
                load_cases(path)
            path.write_text('{"schema":"spire-learn-cases-v1","split":"train","split":"final","cases":[]}')
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_rejects_empty_cases_and_invalid_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(Path(d))
            data = json.loads(path.read_text())
            data['cases'][0]['expected_fingerprint'] = 'g' * 64
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_cases(path)
            data['cases'][0].pop('expected_fingerprint')
            data['cases'] = []
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_repetitions_allowed_with_distinct_execution_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(Path(d))
            data = json.loads(path.read_text())
            repeated = dict(data['cases'][0]['executions'][0], execution_id='e2')
            data['cases'][0]['executions'].append(repeated)
            path.write_text(json.dumps(data))
            manifest = load_cases(path)
            self.assertEqual([e.environment_seed for e in manifest.executions], [1, 1])

    def test_validate_splits_rejects_programmatic_split_disagreement(self):
        execution = CaseExecution('e', 'c', 'final', 'p', 's', 1, 2)
        with self.assertRaises(ValueError):
            validate_splits((CaseManifest(Path('cases.json'), 'train', (execution,)),))

    def test_validate_splits_allows_repeated_case_within_one_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(Path(d), case='repeat', execution='e1')
            data = json.loads(path.read_text())
            data['cases'][0]['executions'].append(
                {'execution_id': 'e2', 'environment_seed': 1, 'policy_seed': 3}
            )
            path.write_text(json.dumps(data))
            manifest = load_cases(path)
            self.assertIsNone(validate_splits((manifest,)))

    def test_manifest_owns_executions_input(self):
        execution = CaseExecution('e', 'c', 'selection', 'p', 's', 1, 2)
        source = [execution]
        manifest = CaseManifest(Path('cases.json'), 'selection', source)
        source.clear()
        self.assertEqual(manifest.executions, (execution,))

    def test_validate_splits_rejects_case_id_repeated_across_manifests(self):
        first = CaseExecution('e1', 'same', 'selection', 'p', 's1', 1, 2)
        second = CaseExecution('e2', 'same', 'selection', 'p', 's2', 2, 3)
        manifests = (CaseManifest(Path('a'), 'selection', (first,)),
                     CaseManifest(Path('b'), 'selection', (second,)))
        with self.assertRaisesRegex(ValueError, 'duplicate case_id'):
            validate_splits(manifests)

    def test_same_scenario_different_seed_is_not_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = load_cases(self.write(root, 'train', env=1, case='a', execution='x'))
            b = load_cases(self.write(root, 'final', env=2, case='b', execution='y'))
            validate_splits((a, b))
