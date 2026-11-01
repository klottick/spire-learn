"""Immutable campaign inputs, tested through the real parsers without backends."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/development/neural-combat/curriculum-campaign-v1/task-1'


def package_fixture():
    """One immutable full generation shared by both focused test modules."""
    api = importlib.import_module('scripts.neural.curriculum_package')
    destination = ARTIFACTS / 'package-fixture'
    if not destination.exists():
        api.author_package(ROOT, destination, campaign_id='curriculum-20261101-a')
    return destination, api.read_json(destination / 'package.json')


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = importlib.import_module('scripts.neural.curriculum_package')
        cls.destination, cls.package = package_fixture()

    def test_actual_264_training_configs_and_64_cpu_cases(self):
        from spire_learn.neural.config import load_neural_config
        p = self.package
        self.assertEqual({k: len(v) for k, v in p['training'].items()},
                         {'fixed': 4, 'randomized': 4, 'mixed': 256})
        self.assertEqual(len(p['panel']['cases']), 64)
        self.assertEqual(len(p['evaluation_slots']), 8)
        self.assertLessEqual(p['total_input_bytes'], 256 * 1024**2)
        for arm, rows in p['training'].items():
            self.assertEqual(len({r['stable_training_sha256'] for r in rows}), 1)
            for index in (0, len(rows) - 1):
                config = load_neural_config(self.destination / rows[index]['config_path'])
                self.assertEqual((config.device, config.rollout_rows, config.rollout_steps), ('cuda:0', 16, 32))
                self.assertEqual(config.scenario_stream['shard_index'], index)
                self.assertEqual(config.config_sha256, rows[index]['config_sha256'])
        for row in p['panel']['cases']:
            config = load_neural_config(self.destination / row['config_path'])
            self.assertEqual((config.device, config.operation, config.rollout_rows, config.rollout_steps),
                             ('cpu', 'evaluate', 1, 1))
            self.assertIsNone(config.scenario_stream)
            self.assertEqual(len(config.cases.cases), 1)

    def test_validation_and_exclusive_publication(self):
        code = '''
import sys
from pathlib import Path
from tests.unit.test_neural_curriculum_integration import blocked_optional
with blocked_optional() as attempts:
    from scripts.neural.curriculum_package import validate_package, read_json
    path=Path(sys.argv[1]); result=validate_package(Path.cwd(),path)
    assert result==read_json(path) and not attempts
print('validated 264 real training configs/66 aligned cycles/64 CPU configs; optional attempts/loaded: []')
'''
        environment = {k: v for k, v in os.environ.items() if not k.startswith('SPIRE_LEARN_RUN_')}
        environment['PYTHONPATH'] = str(ROOT / 'src') + ':' + str(ROOT)
        child = subprocess.run([sys.executable, '-B', '-c', code, str(self.destination / 'package.json')],
                               cwd=ROOT, env=environment, text=True, capture_output=True)
        (ARTIFACTS / 'fresh-validate.stdout').write_text(child.stdout)
        (ARTIFACTS / 'fresh-validate.stderr').write_text(child.stderr)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('optional attempts/loaded: []', child.stdout)
        with self.assertRaises((ValueError, FileExistsError)):
            self.api.author_package(ROOT, self.destination, campaign_id='curriculum-20261101-a')
        with self.assertRaises(ValueError):
            self.api.author_package(ROOT, self.destination.parent / 'wrong', campaign_id='other')

    def test_inventory_covers_every_input_and_rejects_changed_bytes(self):
        emitted = {str(p.relative_to(self.destination)) for p in self.destination.rglob('*') if p.is_file()}
        self.assertEqual(emitted, set(self.package['inventory']) | {'package.json'})
        path = self.destination / self.package['panel']['cases'][0]['config_path']
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                self.api.validate_package(ROOT, self.destination / 'package.json')
        finally:
            path.write_bytes(original)

    def test_rebound_wrong_training_budget_is_not_a_valid_campaign(self):
        """Rehashing a changed config cannot turn a pilot into this campaign."""
        import hashlib
        import tomllib
        from spire_learn.neural.scenario_support import content_sha256
        from spire_learn.neural.stream_runner import _toml
        row = self.package['training']['fixed'][0]
        path = self.destination / row['config_path']
        manifest = self.destination / 'package.json'
        original, package_bytes = path.read_bytes(), manifest.read_bytes()
        document = tomllib.loads(original.decode())
        document['budgets']['max_environment_actions'] = 1024
        changed = _toml(document).encode()
        altered = json.loads(package_bytes)
        altered['inventory'][row['config_path']] = {'sha256': hashlib.sha256(changed).hexdigest(), 'size_bytes': len(changed)}
        altered['training']['fixed'][0]['config_sha256'] = content_sha256(document)
        altered['total_input_bytes'] += len(changed) - len(original)
        try:
            path.write_bytes(changed)
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'campaign training settings'):
                self.api.validate_package(ROOT, manifest)
        finally:
            path.write_bytes(original)
            manifest.write_bytes(package_bytes)


if __name__ == '__main__':
    unittest.main()
