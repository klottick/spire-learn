"""Preparation entry points use only stdlib and create no runtime authority."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from spire_learn.neural.cli import main

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/development/neural-combat'
RECIPE = ROOT / 'configs/neural/curriculum-v1/recipe.json'


class StreamCLITests(unittest.TestCase):
    def test_help_routes_curriculum_and_keeps_old_commands(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(['--help']), 0)
        for command in ('train', 'evaluate', 'calibrate', 'curriculum'):
            self.assertIn(command, out.getvalue())
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(['curriculum', 'prepare', '--help']), 0)

    def test_validation_and_invalid_input_have_zero_publication(self):
        from spire_learn.neural.stream_cli import main as prepare_main
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temporary:
            destination = Path(temporary) / 'prepared'
            args = ['prepare', '--recipe', str(RECIPE), '--destination', str(destination),
                    '--arm', 'mixed', '--shard-index', '0']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(prepare_main(args + ['--validate-only']), 0)
                self.assertFalse(destination.exists())
                self.assertEqual(prepare_main(args[:-1] + ['-1']), 2)
                self.assertFalse(destination.exists())
                destination.mkdir()
                marker = destination / 'user.txt'
                marker.write_text('preserve')
                self.assertEqual(prepare_main(args), 2)
                self.assertEqual(list(destination.iterdir()), [marker])
