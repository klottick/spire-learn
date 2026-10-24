import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.artifacts.provenance import collect_provenance


class ProvenanceTests(unittest.TestCase):
    def test_unborn_git_checkout_and_missing_gym_checkout_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "learner"
            root.mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            value = collect_provenance(learner_root=root, gym_checkout=None)
            self.assertEqual(set(value), {"learner", "gym_checkout", "distributions", "native_build_info"})
            self.assertIsNone(value["learner"]["revision"])
            self.assertEqual(value["learner"]["head_state"], "unborn")
            self.assertIsNone(value["gym_checkout"])
            self.assertIsNone(value["native_build_info"])

    def test_git_subprocesses_disable_optional_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "learner"
            root.mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            real_run = subprocess.run
            commands = []

            def inspect_run(argv, *args, **kwargs):
                commands.append(argv)
                return real_run(argv, *args, **kwargs)

            with patch("spire_learn.artifacts.provenance.subprocess.run", side_effect=inspect_run):
                collect_provenance(learner_root=root, gym_checkout=None)
            self.assertGreaterEqual(len(commands), 2)
            self.assertTrue(all(command[:2] == ["git", "--no-optional-locks"] for command in commands))

    def test_non_repository_does_not_borrow_parent_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = collect_provenance(learner_root=Path(temporary), gym_checkout=Path(temporary) / "absent")
            self.assertIsNone(value["learner"]["revision"])
            self.assertEqual(value["learner"]["head_state"], "unavailable")
            self.assertIsNone(value["gym_checkout"]["revision"])
            json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
