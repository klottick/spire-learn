"""Pure preflight and transport checks; never invoke Docker or Gym."""

import hashlib
import json
from pathlib import Path
import tempfile
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.neural import qualify_gym_release as q
from tests.unit.test_neural_sustained_controls import FakeDocker


class GymReleaseQualificationTests(unittest.TestCase):
    def test_actual_host_entrypoint_imports_before_missing_local_preparation(self):
        script = Path("scripts/neural/qualify_gym_release.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", str(script)], cwd=directory,
                env={**os.environ, "PYTHONPATH": "src"}, capture_output=True,
                text=True, timeout=10.)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertIn("preparation.json", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_launcher_sets_hard_outer_limit_without_running_host(self):
        launcher = Path("scripts/neural/run-gym-release-qualification.sh").resolve()
        self.assertTrue(launcher.is_file(), "bounded qualification launcher is missing")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spy = directory / "timeout"
            spy.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$QUALIFICATION_SPY"\n')
            spy.chmod(0o755)
            record = directory / "argv"
            result = subprocess.run(["bash", str(launcher)], env={
                **os.environ, "PATH": str(directory)+os.pathsep+os.defpath,
                "QUALIFICATION_SPY": str(record),
            }, capture_output=True, text=True, timeout=5.)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(record.read_text().splitlines(), [
                "--signal=TERM", "--kill-after=30s", "90s", "env", "PYTHONPATH=src",
                "python3.12", "-B", "scripts/neural/qualify_gym_release.py",
            ])

    def test_wrong_source_binding_refuses_before_one_use_marker_or_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "qualification"
            domain.mkdir()
            (domain / "preparation.json").write_text("{}")
            (domain / "qualification-release.json").write_text(json.dumps({
                "state": "reviewed", "single_use": True, "source_digest": "wrong",
            }))
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with (patch.object(q, "ROOT", root), patch.object(q, "DOMAIN", domain),
                  patch.object(q.low, "_docker_binary", side_effect=AssertionError("Docker lookup"))):
                with self.assertRaises(AssertionError):
                    q.main()
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((domain / "qualification-started.json").exists())

    def test_exact_cpu_transport_closes_only_owned_id_and_no_gpu_request(self):
        # Use the actual reviewed transport with strict fake command boundaries.
        with tempfile.TemporaryDirectory() as directory:
            client = Path(directory)
            spec = q.low.PhaseSpec(
                "gym-migration-metadata", "spirelearn-gym-migration-20261102-a",
                "gym-migration-20261102", "a" * 64, "b" * 64, "c" * 64,
                80., 120., "/output", ("-B", "/release/verify_guest.py"),
                (("CUDA_VISIBLE_DEVICES", ""), ("NVIDIA_VISIBLE_DEVICES", "none")),
                (("/tmp/owned-output", "/output", False),), (), runtime_class="cpu", pids_limit=128,
            )
            fake = FakeDocker(spec)
            deps = SimpleNamespace(clock=lambda: 0., run=fake.run)
            result = q.transport.execute_job(spec, deps=deps, client=client,
                                            docker="docker", logs=[], progress=None)
            self.assertTrue(result["removal_entered"])
            create = next(argv for argv in fake.calls if argv[5] == "create")
            self.assertNotIn("--gpus", create)
            self.assertEqual(create[create.index("--memory") + 1], str(8 * 1024**3))
            removals = [argv for argv in fake.calls if argv[5] == "rm"]
            self.assertEqual(len(removals), 1)
            self.assertEqual(removals[0][-1], fake.cid)

    def test_name_collision_never_removes_preexisting_container(self):
        with tempfile.TemporaryDirectory() as directory:
            from tests.unit.test_neural_sustained_controls import spec
            fake = FakeDocker(spec(), failure="existing")
            deps = SimpleNamespace(clock=lambda: 0., run=fake.run)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                q.transport.execute_job(fake.spec, deps=deps, client=Path(directory),
                                        docker="docker", logs=[], progress=None)
            self.assertFalse(any(argv[5] in ("create", "start", "rm") for argv in fake.calls))
