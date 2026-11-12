"""Captured Gym 1.0.0 fixtures and the capture script; never imports real Gym."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.neural import capture_gym_descriptors as capture


V1 = "1929e18b9f44372e020877bd62f2a5c9dee3e64e"
FIXTURES = Path("tests/fixtures/neural")
PUBLISHED = {
    "common": "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "standard-combat": "5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    "standard-run": "bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
}


def _fixture(name):
    return json.loads((FIXTURES / f"{name}-1929e18.json").read_text())


class CapturedFixtureTests(unittest.TestCase):
    def test_fixture_bodies_match_published_fingerprints(self):
        for name, fingerprint in PUBLISHED.items():
            with self.subTest(name=name):
                descriptor = _fixture(name)
                self.assertEqual(descriptor["fingerprint"], fingerprint)
                self.assertIs(capture.verify(name, descriptor), descriptor)

    def test_tampered_fixture_is_rejected(self):
        descriptor = _fixture("standard-combat")
        descriptor["profile_id"] = "Other"
        with self.assertRaises(SystemExit):
            capture.verify("standard-combat", descriptor)

    def test_common_view_unchanged_from_afb51f5(self):
        self.assertEqual(_fixture("common"),
                         json.loads((FIXTURES / "common-afb51f5.json").read_text()))

    def test_environment_identities(self):
        combat = _fixture("standard-combat-environment")
        self.assertEqual((combat["profile_id"], combat["schema_id"], combat["mode"]),
                         ("standard-combat-v1", "spire-gym-standard-combat-view-v1",
                          "standalone_combat"))
        self.assertEqual(combat["observation_fingerprint"], PUBLISHED["standard-combat"])
        run = _fixture("standard-run-environment")
        self.assertEqual((run["profile_id"], run["schema_id"], run["combat_view_id"]),
                         ("Standard", "spire-gym-standard-run-observation-v1",
                          "spire-gym-standard-combat-view-v1"))
        self.assertEqual(run["observation_fingerprint"], PUBLISHED["standard-run"])


class CaptureScriptTests(unittest.TestCase):
    def test_capture_mode_never_resets(self):
        closes = []

        class Env:
            def __init__(self, *args, **kwargs):
                self.name = type(self).__name__

            def reset(self, *args, **kwargs):
                raise RuntimeError("unpatched reset")

            step = reset

            def close(self):
                closes.append(self.name)

        class StandardCombatVectorEnv(Env):
            def observation_spec(self):
                return _fixture("standard-combat")

            def environment_spec(self):
                return _fixture("standard-combat-environment")

            def catalog_descriptor(self):
                return _fixture("standard-combat-catalog")

        class RunVectorEnv(Env):
            def observation_spec(self):
                return _fixture("standard-run")

            def environment_spec(self):
                return _fixture("standard-run-environment")

        class RunEnv(RunVectorEnv):
            def combat_observation_spec(self):
                return _fixture("standard-combat")

        class IroncladCombatVectorEnv(Env):
            def observation_spec(self):
                return _fixture("common")

        package = ModuleType("spire_lab")
        package._native = SimpleNamespace(build_info=lambda: {"revision": V1, "dirty": False})
        gymnasium = ModuleType("spire_lab.gymnasium")
        for cls in (StandardCombatVectorEnv, RunVectorEnv, RunEnv, IroncladCombatVectorEnv):
            setattr(gymnasium, cls.__name__, cls)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {"spire_lab": package, "spire_lab.gymnasium": gymnasium}):
            result = capture.capture(Path(directory), V1)
            for cls in (StandardCombatVectorEnv, RunEnv, IroncladCombatVectorEnv):
                with self.subTest(cls=cls.__name__), self.assertRaisesRegex(
                        RuntimeError, "unpatched"):
                    cls().reset()
            written = sorted(path.name for path in Path(directory).iterdir())
        self.assertEqual(result["fingerprints"], PUBLISHED)
        self.assertEqual(sorted(closes), sorted([
            "StandardCombatVectorEnv", "RunVectorEnv", "RunEnv", "IroncladCombatVectorEnv"]))
        self.assertEqual(written, sorted(f"{name}-1929e18.json" for name in (
            "common", "standard-combat", "standard-combat-catalog",
            "standard-combat-environment", "standard-run", "standard-run-environment")))

    def test_reset_during_capture_fails(self):
        class StandardCombatVectorEnv:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self, *args, **kwargs):
                return "reset forwarded"

            step = reset

            def observation_spec(self):
                return self.reset()

            def close(self):
                pass

        package = ModuleType("spire_lab")
        package._native = SimpleNamespace(build_info=lambda: {"revision": V1, "dirty": False})
        gymnasium = ModuleType("spire_lab.gymnasium")
        gymnasium.StandardCombatVectorEnv = StandardCombatVectorEnv
        for name in ("RunVectorEnv", "RunEnv", "IroncladCombatVectorEnv"):
            setattr(gymnasium, name, StandardCombatVectorEnv)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {"spire_lab": package, "spire_lab.gymnasium": gymnasium}):
            with self.assertRaisesRegex(AssertionError, "must not reset or step"):
                capture.capture(Path(directory), V1)

    def test_capture_rejects_dirty_build(self):
        package = ModuleType("spire_lab")
        package._native = SimpleNamespace(build_info=lambda: {"revision": V1, "dirty": True})
        gymnasium = ModuleType("spire_lab.gymnasium")
        for name in ("StandardCombatVectorEnv", "RunVectorEnv", "RunEnv", "IroncladCombatVectorEnv"):
            setattr(gymnasium, name, object)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {"spire_lab": package, "spire_lab.gymnasium": gymnasium}):
            with self.assertRaises(SystemExit):
                capture.capture(Path(directory), V1)


if __name__ == "__main__":
    unittest.main()
