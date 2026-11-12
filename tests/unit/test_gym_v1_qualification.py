"""Offline checks for the Gym 1.0.0 one-reset qualification script."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.neural import qualify_gym_v1 as qualification
from spire_learn.neural import envs
from spire_learn.neural.gym_contracts import GYM_V1_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/neural/gym-v1.0.0"


def _build_info():
    build = GYM_V1_CONTRACT.build
    return {
        "extension": "spire-python",
        "revision": build.revision,
        "dirty": False,
        "profile": build.profile,
        "features": list(build.features),
        "source_digest": build.source_digest,
        "file_digest": build.file_digest,
    }


class FakeDriver:
    def __init__(self, config, *, import_torch=False):
        self.mode = "run" if config.mode.endswith("run") else "standalone"
        self.batch_size = config.rollout_rows
        self.schema = SimpleNamespace(
            schema_fingerprint=(config.standard_view_fingerprint
                                or config.common_view_fingerprint),
            view=SimpleNamespace(view_id="fake-view"),
        )
        self.reset_calls = []
        self.step_calls = 0
        self.close_calls = 0
        self.import_torch = import_torch
        self.rows = tuple(SimpleNamespace(
            frame=SimpleNamespace(schema_fingerprint=self.schema.schema_fingerprint),
            lifecycle=SimpleNamespace(active=True), candidates=(object(),),
            noncombat_indices=(),
        ) for _ in range(self.batch_size))

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        if self.import_torch:
            import torch  # noqa: F401  # must be stopped by the probe's import guard
        return self.rows

    def step(self, *args, **kwargs):
        self.step_calls += 1
        raise AssertionError("qualification must never step the driver")

    def close(self):
        self.close_calls += 1


class GymV1QualificationTests(unittest.TestCase):
    def test_qualification_constructs_four_modes(self):
        drivers = []

        def make_driver(config, cases):
            driver = FakeDriver(config)
            drivers.append(driver)
            return driver

        runtime = SimpleNamespace(native=SimpleNamespace(build_info=_build_info))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(envs, "make_driver", side_effect=make_driver), \
                patch.object(envs, "_load_native_runtime", return_value=runtime):
            result = qualification.run(Path(temporary))
            saved = json.loads((Path(temporary) / "qualification.json").read_text())

        self.assertEqual([item["mode"] for item in result["modes"]], [
            "standalone", "run", "standard-standalone", "standard-run",
        ])
        self.assertEqual(saved, result)
        self.assertEqual(len(drivers), 4)
        for driver in drivers:
            self.assertEqual(len(driver.reset_calls), 1)
            self.assertEqual(driver.step_calls, 0)
            self.assertEqual(driver.close_calls, 1)

    def test_torch_import_fails_and_driver_is_closed(self):
        drivers = []

        def make_driver(config, cases):
            driver = FakeDriver(config, import_torch=True)
            drivers.append(driver)
            return driver

        runtime = SimpleNamespace(native=SimpleNamespace(build_info=_build_info))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(envs, "make_driver", side_effect=make_driver), \
                patch.object(envs, "_load_native_runtime", return_value=runtime):
            with self.assertRaisesRegex(AssertionError, "training import"):
                qualification.run(Path(temporary))

        self.assertEqual(len(drivers), 1)
        self.assertEqual(drivers[0].step_calls, 0)
        self.assertEqual(drivers[0].close_calls, 1)


if __name__ == "__main__":
    unittest.main()
