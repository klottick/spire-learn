"""Fake-driver tests for the Gym 1.0.0 soak utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "soak_gym_v1", ROOT / "scripts/neural/soak_gym_v1.py")
assert SPEC is not None and SPEC.loader is not None
SOAK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOAK)


class _Row:
    def __init__(self, active):
        self.candidates = (type("Candidate", (), {"index": 0})(),) if active else ()
        self.noncombat_indices = (1,)
        self.lifecycle = type("Lifecycle", (), {
            "active": active, "routing": "combat" if active else "noncombat",
            "combats_settled": 0, "last_exit": None,
        })()


class _Transition:
    def __init__(self, source, successor, terminated=False):
        self.source = source
        self.successor = successor
        self.lifecycle = successor.lifecycle
        self.terminated = terminated
        self.truncated = False


class _FakeDriver:
    batch_size = 2

    def __init__(self):
        self.rows = (_Row(True), _Row(True))
        self.reset_masks = []
        self.steps = 0

    def reset(self, *, seeds, entry_ids, reset_mask=None):
        mask = tuple(reset_mask or [True] * self.batch_size)
        self.reset_masks.append(mask)
        rows = list(self.rows)
        for index, selected in enumerate(mask):
            if selected:
                rows[index] = _Row(True)
        self.rows = tuple(rows)
        return self.rows

    def step(self, actions):
        self.steps += 1
        first = _Row(False) if self.steps == 1 else _Row(True)
        second = _Row(True)
        transitions = (_Transition(self.rows[0], first), _Transition(self.rows[1], second))
        self.rows = (first, second)
        return transitions

    def close(self):
        pass


class NeuralGymV1SoakTests(unittest.TestCase):
    def test_standalone_soak_resets_only_finished_rows(self):
        driver = _FakeDriver()
        result = SOAK._drive(driver, mode="standard-standalone", steps=1,
                             rng=random.Random(1), entry_ids=lambda: ["case", "case"])
        self.assertEqual(result["steps"], 1)
        self.assertEqual(driver.reset_masks[-1], (True, False))
        self.assertTrue(driver.rows[1].lifecycle.active)

    def test_run_mode_does_not_import_spire_lab_on_module_import(self):
        self.assertNotIn("spire_lab", SOAK.__dict__)


if __name__ == "__main__":
    unittest.main()
