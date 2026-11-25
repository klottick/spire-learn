"""Scenario support binds to a registered Gym revision, not to one hard-coded build."""
import copy
import dataclasses
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from spire_learn.neural import gym_contracts
from spire_learn.neural.config import ACCEPTED_GYM_REVISION
from spire_learn.neural.scenario_support import ScenarioSupport, load_scenario_support

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "configs/neural/curriculum-v1/support.json"
FAKE = "c" * 40
UNKNOWN = "d" * 40
FAKE_CONTRACT = dataclasses.replace(gym_contracts.GYM_V1_2_4_CONTRACT, source_revision=FAKE,
                                    runtime_revisions=(FAKE,))


def _with_revision(revision):
    raw = json.loads(SUPPORT.read_text())
    raw["gym_revision"] = revision
    raw["provenance"]["revision"] = revision
    return raw


class ScenarioSupportRegistryTest(unittest.TestCase):
    def test_original_support_is_unchanged(self):
        support = load_scenario_support(SUPPORT)
        self.assertEqual(support.payload["gym_revision"], ACCEPTED_GYM_REVISION)
        self.assertIn(ACCEPTED_GYM_REVISION, gym_contracts.CONTRACTS_BY_REVISION)

    def test_registered_revision_is_accepted_and_unregistered_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported source-bound support identity"):
            ScenarioSupport(_with_revision(FAKE))
        with patch.dict(gym_contracts.CONTRACTS_BY_REVISION, {FAKE: FAKE_CONTRACT}):
            support = ScenarioSupport(_with_revision(FAKE))
            self.assertEqual(support.payload["gym_revision"], FAKE)
            with self.assertRaisesRegex(ValueError, "unsupported source-bound support identity"):
                ScenarioSupport(_with_revision(UNKNOWN))
            # Provenance must name the same revision as the support itself.
            raw = _with_revision(FAKE)
            raw["provenance"]["revision"] = ACCEPTED_GYM_REVISION
            with self.assertRaisesRegex(ValueError, "invalid support provenance"):
                ScenarioSupport(raw)

    def test_original_revision_keeps_its_source_pin(self):
        raw = copy.deepcopy(_with_revision(ACCEPTED_GYM_REVISION))
        raw["provenance"]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "accepted public sources"):
            ScenarioSupport(raw)


if __name__ == "__main__":
    unittest.main()
