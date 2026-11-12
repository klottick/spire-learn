"""Combat view registry and Standard combat schema parsing against real 1.0.0 fixtures."""

import copy
import json
from pathlib import Path
import unittest

from spire_learn.contracts import ContractError
from spire_learn.neural import schema
from spire_learn.neural.contracts import BufferArray, CombatFrame
from spire_learn.neural.gym_contracts import CURRENT_CONTRACT, GYM_V1_CONTRACT


V1 = GYM_V1_CONTRACT.source_revision
AFB = CURRENT_CONTRACT.source_revision
FIXTURES = Path("tests/fixtures/neural")


def _load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _lifecycle_info(view_id, outcome="victory", **changes):
    info = {
        "combat_view_id": view_id, "active": False, "routing": "noncombat",
        "selection_owner": None, "active_visit": None, "combats_entered": 1,
        "combats_settled": 1, "last_exit": {
            "visit": 0, "encounter": "cultist", "outcome": outcome,
            "hp_before_healing": 50, "hp_after_cleanup": 56},
        "effective_seed": 12345,
    }
    info.update(changes)
    return info


class ViewRegistryTests(unittest.TestCase):
    def test_registry_values(self):
        from spire_learn.neural.views import COMMON_V1, STANDARD_COMBAT_V1, view_for_id
        self.assertIs(view_for_id("spire-gym-combat-policy-view-v1"), COMMON_V1)
        self.assertIs(view_for_id("spire-gym-standard-combat-view-v1"), STANDARD_COMBAT_V1)
        self.assertEqual(STANDARD_COMBAT_V1.profile_id, "StandardCombat")
        self.assertEqual(STANDARD_COMBAT_V1.fingerprint_domain, b"spire-gym/combat-view/standard/v1\0")
        self.assertEqual(COMMON_V1.fingerprint_domain, b"spire-gym/combat-policy-view/v1\0")
        self.assertEqual(len(STANDARD_COMBAT_V1.groups), 16)
        self.assertIn("escaped", STANDARD_COMBAT_V1.outcomes)
        self.assertNotIn("escaped", COMMON_V1.outcomes)
        self.assertIn("orbs_occupied", STANDARD_COMBAT_V1.tensor_names)
        self.assertNotIn("orbs", COMMON_V1.tensor_names)
        with self.assertRaises(ContractError):
            view_for_id("spire-gym-unknown-view")

    def test_registry_matches_captured_descriptors(self):
        from spire_learn.neural.views import COMMON_V1, STANDARD_COMBAT_V1
        for view, name in ((COMMON_V1, "common-1929e18"), (STANDARD_COMBAT_V1, "standard-combat-1929e18")):
            with self.subTest(view=view.view_id):
                keys = {tensor["key"] for tensor in _load(name)["tensors"]}
                self.assertEqual(keys, view.tensor_names)


class StandardSchemaTests(unittest.TestCase):
    def test_standard_fixture_parses(self):
        from spire_learn.neural.views import STANDARD_COMBAT_V1
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        self.assertIs(parsed.view, STANDARD_COMBAT_V1)
        self.assertEqual(parsed.schema_fingerprint, GYM_V1_CONTRACT.standard_combat_fingerprint)
        self.assertEqual(parsed.gym_revision, V1)
        self.assertIn("orbs", parsed.tensors)
        self.assertEqual(parsed.action_space.n, 2048)

    def test_standard_parse_without_revision_is_unambiguous(self):
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"))
        self.assertEqual(parsed.gym_revision, V1)

    def test_common_fixture_still_parses_under_both_revisions(self):
        from spire_learn.neural.views import COMMON_V1
        old = schema.parse_combat_descriptor(_load("common-afb51f5"), revision=AFB)
        new = schema.parse_combat_descriptor(_load("common-1929e18"), revision=V1)
        self.assertIs(old.view, COMMON_V1)
        self.assertIs(new.view, COMMON_V1)
        self.assertEqual((old.gym_revision, new.gym_revision), (AFB, V1))
        self.assertEqual(old.schema_fingerprint, new.schema_fingerprint)
        with self.assertRaisesRegex(ContractError, "revision is required"):
            schema.parse_combat_descriptor(_load("common-1929e18"))

    def test_view_mismatch_rejected(self):
        with self.assertRaises(ContractError):
            schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=AFB)
        with self.assertRaises(ContractError):
            schema.parse_common_descriptor(_load("standard-combat-1929e18"), revision=V1)

    def test_tampered_standard_descriptor_rejected(self):
        changed = copy.deepcopy(_load("standard-combat-1929e18"))
        changed["profile_id"] = "IroncladCombatPolicy"
        with self.assertRaises(ContractError):
            schema.parse_combat_descriptor(changed, revision=V1)

    def test_standard_feature_plan(self):
        from spire_learn.neural.features import feature_contract
        from spire_learn.neural.views import STANDARD_COMBAT_V1
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        plan = feature_contract(parsed)
        self.assertEqual(set(plan.groups), set(STANDARD_COMBAT_V1.groups))

    def test_standard_authored_frame_validates(self):
        from tests.neural._support import authored_frame
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        frame = authored_frame(parsed)
        schema.validate_frame(frame, parsed)


class FrameAndLifecycleTests(unittest.TestCase):
    def test_frame_rejects_foreign_group(self):
        orbs = BufferArray("int8", (1,), b"\0")
        common = GYM_V1_CONTRACT.common_fingerprint
        with self.assertRaises(ContractError):
            CombatFrame({"orbs_known": orbs}, common)
        frame = CombatFrame({"orbs_known": orbs}, GYM_V1_CONTRACT.standard_combat_fingerprint)
        self.assertIn("orbs_known", frame.tensors)

    def test_lifecycle_view_and_outcome(self):
        from spire_learn.neural.views import COMMON_V1, STANDARD_COMBAT_V1
        standard = STANDARD_COMBAT_V1.lifecycle_view_id
        lifecycle = schema.parse_lifecycle(_lifecycle_info(standard), STANDARD_COMBAT_V1)
        self.assertFalse(hasattr(lifecycle, "effective_seed"))
        with self.assertRaises(ContractError):
            schema.parse_lifecycle(_lifecycle_info(standard), COMMON_V1)
        with self.assertRaises(ContractError):
            schema.parse_lifecycle(_lifecycle_info(COMMON_V1.lifecycle_view_id, outcome="escaped"),
                                   COMMON_V1)
        schema.parse_lifecycle(_lifecycle_info(COMMON_V1.lifecycle_view_id))


if __name__ == "__main__":
    unittest.main()
