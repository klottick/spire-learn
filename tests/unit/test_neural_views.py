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

    def test_malformed_lifecycle_raises_contract_error(self):
        from spire_learn.neural.views import STANDARD_COMBAT_V1
        view_id = STANDARD_COMBAT_V1.lifecycle_view_id
        exit_record = _lifecycle_info(view_id)["last_exit"]
        for info in (
            _lifecycle_info(view_id, routing=["noncombat"]),
            _lifecycle_info(view_id, last_exit=dict(exit_record, outcome=["victory"])),
        ):
            with self.subTest(info=info), self.assertRaises(ContractError):
                schema.parse_lifecycle(info, STANDARD_COMBAT_V1)
        missing = _lifecycle_info(view_id)
        del missing["routing"]
        with self.assertRaisesRegex(ContractError, "missing fields: routing"):
            schema.parse_lifecycle(missing, STANDARD_COMBAT_V1)


if __name__ == "__main__":
    unittest.main()


def _with_unknown_zone_position(frame, parsed, value):
    """Return ``frame`` with occupied combat_cards row 0's zone_position unknown at ``value``."""
    import struct
    from spire_learn.neural.contracts import BufferArray, CombatFrame
    spec = parsed.tensors["combat_cards"]
    columns = len(spec.columns)
    column = next(i for i, c in enumerate(spec.columns) if c.field == "zone_position")
    occupied = frame.tensors[spec.occupied_key].data
    assert occupied[0] == 1, "authored frame must occupy combat_cards row 0"
    tensors = dict(frame.tensors)
    values = list(struct.unpack(f"={len(frame.tensors['combat_cards'].data) // 8}q",
                                frame.tensors["combat_cards"].data))
    known = bytearray(frame.tensors[spec.known_key].data)
    values[column] = value
    known[column] = 0
    tensors["combat_cards"] = BufferArray("int64", spec.shape, struct.pack(f"={len(values)}q", *values))
    tensors[spec.known_key] = BufferArray("int8", spec.shape, bytes(known))
    return CombatFrame(tensors, frame.schema_fingerprint)


class UnknownSentinelTests(unittest.TestCase):
    def test_standard_hidden_draw_zone_position_sentinel(self):
        from tests.neural._support import authored_frame
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        frame = authored_frame(parsed)
        schema.validate_frame(_with_unknown_zone_position(frame, parsed, -1), parsed)
        schema.validate_frame(_with_unknown_zone_position(frame, parsed, 0), parsed)
        with self.assertRaises(ContractError):
            schema.validate_frame(_with_unknown_zone_position(frame, parsed, 5), parsed)

    def test_common_view_keeps_padding_for_unknown_zone_position(self):
        from tests.neural._support import authored_frame
        parsed = schema.parse_combat_descriptor(_load("common-1929e18"), revision=V1)
        frame = authored_frame(parsed)
        schema.validate_frame(_with_unknown_zone_position(frame, parsed, 0), parsed)
        with self.assertRaises(ContractError):
            schema.validate_frame(_with_unknown_zone_position(frame, parsed, -1), parsed)


def _with_cell(frame, parsed, key, row, field, value, known):
    """Return ``frame`` with one cell of an unoccupied ``key`` row rewritten."""
    import struct
    from spire_learn.neural.contracts import BufferArray, CombatFrame
    spec = parsed.tensors[key]
    columns = len(spec.columns)
    assert frame.tensors[spec.occupied_key].data[row] == 0, "row must be unoccupied"
    index = row * columns + next(i for i, c in enumerate(spec.columns) if c.field == field)
    raw = frame.tensors[key].data
    values = list(struct.unpack(f"={len(raw) // 8}q", raw))
    known_bytes = bytearray(frame.tensors[spec.known_key].data)
    values[index] = value
    known_bytes[index] = known
    tensors = dict(frame.tensors)
    tensors[key] = BufferArray("int64", spec.shape, struct.pack(f"={len(values)}q", *values))
    tensors[spec.known_key] = BufferArray("int8", spec.shape, bytes(known_bytes))
    return CombatFrame(tensors, frame.schema_fingerprint)


class EmptySlotTests(unittest.TestCase):
    def test_standard_empty_slots_may_be_known_or_unused_capacity(self):
        from tests.neural._support import authored_frame
        parsed = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        frame = authored_frame(parsed)
        for key, field in (("potions", "potion_kind"), ("orbs", "kind")):
            last = parsed.tensors[key].shape[0] - 1
            with self.subTest(key=key):
                schema.validate_frame(_with_cell(frame, parsed, key, last, field, 0, 1), parsed)
                schema.validate_frame(_with_cell(frame, parsed, key, last, field, 0, 0), parsed)
                with self.assertRaises(ContractError):
                    schema.validate_frame(_with_cell(frame, parsed, key, last, field, 1, 1), parsed)

    def test_common_empty_potion_slots_stay_known(self):
        from tests.neural._support import authored_frame
        parsed = schema.parse_combat_descriptor(_load("common-1929e18"), revision=V1)
        frame = authored_frame(parsed)
        last = parsed.tensors["potions"].shape[0] - 1
        if frame.tensors["potions_occupied"].data[last]:
            self.skipTest("authored common frame occupies every potion slot")
        with self.assertRaises(ContractError):
            schema.validate_frame(_with_cell(frame, parsed, "potions", last, "potion_kind", 0, 0), parsed)
