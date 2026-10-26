"""Structural tests for descriptor-bound neural combat schemas."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
import hashlib
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import contracts as neural_contracts
        from spire_learn.neural import schema
        from spire_learn.neural.contracts import BufferArray, CombatFrame
        from spire_learn.neural.snapshot import snapshot_common
    except ImportError as error:
        test.fail(f"neural schema API is unavailable: {error}")
    return SimpleNamespace(
        schema=schema,
        neural_contracts=neural_contracts,
        BufferArray=BufferArray,
        CombatFrame=CombatFrame,
        snapshot_common=snapshot_common,
    )


def _column(field, unit, low, high, *, vocabulary=None, reference=None):
    result = {"field": field, "unit": unit, "low": low, "high": high}
    if vocabulary is not None:
        result["vocabulary"] = vocabulary
    if reference is not None:
        result["reference"] = reference
        result["unavailable"] = -1
    return result


def _descriptor() -> dict[str, object]:
    vocabularies = {
        "card": ["padding", "strike", "defend"],
        "enemy": ["padding", "cultist", "jaw_worm"],
        "potion": ["padding", "fire_potion", "block_potion"],
        "verb": ["padding", "use_potion", "play_card", "end_turn", "choose", "discard_potion"],
        "row_group": ["none", "potions", "combat_cards", "enemies", "player"],
        "choice": ["padding", "upgrade", "card_reward"],
    }
    tensors: list[dict[str, object]] = []

    def binary(key: str, shape: list[int]) -> None:
        tensors.append({
            "key": key, "dtype": "int8", "shape": shape,
            "low": 0, "high": 1, "space": "MultiBinary", "padding": 0,
        })

    def values(key: str, shape: list[int], columns: list[dict[str, object]]) -> None:
        is_rows = len(shape) == 2
        tensors.append({
            "key": key,
            "dtype": "int64",
            "shape": shape,
            "low": [column["low"] for column in columns],
            "high": [column["high"] for column in columns],
            "space": "Box",
            "fields": [column["field"] for column in columns],
            "units": [column["unit"] for column in columns],
            "columns": columns,
            "padding": 0,
            "known": f"{key}_known",
            "reference_columns": [
                index for index, column in enumerate(columns) if "reference" in column
            ],
        })
        binary(f"{key}_known", shape)
        if is_rows:
            binary(f"{key}_occupied", [shape[0]])

    values("player", [1], [_column("hp", "hp", 0, 100)])
    values("combat_cards", [2, 1], [
        _column("card_kind", "category", 0, 2, vocabulary="card"),
    ])
    values("enemies", [2, 1], [
        _column("enemy_kind", "category", 0, 2, vocabulary="enemy"),
    ])
    values("potions", [3, 1], [
        _column("potion_kind", "category", 0, 2, vocabulary="potion"),
    ])
    action_columns = [
        _column("verb", "category", 0, 5, vocabulary="verb"),
        _column("source_group", "category", 0, 4, vocabulary="row_group"),
        _column("source_row", "source_row", -1, 9, reference="source_group"),
        _column("target_group", "category", 0, 4, vocabulary="row_group"),
        _column("target_row", "target_row", -1, 1, reference="target_group"),
        _column("choice_kind", "category", 0, 2, vocabulary="choice"),
        _column("scalar0", "verb_specific", 0, 9),
        _column("scalar1", "verb_specific", 0, 9),
    ]
    values("actions", [10, 8], action_columns)
    binary("action_mask", [10])
    return {
        "schema_id": "spire-gym-combat-policy-view-v1",
        "profile_id": "IroncladCombatPolicy",
        "array_layout": "owned_c_contiguous_native_endian_row_major",
        "action_space": {"type": "Discrete", "n": 10, "start": 0},
        "capacities": {"combat_cards": 2, "enemies": 2, "potions": 3, "actions": 10},
        "tensors": tensors,
        "vocabularies": vocabularies,
    }


def _native_fingerprint(descriptor: dict[str, object]) -> str:
    """Independent test implementation of the documented native body hash."""
    body = {key: value for key, value in descriptor.items() if key != "fingerprint"}
    canonical = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"spire-gym/combat-policy-view/v1\0" + canonical).hexdigest()


@contextmanager
def _synthetic_schema(api: SimpleNamespace):
    """Parse a small synthetic descriptor with only test-local expected hashes."""
    from spire_learn.neural.config import COMMON_VIEW_FINGERPRINT

    descriptor = _descriptor()
    fingerprint = _native_fingerprint(descriptor)
    if fingerprint == COMMON_VIEW_FINGERPRINT:
        raise AssertionError("synthetic descriptor unexpectedly equals the native accepted hash")
    descriptor["fingerprint"] = fingerprint
    with (
        patch.object(api.schema, "COMMON_VIEW_FINGERPRINT", fingerprint),
        patch.object(api.neural_contracts, "_COMMON_VIEW_FINGERPRINT", fingerprint),
    ):
        yield api.schema.parse_common_descriptor(descriptor)


def _buffer(api: SimpleNamespace, values, shape, dtype="int64"):
    if dtype == "int8":
        data = bytes(values)
    else:
        data = struct.pack("=" + "q" * len(values), *values)
    return api.BufferArray(dtype, shape, data)


def _frame(api: SimpleNamespace, schema, *, legal=(2, 9), verb="use_potion",
           potion_source=0, potion_scalar=0, scalar0_known=None,
           scalar1=0, scalar1_known=False, choice_value=0, choice_known=False,
           target_group_known=False, target_row=None, target_row_known=False,
           enemy_occupied=(1, 0)):
    tensors = {
        "player": _buffer(api, [42], (1,)),
        "player_known": _buffer(api, [1], (1,), "int8"),
        "combat_cards": _buffer(api, [1, 0], (2, 1)),
        "combat_cards_known": _buffer(api, [1, 0], (2, 1), "int8"),
        "combat_cards_occupied": _buffer(api, [1, 0], (2,), "int8"),
        "enemies": _buffer(api, [1, 0], (2, 1)),
        "enemies_known": _buffer(api, [1, 0], (2, 1), "int8"),
        "enemies_occupied": _buffer(api, list(enemy_occupied), (2,), "int8"),
        # Empty potion slots are known zeroes with occupied=0.
        "potions": _buffer(api, [1, 0, 0], (3, 1)),
        "potions_known": _buffer(api, [1, 1, 1], (3, 1), "int8"),
        "potions_occupied": _buffer(api, [1, 0, 0], (3,), "int8"),
    }
    actions = [0] * 80
    known = [0] * 80
    occupied = [0] * 10
    mask = [0] * 10
    verb_ids = {
        "use_potion": 1, "play_card": 2, "end_turn": 3,
        "choose": 4, "discard_potion": 5,
    }
    potion_action = verb in {"use_potion", "discard_potion"}
    scalar0_known = potion_action if scalar0_known is None else scalar0_known
    if 2 in legal:
        source_group = 1 if potion_action else 0
        source_row = potion_source if potion_action else -1
        target_group = 3 if target_group_known else 0
        target_value = -1 if target_row is None else target_row
        actions[16:24] = [
            verb_ids[verb], source_group, source_row, target_group, target_value,
            choice_value if choice_known else 0,
            potion_scalar if scalar0_known else 0,
            scalar1 if scalar1_known else 0,
        ]
        known[16:24] = [
            1, int(potion_action), int(potion_action), int(target_group_known),
            int(target_row_known), int(choice_known), int(scalar0_known), int(scalar1_known),
        ]
        occupied[2] = 1
    if 9 in legal:
        actions[72:80] = [3, 0, -1, 0, -1, 0, 0, 0]
        known[72:80] = [1, 0, 0, 0, 0, 0, 0, 0]
        occupied[9] = 1
    for index in legal:
        mask[index] = 1
    tensors.update({
        "actions": _buffer(api, actions, (10, 8)),
        "actions_known": _buffer(api, known, (10, 8), "int8"),
        "actions_occupied": _buffer(api, occupied, (10,), "int8"),
        "action_mask": _buffer(api, mask, (10,), "int8"),
    })
    return api.CombatFrame(tensors, schema.schema_fingerprint)


class NeuralSchemaTests(unittest.TestCase):
    def test_common_descriptor_requires_exact_identity(self):
        api = _api(self)
        from spire_learn.neural.config import COMMON_VIEW_FINGERPRINT, COMMON_VIEW_ID

        descriptor = {
            "schema_id": "spire-gym-combat-policy-view-v0",
            "profile_id": "IroncladCombatPolicy",
            "array_layout": "owned_c_contiguous_native_endian_row_major",
            "action_space": {"type": "Discrete", "n": 1, "start": 0},
            "capacities": {"actions": 1},
            "tensors": [],
            "vocabularies": {},
            "fingerprint": COMMON_VIEW_FINGERPRINT,
        }
        with self.assertRaises(api.schema.ContractError):
            api.schema.parse_common_descriptor(descriptor)
        descriptor["schema_id"] = COMMON_VIEW_ID
        # A self-asserted accepted tag cannot bless a different descriptor body.
        with self.assertRaises(api.schema.ContractError):
            api.schema.parse_common_descriptor(descriptor)

    def test_unknown_is_not_zero_or_padding(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema)
            api.schema.validate_frame(frame, schema)
            known_empty_potion = (
                frame.tensors["potions"].data[8:16],
                frame.tensors["potions_known"].data[1:2],
                frame.tensors["potions_occupied"].data[1:2],
            )
            unknown_padding = (
                frame.tensors["combat_cards"].data[8:16],
                frame.tensors["combat_cards_known"].data[1:2],
                frame.tensors["combat_cards_occupied"].data[1:2],
            )
            self.assertEqual(known_empty_potion[0], unknown_padding[0])
            self.assertNotEqual(known_empty_potion, unknown_padding)

    def test_reference_must_resolve_to_occupied_row(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(
                api, schema, legal=(2,), target_group_known=True, target_row=1,
                target_row_known=True, enemy_occupied=(1, 0),
            )
            with self.assertRaises(api.schema.ContractError):
                api.schema.validate_frame(frame, schema)

    def test_dispatch_rows_are_not_compacted(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema, legal=(2, 9))
            found = api.schema.candidates(frame, schema)
            self.assertEqual(tuple(candidate.index for candidate in found), (2, 9))

    def test_potion_source_slot_is_typed_and_matches_known_scalar(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema, legal=(2,), potion_source=0, potion_scalar=0)
            candidate = api.schema.candidates(frame, schema)[0]
            self.assertEqual((candidate.source.group, candidate.source.row), ("potions", 0))
            self.assertTrue(candidate.source_applicable)

            bad_frame = _frame(api, schema, legal=(2,), potion_source=0, potion_scalar=1)
            with self.assertRaises(api.schema.ContractError):
                api.schema.candidates(bad_frame, schema)

    def test_candidates_keep_applicability_separate_from_knownness(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            choose = _frame(
                api, schema, legal=(2,), verb="choose", choice_known=False,
            )
            choose_candidate = api.schema.candidates(choose, schema)[0]
            self.assertIsNone(choose_candidate.choice)
            self.assertFalse(choose_candidate.choice_known)
            self.assertTrue(choose_candidate.choice_applicable)
            self.assertFalse(choose_candidate.scalar0_applicable)

            potion = _frame(
                api, schema, legal=(2,), verb="use_potion", scalar0_known=False,
            )
            potion_candidate = api.schema.candidates(potion, schema)[0]
            self.assertIsNone(potion_candidate.scalar0)
            self.assertFalse(potion_candidate.scalar0_known)
            self.assertTrue(potion_candidate.scalar0_applicable)
            self.assertFalse(potion_candidate.scalar1_applicable)

    def test_candidates_reject_known_values_on_inapplicable_fields(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            cases = (
                {"verb": "play_card", "choice_known": True, "choice_value": 1},
                {"verb": "play_card", "scalar0_known": True, "potion_scalar": 1},
                {"verb": "use_potion", "scalar1_known": True, "scalar1": 1},
            )
            for fields in cases:
                with self.subTest(fields=fields), self.assertRaises(api.schema.ContractError):
                    frame = _frame(api, schema, legal=(2,), **fields)
                    api.schema.candidates(frame, schema)

    def test_candidate_record_accepts_applicable_unknown_values(self):
        api = _api(self)
        candidate = api.schema.CombatCandidate(
            index=2,
            verb="choose",
            source=None,
            target=None,
            choice=None,
            choice_known=False,
            choice_applicable=True,
            scalar0=None,
            scalar0_known=False,
            scalar0_applicable=False,
            scalar1=None,
            scalar1_known=False,
            scalar1_applicable=False,
        )
        self.assertTrue(candidate.choice_applicable)
        self.assertFalse(candidate.choice_known)

    def test_public_schema_construction_cannot_claim_parsed_identity(self):
        api = _api(self)
        with _synthetic_schema(api) as parsed:
            public_fields = {
                field.name: getattr(parsed, field.name)
                for field in fields(api.schema.CombatSchema)
            }
            with self.assertRaises((TypeError, api.schema.ContractError)):
                api.schema.CombatSchema(**public_fields)

    def test_dataclasses_replace_cannot_retain_schema_authority(self):
        api = _api(self)
        with _synthetic_schema(api) as parsed:
            changed_tensors = dict(parsed.tensors)
            player = changed_tensors["player"]
            changed_tensors["player"] = replace(
                player,
                columns=(replace(player.columns[0], high=99),),
            )
            changes = (
                {"tensors": changed_tensors},
                {"descriptor_digest": "f" * 64},
            )
            for mutation in changes:
                with self.subTest(mutation=tuple(mutation)), self.assertRaises(
                    (TypeError, ValueError, api.schema.ContractError)
                ):
                    replace(parsed, **mutation)

    def test_parsed_schema_metadata_is_deeply_immutable(self):
        api = _api(self)
        with _synthetic_schema(api) as parsed:
            with self.assertRaises(TypeError):
                parsed.tensors["untrusted"] = parsed.tensors["player"]
            with self.assertRaises(TypeError):
                parsed.capacities["actions"] = 2
            with self.assertRaises(TypeError):
                parsed.vocabularies["verb"][0] = "untrusted"

    def test_missing_and_unknown_targets_keep_applicability_distinct(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            missing = _frame(api, schema, legal=(2,))
            missing_target = api.schema.candidates(missing, schema)[0]
            self.assertFalse(missing_target.target_applicable)
            self.assertIsNone(missing_target.target)

            unknown = _frame(
                api, schema, legal=(2,), target_group_known=True,
                target_row=None, target_row_known=False,
            )
            unknown_target = api.schema.candidates(unknown, schema)[0]
            self.assertTrue(unknown_target.target_applicable)
            self.assertIsNotNone(unknown_target.target)
            self.assertFalse(unknown_target.target.known)
            self.assertIsNone(unknown_target.target.row)

    def test_terminal_empty_actor_keeps_resource_frame_but_has_no_candidates(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema, legal=())
            api.schema.validate_frame(frame, schema)
            self.assertEqual(frame.tensors["player"].data, _buffer(api, [42], (1,)).data)
            with self.assertRaises(api.schema.ContractError):
                api.schema.candidates(frame, schema)

    def test_card_and_potion_selection_owners_are_parsed_as_metadata(self):
        api = _api(self)
        for owner in ("card", "potion", "end_turn"):
            info = {
                "combat_view_id": "spire-gym-combat-policy-view-v1",
                "active": True,
                "routing": "combat_selection",
                "selection_owner": owner,
                "active_visit": 0,
                "combats_entered": 1,
                "combats_settled": 0,
                "last_exit": None,
                "effective_seed": 99,
            }
            lifecycle = api.schema.parse_lifecycle(info)
            self.assertEqual(lifecycle.selection_owner, owner)
            self.assertFalse(hasattr(lifecycle, "effective_seed"))


if __name__ == "__main__":
    unittest.main()
