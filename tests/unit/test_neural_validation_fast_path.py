"""Differential tests: the fast validation path never changes a validation decision.

The reference validator (``_validate_frame_contents_reference``) is the authority.
The fast path may only accept frames the reference accepts; every rejection and
its message still comes from the reference.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import schema
from spire_learn.neural.contracts import BufferArray, CombatFrame
from spire_learn.neural.gym_contracts import GYM_V1_CONTRACT
from tests.neural._support import authored_frame
from tests.unit.test_neural_schema import _api, _buffer, _frame, _synthetic_schema

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/neural"
V1 = GYM_V1_CONTRACT.source_revision


def _load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _outcome(function, frame, parsed):
    try:
        function(frame, parsed)
    except Exception as error:  # the type and message are both part of the decision
        return type(error).__name__, str(error)
    return "accepted"


def _with_value(frame, key, index, value):
    """A new frame with one decoded cell replaced, keeping every other byte."""
    array = frame.tensors[key]
    code = "b" if array.dtype == "int8" else "q"
    size = struct.calcsize("=" + code)
    try:
        cell = struct.pack("=" + code, value)
    except struct.error:
        return None
    data = array.data[:index * size] + cell + array.data[(index + 1) * size:]
    tensors = dict(frame.tensors)
    tensors[key] = BufferArray(array.dtype, array.shape, data)
    return CombatFrame(tensors, frame.schema_fingerprint)


def _probe_values(parsed, spec, column):
    values = {0, 1, -1, spec.padding, column.low, column.high, column.low - 1, column.high + 1,
              2**63 - 1, -(2**63)}
    if column.unavailable is not None:
        values.add(column.unavailable)
    if column.vocabulary is not None:
        size = len(parsed.vocabularies[column.vocabulary])
        values.update({size - 1, size})
    values.update(value for tensor, field, value in parsed.view.unknown_sentinels
                  if tensor == spec.key and field == column.field)
    return sorted(values)


def _sample_rows(frame, spec, limit):
    rows = spec.shape[0] if len(spec.shape) == 2 else 1
    if spec.occupied_key is None:
        return range(rows)
    occupied = frame.tensors[spec.occupied_key].data
    filled = [row for row in range(rows) if occupied[row]]
    empty = [row for row in range(rows) if not occupied[row]]
    after = [row for row in empty if filled and row == filled[-1] + 1]
    picked = filled[:limit] + empty[:1] + after + empty[-1:]
    if limit > 1:  # small synthetic tensors: probe more of the empty rows too
        picked += empty[:2] + empty[len(empty) // 2:len(empty) // 2 + 1]
    return sorted(set(picked))


def _mutations(frame, parsed, *, row_limit):
    """Single-cell changes: value probes per column, plus bit changes on int8 tensors."""
    for key, spec in parsed.tensors.items():
        if spec.columns:
            columns = len(spec.columns)
            for row in _sample_rows(frame, spec, row_limit):
                for index, column in enumerate(spec.columns):
                    cell = row * columns + index
                    for value in _probe_values(parsed, spec, column):
                        yield f"{key}[{row},{column.field}]={value}", _with_value(frame, key, cell, value)
        elif spec.dtype == "int8":
            size = len(frame.tensors[key].data)
            for cell in sorted({0, 1, 2, size // 2, size - 1} & set(range(size))):
                for value in (0, 1, 2, -1):
                    yield f"{key}[{cell}]={value}", _with_value(frame, key, cell, value)
    for key, spec in parsed.tensors.items():
        if spec.known_key is None:
            continue
        known = frame.tensors[spec.known_key]
        columns = len(spec.columns)
        for row in _sample_rows(frame, spec, row_limit):
            for index in range(columns):
                cell = row * columns + index
                flipped = 1 - known.data[cell]
                yield f"{spec.known_key}[{row},{index}]={flipped}", _with_value(
                    frame, spec.known_key, cell, flipped)


class ValidationFastPathTests(unittest.TestCase):
    def fixtures(self):
        """(label, frame, schema) triples, synthetic plus real-capacity descriptors."""
        api = _api(self)
        cases = []
        with _synthetic_schema(api) as synthetic:
            for options in ({}, {"verb": "play_card"}, {"legal": (9,)},
                            {"verb": "choose", "choice_known": True, "choice_value": 1},
                            {"enemy_occupied": (1, 1), "target_group_known": True,
                             "target_row": 1, "target_row_known": True}):
                cases.append((f"synthetic {options}", _frame(api, synthetic, **options), synthetic))
            # The synthetic schema needs its patched fingerprint for every validation call.
            yield from self._run_cases(cases, row_limit=99)
        common = schema.parse_common_descriptor(_load("common-1929e18"), revision=V1)
        standard = schema.parse_combat_descriptor(_load("standard-combat-1929e18"), revision=V1)
        yield from self._run_cases([
            ("common", authored_frame(common), common),
            ("standard", authored_frame(standard), standard),
        ], row_limit=1)

    def _run_cases(self, cases, *, row_limit):
        for label, frame, parsed in cases:
            yield label, frame, parsed, row_limit

    def test_valid_frames_are_accepted_by_the_fast_path_itself(self):
        for label, frame, parsed, _ in self.fixtures():
            with self.subTest(label):
                self.assertEqual(_outcome(schema._validate_frame_contents_reference, frame, parsed),
                                 "accepted")
                self.assertTrue(schema._frame_is_valid_fast(frame, parsed))

    def test_every_single_cell_mutation_gets_the_reference_decision(self):
        checked = rejected = composite = 0
        for label, frame, parsed, row_limit in self.fixtures():
            for name, mutated in _mutations(frame, parsed, row_limit=row_limit):
                if mutated is None:
                    continue
                reference = _outcome(schema._validate_frame_contents_reference, mutated, parsed)
                try:
                    fast = schema._frame_is_valid_fast(mutated, parsed)
                except Exception:
                    fast = False
                # Sound: the fast path never accepts what the reference rejects.
                # Complete: valid frames never fall back to the slow reference.
                if fast != (reference == "accepted"):
                    self.fail(f"{label} {name}: reference={reference!r} fast={fast}")
                if checked % 7 == 0:  # the public composite, including messages, on a fixed sample
                    actual = _outcome(schema._validate_frame_contents, mutated, parsed)
                    if actual != reference:
                        self.fail(f"{label} {name}: reference={reference!r} actual={actual!r}")
                    composite += 1
                checked += 1
                rejected += reference != "accepted"
        # The probes must exercise both decisions heavily, or the comparison proves little.
        self.assertGreater(checked, 4000)
        self.assertGreater(rejected, checked // 4)
        self.assertGreater(checked - rejected, 500)
        self.assertGreater(composite, checked // 8)

    def test_nonzero_padding_is_compared_against_the_tensor_padding(self):
        # Real descriptors pad with 0 today, but the parser accepts any integer padding.
        from tests.unit import test_neural_schema as fixtures
        api = _api(self)
        original = fixtures._descriptor

        def padded():
            descriptor = original()
            for tensor in descriptor["tensors"]:
                if tensor["key"] == "combat_cards":
                    tensor["padding"] = 7
            return descriptor

        with patch.object(fixtures, "_descriptor", padded), _synthetic_schema(api) as parsed:
            self.assertEqual(parsed.tensors["combat_cards"].padding, 7)
            frame = _with_value(_frame(api, parsed), "combat_cards", 1, 7)  # row 1 is unoccupied
            self.assertEqual(_outcome(schema._validate_frame_contents_reference, frame, parsed), "accepted")
            self.assertTrue(schema._frame_is_valid_fast(frame, parsed))
            for value in (0, 6, 8):
                with self.subTest(value=value):
                    wrong = _with_value(frame, "combat_cards", 1, value)
                    self.assertFalse(schema._frame_is_valid_fast(wrong, parsed))
                    self.assertEqual(
                        _outcome(schema._validate_frame_contents, wrong, parsed),
                        _outcome(schema._validate_frame_contents_reference, wrong, parsed))
                    self.assertNotEqual(
                        _outcome(schema._validate_frame_contents_reference, wrong, parsed), "accepted")

    def test_wrong_shape_dtype_and_fingerprint_are_left_to_the_reference(self):
        common = schema.parse_common_descriptor(_load("common-1929e18"), revision=V1)
        frame = authored_frame(common)
        player = frame.tensors["player"]
        for label, tensors in (
            ("dtype", {**frame.tensors, "player": BufferArray("int8", (len(player.data),), player.data)}),
            ("missing", {key: value for key, value in frame.tensors.items() if key != "player"}),
        ):
            with self.subTest(label):
                broken = CombatFrame(tensors, frame.schema_fingerprint)
                self.assertFalse(schema._frame_is_valid_fast(broken, common))
                self.assertEqual(_outcome(schema._validate_frame_contents, broken, common),
                                 _outcome(schema._validate_frame_contents_reference, broken, common))

    def test_a_fast_path_exception_falls_back_to_the_reference(self):
        common = schema.parse_common_descriptor(_load("common-1929e18"), revision=V1)
        frame = authored_frame(common)
        with patch.object(schema, "_frame_is_valid_fast", side_effect=RuntimeError("bug")), \
             patch.object(schema, "_validate_frame_contents_reference",
                          wraps=schema._validate_frame_contents_reference) as reference:
            schema._validate_frame_contents(frame, common)
        self.assertEqual(reference.call_count, 1)

    def test_malformed_companion_tensors_are_left_to_the_reference(self):
        # Parser-rejected layouts from adversarial review: occupancy that is not one int8
        # flag per row must never be read as raw bytes by the fast path.
        api = _api(self)

        def with_specs(parsed, **updates):
            return replace(parsed, tensors={**parsed.tensors, **updates},
                           _parser_authority=schema._PARSER_AUTHORITY)

        with _synthetic_schema(api) as parsed:
            frame = _frame(api, parsed)
            tensors = dict(frame.tensors)
            cases = {}
            cases["int64 occupancy"] = (
                with_specs(parsed, combat_cards_occupied=replace(
                    parsed.tensors["combat_cards_occupied"], dtype="int64")),
                {**tensors, "combat_cards_occupied": _buffer(api, [256, 0], (2,)),
                 "combat_cards": _buffer(api, [0, 1], (2, 1)),
                 "combat_cards_known": _buffer(api, [0, 1], (2, 1), "int8")})
            cases["occupancy aliased to a short tensor"] = (
                with_specs(parsed, combat_cards=replace(
                    parsed.tensors["combat_cards"], occupied_key="player")),
                {**tensors, "player": _buffer(api, [1], (1,))})
            cases["oversized action mask"] = (
                with_specs(parsed,
                           action_mask=replace(parsed.tensors["action_mask"], shape=(11,)),
                           actions_occupied=replace(parsed.tensors["actions_occupied"], shape=(11,))),
                {**tensors, **{key: BufferArray("int8", (11,), tensors[key].data + b"\x01")
                               for key in ("action_mask", "actions_occupied")}})
            for label, (malformed, changed) in cases.items():
                with self.subTest(label):
                    broken = CombatFrame(changed, malformed.schema_fingerprint)
                    if label != "oversized action mask":
                        self.assertFalse(schema._frame_is_valid_fast(broken, malformed))
                    self.assertEqual(
                        _outcome(schema._validate_frame_contents, broken, malformed),
                        _outcome(schema._validate_frame_contents_reference, broken, malformed))
            # Both validators accept the oversized mask; candidates keeps the tuple IndexError.
            malformed, changed = cases["oversized action mask"]
            broken = CombatFrame(changed, malformed.schema_fingerprint)
            self.assertEqual(_outcome(schema.candidates, broken, malformed),
                             ("IndexError", "tuple index out of range"))


if __name__ == "__main__":
    unittest.main()
