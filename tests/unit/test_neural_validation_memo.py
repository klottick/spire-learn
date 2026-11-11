"""Pure tests for exact-pair weak memoization of deep frame validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
import gc
import struct
import unittest
import weakref
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from tests.unit.test_neural_schema import _buffer, _frame, _synthetic_schema


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import contracts as neural_contracts
        from spire_learn.neural import schema
        from spire_learn.neural.contracts import BufferArray, CombatFrame
    except ImportError as error:
        test.fail(f"neural schema API is unavailable: {error}")
    return SimpleNamespace(
        schema=schema, neural_contracts=neural_contracts,
        BufferArray=BufferArray, CombatFrame=CombatFrame,
    )


def _frame_with(api, schema, original, **updates):
    tensors = dict(original.tensors)
    tensors.update(updates)
    return api.CombatFrame(tensors, schema.schema_fingerprint)


class NeuralValidationMemoTests(unittest.TestCase):
    def test_identical_valid_pair_skips_repeated_deep_unpacking(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema)
            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                self.assertIsNone(api.schema.validate_frame(frame, schema))
                first_call_count = unpack.call_count
                self.assertEqual(first_call_count, len(schema.tensors))
                self.assertIsNone(api.schema.validate_frame(frame, schema))
                self.assertEqual(unpack.call_count, first_call_count)

    def test_invalid_padding_domain_known_occupancy_reference_and_mask_never_cache(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            good = _frame(api, schema)
            bad_reference = _frame(
                api,
                schema,
                legal=(2,),
                target_group_known=True,
                target_row=1,
                target_row_known=True,
                enemy_occupied=(1, 0),
            )
            invalid_cases = {
                "padding": _frame_with(
                    api,
                    schema,
                    good,
                    combat_cards=_buffer(api, [1, 1], (2, 1)),
                ),
                "domain": _frame_with(
                    api,
                    schema,
                    good,
                    player=_buffer(api, [101], (1,)),
                ),
                "known": _frame_with(
                    api,
                    schema,
                    good,
                    player_known=_buffer(api, [0], (1,), "int8"),
                ),
                "occupancy": _frame_with(
                    api,
                    schema,
                    good,
                    combat_cards_occupied=_buffer(api, [0, 0], (2,), "int8"),
                ),
                "occupancy_binary": _frame_with(
                    api,
                    schema,
                    good,
                    enemies_occupied=_buffer(api, [1, 2], (2,), "int8"),
                ),
                "reference": bad_reference,
                "mask": _frame_with(
                    api,
                    schema,
                    good,
                    action_mask=_buffer(api, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], (10,), "int8"),
                ),
            }
            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                for name, invalid in invalid_cases.items():
                    with self.subTest(case=name):
                        with self.assertRaises(api.schema.ContractError):
                            api.schema.validate_frame(invalid, schema)
                        first_failure_work = unpack.call_count
                        self.assertGreater(first_failure_work, 0)
                        with self.assertRaises(api.schema.ContractError):
                            api.schema.validate_frame(invalid, schema)
                        self.assertGreater(unpack.call_count, first_failure_work)

    def test_same_fingerprint_different_schema_identity_revalidates(self):
        api = _api(self)
        with _synthetic_schema(api) as first_schema:
            with _synthetic_schema(api) as second_schema:
                self.assertIsNot(first_schema, second_schema)
                self.assertEqual(first_schema.schema_fingerprint, second_schema.schema_fingerprint)
                frame = _frame(api, first_schema)
                with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                    api.schema.validate_frame(frame, first_schema)
                    self.assertEqual(unpack.call_count, len(first_schema.tensors))
                    api.schema.validate_frame(frame, second_schema)
                    self.assertEqual(unpack.call_count, 2 * len(second_schema.tensors))

    def test_dataclass_replacements_get_new_frame_and_buffer_certificates(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema)
            new_player = replace(frame.tensors["player"])
            self.assertIsNot(new_player, frame.tensors["player"])
            changed = replace(frame, tensors={**frame.tensors, "player": new_player})
            copied = replace(changed)
            self.assertIsNot(changed, frame)
            self.assertIsNot(copied, changed)
            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                api.schema.validate_frame(frame, schema)
                api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, len(schema.tensors))
                api.schema.validate_frame(changed, schema)
                self.assertEqual(unpack.call_count, 2 * len(schema.tensors))
                api.schema.validate_frame(copied, schema)
                self.assertEqual(unpack.call_count, 3 * len(schema.tensors))

    def test_raw_source_mutation_cannot_change_owned_immutable_bytes(self):
        api = _api(self)
        raw = bytearray(struct.pack("=q", 42))
        owned = bytes(raw)
        buffer = api.BufferArray("int64", (1,), owned)
        raw[0] ^= 0x7F
        self.assertEqual(buffer.data, owned)
        self.assertIs(type(buffer.data), bytes)

    def test_mutable_shape_tuple_inside_exact_records_is_revalidated(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            original = _frame(api, schema)

            class MutableShape(tuple):
                def __new__(cls, dimensions):
                    shape = super().__new__(cls, dimensions)
                    shape.invalid = False
                    return shape

                def __ne__(self, other):
                    return self.invalid or super().__ne__(other)

            player = original.tensors["player"]
            mutable_shape = MutableShape(player.shape)
            mutable_player = api.BufferArray(player.dtype, mutable_shape, player.data)
            frame = _frame_with(api, schema, original, player=mutable_player)

            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                api.schema.validate_frame(frame, schema)
                first_work = unpack.call_count
                api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

                mutable_shape.invalid = True
                with self.assertRaises(api.schema.ContractError):
                    api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

    def test_mutable_dtype_string_inside_exact_records_is_revalidated(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            original = _frame(api, schema)

            class MutableDType(str):
                def __new__(cls, value):
                    dtype = super().__new__(cls, value)
                    dtype.invalid = False
                    return dtype

                def __ne__(self, other):
                    return self.invalid or super().__ne__(other)

            player = original.tensors["player"]
            mutable_dtype = MutableDType(player.dtype)
            mutable_player = api.BufferArray(mutable_dtype, player.shape, player.data)
            frame = _frame_with(api, schema, original, player=mutable_player)

            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                api.schema.validate_frame(frame, schema)
                first_work = unpack.call_count
                api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

                mutable_dtype.invalid = True
                with self.assertRaises(api.schema.ContractError):
                    api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

    def test_mutable_nested_schema_shape_is_not_admitted(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            frame = _frame(api, schema)

            class MutableShape(tuple):
                def __new__(cls, dimensions):
                    shape = super().__new__(cls, dimensions)
                    shape.invalid = False
                    return shape

                def __ne__(self, other):
                    return self.invalid or super().__ne__(other)

            player_spec = schema.tensors["player"]
            mutable_shape = MutableShape(player_spec.shape)
            mutable_spec = replace(player_spec, shape=mutable_shape)
            object.__setattr__(
                schema,
                "tensors",
                MappingProxyType({**schema.tensors, "player": mutable_spec}),
            )

            with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                api.schema.validate_frame(frame, schema)
                first_work = unpack.call_count
                api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

                mutable_shape.invalid = True
                with self.assertRaises(api.schema.ContractError):
                    api.schema.validate_frame(frame, schema)
                self.assertEqual(unpack.call_count, 2 * first_work)

    def test_weak_cache_does_not_retain_frame_schema_or_tensor_bytes(self):
        api = _api(self)

        self.assertTrue(
            hasattr(api.schema, "_VALIDATION_MEMO"),
            "successful validation memo is not implemented",
        )
        if not hasattr(api.schema, "_VALIDATION_MEMO"):
            return

        def create_pair():
            with _synthetic_schema(api) as schema:
                frame = _frame(api, schema)
                frame_ref = weakref.ref(frame)
                schema_ref = weakref.ref(schema)
                api.schema.validate_frame(frame, schema)
                key = (id(frame), id(schema))
                entry = api.schema._VALIDATION_MEMO[key]
                self.assertEqual(len(entry), 2)
                self.assertTrue(all(type(reference) is weakref.ReferenceType for reference in entry))
                self.assertTrue(all(type(part) is int for part in key))
                return frame_ref, schema_ref, key, id(schema)

        frame_ref, schema_ref, pair_key, schema_key = create_pair()
        gc.collect()
        self.assertIsNone(frame_ref())
        self.assertIsNone(schema_ref())
        with api.schema._VALIDATION_MEMO_LOCK:
            self.assertNotIn(pair_key, api.schema._VALIDATION_MEMO)
            self.assertNotIn(schema_key, api.schema._SCHEMA_IMMUTABILITY_MEMO)

    def test_memo_evicts_oldest_entry_at_its_capacity(self):
        api = _api(self)
        capacity = getattr(api.schema, "_VALIDATION_MEMO_CAPACITY", None)
        self.assertEqual(capacity, 4096)
        if capacity is None:
            return
        with _synthetic_schema(api) as schema:
            frames = [_frame(api, schema) for _ in range(3)]
            with patch.object(api.schema, "_VALIDATION_MEMO_CAPACITY", 2):
                with patch.object(api.schema, "_unpack", wraps=api.schema._unpack) as unpack:
                    for frame in frames:
                        api.schema.validate_frame(frame, schema)
                    self.assertLessEqual(len(api.schema._VALIDATION_MEMO), 2)
                    before_recheck = unpack.call_count
                    api.schema.validate_frame(frames[0], schema)
                    self.assertEqual(unpack.call_count, before_recheck + len(schema.tensors))

    def test_concurrent_validation_and_late_cleanup_do_not_corrupt_replacements(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            self.assertTrue(
                hasattr(api.schema, "_VALIDATION_MEMO"),
                "successful validation memo is not implemented",
            )
            self.assertTrue(
                hasattr(api.schema, "_VALIDATION_MEMO_LOCK"),
                "validation memo synchronization is not implemented",
            )
            if not hasattr(api.schema, "_VALIDATION_MEMO") or not hasattr(
                api.schema, "_VALIDATION_MEMO_LOCK"
            ):
                return
            frame = _frame(api, schema)
            api.schema.validate_frame(frame, schema)
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: api.schema.validate_frame(frame, schema), range(32)))

                def validate_transient(_):
                    transient = _frame(api, schema)
                    api.schema.validate_frame(transient, schema)
                    return weakref.ref(transient)

                transient_refs = list(pool.map(validate_transient, range(32)))
            self.assertTrue(all(result is None for result in results))
            gc.collect()
            self.assertTrue(all(reference() is None for reference in transient_refs))

            key = (id(frame), id(schema))
            previous_entry = api.schema._VALIDATION_MEMO[key]
            old_reference = previous_entry[0]
            old_callback = old_reference.__callback__
            replacement_frame = _frame(api, schema)
            replacement_entry = (weakref.ref(replacement_frame), weakref.ref(schema))
            with api.schema._VALIDATION_MEMO_LOCK:
                api.schema._VALIDATION_MEMO[key] = replacement_entry
            self.assertIsNotNone(old_callback)
            old_callback(old_reference)
            with api.schema._VALIDATION_MEMO_LOCK:
                self.assertIs(api.schema._VALIDATION_MEMO[key], replacement_entry)
                api.schema._VALIDATION_MEMO[key] = previous_entry

    def test_mutable_subclasses_do_not_inherit_success_certificates(self):
        api = _api(self)
        with _synthetic_schema(api) as schema:
            base = _frame(api, schema)

            class MutableFrame(api.CombatFrame):
                pass

            subclass_frame = MutableFrame(base.tensors, base.schema_fingerprint)
            api.schema.validate_frame(subclass_frame, schema)
            object.__setattr__(subclass_frame, "tensors", MappingProxyType({}))
            with self.assertRaises(api.schema.ContractError):
                api.schema.validate_frame(subclass_frame, schema)

            class MutableBuffer(api.BufferArray):
                pass

            mutable_player = MutableBuffer(
                base.tensors["player"].dtype,
                base.tensors["player"].shape,
                base.tensors["player"].data,
            )
            buffer_subclass_frame = _frame_with(
                api, schema, base, player=mutable_player
            )
            api.schema.validate_frame(buffer_subclass_frame, schema)
            object.__setattr__(mutable_player, "data", struct.pack("=q", 101))
            with self.assertRaises(api.schema.ContractError):
                api.schema.validate_frame(buffer_subclass_frame, schema)

            class MutableSchema(api.schema.CombatSchema):
                pass

            schema_subclass = object.__new__(MutableSchema)
            for field in fields(api.schema.CombatSchema):
                object.__setattr__(schema_subclass, field.name, getattr(schema, field.name))
            api.schema.validate_frame(base, schema_subclass)
            object.__setattr__(schema_subclass, "tensors", MappingProxyType({}))
            with self.assertRaises(api.schema.ContractError):
                api.schema.validate_frame(base, schema_subclass)


if __name__ == "__main__":
    unittest.main()
