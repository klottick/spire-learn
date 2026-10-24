"""Pure adapter contract tests; no simulator or optional imports."""

import sys
import math
import struct
from types import ModuleType
import unittest
from unittest.mock import patch

from spire_learn.contracts import ArrayValue, ContractError, SetupBlocked, frozen_mapping
from spire_learn.schema import parse_descriptor
from tests.support import FakeArray, FakeNativeErrors, observation, tiny_descriptor


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        from spire_learn.envs.native import NATIVE_CONTRACT
        self.native = __import__("spire_learn.envs.native", fromlist=["native"])
        self.NATIVE_CONTRACT = NATIVE_CONTRACT
        self.schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))

    def test_import_is_lazy(self):
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("spire_lab", sys.modules)

    def test_snapshot_copies_public_numeric_array(self):
        obs = observation()
        raw = {k: FakeArray(v.dtype, v.shape, v.values) for k, v in obs.items()}
        result = self.native.native_snapshot(raw, self.schema)
        self.assertEqual(result["action_mask"], obs["action_mask"])
        self.assertIsNot(result["action_mask"], obs["action_mask"])

    def test_snapshot_rejects_object_shape_and_endian_mismatch(self):
        obs = observation()
        def raw_with(key, value):
            return {k: (value if k == key else FakeArray(v.dtype, v.shape, v.values)) for k, v in obs.items()}
        for value in (FakeArray("object", (3,), (1, 1, 0)), FakeArray("int8", (2,), (1, 1)), FakeArray("int8", (3,), (1, 1, 0), isnative=False)):
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.native.native_snapshot(raw_with("action_mask", value), self.schema)

    def test_arraylike_shape_must_be_nested_exactly_as_declared(self):
        obs = observation()
        base = {k: FakeArray(v.dtype, v.shape, v.values) for k, v in obs.items()}
        flat = FakeArray("int32", (3, 4), (1,) * 12)
        flat.tolist = lambda: [1] * 12
        bad_scalar = FakeArray("int64", (), (10,))
        bad_scalar.tolist = lambda: [10]
        bad_shape = FakeArray("int32", (1, 3), (1, 1, 0))
        for key, value in (("action_features", flat), ("action_features", bad_shape), ("player", bad_scalar)):
            with self.subTest(key=key, shape=value.shape):
                raw = dict(base)
                raw[key] = value
                with self.assertRaises(ContractError):
                    self.native.native_snapshot(raw, self.schema)
        ragged = FakeArray("int32", (3, 4), (1, 1, 0, 0) * 3)
        ragged.tolist = lambda: [[1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0]]
        raw = dict(base)
        raw["action_features"] = ragged
        with self.assertRaises(ContractError):
            self.native.native_snapshot(raw, self.schema)

    def test_snapshot_validates_values_against_declared_dtype(self):
        obs = observation()
        base = {k: FakeArray(v.dtype, v.shape, v.values) for k, v in obs.items()}
        cases = (
            ("action_mask", FakeArray("int8", (3,), (128, 0, 0))),
            ("action_mask", FakeArray("int8", (3,), (1.5, 0, 0))),
            ("player", FakeArray("int64", (), (1.5,))),
            ("player", FakeArray("int64", (), (2**63,))),
        )
        for key, value in cases:
            with self.subTest(key=key, values=value._values):
                raw = dict(base)
                raw[key] = value
                with self.assertRaises(ContractError):
                    self.native.native_snapshot(raw, self.schema)
        float_descriptor = tiny_descriptor()
        next(t for t in float_descriptor["tensors"] if t["key"] == "player")["dtype"] = "float32"
        float_schema = parse_descriptor(float_descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        for value in (float("inf"), 10**100):
            with self.subTest(float_value=value):
                raw = dict(base)
                raw["player"] = FakeArray("float32", (), (value,))
                with self.assertRaises(ContractError):
                    self.native.native_snapshot(raw, float_schema)

    def test_immutable_array_values_are_revalidated_against_descriptor_dtype(self):
        obs = observation()
        raw = {k: ArrayValue(v.dtype, v.shape, v.values) for k, v in obs.items()}
        raw["action_mask"] = ArrayValue("int8", (3,), (128, 0, 0))
        with self.assertRaises(ContractError):
            self.native.native_snapshot(raw, self.schema)

    def test_float_arrays_require_exact_representable_float_primitives(self):
        for dtype, format_code, values in (
            ("float16", "e", (struct.unpack("=e", struct.pack("=e", 0.1))[0], 2**-24)),
            ("float32", "f", (struct.unpack("=f", struct.pack("=f", 0.1))[0], 2**-149)),
            ("float64", "d", (0.1, -0.0)),
        ):
            descriptor = tiny_descriptor()
            next(t for t in descriptor["tensors"] if t["key"] == "player")["dtype"] = dtype
            schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
            for value in values:
                raw = {k: FakeArray(v.dtype, v.shape, v.values) for k, v in observation().items()}
                raw["player"] = FakeArray(dtype, (), (value,))
                snapshot = self.native.native_snapshot(raw, schema)
                self.assertEqual(snapshot["player"].values, (value,))
                if value == 0.0 and math.copysign(1.0, value) < 0:
                    self.assertLess(math.copysign(1.0, snapshot["player"].values[0]), 0)

        descriptor = tiny_descriptor()
        next(t for t in descriptor["tensors"] if t["key"] == "player")["dtype"] = "float32"
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        for value in (0.1, 1, 1e-50, float("inf"), float("nan")):
            with self.subTest(value=value):
                raw = {k: FakeArray(v.dtype, v.shape, v.values) for k, v in observation().items()}
                raw["player"] = FakeArray("float32", (), (value,))
                with self.assertRaises(ContractError):
                    self.native.native_snapshot(raw, schema)
        raw = {k: ArrayValue(v.dtype, v.shape, v.values) for k, v in observation().items()}
        raw["player"] = ArrayValue("float32", (), (0.1,))
        with self.assertRaises(ContractError):
            self.native.native_snapshot(raw, schema)

    def test_validate_actions_rejects_bool_float_and_negative(self):
        from spire_learn.envs.native import validate_actions
        for actions in ([True], [1.0], [-1]):
            with self.subTest(actions=actions), self.assertRaises(ContractError):
                validate_actions(actions)
        self.assertEqual(validate_actions([0, 2]), (0, 2))

    def test_native_failure_uses_injected_types_only(self):
        from spire_learn.envs.native import NativeErrorTypes, native_failure
        types = NativeErrorTypes(FakeNativeErrors.InputError, FakeNativeErrors.StateError, FakeNativeErrors.SetupError, FakeNativeErrors.FaultError)
        self.assertEqual(native_failure(FakeNativeErrors.FaultError(), error_types=types).kind, "simulator_fault")
        self.assertIsNone(native_failure(RuntimeError("fault"), error_types=types))

    def test_native_failure_resolves_only_already_loaded_documented_binding_types(self):
        from spire_learn.envs.native import native_failure
        module = ModuleType("spire_lab._native")
        names = ("CombatInputError", "CombatStateError", "CombatSetupError", "CombatFaultError")
        classes = tuple(type(name, (Exception,), {}) for name in names)
        for name, error_type in zip(names, classes, strict=True):
            setattr(module, name, error_type)
        with patch.dict(sys.modules, {"spire_lab._native": module}):
            self.assertEqual(native_failure(classes[3]("fault")).kind, "simulator_fault")
            self.assertEqual(native_failure(classes[0]("bad input")).kind, "invalid_action")
            self.assertIsNone(native_failure(RuntimeError("fault")))
        self.assertIsNone(native_failure(classes[3]("not loaded")))

    def test_build_info_reads_only_loaded_binding_and_owns_nested_metadata(self):
        module = ModuleType("spire_lab._native")
        metadata = {"revision": "r1", "nested": {"labels": ["a"]}}
        module.build_info = lambda: metadata
        with patch.dict(sys.modules, {"spire_lab._native": module}):
            result = self.native.public_build_info()
        metadata["nested"]["labels"].append("mutated")
        self.assertEqual(result["nested"]["labels"], ("a",))
        with self.assertRaises(TypeError):
            result["nested"]["new"] = "x"

    def test_info_allowlist_and_fixture_outcome_decode(self):
        value = self.native.normalize_public_info({"outcome": 1, "steps": 4, "seed": 99, "private": "x"}, self.schema)
        self.assertEqual(value, {"outcome": "victory", "steps": 4})

    def test_unsupported_schema_is_setup_error(self):
        descriptor = tiny_descriptor()
        descriptor["schema_id"] = "unsupported"
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(self.NATIVE_CONTRACT.schema_ids, frozenset({"spire-lab-combat-observation-v1"}))


if __name__ == "__main__":
    unittest.main()
