"""Pure adapter contract tests; no simulator or optional imports."""

import sys
import unittest

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
