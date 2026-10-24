import unittest
from dataclasses import replace

from spire_learn.contracts import ArrayValue, ContractError
from spire_learn.schema import candidates_for, parse_descriptor
from tests.support import fixture_contract, observation, tiny_descriptor


class DescriptorSchemaTests(unittest.TestCase):
    def test_action_kind_vocabulary_is_not_candidate_count(self):
        descriptor = tiny_descriptor(action_count=32)
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(schema.action_count, 32)
        self.assertEqual(len(schema.vocabularies["action"]), 3)
        self.assertIsNone(next(t for t in schema.tensors if t.key == "action_mask").mask)
        self.assertEqual(next(t for t in schema.tensors if t.key == "action_kind").mask, "action_mask")

    def test_native_style_int32_action_metadata_is_accepted(self):
        descriptor = tiny_descriptor(action_count=32)
        tensor_by_key = {tensor["key"]: tensor for tensor in descriptor["tensors"]}
        tensor_by_key["action_mask"]["fields"] = ["legal"]
        tensor_by_key["action_kind"].update(dtype="int32", fields=["kind"])
        tensor_by_key["action_source"].update(dtype="int32", mask="action_mask")
        tensor_by_key["action_target"].update(dtype="int32", mask="action_mask")
        tensor_by_key["action_features"].update(
            dtype="int32", shape=[32, 4],
            fields=["cost", "printed_damage", "printed_block", "printed_vulnerable"],
            mask="action_mask",
        )
        raw = observation(mask=(1, 1) + (0,) * 30, kinds=(1, 2) + (0,) * 30)
        native_observation = dict(raw)
        for key in ("action_kind", "action_source", "action_target"):
            value = native_observation[key]
            native_observation[key] = ArrayValue("int32", value.shape, value.values)
        features = tuple(value for index in range(32) for value in ((1, 7, 0, 0) if index == 1 else (0, 0, 0, 0)))
        native_observation["action_features"] = ArrayValue("int32", (32, 4), features)

        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        candidates = candidates_for(native_observation, schema)
        self.assertEqual(schema.action_count, 32)
        self.assertEqual(candidates[1].kind, "play_card")
        self.assertEqual((candidates[1].source, candidates[1].target), (0, 0))
        self.assertEqual(candidates[1].features, (("cost", 1), ("printed_damage", 7), ("printed_block", 0), ("printed_vulnerable", 0)))

    def test_declared_int64_action_metadata_is_accepted_and_observation_dtype_must_match(self):
        descriptor = tiny_descriptor(action_count=32)
        tensors = {tensor["key"]: tensor for tensor in descriptor["tensors"]}
        tensors["action_mask"]["fields"] = ["legal"]
        tensors["action_kind"].update(dtype="int64", fields=["kind"])
        tensors["action_source"].update(dtype="int64", fields=["hand_row"], mask="action_mask")
        tensors["action_target"].update(dtype="int64", fields=["enemy_row"], mask="action_mask")
        tensors["action_features"].update(
            dtype="int64", shape=[32, 4],
            fields=["cost", "printed_damage", "printed_block", "printed_vulnerable"],
            mask="action_mask",
        )
        raw = observation(mask=(1, 1) + (0,) * 30, kinds=(1, 2) + (0,) * 30)
        matching = dict(raw)
        for key in ("action_kind", "action_source", "action_target"):
            value = matching[key]
            matching[key] = ArrayValue("int64", value.shape, value.values)
        features = tuple(value for index in range(32) for value in ((1, 7, 0, 0) if index == 1 else (0, 0, 0, 0)))
        matching["action_features"] = ArrayValue("int64", (32, 4), features)
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(candidates_for(matching, schema)[1].features[1], ("printed_damage", 7))

        mismatched = dict(matching)
        mismatched["action_kind"] = ArrayValue("int32", (32,), matching["action_kind"].values)
        with self.assertRaises(ContractError):
            candidates_for(mismatched, schema)

    def test_rejects_unknown_schema_and_malformed_descriptor(self):
        descriptor = tiny_descriptor()
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset())
        for mutate in (
            lambda d: d.pop("content_id"),
            lambda d: d["tensors"].append(d["tensors"][0]),
            lambda d: d["tensors"][1].update(shape=[9]),
            lambda d: d["tensors"][1].update(axes=["candidate", "field"]),
            lambda d: d["tensors"][1].update(fields=[]),
            lambda d: d["tensors"][2].update(fields=["source"]),
            lambda d: d["tensors"][1].update(fields=["a", "a"]),
            lambda d: d["tensors"][1].update(category="missing"),
        ):
            bad = tiny_descriptor()
            mutate(bad)
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                parse_descriptor(bad, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))

    def test_extracts_only_legal_candidates_and_named_features(self):
        schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        found = candidates_for(observation(), schema)
        self.assertEqual([(c.index, c.kind) for c in found], [(0, "end_turn"), (1, "play_card")])
        self.assertEqual(found[0].features, (("cost", 0), ("printed_damage", 0), ("printed_block", 0), ("printed_vulnerable", 0)))
        self.assertEqual(found[1].features, (("cost", 1), ("printed_damage", 7), ("printed_block", 0), ("printed_vulnerable", 0)))
        self.assertIsNone(found[0].source)
        self.assertIsNone(found[0].target)
        self.assertEqual((found[1].source, found[1].target), (0, 0))

    def test_candidate_mask_width_is_action_count_and_kind_codes_are_vocab_checked(self):
        schema = parse_descriptor(tiny_descriptor(action_count=32), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        obs = observation(mask=(1,) * 32, kinds=(1,) * 31 + (99,))
        with self.assertRaises(ContractError):
            candidates_for(obs, schema)
        short = {"action_mask": ArrayValue("int8", (2,), (1, 0))}
        with self.assertRaises(ContractError):
            candidates_for(short, schema)

    def test_ignores_category_codes_in_masked_candidate_slots(self):
        schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        found = candidates_for(observation(kinds=(1, 2, 99)), schema)
        self.assertEqual(len(found), 2)

    def test_field_names_may_repeat_between_tensors_but_not_within_one(self):
        descriptor = tiny_descriptor()
        descriptor["tensors"].append({"key": "other_hp", "dtype": "int64", "shape": [], "fields": ["hp"], "axes": [], "category": None, "mask": None})
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(len(schema.tensors), 7)

    def test_fixture_contract_is_scoped_to_fixture_only(self):
        contract = fixture_contract()
        self.assertEqual(contract.schema_ids, frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(contract.content_ids, frozenset({"spire-learn-fixture-content-v1"}))
        self.assertEqual(contract.reward_contract_id, "fixture-protocol-v1")

    def test_rejects_malformed_axes_container_even_for_scalar_tensor(self):
        descriptor = tiny_descriptor()
        descriptor["tensors"][-1]["axes"] = None
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))

    def test_qualified_mask_reference_names_field_on_consuming_tensor(self):
        descriptor = tiny_descriptor()
        descriptor["tensors"].append({"key": "enemy_damage_mask", "dtype": "int8", "shape": [3, 1], "fields": ["intent_damage"], "axes": ["candidate", "field"], "category": None, "mask": None})
        descriptor["tensors"].append({"key": "enemy", "dtype": "int64", "shape": [3, 1], "fields": ["intent_damage"], "axes": ["candidate", "field"], "category": None, "mask": "enemy_damage_mask:intent_damage"})
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(next(t for t in schema.tensors if t.key == "enemy").mask, "enemy_damage_mask:intent_damage")
        for reference in ("missing:intent_damage", "enemy_damage_mask:", ":intent_damage", "enemy_damage_mask:intent_damage:extra", "enemy_damage_mask:unknown"):
            bad = tiny_descriptor()
            bad["tensors"].append({"key": "enemy_damage_mask", "dtype": "int8", "shape": [3, 1], "fields": ["intent_damage"], "axes": ["candidate", "field"], "category": None, "mask": None})
            bad["tensors"].append({"key": "enemy", "dtype": "int64", "shape": [3, 1], "fields": ["intent_damage"], "axes": ["candidate", "field"], "category": None, "mask": reference})
            with self.subTest(reference=reference), self.assertRaises(ContractError):
                parse_descriptor(bad, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))

    def test_rejects_missing_required_action_descriptors_and_unsupported_consumed_axes(self):
        for key in ("action_mask", "action_kind", "action_source", "action_target", "action_features"):
            descriptor = tiny_descriptor()
            descriptor["tensors"] = [tensor for tensor in descriptor["tensors"] if tensor["key"] != key]
            with self.subTest(key=key), self.assertRaises(ContractError):
                parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        descriptor = tiny_descriptor()
        features = next(t for t in descriptor["tensors"] if t["key"] == "action_features")
        features["shape"] = [3, 2, 3]
        features["fields"] = ["damage", "block", "cost"]
        features["axes"] = ["candidate", "plane", "field"]
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        valid_schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        unsupported_feature_spec = replace(
            next(tensor for tensor in valid_schema.tensors if tensor.key == "action_features"),
            shape=(3, 2, 3), axes=("candidate", "plane", "field"),
        )
        unsupported_tensors = tuple(
            unsupported_feature_spec if tensor.key == "action_features" else tensor
            for tensor in valid_schema.tensors
        )
        unsupported_schema = replace(valid_schema, tensors=unsupported_tensors)
        obs = dict(observation())
        obs["action_features"] = ArrayValue("int64", (3, 2, 3), tuple(range(18)))
        with self.assertRaises(ContractError):
            candidates_for(obs, unsupported_schema)

    def test_uses_named_public_action_tensors_and_ignores_other_categories(self):
        descriptor = tiny_descriptor()
        descriptor["tensors"].append({"key": "other_category", "dtype": "int64", "shape": [3], "fields": [], "axes": ["candidate"], "category": "outcome", "mask": None})
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        obs = dict(observation())
        obs["other_category"] = ArrayValue("int64", (3,), (1, 2, 1))
        result = candidates_for(obs, schema)
        self.assertEqual([(c.index, c.kind) for c in result], [(0, "end_turn"), (1, "play_card")])
        self.assertEqual(result[1].features, (("cost", 1), ("printed_damage", 7), ("printed_block", 0), ("printed_vulnerable", 0)))

    def test_rejects_legal_padding_kind_and_consumed_dtype_mismatch(self):
        schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        with self.assertRaises(ContractError):
            candidates_for(observation(kinds=(0, 2, 0)), schema)
        obs = dict(observation())
        obs["action_features"] = ArrayValue("float32", (3, 3), (0.0,) * 9)
        with self.assertRaises(ContractError):
            candidates_for(obs, schema)

    def test_rejects_non_numeric_descriptor_dtype_and_invalid_negative_rows(self):
        descriptor = tiny_descriptor()
        descriptor["tensors"][-1]["dtype"] = "object"
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        schema = parse_descriptor(tiny_descriptor(), allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        obs = dict(observation())
        obs["action_source"] = ArrayValue("int32", (3,), (-2, 0, -1))
        with self.assertRaises(ContractError):
            candidates_for(obs, schema)
