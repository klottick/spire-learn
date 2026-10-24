import unittest

from spire_learn.contracts import ArrayValue, ContractError
from spire_learn.schema import candidates_for, fixture_contract, parse_descriptor
from tests.support import observation, tiny_descriptor


class DescriptorSchemaTests(unittest.TestCase):
    def test_action_kind_vocabulary_is_not_candidate_count(self):
        descriptor = tiny_descriptor(action_count=32)
        schema = parse_descriptor(descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(schema.action_count, 32)
        self.assertEqual(len(schema.vocabularies["action"]), 3)
        self.assertIsNone(next(t for t in schema.tensors if t.key == "action_mask").mask)
        self.assertEqual(next(t for t in schema.tensors if t.key == "action_kind").mask, "action_mask")

    def test_rejects_unknown_schema_and_malformed_descriptor(self):
        descriptor = tiny_descriptor()
        with self.assertRaises(ContractError):
            parse_descriptor(descriptor, allowed_schema_ids=frozenset())
        for mutate in (
            lambda d: d.pop("content_id"),
            lambda d: d["tensors"].append(d["tensors"][0]),
            lambda d: d["tensors"][1].update(shape=[9]),
            lambda d: d["tensors"][1].update(axes=["candidate", "field"]),
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
        self.assertEqual(found[0].features, (("hp", 10),))

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
        self.assertEqual(len(schema.tensors), 5)

    def test_fixture_contract_is_scoped_to_fixture_only(self):
        contract = fixture_contract()
        self.assertEqual(contract.schema_ids, frozenset({"spire-learn-fixture-v1"}))
        self.assertEqual(contract.content_ids, frozenset({"spire-learn-fixture-content-v1"}))
        self.assertEqual(contract.reward_contract_id, "fixture-protocol-v1")
