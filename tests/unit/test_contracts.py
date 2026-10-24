import dataclasses
import unittest

from spire_learn.contracts import (
    ArrayValue, CaseExecution, CaseResult, Candidate, ContractError, PublicDecision,
    Route, Transition, frozen_mapping,
)


class ArrayValueTests(unittest.TestCase):
    def test_validates_shape_count_and_finite_numeric_values(self):
        self.assertEqual(ArrayValue("int64", (2,), (1, 2)).values, (1, 2))
        for shape, values in [((2,), (1,)), ((-1,), ()), ((1,), (True,)), ((1,), (float("inf"),)), ((1,), (object(),))]:
            with self.subTest(shape=shape, values=values), self.assertRaises(ContractError):
                ArrayValue("float32", shape, values)

    def test_frozen_mapping_owns_data_and_cannot_be_mutated(self):
        source = {"hp": ArrayValue("int64", (), (10,))}
        result = frozen_mapping(source)
        source["new"] = ArrayValue("int64", (), (0,))
        self.assertEqual(set(result), {"hp"})
        with self.assertRaises(TypeError):
            result["hp"] = ArrayValue("int64", (), (0,))

    def test_records_are_frozen(self):
        record = ArrayValue("int8", (), (1,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.dtype = "int64"

    def test_public_decision_and_transition_snapshot_input_observation_mappings(self):
        original = {"hp": ArrayValue("int64", (), (10,))}
        decision = PublicDecision(original, (), (), "a" * 64, "mapping")
        transition = Transition(Route("x", 0, "e", "d", True), original, 0, 0.0, False, False, original, original, None)
        original["hp"] = ArrayValue("int64", (), (1,))
        original["new"] = ArrayValue("int64", (), (0,))
        self.assertEqual(decision.observation["hp"].values, (10,))
        self.assertEqual(transition.source["hp"].values, (10,))
        self.assertEqual(transition.successor["hp"].values, (10,))
        self.assertEqual(transition.replacement["hp"].values, (10,))
        self.assertNotIn("new", decision.observation)

    def test_candidate_and_case_result_own_collection_and_metadata_inputs(self):
        features = [["damage", 5]]
        candidate = Candidate(0, "play_card", None, None, features)
        resources = {"hp": 10}
        descriptor = {"nested": {"labels": ["visible"]}}
        build_info = {"nested": ["build"]}
        result = CaseResult(CaseExecution("x", "c", "selection", "p", "s", 1, 2), "terminal", True, "victory", 1, 1.0, "done", resources, observed_descriptor=descriptor, native_build_info=build_info)
        features[0][1] = 99
        resources["hp"] = 0
        descriptor["nested"]["labels"].append("hidden")
        build_info["nested"].append("hidden")
        self.assertEqual(candidate.features, (("damage", 5),))
        self.assertEqual(result.final_resources["hp"], 10)
        self.assertEqual(result.observed_descriptor["nested"]["labels"], ("visible",))
        self.assertEqual(result.native_build_info["nested"], ("build",))

    def test_schema_vocabulary_mapping_is_owned(self):
        from spire_learn.schema import ObservationSchema

        vocab = {"action": ["padding", "end_turn"]}
        schema = ObservationSchema("s", "c", "f", 2, (), vocab)
        vocab["action"].append("mutated")
        self.assertEqual(schema.vocabularies["action"], ("padding", "end_turn"))
