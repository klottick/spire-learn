"""Pure descriptor-to-feature contract checks; no optional runtime is imported."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from pathlib import Path

from tests.neural._support import (
    CandidateInput,
    authored_frame,
    schema_from_driver_factory,
    synthetic_schema,
)


class NeuralFeatureContractTests(unittest.TestCase):
    def test_actual_config_cases_flow_through_a_closing_driver_factory_without_runtime_imports(self):
        import builtins
        import sys
        from unittest.mock import patch

        optional_prefixes = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
        original_import = builtins.__import__

        def reject_optional(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in optional_prefixes:
                raise AssertionError(f"optional import attempted in pure config helper: {name}")
            return original_import(name, globals, locals, fromlist, level)

        root = Path(__file__).resolve().parents[2]
        counts = {"factory": 0, "close": 0}

        class Driver:
            schema = object()

            def close(self):
                counts["close"] += 1

        def factory(config, cases):
            counts["factory"] += 1
            self.assertIs(cases, config.cases)
            self.assertEqual(cases.path, config.manifest_path)
            return Driver()

        before = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        with patch.object(builtins, "__import__", new=reject_optional):
            from spire_learn.neural.config import load_neural_config

            config = load_neural_config(root / "configs/neural/run.toml")
            schema = schema_from_driver_factory(config, factory)

        after = {name for name in sys.modules if name.split(".", 1)[0] in optional_prefixes}
        self.assertEqual(after, before)
        self.assertIs(schema, Driver.schema)
        self.assertEqual(counts, {"factory": 1, "close": 1})
        self.assertEqual((config.mode, config.operation, config.split), ("run", "train", "train"))

    def test_learner_authored_candidate_and_power_frames_match_the_parsed_schema(self):
        from spire_learn.neural.schema import validate_frame

        candidates = (
            CandidateInput(2, "play_card", "combat_cards", 0, "enemies", 0),
            CandidateInput(9, "play_card", "combat_cards", 0, "enemies", 1),
        )
        with synthetic_schema() as schema:
            for owners in ((0, 1), (1, 0)):
                validate_frame(
                    authored_frame(schema, candidates=candidates, power_owners=owners),
                    schema,
                )

    def test_descriptor_fields_keep_semantic_types_order_and_vocabulary_identity(self):
        try:
            from spire_learn.neural.features import (
                FeatureContract,
                FeatureField,
                feature_contract,
            )
        except ImportError as error:
            self.fail(f"pure neural feature contract API is unavailable: {error}")

        with synthetic_schema() as schema:
            plan = feature_contract(schema)

            self.assertIsInstance(plan, FeatureContract)
            self.assertEqual(plan.features_id, "spire-learn-combat-features-v1")
            self.assertEqual(plan.schema_fingerprint, schema.schema_fingerprint)
            expected_groups = {
                key: spec for key, spec in schema.tensors.items() if spec.columns
            }
            self.assertEqual(tuple(plan.groups), tuple(expected_groups))

            for group, spec in expected_groups.items():
                fields = plan.groups[group]
                self.assertIsInstance(fields, tuple)
                self.assertEqual(len(fields), len(spec.columns))
                for actual, column in zip(fields, spec.columns, strict=True):
                    self.assertIsInstance(actual, FeatureField)
                    self.assertEqual(actual.field, column.field)
                    if column.reference is not None:
                        self.assertEqual(actual.kind, "reference")
                        self.assertIsNone(actual.vocabulary)
                        self.assertEqual(actual.reference, column.reference)
                    elif column.vocabulary is not None:
                        self.assertEqual(actual.kind, "categorical")
                        self.assertEqual(actual.vocabulary, column.vocabulary)
                        self.assertIsNone(actual.reference)
                        self.assertIn(actual.vocabulary, schema.vocabularies)
                    else:
                        self.assertEqual(actual.kind, "numeric")
                        self.assertIsNone(actual.vocabulary)
                        self.assertIsNone(actual.reference)

            # Visible semantic positions are numeric; reference columns remain
            # typed row gathers, never transport-index scalars.
            self.assertEqual(
                plan.groups["combat_cards"][1],
                FeatureField("zone_position", "numeric", None, None),
            )
            self.assertEqual(
                plan.groups["enemies"][1],
                FeatureField("logical_slot", "numeric", None, None),
            )
            self.assertEqual(
                plan.groups["known_draw"][1],
                FeatureField("offset", "numeric", None, None),
            )
            power_owner = plan.groups["powers"][3]
            self.assertEqual((power_owner.field, power_owner.kind, power_owner.reference),
                             ("owner_row", "reference", "owner_group"))
            action_fields = {field.field: field for field in plan.groups["actions"]}
            self.assertEqual(action_fields["source_row"].kind, "reference")
            self.assertEqual(action_fields["source_row"].reference, "source_group")
            self.assertEqual(action_fields["target_row"].kind, "reference")
            self.assertEqual(action_fields["target_row"].reference, "target_group")

            # Only descriptor-declared public value groups/columns enter the
            # feature plan. No action-slot or metadata identity is encoded.
            self.assertEqual(set(plan.groups), set(expected_groups))
            self.assertNotIn("action_slot", plan.groups)
            self.assertNotIn("seed", plan.groups)
            self.assertNotIn("routing", plan.groups)
            self.assertNotIn("active_visit", plan.groups)
            self.assertNotIn("decision_id", plan.groups)
            with self.assertRaises(TypeError):
                plan.groups["metadata"] = ()
            with self.assertRaises((FrozenInstanceError, AttributeError)):
                plan.features_id = "changed"


if __name__ == "__main__":
    unittest.main()
