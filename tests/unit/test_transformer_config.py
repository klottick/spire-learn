"""The entity transformer (S2) without torch: config, identities, tables, registry, plan and parameter counts."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import config as config_module
from spire_learn.neural import model_families as families
from spire_learn.neural import transformer
from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, NUMBER_ENCODINGS, STANDARD_LEARNER_IDENTITY_IDS, TRANSFORMER_FEATURES_ID,
    TRANSFORMER_LEARNER_IDENTITY_IDS, TRANSFORMER_MODEL_ID, TRANSFORMER_PARAMETER_CAP, TRANSFORMER_SIZES,
    TransformerConfig, distill_config_hash, load_distill_config, load_neural_config, model_config_hash,
    transformer_size,
)
from spire_learn.neural.runner import _config_projection
from spire_learn.neural.schema import parse_combat_descriptor
from spire_learn.neural.gym_contracts import contract_for_revision
from tests.unit.test_distill_config import (
    COMPACT, GYM, HYPER, MAPPING, STUDENT, _identity_toml, _write, distill_toml, eval_toml,
)

ROOT = Path(__file__).resolve().parents[2]
GYM_1_2_0 = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"
SMALL = transformer_size("small", "bucket")
TRANSFORMER_TABLE = ('[transformer]\nlayers = 2\nwidth = 128\nheads = 4\nnumber_encoding = "bucket"\n'
                     "value_heads = 3\n\n")
FF_TABLE = "[model]\nembedding_width = 16\nhidden_width = 64\nmax_parameters = 10000000\nvalue_heads = 3\n\n"
# Every non-action row group of the compact descriptor, and its single-row groups.
ROW_GROUPS = ("permanent_cards", "combat_cards", "enemies", "powers", "relics", "potions", "selection", "orbs",
              "known_draw", "stance_choices", "combat_card_offers")
SINGLE_GROUPS = ("combat_context", "player", "selection_counts", "combat_state")
RESOLVED = {"powers.owner_row", "selection.source_row", "known_draw.combat_card_row", "combat_cards.permanent_row",
            "combat_cards.stasis_enemy_row"}
# The exact counts of transformer_parameter_count on the compact descriptor, recorded at this commit.
COUNTS = {
    ("small", "bucket"): 1_262_372, ("small", "scaled"): 1_030_532, ("small", "sinusoidal"): 1_173_892,
    ("medium", "bucket"): 5_448_612, ("medium", "scaled"): 5_082_372, ("medium", "sinusoidal"): 5_369_092,
    ("large", "bucket"): 15_015_460, ("large", "scaled"): 14_514_820, ("large", "sinusoidal"): 14_944_900,
}


def compact_schema():
    descriptor = json.loads((ROOT / "tests/fixtures/neural/standard-combat-compact-4d19806.json").read_text())
    return parse_combat_descriptor(descriptor, revision=GYM_1_2_0)


def transformer_distill_toml(*, model_table=TRANSFORMER_TABLE, learner=TRANSFORMER_LEARNER_IDENTITY_IDS,
                             model=SMALL, hashes=None) -> str:
    """The distillation training TOML of test_distill_config with a [transformer] student."""
    text = distill_toml(hyper=HYPER)
    head, _, _ = text.partition("[identity]")
    head = head.replace(FF_TABLE, model_table)
    hashes = hashes or {"model": model_config_hash(model), "distill": distill_config_hash(HYPER)}
    return head + _identity_toml(hashes=hashes, learner=learner)


class TransformerConfigTests(unittest.TestCase):
    def test_config_sizes_and_validation(self):
        self.assertEqual(TRANSFORMER_SIZES, {"small": (2, 128, 4), "medium": (4, 256, 8), "large": (6, 384, 8)})
        self.assertEqual(NUMBER_ENCODINGS, ("bucket", "scaled", "sinusoidal"))
        self.assertEqual(TRANSFORMER_PARAMETER_CAP, 16_000_000)
        self.assertEqual((TRANSFORMER_MODEL_ID, TRANSFORMER_FEATURES_ID),
                         ("spire-learn-combat-entity-transformer-v1", "spire-learn-combat-features-entity-v1"))
        self.assertEqual(TRANSFORMER_LEARNER_IDENTITY_IDS[2], "spire-learn-combat-dispatch-v1")
        large = transformer_size("large", "sinusoidal")
        self.assertEqual((large.layers, large.width, large.heads, large.number_encoding, large.value_heads),
                         (6, 384, 8, "sinusoidal", 3))
        # value_heads = 3 is part of the hashed payload; the hash differs from every FF student's.
        self.assertEqual(asdict(SMALL), {"layers": 2, "width": 128, "heads": 4, "number_encoding": "bucket",
                                         "value_heads": 3})
        self.assertNotEqual(model_config_hash(SMALL), model_config_hash(transformer_size("small", "scaled")))
        self.assertNotEqual(model_config_hash(SMALL), model_config_hash(STUDENT))
        for bad in (dict(layers=0), dict(width=130), dict(heads=0), dict(number_encoding="raw"), dict(value_heads=1),
                    dict(layers=2.0), dict(width=True)):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                replace(SMALL, **bad)
        with self.assertRaises(ValueError):
            transformer_size("huge", "bucket")

    def test_transformer_table_in_a_distillation_training_config(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = load_distill_config(_write(folder, "s2.toml", transformer_distill_toml()))
            self.assertEqual(config.model_table, "transformer")
            self.assertEqual(config.model, SMALL)
            self.assertEqual((config.identity.model_id, config.identity.features_id, config.identity.action_id),
                             TRANSFORMER_LEARNER_IDENTITY_IDS)
            self.assertEqual(dict(config.identity.config_hashes),
                             {"model": model_config_hash(SMALL), "distill": distill_config_hash(HYPER)})
            # The FF student is unchanged: [model] still loads with model_table "model".
            ff = load_distill_config(_write(folder, "s1.toml", distill_toml()))
            self.assertEqual((ff.model_table, ff.model), ("model", STUDENT))
            refusals = {
                "an FF identity on [transformer]": transformer_distill_toml(learner=STANDARD_LEARNER_IDENTITY_IDS),
                "a transformer identity on [model]": transformer_distill_toml(
                    model_table=FF_TABLE, hashes={"model": model_config_hash(STUDENT),
                                                  "distill": distill_config_hash(HYPER)}),
                "both tables": transformer_distill_toml(model_table=FF_TABLE + TRANSFORMER_TABLE),
                "an extra key": transformer_distill_toml(model_table=TRANSFORMER_TABLE.replace(
                    "value_heads = 3\n", "value_heads = 3\nmax_parameters = 16000000\n")),
                "a missing key": transformer_distill_toml(model_table=TRANSFORMER_TABLE.replace(
                    'number_encoding = "bucket"\n', "")),
                "one value head": transformer_distill_toml(model_table=TRANSFORMER_TABLE.replace(
                    "value_heads = 3", "value_heads = 1")),
                "an unknown encoding": transformer_distill_toml(model_table=TRANSFORMER_TABLE.replace(
                    '"bucket"', '"raw"')),
                "a stale model hash": transformer_distill_toml(model=transformer_size("small", "scaled")),
            }
            for name, text in refusals.items():
                with self.subTest(refusal=name), self.assertRaises(ValueError):
                    load_distill_config(_write(folder, "bad.toml", text))

    def test_transformer_table_in_an_evaluation_config(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            hashes = {"model": model_config_hash(SMALL)}
            text = eval_toml(folder, model_table=TRANSFORMER_TABLE, hashes=hashes,
                             learner=TRANSFORMER_LEARNER_IDENTITY_IDS)
            config = load_neural_config(_write(folder, "eval.toml", text))
            self.assertEqual((config.model_table, config.model, config.ppo), ("transformer", SMALL, None))
            self.assertEqual(config.accepted_identities[0].model_id, TRANSFORMER_MODEL_ID)
            projection = _config_projection(config)
            self.assertEqual(projection["transformer"], asdict(SMALL))
            self.assertNotIn("model", projection)
            refusals = {
                "a transformer identity on [model]": eval_toml(folder, learner=TRANSFORMER_LEARNER_IDENTITY_IDS),
                "an FF identity on [transformer]": eval_toml(folder, model_table=TRANSFORMER_TABLE, hashes=hashes),
            }
            for name, text in refusals.items():
                with self.subTest(refusal=name), self.assertRaisesRegex(ValueError, "does not configure learner"):
                    load_neural_config(_write(folder, "bad.toml", text))
        # The transformer has no PPO arm (delta 16): its identity takes the distillation objective only.
        raw = {"model_id": TRANSFORMER_MODEL_ID, "features_id": TRANSFORMER_FEATURES_ID,
               "action_id": TRANSFORMER_LEARNER_IDENTITY_IDS[2], "objective_id": "spire-learn-combat-winloss-v1",
               "accepted_raw_mode_mappings": {"standard-standalone": MAPPING},
               "config_hashes": {"model": "a" * 64, "ppo": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "objective only"):
            config_module._identity(raw, model_hash="a" * 64, ppo_hash="b" * 64, contract=contract_for_revision(GYM),
                                    standard_view_fingerprint=COMPACT)


class TransformerFamilyTests(unittest.TestCase):
    def test_family_registered_with_its_cap(self):
        family = families.model_family(TRANSFORMER_MODEL_ID)
        self.assertIs(family, families.TRANSFORMER_FAMILY)
        self.assertEqual((family.family_id, family.model_ids, family.config_type, family.parameter_cap),
                         ("transformer", (TRANSFORMER_MODEL_ID,), TransformerConfig, 16_000_000))
        self.assertTrue(family.check_built)
        # The FF cap stays 10,000,000.
        self.assertEqual(families.model_family(STANDARD_LEARNER_IDENTITY_IDS[0]).parameter_cap, 10_000_000)
        with self.assertRaisesRegex(TypeError, "TransformerConfig"):
            families.check_family_config(TRANSFORMER_MODEL_ID, STUDENT)
        with self.assertRaisesRegex(TypeError, "ModelConfig"):
            families.check_family_config(STANDARD_LEARNER_IDENTITY_IDS[0], SMALL)
        # The family builds with transformer.build_transformer_policy, the caller's arguments unchanged.
        options = dict(seed=3, device="cpu", objective_id=DISTILL_OBJECTIVE_ID, model_id=TRANSFORMER_MODEL_ID,
                       features_id=TRANSFORMER_FEATURES_ID)
        with patch.object(transformer, "build_transformer_policy", return_value="policy") as build:
            self.assertEqual(family.build("schema", SMALL, **options), "policy")
        build.assert_called_once_with("schema", SMALL, **options)
        from spire_learn.neural import features

        with patch.object(features, "encode_frames", return_value="batch") as encode:
            self.assertEqual(family.encode(["frame"], "schema", device="cpu", trim_padding=True), "batch")
        encode.assert_called_once_with(["frame"], "schema", device="cpu", trim_padding=True)

    def test_distillation_trainer_builds_through_the_family(self):
        from spire_learn.neural import distill_train, model

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            s2 = load_distill_config(_write(folder, "s2.toml", transformer_distill_toml()))
            s1 = load_distill_config(_write(folder, "s1.toml", distill_toml()))
        seed = distill_train.domain_seed(s2.seed, "policy")
        with patch.object(transformer, "build_transformer_policy", return_value="s2") as build:
            # build_family_policy checks a non-FF policy; a stand-in skips that check here.
            with patch.object(families, "check_policy", side_effect=lambda policy, **_: policy):
                self.assertEqual(distill_train.build_student(s2, "schema", device="cpu"), "s2")
        build.assert_called_once_with("schema", SMALL, seed=seed, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
                                      model_id=TRANSFORMER_MODEL_ID, features_id=TRANSFORMER_FEATURES_ID)
        # The FF student's build is the historical build_policy call, argument for argument.
        with patch.object(model, "build_policy", return_value="s1") as ff:
            self.assertEqual(distill_train.build_student(s1, "schema", device="cpu"), "s1")
        ff.assert_called_once_with("schema", STUDENT, seed=distill_train.domain_seed(s1.seed, "policy"), device="cpu",
                                   objective_id=DISTILL_OBJECTIVE_ID, model_id=STANDARD_LEARNER_IDENTITY_IDS[0],
                                   features_id=STANDARD_LEARNER_IDENTITY_IDS[1])


class TransformerPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = compact_schema()
        cls.plan = transformer.transformer_plan(cls.schema)

    def test_plan_tokenises_every_row_group_and_resolves_references(self):
        plan, schema = self.plan, self.schema
        # Enumerated from the descriptor: every row group but actions is a token group, every
        # single-row group feeds the context token, and every value group is in the plan.
        descriptor_rows = tuple(name for name, spec in schema.tensors.items()
                                if spec.columns and spec.occupied_key is not None and name != "actions")
        descriptor_singles = tuple(name for name, spec in schema.tensors.items()
                                   if spec.columns and spec.occupied_key is None)
        self.assertEqual(plan.token_groups, descriptor_rows)
        self.assertEqual(plan.token_groups, ROW_GROUPS)
        self.assertEqual(plan.context_groups, descriptor_singles)
        self.assertEqual(plan.context_groups, SINGLE_GROUPS)
        self.assertEqual(set(plan.groups), {*ROW_GROUPS, *SINGLE_GROUPS, "actions"})
        self.assertEqual({f"{ref.group}.{ref.field}" for ref in plan.resolved_references}, RESOLVED)
        self.assertEqual([ref.field for ref in plan.action_references], ["source_row", "target_row",
                                                                           "stance_choice_ref"])
        # Raw row indices are never features: no reference column is numeric or categorical.
        for name, group in plan.groups.items():
            spec = schema.tensors[name]
            references = {index for index, column in enumerate(spec.columns) if column.reference is not None}
            self.assertEqual({ref.field_index for ref in group.references}, references, name)
            self.assertFalse(references & set(group.numeric), name)
            self.assertFalse(references & {index for index, _ in group.categorical}, name)
            self.assertEqual(len(group.numeric) + len(group.categorical) + len(group.references), len(spec.columns))
        # Every target is a token or the context; player/combat_state resolve to the context token,
        # and groups without tokens (actions, map_nodes, offers) are never targets.
        targets = {target for ref in (*plan.resolved_references, *plan.action_references) for _, target in ref.choices}
        self.assertLessEqual(targets, {*ROW_GROUPS, *SINGLE_GROUPS})
        self.assertIn("player", targets)
        self.assertFalse(targets & {"actions", "map_nodes", "offers"})
        self.assertEqual(plan.coverage_width, sum(len(spec.columns) for spec in schema.tensors.values()))

    def test_parameter_counts_within_the_cap(self):
        for (size, encoding), expected in COUNTS.items():
            with self.subTest(size=size, encoding=encoding):
                count = transformer.transformer_parameter_count(self.plan, transformer_size(size, encoding))
                self.assertEqual(count, expected)
                self.assertLessEqual(count, TRANSFORMER_PARAMETER_CAP)
        for encoding in NUMBER_ENCODINGS:
            small, medium, large = (COUNTS[(size, encoding)] for size in ("small", "medium", "large"))
            self.assertLess(small, medium)
            self.assertLess(medium, large)
        # Over the cap is refused before torch is imported (the host has no torch).
        over = TransformerConfig(layers=8, width=512, heads=8, number_encoding="scaled")
        self.assertGreater(transformer.transformer_parameter_count(self.schema, over), TRANSFORMER_PARAMETER_CAP)
        with self.assertRaisesRegex(ContractError, "cap is 16000000"):
            transformer.build_transformer_policy(self.schema, over, seed=0, device="cpu")

    def test_builder_refusals_before_torch(self):
        build = transformer.build_transformer_policy
        refusals = {
            "an FF config": dict(model_config=STUDENT),
            "an FF identity": dict(model_id=STANDARD_LEARNER_IDENTITY_IDS[0],
                                   features_id=STANDARD_LEARNER_IDENTITY_IDS[1]),
            "the S2 model id with another features id": dict(features_id="spire-learn-combat-features-entity-v2"),
            "a PPO objective": dict(objective_id="spire-learn-combat-winloss-v1"),
            "experimental trim compute": dict(experimental_trim_compute=True),
            "a float seed": dict(seed=1.0),
            "no device": dict(device=""),
        }
        for name, change in refusals.items():
            arguments = {"model_config": SMALL, "seed": 0, "device": "cpu", **change}
            config = arguments.pop("model_config")
            with self.subTest(refusal=name), self.assertRaises(ContractError):
                build(self.schema, config, **arguments)
        with self.assertRaises(ContractError):
            transformer.transformer_plan("schema")

    def test_bucket_index(self):
        cases = {0: 43, 1: 44, -1: 42, 15: 58, -15: 28, 16: 59, 31: 59, 32: 60, 1000: 64,
                 2**31 - 1: 85, -(2**31): 0, 2**40: 86}
        self.assertEqual({value: transformer.bucket_index(value) for value in cases}, cases)
        self.assertEqual(transformer.BUCKET_COUNT, 87)

    def test_import_is_torch_free(self):
        code = ("import sys; import spire_learn.neural.transformer, spire_learn.neural.model_families; "
                "sys.exit(int(any(name in sys.modules for name in ('torch', 'numpy'))))")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={"PYTHONPATH": str(ROOT / "src")}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
