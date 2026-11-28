"""Card metadata features for S3 without torch: snapshot loading, tables, config, identity, registry, counts.

The snapshot itself comes from the runtime container (``card_metadata_snapshot_entry.py``), so these
tests use a small synthetic fixture in the real schema shape (``tests/fixtures/neural/card-metadata-synthetic.json``).
"""

from __future__ import annotations

from dataclasses import asdict, replace
import copy
import json
import math
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
    CARD_METADATA_FINGERPRINT, DISTILL_OBJECTIVE_ID, MODEL_FAMILY_TABLES, MODEL_TABLE_LEARNERS,
    STANDARD_LEARNER_IDENTITY_IDS, TRANSFORMER_CARDMETA_FEATURES_ID, TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS,
    TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_FEATURES_ID, TRANSFORMER_LEARNER_IDENTITY_IDS, TRANSFORMER_MODEL_ID,
    TRANSFORMER_PARAMETER_CAP, TRANSFORMER_SIZES, TransformerCardMetaConfig, TransformerConfig,
    distill_config_hash, load_distill_config, load_neural_config, model_config_hash, transformer_cardmeta_size,
    transformer_size,
)
from spire_learn.neural.gym_contracts import contract_for_revision
from spire_learn.neural.runner import _config_projection
from tests.unit.test_distill_config import (
    GYM, HYPER, MAPPING, COMPACT, _identity_toml, _write, distill_toml, eval_toml,
)
from tests.unit.test_transformer_config import compact_schema

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/neural/card-metadata-synthetic.json"
SMALL = transformer_cardmeta_size("small", "bucket")
CARDMETA_TABLE = ('[transformer_cardmeta]\nlayers = 2\nwidth = 128\nheads = 4\nnumber_encoding = "bucket"\n'
                  f'value_heads = 3\ncard_metadata_fingerprint = "{CARD_METADATA_FINGERPRINT}"\n\n')
S2_TABLE = ('[transformer]\nlayers = 2\nwidth = 128\nheads = 4\nnumber_encoding = "bucket"\nvalue_heads = 3\n\n')
FF_TABLE = "[model]\nembedding_width = 16\nhidden_width = 64\nmax_parameters = 10000000\nvalue_heads = 3\n\n"
CARD_GROUPS = ("combat_cards", "permanent_cards", "combat_card_offers")
# The exact counts on the compact descriptor: S2's plus three projections of CARD_METADATA_FEATURES (44) to the width.
COUNTS = {
    ("small", "bucket"): 1_262_372 + 3 * (44 * 128 + 128),
    ("medium", "scaled"): 5_082_372 + 3 * (44 * 256 + 256),
    ("large", "bucket"): 15_015_460 + 3 * (44 * 384 + 384),
}


def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def cardmeta_distill_toml(*, model_table=CARDMETA_TABLE, learner=TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS,
                          model=SMALL, hashes=None) -> str:
    head, _, _ = distill_toml(hyper=HYPER).partition("[identity]")
    head = head.replace(FF_TABLE, model_table)
    hashes = hashes or {"model": model_config_hash(model), "distill": distill_config_hash(HYPER)}
    return head + _identity_toml(hashes=hashes, learner=learner)


class CardMetadataLoadingTests(unittest.TestCase):
    def test_snapshot_path_and_pinned_fingerprint(self):
        self.assertEqual(CARD_METADATA_FINGERPRINT,
                         "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406")
        self.assertEqual(transformer.CARD_METADATA_SNAPSHOT,
                         ROOT / "configs/neural/standard/card-metadata-gym-1.2.4.json")
        # The pin is the Gym release profile's and the name-alias table's.
        profile = json.loads((ROOT / "configs/neural/gym-1.2.5/runtime.json").read_text())
        self.assertEqual(profile["card_metadata_fingerprint"], CARD_METADATA_FINGERPRINT)
        aliases = json.loads((ROOT / "configs/neural/standard/name-aliases.json").read_text())
        self.assertEqual(aliases["gym_card_metadata_fingerprint"], CARD_METADATA_FINGERPRINT)

    def test_loader_refuses_wrong_files(self):
        self.assertEqual(transformer.load_card_metadata(FIXTURE)["fingerprint"], CARD_METADATA_FINGERPRINT)
        document = fixture()
        refused = {
            "another fingerprint": {**document, "fingerprint": "0" * 64},
            "no fingerprint": {key: value for key, value in document.items() if key != "fingerprint"},
            "another schema": {**document, "schema_id": "spire-lab-card-metadata-v2"},
            "no cards": {**document, "cards": {}},
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, value in refused.items():
                with self.subTest(refusal=name):
                    path = _write(Path(folder), "metadata.json", json.dumps(value))
                    with self.assertRaises(ContractError):
                        transformer.load_card_metadata(path)
            with self.assertRaisesRegex(ContractError, "is not JSON"):
                transformer.load_card_metadata(_write(Path(folder), "bad.json", "{"))
            with self.assertRaisesRegex(ContractError, "cannot be read"):
                transformer.load_card_metadata(Path(folder) / "missing.json")
        # The default path is the committed snapshot; it is loaded or refused, never defaulted around.
        if transformer.CARD_METADATA_SNAPSHOT.exists():
            self.assertEqual(transformer.load_card_metadata()["fingerprint"], CARD_METADATA_FINGERPRINT)
        else:
            with self.assertRaisesRegex(ContractError, "cannot be read"):
                transformer.load_card_metadata()


class CardMetadataTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = compact_schema()
        cls.vocabulary = cls.schema.vocabularies["card"]
        cls.tables = transformer.card_metadata_tables(fixture(), cls.vocabulary)

    def static(self, name):
        return self.tables.static[self.vocabulary.index(name)]

    def face(self, name, level):
        return self.tables.faces[level][self.vocabulary.index(name)]

    def test_feature_widths(self):
        self.assertEqual((transformer.CARD_STATIC_FEATURES, transformer.CARD_FACE_FEATURES,
                          transformer.CARD_METADATA_FEATURES), (25, 19, 44))
        self.assertEqual(self.tables.levels, 2)
        self.assertEqual(len(self.tables.static), len(self.vocabulary))
        self.assertTrue(all(len(row) == 25 for row in self.tables.static))
        self.assertTrue(all(len(row) == 19 for level in self.tables.faces for row in level))
        self.assertEqual(self.tables.rows, 10)
        self.assertEqual(self.tables.missing[0], "padding")
        self.assertEqual(self.tables.unused, ())

    def test_static_fields(self):
        row = self.static("strike")
        colors, rarities, types, tags = row[1:9], row[9:16], row[16:21], row[21:25]
        self.assertEqual(row[0], 1.0)
        self.assertEqual(colors, tuple(1.0 if name == "red" else 0.0 for name in transformer.CARD_COLORS))
        self.assertEqual(rarities, tuple(1.0 if name == "basic" else 0.0 for name in transformer.CARD_RARITIES))
        self.assertEqual(types, tuple(1.0 if name == "attack" else 0.0 for name in transformer.CARD_TYPES))
        self.assertEqual(tags, (1.0, 1.0, 0.0, 0.0))  # strike, starter_strike
        self.assertEqual(self.static("defend")[21:25], (0.0, 0.0, 1.0, 0.0))
        self.assertEqual(self.static("wound")[1:9][transformer.CARD_COLORS.index("status")], 1.0)
        self.assertEqual(self.static("pain")[16:21][transformer.CARD_TYPES.index("curse")], 1.0)

    def test_face_choice_by_upgrade_level(self):
        damage = lambda face: (face[13], face[14])  # noqa: E731 - (base_damage present, value)
        strike0, strike1 = self.face("strike", 0), self.face("strike", 1)
        self.assertEqual(damage(strike0), (1.0, math.log1p(6)))
        self.assertEqual(damage(strike1), (1.0, math.log1p(9)))
        # faces[min(max(upgrades, 0), len(faces) - 1)]: a one-face card uses its face at every level.
        self.assertEqual(self.face("wound", 0), self.face("wound", 1))
        self.assertEqual(self.face("wound", 0)[0], 1.0)
        # Keywords follow the face: Apparition is ethereal and exhaust, then exhaust only.
        keywords = lambda face: face[1:5]  # noqa: E731 - ethereal, exhaust, innate, retain
        self.assertEqual(keywords(self.face("apparition", 0)), (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(keywords(self.face("apparition", 1)), (0.0, 1.0, 0.0, 0.0))

    def test_cost_kind_and_nullable_values(self):
        kind = lambda face: face[5:9]  # noqa: E731 - int, x, unplayable, unresolved
        number = lambda face: face[9]  # noqa: E731
        target = lambda face: face[10:13]  # noqa: E731 - enemy, untargeted, unresolved
        strike, whirlwind = self.face("strike", 0), self.face("whirlwind", 0)
        self.assertEqual((kind(strike), number(strike)), ((1.0, 0.0, 0.0, 0.0), math.log1p(1)))
        self.assertEqual(target(strike), (1.0, 0.0, 0.0))
        self.assertEqual((kind(whirlwind), number(whirlwind)), ((0.0, 1.0, 0.0, 0.0), 0.0))
        self.assertEqual(target(whirlwind), (0.0, 1.0, 0.0))
        wound, regret = self.face("wound", 0), self.face("regret", 0)
        self.assertEqual((kind(wound), number(wound), target(wound)), ((0.0, 0.0, 1.0, 0.0), 0.0, (0.0, 0.0, 1.0)))
        self.assertEqual(kind(regret), (0.0, 0.0, 0.0, 1.0))
        # base_damage, base_block, magic_number: a presence bit then the value; null is (0, 0).
        bash = self.face("bash", 0)
        self.assertEqual(bash[13:19], (1.0, math.log1p(8), 0.0, 0.0, 1.0, math.log1p(2)))
        self.assertEqual(self.face("defend", 0)[13:19], (0.0, 0.0, 1.0, math.log1p(5), 0.0, 0.0))
        # A zero is present: the presence bit tells it from null.
        zero = {"upgrades": 0, "keywords": [], "cost": 0, "target": "untargeted", "base_damage": 0,
                "base_block": None, "magic_number": None}
        features = transformer.card_face_features(zero)
        self.assertEqual((features[9], features[13:15]), (0.0, (1.0, 0.0)))
        self.assertEqual(features[5:9], (1.0, 0.0, 0.0, 0.0))

    def test_no_faces_and_no_row_are_zero(self):
        zero = (0.0,) * 19
        for level in (0, 1):
            self.assertEqual(self.face("wish", level), zero)  # a row with no faces: has_face 0
            self.assertEqual(self.face("padding", level), zero)
            self.assertEqual(self.face("burn", level), zero)  # a vocabulary id the snapshot lacks
        self.assertEqual(self.static("wish")[0], 1.0)  # it still has a row
        self.assertEqual(self.static("padding"), (0.0,) * 25)
        self.assertEqual(self.static("burn"), (0.0,) * 25)
        self.assertEqual(self.tables.faces[0][self.vocabulary.index("wish")][0], 0.0)

    def test_unknown_values_are_refused(self):
        def mutated(change):
            document = copy.deepcopy(fixture())
            change(document["cards"][0])
            return document

        changes = {
            "a new color": lambda row: row.update(color="orange"),
            "a new rarity": lambda row: row.update(rarity="legendary"),
            "a new type": lambda row: row.update(type="relic"),
            "a new tag": lambda row: row.update(tags=["strike", "block"]),
            "a new keyword": lambda row: row["faces"][0].update(keywords=["ethereal", "innate", "fleeting"]),
            "a new cost": lambda row: row["faces"][0].update(cost="free"),
            "a boolean cost": lambda row: row["faces"][0].update(cost=True),
            "a new target": lambda row: row["faces"][0].update(target="all_enemies"),
            "a float damage": lambda row: row["faces"][0].update(base_damage=6.5),
            "a repeated id": lambda row: row.update(id="defend"),
            "null faces": lambda row: row.update(faces=None),
            "non-list faces": lambda row: row.update(faces={"upgrades": 0}),
            "no faces key": lambda row: row.pop("faces"),
            "a non-mapping face": lambda row: row.update(faces=[1]),
            "faces out of order": lambda row: row.update(faces=list(reversed(row["faces"]))),
            "a wrong face upgrade": lambda row: row["faces"][1].update(upgrades=2),
            "a missing face upgrade": lambda row: row["faces"][0].pop("upgrades"),
        }
        for name, change in changes.items():
            with self.subTest(change=name), self.assertRaises(ContractError):
                transformer.card_metadata_tables(mutated(change), self.vocabulary)
        with self.assertRaises(ContractError):
            transformer.card_metadata_tables({**fixture(), "fingerprint": "0" * 64}, self.vocabulary)

    def test_malformed_cards_list_is_a_contract_error(self):
        for cards in (None, {}, "x", 3):
            with self.subTest(cards=cards), self.assertRaises(ContractError):
                transformer.card_metadata_tables({**fixture(), "cards": cards}, self.vocabulary)
        with self.assertRaises(ContractError):
            transformer.card_metadata_tables({k: v for k, v in fixture().items() if k != "cards"}, self.vocabulary)
        with self.assertRaises(ContractError):
            transformer.card_metadata_tables({**fixture(), "cards": [3]}, self.vocabulary)

    def test_extra_metadata_ids_are_reported_not_joined(self):
        tables = transformer.card_metadata_tables(fixture(), ["padding", "strike"])
        self.assertEqual((tables.rows, tables.missing), (1, ("padding",)))
        self.assertIn("defend", tables.unused)

    def test_card_groups_come_from_the_descriptor(self):
        groups = transformer.card_metadata_groups(self.schema)
        self.assertEqual([item.group for item in groups], list(CARD_GROUPS))
        by_name = {item.group: item for item in groups}
        for name in CARD_GROUPS:
            spec = self.schema.tensors[name]
            self.assertEqual(spec.columns[by_name[name].card_field_index].vocabulary, "card")
        for name in ("combat_cards", "permanent_cards"):
            spec = self.schema.tensors[name]
            self.assertEqual(spec.columns[by_name[name].upgrades_field_index].field, "upgrades")
        self.assertIsNone(by_name["combat_card_offers"].upgrades_field_index)  # offers are level 0


class CardMetaConfigTests(unittest.TestCase):
    def test_identity_and_config(self):
        self.assertEqual((TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_CARDMETA_FEATURES_ID),
                         ("spire-learn-combat-entity-transformer-cardmeta-v1",
                          "spire-learn-combat-features-entity-cardmeta-v1"))
        self.assertEqual(TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS[2], "spire-learn-combat-dispatch-v1")
        # S2, S3 and FF identities are all distinct in model and in features.
        others = (TRANSFORMER_LEARNER_IDENTITY_IDS, STANDARD_LEARNER_IDENTITY_IDS)
        for other in others:
            self.assertNotEqual(TRANSFORMER_CARDMETA_MODEL_ID, other[0])
            self.assertNotEqual(TRANSFORMER_CARDMETA_FEATURES_ID, other[1])
        self.assertEqual(asdict(SMALL), {"layers": 2, "width": 128, "heads": 4, "number_encoding": "bucket",
                                         "value_heads": 3, "card_metadata_fingerprint": CARD_METADATA_FINGERPRINT})
        self.assertEqual(SMALL.transformer(), transformer_size("small", "bucket"))
        # The fingerprint is part of the hash, and S3's hash is not S2's.
        self.assertNotEqual(model_config_hash(SMALL), model_config_hash(transformer_size("small", "bucket")))
        for bad in (dict(layers=0), dict(width=130), dict(number_encoding="raw"), dict(value_heads=1),
                    dict(card_metadata_fingerprint="0" * 64), dict(card_metadata_fingerprint="")):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                replace(SMALL, **bad)
        with self.assertRaises(ValueError):
            transformer_cardmeta_size("huge", "bucket")

    def test_s2_config_hash_is_unchanged_since_task_10(self):
        # Computed at base 0970605, before Task 11: S3 must not move S2's hash.
        self.assertEqual(model_config_hash(transformer_size("small", "bucket")),
                         "02260177e056a67fd8328a7014c147a7e28873d5e977b020bab24db3bb70baba")
        self.assertEqual(model_config_hash(transformer_size("large", "sinusoidal")),
                         "55865d3d1cff139189f1431a65464fddd0c3bb3ce89285276180731845c239f5")

    def test_cardmeta_table_in_a_distillation_training_config(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = load_distill_config(_write(folder, "s3.toml", cardmeta_distill_toml()))
            self.assertEqual(config.model_table, "transformer_cardmeta")
            self.assertEqual(config.model, SMALL)
            self.assertEqual((config.identity.model_id, config.identity.features_id, config.identity.action_id),
                             TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS)
            self.assertEqual(dict(config.identity.config_hashes),
                             {"model": model_config_hash(SMALL), "distill": distill_config_hash(HYPER)})
            hashes = {"model": model_config_hash(transformer_size("small", "bucket")),
                      "distill": distill_config_hash(HYPER)}
            refusals = {
                "an S2 identity on [transformer_cardmeta]": cardmeta_distill_toml(
                    learner=TRANSFORMER_LEARNER_IDENTITY_IDS),
                "an FF identity on [transformer_cardmeta]": cardmeta_distill_toml(
                    learner=STANDARD_LEARNER_IDENTITY_IDS),
                "an S3 identity on [transformer]": cardmeta_distill_toml(
                    model_table=S2_TABLE, hashes=hashes),
                "both transformer tables": cardmeta_distill_toml(model_table=S2_TABLE + CARDMETA_TABLE),
                "a missing fingerprint": cardmeta_distill_toml(model_table=CARDMETA_TABLE.replace(
                    f'card_metadata_fingerprint = "{CARD_METADATA_FINGERPRINT}"\n', "")),
                "another fingerprint": cardmeta_distill_toml(model_table=CARDMETA_TABLE.replace(
                    CARD_METADATA_FINGERPRINT, "0" * 64)),
                "an extra key": cardmeta_distill_toml(model_table=CARDMETA_TABLE.replace(
                    "value_heads = 3\n", "value_heads = 3\nmax_parameters = 1\n")),
                "a stale model hash": cardmeta_distill_toml(model=transformer_cardmeta_size("small", "scaled")),
            }
            for name, text in refusals.items():
                with self.subTest(refusal=name), self.assertRaises(ValueError):
                    load_distill_config(_write(folder, "bad.toml", text))
            # And the S2 identity does not load on S3's table in the other direction either.
            s2 = load_distill_config(_write(folder, "s2.toml", distill_toml_s2()))
            self.assertEqual(s2.model_table, "transformer")

    def test_cardmeta_table_in_an_evaluation_config(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            hashes = {"model": model_config_hash(SMALL)}
            text = eval_toml(folder, model_table=CARDMETA_TABLE, hashes=hashes,
                             learner=TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS)
            config = load_neural_config(_write(folder, "eval.toml", text))
            self.assertEqual((config.model_table, config.model, config.ppo), ("transformer_cardmeta", SMALL, None))
            self.assertEqual(config.accepted_identities[0].model_id, TRANSFORMER_CARDMETA_MODEL_ID)
            projection = _config_projection(config)
            self.assertEqual(projection["transformer_cardmeta"], asdict(SMALL))
            self.assertNotIn("model", projection)
            self.assertNotIn("transformer", projection)
            refusals = {
                "an S3 identity on [model]": eval_toml(folder, learner=TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS),
                "an S3 identity on [transformer]": eval_toml(
                    folder, model_table=S2_TABLE, learner=TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS,
                    hashes={"model": model_config_hash(transformer_size("small", "bucket"))}),
                "an S2 identity on [transformer_cardmeta]": eval_toml(
                    folder, model_table=CARDMETA_TABLE, hashes=hashes, learner=TRANSFORMER_LEARNER_IDENTITY_IDS),
                "an FF identity on [transformer_cardmeta]": eval_toml(folder, model_table=CARDMETA_TABLE,
                                                                       hashes=hashes),
            }
            for name, text in refusals.items():
                with self.subTest(refusal=name), self.assertRaisesRegex(ValueError, "does not configure learner"):
                    load_neural_config(_write(folder, "bad.toml", text))
        # S3 has no PPO arm (delta 16): its identity takes the distillation objective only.
        raw = {"model_id": TRANSFORMER_CARDMETA_MODEL_ID, "features_id": TRANSFORMER_CARDMETA_FEATURES_ID,
               "action_id": TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS[2], "objective_id": "spire-learn-combat-winloss-v1",
               "accepted_raw_mode_mappings": {"standard-standalone": MAPPING},
               "config_hashes": {"model": "a" * 64, "ppo": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "objective only"):
            config_module._identity(raw, model_hash="a" * 64, ppo_hash="b" * 64, contract=contract_for_revision(GYM),
                                    standard_view_fingerprint=COMPACT)

    def test_tables_registered_with_their_learners(self):
        self.assertIn("transformer_cardmeta", MODEL_FAMILY_TABLES)
        self.assertEqual(MODEL_TABLE_LEARNERS["transformer_cardmeta"], (TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS,))
        self.assertEqual(MODEL_TABLE_LEARNERS["transformer"], (TRANSFORMER_LEARNER_IDENTITY_IDS,))


def distill_toml_s2() -> str:
    head, _, _ = distill_toml(hyper=HYPER).partition("[identity]")
    head = head.replace(FF_TABLE, S2_TABLE)
    hashes = {"model": model_config_hash(transformer_size("small", "bucket")), "distill": distill_config_hash(HYPER)}
    return head + _identity_toml(hashes=hashes, learner=TRANSFORMER_LEARNER_IDENTITY_IDS)


class CardMetaFamilyTests(unittest.TestCase):
    def test_family_registered_with_its_cap(self):
        family = families.model_family(TRANSFORMER_CARDMETA_MODEL_ID)
        self.assertIs(family, families.TRANSFORMER_CARDMETA_FAMILY)
        self.assertEqual((family.family_id, family.model_ids, family.config_type, family.parameter_cap),
                         ("transformer-cardmeta", (TRANSFORMER_CARDMETA_MODEL_ID,), TransformerCardMetaConfig,
                          16_000_000))
        self.assertTrue(family.check_built)
        # S2 and S3 are separate families: neither takes the other's model id or config.
        self.assertIsNot(family, families.model_family(TRANSFORMER_MODEL_ID))
        with self.assertRaisesRegex(TypeError, "TransformerCardMetaConfig"):
            families.check_family_config(TRANSFORMER_CARDMETA_MODEL_ID, transformer_size("small", "bucket"))
        with self.assertRaisesRegex(TypeError, "TransformerConfig"):
            families.check_family_config(TRANSFORMER_MODEL_ID, SMALL)
        options = dict(seed=3, device="cpu", objective_id=DISTILL_OBJECTIVE_ID, model_id=TRANSFORMER_CARDMETA_MODEL_ID,
                       features_id=TRANSFORMER_CARDMETA_FEATURES_ID)
        with patch.object(transformer, "build_transformer_cardmeta_policy", return_value="policy") as build:
            self.assertEqual(family.build("schema", SMALL, **options), "policy")
        build.assert_called_once_with("schema", SMALL, **options)

    def test_distillation_trainer_builds_through_the_family(self):
        from spire_learn.neural import distill_train

        with tempfile.TemporaryDirectory() as folder:
            s3 = load_distill_config(_write(Path(folder), "s3.toml", cardmeta_distill_toml()))
        seed = distill_train.domain_seed(s3.seed, "policy")
        with patch.object(transformer, "build_transformer_cardmeta_policy", return_value="s3") as build:
            with patch.object(families, "check_policy", side_effect=lambda policy, **_: policy):
                self.assertEqual(distill_train.build_student(s3, "schema", device="cpu"), "s3")
        build.assert_called_once_with("schema", SMALL, seed=seed, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
                                      model_id=TRANSFORMER_CARDMETA_MODEL_ID,
                                      features_id=TRANSFORMER_CARDMETA_FEATURES_ID)


class CardMetaBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = compact_schema()

    def test_parameter_counts(self):
        for (size, encoding), expected in COUNTS.items():
            config = transformer_cardmeta_size(size, encoding)
            with self.subTest(size=size, encoding=encoding):
                count = transformer.transformer_cardmeta_parameter_count(self.schema, config)
                self.assertEqual(count, expected)
                self.assertEqual(count, transformer.transformer_parameter_count(self.schema, config.transformer())
                                 + 3 * (transformer.CARD_METADATA_FEATURES * config.width + config.width))
                self.assertLessEqual(count, TRANSFORMER_PARAMETER_CAP)
        # The large bucket model still fits under the cap with its metadata projections.
        self.assertLess(COUNTS[("large", "bucket")], TRANSFORMER_PARAMETER_CAP)

    def test_builder_refusals_before_torch(self):
        options = dict(seed=1, device="cpu")
        refusals = {
            "an S2 config": (transformer_size("small", "bucket"), {}, "TransformerCardMetaConfig"),
            "S2's ids": (SMALL, dict(model_id=TRANSFORMER_MODEL_ID, features_id=TRANSFORMER_FEATURES_ID), "identity pair"),
            "S3 model, S2 features": (SMALL, dict(features_id=TRANSFORMER_FEATURES_ID), "identity pair"),
            "a PPO objective": (SMALL, dict(objective_id="spire-learn-combat-winloss-v1"), "student only"),
            "trim compute": (SMALL, dict(experimental_trim_compute=True), "experimental_trim_compute"),
            "a float seed": (SMALL, dict(seed=1.0), "seed"),
            "another fingerprint": (SMALL, dict(card_metadata={**fixture(), "fingerprint": "0" * 64}), "fingerprint"),
        }
        with patch.dict(sys.modules, {"torch": None}):  # any torch import would fail with ImportError
            for name, (config, changes, message) in refusals.items():
                with self.subTest(refusal=name), self.assertRaisesRegex(ContractError, message):
                    transformer.build_transformer_cardmeta_policy(self.schema, config, **{**options, **changes})
        with patch.dict(sys.modules, {"torch": None}), patch.object(transformer, "TRANSFORMER_PARAMETER_CAP", 1000):
            with self.assertRaisesRegex(ContractError, "its cap is 1000"):
                transformer.build_transformer_cardmeta_policy(self.schema, SMALL, card_metadata=fixture(), **options)
        # Without a snapshot argument the committed snapshot is loaded and refused if absent or other.
        with tempfile.TemporaryDirectory() as folder, patch.object(
                transformer, "CARD_METADATA_SNAPSHOT", Path(folder) / "missing.json"):
            with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ContractError, "cannot be read"):
                transformer.build_transformer_cardmeta_policy(self.schema, SMALL, **options)

    def test_import_is_torch_free(self):
        code = ("import sys; import spire_learn.neural.transformer, spire_learn.neural.model_families; "
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})


class SnapshotEntryTests(unittest.TestCase):
    def run_entry(self, metadata):
        import os

        from scripts.neural import card_metadata_snapshot_entry as entry

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": folder}), \
                patch("spire_learn.neural.standard_names.gym_card_metadata", return_value=metadata), \
                patch("builtins.print") as printed:
            code = entry.main([])
            written = (Path(folder) / "card-metadata-gym-1.2.4.json").read_text()
        return code, written, json.loads(printed.call_args.args[0])

    def test_writes_the_metadata_unchanged(self):
        metadata = fixture()
        code, written, summary = self.run_entry(metadata)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(written), metadata)
        self.assertTrue(written.endswith("\n"))
        self.assertTrue(summary["matches_pin"])
        self.assertEqual((summary["cards"], summary["faces"]), (10, 15))
        # The written file is what the loader accepts.
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(transformer.load_card_metadata(_write(Path(folder), "s.json", written)), metadata)

    def test_a_different_fingerprint_fails_the_entry(self):
        code, written, summary = self.run_entry({**fixture(), "fingerprint": "0" * 64})
        self.assertEqual(code, 1)
        self.assertFalse(summary["matches_pin"])
        with self.assertRaises(ContractError), tempfile.TemporaryDirectory() as folder:
            transformer.load_card_metadata(_write(Path(folder), "s.json", written))


if __name__ == "__main__":
    unittest.main()
