"""Offline contract and profile checks for Spire Gym 1.2.4 (its own rules epoch, gym-1.2.4)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.neural import recipes
from spire_learn.neural import standard_stream as ss
from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.config import ACCEPTED_RUNTIME_GYM_REVISIONS, load_neural_config
from spire_learn.neural.gym_contracts import (
    GYM_V1_2_0_CONTRACT, GYM_V1_2_4_CONTRACT, contract_for_revision, contract_for_run_states,
)
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_gym_1_1_1 import _refuse_torch
from tests.unit.test_neural_standard_stream import GYM_1_2_0, GYM_1_2_4, recipe_document, write_table


ROOT = Path(__file__).resolve().parents[2]
REVISION = GYM_1_2_4
COMPACT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
COMPACT_PLAY = "fbfade07003bd64ad6071243ec3a52e23413c88c5758ca702a515945935439ae"
MODES = ("standalone", "run", "standard-standalone", "standard-run")


class GymV124Tests(unittest.TestCase):
    def test_v1_2_4_contract_identities(self):
        contract = GYM_V1_2_4_CONTRACT
        self.assertIs(contract_for_revision(REVISION), contract)
        self.assertIn(REVISION, ACCEPTED_RUNTIME_GYM_REVISIONS)
        self.assertEqual(contract.runtime_revisions, (REVISION,))
        self.assertEqual(contract.rules_epoch, "gym-1.2.4")
        self.assertEqual(contract.standard_rules_id, "spire-gym-native-standard-rules-v9")
        self.assertEqual(contract.m10_rules_id, "spire-gym-native-m10-rules-v6")
        # Descriptor fingerprints, entry resources and card metadata are unchanged from v1.2.0.
        for field in ("common_fingerprint", "run_fingerprint", "card_metadata_fingerprint",
                      "standard_combat_fingerprint", "standard_run_fingerprint",
                      "standard_compact_combat_fingerprint", "standard_compact_combat_play_fingerprint",
                      "entry_resources_fingerprint"):
            self.assertEqual(getattr(contract, field), getattr(GYM_V1_2_0_CONTRACT, field), field)
        self.assertEqual(contract.standard_compact_combat_fingerprint, COMPACT)
        self.assertEqual(contract.standard_compact_combat_play_fingerprint, COMPACT_PLAY)
        self.assertEqual(contract.entry_resources_fingerprint,
                         "4eafca3b0089e15735fb6aef24d594a084030438daa9738a1ac8cf6eab75dc5a")
        self.assertEqual(contract.build.revision, REVISION)
        self.assertEqual(contract.build.source_digest,
                         "194a901c8bff91a5176eb994f5e1b52c70473c3af74ea0cf1c2672cd679f3757")
        self.assertEqual(contract.build.file_digest,
                         "934fcd6a297196d493e63ea8ae0bf5f0e5f021de877e9ee646cc5e181a611dfb")
        self.assertEqual((contract.build.profile, contract.build.features),
                         ("release", ("spire-python/extension",)))
        self.assertTrue(all(value.endswith("+gym-1.2.4") for value in contract.raw_mode_mappings.values()))
        self.assertEqual(set(contract.raw_mode_mappings), set(MODES))
        self.assertEqual(contract.raw_mode_mappings_for_view(COMPACT)["standard-standalone"],
                         f"spire-learn-standard-combat-compact-v1@{COMPACT}+gym-1.2.4")
        for mode in MODES:
            self.assertNotEqual(contract.raw_mode_mappings[mode], GYM_V1_2_0_CONTRACT.raw_mode_mappings[mode])
        self.assertEqual(len(contract.standard_stream_smoke_tests), 2)
        self.assertEqual(contract_for_run_states(REVISION, "spire-learn-run-states-v1"), contract)

    def test_v1_2_0_is_unchanged(self):
        self.assertEqual(GYM_V1_2_0_CONTRACT.rules_epoch, "gym-1.2.0")
        self.assertEqual(GYM_V1_2_0_CONTRACT.standard_rules_id, "spire-gym-native-standard-rules-v7")
        self.assertEqual(GYM_V1_2_0_CONTRACT.m10_rules_id, "spire-gym-native-m10-rules-v4")
        self.assertEqual(GYM_V1_2_0_CONTRACT.source_revision, GYM_1_2_0)

    def test_checkpoints_do_not_cross_gym_1_2_0_and_gym_1_2_4(self):
        for mode in MODES:
            old = load_neural_config(
                ROOT / "configs/neural/gym-1.2.0" / f"{mode}.toml").accepted_identities[0]
            new = load_neural_config(
                ROOT / "configs/neural/gym-1.2.4" / f"{mode}.toml").accepted_identities[0]
            with patch("builtins.__import__", side_effect=_refuse_torch):
                for expected, actual in ((old, new), (new, old)):
                    with self.subTest(mode=mode), self.assertRaisesRegex(
                            ValueError, "raw-mode mapping"):
                        validate_checkpoint_metadata(
                            _metadata(identity=expected), actual, purpose="evaluate")

    def test_gym_1_2_4_profiles_load(self):
        for name in (*MODES, "standard-stream"):
            config = load_neural_config(ROOT / "configs/neural/gym-1.2.4" / f"{name}.toml")
            self.assertEqual(config.environment_revision, REVISION)
            if name in MODES:
                self.assertEqual(config.mode, name)
                expected = (GYM_V1_2_4_CONTRACT.raw_mode_mappings_for_view(COMPACT)
                            if name == "standard-standalone" else GYM_V1_2_4_CONTRACT.raw_mode_mappings)
                self.assertEqual(config.accepted_identities[0].accepted_raw_mode_mappings[name], expected[name])
            if name in ("standard-standalone", "standard-stream"):
                self.assertEqual((config.standard_view_layout, config.standard_view_preset,
                                  config.standard_view_fingerprint),
                                 ("compact", "standard-combat-compact-v1", COMPACT))
        profile = json.loads((ROOT / "configs/neural/gym-1.2.4/runtime.json").read_text())
        self.assertEqual(profile["schema"], "spire-learn-gym-runtime-profile-v2")
        self.assertEqual(profile["gym_revision"], REVISION)
        self.assertEqual(profile["native_sha256"],
                         "c2b1d5352b95a95c246d5992850300b06bb520e858db9b335d640668f3c6e828")
        self.assertEqual(profile["wheel_sha256"],
                         "5726443b35543f378fdaa6857d6d518132ab460157c1d98e2b0ee3b2a41ce317")
        self.assertEqual(profile["wheel_path"],
                         "artifacts/development/neural-combat/gym-1.2.4/inputs/"
                         "spire_lab-1.2.4-cp312-cp312-manylinux_2_34_x86_64.whl")
        self.assertEqual(profile["package_overlay"], "artifacts/development/neural-combat/gym-1.2.4/site")
        self.assertEqual(profile["delivery_manifest_sha256"],
                         "0ffd08a6d8e284409a89307ee48643c53233029a4a4ae863338a77ae3dab8c64")
        old = json.loads((ROOT / "configs/neural/gym-1.2.0/runtime.json").read_text())
        self.assertEqual(profile["dependency_venv"], old["dependency_venv"])
        self.assertEqual(profile["dependency_lock_sha256"], old["dependency_lock_sha256"])
        self.assertEqual(profile["build_identity"], {
            "dirty": False, "extension": "spire-python",
            "features": ["spire-python/extension"],
            "file_digest": GYM_V1_2_4_CONTRACT.build.file_digest,
            "profile": "release", "revision": REVISION, "rust_version": "1.99.0",
            "source_digest": GYM_V1_2_4_CONTRACT.build.source_digest,
        })
        self.assertEqual(profile["standard_fingerprints"], {
            "combat": GYM_V1_2_4_CONTRACT.standard_combat_fingerprint,
            "run": GYM_V1_2_4_CONTRACT.standard_run_fingerprint,
            "combat_compact": COMPACT,
            "combat_compact_play": COMPACT_PLAY,
        })
        self.assertEqual(profile["card_metadata_fingerprint"], GYM_V1_2_4_CONTRACT.card_metadata_fingerprint)
        for key in ("common_view_fingerprint", "run_observation_fingerprint", "checkpoint_policy"):
            self.assertEqual(profile[key], old[key], key)

    def test_v1_2_0_table_refused_under_v1_2_4(self):
        """A run-states table (and so its recipe) belongs to one Gym revision; tables are regenerated."""
        with tempfile.TemporaryDirectory() as tmp:
            recipes_by_revision = {}
            for revision in (GYM_1_2_0, GYM_1_2_4):
                folder = Path(tmp) / revision[:7]
                folder.mkdir()
                _, sha = write_table(folder, revision=revision)
                path = folder / "recipe.json"
                path.write_text(json.dumps(recipe_document(sha)))
                recipe = ss.load_standard_recipe(path)
                self.assertEqual(recipe.gym_revision, revision)
                recipes_by_revision[revision] = recipe
        recipes.require_recipe_revision(recipes_by_revision[GYM_1_2_4], GYM_1_2_4)
        recipes.require_recipe_revision(recipes_by_revision[GYM_1_2_0], GYM_1_2_0)
        for recipe_revision, config_revision in ((GYM_1_2_0, GYM_1_2_4), (GYM_1_2_4, GYM_1_2_0)):
            with self.subTest(table=recipe_revision, config=config_revision), \
                    self.assertRaisesRegex(ValueError, "run-states table is for Gym"):
                recipes.require_recipe_revision(recipes_by_revision[recipe_revision], config_revision)
        # A stored table's recorded revision cannot be edited to the other release unnoticed:
        # the recipe pins the table's content hash.
        with tempfile.TemporaryDirectory() as tmp:
            table, sha = write_table(Path(tmp), revision=GYM_1_2_0)
            document = json.loads(table.read_bytes())
            document["gym_revision"] = GYM_1_2_4
            table.write_text(json.dumps(document))
            (Path(tmp) / "recipe.json").write_text(json.dumps(recipe_document(sha)))
            with self.assertRaises(ValueError):
                ss.load_standard_recipe(Path(tmp) / "recipe.json")


if __name__ == "__main__":
    unittest.main()
