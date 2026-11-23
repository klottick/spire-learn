"""Offline contract and profile checks for Spire Gym 1.2.0."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.config import ACCEPTED_RUNTIME_GYM_REVISIONS, load_neural_config
from spire_learn.neural.gym_contracts import (
    GYM_V1_1_1_CONTRACT, GYM_V1_2_0_CONTRACT, contract_for_revision,
)
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_gym_1_1_1 import _refuse_torch


ROOT = Path(__file__).resolve().parents[2]
REVISION = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"
COMPACT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
COMPACT_PLAY = "fbfade07003bd64ad6071243ec3a52e23413c88c5758ca702a515945935439ae"
MODES = ("standalone", "run", "standard-standalone", "standard-run")


class GymV120Tests(unittest.TestCase):
    def test_v1_2_0_contract_identities(self):
        contract = GYM_V1_2_0_CONTRACT
        self.assertIs(contract_for_revision(REVISION), contract)
        self.assertIn(REVISION, ACCEPTED_RUNTIME_GYM_REVISIONS)
        self.assertEqual(contract.runtime_revisions, (REVISION,))
        self.assertEqual(contract.rules_epoch, "gym-1.2.0")
        self.assertEqual(contract.standard_rules_id, "spire-gym-native-standard-rules-v7")
        self.assertEqual(contract.m10_rules_id, "spire-gym-native-m10-rules-v4")
        for field in ("common_fingerprint", "run_fingerprint", "card_metadata_fingerprint",
                      "standard_combat_fingerprint", "standard_run_fingerprint"):
            self.assertEqual(getattr(contract, field), getattr(GYM_V1_1_1_CONTRACT, field))
        self.assertEqual(contract.standard_combat_fingerprint,
                         "5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e")
        self.assertEqual(contract.standard_compact_combat_fingerprint, COMPACT)
        self.assertEqual(contract.standard_compact_combat_play_fingerprint, COMPACT_PLAY)
        self.assertIsNone(GYM_V1_1_1_CONTRACT.standard_compact_combat_fingerprint)
        self.assertEqual(contract.build.revision, REVISION)
        self.assertEqual(contract.build.source_digest,
                         "4cd56ee7f22dd8303f7d5ff293d88fd65f6071088a00dab73e5c1b176dbe7508")
        self.assertEqual(contract.build.file_digest,
                         "888d789820040c3dc9a795850f75580eb5e31bfb6760c3e1809933ad966561bd")
        self.assertTrue(all(value.endswith("+gym-1.2.0")
                            for value in contract.raw_mode_mappings.values()))
        self.assertEqual(set(contract.raw_mode_mappings), set(MODES))

    def test_checkpoints_do_not_cross_v1_1_1_and_v1_2_0(self):
        for mode in MODES:
            old = load_neural_config(
                ROOT / "configs/neural/gym-1.1.1" / f"{mode}.toml").accepted_identities[0]
            new = load_neural_config(
                ROOT / "configs/neural/gym-1.2.0" / f"{mode}.toml").accepted_identities[0]
            with patch("builtins.__import__", side_effect=_refuse_torch):
                for expected, actual in ((old, new), (new, old)):
                    with self.subTest(mode=mode), self.assertRaisesRegex(
                            ValueError, "raw-mode mapping"):
                        validate_checkpoint_metadata(
                            _metadata(identity=expected), actual, purpose="evaluate")

    def test_gym_1_2_0_profiles_load(self):
        for name in (*MODES, "standard-stream"):
            path = ROOT / "configs/neural/gym-1.2.0" / f"{name}.toml"
            config = load_neural_config(path)
            self.assertEqual(config.environment_revision, REVISION)
            if name in MODES:
                self.assertEqual(config.mode, name)
                self.assertEqual(config.accepted_identities[0].accepted_raw_mode_mappings[name],
                                 GYM_V1_2_0_CONTRACT.raw_mode_mappings[name])
        profile = json.loads((ROOT / "configs/neural/gym-1.2.0/runtime.json").read_text())
        self.assertEqual(profile["schema"], "spire-learn-gym-runtime-profile-v2")
        self.assertEqual(profile["gym_revision"], REVISION)
        self.assertEqual(profile["native_sha256"],
                         "1f590dcf95d538b88b9b639d71783289281741a37f7884e412be55babc12970b")
        self.assertEqual(profile["wheel_sha256"],
                         "a9d8cdfb26f0e762baea690c9d88639e1dd172e714024a5029109d367978b297")
        self.assertEqual(profile["wheel_path"],
                         "artifacts/development/neural-combat/gym-1.2.0/inputs/"
                         "spire_lab-1.2.0-cp312-cp312-manylinux_2_34_x86_64.whl")
        self.assertEqual(profile["package_overlay"],
                         "artifacts/development/neural-combat/gym-1.2.0/site")
        old = json.loads((ROOT / "configs/neural/gym-1.1.1/runtime.json").read_text())
        self.assertEqual(profile["dependency_venv"], old["dependency_venv"])
        self.assertEqual(profile["dependency_lock_sha256"], old["dependency_lock_sha256"])
        self.assertEqual(profile["build_identity"], {
            "dirty": False, "extension": "spire-python",
            "features": ["spire-python/extension"],
            "file_digest": GYM_V1_2_0_CONTRACT.build.file_digest,
            "profile": "release", "revision": REVISION, "rust_version": "1.99.0",
            "source_digest": GYM_V1_2_0_CONTRACT.build.source_digest,
        })
        self.assertEqual(profile["standard_fingerprints"], {
            "combat": GYM_V1_2_0_CONTRACT.standard_combat_fingerprint,
            "run": GYM_V1_2_0_CONTRACT.standard_run_fingerprint,
            "combat_compact": COMPACT,
            "combat_compact_play": COMPACT_PLAY,
        })
        self.assertEqual(profile["card_metadata_fingerprint"],
                         GYM_V1_2_0_CONTRACT.card_metadata_fingerprint)

    def test_qualification_and_soak_default_profile_directory(self):
        from scripts.neural import qualify_gym_v1, soak_gym_v1
        with patch.object(qualify_gym_v1, "run") as run:
            qualify_gym_v1.main(["--output", "out"])
            self.assertEqual(run.call_args.kwargs["profile_dir"], Path("configs/neural/gym-1.2.0"))
        with patch.object(soak_gym_v1, "run") as run:
            soak_gym_v1.main(["--steps", "1", "--output", "out"])
            self.assertEqual(run.call_args.kwargs["profile_dir"], Path("configs/neural/gym-1.2.0"))


if __name__ == "__main__":
    unittest.main()
