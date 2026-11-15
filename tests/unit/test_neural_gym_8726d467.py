"""Offline contract and profile checks for Spire Gym 8726d467."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import envs, runner
from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.gym_contracts import (
    CURRENT_CONTRACT, LEGACY_CONTRACT, GYM_V1_CONTRACT, GYM_8726D467_CONTRACT,
    contract_for_revision, contracts_for_fingerprint,
)
from tests.unit.test_neural_checkpoint import _metadata


ROOT = Path(__file__).resolve().parents[2]
REVISION = "8726d467ad8579730ddf4fa236b24aeb2206b71a"
COMMON = "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463"


class Gym8726D467Tests(unittest.TestCase):
    def test_release_contract_and_epoch_mappings(self):
        contract = contract_for_revision(REVISION)
        self.assertIs(contract, GYM_8726D467_CONTRACT)
        self.assertEqual(contract.runtime_revisions, (REVISION,))
        self.assertEqual(contract.common_fingerprint, COMMON)
        self.assertEqual(contract.run_fingerprint, GYM_V1_CONTRACT.run_fingerprint)
        self.assertEqual(contract.card_metadata_fingerprint,
                         GYM_V1_CONTRACT.card_metadata_fingerprint)
        self.assertEqual(contract.standard_combat_fingerprint,
                         GYM_V1_CONTRACT.standard_combat_fingerprint)
        self.assertEqual(contract.standard_run_fingerprint,
                         GYM_V1_CONTRACT.standard_run_fingerprint)
        self.assertEqual(contract.build.revision, REVISION)
        self.assertEqual(contract.build.source_digest,
                         "73f554e2a9a2cb20c23efff0daba83e4ff6a6bcd90e0fbd33b48f8534bce1d7a")
        self.assertEqual(contract.build.file_digest,
                         "98db0a9ac92e05c45424fb5658485ed16ee92a0cfcd49063728c2d9a905a6ccf")
        self.assertEqual(contract.build.profile, "release")
        self.assertEqual(contract.build.features, ("spire-python/extension",))
        self.assertEqual(GYM_V1_CONTRACT.standard_rules_id,
                         "spire-gym-native-standard-rules-v3")
        self.assertIsNone(LEGACY_CONTRACT.standard_rules_id)
        self.assertEqual(LEGACY_CONTRACT.m10_rules_id, "spire-gym-native-m10-rules-v1")
        self.assertEqual(contract.m10_rules_id, "spire-gym-native-m10-rules-v2")
        self.assertEqual(contract.standard_rules_id, "spire-gym-native-standard-rules-v5")
        self.assertEqual(contract.rules_epoch, "gym-1.0.0+8726d467")
        self.assertTrue(all(value.endswith("+gym-1.0.0+8726d467")
                            for value in contract.raw_mode_mappings.values()))
        self.assertEqual(contracts_for_fingerprint(COMMON),
                         (CURRENT_CONTRACT, GYM_V1_CONTRACT, contract))
        self.assertEqual(dict(LEGACY_CONTRACT.raw_mode_mappings), {
            "standalone": "spire-learn-standalone-common-v1@02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78",
            "run": "spire-learn-run-policy-v3-common-v1@30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8:02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78",
        })
        self.assertEqual(dict(CURRENT_CONTRACT.raw_mode_mappings), {
            "standalone": f"spire-learn-standalone-common-v1@{COMMON}",
            "run": f"spire-learn-run-policy-v3-common-v1@{CURRENT_CONTRACT.run_fingerprint}:{COMMON}",
        })
        self.assertEqual(GYM_V1_CONTRACT.raw_mode_mappings["standalone"],
                         f"spire-learn-standalone-common-v1@{COMMON}+gym-1.0.0")

    def test_full_build_identity_is_verified(self):
        build = GYM_8726D467_CONTRACT.build
        self.assertIsNotNone(build)
        info = {
            "extension": "spire-python", "revision": REVISION, "dirty": False,
            "profile": "release", "features": ["spire-python/extension"],
            "source_digest": "73f554e2a9a2cb20c23efff0daba83e4ff6a6bcd90e0fbd33b48f8534bce1d7a",
            "file_digest": "98db0a9ac92e05c45424fb5658485ed16ee92a0cfcd49063728c2d9a905a6ccf",
        }
        envs._verify_native_build(SimpleNamespace(build_info=lambda: info), REVISION)
        for key, value in (("dirty", True), ("profile", "debug"),
                           ("source_digest", "0" * 64), ("file_digest", "0" * 64),
                           ("features", ["extra"])):
            changed = {**info, key: value}
            with self.subTest(key=key), self.assertRaises(ContractError):
                envs._verify_native_build(SimpleNamespace(build_info=lambda c=changed: c), REVISION)

    def test_new_templates_validate_and_profile_matches_contract(self):
        expected_modes = ("standalone", "run", "standard-standalone", "standard-run")
        for mode in expected_modes:
            path = ROOT / "configs/neural/gym-8726d467" / f"{mode}.toml"
            config = load_neural_config(path)
            self.assertEqual(config.mode, mode)
            self.assertEqual(config.environment_revision, REVISION)
            self.assertEqual(config.accepted_identities[0].accepted_raw_mode_mappings[mode],
                             GYM_8726D467_CONTRACT.raw_mode_mappings[mode])
            from spire_learn.neural.cli import main
            with patch("sys.stdout"):
                self.assertEqual(main(["train", "--config", str(path), "--validate-only"]), 0)
        profile = json.loads((ROOT / "configs/neural/gym-8726d467/runtime.json").read_text())
        self.assertEqual(profile["schema"], "spire-learn-gym-runtime-profile-v2")
        self.assertEqual(profile["gym_revision"], REVISION)
        self.assertEqual(profile["native_sha256"], "7019351538ae567abc2ffd7f40cc98d5c83419065685fcae12e13dd83867c86d")
        self.assertEqual(profile["wheel_sha256"], "96321606aa43ef73d50af4e6ed8d401aee26b88bc1044b16132935fd9c379fb7")
        self.assertEqual(profile["delivery_manifest_sha256"], "b8b00e5957786f9e965bec41a55f2acf2423d96104a23918124eaa405c35adf1")
        self.assertEqual(profile["build_identity"], {
            "dirty": False, "extension": "spire-python",
            "features": ["spire-python/extension"],
            "file_digest": GYM_8726D467_CONTRACT.build.file_digest,
            "profile": "release", "revision": REVISION,
            "rust_version": "1.99.0",
            "source_digest": GYM_8726D467_CONTRACT.build.source_digest,
        })
        self.assertEqual(profile["common_view_fingerprint"], GYM_8726D467_CONTRACT.common_fingerprint)
        self.assertEqual(profile["run_observation_fingerprint"], GYM_8726D467_CONTRACT.run_fingerprint)
        self.assertEqual(profile["card_metadata_fingerprint"],
                         GYM_8726D467_CONTRACT.card_metadata_fingerprint)
        self.assertEqual(profile["standard_fingerprints"], {
            "combat": GYM_8726D467_CONTRACT.standard_combat_fingerprint,
            "run": GYM_8726D467_CONTRACT.standard_run_fingerprint,
        })

    def test_factory_and_runner_rules_ids_follow_selected_contract(self):
        for mode in ("standalone", "run"):
            for directory, expected in (("gym-v1.0.0", "spire-gym-native-m10-rules-v1"),
                                        ("gym-8726d467", "spire-gym-native-m10-rules-v2")):
                config = load_neural_config(ROOT / "configs/neural" / directory / f"{mode}.toml")
                self.assertEqual(runner._ENVIRONMENT_IDENTITIES[mode](config)["rules_id"], expected)
        for mode in ("standard-standalone", "standard-run"):
            for directory, expected in (("gym-v1.0.0", "spire-gym-native-standard-rules-v3"),
                                        ("gym-8726d467", "spire-gym-native-standard-rules-v5")):
                config = load_neural_config(ROOT / "configs/neural" / directory / f"{mode}.toml")
                self.assertEqual(runner._ENVIRONMENT_IDENTITIES[mode](config)["rules_id"], expected)

    def test_checkpoint_mappings_reject_cross_release_in_both_directions(self):
        old = load_neural_config(ROOT / "configs/neural/gym-v1.0.0/standalone.toml").accepted_identities[0]
        new = load_neural_config(ROOT / "configs/neural/gym-8726d467/standalone.toml").accepted_identities[0]
        with patch("builtins.__import__", side_effect=_refuse_torch):
            for expected, actual in ((old, new), (new, old)):
                with self.assertRaisesRegex(ValueError, "raw-mode mapping"):
                    validate_checkpoint_metadata(_metadata(identity=expected), actual, purpose="evaluate")

    def test_qualification_and_soak_default_profile_directory(self):
        from scripts.neural import qualify_gym_v1, soak_gym_v1
        with patch.object(qualify_gym_v1, "run") as run:
            qualify_gym_v1.main(["--output", "out"])
            self.assertEqual(run.call_args.kwargs["profile_dir"], Path("configs/neural/gym-8726d467"))
        with patch.object(soak_gym_v1, "run") as run:
            soak_gym_v1.main(["--steps", "1", "--output", "out"])
            self.assertEqual(run.call_args.kwargs["profile_dir"], Path("configs/neural/gym-8726d467"))


_real_import = __import__


def _refuse_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("checkpoint compatibility must be decided before torch import")
    return _real_import(name, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
