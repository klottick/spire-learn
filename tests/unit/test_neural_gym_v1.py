"""Gym 1.0.0 release contract, build identity and checkpoint separation."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import envs
from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.gym_contracts import (
    CURRENT_CONTRACT, GYM_V1_CONTRACT, BuildIdentity, contract_for_revision,
    contract_for_schema, contracts_for_fingerprint,
)
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_gym_migration import _config


V1 = "1929e18b9f44372e020877bd62f2a5c9dee3e64e"
COMMON = "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463"
RUN = "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64"
STANDARD_COMBAT = "5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e"
STANDARD_RUN = "bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae"
PUBLISHED_BUILD = {
    "dirty": False,
    "extension": "spire-python",
    "features": ["spire-python/extension"],
    "file_digest": "dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce",
    "profile": "release",
    "revision": V1,
    "rust_version": "1.99.0",
    "source_digest": "d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2",
}


class GymV1ContractTests(unittest.TestCase):
    def test_v1_contract_identities(self):
        contract = contract_for_revision(V1)
        self.assertIs(contract, GYM_V1_CONTRACT)
        self.assertEqual(contract.source_revision, V1)
        self.assertEqual(contract.runtime_revisions, (V1,))
        self.assertEqual(contract.common_fingerprint, COMMON)
        self.assertEqual(contract.run_fingerprint, RUN)
        self.assertEqual(contract.card_metadata_fingerprint,
                         "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406")
        self.assertEqual(contract.standard_combat_fingerprint, STANDARD_COMBAT)
        self.assertEqual(contract.standard_run_fingerprint, STANDARD_RUN)
        self.assertEqual(contract.build, BuildIdentity(
            V1, PUBLISHED_BUILD["source_digest"], PUBLISHED_BUILD["file_digest"],
            "release", ("spire-python/extension",)))

    def test_v1_mappings_carry_rules_epoch(self):
        mappings = GYM_V1_CONTRACT.raw_mode_mappings
        self.assertEqual(set(mappings), {"standalone", "run", "standard-standalone", "standard-run"})
        self.assertEqual(mappings["standalone"],
                         f"spire-learn-standalone-common-v1@{COMMON}+gym-1.0.0")
        self.assertEqual(mappings["run"],
                         f"spire-learn-run-policy-v3-common-v1@{RUN}:{COMMON}+gym-1.0.0")
        self.assertEqual(mappings["standard-standalone"],
                         f"spire-learn-standard-combat-v1@{STANDARD_COMBAT}+gym-1.0.0")
        self.assertEqual(mappings["standard-run"],
                         f"spire-learn-standard-run-v1@{STANDARD_RUN}:{STANDARD_COMBAT}+gym-1.0.0")

    def test_historical_mappings_unchanged(self):
        self.assertEqual(dict(CURRENT_CONTRACT.raw_mode_mappings), {
            "standalone": f"spire-learn-standalone-common-v1@{COMMON}",
            "run": f"spire-learn-run-policy-v3-common-v1@{RUN}:{COMMON}",
        })

    def test_shared_fingerprint_is_ambiguous(self):
        self.assertEqual(contracts_for_fingerprint(COMMON), (CURRENT_CONTRACT, GYM_V1_CONTRACT))
        with self.assertRaisesRegex(ValueError, "revision is required"):
            contract_for_schema(COMMON, None)
        self.assertIs(contract_for_schema(COMMON, V1), GYM_V1_CONTRACT)
        self.assertIs(contract_for_schema(COMMON, CURRENT_CONTRACT.source_revision), CURRENT_CONTRACT)
        with self.assertRaises(ValueError):
            contract_for_schema("e" * 64, V1)

    def test_verify_native_build_full_identity(self):
        envs._verify_native_build(SimpleNamespace(build_info=lambda: dict(PUBLISHED_BUILD)), V1)
        for key, value in (
            ("dirty", True), ("dirty", None), ("profile", "debug"),
            ("source_digest", "0" * 64), ("file_digest", "0" * 64),
            ("features", ["spire-python/extension", "extra"]), ("features", None),
            ("revision", CURRENT_CONTRACT.source_revision), ("extension", "other"),
        ):
            changed = dict(PUBLISHED_BUILD, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                envs._verify_native_build(SimpleNamespace(build_info=lambda c=changed: c), V1)
        missing = {k: v for k, v in PUBLISHED_BUILD.items() if k != "source_digest"}
        with self.assertRaises(ContractError):
            envs._verify_native_build(SimpleNamespace(build_info=lambda: missing), V1)

    def test_v1_checkpoint_rejects_afb51f5_mapping(self):
        old = _config("standalone").accepted_identities[0]
        # NeuralIdentity admits the Standard modes only once their config modes exist.
        new = replace(old, accepted_raw_mode_mappings={
            mode: GYM_V1_CONTRACT.raw_mode_mappings[mode] for mode in ("standalone", "run")})
        self.assertEqual(old.accepted_raw_mode_mappings["standalone"].split("@")[1].split("+")[0],
                         new.accepted_raw_mode_mappings["standalone"].split("@")[1].split("+")[0])
        with patch("builtins.__import__", side_effect=_refuse_torch):
            for purpose in ("evaluate", "warm_start", "resume"):
                with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                    validate_checkpoint_metadata(
                        _metadata(identity=old, training=purpose == "resume"), new, purpose=purpose)


_real_import = __import__


def _refuse_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("checkpoint compatibility must be decided before torch import")
    return _real_import(name, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
