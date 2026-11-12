"""Exact Gym release pairing and fail-closed checkpoint migration tests."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import copy

from spire_learn.contracts import ContractError
from spire_learn.neural import config, envs, schema
from spire_learn.neural.checkpoint import (
    CheckpointCompatibilityError, validate_checkpoint_metadata,
)
from spire_learn.neural.contracts import CombatFrame
from tests.unit.test_neural_checkpoint import _metadata


REVISION = "afb51f5f222b4c4331ed34067c5eebeda606db91"
COMMON = "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463"
RUN = "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64"


def _config(mode, *, replacements=()):
    source = Path(f"configs/neural/{mode}.toml").read_text()
    source = source.replace("../cases/neural/train.json", str(
        Path("configs/cases/neural/train.json").resolve()))
    source = source.replace(f"../../artifacts/development/neural-combat/runs/{mode}", str(
        Path(f"artifacts/development/neural-combat/runs/gym-migration-test/{mode}").resolve()))
    for before, after in (
        (config.ACCEPTED_GYM_REVISION, REVISION),
        (config.COMMON_VIEW_FINGERPRINT, COMMON),
        (config.RUN_OBSERVATION_FINGERPRINT, RUN),
        *replacements,
    ):
        source = source.replace(before, after)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.toml"
        path.write_text(source)
        return config.load_neural_config(path)


class GymMigrationTests(unittest.TestCase):
    def test_real_descriptor_capacity_features_and_body_authentication(self):
        from spire_learn.neural.features import feature_contract
        from tests.neural._support import authored_frame
        raw = json.loads(Path("tests/fixtures/neural/common-afb51f5.json").read_text())
        accepted = schema.parse_common_descriptor(raw, revision=REVISION)
        self.assertEqual(accepted.gym_revision, REVISION)
        self.assertEqual(accepted.tensors["combat_cards"].shape[0], 1024)
        self.assertEqual(accepted.tensors["known_draw"].shape[0], 1024)
        self.assertEqual(len(feature_contract(accepted).groups), 12)
        frame = authored_frame(accepted)
        schema.validate_frame(frame, accepted)
        self.assertTrue(schema.candidates(frame, accepted))
        changed = copy.deepcopy(raw)
        changed["vocabularies"]["card"][1:3] = reversed(changed["vocabularies"]["card"][1:3])
        with self.assertRaises(ContractError):
            schema.parse_common_descriptor(changed)

    def test_real_new_schema_admits_train_rollouts_in_both_modes_without_torch(self):
        from spire_learn.neural.ppo import _validate_admission
        from spire_learn.neural.contracts import (
            CumulativeCounters, PolicyChoice, Rollout, StepRecord, TaskSignal,
        )
        from tests.neural._support import authored_frame
        accepted = schema.parse_common_descriptor(json.loads(
            Path("tests/fixtures/neural/common-afb51f5.json").read_text()), revision=REVISION)
        frame = authored_frame(accepted)
        chosen = schema.candidates(frame, accepted)[0].index
        record = StepRecord("stream", "task", "decision", frame, frame, 0., False, False,
                            TaskSignal(0., False, True, False),
                            PolicyChoice(chosen, -1., 0., 0), True, 0.)
        for mode in ("standalone", "run"):
            current = _config(mode)
            identity = current.accepted_identities[0]
            policy = SimpleNamespace(model_id=identity.model_id, features_id=identity.features_id,
                action_id=identity.action_id, objective_id=identity.objective_id,
                _model_config_hash=identity.config_hashes["model"], _schema=accepted,
                _behavior_policy_version=0)
            rollout = Rollout(mode, "train", identity, (record,), CumulativeCounters(1, 1, 0, 0, 0, 0))
            records, targets = _validate_admission(policy, rollout, current.ppo)
            self.assertEqual(records, (record,))
            self.assertEqual(targets.train_mask, (True,))
            historical = config.load_neural_config(Path(f"configs/neural/{mode}.toml"))
            with self.assertRaisesRegex(ContractError, "raw-mode mapping"):
                _validate_admission(policy, replace(rollout, identity=historical.accepted_identities[0]), current.ppo)

    def test_registry_binds_exact_revision_and_immutable_mapping_ids(self):
        try:
            from spire_learn.neural.gym_contracts import (
                contract_for_revision, contracts_for_fingerprint,
            )
        except ImportError as error:
            self.fail(f"release-contract API missing: {error}")
        current = contract_for_revision(REVISION)
        self.assertEqual(current.common_fingerprint, COMMON)
        self.assertEqual(current.run_fingerprint, RUN)
        self.assertEqual(current.raw_mode_mappings["standalone"],
                         f"spire-learn-standalone-common-v1@{COMMON}")
        self.assertEqual(current.raw_mode_mappings["run"],
                         f"spire-learn-run-policy-v3-common-v1@{RUN}:{COMMON}")
        self.assertIn(current, contracts_for_fingerprint(COMMON))
        with self.assertRaises(TypeError):
            current.raw_mode_mappings["run"] = "replacement"
        for revision in ("0" * 40, True, None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                contract_for_revision(revision)
        for fingerprint in ("0" * 64, True, None):
            with self.subTest(fingerprint=fingerprint):
                self.assertEqual(contracts_for_fingerprint(fingerprint), ())

    def test_both_modes_admit_new_release_without_changing_legacy_identity(self):
        for mode in ("standalone", "run"):
            with self.subTest(mode=mode):
                try:
                    current = _config(mode)
                except ValueError as error:
                    self.fail(f"new exact Gym contract must load: {error}")
                historical = config.load_neural_config(Path(f"configs/neural/{mode}.toml"))
                self.assertEqual(current.environment_revision, REVISION)
                self.assertEqual(current.common_view_fingerprint, COMMON)
                self.assertNotEqual(current.accepted_identities, historical.accepted_identities)
                self.assertEqual(historical.environment_revision, config.ACCEPTED_GYM_REVISION)
                envs._validate_factory_inputs(current, current.cases)

    def test_revision_descriptor_and_mapping_cannot_be_crossed(self):
        substitutions = (
            ((COMMON, config.COMMON_VIEW_FINGERPRINT),),
            ((REVISION, config.ACCEPTED_GYM_REVISION),),
            ((RUN, config.RUN_OBSERVATION_FINGERPRINT),),
            ((f"@{COMMON}", f"@{config.COMMON_VIEW_FINGERPRINT}"),),
        )
        for replacements in substitutions:
            with self.subTest(replacements=replacements), self.assertRaises(ValueError):
                _config("run", replacements=replacements)

    def test_factory_rejects_tampered_pair_before_runtime_import(self):
        try:
            current = _config("standalone")
        except ValueError as error:
            self.fail(f"new exact Gym contract must load: {error}")
        with patch.object(envs, "_load_native_runtime", side_effect=AssertionError("optional import")):
            for changes in (
                {"common_view_fingerprint": config.COMMON_VIEW_FINGERPRINT},
                {"environment_revision": config.ACCEPTED_GYM_REVISION},
            ):
                with self.subTest(changes=changes), self.assertRaises(ContractError):
                    envs.make_driver(replace(current, **changes), current.cases)

    def test_new_frame_fingerprint_is_explicit_and_unknown_hash_stays_rejected(self):
        try:
            self.assertEqual(CombatFrame({}, COMMON).schema_fingerprint, COMMON)
        except ContractError as error:
            self.fail(f"new common-view frame must be admitted: {error}")
        with self.assertRaises(ContractError):
            CombatFrame({}, "e" * 64)

    def test_factory_rejects_empty_or_cross_mode_identity_before_optional_import(self):
        current = _config("standalone")
        identity = current.accepted_identities[0]
        crossed = replace(identity, accepted_raw_mode_mappings={
            **dict(identity.accepted_raw_mode_mappings), "run": config.RUN_RAW_MAPPING_ID,
        })
        with patch.object(envs, "_load_native_runtime", side_effect=AssertionError("optional import")):
            for identities in ((), (crossed,)):
                with self.subTest(identities=identities), self.assertRaises(ContractError):
                    envs.make_driver(replace(current, accepted_identities=identities), current.cases)

    def test_new_selection_owners_and_public_phases_survive_lifecycle_parsing(self):
        for owner in ("card", "potion", "end_turn", "relic", "power"):
            with self.subTest(owner=owner):
                info = dict(combat_view_id="spire-gym-combat-policy-view-v1",
                            active=True, routing="combat_selection", selection_owner=owner,
                            phase="selection", selection_resume_phase="enemy_turn",
                            active_visit=0, combats_entered=1, combats_settled=0, last_exit=None,
                            effective_seed=99, diagnostic={"private": "ignored"})
                try:
                    lifecycle = schema.parse_lifecycle(info)
                except ContractError as error:
                    self.fail(f"public selection owner must be admitted: {error}")
                self.assertEqual(getattr(lifecycle, "phase", None), "selection")
                self.assertEqual(getattr(lifecycle, "selection_resume_phase", None), "enemy_turn")
                self.assertFalse(hasattr(lifecycle, "effective_seed"))
                self.assertFalse(hasattr(lifecycle, "diagnostic"))
                for field in ("phase", "selection_resume_phase"):
                    with self.subTest(field=field), self.assertRaises(ContractError):
                        schema.parse_lifecycle(dict(info, **{field: "unknown_phase"}))

    def test_old_checkpoints_reject_all_new_purposes_before_tensor_loading(self):
        historical = config.load_neural_config(Path("configs/neural/standalone.toml"))
        saved = _metadata(identity=historical.accepted_identities[0], training=True)
        try:
            expected = _config("standalone").accepted_identities[0]
        except ValueError as error:
            self.fail(f"new exact Gym contract must load: {error}")
        for purpose in ("evaluate", "warm_start", "resume"):
            with self.subTest(purpose=purpose), self.assertRaisesRegex(
                CheckpointCompatibilityError, "raw-mode mapping"
            ):
                validate_checkpoint_metadata(saved, expected, purpose=purpose)


if __name__ == "__main__":
    unittest.main()
