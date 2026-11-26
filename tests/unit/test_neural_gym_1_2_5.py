"""Offline contract and profile checks for Spire Gym 1.2.5, a drop-in in the gym-1.2.4 rules epoch."""

import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.neural import baseline_entry
from scripts.neural import boss_winnability as bw
from scripts.neural import search_teacher_entry as entry
from scripts.neural import teacher_audit as audit
from spire_learn.neural import recipes
from spire_learn.neural import standard_stream as ss
from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.config import ACCEPTED_RUNTIME_GYM_REVISIONS, load_neural_config
from spire_learn.neural.scenario_support import canonical_bytes
from spire_learn.neural.gym_contracts import (
    CURRENT_CONTRACT, GYM_V1_2_0_CONTRACT, GYM_V1_2_4_CONTRACT, GYM_V1_2_5_CONTRACT, LEGACY_CONTRACT,
    contract_for_revision, contract_for_run_states, same_rules_epoch, supported_contracts,
)
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_gym_1_1_1 import _refuse_torch
from tests.unit.test_neural_standard_stream import GYM_1_2_0, GYM_1_2_4, recipe_document, write_table
from tests.unit.test_teacher_audit import Fixture, run_main


ROOT = Path(__file__).resolve().parents[2]
REVISION = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"
SOURCE_DIGEST = "19854fcccf6bd09ceef036297d4a6ecf9d068d9bbc37968a38de07482f2ce117"
FILE_DIGEST = "63533371d420314d41b6307169156e5fe715c41920052f4117d225cf758d6a63"
WHEEL_SHA256 = "02346039385b2135f5e7809b3af98315dca89e07cfd855de59eedfeeaad177cf"
NATIVE_SHA256 = "67819e119354e3a48e42102d66260055bcde7bed0f22b9e1eb7a84059580cdf2"
# SHA-256 of the packet's DELIVERY.md, as for every runtime profile since 8726d467.
DELIVERY_SHA256 = "c973774e2bc7af2399fe2332c48a5610c5160ad333c6037ae3d78903ccc8836b"
COMPACT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
MODES = ("standalone", "run", "standard-standalone", "standard-run")
PROFILE_DIR = ROOT / "configs/neural/gym-1.2.5"
BUILD = {"dirty": False, "extension": "spire-python", "features": ["spire-python/extension"],
         "file_digest": FILE_DIGEST, "profile": "release", "revision": REVISION, "rust_version": "1.99.0",
         "source_digest": SOURCE_DIGEST}


def _table_recipe(folder: Path, revision: str):
    """A synthetic table recorded under ``revision`` (a v1.2.5 one carries the entry-resource pin too)."""
    folder.mkdir()
    table, sha = write_table(folder, revision=GYM_1_2_4 if revision == REVISION else revision)
    if revision == REVISION:
        document = json.loads(table.read_bytes())
        document["gym_revision"] = REVISION
        data = canonical_bytes(document)
        table.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
    path = folder / "recipe.json"
    path.write_text(json.dumps(recipe_document(sha)))
    return ss.load_standard_recipe(path)


def run_entry(directory, *, recipe_revision, build):
    """``_run`` of the committed ``current`` teacher on one fake selection fight."""
    recipe = Mock(spec=entry.StandardRecipe, gym_revision=recipe_revision, table_sha256="table",
                  character="ironclad", ascension=20)
    gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: dict(build)))
    panel = {"manifest": {"cases": [{"case_id": "c", "entry": {"id": "c"}, "environment_seed": 1}]},
             "panel_identity": "panel", "panel_sha256": "p", "manifest_sha256": "m"}
    descriptor = {"profile_id": "profile", "schema_id": "schema", "fingerprint": "fingerprint"}
    fight = {"case_id": "c", "outcome": "victory", "cpu_seconds": 0.0}
    recipe_path = Path(directory) / "recipe.json"
    recipe_path.write_text("{}")
    args = SimpleNamespace(teachers=ROOT / "configs/neural/search-teacher-gym-1.2.4/teachers.json",
                           teacher="current", panel="selection", recipe=recipe_path, train_sample=None, seed=1,
                           limit=None, max_steps=1024, max_turn_actions=500, parallel=1)
    output = Path(directory) / f"output-{build['revision'][:7]}"
    with (patch.dict(sys.modules, {"spire_lab": gym}),
          patch.object(entry, "load_recipe", return_value=recipe),
          patch.object(entry, "build_panel", return_value=panel),
          patch.object(entry, "preflight", return_value=descriptor),
          patch.object(entry, "_results", return_value=[(fight, [])])):
        return entry._run(args, output)


class GymV125Tests(unittest.TestCase):
    def test_v1_2_5_contract_identities(self):
        contract = GYM_V1_2_5_CONTRACT
        self.assertIs(contract_for_revision(REVISION), contract)
        self.assertIn(REVISION, ACCEPTED_RUNTIME_GYM_REVISIONS)
        self.assertEqual((contract.source_revision, contract.runtime_revisions), (REVISION, (REVISION,)))
        self.assertEqual(contract.build.revision, REVISION)
        self.assertEqual(contract.build.source_digest, SOURCE_DIGEST)
        self.assertEqual(contract.build.file_digest, FILE_DIGEST)
        self.assertEqual((contract.build.profile, contract.build.features), ("release", ("spire-python/extension",)))
        self.assertEqual(contract_for_run_states(REVISION, "spire-learn-run-states-v1"), contract)
        self.assertEqual(len(contract.standard_stream_smoke_tests), 3)
        self.assertIn("test_standard_stream_smoke_gym_1_2_5", contract.standard_stream_smoke_tests[0])
        profile = json.loads((PROFILE_DIR / "runtime.json").read_text())
        old = json.loads((ROOT / "configs/neural/gym-1.2.4/runtime.json").read_text())
        self.assertEqual(set(profile), set(old))
        self.assertEqual(profile["gym_revision"], REVISION)
        self.assertEqual(profile["native_sha256"], NATIVE_SHA256)
        self.assertEqual(profile["wheel_sha256"], WHEEL_SHA256)
        self.assertEqual(profile["wheel_path"], "artifacts/development/neural-combat/gym-1.2.5/inputs/"
                                                "spire_lab-1.2.5-cp312-cp312-manylinux_2_34_x86_64.whl")
        self.assertEqual(profile["package_overlay"], "artifacts/development/neural-combat/gym-1.2.5/site")
        self.assertEqual(profile["delivery_manifest_sha256"], DELIVERY_SHA256)
        self.assertEqual(profile["build_identity"], BUILD)
        # Everything else (dependency venv and lock, fingerprints, policies) is the v1.2.4 profile's.
        changed = {"gym_revision", "native_sha256", "wheel_sha256", "wheel_path", "package_overlay",
                   "delivery_manifest_sha256", "build_identity"}
        for key in set(old) - changed:
            self.assertEqual(profile[key], old[key], key)

    def test_v1_2_5_shares_epoch_and_fingerprints_with_v1_2_4(self):
        new, old = GYM_V1_2_5_CONTRACT, GYM_V1_2_4_CONTRACT
        self.assertEqual(new.rules_epoch, "gym-1.2.4")
        for field in dataclasses.fields(new):
            if field.name not in ("source_revision", "runtime_revisions", "build", "standard_stream_smoke_tests",
                                  "tiered_playouts"):
                self.assertEqual(getattr(new, field.name), getattr(old, field.name), field.name)
        self.assertEqual(new.standard_rules_id, "spire-gym-native-standard-rules-v9")
        self.assertEqual(new.m10_rules_id, "spire-gym-native-m10-rules-v6")
        self.assertEqual(dict(new.raw_mode_mappings), dict(old.raw_mode_mappings))
        self.assertEqual(dict(new.raw_mode_mappings_for_view(COMPACT)), dict(old.raw_mode_mappings_for_view(COMPACT)))
        self.assertNotEqual(new.build, old.build)

    def test_same_rules_epoch(self):
        v120, v124 = GYM_V1_2_0_CONTRACT.source_revision, GYM_V1_2_4_CONTRACT.source_revision
        for a, b in ((v124, REVISION), (REVISION, v124), (REVISION, REVISION), (v124, v124), (v120, v120)):
            self.assertTrue(same_rules_epoch(a, b), (a, b))
        for a, b in ((v120, REVISION), (REVISION, v120), (v120, v124), (REVISION, "d" * 40), ("d" * 40, "d" * 40),
                     (REVISION, None), (None, None)):
            self.assertFalse(same_rules_epoch(a, b), (a, b))
        # Contracts without an epoch share none, not even with themselves.
        for contract in (LEGACY_CONTRACT, CURRENT_CONTRACT):
            self.assertIsNone(contract.rules_epoch)
            self.assertFalse(same_rules_epoch(contract.source_revision, contract.source_revision))

    def test_gym_1_2_4_table_and_recipe_accepted_on_v1_2_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            table_124 = _table_recipe(Path(tmp) / "124", GYM_1_2_4)
            table_125 = _table_recipe(Path(tmp) / "125", REVISION)
            table_120 = _table_recipe(Path(tmp) / "120", GYM_1_2_0)
        for recipe in (table_124, table_125):
            for config in (GYM_1_2_4, REVISION):
                recipes.require_recipe_revision(recipe, config)
                with patch.object(recipes, "_VALIDATOR", SimpleNamespace(live_gym_revision=config)):
                    self.assertIsNotNone(recipes.gym_validator("test", recipe))
        for recipe, config in ((table_120, REVISION), (table_124, GYM_1_2_0), (table_125, GYM_1_2_0)):
            with self.subTest(table=recipe.gym_revision, config=config):
                with self.assertRaisesRegex(ValueError, "run-states table is for Gym"):
                    recipes.require_recipe_revision(recipe, config)
                with patch.object(recipes, "_VALIDATOR", SimpleNamespace(live_gym_revision=config)), \
                        self.assertRaisesRegex(ValueError, "run it in that Gym's runtime"):
                    recipes.gym_validator("test", recipe)
        # The committed gym-1.2.4 recipe (its table is local data, so this part may skip).
        committed = ROOT / "configs/neural/baalorlord-standard-gym-1.2.4/recipe.json"
        table = ROOT / "artifacts/development/neural-combat/run-states/run-states-gym-1.2.4.json"
        if table.is_file():
            recipe = ss.load_standard_recipe(committed)
            self.assertEqual(recipe.gym_revision, GYM_1_2_4)
            recipes.require_recipe_revision(recipe, REVISION)

    def test_gym_1_2_5_profiles_load(self):
        for name in (*MODES, "standard-stream"):
            config = load_neural_config(PROFILE_DIR / f"{name}.toml")
            old = load_neural_config(ROOT / "configs/neural/gym-1.2.4" / f"{name}.toml")
            self.assertEqual(config.environment_revision, REVISION)
            self.assertEqual(config.accepted_identities, old.accepted_identities, name)
            self.assertEqual((config.mode, config.standard_view_layout, config.standard_view_preset,
                              config.standard_view_fingerprint),
                             (old.mode, old.standard_view_layout, old.standard_view_preset,
                              old.standard_view_fingerprint))
            self.assertIn("gym-1.2.5", str(config.output_root))
        for name in (*MODES, "standard-stream"):
            text = (PROFILE_DIR / f"{name}.toml").read_text()
            old = (ROOT / "configs/neural/gym-1.2.4" / f"{name}.toml").read_text()
            self.assertEqual(text, old.replace(GYM_1_2_4, REVISION).replace(
                "runs/gym-1.2.4/", "runs/gym-1.2.5/"), name)

    def test_checkpoints_cross_v1_2_4_and_v1_2_5(self):
        for mode in MODES:
            identities = {
                name: load_neural_config(ROOT / "configs/neural" / name / f"{mode}.toml").accepted_identities[0]
                for name in ("gym-1.2.0", "gym-1.2.4", "gym-1.2.5")}
            with patch("builtins.__import__", side_effect=_refuse_torch):
                for saved, expected in (("gym-1.2.4", "gym-1.2.5"), ("gym-1.2.5", "gym-1.2.4")):
                    with self.subTest(mode=mode, saved=saved):
                        validate_checkpoint_metadata(
                            _metadata(identity=identities[saved]), identities[expected], purpose="evaluate")
                for saved, expected in (("gym-1.2.0", "gym-1.2.5"), ("gym-1.2.5", "gym-1.2.0")):
                    with self.subTest(mode=mode, saved=saved), self.assertRaisesRegex(ValueError, "raw-mode mapping"):
                        validate_checkpoint_metadata(
                            _metadata(identity=identities[saved]), identities[expected], purpose="evaluate")

    def test_other_epochs_still_refused(self):
        v120 = GYM_V1_2_0_CONTRACT.source_revision
        # The teacher entry: a gym-1.2.0 recipe on the v1.2.5 Gym, and a gym-1.2.4 recipe on v1.2.0.
        for recipe_revision, live in ((v120, REVISION), (GYM_1_2_4, v120)):
            with self.subTest(recipe=recipe_revision, live=live), tempfile.TemporaryDirectory() as tmp, \
                    self.assertRaisesRegex(ValueError, "Gym revision"):
                run_entry(tmp, recipe_revision=recipe_revision, build={**BUILD, "revision": live})
        # The boss and descent-cost entries' live check.
        bw._check_live_revision(GYM_1_2_4, build_info=lambda: {"revision": REVISION})
        bw._check_live_revision(REVISION, build_info=lambda: {"revision": GYM_1_2_4})
        for table, live in ((v120, REVISION), (REVISION, v120), (GYM_1_2_4, "d" * 40)):
            with self.assertRaisesRegex(ValueError, "run-states table"):
                bw._check_live_revision(table, build_info=lambda: {"revision": live})
        # The baseline panel recipe against the plan's revision.
        with tempfile.TemporaryDirectory() as tmp:
            for recipe_revision, accepted in ((GYM_1_2_4, True), (v120, False)):
                stop = RuntimeError("past the gate")
                deps = {"load_recipe": lambda path: SimpleNamespace(gym_revision=recipe_revision),
                        "build_panel": Mock(side_effect=stop)}
                with self.subTest(recipe=recipe_revision), self.assertRaisesRegex(
                        RuntimeError if accepted else ValueError, "past the gate" if accepted else "is for Gym"):
                    baseline_entry._evaluate({}, deps, {"gym_revision": REVISION}, None, Path("recipe.json"),
                                             Path(tmp) / recipe_revision[:7], None, None, None, None, None,
                                             labels=[], evaluation={})
        # Audit runs from two epochs.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            for teacher in audit.TEACHERS:
                self._set_revision(fixture, teacher, REVISION, BUILD)
            self._set_revision(fixture, "current", v120, epoch="gym-1.2.0")
            with self.assertRaisesRegex(audit.AuditInputError, "rules_epoch"):
                run_main(fixture.root, fixture.argv())

    def test_tiered_playouts_flag_only_on_v1_2_5(self):
        self.assertTrue(GYM_V1_2_5_CONTRACT.tiered_playouts)
        for contract in supported_contracts():
            self.assertIs(contract.tiered_playouts, contract is GYM_V1_2_5_CONTRACT, contract.source_revision)

    def test_entry_records_live_revision(self):
        old_build = {**BUILD, "revision": GYM_1_2_4, "source_digest": GYM_V1_2_4_CONTRACT.build.source_digest,
                     "file_digest": GYM_V1_2_4_CONTRACT.build.file_digest}
        with tempfile.TemporaryDirectory() as tmp:
            new = run_entry(tmp, recipe_revision=GYM_1_2_4, build=BUILD)["provenance"]
            old = run_entry(tmp, recipe_revision=GYM_1_2_4, build=old_build)["provenance"]
        self.assertEqual((new["gym_revision"], new["gym_build"]), (REVISION, BUILD))
        self.assertEqual((old["gym_revision"], old["gym_build"]), (GYM_1_2_4, old_build))
        for provenance in (new, old):
            self.assertEqual(provenance["rules_epoch"], "gym-1.2.4")
            self.assertEqual(provenance["states_table_sha256"], "table")
        self.assertEqual({k: v for k, v in new.items() if k not in ("gym_revision", "gym_build")},
                         {k: v for k, v in old.items() if k not in ("gym_revision", "gym_build")})

    @staticmethod
    def _set_revision(fixture, teacher, revision, build=None, epoch=None):
        for (name, _), path in fixture.runs.items():
            if name == teacher:
                summary = json.loads((path / "summary.json").read_text())
                summary["provenance"]["gym_revision"] = revision
                if epoch is not None:
                    summary["provenance"]["rules_epoch"] = epoch
                summary["provenance"]["gym_build"] = build or {"revision": revision, "dirty": False}
                (path / "summary.json").write_text(json.dumps(summary))

    def test_audit_accepts_mixed_revisions_in_one_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            for teacher in audit.TEACHERS:
                self._set_revision(fixture, teacher, GYM_1_2_4)
            self._set_revision(fixture, "tree-3x", REVISION, BUILD)
            run_main(fixture.root, fixture.argv())
            report = json.loads((fixture.root / "artifacts/report.json").read_text())
            provenance = report["provenance"]
            self.assertEqual(provenance["rules_epoch"], "gym-1.2.4")
            self.assertNotIn("gym_revision", provenance)
            self.assertNotIn("gym_build", provenance)
            for key, run in provenance["runs"].items():
                expected = REVISION if key.startswith("tree-3x:") else GYM_1_2_4
                self.assertEqual(run["gym_revision"], expected, key)
                self.assertEqual(run["gym_build"]["revision"], expected, key)
            self.assertEqual(provenance["runs"]["tree-3x:boss"]["gym_build"], BUILD)
            # One revision with two different builds is still refused.
            self._set_revision(fixture, "tree-1x", REVISION, {**BUILD, "dirty": True})
            with self.assertRaisesRegex(audit.AuditInputError, "gym_build"):
                run_main(fixture.root, fixture.argv("artifacts/report-2.json"))
            # So is a run whose recorded epoch is not its revision's.
            self._set_revision(fixture, "tree-1x", GYM_V1_2_0_CONTRACT.source_revision)
            with self.assertRaisesRegex(audit.AuditInputError, "gym_revision"):
                run_main(fixture.root, fixture.argv("artifacts/report-3.json"))


if __name__ == "__main__":
    unittest.main()
