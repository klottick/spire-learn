"""Every revision gate outside the standard stream derives from the Gym contract registry."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.neural import boss_winnability as bw
from scripts.neural import search_teacher_entry as teacher_entry
from scripts.neural import teacher_descent_cost_entry as descent
from spire_learn.contracts import ContractError
from spire_learn.neural import envs, gym_contracts
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.views import STANDARD_COMBAT_COMPACT_V1, STANDARD_COMBAT_V1, view_for_fingerprint

ROOT = Path(__file__).resolve().parents[2]
FAKE = "c" * 40
UNKNOWN = "d" * 40
REAL = gym_contracts.GYM_V1_2_4_CONTRACT
FAKE_CONTRACT = dataclasses.replace(
    REAL, source_revision=FAKE, runtime_revisions=(FAKE,),
    standard_combat_fingerprint="2" * 64,
    standard_compact_combat_fingerprint="3" * 64, standard_compact_combat_play_fingerprint="4" * 64,
)


def registered():
    return patch.dict(gym_contracts.CONTRACTS_BY_REVISION, {FAKE: FAKE_CONTRACT})


class RegistrySitesTest(unittest.TestCase):
    def test_config_and_driver_factory_gates(self):
        source = (ROOT / "configs/neural/gym-1.2.4/standalone.toml").read_text()
        source = source.replace("../../cases/neural/train.json", str(ROOT / "configs/cases/neural/train.json"))
        source = source.replace("../../../artifacts", str(ROOT / "artifacts"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(source.replace(REAL.source_revision, FAKE))
            with self.assertRaisesRegex(ValueError, "unsupported Gym revision"):
                load_neural_config(path)
            with registered():
                config = load_neural_config(path)
                self.assertEqual(config.environment_revision, FAKE)
                envs._validate_factory_inputs(config, config.cases)
                bad = dataclasses.replace(config, environment_revision=UNKNOWN)
                with self.assertRaisesRegex(ContractError, "accepted Gym revision"):
                    envs._validate_factory_inputs(bad, config.cases)
            path.write_text(source.replace(REAL.source_revision, UNKNOWN))
            with registered(), self.assertRaisesRegex(ValueError, "unsupported Gym revision"):
                load_neural_config(path)

    def test_view_for_fingerprint_follows_registry(self):
        for fingerprint, view in (("2" * 64, STANDARD_COMBAT_V1),
                                  ("3" * 64, STANDARD_COMBAT_COMPACT_V1)):
            with self.assertRaises(ContractError):
                view_for_fingerprint(fingerprint)
            with registered():
                self.assertIs(view_for_fingerprint(fingerprint), view)
            with self.assertRaises(ContractError):
                view_for_fingerprint(fingerprint)

    def test_search_teacher_entry_gate(self):
        def run(recipe_revision, live=FAKE):
            recipe = Mock(spec=teacher_entry.StandardRecipe, gym_revision=recipe_revision)
            gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: {"revision": live}))
            stop = RuntimeError("past the gate")
            with patch.dict("sys.modules", {"spire_lab": gym}), \
                 patch.object(teacher_entry, "_teacher", return_value=(None, None, True)), \
                 patch.object(teacher_entry, "load_recipe", return_value=recipe), \
                 patch.object(teacher_entry, "build_panel", side_effect=stop):
                teacher_entry._run(SimpleNamespace(teachers="t", teacher="current", panel="boss", recipe=Path("r")),
                                   Path("o"))
        with registered():
            with self.assertRaisesRegex(RuntimeError, "past the gate"):
                run(FAKE)
            with self.assertRaisesRegex(ValueError, "supported Gym revision"):
                run(UNKNOWN, live=UNKNOWN)
            with self.assertRaisesRegex(ValueError, "Gym revision"):
                run(FAKE, live=REAL.source_revision)
        with self.assertRaisesRegex(ValueError, "supported Gym revision"):
            run(FAKE)

    def test_boss_provenance_uses_recipe_contract(self):
        descriptor = {"fingerprint": "f" * 64, "profile_id": "p"}
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "recipe.json"
            recipe.write_text(json.dumps({"states_table": {"sha256": "s"}}))
            gym = SimpleNamespace(_native=SimpleNamespace(build_info=lambda: {"revision": FAKE}))
            with patch.dict("sys.modules", {"spire_lab": gym}):
                for revision in (FAKE, UNKNOWN):
                    with patch.object(bw, "_table_states", return_value=(revision, [])), registered():
                        if revision == FAKE:
                            result = bw._search_measurement_provenance(recipe, descriptor, "compact")
                            self.assertEqual(result["branch_observation_fingerprint"], "3" * 64)
                        else:
                            with self.assertRaisesRegex(ValueError, "unsupported Gym revision"):
                                bw._search_measurement_provenance(recipe, descriptor, "compact")

    def test_descent_cost_gate_derives_from_recipe_table(self):
        stop = RuntimeError("past the gate")
        for revision, accepted in ((FAKE, True), (UNKNOWN, False)):
            with registered(), patch.object(bw, "_table_states", return_value=(revision, [])), \
                 patch.object(bw, "_check_live_revision", side_effect=stop):
                if accepted:
                    with self.assertRaisesRegex(RuntimeError, "past the gate"):
                        descent.run(Path("out"))
                else:
                    with self.assertRaisesRegex(ValueError, "Unsupported Gym table revision"):
                        descent.run(Path("out"))


if __name__ == "__main__":
    unittest.main()
