"""Standard Gym 1.0.0 config and case-manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.neural import envs
from spire_learn.neural.cases import load_neural_cases
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.gym_contracts import GYM_V1_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
V1 = GYM_V1_CONTRACT.source_revision
STANDARD_VIEW = GYM_V1_CONTRACT.standard_combat_fingerprint
STANDARD_RUN = GYM_V1_CONTRACT.standard_run_fingerprint


def _standard_standalone_case(seed: int = 0) -> dict[str, object]:
    fixture = json.loads((ROOT / "tests/fixtures/neural/standard-combat-catalog-1929e18.json").read_text())
    entry = fixture["entries"][0]
    entry.pop("start_kind")
    return {
        "case_id": "standard-combat-case",
        "lineage_id": "standard:combat:ironclad:a0",
        "mode": "standard-standalone",
        "environment_seed": seed,
        "cohort": "standard_combat",
        "entry": entry,
    }


def _standard_run_case(seed: int = 0) -> dict[str, object]:
    return {
        "case_id": "standard-run-case",
        "lineage_id": "standard:run:ironclad:a0",
        "mode": "standard-run",
        "environment_seed": seed,
        "cohort": "standard_run",
        "character": "ironclad",
        "ascension": 0,
        "creation": {
            "neow": "full",
            "seen_bosses": [["Guardian", "Hexaghost", "SlimeBoss"]],
            "note": {"card": "iron_wave", "upgrades": 0},
        },
    }


def _write_case_manifest(path: Path, case: dict[str, object], *, split: str = "train") -> None:
    path.write_text(json.dumps({
        "schema": "spire-learn-neural-cases-v1",
        "split": split,
        "partition": split,
        "cases": [case],
    }))


class StandardNeuralConfigTests(unittest.TestCase):
    def test_standard_templates_validate(self):
        expected = {
            "standard-standalone": STANDARD_VIEW,
            "standard-run": STANDARD_RUN,
        }
        for mode, fingerprint in expected.items():
            with self.subTest(mode=mode):
                config_path = ROOT / f"configs/neural/gym-v1.0.0/{mode}.toml"
                config = load_neural_config(config_path)
                self.assertEqual(config.mode, mode)
                self.assertEqual(config.environment_revision, V1)
                self.assertEqual(config.standard_view_fingerprint, STANDARD_VIEW)
                self.assertEqual(config.standard_run_fingerprint,
                                 STANDARD_RUN if mode == "standard-run" else None)
                self.assertEqual(config.workers, 1)
                self.assertEqual(config.accepted_identities[0].accepted_raw_mode_mappings[mode],
                                 GYM_V1_CONTRACT.raw_mode_mappings[mode])
                from spire_learn.neural.cli import main
                with patch("sys.stdout") as stdout:
                    self.assertEqual(main(["train", "--config", str(config_path), "--validate-only"]), 0)
                self.assertIn(
                    "configuration valid; no neural runtime or artifacts created",
                    "".join(call.args[0] for call in stdout.write.call_args_list),
                )

    def test_standard_requires_v1_revision_before_runtime_import(self):
        config = load_neural_config(ROOT / "configs/neural/gym-v1.0.0/standard-standalone.toml")
        bad = __import__("dataclasses").replace(
            config, environment_revision="afb51f5f222b4c4331ed34067c5eebeda606db91"
        )
        with patch.object(envs, "_load_native_runtime", side_effect=AssertionError("runtime imported")) as loader:
            with self.assertRaises(ValueError):
                envs.make_driver(bad, bad.cases)
            loader.assert_not_called()

    def test_standard_fingerprint_must_match_contract(self):
        source = (ROOT / "configs/neural/gym-v1.0.0/standard-standalone.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "cases.json"
            _write_case_manifest(case_path, _standard_standalone_case())
            config_path = root / "config.toml"
            source = source.replace(
                'manifest_path = "../../cases/neural/standard-standalone-train.json"',
                f'manifest_path = "{case_path}"',
            ).replace(
                f'output_root = "../../../artifacts/development/neural-combat/runs/gym-v1.0.0/standard-standalone"',
                f'output_root = "{(ROOT / "artifacts/development/neural-combat/runs/test-standard-config").resolve()}"',
            )
            config_path.write_text(source.replace(STANDARD_VIEW, "0" * 64, 1))
            with self.assertRaisesRegex(ValueError, "standard.*fingerprint"):
                load_neural_config(config_path)

    def test_legacy_modes_still_ironclad_a0(self):
        manifest = json.loads((ROOT / "configs/cases/neural/train.json").read_text())
        run_case = next(case for case in manifest["cases"] if case["mode"] == "run")
        run_case["character"] = "silent"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-run.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Ironclad A0"):
                load_neural_cases(path)

    def test_workers_bounds(self):
        config_path = ROOT / "configs/neural/gym-v1.0.0/standard-standalone.toml"
        source = config_path.read_text()
        for workers in (0, 257):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(source.replace("workers = 1", f"workers = {workers}"))
                with self.assertRaisesRegex(ValueError, "workers"):
                    load_neural_config(path)

    def test_workers_default_and_standard_run_fingerprint(self):
        for mode in ("standard-standalone", "standard-run"):
            source = (ROOT / f"configs/neural/gym-v1.0.0/{mode}.toml").read_text()
            source = source.replace("workers = 1\n", "")
            source = source.replace(
                f'manifest_path = "../../cases/neural/{mode}-train.json"',
                f'manifest_path = "{(ROOT / f"configs/cases/neural/{mode}-train.json").resolve()}"',
            )
            source = source.replace(
                f'output_root = "../../../artifacts/development/neural-combat/runs/gym-v1.0.0/{mode}"',
                f'output_root = "{(ROOT / "artifacts/development/neural-combat/runs/test-standard-default").resolve()}"',
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(source)
                self.assertEqual(load_neural_config(path).workers, 1)
        source = (ROOT / "configs/neural/gym-v1.0.0/standard-run.toml").read_text()
        source = source.replace(
            'manifest_path = "../../cases/neural/standard-run-train.json"',
            f'manifest_path = "{(ROOT / "configs/cases/neural/standard-run-train.json").resolve()}"',
        ).replace(
            'output_root = "../../../artifacts/development/neural-combat/runs/gym-v1.0.0/standard-run"',
            f'output_root = "{(ROOT / "artifacts/development/neural-combat/runs/test-standard-default").resolve()}"',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(source.replace(f'standard_run_fingerprint = "{STANDARD_RUN}"\n', ""))
            with self.assertRaisesRegex(ValueError, "environment.*exactly"):
                load_neural_config(path)

    def test_standard_cases_preserve_multi_character_and_creation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standalone_path = root / "standard-standalone.json"
            run_path = root / "standard-run.json"
            _write_case_manifest(standalone_path, _standard_standalone_case())
            _write_case_manifest(run_path, _standard_run_case())
            standalone = load_neural_cases(standalone_path).cases[0]
            run = load_neural_cases(run_path).cases[0]
            self.assertEqual((standalone.mode, standalone.character, standalone.ascension),
                             ("standard-standalone", "ironclad", 0))
            self.assertEqual((run.mode, run.character, run.ascension),
                             ("standard-run", "ironclad", 0))
            self.assertEqual(run.creation["note"]["card"], "iron_wave")

    def test_standard_case_character_and_ascension_bounds(self):
        for character, ascension in (("silent", 20), ("defect", 0), ("watcher", 17)):
            case = _standard_standalone_case()
            case["entry"]["character"] = character
            case["entry"]["ascension"] = ascension
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.json"
                _write_case_manifest(path, case)
                loaded = load_neural_cases(path).cases[0]
                self.assertEqual((loaded.character, loaded.ascension), (character, ascension))
        for character, ascension in (("ironclad", 21), ("ironclad", -1), ("colourless", 0)):
            case = _standard_standalone_case()
            case["entry"]["character"] = character
            case["entry"]["ascension"] = ascension
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.json"
                _write_case_manifest(path, case)
                with self.assertRaises(ValueError):
                    load_neural_cases(path)

    def test_standard_run_creation_uses_input_note_key_and_shape(self):
        for creation in (
            {"neow": "full", "seen_bosses": [], "stored_note": {"card": "iron_wave", "upgrades": 0}},
            {"neow": "full", "seen_bosses": []},
            {"neow": "full", "seen_bosses": [], "note": {"card": "iron_wave", "upgrades": True}},
        ):
            case = _standard_run_case()
            case["creation"] = creation
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.json"
                _write_case_manifest(path, case)
                with self.assertRaises(ValueError):
                    load_neural_cases(path)


if __name__ == "__main__":
    unittest.main()
