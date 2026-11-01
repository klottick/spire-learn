"""Strict authored neural configuration and case-lineage tests."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import copy
import json
import tempfile
import re
import unittest

from spire_learn.neural.cases import (
    NeuralCases,
    environment_seed_for,
    load_neural_cases,
    validate_neural_case_lineage,
)
from spire_learn.neural.config import load_neural_config


class NeuralConfigTests(unittest.TestCase):
    def test_fixed_gym_revision_loads_without_rewriting_historical_configs(self):
        old_revision = "533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
        new_revision = "a86df922a9dc598645820cc6544df36aefb3d2cc"
        for mode in ("standalone", "run"):
            base = Path(f"configs/neural/{mode}.toml")
            source = base.read_text().replace(
                '../cases/neural/train.json',
                str(Path("configs/cases/neural/train.json").resolve()),
            ).replace(
                f'../../artifacts/development/neural-combat/runs/{mode}',
                str(Path(f"artifacts/development/neural-combat/runs/{mode}").resolve()),
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(source.replace(old_revision, new_revision))
                try:
                    updated = load_neural_config(path)
                except ValueError as error:
                    self.fail(f"the delivered fixed Gym revision must load: {error}")
                self.assertEqual(updated.environment_revision, new_revision)
                historical = load_neural_config(base)
                self.assertEqual(historical.environment_revision, old_revision)
                self.assertEqual(updated.accepted_identities, historical.accepted_identities)
                self.assertNotEqual(updated.config_sha256, historical.config_sha256)
                path.write_text(source.replace(old_revision, "0" * 40))
                with self.assertRaisesRegex(ValueError, "unsupported Gym revision"):
                    load_neural_config(path)

    def test_environment_seed_domains_and_train_replica_offsets(self):
        self.assertEqual(environment_seed_for("train", replica=0, ordinal=17), 17)
        self.assertEqual(environment_seed_for("train", replica=2, ordinal=17), 2 * 2**32 + 17)
        self.assertEqual(environment_seed_for("selection", replica=0, ordinal=0), 2**40)
        self.assertEqual(environment_seed_for("final", replica=0, ordinal=0), 2**48)
        invalid = (
            ("train", 256, 0),
            ("train", 0, 2**32),
            ("selection", 1, 0),
            ("final", 0, 2**64 - 2**48),
        )
        for split, replica, ordinal in invalid:
            with self.subTest(split=split, replica=replica, ordinal=ordinal), self.assertRaises(ValueError):
                environment_seed_for(split, replica=replica, ordinal=ordinal)

    def test_authored_configs_resolve_profiles_paths_and_hashes(self):
        standalone = load_neural_config(Path("configs/neural/standalone.toml"))
        run = load_neural_config(Path("configs/neural/run.toml"))
        self.assertEqual((standalone.mode, run.mode), ("standalone", "run"))
        self.assertEqual(standalone.split, run.split)
        self.assertEqual(standalone.partition, "train")
        self.assertEqual(run.profile_id, "m10-ironclad-policy")
        self.assertEqual((run.observation_version, run.replay_version), (3, 3))
        self.assertIsNone(standalone.scenario_stream)
        self.assertIsNone(run.scenario_stream)
        self.assertEqual(len(standalone.config_sha256), 64)
        self.assertEqual(len(standalone.manifest_sha256), 64)
        self.assertEqual(standalone.manifest_path, Path("configs/cases/neural/train.json").resolve())
        self.assertEqual(standalone.output_root, Path(
            "artifacts/development/neural-combat/runs/standalone"
        ).resolve())
        self.assertEqual(run.accepted_identities[0].accepted_raw_mode_mappings.keys(),
                         {"standalone", "run"})

    def test_general_cuda_config_requires_only_the_qualified_runtime_uuid(self):
        gpu_uuid = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
        source = Path("configs/neural/standalone.toml").read_text()
        source = source.replace(
            'manifest_path = "../cases/neural/train.json"',
            f'manifest_path = "{Path("configs/cases/neural/train.json").resolve()}"',
        ).replace(
            'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
            f'output_root = "{Path("artifacts/development/neural-combat/runs/config-tests").resolve()}"',
        )
        valid = source.replace('device = "cpu"', 'device = "cuda:0"')
        valid = f'{valid}\n[runtime]\ngpu_uuid = "{gpu_uuid}"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.toml"
            path.write_text(valid)
            config = load_neural_config(path)
            self.assertEqual(config.device, "cuda:0")
            self.assertEqual(config.runtime_gpu_uuid, gpu_uuid)

            invalid_configs = (
                valid.rsplit("\n[runtime]\n", 1)[0],
                valid.replace(gpu_uuid, "GPU-00000000-0000-0000-0000-000000000000"),
                valid.replace(gpu_uuid, "not-a-uuid"),
                valid.replace(f'gpu_uuid = "{gpu_uuid}"', "gpu_uuid = true"),
                valid.replace(f'gpu_uuid = "{gpu_uuid}"', f'gpu_uuid = "{gpu_uuid}"\nextra = 1'),
                valid.replace('device = "cuda:0"', 'device = "cuda"'),
                valid.replace('device = "cuda:0"', 'device = "cuda:1"'),
                source + f'\n[runtime]\ngpu_uuid = "{gpu_uuid}"\n',
            )
            for invalid in invalid_configs:
                with self.subTest(invalid=invalid[-80:]):
                    path.write_text(invalid)
                    with self.assertRaises(ValueError):
                        load_neural_config(path)

        cpu = load_neural_config(Path("configs/neural/standalone.toml"))
        self.assertEqual(cpu.device, "cpu")
        self.assertIsNone(cpu.runtime_gpu_uuid)
        calibration = load_neural_config(Path("configs/neural/calibration.toml"))
        self.assertEqual(calibration.calibration_gpu_uuid, gpu_uuid)
        self.assertIsNone(calibration.runtime_gpu_uuid)

    def test_authored_learner_identity_and_architecture_contract(self):
        standalone = load_neural_config(Path("configs/neural/standalone.toml"))
        run = load_neural_config(Path("configs/neural/run.toml"))
        identity = standalone.accepted_identities[0]
        self.assertEqual(
            (identity.model_id, identity.features_id, identity.action_id, identity.objective_id),
            (
                "spire-learn-combat-ff-v1",
                "spire-learn-combat-features-v1",
                "spire-learn-combat-dispatch-v1",
                "spire-learn-combat-winloss-v1",
            ),
        )
        self.assertEqual(standalone.common_view_id, "spire-gym-combat-policy-view-v1")
        self.assertEqual(
            identity.accepted_raw_mode_mappings,
            {
                "standalone": "spire-learn-standalone-common-v1@"
                "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78",
                "run": "spire-learn-run-policy-v3-common-v1@"
                "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8:"
                "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78",
            },
        )
        self.assertEqual(run.run_observation_fingerprint,
                         "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8")
        self.assertEqual(
            (standalone.model.embedding_width, standalone.model.hidden_width,
             standalone.model.max_parameters),
            (16, 64, 10_000_000),
        )
        self.assertFalse(hasattr(standalone.model, "activation"))
        self.assertEqual(standalone.ppo.adam_eps, 0.00001)
        self.assertFalse(hasattr(standalone.ppo, "target_kl"))

    def test_config_rejects_unsupported_learner_identity(self):
        source = Path("configs/neural/standalone.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "standalone.toml"
            (root / "train.json").write_text(Path("configs/cases/neural/train.json").read_text())
            text = source.replace(
                'manifest_path = "../cases/neural/train.json"',
                'manifest_path = "train.json"',
            ).replace(
                'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
                f'output_root = "{Path("artifacts/development/neural-combat/runs/identity-test").resolve()}"',
            ).replace(
                'model_id = "spire-learn-combat-ff-v1"',
                'model_id = "unsupported-model-v1"',
            )
            config_path.write_text(text)
            with self.assertRaises(ValueError):
                load_neural_config(config_path)

    def test_authored_rollout_profiles_and_action_budgets_are_bounded(self):
        standalone = load_neural_config(Path("configs/neural/standalone.toml"))
        run = load_neural_config(Path("configs/neural/run.toml"))
        self.assertEqual((standalone.rollout_rows, standalone.rollout_steps), (4, 32))
        self.assertEqual((run.rollout_rows, run.rollout_steps), (1, 128))
        for config in (standalone, run):
            self.assertEqual(config.budgets.max_training_decisions, 256)
            self.assertEqual(config.budgets.max_environment_actions, 1024)
            self.assertEqual(config.budgets.max_optimizer_steps, 16)
            self.assertEqual(config.budgets.max_wall_seconds, 120.0)

    def test_run_parallel_slots_and_rollout_batch_caps_are_enforced(self):
        standalone_source = Path("configs/neural/standalone.toml").read_text()
        run_source = Path("configs/neural/run.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.json").write_text(Path("configs/cases/neural/train.json").read_text())
            output = Path("artifacts/development/neural-combat/runs/config-tests").resolve()

            def write_config(name: str, source: str, mode: str) -> Path:
                text = source.replace(
                    'manifest_path = "../cases/neural/train.json"',
                    'manifest_path = "train.json"',
                )
                text = text.replace(
                    f'output_root = "../../artifacts/development/neural-combat/runs/{mode}"',
                    f'output_root = "{output}"',
                )
                path = root / name
                path.write_text(text)
                return path

            run_path = write_config("run.toml", run_source, "run")
            run_text = run_path.read_text()
            run_path.write_text(re.sub(r"(?m)^rows = \d+$", "rows = 2", run_text, count=1))
            with self.assertRaises(ValueError):
                load_neural_config(run_path)

            standalone_path = write_config("standalone.toml", standalone_source, "standalone")
            standalone_text = standalone_path.read_text()
            standalone_text = re.sub(r"(?m)^rows = \d+$", "rows = 4", standalone_text, count=1)
            standalone_text = re.sub(r"(?m)^steps = \d+$", "steps = 32", standalone_text, count=1)
            standalone_text = standalone_text.replace(
                "max_training_decisions = 256", "max_training_decisions = 127"
            )
            standalone_path.write_text(standalone_text)
            with self.assertRaises(ValueError):
                load_neural_config(standalone_path)

            standalone_valid = write_config("standalone-valid.toml", standalone_source, "standalone")
            valid_text = standalone_valid.read_text()
            invalid_action_caps = (
                valid_text.replace("max_environment_actions = 1024\n", ""),
                valid_text.replace("max_environment_actions = 1024", "max_environment_actions = 0"),
                valid_text.replace("max_environment_actions = 1024", "max_environment_actions = 127"),
            )
            for invalid_text in invalid_action_caps:
                with self.subTest(action_cap=invalid_text), self.assertRaises(ValueError):
                    standalone_valid.write_text(invalid_text)
                    load_neural_config(standalone_valid)

            wide_slots = re.sub(r"(?m)^rows = \d+$", "rows = 257", valid_text, count=1)
            wide_slots = re.sub(r"(?m)^steps = \d+$", "steps = 1", wide_slots, count=1)
            wide_slots = wide_slots.replace(
                "max_training_decisions = 256", "max_training_decisions = 500"
            ).replace("max_environment_actions = 1024", "max_environment_actions = 500")
            standalone_valid.write_text(wide_slots)
            with self.assertRaises(ValueError):
                load_neural_config(standalone_valid)

    def test_train_manifest_has_24_exact_standalone_entries_and_explicit_run_case(self):
        manifest = load_neural_cases(Path("configs/cases/neural/train.json"))
        standalone = tuple(case for case in manifest.cases if case.mode == "standalone")
        runs = tuple(case for case in manifest.cases if case.mode == "run")
        self.assertEqual(len(standalone), 24)
        self.assertEqual(len(runs), 1)
        self.assertEqual(
            {case.entry["encounter"] for case in standalone},
            {"cultist", "jaw_worm", "two_louse", "small_slimes"},
        )
        self.assertEqual({case.entry["hp"] for case in standalone}, {64, 80})
        self.assertTrue(all(case.entry["max_hp"] == 80 for case in standalone))
        self.assertTrue(all(case.entry_id == case.entry["id"] for case in standalone))
        self.assertTrue(all(case.entry["relics"] == ("burning_blood",) for case in standalone))
        self.assertTrue(all(case.entry["potions"] == (None, None, None) for case in standalone))
        expected_decks = {
            "starting": Counter({"strike": 5, "defend": 4, "bash": 1}),
            "immolate_armaments": Counter({
                "strike": 5, "defend": 4, "bash": 1, "immolate": 1, "armaments": 1,
            }),
            "warcry_whirlwind": Counter({
                "strike": 5, "defend": 4, "bash": 1, "warcry": 1, "whirlwind": 1,
            }),
        }
        for case in standalone:
            deck_counts = Counter(row["card"] for row in case.entry["deck"])
            self.assertEqual(deck_counts, expected_decks[case.case_id.split("-")[-3]])
        self.assertTrue(all(type(row["upgrades"]) is int for case in standalone
                            for row in case.entry["deck"]))
        self.assertEqual(runs[0].profile_id, "m10-ironclad-policy")
        self.assertIsNone(runs[0].entry)

    def test_native_entry_loader_is_open_to_future_ids_but_strict_about_structure(self):
        base_entry = {
            "id": "Native_Entry-001",
            "encounter": "future_public_encounter",
            "hp": 37,
            "max_hp": 99,
            "deck": [{"card": "future_public_card", "upgrades": 0}],
            "relics": ["future_public_relic"],
            "potions": [None, "future_public_potion", None],
        }

        def write_manifest(root: Path, entry: dict[str, object], entry_id: str | None = None) -> Path:
            case_entry_id = entry["id"] if entry_id is None else entry_id
            manifest = {
                "schema": "spire-learn-neural-cases-v1",
                "split": "train",
                "partition": "train",
                "cases": [{
                    "case_id": "future-shape-case",
                    "lineage_id": "future-shape-lineage",
                    "mode": "standalone",
                    "environment_seed": 17,
                    "cohort": "future_public_cohort",
                    "entry_id": case_entry_id,
                    "entry": entry,
                }],
            }
            path = root / "cases.json"
            path.write_text(json.dumps(manifest))
            return path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = write_manifest(root, copy.deepcopy(base_entry))
            loaded = load_neural_cases(valid_path)
            future_case = loaded.cases[0]
            self.assertEqual(future_case.entry_id, "Native_Entry-001")
            self.assertEqual(future_case.entry["encounter"], "future_public_encounter")

            counter_entry = copy.deepcopy(base_entry)
            counter_entry["relics"] = ["future_public_relic", "neows_lament"]
            counter_entry["relic_counters"] = {"neows_lament": 2}
            self.assertEqual(
                load_neural_cases(write_manifest(root, counter_entry)).cases[0]
                .entry["relic_counters"]["neows_lament"],
                2,
            )

            invalid_entries: list[tuple[str, dict[str, object], str | None]] = []

            entry = copy.deepcopy(base_entry)
            entry["unexpected"] = True
            invalid_entries.append(("unknown entry field", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["hp"] = True
            invalid_entries.append(("boolean hp", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["hp"] = 0
            invalid_entries.append(("hp lower bound", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["max_hp"] = 1000
            invalid_entries.append(("max hp upper bound", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["deck"] = []
            invalid_entries.append(("empty deck", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["deck"] = [{"card": "future_card", "upgrades": 0}] * 129
            invalid_entries.append(("oversize deck", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["deck"][0]["unexpected"] = 1
            invalid_entries.append(("unknown deck row field", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["deck"][0]["upgrades"] = True
            invalid_entries.append(("boolean upgrade", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["deck"][0]["upgrades"] = 11
            invalid_entries.append(("upgrade upper bound", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["relics"] = ["duplicate", "duplicate"]
            invalid_entries.append(("duplicate relic", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["potions"] = [None, None]
            invalid_entries.append(("potion slot count", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["potions"] = [None, "", None]
            invalid_entries.append(("empty potion id", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["relic_counters"] = {"unknown_relic": 1}
            invalid_entries.append(("unknown relic counter", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["relic_counters"] = {"neows_lament": False}
            invalid_entries.append(("boolean relic counter", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["relic_counters"] = {"neows_lament": 0}
            invalid_entries.append(("relic counter value", entry, None))
            entry = copy.deepcopy(base_entry)
            entry["relic_counters"] = {"neows_lament": 1}
            invalid_entries.append(("counter without equipped relic", entry, None))
            entry = copy.deepcopy(base_entry)
            invalid_entries.append(("mismatched outer id", entry, "different-id"))
            entry = copy.deepcopy(base_entry)
            entry["id"] = "invalid.entry.id"
            invalid_entries.append(("entry id alphabet", entry, None))

            for label, entry, outer_id in invalid_entries:
                with self.subTest(invalid=label), self.assertRaises(ValueError):
                    load_neural_cases(write_manifest(root, entry, outer_id))

    def test_empty_relic_counter_table_is_valid_without_neows_lament(self):
        entry = {
            "id": "empty-counter-entry",
            "encounter": "future_public_encounter",
            "hp": 50,
            "max_hp": 80,
            "deck": [{"card": "future_public_card", "upgrades": 0}],
            "relics": ["future_public_relic"],
            "potions": [None, None, None],
            "relic_counters": {},
        }
        manifest = {
            "schema": "spire-learn-neural-cases-v1",
            "split": "train",
            "partition": "train",
            "cases": [{
                "case_id": "empty-counter-case",
                "lineage_id": "empty-counter-lineage",
                "mode": "standalone",
                "environment_seed": 29,
                "cohort": "future_public_cohort",
                "entry_id": entry["id"],
                "entry": entry,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(manifest))
            loaded = load_neural_cases(path)
        self.assertEqual(loaded.cases[0].entry["relic_counters"], {})

    def test_selection_uses_new_ids_held_out_seeds_and_separate_unseen_cohort(self):
        train = load_neural_cases(Path("configs/cases/neural/train.json"))
        selection = load_neural_cases(Path("configs/cases/neural/selection.json"))
        seen_train = {case.case_id for case in train.cases}
        self.assertTrue(all(case.case_id not in seen_train for case in selection.cases))
        standalone = tuple(case for case in selection.cases if case.mode == "standalone")
        seen = {case.entry["encounter"] for case in standalone if case.cohort == "seen"}
        held_out = {case.entry["encounter"] for case in standalone if case.cohort == "held_out_encounter"}
        self.assertEqual(seen, {"cultist", "jaw_worm", "two_louse", "small_slimes"})
        self.assertEqual(held_out, {"large_slime", "three_louse"})
        self.assertTrue(all(case.environment_seed >= 2**40 for case in selection.cases))
        self.assertTrue(all(case.environment_seed < 2**48 for case in selection.cases))

    def test_invalid_config_and_split_lineage(self):
        source = Path("configs/neural/standalone.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "bad.toml"
            manifest_path = root / "train.json"
            manifest_path.write_text(Path("configs/cases/neural/train.json").read_text())
            for mutation in (
                source.replace('mode = "standalone"', 'mode = "other"'),
                source + "\nunknown_key = 1\n",
                source.replace("rows = 4", "rows = true"),
                source.replace("learning_rate = 0.0003", "learning_rate = nan"),
                source.replace('mode = "standalone"', 'mode = "run"'),
            ):
                text = mutation.replace(
                    'manifest_path = "../cases/neural/train.json"',
                    'manifest_path = "train.json"',
                )
                config_path.write_text(text)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_neural_config(config_path)

        train = load_neural_cases(Path("configs/cases/neural/train.json"))
        selection = load_neural_cases(Path("configs/cases/neural/selection.json"))
        duplicated = NeuralCases(
            Path("duplicate.json"),
            selection.split,
            selection.partition,
            (selection.cases[0].__class__(
                **{
                    **selection.cases[0].__dict__,
                    "lineage_id": train.cases[0].lineage_id,
                }
            ), *selection.cases[1:]),
            selection.sha256,
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            validate_neural_case_lineage((train, duplicated))
        with self.assertRaisesRegex(ValueError, "final.*training|training.*final"):
            validate_neural_case_lineage(
                (load_neural_cases(Path("configs/cases/neural/final.json")),),
                training=True,
            )

    def test_run_profile_schema_and_fingerprint_fail_closed(self):
        source = Path("configs/neural/run.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "run.toml"
            (root / "train.json").write_text(Path("configs/cases/neural/train.json").read_text())
            base = source.replace(
                'manifest_path = "../cases/neural/train.json"',
                'manifest_path = "train.json"',
            ).replace(
                'output_root = "../../artifacts/development/neural-combat/runs/run"',
                f'output_root = "{Path("artifacts/development/neural-combat/runs/run-test").resolve()}"',
            )
            invalid_fingerprint = "0" * 64
            for mutation in (
                base.replace('profile_id = "m10-ironclad-policy"', 'profile_id = "m10-ironclad-heart"'),
                base.replace("observation_version = 3", "observation_version = 2"),
                base.replace("replay_version = 3", "replay_version = 2"),
                base.replace(
                    "common_view_fingerprint = \"02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78\"",
                    f'common_view_fingerprint = "{invalid_fingerprint}"',
                ),
                base.replace(
                    "run_observation_fingerprint = \"30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8\"",
                    f'run_observation_fingerprint = "{invalid_fingerprint}"',
                ),
            ):
                config_path.write_text(mutation)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_neural_config(config_path)

    def test_final_manifest_is_reserved_and_config_is_evaluation_only(self):
        final = load_neural_cases(Path("configs/cases/neural/final.json"))
        self.assertEqual(final.split, "final")
        self.assertTrue(final.cases)
        source = Path("configs/neural/standalone.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "final.toml"
            manifest_path = root / "final.json"
            manifest_path.write_text(Path("configs/cases/neural/final.json").read_text())
            text = source.replace('operation = "train"', 'operation = "evaluate"')
            text = text.replace('split = "train"', 'split = "final"')
            text = text.replace('partition = "train"', 'partition = "final"')
            text = text.replace(
                'manifest_path = "../cases/neural/train.json"',
                'manifest_path = "final.json"',
            )
            text = text.replace(
                'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
                f'output_root = "{Path("artifacts/development/neural-combat/runs/final-test").resolve()}"',
            )
            config_path.write_text(text)
            self.assertTrue(load_neural_config(config_path).final_requires_opt_in)


if __name__ == "__main__":
    unittest.main()
