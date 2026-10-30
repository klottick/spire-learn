"""Focused pure validation for the authored Task 9 study package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    from scripts.neural import task9_study_package
except ImportError as exc:
    task9_study_package = None
    PACKAGE_IMPORT_ERROR = str(exc)
else:
    PACKAGE_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
STUDY_RELATIVE = Path("configs/neural/experiments/task9-v1/study.json")
TRAINING_ORDER = ("standalone-a", "run-a", "standalone-b", "run-b")
POLICY_ORDER = (
    "standalone-a", "run-a", "standalone-b", "run-b",
    "legal-random", "end-turn-last-v1",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_source_package(destination: Path) -> None:
    """Copy only authored source/config inputs; no ignored runtime artifacts."""
    shutil.copytree(ROOT / "src/spire_learn", destination / "src/spire_learn")
    shutil.copytree(ROOT / "scripts/neural", destination / "scripts/neural",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs/neural/experiments/task9-v1",
                    destination / "configs/neural/experiments/task9-v1")
    shutil.copytree(ROOT / "configs/runtime", destination / "configs/runtime")
    cases = destination / "configs/cases/neural"
    cases.mkdir(parents=True)
    for name in ("train.json", "selection.json", "final.json"):
        shutil.copy2(ROOT / "configs/cases/neural" / name, cases / name)
    for path in (ROOT / "configs/cases/neural").glob("task9-selection*.json"):
        shutil.copy2(path, cases / path.name)


def _validate_in_subprocess(root: Path, *, import_guard: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script_path = root / "scripts/neural/task9_study_package.py"
    args = [
        "--root", str(root), "--study", STUDY_RELATIVE.as_posix(), "--validate",
    ]
    if import_guard:
        guard_code = r'''
import importlib.abc
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts/neural"))
blocked = ("torch", "numpy", "tensordict", "torchrl", "gymnasium", "spire_lab")
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            attempts.append(fullname)
            raise RuntimeError("optional import attempted: " + fullname)
sys.meta_path.insert(0, Guard())
from task9_study_package import main
status = main(["--root", str(root), "--study", "configs/neural/experiments/task9-v1/study.json",
               "--validate"])
assert status == 0, status
assert not attempts, attempts
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in blocked)
print("optional import guard passed: 6 prefixes, no attempted or loaded imports")
'''
        return subprocess.run([sys.executable, "-B", "-c", guard_code, str(root)], cwd=root,
                              env=env, text=True, capture_output=True, timeout=30)
    return subprocess.run([sys.executable, "-B", str(script_path), *args], cwd=root,
                          env=env, text=True, capture_output=True, timeout=30)


class StudyPackageTests(unittest.TestCase):
    def api(self):
        if task9_study_package is None:
            self.fail(f"Task A package API is missing: {PACKAGE_IMPORT_ERROR}")
        return task9_study_package

    def validated(self) -> dict[str, object]:
        return self.api().validate_study(ROOT, ROOT / STUDY_RELATIVE)

    def assert_mutation_rejected(self, mutate, expected: str | None = None) -> None:
        if task9_study_package is None:
            self.skipTest(f"validator module not present yet: {PACKAGE_IMPORT_ERROR}")
        with tempfile.TemporaryDirectory(prefix="spire-task9-study-") as temporary:
            root = Path(temporary)
            _copy_source_package(root)
            mutate(root)
            result = _validate_in_subprocess(root)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        if expected is not None:
            self.assertIn(expected, result.stderr + result.stdout)

    def test_validation_returns_frozen_protocol_and_exact_inventory(self):
        study = self.validated()
        self.assertEqual(study["schema"], "spire-learn-task9-study-v1")
        self.assertEqual(study["state"], "prepared_not_released")
        self.assertEqual(tuple(run["policy_id"] for run in study["training"]["runs"]),
                         TRAINING_ORDER)
        self.assertEqual(tuple(study["selection"]["policy_order"]), POLICY_ORDER)
        self.assertEqual(len(study["selection"]["case_bindings"]), 32)
        self.assertEqual(len(study["selection"]["schedule"]), 192)

    def test_all_caps_seeds_defaults_and_backend_pins_match_the_spec(self):
        study = self.validated()
        budgets = study["budgets"]
        self.assertEqual(budgets["training"], {
            "learned_decisions": 16384,
            "environment_step_slots": 65536,
            "optimizer_attempts": 1024,
            "soft_wall_seconds": 600,
            "phase_wall_seconds": 900,
            "output_ceiling_bytes": 1073741824,
            "final_checkpoint_slots": 1,
            "midpoint_checkpoint_slots": 0,
            "whole_phase_includes_setup_cleanup": True,
        })
        self.assertEqual(budgets["selection"]["step_slots_per_operation"], 256)
        self.assertEqual(budgets["selection"]["reserved_step_slots"], 49152)
        self.assertEqual(budgets["selection"]["max_reserved_step_slots"], 65536)
        self.assertEqual(budgets["selection"]["soft_wall_seconds"], 600)
        self.assertEqual(budgets["selection"]["phase_wall_seconds"], 900)
        self.assertEqual(budgets["selection"]["output_ceiling_bytes"], 2147483648)
        self.assertEqual(budgets["selection"]["shared_soft_deadline_start"], "before_setup")
        self.assertTrue(budgets["selection"]["phase_wall_limit_includes_cleanup"])
        self.assertEqual(budgets["campaign"]["phase_count"], 5)
        self.assertEqual(budgets["campaign"]["whole_wall_seconds"], 4500)
        self.assertEqual(budgets["storage"]["total_output_ceiling_bytes"], 6442450944)
        self.assertEqual(budgets["storage"]["retained_aggregate_ceiling_bytes"], 42949672960)
        self.assertEqual(budgets["storage"]["minimum_free_bytes"], 32212254720)

        pins = study["runtime_pins"]
        self.assertEqual(pins["python"], "3.12")
        self.assertEqual(pins["torch"], "2.14.1+cu130")
        self.assertEqual(pins["gym_revision"], "533dad1c8aec86df624b3742b6b3ad1dbf1d0033")
        self.assertEqual(pins["gpu_uuid"], "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d")
        self.assertEqual(pins["train_device"], "cuda:0")
        self.assertEqual(pins["selection_device"], "cpu")
        self.assertEqual(pins["cpu_threads"], 4)
        self.assertEqual(pins["interop_threads"], 1)

        expected_seeds = {
            "standalone-a": (101103, 102207, 103301, 104409),
            "run-a": (301103, 302207, 303301, 304409),
            "standalone-b": (201103, 202207, 203301, 204409),
            "run-b": (401103, 402207, 403301, 404409),
        }
        for run in study["training"]["runs"]:
            seeds = run["seeds"]
            self.assertEqual(tuple(seeds[name] for name in
                                   ("policy", "optimizer", "curriculum", "noncombat")),
                             expected_seeds[run["policy_id"]])
            self.assertEqual(run["rollout"]["rows"],
                             16 if run["mode"] == "standalone" else 1)
            self.assertEqual(run["rollout"]["steps"],
                             32 if run["mode"] == "standalone" else 128)
        self.assertEqual(study["selection"]["eval_seeds"], {
            "policy": 91003, "optimizer": 92009, "curriculum": 93011, "noncombat": 94007,
        })
        self.assertEqual(study["interpretation"]["final_split"], "not_used")

    def test_catalog_labels_and_parent_order_are_true_and_disjoint(self):
        study = self.validated()
        cases = study["selection"]["case_bindings"]
        self.assertEqual([case["mode"] for case in cases],
                         [mode for _ in range(16) for mode in ("standalone", "run")])
        self.assertEqual([case["case_index"] for case in cases],
                         [index for index in range(16) for _ in (0, 1)])
        standalone = [case for case in cases if case["mode"] == "standalone"]
        run = [case for case in cases if case["mode"] == "run"]
        self.assertEqual([case["encounter"] for case in standalone], [
            "cultist", "large_slime", "jaw_worm", "three_louse", "two_louse",
            "large_slime", "small_slimes", "three_louse", "cultist", "large_slime",
            "jaw_worm", "three_louse", "two_louse", "large_slime", "small_slimes",
            "three_louse",
        ])
        self.assertEqual([case["cohort"] for case in standalone], [
            "seen", "held_out_encounter", "seen", "held_out_encounter", "seen",
            "held_out_encounter", "seen", "held_out_encounter", "seen",
            "held_out_encounter", "seen", "held_out_encounter", "seen",
            "held_out_encounter", "seen", "held_out_encounter",
        ])
        self.assertTrue(all(case["cohort"] == "normal_run" for case in run))
        self.assertEqual(len({case["case_id"] for case in cases}), 32)
        self.assertEqual(len({case["lineage_id"] for case in cases}), 32)
        self.assertEqual(len({case["entry_id"] for case in standalone}), 16)
        self.assertEqual([case["environment_seed"] for case in standalone],
                         [2**40 + 4096 + i for i in range(16)])
        self.assertEqual([case["environment_seed"] for case in run],
                         [2**40 + 4096 + i for i in range(16, 32)])

    def test_round_robin_has_192_unique_policy_case_pairs_and_alternating_modes(self):
        study = self.validated()
        selection = study["selection"]
        schedule = selection["schedule"]
        rebuilt = self.api().build_round_robin(selection["case_bindings"],
                                               selection["policy_order"])
        self.assertEqual(tuple(schedule), rebuilt)
        self.assertEqual(len(rebuilt), 192)
        self.assertEqual([row["operation_id"] for row in rebuilt],
                         [f"selection-{index:03d}" for index in range(192)])
        pairs = [(row["policy_id"], row["case_id"]) for row in rebuilt]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(set(pairs), {
            (policy, case["case_id"])
            for policy in POLICY_ORDER
            for case in selection["case_bindings"]
        })
        self.assertEqual([(row["policy_id"], row["mode"]) for row in rebuilt[:12]], [
            (policy, mode) for policy in POLICY_ORDER for mode in ("standalone", "run")
        ])
        self.assertEqual([(row["policy_id"], row["mode"]) for row in rebuilt[12:24]], [
            (policy, mode) for policy in ("run-a", "standalone-b", "run-b",
                                          "legal-random", "end-turn-last-v1", "standalone-a")
            for mode in ("run", "standalone")
        ])
        self.assertTrue(all(row["config_sha256"] and row["config_raw_sha256"]
                            and row["one_case_manifest_sha256"]
                            and row["parent_selection_sha256"] for row in rebuilt))

    def test_round_robin_rejects_boolean_case_index(self):
        study = self.validated()
        cases = [dict(case) for case in study["selection"]["case_bindings"]]
        cases[0]["case_index"] = False
        with self.assertRaises(ValueError):
            self.api().build_round_robin(cases, list(POLICY_ORDER))

    def test_future_checkpoint_binding_contains_no_fabricated_digests(self):
        study = self.validated()
        checkpoint = study["checkpoint_binding"]
        self.assertEqual(checkpoint["state"], "future_inventory_required")
        self.assertEqual(checkpoint["inventory_policy"], "post-run-inventory")
        self.assertEqual(checkpoint["final_slot"], "final")
        self.assertEqual(len(checkpoint["policies"]), 4)
        for binding in checkpoint["policies"]:
            self.assertNotIn("checkpoint_sha256", binding)
            self.assertNotIn("metadata_sha256", binding)
            self.assertNotIn("payload_sha256", binding)
            self.assertEqual(binding["training_run_id_source"],
                             "successful-training-receipt.run_id")

    def test_invented_checkpoint_digest_is_rejected(self):
        def mutate(root: Path) -> None:
            path = root / STUDY_RELATIVE
            study = json.loads(path.read_text(encoding="utf-8"))
            study["checkpoint_binding"]["policies"][0]["checkpoint_sha256"] = "0" * 64
            _write_json(path, study)

        self.assert_mutation_rejected(mutate, "invented binding")

    def test_cli_validate_and_describe_are_pure_and_import_guarded(self):
        if task9_study_package is None:
            self.skipTest(f"validator module not present yet: {PACKAGE_IMPORT_ERROR}")
        with tempfile.TemporaryDirectory(prefix="spire-task9-copy-") as temporary:
            root = Path(temporary)
            _copy_source_package(root)
            self.assertFalse((root / "artifacts").exists())
            result = _validate_in_subprocess(root, import_guard=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"schedule_count": 192', result.stdout)
            self.assertIn("optional import guard passed: 6 prefixes", result.stdout)
            self.assertFalse((root / "artifacts").exists())
            describe = subprocess.run(
                [sys.executable, "-B", str(root / "scripts/neural/task9_study_package.py"),
                 "--describe"], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "src"),
                     "PYTHONDONTWRITEBYTECODE": "1"},
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(describe.returncode, 0, describe.stderr)
            self.assertIn("spire-learn-task9-study-v1", describe.stdout)
            self.assertFalse((root / "artifacts").exists())

    def test_altered_config_canonical_hash_is_rejected(self):
        def mutate(root: Path) -> None:
            path = root / STUDY_RELATIVE
            study = json.loads(path.read_text(encoding="utf-8"))
            study["training"]["runs"][0]["config"]["config_sha256"] = "0" * 64
            _write_json(path, study)

        self.assert_mutation_rejected(mutate, "canonical SHA-256")

    def test_config_raw_hash_is_bound_separately_from_canonical_digest(self):
        def mutate(root: Path) -> None:
            path = root / "configs/neural/experiments/task9-v1/standalone-a.toml"
            path.write_text(path.read_text(encoding="utf-8") + "# raw-only change\n",
                            encoding="utf-8")

        self.assert_mutation_rejected(mutate, "raw SHA-256")

    def test_semantic_config_change_is_rejected_by_loader_canonical_hash(self):
        def mutate(root: Path) -> None:
            study_path = root / STUDY_RELATIVE
            study = json.loads(study_path.read_text(encoding="utf-8"))
            binding = study["training"]["runs"][0]["config"]
            config_path = root / binding["path"]
            text = config_path.read_text(encoding="utf-8")
            self.assertIn("max_environment_actions = 65536", text)
            text = text.replace("max_environment_actions = 65536",
                                "max_environment_actions = 65535", 1)
            config_path.write_text(text, encoding="utf-8")
            binding["raw_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            _write_json(study_path, study)

        self.assert_mutation_rejected(mutate, "canonical SHA-256 mismatch")

    def test_reordered_training_inventory_is_rejected(self):
        def mutate(root: Path) -> None:
            path = root / STUDY_RELATIVE
            study = json.loads(path.read_text(encoding="utf-8"))
            study["training"]["runs"][0], study["training"]["runs"][1] = (
                study["training"]["runs"][1], study["training"]["runs"][0]
            )
            _write_json(path, study)

        self.assert_mutation_rejected(mutate, "training run inventory/order")

    def test_child_case_membership_must_equal_the_bound_parent_record(self):
        def mutate(root: Path) -> None:
            study_path = root / STUDY_RELATIVE
            study = json.loads(study_path.read_text(encoding="utf-8"))
            binding = study["selection"]["case_bindings"][0]
            manifest_path = root / binding["manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["entry"]["hp"] -= 1
            _write_json(manifest_path, manifest)
            raw = manifest_path.read_bytes()
            binding["manifest_sha256"] = _canonical_sha256(manifest)
            binding["manifest_raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(study_path, study)

        self.assert_mutation_rejected(mutate, "exact parent record")

    def test_train_selection_lineage_leakage_is_rejected(self):
        def mutate(root: Path) -> None:
            study_path = root / STUDY_RELATIVE
            study = json.loads(study_path.read_text(encoding="utf-8"))
            train_path = root / "configs/cases/neural/train.json"
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train["cases"][0]["lineage_id"] = (
                study["selection"]["case_bindings"][0]["lineage_id"]
            )
            _write_json(train_path, train)
            raw = train_path.read_bytes()
            study["training"]["manifest"]["canonical_sha256"] = _canonical_sha256(train)
            study["training"]["manifest"]["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            _write_json(study_path, study)

        self.assert_mutation_rejected(mutate, "duplicate lineage")

    def test_policy_substitution_is_rejected(self):
        def mutate(root: Path) -> None:
            path = root / STUDY_RELATIVE
            study = json.loads(path.read_text(encoding="utf-8"))
            study["selection"]["policy_order"][0] = "unapproved-policy"
            _write_json(path, study)

        self.assert_mutation_rejected(mutate, "policy order")

    def test_supported_binding_path_rejects_symlink(self):
        def mutate(root: Path) -> None:
            binding = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))[
                "selection"]["case_bindings"][0]
            path = root / binding["manifest_path"]
            target = path.with_name("unbound-case-target.json")
            path.rename(target)
            path.symlink_to(target.name)

        self.assert_mutation_rejected(mutate, "symlink")

    def test_unknown_study_binding_fields_are_rejected(self):
        def mutate(root: Path) -> None:
            path = root / STUDY_RELATIVE
            study = json.loads(path.read_text(encoding="utf-8"))
            study["release"] = "live"
            _write_json(path, study)

        self.assert_mutation_rejected(mutate, "exactly")


if __name__ == "__main__":
    unittest.main()
