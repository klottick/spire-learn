"""Pure parser, result-gate, and container-boundary tests for Task 5."""

from __future__ import annotations

import builtins
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from scripts.neural.task5_collection_control import (
    EXPECTED_COMMON_FINGERPRINT,
    EXPECTED_GYM_REVISION,
    EXPECTED_RUN_FINGERPRINT,
    EXPECTED_TEST_IDS,
    classify_collection_result,
    parse_text_test_result,
    pre_dispatch_manifest,
    validate_collection_evidence,
    verify_container_inspect,
)
from scripts.neural import task5_collection_control as control


ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_FRAME_HASH = "d" * 64


def _run_boundary_fixture(*, outcome: str = "victory", terminal: bool = False) -> dict[str, object]:
    counters_before = {
        "attempted_environment_slots": 10,
        "forwarded_environment_slots": 10,
        "known_accepted_environment_slots": 10,
        "delivered_environment_slots": 10,
        "uncertain_environment_slots": 0,
        "attempted_reset_slots": 1,
        "forwarded_reset_slots": 1,
        "known_accepted_reset_slots": 1,
        "uncertain_reset_slots": 0,
        "reset_operations": 1,
        "reset_slots": 1,
        "delivered_reset_slots": 1,
        "next_seed_ordinal": 1,
        "tasks_completed": 0,
        "task_terminals": 0,
    }
    counters_settled = dict(counters_before)
    counters_settled.update({
        "attempted_environment_slots": 11,
        "forwarded_environment_slots": 11,
        "known_accepted_environment_slots": 11,
        "delivered_environment_slots": 11,
        "tasks_completed": 1,
        "task_terminals": 1,
    })
    counters_continued = dict(counters_settled)
    counters_continued.update({
        "attempted_environment_slots": 12,
        "forwarded_environment_slots": 12,
        "known_accepted_environment_slots": 12,
        "delivered_environment_slots": 12,
        "tasks_completed": 1,
        "task_terminals": 1,
    })
    return {
        "policy_metadata": {
            "kind": "public-diagnostic-lowest-legal-v1",
            "behavior_policy_version": "public-diagnostic-lowest-legal-v1",
            "distribution": "point-mass-on-minimum-legal-action-mask-index",
            "value_estimator": "constant-zero-v1",
            "policy_spec": control.PUBLIC_DIAGNOSTIC_SPEC,
            "identity": {
                "model_id": "spire-learn-public-diagnostic-lowest-legal-v1",
                "features_id": "spire-learn-combat-features-v1",
                "action_id": "spire-learn-public-lowest-legal-action-v1",
                "objective_id": "spire-learn-combat-winloss-v1",
            "policy_spec_sha256": control.PUBLIC_DIAGNOSTIC_SPEC_SHA256,
            },
        },
        "settlement": {
            "sequence": 7,
            "stream_id": "stream-0",
            "task_id": "task-0-run-0-visit-0",
            "decision_id": "decision-10",
            "visit": 0,
            "outcome": outcome,
            "encounter": "cultist",
            "hp_before_healing": 51,
            "hp_after_cleanup": 57,
            "task_terminal": True,
            "native_terminated": terminal,
            "native_truncated": False,
            "actor_training_mask": True,
            "source_frame_sha256": "b" * 64,
            "successor_frame_sha256": _PUBLIC_FRAME_HASH,
            "immediate_checkpoint_sha256": _PUBLIC_FRAME_HASH,
            "log_probability": 0.0,
            "value": 0.0,
            "behavior_policy_version": "public-diagnostic-lowest-legal-v1",
            "reset_operations_before": 1,
            "reset_operations_after": 1,
            "counters_before": counters_before,
            "counters_after": counters_settled,
        },
        "continuation": {
            "sequence": 8,
            "stream_id": "stream-0",
            "task_id": "task-0-run-0-visit-1",
            "visit": 1,
            "decision_id": "decision-11",
            "continuation_kind": "fixed_noncombat",
            "actor_training_mask": False,
            "native_terminated": False,
            "native_truncated": False,
            "source_frame_sha256": _PUBLIC_FRAME_HASH,
            "successor_frame_sha256": "c" * 64,
            "log_probability": None,
            "value": None,
            "behavior_policy_version": None,
            "reset_operations_before": 1,
            "reset_operations_after": 1,
            "counters_before": counters_settled,
            "counters_after": counters_continued,
        },
    }


def collection_record(mode: str) -> dict[str, object]:
    test_id = next(
        item for item in EXPECTED_TEST_IDS
        if ("test_collect_run_" in item) == (mode == "run")
    )
    reset_operations = 1 if mode == "run" else 2
    reset_slots = (1 if mode == "run" else 4) + (reset_operations - 1)
    record: dict[str, object] = {
        "test_id": test_id,
        "mode": mode,
        "environment_revision": EXPECTED_GYM_REVISION,
        "common_view_fingerprint": EXPECTED_COMMON_FINGERPRINT,
        "run_profile_id": "m10-ironclad-policy" if mode == "run" else None,
        "run_observation_fingerprint": EXPECTED_RUN_FINGERPRINT if mode == "run" else None,
        "attempted_environment_slots": 12 if mode == "run" else 4,
        "forwarded_environment_slots": 12 if mode == "run" else 4,
        "known_accepted_environment_slots": 12 if mode == "run" else 4,
        "delivered_environment_slots": 12 if mode == "run" else 4,
        "uncertain_environment_slots": 0,
        "reset_operations": reset_operations,
        "reset_slots": reset_slots,
        "attempted_reset_slots": reset_slots,
        "forwarded_reset_slots": reset_slots,
        "known_accepted_reset_slots": reset_slots,
        "delivered_reset_slots": reset_slots,
        "uncertain_reset_slots": 0,
        "next_seed_ordinal": reset_slots,
        "tasks_completed": 1,
        "task_terminals": 1,
        "faults": 0,
        "optimizer_updates": 0,
        "shared_budget_id": control.SHARED_COHORT_BUDGET_ID,
        "wall_budget_limit_seconds": 120.0,
        "elapsed_seconds": 1.25 if mode == "run" else 2.5,
        "wall_overshoot_seconds": 0.0,
        "elapsed_seconds_is_fallback": False,
        "resource_checkpoints": [{
            "stream_id": "stream-0",
            "task_id": "task-0-run-0-visit-0" if mode == "run" else "task-0-0",
            "visit": 0,
            "outcome": "victory",
            "encounter": "cultist",
            "hp_before_healing": 51,
            "hp_after_cleanup": 57,
            "public_frame_sha256": _PUBLIC_FRAME_HASH,
        }],
        "run_continuation": _run_boundary_fixture() if mode == "run" else None,
        "manual_reset_boundary": ({
            "reset_operations_before": 1,
            "reset_operations_after": 2,
            "tasks_completed_before": 1,
            "delivered_reset_slots_before": 4,
            "delivered_reset_slots_after": 5,
            "next_seed_ordinal_before": 4,
            "next_seed_ordinal_after": 5,
            "reset_slots_before": 4,
            "reset_slots_after": 5,
            "seed_ordinals": [4],
        } if mode == "standalone" else None),
        "seed_allocations": [
            {
                "case_id": f"case-{index}",
                "allocation_policy_id": "environment-seed-for-replica0-v1",
                "actual_seed": 100 + index,
                "authored_reference_seed": 1 + index,
                "ordinal": index,
                "delivery": "delivered",
            }
            for index in range(reset_slots)
        ],
    }
    return record


def evidence() -> dict[str, object]:
    return {
        "schema": "spire-learn-neural-collection-evidence-v1",
        "tests": [collection_record("run"), collection_record("standalone")],
    }


class _ExpectedLabelCase(unittest.TestCase):
    def test_collect_run_natural_task_end_without_run_reset(self):
        pass

    def test_collect_standalone_manual_resets(self):
        pass


_ExpectedLabelCase.__name__ = "NeuralCollectionIntegrationTests"
_ExpectedLabelCase.__qualname__ = "NeuralCollectionIntegrationTests"
_ExpectedLabelCase.__module__ = "tests.integration.test_neural_collection"


def verbose_success_text() -> str:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_ExpectedLabelCase)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("authored TextTestRunner labels did not pass")
    return stream.getvalue()


def inspect_fixture(
    root: Path, venv: Path, run_root: Path, expected_command: str,
) -> dict[str, object]:
    return {
        "image_id": "sha256:" + "a" * 64,
        "user": "1000:1000",
        "workdir": "/learner",
        "entrypoint": ["/bin/sh"],
        "cmd": ["-ec", expected_command],
        "readonly_rootfs": True,
        "privileged": False,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "nano_cpus": 4_000_000_000,
        "memory": 8 * 1024**3,
        "pids_limit": 512,
        "network": "none",
        "tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
        "env": [
            "PYTHONPATH=/learner/src", "PYTHONDONTWRITEBYTECODE=1",
            "PIP_NO_INDEX=1", "PIP_NO_CACHE_DIR=1",
            "PIP_DISABLE_PIP_VERSION_CHECK=1", "HOME=/run/home", "TMPDIR=/run/tmp",
        ],
        "mounts": [
            {"Source": str(root / "src"), "Destination": "/learner/src", "RW": False},
            {"Source": str(root / "tests"), "Destination": "/learner/tests", "RW": False},
            {"Source": str(root / "configs"), "Destination": "/learner/configs", "RW": False},
            {"Type": "bind",
             "Source": str(root / "scripts/neural/task5_collection_control.py"),
             "Destination": "/learner/scripts/neural/task5_collection_control.py", "RW": False},
            {"Source": str(venv), "Destination": "/work/venv-cpu", "RW": False},
            {"Source": str(venv.parent / "requirements-cpu.lock"),
             "Destination": "/work/requirements-cpu.lock", "RW": False},
            {"Source": str(run_root), "Destination": "/run", "RW": True},
            {"Source": str(run_root),
             "Destination": "/learner/artifacts/development/neural-combat", "RW": True},
        ],
    }


class NeuralCollectionControlTests(unittest.TestCase):
    def test_parser_accepts_full_stdlib_texttestrunner_ids_and_success(self):
        output = verbose_success_text()
        result = parse_text_test_result("", output, exit_code=0)
        self.assertTrue(result["passed"])
        self.assertEqual(tuple(result["observed_test_ids"]), EXPECTED_TEST_IDS)
        classified = classify_collection_result("", output, evidence(), exit_code=0)
        self.assertEqual(classified["classification"], "passed")
        self.assertEqual(classified["collection_evidence"]["total_attempted_environment_slots"], 16)

    def test_pre_dispatch_manifest_records_two_mode_limits_and_provenance(self):
        snapshot = {
            "config_sha256": {"run": "a" * 64, "standalone": "b" * 64},
            "cpu_lock_sha256": "c" * 64,
            "gym_native_sha256": "d" * 64,
            "gym_source_map_sha256": "e" * 64,
            "gym_durable_source_files": 636,
            "gym_original_post_build_source_hash_available": False,
            "learner_head": "f" * 40,
            "dirty_status": ["?? task-owned-file"],
            "input_tree_sha256": "1" * 64,
        }
        manifest = pre_dispatch_manifest(snapshot)
        self.assertEqual(tuple(manifest["expected_test_ids"]), EXPECTED_TEST_IDS)
        self.assertEqual(manifest["limits"]["standalone"]["attempted_environment_slots"], 2048)
        self.assertEqual(manifest["limits"]["run"]["attempted_environment_slots"], 2048)
        self.assertEqual(manifest["limits"]["combined_attempted_environment_slots"], 4096)
        self.assertEqual(manifest["limits"]["whole_wrapper_active_seconds"], 210)
        self.assertEqual(manifest["limits"]["whole_wrapper_cleanup_seconds"], 90)
        self.assertEqual(manifest["limits"]["shared_budget_id"], control.SHARED_COHORT_BUDGET_ID)
        self.assertEqual(manifest["optimizer_updates"], 0)
        self.assertEqual(manifest["initial_reset_counters"]["run"]["delivered_reset_slots"], 0)
        self.assertFalse(manifest["provenance_qualification"]["original_post_build_source_hash_available"])

    def test_parser_rejects_skips_failures_duplicate_ids_and_invented_labels(self):
        success = verbose_success_text()
        self.assertFalse(parse_text_test_result(
            "", success.replace(" ... ok", " ... skipped 'disabled'", 1), exit_code=0
        )["passed"])
        self.assertFalse(parse_text_test_result("", success, exit_code=1)["passed"])
        invented = success.replace(EXPECTED_TEST_IDS[0], "tests.integration.short_name")
        self.assertFalse(parse_text_test_result("", invented, exit_code=0)["passed"])
        duplicated = success + "\n" + success.splitlines()[0] + "\n"
        self.assertFalse(parse_text_test_result("", duplicated, exit_code=0)["passed"])

    def test_evidence_rejects_action_overshoot_faults_updates_and_missing_boundary(self):
        record = evidence()
        record["tests"][1]["attempted_environment_slots"] = 2049
        with self.assertRaisesRegex(ValueError, "per-mode"):
            validate_collection_evidence(record)
        record = evidence()
        record["tests"][0]["optimizer_updates"] = 1
        with self.assertRaisesRegex(ValueError, "optimizer updates"):
            validate_collection_evidence(record)
        record = evidence()
        record["tests"][0]["resource_checkpoints"] = []
        with self.assertRaisesRegex(ValueError, "settlement resource"):
            validate_collection_evidence(record)
        record = evidence()
        record["tests"][0]["uncertain_reset_slots"] = 1
        with self.assertRaisesRegex(ValueError, "reset evidence is uncertain"):
            validate_collection_evidence(record)
        record = evidence()
        record["tests"][0]["run_continuation"]["settlement"]["outcome"] = "death"
        with self.assertRaisesRegex(ValueError, "nonterminal victory"):
            validate_collection_evidence(record)
        record = evidence()
        continuation_before = dict(
            record["tests"][0]["run_continuation"]["continuation"]["counters_before"]
        )
        continuation_before["reset_operations"] = 2
        record["tests"][0]["run_continuation"]["continuation"]["counters_before"] = continuation_before
        record["tests"][0]["run_continuation"]["continuation"]["reset_operations_before"] = 2
        with self.assertRaisesRegex(ValueError, "intervening action or reset"):
            validate_collection_evidence(record)
        record = evidence()
        record["tests"][1]["manual_reset_boundary"]["seed_ordinals"] = []
        with self.assertRaisesRegex(ValueError, "manual reset boundary"):
            validate_collection_evidence(record)

    def test_evidence_accepts_exact_combined_slot_action_cap(self):
        record = evidence()
        for item in record["tests"]:
            item["attempted_environment_slots"] = 2048
            item["forwarded_environment_slots"] = 2048
            item["known_accepted_environment_slots"] = 2048
            item["delivered_environment_slots"] = 2048
        run_boundary = record["tests"][0]["run_continuation"]
        action_fields = (
            "attempted_environment_slots", "forwarded_environment_slots",
            "known_accepted_environment_slots", "delivered_environment_slots",
        )
        for event, total in ((run_boundary["settlement"], 2047),
                             (run_boundary["continuation"], 2048)):
            for name in action_fields:
                event["counters_after"][name] = total
        for name in action_fields:
            run_boundary["settlement"]["counters_before"][name] = 2046
            run_boundary["continuation"]["counters_before"][name] = 2047
        self.assertEqual(validate_collection_evidence(record)["total_attempted_environment_slots"], 4096)

    def test_container_inspect_requires_exact_readonly_inputs_and_fresh_output_child(self):
        root = Path("/learner-root")
        venv = Path("/protected/runtime/data/venv-cpu")
        run_root = Path("/fresh/task-5/run-a")
        expected_command = control.extract_container_command(
            (ROOT / "scripts/neural/run-task5-collection.sh").read_text(encoding="utf-8")
        )
        fixture = inspect_fixture(root, venv, run_root, expected_command)
        verified = verify_container_inspect(
            json.dumps(fixture), root=root, venv=venv, run_root=run_root,
            image_id="sha256:" + "a" * 64, expected_command=expected_command,
        )
        self.assertTrue(verified["passed"])
        bad_command = inspect_fixture(root, venv, run_root, expected_command)
        bad_command["cmd"][1] += "\n# mutated"
        with self.assertRaisesRegex(ValueError, "Cmd"):
            verify_container_inspect(json.dumps(bad_command), root=root, venv=venv,
                                     run_root=run_root, image_id="sha256:" + "a" * 64,
                                     expected_command=expected_command)
        bad_argv = inspect_fixture(root, venv, run_root, expected_command)
        bad_argv["cmd"][0] = "-c"
        with self.assertRaisesRegex(ValueError, "Cmd"):
            verify_container_inspect(json.dumps(bad_argv), root=root, venv=venv,
                                     run_root=run_root, image_id="sha256:" + "a" * 64,
                                     expected_command=expected_command)
        bad_mounts = inspect_fixture(root, venv, run_root, expected_command)
        bad_mounts["mounts"][0]["RW"] = True
        with self.assertRaisesRegex(ValueError, "mounts"):
            verify_container_inspect(json.dumps(bad_mounts), root=root, venv=venv,
                                     run_root=run_root, image_id="sha256:" + "a" * 64,
                                     expected_command=expected_command)
        duplicate_mount = inspect_fixture(root, venv, run_root, expected_command)
        duplicate_mount["mounts"].append(dict(duplicate_mount["mounts"][0]))
        with self.assertRaisesRegex(ValueError, "mounts"):
            verify_container_inspect(json.dumps(duplicate_mount), root=root, venv=venv,
                                     run_root=run_root, image_id="sha256:" + "a" * 64,
                                     expected_command=expected_command)
        bad_env = inspect_fixture(root, venv, run_root, expected_command)
        bad_env["env"].append("AWS_SECRET_ACCESS_KEY=not-a-real-secret")
        with self.assertRaisesRegex(ValueError, "credential-like"):
            verify_container_inspect(json.dumps(bad_env), root=root, venv=venv,
                                     run_root=run_root, image_id="sha256:" + "a" * 64,
                                     expected_command=expected_command)

    def test_container_inspect_requires_exact_readonly_control_file_mount(self):
        root = Path("/learner-root")
        venv = Path("/protected/runtime/data/venv-cpu")
        run_root = Path("/fresh/task-5/run-a")
        expected_command = control.extract_container_command(
            (ROOT / "scripts/neural/run-task5-collection.sh").read_text(encoding="utf-8")
        )
        correct = inspect_fixture(root, venv, run_root, expected_command)
        expected_mount = {
            "Type": "bind",
            "Source": str(root / "scripts/neural/task5_collection_control.py"),
            "Destination": "/learner/scripts/neural/task5_collection_control.py",
            "RW": False,
        }
        self.assertIn(expected_mount, correct["mounts"])
        self.assertTrue(verify_container_inspect(
            json.dumps(correct), root=root, venv=venv, run_root=run_root,
            image_id="sha256:" + "a" * 64, expected_command=expected_command,
        )["passed"])

        rejected = []
        writable = inspect_fixture(root, venv, run_root, expected_command)
        writable["mounts"][3]["RW"] = True
        rejected.append(writable)
        wrong_source = inspect_fixture(root, venv, run_root, expected_command)
        wrong_source["mounts"][3]["Source"] = str(root / "scripts")
        rejected.append(wrong_source)
        wrong_destination = inspect_fixture(root, venv, run_root, expected_command)
        wrong_destination["mounts"][3]["Destination"] = "/learner/scripts/neural/control.py"
        rejected.append(wrong_destination)
        wrong_type = inspect_fixture(root, venv, run_root, expected_command)
        wrong_type["mounts"][3]["Type"] = "volume"
        rejected.append(wrong_type)
        missing = inspect_fixture(root, venv, run_root, expected_command)
        missing["mounts"].pop(3)
        rejected.append(missing)
        extra = inspect_fixture(root, venv, run_root, expected_command)
        extra["mounts"].append({
            "Source": str(root / "scripts"), "Destination": "/learner/scripts", "RW": False,
        })
        rejected.append(extra)

        for fixture in rejected:
            with self.subTest(mounts=fixture["mounts"]):
                with self.assertRaisesRegex(ValueError, "mounts"):
                    verify_container_inspect(
                        json.dumps(fixture), root=root, venv=venv, run_root=run_root,
                        image_id="sha256:" + "a" * 64, expected_command=expected_command,
                    )

    def test_gated_native_methods_discover_from_reviewed_mount_layout_with_optional_imports_blocked(self):
        with tempfile.TemporaryDirectory(prefix="task5-mounted-layout-") as temporary:
            learner = Path(temporary) / "learner"
            learner.mkdir()
            for mount_name in ("src", "tests", "configs"):
                (learner / mount_name).symlink_to(ROOT / mount_name, target_is_directory=True)
            mounted_control = learner / "scripts/neural/task5_collection_control.py"
            mounted_control.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/neural/task5_collection_control.py", mounted_control)
            mounted_control.chmod(0o444)
            self.assertEqual(
                sorted(path.relative_to(learner / "scripts").as_posix()
                       for path in (learner / "scripts").rglob("*")),
                ["neural", "neural/task5_collection_control.py"],
            )

            bootstrap = r'''
import builtins, io, json, pathlib, sys, unittest

blocked_roots = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab", "spire_learn"}
attempts = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked_roots:
        attempts.append(name)
        raise RuntimeError("optional import blocked: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

module_name = "tests.integration.test_neural_collection"
suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
ids = []
def collect_ids(item):
    if hasattr(item, "_tests"):
        for child in item:
            collect_ids(child)
    else:
        ids.append(item.id())
collect_ids(suite)
expected = [
    "tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_run_natural_task_end_without_run_reset",
    "tests.integration.test_neural_collection.NeuralCollectionIntegrationTests.test_collect_standalone_manual_resets",
]
assert ids == expected, ids
module = sys.modules[module_name]
control = sys.modules["scripts.neural.task5_collection_control"]
assert module._ENABLED is False
assert pathlib.Path(control.__file__).absolute() == pathlib.Path.cwd() / "scripts/neural/task5_collection_control.py"
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
assert result.testsRun == 0, result.testsRun
assert len(result.skipped) == 1, result.skipped
assert not result.errors and not result.failures
assert attempts == [], attempts
print(json.dumps({"discovered_ids": ids, "tests_run": result.testsRun,
                  "class_skips": len(result.skipped), "optional_import_attempts": attempts,
                  "control_module": str(pathlib.Path(control.__file__).absolute())}, sort_keys=True))
'''
            environment = {
                name: os.environ[name]
                for name in ("PATH", "LANG", "LC_ALL")
                if name in os.environ
            }
            environment.update({
                "PYTHONPATH": str(learner / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            })
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", bootstrap], cwd=learner,
                env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            availability = json.loads(result.stdout)
            self.assertEqual(tuple(availability["discovered_ids"]), EXPECTED_TEST_IDS)
            self.assertEqual(availability["tests_run"], 0)
            self.assertEqual(availability["class_skips"], 1)
            self.assertEqual(availability["optional_import_attempts"], [])
            self.assertEqual(Path(availability["control_module"]), mounted_control)

    def test_enabled_harness_function_aliases_are_static_for_instance_calls(self):
        optional_prefixes = {
            "numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab",
        }
        optional_import_attempts = []
        original_import = builtins.__import__

        def block_optional_imports(name, *args, **kwargs):
            prefix = name.split(".", 1)[0]
            if prefix in optional_prefixes:
                optional_import_attempts.append(name)
                raise AssertionError(f"optional import attempted in pure harness wiring: {name}")
            return original_import(name, *args, **kwargs)

        def package(name):
            result = types.ModuleType(name)
            result.__path__ = []
            return result

        budget_sentinel = object()
        collector_sentinel = object()
        cases_sentinel = object()
        model_sentinel = object()
        seed_sentinel = object()
        device_sentinel = object()
        config_sentinel = types.SimpleNamespace(
            cases=cases_sentinel,
            model=model_sentinel,
            seed_streams=types.SimpleNamespace(policy=seed_sentinel),
            device=device_sentinel,
        )
        schema_sentinel = object()
        driver_sentinel = types.SimpleNamespace(schema=schema_sentinel)
        policy_sentinel = object()
        config_path = Path("/pure-fixture/configs/neural/standalone.toml")
        monotonic = time.monotonic
        calls = []

        def strict_budget(seconds, clock, /):
            self.assertEqual(seconds, 120.0)
            self.assertIs(clock, monotonic)
            calls.append(("Budget", (seconds, clock), {}))
            return budget_sentinel

        def strict_load_neural_config(path, /):
            self.assertEqual(path, config_path)
            calls.append(("load_neural_config", (path,), {}))
            return config_sentinel

        def strict_make_driver(config, cases, /):
            self.assertIs(config, config_sentinel)
            self.assertIs(cases, cases_sentinel)
            calls.append(("make_driver", (config, cases), {}))
            return driver_sentinel

        def strict_build_policy(schema, model, /, *, seed, device):
            self.assertIs(schema, schema_sentinel)
            self.assertIs(model, model_sentinel)
            self.assertIs(seed, seed_sentinel)
            self.assertIs(device, device_sentinel)
            calls.append(("build_policy", (schema, model), {"seed": seed, "device": device}))
            return policy_sentinel

        spire_learn = package("spire_learn")
        experiments = package("spire_learn.experiments")
        neural = package("spire_learn.neural")
        budget_module = types.ModuleType("spire_learn.experiments.budget")
        budget_module.Budget = strict_budget
        collector_module = types.ModuleType("spire_learn.neural.collector")
        collector_module.Collector = collector_sentinel
        neural_config_module = types.ModuleType("spire_learn.neural.config")
        neural_config_module.load_neural_config = strict_load_neural_config
        envs_module = types.ModuleType("spire_learn.neural.envs")
        envs_module.make_driver = strict_make_driver
        model_module = types.ModuleType("spire_learn.neural.model")
        model_module.build_policy = strict_build_policy
        spire_learn.experiments = experiments
        spire_learn.neural = neural
        experiments.budget = budget_module
        neural.collector = collector_module
        neural.config = neural_config_module
        neural.envs = envs_module
        neural.model = model_module
        fake_modules = {
            "spire_learn": spire_learn,
            "spire_learn.experiments": experiments,
            "spire_learn.experiments.budget": budget_module,
            "spire_learn.neural": neural,
            "spire_learn.neural.collector": collector_module,
            "spire_learn.neural.config": neural_config_module,
            "spire_learn.neural.envs": envs_module,
            "spire_learn.neural.model": model_module,
        }

        with (
            mock.patch("builtins.__import__", block_optional_imports),
            mock.patch.dict(sys.modules, fake_modules),
        ):
            config_module = importlib.import_module("tests.integration.test_neural_collection")
            original_case = config_module.NeuralCollectionIntegrationTests
            alias_names = ("load_neural_config", "make_driver", "build_policy")
            missing = object()
            original_alias_state = {
                name: original_case.__dict__.get(name, missing) for name in alias_names
            }
            probe_case = type(
                "EnabledHarnessAliasProbe", (original_case,), {"__module__": __name__},
            )
            with (
                mock.patch.object(config_module, "_ENABLED", True),
                mock.patch.object(config_module, "_EVIDENCE_PATH", "/pure-fixture/evidence.json"),
            ):
                probe_case.setUpClass()
                instance = probe_case("test_collect_standalone_manual_resets")
                self.assertIs(probe_case.shared_budget, budget_sentinel)
                self.assertEqual(probe_case.shared_budget_id, control.SHARED_COHORT_BUDGET_ID)
                self.assertIs(probe_case.Collector, collector_sentinel)

                loaded_config = instance.load_neural_config(config_path)
                made_driver = instance.make_driver(loaded_config, loaded_config.cases)
                made_policy = instance.build_policy(
                    made_driver.schema,
                    loaded_config.model,
                    seed=loaded_config.seed_streams.policy,
                    device=loaded_config.device,
                )

                self.assertIs(loaded_config, config_sentinel)
                self.assertIs(made_driver, driver_sentinel)
                self.assertIs(made_policy, policy_sentinel)
                self.assertEqual([call[0] for call in calls], [
                    "Budget", "load_neural_config", "make_driver", "build_policy",
                ])
                self.assertEqual(calls[1], ("load_neural_config", (config_path,), {}))
                self.assertEqual(calls[2], ("make_driver", (config_sentinel, cases_sentinel), {}))
                self.assertEqual(calls[3], (
                    "build_policy", (schema_sentinel, model_sentinel),
                    {"seed": seed_sentinel, "device": device_sentinel},
                ))
                self.assertEqual(optional_import_attempts, [])

            self.assertEqual(
                {name: original_case.__dict__.get(name, missing) for name in alias_names},
                original_alias_state,
            )

    def test_help_is_zero_effect_and_unreviewed_run_stops_before_docker(self):
        script = ROOT / "scripts/neural/run-task5-collection.sh"
        help_result = subprocess.run(
            ["bash", str(script), "--help"], capture_output=True, text=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--help", help_result.stdout)
        environment = dict(os.environ)
        environment.pop("SPIRE_LEARN_TASK5_REVIEWED", None)
        environment.pop("SPIRE_LEARN_TASK5_RUN_ID", None)
        held_result = subprocess.run(
            ["bash", str(script), "--run"], capture_output=True, text=True,
            check=False, env=environment,
        )
        self.assertEqual(held_result.returncode, 2)
        self.assertIn("held for internal parent and independent review", held_result.stderr)

    def test_shell_wrapper_and_embedded_container_command_parse(self):
        script = ROOT / "scripts/neural/run-task5-collection.sh"
        bash_result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False,
        )
        self.assertEqual(bash_result.returncode, 0, bash_result.stderr)
        source = script.read_text(encoding="utf-8")
        self.assertIn(
            'CONTROL_TOOL="$ROOT/scripts/neural/task5_collection_control.py"', source,
        )
        self.assertIn(
            '--mount "type=bind,src=$CONTROL_TOOL,dst=/learner/scripts/neural/task5_collection_control.py,readonly"',
            source,
        )
        self.assertNotIn('--mount "type=bind,src=$ROOT/scripts,dst=/learner/scripts', source)
        match = re.search(
            r"CONTAINER_COMMAND=\$\(cat <<'COMMAND'\n(.*?)\nCOMMAND\n\)",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        shell_result = subprocess.run(
            ["/bin/sh", "-n"], input=match.group(1), capture_output=True,
            text=True, check=False,
        )
        self.assertEqual(shell_result.returncode, 0, shell_result.stderr)

    def test_cleanup_timeout_sum_and_signal_traps_leave_room_inside_ninety_seconds(self):
        source = (ROOT / "scripts/neural/run-task5-collection.sh").read_text(encoding="utf-8")
        self.assertIn("timeout --signal=TERM --kill-after=90s 210s", source)
        self.assertIn("trap '' INT TERM", source)
        self.assertIn("trap - EXIT", source)
        self.assertEqual(control.cleanup_timeout_worst_case_seconds(source), 89)
        mutated = source.replace("if record source-compare", "if record source-after", 1)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            control.cleanup_timeout_worst_case_seconds(mutated)


if __name__ == "__main__":
    unittest.main()
