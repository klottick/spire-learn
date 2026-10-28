from __future__ import annotations

import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/neural"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import task8_calibration_run_helpers as helpers


HELPER_PATH = SCRIPTS / "task8_calibration_run_helpers.py"


class CalibrationHostGateTests(unittest.TestCase):
    def test_storage_preflight_allows_exact_cuda_and_cpu_venv_links_and_reserves_remaining_cap(self):
        with _StorageCheckout() as checkout:
            budget = checkout.verify_storage()

            self.assertGreater(budget["new_gpu_bytes"], 0)
            self.assertEqual(
                budget["new_gpu_bytes"] + budget["reservation_bytes"], 1024**3,
            )
            self.assertLessEqual(budget["neural_projected_bytes"], 40 * 1024**3)
            self.assertEqual(budget["free_floor_bytes"], 30 * 1024**3)
            self.assertEqual(os.readlink(checkout.runtime_venv / "lib64"), "lib")
            self.assertEqual(os.readlink(checkout.cpu_venv / "lib64"), "lib")
            self.assertEqual(os.readlink(checkout.cpu_venv / "bin/python"), "/usr/local/bin/python")
            self.assertEqual(os.readlink(checkout.cpu_venv / "bin/python3"), "python")
            self.assertEqual(os.readlink(checkout.cpu_venv / "bin/python3.12"), "python")

    def test_storage_preflight_reserves_exact_cap_and_rejects_oversize(self):
        with _StorageCheckout() as checkout:
            checkout.set_log_size(1024**3)
            exact = checkout.verify_storage()
            self.assertEqual(exact["new_gpu_bytes"], 1024**3)
            self.assertEqual(exact["reservation_bytes"], 0)

            checkout.set_log_size(1024**3 + 1)
            with self.assertRaisesRegex(helpers.CalibrationControlError, "1 GiB"):
                checkout.verify_storage()

    def test_wrapper_storage_argv_runs_through_real_cli_with_temporary_checkout(self):
        with _StorageCheckout() as checkout:
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(HELPER_PATH), "verify-storage",
                    str(checkout.root), str(checkout.run_root), str(checkout.control_root),
                    str(checkout.admin_root), str(checkout.output_child),
                ],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            budget = json.loads(completed.stdout)
            self.assertEqual(
                budget["new_gpu_bytes"] + budget["reservation_bytes"], 1024**3,
            )
            self.assertGreaterEqual(budget["free_bytes"], 30 * 1024**3)
            self.assertLessEqual(budget["neural_projected_bytes"], 40 * 1024**3)

    def test_final_storage_check_keeps_explicit_inventory_reservation(self):
        with _StorageCheckout() as checkout:
            checkout.set_log_size(1024**3 - 16 * 1024**2)
            exact = checkout.verify_storage(reservation_bytes=16 * 1024**2)
            self.assertEqual(exact["reservation_bytes"], 16 * 1024**2)

            checkout.set_log_size(1024**3 - 16 * 1024**2 + 1)
            with self.assertRaisesRegex(helpers.CalibrationControlError, "1 GiB"):
                checkout.verify_storage(reservation_bytes=16 * 1024**2)

    def test_storage_preflight_rejects_non_sibling_lib64_targets(self):
        with _StorageCheckout() as checkout:
            (checkout.cpu_venv / "lib64").unlink()
            (checkout.cpu_venv / "lib64").symlink_to(checkout.runtime_venv / "lib")
            with self.assertRaisesRegex(helpers.CalibrationControlError, "lib64|symlink"):
                checkout.verify_storage()

    def test_wrapper_inventory_argv_runs_through_real_cli_and_hashes_closed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runtime/run-01"
            control_root = root / "runtime-control/run-01"
            admin_root = root / "runtime-admin/run-01"
            run_root.mkdir(parents=True)
            control_root.mkdir(parents=True)
            admin_root.mkdir(parents=True)
            run_file = run_root / "closed-run.log"
            control_file = control_root / "closed-control.log"
            run_file.write_bytes(b"closed run output\n")
            control_file.write_bytes(b"closed control output\n")
            inventory = admin_root / "evidence-inventory.json"
            argv = [
                sys.executable, "-B", str(HELPER_PATH), "inventory-closed", str(inventory),
                "--logs-closed", str(run_root), str(control_root),
            ]
            completed = subprocess.run(
                argv, cwd=ROOT, text=True, capture_output=True, check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(completed.stdout)
            record = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertTrue(record["logs_closed_before_inventory"])
            self.assertEqual(result["sha256"], hashlib.sha256(inventory.read_bytes()).hexdigest())
            for evidence_file in (run_file, control_file):
                entry = next(
                    item for item in record["entries"]
                    if item["root"] == str(evidence_file.parent)
                    and item["path"] == evidence_file.name
                )
                self.assertEqual(entry["sha256"], hashlib.sha256(evidence_file.read_bytes()).hexdigest())

            missing_flag_output = admin_root / "missing-flag-inventory.json"
            missing_flag = subprocess.run(
                [sys.executable, "-B", str(HELPER_PATH), "inventory-closed",
                 str(missing_flag_output), str(run_root), str(control_root)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(missing_flag.returncode, 2)
            self.assertIn("--logs-closed", missing_flag.stderr)
            self.assertFalse(missing_flag_output.exists())

    def test_wrapper_lifecycle_numeric_argv_runs_through_real_cli_with_release_provenance(self):
        binding = {
            "learner_head": "d" * 40,
            "learner_dirty": True,
            "release_sha256": "b" * 64,
            **_release_binding(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding_path = root / "release-binding.json"
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            for suffix, wrapper_status, cleanup_failed, expected_status in (
                ("success", 0, 0, "complete"),
                ("failure", 2, 1, "failed"),
            ):
                output = root / f"host-lifecycle-{suffix}.json"
                argv = [
                    sys.executable, "-B", str(HELPER_PATH), "write-host-lifecycle",
                    str(output), str(time.time()), str(wrapper_status), str(cleanup_failed),
                    "--release-binding", str(binding_path),
                ]
                completed = subprocess.run(
                    argv, cwd=ROOT, text=True, capture_output=True, check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                record = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(completed.stdout), record)
                self.assertEqual(record["status"], expected_status)
                self.assertEqual(record["wrapper_exit_code"], wrapper_status)
                self.assertIs(record["cleanup_failed"], bool(cleanup_failed))
                self.assertEqual(record["learner_head"], binding["learner_head"])
                self.assertTrue(record["learner_dirty"])
                self.assertEqual(record["provenance_status"], "verified_parent_release")
                self.assertEqual(record["source_snapshot_sha256"], binding["source_snapshot_sha256"])
                self.assertEqual(record["release_sha256"], binding["release_sha256"])
                self.assertEqual(record["release_binding_sha256"], hashlib.sha256(
                    binding_path.read_bytes()).hexdigest())

    def test_wrapper_and_runbook_use_fixed_v3_continuation_paths(self):
        wrapper = (SCRIPTS / "run-task8-calibration.sh").read_text(encoding="utf-8")
        runbook = (SCRIPTS / "task8-calibration-runbook.md").read_text(encoding="utf-8")
        for name in (
            "calibration-source-snapshot-v3.json", "phase-release-v3.json",
            "phase-release-continuation-1.claim.json",
        ):
            self.assertIn(name, wrapper)
            self.assertIn(name, runbook)
        self.assertIn("claim-continuation", wrapper)
        self.assertIn(
            'RELEASE_CLAIM="$CALIBRATION_ROOT/phase-release-continuation-1.claim.json"', wrapper,
        )

    def test_authored_config_and_retained_qualification_match_exact_hashes(self):
        paths = helpers._expected_input_paths(ROOT)
        config = helpers.verify_calibration_config(paths["calibration_config"], paths["train_manifest"])
        qualification = helpers.verify_qualification(
            paths["qualification_summary"], paths["qualification_report"], paths["cuda_lock"],
        )
        self.assertEqual(config["train_manifest_sha256"], helpers._sha256(paths["train_manifest"]))
        self.assertEqual(qualification["cuda_lock_sha256"], helpers.LOCK_SHA256)
        self.assertEqual(qualification["native_sha256"], helpers.NATIVE_SHA256)
        self.assertEqual(qualification["qualification_summary_sha256"], helpers._sha256(paths["qualification_summary"]))

    def test_release_absence_fails_before_any_container_creation(self):
        with tempfile.TemporaryDirectory(prefix="task8-release-absence-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert test checkout marker", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration_root = task_root / "calibration"
            calibration_root.mkdir(parents=True)
            release_path = calibration_root / "phase-release-v3.json"
            snapshot_path = calibration_root / "calibration-source-snapshot-v3.json"

            # A parent may create valid v3 inputs in ROOT after review. This
            # negative case is deliberately bound to its isolated checkout.
            self.assertNotEqual(root, ROOT)
            self.assertNotEqual(
                release_path,
                ROOT / helpers.TASK_RELATIVE / "calibration/phase-release-v3.json",
            )
            with self.assertRaises(helpers.CalibrationControlError):
                helpers.verify_release(root, release_path, snapshot_path)
            self.assertFalse(release_path.exists())
            self.assertFalse(snapshot_path.exists())

    def test_gpu_inventory_requires_exact_device_and_twelve_gib_free(self):
        accepted = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d, NVIDIA GeForce RTX 5090, 595.91.07, 32607, 12288\n"
        self.assertEqual(helpers.verify_gpu_inventory_csv(accepted)["memory_free_mib"], 12288)
        for invalid in (
            accepted.replace("12288", "12287"),
            accepted.replace("GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d", "GPU-other"),
            accepted + accepted,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(helpers.CalibrationControlError):
                helpers.verify_gpu_inventory_csv(invalid)

    def test_exact_name_and_create_id_ownership_gates_fail_closed(self):
        helpers.assert_name_absent("\n")
        with self.assertRaises(helpers.CalibrationControlError):
            helpers.assert_name_absent("0123456789abcdef\n")
        self.assertEqual(helpers.parse_created_container_id("a" * 64 + "\n"), "a" * 64)
        for output in ("", "a" * 12, "a" * 64 + "\n" + "b" * 64):
            with self.subTest(output=output), self.assertRaises(helpers.CalibrationControlError):
                helpers.parse_created_container_id(output)

    def test_calibration_container_argv_has_one_narrow_writable_mount(self):
        with _ProjectedCheckout() as checkout:
            binding = _release_binding()
            argv = helpers.build_container_argv(
                checkout.root, checkout.output_child, "run-01",
                image_id="sha256:" + "b" * 64, release_binding=binding,
            )
            self.assertEqual(argv[:4], ("docker", "create", "--pull=never", "--name"))
            for value in (
                "--network", "none", "--read-only", "--user", "1000:1000", "--cpus", "4",
                "--memory", "8g", "--memory-swap", "8g", "--pids-limit", "512", "--cap-drop",
                "ALL", "--security-opt", "no-new-privileges", "--gpus",
                f"device={helpers.UUID}", "--entrypoint", "/bin/sh", helpers.IMAGE_ALIAS,
            ):
                self.assertIn(value, argv)
            env_values = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--env"]
            self.assertFalse(any("previous_" in value or "continuation_generation" in value
                                 for value in env_values))
            self.assertNotIn("--privileged", argv)
            self.assertNotIn("--network=host", argv)
            self.assertNotIn("/var/run/docker.sock", argv)
            self.assertNotIn(str(checkout.root), argv)
            mount_values = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--mount"]
            writable = [item for item in mount_values if not item.endswith(",readonly")]
            self.assertEqual(len(writable), 1)
            self.assertEqual(
                writable[0],
                f"type=bind,src={checkout.output_child},dst=/learner/artifacts/development/neural-combat/task-8/calibration/outputs",
            )
            self.assertTrue(any("dst=/runtime/venv-cuda,readonly" in item for item in mount_values))
            self.assertTrue(any("dst=/learner/evidence/requirements-cuda.lock,readonly" in item for item in mount_values))

    def test_inspect_fixture_proves_owner_limits_exact_mounts_and_uuid(self):
        with _ProjectedCheckout() as checkout:
            output_child = checkout.root / "artifacts/development/neural-combat/task-8/calibration/runtime/run-02/output-child"
            output_child.mkdir(parents=True)
            binding = _release_binding()
            argv = helpers.build_container_argv(
                checkout.root, output_child, "run-02",
                image_id="sha256:" + "b" * 64, release_binding=binding,
            )
            mounts = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--mount"]
            env = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--env"]
            mount_rows = []
            for mount in mounts:
                parts = dict(item.split("=", 1) for item in mount.split(",") if "=" in item)
                mount_rows.append({
                    "Type": "bind", "Source": parts["src"], "Destination": parts["dst"],
                    "RW": "readonly" not in mount,
                })
            container_id = "c" * 64
            value = [{
                "Id": container_id,
                "Name": "/" + helpers.container_name("run-02"),
                "Image": "sha256:" + "b" * 64,
                "Config": {
                    "User": "1000:1000", "WorkingDir": "/learner", "Entrypoint": ["/bin/sh"],
                    "Cmd": ["-ec", helpers.CALIBRATION_SCRIPT], "Env": [*env, "PATH=/usr/local/bin"],
                    "Labels": {"spire-learn.task": "task8-calibration", "spire-learn.run-id": "run-02"},
                },
                "HostConfig": {
                    "NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                    "AutoRemove": False, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                    "NanoCpus": 4_000_000_000, "Memory": 8 * 1024**3, "MemorySwap": 8 * 1024**3,
                    "PidsLimit": 512, "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
                    "DeviceRequests": [{"DeviceIDs": [helpers.UUID], "Capabilities": [["gpu"]],
                                        "Count": 0, "Options": {}}],
                },
                "Mounts": mount_rows,
            }]
            raw = json.dumps(value)
            checked = helpers.assert_container_inspect(
                raw, container_id=container_id, run_id="run-02", image_id="sha256:" + "b" * 64,
                root=checkout.root, output_child=output_child,
            )
            self.assertEqual(checked["gpu_uuid"], helpers.UUID)
            self.assertTrue(checked["read_only_rootfs"])
            self.assertTrue(helpers.assert_container_owner(raw, container_id=container_id, run_id="run-02")["owned"])
            value[0]["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-wrong"]
            with self.assertRaises(helpers.CalibrationControlError):
                helpers.assert_container_inspect(
                    json.dumps(value), container_id=container_id, run_id="run-02",
                    image_id="sha256:" + "b" * 64, root=checkout.root,
                    output_child=output_child,
                )

    def test_container_remove_requires_exited_owner_and_zero_exit(self):
        container_id = "c" * 64
        raw = json.dumps([{
            "Id": container_id,
            "Name": "/" + helpers.container_name("run-03"),
            "Config": {"Labels": {"spire-learn.task": "task8-calibration", "spire-learn.run-id": "run-03"}},
            "State": {"Status": "exited", "Running": False, "OOMKilled": False, "ExitCode": 0},
        }])
        self.assertEqual(helpers.assert_container_finished(raw, container_id=container_id, run_id="run-03")["exit_code"], 0)
        self.assertFalse(helpers.container_is_running(raw, container_id=container_id, run_id="run-03"))
        with self.assertRaises(helpers.CalibrationControlError):
            helpers.assert_container_finished(raw, container_id="d" * 64, run_id="run-03")
        changed = json.loads(raw)
        changed[0]["State"]["ExitCode"] = 143
        with self.assertRaises(helpers.CalibrationControlError):
            helpers.assert_container_finished(json.dumps(changed), container_id=container_id, run_id="run-03")

    def test_transcript_requires_one_executed_exact_label_and_zero_skips(self):
        label = helpers.TEST_LABEL
        good = f"{label} ... ok\n\nRan 1 test in 0.010s\n\nOK\n"
        self.assertEqual(helpers.verify_unittest_transcript(good)["observed_labels"], [label])
        for bad in (
            good.replace("Ran 1 test", "Ran 2 tests"),
            good.replace("... ok", "... skipped 'gate disabled'"),
            good.replace("... ok", "... FAIL"),
            good.replace(label, "test_other (tests.integration.Other.test_other)"),
        ):
            with self.subTest(bad=bad), self.assertRaises(helpers.CalibrationControlError):
                helpers.verify_unittest_transcript(bad)

    def test_closed_inventory_requires_closed_logs_and_excludes_its_own_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            (root / "transcript.log").write_text("closed", encoding="utf-8")
            inventory = Path(directory) / "admin/evidence-inventory.json"
            inventory.parent.mkdir()
            with self.assertRaises(helpers.CalibrationControlError):
                helpers.write_closed_inventory((root,), inventory, all_logs_closed=False)
            result = helpers.write_closed_inventory((root,), inventory, all_logs_closed=True)
            value = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertTrue(value["logs_closed_before_inventory"])
            self.assertEqual(value["roots"], [str(root)])
            self.assertEqual(result["entry_count"], len(value["entries"]))
            self.assertFalse(any(entry["path"].endswith("evidence-inventory.json") for entry in value["entries"]))
            (root / "nested").mkdir()
            with self.assertRaises(helpers.CalibrationControlError):
                helpers.write_closed_inventory((root, root / "nested"), Path(directory) / "admin/other.json",
                                               all_logs_closed=True)

    def test_host_lifecycle_records_one_deadline_through_cleanup_and_inventory(self):
        record = helpers.host_lifecycle_record(
            100.0, 710.0, wrapper_exit_code=2, cleanup_failed=True,
        )
        self.assertEqual(record["elapsed_seconds_including_container_cleanup_and_inventory"], 610.0)
        self.assertEqual(record["wrapper_elapsed_after_300_seconds"], 310.0)
        self.assertEqual(record["hard_overshoot_seconds"], 10.0)
        self.assertEqual(record["wrapper_term_deadline_seconds"], 510)
        self.assertEqual(record["wrapper_kill_deadline_seconds"], 600)
        self.assertEqual(record["status"], "failed")
        with self.assertRaises(helpers.CalibrationControlError):
            helpers.host_lifecycle_record(101.0, 100.0, wrapper_exit_code=0, cleanup_failed=False)

    def test_learner_git_identity_uses_bounded_read_only_host_calls(self):
        head = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("fixture metadata marker", encoding="utf-8")
            results = (
                subprocess.CompletedProcess([], 0, stdout=head + "\n", stderr=""),
                subprocess.CompletedProcess([], 0, stdout=" M src/spire_learn/neural/calibration.py\n", stderr=""),
            )
            with patch.object(helpers.subprocess, "run", side_effect=results) as run:
                identity = helpers.read_learner_git_identity(root)

        self.assertEqual(identity, {"learner_head": head, "learner_dirty": True})
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], ["git", "rev-parse", "--verify", "HEAD"])
        self.assertEqual(run.call_args_list[1].args[0], [
            "git", "status", "--porcelain=v1", "--untracked-files=all",
        ])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 5)
            self.assertEqual(call.kwargs["capture_output"], True)
            self.assertEqual(call.kwargs["cwd"], root)
            self.assertEqual(call.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")

    def test_learner_git_release_binding_rejects_stale_or_malformed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("fixture metadata marker", encoding="utf-8")
            identity = {"learner_head": "b" * 40, "learner_dirty": True}
            with patch.object(helpers, "read_learner_git_identity", return_value=identity):
                self.assertEqual(helpers.verify_learner_git_release(root, identity), identity)
                for stale in (
                    {"learner_head": "c" * 40, "learner_dirty": True},
                    {"learner_head": "b" * 40, "learner_dirty": False},
                    {"learner_head": "b" * 40, "learner_dirty": "true"},
                ):
                    with self.subTest(stale=stale), self.assertRaises(helpers.CalibrationControlError):
                        helpers.verify_learner_git_release(root, stale)

    def test_host_lifecycle_retains_git_and_release_hash_provenance(self):
        binding = {
            "learner_head": "d" * 40,
            "learner_dirty": True,
            "release_sha256": "b" * 64,
            "release_binding_sha256": "c" * 64,
            **_release_binding(),
        }
        record = helpers.host_lifecycle_record(
            100.0, 120.0, wrapper_exit_code=0, cleanup_failed=False,
            release_binding=binding,
        )
        for key in (
            "learner_head", "learner_dirty", "source_snapshot_sha256", "release_sha256",
            "release_binding_sha256",
        ):
            value = binding[key]
            self.assertEqual(record[key], value)
        self.assertEqual(record["provenance_status"], "verified_parent_release")
        self.assertEqual(record["continuation"]["generation"], 1)
        self.assertEqual(record["continuation"]["previous_run_id"], helpers.PREDECESSOR_RUN_ID)
        for key, value in helpers.PREDECESSOR_EVIDENCE_HASHES.items():
            self.assertEqual(record["continuation"][key], value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding_path = root / "release-binding.json"
            binding_path.write_text(json.dumps({key: value for key, value in binding.items()
                                                if key != "release_binding_sha256"}), encoding="utf-8")
            output_path = root / "host-lifecycle.json"
            written = helpers.write_host_lifecycle(
                output_path, 100.0, wrapper_exit_code=0, cleanup_failed=False,
                finished_epoch=120.0, release_binding_path=binding_path,
            )
            self.assertEqual(written["release_binding_sha256"], helpers._sha256(binding_path))
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["learner_head"],
                binding["learner_head"],
            )

    def test_new_host_module_imports_without_optional_runtime_import_attempts(self):
        program = r'''
import builtins
import sys
sys.path.insert(0, "scripts/neural")
blocked = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        attempts.append(name)
        raise AssertionError("optional import attempted: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import task8_calibration_run_helpers
assert attempts == [], attempts
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", program], cwd=ROOT, env=env,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_enabled_harness_binds_one_calibrate_call_in_projected_import_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            guest = Path(directory) / "learner"
            evidence = Path(directory) / "evidence"
            (guest / "configs/neural").mkdir(parents=True)
            (guest / "configs/cases/neural").mkdir(parents=True)
            (guest / "scripts/neural").mkdir(parents=True)
            evidence.mkdir()
            (guest / "src").symlink_to(ROOT / "src", target_is_directory=True)
            (guest / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
            for relative in (
                "configs/neural/calibration.toml",
                "configs/cases/neural/train.json",
                "scripts/neural/task8_calibration_run_helpers.py",
                "scripts/neural/task8_gpu_setup_helpers.py",
            ):
                (guest / relative).symlink_to(ROOT / relative)
            paths = helpers._expected_input_paths(ROOT)
            evidence_files = {
                "requirements-cuda.lock": paths["cuda_lock"],
                "task-8-gpu-qualification-report.md": paths["qualification_report"],
                "task-8-gpu-qualification-summary.json": paths["qualification_summary"],
            }
            for name, source in evidence_files.items():
                (evidence / name).symlink_to(source)

            run_dir = Path(directory) / "fake-calibration-output"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "lifecycle.json").write_text("{}\n", encoding="utf-8")
            binding = {
                "source_snapshot_sha256": "a" * 64,
                "runbook_sha256": "b" * 64,
                "calibration_config_file_sha256": helpers._sha256(paths["calibration_config"]),
                "train_manifest_sha256": helpers._sha256(paths["train_manifest"]),
                "qualification_report_sha256": helpers._sha256(paths["qualification_report"]),
                "qualification_summary_sha256": helpers._sha256(paths["qualification_summary"]),
                "cuda_lock_sha256": helpers.LOCK_SHA256,
                "native_sha256": helpers.NATIVE_SHA256,
            }
            program = r'''
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

numpy = types.ModuleType("numpy")
numpy.__version__ = "2.5.4"
gymnasium = types.ModuleType("gymnasium")
gymnasium.__version__ = "1.4.0"
torch = types.ModuleType("torch")
torch.__version__ = "2.14.1+cu130"
torch.version = types.SimpleNamespace(cuda="13.0")
thread_counts = {"intra": 0, "inter": 0}
torch.set_num_threads = lambda value: thread_counts.__setitem__("intra", value)
torch.set_num_interop_threads = lambda value: thread_counts.__setitem__("inter", value)
torch.get_num_threads = lambda: thread_counts["intra"]
torch.get_num_interop_threads = lambda: thread_counts["inter"]
sys.modules.update({"numpy": numpy, "gymnasium": gymnasium, "torch": torch})
versions = {"torch": "2.14.1+cu130", "numpy": "2.5.4", "gymnasium": "1.4.0", "spire-lab": "0.1.0"}
importlib.metadata.version = lambda name: versions[name]

from spire_learn.neural.runner import NeuralReport
from tests.integration.test_neural_calibration import NeuralCalibrationIntegrationTests
names = ("standalone-b4-h32", "standalone-b8-h32", "standalone-b16-h32", "run-b1-h128")
slots = (128, 256, 512, 128)
trials = [{"trial_id": name, "fully_tested": True, "status": "complete",
           "reserved_action_slots": reserved,
           "collection": {"delivered_environment_slots": reserved,
                          "attempted_environment_slots": reserved, "faults": 0,
                          "uncertain_environment_slots": 0, "environment_truncations": 0,
                          "environment_terminations": 0, "stop_reason": "horizon"},
           "optimizer_probe": None if index == 3 else {
               "attempted_steps": 1, "accepted_steps": 1, "uncertain_steps": 0,
               "finite": True, "parameters_changed": True}}
          for index, (name, reserved) in enumerate(zip(names, slots))]
summary = {"status": "complete", "cohort": {"reserved_action_slots": 1024,
           "attempted_environment_slots": 1024, "delivered_environment_slots": 1024,
           "uncertain_environment_slots": 0, "optimizer_accepted_steps": 3},
           "trials": trials,
           "selection": {"trial_id": "standalone-b16-h32"}}
report = NeuralReport(summary, 0, Path(os.environ["FAKE_CALIBRATION_RUN_DIR"]))
suite = unittest.defaultTestLoader.loadTestsFromTestCase(NeuralCalibrationIntegrationTests)
with patch("spire_learn.neural.runner.calibrate", return_value=report) as calibrate:
    result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.testsRun == 1 and result.failures == [] and result.errors == []
assert not result.skipped, result.skipped
assert calibrate.call_count == 1
assert calibrate.call_args.args[0].operation == "calibrate"
assert calibrate.call_args.args[0].split == "train"
'''
            env = dict(os.environ)
            env.update({
                "PYTHONPATH": f"{guest / 'src'}:{guest}",
                "SPIRE_LEARN_RUN_TASK8_CALIBRATION": "1",
                "SPIRE_LEARN_TASK8_GPU_UUID": helpers.UUID,
                "SPIRE_LEARN_TASK8_EVIDENCE_DIR": str(evidence),
                "FAKE_CALIBRATION_RUN_DIR": str(run_dir),
                **{key: value for key, value in (
                    ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                    ("OPENBLAS_NUM_THREADS", "4"), ("NUMEXPR_NUM_THREADS", "4"),
                )},
            })
            env.update({
                "SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256": binding["source_snapshot_sha256"],
                "SPIRE_LEARN_TASK8_RUNBOOK_SHA256": binding["runbook_sha256"],
                "SPIRE_LEARN_TASK8_CONFIG_SHA256": binding["calibration_config_file_sha256"],
                "SPIRE_LEARN_TASK8_MANIFEST_SHA256": binding["train_manifest_sha256"],
                "SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256": binding["qualification_report_sha256"],
                "SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256": binding["qualification_summary_sha256"],
                "SPIRE_LEARN_TASK8_LOCK_SHA256": binding["cuda_lock_sha256"],
                "SPIRE_LEARN_TASK8_NATIVE_SHA256": binding["native_sha256"],
            })
            completed = subprocess.run(
                [sys.executable, "-B", "-c", program], cwd=guest, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class CalibrationContinuationControlTests(unittest.TestCase):
    def test_real_named_predecessor_is_read_only_and_matches_the_reviewed_failure(self):
        evidence_paths = {
            key: ROOT / helpers.TASK_RELATIVE / relative
            for key, relative in helpers.PREDECESSOR_EVIDENCE_PATHS.items()
        }
        before = {key: path.read_bytes() for key, path in evidence_paths.items()}

        evidence = helpers.verify_predecessor_evidence(ROOT)

        self.assertEqual(evidence["previous_run_id"], helpers.PREDECESSOR_RUN_ID)
        self.assertEqual(evidence["previous_release_sha256"], helpers.PREDECESSOR_RELEASE_SHA256)
        self.assertEqual({key: path.read_bytes() for key, path in evidence_paths.items()}, before)
        completed = subprocess.run(
            [sys.executable, "-B", str(HELPER_PATH), "verify-predecessor", str(ROOT)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(json.loads(completed.stdout), evidence)

    def test_predecessor_rejects_missing_modified_symlink_wrong_claim_and_effects(self):
        with tempfile.TemporaryDirectory(prefix="task8-predecessor-fixtures-") as directory:
            root = _copy_predecessor_fixture(Path(directory) / "missing")
            claim = root / helpers.TASK_RELATIVE / helpers.PREDECESSOR_EVIDENCE_PATHS[
                "previous_claim_sha256"
            ]
            claim.unlink()
            with self.assertRaisesRegex(helpers.CalibrationControlError, "missing"):
                helpers.verify_predecessor_evidence(root, hash_file=_fixture_hash)

            root = _copy_predecessor_fixture(Path(directory) / "modified")
            proof = root / helpers.TASK_RELATIVE / helpers.PREDECESSOR_EVIDENCE_PATHS[
                "previous_failure_proof_sha256"
            ]
            proof.write_bytes(proof.read_bytes() + b"\nmodified\n")
            with self.assertRaisesRegex(helpers.CalibrationControlError, "hash differs"):
                helpers.verify_predecessor_evidence(root, hash_file=_fixture_hash_modified)

            root = _copy_predecessor_fixture(Path(directory) / "symlink")
            claim = root / helpers.TASK_RELATIVE / helpers.PREDECESSOR_EVIDENCE_PATHS[
                "previous_claim_sha256"
            ]
            elsewhere = root / "claim-copy.json"
            shutil.copyfile(claim, elsewhere)
            claim.unlink()
            claim.symlink_to(elsewhere)
            with self.assertRaisesRegex(helpers.CalibrationControlError, "symlink"):
                helpers.verify_predecessor_evidence(root, hash_file=_fixture_hash)

            for mutation in ("wrong-run", "wrong-release", "positive-effect"):
                with self.subTest(mutation=mutation):
                    root = _copy_predecessor_fixture(Path(directory) / mutation)
                    task_root = root / helpers.TASK_RELATIVE
                    claim_path = task_root / helpers.PREDECESSOR_EVIDENCE_PATHS[
                        "previous_claim_sha256"
                    ]
                    proof_path = task_root / helpers.PREDECESSOR_EVIDENCE_PATHS[
                        "previous_failure_proof_sha256"
                    ]
                    if mutation in ("wrong-run", "wrong-release"):
                        claim_value = json.loads(claim_path.read_text(encoding="utf-8"))
                        claim_value["run_id" if mutation == "wrong-run" else "release_sha256"] = (
                            "wrong-run" if mutation == "wrong-run" else "0" * 64
                        )
                        claim_path.write_text(json.dumps(claim_value), encoding="utf-8")
                        expected_error = "exact failed run/release"
                    else:
                        proof_value = json.loads(proof_path.read_text(encoding="utf-8"))
                        proof_value["optimizer_calls"] = 1
                        proof_path.write_text(json.dumps(proof_value), encoding="utf-8")
                        expected_error = "calibration or native effect"
                    with self.assertRaisesRegex(helpers.CalibrationControlError, expected_error):
                        helpers.verify_predecessor_evidence(root, hash_file=_fixture_hash)

    def test_claim_continuation_cli_parses_and_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory(prefix="task8-claim-cli-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert fixture", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration_root = task_root / "calibration"
            calibration_root.mkdir(parents=True)
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(HELPER_PATH), "claim-continuation", str(root),
                    str(task_root / helpers.CONTINUATION_RELEASE_PATH),
                    str(task_root / helpers.CONTINUATION_SNAPSHOT_PATH),
                    str(task_root / helpers.CONTINUATION_CLAIM_PATH), "calibration-continuation-test",
                    "--expected-release-sha256", "a" * 64,
                ],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("required path is missing", completed.stderr)
            self.assertFalse((task_root / helpers.CONTINUATION_CLAIM_PATH).exists())

    def test_claim_continuation_cli_requires_a_captured_release_hash(self):
        with tempfile.TemporaryDirectory(prefix="task8-claim-cli-binding-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert fixture", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration_root = task_root / "calibration"
            calibration_root.mkdir(parents=True)
            base = [
                sys.executable, "-B", str(HELPER_PATH), "claim-continuation", str(root),
                str(task_root / helpers.CONTINUATION_RELEASE_PATH),
                str(task_root / helpers.CONTINUATION_SNAPSHOT_PATH),
                str(task_root / helpers.CONTINUATION_CLAIM_PATH), "calibration-continuation-test",
            ]
            cases = (
                (base, "--expected-release-sha256"),
                (base + ["--expected-release-sha256", "malformed"], "captured release SHA256"),
                (base + ["--expected-release-sha256", "a" * 64], "required path is missing"),
            )
            for argv, expected_error in cases:
                with self.subTest(expected_error=expected_error):
                    completed = subprocess.run(
                        argv, cwd=ROOT, text=True, capture_output=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn(expected_error, completed.stderr)
                    self.assertFalse((task_root / helpers.CONTINUATION_CLAIM_PATH).exists())

    def test_claim_output_cli_compares_fake_claim_to_captured_release_hash(self):
        with tempfile.TemporaryDirectory(prefix="task8-claim-output-cli-") as directory:
            claim_path = Path(directory) / "claim.stdout.json"
            captured = "a" * 64
            claim_path.write_text(json.dumps({"release_sha256": captured}), encoding="utf-8")
            claim_bytes = claim_path.read_bytes()
            completed = subprocess.run(
                [sys.executable, "-B", str(HELPER_PATH), "assert-claim-release-binding",
                 str(claim_path), captured],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            mismatch = subprocess.run(
                [sys.executable, "-B", str(HELPER_PATH), "assert-claim-release-binding",
                 str(claim_path), "b" * 64],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("differs from captured binding", mismatch.stderr)
            self.assertEqual(claim_path.read_bytes(), claim_bytes)

    def test_continuation_claim_is_exclusive_binds_predecessor_and_preserves_original(self):
        self.assertIn(
            "expected_release_sha256",
            inspect.signature(helpers.claim_continuation_release).parameters,
        )
        with tempfile.TemporaryDirectory(prefix="task8-claim-exclusive-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert fixture", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration_root = task_root / "calibration"
            calibration_root.mkdir(parents=True)
            original_claim = calibration_root / "phase-release.claim.json"
            original_claim.write_bytes(b"consumed predecessor claim\n")
            release_path = task_root / helpers.CONTINUATION_RELEASE_PATH
            snapshot_path = task_root / helpers.CONTINUATION_SNAPSHOT_PATH
            marker_path = task_root / helpers.CONTINUATION_CLAIM_PATH
            binding = {
                "release_sha256": "f" * 64,
                "previous_run_id": helpers.PREDECESSOR_RUN_ID,
                "previous_release_sha256": helpers.PREDECESSOR_RELEASE_SHA256,
                **helpers.PREDECESSOR_EVIDENCE_HASHES,
            }
            captured_hash = "e" * 64
            claim_argv = [
                str(HELPER_PATH), "claim-continuation", str(root), str(release_path),
                str(snapshot_path), str(marker_path), "calibration-continuation-c",
                "--expected-release-sha256", captured_hash,
            ]
            mismatch_stderr = io.StringIO()
            with (
                patch.object(helpers, "verify_release", return_value=binding),
                patch.object(sys, "argv", claim_argv),
                patch.object(sys, "stderr", mismatch_stderr),
            ):
                self.assertEqual(helpers._cmd_main(), 2)
            self.assertIn("differs from captured binding", mismatch_stderr.getvalue())
            self.assertFalse(marker_path.exists())
            output = io.StringIO()
            with (
                patch.object(helpers, "verify_release", return_value=binding) as verify,
                patch.object(sys, "argv", claim_argv[:-1] + [binding["release_sha256"]]),
                patch.object(sys, "stdout", output),
            ):
                self.assertEqual(helpers._cmd_main(), 0)
                verify.assert_called_once_with(root, release_path, snapshot_path)
                claim = json.loads(output.getvalue())
                self.assertEqual(claim["release_sha256"], binding["release_sha256"])
                self.assertEqual(claim["previous_run_id"], helpers.PREDECESSOR_RUN_ID)
                for key, value in helpers.PREDECESSOR_EVIDENCE_HASHES.items():
                    self.assertEqual(claim[key], value)
                original_bytes = original_claim.read_bytes()
            with patch.object(helpers, "verify_release", return_value=binding):
                with self.assertRaisesRegex(helpers.CalibrationControlError, "already exists"):
                    helpers.claim_continuation_release(
                        root, release_path, snapshot_path, marker_path, "calibration-continuation-c",
                        expected_release_sha256=binding["release_sha256"],
                    )
                claimed_bytes = marker_path.read_bytes()
                with self.assertRaisesRegex(helpers.CalibrationControlError, "fresh run ID"):
                    helpers.claim_continuation_release(
                        root, release_path, snapshot_path, marker_path, helpers.PREDECESSOR_RUN_ID,
                        expected_release_sha256=binding["release_sha256"],
                    )
                self.assertEqual(marker_path.read_bytes(), claimed_bytes)
                alternate_marker = calibration_root / "phase-release-continuation-2.claim.json"
                with self.assertRaisesRegex(helpers.CalibrationControlError, "fixed generation-one"):
                    helpers.claim_continuation_release(
                        root, release_path, snapshot_path, alternate_marker, "calibration-continuation-d",
                        expected_release_sha256=binding["release_sha256"],
                    )
            self.assertEqual(original_claim.read_bytes(), original_bytes)

    def test_exclusive_write_fsyncs_file_then_parent_directory(self):
        with tempfile.TemporaryDirectory(prefix="task8-exclusive-fsync-order-") as directory:
            output = Path(directory) / "claim.json"
            observed = []

            def observe_fsync(fd):
                kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
                observed.append(kind)

            with patch.object(helpers.os, "fsync", side_effect=observe_fsync):
                helpers._write_exclusive(output, b"one-use claim\n")

            self.assertEqual(observed, ["file", "directory"])
            self.assertEqual(output.read_bytes(), b"one-use claim\n")

    def test_directory_fsync_failure_keeps_claim_and_blocks_precreate_gate(self):
        with tempfile.TemporaryDirectory(prefix="task8-exclusive-fsync-failure-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert fixture", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration_root = task_root / "calibration"
            calibration_root.mkdir(parents=True)
            original_claim = calibration_root / "phase-release.claim.json"
            original_claim.write_bytes(b"consumed predecessor claim\n")
            release_path = task_root / helpers.CONTINUATION_RELEASE_PATH
            snapshot_path = task_root / helpers.CONTINUATION_SNAPSHOT_PATH
            marker_path = task_root / helpers.CONTINUATION_CLAIM_PATH
            captured_hash = "a" * 64
            binding = {
                "release_sha256": captured_hash,
                "previous_run_id": helpers.PREDECESSOR_RUN_ID,
                "previous_release_sha256": helpers.PREDECESSOR_RELEASE_SHA256,
                **helpers.PREDECESSOR_EVIDENCE_HASHES,
            }
            argv = [
                str(HELPER_PATH), "claim-continuation", str(root), str(release_path),
                str(snapshot_path), str(marker_path), "calibration-continuation-fsync",
                "--expected-release-sha256", captured_hash,
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(helpers, "verify_release", return_value=binding),
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", stdout),
                patch.object(sys, "stderr", stderr),
                patch.object(helpers.os, "fsync", side_effect=[None, OSError("directory fsync failed")]),
            ):
                self.assertEqual(helpers._cmd_main(), 2)
            self.assertIn("parent directory", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            claim_bytes = marker_path.read_bytes()
            claim = json.loads(claim_bytes)
            self.assertEqual(claim["release_sha256"], captured_hash)
            self.assertEqual(original_claim.read_bytes(), b"consumed predecessor claim\n")

            second_stderr = io.StringIO()
            with (
                patch.object(helpers, "verify_release", return_value=binding),
                patch.object(sys, "argv", argv),
                patch.object(sys, "stderr", second_stderr),
            ):
                self.assertEqual(helpers._cmd_main(), 2)
            self.assertIn("already exists", second_stderr.getvalue())
            self.assertEqual(marker_path.read_bytes(), claim_bytes)

            wrapper = (SCRIPTS / "run-task8-calibration.sh").read_text(encoding="utf-8")
            self.assertIn("record release-claim-binding-assert python3.12", wrapper)
            claim_call = wrapper.index('record release-single-use-claim python3.12')
            claim_assertion = wrapper.index('record release-claim-binding-assert python3.12')
            create_attempt = wrapper.index("CREATE_ATTEMPTED=1")
            self.assertLess(claim_call, claim_assertion)
            self.assertLess(claim_assertion, create_attempt)
            self.assertIn("--expected-release-sha256", wrapper[claim_call:claim_assertion])

    def test_v3_release_schema_binds_exact_predecessor_hashes(self):
        with tempfile.TemporaryDirectory(prefix="task8-v3-release-") as directory:
            root = Path(directory).resolve()
            (root / ".git").write_text("inert fixture", encoding="utf-8")
            task_root = root / helpers.TASK_RELATIVE
            calibration = task_root / "calibration"
            calibration.mkdir(parents=True)
            snapshot_path = task_root / helpers.CONTINUATION_SNAPSHOT_PATH
            release_path = task_root / helpers.CONTINUATION_RELEASE_PATH
            snapshot_path.write_text(
                json.dumps({"scope_roots": list(helpers.SOURCE_SCOPES)}), encoding="utf-8",
            )
            runbook_path = root / "scripts/neural/task8-calibration-runbook.md"
            runbook_path.parent.mkdir(parents=True)
            runbook_path.write_text("fixture runbook\n", encoding="utf-8")
            calibration_hashes = {
                "calibration_config_sha256": "c" * 64,
                "train_manifest_sha256": "d" * 64,
            }
            qualification_hashes = {
                "qualification_report_sha256": "e" * 64,
                "qualification_summary_sha256": "f" * 64,
                "cuda_lock_sha256": helpers.LOCK_SHA256,
                "native_sha256": helpers.NATIVE_SHA256,
            }
            git_identity = {"learner_head": "a" * 40, "learner_dirty": True}
            release = {
                "decision": "one_task8_calibration_cohort",
                "parent_source_review": "passed",
                "independent_review": "passed",
                "single_use": True,
                "release_id": "continuation-review-1",
                "source_snapshot_sha256": helpers._sha256(snapshot_path),
                "runbook_sha256": helpers._sha256(runbook_path),
                **calibration_hashes,
                **qualification_hashes,
                "gpu_uuid": helpers.UUID,
                **git_identity,
                **helpers.PREDECESSOR_EVIDENCE_HASHES,
            }
            self.assertEqual(set(release), helpers.RELEASE_FIELDS)
            release_path.write_text(json.dumps(release), encoding="utf-8")
            predecessor = {
                "previous_run_id": helpers.PREDECESSOR_RUN_ID,
                "previous_release_sha256": helpers.PREDECESSOR_RELEASE_SHA256,
                **helpers.PREDECESSOR_EVIDENCE_HASHES,
            }
            with (
                patch.object(helpers.setup, "validate_safe_task8_inputs",
                            return_value=(release_path, snapshot_path)),
                patch.object(helpers.setup, "verify_source_snapshot", return_value={"file_count": 4}),
                patch.object(helpers, "verify_predecessor_evidence", return_value=predecessor),
                patch.object(helpers, "verify_calibration_config", return_value=calibration_hashes),
                patch.object(helpers, "verify_qualification", return_value=qualification_hashes),
                patch.object(helpers, "verify_learner_git_release", return_value=git_identity),
            ):
                verified = helpers.verify_release(root, release_path, snapshot_path)
                self.assertEqual(verified["continuation_generation"], 1)
                self.assertEqual(verified["previous_run_id"], helpers.PREDECESSOR_RUN_ID)
                for key, value in helpers.PREDECESSOR_EVIDENCE_HASHES.items():
                    self.assertEqual(verified[key], value)

                release["previous_inventory_sha256"] = "0" * 64
                release_path.write_text(json.dumps(release), encoding="utf-8")
                with self.assertRaisesRegex(helpers.CalibrationControlError, "does not bind"):
                    helpers.verify_release(root, release_path, snapshot_path)


def _fixture_hash(path: Path) -> str:
    for key, expected_path in helpers.PREDECESSOR_EVIDENCE_PATHS.items():
        suffix = (helpers.TASK_RELATIVE / expected_path).parts
        if path.parts[-len(suffix):] == suffix:
            return helpers.PREDECESSOR_EVIDENCE_HASHES[key]
    raise AssertionError(f"unexpected predecessor fixture path: {path}")


def _fixture_hash_modified(path: Path) -> str:
    if path.name == "runtime-parent-proof.json":
        return "0" * 64
    return _fixture_hash(path)


def _copy_predecessor_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / ".git").write_text("inert fixture metadata", encoding="utf-8")
    task_root = root / helpers.TASK_RELATIVE
    for relative in helpers.PREDECESSOR_EVIDENCE_PATHS.values():
        destination = task_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / helpers.TASK_RELATIVE / relative
        shutil.copyfile(source, destination)
    return root


def _release_binding() -> dict[str, object]:
    return {
        name: (helpers.LOCK_SHA256 if name == "cuda_lock_sha256" else
               helpers.NATIVE_SHA256 if name == "native_sha256" else "a" * 64)
        for name in (
            "source_snapshot_sha256", "runbook_sha256", "calibration_config_sha256",
            "train_manifest_sha256", "qualification_report_sha256",
            "qualification_summary_sha256", "cuda_lock_sha256", "native_sha256",
        )
    } | {
        "continuation_generation": helpers.CONTINUATION_GENERATION,
        "previous_run_id": helpers.PREDECESSOR_RUN_ID,
        "previous_release_sha256": helpers.PREDECESSOR_RELEASE_SHA256,
        **helpers.PREDECESSOR_EVIDENCE_HASHES,
    }


class _ProjectedCheckout:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "checkout"

    def __enter__(self):
        self.root.mkdir()
        (self.root / ".git").write_text("test", encoding="utf-8")
        for relative in (
            "src", "tests", "configs/neural", "configs/cases/neural",
            "scripts/neural", "artifacts/development/neural-combat/task-8/runtime-output/"
            "cuda-setup-20261028-a/install-work/venv-cuda",
            "artifacts/development/neural-combat/task-8/runtime-output/"
            "cuda-setup-20261028-a/resolve-work",
            "artifacts/development/neural-combat/task-8/calibration/runtime/run-01/output-child",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        files = (
            "configs/neural/calibration.toml", "configs/cases/neural/train.json",
            "scripts/neural/task8_calibration_run_helpers.py", "scripts/neural/task8_gpu_setup_helpers.py",
            "artifacts/development/neural-combat/task-8/task-8-gpu-qualification-report.md",
            "artifacts/development/neural-combat/task-8/task-8-gpu-qualification-summary.json",
            "artifacts/development/neural-combat/task-8/runtime-output/cuda-setup-20261028-a/resolve-work/requirements-cuda.lock",
        )
        for relative in files:
            (self.root / relative).write_text("fixture", encoding="utf-8")
        (self.root / "scripts/neural/run-task8-calibration.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.root / "scripts/neural/task8-calibration-runbook.md").write_text("fixture\n", encoding="utf-8")
        self.output_child = self.root / "artifacts/development/neural-combat/task-8/calibration/runtime/run-01/output-child"
        return self

    def __exit__(self, exc_type, exc, tb):
        self.temp.cleanup()


class _StorageCheckout:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name) / "checkout"

    def __enter__(self):
        self.root.mkdir()
        (self.root / ".git").write_text("fixture metadata marker", encoding="utf-8")
        self.neural_root = self.root / "artifacts/development/neural-combat"
        task_root = self.neural_root / helpers.TASK_RELATIVE.name
        calibration_root = task_root / "calibration"
        self.run_root = calibration_root / "runtime/round2-run"
        self.control_root = calibration_root / "runtime-control/round2-run"
        self.admin_root = calibration_root / "runtime-admin/round2-run"
        self.output_child = self.run_root / "output-child"
        self.cpu_venv = (
            self.neural_root / "runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc/data/venv-cpu"
        )
        self.runtime_venv = (
            task_root / "runtime-output" / helpers.CUDA_RUN_ID / "install-work/venv-cuda"
        )
        for path in (
            self.run_root / "logs", self.control_root, self.admin_root, self.output_child,
            self.cpu_venv / "lib", self.cpu_venv / "bin", self.runtime_venv / "lib",
        ):
            path.mkdir(parents=True, exist_ok=True)
        (self.cpu_venv / "lib64").symlink_to("lib")
        (self.runtime_venv / "lib64").symlink_to("lib")
        for name, target in (
            ("python", "/usr/local/bin/python"),
            ("python3", "python"),
            ("python3.12", "python"),
        ):
            (self.cpu_venv / "bin" / name).symlink_to(target)
        self.log_path = self.run_root / "logs/storage-preflight.stdout.log"
        self.log_path.write_bytes(b"initial wrapper log\n")
        return self

    def set_log_size(self, size: int):
        with self.log_path.open("wb") as stream:
            stream.truncate(size)

    def verify_storage(self, *, reservation_bytes=None):
        gib = 1024**3
        disk = SimpleNamespace(total=200 * gib, used=100 * gib, free=100 * gib)
        with patch.object(helpers.setup.shutil, "disk_usage", return_value=disk):
            return helpers.verify_host_storage(
                self.root, self.run_root, self.control_root, self.admin_root, self.output_child,
                reservation_bytes=reservation_bytes,
            )

    def __exit__(self, exc_type, exc, tb):
        self.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
