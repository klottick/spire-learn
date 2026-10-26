"""Pure tests for the held Task 7 runtime wrapper's safety helpers."""

from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.neural import task7_training_run_helpers as helpers


def _process_identity(role, boundary):
    return {
        "role": role, "process_id": os.getpid(),
        "python": "3.12.15", "torch": "2.14.1+cpu", "torch_cuda": None,
        "torch_num_threads": 4, "torch_num_interop_threads": 1,
        "thread_environment": helpers.THREAD_ENVIRONMENT,
        "configuration_boundary": boundary,
    }


def _preflight_runtime_identity():
    return {
        "python": "3.12.15", "torch": "2.14.1+cpu", "numpy": "2.5.4",
        "gymnasium": "1.4.0", "torch_cuda": None,
        "preflight_process": _process_identity(
            "preflight_process", "preflight_process_before_pip_check",
        ),
        "training_process": None, "pip_check": "passed",
        "check_order": [
            "installed_versions", "preflight_process_thread_gate", "pip_check",
            "training_process_thread_gate", "unittest",
        ],
    }


def _qualified_runtime_identity():
    identity = _preflight_runtime_identity()
    identity["training_process"] = _process_identity(
        "unittest_training_process", "setUpClass_before_model_or_environment_factory",
    )
    return identity


class Task7RunHelperTests(unittest.TestCase):
    def test_output_scaffold_creates_all_parents_for_a_fresh_run_child(self):
        with tempfile.TemporaryDirectory() as directory:
            output_child = Path(directory) / "fresh-output"
            output_child.mkdir()
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(helpers.main(["prepare-output-tree", str(output_child)]), 0)
            expected = {
                output_child / "development",
                output_child / "development/neural-combat",
                output_child / "development/neural-combat/runs",
                output_child / "development/neural-combat/runs/standalone",
                output_child / "development/neural-combat/runs/run",
            }
            self.assertEqual(set(json.loads(output.getvalue())["created"]),
                             {str(path) for path in expected})
            self.assertTrue(all(path.is_dir() for path in expected))
            wrapper = (Path(__file__).resolve().parents[2]
                       / "scripts/neural/run-task7-training-smoke.sh").read_text(encoding="utf-8")
            self.assertLess(wrapper.index("trap cleanup EXIT"),
                            wrapper.index('mkdir -p "$TASK_ROOT/runtime"'))
            self.assertIn('python3.12 -B "$HELPER" prepare-output-tree "$OUTPUT_CHILD"', wrapper)
            self.assertNotIn('mkdir -m 700 "$OUTPUT_CHILD/development/neural-combat/runs/standalone"',
                             wrapper)

    def test_smoke_transcript_requires_exact_two_methods_and_zero_skips(self):
        stdout = """test_run_bounded_training_and_reload_parity (tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_run_bounded_training_and_reload_parity) ... ok
test_standalone_bounded_training_and_reload_parity (tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_standalone_bounded_training_and_reload_parity) ... ok

----------------------------------------------------------------------
Ran 2 tests in 1.23s

OK
"""
        identity = _preflight_runtime_identity()
        training_process = _qualified_runtime_identity()["training_process"]
        stderr = (
            "TASK7_RUNTIME_IDENTITY=" + json.dumps(identity, sort_keys=True) +
            "\nTASK7_PREFLIGHT_PROCESS_IDENTITY="
            + json.dumps(identity["preflight_process"], sort_keys=True) +
            "\nTASK7_PIP_CHECK=passed\nTASK7_TRAINING_PROCESS_IDENTITY="
            + json.dumps(training_process, sort_keys=True) + "\n" + stdout
        )
        self.assertEqual(helpers.parse_transcript("", stderr)["skipped"], 0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            helpers.parse_transcript("", stderr.replace("Ran 2 tests", "Ran 3 tests") +
                                     "test_unexpected (tests.Other.test_unexpected) ... ok\n")
        with self.assertRaisesRegex(ValueError, "skip"):
            helpers.parse_transcript("", stderr.replace("... ok", "... skipped 'gate'", 1))
        with self.assertRaisesRegex(ValueError, "preflight marker"):
            helpers.parse_transcript("", stdout)
        with self.assertRaisesRegex(ValueError, "before the first"):
            tests = stdout
            markers = stderr[:stderr.index("test_run_")]
            helpers.parse_transcript("", tests + markers)

    def test_transcript_rejects_preflight_only_thread_identity(self):
        stdout = """test_run_bounded_training_and_reload_parity (tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_run_bounded_training_and_reload_parity) ... ok
test_standalone_bounded_training_and_reload_parity (tests.integration.test_neural_training.NeuralTrainingSmokeTests.test_standalone_bounded_training_and_reload_parity) ... ok

----------------------------------------------------------------------
Ran 2 tests in 1.23s

OK
"""
        identity = _preflight_runtime_identity()
        stderr = (
            "TASK7_RUNTIME_IDENTITY=" + json.dumps(identity, sort_keys=True) +
            "\nTASK7_PREFLIGHT_PROCESS_IDENTITY="
            + json.dumps(identity["preflight_process"], sort_keys=True)
            + "\nTASK7_PIP_CHECK=passed\n" + stdout
        )
        with self.assertRaisesRegex(ValueError, "training process"):
            helpers.parse_transcript("", stderr)

    def test_training_process_identity_configures_strict_fake_before_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            identity_path = Path(directory) / "runtime-identity.json"
            identity_path.write_text(json.dumps(_preflight_runtime_identity()), encoding="utf-8")

            class StrictTorch:
                __version__ = "2.14.1+cpu"
                version = SimpleNamespace(cuda=None)

                def __init__(self):
                    self.calls = []
                    self.intra = None
                    self.inter = None

                def set_num_threads(self, value):
                    if value != 4:
                        raise AssertionError("training process must set four intra-op threads")
                    self.calls.append(("set_num_threads", value))
                    self.intra = value

                def set_num_interop_threads(self, value):
                    if value != 1:
                        raise AssertionError("training process must set one inter-op thread")
                    self.calls.append(("set_num_interop_threads", value))
                    self.inter = value

                def get_num_threads(self):
                    self.calls.append(("get_num_threads",))
                    return self.intra

                def get_num_interop_threads(self):
                    self.calls.append(("get_num_interop_threads",))
                    return self.inter

            torch = StrictTorch()
            original_import = builtins.__import__
            optional_prefixes = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}

            def blocked_import(name, *args, **kwargs):
                if name.split(".", 1)[0] in optional_prefixes:
                    raise AssertionError(f"optional import attempted: {name}")
                return original_import(name, *args, **kwargs)

            with patch.dict(os.environ, helpers.THREAD_ENVIRONMENT), \
                    patch("builtins.__import__", side_effect=blocked_import):
                record = helpers.record_training_process_identity(torch, identity_path)

            self.assertEqual(torch.calls, [
                ("set_num_threads", 4), ("set_num_interop_threads", 1),
                ("get_num_threads",), ("get_num_interop_threads",),
            ])
            self.assertEqual(record["role"], "unittest_training_process")
            self.assertEqual(
                record["configuration_boundary"],
                "setUpClass_before_model_or_environment_factory",
            )
            self.assertEqual(record["torch_num_threads"], 4)
            self.assertEqual(record["torch_num_interop_threads"], 1)
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            self.assertEqual(identity["training_process"], record)
            self.assertTrue(helpers.runtime_identity_is_qualified(identity))

    def test_source_bound_release_requires_review_and_exact_snapshot_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = {"head": "a" * 40, "task7_files_sha256": {"file": "b" * 64}}
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            digest = hashlib.sha256(
                json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            release = {
                "decision": "internal_cpu_smoke_release",
                "first_delivery_review": "passed",
                "fresh_independent_review": "passed",
                "task3_native_conformance": "accepted",
                "task5_native_collection": "accepted",
                "task7_source_snapshot_sha256": digest,
            }
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            self.assertEqual(helpers.verify_release(release_path, snapshot_path)["decision"],
                             "internal_cpu_smoke_release")
            release["fresh_independent_review"] = "pending"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "release record"):
                helpers.verify_release(release_path, snapshot_path)

    def test_pinned_image_alias_and_container_id_parsers_are_strict(self):
        helpers.verify_image_repo_digests(json.dumps([
            "python@sha256:" + helpers.IMAGE_DIGEST,
        ]))
        with self.assertRaisesRegex(ValueError, "digest"):
            helpers.verify_image_repo_digests(json.dumps(["python:3.12-slim"]))
        self.assertEqual(helpers.resolve_container_id("a" * 64 + "\n"), "a" * 64)
        self.assertEqual(helpers.resolve_container_id(""), "")
        with self.assertRaisesRegex(ValueError, "zero or one"):
            helpers.resolve_container_id("a" * 64 + "\n" + "b" * 64)

    def test_immutable_image_identity_requires_digest_and_linux_amd64(self):
        value = helpers.verify_image_identity(
            "sha256:" + "a" * 64 + "|linux|amd64|[\"python@sha256:" + helpers.IMAGE_DIGEST + "\"]\n"
        )
        self.assertEqual(value["architecture"], "amd64")
        with self.assertRaisesRegex(ValueError, "linux/amd64"):
            helpers.verify_image_identity(
                "sha256:" + "a" * 64 + "|linux|arm64|[\"python@sha256:" + helpers.IMAGE_DIGEST + "\"]"
            )

    def test_container_inspection_requires_exact_read_only_mounts_and_resource_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for child in ("src", "tests", "configs", "venv", "lock", "outputs"):
                (root / child).mkdir()
            helper_source = root / "scripts/neural/task7_training_run_helpers.py"
            helper_source.parent.mkdir(parents=True)
            helper_source.write_text("# projected helper source\n", encoding="utf-8")
            mounts = [
                {"Type": "bind", "Source": str(root / "src"), "Destination": "/learner/src", "RW": False},
                {"Type": "bind", "Source": str(root / "tests"), "Destination": "/learner/tests", "RW": False},
                {"Type": "bind", "Source": str(root / "configs"), "Destination": "/learner/configs", "RW": False},
                {"Type": "bind", "Source": str(root / "outputs"), "Destination": "/learner/artifacts", "RW": True},
                {"Type": "bind", "Source": str(root / "venv"), "Destination": "/work/venv-cpu", "RW": False},
                {"Type": "bind", "Source": str(root / "lock"), "Destination": "/work/requirements-cpu.lock", "RW": False},
                {"Type": "bind", "Source": str(helper_source),
                 "Destination": "/learner/scripts/neural/task7_training_run_helpers.py", "RW": False},
            ]
            config = {
                "Image": helpers.IMAGE_REFERENCE,
                "User": "1000:1000",
                "WorkingDir": "/learner",
                "Entrypoint": ["/bin/sh"],
                "Cmd": ["-ec", helpers.CONTAINER_SCRIPT, "task7", helpers.RUNTIME_PREFLIGHT_CODE],
                "Env": ["SPIRE_LEARN_RUN_NEURAL_NATIVE=1", "SPIRE_LEARN_RUN_NEURAL_TENSOR=1",
                        "PYTHONPATH=/learner/src", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
                        "HOME=/tmp",
                        *(f"{name}={value}" for name, value in helpers.THREAD_ENVIRONMENT.items())],
                "Labels": {"spire-learn.task": "task7-training-smoke", "spire-learn.run-id": "run-1"},
            }
            host = {
                "NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"], "NanoCpus": 4_000_000_000,
                "Memory": 8 * 1024**3, "PidsLimit": 512, "AutoRemove": False,
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"}, "Privileged": False,
                "Devices": [], "DeviceRequests": [],
            }
            image_id = "sha256:" + "d" * 64
            image_identity = {
                "image_id": image_id, "os": "linux", "architecture": "amd64",
                "repo_digests": ["python@sha256:" + helpers.IMAGE_DIGEST],
                "image_reference": helpers.IMAGE_REFERENCE,
            }
            inspect = [{"Id": "c" * 64, "Image": image_id,
                        "Config": config, "HostConfig": host, "Mounts": mounts}]
            result = helpers.assert_container_inspect(
                inspect, run_id="run-1", container_id="c" * 64,
                image_identity=image_identity,
                repo_root=root, venv=root / "venv", lock=root / "lock", output_child=root / "outputs",
            )
            self.assertEqual(result["network"], "none")
            self.assertEqual(result["platform"], "linux/amd64")
            helper_mount_index = len(mounts) - 1
            invalid_mounts = {
                "missing helper": mounts[:-1],
                "writable helper": [
                    {**mount, "RW": True} if index == helper_mount_index else dict(mount)
                    for index, mount in enumerate(mounts)
                ],
                "wrong helper source": [
                    {**mount, "Source": str(root / "scripts/neural/other.py")}
                    if index == helper_mount_index else dict(mount)
                    for index, mount in enumerate(mounts)
                ],
                "wrong helper destination": [
                    {**mount, "Destination": "/learner/scripts/task7_training_run_helpers.py"}
                    if index == helper_mount_index else dict(mount)
                    for index, mount in enumerate(mounts)
                ],
                "non-bind helper": [
                    {**mount, "Type": "volume"} if index == helper_mount_index else dict(mount)
                    for index, mount in enumerate(mounts)
                ],
                "duplicate helper destination": mounts + [dict(mounts[helper_mount_index])],
            }
            for case, invalid in invalid_mounts.items():
                with self.subTest(case=case):
                    invalid_payload = copy.deepcopy(inspect)
                    invalid_payload[0]["Mounts"] = invalid
                    with self.assertRaises(ValueError):
                        helpers.assert_container_inspect(
                            invalid_payload, run_id="run-1", container_id="c" * 64,
                            image_identity=image_identity,
                            repo_root=root, venv=root / "venv", lock=root / "lock",
                            output_child=root / "outputs",
                        )
            inspect[0]["Mounts"].append({"Type": "bind", "Source": str(root),
                                         "Destination": "/secret", "RW": False})
            with self.assertRaisesRegex(ValueError, "mount table"):
                helpers.assert_container_inspect(
                    inspect, run_id="run-1", container_id="c" * 64,
                    image_identity=image_identity,
                    repo_root=root, venv=root / "venv", lock=root / "lock", output_child=root / "outputs",
                )
            inspect[0]["Mounts"].pop()
            inspect[0]["HostConfig"]["Privileged"] = True
            with self.assertRaisesRegex(ValueError, "not be privileged"):
                helpers.assert_container_inspect(
                    inspect, run_id="run-1", container_id="c" * 64,
                    image_identity=image_identity,
                    repo_root=root, venv=root / "venv", lock=root / "lock", output_child=root / "outputs",
                )
            inspect[0]["HostConfig"]["Privileged"] = False
            inspect[0]["HostConfig"]["DeviceRequests"] = [{"Count": -1}]
            with self.assertRaisesRegex(ValueError, "host devices"):
                helpers.assert_container_inspect(
                    inspect, run_id="run-1", container_id="c" * 64,
                    image_identity=image_identity,
                    repo_root=root, venv=root / "venv", lock=root / "lock", output_child=root / "outputs",
                )

    def test_projected_container_layout_import_uses_only_the_single_file_helper_bind(self):
        repo_root = Path(__file__).resolve().parents[2]
        helper_source = repo_root / "scripts/neural/task7_training_run_helpers.py"
        wrapper_path = repo_root / "scripts/neural/run-task7-training-smoke.sh"
        wrapper_lines = wrapper_path.read_text(encoding="utf-8").splitlines()
        self.assertIn('HELPER="$ROOT/scripts/neural/task7_training_run_helpers.py"', wrapper_lines)

        mount_specs = []
        for line in wrapper_lines:
            stripped = line.strip()
            if not stripped.startswith('--mount "'):
                continue
            spec = stripped.split('"', 2)[1]
            fields = spec.split(",")
            values = dict(field.split("=", 1) for field in fields if "=" in field)
            mount_specs.append({
                "type": values.get("type"),
                "source": values.get("src"),
                "destination": values.get("dst"),
                "read_only": "readonly" in fields,
            })

        guest_helper = "/learner/scripts/neural/task7_training_run_helpers.py"
        script_mounts = [
            mount for mount in mount_specs
            if isinstance(mount["destination"], str)
            and mount["destination"].startswith("/learner/scripts")
        ]
        self.assertEqual(script_mounts, [{
            "type": "bind", "source": "$HELPER", "destination": guest_helper,
            "read_only": True,
        }])
        self.assertTrue(helper_source.is_file())

        import_program = """
import builtins
import importlib
from pathlib import Path
import sys
import sysconfig

optional_prefixes = {"numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab"}
optional_attempts = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in optional_prefixes:
        optional_attempts.append(name)
        raise AssertionError(f"optional import attempted: {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

module = importlib.import_module("scripts.neural.task7_training_run_helpers")
root = Path.cwd().resolve()
expected_helper = Path(sys.argv[1]).resolve()
host_repo = Path(sys.argv[2]).resolve()
assert Path(module.__file__).resolve() == expected_helper, module.__file__
assert not optional_attempts, optional_attempts
stdlib = Path(sysconfig.get_path("stdlib")).resolve()
extensions = Path(sysconfig.get_config_var("DESTSHARED")).resolve()
stdlib_zip = Path(sys.base_prefix) / "lib" / (
    f"python{sys.version_info.major}{sys.version_info.minor}.zip"
)
reviewed_inputs = {root, root / "src"}
for raw in sys.path:
    entry = Path(raw or ".").resolve()
    if entry in reviewed_inputs or entry == stdlib_zip:
        continue
    if entry == host_repo or entry.is_relative_to(host_repo):
        raise AssertionError(f"host repository leaked onto sys.path: {entry}")
    if entry == stdlib or entry.is_relative_to(stdlib):
        continue
    if entry == extensions or entry.is_relative_to(extensions):
        continue
    raise AssertionError(f"unreviewed sys.path entry: {entry}")
"""

        with tempfile.TemporaryDirectory() as directory:
            projected_root = Path(directory) / "learner"
            projected_src = projected_root / "src"
            projected_helper = projected_root / Path(guest_helper).relative_to("/learner")
            projected_root.mkdir()
            shutil.copytree(
                repo_root / "src", projected_src,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            projected_helper.parent.mkdir(parents=True)
            projected_helper.write_bytes(helper_source.read_bytes())
            environment = {
                "PYTHONPATH": str(projected_src),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            def import_helper():
                return subprocess.run(
                    [sys.executable, "-B", "-S", "-c", import_program,
                     str(projected_helper), str(repo_root)],
                    cwd=projected_root, env=environment, capture_output=True, text=True,
                )

            imported = import_helper()
            self.assertEqual(imported.returncode, 0, imported.stderr)

            projected_helper.unlink()
            missing = import_helper()
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("ModuleNotFoundError", missing.stderr)

    def test_combined_smoke_artifacts_enforce_two_mode_and_global_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _qualified_runtime_identity()
            identity_path = root / "development/neural-combat/runtime-identity.json"
            identity_path.parent.mkdir(parents=True)
            identity_path.write_text(json.dumps(identity), encoding="utf-8")
            for mode in ("run", "standalone"):
                run = root / mode / "unique-run"
                checkpoint = run / "checkpoint-final"
                checkpoint.mkdir(parents=True)
                for name in ("metadata.json", "model.pt", "training.pt"):
                    (checkpoint / name).write_bytes(b"artifact")
                summary = {
                    "status": "complete", "mode": mode,
                    "optimizer": {"updates": 2, "attempted_steps": 16,
                                  "accepted_steps": 16, "uncertain_steps": 0},
                    "exit_code": 1, "lifecycle_receipt_required": True,
                    "collection": {"learned_decisions": 256,
                                   "attempted_environment_slots": 1024},
                }
                (run / "summary.json").write_text(
                    json.dumps({"summary": summary}), encoding="utf-8"
                )
                (run / "lifecycle.json").write_text(json.dumps({
                    "schema_id": "spire-learn-neural-lifecycle-v1", "status": "complete",
                    "exit_code": 0, "elapsed_seconds": 12.0, "wall_overshoot_seconds": 0.0,
                    "timing_boundary": "after runtime and artifact writer close; before receipt write",
                    "receipt_path": "lifecycle.json",
                }), encoding="utf-8")
                (run / "smoke-checks.json").write_text(json.dumps({
                    "schema_id": "spire-learn-task7-smoke-checks-v1",
                    "all_final_weights_finite": True, "at_least_one_weight_changed": True,
                    "deterministic_reload_parity": True, "resume_round_trip": True,
                }), encoding="utf-8")
            result = helpers.assert_smoke_artifacts(root)
            self.assertEqual(result["totals"], {
                "learned_decisions": 512, "environment_slots": 2048,
                "optimizer_attempted_steps": 32, "optimizer_accepted_steps": 32,
            })

    def test_full_mounted_input_inventory_binds_shared_source_configs_and_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("src/spire_learn", "tests/unit", "configs/neural"):
                (root / folder).mkdir(parents=True)
            producer = root / "src/spire_learn/shared.py"
            outside_test = root / "tests/unit/test_shared.py"
            outside_config = root / "configs/neural/shared.toml"
            producer.write_text("value = 1\n", encoding="utf-8")
            outside_test.write_text("assert True\n", encoding="utf-8")
            outside_config.write_text("seed = 1\n", encoding="utf-8")
            before = helpers.mounted_input_inventory(root)
            self.assertEqual(before["src/spire_learn/shared.py"]["mode"], "100644")
            producer.chmod(0o755)
            executable = helpers.mounted_input_inventory(root)
            self.assertEqual(executable["src/spire_learn/shared.py"]["mode"], "100755")
            outside_config.write_text("seed = 2\n", encoding="utf-8")
            after = helpers.mounted_input_inventory(root)
            self.assertNotEqual(before, after)
            self.assertNotEqual(executable, after)
            (root / "tests/unit/escape.py").symlink_to(producer)
            with self.assertRaisesRegex(ValueError, "symlink"):
                helpers.mounted_input_inventory(root)

    def test_final_inventory_excludes_only_its_narrow_writer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, control, admin = (root / name for name in ("run", "control", "admin"))
            for path in (run, control, admin):
                path.mkdir()
            (run / "stdout.log").write_text("test", encoding="utf-8")
            (control / "cleanup.json").write_text("{}", encoding="utf-8")
            (admin / "release.json").write_text("{}", encoding="utf-8")
            (admin / "evidence-inventory.json").write_text("self", encoding="utf-8")
            inventory = helpers.inventory(run, control, admin)
            recorded = {(item["tree"], item["path"]) for item in inventory["files"]}
            self.assertIn(("run", "stdout.log"), recorded)
            self.assertIn(("control", "cleanup.json"), recorded)
            self.assertIn(("admin", "release.json"), recorded)
            self.assertNotIn(("admin", "evidence-inventory.json"), recorded)


if __name__ == "__main__":
    unittest.main()
