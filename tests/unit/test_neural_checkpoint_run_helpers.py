"""Zero-effect syntax, runtime-availability and inspect-control tests."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

from scripts.neural import task6_checkpoint_run_helpers as run_helpers
from scripts.neural.task6_checkpoint_run_helpers import (
    IMAGE,
    assert_container_inspect,
    parse_container_command,
    runtime_identity,
)


class NeuralCheckpointRunHelperTests(unittest.TestCase):
    def test_runner_and_embedded_container_command_pass_syntax_only_checks(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "scripts/neural/run-task6-checkpoint-tests.sh"
        shell = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        parsed = parse_container_command(script)
        self.assertTrue(parsed["parsed"], parsed["stderr"])
        body = parsed["command"]
        self.assertIn("120s", body)
        self.assertIn("unittest tests.neural.test_checkpoint -v", body)
        self.assertNotIn("unittest discover", body)
        self.assertNotIn("docker ", body)
        source = script.read_text(encoding="utf-8")
        self.assertIn("--pull=never", source)
        self.assertIn("docker image inspect", source)
        self.assertIn("container-create-recovery", source)
        self.assertIn("wrapper-preinventory-status", source)
        self.assertIn('wrapper.stdout.log" 2>> "$OUTER_LOG_ROOT/wrapper.stderr.log', source)
        self.assertIn("timeout --signal=TERM --kill-after=90s 210s", source)
        finalize_start = source.index("finalize() {")
        finalize_end = source.index("\n}\n\ntrap finalize EXIT", finalize_start)
        cleanup_budgets = [
            int(timeout_seconds) + int(kill_seconds)
            for kill_seconds, timeout_seconds in re.findall(
                r"timeout --signal=TERM --kill-after=(\d+)s (\d+)s",
                source[finalize_start:finalize_end],
            )
        ]
        self.assertLessEqual(sum(cleanup_budgets), 75, cleanup_budgets)
        recovery_start = source.index("container-create-recovery-lookup")
        recovery_end = source.index("container-final-inventory", recovery_start)
        recovery = source[recovery_start:recovery_end]
        self.assertIn('docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"', recovery)
        self.assertIn('docker inspect "$recovered_id"', recovery)
        self.assertIn('--container-id "$recovered_id"', recovery)
        self.assertLess(recovery.index("--container-id \"$recovered_id\""),
                        recovery.index('docker rm -f "$CONTAINER_ID"'))
        self.assertIn("preserve any uncertain or unowned object", recovery)

    def test_read_only_availability_check_matches_retained_cpu_runtime(self):
        root = Path(__file__).resolve().parents[2]
        record = runtime_identity(root)
        self.assertEqual(
            record["cpu_lock_sha256"],
            "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0",
        )
        self.assertEqual(
            record["native_extension_sha256"],
            "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9",
        )
        self.assertEqual(record["gym_revision"], "533dad1c8aec86df624b3742b6b3ad1dbf1d0033")
        self.assertIs(record["gym_original_source_posthash_available"], False)

    def test_exact_container_inspection_accepts_only_reviewed_limits_and_mounts(self):
        command = "set -eu\nexec timeout 120s python -B -m unittest tests.neural.test_checkpoint -v"
        environment = [
            "PYTHONPATH=/learner/src", "PYTHONDONTWRITEBYTECODE=1", "PIP_NO_INDEX=1",
            "PIP_NO_CACHE_DIR=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", "HOME=/run/home",
            "TMPDIR=/run/tmp", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
            "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
            "SPIRE_LEARN_RUN_NEURAL_TENSOR=1", "SPIRE_LEARN_TENSOR_TASK=task6",
            "SPIRE_LEARN_TENSOR_RESERVATION_ID=task6-run-a",
            "SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64",
            "SPIRE_LEARN_TENSOR_EVIDENCE_PATH=/run/tensor-sampling.json",
        ]
        inspect = [{
            "Id": "a" * 64,
            "Name": "/spirelearn-task6-run-a",
            "Config": {
                "Image": IMAGE,
                "User": "1000:1000",
                "Entrypoint": ["/bin/sh"],
                "Cmd": ["-ec", command],
                "WorkingDir": "/learner",
                "Env": environment,
                "Labels": {
                    "spire-learn.phase": "task6-checkpoint-cpu-tests",
                    "spire-learn.owner": "run-a",
                },
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "NetworkMode": "none",
                "Memory": 8 * 1024**3,
                "NanoCpus": 4_000_000_000,
                "PidsLimit": 512,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
                "DeviceRequests": [],
                "Devices": [],
                "Privileged": False,
            },
            "Mounts": [],
        }]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("src", "tests", "venv", "lock", "run")]
            for path in paths:
                path.mkdir()
            inspect[0]["Mounts"] = [
                {"Type": "bind", "Source": str(paths[0]), "Destination": "/learner/src", "RW": False},
                {"Type": "bind", "Source": str(paths[1]), "Destination": "/learner/tests", "RW": False},
                {"Type": "bind", "Source": str(paths[2]), "Destination": "/work/venv-cpu", "RW": False},
                {"Type": "bind", "Source": str(paths[3]), "Destination": "/work/requirements-cpu.lock", "RW": False},
                {"Type": "bind", "Source": str(paths[4]), "Destination": "/run", "RW": True},
            ]
            result = assert_container_inspect(
                inspect,
                run_id="run-a",
                source_root=paths[0],
                tests_root=paths[1],
                venv=paths[2],
                lock=paths[3],
                writable=paths[4],
                command=command,
                container_id="a" * 64,
            )
            self.assertEqual(result["pids_limit"], 512)
            self.assertEqual(result["network"], "none")

            inspect[0]["Mounts"][0]["RW"] = True
            with self.assertRaisesRegex(ValueError, "mount table"):
                assert_container_inspect(
                    inspect,
                    run_id="run-a",
                    source_root=paths[0],
                    tests_root=paths[1],
                    venv=paths[2],
                    lock=paths[3],
                    writable=paths[4],
                    command=command,
                    container_id="a" * 64,
                )

            inspect[0]["Mounts"][0]["RW"] = False
            inspect[0]["Config"]["Labels"]["spire-learn.owner"] = "somebody-else"
            with self.assertRaisesRegex(ValueError, "ownership labels"):
                assert_container_inspect(
                    inspect,
                    run_id="run-a",
                    source_root=paths[0],
                    tests_root=paths[1],
                    venv=paths[2],
                    lock=paths[3],
                    writable=paths[4],
                    command=command,
                    container_id="a" * 64,
                )
            inspect[0]["Config"]["Labels"]["spire-learn.owner"] = "run-a"
            with self.assertRaisesRegex(ValueError, "container ID"):
                assert_container_inspect(
                    inspect,
                    run_id="run-a",
                    source_root=paths[0],
                    tests_root=paths[1],
                    venv=paths[2],
                    lock=paths[3],
                    writable=paths[4],
                    command=command,
                    container_id="b" * 64,
                )

    def test_runtime_identity_and_exact_required_tests_gate_green_and_red(self):
        ids = run_helpers.expected_checkpoint_test_ids()
        self.assertEqual(len(ids), 4)
        runtime = self._runtime_line()

        green = self._test_transcript(ids, ["ok"] * 4, runtime=runtime)
        assessed = run_helpers.assess_checkpoint_test_run(
            green, "", 0, expected_outcome="green"
        )
        self.assertTrue(assessed["accepted"], assessed)
        self.assertEqual(assessed["classification"], "green")
        self.assertEqual(assessed["tests_ran"], 4)
        self.assertEqual(assessed["skipped_tests"], 0)

        rejected_green_cases = (
            ("empty", "Ran 0 tests in 0.000s\n\nOK\n", runtime, 0),
            ("skipped", self._test_transcript(ids, ["ok", "ok", "ok", "skipped"], runtime=runtime,
                                               footer="OK (skipped=1)"), runtime, 0),
            ("missing-runtime", self._test_transcript(ids, ["ok"] * 4, runtime=""), "", 0),
            ("wrong-id", self._test_transcript(
                ["test_unexpected (tests.neural.test_checkpoint.Other.test_unexpected)"] * 4,
                ["ok"] * 4, runtime=runtime,
            ), runtime, 0),
        )
        for name, transcript, _runtime, exit_code in rejected_green_cases:
            with self.subTest(name=name):
                result = run_helpers.assess_checkpoint_test_run(
                    transcript, "", exit_code, expected_outcome="green"
                )
                self.assertFalse(result["accepted"], result)

        api_red = (
            runtime
            + "\nsetUpClass (tests.neural.test_checkpoint.NeuralCheckpointTensorTests) ... ERROR\n\n"
            + "======================================================================\n"
            + "ERROR: setUpClass (tests.neural.test_checkpoint.NeuralCheckpointTensorTests)\n"
            + "RuntimeError: TASK6_CHECKPOINT_API_MISSING module=spire_learn.neural.checkpoint names=load_checkpoint,save_checkpoint\n"
            + "----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nFAILED (errors=1)\n"
        )
        result = run_helpers.assess_checkpoint_test_run(
            api_red, "", 1, expected_outcome="red"
        )
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["classification"], "product_api_missing")
        self.assertEqual(result["tests_ran"], 0)

        behavioral_red = self._test_transcript(
            ids, ["FAIL", "ok", "ok", "ok"], runtime=runtime,
            footer="FAILED (failures=1)",
        )
        result = run_helpers.assess_checkpoint_test_run(
            behavioral_red, "", 1, expected_outcome="red"
        )
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["classification"], "behavioral_red")

        rejected_red_cases = (
            ("torch-import-error", runtime + "\nsetUpClass (tests.neural.test_checkpoint.NeuralCheckpointTensorTests) ... ERROR\n"
             + "ImportError: No module named torch\nRan 0 tests in 0.000s\nFAILED (errors=1)\n", 1),
            ("api-marker-wrong-names", api_red.replace(
                "names=load_checkpoint,save_checkpoint", "names=load_checkpoint"
            ), 1),
            ("api-marker-without-runtime", api_red.replace(runtime + "\n", ""), 1),
            ("behavior-exit-zero", behavioral_red, 0),
            ("unexpected-test-error", self._test_transcript(
                ids, ["ERROR", "ok", "ok", "ok"], runtime=runtime,
                footer="FAILED (errors=1)",
            ), 1),
        )
        for name, transcript, exit_code in rejected_red_cases:
            with self.subTest(name=name):
                result = run_helpers.assess_checkpoint_test_run(
                    transcript, "", exit_code, expected_outcome="red"
                )
                self.assertFalse(result["accepted"], result)

    def test_expected_test_ids_match_real_stdlib_verbose_runner_labels(self):
        expected = set(run_helpers.expected_checkpoint_test_ids())
        methods = run_helpers.EXPECTED_CHECKPOINT_TEST_METHODS
        namespace = {"__module__": "tests.neural.test_checkpoint"}
        for name in methods:
            def passing_test(self):
                return None
            namespace[name] = passing_test
        test_case = type("NeuralCheckpointTensorTests", (unittest.TestCase,), namespace)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        )
        self.assertTrue(result.wasSuccessful(), stream.getvalue())
        labels = {
            line.rsplit(" ... ", 1)[0]
            for line in stream.getvalue().splitlines()
            if " ... " in line and line.endswith(" ... ok")
        }
        self.assertEqual(labels, expected)

    def test_runner_records_child_status_without_changing_errexit(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "scripts/neural/run-task6-checkpoint-tests.sh").read_text(encoding="utf-8")
        start = source.index("record() {")
        end = source.index("\nfinalize() {", start)
        functions = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "run"
            outer = Path(directory) / "outer"
            logs.mkdir()
            outer.mkdir()
            script = (
                "set -Eeuo pipefail\n"
                f"LOG_ROOT={shlex.quote(str(logs))}\n"
                f"OUTER_LOG_ROOT={shlex.quote(str(outer))}\n"
                + functions
                + "\ncase $- in *e*) before_on=1 ;; *) before_on=0 ;; esac\n"
                + "if record child-seven bash -c 'exit 7'; then status_on=0; else status_on=$?; fi\n"
                + "case $- in *e*) after_on=1 ;; *) after_on=0 ;; esac\n"
                + "set +e\n"
                + "case $- in *e*) before_off=1 ;; *) before_off=0 ;; esac\n"
                + "record_outer child-nine bash -c 'exit 9'\nstatus_off=$?\n"
                + "case $- in *e*) after_off=1 ;; *) after_off=0 ;; esac\n"
                + "printf 'on=%s,%s status=%s off=%s,%s status=%s\\n' \\\n"
                + '  "$before_on" "$after_on" "$status_on" "$before_off" "$after_off" "$status_off"\n'
            )
            completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "on=1,1 status=7 off=0,0 status=9\n")
            self.assertEqual((logs / "child-seven.exit.log").read_text().strip(), "7")
            self.assertEqual((outer / "child-nine.exit.log").read_text().strip(), "9")

    def test_final_inventory_writer_failure_is_exact_and_fails_wrapper(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "scripts/neural/run-task6-checkpoint-tests.sh").read_text(encoding="utf-8")
        start = source.index("write_final_inventory() {")
        end = source.index("\nfinalize() {", start)
        function = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            admin = Path(directory)
            script = (
                "set -Eeuo pipefail\n"
                f"ADMIN={shlex.quote(str(admin))}\n"
                + function
                + "\nif write_final_inventory \"$ADMIN\" 0 bash -c 'exit 17'; then status=0; else status=$?; fi\n"
                + "printf 'status=%s inventory=%s wrapper=%s\\n' \\\n"
                + '  "$status" "$FINAL_INVENTORY_STATUS" "$FINAL_WRAPPER_STATUS"\n'
            )
            completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "status=2 inventory=17 wrapper=2\n")
            self.assertEqual((admin / "evidence-inventory.exit.log").read_text(),
                             "inventory_command_exit=17\nwrapper_exit=2\n")
            self.assertIn("exit", (admin / "evidence-inventory.command.log").read_text())
            self.assertEqual((admin / "evidence-inventory.stdout.log").read_text(), "")
            self.assertEqual((admin / "evidence-inventory.stderr.log").read_text(), "")

    def test_durable_run_summary_keeps_the_exact_child_exit_and_rejects_disagreement(self):
        transcript = (
            self._runtime_line()
            + "\nsetUpClass (tests.neural.test_checkpoint.NeuralCheckpointTensorTests) ... ERROR\n"
            + "RuntimeError: TASK6_CHECKPOINT_API_MISSING module=spire_learn.neural.checkpoint names=load_checkpoint,save_checkpoint\n"
            + "Ran 0 tests in 0.000s\nFAILED (errors=1)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            logs = run_root / "logs"
            logs.mkdir()
            (logs / "container-start.stdout.log").write_text(transcript, encoding="utf-8")
            (logs / "container-start.stderr.log").write_text("", encoding="utf-8")
            (logs / "container-start.exit.log").write_text("1\n", encoding="utf-8")
            result = run_helpers.run_summary(
                run_root, expected_outcome="red", container_created=True,
                container_started=True, start_status=1, cleanup_failed=False,
            )
            self.assertTrue(result["accepted"])
            self.assertEqual(result["container_start_exit"], 1)
            self.assertEqual(result["container_start_exit_recorded"], 1)
            durable = json.loads((run_root / "run-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(durable["classification"], "product_api_missing")
            (logs / "container-start.exit.log").write_text("143\n", encoding="utf-8")
            (run_root / "run-summary.json").unlink()
            result = run_helpers.run_summary(
                run_root, expected_outcome="red", container_created=True,
                container_started=True, start_status=1, cleanup_failed=False,
            )
            self.assertFalse(result["accepted"])
            self.assertEqual(result["container_start_exit_recorded"], 143)
            (run_root / "run-summary.json").unlink()
            (logs / "container-start.exit.log").write_text("1\n", encoding="utf-8")
            cli_output = io.StringIO()
            with redirect_stdout(cli_output):
                accepted_exit = run_helpers._main([
                    "run-summary", str(run_root), "--expected-outcome", "red",
                    "--container-created", "--container-started", "--start-status", "1",
                ])
            self.assertEqual(accepted_exit, 0)
            self.assertTrue(json.loads(cli_output.getvalue())["accepted"])
            (run_root / "run-summary.json").unlink()
            cli_output = io.StringIO()
            with redirect_stdout(cli_output):
                rejected_exit = run_helpers._main([
                    "run-summary", str(run_root), "--expected-outcome", "green",
                    "--container-created", "--container-started", "--start-status", "1",
                ])
            self.assertEqual(rejected_exit, 2)
            self.assertFalse(json.loads(cli_output.getvalue())["accepted"])

    def test_local_pinned_image_and_uncertain_create_candidate_require_exact_identity(self):
        pinned_digest = IMAGE.split("@", 1)[1]
        canonical_alias = "python@" + pinned_digest
        accepted = [{
            "Id": "sha256:" + "1" * 64,
            "RepoDigests": [IMAGE],
            "Os": "linux",
            "Architecture": "amd64",
        }]
        identity = run_helpers.assert_local_image_identity(accepted)
        self.assertEqual(identity["image"], IMAGE)
        self.assertEqual(identity["architecture"], "amd64")
        canonical = [{**accepted[0], "RepoDigests": [canonical_alias]}]
        canonical_identity = run_helpers.assert_local_image_identity(canonical)
        self.assertEqual(canonical_identity["image"], IMAGE)
        self.assertEqual(canonical_identity["repo_digests"], [canonical_alias])
        for bad in (
            [{**accepted[0], "RepoDigests": ["ghcr.io/library/python@" + pinned_digest]}],
            [{**accepted[0], "RepoDigests": ["docker.io/library/python@sha256:" + "0" * 64]}],
            [{**accepted[0], "RepoDigests": ["python@sha256:" + "0" * 64]}],
            [{**accepted[0], "Architecture": "arm64"}],
            [{**accepted[0], "Id": "not-an-immutable-image-id"}],
            [],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_helpers.assert_local_image_identity(bad)

        self.assertIsNone(run_helpers.resolve_container_id(""))
        self.assertEqual(run_helpers.resolve_container_id("a" * 64 + "\n"), "a" * 64)
        for candidate in ("bad-id\n", "a" * 64 + "\n" + "b" * 64 + "\n"):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    run_helpers.resolve_container_id(candidate)

    def test_final_inventory_covers_run_and_control_roots_and_documents_only_admin_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_root, control_root, admin_root = (base / name for name in ("run", "control", "admin"))
            run_root.mkdir()
            control_root.mkdir()
            admin_root.mkdir()
            (run_root / "child.log").write_text("child evidence", encoding="utf-8")
            (control_root / "wrapper.exit.log").write_text("0\n", encoding="utf-8")
            output = admin_root / "evidence-inventory.json"
            record = run_helpers.close_inventory(
                run_root, control_root, output, admin_root=admin_root
            )
            self.assertEqual(set(record["files"]), {"run/child.log", "control/wrapper.exit.log"})
            self.assertEqual(record["excluded_final_writer"]["root"], str(admin_root.resolve()))
            self.assertEqual(
                set(record["excluded_final_writer"]["bookkeeping_files"]),
                {
                    "evidence-inventory.json", "evidence-inventory.command.log",
                    "evidence-inventory.stdout.log", "evidence-inventory.stderr.log",
                    "evidence-inventory.exit.log",
                },
            )
            for name in (
                "evidence-inventory.command.log", "evidence-inventory.stdout.log",
                "evidence-inventory.stderr.log", "evidence-inventory.exit.log",
            ):
                (admin_root / name).write_text("final writer bookkeeping", encoding="utf-8")
            self.assertTrue(run_helpers.verify_inventory(record, run_root, control_root, admin_root))
            (control_root / "wrapper.exit.log").write_text("2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                run_helpers.verify_inventory(record, run_root, control_root, admin_root)
            (control_root / "wrapper.exit.log").write_text("0\n", encoding="utf-8")
            (admin_root / "unrelated.txt").write_text("not final-writer bookkeeping", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "admin output"):
                run_helpers.verify_inventory(record, run_root, control_root, admin_root)

    @staticmethod
    def _runtime_line():
        identity = {
            "packages": {
                "spire-lab": "0.1.0", "gymnasium": "1.4.0",
                "numpy": "2.5.4", "torch": "2.14.1+cpu",
            },
            "lock_sha256": "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0",
            "native_sha256": "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9",
        }
        return "TASK6_RUNTIME_IDENTITY=" + json.dumps(identity, sort_keys=True)

    @staticmethod
    def _test_transcript(ids, statuses, *, runtime="", footer="OK"):
        lines = []
        if runtime:
            lines.append(runtime)
        lines.extend(f"{label} ... {status}" for label, status in zip(ids, statuses))
        lines.extend(("", "----------------------------------------------------------------------", "Ran 4 tests in 0.001s", "", footer))
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
