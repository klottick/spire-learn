import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from spire_learn.contracts import Failure


def _cli_api(test_case):
    try:
        module = import_module("spire_learn.cli")
    except ImportError as error:
        test_case.fail(f"spire-learn CLI API is missing: {error}")
    main = getattr(module, "main", None)
    test_case.assertTrue(callable(main), "spire-learn CLI main() API is missing")
    return module, main


def _write_manifest(path: Path, *, split: str = "selection") -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "spire-learn-cases-v1",
                "split": split,
                "cases": [
                    {
                        "case_id": "case-a",
                        "scenario_id": "scenario-a",
                        "partition": "selection",
                        "executions": [
                            {"execution_id": "execution-a", "environment_seed": 7, "policy_seed": 101}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, *, split: str = "selection", mode: str = "evaluate") -> Path:
    _write_manifest(path.parent / "cases.json", split=split)
    path.write_text(
        "\n".join(
            (
                f'mode = "{mode}"',
                'profile = "smoke"',
                'agent = "random"',
                'cases = "cases.json"',
                'output_root = "runs"',
                "per_case_decisions = 10" if mode == "evaluate" else "max_batch_steps = 10",
                "max_wall_seconds = 10.0",
                "traces = false",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


class _ProbeWriter:
    def __init__(self, *, close_error: BaseException | None = None):
        self.close_error = close_error
        self.close_calls = 0

    def write_start(self, config, manifest):
        self.start = (config, manifest)

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class EvaluationCliTests(unittest.TestCase):
    def test_import_and_help_do_not_import_optional_dependencies(self):
        code = r'''
import builtins
import sys

blocked = ("numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab")
attempts = []
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        attempts.append(name)
        raise AssertionError("optional dependency import attempted: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from spire_learn.cli import main
import spire_learn.__main__
assert main(["--help"]) == 0
assert main(["not-a-command"]) == 2
assert attempts == [], attempts
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in blocked
)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path.cwd(),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_validate_only_has_no_run_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(root / "run.toml")
            module, main = _cli_api(self)
            with patch.object(module, "evaluate", side_effect=AssertionError("evaluation must not run")):
                result = main(["evaluate", "--config", str(config_path), "--validate-only"])

            self.assertEqual(result, 0)
            self.assertFalse((root / "runs").exists())

    def test_probe_validate_only_does_not_collect_provenance_or_create_writer(self):
        module, main = _cli_api(self)
        for command in ("probe-vector", "probe-torchrl"):
            with self.subTest(command=command):
                config_path = Path("configs/probes") / f"{command.removeprefix('probe-')}-smoke.toml"
                with (
                    patch.object(module, "collect_provenance") as collect,
                    patch.object(module.RunWriter, "create") as create,
                    redirect_stdout(StringIO()),
                    redirect_stderr(StringIO()),
                ):
                    result = main(
                        [command, "--config", str(config_path), "--validate-only"]
                    )

                self.assertEqual(result, 0)
                collect.assert_not_called()
                create.assert_not_called()

    def test_final_flag_is_evaluation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(root / "final.toml", split="final")
            module, main = _cli_api(self)
            with patch.object(module, "evaluate", side_effect=AssertionError("validate-only must not run")):
                self.assertEqual(
                    main(["evaluate", "--config", str(config_path), "--allow-final", "--validate-only"]),
                    0,
                )
                self.assertEqual(
                    main(["probe-vector", "--config", str(config_path), "--allow-final"]),
                    2,
                )
            self.assertFalse((root / "runs").exists())

    def test_exit_codes_distinguish_blocked_and_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_config = _write_config(root / "final.toml", split="final")
            module, main = _cli_api(self)

            self.assertEqual(main(["evaluate", "--config", str(final_config)]), 2)
            self.assertFalse((root / "runs").exists())

            selection_config = _write_config(root / "selection.toml")
            incomplete = module.EvaluationReport(
                summary={"scheduled": 1, "started": 1, "not_started": 0},
                exit_code=1,
                run_dir=None,
                fatal_error=None,
            )
            with patch.object(module, "evaluate", return_value=incomplete) as evaluate:
                self.assertEqual(main(["evaluate", "--config", str(selection_config)]), 1)
            evaluate.assert_called_once()

    def test_evaluation_output_failure_is_execution_reliability_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_config(Path(directory) / "selection.toml")
            module, main = _cli_api(self)
            report = module.EvaluationReport(
                summary={"scheduled": 1},
                exit_code=0,
                run_dir=None,
                fatal_error=None,
            )
            with (
                patch.object(module, "evaluate", return_value=report),
                patch.object(module, "_print_evaluation_report", side_effect=OSError("stdout unavailable")),
            ):
                self.assertEqual(main(["evaluate", "--config", str(config_path)]), 1)

    def test_evaluation_execution_failure_is_not_preflight_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_config(Path(directory) / "selection.toml")
            module, main = _cli_api(self)
            with patch.object(module, "evaluate", side_effect=OSError("evaluation I/O failure")):
                self.assertEqual(main(["evaluate", "--config", str(config_path)]), 1)

    def test_probe_artifact_creation_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            module, _ = _cli_api(self)
            config = SimpleNamespace(
                output_root=Path(directory) / "runs",
                gym_checkout=None,
            )
            with (
                patch.object(module, "collect_provenance", return_value={}, create=True),
                patch.object(module.RunWriter, "create", side_effect=OSError("artifact root unavailable")),
            ):
                status = module._run_probe("probe-vector", config, None, None)

            self.assertEqual(status, 1)

    def test_probe_start_records_provenance_and_runtime_qualifications_for_both_modes(self):
        module, _ = _cli_api(self)
        provenance = {
            "learner": {"revision": "learner-revision", "head_state": "present", "dirty": False},
            "gym_checkout": {"revision": "gym-revision", "dirty": True},
            "distributions": {"spire-lab": None},
            "native_build_info": None,
        }
        report = SimpleNamespace(
            verdict="compatible",
            failures=(),
            batch_steps=1,
            slot_decisions=2,
            replacement_decisions=0,
        )
        vector_module = ModuleType("spire_learn.probes.vector")
        vector_module.run_vector_probe = lambda *args, **kwargs: report
        vector_module.probe_report_to_json = lambda value: {"verdict": value.verdict}
        torchrl_module = ModuleType("spire_learn.probes.torchrl")
        torchrl_module.run_torchrl_probe = lambda *args, **kwargs: report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for command, mode in (
                ("probe-vector", "probe-vector"),
                ("probe-torchrl", "probe-torchrl"),
            ):
                with self.subTest(mode=mode):
                    gym_checkout = root / "gym-checkout" if command == "probe-vector" else None
                    run_provenance = {
                        **provenance,
                        "gym_checkout": (
                            provenance["gym_checkout"] if gym_checkout is not None else None
                        ),
                    }
                    writer = _ProbeWriter()
                    config = SimpleNamespace(
                        mode=mode,
                        profile="smoke",
                        agent="random",
                        cases=root / "cases.json",
                        output_root=root / "runs",
                        per_case_decisions=None,
                        max_batch_steps=10,
                        max_wall_seconds=10.0,
                        traces=False,
                        expected_partition=None,
                        expected_schema_id=None,
                        expected_fingerprint=None,
                        gym_checkout=gym_checkout,
                        split_guard_manifests=(),
                        probe=SimpleNamespace(num_envs=2, workers=1),
                    )
                    manifest = SimpleNamespace(
                        path=config.cases,
                        split="selection",
                        executions=(),
                    )
                    with (
                        patch.object(module.RunWriter, "create", return_value=writer),
                        patch.object(
                            module,
                            "collect_provenance",
                            return_value=run_provenance,
                            create=True,
                        ) as collect,
                        patch.dict(
                            sys.modules,
                            {
                                "spire_learn.probes.vector": vector_module,
                                "spire_learn.probes.torchrl": torchrl_module,
                            },
                        ),
                        redirect_stdout(StringIO()),
                        redirect_stderr(StringIO()),
                    ):
                        status = module._run_probe(command, config, manifest, object())

                    collect.assert_called_once_with(
                        learner_root=Path(module.__file__).resolve().parents[2],
                        gym_checkout=gym_checkout,
                    )
                    self.assertEqual(status, 0)
                    config_record = writer.start[0]
                    self.assertEqual(config_record["provenance"], run_provenance)
                    self.assertIsNone(config_record["provenance"]["native_build_info"])
                    self.assertTrue(config_record["python_version"])
                    self.assertTrue(config_record["platform"])
                    self.assertIn("original-game RNG parity", config_record["rng_001"])
                    self.assertIn("Secret Portal", config_record["time_001"])

    def test_probe_exit_codes_use_failures_and_delivered_counters(self):
        module, _ = _cli_api(self)
        reports = (
            (
                SimpleNamespace(
                    verdict="blocked",
                    failures=(Failure("simulator_fault", "fixture", "step"),),
                    batch_steps=1,
                    slot_decisions=2,
                    replacement_decisions=0,
                ),
                1,
            ),
            (
                SimpleNamespace(
                    verdict="blocked",
                    failures=(Failure("setup_error", "runtime unavailable", "construction"),),
                    batch_steps=0,
                    slot_decisions=0,
                    replacement_decisions=0,
                ),
                2,
            ),
            (
                SimpleNamespace(
                    verdict="blocked",
                    failures=(),
                    batch_steps=2,
                    slot_decisions=4,
                    replacement_decisions=0,
                ),
                1,
            ),
            (
                SimpleNamespace(
                    verdict="blocked",
                    failures=(Failure("operational_error", "probe operation failed", "step"),),
                    batch_steps=0,
                    slot_decisions=0,
                    replacement_decisions=0,
                ),
                1,
            ),
            (
                SimpleNamespace(
                    verdict="adaptable",
                    failures=(),
                    batch_steps=2,
                    slot_decisions=4,
                    replacement_decisions=1,
                ),
                0,
            ),
        )

        for report, expected in reports:
            with self.subTest(verdict=report.verdict, failures=report.failures):
                self.assertEqual(module._probe_exit_code(report), expected)

    def test_probe_primary_process_control_survives_close_process_control(self):
        with tempfile.TemporaryDirectory() as directory:
            module, _ = _cli_api(self)
            primary = KeyboardInterrupt("probe interrupted")
            close_control = SystemExit("close interrupted")
            writer = _ProbeWriter(close_error=close_control)
            probe_module = ModuleType("spire_learn.probes.vector")

            def run_probe(*args, **kwargs):
                raise primary

            probe_module.run_vector_probe = run_probe
            probe_module.probe_report_to_json = lambda report: {"verdict": report.verdict}
            config = SimpleNamespace(
                mode="probe-vector",
                profile="smoke",
                agent="random",
                cases=Path(directory) / "cases.json",
                output_root=Path(directory) / "runs",
                per_case_decisions=None,
                max_batch_steps=10,
                max_wall_seconds=10.0,
                traces=False,
                expected_partition=None,
                expected_schema_id=None,
                expected_fingerprint=None,
                gym_checkout=None,
                split_guard_manifests=(),
                probe=SimpleNamespace(num_envs=2, workers=1),
            )
            manifest = SimpleNamespace(path=config.cases, split="selection", executions=())
            probe_stderr = StringIO()
            with (
                patch.object(module.RunWriter, "create", return_value=writer),
                patch.dict(sys.modules, {"spire_learn.probes.vector": probe_module}),
                redirect_stdout(StringIO()),
                redirect_stderr(probe_stderr),
            ):
                caught = None
                try:
                    module._run_probe("probe-vector", config, manifest, object())
                except BaseException as error:
                    caught = error

            self.assertIs(caught, primary)
            self.assertEqual(writer.close_calls, 1)
            self.assertTrue(
                any(
                    "SystemExit: close interrupted" in note
                    for note in getattr(primary, "__notes__", ())
                )
            )

    def test_probe_close_process_control_propagates_without_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            module, _ = _cli_api(self)
            close_control = KeyboardInterrupt("close interrupted")
            writer = _ProbeWriter(close_error=close_control)
            report = SimpleNamespace(
                verdict="compatible",
                failures=(),
                batch_steps=2,
                slot_decisions=4,
                replacement_decisions=0,
            )
            probe_module = ModuleType("spire_learn.probes.vector")
            probe_module.run_vector_probe = lambda *args, **kwargs: report
            probe_module.probe_report_to_json = lambda value: {"verdict": value.verdict}
            config = SimpleNamespace(
                mode="probe-vector",
                profile="smoke",
                agent="random",
                cases=Path(directory) / "cases.json",
                output_root=Path(directory) / "runs",
                per_case_decisions=None,
                max_batch_steps=10,
                max_wall_seconds=10.0,
                traces=False,
                expected_partition=None,
                expected_schema_id=None,
                expected_fingerprint=None,
                gym_checkout=None,
                split_guard_manifests=(),
                probe=SimpleNamespace(num_envs=2, workers=1),
            )
            manifest = SimpleNamespace(path=config.cases, split="selection", executions=())
            with (
                patch.object(module.RunWriter, "create", return_value=writer),
                patch.dict(sys.modules, {"spire_learn.probes.vector": probe_module}),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    module._run_probe("probe-vector", config, manifest, object())

            self.assertIs(raised.exception, close_control)
            self.assertEqual(writer.close_calls, 1)

    def test_probe_ordinary_failure_and_close_error_keep_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            module, _ = _cli_api(self)
            writer = _ProbeWriter(close_error=OSError("close I/O error"))
            probe_module = ModuleType("spire_learn.probes.vector")

            def run_probe(*args, **kwargs):
                raise OSError("probe I/O error")

            probe_module.run_vector_probe = run_probe
            probe_module.probe_report_to_json = lambda report: {}
            config = SimpleNamespace(
                mode="probe-vector",
                profile="smoke",
                agent="random",
                cases=Path(directory) / "cases.json",
                output_root=Path(directory) / "runs",
                per_case_decisions=None,
                max_batch_steps=10,
                max_wall_seconds=10.0,
                traces=False,
                expected_partition=None,
                expected_schema_id=None,
                expected_fingerprint=None,
                gym_checkout=None,
                split_guard_manifests=(),
                probe=SimpleNamespace(num_envs=2, workers=1),
            )
            manifest = SimpleNamespace(path=config.cases, split="selection", executions=())
            errors = StringIO()
            with (
                patch.object(module.RunWriter, "create", return_value=writer),
                patch.dict(sys.modules, {"spire_learn.probes.vector": probe_module}),
                redirect_stdout(StringIO()),
                redirect_stderr(errors),
            ):
                status = module._run_probe("probe-vector", config, manifest, object())

            self.assertEqual(status, 1)
            self.assertEqual(writer.close_calls, 1)
            self.assertIn("probe I/O error", errors.getvalue())
            self.assertIn("close I/O error", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
