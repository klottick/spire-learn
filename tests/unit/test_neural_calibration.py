"""Pure contract checks for the bounded Task 8 calibration path."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.neural.config import load_neural_config


ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"


def _calibration_text(*, operation: str = "calibrate", mode: str = "standalone",
                      device: str = "cuda:0", table: str | None = None) -> str:
    source = (ROOT / "configs/neural/standalone.toml").read_text(encoding="utf-8")
    source = source.replace('mode = "standalone"', f'mode = "{mode}"', 1)
    source = source.replace('operation = "train"', f'operation = "{operation}"', 1)
    source = source.replace('device = "cpu"', f'device = "{device}"', 1)
    source = source.replace("max_optimizer_steps = 16", "max_optimizer_steps = 3", 1)
    source = source.replace("max_training_decisions = 256", "max_training_decisions = 1024", 1)
    source = source.replace("max_environment_actions = 1024", "max_environment_actions = 1024", 1)
    source = source.replace("max_wall_seconds = 120.0", "max_wall_seconds = 300.0", 1)
    if table is not None:
        source += "\n" + table.rstrip() + "\n"
    return source


def _write_config(directory: Path, text: str) -> Path:
    path = directory / "calibration.toml"
    manifest = ROOT / "configs/cases/neural/train.json"
    output = ROOT / "artifacts/development/neural-combat/task-8/calibration/test-output"
    text = text.replace(
        'manifest_path = "../cases/neural/train.json"',
        f'manifest_path = "{manifest}"',
    ).replace(
        'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
        f'output_root = "{output}"',
    )
    path.write_text(text, encoding="utf-8")
    return path


class NeuralCalibrationConfigTests(unittest.TestCase):
    def test_exact_calibration_config_loads_with_qualified_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(Path(directory), _calibration_text(
                table=f'[calibration]\ngpu_uuid = "{GPU_UUID}"',
            ))
            config = load_neural_config(path)

        self.assertEqual(config.operation, "calibrate")
        self.assertEqual(config.split, "train")
        self.assertEqual(config.partition, "train")
        self.assertEqual(config.mode, "standalone")
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.calibration_gpu_uuid, GPU_UUID)
        self.assertEqual(config.budgets.max_optimizer_steps, 3)
        self.assertEqual(config.budgets.max_environment_actions, 1024)
        self.assertEqual(config.budgets.max_wall_seconds, 300.0)

    def test_calibration_uuid_table_is_exact_and_rejected_elsewhere(self):
        invalid_tables = (
            "",
            f'[calibration]\ngpu_uuid = "{GPU_UUID}"\nextra = "unexpected"',
            f'[calibration]\ngpu_uuid = true',
            '[calibration]\ngpu_uuid = "GPU-not-a-uuid"',
            '[calibration]\ngpu_uuid = "GPU-00000000-0000-0000-0000-000000000000"',
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, table in enumerate(invalid_tables):
                path = _write_config(Path(directory), _calibration_text(table=table))
                with self.subTest(table=table), self.assertRaises(ValueError):
                    load_neural_config(path)

            for operation in ("train", "evaluate"):
                path = _write_config(Path(directory), _calibration_text(
                    operation=operation,
                    table=f'[calibration]\ngpu_uuid = "{GPU_UUID}"',
                ))
                with self.subTest(operation=operation), self.assertRaises(ValueError):
                    load_neural_config(path)

            for mode, device in (("run", "cuda:0"), ("standalone", "cpu"),
                                 ("standalone", "cuda:1")):
                path = _write_config(Path(directory), _calibration_text(
                    mode=mode,
                    device=device,
                    table=f'[calibration]\ngpu_uuid = "{GPU_UUID}"',
                ))
                with self.subTest(mode=mode, device=device), self.assertRaises(ValueError):
                    load_neural_config(path)

    def test_authored_calibration_config_is_the_fixed_cohort_input(self):
        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        self.assertEqual(config.manifest_path, (ROOT / "configs/cases/neural/train.json").resolve())
        self.assertEqual(config.rollout_rows, 4)
        self.assertEqual(config.rollout_steps, 32)
        self.assertEqual(
            (config.model.embedding_width, config.model.hidden_width, config.model.max_parameters),
            (16, 64, 10_000_000),
        )


class NeuralCalibrationControlTests(unittest.TestCase):
    def test_manifest_release_binding_preserves_file_and_canonical_identity_domains(self):
        from spire_learn.neural.calibration import (
            CalibrationPreflightError,
            _validate_manifest_release_binding,
        )

        config_path = ROOT / "configs/neural/calibration.toml"
        manifest_path = ROOT / "configs/cases/neural/train.json"
        config = load_neural_config(config_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        config_file_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            formatted_path = Path(directory) / "train-formatted.json"
            formatted_bytes = (
                json.dumps(json.loads(manifest_bytes), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            formatted_path.write_bytes(formatted_bytes)
            formatted_config_path = Path(directory) / "calibration-formatted.toml"
            formatted_config_text = config_path.read_text(encoding="utf-8").replace(
                'manifest_path = "../cases/neural/train.json"',
                f'manifest_path = "{formatted_path}"',
            ).replace(
                'output_root = "../../artifacts/development/neural-combat/task-8/calibration/outputs"',
                f'output_root = "{config.output_root}"',
            )
            formatted_config_path.write_text(formatted_config_text, encoding="utf-8")
            formatted_config = load_neural_config(formatted_config_path)
            formatted_file_sha256 = hashlib.sha256(formatted_bytes).hexdigest()
            formatted_config_file_sha256 = hashlib.sha256(formatted_config_path.read_bytes()).hexdigest()

            self.assertEqual(formatted_config.manifest_sha256, config.manifest_sha256)
            self.assertNotEqual(formatted_file_sha256, manifest_file_sha256)

            release_binding = {
                "train_manifest_sha256": manifest_file_sha256,
                "calibration_config_file_sha256": config_file_sha256,
            }
            identities = _validate_manifest_release_binding(config, release_binding)
            self.assertEqual(identities, {
                "train_manifest_file_sha256": manifest_file_sha256,
                "train_manifest_canonical_sha256": config.manifest_sha256,
                "calibration_config_file_sha256": config_file_sha256,
                "calibration_config_canonical_sha256": config.config_sha256,
            })
            self.assertNotEqual(identities["train_manifest_file_sha256"],
                                identities["train_manifest_canonical_sha256"])
            self.assertNotEqual(identities["calibration_config_file_sha256"],
                                identities["calibration_config_canonical_sha256"])

            formatted_binding = {
                "train_manifest_sha256": formatted_file_sha256,
                "calibration_config_file_sha256": formatted_config_file_sha256,
            }
            formatted_identities = _validate_manifest_release_binding(formatted_config, formatted_binding)
            self.assertEqual(formatted_identities["train_manifest_canonical_sha256"],
                             identities["train_manifest_canonical_sha256"])
            self.assertEqual(formatted_identities["train_manifest_file_sha256"], formatted_file_sha256)
            with self.assertRaisesRegex(CalibrationPreflightError, "manifest file binding differs"):
                _validate_manifest_release_binding(formatted_config, release_binding)

    def test_verify_device_accepts_raw_manifest_release_binding_before_torch_import(self):
        from spire_learn.neural.calibration import (
            CUDA_LOCK_SHA256,
            NATIVE_SHA256,
            CalibrationPreflightError,
            _CudaCalibrationDependencies,
        )

        config_path = ROOT / "configs/neural/calibration.toml"
        config = load_neural_config(config_path)
        manifest_file_sha256 = hashlib.sha256(config.manifest_path.read_bytes()).hexdigest()
        runtime_environment = {
            "SPIRE_LEARN_TASK8_GPU_UUID": GPU_UUID,
            "SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256": "a" * 64,
            "SPIRE_LEARN_TASK8_RUNBOOK_SHA256": "b" * 64,
            "SPIRE_LEARN_TASK8_CONFIG_SHA256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "SPIRE_LEARN_TASK8_MANIFEST_SHA256": manifest_file_sha256,
            "SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256": "c" * 64,
            "SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256": "d" * 64,
            "SPIRE_LEARN_TASK8_LOCK_SHA256": CUDA_LOCK_SHA256,
            "SPIRE_LEARN_TASK8_NATIVE_SHA256": NATIVE_SHA256,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
        attempted_optional_imports = []
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                attempted_optional_imports.append(name)
                raise RuntimeError("test stops at the optional Torch import boundary")
            return real_import(name, *args, **kwargs)

        dependencies = _CudaCalibrationDependencies()
        with patch.dict(os.environ, runtime_environment, clear=True), \
                patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(CalibrationPreflightError,
                                        "qualified CUDA Torch could not be imported"):
                dependencies.verify_device(config, object())
        self.assertEqual(attempted_optional_imports, ["torch"])

    def test_verify_device_rejects_canonical_manifest_hash_as_file_binding_before_torch_import(self):
        from spire_learn.neural.calibration import (
            CUDA_LOCK_SHA256,
            NATIVE_SHA256,
            CalibrationPreflightError,
            _CudaCalibrationDependencies,
        )

        config_path = ROOT / "configs/neural/calibration.toml"
        config = load_neural_config(config_path)
        runtime_environment = {
            "SPIRE_LEARN_TASK8_GPU_UUID": GPU_UUID,
            "SPIRE_LEARN_TASK8_SOURCE_SNAPSHOT_SHA256": "a" * 64,
            "SPIRE_LEARN_TASK8_RUNBOOK_SHA256": "b" * 64,
            "SPIRE_LEARN_TASK8_CONFIG_SHA256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "SPIRE_LEARN_TASK8_MANIFEST_SHA256": config.manifest_sha256,
            "SPIRE_LEARN_TASK8_QUALIFICATION_REPORT_SHA256": "c" * 64,
            "SPIRE_LEARN_TASK8_QUALIFICATION_SUMMARY_SHA256": "d" * 64,
            "SPIRE_LEARN_TASK8_LOCK_SHA256": CUDA_LOCK_SHA256,
            "SPIRE_LEARN_TASK8_NATIVE_SHA256": NATIVE_SHA256,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
        attempted_optional_imports = []
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                attempted_optional_imports.append(name)
                raise RuntimeError("test stops at the optional Torch import boundary")
            return real_import(name, *args, **kwargs)

        dependencies = _CudaCalibrationDependencies()
        with patch.dict(os.environ, runtime_environment, clear=True), \
                patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(CalibrationPreflightError,
                                        "parent-reviewed manifest file binding differs"):
                dependencies.verify_device(config, object())
        self.assertEqual(attempted_optional_imports, [])

    def test_native_payload_identity_reads_one_distribution_file_without_importing_it(self):
        from spire_learn.neural.calibration import _CudaCalibrationDependencies, CalibrationPreflightError
        from unittest.mock import patch

        class Distribution:
            def __init__(self, root, files):
                self.root = root
                self.files = tuple(files)

            def locate_file(self, entry):
                return self.root / entry

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            native = root / "spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
            native.parent.mkdir()
            native.write_bytes(b"accepted native fixture")
            distribution = Distribution(root, (Path("spire_lab/_native.cpython-312-x86_64-linux-gnu.so"),))
            with patch("spire_learn.neural.calibration.sys.prefix", str(root)):
                actual = _CudaCalibrationDependencies._native_payload_sha256(distribution)
                self.assertEqual(len(actual), 64)
                with self.assertRaises(CalibrationPreflightError):
                    _CudaCalibrationDependencies._native_payload_sha256(Distribution(
                        root, (*distribution.files, Path("spire_lab/_native-copy.so")),
                    ))

    def test_fixed_trial_reservations_and_derived_run_profile(self):
        from spire_learn.neural.calibration import TRIALS, _trial_config

        self.assertEqual(
            [(item.mode, item.batch_size, item.horizon, item.action_slots) for item in TRIALS],
            [("standalone", 4, 32, 128), ("standalone", 8, 32, 256),
             ("standalone", 16, 32, 512), ("run", 1, 128, 128)],
        )
        self.assertEqual([item.seed_ordinal for item in TRIALS], [0, 1024, 2048, 3072])
        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        run_config = _trial_config(config, TRIALS[-1], trial_index=3)
        self.assertEqual(run_config.operation, "calibrate")
        self.assertEqual(run_config.mode, "run")
        self.assertEqual(run_config.profile_id, "m10-ironclad-policy")
        self.assertEqual(run_config.observation_version, 3)
        self.assertEqual(run_config.replay_version, 3)
        self.assertEqual(
            run_config.run_observation_fingerprint,
            "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8",
        )
        self.assertEqual(run_config.manifest_sha256, config.manifest_sha256)
        self.assertEqual(run_config.config_sha256, config.config_sha256)
        self.assertEqual(run_config.rollout_rows, 1)
        self.assertEqual(run_config.rollout_steps, 128)
        self.assertEqual(len(set((run_config.seed_streams.policy, run_config.seed_streams.optimizer,
                                  run_config.seed_streams.curriculum, run_config.seed_streams.noncombat))), 4)

    def test_calibration_preflight_rejects_invalid_operation_device_and_budgets_before_factories(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        for invalid in (
            replace(config, operation="train"),
            replace(config, split="selection"),
            replace(config, device="cuda:1"),
            replace(config, calibration_gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"),
            replace(config, budgets=replace(config.budgets, max_environment_actions=1025)),
        ):
            dependencies = _NoFactoryDependencies()
            with self.subTest(operation=invalid.operation, split=invalid.split, device=invalid.device), \
                    self.assertRaises((TypeError, ValueError)):
                _calibrate_with_dependencies(invalid, dependencies, clock=lambda: 0.0)
            self.assertEqual(dependencies.calls, [])

    def test_action_and_optimizer_reservations_are_cumulative_and_never_refunded(self):
        from spire_learn.neural.calibration import _CalibrationLedger

        ledger = _CalibrationLedger()
        for index, size in enumerate((128, 256, 512, 128)):
            ledger.reserve_actions(size, name=f"trial-{index}")
        self.assertEqual(ledger.snapshot()["reserved_action_slots"], 1024)
        with self.assertRaises(ValueError):
            ledger.reserve_actions(1, name="over-cap")
        for index in range(3):
            reservation = ledger.reserve_optimizer_call(trial_id=f"standalone-{index}")
            ledger.mark_optimizer_attempted(reservation)
            ledger.mark_optimizer_accepted(reservation)
        self.assertEqual(ledger.snapshot()["optimizer_attempted_steps"], 3)
        self.assertEqual(ledger.snapshot()["optimizer_accepted_steps"], 3)
        with self.assertRaises(ValueError):
            ledger.reserve_optimizer_call(trial_id="run-must-not-update")

    def test_sample_reservations_reconcile_pre_sync_inner_and_post_sync_failures(self):
        from spire_learn.neural.calibration import _CalibrationLedger, _TimedPolicy, _TrialTiming

        class Budget:
            def check(self, _phase):
                return None

        class Policy:
            def __init__(self, error=None):
                self.error = error
                self.calls = 0

            def act(self, frames, *, generator, deterministic=False):
                del generator, deterministic
                self.calls += 1
                if self.error is not None:
                    raise self.error
                return ("sample",) * len(frames)

        def make_adapter(policy, ledger, *, timer=lambda: 0.0, synchronize=None):
            return _TimedPolicy(
                policy,
                ledger=ledger,
                trial_id="test-trial",
                timing=_TrialTiming(),
                timer=timer,
                budget=Budget(),
                synchronize=synchronize,
            )

        pre_ledger = _CalibrationLedger()
        pre_calls = []
        pre_error = RuntimeError("pre-call sync failed")

        def fail_before_call():
            pre_calls.append("sync")
            raise pre_error

        pre_policy = Policy()
        with self.assertRaises(RuntimeError) as caught:
            make_adapter(pre_policy, pre_ledger, synchronize=fail_before_call).act(
                (object(), object()), generator=None,
            )
        self.assertIs(caught.exception, pre_error)
        self.assertEqual(pre_policy.calls, 0)
        self.assertEqual(pre_ledger.snapshot()["reserved_sample_rows"], 2)
        self.assertEqual(pre_ledger.snapshot()["attempted_sample_rows"], 0)
        self.assertEqual(pre_ledger.snapshot()["delivered_sample_rows"], 0)
        self.assertEqual(pre_ledger.snapshot()["uncertain_sample_rows"], 0)
        self.assertEqual(pre_ledger.snapshot()["undispatched_sample_rows"], 2)

        inner_ledger = _CalibrationLedger()
        inner_error = KeyboardInterrupt("inner act failed")
        inner_policy = Policy(inner_error)
        with self.assertRaises(KeyboardInterrupt) as caught:
            make_adapter(inner_policy, inner_ledger).act((object(), object()), generator=None)
        self.assertIs(caught.exception, inner_error)
        self.assertEqual(inner_ledger.snapshot()["attempted_sample_rows"], 2)
        self.assertEqual(inner_ledger.snapshot()["delivered_sample_rows"], 0)
        self.assertEqual(inner_ledger.snapshot()["uncertain_sample_rows"], 2)
        self.assertEqual(inner_ledger.snapshot()["undispatched_sample_rows"], 0)

        post_ledger = _CalibrationLedger()
        post_calls = []
        post_error = RuntimeError("post-call sync failed")

        def fail_after_call():
            post_calls.append("sync")
            if len(post_calls) == 2:
                raise post_error

        post_policy = Policy()
        post_timing = _TrialTiming()
        adapter = _TimedPolicy(
            post_policy, ledger=post_ledger, trial_id="post-sync-test", timing=post_timing,
            timer=lambda: 0.0, budget=Budget(), synchronize=fail_after_call,
        )
        returned = adapter.act((object(), object()), generator=None)
        self.assertEqual(returned, ("sample", "sample"))
        with self.assertRaises(RuntimeError) as caught:
            post_timing.raise_pending_instrumentation_error()
        self.assertIs(caught.exception, post_error)
        self.assertEqual(post_ledger.snapshot()["attempted_sample_rows"], 2)
        self.assertEqual(post_ledger.snapshot()["delivered_sample_rows"], 2)
        self.assertEqual(post_ledger.snapshot()["uncertain_sample_rows"], 0)
        self.assertEqual(post_ledger.snapshot()["undispatched_sample_rows"], 0)

    def test_sample_return_survives_timer_failure_after_the_inner_call(self):
        from spire_learn.neural.calibration import _CalibrationLedger, _TimedPolicy, _TrialTiming

        class Budget:
            def check(self, _phase):
                return None

        class Policy:
            def act(self, frames, *, generator, deterministic=False):
                del generator, deterministic
                return ("sample",) * len(frames)

        calls = 0
        timer_error = OSError("timer unavailable after act")

        def timer():
            nonlocal calls
            calls += 1
            if calls == 1:
                return 1.0
            raise timer_error

        ledger = _CalibrationLedger()
        timing = _TrialTiming()
        adapter = _TimedPolicy(
            Policy(), ledger=ledger, trial_id="timer-test", timing=timing,
            timer=timer, budget=Budget(), synchronize=None,
        )
        result = adapter.act((object(), object()), generator=None)
        self.assertEqual(result, ("sample", "sample"))
        self.assertEqual(ledger.snapshot()["delivered_sample_rows"], 2)
        with self.assertRaises(OSError) as caught:
            timing.raise_pending_instrumentation_error()
        self.assertIs(caught.exception, timer_error)

    def test_forward_backward_probe_admits_each_phase_and_reports_separate_time(self):
        from spire_learn.neural.calibration import _forward_backward_probe

        events = []

        class Budget:
            def check(self, phase):
                events.append(f"admit:{phase}")

        class Loss:
            def backward(self):
                events.append("backward")

        class Model:
            def forward(self, batch):
                events.append("forward")
                return ("output", batch)

        now = 0.0

        def timer():
            nonlocal now
            now += 0.25
            return now

        def objective(output, batch):
            events.append("objective")
            self.assertEqual(output, ("output", batch))
            return Loss()

        output, loss, timing = _forward_backward_probe(
            Model(), "owned-frames", objective, budget=Budget(), phase="cpu",
            timer=timer,
        )
        self.assertEqual(output, ("output", "owned-frames"))
        self.assertIsInstance(loss, Loss)
        self.assertEqual(events, ["admit:cpu_forward", "forward", "objective",
                                  "admit:cpu_backward", "backward"])
        self.assertEqual(timing, {"forward_seconds": 0.25, "backward_seconds": 0.25})

    def test_stored_frame_cpu_gpu_ratio_has_named_boundary_and_zero_guard(self):
        from spire_learn.neural.calibration import _stored_frame_cpu_gpu_ratio

        ratio = _stored_frame_cpu_gpu_ratio(1.0, 3.0, 0.5, 1.5)
        self.assertEqual(ratio["cpu_over_gpu"], 2.0)
        self.assertEqual(ratio["numerator_seconds"], 4.0)
        self.assertEqual(ratio["denominator_seconds"], 2.0)
        self.assertIn("CPU collation and GPU H2D excluded", ratio["measurement_boundary"])
        zero = _stored_frame_cpu_gpu_ratio(1.0, 0.0, 0.0, 0.0)
        self.assertIsNone(zero["cpu_over_gpu"])
        self.assertEqual(zero["unavailable_reason"], "zero_gpu_forward_backward_seconds")

    def test_logit_difference_metric_covers_all_candidates_by_name(self):
        from spire_learn.neural.calibration import _all_candidate_logit_difference

        metric = _all_candidate_logit_difference(((1.0, 100.0),), ((2.0, -500.0),))
        self.assertEqual(metric, {"all_candidate_logits_max_abs_difference": 600.0})

    def test_core_uses_real_collector_reservations_and_keeps_sample_source_frames(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _SuccessfulFakeDependencies()
        report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)

        self.assertEqual(report.summary["status"], "complete")
        self.assertEqual(report.summary["cohort"]["reserved_action_slots"], 1024)
        self.assertEqual(report.summary["cohort"]["delivered_environment_slots"], 1024)
        self.assertEqual(report.summary["cohort"]["optimizer_accepted_steps"], 3)
        self.assertEqual(report.summary["selection"]["trial_id"], "standalone-b16-h32")
        self.assertEqual(dependencies.factory_order, [item.spec.trial_id for item in dependencies.trials])
        self.assertEqual(len(dependencies.comparison_frames), 3)
        for frames, update_frames, expected_size in zip(
            dependencies.comparison_frames, dependencies.update_frames, (4, 8, 16), strict=True,
        ):
            self.assertIs(frames, update_frames)
            self.assertEqual(len(frames), expected_size)
        self.assertEqual([trial.close_count for trial in dependencies.trials], [1, 1, 1, 1])

    def test_wall_exhaustion_after_reset_keeps_reset_delivery_and_dispatches_nothing(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _ResetExhaustionDependencies()
        report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        first = report.summary["trials"][0]
        self.assertEqual(report.summary["status"], "incomplete")
        self.assertEqual(first["collection"]["delivered_reset_slots"], 4)
        self.assertEqual(first["collection"]["attempted_environment_slots"], 0)
        self.assertEqual(first["collection"]["learned_decisions"], 0)
        self.assertEqual(first["collection"]["stop_reason"], "wall_budget")
        self.assertEqual(first["status"], "incomplete")
        self.assertIsNone(report.summary["selection"])

    def test_first_trial_failure_preserves_partial_counts_and_closes_without_retry(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _PartialFailureDependencies()
        report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertEqual(report.summary["status"], "operational_failure")
        self.assertEqual(len(report.summary["trials"]), 1)
        partial = report.summary["trials"][0]
        self.assertEqual(partial["collection"]["attempted_environment_slots"], 8)
        self.assertEqual(partial["collection"]["delivered_environment_slots"], 4)
        self.assertEqual(partial["collection"]["uncertain_environment_slots"], 4)
        self.assertEqual(dependencies.factory_order, ["standalone-b4-h32"])
        self.assertEqual(dependencies.trials[0].close_count, 1)

    def test_selection_ignores_size_that_failed_before_all_probes(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _ComparisonFailureDependencies(fail_batch=8)
        report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertEqual(report.summary["status"], "operational_failure")
        self.assertEqual(report.summary["selection"]["trial_id"], "standalone-b4-h32")
        self.assertEqual([row["trial_id"] for row in report.summary["trials"]], [
            "standalone-b4-h32", "standalone-b8-h32",
        ])
        self.assertTrue(report.summary["trials"][0]["fully_tested"])
        self.assertFalse(report.summary["trials"][1]["fully_tested"])

    def test_collector_uses_mixed_terminal_manual_reset_and_recovery_ordinals(self):
        from spire_learn.neural.calibration import (
            _CalibrationBudget, _CalibrationLedger, _assemble_trial, _trial_config, TRIALS,
        )
        from spire_learn.neural.collector import CollectorRecovery

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        trial_config = _trial_config(config, TRIALS[0], trial_index=0)
        clock = _FakeClock()
        with _synthetic_schema() as schema:
            driver = _FakeDriver(trial_config, schema, clock=clock, terminate_slot_once=1)
            session = _assemble_trial(
                trial_config,
                TRIALS[0],
                CollectorRecovery(10, 77, (40, 40, 40, 40)),
                _CalibrationBudget(clock),
                _CalibrationLedger(),
                driver_factory=lambda _config, _cases: driver,
                policy_factory=lambda _schema, _model, **_kwargs: _FakePolicy(),
                generator_factory=lambda _config: object(),
            )
            rollout = session.collector.collect(
                max_learned_decisions=8,
                max_env_actions=8,
                policy_version=0,
            )
            diagnostics = rollout.collection

        self.assertEqual(diagnostics.reset_operations, 2)
        self.assertEqual(diagnostics.reset_slots, 5)
        self.assertEqual(diagnostics.delivered_environment_slots, 8)
        self.assertEqual(
            [allocation.ordinal for allocation in diagnostics.seed_allocations],
            [10, 11, 12, 13, 14],
        )
        self.assertEqual(driver.reset_masks[1], (False, True, False, False))
        self.assertEqual(session.collector.recovery_state.next_seed_ordinal, 15)
        session.close()

    def test_process_control_propagates_after_trial_and_writer_close_once(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _ControlFailureDependencies()
        first_control = KeyboardInterrupt("first trial control")
        later_control = SystemExit("later writer control")

        def interrupt_comparison(*_args):
            raise first_control

        def interrupt_writer(*_args):
            raise later_control

        dependencies.compare = interrupt_comparison
        dependencies.writer.event = interrupt_writer
        with self.assertRaises(KeyboardInterrupt) as caught:
            _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertIs(caught.exception, first_control)
        self.assertTrue(any("SystemExit" in note for note in getattr(caught.exception, "__notes__", ())))
        self.assertEqual(dependencies.trials[0].close_count, 1)
        self.assertEqual(dependencies.writer.close_count, 1)

    def test_first_control_after_ordinary_failure_carries_the_ordinary_error(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _SuccessfulFakeDependencies()
        ordinary = RuntimeError("comparison failed first")
        control = KeyboardInterrupt("writer control followed")

        def fail_comparison(*_args):
            raise ordinary

        def fail_writer(*_args):
            raise control

        dependencies.compare = fail_comparison
        dependencies.writer.event = fail_writer
        with self.assertRaises(KeyboardInterrupt) as caught:
            _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertIs(caught.exception, control)
        self.assertTrue(any("comparison failed first" in note
                            for note in getattr(caught.exception, "__notes__", ())))
        self.assertEqual(dependencies.trials[0].close_count, 1)
        self.assertEqual(dependencies.writer.close_count, 1)

    def test_process_control_from_trial_close_supersedes_and_notes_ordinary_failure(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _SuccessfulFakeDependencies()
        ordinary = RuntimeError("comparison failed before close")
        control = SystemExit("trial close control")

        def fail_comparison(*_args):
            raise ordinary

        original_build = dependencies.build_trial

        def build_with_control_close(*args):
            session = original_build(*args)
            original_close = session.close

            def close_then_control():
                original_close()
                raise control

            session.close = close_then_control
            return session

        dependencies.compare = fail_comparison
        dependencies.build_trial = build_with_control_close
        with self.assertRaises(SystemExit) as caught:
            _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertIs(caught.exception, control)
        self.assertTrue(any("comparison failed before close" in note
                            for note in getattr(caught.exception, "__notes__", ())))
        self.assertEqual(dependencies.trials[0].close_count, 1)
        self.assertEqual(dependencies.writer.close_count, 1)

    def test_collector_fault_close_control_is_recovered_without_second_close(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _PartialFailureDependencies()
        close_control = KeyboardInterrupt("driver close control after collector fault")
        original_build = dependencies.build_trial
        close_calls = []

        def build_with_control_driver_close(*args):
            session = original_build(*args)
            original_close = session.driver.close

            def close_then_control():
                close_calls.append("driver-close")
                original_close()
                raise close_control

            session.driver.close = close_then_control
            return session

        dependencies.build_trial = build_with_control_driver_close
        with self.assertRaises(KeyboardInterrupt) as caught:
            _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        self.assertIs(caught.exception, close_control)
        self.assertTrue(any("injected fake accepted-step fault" in note
                            for note in getattr(caught.exception, "__notes__", ())))
        self.assertEqual(close_calls, ["driver-close"])
        self.assertEqual(dependencies.trials[0].close_count, 1)
        self.assertEqual(dependencies.writer.close_count, 1)

    def test_memory_and_nvidia_query_preserve_process_control(self):
        from types import SimpleNamespace
        from spire_learn.neural.calibration import _CudaCalibrationDependencies

        memory_control = KeyboardInterrupt("memory sync control")

        def interrupt_sync(*_args):
            raise memory_control

        dependencies = _CudaCalibrationDependencies()
        dependencies.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=interrupt_sync))
        with self.assertRaises(KeyboardInterrupt) as caught:
            dependencies.check_memory()
        self.assertIs(caught.exception, memory_control)

        query_control = SystemExit("nvidia query control")
        with patch("spire_learn.neural.calibration.subprocess.run", side_effect=query_control):
            with self.assertRaises(SystemExit) as caught:
                dependencies._nvidia_query("gpu=uuid")
        self.assertIs(caught.exception, query_control)

    def test_final_clock_failure_is_nonzero_and_explicitly_unavailable(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _SuccessfulFakeDependencies()
        with tempfile.TemporaryDirectory() as directory:
            dependencies.writer.run_dir = Path(directory)
            original_verify = dependencies.verify_device
            budget_holder = {}

            def capture_budget(selected_config, budget):
                budget_holder["budget"] = budget
                return original_verify(selected_config, budget)

            dependencies.verify_device = capture_budget
            clock_error = RuntimeError("final clock unavailable")

            def fail_final_clock():
                raise clock_error

            original_close = dependencies.writer.close

            def close_then_break_clock():
                original_close()
                budget_holder["budget"]._clock = fail_final_clock

            dependencies.writer.close = close_then_break_clock
            report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
            lifecycle_path = dependencies.writer.run_dir / "lifecycle.json"
            lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
            self.assertNotEqual(report.exit_code, 0)
            self.assertEqual(report.summary["status"], "operational_failure")
            self.assertEqual(lifecycle["status"], "operational_failure")
            self.assertIsNone(lifecycle["elapsed_seconds"])
            self.assertFalse(lifecycle["timing_available"])
            self.assertEqual(lifecycle["timing_unavailable"]["error"]["type"], "RuntimeError")
            self.assertEqual(dependencies.writer.close_count, 1)

    def test_final_clock_control_propagates_after_close_and_marks_receipt_unavailable(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        dependencies = _SuccessfulFakeDependencies()
        with tempfile.TemporaryDirectory() as directory:
            dependencies.writer.run_dir = Path(directory)
            original_verify = dependencies.verify_device
            budget_holder = {}

            def capture_budget(selected_config, budget):
                budget_holder["budget"] = budget
                return original_verify(selected_config, budget)

            dependencies.verify_device = capture_budget
            control = KeyboardInterrupt("final clock control")

            def interrupt_final_clock():
                raise control

            original_close = dependencies.writer.close

            def close_then_interrupt_clock():
                original_close()
                budget_holder["budget"]._clock = interrupt_final_clock

            dependencies.writer.close = close_then_interrupt_clock
            with self.assertRaises(KeyboardInterrupt) as caught:
                _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
            self.assertIs(caught.exception, control)
            self.assertEqual(dependencies.writer.close_count, 1)
            lifecycle = json.loads((Path(directory) / "lifecycle.json").read_text(encoding="utf-8"))
            self.assertEqual(lifecycle["status"], "interrupted")
            self.assertIsNone(lifecycle["elapsed_seconds"])
            self.assertFalse(lifecycle["timing_available"])

    def test_enabled_harness_accepts_horizon_and_true_early_run_end_only(self):
        from types import SimpleNamespace
        from tests.integration.test_neural_calibration import _assert_completed_calibration_report

        def report_for(run_delivered, *, run_stop="horizon", run_terminals=0, run_faults=0,
                       exit_code=0, status="complete"):
            reservations = (128, 256, 512, 128)
            trial_ids = ("standalone-b4-h32", "standalone-b8-h32",
                         "standalone-b16-h32", "run-b1-h128")
            trials = []
            for index, (trial_id, reserved) in enumerate(zip(trial_ids, reservations, strict=True)):
                run = index == 3
                delivered = run_delivered if run else reserved
                trials.append({
                    "trial_id": trial_id,
                    "fully_tested": True,
                    "status": "complete",
                    "reserved_action_slots": reserved,
                    "collection": {
                        "delivered_environment_slots": delivered,
                        "attempted_environment_slots": delivered,
                        "faults": run_faults if run else 0,
                        "stop_reason": run_stop if run else "horizon",
                        "task_terminals": run_terminals if run else 0,
                        "environment_terminations": run_terminals if run else 0,
                        "environment_truncations": 0,
                    },
                    "optimizer_probe": None if run else {
                        "attempted_steps": 1, "accepted_steps": 1, "uncertain_steps": 0,
                        "finite": True, "parameters_changed": True,
                    },
                })
            delivered_total = sum(row["collection"]["delivered_environment_slots"] for row in trials)
            return SimpleNamespace(exit_code=exit_code, summary={
                "status": status,
                "cohort": {
                    "reserved_action_slots": 1024,
                    "attempted_environment_slots": delivered_total,
                    "delivered_environment_slots": delivered_total,
                    "optimizer_accepted_steps": 3,
                },
                "trials": trials,
            })

        _assert_completed_calibration_report(self, report_for(128))
        _assert_completed_calibration_report(
            self, report_for(39, run_stop="run_episode_terminal", run_terminals=1),
        )
        with self.assertRaises(AssertionError):
            _assert_completed_calibration_report(
                self, report_for(128, run_stop="fault", run_faults=1),
            )
        with self.assertRaises(AssertionError):
            _assert_completed_calibration_report(
                self, report_for(39, run_stop="wall_budget", status="incomplete", exit_code=1),
            )

    def test_trial_identity_uses_actual_collector_local_ids(self):
        from spire_learn.neural.calibration import _calibrate_with_dependencies, _manifest_projection

        config = load_neural_config(ROOT / "configs/neural/calibration.toml")
        manifest = _manifest_projection(config, {"gpu_uuid": GPU_UUID})
        identity = manifest["trial_identity"]
        self.assertEqual(identity["collector_stream_id_format"], "stream-{slot}")
        self.assertEqual(identity["collector_task_id_formats"]["standalone"], "task-{slot}-{episode}")
        self.assertEqual(
            identity["collector_task_id_formats"]["run"],
            "task-{slot}-run-{episode}-visit-{visit}",
        )
        self.assertEqual(identity["qualified_identity_format"], "{trial_id}/{stream_id}/{task_id}")
        self.assertNotIn("stream_id_prefix", str(manifest))

        dependencies = _SuccessfulFakeDependencies()
        report = _calibrate_with_dependencies(config, dependencies, clock=dependencies.clock)
        first_identity = report.summary["trials"][0]["collector_local_identities"][0]
        self.assertEqual(first_identity["stream_id"], "stream-0")
        self.assertEqual(first_identity["task_id"], "task-0-0")
        self.assertEqual(first_identity["qualified_id"], "standalone-b4-h32/stream-0/task-0-0")

    def test_cli_help_validation_and_gate_off_discovery_are_optional_import_free(self):
        program = r'''
import builtins
import io
import sys
import unittest
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
from spire_learn.neural.cli import main
from spire_learn.neural.runner import calibrate
import task8_calibration_run_helpers
assert main(["--help"]) == 0
assert main(["calibrate", "--config", "configs/neural/calibration.toml", "--validate-only"]) == 0
suite = unittest.defaultTestLoader.discover("tests/integration", pattern="test_neural_calibration.py")
assert suite.countTestCases() == 1
result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
assert result.testsRun == 0
assert len(result.skipped) == 1
assert result.failures == [] and result.errors == []
assert attempts == [], attempts
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        environment.pop("SPIRE_LEARN_RUN_TASK8_CALIBRATION", None)
        environment.pop("SPIRE_LEARN_TASK8_GPU_UUID", None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", program], cwd=ROOT, env=environment,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class _NoFactoryDependencies:
    def __init__(self):
        self.calls = []

    def verify_device(self, *args, **kwargs):
        self.calls.append("verify_device")

    def create_writer(self, *args, **kwargs):
        self.calls.append("create_writer")

    def build_trial(self, *args, **kwargs):
        self.calls.append("build_trial")


class _FakeClock:
    def __init__(self):
        self.value = 0.0
        self.reset_advance = 0.0

    def __call__(self):
        return self.value

    def advance(self, amount: float):
        self.value += amount


class _FakeWriter:
    run_dir = None

    def __init__(self):
        self.events = []
        self.close_count = 0
        self.started = False
        self.finalized = False

    def start(self, config, manifest):
        self.started = True
        self.config = config
        self.manifest = manifest

    def event(self, value):
        self.events.append(value)

    def finalize(self, summary):
        self.finalized = True
        self.summary = summary

    def close(self):
        self.close_count += 1


class _FakePolicy:
    model_id = "spire-learn-combat-ff-v1"
    features_id = "spire-learn-combat-features-v1"
    action_id = "spire-learn-combat-dispatch-v1"
    objective_id = "spire-learn-combat-winloss-v1"
    _behavior_policy_version = 0
    _parameter_count = 265_122

    def act(self, frames, *, generator, deterministic=False):
        from spire_learn.neural.contracts import PolicyChoice

        del generator, deterministic
        choices = []
        for frame in frames:
            mask = frame.tensors["action_mask"].data
            action = next(index for index, value in enumerate(mask) if value)
            choices.append(PolicyChoice(action, 0.0, 0.0, self._behavior_policy_version))
        return tuple(choices)

    def values(self, frames):
        return (0.0,) * len(frames)


def _synthetic_schema():
    from tests.neural._support import synthetic_schema

    return synthetic_schema()


class _FakeDriver:
    def __init__(self, config, schema, *, clock, terminate_slot_once=None, fail_on_step=None,
                 on_close=None):
        from spire_learn.neural.envs import DriverRow
        from spire_learn.neural.schema import candidates

        self.mode = config.mode
        self.batch_size = config.rollout_rows
        self.schema = schema
        self._rows = (None,) * self.batch_size
        self._accepted_steps = 0
        self._closed = False
        self.close_calls = 0
        self.clock = clock
        self.terminate_slot_once = terminate_slot_once
        self.terminated = False
        self.fail_on_step = fail_on_step
        self.on_close = on_close
        self.step_calls = 0
        self.reset_masks = []
        self.DriverRow = DriverRow
        self.candidates = candidates

    @property
    def rows(self):
        return self._rows

    @property
    def accepted_steps(self):
        return self._accepted_steps

    def _row(self, *, active=True, settled=False):
        from spire_learn.neural.contracts import Lifecycle
        from tests.neural._support import authored_frame

        frame = authored_frame(self.schema)
        lifecycle = (
            Lifecycle(True, "combat", None, 0, 1, 0, None)
            if active else Lifecycle(
                False, "noncombat", None, None, 1, 1,
                {"visit": 0, "encounter": "cultist", "outcome": "victory",
                 "hp_before_healing": 40, "hp_after_cleanup": 40},
            )
        )
        return self.DriverRow(
            frame,
            lifecycle,
            self.candidates(frame, self.schema) if active else (),
            (),
        )

    def reset(self, *, seeds, entry_ids, reset_mask=None):
        mask = tuple(reset_mask or (True,) * self.batch_size)
        self.reset_masks.append(mask)
        rows = list(self._rows)
        for index, selected in enumerate(mask):
            if selected:
                rows[index] = self._row()
        self._rows = tuple(rows)
        self.clock.advance(self.clock.reset_advance)
        return self._rows

    def step(self, actions):
        from spire_learn.neural.envs import DriverFault, DriverFaultEvidence, DriverTransition

        self.step_calls += 1
        if self.fail_on_step == self.step_calls:
            raise DriverFault(DriverFaultEvidence(
                phase="step", accepted_prefix=self._accepted_steps,
                failed_operation=self._accepted_steps + 2, delivery="unknown",
                exception_type="RuntimeError", message="injected fake accepted-step fault",
                operation_slots=self.batch_size,
                operation_slot_indices=tuple(range(self.batch_size)),
                uncertain_current_slots=self.batch_size,
            ))
        transitions = []
        successors = []
        for slot, source in enumerate(self._rows):
            assert source is not None
            terminate = self.terminate_slot_once == slot and not self.terminated
            successor = self._row(active=not terminate)
            transitions.append(DriverTransition(source, successor, 0.0, terminate, False))
            successors.append(successor)
            if terminate:
                self.terminated = True
        self._rows = tuple(successors)
        self._accepted_steps += 1
        self.clock.advance(0.001)
        return tuple(transitions)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close(None, None, None)


class _SuccessfulFakeDependencies:
    def __init__(self):
        self.clock = _FakeClock()
        self.writer = _FakeWriter()
        self.factory_order = []
        self.trials = []
        self.comparison_frames = []
        self.update_frames = []
        self.contexts = []
        self.fail_batch = None

    def verify_device(self, config, budget):
        del budget
        return {"gpu_uuid": config.calibration_gpu_uuid, "fake": True}

    def create_writer(self, config, run_id):
        del config, run_id
        return self.writer

    def build_trial(self, config, spec, recovery, budget, ledger, timer):
        from spire_learn.neural.calibration import _assemble_trial

        self.factory_order.append(spec.trial_id)
        context = _synthetic_schema()
        schema = context.__enter__()
        self.contexts.append(context)
        driver = _FakeDriver(
            config, schema, clock=self.clock,
            fail_on_step=2 if self.fail_batch == spec.batch_size else None,
            on_close=context.__exit__,
        )
        trial = _assemble_trial(
            config, spec, recovery, budget, ledger,
            driver_factory=lambda _config, _cases: driver,
            policy_factory=lambda _schema, _model, **_kwargs: _FakePolicy(),
            generator_factory=lambda _config: object(),
            timer=timer,
        )
        self.trials.append(trial)
        return trial

    def compare(self, trial, frames, budget):
        from spire_learn.neural.calibration import _frame_sha256

        budget.check("fake_comparison")
        self.comparison_frames.append(frames)
        self.clock.advance(0.05)
        if self.fail_batch == trial.spec.batch_size:
            raise RuntimeError("injected comparison failure")
        return {
            "measurement_complete": True,
            "cpu_finite": True,
            "gpu_finite": True,
            "frame_sha256": [_frame_sha256(frame) for frame in frames],
            "same_initial_parameter_sha256": True,
        }

    def optimizer_probe(self, trial, frames, budget, ledger, reservation):
        budget.check("fake_optimizer")
        self.update_frames.append(frames)
        self.clock.advance(0.01)
        ledger.mark_optimizer_attempted(reservation)
        ledger.mark_optimizer_accepted(reservation)
        return {
            "attempted_steps": 1, "accepted_steps": 1, "uncertain_steps": 0,
            "finite": True, "parameters_changed": True,
        }

    def memory_report(self):
        return {"fake": True}


class _ResetExhaustionDependencies(_SuccessfulFakeDependencies):
    def build_trial(self, config, spec, recovery, budget, ledger, timer):
        trial = super().build_trial(config, spec, recovery, budget, ledger, timer)
        trial.driver.clock.reset_advance = 301.0
        return trial


class _PartialFailureDependencies(_SuccessfulFakeDependencies):
    def build_trial(self, config, spec, recovery, budget, ledger, timer):
        self.fail_batch = spec.batch_size
        return super().build_trial(config, spec, recovery, budget, ledger, timer)

    def compare(self, trial, frames, budget):
        raise AssertionError("faulted collection must not proceed to comparison")


class _ComparisonFailureDependencies(_SuccessfulFakeDependencies):
    def __init__(self, *, fail_batch):
        super().__init__()
        self.fail_batch = fail_batch


class _ControlFailureDependencies(_SuccessfulFakeDependencies):
    def compare(self, trial, frames, budget):
        self.comparison_frames.append(frames)
        raise KeyboardInterrupt()


if __name__ == "__main__":
    unittest.main()
