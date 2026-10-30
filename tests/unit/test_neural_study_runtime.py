"""Strict fake-boundary tests for the Task 9 outer study runtime."""

from __future__ import annotations

import importlib
import importlib.abc
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
STUDY_RELATIVE = Path("configs/neural/experiments/task9-v1/study.json")


def _copy_study_checkout(root: Path) -> Path:
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "configs", root / "configs")
    scripts = root / "scripts" / "neural"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/neural/task9_study_package.py", scripts)
    shutil.copy2(ROOT / "scripts/neural/task9_study_runtime.py", scripts)
    shutil.copy2(ROOT / "scripts/neural/task9_nvidia_cdi.py", scripts)
    return root


def _write_valid_checkpoint(path: Path, config, runtime) -> None:
    from spire_learn.neural.checkpoint import CHECKPOINT_SCHEMA_ID, identity_payload

    path.mkdir()
    payloads = {}
    for name, data in (("model.pt", b"model-payload"), ("training.pt", b"training-payload")):
        payload_path = path / name
        with zipfile.ZipFile(payload_path, "w") as archive:
            archive.writestr("weights/data.bin", data)
        raw = payload_path.read_bytes()
        payloads[name] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    diagnostics = runtime.collector.last_rollout.collection if runtime.collector.last_rollout else None
    progress = {
        name: (getattr(diagnostics, name, 0) if diagnostics is not None else 0)
        for name in (
            "attempted_environment_slots", "forwarded_environment_slots",
            "known_accepted_environment_slots", "delivered_environment_slots",
            "uncertain_environment_slots", "attempted_reset_slots",
            "forwarded_reset_slots", "known_accepted_reset_slots",
            "delivered_reset_slots", "uncertain_reset_slots", "learned_decisions",
            "fixed_decisions", "incomplete_decisions",
        )
    }
    progress.update({
        "optimizer_attempted_steps": runtime.optimizer_attempted_steps,
        "optimizer_accepted_steps": runtime.optimizer_accepted_steps,
        "optimizer_uncertain_steps": runtime.optimizer_uncertain_steps,
        "update_count": runtime.update_count,
        "elapsed_seconds": 0.0,
    })
    identity = next(item for item in config.accepted_identities
                    if config.mode in item.accepted_raw_mode_mappings)
    metadata = {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "identity": identity_payload(identity),
        "payload_inventory": payloads,
        "model_tensors": [{
            "name": "test.weight", "dtype": "float32", "shape": [1],
            "sha256": "e" * 64,
        }],
        "training_state": {
            "optimizer_step": runtime.optimizer_accepted_steps,
            "hyperparameter_id": config.accepted_identities[0].config_hashes["ppo"],
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": progress,
            "update_id": f"update-{runtime.update_count}",
            "behavior_policy_version": runtime.collector.policy._behavior_policy_version,
            "next_environment_seed_ordinal": progress["delivered_reset_slots"],
            "next_decision_sequence": progress["delivered_environment_slots"],
            "saved_inflight_record_count": 0,
        },
        "run_metadata": {
            "mode": config.mode, "operation": "train", "split": "train",
            "partition": "train", "config_sha256": config.config_sha256,
            "manifest_sha256": config.manifest_sha256,
            "environment_revision": config.environment_revision,
            "cumulative_prefix": progress,
        },
        "parent_checkpoint": None,
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class _StudyCollector:
    def __init__(self, config, rollouts):
        from spire_learn.neural.collector import CollectorRecovery

        self.config = config
        self.rollouts = list(rollouts)
        self.last_rollout = None
        self.collect_calls = []
        self.learned_combat_policy = True
        self.recovery_state = CollectorRecovery(0, 0, (0,) * config.rollout_rows)
        self.policy = SimpleNamespace(_behavior_policy_version=0)
        self.driver = SimpleNamespace(schema=None)
        self.budget = None
        self.close_calls = 0

    def collect(self, *, max_learned_decisions, max_env_actions, policy_version):
        self.collect_calls.append((max_learned_decisions, max_env_actions, policy_version))
        if not self.rollouts:
            raise AssertionError("unexpected collector dispatch")
        self.last_rollout = self.rollouts.pop(0)
        return self.last_rollout

    def close(self):
        self.close_calls += 1


class _StudyRuntime:
    def __init__(self, config, collector):
        from spire_learn.neural.runner import RuntimeOperationLimits

        self.config = config
        self.collector = collector
        self.operation_limits = RuntimeOperationLimits()
        self.recovered_progress = {}
        self.optimizer_attempted_steps = 0
        self.optimizer_accepted_steps = 0
        self.optimizer_uncertain_steps = 0
        self.update_count = 0
        self.policy_kind = "checkpoint"
        self.action_selection = "seeded-sampling"
        self.controller_id = "strict-fake-runtime-v1"
        self.parent_checkpoint = None
        self.device_report = None
        self.budget = None
        self.close_calls = 0
        self.update_calls = 0
        self.saved = []

    def set_budget(self, budget):
        self.budget = budget
        self.collector.budget = budget

    def update(self, rollout):
        from spire_learn.neural.ppo import UpdateMetrics

        actors = sum(record.actor_training_mask for record in rollout.records)
        steps = self.config.ppo.epochs * (
            (actors + self.config.ppo.minibatch_rows - 1) // self.config.ppo.minibatch_rows
        )
        version = self.collector.policy._behavior_policy_version
        self.collector.policy._behavior_policy_version += int(steps > 0)
        self.update_calls += 1
        return UpdateMetrics(0.0, 0.0, 0.0, steps, steps, None,
                             version, self.collector.policy._behavior_policy_version, actors)

    def record_update(self, metrics):
        self.optimizer_attempted_steps += metrics.attempted_steps
        self.optimizer_accepted_steps += metrics.accepted_steps
        self.update_count += 1

    def save(self, path, purpose):
        self.saved.append((Path(path), purpose))
        _write_valid_checkpoint(Path(path), self.config, self)

    def close(self):
        self.close_calls += 1
        self.collector.close()


def _training_rollout(config):
    from tests.unit.test_neural_runner import _allocation, _record, _rollout

    cases = config.cases.for_mode(config.mode)
    records = []
    allocations = []
    for slot in range(config.rollout_rows):
        allocations.append(_allocation(config, slot, slot, case_index=slot % len(cases)))
        records.append(_record(
            config, slot=slot, decision=0, version=0,
            terminal=config.mode == "standalone",
        ))
    return _rollout(config, records, allocations=allocations,
                    stop_reason="inactive_reset_boundary")


def _empty_evaluation_rollout(config):
    from tests.unit.test_neural_runner import _rollout

    return _rollout(config, (), stop_reason="inactive_reset_boundary")


def _evaluation_rollout(config, *, learned: bool):
    from dataclasses import replace
    from tests.unit.test_neural_runner import _allocation, _record, _rollout

    allocation = _allocation(config, 0, 0, case_index=0)
    record = _record(
        config, slot=0, decision=0, version=0,
        terminal=True, learned=learned,
        task_id="task-0-run-0" if config.mode == "run" else None,
    )
    rollout = _rollout(
        config, (record,), allocations=(allocation,),
        stop_reason="inactive_reset_boundary",
    )
    if config.mode == "run":
        record = replace(record, terminated=True)
        counters = replace(rollout.counters, environment_terminations=1)
        rollout = replace(rollout, records=(record,), counters=counters)
    return rollout


def _training_receipt(root: Path, policy_id: str, *, deadline: float = 10_000.0,
                      clock=lambda: 0.0, factory_hook=None):
    from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
    from spire_learn.neural import config as config_module, runner
    from spire_learn.neural.config import load_neural_config

    study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
    record = next(item for item in study["training"]["runs"] if item["policy_id"] == policy_id)
    with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
        config = load_neural_config(root / record["config"]["path"])
        runtime = _StudyRuntime(config, _StudyCollector(config, [_training_rollout(config)]))

        def factory(actual_config, **kwargs):
            if factory_hook is not None:
                factory_hook(actual_config, runtime, kwargs)
            return runtime

        receipt_path = root / record["output_root"] / "operation-receipt.json"
        receipt = run_training_operation(
            root, STUDY_RELATIVE, policy_id, receipt_path, deadline=deadline,
            clock=clock, dependencies=RuntimeDependencies(factory=factory, train=runner.train),
        )
    return receipt


def _freeze_four_training_bindings(root: Path):
    from scripts.neural.task9_study_runtime import freeze_checkpoint_bindings

    study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
    receipts = {
        item["policy_id"]: root / item["output_root"] / "operation-receipt.json"
        for item in study["training"]["runs"]
    }
    destination = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection/control/checkpoint-bindings.json"
    return freeze_checkpoint_bindings(root, STUDY_RELATIVE, receipts, destination), receipts, destination


def _freeze_training_policies(root: Path, available: tuple[str, ...]):
    from scripts.neural.task9_study_runtime import freeze_checkpoint_bindings

    study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
    receipt_paths = {}
    for item in study["training"]["runs"]:
        policy_id = item["policy_id"]
        receipt_paths[policy_id] = (
            root / item["output_root"] / "operation-receipt.json"
            if policy_id in available else None
        )
    destination = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection/control/checkpoint-bindings.json"
    return freeze_checkpoint_bindings(root, STUDY_RELATIVE, receipt_paths, destination), receipt_paths, destination


def _selection_receipt_path(root: Path, operation_id: str) -> Path:
    return root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection/control/operations" / f"{operation_id}.json"


def _selection_reservation_path(root: Path, operation_id: str) -> Path:
    return root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection/control/operations" / f"{operation_id}.reservation.json"


class NeuralStudyRuntimeTests(unittest.TestCase):
    def test_public_study_runtime_api_exists(self):
        module = importlib.import_module("scripts.neural.task9_study_runtime")
        self.assertTrue(callable(module.run_training_operation))
        self.assertTrue(callable(module.run_selection_operation))
        self.assertTrue(callable(module.run_selection_schedule))
        self.assertTrue(callable(module.freeze_checkpoint_bindings))

    def test_training_factory_time_uses_the_same_deadline_and_actual_runner_closes_once(self):
        from spire_learn.neural import runner
        from spire_learn.neural.config import load_neural_config
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study_path = STUDY_RELATIVE
            study = json.loads((root / study_path).read_text(encoding="utf-8"))
            training = study["training"]["runs"][0]
            from spire_learn.neural import config as config_module
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / training["config"]["path"])
                collector = _StudyCollector(config, ())
                runtime = _StudyRuntime(config, collector)
                now = [0.0]
                factory_calls = []

                def factory(actual_config, **kwargs):
                    factory_calls.append((kwargs, now[0]))
                    scratch = actual_config.output_root / ".runtime"
                    expected = {
                        "tmp", "cache", "torch", "torch-extensions",
                        "cuda", "triton", "home",
                    }
                    self.assertEqual({path.name for path in scratch.iterdir()}, expected)
                    self.assertTrue(all(path.is_dir() and not path.is_symlink()
                                        and not any(path.iterdir())
                                        for path in scratch.iterdir()))
                    now[0] = 5.0
                    return runtime

                def train(actual_config, *, runtime, clock):
                    self.assertEqual(runtime.operation_limits.monotonic_deadline, 4.0)
                    return runner.train(actual_config, runtime=runtime, clock=clock)

                receipt_path = root / training["output_root"] / "operation-receipt.json"
                receipt = run_training_operation(
                    root, study_path, "standalone-a", receipt_path,
                    deadline=4.0, clock=lambda: now[0],
                    dependencies=RuntimeDependencies(factory=factory, train=train),
                )

            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(factory_calls[0][1], 0.0)
            self.assertIsNone(factory_calls[0][0]["checkpoint"])
            self.assertEqual(receipt["status"], "unavailable")
            self.assertEqual(receipt["unavailable_reason"], "zero_accepted_optimizer_updates")
            self.assertEqual(receipt["report"]["optimizer"]["accepted_steps"], 0)
            self.assertEqual(collector.collect_calls, [])
            self.assertEqual(runtime.update_calls, 0)
            self.assertEqual(runtime.close_calls, 1)
            self.assertEqual(len(runtime.saved), 1)
            self.assertEqual(receipt["lifecycle"]["status"], "complete")
            self.assertEqual(receipt["run_id"], Path(receipt["run_directory"]).name)
            self.assertEqual(len(receipt["summary_sha256"]), 64)
            self.assertEqual(len(receipt["lifecycle_sha256"]), 64)

    def test_exhausted_training_deadline_never_enters_factory(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            record = study["training"]["runs"][0]
            calls = []
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a",
                    root / record["output_root"] / "operation-receipt.json",
                    deadline=10.0, clock=lambda: 10.0,
                    dependencies=RuntimeDependencies(factory=lambda *a, **k: calls.append((a, k))),
                )

            self.assertEqual(calls, [])
            self.assertEqual(receipt["status"], "not_started")
            self.assertFalse(receipt["attempted_effects"])
            self.assertEqual(receipt["outer_timing"]["elapsed_seconds"], 0.0)

    def test_training_admission_rejects_preexisting_runtime_scratch_or_output(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module

        for kind in ("runtime_directory", "runtime_symlink", "other_output"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = _copy_study_checkout(Path(directory) / "checkout")
                record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
                output_root = root / record["output_root"]
                output_root.mkdir(parents=True)
                if kind == "runtime_directory":
                    (output_root / ".runtime").mkdir()
                elif kind == "runtime_symlink":
                    external = root / "outside-scratch"
                    external.mkdir()
                    (output_root / ".runtime").symlink_to(external, target_is_directory=True)
                else:
                    (output_root / "prior-run.json").write_text("{}\n", encoding="utf-8")
                factory_calls = []
                with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                    with self.assertRaises(FileExistsError):
                        run_training_operation(
                            root, STUDY_RELATIVE, "standalone-a",
                            output_root / "operation-receipt.json",
                            deadline=10_000.0, clock=lambda: 0.0,
                            dependencies=RuntimeDependencies(
                                factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
                            ),
                        )

                self.assertEqual(factory_calls, [])
                self.assertFalse((output_root / "operation-reservation.json").exists())

    def test_default_factory_gates_neural_torch_and_keeps_reference_torch_free(self):
        from scripts.neural.task9_study_runtime import _default_factory
        from spire_learn.neural import config as config_module, device, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            training = study["training"]["runs"][0]
            selection_row = next(
                row for row in study["selection"]["schedule"] if row["policy_id"] == "legal-random"
            )
            events = []
            torch_module = ModuleType("torch")
            thread_state = {"threads": None, "interop": None}
            torch_module.set_num_threads = lambda value: thread_state.update(threads=value)
            torch_module.set_num_interop_threads = lambda value: thread_state.update(interop=value)
            torch_module.get_num_threads = lambda: thread_state["threads"]
            torch_module.get_num_interop_threads = lambda: thread_state["interop"]
            original_import = builtins.__import__
            sentinel = SimpleNamespace()

            def fake_import(name, *args, **kwargs):
                if name == "torch":
                    events.append("torch")
                    return torch_module
                return original_import(name, *args, **kwargs)

            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                neural_config = load_neural_config(root / training["config"]["path"])
                reference_config = load_neural_config(root / selection_row["config_path"])
                with patch.object(device, "plan_runtime_device", side_effect=lambda config: events.append("plan") or object()):
                    with patch.object(device, "require_runtime_execution_gate", side_effect=lambda plan: events.append("gate")):
                        with patch.object(runner, "create_runtime", side_effect=lambda *a, **k: events.append("neural_create") or sentinel):
                            with patch("builtins.__import__", side_effect=fake_import):
                                self.assertIs(_default_factory(
                                    neural_config, purpose="train", checkpoint=None,
                                    policy_kind="checkpoint", action_selection="seeded-sampling",
                                ), sentinel)
                self.assertEqual(events, ["plan", "gate", "torch", "neural_create"])
                self.assertEqual(thread_state, {"threads": 4, "interop": 1})
                self.assertEqual(sentinel.task9_thread_settings["intraop_threads"], 4)
                self.assertEqual(sentinel.task9_thread_settings["interop_threads"], 1)
                self.assertTrue(sentinel.task9_thread_settings["verified_after_runtime_assembly"])
                events.clear()
                with patch.object(device, "plan_runtime_device", side_effect=AssertionError("reference must skip device gate")):
                    with patch.object(device, "require_runtime_execution_gate", side_effect=AssertionError("reference must skip execution gate")):
                        with patch.object(runner, "create_runtime", side_effect=lambda *a, **k: events.append("reference_create") or sentinel):
                            with patch("builtins.__import__", side_effect=fake_import):
                                self.assertIs(_default_factory(
                                    reference_config, purpose="evaluate", checkpoint=None,
                                    policy_kind="legal-random", action_selection="seeded-sampling",
                                ), sentinel)
            self.assertEqual(events, ["reference_create"])

    def test_default_factory_gate_failure_prevents_torch_import(self):
        from scripts.neural.task9_study_runtime import _default_factory
        from spire_learn.neural import config as config_module, device, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
            attempts = []
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "torch":
                    attempts.append(name)
                    raise AssertionError("Torch must remain outside a failed device gate")
                return original_import(name, *args, **kwargs)

            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / record["config"]["path"])
                with patch.object(device, "plan_runtime_device", return_value=object()):
                    with patch.object(device, "require_runtime_execution_gate", side_effect=RuntimeError("gate closed")):
                        with patch.object(runner, "create_runtime", side_effect=AssertionError("runtime factory must not run")):
                            with patch("builtins.__import__", side_effect=guarded_import):
                                with self.assertRaisesRegex(RuntimeError, "gate closed"):
                                    _default_factory(
                                        config, purpose="train", checkpoint=None,
                                        policy_kind="checkpoint", action_selection="seeded-sampling",
                                    )
            self.assertEqual(attempts, [])

    def test_training_setup_failure_closes_owned_runtime_and_publishes_fault_receipt(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module
        from spire_learn.neural.runner import RuntimeOperationLimits

        class SetupFailureRuntime:
            def __init__(self):
                self.close_calls = 0

            @property
            def operation_limits(self):
                return RuntimeOperationLimits()

            @operation_limits.setter
            def operation_limits(self, value):
                raise RuntimeError("injected deadline installation failure")

            def close(self):
                self.close_calls += 1
                raise OSError("injected setup cleanup failure")

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
            runtime = SetupFailureRuntime()
            trainer_calls = []
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a",
                    root / record["output_root"] / "operation-receipt.json",
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime,
                        train=lambda *args, **kwargs: trainer_calls.append((args, kwargs)),
                    ),
                )

            self.assertEqual(receipt["status"], "operational_failure")
            self.assertEqual(receipt["error"]["message"], "injected deadline installation failure")
            self.assertFalse(receipt["attempted_effects"])
            self.assertEqual(runtime.close_calls, 1)
            self.assertEqual(trainer_calls, [])

    def test_runner_save_fault_is_an_operation_failure_without_checkpoint_substitution(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / record["config"]["path"])
                runtime = _StudyRuntime(config, _StudyCollector(config, [_training_rollout(config)]))
                runtime.save = lambda path, purpose: (_ for _ in ()).throw(OSError("injected checkpoint save fault"))
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a",
                    root / record["output_root"] / "operation-receipt.json",
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime, train=runner.train,
                    ),
                )

            self.assertEqual(receipt["status"], "operational_failure")
            self.assertIsNone(receipt["checkpoint"])
            self.assertEqual(receipt["lifecycle"]["status"], "artifact_failure")
            self.assertEqual(runtime.close_calls, 1)

    def test_runner_close_fault_is_reported_by_the_outer_training_receipt(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / record["config"]["path"])
                runtime = _StudyRuntime(config, _StudyCollector(config, [_training_rollout(config)]))
                actual_close = runtime.close

                def close_then_fail():
                    actual_close()
                    raise OSError("injected runtime close fault")

                runtime.close = close_then_fail
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a",
                    root / record["output_root"] / "operation-receipt.json",
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime, train=runner.train,
                    ),
                )

            self.assertEqual(receipt["status"], "operational_failure")
            self.assertEqual(receipt["lifecycle"]["status"], "operational_failure")
            self.assertEqual(receipt["close_status"], "reported_failure")
            self.assertEqual(runtime.close_calls, 1)

    def test_training_receipt_publication_failure_raises_after_runner_close(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / record["config"]["path"])
                runtime = _StudyRuntime(config, _StudyCollector(config, [_training_rollout(config)]))
                module = importlib.import_module("scripts.neural.task9_study_runtime")
                original_exclusive_json = module._exclusive_json

                def fail_receipt_publication(path, value):
                    if Path(path).name == "operation-receipt.json":
                        raise OSError("injected training receipt publication failure")
                    return original_exclusive_json(path, value)

                receipt_path = root / record["output_root"] / "operation-receipt.json"
                with patch.object(module, "_exclusive_json", side_effect=fail_receipt_publication):
                    with self.assertRaisesRegex(OSError, "training receipt publication failure"):
                        run_training_operation(
                            root, STUDY_RELATIVE, "standalone-a", receipt_path,
                            deadline=10_000.0, clock=lambda: 0.0,
                            dependencies=RuntimeDependencies(
                                factory=lambda *args, **kwargs: runtime, train=runner.train,
                            ),
                        )

            self.assertEqual(runtime.close_calls, 1)
            self.assertFalse(receipt_path.exists())
            self.assertTrue(list(receipt_path.parent.glob("*/checkpoint-final")))

    def test_unavailable_frozen_model_is_not_started_without_runtime_factory(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_operation
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                row = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"][0]
                factory_calls = []
                receipt_path = _selection_receipt_path(root, row["operation_id"])
                receipt = run_selection_operation(
                    root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                    receipt_path, deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(factory=lambda *a, **k: factory_calls.append((a, k))),
                )

            self.assertEqual(receipt["status"], "not_started")
            self.assertEqual(receipt["transport_exit_code"], 0)
            self.assertFalse(receipt["attempted_effects"])
            self.assertEqual(factory_calls, [])
            self.assertFalse(_selection_reservation_path(root, row["operation_id"]).exists())

    def test_accepted_runner_receipt_binds_final_checkpoint_and_real_lifecycle(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_training_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            record = study["training"]["runs"][0]
            now = [0.0]

            def clock():
                now[0] += 0.1
                return now[0]

            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / record["config"]["path"])
                runtime = _StudyRuntime(config, _StudyCollector(config, [_training_rollout(config)]))
                runtime.task9_thread_settings = {
                    "intraop_threads": 4, "interop_threads": 1,
                    "process_id": os.getpid(),
                    "verified_before_runtime_assembly": True,
                    "verified_after_runtime_assembly": True,
                }
                receipt_path = root / record["output_root"] / "operation-receipt.json"
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a", receipt_path,
                    deadline=10_000.0, clock=clock,
                    dependencies=RuntimeDependencies(
                        factory=lambda actual_config, **kwargs: runtime,
                        train=runner.train,
                    ),
                )

            self.assertEqual(receipt["status"], "eligible")
            self.assertEqual(receipt["lifecycle"]["status"], "complete")
            self.assertEqual(receipt["runtime_thread_settings"]["intraop_threads"], 4)
            self.assertEqual(receipt["runtime_thread_settings"]["interop_threads"], 1)
            self.assertEqual(receipt["runtime_thread_settings"]["process_id"], os.getpid())
            scratch_root = (Path(record["output_root"]) / ".runtime").as_posix()
            self.assertEqual(receipt["runtime_scratch"]["root"], scratch_root)
            self.assertEqual(
                receipt["runtime_scratch"]["directories"],
                {
                    name: f"{scratch_root}/{name}" for name in (
                        "tmp", "cache", "torch", "torch-extensions",
                        "cuda", "triton", "home",
                    )
                },
            )
            self.assertEqual(receipt["report"]["optimizer"]["accepted_steps"], 4)
            self.assertEqual(receipt["report"]["optimizer"]["uncertain_steps"], 0)
            self.assertEqual(receipt["attempted_effects"], True)
            self.assertEqual(
                {name: receipt["counts"][name] for name in (
                    "attempted_environment_slots", "forwarded_environment_slots",
                    "known_accepted_environment_slots", "delivered_environment_slots",
                    "uncertain_environment_slots", "attempted_reset_slots",
                    "forwarded_reset_slots", "known_accepted_reset_slots",
                    "delivered_reset_slots", "uncertain_reset_slots",
                )},
                {
                    "attempted_environment_slots": 16, "forwarded_environment_slots": 16,
                    "known_accepted_environment_slots": 16, "delivered_environment_slots": 16,
                    "uncertain_environment_slots": 0, "attempted_reset_slots": 16,
                    "forwarded_reset_slots": 16, "known_accepted_reset_slots": 16,
                    "delivered_reset_slots": 16, "uncertain_reset_slots": 0,
                },
            )
            self.assertEqual(receipt["checkpoint"]["directory"],
                             (Path(receipt["run_directory"]) / "checkpoint-final").as_posix())
            self.assertEqual(set(receipt["checkpoint"]["inventory"]),
                             {"metadata.json", "model.pt", "training.pt"})
            self.assertEqual(receipt["checkpoint"]["model_tensors"][0]["name"], "test.weight")
            self.assertNotEqual(
                receipt["checkpoint"]["metadata"]["training_state"]["counters"]["elapsed_seconds"],
                receipt["lifecycle"]["elapsed_seconds"],
            )

    def test_binding_freeze_uses_four_receipts_and_actual_immutable_preflight(self):
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                receipts = {
                    policy_id: _training_receipt(root, policy_id)
                    for policy_id in ("standalone-a", "run-a", "standalone-b", "run-b")
                }
                bindings, receipt_paths, destination = _freeze_four_training_bindings(root)

            self.assertEqual(bindings["schema_id"], "spire-learn-task9-checkpoint-bindings-v1")
            self.assertEqual(bindings["state"], "frozen_before_selection")
            self.assertEqual([item["policy_id"] for item in bindings["policies"]],
                             ["standalone-a", "run-a", "standalone-b", "run-b"])
            for policy in bindings["policies"]:
                with self.subTest(policy=policy["policy_id"]):
                    self.assertEqual(policy["status"], "available")
                    source = receipt_paths[policy["policy_id"]]
                    self.assertEqual(policy["training_receipt"]["raw_sha256"],
                                     hashlib.sha256(source.read_bytes()).hexdigest())
                    self.assertEqual(policy["training_receipt"]["run_id"],
                                     receipts[policy["policy_id"]]["run_id"])
                    self.assertTrue(policy["checkpoint_directory"].startswith(
                        "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/training/"))
                    self.assertEqual(set(policy["inventory"]),
                                     {"metadata.json", "model.pt", "training.pt"})
                    self.assertEqual(len(policy["inventory"]["model.pt"]["sha256"]), 64)
                    self.assertEqual(policy["model_tensors"][0]["name"], "test.weight")
            self.assertTrue(destination.is_file())

    def test_selection_runner_accepts_healthy_incomplete_and_keeps_transport_status_separate(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation,
        )
        from spire_learn.neural import config as config_module, runner

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                for policy_id in ("standalone-a", "run-a", "standalone-b", "run-b"):
                    _training_receipt(root, policy_id)
                bindings, _, binding_path = _freeze_four_training_bindings(root)
                row = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"][0]
                case_config = root / row["config_path"]

                def factory(config, *, purpose, checkpoint, policy_kind, action_selection):
                    runtime = _StudyRuntime(config, _StudyCollector(config, [_empty_evaluation_rollout(config)]))
                    runtime.policy_kind = policy_kind
                    runtime.action_selection = action_selection
                    runtime.parent_checkpoint = checkpoint
                    runtime.collector.learned_combat_policy = policy_kind == "checkpoint"
                    return runtime

                receipt = run_selection_operation(
                    root, STUDY_RELATIVE, row["operation_id"], binding_path,
                    _selection_receipt_path(root, row["operation_id"]),
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(factory=factory, evaluate=runner.evaluate_policy),
                )

                with self.assertRaises(FileExistsError):
                    run_selection_operation(
                        root, STUDY_RELATIVE, row["operation_id"], binding_path,
                        _selection_receipt_path(root, row["operation_id"]),
                        deadline=10_000.0, clock=lambda: 0.0,
                        dependencies=RuntimeDependencies(factory=factory, evaluate=runner.evaluate_policy),
                    )

            self.assertEqual(bindings["state"], "frozen_before_selection")
            self.assertEqual(receipt["status"], "incomplete")
            self.assertEqual(receipt["runner_exit_code"], 1)
            self.assertEqual(receipt["transport_exit_code"], 0)
            self.assertEqual(receipt["lifecycle"]["status"], "incomplete")
            self.assertEqual(receipt["case_accounting"]["scheduled"], 1)
            self.assertEqual(receipt["case_accounting"]["complete"], 0)
            self.assertEqual(receipt["optimizer"]["attempted_steps"], 0)
            self.assertEqual(receipt["checkpoint"]["before"], receipt["checkpoint"]["after"])
            self.assertTrue(case_config.is_file())

    def test_bad_checkpoint_metadata_preflight_stops_before_runtime_factory(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation,
        )
        from spire_learn.neural import config as config_module
        from spire_learn.neural.checkpoint import preflight_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _training_receipt(root, "standalone-a")
                _, _, binding_path = _freeze_training_policies(root, ("standalone-a",))
                study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
                row = study["selection"]["schedule"][0]
                binding = json.loads(binding_path.read_text(encoding="utf-8"))
                checkpoint = root / binding["policies"][0]["checkpoint_directory"]
                (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
                calls = []
                factory_calls = []

                def preflight(path, expected, *, purpose):
                    calls.append((Path(path), purpose))
                    return preflight_checkpoint(path, expected, purpose=purpose)

                def factory(*args, **kwargs):
                    factory_calls.append((args, kwargs))
                    raise AssertionError("bad metadata must not enter the runtime factory")

                prefixes = ("torch", "numpy", "tensordict", "torchrl", "gymnasium", "spire_lab")
                import_attempts = []

                class OptionalImportGuard(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if any(fullname == prefix or fullname.startswith(prefix + ".")
                               for prefix in prefixes):
                            import_attempts.append(fullname)
                            raise AssertionError(f"optional import entered before bad metadata rejection: {fullname}")
                        return None

                guard = OptionalImportGuard()
                sys.meta_path.insert(0, guard)

                try:
                    receipt = run_selection_operation(
                        root, STUDY_RELATIVE, row["operation_id"], binding_path,
                        _selection_receipt_path(root, row["operation_id"]),
                        deadline=10_000.0, clock=lambda: 0.0,
                        dependencies=RuntimeDependencies(
                            factory=factory, checkpoint_preflight=preflight,
                        ),
                    )
                finally:
                    sys.meta_path.remove(guard)

            self.assertEqual(calls, [(checkpoint, "evaluate")])
            self.assertEqual(factory_calls, [])
            self.assertEqual(import_attempts, [])
            self.assertEqual(receipt["status"], "operational_failure")
            self.assertEqual(receipt["attempted_effects"], False)
            self.assertEqual(receipt["checkpoint"]["before"], None)

    def test_full_selection_schedule_uses_all_192_actual_operation_receipts(self):
        import subprocess

        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation, run_selection_schedule,
        )
        from spire_learn.neural import config as config_module, runner

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                for policy_id in ("standalone-a", "run-a", "standalone-b", "run-b"):
                    _training_receipt(root, policy_id)
                _, _, binding_path = _freeze_four_training_bindings(root)
                study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
                schedule = study["selection"]["schedule"]
                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                selection_calls = []

                def factory(config, *, purpose, checkpoint, policy_kind, action_selection):
                    selection_calls.append((policy_kind, action_selection, checkpoint))
                    runtime = _StudyRuntime(config, _StudyCollector(
                        config, [_evaluation_rollout(config, learned=policy_kind == "checkpoint")],
                    ))
                    runtime.policy_kind = policy_kind
                    runtime.action_selection = action_selection
                    runtime.parent_checkpoint = checkpoint
                    runtime.collector.learned_combat_policy = policy_kind == "checkpoint"
                    return runtime

                operation_dependencies = RuntimeDependencies(
                    factory=factory, evaluate=runner.evaluate_policy,
                )

                def child_process(argv, *, capture_output, text, timeout, check,
                                  output_limit_bytes, progress_check):
                    args = {arg: argv[index + 1] for index, arg in enumerate(argv[:-1])
                            if arg.startswith("--")}
                    operation_id = args["--operation-id"]
                    self.assertGreater(timeout, 0)
                    progress_check()
                    run_selection_operation(
                        root, STUDY_RELATIVE, operation_id, binding_path,
                        Path(args["--receipt"]), deadline=float(args["--deadline"]),
                        clock=lambda: 0.0, dependencies=operation_dependencies,
                    )
                    progress_check()
                    return subprocess.CompletedProcess(argv, 0, "", "")

                result = run_selection_schedule(
                    root, STUDY_RELATIVE, binding_path, output_dir,
                    deadline=600.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        child_process=child_process, phase_context=root / "phase-context.json",
                    ),
                )

            self.assertEqual(result["schema_id"], "spire-learn-task9-selection-schedule-result-v1")
            self.assertEqual(len(result["rows"]), 192)
            self.assertEqual(len({(row["policy_id"], row["case_id"]) for row in result["rows"]}), 192)
            self.assertEqual(result["attempted_operation_count"], 192)
            self.assertEqual(result["complete_operation_count"], 192)
            self.assertEqual(result["incomplete_operation_count"], 0)
            self.assertEqual(result["not_started_operation_count"], 0)
            self.assertEqual(result["reserved_step_slots"], 49_152)
            self.assertNotIn("win_rate", result)
            self.assertEqual(len(selection_calls), 192)
            for name in (
                "attempted_reset_slots", "forwarded_reset_slots",
                "known_accepted_reset_slots", "delivered_reset_slots",
                "uncertain_reset_slots",
            ):
                expected_sum = 0 if name == "uncertain_reset_slots" else 192
                self.assertEqual(
                    result["progress_totals"][name],
                    {"known_sum": expected_sum, "unknown_rows": 0},
                )
            self.assertTrue((output_dir / "control/schedule-plan.json").is_file())
            self.assertTrue((output_dir / "control/schedule-result.json").is_file())
            for row in schedule:
                receipt_path = _selection_receipt_path(root, row["operation_id"])
                self.assertTrue(receipt_path.is_file())
                self.assertTrue(_selection_reservation_path(root, row["operation_id"]).is_file())
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    {name: receipt["counts"][name] for name in (
                        "attempted_environment_slots", "forwarded_environment_slots",
                        "known_accepted_environment_slots", "delivered_environment_slots",
                        "uncertain_environment_slots",
                    )},
                    {
                        "attempted_environment_slots": 1, "forwarded_environment_slots": 1,
                        "known_accepted_environment_slots": 1, "delivered_environment_slots": 1,
                        "uncertain_environment_slots": 0,
                    },
                )
                if row["policy_id"] in {"legal-random", "end-turn-last-v1"}:
                    self.assertEqual(receipt["counts"]["learned_decisions"], 0)
                    self.assertEqual(receipt["counts"]["fixed_decisions"], 1)
                    guard = receipt["reference_import_guard"]
                    self.assertTrue(guard["installed"])
                    self.assertEqual(guard["attempts"], [])
                    self.assertEqual(guard["loaded_modules"], [])
                else:
                    self.assertEqual(receipt["counts"]["learned_decisions"], 1)
                    self.assertEqual(receipt["counts"]["fixed_decisions"], 0)
                    self.assertEqual(receipt["optimizer"]["attempted_steps"], 0)
                    self.assertEqual(receipt["optimizer"]["accepted_steps"], 0)
                    self.assertEqual(receipt["optimizer"]["uncertain_steps"], 0)
                    self.assertEqual(receipt["optimizer"]["updates"], 0)
                    self.assertEqual(receipt["checkpoint"]["factory_checkpoint_path"],
                                     receipt["checkpoint"]["runtime_parent_checkpoint"])

    def test_faulted_training_binding_blocks_direct_and_scheduled_selection(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation, run_selection_schedule,
            run_training_operation,
        )
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            training = study["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                fault_receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a",
                    root / training["output_root"] / "operation-receipt.json",
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError("injected training factory fault")
                        ),
                    ),
                )
                self.assertEqual(fault_receipt["status"], "operational_failure")
                bindings, _, bindings_path = _freeze_training_policies(root, ("standalone-a",))
                self.assertEqual(bindings["stop_required_policy_ids"], ["standalone-a"])

                selection = study["selection"]["schedule"]
                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                child_calls = []
                result = run_selection_schedule(
                    root, STUDY_RELATIVE, bindings_path, output_dir,
                    deadline=600.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        phase_context=root / "phase-context.json",
                        child_process=lambda *args, **kwargs: child_calls.append(args),
                    ),
                )
                reference_row = next(row for row in selection if row["policy_id"] == "legal-random")
                factory_calls = []
                direct = run_selection_operation(
                    root, STUDY_RELATIVE, reference_row["operation_id"], bindings_path,
                    _selection_receipt_path(root, reference_row["operation_id"]),
                    deadline=600.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
                    ),
                )

            self.assertEqual(child_calls, [])
            self.assertEqual(factory_calls, [])
            self.assertEqual(result["not_started_operation_count"], 192)
            self.assertEqual(result["reserved_step_slots"], 0)
            self.assertTrue(all(row["status"] == "not_started"
                                and row["reason"].startswith("prior_training_fault")
                                for row in result["rows"]))
            self.assertEqual(direct["status"], "not_started")
            self.assertTrue(direct["reason"].startswith("prior_training_fault"))
            self.assertFalse(_selection_reservation_path(root, reference_row["operation_id"]).exists())

    def test_binding_stop_required_ids_must_be_known_unique_and_in_training_order(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, RuntimeProtocolError, run_selection_operation,
        )
        from spire_learn.neural import config as config_module

        for malformed in (["unknown-policy"], ["run-a", "standalone-a"],
                          ["standalone-a", "standalone-a"]):
            with self.subTest(stop_required=malformed), tempfile.TemporaryDirectory() as directory:
                root = _copy_study_checkout(Path(directory) / "checkout")
                with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                    _, _, bindings_path = _freeze_training_policies(root, ())
                    study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
                    row = next(item for item in study["selection"]["schedule"]
                               if item["policy_id"] == "legal-random")
                    binding = json.loads(bindings_path.read_text(encoding="utf-8"))
                    binding["stop_required_policy_ids"] = malformed
                    bindings_path.write_text(json.dumps(binding), encoding="utf-8")
                    factory_calls = []

                    def factory(*args, **kwargs):
                        factory_calls.append(True)
                        raise AssertionError("malformed frozen fault IDs must stop before factory")

                    with self.assertRaises(RuntimeProtocolError):
                        run_selection_operation(
                            root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                            _selection_receipt_path(root, row["operation_id"]),
                            deadline=10_000.0, clock=lambda: 0.0,
                            dependencies=RuntimeDependencies(factory=factory),
                        )
                self.assertEqual(factory_calls, [])
                self.assertFalse(_selection_reservation_path(root, row["operation_id"]).exists())

    def test_selection_reset_only_rollout_keeps_known_reset_effects(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config
        from tests.unit.test_neural_runner import _allocation, _rollout

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                row = next(
                    item for item in json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"]
                    if item["policy_id"] == "legal-random"
                )
                config = load_neural_config(root / row["config_path"])
                rollout = _rollout(
                    config, (), allocations=(_allocation(config, 0, 0, case_index=0),),
                    stop_reason="inactive_reset_boundary",
                )
                runtime = _StudyRuntime(config, _StudyCollector(config, [rollout]))
                runtime.policy_kind = "legal-random"
                runtime.action_selection = "seeded-sampling"
                runtime.collector.learned_combat_policy = False
                receipt = run_selection_operation(
                    root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                    _selection_receipt_path(root, row["operation_id"]),
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime,
                        evaluate=runner.evaluate_policy,
                    ),
                )

            self.assertEqual(receipt["status"], "incomplete")
            self.assertTrue(receipt["attempted_effects"])
            self.assertEqual(
                {name: receipt["counts"][name] for name in (
                    "attempted_reset_slots", "forwarded_reset_slots",
                    "known_accepted_reset_slots", "delivered_reset_slots",
                    "uncertain_reset_slots",
                )},
                {
                    "attempted_reset_slots": 1, "forwarded_reset_slots": 1,
                    "known_accepted_reset_slots": 1, "delivered_reset_slots": 1,
                    "uncertain_reset_slots": 0,
                },
            )

    def test_selection_reset_fault_keeps_known_partial_reset_effects(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_operation
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config
        from tests.unit.test_neural_runner import _allocation, _rollout

        class ResetFaultCollector(_StudyCollector):
            def __init__(self, config, partial):
                super().__init__(config, ())
                self.partial_rollout = partial

            def collect(self, *, max_learned_decisions, max_env_actions, policy_version):
                self.collect_calls.append((max_learned_decisions, max_env_actions, policy_version))
                raise RuntimeError("injected failure after accepted reset")

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                row = next(
                    item for item in json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"]
                    if item["policy_id"] == "legal-random"
                )
                config = load_neural_config(root / row["config_path"])
                partial = _rollout(
                    config, (), allocations=(_allocation(config, 0, 0, case_index=0),),
                    stop_reason="inactive_reset_boundary",
                )
                runtime = _StudyRuntime(config, ResetFaultCollector(config, partial))
                runtime.policy_kind = "legal-random"
                runtime.action_selection = "seeded-sampling"
                runtime.collector.learned_combat_policy = False
                receipt = run_selection_operation(
                    root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                    _selection_receipt_path(root, row["operation_id"]),
                    deadline=10_000.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime,
                        evaluate=runner.evaluate_policy,
                    ),
                )

            self.assertEqual(receipt["status"], "operational_failure")
            self.assertTrue(receipt["attempted_effects"])
            self.assertEqual(receipt["counts"]["attempted_environment_slots"], 0)
            self.assertEqual(receipt["counts"]["known_accepted_reset_slots"], 1)
            self.assertEqual(receipt["counts"]["delivered_reset_slots"], 1)

    def test_runner_control_without_report_publishes_unknown_effects(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_operation, run_training_operation
        from spire_learn.neural import config as config_module
        from spire_learn.neural.config import load_neural_config

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt("injected runner control exception")

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            training = study["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / training["config"]["path"])
                train_runtime = _StudyRuntime(config, _StudyCollector(config, ()))
                train_receipt_path = root / training["output_root"] / "operation-receipt.json"
                with self.assertRaises(KeyboardInterrupt):
                    run_training_operation(
                        root, STUDY_RELATIVE, "standalone-a", train_receipt_path,
                        deadline=10_000.0, clock=lambda: 0.0,
                        dependencies=RuntimeDependencies(
                            factory=lambda *args, **kwargs: train_runtime, train=interrupt,
                        ),
                    )

                _, _, bindings_path = _freeze_training_policies(root, ())
                row = next(
                    item for item in study["selection"]["schedule"]
                    if item["policy_id"] == "legal-random"
                )
                selection_config = load_neural_config(root / row["config_path"])
                selection_runtime = _StudyRuntime(selection_config, _StudyCollector(selection_config, ()))
                selection_runtime.policy_kind = "legal-random"
                selection_runtime.action_selection = "seeded-sampling"
                selection_runtime.collector.learned_combat_policy = False
                selection_receipt_path = _selection_receipt_path(root, row["operation_id"])
                with self.assertRaises(KeyboardInterrupt):
                    run_selection_operation(
                        root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                        selection_receipt_path, deadline=10_000.0, clock=lambda: 0.0,
                        dependencies=RuntimeDependencies(
                            factory=lambda *args, **kwargs: selection_runtime, evaluate=interrupt,
                        ),
                    )

            train_receipt = json.loads(train_receipt_path.read_text(encoding="utf-8"))
            selection_receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
            for receipt in (train_receipt, selection_receipt):
                self.assertEqual(receipt["status"], "interrupted")
                self.assertIsNone(receipt["attempted_effects"])
                self.assertIsNone(receipt["counts"]["attempted_environment_slots"])
                self.assertIsNone(receipt["counts"]["attempted_reset_slots"])

    def test_final_clock_fault_overrides_unavailable_and_not_started_statuses(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation, run_selection_schedule,
            run_training_operation,
        )
        from spire_learn.neural import config as config_module, runner
        from spire_learn.neural.config import load_neural_config

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            study = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))
            training = study["training"]["runs"][0]
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                config = load_neural_config(root / training["config"]["path"])
                runtime = _StudyRuntime(config, _StudyCollector(config, ()))
                clock_calls = []

                def training_clock():
                    clock_calls.append("call")
                    if len(clock_calls) == 3:
                        raise RuntimeError("injected final training clock failure")
                    return 0.0

                def train(actual_config, *, runtime, clock):
                    return runner.train(actual_config, runtime=runtime, clock=lambda: 10_000.0)

                receipt_path = root / training["output_root"] / "operation-receipt.json"
                receipt = run_training_operation(
                    root, STUDY_RELATIVE, "standalone-a", receipt_path,
                    deadline=10_000.0, clock=training_clock,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: runtime, train=train,
                    ),
                )
                _, _, bindings_path = _freeze_training_policies(root, ("standalone-a",))
                bindings = json.loads(bindings_path.read_text(encoding="utf-8"))

            self.assertEqual(receipt["status"], "operational_failure")
            self.assertEqual(receipt["reason"], "final_clock_error")
            self.assertEqual(receipt["transport_exit_code"], 1)
            self.assertEqual(bindings["stop_required_policy_ids"], ["standalone-a"])

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                row = next(
                    item for item in json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"]
                    if item["policy_id"] == "legal-random"
                )
                calls = []
                factory_calls = []

                def selection_clock():
                    calls.append("call")
                    if len(calls) == 5:
                        raise RuntimeError("injected final selection clock failure")
                    return 1.0 if len(calls) == 4 else 0.0

                selection_receipt = run_selection_operation(
                    root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                    _selection_receipt_path(root, row["operation_id"]),
                    deadline=1.0, clock=selection_clock,
                    dependencies=RuntimeDependencies(
                        factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
                    ),
                )

            self.assertEqual(factory_calls, [])
            self.assertEqual(selection_receipt["status"], "operational_failure")
            self.assertEqual(selection_receipt["reason"], "final_clock_error")
            self.assertEqual(selection_receipt["transport_exit_code"], 1)

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                calls = []
                child_calls = []

                def schedule_clock():
                    calls.append("call")
                    if len(calls) == 3:
                        raise RuntimeError("injected final schedule clock failure")
                    return 0.0

                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                schedule_result = run_selection_schedule(
                    root, STUDY_RELATIVE, bindings_path, output_dir,
                    deadline=0.0, clock=schedule_clock,
                    dependencies=RuntimeDependencies(
                        phase_context=root / "phase-context.json",
                        child_process=lambda *args, **kwargs: child_calls.append(args),
                    ),
                )

            self.assertEqual(child_calls, [])
            self.assertEqual(schedule_result["status"], "operational_failure")
            self.assertEqual(schedule_result["transport_exit_code"], 1)
            self.assertEqual(schedule_result["not_started_operation_count"], 192)
            self.assertEqual(schedule_result["reserved_step_slots"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                calls = []

                def missing_context_clock():
                    calls.append("call")
                    if len(calls) == 2:
                        raise RuntimeError("injected final clock failure after context refusal")
                    return 0.0

                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                missing_context_result = run_selection_schedule(
                    root, STUDY_RELATIVE, bindings_path, output_dir,
                    deadline=600.0, clock=missing_context_clock,
                    dependencies=RuntimeDependencies(),
                )

            self.assertEqual(missing_context_result["status"], "operational_failure")
            self.assertEqual(missing_context_result["transport_exit_code"], 1)
            self.assertEqual(missing_context_result["not_started_operation_count"], 192)

    def test_train_and_selection_receipt_writers_preserve_process_control_precedence(self):
        from scripts.neural.task9_study_runtime import (
            RuntimeDependencies, run_selection_operation, run_training_operation,
        )
        from spire_learn.neural import config as config_module

        cases = (
            ("training", RuntimeError("training operation fault"),
             KeyboardInterrupt("training receipt writer control"), KeyboardInterrupt),
            ("training", KeyboardInterrupt("training operation control"),
             OSError("training receipt writer fault"), KeyboardInterrupt),
            ("selection", RuntimeError("selection operation fault"),
             KeyboardInterrupt("selection receipt writer control"), KeyboardInterrupt),
            ("selection", KeyboardInterrupt("selection operation control"),
             OSError("selection receipt writer fault"), KeyboardInterrupt),
        )
        module = importlib.import_module("scripts.neural.task9_study_runtime")
        for operation, operation_error, writer_error, expected_type in cases:
            with self.subTest(operation=operation, operation_error=type(operation_error).__name__), \
                    tempfile.TemporaryDirectory() as directory:
                root = _copy_study_checkout(Path(directory) / "checkout")
                with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                    if operation == "training":
                        record = json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["training"]["runs"][0]
                        receipt_path = root / record["output_root"] / "operation-receipt.json"

                        def invoke():
                            def factory(*args, **kwargs):
                                raise operation_error

                            return run_training_operation(
                                root, STUDY_RELATIVE, "standalone-a", receipt_path,
                                deadline=10_000.0, clock=lambda: 0.0,
                                dependencies=RuntimeDependencies(factory=factory),
                            )
                    else:
                        _, _, bindings_path = _freeze_training_policies(root, ())
                        row = next(
                            item for item in json.loads((root / STUDY_RELATIVE).read_text(encoding="utf-8"))["selection"]["schedule"]
                            if item["policy_id"] == "legal-random"
                        )
                        receipt_path = _selection_receipt_path(root, row["operation_id"])

                        def invoke():
                            def factory(*args, **kwargs):
                                raise operation_error

                            return run_selection_operation(
                                root, STUDY_RELATIVE, row["operation_id"], bindings_path,
                                receipt_path, deadline=10_000.0, clock=lambda: 0.0,
                                dependencies=RuntimeDependencies(factory=factory),
                            )

                    original_exclusive_json = module._exclusive_json

                    def fail_receipt_write(path, value):
                        if Path(path) == receipt_path:
                            raise writer_error
                        return original_exclusive_json(path, value)

                    with patch.object(module, "_exclusive_json", side_effect=fail_receipt_write):
                        with self.assertRaises(expected_type) as caught:
                            invoke()

                notes = "\n".join(getattr(caught.exception, "__notes__", ()))
                if isinstance(operation_error, RuntimeError):
                    self.assertIs(caught.exception, writer_error)
                    self.assertIn("RuntimeError", notes)
                else:
                    self.assertIs(caught.exception, operation_error)
                    self.assertIn("OSError", notes)

    def test_expired_selection_deadline_writes_all_not_started_without_child_or_reservation(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_schedule
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                child_calls = []
                result = run_selection_schedule(
                    root, STUDY_RELATIVE, bindings_path, output_dir,
                    deadline=0.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        phase_context=root / "phase-context.json",
                        child_process=lambda *args, **kwargs: child_calls.append(args),
                    ),
                )

            self.assertEqual(child_calls, [])
            self.assertEqual(result["not_started_operation_count"], 192)
            self.assertEqual(result["reserved_step_slots"], 0)
            self.assertTrue(all(row["status"] == "not_started" for row in result["rows"]))
            self.assertFalse((output_dir / "control/operations/selection-000.reservation.json").exists())

    def test_child_transport_timeout_keeps_reserved_row_unresolved_and_finalizes_remainder(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_schedule
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _training_receipt(root, "standalone-a")
                _, _, bindings_path = _freeze_training_policies(root, ("standalone-a",))
                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"

                def child_timeout(*args, **kwargs):
                    self.assertEqual(kwargs["output_limit_bytes"], 64 * 1024)
                    self.assertGreater(kwargs["timeout"], 0)
                    raise TimeoutError("child result was lost after reservation")

                result = run_selection_schedule(
                    root, STUDY_RELATIVE, bindings_path, output_dir,
                    deadline=600.0, clock=lambda: 0.0,
                    dependencies=RuntimeDependencies(
                        phase_context=root / "phase-context.json", child_process=child_timeout,
                    ),
                )

            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["unresolved_operation_count"], 1)
            self.assertEqual(result["not_started_operation_count"], 191)
            self.assertEqual(result["reserved_step_slots"], 256)
            first = result["rows"][0]
            self.assertEqual(first["status"], "unresolved")
            self.assertIsNone(first["attempted_effects"])
            self.assertTrue(_selection_reservation_path(root, "selection-000").is_file())
            self.assertFalse(_selection_receipt_path(root, "selection-000").exists())
            self.assertTrue((output_dir / "control/schedule-result.json").is_file())

    def test_schedule_finalization_write_error_does_not_swallow_process_control(self):
        from scripts.neural.task9_study_runtime import RuntimeDependencies, run_selection_schedule
        from spire_learn.neural import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            with patch.object(config_module, "__file__", str(root / "src/spire_learn/neural/config.py")):
                _, _, bindings_path = _freeze_training_policies(root, ())
                output_dir = root / "artifacts/development/neural-combat/experiments/task9-v1-attempt-b/selection"
                original_exclusive_json = importlib.import_module(
                    "scripts.neural.task9_study_runtime"
                )._exclusive_json

                def failing_final_write(path, value):
                    if Path(path).name == "schedule-result.json":
                        raise OSError("injected final schedule publication failure")
                    return original_exclusive_json(path, value)

                with patch("scripts.neural.task9_study_runtime._exclusive_json", side_effect=failing_final_write):
                    with self.assertRaises(KeyboardInterrupt):
                        run_selection_schedule(
                            root, STUDY_RELATIVE, bindings_path, output_dir,
                            deadline=600.0, clock=lambda: 0.0,
                            dependencies=RuntimeDependencies(
                                phase_context=root / "phase-context.json",
                                child_process=lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                            ),
                        )
            self.assertTrue((output_dir / "control/schedule-plan.json").is_file())
            self.assertFalse((output_dir / "control/schedule-result.json").exists())

    def test_real_reservation_directory_fsync_fault_retains_256_without_dispatch(self):
        from scripts.neural import task9_study_runtime as b
        from scripts.neural import task9_study_controls as c
        from spire_learn.neural import config as config_module
        from unittest.mock import Mock
        for kind in ('directory','control','partial','unreadable'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=_copy_study_checkout(Path(d)/'checkout')
                with patch.object(config_module,'__file__',str(root/'src/spire_learn/neural/config.py')):
                    _training_receipt(root,'standalone-a')
                    _,_,binding=_freeze_training_policies(root,('standalone-a',))
                    output=root/b.SELECTION_OUTPUT_RELATIVE
                    reservation=_selection_reservation_path(root,'selection-000')
                    child=Mock(side_effect=AssertionError('must not dispatch'))
                    sync=b._fsync_directory;fsync=os.fsync;read=b._read_json;synced=[]
                    def file_sync(fd):
                        synced.append(Path(os.readlink(f'/proc/self/fd/{fd}')));return fsync(fd)
                    def directory_sync(path):
                        if Path(path)==reservation.parent:
                            self.assertIn(reservation,synced)
                            self.assertTrue(reservation.is_file())
                            if kind=='partial':reservation.write_text('{partial')
                            if kind=='control':raise KeyboardInterrupt('reservation directory control')
                            raise OSError('reservation directory durability unknown')
                        return sync(path)
                    def read_footprint(path,**kwargs):
                        if kind=='unreadable' and Path(path)==reservation:raise PermissionError('unreadable footprint')
                        return read(path,**kwargs)
                    with patch.object(b.os,'fsync',side_effect=file_sync), \
                         patch.object(b,'_fsync_directory',side_effect=directory_sync), \
                         patch.object(b,'_read_json',side_effect=read_footprint):
                        if kind=='control':
                            with self.assertRaises(KeyboardInterrupt):
                                b.run_selection_schedule(root,STUDY_RELATIVE,binding,output,deadline=600,clock=lambda:0,
                                    dependencies=b.RuntimeDependencies(phase_context=root/'context.json',child_process=child))
                            result=json.loads((output/'control/schedule-result.json').read_text())
                        else:
                            result=b.run_selection_schedule(root,STUDY_RELATIVE,binding,output,deadline=600,clock=lambda:0,
                                dependencies=b.RuntimeDependencies(phase_context=root/'context.json',child_process=child))
                    child.assert_not_called()
                    self.assertEqual(result['status'],'stopped')
                    self.assertEqual(result['reserved_step_slots'],256)
                    self.assertEqual(result['reserved_operation_count'],1)
                    self.assertEqual(result['attempted_operation_count'],0)
                    row=result['rows'][0]
                    self.assertEqual(row['status'],'unresolved')
                    self.assertEqual(row['dispatch_state'],'not_dispatched')
                    self.assertIs(row['attempted_effects'],False)
                    self.assertEqual(row['reservation_publication']['state'],'uncertain')
                    self.assertEqual(result['error']['exception_type'],'KeyboardInterrupt' if kind=='control' else 'OSError')
                    self.assertTrue(all(r['status']=='not_started' for r in result['rows'][1:]))
                    study=json.loads((root/STUDY_RELATIVE).read_text())
                    audit=c._reservation_audit(root,study)
                    self.assertTrue(audit[0]['unresolved']);self.assertEqual(audit[0]['reserved_step_slots_upper_bound'],256)
                    with self.assertRaises(FileExistsError):
                        b.run_selection_schedule(root,STUDY_RELATIVE,binding,output,deadline=600,clock=lambda:0,
                            dependencies=b.RuntimeDependencies(phase_context=root/'context.json',child_process=child))

    def test_dispatch_timeout_charges_bookkeeping_and_stops_at_absolute_grace(self):
        from scripts.neural import task9_study_runtime as b
        from spire_learn.neural import config as config_module
        for exhausted in (False,True):
            with self.subTest(exhausted=exhausted),tempfile.TemporaryDirectory() as d:
                root=_copy_study_checkout(Path(d)/'checkout')
                with patch.object(config_module,'__file__',str(root/'src/spire_learn/neural/config.py')):
                    _training_receipt(root,'standalone-a')
                    _,_,binding=_freeze_training_policies(root,('standalone-a',))
                    output=root/b.SELECTION_OUTPUT_RELATIVE;now=[0.0];calls=[]
                    account=b._directory_bytes;reserve=b._reserve_scheduled_operation
                    path=b._operation_receipt_file;after_reserve=[False]
                    def accounting(p):
                        value=account(p);now[0]+=10;return value
                    def reservation(*args,**kwargs):
                        value=reserve(*args,**kwargs)
                        now[0]=661 if exhausted else now[0]+20
                        after_reserve[0]=True;return value
                    def receipt_path(*args):
                        value=path(*args)
                        if after_reserve[0]:
                            now[0]+=10;after_reserve[0]=False
                        return value
                    def child(argv,**kwargs):
                        calls.append((argv,kwargs['timeout']))
                        raise TimeoutError('intentional child stop')
                    with patch.object(b,'_directory_bytes',side_effect=accounting), \
                         patch.object(b,'_reserve_scheduled_operation',side_effect=reservation), \
                         patch.object(b,'_operation_receipt_file',side_effect=receipt_path):
                        result=b.run_selection_schedule(root,STUDY_RELATIVE,binding,output,deadline=600,clock=lambda:now[0],
                            dependencies=b.RuntimeDependencies(phase_context=root/'context.json',child_process=child))
                self.assertEqual(result['reserved_step_slots'],256)
                if exhausted:
                    self.assertFalse(calls)
                    self.assertEqual(result['rows'][0]['dispatch_state'],'not_dispatched')
                    self.assertIs(result['rows'][0]['attempted_effects'],False)
                    self.assertEqual(result['attempted_operation_count'],0)
                else:
                    self.assertEqual(len(calls),1)
                    self.assertEqual(calls[0][1],620.0)
                    self.assertEqual(float(calls[0][0][calls[0][0].index('--deadline')+1]),600.0)
                self.assertEqual(result['rows'][0]['status'],'unresolved')
                self.assertTrue(all(r['status']=='not_started' for r in result['rows'][1:]))

    def test_reference_import_guard_fails_a_caught_torch_attempt_in_fresh_process(self):
        source = textwrap.dedent("""\
            import importlib
            from scripts.neural.task9_study_runtime import _reference_import_guard, ReferenceImportViolation
            try:
                with _reference_import_guard() as guard:
                    try:
                        importlib.import_module("torch")
                    except ImportError:
                        pass
                    guard.check()
            except ReferenceImportViolation:
                print("reference-import-blocked")
            else:
                raise AssertionError("caught import attempt escaped the guard")
        """)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", source], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("reference-import-blocked", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_cli_help_is_lazy_under_six_prefix_import_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            script = root / "scripts/neural/task9_study_runtime.py"
            guard = textwrap.dedent("""\
                import importlib.abc, runpy, sys
                prefixes = ("torch", "numpy", "tensordict", "torchrl", "gymnasium", "spire_lab")
                attempted = []
                class Guard(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if any(fullname == p or fullname.startswith(p + ".") for p in prefixes):
                            attempted.append(fullname)
                            raise AssertionError("guarded optional import: " + fullname)
                        return None
                sys.meta_path.insert(0, Guard())
                before = {name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in prefixes)}
                sys.argv = [sys.argv[1], "--help"]
                try:
                    runpy.run_path(sys.argv[0], run_name="__main__")
                except SystemExit as error:
                    assert error.code == 0, error.code
                after = {name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in prefixes)}
                assert not attempted, attempted
                assert after == before == set(), (before, after)
            """)
            completed = subprocess.run(
                [sys.executable, "-B", "-c", guard, str(script)],
                cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("selection-child", completed.stdout)
            self.assertEqual(completed.stderr, "")

    def test_cli_projects_context_arguments_before_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _copy_study_checkout(Path(directory) / "checkout")
            controls = root / "scripts/neural/task9_study_controls.py"
            controls.write_text(textwrap.dedent("""\
                import json, os, sys
                from pathlib import Path
                def verify_guest_context(**kwargs):
                    payload = {key: (str(value) if isinstance(value, Path) else value)
                               for key, value in kwargs.items()}
                    payload["torch_loaded"] = any(name == "torch" or name.startswith("torch.")
                                                  for name in sys.modules)
                    Path(os.environ["TASK9_VERIFY_LOG"]).write_text(json.dumps(payload), encoding="utf-8")
                    raise RuntimeError("test verifier stop")
            """), encoding="utf-8")
            log_path = root / "verify-call.json"
            phase_context = root / "phase-context.json"
            bindings = root / "bindings.json"
            receipt = root / "receipt.json"
            command = [
                sys.executable, "-B", str(root / "scripts/neural/task9_study_runtime.py"),
                "selection-child", "--root", str(root), "--study", str(STUDY_RELATIVE),
                "--phase-context", str(phase_context), "--deadline", "600",
                "--operation-id", "selection-000", "--bindings", str(bindings),
                "--receipt", str(receipt),
            ]
            completed = subprocess.run(
                command, cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "src"), "TASK9_VERIFY_LOG": str(log_path)},
                capture_output=True, text=True, check=False,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("test verifier stop", completed.stderr)
            projected = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(set(projected), {
                "root", "study_path", "phase_context", "phase", "deadline", "policy_id",
                "operation_id", "bindings_path", "receipt_path", "output_dir", "torch_loaded",
            })
            self.assertEqual(projected["phase"], "selection")
            self.assertEqual(projected["operation_id"], "selection-000")
            self.assertEqual(projected["deadline"], 600.0)
            self.assertFalse(projected["torch_loaded"])


if __name__ == "__main__":
    unittest.main()
