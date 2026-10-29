"""Two gated, cumulative CUDA runtime proofs; importing this file stays CPU-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
_GATE = os.environ.get("SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME") == "1"
_HELPER_PATH = Path(os.environ.get(
    "SPIRE_LEARN_GPU_RUNTIME_PROOF_HELPER",
    str(ROOT / "scripts/neural/gpu_runtime_proof_helpers.py"),
))
if not _HELPER_PATH.is_file():
    raise ImportError(f"Task 9 GPU proof helper is missing: {_HELPER_PATH}")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_spire_learn_gpu_runtime_proof_helpers", _HELPER_PATH,
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"Task 9 GPU proof helper cannot be loaded: {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPER
_HELPER_SPEC.loader.exec_module(_HELPER)


def _inner_policy(policy):
    return policy.inner if isinstance(policy, _HELPER.GuardedPolicy) else policy


def _unwrap_optimizer(optimizer):
    while hasattr(optimizer, "inner"):
        optimizer = optimizer.inner
    return optimizer


def _assert_fresh_warm_optimizer(test, optimizer):
    """Check runner counters on the wrapper and Adam state on the inner optimizer."""
    test.assertEqual(optimizer.attempted, 0)
    test.assertEqual(optimizer.accepted, 0)
    test.assertEqual(len(_unwrap_optimizer(optimizer).state), 0)


def _assert_fresh_resume_optimizer(test, optimizer):
    """Check per-invocation counters on a resumed wrapper with restored Adam state."""
    test.assertEqual(optimizer.attempted, 0)
    test.assertEqual(optimizer.accepted, 0)


def _state_hash(state, torch) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_groups(state, torch) -> dict[str, str]:
    actor_head = {name: value for name, value in state.items() if name.startswith("candidate_")}
    critic_head = {name: value for name, value in state.items() if name.startswith("value_head.")}
    shared_body = {
        name: value for name, value in state.items()
        if name not in actor_head and name not in critic_head
    }
    if not actor_head or not critic_head or not shared_body:
        raise AssertionError("actor head, critic head, and shared model body must all be present")
    return {
        "model_sha256": _state_hash(state, torch),
        "shared_body_sha256": _state_hash(shared_body, torch),
        "actor_head_sha256": _state_hash(actor_head, torch),
        "critic_head_sha256": _state_hash(critic_head, torch),
    }


def _tree_equal(left, right, torch) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.device.type == right.device.type and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tree_equal(left[key], right[key], torch) for key in left
        )
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _tree_equal(a, b, torch) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _choice_tuple(choice):
    return (
        choice.original_action_index,
        choice.log_probability,
        choice.value,
        choice.behavior_policy_version,
    )


def _assert_exact_choices(test, left, right):
    test.assertEqual(len(left), len(right))
    for first, second in zip(left, right, strict=True):
        test.assertEqual(_choice_tuple(first), _choice_tuple(second))


def _assert_portable_choices(test, cpu, cuda):
    test.assertEqual(len(cpu), len(cuda))
    for first, second in zip(cpu, cuda, strict=True):
        test.assertEqual(first.original_action_index, second.original_action_index)
        test.assertTrue(math.isclose(first.log_probability, second.log_probability, rel_tol=0.0, abs_tol=1e-4))
        test.assertTrue(math.isclose(first.value, second.value, rel_tol=0.0, abs_tol=1e-4))


def _assert_cpu_model_outputs(test, warm, frozen):
    test.assertEqual(len(warm), len(frozen))
    for fresh, checkpointed in zip(warm, frozen, strict=True):
        test.assertEqual(fresh.original_action_index, checkpointed.original_action_index)
        test.assertEqual(fresh.log_probability, checkpointed.log_probability)
        test.assertEqual(fresh.value, checkpointed.value)


def _install_runner_api_aliases(
    target, *, load_neural_config, create_runtime, evaluate_policy, train,
):
    """Keep imported module functions unbound when accessed through test instances."""
    target.load_neural_config = staticmethod(load_neural_config)
    target.create_runtime = staticmethod(create_runtime)
    target.evaluate_policy = staticmethod(evaluate_policy)
    target.train = staticmethod(train)


def _close_owned_runtime(runtime, primary):
    """Close one acquired runtime while preserving an earlier failure."""
    if getattr(runtime, "_closed", False):
        return
    try:
        runtime.close()
    except BaseException as close_error:
        promoted = _preserve_owned_close_failure(primary, close_error)
        if promoted is not primary:
            raise promoted


def _preserve_owned_close_failure(primary, close_error):
    """Match runner cleanup: process-control failures supersede ordinary errors."""
    if primary is None:
        return close_error
    if _owned_is_process_control(close_error) and not _owned_is_process_control(primary):
        add_note = getattr(close_error, "add_note", None)
        if callable(add_note):
            add_note(
                f"owned runtime close process control superseded prior "
                f"{type(primary).__name__}: {_owned_error_message(primary)}"
            )
        return close_error
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(
            f"owned runtime close also failed: {type(close_error).__name__}: "
            f"{_owned_error_message(close_error)}"
        )
    return primary


def _owned_is_process_control(error):
    return not isinstance(error, Exception)


def _owned_error_message(error):
    try:
        return str(error).replace("\x00", " ").replace("\r", " ").replace("\n", " ")[:384]
    except BaseException:
        return "exception message unavailable"


def _close_owned_runtimes(runtimes, primary):
    """Close all acquired runtimes and retain the first failure."""
    first = primary
    for runtime in runtimes:
        if runtime is None:
            continue
        try:
            if not getattr(runtime, "_closed", False):
                runtime.close()
        except BaseException as close_error:
            first = _preserve_owned_close_failure(first, close_error)
    if first is not primary:
        raise first


class _InertDriver:
    """create_runtime-only replacement; a checkpoint comparison cannot touch Gym."""

    def __init__(self, mode, batch_size, schema):
        self.mode = mode
        self.batch_size = batch_size
        self.rows = (None,) * batch_size
        self.accepted_steps = 0
        self.schema = schema
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0

    def reset(self, **_kwargs):
        self.reset_calls += 1
        raise AssertionError("checkpoint comparison attempted an environment reset")

    def step(self, *_args, **_kwargs):
        self.step_calls += 1
        raise AssertionError("checkpoint comparison attempted an environment step")

    def close(self):
        self.close_calls += 1


@unittest.skipUnless(
    _GATE,
    "Task 9 CUDA workload is held for reviewed source, runbook, and explicit parent release",
)
class NeuralGpuRuntimeProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _GATE:
            raise RuntimeError("protected proof setup was reached while the opt-in gate is off")
        # This timestamp precedes optional imports, config loading, and device qualification.
        cls.started = time.monotonic()
        cls._first_mode_failure = None
        raw_deadline = os.environ.get("SPIRE_LEARN_NEURAL_GPU_SOFT_DEADLINE")
        if raw_deadline is None:
            raise RuntimeError("protected runner did not supply its shared absolute deadline")
        cls.shared_deadline = min(float(raw_deadline), cls.started + 120.0)
        if not math.isfinite(cls.shared_deadline) or cls.shared_deadline <= cls.started:
            raise _HELPER.ProofDeadlineExceeded("the shared proof deadline elapsed before test setup")
        if os.environ.get("SPIRE_LEARN_NEURAL_GPU_UUID") != _HELPER.GPU_UUID:
            raise RuntimeError("integration test is not bound to the released GPU UUID")

        sys.path.insert(0, str(ROOT / "src"))
        import torch

        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.runner import (
            RuntimeOperationLimits,
            create_runtime,
            evaluate_policy,
            train,
        )

        cls.torch = torch
        cls.RuntimeOperationLimits = RuntimeOperationLimits
        _install_runner_api_aliases(
            cls,
            load_neural_config=load_neural_config,
            create_runtime=create_runtime,
            evaluate_policy=evaluate_policy,
            train=train,
        )
        cls.exports = _HELPER.verify_test_exports(ROOT)
        cls.ledger = _HELPER.EffectLedger(cls.shared_deadline)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("proof worker thread limits differ from the released controls")
        properties = torch.cuda.get_device_properties(0)
        if properties.name != "NVIDIA GeForce RTX 5090":
            raise RuntimeError("CUDA device name differs from the Task 8 qualified GPU")
        allocator_fraction = min(1.0, (8 * 1024**3) / float(properties.total_memory))
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)
        cls.device_identity = {
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
            "allocator_fraction": allocator_fraction,
            "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
        cls.configs = {
            mode: load_neural_config(ROOT / "configs/neural" / f"{mode}-gpu-smoke.toml")
            for mode in ("standalone", "run")
        }
        for mode, config in cls.configs.items():
            expected = (4, 8) if mode == "standalone" else (1, 32)
            if (config.mode, config.operation, config.split, config.partition) != (mode, "train", "train", "train"):
                raise RuntimeError(f"{mode} proof config is not train-only")
            if (config.rollout_rows, config.rollout_steps) != expected:
                raise RuntimeError(f"{mode} proof rollout geometry differs")
            if config.runtime_gpu_uuid != _HELPER.GPU_UUID:
                raise RuntimeError(f"{mode} proof config GPU UUID differs")
        cls.ledger.check_deadline()

    def _write_mode_evidence(self, config, result):
        output_root = Path(config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        _HELPER.write_json_exclusive(output_root / "gpu-proof-checks.json", result)

    def _compare_optimizer_device(self, optimizer, expected_device):
        adam = _unwrap_optimizer(optimizer)
        self.assertTrue(adam.state, "one accepted Adam update must publish moments")
        for parameter_state in adam.state.values():
            self.assertEqual(parameter_state["step"].device.type, "cpu")
            self.assertEqual(parameter_state["exp_avg"].device.type, expected_device)
            self.assertEqual(parameter_state["exp_avg_sq"].device.type, expected_device)

    def _create_inert_runtime(self, config, *, purpose, checkpoint, policy_kind="checkpoint"):
        from spire_learn.neural import envs

        schema = self._trained_driver.schema
        driver = _InertDriver(config.mode, config.rollout_rows, schema)
        runtime = None
        try:
            with patch.object(envs, "make_driver", return_value=driver) as factory:
                runtime = self.create_runtime(
                    config, purpose=purpose, checkpoint=checkpoint,
                    policy_kind=policy_kind,
                    action_selection="deterministic" if purpose == "evaluate" else "seeded-sampling",
                )
                self.assertEqual(factory.call_count, 1)
        except BaseException as error:
            if runtime is not None:
                _close_owned_runtime(runtime, error)
            raise
        assert runtime is not None
        return runtime, driver

    def _compare_checkpoint_portability(self, mode, config, checkpoint, loaded, public_frame, gpu_choices):
        self.ledger.check_deadline()
        cpu_train = self.load_neural_config(ROOT / "configs/neural" / f"{mode}.toml")
        warm = frozen = None
        warm_driver = eval_driver = None
        try:
            warm, warm_driver = self._create_inert_runtime(
                cpu_train, purpose="warm_start", checkpoint=checkpoint,
            )
            selection = self.load_neural_config(ROOT / "configs/neural" / f"{mode}-selection.toml")
            frozen, eval_driver = self._create_inert_runtime(
                selection, purpose="evaluate", checkpoint=checkpoint,
            )
            self.assertEqual(cpu_train.operation, "train")
            self.assertEqual(selection.operation, "evaluate")
            self.assertEqual(warm.device_plan.device, "cpu")
            self.assertEqual(frozen.device_plan.device, "cpu")
            saved_weights = dict(loaded.model_tensors)
            warm_weights = warm.policy.state_dict()
            frozen_weights = frozen.policy.state_dict()
            self.assertTrue(_tree_equal(saved_weights, warm_weights, self.torch))
            self.assertTrue(_tree_equal(saved_weights, frozen_weights, self.torch))
            self.assertIsNotNone(warm.optimizer)
            _assert_fresh_warm_optimizer(self, warm.optimizer)
            self.assertEqual(warm.recovered_progress, {})
            self.assertEqual(warm.update_count, 0)
            self.assertEqual(warm.policy._behavior_policy_version, 0)
            self.assertIsNone(frozen.optimizer)
            self.assertEqual(frozen.policy._behavior_policy_version,
                             loaded.metadata["training_state"]["behavior_policy_version"])
            policy_generator = self.torch.Generator(device="cpu")
            policy_generator.manual_seed(cpu_train.seed_streams.policy)
            shuffle_generator = self.torch.Generator(device="cpu")
            shuffle_generator.manual_seed(cpu_train.seed_streams.optimizer)
            self.assertTrue(self.torch.equal(warm.policy_generator.get_state(), policy_generator.get_state()))
            self.assertTrue(self.torch.equal(warm.shuffle_generator.get_state(), shuffle_generator.get_state()))
            self.assertEqual(warm.curriculum_rng.getstate(), random.Random(cpu_train.seed_streams.curriculum).getstate())
            self.assertEqual(warm.controller_rng.getstate(), random.Random(cpu_train.seed_streams.noncombat).getstate())
            frame = (public_frame,)
            warm_choices = warm.policy.act(
                frame, generator=warm.policy_generator, deterministic=True,
            )
            frozen_choices = frozen.policy.act(
                frame, generator=frozen.policy_generator, deterministic=True,
            )
            _assert_cpu_model_outputs(self, warm_choices, frozen_choices)
            _assert_portable_choices(self, warm_choices, gpu_choices)
            self.assertEqual(warm_driver.reset_calls + eval_driver.reset_calls, 0)
            self.assertEqual(warm_driver.step_calls + eval_driver.step_calls, 0)
            self.assertEqual(warm_driver.close_calls, 0)
            self.assertEqual(eval_driver.close_calls, 0)
        finally:
            _close_owned_runtimes((warm, frozen), sys.exc_info()[1])
        assert warm is not None and frozen is not None
        assert warm_driver is not None and eval_driver is not None
        self.assertEqual(warm_driver.close_calls, 1)
        self.assertEqual(eval_driver.close_calls, 1)
        return {
            "warm_start_optimizer_fresh": True,
            "warm_start_rng_fresh": True,
            "warm_start_behavior_version": warm.policy._behavior_policy_version,
            "evaluation_optimizer_absent": frozen.optimizer is None,
            "evaluation_checkpoint_behavior_version": frozen.policy._behavior_policy_version,
            "cpu_driver_resets": warm_driver.reset_calls + eval_driver.reset_calls,
            "cpu_driver_steps": warm_driver.step_calls + eval_driver.step_calls,
        }

    def _run_mode(self, mode, test_name, expected_rollout):
        self.ledger.check_deadline()
        config = self.configs[mode]
        self.assertEqual((config.rollout_rows, config.rollout_steps), expected_rollout)
        self.assertEqual(config.budgets.max_training_decisions, 32)
        self.assertEqual(config.budgets.max_environment_actions, 32)
        self.assertEqual(config.budgets.max_optimizer_steps, 4)
        self.assertEqual(config.budgets.max_wall_seconds, 120.0)

        primary = None
        runtime = self.create_runtime(config, purpose="train")
        try:
            return self._run_mode_after_acquisition(
                mode, test_name, expected_rollout, config, runtime,
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            _close_owned_runtime(runtime, primary)

    def _run_mode_after_acquisition(
        self, mode, test_name, expected_rollout, config, runtime,
    ):
        from spire_learn.neural import runner as neural_runner

        self._trained_driver = runtime.collector.driver
        guarded_driver = _HELPER.GuardedDriver(runtime.collector.driver, self.ledger, mode)
        underlying_policy = runtime.policy
        guarded_policy = _HELPER.GuardedPolicy(underlying_policy, self.ledger, mode)
        self.assertIs(runtime.policy, underlying_policy)
        runtime.collector.driver = guarded_driver
        runtime.collector.policy = guarded_policy
        self.assertIs(runtime.collector.policy.inner, runtime.policy)
        runtime.optimizer = _HELPER.GuardedOptimizer(runtime.optimizer, self.ledger, mode)
        initial = {
            name: tensor.detach().cpu().clone()
            for name, tensor in underlying_policy.state_dict().items()
        }
        frames = []
        collect = runtime.collector.collect

        def collect_and_retain(**kwargs):
            rollout = collect(**kwargs)
            frames.extend(
                record.source for record in rollout.records if record.actor_training_mask
            )
            return rollout

        runtime.collector.collect = collect_and_retain
        update_metrics = []
        original_update = runtime.update

        def capture_update(rollout):
            metrics = original_update(rollout)
            update_metrics.append(metrics)
            return metrics

        runtime.update = capture_update
        report = None
        save_capture = {}
        save_writer_calls = 0
        original_save = runtime.save
        checkpoint_writer = neural_runner.save_checkpoint

        def recording_writer(*args, **kwargs):
            nonlocal save_writer_calls
            if save_writer_calls != 0:
                raise AssertionError("training invoked the checkpoint writer more than once")
            save_writer_calls += 1
            metadata = kwargs["metadata"]
            save_capture["training_counters"] = dict(metadata["training_state"]["counters"])
            save_capture["cumulative_prefix"] = dict(metadata["run_metadata"]["cumulative_prefix"])
            save_capture["collector_recovery"] = runtime.collector.recovery_state
            return checkpoint_writer(*args, **kwargs)

        def save_with_boundary(path, purpose):
            prefix_elapsed = float(runtime.recovered_progress.get("elapsed_seconds", 0.0))
            started_elapsed = prefix_elapsed + float(runtime.budget.elapsed())
            try:
                with patch.object(neural_runner, "save_checkpoint", new=recording_writer):
                    return original_save(path, purpose)
            finally:
                completed_elapsed = prefix_elapsed + float(runtime.budget.elapsed())
                save_capture["elapsed_bracket"] = (started_elapsed, completed_elapsed)

        runtime.save = save_with_boundary
        runtime.operation_limits = self.RuntimeOperationLimits(
            max_updates=1, monotonic_deadline=self.shared_deadline,
        )
        self.ledger.check_deadline()
        report = self.train(config, runtime=runtime, clock=time.monotonic)
        self.assertIsNotNone(report)
        self.assertEqual(report.exit_code, 0, report.summary)
        self.assertEqual(report.summary["status"], "complete", report.summary)
        lifecycle = report.summary.get("lifecycle")
        self.assertIsInstance(lifecycle, Mapping)
        self.assertEqual(lifecycle.get("status"), "complete")
        self.assertEqual(lifecycle.get("exit_code"), 0)
        self.assertEqual(lifecycle.get("close_errors"), [])
        self.assertEqual(lifecycle.get("wall_overshoot_seconds"), 0.0)
        self.assertEqual(report.summary.get("wall_overshoot_seconds"), 0.0)
        lifecycle_elapsed = lifecycle.get("elapsed_seconds")
        summary_elapsed = report.summary.get("elapsed_seconds")
        self.assertIs(type(lifecycle_elapsed), float)
        self.assertTrue(math.isfinite(lifecycle_elapsed) and lifecycle_elapsed >= 0.0)
        self.assertEqual(lifecycle_elapsed, summary_elapsed)
        self.assertTrue(report.summary.get("lifecycle_receipt_required"))
        self.assertNotIn("lifecycle_receipt_error", report.summary)
        self.assertEqual(save_writer_calls, 1, "the final checkpoint must be delegated exactly once")
        self.assertEqual(report.summary["optimizer"]["updates"], 1, report.summary)
        self.assertEqual(len(update_metrics), 1)
        self.assertTrue(all(math.isfinite(getattr(update_metrics[0], name)) for name in (
            "policy_loss", "value_loss", "entropy",
        )))
        self.assertGreater(report.summary["optimizer"]["accepted_steps"], 0, report.summary)
        self.assertEqual(report.summary["optimizer"]["attempted_steps"],
                         report.summary["optimizer"]["accepted_steps"])
        self.assertEqual(report.summary["optimizer"]["uncertain_steps"], 0)
        collection = report.summary["collection"]
        self.assertLessEqual(collection["learned_decisions"], 32)
        self.assertLessEqual(collection["attempted_environment_slots"], 32)
        self.assertLessEqual(collection["delivered_environment_slots"], 32)
        self.assertLessEqual(collection["attempted_reset_slots"], 36 if mode == "standalone" else 33)
        self.assertLessEqual(collection["delivered_reset_slots"], 36 if mode == "standalone" else 33)
        self.assertEqual(collection["faults"], 0, collection)
        self.assertEqual(collection["uncertain_environment_slots"], 0, collection)
        self.assertEqual(collection["uncertain_reset_slots"], 0, collection)
        self.assertEqual(collection["incomplete_environment_slots"], 0, collection)
        self.assertIsNotNone(report.run_dir)
        self.assertTrue(frames, "training must retain a public actor frame for parity checks")
        self.assertEqual(guarded_driver.close_calls, 1, "training driver must close exactly once")
        self.assertLess(time.monotonic(), self.shared_deadline)

        final_state = underlying_policy.state_dict()
        self.assertEqual(set(initial), set(final_state))
        self.assertTrue(all(self.torch.isfinite(value).all().item() for value in final_state.values()))
        changed = [name for name in initial if not self.torch.equal(initial[name], final_state[name].detach().cpu())]
        self.assertTrue(changed, "the accepted finite update must change at least one parameter")
        checkpoint = Path(report.run_dir) / "checkpoint-final"
        self.assertTrue(checkpoint.is_dir())

        from spire_learn.neural.checkpoint import load_checkpoint
        from spire_learn.neural.collector import CollectorRecovery

        saved = load_checkpoint(
            checkpoint, config.accepted_identities[0], purpose="resume", device="cpu",
        )
        training_state = saved.metadata["training_state"]
        run_metadata = saved.metadata["run_metadata"]
        self.assertEqual(training_state["counters"], save_capture["training_counters"])
        self.assertEqual(run_metadata["cumulative_prefix"], save_capture["cumulative_prefix"])
        self.assertEqual(training_state["counters"], run_metadata["cumulative_prefix"])
        self.assertEqual(run_metadata["runtime_device"], "cuda:0")
        self.assertEqual(run_metadata["runtime_gpu_uuid"], _HELPER.GPU_UUID)
        recovery_payload = run_metadata["collector_recovery"]
        expected_recovery = CollectorRecovery(
            recovery_payload["next_seed_ordinal"],
            recovery_payload["next_decision_sequence"],
            tuple(recovery_payload["next_episode_ordinals"]),
        )
        self.assertEqual(save_capture["collector_recovery"], expected_recovery)
        owned_recovery_payload = {
            "next_seed_ordinal": expected_recovery.next_seed_ordinal,
            "next_decision_sequence": expected_recovery.next_decision_sequence,
            "next_episode_ordinals": list(expected_recovery.next_episode_ordinals),
        }
        saved_counters = save_capture["training_counters"]
        saved_elapsed = saved_counters.get("elapsed_seconds")
        self.assertIs(type(saved_elapsed), float)
        self.assertTrue(math.isfinite(saved_elapsed) and saved_elapsed >= 0.0)
        bracket_start, bracket_end = save_capture["elapsed_bracket"]
        self.assertTrue(math.isfinite(bracket_start) and bracket_start >= 0.0)
        self.assertTrue(math.isfinite(bracket_end) and bracket_end >= bracket_start)
        self.assertLessEqual(bracket_start, saved_elapsed)
        self.assertLessEqual(saved_elapsed, bracket_end)
        remaining_configured_wall = config.budgets.max_wall_seconds - saved_elapsed
        self.assertTrue(
            math.isfinite(remaining_configured_wall) and remaining_configured_wall > 0.0,
            "saved checkpoint prefix exhausted the configured wall allowance",
        )
        remaining_shared_deadline = self.shared_deadline - time.monotonic()
        self.assertTrue(
            math.isfinite(remaining_shared_deadline) and remaining_shared_deadline > 0.0,
            "the earlier shared proof deadline elapsed before resume comparison",
        )
        saved_version = training_state["behavior_policy_version"]
        frames_for_compare = (frames[0],)
        deterministic_original = underlying_policy.act(
            frames_for_compare, generator=runtime.policy_generator, deterministic=True,
        )
        rng_before = self.ledger.mode_snapshot(mode)
        inert_resume, resume_driver = self._create_inert_runtime(
            config, purpose="resume", checkpoint=checkpoint,
        )
        resume_primary = None
        try:
            self.assertTrue(_tree_equal(underlying_policy.state_dict(), inert_resume.policy.state_dict(), self.torch))
            self.assertTrue(_tree_equal(runtime.optimizer.state_dict(), inert_resume.optimizer.state_dict(), self.torch))
            self.assertTrue(self.torch.equal(runtime.policy_generator.get_state(),
                                             inert_resume.policy_generator.get_state()))
            self.assertTrue(self.torch.equal(runtime.shuffle_generator.get_state(),
                                             inert_resume.shuffle_generator.get_state()))
            self.assertEqual(runtime.curriculum_rng.getstate(), inert_resume.curriculum_rng.getstate())
            self.assertEqual(runtime.controller_rng.getstate(), inert_resume.controller_rng.getstate())
            self.assertEqual(inert_resume.recovered_progress, training_state["counters"])
            self.assertEqual(inert_resume.update_count, training_state["counters"]["update_count"])
            self.assertEqual(inert_resume.policy._behavior_policy_version, saved_version)
            self.assertEqual(inert_resume.collector.recovery_state, expected_recovery)
            self.assertEqual(inert_resume.optimizer_attempted_steps, training_state["counters"]["optimizer_attempted_steps"])
            _assert_fresh_resume_optimizer(self, inert_resume.optimizer)
            later_counters = runtime._latest_counters()
            self.assertEqual(
                {name: value for name, value in later_counters.items() if name != "elapsed_seconds"},
                {name: value for name, value in saved_counters.items() if name != "elapsed_seconds"},
            )
            later_elapsed = later_counters.get("elapsed_seconds")
            self.assertIs(type(later_elapsed), float)
            self.assertTrue(math.isfinite(later_elapsed) and later_elapsed >= saved_elapsed)
            self.assertEqual(inert_resume.collector.recovery_state, save_capture["collector_recovery"])
            self._compare_optimizer_device(runtime.optimizer, "cuda")
            self._compare_optimizer_device(inert_resume.optimizer, "cuda")
            deterministic_restored = inert_resume.policy.act(
                frames_for_compare, generator=inert_resume.policy_generator, deterministic=True,
            )
            _assert_exact_choices(self, deterministic_original, deterministic_restored)

            self.ledger.reserve_diagnostic_pair(mode)
            original_generator = self.torch.Generator(device=config.device)
            restored_generator = self.torch.Generator(device=config.device)
            original_generator.set_state(runtime.policy_generator.get_state())
            restored_generator.set_state(inert_resume.policy_generator.get_state())
            original_sample = _HELPER.GuardedPolicy(
                underlying_policy, self.ledger, mode, diagnostic=True,
            ).act(frames_for_compare, generator=original_generator, deterministic=False)
            restored_sample = _HELPER.GuardedPolicy(
                inert_resume.policy, self.ledger, mode, diagnostic=True,
            ).act(frames_for_compare, generator=restored_generator, deterministic=False)
            _assert_exact_choices(self, original_sample, restored_sample)
            self.assertTrue(self.torch.equal(original_generator.get_state(), restored_generator.get_state()))
            self.assertEqual(resume_driver.reset_calls, 0)
            self.assertEqual(resume_driver.step_calls, 0)
        except BaseException as error:
            resume_primary = error
            raise
        finally:
            _close_owned_runtime(inert_resume, resume_primary)
        self.assertEqual(resume_driver.close_calls, 1)
        self.assertEqual(self.ledger.mode_snapshot(mode)["optimizer_attempted"], rng_before.get("optimizer_attempted", 0))

        portable = self._compare_checkpoint_portability(
            mode, config, checkpoint, saved, frames[0], deterministic_original,
        )
        mode_counters = {
            **_HELPER.project_mode_counters(
                self.ledger,
                mode,
                faults=collection["faults"],
                incomplete_environment_slots=collection["incomplete_environment_slots"],
            ),
            "diagnostic_rows_unclaimed": self.ledger.diagnostic_reserved[mode],
        }
        self.assertEqual(mode_counters["diagnostic_rows_unclaimed"], 0)
        checks = {
            "one_shared_deadline": time.monotonic() < self.shared_deadline,
            "accepted_finite_update_and_changed_weights": True,
            "final_checkpoint_saved": checkpoint.is_dir(),
            "same_backend_weights_and_adam": True,
            "adam_scalar_steps_cpu_moments_cuda": True,
            "same_backend_torch_rng_and_python_rng": True,
            "same_backend_counters_version_and_recovery": True,
            "same_frame_deterministic_and_sampled_full_choice_parity": True,
            "portable_cpu_weights_and_tolerant_outputs": True,
            "warm_start_fresh_state_and_frozen_evaluation": True,
            "cpu_comparisons_used_inert_drivers_without_steps": (
                portable["cpu_driver_resets"] == 0 and portable["cpu_driver_steps"] == 0
            ),
            "training_driver_closed_once": guarded_driver.close_calls == 1,
            "actor_critic_and_shared_hashes_recorded": True,
            "zero_fault_uncertainty_and_incomplete_effects": True,
        }
        result = {
            "schema_id": _HELPER.MODE_RESULT_SCHEMA,
            "status": "passed",
            "mode": mode,
            "test_id": test_name,
            "exit_code": 0,
            "skipped": False,
            "updates": {
                "accepted": report.summary["optimizer"]["accepted_steps"],
                "finite": all(math.isfinite(getattr(update_metrics[0], name)) for name in (
                    "policy_loss", "value_loss", "entropy",
                )),
                "weights_changed": bool(changed),
            },
            "weights": _state_groups(final_state, self.torch),
            "counters": {
                **{key: value for key, value in mode_counters.items() if key != "diagnostic_rows_unclaimed"},
            },
            "checks": checks,
            "portable_cpu": portable,
            "checkpoint": {
                "path": str(checkpoint.relative_to(ROOT)),
                "metadata_raw_sha256": _HELPER.sha256_file(checkpoint / "metadata.json"),
                "metadata_canonical_sha256": _HELPER.canonical_sha256(
                    _HELPER.load_json(checkpoint / "metadata.json")
                ),
                "policy_version": saved_version,
                "collector_recovery": owned_recovery_payload,
            },
            "device": self.device_identity,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        _HELPER.validate_mode_result(result, mode=mode, test_id=test_name)
        self._write_mode_evidence(config, result)
        self.ledger.check_deadline()
        return result

    def test_01_standalone_b4_h8(self):
        try:
            result = self._run_mode(
                "standalone",
                "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                (4, 8),
            )
        except BaseException as error:
            type(self)._first_mode_failure = f"{type(error).__name__}: {str(error)[:256]}"
            raise
        self.assertEqual(result["status"], "passed")

    def test_02_run_b1_h32(self):
        if getattr(type(self), "_first_mode_failure", None) is not None:
            raise RuntimeError(
                "standalone proof failed; run mode is halted without another workload: "
                + str(type(self)._first_mode_failure)
            )
        result = self._run_mode(
            "run",
            "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32",
            (1, 32),
        )
        self.assertEqual(result["status"], "passed")
        totals = self.ledger.snapshot()
        self.assertEqual(totals["steps"], totals["steps_delivered"])
        self.assertEqual(totals["resets"], totals["resets_delivered"])
        self.assertLessEqual(totals["steps"], 64)
        self.assertLessEqual(totals["resets"], 69)
        self.assertLessEqual(totals["optimizer_attempted"], 8)
        self.assertEqual(totals["optimizer_attempted"], totals["optimizer_accepted"])
        self.assertLessEqual(totals["actor_samples"], 64)
        self.assertEqual(totals["diagnostic_samples"], 4)
        self.assertEqual(totals["diagnostic_samples_delivered"], 4)
        self.assertEqual(set(totals["diagnostic_rows_unclaimed"].values()), {0})


if __name__ == "__main__":
    unittest.main()
