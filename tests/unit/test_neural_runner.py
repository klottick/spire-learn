"""Pure runner tests with scripted rollouts and artifact writers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import builtins
import copy
import os
from pathlib import Path
import random
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spire_learn.neural.collector import CollectorRecovery
from spire_learn.neural.config import NeuralBudget, load_neural_config
from spire_learn.neural.contracts import (
    BufferArray,
    CollectionDiagnostics,
    CombatFrame,
    CumulativeCounters,
    PolicyChoice,
    ResourceCheckpoint,
    Rollout,
    SeedAllocation,
    StepRecord,
    TaskSignal,
)
from spire_learn.neural.ppo import UpdateMetrics


ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"
GPU_UUID = "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d"
GPU_NAME = "NVIDIA GeForce RTX 5090"
GPU_DRIVER = "595.91.07"


class _AssemblyGenerator:
    def __init__(self, device, events):
        self.device = device
        self.events = events
        self.seed = None
        self.states = []

    def manual_seed(self, seed):
        self.seed = seed
        self.events.append(("manual_seed", self.device, seed))

    def set_state(self, state):
        self.states.append(state)
        self.events.append(("set_state", self.device, state))

    def get_state(self):
        return f"state:{self.device}:{self.seed}"


class _AssemblyCuda:
    def __init__(self, events):
        self.events = events

    def is_available(self):
        self.events.append("cuda_available")
        return True

    def device_count(self):
        self.events.append("cuda_device_count")
        return 1

    def set_device(self, index):
        self.events.append(("cuda_set_device", index))

    def init(self):
        self.events.append("cuda_init")

    def get_device_properties(self, index):
        self.events.append(("cuda_properties", index))
        return SimpleNamespace(name=GPU_NAME, total_memory=32 * 1024**3)

    def get_device_capability(self, index):
        self.events.append(("cuda_capability", index))
        return (12, 0)

    def get_arch_list(self):
        self.events.append("cuda_arch_list")
        return ["sm_120"]

    def mem_get_info(self, index):
        self.events.append(("cuda_mem_get_info", index))
        return 16 * 1024**3, 32 * 1024**3

    def set_per_process_memory_fraction(self, fraction, index):
        self.events.append(("cuda_memory_fraction", fraction, index))


class _AssemblyTorch:
    __version__ = "2.14.1+cu130"

    def __init__(self, events):
        self.events = events
        self.version = SimpleNamespace(cuda="13.0")
        self.cuda = _AssemblyCuda(events)
        self.intraop = 4
        self.interop = 1
        self.generators = []
        self.optim = SimpleNamespace(Adam=self._adam)
        self.adams = []

    def Generator(self, *, device):
        self.events.append(("generator", device))
        generator = _AssemblyGenerator(device, self.events)
        self.generators.append(generator)
        return generator

    def get_num_threads(self):
        return self.intraop

    def set_num_threads(self, count):
        self.events.append(("set_num_threads", count))
        self.intraop = count

    def get_num_interop_threads(self):
        return self.interop

    def set_num_interop_threads(self, count):
        self.events.append(("set_num_interop_threads", count))
        self.interop = count

    def _adam(self, parameters, *, lr, eps):
        optimizer = _AssemblyAdam()
        self.adams.append(optimizer)
        self.events.append(("adam", lr, eps))
        return optimizer


class _AssemblyAdam:
    def __init__(self):
        self.param_groups = []
        self.loaded_states = []

    def load_state_dict(self, state):
        self.loaded_states.append(state)


class _AssemblyPolicy:
    def __init__(self):
        self._behavior_policy_version = 0
        self.loaded_weights = []

    def parameters(self):
        return (object(),)

    def load_state_dict(self, weights, *, strict):
        self.loaded_weights.append((weights, strict))


class _AssemblyDriver:
    def __init__(self, events):
        self.events = events
        self.schema = SimpleNamespace(descriptor_digest="fake-schema")
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.events.append("driver_close")


class _AssemblyCollector:
    def __init__(self, driver, policy, config, **kwargs):
        self.driver = driver
        self.policy = policy
        self.config = config
        self.partial_rollout = None
        self.learned_combat_policy = kwargs.get("learned_combat_policy", True)
        self.budget = kwargs["budget"]
        self._recovery = kwargs.get("recovery") or CollectorRecovery(
            0, 0, (0,) * config.rollout_rows
        )
        self.close_calls = 0

    @property
    def recovery_state(self):
        return self._recovery

    def close(self):
        self.close_calls += 1
        self.driver.close()


@contextmanager
def _patch_runtime_assembly(
    events,
    *,
    torch_module=None,
    driver=None,
    policy=None,
    policy_factory=None,
    preflight=None,
    loader=None,
    reject_torch_import=False,
):
    from spire_learn.neural import model, runner
    import spire_learn.neural.envs as envs

    selected_driver = driver or _AssemblyDriver(events)
    selected_policy = policy or _AssemblyPolicy()
    original_import = builtins.__import__

    def traced_import(name, *args, **kwargs):
        if name == "torch":
            events.append("torch_import")
            if reject_torch_import:
                raise AssertionError("unexpected Torch import")
        return original_import(name, *args, **kwargs)

    def make_driver(config, cases):
        events.append("make_driver")
        return selected_driver

    def build_policy(schema, model_config, *, seed, device):
        events.append(("build_policy", seed, device))
        if policy_factory is not None:
            return policy_factory(schema, model_config, seed=seed, device=device)
        return selected_policy

    def nvidia_query(argv, **kwargs):
        events.append("nvidia_smi")
        return SimpleNamespace(stdout=f"{GPU_UUID}, {GPU_NAME}, {GPU_DRIVER}, 32768, 16384\n")

    with ExitStack() as stack:
        if torch_module is not None:
            stack.enter_context(patch.dict(sys.modules, {"torch": torch_module}))
        stack.enter_context(patch("builtins.__import__", side_effect=traced_import))
        stack.enter_context(patch.object(envs, "make_driver", side_effect=make_driver))
        stack.enter_context(patch.object(model, "build_policy", side_effect=build_policy))
        stack.enter_context(patch.object(runner, "Collector", _AssemblyCollector))
        stack.enter_context(patch("spire_learn.neural.device.subprocess.run", side_effect=nvidia_query))
        if preflight is not None:
            stack.enter_context(patch.object(runner, "preflight_checkpoint", side_effect=preflight, create=True))
        if loader is not None:
            stack.enter_context(patch.object(runner, "load_checkpoint", side_effect=loader))
        yield selected_driver, selected_policy


def _resume_bundle(config, run_metadata_fields):
    from spire_learn.neural import runner

    progress = {name: 0 for name in runner._PROGRESS_FIELDS}
    progress["elapsed_seconds"] = 0.0
    recovery = {
        "next_seed_ordinal": 0,
        "next_decision_sequence": 0,
        "next_episode_ordinals": [0] * config.rollout_rows,
    }
    run_metadata = {
        "mode": config.mode,
        "operation": "train",
        "split": "train",
        "partition": "train",
        "config_sha256": config.config_sha256,
        "manifest_sha256": config.manifest_sha256,
        "environment_revision": config.environment_revision,
        "collector_recovery": recovery,
        "cumulative_prefix": dict(progress),
        **run_metadata_fields,
    }
    training_state = {
        "optimizer_step": 0,
        "next_environment_seed_ordinal": 0,
        "next_decision_sequence": 0,
        "counters": dict(progress),
        "behavior_policy_version": 0,
    }
    metadata = {"run_metadata": run_metadata, "training_state": training_state}
    policy_state = bytearray(b"policy-rng-state")
    minibatch_state = bytearray(b"minibatch-rng-state")
    curriculum_state = random.Random(31).getstate()
    controller_state = random.Random(47).getstate()
    optimizer_state = {"state": {"unmodified": object()}}
    loaded = SimpleNamespace(
        metadata=metadata,
        model_tensors={"weight": object()},
        rng_state={
            "policy": policy_state,
            "minibatch": minibatch_state,
            "curriculum": curriculum_state,
            "controller": controller_state,
        },
        optimizer_state=optimizer_state,
    )
    return metadata, loaded, policy_state, minibatch_state, curriculum_state, controller_state, optimizer_state


def _frame(value: int) -> CombatFrame:
    return CombatFrame({"player": BufferArray("int8", (1,), bytes((value,)))}, FINGERPRINT)


def _allocation(config, slot: int, ordinal: int, *, case_index: int | None = None) -> SeedAllocation:
    case = config.cases.for_mode(config.mode)[case_index if case_index is not None else slot % len(config.cases.for_mode(config.mode))]
    return SeedAllocation(
        stream_id=f"stream-{slot}", slot=slot, case_id=case.case_id,
        entry_id=case.entry_id, authored_reference_seed=case.environment_seed,
        actual_seed=case.environment_seed if config.split != "train" else ordinal,
        ordinal=ordinal,
        allocation_policy_id=("environment-seed-for-replica0-v1" if config.split == "train"
                              else "authored-reference-seed-v1"),
        delivery="delivered",
    )


def _record(config, *, slot: int, decision: int, version: int, terminal: bool,
            learned: bool = True, task_id: str | None = None) -> StepRecord:
    source = _frame(slot + decision + 1)
    successor = _frame(slot + decision + 2)
    choice = (PolicyChoice(0, -0.25, 0.5, version) if learned else PolicyChoice(0))
    signal = (TaskSignal(1.0, True, False, False) if terminal
              else TaskSignal(0.0, False, True, True))
    return StepRecord(
        stream_id=f"stream-{slot}", task_id=task_id or f"task-{slot}-0",
        decision_id=f"decision-{decision}", source=source, successor=successor,
        native_reward=0.0, terminated=False, truncated=False, task_signal=signal,
        choice=choice, actor_training_mask=learned, next_value=0.25 if learned else None,
    )


def _rollout(config, records, *, allocations=(), stop_reason="horizon"):
    records = tuple(records)
    actor_rows = sum(record.actor_training_mask for record in records)
    fixed_rows = len(records) - actor_rows
    terminals = sum(record.task_signal.task_terminal for record in records)
    environment_slots = len(records)
    reset_slots = len(allocations)
    diagnostics = CollectionDiagnostics(
        attempted_environment_slots=environment_slots,
        forwarded_environment_slots=environment_slots,
        known_accepted_environment_slots=environment_slots,
        delivered_environment_slots=environment_slots,
        reset_operations=int(reset_slots > 0),
        reset_slots=reset_slots,
        attempted_reset_slots=reset_slots,
        forwarded_reset_slots=reset_slots,
        known_accepted_reset_slots=reset_slots,
        delivered_reset_slots=reset_slots,
        learned_decisions=actor_rows,
        fixed_decisions=fixed_rows,
        tasks_started=reset_slots,
        task_terminals=terminals,
        tasks_completed=terminals,
        victories=terminals,
        next_seed_ordinal=reset_slots,
        seed_allocations=tuple(allocations),
        stop_reason=stop_reason,
    )
    counters = CumulativeCounters(
        rows=len(records), actor_rows=actor_rows, fixed_rows=fixed_rows,
        task_terminals=terminals, environment_terminations=0,
        environment_truncations=0,
    )
    return Rollout(config.mode, config.split, config.accepted_identities[0], records,
                   counters, diagnostics)


class FakeCollector:
    def __init__(self, config, rollouts=(), *, recovery=None, collect_error=None):
        self.config = config
        self.rollouts = list(rollouts)
        self.collect_error = collect_error
        self.collect_calls = []
        self.partial_rollout = None
        self.learned_combat_policy = True
        self.policy = SimpleNamespace(_behavior_policy_version=0)
        self.driver = SimpleNamespace(schema=None)
        self._recovery = recovery or CollectorRecovery(
            0, 0, (0,) * config.rollout_rows
        )
        self.close_calls = 0
        self.budget = None

    @property
    def recovery_state(self):
        return self._recovery

    def collect(self, *, max_learned_decisions, max_env_actions, policy_version):
        self.collect_calls.append((max_learned_decisions, max_env_actions, policy_version))
        if self.collect_error is not None:
            raise self.collect_error
        if not self.rollouts:
            raise AssertionError("unexpected collector dispatch")
        rollout = self.rollouts.pop(0)
        if callable(rollout):
            rollout = rollout(policy_version)
        self.partial_rollout = rollout
        return rollout

    def close(self):
        self.close_calls += 1


class FakeRuntime:
    def __init__(self, config, collector, *, prefix=None, close_error=None,
                 save_hook=None, close_hook=None):
        self.config = config
        self.collector = collector
        self.recovered_progress = dict(prefix or {})
        self.optimizer_attempted_steps = int(self.recovered_progress.get("optimizer_attempted_steps", 0))
        self.optimizer_accepted_steps = int(self.recovered_progress.get("optimizer_accepted_steps", 0))
        self.optimizer_uncertain_steps = int(self.recovered_progress.get("optimizer_uncertain_steps", 0))
        self.update_count = int(self.recovered_progress.get("update_count", 0))
        self.policy_kind = "checkpoint"
        self.action_selection = "deterministic"
        self.controller_id = "fake-controller-v1"
        self.parent_checkpoint = None
        self.update_calls = []
        self.saved = []
        self.close_calls = 0
        self.close_error = close_error
        self.save_hook = save_hook
        self.close_hook = close_hook
        self.budget = None

    def update(self, rollout):
        version = self.collector.policy._behavior_policy_version
        actor = tuple(record for record in rollout.records if record.actor_training_mask)
        if any(record.choice.behavior_policy_version != version for record in actor):
            raise RuntimeError("stale-policy rollout reached updater")
        self.update_calls.append(rollout)
        steps = self.config.ppo.epochs * ((len(actor) + self.config.ppo.minibatch_rows - 1)
                                           // self.config.ppo.minibatch_rows)
        resulting = version + int(steps > 0)
        self.collector.policy._behavior_policy_version = resulting
        return UpdateMetrics(
            0.0, 0.0, 0.0, steps, steps, None, version, resulting, len(actor)
        )

    def record_update(self, metrics):
        self.optimizer_attempted_steps += metrics.attempted_steps
        self.optimizer_accepted_steps += metrics.accepted_steps
        self.update_count += 1

    def save(self, path, purpose):
        self.saved.append((Path(path), purpose))
        if self.save_hook is not None:
            self.save_hook()

    def close(self):
        self.close_calls += 1
        if self.close_hook is not None:
            self.close_hook()
        if self.close_error is not None:
            raise self.close_error


class FakeWriter:
    def __init__(self, *, event_error=None, close_error=None, close_hook=None):
        self.event_error = event_error
        self.close_error = close_error
        self.close_hook = close_hook
        self.events = []
        self.summaries = []
        self.lifecycle_receipts = []
        self.close_calls = 0

    def event(self, event):
        if self.event_error is not None:
            raise self.event_error
        self.events.append(event)

    def finalize(self, summary):
        self.summaries.append(summary)

    def close(self):
        self.close_calls += 1
        if self.close_hook is not None:
            self.close_hook()
        if self.close_error is not None:
            raise self.close_error

    def write_lifecycle_receipt(self, receipt):
        self.lifecycle_receipts.append(dict(receipt))


class NeuralRunnerTests(unittest.TestCase):
    def setUp(self):
        self.train_config = replace(
            load_neural_config(ROOT / "configs/neural/standalone.toml"),
            rollout_rows=1,
            rollout_steps=1,
            budgets=NeuralBudget(8, 4, 4, 60.0),
        )
        self.selection_config = replace(
            load_neural_config(ROOT / "configs/neural/standalone-selection.toml"),
            rollout_rows=1,
            rollout_steps=1,
        )

    def _run_train(self, config, runtime, writer=None):
        from spire_learn.neural import runner

        selected_writer = writer or FakeWriter()
        with patch.object(runner, "_writer_start",
                          return_value=(selected_writer, Path("/tmp/neural-test-run"))):
            report = runner.train(config, runtime=runtime, clock=lambda: 0.0)
        return report, selected_writer

    def test_operation_gate_rejects_evaluation_before_collection_or_update(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, operation="evaluate", split="selection", partition="selection")
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)
        report = runner.train(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(collector.collect_calls, [])
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(runtime.close_calls, 1)

    def test_optimizer_budget_is_checked_before_any_collection_or_update(self):
        config = replace(self.train_config, budgets=NeuralBudget(3, 4, 4, 60.0))
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(collector.collect_calls, [])
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(runtime.saved[0][1], "final")

    def test_empty_collection_stops_without_repolling_until_wall_budget(self):
        config = self.train_config
        empty = _rollout(config, (), stop_reason="inactive_reset_boundary")
        collector = FakeCollector(config, (empty,))
        runtime = FakeRuntime(config, collector)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.summary["stop_reason"], "inactive_reset_boundary")
        self.assertEqual(len(collector.collect_calls), 1)
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(runtime.saved[0][1], "final")

    def test_runtime_update_ceiling_stops_before_a_third_collection(self):
        from spire_learn.neural.runner import RuntimeOperationLimits

        config = replace(self.train_config, budgets=NeuralBudget(16, 4, 4, 60.0))
        rollouts = tuple(
            _rollout(config, (_record(config, slot=0, decision=index, version=index,
                                      terminal=False, task_id=f"task-0-{index}"),))
            for index in range(4)
        )
        collector = FakeCollector(config, rollouts)
        runtime = FakeRuntime(config, collector)
        runtime.operation_limits = RuntimeOperationLimits(max_updates=2)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(len(collector.collect_calls), 2)
        self.assertEqual(len(runtime.update_calls), 2)
        self.assertEqual(report.summary["optimizer"]["updates"], 2)

    def test_segment_environment_ceiling_bounds_real_fixed_only_run_collector(self):
        from spire_learn.neural.collector import Collector
        from spire_learn.neural.runner import RuntimeOperationLimits
        from spire_learn.experiments.budget import Budget
        from tests.unit.test_neural_collector import ScriptedDriver, FakePolicy, row, transition

        for operational, lifetime, expected in ((103, 200, 3), (100, 200, 0), (103, 102, 2)):
            with self.subTest(operational=operational, lifetime=lifetime):
                config = replace(load_neural_config(ROOT / "configs/neural/run.toml"),
                                 rollout_steps=128, budgets=NeuralBudget(16, 100, lifetime, 60.0))
                boundary = row(1, active=False)
                driver = ScriptedDriver("run", 1, reset_rows=lambda *args: boundary,
                    steps=[lambda before, actions: (transition(before[0], boundary),)] * 8)
                policy = FakePolicy(config)
                policy._behavior_policy_version = 0
                collector = Collector(driver, policy, config, controller_rng=random.Random(1),
                    curriculum_rng=random.Random(2), budget=Budget(60.0, lambda: 0.0))
                runtime = FakeRuntime(config, collector, prefix={"attempted_environment_slots": 100})
                runtime.operation_limits = RuntimeOperationLimits(max_environment_actions=operational)
                report, _ = self._run_train(config, runtime)
                self.assertEqual(report.exit_code, 0, report.summary)
                self.assertEqual(len(driver.step_calls), expected)
                self.assertEqual(report.summary["stop_reason"], "environment_action_budget")
                self.assertEqual(runtime.update_calls, [])

    def test_runtime_limit_positional_deadline_contract_is_preserved(self):
        from spire_learn.neural.runner import RuntimeOperationLimits
        limits = RuntimeOperationLimits(2, 15.0)
        self.assertEqual(limits.monotonic_deadline, 15.0)
        self.assertIsNone(limits.max_environment_actions)

    def test_operational_environment_ceiling_rejects_invalid_counts(self):
        from spire_learn.neural.runner import RuntimeOperationLimits
        for value in (-1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RuntimeOperationLimits(max_environment_actions=value)

    def test_shared_deadline_uses_remaining_time_and_zero_stops_before_dispatch(self):
        from spire_learn.neural.runner import RuntimeOperationLimits

        config = self.train_config
        clock_value = [10.0]
        clock = lambda: clock_value[0]
        first = FakeRuntime(config, FakeCollector(config, (_rollout(config, ()),)))
        first.operation_limits = RuntimeOperationLimits(monotonic_deadline=15.0)
        report, _ = self._run_train_with_clock(config, first, clock)
        self.assertEqual(report.exit_code, 0)
        self.assertAlmostEqual(first.budget.remaining(), 5.0)

        clock_value[0] = 13.0
        second_collector = FakeCollector(config, (_rollout(config, ()),))
        second = FakeRuntime(config, second_collector)
        second.operation_limits = RuntimeOperationLimits(monotonic_deadline=15.0)
        report, _ = self._run_train_with_clock(config, second, clock)
        self.assertEqual(report.exit_code, 0)
        self.assertAlmostEqual(second.budget.remaining(), 2.0)

        clock_value[0] = 15.0
        expired_collector = FakeCollector(config)
        expired = FakeRuntime(config, expired_collector)
        expired.operation_limits = RuntimeOperationLimits(monotonic_deadline=15.0)
        report, _ = self._run_train_with_clock(config, expired, clock)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(expired_collector.collect_calls, [])
        self.assertEqual(expired.update_calls, [])
        self.assertEqual(expired.saved[0][1], "final")
        self.assertEqual(expired.budget.remaining(), 0.0)

    def test_evaluate_consumes_an_exhausted_absolute_deadline_before_collection(self):
        from spire_learn.neural import runner
        from spire_learn.neural.runner import RuntimeOperationLimits

        config = self.selection_config
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)
        runtime.operation_limits = RuntimeOperationLimits(monotonic_deadline=5.0)
        writer = FakeWriter()
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-eval-deadline-test"))):
            report = runner.evaluate_policy(config, runtime=runtime, clock=lambda: 5.0)

        self.assertEqual(report.exit_code, 1)
        self.assertEqual(collector.collect_calls, [])
        self.assertEqual(runtime.close_calls, 1)
        self.assertEqual(writer.close_calls, 1)

    def test_runner_budgets_satisfy_the_accepted_ppo_budget_boundary(self):
        from spire_learn.experiments.budget import Budget
        from spire_learn.neural.ppo import _budget_exhausted
        from spire_learn.neural.runner import _RunnerBudget

        now = [10.0]
        positive = _RunnerBudget(5.0, lambda: now[0], None)
        self.assertIsInstance(positive, Budget)
        self.assertFalse(_budget_exhausted(positive))

        expired_deadline = _RunnerBudget(5.0, lambda: now[0], 10.0)
        self.assertIsInstance(expired_deadline, Budget)
        self.assertEqual(expired_deadline.allowed_seconds, 0.0)
        self.assertEqual(expired_deadline.remaining(), 0.0)
        self.assertTrue(_budget_exhausted(expired_deadline))

        zero = _RunnerBudget(0.0, lambda: now[0], None)
        self.assertIsInstance(zero, Budget)
        self.assertEqual(zero.allowed_seconds, 0.0)
        self.assertEqual(zero.remaining(), 0.0)
        self.assertTrue(_budget_exhausted(zero))

    def test_train_passes_a_budget_instance_to_a_strict_update_boundary(self):
        from spire_learn.experiments.budget import Budget
        from spire_learn.neural.ppo import _budget_exhausted
        from spire_learn.neural.runner import RuntimeOperationLimits

        class StrictBudgetRuntime(FakeRuntime):
            def update(self, rollout):
                if not isinstance(self.budget, Budget):
                    raise AssertionError("runner supplied a non-Budget object to update")
                self.asserted_budget_exhausted = _budget_exhausted(self.budget)
                return super().update(rollout)

        config = self.train_config
        allocation = _allocation(config, 0, 0)
        rollout = _rollout(config, (
            _record(config, slot=0, decision=0, version=0, terminal=False),
        ), allocations=(allocation,))
        collector = FakeCollector(config, (rollout,))
        runtime = StrictBudgetRuntime(config, collector)
        runtime.operation_limits = RuntimeOperationLimits(max_updates=1)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(report.exit_code, 0, report.summary)
        self.assertEqual(len(runtime.update_calls), 1)
        self.assertFalse(runtime.asserted_budget_exhausted)

    def _run_train_with_clock(self, config, runtime, clock):
        from spire_learn.neural import runner

        writer = FakeWriter()
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-deadline-test"))):
            report = runner.train(config, runtime=runtime, clock=clock)
        return report, writer

    def _control_operation_inputs(self, operation, *, first_control=None):
        config = self.train_config if operation == "train" else self.selection_config
        if first_control is None:
            allocation = _allocation(config, 0, 0)
            rollout = _rollout(config, (
                _record(config, slot=0, decision=0, version=0, terminal=False),
            ), allocations=(allocation,))
            collector = FakeCollector(config, (rollout,))
            writer = FakeWriter(event_error=OSError("ordinary event failure"))
        else:
            collector = FakeCollector(config, collect_error=first_control)
            writer = FakeWriter()
        return config, FakeRuntime(config, collector), writer

    def _run_with_finalization_controls(
        self, operation, config, runtime, writer, *, clock, receipt_control=None,
    ):
        from spire_learn.neural import runner

        invoke = runner.train if operation == "train" else runner.evaluate_policy
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path(f"/tmp/neural-fix2-{operation}"))):
            if receipt_control is None:
                return invoke(config, runtime=runtime, clock=clock)
            with patch.object(runner, "_write_lifecycle_receipt", side_effect=receipt_control):
                return invoke(config, runtime=runtime, clock=clock)

    def test_final_clock_process_control_supersedes_ordinary_failure_in_train_and_evaluate(self):
        for operation in ("train", "evaluate"):
            with self.subTest(operation=operation):
                config, runtime, writer = self._control_operation_inputs(operation)
                control = KeyboardInterrupt(f"{operation} final clock control")

                def clock():
                    if writer.close_calls:
                        raise control
                    return 0.0

                with self.assertRaises(KeyboardInterrupt) as caught:
                    self._run_with_finalization_controls(
                        operation, config, runtime, writer, clock=clock,
                    )
                self.assertIs(caught.exception, control)
                notes = "\n".join(getattr(control, "__notes__", ()))
                self.assertIn("ordinary event failure", notes)
                self.assertIn("OSError", notes)
                self.assertEqual(runtime.close_calls, 1)
                self.assertEqual(writer.close_calls, 1)

    def test_receipt_process_control_supersedes_ordinary_failure_in_train_and_evaluate(self):
        for operation in ("train", "evaluate"):
            with self.subTest(operation=operation):
                config, runtime, writer = self._control_operation_inputs(operation)
                control = SystemExit(f"{operation} receipt control")
                with self.assertRaises(SystemExit) as caught:
                    self._run_with_finalization_controls(
                        operation, config, runtime, writer,
                        clock=lambda: 0.0, receipt_control=control,
                    )
                self.assertIs(caught.exception, control)
                notes = "\n".join(getattr(control, "__notes__", ()))
                self.assertIn("ordinary event failure", notes)
                self.assertIn("OSError", notes)
                self.assertEqual(runtime.close_calls, 1)
                self.assertEqual(writer.close_calls, 1)

    def test_first_process_control_is_preserved_and_later_control_is_noted(self):
        for operation in ("train", "evaluate"):
            with self.subTest(operation=operation):
                first = KeyboardInterrupt(f"{operation} original operation control")
                later = SystemExit(f"{operation} later receipt control")
                config, runtime, writer = self._control_operation_inputs(
                    operation, first_control=first,
                )
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self._run_with_finalization_controls(
                        operation, config, runtime, writer,
                        clock=lambda: 0.0, receipt_control=later,
                    )
                self.assertIs(caught.exception, first)
                notes = "\n".join(getattr(first, "__notes__", ()))
                self.assertIn("SystemExit", notes)
                self.assertIn("later receipt control", notes)
                self.assertEqual(runtime.close_calls, 1)
                self.assertEqual(writer.close_calls, 1)

    def test_reloaded_choice_parity_checks_an_earlier_log_probability_mismatch(self):
        from tests.integration.test_neural_training import _assert_choice_parity

        actual = (
            SimpleNamespace(original_action_index=1, value=0.5, log_probability=-0.75),
            SimpleNamespace(original_action_index=2, value=0.25, log_probability=-0.5),
        )
        expected = (
            SimpleNamespace(original_action_index=1, value=0.5, log_probability=-0.5),
            SimpleNamespace(original_action_index=2, value=0.25, log_probability=-0.5),
        )
        with self.assertRaisesRegex(AssertionError, "choice 0 log_probability"):
            _assert_choice_parity(actual, expected)

    def test_resume_prefix_offsets_each_budget_exactly_once(self):
        from spire_learn.neural.runner import _PROGRESS_FIELDS

        config = replace(self.train_config, budgets=NeuralBudget(8, 4, 4, 60.0))
        prefix = {name: 0 for name in _PROGRESS_FIELDS}
        prefix.update({
            "attempted_environment_slots": 2,
            "forwarded_environment_slots": 2,
            "known_accepted_environment_slots": 2,
            "delivered_environment_slots": 2,
            "delivered_reset_slots": 2,
            "learned_decisions": 3,
            "optimizer_attempted_steps": 4,
            "optimizer_accepted_steps": 4,
        })
        prefix["elapsed_seconds"] = 5.0
        case = config.cases.for_mode(config.mode)[0]
        recovery = CollectorRecovery(2, 2, (2,))
        allocation = SeedAllocation(
            "stream-0", 0, case.case_id, case.entry_id, case.environment_seed,
            7, 2, "environment-seed-for-replica0-v1", "delivered",
        )
        rollout = _rollout(
            config, (_record(config, slot=0, decision=2, version=0, terminal=False,
                             task_id="task-0-2"),), allocations=(allocation,),
        )
        collector = FakeCollector(config, (rollout,), recovery=recovery)
        runtime = FakeRuntime(config, collector, prefix=prefix)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(collector.collect_calls, [(1, 2, 0)])
        self.assertEqual(len(runtime.update_calls), 1)
        self.assertEqual(runtime.optimizer_attempted_steps, 8)
        self.assertEqual(report.summary["cumulative_prefix"]["learned_decisions"], 3)
        self.assertEqual(report.summary["total_elapsed_seconds_with_resume_prefix"], 5.0)

    def test_checkpoint_recovery_requires_consistent_explicit_metadata(self):
        from spire_learn.neural.runner import _PROGRESS_FIELDS, _checkpoint_recovery

        config = replace(self.train_config, budgets=NeuralBudget(8, 4, 4, 60.0))
        progress = {name: 0 for name in _PROGRESS_FIELDS}
        progress.update({
            "attempted_environment_slots": 2,
            "forwarded_environment_slots": 2,
            "known_accepted_environment_slots": 2,
            "delivered_environment_slots": 2,
            "delivered_reset_slots": 2,
            "learned_decisions": 2,
        })
        progress["elapsed_seconds"] = 7.5
        recovery_projection = {
            "next_seed_ordinal": 2,
            "next_decision_sequence": 2,
            "next_episode_ordinals": [2],
        }
        metadata = {
            "run_metadata": {
                "mode": config.mode,
                "operation": "train",
                "split": "train",
                "partition": "train",
                "config_sha256": config.config_sha256,
                "manifest_sha256": config.manifest_sha256,
                "environment_revision": config.environment_revision,
                "collector_recovery": recovery_projection,
                "cumulative_prefix": dict(progress),
            },
            "training_state": {
                "next_environment_seed_ordinal": 2,
                "next_decision_sequence": 2,
                "counters": dict(progress),
                "optimizer_step": 0,
            },
        }
        loaded = SimpleNamespace(metadata=metadata)
        recovery, restored = _checkpoint_recovery(loaded, config)
        self.assertEqual(recovery, CollectorRecovery(2, 2, (2,)))
        self.assertEqual(restored, progress)

        bad = SimpleNamespace(metadata={
            **metadata,
            "run_metadata": {
                **metadata["run_metadata"],
                "collector_recovery": {**recovery_projection, "next_episode_ordinals": [1]},
            },
        })
        with self.assertRaisesRegex(ValueError, "episode progress disagrees"):
            _checkpoint_recovery(bad, config)

        bad = SimpleNamespace(metadata={
            **metadata,
            "run_metadata": {**metadata["run_metadata"], "collector_recovery": None},
        })
        with self.assertRaisesRegex(ValueError, "explicit collector recovery"):
            _checkpoint_recovery(bad, config)

        bad = SimpleNamespace(metadata={
            **metadata,
            "training_state": {**metadata["training_state"], "next_decision_sequence": 3},
        })
        with self.assertRaisesRegex(ValueError, "recovery scalars disagree"):
            _checkpoint_recovery(bad, config)

    def test_fixed_manifest_resume_still_rejects_changed_config_hash(self):
        from spire_learn.neural import runner
        _metadata, loaded, *_rest = _resume_bundle(self.train_config, {})
        loaded.metadata["run_metadata"]["config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "config hash does not match"):
            runner._checkpoint_recovery(loaded, self.train_config)

    def test_stream_collection_fault_preserves_prefix_and_forwards_no_salvage_update(self):
        for stream in (None, {}):
            with self.subTest(stream=stream):
                config = replace(self.train_config, scenario_stream=stream)
                original = ValueError("first collection fault")
                collector = FakeCollector(config, collect_error=original)
                collector.partial_rollout = _rollout(config, (_record(config, slot=0, decision=0, version=0, terminal=False),))
                runtime = FakeRuntime(config, collector)
                # Isolate post-admission collection behavior with this marker.
                # Real loader/admission checks live in test_neural_stream_resume.
                with patch("spire_learn.neural.stream_resume.stream_identity") as admission:
                    report, writer = self._run_train(config, runtime)
                self.assertEqual(admission.call_count, int(stream is not None))
                if stream is not None:
                    admission.assert_called_once_with(config)
                self.assertEqual(report.exit_code, 1)
                self.assertEqual(len(runtime.update_calls), int(stream is None))
                self.assertEqual(runtime.saved, [])
                self.assertTrue(any(event.get("event") == "accepted_action" for event in writer.events))

    def test_stream_partial_event_failure_preserves_first_collection_error(self):
        config = replace(self.train_config, scenario_stream={})
        original = ValueError("first collection fault")
        collector = FakeCollector(config, collect_error=original)
        collector.partial_rollout = _rollout(config, (_record(
            config, slot=0, decision=0, version=0, terminal=False),))
        runtime = FakeRuntime(config, collector)
        # This marker isolates post-admission behavior; actual invalid stream
        # rejection is covered through all entries in test_neural_stream_resume.
        with patch("spire_learn.neural.stream_resume.stream_identity") as admission:
            report, _ = self._run_train(config, runtime, FakeWriter(event_error=OSError("event failed")))
        admission.assert_called_once_with(config)
        self.assertEqual(runtime.update_calls, [])
        self.assertIn("first collection fault", str(report.summary))
        self.assertTrue(any("partial collection event also failed" in note
                            for note in original.__notes__))

    def test_partial_rollout_updates_only_actor_eligible_rows_and_uses_fresh_versions(self):
        config = replace(self.train_config, rollout_steps=2,
                         budgets=NeuralBudget(8, 4, 4, 60.0))
        first = _rollout(config, (
            _record(config, slot=0, decision=0, version=0, terminal=False, learned=True),
            _record(config, slot=1, decision=0, version=0, terminal=False, learned=False),
        ))
        second = lambda version: _rollout(config, (
            _record(config, slot=0, decision=1, version=version, terminal=False, learned=True),
        ))
        collector = FakeCollector(config, (first, second))
        runtime = FakeRuntime(config, collector)
        report, _ = self._run_train(config, runtime)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual([call[2] for call in collector.collect_calls], [0, 1])
        self.assertEqual(len(runtime.update_calls), 2)
        self.assertEqual(sum(record.actor_training_mask for record in runtime.update_calls[0].records), 1)
        self.assertEqual(sum(not record.actor_training_mask for record in runtime.update_calls[0].records), 1)
        self.assertEqual(runtime.update_calls[1].records[0].choice.behavior_policy_version, 1)

    def test_artifact_failure_stops_after_the_first_collector_dispatch(self):
        config = self.train_config
        allocation = _allocation(config, 0, 0)
        rollout = _rollout(config, (
            _record(config, slot=0, decision=0, version=0, terminal=False),
        ), allocations=(allocation,))
        collector = FakeCollector(config, (rollout, rollout))
        runtime = FakeRuntime(config, collector)
        writer = FakeWriter(event_error=OSError("event write failed"))
        report, _ = self._run_train(config, runtime, writer)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(len(collector.collect_calls), 1)
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(runtime.saved, [])
        self.assertEqual(runtime.close_calls, 1)

    def test_final_evaluation_requires_opt_in_before_dispatch(self):
        from spire_learn.neural.runner import evaluate_policy

        config = load_neural_config(ROOT / "configs/neural/standalone-final.toml")
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)
        report = evaluate_policy(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(collector.collect_calls, [])
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(runtime.close_calls, 1)

    def test_selection_evaluation_is_frozen_and_accounts_every_fixed_case(self):
        from spire_learn.neural.runner import evaluate_policy

        config = self.selection_config
        cases = config.cases.for_mode(config.mode)
        allocations = tuple(_allocation(config, 0, ordinal, case_index=ordinal)
                            for ordinal in range(len(cases)))
        rollouts = tuple(
            _rollout(config, (_record(config, slot=0, decision=ordinal, version=0,
                                      terminal=True, task_id=f"task-0-{ordinal}"),),
                     allocations=allocations[:ordinal + 1])
            for ordinal in range(len(cases))
        )
        collector = FakeCollector(config, rollouts)
        runtime = FakeRuntime(config, collector)
        writer = FakeWriter()
        from spire_learn.neural import runner
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-eval-test"))):
            report = evaluate_policy(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(runtime.update_calls, [])
        self.assertEqual(len(collector.collect_calls), 6)
        accounting = report.summary["case_accounting"]
        self.assertEqual(accounting["scheduled"], 6)
        self.assertEqual(accounting["complete"], 6)
        self.assertEqual(accounting["incomplete"], 0)
        self.assertEqual(accounting["not_started"], 0)

    def test_fixed_case_evaluation_rejects_multirow_before_runtime_construction(self):
        from spire_learn.neural.runner import create_runtime, evaluate_policy

        config = replace(self.selection_config, rollout_rows=2)
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)
        report = evaluate_policy(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(collector.collect_calls, [])

        with patch("spire_learn.neural.envs.make_driver") as make_driver:
            with self.assertRaisesRegex(ValueError, "rollout.rows=1"):
                create_runtime(config, purpose="evaluate", policy_kind="legal-random",
                               action_selection="seeded-sampling")
        make_driver.assert_not_called()

    def test_case_accounting_is_exhaustive_for_started_incomplete_and_not_started(self):
        from spire_learn.neural.runner import _case_accounting

        config = self.selection_config
        allocation = _allocation(config, 0, 0)
        incomplete = _record(config, slot=0, decision=0, version=0, terminal=False)
        diagnostics = CollectionDiagnostics(seed_allocations=(allocation,))
        report = _case_accounting(
            config, (incomplete,), diagnostics, CollectorRecovery(0, 0, (0,) * config.rollout_rows)
        )
        self.assertEqual(report["scheduled"], 6)
        self.assertEqual(report["complete"], 0)
        self.assertEqual(report["incomplete"], 1)
        self.assertEqual(report["not_started"], 5)
        self.assertEqual(len(report["cases"]), report["scheduled"])
        self.assertEqual({item["status"] for item in report["cases"]},
                         {"incomplete", "not_started"})

    def test_evaluation_empty_collection_stops_as_incomplete(self):
        from spire_learn.neural.runner import evaluate_policy

        config = self.selection_config
        empty = _rollout(config, (), stop_reason="inactive_reset_boundary")
        collector = FakeCollector(config, (empty,))
        runtime = FakeRuntime(config, collector)
        writer = FakeWriter()
        from spire_learn.neural import runner
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-eval-empty"))):
            report = evaluate_policy(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.summary["status"], "incomplete")
        self.assertEqual(len(collector.collect_calls), 1)
        self.assertEqual(runtime.update_calls, [])

    def test_close_failures_are_attached_without_replacing_keyboard_interrupt(self):
        config = self.train_config
        original = KeyboardInterrupt("stop requested")
        collector = FakeCollector(config, collect_error=original)
        runtime = FakeRuntime(config, collector, close_error=OSError("runtime close failed"))
        writer = FakeWriter(close_error=OSError("writer close failed"))
        from spire_learn.neural import runner
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-control-test"))):
            with self.assertRaises(KeyboardInterrupt) as caught:
                runner.train(config, runtime=runtime, clock=lambda: 0.0)
        self.assertIs(caught.exception, original)
        notes = "\n".join(getattr(original, "__notes__", ()))
        self.assertIn("runtime close", notes)
        self.assertIn("artifact writer close", notes)

    def test_clock_construction_failure_closes_runtime_before_returning(self):
        config = self.train_config
        collector = FakeCollector(config)
        runtime = FakeRuntime(config, collector)

        def broken_clock():
            raise RuntimeError("clock initialization failed")

        report, _ = self._run_train_with_clock(config, runtime, broken_clock)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(runtime.close_calls, 1)
        self.assertEqual(collector.collect_calls, [])

    def test_final_lifecycle_receipt_includes_save_runtime_and_writer_close(self):
        clock_value = [0.0]
        clock = lambda: clock_value[0]
        config = self.train_config
        collector = FakeCollector(config, (_rollout(config, ()),))
        runtime = FakeRuntime(
            config, collector,
            save_hook=lambda: clock_value.__setitem__(0, 3.0),
            close_hook=lambda: clock_value.__setitem__(0, 5.0),
        )
        writer = FakeWriter(close_hook=lambda: clock_value.__setitem__(0, 7.0))
        from spire_learn.neural import runner
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-lifecycle-test"))):
            report = runner.train(config, runtime=runtime, clock=clock)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.summary["status"], "complete")
        self.assertEqual(report.summary["elapsed_seconds"], 7.0)
        self.assertEqual(writer.lifecycle_receipts[0]["elapsed_seconds"], 7.0)
        self.assertIn("after runtime and artifact writer close", writer.lifecycle_receipts[0]["timing_boundary"])
        self.assertEqual(runtime.close_calls, 1)
        self.assertEqual(writer.close_calls, 1)

    def test_evaluation_lifecycle_receipt_includes_runtime_and_writer_close(self):
        from spire_learn.neural import runner

        config = self.selection_config
        cases = config.cases.for_mode(config.mode)
        allocations = tuple(_allocation(config, 0, ordinal, case_index=ordinal)
                            for ordinal in range(len(cases)))
        rollouts = tuple(
            _rollout(config, (_record(config, slot=0, decision=ordinal, version=0,
                                      terminal=True, task_id=f"task-0-{ordinal}"),),
                     allocations=allocations[:ordinal + 1])
            for ordinal in range(len(cases))
        )
        clock_value = [0.0]
        runtime = FakeRuntime(
            config, FakeCollector(config, rollouts),
            close_hook=lambda: clock_value.__setitem__(0, 5.0),
        )
        writer = FakeWriter(close_hook=lambda: clock_value.__setitem__(0, 7.0))
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-eval-lifecycle-test"))):
            report = runner.evaluate_policy(config, runtime=runtime,
                                            clock=lambda: clock_value[0])
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.summary["elapsed_seconds"], 7.0)
        self.assertEqual(writer.lifecycle_receipts[0]["elapsed_seconds"], 7.0)
        self.assertEqual(report.summary["case_accounting"]["complete"], len(cases))

    def test_save_failure_records_possibly_published_checkpoint_and_close_failure(self):
        config = self.train_config
        collector = FakeCollector(config, (_rollout(config, ()),))

        def fail_save():
            raise OSError("directory fsync failed after publication")

        runtime = FakeRuntime(config, collector, save_hook=fail_save,
                              close_error=OSError("runtime close failed"))
        writer = FakeWriter()
        from spire_learn.neural import runner
        with patch.object(runner, "_writer_start",
                          return_value=(writer, Path("/tmp/neural-save-failure-test"))):
            report = runner.train(config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.summary["status"], "artifact_failure")
        self.assertEqual(report.summary["checkpoint_publication"]["publication_state"], "possibly_published")
        self.assertEqual(len(runtime.saved), 1)
        self.assertEqual(writer.lifecycle_receipts[0]["status"], "artifact_failure")
        self.assertEqual(len(writer.lifecycle_receipts[0]["close_errors"]), 1)

    def test_preflight_close_process_control_is_propagated(self):
        config = replace(self.train_config, operation="evaluate", split="selection", partition="selection")
        original = KeyboardInterrupt("close control")
        runtime = FakeRuntime(config, FakeCollector(config), close_error=original)
        from spire_learn.neural import runner
        with self.assertRaises(KeyboardInterrupt) as caught:
            runner.train(config, runtime=runtime, clock=lambda: 0.0)
        self.assertIs(caught.exception, original)

    def test_preflight_close_failure_is_reported_and_setup_failure_closes_runtime(self):
        from spire_learn.neural import runner

        invalid_config = replace(self.train_config, operation="evaluate", split="selection",
                                 partition="selection")
        closing = FakeRuntime(invalid_config, FakeCollector(invalid_config),
                              close_error=OSError("preflight close failed"))
        report = runner.train(invalid_config, runtime=closing, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.summary["stop_reason"], "preflight_close_failure")
        self.assertEqual(closing.close_calls, 1)

        class SetupFailureRuntime(FakeRuntime):
            def set_budget(self, budget):
                raise RuntimeError("runtime setup failed")

        collector = FakeCollector(self.train_config)
        runtime = SetupFailureRuntime(self.train_config, collector)
        report = runner.train(self.train_config, runtime=runtime, clock=lambda: 0.0)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(runtime.close_calls, 1)
        self.assertEqual(collector.collect_calls, [])

    def test_reference_policies_use_legal_mask_and_public_action_verbs(self):
        from spire_learn.neural import evaluation

        schema_type = type("SchemaFake", (), {})
        schema = schema_type()
        schema.schema_fingerprint = FINGERPRINT
        schema.action_space = SimpleNamespace(n=5)
        spec = SimpleNamespace(
            shape=(5, 1), columns=(SimpleNamespace(field="verb", vocabulary="verb"),),
            known_key="actions_known",
        )
        schema.tensors = {"actions": spec}
        schema.vocabularies = {"verb": ("padding", "play_card", "end_turn", "use_potion", "choose")}
        verbs = (1, 2, 3, 4, 2)
        actions = b"".join(struct.pack("=q", value) for value in verbs)
        frame = CombatFrame({
            "action_mask": BufferArray("int8", (5,), bytes((1, 1, 0, 1, 1))),
            "actions": BufferArray("int64", (5, 1), actions),
            "actions_known": BufferArray("int8", (5, 1), bytes((1, 1, 1, 1, 1))),
        }, FINGERPRINT)
        with patch.object(evaluation, "CombatSchema", schema_type):
            self.assertEqual(evaluation.choose_reference(
                frame, schema, kind="end-turn-last-v1", rng=random.Random(1)
            ), 0)
            self.assertIn(evaluation.choose_reference(
                frame, schema, kind="legal-random", rng=random.Random(4)
            ), {0, 1, 3, 4})

    def test_cuda_runtime_uses_cuda_policy_rng_and_cpu_shuffle_rng_after_qualification(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        events = []
        torch_module = _AssemblyTorch(events)
        with patch.dict(os.environ, {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        }):
            with _patch_runtime_assembly(events, torch_module=torch_module) as (driver, policy):
                runtime = runner.create_runtime(config, purpose="train")
                self.assertLess(events.index("torch_import"), events.index("nvidia_smi"))
                self.assertLess(events.index("nvidia_smi"), events.index("make_driver"))
                self.assertLess(events.index("make_driver"), events.index(("build_policy", config.seed_streams.policy, "cuda:0")))
                self.assertEqual(runtime.policy_generator.device, "cuda:0")
                self.assertEqual(runtime.shuffle_generator.device, "cpu")
                self.assertEqual(runtime.device_report["gpu_uuid"], GPU_UUID)
                self.assertEqual([item.device for item in torch_module.generators], ["cuda:0", "cpu"])
                self.assertEqual(
                    [item.seed for item in torch_module.generators],
                    [config.seed_streams.policy, config.seed_streams.optimizer],
                )
                self.assertEqual(policy.loaded_weights, [])
                runtime.close()
                runtime.close()
        self.assertEqual(driver.close_calls, 1)

    def test_cuda_runtime_execution_gate_rejects_before_torch_import_or_driver(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        events = []
        with patch.dict(os.environ, {}, clear=True):
            with _patch_runtime_assembly(events, reject_torch_import=True) as (driver, _policy):
                with self.assertRaisesRegex(RuntimeError, "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME=1"):
                    runner.create_runtime(config, purpose="train")
        self.assertNotIn("torch_import", events)
        self.assertNotIn("make_driver", events)
        self.assertEqual(driver.close_calls, 0)

    def test_resume_preflights_before_torch_and_keeps_rng_and_adam_payloads_unchanged(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        fields = {
            "runtime_device": "cuda:0",
            "policy_rng_backend": "cuda",
            "runtime_gpu_uuid": GPU_UUID,
        }
        metadata, loaded, policy_state, minibatch_state, curriculum_state, controller_state, optimizer_state = (
            _resume_bundle(config, fields)
        )
        events = []

        def preflight(path, expected, *, purpose):
            events.append("preflight")
            self.assertEqual(purpose, "resume")
            return SimpleNamespace(metadata=metadata)

        def load(path, expected, *, purpose, device):
            events.append("load_checkpoint")
            self.assertEqual((purpose, device), ("resume", "cpu"))
            return loaded

        with patch.dict(os.environ, {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        }):
            torch_module = _AssemblyTorch(events)
            with _patch_runtime_assembly(
                events, torch_module=torch_module, preflight=preflight, loader=load,
            ) as (driver, policy):
                runtime = runner.create_runtime(
                    config, purpose="resume", checkpoint=Path("/tmp/fake-resume"),
                )
                self.assertLess(events.index("preflight"), events.index("torch_import"))
                self.assertLess(events.index("nvidia_smi"), events.index("load_checkpoint"))
                self.assertLess(events.index("load_checkpoint"), events.index("make_driver"))
                self.assertEqual(policy.loaded_weights[0][1], True)
                self.assertEqual(set(policy.loaded_weights[0][0]), {"weight"})
                self.assertIs(runtime.policy_generator.states[0], policy_state)
                self.assertIs(runtime.shuffle_generator.states[0], minibatch_state)
                self.assertEqual(runtime.curriculum_rng.getstate(), curriculum_state)
                self.assertEqual(runtime.controller_rng.getstate(), controller_state)
                self.assertIs(torch_module.adams[0].loaded_states[0], optimizer_state)
                self.assertEqual(runtime.device_report["gpu_uuid"], GPU_UUID)

                with patch.object(runner, "save_checkpoint") as save:
                    runtime.save(Path("/tmp/fake-checkpoint-output"), "resume")
                saved_metadata = save.call_args.kwargs["metadata"]
                saved_run_metadata = saved_metadata["run_metadata"]
                self.assertEqual(
                    {name: saved_run_metadata[name] for name in (
                        "runtime_device", "policy_rng_backend", "runtime_gpu_uuid",
                    )},
                    fields,
                )
                manifest = runner._manifest_projection(
                    config, runtime, policy_kind="checkpoint",
                    action_selection="seeded-sampling", allow_final=False,
                )
                self.assertEqual(manifest["runtime_device_report"], runtime.device_report)
                summary = runner._base_summary(
                    config, runtime, status="complete", stop_reason="test",
                    run_dir=None, elapsed=0.0, overshoot=0.0, records=(),
                    diagnostics=None, recovery=runtime.collector_recovery,
                    fatal_error=None, policy_kind="checkpoint",
                    action_selection="seeded-sampling", allow_final=False,
                )
                self.assertEqual(summary["runtime_device_report"], runtime.device_report)
                runtime.close()
        self.assertEqual(driver.close_calls, 1)

    def test_resume_device_mismatch_rejects_before_torch_gate_or_driver(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        mismatched_fields = {
            "runtime_device": "cpu", "policy_rng_backend": "cpu", "runtime_gpu_uuid": None,
        }
        metadata, _loaded, *_rest = _resume_bundle(config, mismatched_fields)
        events = []

        def preflight(path, expected, *, purpose):
            events.append("preflight")
            return SimpleNamespace(metadata=metadata)

        with patch.dict(os.environ, {}, clear=True):
            with _patch_runtime_assembly(
                events, preflight=preflight, reject_torch_import=True,
            ) as (driver, _policy):
                with self.assertRaisesRegex(ValueError, "does not match the selected config"):
                    runner.create_runtime(
                        config, purpose="resume", checkpoint=Path("/tmp/fake-resume"),
                    )
        self.assertEqual(events, ["preflight"])
        self.assertEqual(driver.close_calls, 0)

    def test_resume_manifest_mismatch_rejects_from_preflight_metadata_before_torch(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        fields = {
            "runtime_device": "cuda:0",
            "policy_rng_backend": "cuda",
            "runtime_gpu_uuid": GPU_UUID,
        }
        metadata, _loaded, *_rest = _resume_bundle(config, fields)
        preflight_metadata = copy.deepcopy(metadata)
        preflight_metadata["run_metadata"]["manifest_sha256"] = "f" * 64
        events = []

        def preflight(path, expected, *, purpose):
            events.append("preflight")
            return SimpleNamespace(metadata=preflight_metadata)

        with patch.dict(os.environ, {}, clear=True):
            with _patch_runtime_assembly(
                events, preflight=preflight, reject_torch_import=True,
            ) as (driver, _policy):
                with self.assertRaisesRegex(ValueError, "manifest hash does not match"):
                    runner.create_runtime(
                        config, purpose="resume", checkpoint=Path("/tmp/fake-resume"),
                    )
        self.assertEqual(events, ["preflight"])
        self.assertEqual(driver.close_calls, 0)

    def test_loaded_resume_metadata_is_rechecked_before_native_driver_factory(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        fields = {
            "runtime_device": "cuda:0",
            "policy_rng_backend": "cuda",
            "runtime_gpu_uuid": GPU_UUID,
        }
        metadata, loaded, *_rest = _resume_bundle(config, fields)
        preflight_metadata = copy.deepcopy(metadata)
        loaded.metadata["run_metadata"].update({
            "runtime_device": "cpu", "policy_rng_backend": "cpu", "runtime_gpu_uuid": None,
        })
        events = []

        def preflight(path, expected, *, purpose):
            events.append("preflight")
            return SimpleNamespace(metadata=preflight_metadata)

        def load(path, expected, *, purpose, device):
            events.append("load_checkpoint")
            return loaded

        with patch.dict(os.environ, {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        }):
            torch_module = _AssemblyTorch(events)
            with _patch_runtime_assembly(
                events, torch_module=torch_module, preflight=preflight, loader=load,
            ) as (driver, _policy):
                with self.assertRaisesRegex(ValueError, "does not match the selected config"):
                    runner.create_runtime(
                        config, purpose="resume", checkpoint=Path("/tmp/fake-resume"),
                    )
        self.assertIn("load_checkpoint", events)
        self.assertNotIn("make_driver", events)
        self.assertEqual(driver.close_calls, 0)

    def test_warm_start_and_evaluation_load_cpu_weights_without_restoring_training_state(self):
        from spire_learn.neural import runner

        cases = (
            (replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID), "warm_start"),
            (replace(self.selection_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID), "evaluate"),
        )
        for config, purpose in cases:
            with self.subTest(purpose=purpose):
                events = []
                loaded = SimpleNamespace(metadata={}, model_tensors={"weight": object()})
                load_calls = []

                def preflight(path, expected, *, purpose):
                    return SimpleNamespace(metadata={})

                def load(path, expected, *, purpose, device):
                    load_calls.append((purpose, device))
                    return loaded

                with patch.dict(os.environ, {
                    "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
                    "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
                }):
                    torch_module = _AssemblyTorch(events)
                    with _patch_runtime_assembly(
                        events, torch_module=torch_module, preflight=preflight, loader=load,
                    ) as (driver, policy):
                        runtime = runner.create_runtime(
                            config, purpose=purpose, checkpoint=Path("/tmp/fake-portable"),
                            action_selection="deterministic",
                        )
                        self.assertEqual(load_calls, [(purpose, "cpu")])
                        self.assertEqual(policy.loaded_weights[0][1], True)
                        self.assertEqual(runtime.policy_generator.device, "cuda:0")
                        self.assertEqual(runtime.shuffle_generator.device, "cpu")
                        self.assertEqual(torch_module.generators[0].seed, config.seed_streams.policy)
                        self.assertEqual(torch_module.generators[1].seed, config.seed_streams.optimizer)
                        if purpose == "warm_start":
                            self.assertIsNotNone(runtime.optimizer)
                            self.assertEqual(torch_module.adams[0].loaded_states, [])
                        else:
                            self.assertIsNone(runtime.optimizer)
                            self.assertEqual(torch_module.adams, [])
                        runtime.close()
                self.assertEqual(driver.close_calls, 1)

    def test_reference_policy_stays_torch_free_and_closes_driver_once(self):
        from spire_learn.neural import runner

        events = []
        with _patch_runtime_assembly(events, reject_torch_import=True) as (driver, _policy):
            runtime = runner.create_runtime(
                self.selection_config, purpose="evaluate", policy_kind="legal-random",
                action_selection="seeded-sampling",
            )
            self.assertIsNone(runtime.torch)
            self.assertIsNone(runtime.optimizer)
            self.assertIsInstance(runtime.policy_generator, random.Random)
            self.assertEqual(runtime.device_report["device"], "cpu")
            runtime.close()
            runtime.close()
        self.assertNotIn("torch_import", events)
        self.assertNotIn("nvidia_smi", events)
        self.assertEqual(driver.close_calls, 1)

    def test_cuda_assembly_failure_preserves_keyboard_interrupt_and_closes_driver_once(self):
        from spire_learn.neural import runner

        config = replace(self.train_config, device="cuda:0", runtime_gpu_uuid=GPU_UUID)
        events = []
        interrupt = KeyboardInterrupt("stop requested")

        def fail_policy(*args, **kwargs):
            raise interrupt

        with patch.dict(os.environ, {
            "SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME": "1",
            "SPIRE_LEARN_NEURAL_GPU_UUID": GPU_UUID,
        }):
            torch_module = _AssemblyTorch(events)
            with _patch_runtime_assembly(
                events, torch_module=torch_module, policy_factory=fail_policy,
            ) as (driver, _policy):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    runner.create_runtime(config, purpose="train")
        self.assertIs(caught.exception, interrupt)
        self.assertEqual(driver.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
